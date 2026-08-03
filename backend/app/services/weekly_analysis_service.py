"""Production-only composition and verified local-data adapter for V2."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from pathlib import Path
from typing import Callable, Mapping, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ..models.models import Instrument, MarketPrice, ValuationRecord
from ..weekly_analysis.aggregation import aggregate_bars
from ..weekly_analysis.domain import DailyBar
from ..weekly_analysis.engine import (
    ANALYSIS_YEARS,
    AnalysisRequest,
    MODEL_LINE,
    DatasetQualityError,
    JudgmentPolicy,
    VerifiedDataset,
    WeeklyJudgment,
    WARMUP_WEEKS,
    WeeklyAnalysisEngine,
)
from ..weekly_analysis.features import FeatureSnapshot, ValuationPoint
from ..weekly_analysis.optimizer import TradingCalendar, WorkState
from ..weekly_analysis.repository import WeeklyAnalysisRepository
from ..weekly_analysis.tasks import WeeklyAnalysisTaskManager
from .market_calendar import CalendarProvider, ExchangeCalendarProvider


class WeeklyAnalysisUnavailable(RuntimeError):
    """The production quality/scoring boundary is not configured."""


class PositionPercentProvider(Protocol):
    def __call__(self, symbol: str) -> int | None: ...


class UnavailableJudgmentPolicy:
    """Fail closed until an audited scorer is explicitly composed."""

    def judge(
        self,
        *,
        model,
        features,
        sample,
        current_position,
    ):
        raise WeeklyAnalysisUnavailable(
            "No audited weekly judgment scorer is available"
        )


class RuleBasedJudgmentPolicy:
    """Deterministic weekly scorer driven by the progressively learned weights.

    The policy performs no network, AI, random, persistence, or future-data
    access. User position is validated at the boundary but deliberately does
    not participate in the market score, probability, or target calculation.
    """

    counterfactual_evaluator_identity = (
        "rule-based-weekly-counterfactual-v1"
    )

    def judge(
        self,
        *,
        model: WorkState,
        features: FeatureSnapshot,
        sample,
        current_position: int | None,
    ) -> WeeklyJudgment:
        _validate_rule_inputs(
            model=model,
            features=features,
            sample=sample,
            current_position=current_position,
        )
        values = features.features
        valuation = _feature_decimal(values, "valuation_percentile")
        dif = _feature_decimal(values, "dif")
        dea = _feature_decimal(values, "dea")
        histogram = _feature_decimal(values, "macd_histogram")
        ema_12 = _feature_decimal(values, "ema_12")
        ma_20 = _feature_decimal(values, "ma_20")
        ma_60 = _feature_decimal(values, "ma_60")
        golden = values.get("golden_point") is True
        black = values.get("black_point") is True
        golden_strength = _feature_integer(values, "golden_strength")
        black_strength = _feature_integer(values, "black_strength")

        strong_black = black and black_strength >= 4
        strong_golden = (
            golden
            and golden_strength >= 4
            and _greater(dif, dea)
            and _nonnegative(histogram)
            and _at_or_above(ema_12, ma_20)
        )
        overvalued_trend = (
            valuation is not None
            and valuation >= Decimal("85")
            and not black
            and _positive(dif)
            and _positive(dea)
            and _at_or_above(ema_12, ma_20)
            and _at_or_above(ema_12, ma_60)
        )

        if (
            valuation is not None
            and valuation >= Decimal("85")
            and strong_black
        ):
            return _judgment(
                direction="down",
                probability=Decimal("81"),
                confidence=Decimal("84"),
                target_position=15,
                confirmed=True,
            )
        if overvalued_trend:
            return _judgment(
                direction="up",
                probability=Decimal("70"),
                confidence=Decimal("70"),
                target_position=70,
                confirmed=False,
            )
        if (
            valuation is not None
            and valuation <= Decimal("30")
            and strong_golden
        ):
            return _judgment(
                direction="up",
                probability=Decimal("78"),
                confidence=Decimal("80"),
                target_position=80,
                confirmed=True,
            )
        if valuation is not None and valuation <= Decimal("30"):
            return _judgment(
                direction="up",
                probability=Decimal("58"),
                confidence=Decimal("61"),
                target_position=20,
                confirmed=False,
            )

        component_scores = _component_scores(features)
        score = sum(
            (
                component_scores[name]
                * model.current_model_weights[name]
                / Decimal("100")
            )
            for name in model.current_model_weights
        )
        if score >= Decimal("12"):
            direction = "up"
        elif score <= Decimal("-12"):
            direction = "down"
        else:
            direction = "neutral"
        target = _target_from_score(score)
        if direction == "neutral":
            probability = _rounded_clamp(
                Decimal("58") - abs(score) / Decimal("2"),
                Decimal("52"),
                Decimal("58"),
            )
        else:
            probability = _rounded_clamp(
                Decimal("52") + abs(score) * Decimal("0.45"),
                Decimal("55"),
                Decimal("90"),
            )
        confidence = _rounded_clamp(
            Decimal("55") + abs(score) * Decimal("0.35"),
            Decimal("55"),
            Decimal("90"),
        )
        return _judgment(
            direction=direction,
            probability=probability,
            confidence=confidence,
            target_position=target,
            confirmed=strong_black or strong_golden,
        )


class LocalVerifiedDatasetProvider:
    """Read verified local rows without refresh, demo fallback, or network I/O."""

    def __init__(
        self,
        sessions: sessionmaker[Session] | Callable[[], Session],
        *,
        calendar_provider: CalendarProvider | None = None,
        current_position_provider: PositionPercentProvider | None = None,
    ) -> None:
        if not callable(sessions):
            raise TypeError("sessions must be a session factory")
        self._sessions = sessions
        self._calendar = calendar_provider or ExchangeCalendarProvider()
        self._current_position = (
            current_position_provider or (lambda _symbol: None)
        )

    def load_verified_window(
        self,
        symbol: str,
        *,
        as_of: date,
        years: int,
        warmup_weeks: int,
    ) -> VerifiedDataset:
        if symbol not in {"399006", "NDX"}:
            raise DatasetQualityError("Only 399006 and NDX are supported")
        if years != ANALYSIS_YEARS or warmup_weeks != WARMUP_WEEKS:
            raise DatasetQualityError(
                "Production V2 requires years=10 and warmup_weeks=60"
            )
        if not isinstance(as_of, date):
            raise TypeError("as_of must be a date")
        with self._sessions() as session:
            instrument_id = session.scalar(
                select(Instrument.id).where(Instrument.code == symbol)
            )
            if instrument_id is None:
                raise DatasetQualityError(
                    f"Unknown direct-index instrument: {symbol}"
                )
            latest_local_date = session.scalar(
                select(func.max(MarketPrice.trade_date)).where(
                    MarketPrice.instrument_id == instrument_id,
                    MarketPrice.timeframe == "daily",
                    MarketPrice.trade_date <= as_of,
                )
            )
        if latest_local_date is None:
            raise DatasetQualityError(
                f"No local verified daily history exists for {symbol}"
            )
        # The caller's date may be a trading day whose close has not reached
        # the local cache yet. Exclude that unfinished week instead of treating
        # the absent future close as a historical data gap.
        as_of = min(as_of, latest_local_date)
        boundary = _subtract_years(as_of, years)
        calendar_start = boundary - timedelta(
            weeks=warmup_weeks + 20
        )
        calendar_end = as_of + timedelta(weeks=14)
        expected_sessions = self._calendar.sessions(
            symbol, calendar_start, calendar_end
        )
        if not expected_sessions:
            raise DatasetQualityError(
                "Exchange calendar returned no verified sessions"
            )

        with self._sessions() as session:
            prices = tuple(
                session.scalars(
                    select(MarketPrice)
                    .where(
                        MarketPrice.instrument_id == instrument_id,
                        MarketPrice.timeframe == "daily",
                        MarketPrice.trade_date >= calendar_start,
                        MarketPrice.trade_date <= as_of,
                    )
                    .order_by(MarketPrice.trade_date)
                )
            )
            valuation_rows = tuple(
                session.scalars(
                    select(ValuationRecord)
                    .where(
                        ValuationRecord.instrument_id == instrument_id,
                        ValuationRecord.valuation_date >= calendar_start,
                        ValuationRecord.valuation_date <= as_of,
                    )
                    .order_by(ValuationRecord.valuation_date)
                )
            )
        if not prices:
            raise DatasetQualityError(
                f"No local verified daily history exists for {symbol}"
            )
        daily = tuple(_daily_bar(symbol, row) for row in prices)
        aggregation = aggregate_bars(
            daily,
            "weekly",
            symbol,
            expected_trade_dates=expected_sessions,
            as_of=as_of,
        )
        if not aggregation.report.is_publishable:
            codes = ",".join(
                issue.code
                for issue in aggregation.report.issues
                if issue.severity == "blocking"
            )
            raise DatasetQualityError(
                f"Local weekly aggregation failed quality gates: {codes}"
            )
        weekly = aggregation.bars
        analysis_rows = tuple(
            row for row in weekly if row.period_start >= boundary
        )
        if not analysis_rows:
            raise DatasetQualityError(
                "Local history does not reach the ten-year analysis window"
            )
        first_analysis_index = weekly.index(analysis_rows[0])
        if first_analysis_index < warmup_weeks:
            raise DatasetQualityError(
                "Local history lacks 60 complete warmup weeks"
            )
        valuations = tuple(
            _valuation_point(row) for row in valuation_rows
        )
        valuations = tuple(
            point for point in valuations if point is not None
        )
        # Direct-index valuation providers are allowed to be unavailable or
        # sparse.  Preserve only their real observations: feature generation
        # reports VALUATION_UNAVAILABLE/INSUFFICIENT_VALUATION_HISTORY as
        # non-blocking warnings and continues with weekly technical signals.
        # Never manufacture a ten-year PE/PB series by forward-filling prices,
        # ETF proxies, or long gaps between observed valuation dates.
        return VerifiedDataset(
            instrument_code=symbol,
            daily_bars=daily,
            completed_weekly_bars=weekly,
            valuations=valuations,
            trading_calendar=TradingCalendar(
                instrument_code=symbol,
                expected_trade_dates=expected_sessions,
                coverage_start=expected_sessions[0],
                coverage_end=expected_sessions[-1],
            ),
            complete_week_keys=tuple(
                _week_key(row.period_end) for row in weekly
            ),
            current_position=None,
        )

    def latest_complete_week(self, symbol: str) -> str:
        with self._sessions() as session:
            latest_local_date = session.scalar(
                select(func.max(MarketPrice.trade_date))
                .join(Instrument)
                .where(
                    Instrument.code == symbol,
                    MarketPrice.timeframe == "daily",
                )
            )
        if latest_local_date is None:
            raise DatasetQualityError(
                f"No local verified daily history exists for {symbol}"
            )
        dataset = self.load_verified_window(
            symbol,
            # The public source may not have published today's close yet.
            # Anchor completion to the latest locally visible direct-index
            # session so an unfinished current week is excluded cleanly.
            as_of=latest_local_date,
            years=ANALYSIS_YEARS,
            warmup_weeks=WARMUP_WEEKS,
        )
        if not dataset.complete_week_keys:
            raise DatasetQualityError("No complete trading week is available")
        return dataset.complete_week_keys[-1]

    def capture_analysis_request(
        self,
        symbol: str,
        *,
        model_line: str,
    ) -> AnalysisRequest:
        latest = self.latest_complete_week(symbol)
        position = self._current_position(symbol)
        if position is not None and (
            type(position) is not int or not 0 <= position <= 100
        ):
            raise DatasetQualityError(
                "current_position must be an integer percent from 0 to 100"
            )
        payload = "unset" if position is None else str(position)
        return AnalysisRequest(
            latest_complete_week=latest,
            model_line=model_line,
            current_position=position,
            position_snapshot_key=hashlib.sha256(
                payload.encode("ascii")
            ).hexdigest(),
        )

    def position_snapshot_key(self, symbol: str) -> str:
        return self.capture_analysis_request(
            symbol, model_line=MODEL_LINE
        ).position_snapshot_key


class WeeklyAnalysisService:
    """Production facade intentionally omitting the test-only ``as_of``."""

    def __init__(
        self,
        *,
        provider: LocalVerifiedDatasetProvider,
        task_manager: WeeklyAnalysisTaskManager,
    ) -> None:
        self.provider = provider
        self.task_manager = task_manager

    def start(self, symbol: str):
        request = self.provider.capture_analysis_request(
            symbol,
            model_line=MODEL_LINE,
        )
        return self.task_manager.submit(
            symbol,
            latest_complete_week=request.latest_complete_week,
            model_line=request.model_line,
            advice_request_key=request.position_snapshot_key,
            analysis_request=request,
        )

    def get(self, task_id: str):
        return self.task_manager.get(task_id)

    def shutdown(self, *, wait: bool = True) -> None:
        self.task_manager.shutdown(wait=wait)


def build_weekly_analysis_service(
    sessions: sessionmaker[Session],
    *,
    judgment_policy: JudgmentPolicy | None = None,
    calendar_provider: CalendarProvider | None = None,
    current_position_provider: PositionPercentProvider | None = None,
    audit_root: Path | str = Path("data") / "weekly_analysis_v2",
) -> WeeklyAnalysisService:
    """Compose local production adapters without starting a task or network I/O."""

    provider = LocalVerifiedDatasetProvider(
        sessions,
        calendar_provider=calendar_provider,
        current_position_provider=current_position_provider,
    )
    repository = WeeklyAnalysisRepository(
        sessions,
        audit_root=audit_root,
    )
    engine = WeeklyAnalysisEngine(
        provider=provider,
        repository=repository,
        judgment_policy=(
            judgment_policy or RuleBasedJudgmentPolicy()
        ),
    )
    manager = WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
    )
    return WeeklyAnalysisService(
        provider=provider,
        task_manager=manager,
    )


def _daily_bar(symbol: str, row: MarketPrice) -> DailyBar:
    source = (row.source or "").upper()
    if (
        not source
        or source.startswith("DEMO")
        or any(marker in source for marker in ("ETF", "FUND", "QQQ", "159941"))
    ):
        raise DatasetQualityError(
            "Local direct-index rows contain demo/proxy/fund sources"
        )
    multiplier = Decimal(row.volume_multiplier)
    volume = (
        None
        if symbol == "NDX" or row.volume is None
        else row.volume * multiplier
    )
    return DailyBar(
        trade_date=row.trade_date,
        open_price=row.open_price,
        high_price=row.high_price,
        low_price=row.low_price,
        close_price=row.close_price,
        adjusted_close_price=row.adjusted_close_price,
        volume=volume,
        turnover=None,
        source=row.source,
    )


def _valuation_point(row: ValuationRecord) -> ValuationPoint | None:
    raw = dict(row.raw_values)
    source = str(raw.get("source", "")).strip().upper()
    direct_index_sources = {
        "AKSHARE_CSINDEX",
        "OFFICIAL_DIRECT_INDEX",
        "VERIFIED_DIRECT_INDEX_CACHE",
    }
    if bool(raw.get("proxy", False)) or source not in direct_index_sources:
        raise DatasetQualityError(
            "valuation source is not a verified direct index source"
        )
    value = row.pe_ratio if row.pe_ratio is not None else row.pb_ratio
    if value is None:
        return None
    return ValuationPoint(row.valuation_date, value)


def _quality_checked_valuations(
    valuations: tuple[ValuationPoint, ...],
    *,
    expected_sessions: tuple[date, ...],
    analysis_start: date,
    analysis_end: date,
) -> tuple[ValuationPoint, ...]:
    """Forward-estimate at most two isolated sessions, then fail closed."""

    sessions = tuple(
        day
        for day in expected_sessions
        if analysis_start <= day <= analysis_end
    )
    if not sessions:
        raise DatasetQualityError(
            "valuation coverage has no expected trading sessions"
        )
    ordered = tuple(sorted(valuations, key=lambda point: point.valuation_date))
    if len({point.valuation_date for point in ordered}) != len(ordered):
        raise DatasetQualityError("valuation history contains duplicate dates")
    by_date = {point.valuation_date: point for point in ordered}
    prior = tuple(
        point for point in ordered if point.valuation_date < sessions[0]
    )
    last_value = prior[-1].value if prior else None
    gap_length = 0
    checked: list[ValuationPoint] = []
    for session in sessions:
        observed = by_date.get(session)
        if observed is not None:
            checked.append(observed)
            last_value = observed.value
            gap_length = 0
            continue
        gap_length += 1
        if last_value is None or gap_length > 2:
            kind = "stale terminal" if session == sessions[-1] else "gap"
            raise DatasetQualityError(
                f"valuation coverage {kind} exceeds two trading sessions"
            )
        checked.append(
            ValuationPoint(
                valuation_date=session,
                value=last_value,
                estimated=True,
            )
        )
    return prior + tuple(checked)


def _week_key(day: date) -> str:
    iso_year, iso_week, _weekday = day.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _validate_rule_inputs(
    *,
    model: WorkState,
    features: FeatureSnapshot,
    sample,
    current_position: int | None,
) -> None:
    if not isinstance(model, WorkState):
        raise TypeError("model must be a WorkState")
    if not isinstance(features, FeatureSnapshot):
        raise TypeError("features must be a FeatureSnapshot")
    if (
        features.instrument_code != model.instrument_code
        or getattr(sample, "symbol", None) != model.instrument_code
    ):
        raise DatasetQualityError(
            "Rule policy model, features, and sample markets must match"
        )
    if not features.quality_report.is_publishable:
        raise DatasetQualityError(
            "Rule policy requires a publishable feature snapshot"
        )
    if current_position is not None and (
        type(current_position) is not int
        or not 0 <= current_position <= 100
    ):
        raise DatasetQualityError(
            "current_position must be an integer percent from 0 to 100"
        )


def _component_scores(
    snapshot: FeatureSnapshot,
) -> dict[str, Decimal]:
    values = snapshot.features
    valuation = _feature_decimal(values, "valuation_percentile")
    dif = _feature_decimal(values, "dif")
    dea = _feature_decimal(values, "dea")
    histogram = _feature_decimal(values, "macd_histogram")
    ema_12 = _feature_decimal(values, "ema_12")
    ma_20 = _feature_decimal(values, "ma_20")
    ma_60 = _feature_decimal(values, "ma_60")
    rsi = _feature_decimal(values, "rsi_6")
    volume_ratio = _feature_decimal(values, "volume_ratio")
    volatility = _feature_decimal(values, "volatility_20")
    drawdown = _feature_decimal(values, "current_drawdown")
    golden_strength = _feature_integer(values, "golden_strength")
    black_strength = _feature_integer(values, "black_strength")

    valuation_score = (
        Decimal()
        if valuation is None
        else _clamp((Decimal("50") - valuation) * Decimal("2"))
    )
    dea_score = _clamp(
        _sign_score(dea, Decimal("50"))
        + _sign_score(histogram, Decimal("25"))
        + _comparison_score(dif, dea, Decimal("25"))
    )
    dif_score = _clamp(
        _sign_score(dif, Decimal("45"))
        + _comparison_score(dif, dea, Decimal("35"))
        + _sign_score(histogram, Decimal("20"))
    )
    reversal_score = _clamp(
        Decimal(golden_strength - black_strength) * Decimal("20")
    )
    momentum_score = _clamp(
        _comparison_score(ema_12, ma_20, Decimal("50"))
        + _comparison_score(ema_12, ma_60, Decimal("50"))
    )

    risk_score = Decimal()
    if rsi is not None:
        risk_score += _clamp(
            (rsi - Decimal("50")) * Decimal("2"),
            Decimal("-40"),
            Decimal("40"),
        )
    if volume_ratio is not None:
        risk_score += _clamp(
            (volume_ratio - Decimal("1")) * Decimal("30"),
            Decimal("-15"),
            Decimal("15"),
        )
    if drawdown is not None:
        if drawdown <= Decimal("-0.20"):
            risk_score -= Decimal("35")
        elif drawdown <= Decimal("-0.10"):
            risk_score -= Decimal("15")
    if volatility is not None:
        if volatility >= Decimal("0.10"):
            risk_score -= Decimal("30")
        elif volatility >= Decimal("0.06"):
            risk_score -= Decimal("15")
    return {
        "valuation": valuation_score,
        "dea_trend": dea_score,
        "dif_trend": dif_score,
        "reversal": reversal_score,
        "price_momentum": momentum_score,
        "risk_regime": _clamp(risk_score),
    }


def _judgment(
    *,
    direction: str,
    probability: Decimal,
    confidence: Decimal,
    target_position: int,
    confirmed: bool,
) -> WeeklyJudgment:
    remainder = Decimal("100") - probability
    half = remainder / Decimal("2")
    if direction == "up":
        probabilities = {
            "up": probability,
            "down": half,
            "neutral": half,
        }
    elif direction == "down":
        probabilities = {
            "up": half,
            "down": probability,
            "neutral": half,
        }
    else:
        probabilities = {
            "up": half,
            "down": half,
            "neutral": probability,
        }
    return WeeklyJudgment(
        direction=direction,
        probability=probability,
        confidence=confidence,
        target_position=target_position,
        direction_probabilities=probabilities,
        confirmation_conditions_met=confirmed,
    )


def _feature_decimal(
    values: Mapping[str, object],
    name: str,
) -> Decimal | None:
    raw = values.get(name)
    if isinstance(raw, bool) or not isinstance(raw, (int, Decimal)):
        return None
    result = Decimal(raw)
    return result if result.is_finite() else None


def _feature_integer(values: Mapping[str, object], name: str) -> int:
    raw = values.get(name)
    if isinstance(raw, bool) or not isinstance(raw, (int, Decimal)):
        return 0
    value = Decimal(raw)
    if not value.is_finite():
        return 0
    return max(0, min(5, int(value)))


def _target_from_score(score: Decimal) -> int:
    for threshold, target in (
        (Decimal("65"), 85),
        (Decimal("40"), 75),
        (Decimal("20"), 65),
        (Decimal("8"), 60),
        (Decimal("-8"), 50),
        (Decimal("-20"), 40),
        (Decimal("-40"), 30),
        (Decimal("-65"), 20),
    ):
        if score >= threshold:
            return target
    return 15


def _sign_score(
    value: Decimal | None,
    magnitude: Decimal,
) -> Decimal:
    if value is None or value == 0:
        return Decimal()
    return magnitude if value > 0 else -magnitude


def _comparison_score(
    left: Decimal | None,
    right: Decimal | None,
    magnitude: Decimal,
) -> Decimal:
    if left is None or right is None or left == right:
        return Decimal()
    return magnitude if left > right else -magnitude


def _positive(value: Decimal | None) -> bool:
    return value is not None and value > 0


def _nonnegative(value: Decimal | None) -> bool:
    return value is not None and value >= 0


def _greater(
    left: Decimal | None,
    right: Decimal | None,
) -> bool:
    return left is not None and right is not None and left > right


def _at_or_above(
    left: Decimal | None,
    right: Decimal | None,
) -> bool:
    return left is not None and right is not None and left >= right


def _clamp(
    value: Decimal,
    minimum: Decimal = Decimal("-100"),
    maximum: Decimal = Decimal("100"),
) -> Decimal:
    return max(minimum, min(maximum, value))


def _rounded_clamp(
    value: Decimal,
    minimum: Decimal,
    maximum: Decimal,
) -> Decimal:
    return _clamp(value, minimum, maximum).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )


__all__ = [
    "LocalVerifiedDatasetProvider",
    "RuleBasedJudgmentPolicy",
    "UnavailableJudgmentPolicy",
    "WeeklyAnalysisService",
    "WeeklyAnalysisUnavailable",
    "build_weekly_analysis_service",
]
