"""Leakage-safe V3.3 feature engine for the 20-week hybrid model.

The engine deliberately keeps feature construction separate from persistence.
It can consume SQLAlchemy rows, immutable dataclasses, or point-in-time
observations supplied by the V3.3 repository.  Every non-price observation is
filtered by ``available_at <= cutoff``; a reporting/effective date alone is
never treated as proof that the value was known.

V3.3 predicts two *independent* tradable targets:

* ``399006`` - ChiNext direct index data.
* ``159941`` - GF Nasdaq-100 ETF CNY market data.  ``NDX`` is an explanatory
  benchmark only and can never be substituted for the ETF target/label.

Missing fundamentals and macro observations remain ``None`` and receive an
explicit missing mask.  They are not silently replaced with zero or a neutral
percentile in this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from datetime import timedelta
from decimal import Decimal
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.models import Instrument, MarketPrice, ValuationRecord


SUPPORTED_MARKETS = ("399006", "159941")
BENCHMARK_BY_MARKET: Mapping[str, str | None] = {
    "399006": None,
    "159941": "NDX",
}
HORIZON_WEEKS = 20
DAILY_WINDOW = 100
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
WEEKLY_MACD_STABLE_BARS = MACD_SLOW + MACD_SIGNAL

# Point-in-time families required by the approved V3.3 design.  Individual
# series may be absent, but the absence must be visible to the model and UI.
OBSERVATION_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "valuation": ("pe", "pb"),
    "earnings": ("earnings_growth",),
    "rates": ("short_rate", "long_rate", "real_rate", "term_spread"),
    "fund_flow": ("fund_flow", "market_breadth"),
    "macro": ("liquidity", "inflation", "pmi", "employment", "volatility_index"),
    "qdii": (
        "nav",
        "estimated_nav",
        "fx_usdcny",
        "fund_shares",
        "aum",
        "subscriptions",
    ),
}
DEFAULT_PIT_SERIES_CODES = tuple(
    series for names in OBSERVATION_FAMILIES.values() for series in names
)


class V33FeatureError(RuntimeError):
    """Feature construction cannot satisfy the V3.3 audit contract."""


class FutureLeakageError(V33FeatureError):
    """Raised when a caller attempts to provide information from the future."""


@dataclass(frozen=True, slots=True)
class PriceBar:
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    # Unadjusted exchange close is retained separately for price-vs-NAV
    # quantities.  ``close`` remains the qfq series used by technical returns.
    raw_close: float | None = None
    volume: float | None = None
    turnover: float | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class PointInTimeObservation:
    """A value together with when it was truly usable by the application."""

    market: str
    series: str
    value: float
    effective_date: date
    published_at: datetime
    available_at: datetime
    source: str
    unit: str | None = None
    vintage: str | None = None
    quality_status: str = "verified"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    market: str
    cutoff_date: date
    source_data_max_date: date
    daily_as_of: date
    weekly_as_of: date
    weekly: Mapping[str, float | None]
    daily: Mapping[str, float | None]
    daily_sequence: tuple[Mapping[str, float | str | None], ...]
    missing_masks: Mapping[str, int]
    provenance: Mapping[str, Any]
    derivative_turn: Mapping[str, Any]
    benchmark_market: str | None = None


def _number(value: Decimal | float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _safe_return(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0.0):
        return None
    return current / previous - 1.0


def _valid_ohlc(bar: PriceBar) -> bool:
    """Accept only machine-rounding excursions, not genuine OHLC violations."""

    values = (bar.open, bar.high, bar.low, bar.close)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        return False
    scale = max(1.0, *(abs(float(value)) for value in values))
    tolerance = max(1e-9, scale * 1e-12)
    upper = max(bar.open, bar.close)
    lower = min(bar.open, bar.close)
    return (
        bar.high + tolerance >= upper
        and bar.low - tolerance <= lower
        and bar.high + tolerance >= bar.low
    )


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _std(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.pstdev(clean) if len(clean) >= 2 else None


def _ema(values: Sequence[float], period: int) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    output = np.empty(len(values), dtype=float)
    output[0] = float(values[0])
    alpha = 2.0 / (period + 1.0)
    for index in range(1, len(values)):
        output[index] = alpha * float(values[index]) + (1.0 - alpha) * output[index - 1]
    return output


def macd(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standard DIF, DEA and ``2 * (DIF - DEA)`` sequences."""

    fast = _ema(values, MACD_FAST)
    slow = _ema(values, MACD_SLOW)
    dif = fast - slow
    dea = _ema(dif.tolist(), MACD_SIGNAL)
    histogram = 2.0 * (dif - dea)
    return dif, dea, histogram


def _linear_slope(values: Sequence[float], span: int, scale: float = 1.0) -> float | None:
    if len(values) < span or span < 2:
        return None
    y = np.asarray(values[-span:], dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    slope = float(np.polyfit(np.arange(span, dtype=float), y, 1)[0])
    return slope / max(abs(scale), 1e-12)


def _curvature(values: Sequence[float], span: int, scale: float = 1.0) -> float | None:
    if len(values) < span or span < 3:
        return None
    y = np.asarray(values[-span:], dtype=float)
    if not np.all(np.isfinite(y)):
        return None
    a = float(np.polyfit(np.arange(span, dtype=float), y, 2)[0])
    # The second derivative of a*t^2+b*t+c is 2*a.
    return 2.0 * a / max(abs(scale), 1e-12)


def _quadratic_turn(values: Sequence[float], span: int, scale: float) -> dict[str, Any] | None:
    """Fit a local quadratic and return its derivative-zero forecast.

    ``x=0`` is the latest observation, so a positive ``days_ahead`` is a
    future turning point.  The zero of DIF itself is separately reported and
    never confused with the zero of its derivative.
    """

    if len(values) < span:
        return None
    y = np.asarray(values[-span:], dtype=float) / max(abs(scale), 1e-12)
    x = np.arange(-span + 1, 1, dtype=float)
    a, b, c = (float(value) for value in np.polyfit(x, y, 2))
    derivative_zero = None if abs(a) < 1e-10 else -b / (2.0 * a)
    kind = "flat"
    if derivative_zero is not None:
        kind = "bottom" if a > 0 else "top"
    roots = np.roots([a, b, c]) if abs(a) >= 1e-10 else np.roots([b, c]) if abs(b) >= 1e-10 else []
    future_axis_roots = sorted(
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1e-7 and 0.0 <= float(root.real) <= 100.0
    )
    return {
        "window": span,
        "a": a,
        "b": b,
        "c": c,
        "derivative_zero_days": (
            None
            if derivative_zero is None or derivative_zero < 0.0 or derivative_zero > 100.0
            else float(derivative_zero)
        ),
        "axis_zero_days": None if not future_axis_roots else future_axis_roots[0],
        "kind": kind,
        "second_derivative": 2.0 * a,
    }


def _turn_consensus(values: Sequence[float], windows: Sequence[int], scale: float) -> dict[str, Any]:
    fits = [fit for span in windows if (fit := _quadratic_turn(values, span, scale)) is not None]
    usable = [fit for fit in fits if fit["derivative_zero_days"] is not None and fit["kind"] != "flat"]
    directions = {fit["kind"] for fit in usable}
    days = [float(fit["derivative_zero_days"]) for fit in usable]
    dispersion = _std(days)
    stable = bool(usable) and len(directions) == 1 and (dispersion is None or dispersion <= 5.0)
    return {
        "fits": fits,
        "stable": stable,
        "kind": next(iter(directions)) if stable else "unstable",
        "days_ahead": _mean(days) if stable else None,
        "dispersion": dispersion,
    }


_TURN_KIND_VALUE = {"top": -1.0, "flat": 0.0, "unstable": 0.0, "bottom": 1.0}


def _turn_model_features(
    prefix: str,
    consensus: Mapping[str, Any],
    windows: Sequence[int],
) -> dict[str, float | None]:
    """Project audited DIF quadratic fits into stable scalar model inputs.

    ``FeatureSnapshot.derivative_turn`` retains the structured explanation for
    the UI and audit trail.  The model, however, consumes the scalar mapping
    returned here.  Keeping the projection explicit prevents the historical
    bug where the fits were calculated and persisted but never reached the
    weekly or daily design matrices.
    """

    fits_by_window = {
        int(fit["window"]): fit
        for fit in consensus.get("fits", ())
        if isinstance(fit, Mapping) and fit.get("window") is not None
    }
    output: dict[str, float | None] = {}
    for span in windows:
        fit = fits_by_window.get(int(span))
        stem = f"{prefix}_turn_{int(span):02d}"
        for field in (
            "a",
            "b",
            "c",
            "derivative_zero_days",
            "axis_zero_days",
            "second_derivative",
        ):
            value = None if fit is None else fit.get(field)
            output[f"{stem}_{field}"] = None if value is None else float(value)
        kind = None if fit is None else str(fit.get("kind", "flat"))
        output[f"{stem}_kind"] = (
            None if kind is None else _TURN_KIND_VALUE.get(kind, 0.0)
        )

    has_fits = bool(fits_by_window)
    consensus_kind = str(consensus.get("kind", "unstable"))
    output[f"{prefix}_turn_consensus_stable"] = (
        None if not has_fits else float(bool(consensus.get("stable")))
    )
    output[f"{prefix}_turn_consensus_kind"] = (
        None if not has_fits else _TURN_KIND_VALUE.get(consensus_kind, 0.0)
    )
    days_ahead = consensus.get("days_ahead")
    output[f"{prefix}_turn_consensus_derivative_zero_days"] = (
        None if days_ahead is None else float(days_ahead)
    )
    dispersion = consensus.get("dispersion")
    output[f"{prefix}_turn_consensus_dispersion"] = (
        None if dispersion is None else float(dispersion)
    )
    return output


def _cross_duration(dif: Sequence[float], dea: Sequence[float]) -> tuple[int, int]:
    if not dif or not dea:
        return 0, 0
    state = 1 if dif[-1] >= dea[-1] else -1
    duration = 0
    for left, right in zip(reversed(dif), reversed(dea)):
        if (1 if left >= right else -1) != state:
            break
        duration += 1
    return state, duration


def _divergence(price: Sequence[float], oscillator: Sequence[float], span: int) -> float | None:
    price_slope = _linear_slope(price, span, price[-1]) if price else None
    osc_slope = _linear_slope(oscillator, span, price[-1]) if price else None
    if price_slope is None or osc_slope is None:
        return None
    # Positive: confirming slopes. Negative: price/MACD divergence.
    return math.copysign(min(abs(price_slope - osc_slope), 1.0), price_slope * osc_slope)


def _rsi(values: Sequence[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    changes = np.diff(np.asarray(values[-(period + 1) :], dtype=float))
    gain = float(np.mean(np.maximum(changes, 0.0)))
    loss = float(np.mean(np.maximum(-changes, 0.0)))
    if loss == 0.0:
        return 100.0 if gain else 50.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def _atr(bars: Sequence[PriceBar], period: int = 14) -> float | None:
    if len(bars) <= period:
        return None
    ranges: list[float] = []
    for previous, current in zip(bars[-(period + 1) : -1], bars[-period:]):
        ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return _mean(ranges)


def _max_drawdown(values: Sequence[float]) -> float | None:
    if not values:
        return None
    peak = float(values[0])
    worst = 0.0
    for value in values:
        peak = max(peak, float(value))
        if peak:
            worst = min(worst, float(value) / peak - 1.0)
    return worst


def _window_return(values: Sequence[float], span: int) -> float | None:
    return None if len(values) <= span else _safe_return(values[-1], values[-span - 1])


def _volatility(values: Sequence[float], span: int, periods: int) -> float | None:
    if len(values) <= span:
        return None
    returns = [_safe_return(right, left) for left, right in zip(values[-(span + 1) : -1], values[-span:])]
    result = _std(returns)
    return None if result is None else result * math.sqrt(periods)


def _change(values: Sequence[float], span: int) -> float | None:
    return None if len(values) <= span else _safe_return(values[-1], values[-span - 1])


def _zscore(values: Sequence[float], span: int) -> float | None:
    if len(values) < span:
        return None
    sample = [float(value) for value in values[-span:]]
    deviation = _std(sample)
    average = _mean(sample)
    return None if deviation in (None, 0.0) or average is None else (sample[-1] - average) / deviation


def _obv(closes: Sequence[float], volumes: Sequence[float | None]) -> list[float]:
    output = [0.0]
    for previous, current, volume in zip(closes[:-1], closes[1:], volumes[1:]):
        signed = 0.0 if volume is None else float(volume) * (1.0 if current > previous else -1.0 if current < previous else 0.0)
        output.append(output[-1] + signed)
    return output


def _correlation(left: Sequence[float | None], right: Sequence[float | None], span: int) -> float | None:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left[-span:], right[-span:])
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < span:
        return None
    x, y = np.asarray(pairs, dtype=float).T
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _aggregate_weekly(bars: Sequence[PriceBar]) -> list[PriceBar]:
    grouped: dict[tuple[int, int], list[PriceBar]] = defaultdict(list)
    for bar in bars:
        year, week, _ = bar.trade_date.isocalendar()
        grouped[(year, week)].append(bar)
    weekly: list[PriceBar] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: row.trade_date)
        volumes = [row.volume for row in rows]
        turnovers = [row.turnover for row in rows]
        weekly.append(
            PriceBar(
                trade_date=rows[-1].trade_date,
                open=rows[0].open,
                high=max(row.high for row in rows),
                low=min(row.low for row in rows),
                close=rows[-1].close,
                volume=None if any(value is None for value in volumes) else sum(float(value) for value in volumes if value is not None),
                turnover=None if any(value is None for value in turnovers) else sum(float(value) for value in turnovers if value is not None),
                source=rows[-1].source,
            )
        )
    return sorted(weekly, key=lambda row: row.trade_date)


def _latest_observation(
    observations: Sequence[PointInTimeObservation], series: str
) -> PointInTimeObservation | None:
    matching = [row for row in observations if row.series == series]
    return max(matching, key=lambda row: (row.effective_date, row.available_at)) if matching else None


def _observation_changes(
    observations: Sequence[PointInTimeObservation], series: str, spans: Sequence[int]
) -> dict[str, float | None]:
    rows = sorted((row for row in observations if row.series == series), key=lambda row: row.effective_date)
    values = [row.value for row in rows]
    output: dict[str, float | None] = {series: None if not values else values[-1]}
    for span in spans:
        prior = None
        if rows:
            threshold = rows[-1].effective_date - timedelta(weeks=span)
            eligible = [row for row in rows[:-1] if row.effective_date <= threshold]
            prior = eligible[-1].value if eligible else None
        output[f"{series}_change_{span}"] = _safe_return(values[-1], prior) if values else None
    if len(values) >= 3:
        last_change = _safe_return(values[-1], values[-2])
        previous_change = _safe_return(values[-2], values[-3])
        output[f"{series}_acceleration"] = (
            None if last_change is None or previous_change is None else last_change - previous_change
        )
    else:
        output[f"{series}_acceleration"] = None
    return output


class V33FeatureService:
    """Construct weekly, full-100-session daily and point-in-time features."""

    def __init__(self, *, require_weekly_bars: int = 20, daily_window: int = DAILY_WINDOW) -> None:
        self.require_weekly_bars = require_weekly_bars
        self.daily_window = daily_window

    @staticmethod
    def validate_market(market: str) -> None:
        if market not in SUPPORTED_MARKETS:
            raise V33FeatureError("V3.3 supports only 399006 and 159941")

    def load_snapshot(
        self,
        session: Session,
        market: str,
        cutoff_date: date,
        *,
        observations: Iterable[PointInTimeObservation] = (),
        weekly_cutoff_date: date | None = None,
    ) -> FeatureSnapshot:
        """Load legacy market rows without binding to the forthcoming V3.3 tables."""

        self.validate_market(market)
        instrument = session.scalar(select(Instrument).where(Instrument.code == market))
        if instrument is None:
            raise V33FeatureError(f"Missing instrument {market}")
        rows = session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= cutoff_date,
            )
            .order_by(MarketPrice.trade_date)
        ).all()
        bars = [self._row_to_bar(row) for row in rows]

        benchmark_bars: list[PriceBar] = []
        if market == "159941":
            benchmark = session.scalar(select(Instrument).where(Instrument.code == "NDX"))
            if benchmark is not None:
                benchmark_rows = session.scalars(
                    select(MarketPrice)
                    .where(
                        MarketPrice.instrument_id == benchmark.id,
                        MarketPrice.timeframe == "daily",
                        MarketPrice.trade_date <= cutoff_date,
                    )
                    .order_by(MarketPrice.trade_date)
                ).all()
                benchmark_bars = [self._row_to_bar(row) for row in benchmark_rows]

        valuation_rows = session.scalars(
            select(ValuationRecord)
            .where(
                ValuationRecord.instrument_id == instrument.id,
                ValuationRecord.valuation_date <= cutoff_date,
            )
            .order_by(ValuationRecord.valuation_date)
        ).all()
        combined = list(observations)
        for row in valuation_rows:
            raw = dict(row.raw_values or {})
            published_at = self._parse_datetime(raw.get("published_at"), row.valuation_date)
            available_at = self._parse_datetime(raw.get("available_at"), row.valuation_date)
            # Legacy valuation rows without point-in-time timestamps are usable
            # only from their valuation date.  They are explicitly labelled.
            for series, value in (("pe", row.pe_ratio), ("pb", row.pb_ratio)):
                number = _number(value)
                if number is not None:
                    combined.append(
                        PointInTimeObservation(
                            market=market,
                            series=series,
                            value=number,
                            effective_date=row.valuation_date,
                            published_at=published_at,
                            available_at=available_at,
                            source=str(raw.get("source") or "legacy_valuation_records"),
                            vintage=None if raw.get("vintage") is None else str(raw["vintage"]),
                            quality_status=str(raw.get("quality_status") or "legacy_date_only"),
                        )
                    )
        return self.build_snapshot(
            market,
            bars,
            cutoff_date,
            observations=combined,
            benchmark_bars=benchmark_bars,
            weekly_cutoff_date=weekly_cutoff_date,
        )

    def load_repository_snapshot(
        self,
        repository: Any,
        market: str,
        cutoff_at: datetime,
        *,
        weekly_cutoff_date: date | None = None,
        series_codes: Sequence[str] = DEFAULT_PIT_SERIES_CODES,
    ) -> FeatureSnapshot:
        """Duck-typed adapter for :class:`V33DataRepository`.

        This keeps the model core import-independent from the storage service
        while using its exact ``available_at`` filtered reads in production.
        """

        self.validate_market(market)
        target_rows = repository.market_bars_as_of(market, cutoff_at, timeframe="daily")
        benchmark_rows = (
            repository.market_bars_as_of("NDX", cutoff_at, timeframe="daily")
            if market == "159941"
            else []
        )
        observation_rows = repository.observations_as_of(tuple(series_codes), cutoff_at)
        observations = [
            self._pit_row_to_observation(row, market)
            for row in observation_rows
            if _number(getattr(row, "numeric_value", None)) is not None
        ]
        return self.build_snapshot(
            market,
            [self._row_to_bar(row) for row in target_rows],
            cutoff_at.date(),
            observations=observations,
            benchmark_bars=[self._row_to_bar(row) for row in benchmark_rows],
            weekly_cutoff_date=weekly_cutoff_date,
            cutoff_at=cutoff_at,
        )

    def build_snapshot(
        self,
        market: str,
        bars: Sequence[PriceBar],
        cutoff_date: date,
        *,
        observations: Iterable[PointInTimeObservation] = (),
        benchmark_bars: Sequence[PriceBar] = (),
        weekly_cutoff_date: date | None = None,
        cutoff_at: datetime | None = None,
    ) -> FeatureSnapshot:
        self.validate_market(market)
        ordered = sorted(bars, key=lambda row: row.trade_date)
        if any(bar.trade_date > cutoff_date for bar in ordered):
            raise FutureLeakageError("price source_data_max_date exceeds cutoff")
        if any(left.trade_date >= right.trade_date for left, right in zip(ordered, ordered[1:])):
            raise V33FeatureError("daily bars must have unique strictly increasing dates")
        if len(ordered) < self.daily_window:
            raise V33FeatureError(
                f"{market} requires {self.daily_window} visible daily sessions, got {len(ordered)}"
            )
        if any(not _valid_ohlc(bar) for bar in ordered):
            raise V33FeatureError("invalid OHLC data")

        weekly_cutoff = cutoff_date if weekly_cutoff_date is None else weekly_cutoff_date
        if weekly_cutoff > cutoff_date:
            raise FutureLeakageError("weekly cutoff exceeds daily/cutoff date")
        weekly_bars = _aggregate_weekly(
            [bar for bar in ordered if bar.trade_date <= weekly_cutoff]
        )
        if len(weekly_bars) < self.require_weekly_bars:
            raise V33FeatureError(
                f"{market} requires {self.require_weekly_bars} weekly bars, got {len(weekly_bars)}"
            )

        effective_cutoff_at = (
            datetime.combine(cutoff_date, time.max, tzinfo=timezone.utc)
            if cutoff_at is None
            else cutoff_at
        )
        if effective_cutoff_at.tzinfo is None:
            effective_cutoff_at = effective_cutoff_at.replace(tzinfo=timezone.utc)
        if effective_cutoff_at.date() < cutoff_date:
            raise FutureLeakageError("cutoff_at precedes cutoff_date")
        visible_observations: list[PointInTimeObservation] = []
        excluded_future = 0
        for observation in observations:
            if observation.market not in (market, "GLOBAL", BENCHMARK_BY_MARKET[market]):
                continue
            available = observation.available_at
            if available.tzinfo is None:
                available = available.replace(tzinfo=timezone.utc)
            if available > effective_cutoff_at or observation.effective_date > cutoff_date:
                excluded_future += 1
                continue
            visible_observations.append(observation)
        latest_vintages: dict[tuple[str, date], PointInTimeObservation] = {}
        for observation in visible_observations:
            key = (observation.series, observation.effective_date)
            previous = latest_vintages.get(key)
            if previous is None or observation.available_at >= previous.available_at:
                latest_vintages[key] = observation
        visible_observations = sorted(
            latest_vintages.values(),
            key=lambda row: (row.series, row.effective_date, row.available_at),
        )

        ordered_benchmark = sorted(
            (bar for bar in benchmark_bars if bar.trade_date <= cutoff_date),
            key=lambda row: row.trade_date,
        )
        if any(bar.trade_date > cutoff_date for bar in ordered_benchmark):
            raise FutureLeakageError("benchmark source_data_max_date exceeds cutoff")

        daily_features, daily_sequence, daily_turn = self._daily_features(ordered)
        weekly_features, weekly_turn = self._weekly_features(weekly_bars)
        daily_features.update(
            _turn_model_features("daily_dif", daily_turn, (9, 13, 21))
        )
        weekly_features.update(
            _turn_model_features("weekly_dif", weekly_turn, (5, 9, 13))
        )
        fundamental_features, missing_masks, obs_provenance = self._observation_features(
            market, visible_observations
        )
        qdii_features, qdii_masks = self._qdii_features(
            market,
            ordered,
            ordered_benchmark,
            visible_observations,
        )
        weekly_features.update(fundamental_features)
        weekly_features.update(qdii_features)
        daily_features.update({key: value for key, value in qdii_features.items() if key.startswith("qdii_")})
        missing_masks.update(qdii_masks)

        for name, value in {**weekly_features, **daily_features}.items():
            # A missingness indicator is a separate model input.  Reusing the
            # feature name here would make ``flatten_snapshot`` overwrite the
            # real DIF/DEA/PE/etc. value with 0 or 1.
            missing_masks.setdefault(f"missing_{name}", int(value is None))

        derivative_turn = {
            "daily_dif": daily_turn,
            "weekly_dif": weekly_turn,
            "semantics": {
                "derivative_zero": "d(DIF)/dt = 0, local top/bottom candidate",
                "axis_zero": "DIF = 0, zero-axis crossing candidate",
            },
        }
        return FeatureSnapshot(
            market=market,
            cutoff_date=cutoff_date,
            source_data_max_date=ordered[-1].trade_date,
            daily_as_of=ordered[-1].trade_date,
            weekly_as_of=weekly_bars[-1].trade_date,
            weekly=weekly_features,
            daily=daily_features,
            daily_sequence=daily_sequence,
            missing_masks=missing_masks,
            provenance={
                "target": market,
                "benchmark": BENCHMARK_BY_MARKET[market],
                "target_price_sources": sorted({bar.source or "UNKNOWN" for bar in ordered}),
                "observations": obs_provenance,
                "excluded_future_observations": excluded_future,
                "daily_session_count": len(ordered),
                "weekly_bar_count": len(weekly_bars),
                "weekly_cutoff_date": weekly_cutoff.isoformat(),
                "daily_window_used": self.daily_window,
                **(
                    {"qdii_nav_premium_price_basis": "unadjusted_exchange_close"}
                    if market == "159941"
                    else {}
                ),
            },
            derivative_turn=derivative_turn,
            benchmark_market=BENCHMARK_BY_MARKET[market],
        )

    def _daily_features(
        self, bars: Sequence[PriceBar]
    ) -> tuple[dict[str, float | None], tuple[Mapping[str, float | str | None], ...], dict[str, Any]]:
        closes = [bar.close for bar in bars]
        opens = [bar.open for bar in bars]
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]
        volumes = [bar.volume for bar in bars]
        dif, dea, histogram = macd(closes)
        spread = dif - dea
        scale = closes[-1]
        cross_state, cross_duration = _cross_duration(dif.tolist(), dea.tolist())
        price_returns = [_safe_return(right, left) for left, right in zip(closes[:-1], closes[1:])]
        volume_changes = [_safe_return(right, left) for left, right in zip(volumes[:-1], volumes[1:])]
        obv = _obv(closes, volumes)
        latest = bars[-1]
        full_range = max(latest.high - latest.low, 1e-12)
        features: dict[str, float | None] = {
            "daily_dif": float(dif[-1] / scale),
            "daily_dea": float(dea[-1] / scale),
            "daily_macd_histogram": float(histogram[-1] / scale),
            "daily_dif_dea_relative_gap": float((dif[-1] - dea[-1]) / (abs(dif[-1]) + abs(dea[-1]) + 1e-12)),
            "daily_cross_state": float(cross_state),
            "daily_cross_duration": float(cross_duration),
            "daily_zero_axis_state": float(1 if dif[-1] >= 0.0 else -1),
            "daily_spread_speed_1": _linear_slope(spread.tolist(), 2, scale),
            "daily_spread_speed_3": _linear_slope(spread.tolist(), 3, scale),
            "daily_spread_speed_5": _linear_slope(spread.tolist(), 5, scale),
            "daily_histogram_speed_3": _linear_slope(histogram.tolist(), 3, scale),
            "daily_histogram_acceleration_5": _curvature(histogram.tolist(), 5, scale),
            "daily_return_1": _window_return(closes, 1),
            "daily_return_5": _window_return(closes, 5),
            "daily_return_20": _window_return(closes, 20),
            "daily_return_60": _window_return(closes, 60),
            "daily_volatility_20": _volatility(closes, 20, 252),
            "daily_volatility_60": _volatility(closes, 60, 252),
            "daily_rsi14": _rsi(closes),
            "daily_atr14_ratio": None if (atr := _atr(bars)) is None else atr / scale,
            "daily_ma5_distance": _safe_return(scale, _mean(closes[-5:])),
            "daily_ma10_distance": _safe_return(scale, _mean(closes[-10:])),
            "daily_ma20_distance": _safe_return(scale, _mean(closes[-20:])),
            "daily_ma60_distance": _safe_return(scale, _mean(closes[-60:])),
            "daily_body_ratio": (latest.close - latest.open) / full_range,
            "daily_upper_shadow_ratio": (latest.high - max(latest.open, latest.close)) / full_range,
            "daily_lower_shadow_ratio": (min(latest.open, latest.close) - latest.low) / full_range,
            "daily_close_location": (latest.close - latest.low) / full_range,
            "daily_gap": _safe_return(latest.open, bars[-2].close),
            "daily_volume_change_1": _change([float(value) for value in volumes if value is not None], 1) if all(value is not None for value in volumes[-2:]) else None,
            "daily_volume_change_5": _change([float(value) for value in volumes if value is not None], 5) if all(value is not None for value in volumes[-6:]) else None,
            "daily_volume_change_20": _change([float(value) for value in volumes if value is not None], 20) if all(value is not None for value in volumes[-21:]) else None,
            "daily_volume_zscore_20": _zscore([float(value) for value in volumes if value is not None], 20) if all(value is not None for value in volumes[-20:]) else None,
            "daily_obv_slope_20": _linear_slope(obv, 20, max(abs(obv[-1]), 1.0)),
            "daily_price_volume_correlation_20": _correlation(price_returns, volume_changes, 20),
        }
        for span in (1, 3, 5, 10):
            features[f"daily_dif_slope_{span}"] = _linear_slope(dif.tolist(), span, scale)
            features[f"daily_dea_slope_{span}"] = _linear_slope(dea.tolist(), span, scale)
        for span in (3, 5, 10):
            features[f"daily_dif_curvature_{span}"] = _curvature(dif.tolist(), span, scale)
            features[f"daily_dea_curvature_{span}"] = _curvature(dea.tolist(), span, scale)
        for span in (20, 40, 60):
            features[f"daily_price_dif_divergence_{span}"] = _divergence(closes, dif.tolist(), span)
            features[f"daily_price_macd_divergence_{span}"] = _divergence(closes, histogram.tolist(), span)

        sequence: list[Mapping[str, float | str | None]] = []
        start = len(bars) - self.daily_window
        for index in range(start, len(bars)):
            bar = bars[index]
            day_range = max(bar.high - bar.low, 1e-12)
            sequence.append(
                {
                    "trade_date": bar.trade_date.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "turnover": bar.turnover,
                    "dif": float(dif[index]),
                    "dea": float(dea[index]),
                    "macd_histogram": float(histogram[index]),
                    "dif_slope_1": None if index == 0 else float((dif[index] - dif[index - 1]) / max(abs(bar.close), 1e-12)),
                    "dea_slope_1": None if index == 0 else float((dea[index] - dea[index - 1]) / max(abs(bar.close), 1e-12)),
                    "body_ratio": (bar.close - bar.open) / day_range,
                    "upper_shadow_ratio": (bar.high - max(bar.open, bar.close)) / day_range,
                    "lower_shadow_ratio": (min(bar.open, bar.close) - bar.low) / day_range,
                }
            )
        return features, tuple(sequence), _turn_consensus(dif.tolist(), (9, 13, 21), scale)

    def _weekly_features(
        self, bars: Sequence[PriceBar]
    ) -> tuple[dict[str, float | None], dict[str, Any]]:
        closes = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]
        dif, dea, histogram = macd(closes)
        spread = dif - dea
        scale = closes[-1]
        cross_state, cross_duration = _cross_duration(dif.tolist(), dea.tolist())
        price_returns = [_safe_return(right, left) for left, right in zip(closes[:-1], closes[1:])]
        volume_changes = [_safe_return(right, left) for left, right in zip(volumes[:-1], volumes[1:])]
        obv = _obv(closes, volumes)
        latest = bars[-1]
        week_range = max(latest.high - latest.low, 1e-12)
        macd_stable = len(bars) >= WEEKLY_MACD_STABLE_BARS
        features: dict[str, float | None] = {
            "weekly_dif": float(dif[-1] / scale) if macd_stable else None,
            "weekly_dea": float(dea[-1] / scale) if macd_stable else None,
            "weekly_macd_histogram": float(histogram[-1] / scale) if macd_stable else None,
            "weekly_dif_dea_relative_gap": (
                float((dif[-1] - dea[-1]) / (abs(dif[-1]) + abs(dea[-1]) + 1e-12))
                if macd_stable
                else None
            ),
            "weekly_cross_state": float(cross_state) if macd_stable else None,
            "weekly_cross_duration": float(cross_duration) if macd_stable else None,
            "weekly_zero_axis_state": float(1 if dif[-1] >= 0 else -1) if macd_stable else None,
            "weekly_spread_speed_1": _linear_slope(spread.tolist(), 2, scale) if macd_stable else None,
            "weekly_spread_speed_3": _linear_slope(spread.tolist(), 3, scale) if macd_stable else None,
            "weekly_spread_speed_5": _linear_slope(spread.tolist(), 5, scale) if macd_stable else None,
            "weekly_histogram_speed_3": _linear_slope(histogram.tolist(), 3, scale) if macd_stable else None,
            "weekly_histogram_acceleration_5": _curvature(histogram.tolist(), 5, scale) if macd_stable else None,
            "weekly_atr14_ratio": None if (atr := _atr(bars)) is None else atr / scale,
            "weekly_drawdown_60": (
                _max_drawdown(closes[-60:]) if len(closes) >= 60 else None
            ),
            "weekly_body_ratio": (latest.close - latest.open) / week_range,
            "weekly_upper_shadow_ratio": (latest.high - max(latest.open, latest.close)) / week_range,
            "weekly_lower_shadow_ratio": (min(latest.open, latest.close) - latest.low) / week_range,
            "weekly_close_location": (latest.close - latest.low) / week_range,
            "weekly_gap": _safe_return(latest.open, bars[-2].close),
            "weekly_volume_change_1": _change([float(value) for value in volumes if value is not None], 1) if all(value is not None for value in volumes[-2:]) else None,
            "weekly_volume_change_4": _change([float(value) for value in volumes if value is not None], 4) if all(value is not None for value in volumes[-5:]) else None,
            "weekly_volume_change_13": _change([float(value) for value in volumes if value is not None], 13) if all(value is not None for value in volumes[-14:]) else None,
            "weekly_volume_zscore_20": _zscore([float(value) for value in volumes if value is not None], 20) if all(value is not None for value in volumes[-20:]) else None,
            "weekly_volume_zscore_52": _zscore([float(value) for value in volumes if value is not None], 52) if all(value is not None for value in volumes[-52:]) else None,
            "weekly_obv_slope_13": _linear_slope(obv, 13, max(abs(obv[-1]), 1.0)),
            "weekly_price_volume_correlation_13": _correlation(price_returns, volume_changes, 13),
        }
        for span in (1, 2, 4, 8, 13, 20):
            features[f"weekly_return_{span}"] = _window_return(closes, span)
        for span in (4, 13, 20):
            features[f"weekly_volatility_{span}"] = _volatility(closes, span, 52)
        for span in (5, 10, 20, 40):
            features[f"weekly_ma{span}_distance"] = (
                _safe_return(scale, _mean(closes[-span:]))
                if len(closes) >= span
                else None
            )
        for span in (1, 3, 5, 8):
            features[f"weekly_dif_slope_{span}"] = (
                _linear_slope(dif.tolist(), max(2, span), scale) if macd_stable else None
            )
            features[f"weekly_dea_slope_{span}"] = (
                _linear_slope(dea.tolist(), max(2, span), scale) if macd_stable else None
            )
        for span in (3, 5, 8):
            features[f"weekly_dif_curvature_{span}"] = (
                _curvature(dif.tolist(), span, scale) if macd_stable else None
            )
            features[f"weekly_dea_curvature_{span}"] = (
                _curvature(dea.tolist(), span, scale) if macd_stable else None
            )
        for span in (4, 8, 13, 20):
            features[f"weekly_price_dif_divergence_{span}"] = (
                _divergence(closes, dif.tolist(), span) if macd_stable else None
            )
            features[f"weekly_price_macd_divergence_{span}"] = (
                _divergence(closes, histogram.tolist(), span) if macd_stable else None
            )
        return features, _turn_consensus(
            dif.tolist() if macd_stable else (), (5, 9, 13), scale
        )

    @staticmethod
    def _observation_features(
        market: str, observations: Sequence[PointInTimeObservation]
    ) -> tuple[dict[str, float | None], dict[str, int], dict[str, Any]]:
        features: dict[str, float | None] = {}
        masks: dict[str, int] = {}
        provenance: dict[str, Any] = {}
        for family, series_names in OBSERVATION_FAMILIES.items():
            if family == "qdii":
                continue
            family_present = False
            for series in series_names:
                latest = _latest_observation(observations, series)
                if series in ("pe", "pb"):
                    changes = _observation_changes(observations, series, (4, 13, 26))
                    features.update(changes)
                    if latest is not None and latest.value > 0:
                        features[f"{series}_log"] = math.log(latest.value)
                        history = sorted(
                            row.value for row in observations if row.series == series and row.value > 0
                        )
                        rank = sum(value <= latest.value for value in history)
                        features[f"{series}_expanding_percentile"] = 100.0 * rank / len(history)
                    else:
                        features[f"{series}_log"] = None
                        features[f"{series}_expanding_percentile"] = None
                else:
                    features.update(_observation_changes(observations, series, (1, 4, 13)))
                masks[f"missing_{series}"] = 1 if latest is None else 0
                family_present = family_present or latest is not None
                if latest is not None:
                    provenance[series] = {
                        "source": latest.source,
                        "effective_date": latest.effective_date.isoformat(),
                        "available_at": latest.available_at.isoformat(),
                        "vintage": latest.vintage,
                        "quality_status": latest.quality_status,
                    }
            masks[f"missing_family_{family}"] = 0 if family_present else 1
        pe = features.get("pe")
        features["earnings_yield"] = None if pe in (None, 0.0) else 1.0 / float(pe)
        return features, masks, provenance

    @staticmethod
    def _qdii_features(
        market: str,
        target: Sequence[PriceBar],
        benchmark: Sequence[PriceBar],
        observations: Sequence[PointInTimeObservation],
    ) -> tuple[dict[str, float | None], dict[str, int]]:
        if market != "159941":
            return {}, {}
        features: dict[str, float | None] = {}
        masks: dict[str, int] = {}
        latest_by_series = {series: _latest_observation(observations, series) for series in OBSERVATION_FAMILIES["qdii"]}
        # Premium/discount is a same-day tradable-price comparison.  Using a
        # qfq close here changes the price unit and can manufacture a premium.
        # Technical returns and tracking error intentionally continue to use
        # ``PriceBar.close`` (the qfq series).
        target_close = (
            target[-1].raw_close
            if target[-1].raw_close is not None
            else target[-1].close
        )
        nav = latest_by_series["nav"]
        estimated_nav = latest_by_series["estimated_nav"]
        features["qdii_nav"] = None if nav is None else nav.value
        features["qdii_estimated_nav"] = None if estimated_nav is None else estimated_nav.value
        features["qdii_raw_close"] = target_close
        features["qdii_nav_premium"] = None if nav is None or nav.value == 0 else target_close / nav.value - 1.0
        features["qdii_estimated_nav_premium"] = (
            None if estimated_nav is None or estimated_nav.value == 0 else target_close / estimated_nav.value - 1.0
        )
        for series in ("fx_usdcny", "fund_shares", "aum", "subscriptions"):
            features.update(_observation_changes(observations, series, (1, 5, 20)))
        for series, row in latest_by_series.items():
            masks[f"missing_{series}"] = 1 if row is None else 0

        # A US close labelled D is not available when 159941 closes at 15:00
        # China time on D.  With legacy bars (no intraday available_at), use
        # only strictly earlier NDX sessions.  Point-in-time V3.3 observations
        # can later provide a more precise timestamp without weakening this
        # conservative boundary.
        usable_benchmark = [bar for bar in benchmark if bar.trade_date < target[-1].trade_date]
        benchmark_closes = [bar.close for bar in usable_benchmark]
        for span in (5, 20, 60):
            features[f"qdii_ndx_return_{span}"] = _window_return(benchmark_closes, span)
        target_returns: list[float | None] = []
        benchmark_returns: list[float | None] = []
        benchmark_index = 0
        previous_benchmark_close: float | None = None
        previous_target_close: float | None = None
        for target_bar in target:
            while (
                benchmark_index < len(usable_benchmark)
                and usable_benchmark[benchmark_index].trade_date < target_bar.trade_date
            ):
                previous_benchmark_close = (
                    usable_benchmark[benchmark_index - 1].close
                    if benchmark_index > 0
                    else previous_benchmark_close
                )
                benchmark_index += 1
            if benchmark_index == 0:
                previous_target_close = target_bar.close
                continue
            current_benchmark = usable_benchmark[benchmark_index - 1].close
            if previous_target_close is not None and previous_benchmark_close is not None:
                target_returns.append(_safe_return(target_bar.close, previous_target_close))
                benchmark_returns.append(_safe_return(current_benchmark, previous_benchmark_close))
            previous_target_close = target_bar.close
            previous_benchmark_close = current_benchmark
        differences = [
            float(left) - float(right)
            for left, right in zip(target_returns, benchmark_returns)
            if left is not None and right is not None
        ]
        for span in (20, 60, 120):
            deviation = _std(differences[-span:]) if len(differences) >= span else None
            features[f"qdii_tracking_error_{span}"] = None if deviation is None else deviation * math.sqrt(252)
        masks["missing_ndx_benchmark"] = 0 if usable_benchmark else 1
        masks["missing_qdii_tracking_error"] = 1 if features["qdii_tracking_error_20"] is None else 0
        return features, masks

    @staticmethod
    def _row_to_bar(row: MarketPrice) -> PriceBar:
        raw_close = _number(row.close_price)
        adjusted_close = _number(row.adjusted_close_price)
        if raw_close is None or raw_close <= 0.0:
            raise V33FeatureError("market price row has no close")
        # 159941 is persisted with a CNY qfq adjusted close.  Scaling only the
        # close would corrupt high/low validation and every candlestick ratio.
        # Apply the exact same factor to all OHLC fields so shapes are invariant.
        close = adjusted_close if adjusted_close is not None else raw_close
        factor = close / raw_close if raw_close else 1.0
        open_price = _number(row.open_price)
        high = _number(row.high_price)
        low = _number(row.low_price)
        raw_volume = _number(row.volume)
        volume_multiplier = float(getattr(row, "volume_multiplier", 1) or 1)
        return PriceBar(
            trade_date=row.trade_date,
            open=close if open_price is None else open_price * factor,
            high=close if high is None else high * factor,
            low=close if low is None else low * factor,
            close=close,
            raw_close=raw_close,
            volume=None if raw_volume is None else raw_volume * volume_multiplier,
            turnover=_number(row.turnover),
            source=row.source,
        )

    @staticmethod
    def _pit_row_to_observation(row: Any, market: str) -> PointInTimeObservation:
        value = _number(getattr(row, "numeric_value", None))
        if value is None:
            raise V33FeatureError("point-in-time observation has no numeric value")
        original_series = str(row.series_code)
        series = V33FeatureService._canonical_series(original_series)
        return PointInTimeObservation(
            market=market,
            series=series,
            value=value,
            effective_date=row.effective_date,
            published_at=row.published_at,
            available_at=row.available_at,
            source=row.source,
            unit=getattr(row, "unit", None),
            vintage=getattr(row, "vintage", None),
            quality_status=getattr(row, "quality_status", "verified"),
            metadata={
                "original_series_code": original_series,
                "dataset": getattr(row, "dataset", None),
                "raw_payload_hash": getattr(row, "raw_payload_hash", None),
            },
        )

    @staticmethod
    def _canonical_series(series_code: str) -> str:
        key = series_code.strip().lower().replace("-", "_").replace(".", "_").replace("/", "_")
        if key in DEFAULT_PIT_SERIES_CODES:
            return key
        rules = (
            ("estimated_nav", ("estimated_nav", "iopv", "estimate_nav")),
            ("fx_usdcny", ("usdcny", "usd_cny", "cnyusd")),
            ("fund_shares", ("fund_share", "shares_outstanding")),
            ("subscriptions", ("subscription", "redemption", "creation_units")),
            ("earnings_growth", ("earnings_growth", "profit_growth", "eps_growth")),
            ("term_spread", ("term_spread", "yield_curve")),
            ("real_rate", ("real_rate", "real_yield")),
            ("short_rate", ("dr007", "lpr", "fed_funds", "sofr", "short_rate")),
            ("long_rate", ("10y", "long_rate", "treasury_yield")),
            ("market_breadth", ("market_breadth", "advance_decline", "breadth")),
            ("fund_flow", ("northbound", "margin_balance", "fund_flow", "cftc")),
            ("liquidity", ("m2", "social_financing", "credit_impulse", "liquidity")),
            ("inflation", ("cpi", "ppi", "pce", "inflation")),
            ("pmi", ("pmi", "ism")),
            ("employment", ("employment", "unemployment", "payroll")),
            ("volatility_index", ("vix", "volatility_index")),
            ("aum", ("aum", "assets_under_management")),
            ("nav", ("nav", "unit_net_value")),
            ("pe", ("pe_ttm", "price_earnings", "_pe")),
            ("pb", ("price_book", "_pb")),
        )
        for canonical, markers in rules:
            if any(marker in key for marker in markers):
                return canonical
        return key

    @staticmethod
    def _parse_datetime(value: Any, fallback: date) -> datetime:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, str):
            try:
                result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                result = datetime.combine(fallback, time.max, tzinfo=timezone.utc)
        else:
            result = datetime.combine(fallback, time.max, tzinfo=timezone.utc)
        return result if result.tzinfo is not None else result.replace(tzinfo=timezone.utc)


def flatten_snapshot(snapshot: FeatureSnapshot) -> dict[str, float | None]:
    """Return deterministic scalar features including missing masks.

    The 100-session raw sequence is retained on :class:`FeatureSnapshot`; its
    curve descriptors are already represented by slopes, curvature,
    divergences and candlestick/volume features in this scalar projection.
    """

    output: dict[str, float | None] = {}
    output.update(snapshot.weekly)
    output.update(snapshot.daily)
    output.update({name: float(value) for name, value in snapshot.missing_masks.items()})
    # Preserve the *entire* 100-session curve for the model instead of reducing
    # it to the latest six indicators as V3.2 did.  Prices and oscillators are
    # scale-normalised so 399006 and 159941 remain numerically well conditioned;
    # the models themselves still remain strictly market-specific.
    latest_close = float(snapshot.daily_sequence[-1]["close"])
    visible_volumes = [
        float(row["volume"])
        for row in snapshot.daily_sequence
        if row.get("volume") is not None
    ]
    volume_scale = float(np.median(visible_volumes)) if visible_volumes else None
    sequence_fields = (
        "open",
        "high",
        "low",
        "close",
        "dif",
        "dea",
        "macd_histogram",
        "volume",
        "dif_slope_1",
        "dea_slope_1",
        "body_ratio",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
    )
    # A 12-term DCT projection consumes every one of the 100 sessions while
    # keeping progressive training computationally bounded.  Unlike a handful
    # of latest-value summaries, these orthogonal coefficients retain curve
    # level, trend and progressively finer shapes across the entire window.
    x = np.arange(len(snapshot.daily_sequence), dtype=float)
    for field_name in sequence_fields:
        normalizer = (
            volume_scale
            if field_name == "volume"
            else latest_close
            if field_name in {"open", "high", "low", "close", "dif", "dea", "macd_histogram"}
            else 1.0
        )
        raw_values = [row.get(field_name) for row in snapshot.daily_sequence]
        missing = np.asarray([value is None for value in raw_values], dtype=bool)
        clean = [float(value) for value in raw_values if value is not None]
        output[f"seq100_missing_fraction_{field_name}"] = float(np.mean(missing))
        if not clean or normalizer in (None, 0.0):
            for coefficient in range(12):
                output[f"seq100_dct_{field_name}_{coefficient:02d}"] = None
            continue
        fill = float(np.median(clean))
        values = np.asarray(
            [fill if value is None else float(value) for value in raw_values], dtype=float
        ) / max(abs(float(normalizer)), 1e-12)
        for coefficient in range(12):
            basis = np.cos(math.pi * (x + 0.5) * coefficient / len(values))
            output[f"seq100_dct_{field_name}_{coefficient:02d}"] = float(
                np.dot(values, basis) / len(values)
            )
    return dict(sorted(output.items()))
