"""Deterministic V3.1 weekly-path analysis with a 100-session daily corrector."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import hashlib
import math
import statistics
from threading import Lock, Thread
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.errors import PublicConflictError, PublicDataUnavailableError, PublicValidationError
from backend.app.models.models import (
    IndicatorRecord,
    Instrument,
    MarketPrice,
    V31AnalysisRun,
    V31DailyCorrection,
    V31ModelEvaluation,
    V31ModelVersion,
    V31WeeklyForecast,
    ValuationRecord,
    utc_now,
)
from backend.app.schemas.market import ProviderResult, ProviderStatus
from backend.app.schemas.valuation import ValuationProviderResult
from backend.app.services.indicator_service import IndicatorCalculationStatus, IndicatorService
from backend.app.services.investment_calendar_service import InvestmentCalendarService
from backend.app.services.market_calendar import CalendarProvider, ExchangeCalendarProvider
from backend.app.services.market_data import MarketDataService
from backend.app.services.providers import AkShareIndexProvider, AkShareValuationProvider, YahooDirectIndexProvider
from backend.app.services.valuation_service import ValuationService


SUPPORTED = {"399006", "NDX"}
FEATURE_VERSION = "V3.1-WEEKLY-DAILY100-1"
METHODOLOGY_VERSION = "V3.1-WALK-FORWARD-PATH-1"
STAGES = (
    "等待执行",
    "刷新行情",
    "校验数据",
    "计算周K信号",
    "计算日K修正",
    "生成概率路径",
    "读取当前仓位",
    "生成投资建议",
    "已完成",
)


class DataGateBlocked(RuntimeError):
    def __init__(self, code: str, message: str, *, source: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.source = source


def _call_with_timeout(
    call: Callable[[], Any],
    timeout_seconds: float,
    timeout_value: Callable[[], Any],
) -> Any:
    """Bound a blocking public-data adapter without risking process shutdown."""

    values: list[Any] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            values.append(call())
        except BaseException as error:  # re-raised in the request thread
            errors.append(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        return timeout_value()
    if errors:
        raise errors[0]
    return values[0]


class _TimedMarketProvider:
    def __init__(self, delegate: Any, timeout_seconds: float) -> None:
        self.delegate = delegate
        self.timeout_seconds = timeout_seconds
        self.source = delegate.source

    def fetch(
        self,
        instrument_code: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> ProviderResult:
        return _call_with_timeout(
            lambda: self.delegate.fetch(instrument_code, start_date, end_date),
            self.timeout_seconds,
            lambda: ProviderResult.failed(
                self.source,
                f"Public index request timed out after {self.timeout_seconds:g}s",
                volume_availability=(
                    "not_available_for_direct_index"
                    if instrument_code == "NDX"
                    else "available"
                ),
            ),
        )


class _TimedValuationProvider:
    def __init__(self, delegate: Any, timeout_seconds: float) -> None:
        self.delegate = delegate
        self.timeout_seconds = timeout_seconds
        self.source = delegate.source

    def fetch(self, instrument_code: str) -> ValuationProviderResult:
        return _call_with_timeout(
            lambda: self.delegate.fetch(instrument_code),
            self.timeout_seconds,
            lambda: ValuationProviderResult.failed(
                self.source,
                f"Public valuation request timed out after {self.timeout_seconds:g}s",
            ),
        )


@dataclass(frozen=True)
class GateSnapshot:
    symbol: str
    latest_available: date
    price_as_of: date
    valuation_as_of: date | None
    complete_week_as_of: date
    incomplete_week_as_of: date | None
    volume_as_of: date | None
    source: str
    volume_source: str
    degraded: tuple[str, ...]

    @property
    def model_cutoff(self) -> date:
        # Valuation is a soft input. A stale observation reduces confidence but
        # must not silently roll the price model back by years.
        return self.complete_week_as_of


def _f(value: Decimal | float | int) -> float:
    return float(value)


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _snap5(value: float) -> int:
    return int(_clamp(round(value / 5) * 5, 0, 100))


def _ema(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    output = [values[0]]
    for value in values[1:]:
        output.append(alpha * value + (1 - alpha) * output[-1])
    return output


def _macd(values: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
    fast, slow = _ema(values, 12), _ema(values, 26)
    dif = [left - right for left, right in zip(fast, slow)]
    dea = _ema(dif, 9)
    return dif, dea, [2 * (left - right) for left, right in zip(dif, dea)]


def _returns(values: Sequence[float]) -> list[float]:
    return [values[index] / values[index - 1] - 1 for index in range(1, len(values))]


def _window_return(values: Sequence[float], periods: int) -> float:
    return 0.0 if len(values) <= periods else values[-1] / values[-1 - periods] - 1


def _volatility(values: Sequence[float], periods: int, annualizer: float) -> float:
    window = _returns(values[-(periods + 1):])
    return 0.0 if len(window) < 2 else statistics.pstdev(window) * math.sqrt(annualizer)


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def _rsi(values: Sequence[float], period: int = 14) -> float:
    changes = [values[index] - values[index - 1] for index in range(1, len(values))]
    if len(changes) < period:
        return 50.0
    sample = changes[-period:]
    gain = sum(max(change, 0) for change in sample) / period
    loss = sum(max(-change, 0) for change in sample) / period
    if loss == 0:
        return 100.0
    return 100 - 100 / (1 + gain / loss)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] * (right - position) + ordered[right] * (position - left)


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class V31AnalysisService:
    """Local, auditable analysis pipeline. It never calls an AI/LLM service."""

    def __init__(
        self,
        sessions: Callable[[], Session],
        *,
        market: MarketDataService,
        valuations: ValuationService,
        indicators: IndicatorService,
        calendar: CalendarProvider | None = None,
        investment_calendar: InvestmentCalendarService,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.sessions = sessions
        self.market = market
        self.valuations = valuations
        self.indicators = indicators
        self.calendar = calendar or ExchangeCalendarProvider()
        self.investment_calendar = investment_calendar
        self.now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._locks = {symbol: Lock() for symbol in SUPPORTED}

    def run(self, symbol: str, *, refresh: bool = True) -> dict[str, Any]:
        self._validate_symbol(symbol)
        if not self._locks[symbol].acquire(blocking=False):
            raise PublicConflictError(f"{symbol} 的V3.1分析正在执行，请勿重复点击")
        run_id = f"V31-{symbol}-{uuid4().hex[:16]}"
        self._create_run(run_id, symbol)
        context: dict[str, Any] = {}
        try:
            self._stage(run_id, "刷新行情", lambda: context.update(self._refresh(symbol, refresh)))
            gate = self._stage(run_id, "校验数据", lambda: self._data_gate(symbol, context))
            context["gate"] = gate
            models = self._ensure_models(symbol, gate.complete_week_as_of)
            weekly = self._stage(run_id, "计算周K信号", lambda: self._weekly_signal(symbol, gate, models["weekly"]))
            daily = self._stage(run_id, "计算日K修正", lambda: self._daily_correction(symbol, gate, weekly))
            path = self._stage(run_id, "生成概率路径", lambda: self._probability_path(symbol, gate, weekly, models["weekly"]))
            position = self._stage(run_id, "读取当前仓位", lambda: self._position(symbol))
            advice = self._stage(run_id, "生成投资建议", lambda: self._advice(symbol, gate, weekly, daily, path, position))
            snapshot_id = self._snapshot_id(symbol, gate, models)
            forecast_id = self._persist_forecast(symbol, gate, models, weekly, daily, path, snapshot_id)
            evaluations = self._evaluate_mature(symbol)
            result = {
                "run_id": run_id,
                "market": symbol,
                "model": models,
                "freshness": self._gate_payload(gate),
                "data_gate": {
                    "status": "degraded" if gate.degraded else "passed",
                    "hard_blocked": False,
                    "degraded_reasons": list(gate.degraded),
                    "confidence_deduction": min(20, len(gate.degraded) * 5),
                },
                "weekly": weekly,
                "daily": daily,
                "path": path,
                "position": position,
                "advice": advice,
                "evaluation": evaluations,
                "forecast_id": forecast_id,
                "data_snapshot_id": snapshot_id,
                "generated_at": utc_now().isoformat(),
            }
            self._complete_run(run_id, result, gate, models["weekly"]["version"])
            result["stages"] = self.get_run(run_id)["stages"]
            return result
        except Exception as error:
            self._fail_run(run_id, error, context.get("gate"), context)
            return self.get_run(run_id)
        finally:
            self._locks[symbol].release()

    def latest(self, symbol: str) -> dict[str, Any] | None:
        self._validate_symbol(symbol)
        with self.sessions() as session:
            row = session.scalar(
                select(V31AnalysisRun)
                .where(V31AnalysisRun.market == symbol)
                .order_by(V31AnalysisRun.started_at.desc(), V31AnalysisRun.id.desc())
            )
            return None if row is None else self._run_payload(row)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            row = session.get(V31AnalysisRun, run_id)
            if row is None:
                raise PublicDataUnavailableError("V3.1分析运行不存在")
            return self._run_payload(row)

    @staticmethod
    def _validate_symbol(symbol: str) -> None:
        if symbol not in SUPPORTED:
            raise PublicValidationError("V3.1仅支持399006和NDX")

    def _create_run(self, run_id: str, symbol: str) -> None:
        with self.sessions() as session, session.begin():
            session.add(V31AnalysisRun(
                id=run_id,
                status="running",
                current_stage="等待执行",
                market=symbol,
                data_gate_status="pending",
                degraded_reasons_json=[],
                stages_json=[{"stage": "等待执行", "status": "completed", "duration_ms": 0}],
                result_json={},
            ))

    def _stage(self, run_id: str, name: str, operation: Callable[[], Any]) -> Any:
        started = perf_counter()
        self._update_run(run_id, current_stage=name)
        try:
            result = operation()
        except Exception as error:
            self._append_stage(run_id, name, "failed", started, str(error))
            raise
        self._append_stage(run_id, name, "completed", started, None)
        return result

    def _append_stage(self, run_id: str, name: str, status: str, started: float, error: str | None) -> None:
        with self.sessions() as session, session.begin():
            row = session.get(V31AnalysisRun, run_id)
            if row is None:
                return
            stages = list(row.stages_json or [])
            stages.append({
                "stage": name,
                "status": status,
                "duration_ms": round((perf_counter() - started) * 1000),
                "error": error,
            })
            row.stages_json = stages

    def _update_run(self, run_id: str, **values: Any) -> None:
        with self.sessions() as session, session.begin():
            row = session.get(V31AnalysisRun, run_id)
            if row is not None:
                for name, value in values.items():
                    setattr(row, name, value)

    def _refresh(self, symbol: str, enabled: bool) -> dict[str, Any]:
        refresh_result: dict[str, Any] = {"refresh_requested": enabled}
        if enabled:
            latest = self.market.latest_daily_date(symbol)
            start = None if latest is None else latest - timedelta(days=14)
            end = self._latest_available_session(symbol)
            provider = _TimedMarketProvider(
                AkShareIndexProvider() if symbol == "399006" else YahooDirectIndexProvider(),
                timeout_seconds=35,
            )
            market_result = self.market.update_from_provider(symbol, provider, start, end)
            valuation_result = self.valuations.refresh(
                symbol,
                _TimedValuationProvider(AkShareValuationProvider(), timeout_seconds=25),
            )
            refresh_result.update({
                "market_refresh": market_result.model_dump(mode="json", exclude={"records"}),
                "valuation_refresh": valuation_result.model_dump(mode="json"),
            })
        latest_local = self.market.latest_daily_date(symbol)
        if latest_local is None:
            raise DataGateBlocked("NO_DAILY_DATA", f"{symbol}没有可用日K")
        # Let the storage service derive the exchange calendar from the full
        # local daily bounds. Passing only the ten-year modeling window would
        # incorrectly flag legitimate older rows as unexpected sessions.
        aggregation = self.market.aggregate_periods(
            symbol,
            expected_trade_dates=None,
            as_of=self._latest_available_session(symbol),
        )
        refresh_result["aggregation"] = aggregation
        indicator_results = self.indicators.recalculate_all_timeframes(symbol)
        refresh_result["indicators"] = [
            {
                "timeframe": result.timeframe,
                "status": result.status.value,
                "cutoff": _iso(result.data_cutoff),
                "error": result.error,
            }
            for result in indicator_results
        ]
        if any(result.status == IndicatorCalculationStatus.VALIDATION_ERROR for result in indicator_results):
            raise DataGateBlocked("INDICATOR_FAILURE", "技术指标计算失败")
        return refresh_result

    def _latest_available_session(self, symbol: str) -> date:
        now = self.now_provider().astimezone(ZoneInfo("Asia/Shanghai") if symbol == "399006" else ZoneInfo("America/New_York"))
        cutoff = time(16, 15) if symbol == "399006" else time(17, 30)
        candidate = now.date() if now.time() >= cutoff else now.date() - timedelta(days=1)
        sessions = self.calendar.sessions(symbol, candidate - timedelta(days=15), candidate)
        if not sessions:
            raise DataGateBlocked("CALENDAR_EMPTY", f"{symbol}交易日历不可用")
        return sessions[-1]

    def _data_gate(self, symbol: str, context: dict[str, Any]) -> GateSnapshot:
        latest_available = self._latest_available_session(symbol)
        with self.sessions() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == symbol))
            if instrument is None:
                raise DataGateBlocked("UNKNOWN_MARKET", f"指数不存在：{symbol}")
            daily = list(session.scalars(select(MarketPrice).where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
            ).order_by(MarketPrice.trade_date)))
            weekly = list(session.scalars(select(MarketPrice).where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "weekly",
            ).order_by(MarketPrice.trade_date)))
            valuation_as_of = session.scalar(select(func.max(ValuationRecord.valuation_date)).where(
                ValuationRecord.instrument_id == instrument.id,
                ValuationRecord.valuation_date <= latest_available,
            ))
        if not daily or not weekly:
            raise DataGateBlocked("MISSING_PERIOD_DATA", "日K或完整周K为空")
        dates = [row.trade_date for row in daily]
        if len(dates) != len(set(dates)) or dates != sorted(dates):
            raise DataGateBlocked("DUPLICATE_OR_UNSORTED", "行情日期重复或顺序错误")
        for row in daily[-260:]:
            values = (row.open_price, row.high_price, row.low_price, row.close_price)
            if any(value is None or value <= 0 for value in values):
                raise DataGateBlocked("INVALID_OHLC", f"{row.trade_date}价格为空或非正数")
            if row.high_price < max(row.open_price, row.close_price) or row.low_price > min(row.open_price, row.close_price):
                raise DataGateBlocked("INVALID_OHLC", f"{row.trade_date}不满足OHLC约束")
            source = (row.source or "").upper()
            if not source or "DEMO" in source or "ETF" in source or "FUND" in source:
                raise DataGateBlocked("PROXY_SOURCE", f"{row.trade_date}不是直接指数数据", source=row.source)
        price_as_of = dates[-1]
        if price_as_of < latest_available:
            raise DataGateBlocked("STALE_DAILY_DATA", f"行情截止{price_as_of}，应更新至{latest_available}", source=daily[-1].source)
        recent_expected = set(self.calendar.sessions(symbol, latest_available - timedelta(days=40), latest_available))
        recent_actual = {day for day in dates if day >= latest_available - timedelta(days=40)}
        missing = sorted(recent_expected - recent_actual)
        if missing:
            raise DataGateBlocked("RECENT_SESSION_GAP", f"最近行情缺少交易日：{', '.join(map(str, missing[:5]))}")
        week_start = latest_available - timedelta(days=latest_available.weekday())
        week_sessions = self.calendar.sessions(symbol, week_start, week_start + timedelta(days=6))
        current_week_complete = bool(week_sessions) and latest_available >= week_sessions[-1]
        eligible_weekly = [
            row
            for row in weekly
            if current_week_complete
            or row.trade_date.isocalendar()[:2] != latest_available.isocalendar()[:2]
        ]
        if not eligible_weekly:
            raise DataGateBlocked("NO_COMPLETE_WEEK", "没有可供主模型使用的完整交易周")
        complete_week = eligible_weekly[-1].trade_date
        if complete_week > price_as_of:
            raise DataGateBlocked("FUTURE_LEAKAGE", "周K截止日超过日K截止日")
        degraded: list[str] = []
        if valuation_as_of is None:
            degraded.append("PE/PB估值不可用")
        elif (latest_available - valuation_as_of).days > 10:
            degraded.append(f"估值滞后至{valuation_as_of}")
        if symbol == "NDX":
            volume_source = "DISABLED_DIRECT_INDEX"
            degraded.append("NDX直接指数无可靠成交量，成交量特征已禁用")
            volume_as_of = None
        else:
            missing_volume = [row.trade_date for row in weekly[-60:] if row.volume is None]
            if missing_volume:
                raise DataGateBlocked("WEEKLY_VOLUME_MISSING", f"创业板周成交量缺失：{missing_volume[-1]}")
            volume_source = weekly[-1].volume_source or "DAILY_SUM"
            volume_as_of = complete_week
        incomplete = price_as_of if not current_week_complete and price_as_of > complete_week else None
        return GateSnapshot(symbol, latest_available, price_as_of, valuation_as_of, complete_week, incomplete, volume_as_of, daily[-1].source or "UNKNOWN", volume_source, tuple(degraded))

    def _model_parameters(self, symbol: str) -> dict[str, Any]:
        if symbol == "399006":
            return {"momentum": .30, "trend": .25, "macd": .20, "valuation": .15, "risk": .10, "base_threshold": .045, "volatility_reference": .34, "seed": 39900631}
        return {"momentum": .34, "trend": .28, "macd": .22, "valuation": .06, "risk": .10, "base_threshold": .030, "volatility_reference": .24, "seed": 10031}

    def _ensure_models(self, symbol: str, cutoff: date) -> dict[str, Any]:
        prefix = "CYB" if symbol == "399006" else "NDX"
        parameters = self._model_parameters(symbol)
        definitions = (
            ("weekly", f"{prefix}_WEEKLY_V3.1.0", parameters, parameters["seed"]),
            ("daily", f"{prefix}_DAILY_CORRECTOR_V3.1.0", {"window": 100, "confidence_limit": 10, "position_limit": 10}, parameters["seed"] + 1),
        )
        with self.sessions() as session, session.begin():
            for model_type, version, payload, seed in definitions:
                model = session.scalar(select(V31ModelVersion).where(
                    V31ModelVersion.market == symbol,
                    V31ModelVersion.model_type == model_type,
                    V31ModelVersion.status == "champion",
                ).order_by(V31ModelVersion.created_at.desc()))
                if model is None:
                    model = V31ModelVersion(
                        id=f"{symbol}:{model_type}:{version}", market=symbol, model_type=model_type,
                        version=version, feature_version=FEATURE_VERSION, methodology_version=METHODOLOGY_VERSION,
                        training_end_date=cutoff, validation_start_date=None, validation_end_date=cutoff,
                        random_seed=seed, parameters_json=payload, metrics_json={"validation_sample_count": 0}, status="champion",
                    )
                    session.add(model)
        return {
            "weekly": {"version": definitions[0][1], "type": "champion", "feature_version": FEATURE_VERSION, "parameters": parameters},
            "daily": {"version": definitions[1][1], "type": "champion", "feature_version": FEATURE_VERSION, "window": 100},
            "challenger": {"candidate_count": 0, "promoted": False, "reason": "本次仅执行周度预测；预测运行与重新训练已分离"},
        }

    def _weekly_rows(self, symbol: str, cutoff: date) -> tuple[list[MarketPrice], list[float]]:
        with self.sessions() as session:
            rows = list(session.scalars(select(MarketPrice).join(Instrument).where(
                Instrument.code == symbol,
                MarketPrice.timeframe == "weekly",
                MarketPrice.trade_date <= cutoff,
            ).order_by(MarketPrice.trade_date)))
        closes = [_f(row.adjusted_close_price or row.close_price) for row in rows]
        if len(closes) < 65:
            raise DataGateBlocked("WEEKLY_HISTORY_SHORT", "周K历史不足65周")
        return rows, closes

    def _valuation_percentile(self, symbol: str, cutoff: date) -> tuple[float | None, float | None, float | None]:
        with self.sessions() as session:
            rows = list(session.scalars(select(ValuationRecord).join(Instrument).where(
                Instrument.code == symbol,
                ValuationRecord.valuation_date <= cutoff,
            ).order_by(ValuationRecord.valuation_date)))
        values = [_f(row.pe_ratio if row.pe_ratio is not None else row.pb_ratio) for row in rows if row.pe_ratio is not None or row.pb_ratio is not None]
        if not values:
            return None, None, None
        latest = values[-1]
        percentile = sum(value <= latest for value in values) / len(values) * 100
        return latest, percentile, (values[-1] / values[-5] - 1) if len(values) >= 5 and values[-5] else None

    def _weekly_signal(self, symbol: str, gate: GateSnapshot, model: dict[str, Any]) -> dict[str, Any]:
        rows, closes = self._weekly_rows(symbol, gate.complete_week_as_of)
        dif, dea, hist = _macd(closes)
        vol4 = _volatility(closes, 4, 52)
        vol13 = _volatility(closes, 13, 52)
        ma20 = statistics.fmean(closes[-20:])
        ma60 = statistics.fmean(closes[-60:])
        ret4, ret8, ret13 = (_window_return(closes, span) for span in (4, 8, 13))
        valuation, valuation_percentile, valuation_speed = self._valuation_percentile(symbol, gate.complete_week_as_of)
        scale = max(vol13 / math.sqrt(52), .01)
        momentum = _clamp((ret4 * .5 + ret8 * .3 + ret13 * .2) / (scale * 3), -1, 1)
        trend = _clamp(((closes[-1] / ma20 - 1) * .65 + (closes[-1] / ma60 - 1) * .35) / max(scale * 2, .02), -1, 1)
        macd_score = _clamp(((dif[-1] - dea[-1]) / closes[-1]) / max(scale / 4, .002), -1, 1)
        valuation_score = 0.0 if valuation_percentile is None else _clamp((50 - valuation_percentile) / 50, -1, 1)
        drawdown = _max_drawdown(closes[-60:])
        risk_score = _clamp(1 + drawdown / max(vol13, .08), -1, 1)
        weights = model["parameters"]
        score = _clamp(momentum * weights["momentum"] + trend * weights["trend"] + macd_score * weights["macd"] + valuation_score * weights["valuation"] + risk_score * weights["risk"], -1, 1)
        direction = "up" if score >= .08 else "down" if score <= -.08 else "sideways"
        if score >= .70: state, position_range = "极强看多", [90, 100]
        elif score >= .45: state, position_range = "强看多", [75, 90]
        elif score >= .20: state, position_range = "中度看多", [60, 75]
        elif score >= .05: state, position_range = "震荡偏多", [50, 65]
        elif score > -.05: state, position_range = "中性震荡", [40, 55]
        elif score > -.20: state, position_range = "震荡偏空", [30, 45]
        elif score > -.45: state, position_range = "中度看空", [20, 35]
        else: state, position_range = "强看空", [10, 20]
        if valuation_percentile is not None and valuation_percentile >= 90 and score > .2:
            market_state = "高估但强势" if hist[-1] >= hist[-2] else "高估且动能减弱"
        elif valuation_percentile is not None and valuation_percentile <= 15 and score < -.08:
            market_state = "低估但止跌" if hist[-1] > hist[-2] else "低估且弱势"
        elif score < -.45: market_state = "下跌加速"
        elif score > .2: market_state = "合理估值上涨"
        else: market_state = "合理估值震荡"
        confidence = _clamp(56 + abs(score) * 30 - min(20, len(gate.degraded) * 5), 35, 90)
        positive = []
        risks = []
        (positive if momentum > 0 else risks).append(f"4/8/13周动量综合{momentum:+.2f}")
        (positive if trend > 0 else risks).append("价格位于关键周均线上方" if trend > 0 else "价格位于关键周均线下方")
        (positive if macd_score > 0 else risks).append("周K DIF高于DEA" if macd_score > 0 else "周K DIF低于DEA")
        if valuation_percentile is not None:
            (positive if valuation_percentile < 35 else risks).append(f"估值历史分位{valuation_percentile:.1f}%")
        return {
            "direction": direction, "score": _round(score * 100, 2), "confidence": _round(confidence, 1),
            "market_state": market_state, "state": state, "base_target_position": _snap5(statistics.fmean(position_range)),
            "position_range": position_range, "positive_factors": positive, "risk_factors": risks,
            "features": {
                "dif": _round(dif[-1]), "dea": _round(dea[-1]), "macd_histogram": _round(hist[-1]),
                "dif_slope": _round(dif[-1] - dif[-2]), "dea_slope": _round(dea[-1] - dea[-2]),
                "macd_histogram_speed": _round(hist[-1] - hist[-2]), "cross": "golden" if dif[-1] >= dea[-1] else "death",
                "zero_axis": "above" if dif[-1] >= 0 and dea[-1] >= 0 else "below",
                "return_4w": _round(ret4 * 100), "return_8w": _round(ret8 * 100), "return_13w": _round(ret13 * 100),
                "volatility_4w": _round(vol4 * 100), "volatility_13w": _round(vol13 * 100),
                "ma20": _round(ma20), "ma60": _round(ma60), "ma20_distance": _round((closes[-1] / ma20 - 1) * 100),
                "ma60_distance": _round((closes[-1] / ma60 - 1) * 100), "drawdown_60w": _round(drawdown * 100),
                "valuation": None if valuation is None else _round(valuation), "valuation_percentile": None if valuation_percentile is None else _round(valuation_percentile, 1),
                "valuation_speed": None if valuation_speed is None else _round(valuation_speed * 100),
                "volume_change": None if symbol == "NDX" or rows[-2].volume in (None, 0) or rows[-1].volume is None else _round((_f(rows[-1].volume) / _f(rows[-2].volume) - 1) * 100),
            },
        }

    def _daily_correction(self, symbol: str, gate: GateSnapshot, weekly: dict[str, Any]) -> dict[str, Any]:
        with self.sessions() as session:
            rows = list(session.scalars(select(MarketPrice).join(Instrument).where(
                Instrument.code == symbol, MarketPrice.timeframe == "daily", MarketPrice.trade_date <= gate.price_as_of,
            ).order_by(MarketPrice.trade_date.desc()).limit(100)))
        rows.reverse()
        if len(rows) < 100:
            raise DataGateBlocked("DAILY_WINDOW_SHORT", "最近100个交易日日K不足")
        closes = [_f(row.adjusted_close_price or row.close_price) for row in rows]
        dif, dea, hist = _macd(closes)
        rsi = _rsi(closes)
        ma20, ma60 = statistics.fmean(closes[-20:]), statistics.fmean(closes[-60:])
        overheat = rsi >= 70 or closes[-1] / ma20 - 1 >= .07
        oversold = rsi <= 30 or closes[-1] / ma20 - 1 <= -.07
        daily_direction = "up" if dif[-1] > dea[-1] and closes[-1] > ma20 else "down" if dif[-1] < dea[-1] and closes[-1] < ma20 else "sideways"
        consistent = daily_direction == weekly["direction"] or daily_direction == "sideways"
        confidence_adjustment = 3 if consistent and daily_direction != "sideways" else -4 if not consistent else 0
        position_adjustment = 0
        state = "节奏中性"
        if overheat:
            state = "短线过热"
            position_adjustment = -5
            confidence_adjustment = min(confidence_adjustment, -3)
        elif oversold:
            state = "短线超跌"
            position_adjustment = 5
            confidence_adjustment = min(confidence_adjustment, -2) if weekly["direction"] == "down" else max(confidence_adjustment, 2)
        if weekly["direction"] == "up" and daily_direction == "down":
            state = "周线看多、日线等待确认"
            position_adjustment = min(position_adjustment, -5)
        elif weekly["direction"] == "down" and daily_direction == "up":
            state = "周线看空、日线超跌反弹"
            position_adjustment = max(position_adjustment, 5)
        position_adjustment = int(_clamp(position_adjustment, -10, 10))
        confidence_adjustment = int(_clamp(confidence_adjustment, -10, 10))
        sessions = self.calendar.sessions(symbol, gate.price_as_of + timedelta(days=1), gate.price_as_of + timedelta(days=20))
        window = sessions[:3]
        trigger = "日K收盘站上20日均线且DIF不低于DEA" if weekly["direction"] == "up" else "日K收盘跌破20日均线且DIF不高于DEA" if weekly["direction"] == "down" else "周K方向突破震荡区间后再执行"
        invalidation = "日K跌破20日均线且DIF下穿DEA" if weekly["direction"] == "up" else "日K重新站上20日均线且DIF上穿DEA" if weekly["direction"] == "down" else "周K继续维持震荡"
        return {
            "window_sessions": 100, "state": state, "direction": daily_direction, "overheat": overheat, "oversold": oversold,
            "consistent_with_weekly": consistent, "confidence_adjustment": confidence_adjustment, "position_adjustment": position_adjustment,
            "execution_speed": "slow" if not consistent or overheat or oversold else "normal",
            "first_execution_window": [_iso(window[0]) if window else None, _iso(window[-1]) if window else None],
            "trigger_conditions": [trigger], "invalidation_conditions": [invalidation],
            "features": {"dif": _round(dif[-1]), "dea": _round(dea[-1]), "macd_histogram": _round(hist[-1]), "rsi": _round(rsi, 1),
                "ma5": _round(statistics.fmean(closes[-5:])), "ma10": _round(statistics.fmean(closes[-10:])), "ma20": _round(ma20), "ma60": _round(ma60),
                "return_5d": _round(_window_return(closes, 5) * 100), "return_20d": _round(_window_return(closes, 20) * 100), "return_60d": _round(_window_return(closes, 60) * 100),
                "drawdown_20d": _round(_max_drawdown(closes[-20:]) * 100), "drawdown_60d": _round(_max_drawdown(closes[-60:]) * 100),
                "volatility_20d": _round(_volatility(closes, 20, 252) * 100), "volatility_60d": _round(_volatility(closes, 60, 252) * 100),
                "distance_100d_high": _round((closes[-1] / max(closes) - 1) * 100), "distance_100d_low": _round((closes[-1] / min(closes) - 1) * 100),
                "ma20_distance": _round((closes[-1] / ma20 - 1) * 100)},
        }

    def _probability_path(self, symbol: str, gate: GateSnapshot, weekly: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
        _, closes = self._weekly_rows(symbol, gate.complete_week_as_of)
        paths: list[list[float]] = []
        current_signature = (_window_return(closes, 4), _window_return(closes, 13), _volatility(closes, 13, 52), _max_drawdown(closes[-60:]))
        candidates: list[tuple[float, list[float]]] = []
        for index in range(60, len(closes) - 13):
            history = closes[: index + 1]
            signature = (_window_return(history, 4), _window_return(history, 13), _volatility(history, 13, 52), _max_drawdown(history[-60:]))
            distance = sum(abs(left - right) / scale for left, right, scale in zip(current_signature, signature, (.12, .25, .30, .35)))
            base = closes[index]
            candidate = [closes[index + week] / base - 1 for week in range(1, 14)]
            candidates.append((distance, candidate))
        paths = [path for _, path in sorted(candidates, key=lambda item: item[0])[:120]]
        if len(paths) < 20:
            raise DataGateBlocked("PATH_SAMPLE_SHORT", "可用13周历史情景不足20组")
        p10 = [_percentile([path[week] for path in paths], .10) for week in range(13)]
        p50 = [_percentile([path[week] for path in paths], .50) for week in range(13)]
        p90 = [_percentile([path[week] for path in paths], .90) for week in range(13)]
        expected = [statistics.fmean(path[week] for path in paths) for week in range(13)]
        terminals = [path[-1] for path in paths]
        volatility = weekly["features"]["volatility_13w"] / 100
        parameters = model["parameters"]
        threshold = parameters["base_threshold"] * max(.65, min(1.8, volatility / parameters["volatility_reference"]))
        up = sum(value > threshold for value in terminals) / len(terminals) * 100
        down = sum(value < -threshold for value in terminals) / len(terminals) * 100
        sideways = 100 - up - down
        maxima = [max(range(13), key=lambda week: path[week]) + 1 for path in paths]
        minima = [min(range(13), key=lambda week: path[week]) + 1 for path in paths]
        expected_drawdowns = []
        for path in paths:
            levels = [1 + value for value in path]
            expected_drawdowns.append(_max_drawdown(levels))
        points = [{"horizon_week": week + 1, "p10_cumulative_return": _round(p10[week] * 100), "p50_cumulative_return": _round(p50[week] * 100), "p90_cumulative_return": _round(p90[week] * 100), "expected_cumulative_return": _round(expected[week] * 100)} for week in range(13)]
        return {
            "points": points, "up_probability": _round(up, 1), "sideways_probability": _round(sideways, 1), "down_probability": _round(down, 1),
            "direction_threshold": _round(threshold * 100, 2), "expected_max_drawdown": _round(statistics.fmean(expected_drawdowns) * 100),
            "high_week_range": f"第{max(1, round(_percentile(maxima, .35)))}—{min(13, round(_percentile(maxima, .65)))}周",
            "low_week_range": f"第{max(1, round(_percentile(minima, .35)))}—{min(13, round(_percentile(minima, .65)))}周",
            "high_range_probability": _round(sum(4 <= week <= 8 for week in maxima) / len(maxima) * 100, 1),
            "low_range_probability": _round(sum(6 <= week <= 13 for week in minima) / len(minima) * 100, 1),
            "scenario_count": len(paths), "method": "历史相似情景路径+逐周分位数", "random_seed": model["parameters"]["seed"],
        }

    def _position(self, symbol: str) -> dict[str, Any]:
        positions = self.investment_calendar.current_positions()
        confirmed = positions.get(symbol)
        if confirmed is None:
            confirmed = 0
        return {"confirmed_position": confirmed, "in_transit_adjustment": 0, "settlement_position": confirmed, "advice_basis_position": confirmed, "defaulted_to_zero": not bool(self.investment_calendar.list_entries(symbol))}

    def _next_sessions(self, symbol: str, after: date, count: int) -> list[date]:
        days = self.calendar.sessions(symbol, after + timedelta(days=1), after + timedelta(days=120))
        return list(days[:count])

    @staticmethod
    def _ratio(score: float) -> str:
        return "7:3" if score >= 65 else "6:4" if score >= 20 else "5:5" if score > -20 else "4:6" if score > -60 else "3:7"

    @staticmethod
    def _allocation(total: int, ratio: str) -> dict[str, Any]:
        fund_ratio = int(ratio[0]) / 10
        candidates = [(fund, total - fund) for fund in range(0, total + 1, 5)]
        fund, etf = min(candidates, key=lambda pair: abs(pair[0] - total * fund_ratio)) if candidates else (0, 0)
        actual = 0 if total == 0 else fund / total * 100
        return {"target_ratio": ratio, "fund_position": fund, "etf_position": etf, "actual_ratio": f"{actual:.1f}:{100-actual:.1f}", "rounding_deviation_points": _round(abs(actual - fund_ratio * 100), 1)}

    def _advice(self, symbol: str, gate: GateSnapshot, weekly: dict[str, Any], daily: dict[str, Any], path: dict[str, Any], position: dict[str, Any]) -> dict[str, Any]:
        current = position["advice_basis_position"]
        target = _snap5(weekly["base_target_position"] + daily["position_adjustment"])
        if current == 0 and weekly["direction"] != "up":
            target = 0
        change = target - current
        side = "buy" if change > 0 else "sell" if change < 0 else "hold"
        total = abs(change)
        batch_count = 0 if total == 0 else min(4, max(1, math.ceil(total / 20)))
        sizes: list[int] = []
        remaining = total
        for index in range(batch_count):
            slots = batch_count - index
            size = _snap5(remaining / slots)
            size = max(5, size)
            sizes.append(size)
            remaining -= size
        if sizes and remaining:
            sizes[-1] += remaining
        sessions = self._next_sessions(symbol, gate.price_as_of, max(20, batch_count * 6 + 3))
        batches = []
        after = current
        for index, size in enumerate(sizes):
            session_index = min(index * 5, len(sessions) - 1)
            execution = sessions[session_index]
            window_end = sessions[min(session_index + 2, len(sessions) - 1)]
            after += size if side == "buy" else -size
            trigger = daily["trigger_conditions"][0] if index == 0 else ("上一个完整周K方向保持不变，且DEA未发生反向拐头" if side == "buy" else "上一个完整周K方向保持不变，且DEA继续转弱")
            invalidation = daily["invalidation_conditions"][0]
            batches.append({
                "sequence": index + 1, "side": side, "position_points": size,
                "relative_holding_percent": None if side != "sell" or current == 0 else _round(size / current * 100, 2),
                "execution_date": execution.isoformat(), "execution_window": [execution.isoformat(), window_end.isoformat()],
                "review_date": sessions[min(session_index + 5, len(sessions) - 1)].isoformat(), "trigger_condition": trigger,
                "invalidation_condition": invalidation, "position_after": after, "conditional": True,
            })
        ratio = self._ratio(weekly["score"])
        confidence = _clamp(weekly["confidence"] + daily["confidence_adjustment"], 0, 100)
        return {
            "current_position": current, "weekly_base_target_position": weekly["base_target_position"], "daily_position_adjustment": daily["position_adjustment"],
            "target_position": target, "side": side, "total_adjustment_points": total, "confidence": _round(confidence, 1),
            "batches": batches, "fund_etf": self._allocation(target, ratio),
            "summary": "维持0%仓位，等待周K看多确认" if total == 0 and current == 0 else f"分{batch_count}批{'增加' if side == 'buy' else '减少'}{total}个仓位百分点" if total else "维持当前仓位",
            "percentage_basis": "占总可投资资金的仓位百分点", "execution_note": "日期为条件复核窗口；条件未触发则不执行，不代表预测顶部或底部。",
        }

    def _snapshot_id(self, symbol: str, gate: GateSnapshot, models: dict[str, Any]) -> str:
        payload = "|".join((symbol, gate.price_as_of.isoformat(), gate.complete_week_as_of.isoformat(), _iso(gate.valuation_as_of) or "none", models["weekly"]["version"], FEATURE_VERSION))
        return "DS-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _persist_forecast(self, symbol: str, gate: GateSnapshot, models: dict[str, Any], weekly: dict[str, Any], daily: dict[str, Any], path: dict[str, Any], snapshot_id: str) -> int:
        points = path["points"]
        with self.sessions() as session, session.begin():
            forecast = session.scalar(select(V31WeeklyForecast).where(
                V31WeeklyForecast.market == symbol,
                V31WeeklyForecast.forecast_date == gate.complete_week_as_of,
                V31WeeklyForecast.model_version == models["weekly"]["version"],
            ))
            if forecast is None:
                forecast = V31WeeklyForecast(
                    market=symbol, forecast_date=gate.complete_week_as_of, model_version=models["weekly"]["version"], data_snapshot_id=snapshot_id,
                    weekly_direction=weekly["direction"], weekly_confidence=Decimal(str(weekly["confidence"])), base_target_position=weekly["base_target_position"],
                    p10_path_json=[point["p10_cumulative_return"] for point in points], p50_path_json=[point["p50_cumulative_return"] for point in points],
                    p90_path_json=[point["p90_cumulative_return"] for point in points], expected_path_json=[point["expected_cumulative_return"] for point in points],
                    up_probability=Decimal(str(path["up_probability"])), sideways_probability=Decimal(str(path["sideways_probability"])), down_probability=Decimal(str(path["down_probability"])),
                    expected_max_drawdown=Decimal(str(path["expected_max_drawdown"])), high_week_range=path["high_week_range"], low_week_range=path["low_week_range"],
                    payload_json={"scenario_count": path["scenario_count"], "direction_threshold": path["direction_threshold"], "volatility": weekly["features"]["volatility_13w"]},
                )
                session.add(forecast)
                session.flush()
            correction = session.scalar(select(V31DailyCorrection).where(V31DailyCorrection.weekly_forecast_id == forecast.id))
            window = daily["first_execution_window"]
            if correction is None:
                correction = V31DailyCorrection(weekly_forecast_id=forecast.id, correction_date=gate.price_as_of, daily_state=daily["state"], confidence_adjustment=daily["confidence_adjustment"], position_adjustment=daily["position_adjustment"], execution_window_start=date.fromisoformat(window[0]) if window[0] else None, execution_window_end=date.fromisoformat(window[1]) if window[1] else None, trigger_conditions_json=daily["trigger_conditions"], invalidation_conditions_json=daily["invalidation_conditions"], payload_json=daily["features"])
                session.add(correction)
            return forecast.id

    def _evaluate_mature(self, symbol: str) -> dict[str, Any]:
        with self.sessions() as session, session.begin():
            forecasts = list(session.scalars(select(V31WeeklyForecast).where(V31WeeklyForecast.market == symbol).order_by(V31WeeklyForecast.forecast_date)))
            for forecast in forecasts:
                if session.scalar(select(V31ModelEvaluation).where(V31ModelEvaluation.forecast_id == forecast.id)) is not None:
                    continue
                actual_rows = list(session.scalars(select(MarketPrice).join(Instrument).where(
                    Instrument.code == symbol, MarketPrice.timeframe == "weekly", MarketPrice.trade_date > forecast.forecast_date,
                ).order_by(MarketPrice.trade_date).limit(13)))
                if len(actual_rows) < 13:
                    continue
                base_row = session.scalar(select(MarketPrice).join(Instrument).where(Instrument.code == symbol, MarketPrice.timeframe == "weekly", MarketPrice.trade_date == forecast.forecast_date))
                if base_row is None:
                    continue
                base = _f(base_row.adjusted_close_price or base_row.close_price)
                actual = [(_f(row.adjusted_close_price or row.close_price) / base - 1) * 100 for row in actual_rows]
                p10, p50, p90 = list(map(float, forecast.p10_path_json)), list(map(float, forecast.p50_path_json)), list(map(float, forecast.p90_path_json))
                scale = max(float((forecast.payload_json or {}).get("volatility", 20)), 5)
                path_error = math.sqrt(statistics.fmean((actual[index] - p50[index]) ** 2 for index in range(13))) / scale
                coverage = sum(p10[index] <= actual[index] <= p90[index] for index in range(13)) / 13 * 100
                actual_high, actual_low = actual.index(max(actual)) + 1, actual.index(min(actual)) + 1
                predicted_high, predicted_low = p50.index(max(p50)) + 1, p50.index(min(p50)) + 1
                direction_actual = "up" if actual[-1] > float((forecast.payload_json or {}).get("direction_threshold", 3)) else "down" if actual[-1] < -float((forecast.payload_json or {}).get("direction_threshold", 3)) else "sideways"
                probability = {"up": _f(forecast.up_probability), "down": _f(forecast.down_probability), "sideways": _f(forecast.sideways_probability)}[direction_actual] / 100
                session.add(V31ModelEvaluation(forecast_id=forecast.id, maturity_status="full", path_error=Decimal(str(_round(path_error))), terminal_return_error=Decimal(str(_round(abs(actual[-1] - p50[-1])))), direction_score=Decimal(str(_round((1 - probability) ** 2))), quantile_loss=Decimal(str(_round(statistics.fmean(max(p10[i] - actual[i], 0) + max(actual[i] - p90[i], 0) for i in range(13))))), interval_coverage=Decimal(str(_round(coverage))), high_date_error=abs(actual_high - predicted_high), low_date_error=abs(actual_low - predicted_low), turnover=Decimal("0"), net_strategy_return=Decimal(str(_round(actual[-1]))), payload_json={"actual_path": [_round(value) for value in actual], "predicted_path": p50}))
        with self.sessions() as session:
            rows = list(session.scalars(select(V31ModelEvaluation).join(V31WeeklyForecast).where(V31WeeklyForecast.market == symbol).order_by(V31ModelEvaluation.evaluated_at.desc())))
        def metrics(window: int | None) -> dict[str, Any]:
            sample = rows if window is None else rows[:window]
            return {"sample_count": len(sample), "path_error": None if not sample else _round(statistics.fmean(_f(row.path_error) for row in sample)), "terminal_return_error": None if not sample else _round(statistics.fmean(_f(row.terminal_return_error) for row in sample)), "probability_score": None if not sample else _round(statistics.fmean(_f(row.direction_score) for row in sample)), "interval_coverage": None if not sample else _round(statistics.fmean(_f(row.interval_coverage) for row in sample), 1), "high_week_error": None if not sample else _round(statistics.fmean(row.high_date_error or 0 for row in sample), 2), "low_week_error": None if not sample else _round(statistics.fmean(row.low_date_error or 0 for row in sample), 2)}
        return {"out_of_sample_only": True, "recent_20": metrics(20), "recent_52": metrics(52), "all": metrics(None), "training_error": None, "validation_error": None, "note": "完整13周预测仅在13周后计入样本外指标"}

    def _complete_run(self, run_id: str, result: dict[str, Any], gate: GateSnapshot, model_version: str) -> None:
        self._append_stage(run_id, "已完成", "completed", perf_counter(), None)
        self._update_run(run_id, completed_at=utc_now(), status="completed", current_stage="已完成", price_data_as_of=gate.price_as_of, valuation_data_as_of=gate.valuation_as_of, weekly_data_as_of=gate.complete_week_as_of, data_gate_status="degraded" if gate.degraded else "passed", degraded_reasons_json=list(gate.degraded), model_version=model_version, data_snapshot_id=result["data_snapshot_id"], result_json=result)

    def _fail_run(self, run_id: str, error: Exception, gate: GateSnapshot | None, context: dict[str, Any]) -> None:
        code = error.code if isinstance(error, DataGateBlocked) else type(error).__name__
        source = error.source if isinstance(error, DataGateBlocked) else None
        message = str(error)
        payload = {"failure_stage": self.get_run(run_id)["current_stage"], "failure_reason": message, "data_source": source or context.get("market_refresh", {}).get("source"), "last_successful_data_date": _iso(self.market.latest_daily_date(self.get_run(run_id)["market"])), "run_id": run_id, "fallback_allowed": False}
        self._update_run(run_id, completed_at=utc_now(), status="failed", current_stage="执行失败", data_gate_status="blocked", error_code=code, error_message=message, degraded_reasons_json=[] if gate is None else list(gate.degraded), result_json=payload)

    @staticmethod
    def _gate_payload(gate: GateSnapshot) -> dict[str, Any]:
        return {"latest_available_trade_date": gate.latest_available.isoformat(), "price_data_as_of": gate.price_as_of.isoformat(), "valuation_data_as_of": _iso(gate.valuation_as_of), "latest_complete_week_as_of": gate.complete_week_as_of.isoformat(), "incomplete_week_data_as_of": _iso(gate.incomplete_week_as_of), "model_data_cutoff": gate.model_cutoff.isoformat(), "volume_data_as_of": _iso(gate.volume_as_of), "price_source": gate.source, "volume_source": gate.volume_source, "current_week_complete": gate.incomplete_week_as_of is None}

    @staticmethod
    def _run_payload(row: V31AnalysisRun) -> dict[str, Any]:
        return {"implementation_revision": "V31-LATEST-ALL-1", "id": row.id, "market": row.market, "status": row.status, "current_stage": row.current_stage, "started_at": _iso(row.started_at), "completed_at": _iso(row.completed_at), "price_data_as_of": _iso(row.price_data_as_of), "valuation_data_as_of": _iso(row.valuation_data_as_of), "weekly_data_as_of": _iso(row.weekly_data_as_of), "data_gate_status": row.data_gate_status, "degraded_reasons": row.degraded_reasons_json, "error_code": row.error_code, "error_message": row.error_message, "model_version": row.model_version, "data_snapshot_id": row.data_snapshot_id, "stages": row.stages_json, "result": row.result_json}
