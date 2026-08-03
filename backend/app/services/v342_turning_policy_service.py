"""Independent V3.4.2 turning-point, position-policy, and replay layer.

The service consumes immutable V3.4.1 forecasts.  It never updates a V3.4.1
row and it never uses a future realised price to construct a historical
decision.  State transitions and replay outputs are append-only V3.4.2 data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V2PositionEvent,
    V2PositionSnapshot,
    V341FeatureSnapshot,
    V341Forecast,
    V341ForecastEvaluation,
    V341TrainingIteration,
    V342PolicyBatch,
    V342PolicyBatchEvent,
    V342PolicyRun,
    V342PolicyVersion,
    V342StrategyBacktestPoint,
    V342StrategyBacktestRun,
    V342TurningAssessment,
    V342TurningCandidate,
    utc_now,
)
from backend.app.services.market_calendar import CalendarProvider
from backend.app.services.instrument_universe import MODEL_MARKETS
from backend.app.services.v341_runtime_service import EVALUATION_VERSION
from backend.app.services.v341_training_service import (
    PROTOCOL_VERSION as V341_PROTOCOL_VERSION,
)


PROTOCOL_VERSION = "V3.4.2_TURNING_POLICY"
POLICY_VERSION = "V3.4.2_POLICY_1"
RETROSPECTIVE_POLICY_VERSION = "V3.4.2_POLICY_1_RETROSPECTIVE"
POLICY_RELEASE_DATE = date(2026, 8, 2)
RETROSPECTIVE_EFFECTIVE_FROM = date(1900, 1, 1)
ASSESSMENT_VERSION = "V3.4.2_TURNING_1"
TURN_THRESHOLD_VERSION = "V3.4.2_ROBUST_MAD_1"
COST_VERSION = "V3.4.2_LINEAR_BPS_1"
BACKTEST_VERSION = "V3.4.2_FROZEN_FORECAST_REPLAY_1"
FIXED_DCA_BENCHMARK = "FIXED_5PP_WEEKLY_EXPOSURE_RAMP"
POSITION_GRID = 5
MAX_BATCHES = 4
MAX_SINGLE_CHANGE_PP = 20
DEFAULT_FIRST_BATCH_PP = 10
MINIMUM_COOLDOWN_SESSIONS = 5
TRANSACTION_COST_BPS = 10
DEGRADED_MAX_POSITION_PP = 30
ALLOWED_RATIOS = ("7:3", "6:4", "5:5", "4:6", "3:7")


class V342PolicyError(RuntimeError):
    pass


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default))


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(float(value), 8)))


def _snap5(value: float) -> int:
    return max(0, min(100, int(round(float(value) / POSITION_GRID) * POSITION_GRID)))


def _mad(values: Sequence[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    if not len(array):
        return 0.0
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _ema(values: Sequence[float], span: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.empty(len(array), dtype=float)
    if not len(array):
        return output
    alpha = 2.0 / (span + 1.0)
    output[0] = array[0]
    for index in range(1, len(array)):
        output[index] = alpha * array[index] + (1.0 - alpha) * output[index - 1]
    return output


def _macd(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closes = np.asarray(values, dtype=float)
    dif = _ema(closes, 12) - _ema(closes, 26)
    dea = _ema(dif, 9)
    return dif, dea, 2.0 * (dif - dea)


def _scaled_market_ohlc(row: MarketPrice) -> Mapping[str, float]:
    """Return one internally consistent ETF OHLC bar on the stored qfq scale.

    The adjusted close is still the direct ETF series stored by the market-data
    layer; no index, FX, NAV, or premium/discount proxy is introduced.  When a
    qfq factor exists it must scale O/H/L as well as close, otherwise ATR and
    candlestick extrema would mix incompatible units.
    """

    raw_close = float(row.close_price)
    if raw_close <= 0.0:
        raise V342PolicyError("market price close must be positive")
    adjusted_close = float(row.adjusted_close_price or raw_close)
    factor = adjusted_close / raw_close
    return {
        "open": float(row.open_price or raw_close) * factor,
        "high": float(row.high_price or raw_close) * factor,
        "low": float(row.low_price or raw_close) * factor,
        "close": adjusted_close,
    }


@dataclass(frozen=True)
class TurningCandidate:
    signal_kind: str
    turn_kind: str
    candidate_week: int
    window_start_date: date
    window_end_date: date
    classification: str
    local_extremum_met: bool
    direction_reversal_met: bool
    prominence_value: float
    minimum_prominence: float
    persistence_periods: int
    minimum_persistence_periods: int
    move_magnitude: float
    minimum_move_magnitude: float
    neutral_threshold: float
    confirmation_periods: int
    confirmation_status: str
    evidence: Mapping[str, Any]

    @property
    def candidate_hash(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class TurningAssessmentResult:
    window_extrema: Mapping[str, Any]
    candidates: tuple[TurningCandidate, ...]
    price_turn_status: str
    dif_turn_status: str
    consistency_status: str
    top_bottom_conflict: bool
    criteria: Mapping[str, Any]
    input_hash: str
    assessment_hash: str


@dataclass(frozen=True)
class PositionState:
    position_percent: int
    source_kind: str
    event_id: int | None = None
    effective_date: date | None = None
    updated_at: datetime | None = None
    recorded_at: datetime | None = None
    direction: str | None = None
    change_percent: int | None = None


@dataclass(frozen=True)
class PolicyDecision:
    current_position: int
    uncapped_target: int
    target_position: int
    next_executable_position: int
    action: str
    total_change: int
    batches: tuple[Mapping[str, Any], ...]
    fund_etf_ratio: str
    cooldown_eligible: bool
    observed_confirmation_met: bool
    reasons: tuple[str, ...]
    policy_status: str


@dataclass(frozen=True)
class BacktestResult:
    metrics: Mapping[str, Any]
    points: tuple[Mapping[str, Any], ...]


def _history_statistics(
    history: Sequence[Mapping[str, Any]],
    anchor_close: float,
) -> Mapping[str, float]:
    closes = np.asarray([float(row["close"]) for row in history[-60:]], dtype=float)
    if len(closes) < 4:
        closes = np.asarray([anchor_close] * 4, dtype=float)
    dif, _dea, _histogram = _macd(closes)
    d1 = np.diff(dif)[-52:]
    dif_scale = max(1.4826 * _mad(d1), 1e-10 * anchor_close)
    returns = np.diff(np.log(np.maximum(closes, 1e-12)))[-52:]
    return_scale = max(1.4826 * _mad(returns), 0.003)
    true_ranges: list[float] = []
    for index, row in enumerate(history[-14:]):
        close = float(row["close"])
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        previous = close if index == 0 else float(history[-14:][index - 1]["close"])
        true_ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
    atr_ratio = (float(np.mean(true_ranges)) / anchor_close) if true_ranges else 0.0
    return {
        "dif_scale": dif_scale,
        "dif_epsilon": 0.25 * dif_scale,
        "return_scale": return_scale,
        "minimum_prominence": max(1.5 * return_scale, 0.5 * atr_ratio),
    }


def assess_turning_points(
    representative_ohlcv: Sequence[Mapping[str, Any]],
    indicators: Sequence[Mapping[str, Any]],
    historical_weekly: Sequence[Mapping[str, Any]],
) -> TurningAssessmentResult:
    """Separate extrema from confirmed price/DIF turns using frozen inputs."""

    if len(representative_ohlcv) != 13 or len(indicators) != 13:
        raise V342PolicyError("turning assessment requires exactly 13 forecast weeks")
    closes = np.asarray([float(row["close"]) for row in representative_ohlcv])
    highs = np.asarray([float(row.get("high", row["close"])) for row in representative_ohlcv])
    lows = np.asarray([float(row.get("low", row["close"])) for row in representative_ohlcv])
    if np.any(~np.isfinite(closes)) or np.any(closes <= 0.0):
        raise V342PolicyError("forecast closes must be finite and positive")
    if (
        np.any(~np.isfinite(highs))
        or np.any(~np.isfinite(lows))
        or np.any(lows <= 0.0)
        or np.any(highs < lows)
    ):
        raise V342PolicyError("forecast high/low values must be finite, positive, and ordered")
    anchor_close = (
        float(historical_weekly[-1]["close"])
        if historical_weekly
        else float(representative_ohlcv[0]["open"])
    )
    thresholds = _history_statistics(historical_weekly, anchor_close)
    dates = [date.fromisoformat(str(row["week_end"])) for row in representative_ohlcv]
    candidates: list[TurningCandidate] = []

    low_index = int(np.argmin(lows))
    high_index = int(np.argmax(highs))
    extrema = {
        "lowest": {
            "kind": "WINDOW_LOW",
            "week": low_index + 1,
            "date": dates[low_index].isoformat(),
            "price": float(lows[low_index]),
            "close": float(closes[low_index]),
            "is_valid_turn": False,
        },
        "highest": {
            "kind": "WINDOW_HIGH",
            "week": high_index + 1,
            "date": dates[high_index].isoformat(),
            "price": float(highs[high_index]),
            "close": float(closes[high_index]),
            "is_valid_turn": False,
        },
        "semantics": (
            "13-week OHLC intraperiod lowest low and highest high; "
            "not automatically tradable turning points"
        ),
    }
    for index, kind in ((low_index, "BOTTOM"), (high_index, "TOP")):
        candidates.append(
            TurningCandidate(
                signal_kind="PRICE",
                turn_kind=kind,
                candidate_week=index + 1,
                window_start_date=dates[index],
                window_end_date=dates[index],
                classification="WINDOW_EXTREME",
                local_extremum_met=False,
                direction_reversal_met=False,
                prominence_value=0.0,
                minimum_prominence=float(thresholds["minimum_prominence"]),
                persistence_periods=0,
                minimum_persistence_periods=1,
                move_magnitude=0.0,
                minimum_move_magnitude=float(thresholds["return_scale"]),
                neutral_threshold=0.0,
                confirmation_periods=0,
                confirmation_status="NOT_REQUIRED",
                evidence={"endpoint_or_global_extreme": True},
            )
        )

    price_turns: list[TurningCandidate] = []
    sr = float(thresholds["return_scale"])
    min_prominence = float(thresholds["minimum_prominence"])
    for index in range(1, 12):
        before = math.log(closes[index] / closes[index - 1])
        after = math.log(closes[index + 1] / closes[index])
        for kind in ("BOTTOM", "TOP"):
            bottom = kind == "BOTTOM"
            local = (
                closes[index] < closes[index - 1] and closes[index] < closes[index + 1]
                if bottom
                else closes[index] > closes[index - 1] and closes[index] > closes[index + 1]
            )
            reversal = (
                before <= -sr and after >= sr
                if bottom
                else before >= sr and after <= -sr
            )
            prominence = (
                min(
                    math.log(closes[index - 1] / closes[index]),
                    math.log(closes[index + 1] / closes[index]),
                )
                if bottom
                else min(
                    math.log(closes[index] / closes[index - 1]),
                    math.log(closes[index] / closes[index + 1]),
                )
            )
            move = abs(before) + abs(after)
            has_following = index + 2 < len(closes)
            persistence = False
            if has_following:
                persistence = (
                    closes[index + 2] >= closes[index] * math.exp(0.5 * sr)
                    if bottom
                    else closes[index + 2] <= closes[index] * math.exp(-0.5 * sr)
                )
            qualifies = local and reversal and prominence >= min_prominence and move >= 2 * sr
            status = "CONFIRMED" if qualifies and persistence else "PENDING" if qualifies else "REJECTED"
            if qualifies or local:
                candidate = TurningCandidate(
                    signal_kind="PRICE",
                    turn_kind=kind,
                    candidate_week=index + 1,
                    window_start_date=dates[max(0, index - 1)],
                    window_end_date=dates[min(12, index + 1)],
                    classification="VALID_TURN" if status == "CONFIRMED" else "CANDIDATE",
                    local_extremum_met=bool(local),
                    direction_reversal_met=bool(reversal),
                    prominence_value=max(0.0, prominence),
                    minimum_prominence=min_prominence,
                    persistence_periods=1 if persistence else 0,
                    minimum_persistence_periods=1,
                    move_magnitude=move,
                    minimum_move_magnitude=2 * sr,
                    neutral_threshold=sr,
                    confirmation_periods=1 if persistence else 0,
                    confirmation_status=status,
                    evidence={
                        "before_log_return": before,
                        "after_log_return": after,
                        "following_period_available": bool(has_following),
                        "following_period_held": bool(persistence),
                    },
                )
                candidates.append(candidate)
                if status == "CONFIRMED":
                    price_turns.append(candidate)

    dif_values = np.asarray([float(row["dif"]) for row in indicators])
    dea_values = np.asarray([float(row["dea"]) for row in indicators])
    macd_values = np.asarray([float(row["macd"]) for row in indicators])
    d1 = np.asarray([float(row["dif_first_change"]) for row in indicators])
    d2 = np.asarray([float(row["dif_second_change"]) for row in indicators])
    epsilon = float(thresholds["dif_epsilon"])
    dif_turns: list[TurningCandidate] = []
    for index in range(1, 13):
        for kind in ("BOTTOM", "TOP"):
            bottom = kind == "BOTTOM"
            crossed = (
                d1[index - 1] <= -epsilon and d1[index] >= epsilon and d2[index] > 0.0
                if bottom
                else d1[index - 1] >= epsilon and d1[index] <= -epsilon and d2[index] < 0.0
            )
            if not crossed:
                continue
            has_following = index + 1 < len(d1)
            confirmed = False
            if has_following:
                confirmed = (
                    d1[index + 1] >= 0.5 * epsilon
                    and dif_values[index + 1] > dif_values[index]
                    and dea_values[index + 1] >= dea_values[index]
                    and macd_values[index + 1] > macd_values[index]
                    if bottom
                    else d1[index + 1] <= -0.5 * epsilon
                    and dif_values[index + 1] < dif_values[index]
                    and dea_values[index + 1] <= dea_values[index]
                    and macd_values[index + 1] < macd_values[index]
                )
            status = "CONFIRMED" if confirmed else "PENDING" if not has_following else "REJECTED"
            candidate = TurningCandidate(
                signal_kind="DIF",
                turn_kind=kind,
                candidate_week=index + 1,
                window_start_date=dates[max(0, index - 1)],
                window_end_date=dates[min(12, index + 1)],
                classification="VALID_TURN" if confirmed else "CANDIDATE",
                local_extremum_met=True,
                direction_reversal_met=True,
                prominence_value=abs(float(d1[index] - d1[index - 1])),
                minimum_prominence=2.0 * epsilon,
                persistence_periods=1 if confirmed else 0,
                minimum_persistence_periods=1,
                move_magnitude=abs(float(d1[index] - d1[index - 1])),
                minimum_move_magnitude=2.0 * epsilon,
                neutral_threshold=epsilon,
                confirmation_periods=1 if confirmed else 0,
                confirmation_status=status,
                evidence={
                    "previous_dif_first_change": float(d1[index - 1]),
                    "current_dif_first_change": float(d1[index]),
                    "dif_second_change": float(d2[index]),
                    "following_period_available": bool(has_following),
                    "dea_confirmation": bool(confirmed),
                    "macd_confirmation": bool(confirmed),
                },
            )
            candidates.append(candidate)
            if confirmed:
                dif_turns.append(candidate)

    # One signal/kind/week has one auditable row.  If a global window extreme
    # is also a local candidate, retain the more informative classification.
    priority = {"WINDOW_EXTREME": 0, "CANDIDATE": 1, "VALID_TURN": 2}
    unique_candidates: dict[tuple[str, str, int], TurningCandidate] = {}
    for candidate in candidates:
        key = (candidate.signal_kind, candidate.turn_kind, candidate.candidate_week)
        previous = unique_candidates.get(key)
        if previous is None or priority[candidate.classification] > priority[previous.classification]:
            unique_candidates[key] = candidate
    candidates = sorted(
        unique_candidates.values(),
        key=lambda item: (item.candidate_week, item.signal_kind, item.turn_kind),
    )

    price_kinds = {candidate.turn_kind for candidate in price_turns}
    dif_kinds = {candidate.turn_kind for candidate in dif_turns}
    ambiguous = len(price_kinds) > 1 or len(dif_kinds) > 1
    conflict = bool(price_kinds and dif_kinds and price_kinds != dif_kinds)
    if ambiguous:
        consistency = "AMBIGUOUS_TWO_SIDED"
    elif not price_turns or not dif_turns:
        consistency = "UNCONFIRMED"
    elif conflict:
        consistency = "TYPE_CONFLICT"
    else:
        price_turn = max(price_turns, key=lambda item: item.prominence_value)
        dif_turn = max(dif_turns, key=lambda item: item.prominence_value)
        consistency = (
            "TEMPORALLY_CONSISTENT"
            if abs(price_turn.candidate_week - dif_turn.candidate_week) <= 2
            else "TIME_CONFLICT"
        )
    price_status = (
        "CONFIRMED" if price_turns else "PENDING" if any(
            item.signal_kind == "PRICE" and item.confirmation_status == "PENDING"
            for item in candidates
        ) else "NO_VALID_TURN"
    )
    dif_status = (
        "CONFIRMED" if dif_turns else "PENDING" if any(
            item.signal_kind == "DIF" and item.confirmation_status == "PENDING"
            for item in candidates
        ) else "NO_VALID_TURN"
    )
    criteria = {
        "version": TURN_THRESHOLD_VERSION,
        **thresholds,
        "minimum_price_persistence_periods": 1,
        "minimum_dif_confirmation_periods": 1,
        "consistency_max_gap_weeks": 2,
        "endpoint_semantics": "window_extreme_only",
    }
    input_payload = {
        "representative_ohlcv": list(representative_ohlcv),
        "indicators": list(indicators),
        "historical_weekly": list(historical_weekly[-60:]),
        "criteria": criteria,
    }
    input_hash = _hash(input_payload)
    result_payload = {
        "window_extrema": extrema,
        "candidates": [asdict(item) for item in candidates],
        "price_turn_status": price_status,
        "dif_turn_status": dif_status,
        "consistency_status": consistency,
        "top_bottom_conflict": conflict,
        "criteria": criteria,
        "input_hash": input_hash,
    }
    return TurningAssessmentResult(
        window_extrema=extrema,
        candidates=tuple(candidates),
        price_turn_status=price_status,
        dif_turn_status=dif_status,
        consistency_status=consistency,
        top_bottom_conflict=conflict,
        criteria=criteria,
        input_hash=input_hash,
        assessment_hash=_hash(result_payload),
    )


def _confirmed_turn_window(
    assessment: TurningAssessmentResult,
) -> tuple[str, date, date] | None:
    if assessment.consistency_status != "TEMPORALLY_CONSISTENT":
        return None
    confirmed = [
        item
        for item in assessment.candidates
        if item.classification == "VALID_TURN" and item.confirmation_status == "CONFIRMED"
    ]
    kinds = {item.turn_kind for item in confirmed}
    if len(kinds) != 1 or {item.signal_kind for item in confirmed} < {"PRICE", "DIF"}:
        return None
    kind = next(iter(kinds))
    strongest = {
        signal: max(
            (item for item in confirmed if item.signal_kind == signal),
            key=lambda item: item.prominence_value,
        )
        for signal in ("PRICE", "DIF")
    }
    # Do not act before both independent signals' valid windows begin.  The
    # intersection is the conservative agreement window; a two-week-consistent
    # non-overlap falls back to the union end and remains conditional.
    window_start = max(item.window_start_date for item in strongest.values())
    window_end = min(item.window_end_date for item in strongest.values())
    if window_end < window_start:
        window_end = max(item.window_end_date for item in strongest.values())
    return kind, window_start, window_end


def _confirmed_kind(assessment: TurningAssessmentResult) -> str | None:
    confirmed = _confirmed_turn_window(assessment)
    return None if confirmed is None else confirmed[0]


def _ratio(up: float, down: float) -> str:
    edge = up - down
    if edge >= 0.30:
        return "7:3"
    if edge >= 0.10:
        return "6:4"
    if edge > -0.10:
        return "5:5"
    if edge > -0.30:
        return "4:6"
    return "3:7"


def _split_batches(amount: int, first_limit: int) -> tuple[int, ...]:
    if amount <= 0:
        return ()
    if amount % POSITION_GRID:
        raise V342PolicyError("position delta must use the five-point grid")
    output: list[int] = []
    remaining = amount
    while remaining and len(output) < MAX_BATCHES:
        limit = first_limit if not output else MAX_SINGLE_CHANGE_PP
        later_slots = MAX_BATCHES - len(output) - 1
        minimum_reserved = min(remaining, later_slots * POSITION_GRID)
        value = min(limit, remaining - minimum_reserved if remaining > minimum_reserved else remaining)
        value = max(POSITION_GRID, (value // POSITION_GRID) * POSITION_GRID)
        output.append(value)
        remaining -= value
    if remaining:
        output[-1] += remaining
    return tuple(output)


def build_policy_decision(
    *,
    current_position: int,
    horizon_probabilities: Mapping[str | int, Any],
    reliability_score: float,
    health_status: str,
    assessment: TurningAssessmentResult,
    valuation_percentile: float | None = None,
    cooldown_eligible: bool = True,
    observed_confirmation_met: bool = True,
    anchor_date: date,
) -> PolicyDecision:
    """Map a forecast to a constrained plan; probability alone cannot trade."""

    current = _snap5(current_position)
    horizon = horizon_probabilities.get(13, horizon_probabilities.get("13", {}))
    calibrated = horizon.get("calibrated", horizon) if isinstance(horizon, Mapping) else {}
    up = float(calibrated.get("up", 0.0))
    down = float(calibrated.get("down", 0.0))
    sideways = float(calibrated.get("sideways", max(0.0, 1.0 - up - down)))
    edge = up - down
    reasons = [f"13w direction edge={edge:.4f}", f"sideways={sideways:.4f}"]
    uncertain = abs(edge) < 0.10 or sideways >= max(up, down) or reliability_score < 40.0
    if uncertain:
        raw_target = current
        reasons.append("uncertain state: hold")
    elif edge >= 0.35:
        raw_target = 80
    elif edge >= 0.20:
        raw_target = 70
    elif edge >= 0.10:
        raw_target = 60
    elif edge <= -0.35:
        raw_target = 20
    elif edge <= -0.20:
        raw_target = 30
    else:
        raw_target = 40
    uncapped = _snap5(raw_target)
    confirmed_turn = _confirmed_turn_window(assessment)
    confirmed_kind = None if confirmed_turn is None else confirmed_turn[0]
    target = uncapped
    if valuation_percentile is not None:
        if valuation_percentile >= 90.0:
            target = min(target, 70 if confirmed_kind != "TOP" else 30)
            reasons.append("high valuation risk cap")
        elif valuation_percentile <= 15.0 and target > current and confirmed_kind != "BOTTOM":
            target = min(target, current + 10)
            reasons.append("low valuation without confirmed reversal: probe only")
    if health_status in {"MODEL_DEGRADED", "MODEL_OUT_OF_DISTRIBUTION"}:
        target = min(current, DEGRADED_MAX_POSITION_PP) if current > DEGRADED_MAX_POSITION_PP else current
        reasons.append("degraded/OOD model: hold or reduce only")
    elif target > current and confirmed_kind != "BOTTOM":
        target = current
        reasons.append("increase blocked until price and DIF bottom agree")
    elif target < current and confirmed_kind not in {"TOP", None}:
        target = current
        reasons.append("reduction direction conflicts with confirmed bottom")
    if assessment.consistency_status in {
        "TYPE_CONFLICT", "TIME_CONFLICT", "AMBIGUOUS_TWO_SIDED"
    } and target > current:
        target = current
        reasons.append("turning conflict: no increased exposure")
    target = _snap5(target)
    strong = (
        assessment.consistency_status == "TEMPORALLY_CONSISTENT"
        and reliability_score >= 70.0
        and abs(edge) >= 0.35
    )
    first_limit = MAX_SINGLE_CHANGE_PP if strong else DEFAULT_FIRST_BATCH_PP
    delta = target - current
    maximum_plan_change = first_limit + (MAX_BATCHES - 1) * MAX_SINGLE_CHANGE_PP
    if abs(delta) > maximum_plan_change:
        target = current + (maximum_plan_change if delta > 0 else -maximum_plan_change)
        target = _snap5(target)
        delta = target - current
        reasons.append("four-batch safety horizon capped the conditional plan")
    action = "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD"
    amounts = _split_batches(abs(delta), first_limit) if delta else ()
    if confirmed_turn is None:
        first_window_start = anchor_date + timedelta(days=7)
        first_window_end = anchor_date + timedelta(days=20)
    else:
        _, first_window_start, first_window_end = confirmed_turn
    first_ready = (
        cooldown_eligible
        and observed_confirmation_met
        and anchor_date >= first_window_start
    )
    if amounts and not observed_confirmation_met:
        reasons.append("first batch waits for observed market confirmation")
    first_executable = amounts[0] if amounts and first_ready else 0
    next_position = current + (first_executable if action == "BUY" else -first_executable)
    batches: list[Mapping[str, Any]] = []
    running = current
    for index, amount in enumerate(amounts):
        running += amount if action == "BUY" else -amount
        batches.append(
            {
                "batch_number": index + 1,
                "action": action,
                "change_pp": amount,
                "target_after_pp": running,
                "window_start_date": (
                    first_window_start + timedelta(days=7 * index)
                ).isoformat(),
                "window_end_date": (
                    first_window_end + timedelta(days=7 * index)
                ).isoformat(),
                "initial_state": (
                    "ELIGIBLE" if index == 0 and first_ready else "WAITING_CONFIRMATION"
                ),
                "trigger": {
                    "requires_previous_executed": index > 0,
                    "minimum_cooldown_sessions": MINIMUM_COOLDOWN_SESSIONS,
                    "required_turn_kind": "BOTTOM" if action == "BUY" else "TOP",
                    "requires_price_dif_agreement": True,
                    "requires_latest_real_indicator_confirmation": True,
                },
                "invalidation": {
                    "opposite_confirmed_turn": True,
                    "model_health_degraded_blocks_increase": action == "BUY",
                    "window_expiry_cancels": True,
                },
            }
        )
    ratio = _ratio(up, down)
    if ratio not in ALLOWED_RATIOS:
        raise V342PolicyError("fund/ETF ratio escaped the approved set")
    return PolicyDecision(
        current_position=current,
        uncapped_target=uncapped,
        target_position=target,
        next_executable_position=_snap5(next_position),
        action=action,
        total_change=abs(delta),
        batches=tuple(batches),
        fund_etf_ratio=ratio,
        cooldown_eligible=cooldown_eligible,
        observed_confirmation_met=observed_confirmation_met,
        reasons=tuple(reasons),
        policy_status="HOLD_UNCERTAIN" if action == "HOLD" else "CONDITIONAL_PLAN",
    )


def observed_market_confirmation(
    historical_weekly: Sequence[Mapping[str, Any]],
    *,
    action: str,
    dif_epsilon: float,
) -> bool:
    """Confirm a batch only from complete market data available at the anchor."""

    if action == "HOLD" or len(historical_weekly) < 30:
        return False
    closes = [float(row["close"]) for row in historical_weekly]
    dif, dea, histogram = _macd(closes)
    d1 = np.diff(dif)
    if len(d1) < 3:
        return False
    epsilon = max(float(dif_epsilon), 1e-12)
    if action == "BUY":
        return bool(
            d1[-3] <= -epsilon
            and d1[-2] >= epsilon
            and d1[-1] >= 0.5 * epsilon
            and dif[-1] > dif[-2]
            and dea[-1] >= dea[-2]
            and histogram[-1] > histogram[-2]
            and closes[-1] >= closes[-2]
        )
    return bool(
        d1[-3] >= epsilon
        and d1[-2] <= -epsilon
        and d1[-1] <= -0.5 * epsilon
        and dif[-1] < dif[-2]
        and dea[-1] <= dea[-2]
        and histogram[-1] < histogram[-2]
        and closes[-1] <= closes[-2]
    )


def transition_batch_state(
    *,
    current_state: str,
    previous_executed: bool,
    cooldown_sessions: int,
    confirmation_met: bool,
    invalidated: bool,
    expired: bool,
    execution_observed: bool,
    not_started: bool = False,
) -> str:
    """Deterministic append-only state-machine transition."""

    terminal = {"CANCELLED", "EXPIRED", "SUPERSEDED", "EXECUTED_OBSERVED"}
    if current_state in terminal:
        return current_state
    if invalidated:
        return "CANCELLED"
    if expired:
        return "EXPIRED"
    if not_started:
        return "WAITING_CONFIRMATION"
    if execution_observed and current_state == "ELIGIBLE":
        return "EXECUTED_OBSERVED"
    if not previous_executed or cooldown_sessions < MINIMUM_COOLDOWN_SESSIONS:
        return "DEFERRED"
    return "ELIGIBLE" if confirmation_met else "DEFERRED"


def backtest_policy(
    decisions: Sequence[Mapping[str, Any]],
    weekly_prices: Sequence[Mapping[str, Any]],
    *,
    transaction_cost_bps: int = TRANSACTION_COST_BPS,
    decision_builder: Callable[
        [Mapping[str, Any], int], PolicyDecision | Mapping[str, Any]
    ] | None = None,
) -> BacktestResult:
    """Replay the complete conditional policy with next-week-open execution.

    A decision is formed only after the anchor week's close.  An eligible
    batch is therefore filled at the following weekly bar's open.  Overnight
    close-to-open return remains on the old position and open-to-close return
    uses the new position, preventing an untradeable close-to-close gain from
    being assigned to a newly recommended position.
    """

    if transaction_cost_bps < 0:
        raise V342PolicyError("transaction cost must be nonnegative")
    prices = sorted(weekly_prices, key=lambda row: str(row["date"]))
    if len(prices) < 2:
        raise V342PolicyError("backtest requires at least two weekly prices")
    by_anchor = {str(item["anchor_date"]): item for item in decisions}
    matched_indices = [
        index for index, row in enumerate(prices) if str(row["date"]) in by_anchor
    ]
    if not matched_indices:
        raise V342PolicyError("no decision anchor matches the supplied weekly prices")
    start_index = min(matched_indices)
    replay_sessions: set[date] = set()
    for row in prices:
        close_day = date.fromisoformat(str(row["date"]))
        open_day = date.fromisoformat(str(row.get("open_date", row["date"])))
        declared_sessions = row.get("trading_dates")
        if declared_sessions:
            replay_sessions.update(date.fromisoformat(str(value)) for value in declared_sessions)
        else:
            cursor = open_day
            while cursor <= close_day:
                if cursor.weekday() < 5:
                    replay_sessions.add(cursor)
                cursor += timedelta(days=1)
    ordered_replay_sessions = tuple(sorted(replay_sessions))
    position = 0
    strategy_equity = 1.0
    buy_hold_equity = 1.0
    fixed_equity = 1.0
    peak = 1.0
    turnover = 0.0
    total_cost = 0.0
    points: list[Mapping[str, Any]] = []
    returns: list[float] = []
    active_plan: dict[str, Any] | None = None
    last_execution_date: date | None = None
    fixed_position = 0
    for index in range(start_index, len(prices) - 1):
        previous = prices[index]
        current = prices[index + 1]
        previous_date = date.fromisoformat(str(previous["date"]))
        current_date = date.fromisoformat(str(current["date"]))
        current_open_date = date.fromisoformat(
            str(current.get("open_date", current["date"]))
        )
        if current_open_date > current_date:
            raise V342PolicyError("weekly open_date cannot follow its close date")
        policy_events: list[Mapping[str, Any]] = []

        decision_input = by_anchor.get(str(previous["date"]))
        if decision_input is not None:
            if active_plan is not None:
                for prior in active_plan["batches"]:
                    if prior["state"] not in {
                        "CANCELLED", "EXPIRED", "SUPERSEDED", "EXECUTED_OBSERVED"
                    }:
                        prior["state"] = "SUPERSEDED"
                        policy_events.append(
                            {
                                "event": "SUPERSEDED_BY_NEW_DECISION",
                                "batch_number": prior["batch_number"],
                                "forecast_id": active_plan["forecast_id"],
                            }
                        )
            built = (
                decision_builder(decision_input, position)
                if decision_builder is not None
                else decision_input
            )
            built_payload = asdict(built) if isinstance(built, PolicyDecision) else dict(built)
            batch_payloads = [dict(item) for item in built_payload.get("batches", ())]
            legacy_immediate = False
            if not batch_payloads:
                legacy_target = int(built_payload.get("next_executable_position", position))
                if legacy_target != position:
                    legacy_immediate = True
                    legacy_action = "BUY" if legacy_target > position else "SELL"
                    batch_payloads = [
                        {
                            "batch_number": 1,
                            "action": legacy_action,
                            "change_pp": abs(legacy_target - position),
                            "target_after_pp": legacy_target,
                            "window_start_date": previous_date.isoformat(),
                            "window_end_date": current_date.isoformat(),
                            "initial_state": "WAITING_CONFIRMATION",
                            "trigger": {},
                        }
                    ]
            active_plan = {
                "forecast_id": int(decision_input["forecast_id"]),
                "action": str(built_payload.get("action", "HOLD")),
                "cooldown_eligible": bool(built_payload.get("cooldown_eligible", True)),
                "dif_epsilon": float(decision_input.get("dif_epsilon", 0.0)),
                "legacy_immediate": legacy_immediate,
                "batches": [
                    {
                        **item,
                        "state": str(item.get("initial_state", "WAITING_CONFIRMATION")),
                        "executed_date": None,
                    }
                    for item in batch_payloads
                ],
            }
            policy_events.append(
                {
                    "event": "POLICY_DECISION_FORMED_AT_CLOSE",
                    "forecast_id": active_plan["forecast_id"],
                    "batch_count": len(active_plan["batches"]),
                }
            )

        pending_execution: dict[str, Any] | None = None
        if active_plan is not None:
            for batch_index, batch in enumerate(active_plan["batches"]):
                if batch["state"] in {
                    "CANCELLED", "EXPIRED", "SUPERSEDED", "EXECUTED_OBSERVED"
                }:
                    continue
                dependency = (
                    None if batch_index == 0 else active_plan["batches"][batch_index - 1]
                )
                previous_executed = dependency is None or dependency["state"] == "EXECUTED_OBSERVED"
                if dependency is None:
                    if last_execution_date is None:
                        cooldown_sessions = (
                            MINIMUM_COOLDOWN_SESSIONS
                            if active_plan["cooldown_eligible"]
                            else 0
                        )
                    else:
                        cooldown_sessions = sum(
                            last_execution_date < session_date <= current_open_date
                            for session_date in ordered_replay_sessions
                        )
                elif dependency["executed_date"] is None:
                    cooldown_sessions = 0
                else:
                    executed_date = date.fromisoformat(str(dependency["executed_date"]))
                    cooldown_sessions = sum(
                        executed_date < session_date <= current_open_date
                        for session_date in ordered_replay_sessions
                    )
                window_start = date.fromisoformat(str(batch["window_start_date"]))
                window_end = date.fromisoformat(str(batch["window_end_date"]))
                # The window constrains the actual fill date, not merely the
                # prior close at which eligibility was observed.  A signal on
                # the last window day cannot authorize an out-of-window fill
                # at the following week's open.
                not_started = current_open_date < window_start
                expired = current_open_date > window_end
                confirmation = (
                    True
                    if active_plan["legacy_immediate"]
                    else observed_market_confirmation(
                        prices[: index + 1],
                        action=str(batch["action"]),
                        dif_epsilon=float(active_plan["dif_epsilon"]),
                    )
                )
                old_state = str(batch["state"])
                new_state = transition_batch_state(
                    current_state=old_state,
                    previous_executed=previous_executed,
                    cooldown_sessions=cooldown_sessions,
                    confirmation_met=confirmation,
                    invalidated=False,
                    expired=expired,
                    execution_observed=False,
                    not_started=not_started,
                )
                batch["state"] = new_state
                if new_state != old_state:
                    policy_events.append(
                        {
                            "event": "BATCH_STATE_TRANSITION",
                            "batch_number": batch["batch_number"],
                            "from_state": old_state,
                            "to_state": new_state,
                            "evaluation_date": previous_date.isoformat(),
                        }
                    )
                if new_state == "ELIGIBLE":
                    pending_execution = {
                        "batch": batch,
                        "forecast_id": active_plan["forecast_id"],
                    }
                    break

        old_position = position
        next_open = float(current.get("open", current["close"]))
        previous_close = float(previous["close"])
        next_close = float(current["close"])
        if min(previous_close, next_open, next_close) <= 0.0:
            raise V342PolicyError("backtest prices must be positive")
        overnight_return = next_open / previous_close - 1.0
        change = 0
        executed_forecast_id: int | None = None
        if pending_execution is not None:
            batch = pending_execution["batch"]
            target = _snap5(int(batch["target_after_pp"]))
            change = target - position
            position = target
            batch["state"] = "EXECUTED_OBSERVED"
            batch["executed_date"] = current_open_date.isoformat()
            last_execution_date = current_open_date
            executed_forecast_id = int(pending_execution["forecast_id"])
            policy_events.append(
                {
                    "event": "SIMULATED_BATCH_FILLED_AT_NEXT_WEEK_OPEN",
                    "batch_number": batch["batch_number"],
                    "execution_date": current_open_date.isoformat(),
                    "target_after_pp": position,
                }
            )
        session_return = next_close / next_open - 1.0
        gross = (
            (1.0 + old_position / 100.0 * overnight_return)
            * (1.0 + position / 100.0 * session_return)
            - 1.0
        )
        one_way_turnover = abs(change) / 100.0
        cost = one_way_turnover * transaction_cost_bps / 10_000.0
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        strategy_equity *= 1.0 + net
        period_return = next_close / previous_close - 1.0
        buy_hold_equity *= 1.0 + period_return
        old_fixed_position = fixed_position
        fixed_position = min(100, fixed_position + 5)
        fixed_period_return = (
            (1.0 + old_fixed_position / 100.0 * overnight_return)
            * (1.0 + fixed_position / 100.0 * session_return)
            - 1.0
        )
        fixed_equity *= 1.0 + fixed_period_return
        peak = max(peak, strategy_equity)
        drawdown = strategy_equity / peak - 1.0
        turnover += one_way_turnover
        total_cost += cost
        returns.append(net)
        point = {
            "sequence": len(points) + 1,
            "date": str(current["date"]),
            "forecast_id": executed_forecast_id,
            "position_pp": position,
            "position_change_pp": change,
            "gross_return": gross,
            "net_return": net,
            "turnover": one_way_turnover,
            "transaction_cost": cost,
            "strategy_equity": strategy_equity,
            "strategy_drawdown": drawdown,
            "buy_hold_equity": buy_hold_equity,
            "fixed_dca_equity": fixed_equity,
            "old_position_pp": old_position,
            "overnight_return": overnight_return,
            "open_to_close_return": session_return,
            "policy_events": policy_events,
        }
        point["point_hash"] = _hash(point)
        points.append(point)
    annualized = (
        math.sqrt(52.0) * float(np.mean(returns)) / float(np.std(returns, ddof=1))
        if len(returns) > 1 and float(np.std(returns, ddof=1)) > 0.0
        else 0.0
    )
    strategy_return = strategy_equity - 1.0
    buy_hold_return = buy_hold_equity - 1.0
    fixed_return = fixed_equity - 1.0
    metrics = {
        "version": BACKTEST_VERSION,
        "realized_return": strategy_return,
        "maximum_drawdown": min(float(row["strategy_drawdown"]) for row in points),
        "risk_adjusted_return": annualized,
        "risk_adjusted_metric": "ANNUALIZED_WEEKLY_SHARPE_RF_0",
        "turnover": turnover,
        "transaction_cost": total_cost,
        "transaction_cost_bps": transaction_cost_bps,
        "buy_hold_return": buy_hold_return,
        "fixed_dca_return": fixed_return,
        "fixed_dca_semantics": FIXED_DCA_BENCHMARK,
        "excess_vs_buy_hold": strategy_return - buy_hold_return,
        "excess_vs_fixed_dca": strategy_return - fixed_return,
        "decision_count": len(decisions),
        "point_count": len(points),
        "execution_timing": (
            "decision and eligibility after week t close; fill at week t+1 open; "
            "old position receives close-to-open return and new position receives open-to-close return"
        ),
        "conditional_batch_replay": True,
        "cooldown_replay": "exact stored trading-session dates counted from each open fill",
    }
    return BacktestResult(metrics=metrics, points=tuple(points))


class V342TurningPolicyService:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        calendar: CalendarProvider,
        now_provider: Callable[[], datetime] = utc_now,
    ) -> None:
        self.session_factory = session_factory
        self.calendar = calendar
        self.now_provider = now_provider

    @staticmethod
    def validate_market(market: str) -> None:
        """Keep chart-only instruments outside every direct policy entry point."""

        if market not in MODEL_MARKETS:
            raise V342PolicyError(
                f"V3.4.2 policy supports only 399006 and 159941, got {market!r}"
            )

    @staticmethod
    def _policy_config(*, retrospective: bool = False) -> Mapping[str, Any]:
        return {
            "policy_version": (
                RETROSPECTIVE_POLICY_VERSION if retrospective else POLICY_VERSION
            ),
            "evaluation_mode": "RETROSPECTIVE" if retrospective else "LIVE_OOS",
            "effective_from_date": (
                RETROSPECTIVE_EFFECTIVE_FROM if retrospective else POLICY_RELEASE_DATE
            ).isoformat(),
            "turning_assessment_version": ASSESSMENT_VERSION,
            "turn_threshold_version": TURN_THRESHOLD_VERSION,
            "cost_version": COST_VERSION,
            "position_grid_pp": POSITION_GRID,
            "maximum_batches": MAX_BATCHES,
            "max_single_change_pp": MAX_SINGLE_CHANGE_PP,
            "default_first_batch_pp": DEFAULT_FIRST_BATCH_PP,
            "minimum_cooldown_sessions": MINIMUM_COOLDOWN_SESSIONS,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "degraded_max_position_pp": DEGRADED_MAX_POSITION_PP,
            "uncertain_action": "HOLD",
            "fixed_dca_benchmark": FIXED_DCA_BENCHMARK,
        }

    def _ensure_policy(
        self,
        session: Session,
        market: str,
        *,
        retrospective: bool = False,
    ) -> V342PolicyVersion:
        version = RETROSPECTIVE_POLICY_VERSION if retrospective else POLICY_VERSION
        effective_from = (
            RETROSPECTIVE_EFFECTIVE_FROM if retrospective else POLICY_RELEASE_DATE
        )
        status = "RETROSPECTIVE" if retrospective else "CHAMPION"
        row = session.scalar(
            select(V342PolicyVersion).where(
                V342PolicyVersion.protocol_version == PROTOCOL_VERSION,
                V342PolicyVersion.model_market == market,
                V342PolicyVersion.version == version,
            )
        )
        config = dict(self._policy_config(retrospective=retrospective))
        digest = _hash(config)
        if row is not None:
            if (
                row.config_hash != digest
                or row.effective_from_date != effective_from
                or row.status != status
            ):
                raise V342PolicyError("frozen V3.4.2 policy configuration changed")
            return row
        row = V342PolicyVersion(
            id=f"{PROTOCOL_VERSION}:{market}:{version}",
            protocol_version=PROTOCOL_VERSION,
            model_market=market,
            version=version,
            parent_policy_id=None,
            status=status,
            effective_from_date=effective_from,
            max_single_change_pp=MAX_SINGLE_CHANGE_PP,
            minimum_cooldown_sessions=MINIMUM_COOLDOWN_SESSIONS,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            degraded_max_position_pp=DEGRADED_MAX_POSITION_PP,
            uncertain_action="HOLD",
            config_json=config,
            config_hash=digest,
            created_at=self.now_provider(),
        )
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def _position(
        session: Session,
        instrument_id: int,
        *,
        as_of: date | None = None,
    ) -> PositionState:
        query = (
            select(V2PositionEvent, V2PositionSnapshot)
            .join(
                V2PositionSnapshot,
                V2PositionSnapshot.position_event_id == V2PositionEvent.id,
            )
            .where(V2PositionEvent.instrument_id == instrument_id)
        )
        if as_of is not None:
            query = query.where(V2PositionEvent.operation_date <= as_of)
        result = session.execute(
            query.order_by(
                V2PositionEvent.operation_date.desc(),
                V2PositionEvent.sequence.desc(),
                V2PositionEvent.created_at.desc(),
                V2PositionEvent.id.desc(),
            ).limit(1)
        ).first()
        if result is None:
            return PositionState(0, "DEFAULT_CLEARED")
        event, snapshot = result
        return PositionState(
            position_percent=_snap5(float(snapshot.position_percent)),
            source_kind="V2_POSITION_EVENT",
            event_id=event.id,
            effective_date=event.operation_date,
            updated_at=event.updated_at,
            recorded_at=event.created_at,
            direction=event.direction,
            change_percent=_snap5(float(event.change_percent)),
        )

    @staticmethod
    def _weekly_history(
        session: Session,
        instrument_id: int,
        anchor: date,
        *,
        through: date | None = None,
    ) -> list[Mapping[str, Any]]:
        end = through or anchor
        weekly_rows = session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.timeframe == "weekly",
                MarketPrice.trade_date <= end,
            )
            .order_by(MarketPrice.trade_date)
        ).all()
        daily = session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= end,
            )
            .order_by(MarketPrice.trade_date)
        ).all()
        grouped: dict[tuple[int, int], list[MarketPrice]] = {}
        for row in daily:
            iso = row.trade_date.isocalendar()
            grouped.setdefault((iso.year, iso.week), []).append(row)
        if weekly_rows:
            output = []
            for row in weekly_rows:
                scaled = _scaled_market_ohlc(row)
                iso = row.trade_date.isocalendar()
                daily_rows = grouped.get((iso.year, iso.week), [])
                open_date = daily_rows[0].trade_date if daily_rows else row.trade_date
                output.append(
                    {
                        "date": row.trade_date.isoformat(),
                        "close_date": row.trade_date.isoformat(),
                        "open_date": open_date.isoformat(),
                        "trading_dates": [item.trade_date.isoformat() for item in daily_rows],
                        "open_date_source": (
                            "DAILY_SESSION_CALENDAR" if daily_rows else "WEEK_END_FALLBACK"
                        ),
                        **scaled,
                        "volume": float(row.volume or 0),
                    }
                )
            return output
        output: list[Mapping[str, Any]] = []
        for rows in grouped.values():
            scaled_rows = [_scaled_market_ohlc(row) for row in rows]
            output.append(
                {
                    "date": rows[-1].trade_date.isoformat(),
                    "close_date": rows[-1].trade_date.isoformat(),
                    "open_date": rows[0].trade_date.isoformat(),
                    "trading_dates": [row.trade_date.isoformat() for row in rows],
                    "open_date_source": "DAILY_SESSION_CALENDAR",
                    "open": scaled_rows[0]["open"],
                    "high": max(row["high"] for row in scaled_rows),
                    "low": min(row["low"] for row in scaled_rows),
                    "close": scaled_rows[-1]["close"],
                    "volume": sum(float(row.volume or 0) for row in rows),
                }
            )
        return output

    @staticmethod
    def _valuation(snapshot: V341FeatureSnapshot) -> float | None:
        weekly = dict(snapshot.feature_json.get("weekly", snapshot.feature_json))
        values = [
            float(value)
            for key in ("pe_expanding_percentile", "pb_expanding_percentile")
            if (value := weekly.get(key)) is not None
        ]
        return float(np.mean(values)) if values else None

    def _cooldown_eligible(self, market: str, position: PositionState, anchor: date) -> bool:
        if position.effective_date is None:
            return True
        if position.effective_date >= anchor:
            return False
        sessions = tuple(self.calendar.sessions(market, position.effective_date, anchor))
        return len(sessions) - 1 >= MINIMUM_COOLDOWN_SESSIONS

    def _persist_assessment(
        self,
        session: Session,
        forecast: V341Forecast,
        policy: V342PolicyVersion,
        result: TurningAssessmentResult,
    ) -> V342TurningAssessment:
        existing = session.scalar(
            select(V342TurningAssessment).where(
                V342TurningAssessment.forecast_id == forecast.id,
                V342TurningAssessment.policy_version_id == policy.id,
                V342TurningAssessment.assessment_version == ASSESSMENT_VERSION,
            )
        )
        if existing is not None:
            if existing.assessment_hash != result.assessment_hash:
                raise V342PolicyError("frozen turning assessment cannot be overwritten")
            return existing
        row = V342TurningAssessment(
            id=f"v342-turn:{forecast.id}:{policy.config_hash[:16]}",
            protocol_version=PROTOCOL_VERSION,
            model_market=forecast.model_market,
            forecast_id=forecast.id,
            policy_version_id=policy.id,
            forecast_anchor_date=forecast.forecast_anchor_date,
            assessment_version=ASSESSMENT_VERSION,
            window_extrema_json=dict(result.window_extrema),
            price_turn_status=result.price_turn_status,
            dif_turn_status=result.dif_turn_status,
            consistency_status=result.consistency_status,
            top_bottom_conflict=result.top_bottom_conflict,
            criteria_json=dict(result.criteria),
            input_hash=result.input_hash,
            assessment_hash=result.assessment_hash,
            created_at=self.now_provider(),
        )
        session.add(row)
        session.flush()
        for candidate in result.candidates:
            session.add(
                V342TurningCandidate(
                    assessment_id=row.id,
                    signal_kind=candidate.signal_kind,
                    turn_kind=candidate.turn_kind,
                    candidate_week=candidate.candidate_week,
                    window_start_date=candidate.window_start_date,
                    window_end_date=candidate.window_end_date,
                    classification=candidate.classification,
                    local_extremum_met=bool(candidate.local_extremum_met),
                    direction_reversal_met=bool(candidate.direction_reversal_met),
                    prominence_value=_decimal(candidate.prominence_value),
                    minimum_prominence=_decimal(candidate.minimum_prominence),
                    persistence_periods=candidate.persistence_periods,
                    minimum_persistence_periods=candidate.minimum_persistence_periods,
                    move_magnitude=_decimal(candidate.move_magnitude),
                    minimum_move_magnitude=_decimal(candidate.minimum_move_magnitude),
                    neutral_threshold=_decimal(candidate.neutral_threshold),
                    confirmation_periods=candidate.confirmation_periods,
                    confirmation_status=candidate.confirmation_status,
                    evidence_json=_jsonable(candidate.evidence),
                    candidate_hash=candidate.candidate_hash,
                    created_at=self.now_provider(),
                )
            )
        session.flush()
        return row

    def _supersede_previous_runs(
        self,
        session: Session,
        *,
        market: str,
        new_run_id: str,
        evaluation_date: date,
    ) -> None:
        terminal = {"CANCELLED", "EXPIRED", "SUPERSEDED", "EXECUTED_OBSERVED"}
        old_batches = session.scalars(
            select(V342PolicyBatch)
            .join(V342PolicyRun, V342PolicyRun.id == V342PolicyBatch.policy_run_id)
            .where(
                V342PolicyRun.model_market == market,
                V342PolicyRun.id != new_run_id,
            )
        ).all()
        for batch in old_batches:
            latest = session.scalar(
                select(V342PolicyBatchEvent)
                .where(V342PolicyBatchEvent.batch_id == batch.id)
                .order_by(V342PolicyBatchEvent.event_at.desc(), V342PolicyBatchEvent.id.desc())
                .limit(1)
            )
            if latest is None or latest.to_state in terminal:
                continue
            payload = {
                "batch_id": batch.id,
                "from_state": latest.to_state,
                "to_state": "SUPERSEDED",
                "new_policy_run_id": new_run_id,
                "evaluated_anchor_date": evaluation_date.isoformat(),
            }
            session.add(
                V342PolicyBatchEvent(
                    batch_id=batch.id,
                    from_state=latest.to_state,
                    to_state="SUPERSEDED",
                    evaluated_anchor_date=evaluation_date,
                    event_at=self.now_provider(),
                    trigger_snapshot_json=payload,
                    reason_json={
                        "reason": "new frozen policy run supersedes all older open batches"
                    },
                    event_hash=_hash(payload),
                    created_at=self.now_provider(),
                )
            )
        session.flush()

    def analyze(self, market: str) -> Mapping[str, Any]:
        self.validate_market(market)
        with self.session_factory() as session:
            forecast = session.scalar(
                select(V341Forecast)
                .where(V341Forecast.model_market == market)
                .order_by(V341Forecast.forecast_anchor_date.desc(), V341Forecast.id.desc())
                .limit(1)
            )
            if forecast is None:
                raise V342PolicyError("no frozen V3.4.1 forecast is available")
            snapshot = session.get(V341FeatureSnapshot, forecast.feature_snapshot_id)
            if snapshot is None:
                raise V342PolicyError("forecast feature snapshot is missing")
            policy = self._ensure_policy(session, market)
            history = self._weekly_history(
                session, forecast.target_instrument_id, forecast.forecast_anchor_date
            )
            assessment_result = assess_turning_points(
                forecast.representative_ohlcv_json,
                forecast.indicator_path_json,
                history,
            )
            assessment = self._persist_assessment(
                session, forecast, policy, assessment_result
            )
            position = self._position(session, forecast.target_instrument_id)
            cooldown = self._cooldown_eligible(
                market, position, forecast.forecast_anchor_date
            )
            provisional = build_policy_decision(
                current_position=position.position_percent,
                horizon_probabilities=forecast.horizon_probabilities_json,
                reliability_score=float(forecast.model_reliability_score),
                health_status=forecast.health_status,
                assessment=assessment_result,
                valuation_percentile=self._valuation(snapshot),
                cooldown_eligible=cooldown,
                anchor_date=forecast.forecast_anchor_date,
            )
            observed_confirmation = observed_market_confirmation(
                history,
                action=provisional.action,
                dif_epsilon=float(assessment_result.criteria["dif_epsilon"]),
            )
            decision = build_policy_decision(
                current_position=position.position_percent,
                horizon_probabilities=forecast.horizon_probabilities_json,
                reliability_score=float(forecast.model_reliability_score),
                health_status=forecast.health_status,
                assessment=assessment_result,
                valuation_percentile=self._valuation(snapshot),
                cooldown_eligible=cooldown,
                observed_confirmation_met=observed_confirmation,
                anchor_date=forecast.forecast_anchor_date,
            )
            identity = {
                "forecast_id": forecast.id,
                "forecast_hash": forecast.forecast_hash,
                "policy_id": policy.id,
                "policy_hash": policy.config_hash,
                "assessment_hash": assessment.assessment_hash,
                "position": asdict(position),
            }
            identity_hash = _hash(identity)
            existing = session.scalar(
                select(V342PolicyRun).where(
                    V342PolicyRun.protocol_version == PROTOCOL_VERSION,
                    V342PolicyRun.model_market == market,
                    V342PolicyRun.input_identity_hash == identity_hash,
                )
            )
            if existing is not None:
                return dict(existing.result_json)
            result = _jsonable({
                "protocol_version": PROTOCOL_VERSION,
                "market": market,
                "forecast_id": forecast.id,
                "forecast_anchor_date": forecast.forecast_anchor_date.isoformat(),
                "forecast_hash": forecast.forecast_hash,
                "forecast_model_version_id": forecast.model_version_id,
                "policy_version": policy.version,
                "policy_hash": policy.config_hash,
                "turning_assessment": {
                    "window_extrema": dict(assessment_result.window_extrema),
                    "price_turn_status": assessment_result.price_turn_status,
                    "dif_turn_status": assessment_result.dif_turn_status,
                    "consistency_status": assessment_result.consistency_status,
                    "top_bottom_conflict": assessment_result.top_bottom_conflict,
                    "criteria": dict(assessment_result.criteria),
                    "candidates": [asdict(item) for item in assessment_result.candidates],
                },
                "position_source": asdict(position),
                "decision": asdict(decision),
                "semantics": {
                    "window_extrema_are_not_trading_turns": True,
                    "reliability_is_not_probability": True,
                    "later_batches_require_observed_confirmation": True,
                    "system_does_not_auto_execute_trades": True,
                },
            })
            run_hash = _hash(result)
            run = V342PolicyRun(
                id=f"v342-policy:{identity_hash}",
                protocol_version=PROTOCOL_VERSION,
                model_market=market,
                run_kind="LIVE_ANALYSIS",
                forecast_id=forecast.id,
                policy_version_id=policy.id,
                turning_assessment_id=assessment.id,
                forecast_anchor_date=forecast.forecast_anchor_date,
                position_source_kind=position.source_kind,
                position_source_event_id=position.event_id,
                position_effective_date=position.effective_date,
                position_updated_at=position.updated_at,
                current_position_pp=decision.current_position,
                uncapped_target_pp=decision.uncapped_target,
                target_position_pp=decision.target_position,
                next_executable_position_pp=decision.next_executable_position,
                total_change_pp=decision.total_change,
                action=decision.action,
                fund_etf_ratio=decision.fund_etf_ratio,
                health_status=forecast.health_status,
                reliability_score=int(round(float(forecast.model_reliability_score))),
                cooldown_eligible=decision.cooldown_eligible,
                result_json=result,
                input_identity_hash=identity_hash,
                run_hash=run_hash,
                created_at=self.now_provider(),
            )
            session.add(run)
            session.flush()
            self._supersede_previous_runs(
                session,
                market=market,
                new_run_id=run.id,
                evaluation_date=forecast.forecast_anchor_date,
            )
            previous_row: V342PolicyBatch | None = None
            for batch in decision.batches:
                batch_payload = {
                    **dict(batch),
                    "policy_run_id": run.id,
                    "depends_on_batch_id": None if previous_row is None else previous_row.id,
                }
                batch_row = V342PolicyBatch(
                    policy_run_id=run.id,
                    batch_number=int(batch["batch_number"]),
                    action=str(batch["action"]),
                    change_pp=int(batch["change_pp"]),
                    target_after_pp=int(batch["target_after_pp"]),
                    window_start_date=date.fromisoformat(str(batch["window_start_date"])),
                    window_end_date=date.fromisoformat(str(batch["window_end_date"])),
                    initial_state=str(batch["initial_state"]),
                    depends_on_batch_id=None if previous_row is None else previous_row.id,
                    trigger_definition_json=dict(batch["trigger"]),
                    invalidation_definition_json=dict(batch["invalidation"]),
                    batch_hash=_hash(batch_payload),
                    created_at=self.now_provider(),
                )
                session.add(batch_row)
                session.flush()
                initial_event = {
                    "batch_id": batch_row.id,
                    "from_state": None,
                    "to_state": batch_row.initial_state,
                    "evaluated_anchor_date": forecast.forecast_anchor_date.isoformat(),
                    "reason": "frozen policy plan created",
                }
                session.add(
                    V342PolicyBatchEvent(
                        batch_id=batch_row.id,
                        from_state=None,
                        to_state=batch_row.initial_state,
                        evaluated_anchor_date=forecast.forecast_anchor_date,
                        event_at=self.now_provider(),
                        trigger_snapshot_json={
                            "forecast_hash": forecast.forecast_hash,
                            "assessment_hash": assessment.assessment_hash,
                        },
                        reason_json={"reason": "frozen policy plan created"},
                        event_hash=_hash(initial_event),
                        created_at=self.now_provider(),
                    )
                )
                previous_row = batch_row
            session.commit()
            return result

    def latest(self, market: str) -> Mapping[str, Any] | None:
        self.validate_market(market)
        with self.session_factory() as session:
            row = session.scalar(
                select(V342PolicyRun)
                .where(V342PolicyRun.model_market == market)
                .order_by(V342PolicyRun.forecast_anchor_date.desc(), V342PolicyRun.created_at.desc())
                .limit(1)
            )
            return None if row is None else dict(row.result_json)

    def evaluate_batches(
        self,
        market: str,
        *,
        as_of: date | None = None,
    ) -> Mapping[str, Any]:
        """Append observed transitions; never execute or edit a planned batch."""

        self.validate_market(market)

        evaluation_date = as_of or self.now_provider().date()
        with self.session_factory() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == market))
            if instrument is None:
                raise V342PolicyError(f"unknown market: {market}")
            latest_run = session.scalar(
                select(V342PolicyRun)
                .where(V342PolicyRun.model_market == market)
                .order_by(V342PolicyRun.forecast_anchor_date.desc(), V342PolicyRun.created_at.desc())
                .limit(1)
            )
            if latest_run is None:
                return {"market": market, "as_of": evaluation_date.isoformat(), "batches": []}
            rows = session.execute(
                select(V342PolicyBatch, V342PolicyRun)
                .join(V342PolicyRun, V342PolicyRun.id == V342PolicyBatch.policy_run_id)
                .where(V342PolicyRun.id == latest_run.id)
                .order_by(V342PolicyRun.forecast_anchor_date, V342PolicyBatch.batch_number)
            ).all()
            if not rows:
                return {"market": market, "as_of": evaluation_date.isoformat(), "batches": []}
            position = self._position(session, instrument.id, as_of=evaluation_date)
            history = self._weekly_history(
                session, instrument.id, evaluation_date, through=evaluation_date
            )
            output: list[Mapping[str, Any]] = []
            for batch, run in rows:
                latest_event = session.scalar(
                    select(V342PolicyBatchEvent)
                    .where(V342PolicyBatchEvent.batch_id == batch.id)
                    .order_by(V342PolicyBatchEvent.event_at.desc(), V342PolicyBatchEvent.id.desc())
                    .limit(1)
                )
                if latest_event is None:
                    raise V342PolicyError("batch has no initial append-only event")
                previous_executed = batch.depends_on_batch_id is None
                cooldown_start = run.position_effective_date or run.forecast_anchor_date
                if batch.depends_on_batch_id is not None:
                    dependency_event = session.scalar(
                        select(V342PolicyBatchEvent)
                        .where(
                            V342PolicyBatchEvent.batch_id == batch.depends_on_batch_id,
                            V342PolicyBatchEvent.to_state == "EXECUTED_OBSERVED",
                        )
                        .order_by(V342PolicyBatchEvent.event_at.desc())
                        .limit(1)
                    )
                    previous_executed = dependency_event is not None
                    if dependency_event is not None:
                        cooldown_start = dependency_event.event_at.date()
                sessions = tuple(self.calendar.sessions(market, cooldown_start, evaluation_date))
                cooldown_sessions = max(0, len(sessions) - 1)
                assessment = session.get(V342TurningAssessment, run.turning_assessment_id)
                if assessment is None:
                    raise V342PolicyError("batch turning assessment is missing")
                confirmation = observed_market_confirmation(
                    history,
                    action=batch.action,
                    dif_epsilon=float(assessment.criteria_json.get("dif_epsilon", 0.0)),
                )
                expected_direction = "increase" if batch.action == "BUY" else "decrease"
                reached_target = (
                    position.position_percent >= batch.target_after_pp
                    if batch.action == "BUY"
                    else position.position_percent <= batch.target_after_pp
                )
                execution_observed = bool(
                    latest_event.to_state == "ELIGIBLE"
                    and position.event_id is not None
                    and position.event_id != run.position_source_event_id
                    and position.effective_date is not None
                    and position.effective_date >= batch.window_start_date
                    and position.effective_date >= latest_event.evaluated_anchor_date
                    and position.recorded_at is not None
                    and position.recorded_at >= run.created_at
                    and position.direction == expected_direction
                    and (position.change_percent or 0) >= batch.change_pp
                    and reached_target
                )
                invalidated = latest_run.action not in {batch.action, "HOLD"}
                not_started = evaluation_date < batch.window_start_date
                expired = evaluation_date > batch.window_end_date
                new_state = transition_batch_state(
                    current_state=latest_event.to_state,
                    previous_executed=previous_executed,
                    cooldown_sessions=cooldown_sessions,
                    confirmation_met=confirmation,
                    invalidated=invalidated,
                    expired=expired,
                    execution_observed=execution_observed,
                    not_started=not_started,
                )
                if new_state != latest_event.to_state:
                    event_payload = {
                        "batch_id": batch.id,
                        "from_state": latest_event.to_state,
                        "to_state": new_state,
                        "evaluated_anchor_date": evaluation_date.isoformat(),
                        "position_event_id": position.event_id,
                        "position_percent": position.position_percent,
                        "previous_executed": previous_executed,
                        "cooldown_sessions": cooldown_sessions,
                        "confirmation_met": confirmation,
                        "invalidated": invalidated,
                        "expired": expired,
                        "not_started": not_started,
                    }
                    session.add(
                        V342PolicyBatchEvent(
                            batch_id=batch.id,
                            from_state=latest_event.to_state,
                            to_state=new_state,
                            evaluated_anchor_date=evaluation_date,
                            event_at=self.now_provider(),
                            trigger_snapshot_json=_jsonable(event_payload),
                            reason_json={
                                "system_does_not_auto_execute": True,
                                "execution_requires_observed_ledger_change": True,
                            },
                            event_hash=_hash(event_payload),
                            created_at=self.now_provider(),
                        )
                    )
                output.append(
                    {
                        "batch_id": batch.id,
                        "batch_number": batch.batch_number,
                        "policy_run_id": run.id,
                        "state_before": latest_event.to_state,
                        "state_after": new_state,
                        "confirmation_met": confirmation,
                        "cooldown_sessions": cooldown_sessions,
                        "previous_executed": previous_executed,
                        "execution_observed": execution_observed,
                    }
                )
            session.commit()
            return {
                "market": market,
                "as_of": evaluation_date.isoformat(),
                "position_source": _jsonable(asdict(position)),
                "batches": output,
            }

    def run_backtest(
        self,
        market: str,
        *,
        evaluation_available_through: date | None = None,
        retrospective: bool = True,
    ) -> Mapping[str, Any]:
        self.validate_market(market)
        through = evaluation_available_through or self.now_provider().date()
        with self.session_factory() as session:
            policy = self._ensure_policy(session, market, retrospective=retrospective)
            iteration_query = (
                select(V341TrainingIteration, V341Forecast)
                .join(V341Forecast, V341Forecast.id == V341TrainingIteration.forecast_id)
                .where(
                    V341TrainingIteration.protocol_version == V341_PROTOCOL_VERSION,
                    V341TrainingIteration.model_market == market,
                    V341Forecast.protocol_version == V341_PROTOCOL_VERSION,
                    V341Forecast.model_market == market,
                    V341Forecast.label_end_date <= through,
                )
                .order_by(V341TrainingIteration.weekly_iteration_number)
            )
            if not retrospective:
                iteration_query = iteration_query.where(
                    V341Forecast.forecast_anchor_date >= policy.effective_from_date
                )
            iteration_pairs = list(session.execute(iteration_query))
            numbers = [pair[0].weekly_iteration_number for pair in iteration_pairs]
            if any(right != left + 1 for left, right in zip(numbers, numbers[1:])):
                raise V342PolicyError(
                    "INCOMPLETE_UPSTREAM_FORECAST_SET: non-contiguous formal iterations"
                )
            if retrospective and numbers and numbers[0] != 1:
                raise V342PolicyError(
                    "INCOMPLETE_UPSTREAM_FORECAST_SET: retrospective replay does not start at iteration 1"
                )
            candidate_forecasts = [pair[1] for pair in iteration_pairs]
            forecasts: list[V341Forecast] = []
            evaluation_identity: list[tuple[int, int, str, date]] = []
            for forecast in candidate_forecasts:
                evaluations = session.scalars(
                    select(V341ForecastEvaluation)
                    .where(
                        V341ForecastEvaluation.forecast_id == forecast.id,
                        V341ForecastEvaluation.evaluation_available_date <= through,
                        V341ForecastEvaluation.evaluation_version == EVALUATION_VERSION,
                    )
                    .order_by(V341ForecastEvaluation.horizon_weeks)
                ).all()
                horizons = {row.horizon_weeks for row in evaluations}
                if (
                    forecast.maturity_status != "FULLY_MATURE_13W"
                    or horizons != {4, 8, 13}
                ):
                    raise V342PolicyError(
                        "INCOMPLETE_UPSTREAM_FORECAST_SET: "
                        f"forecast {forecast.id} at {forecast.forecast_anchor_date} "
                        "does not have the frozen 4/8/13 evaluation set"
                    )
                forecasts.append(forecast)
                evaluation_identity.extend(
                    (
                        forecast.id,
                        row.horizon_weeks,
                        row.evaluation_hash,
                        row.evaluation_available_date,
                    )
                    for row in evaluations
                )
            if not forecasts:
                if not retrospective:
                    session.commit()
                    return {
                        "protocol_version": PROTOCOL_VERSION,
                        "market": market,
                        "policy_version": policy.version,
                        "policy_hash": policy.config_hash,
                        "status": "NOT_DUE",
                        "reason": "no post-effective forecast has all frozen 4/8/13 evaluations",
                        "evaluation_available_through": through.isoformat(),
                    }
                raise V342PolicyError(
                    "no fully mature forecast with frozen 4/8/13 evaluations is available"
                )
            history = self._weekly_history(
                session,
                forecasts[0].target_instrument_id,
                forecasts[0].forecast_anchor_date,
                through=through,
            )
            if len(history) < 2:
                raise V342PolicyError("weekly market history is unavailable for replay")
            decisions: list[Mapping[str, Any]] = []
            replay_inputs: dict[int, Mapping[str, Any]] = {}
            for forecast in forecasts:
                snapshot = session.get(V341FeatureSnapshot, forecast.feature_snapshot_id)
                if snapshot is None:
                    raise V342PolicyError("forecast feature snapshot is missing")
                visible_history = [
                    row for row in history if str(row["date"]) <= forecast.forecast_anchor_date.isoformat()
                ]
                assessment = assess_turning_points(
                    forecast.representative_ohlcv_json,
                    forecast.indicator_path_json,
                    visible_history,
                )
                replay_inputs[forecast.id] = {
                    "forecast": forecast,
                    "assessment": assessment,
                    "valuation_percentile": self._valuation(snapshot),
                }
                decisions.append(
                    {
                        "forecast_id": forecast.id,
                        "anchor_date": forecast.forecast_anchor_date.isoformat(),
                        "dif_epsilon": float(assessment.criteria["dif_epsilon"]),
                        "forecast_hash": forecast.forecast_hash,
                        "assessment_hash": assessment.assessment_hash,
                    }
                )

            def replay_decision_builder(
                decision_input: Mapping[str, Any], current_position: int
            ) -> PolicyDecision:
                replay_input = replay_inputs[int(decision_input["forecast_id"])]
                replay_forecast = replay_input["forecast"]
                replay_assessment = replay_input["assessment"]
                return build_policy_decision(
                    current_position=current_position,
                    horizon_probabilities=replay_forecast.horizon_probabilities_json,
                    reliability_score=float(replay_forecast.model_reliability_score),
                    health_status=replay_forecast.health_status,
                    assessment=replay_assessment,
                    valuation_percentile=replay_input["valuation_percentile"],
                    cooldown_eligible=True,
                    # Eligibility is replayed point-in-time at each later close;
                    # it is not borrowed from the forecast anchor.
                    observed_confirmation_met=False,
                    anchor_date=replay_forecast.forecast_anchor_date,
                )

            first_anchor = forecasts[0].forecast_anchor_date.isoformat()
            in_replay_window = [row for row in history if str(row["date"]) >= first_anchor]
            missing_exact_open_dates = [
                str(row["date"])
                for row in in_replay_window
                if row.get("open_date_source") != "DAILY_SESSION_CALENDAR"
            ]
            if missing_exact_open_dates:
                raise V342PolicyError(
                    "exact daily-session open_date is required for policy replay: "
                    + ",".join(missing_exact_open_dates[:5])
                )
            result = backtest_policy(
                decisions,
                history,
                transaction_cost_bps=policy.transaction_cost_bps,
                decision_builder=replay_decision_builder,
            )
            forecast_set_hash = _hash(
                {
                    "forecasts": [
                        (row.id, row.forecast_hash, row.forecast_anchor_date)
                        for row in forecasts
                    ],
                    "evaluations": evaluation_identity,
                }
            )
            cost_hash = _hash(
                {"version": COST_VERSION, "transaction_cost_bps": policy.transaction_cost_bps}
            )
            price_set_hash = _hash(
                [
                    (
                        row["date"],
                        row.get("open_date"),
                        row.get("trading_dates"),
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                    )
                    for row in history
                ]
            )
            existing = session.scalar(
                select(V342StrategyBacktestRun).where(
                    V342StrategyBacktestRun.policy_version_id == policy.id,
                    V342StrategyBacktestRun.model_market == market,
                    V342StrategyBacktestRun.forecast_set_hash == forecast_set_hash,
                    V342StrategyBacktestRun.cost_model_hash == cost_hash,
                    V342StrategyBacktestRun.evaluation_available_through == through,
                    V342StrategyBacktestRun.price_set_hash == price_set_hash,
                )
            )
            evaluation_mode = "RETROSPECTIVE" if retrospective else "LIVE_OOS"
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "market": market,
                "policy_version": policy.version,
                "policy_hash": policy.config_hash,
                "forecast_set_hash": forecast_set_hash,
                "price_set_hash": price_set_hash,
                "evaluation_mode": evaluation_mode,
                "evaluation_available_through": through.isoformat(),
                "metrics": dict(result.metrics),
                "points": list(result.points),
                "leakage_audit": {
                    "decision_inputs": "frozen V3.4.1 forecast at each anchor",
                    "historical_position": "internal simulated state only",
                    "realized_prices_used_only_after_each_anchor": True,
                    "policy_application": (
                        "RETROSPECTIVE_ON_FROZEN_MODEL_FORECASTS"
                        if retrospective
                        else "POLICY_VERSION_AS_OF_LIVE_OOS"
                    ),
                    "actual_best_date_fields_used": False,
                },
            }
            if existing is not None:
                if existing.run_hash != _hash(payload):
                    raise V342PolicyError("frozen strategy backtest cannot be overwritten")
                return payload
            now = self.now_provider()
            run = V342StrategyBacktestRun(
                id=(
                    f"v342-backtest:{market}:"
                    f"{_hash((policy.id, forecast_set_hash, cost_hash, through, price_set_hash))[:32]}"
                ),
                protocol_version=PROTOCOL_VERSION,
                model_market=market,
                policy_version_id=policy.id,
                start_anchor_date=forecasts[0].forecast_anchor_date,
                end_anchor_date=forecasts[-1].forecast_anchor_date,
                evaluation_available_through=through,
                forecast_set_hash=forecast_set_hash,
                cost_model_hash=cost_hash,
                price_set_hash=price_set_hash,
                status=(
                    "COMPLETED_RETROSPECTIVE" if retrospective else "COMPLETED_LIVE_OOS"
                ),
                decision_count=len(decisions),
                realized_return=_decimal(float(result.metrics["realized_return"])),
                maximum_drawdown=_decimal(float(result.metrics["maximum_drawdown"])),
                risk_adjusted_return=_decimal(float(result.metrics["risk_adjusted_return"])),
                risk_adjusted_metric=str(result.metrics["risk_adjusted_metric"]),
                turnover=_decimal(float(result.metrics["turnover"])),
                transaction_cost=_decimal(float(result.metrics["transaction_cost"])),
                buy_hold_return=_decimal(float(result.metrics["buy_hold_return"])),
                fixed_dca_return=_decimal(float(result.metrics["fixed_dca_return"])),
                excess_vs_buy_hold=_decimal(float(result.metrics["excess_vs_buy_hold"])),
                excess_vs_fixed_dca=_decimal(float(result.metrics["excess_vs_fixed_dca"])),
                metrics_json=dict(result.metrics),
                leakage_audit_json=dict(payload["leakage_audit"]),
                run_hash=_hash(payload),
                started_at=now,
                completed_at=now,
            )
            session.add(run)
            session.flush()
            for point in result.points:
                session.add(
                    V342StrategyBacktestPoint(
                        backtest_run_id=run.id,
                        sequence=int(point["sequence"]),
                        point_date=date.fromisoformat(str(point["date"])),
                        forecast_id=point["forecast_id"],
                        position_pp=int(point["position_pp"]),
                        position_change_pp=int(point["position_change_pp"]),
                        gross_return=_decimal(float(point["gross_return"])),
                        net_return=_decimal(float(point["net_return"])),
                        turnover=_decimal(float(point["turnover"])),
                        transaction_cost=_decimal(float(point["transaction_cost"])),
                        strategy_equity=_decimal(float(point["strategy_equity"])),
                        strategy_drawdown=_decimal(float(point["strategy_drawdown"])),
                        buy_hold_equity=_decimal(float(point["buy_hold_equity"])),
                        fixed_dca_equity=_decimal(float(point["fixed_dca_equity"])),
                        event_json={
                            "execution_timing": result.metrics["execution_timing"],
                            "policy_events": list(point.get("policy_events", ())),
                            "old_position_pp": point.get("old_position_pp"),
                            "overnight_return": point.get("overnight_return"),
                            "open_to_close_return": point.get("open_to_close_return"),
                        },
                        point_hash=str(point["point_hash"]),
                    )
                )
            session.commit()
            return payload

    def run_live_oos_backtest(
        self,
        market: str,
        *,
        evaluation_available_through: date | None = None,
    ) -> Mapping[str, Any]:
        return self.run_backtest(
            market,
            evaluation_available_through=evaluation_available_through,
            retrospective=False,
        )
