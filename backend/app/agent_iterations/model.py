from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from typing import Mapping


ZERO = Decimal("0")
ONE_HUNDRED = Decimal("100")
COMPONENTS = ("valuation", "eda", "dip", "signal_point", "trend", "market_regime")


def _decimal(value: Decimal | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _clamp(value: Decimal, low: Decimal = Decimal("-100"), high: Decimal = ONE_HUNDRED) -> Decimal:
    return min(high, max(low, value))


def _integer(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class WeeklyObservation:
    trade_date: date
    close: Decimal
    volume: Decimal | None
    volume_ratio: Decimal | None
    dif: Decimal | None
    dea: Decimal | None
    dif_slope: Decimal | None
    dea_slope: Decimal | None
    macd_histogram: Decimal | None
    rsi: Decimal | None
    ma_20: Decimal | None
    ma_60: Decimal | None
    valuation_percentile: Decimal | None
    volatility: Decimal | None
    drawdown: Decimal | None
    golden_strength: int = 0
    black_strength: int = 0


@dataclass(frozen=True)
class PositionPlan:
    current_position: int
    target_position: int
    adjustment: int
    mode: str
    batches: list[int]
    conditional_batches: list[int]


@dataclass(frozen=True)
class WeeklyAdvice:
    as_of_date: date
    recommendation: str
    state_label: str
    total_score: Decimal
    confidence: int
    probability_up_15d: int
    current_position: int
    target_position: int
    adjustment: int
    batches: list[int]
    conditional_batches: list[int]
    fund_etf_ratio: str
    fund_allocation: int
    etf_allocation: int
    component_scores: dict[str, Decimal]
    conservative_mode: bool


@dataclass(frozen=True)
class ModelState:
    instrument_code: str
    market: str
    iteration: int
    version: str
    parent_version: str | None
    weights: dict[str, Decimal]
    confirmation_offset: int = 0
    confidence_bias: Decimal = ZERO
    feedback_count: int = 0
    recent_errors: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "instrument_code": self.instrument_code,
            "market": self.market,
            "iteration": self.iteration,
            "version": self.version,
            "parent_version": self.parent_version,
            "weights": {key: str(value) for key, value in self.weights.items()},
            "confirmation_offset": self.confirmation_offset,
            "confidence_bias": str(self.confidence_bias),
            "feedback_count": self.feedback_count,
            "recent_errors": list(self.recent_errors),
            "indicator_aliases": {"DIP": "DIF", "EDA": "DEA"},
        }


@dataclass(frozen=True)
class IterationFeedback:
    actual_direction: int
    predicted_direction: int
    deviation_days: int
    probability: Decimal
    component_scores: dict[str, Decimal]

    @property
    def direction_correct(self) -> bool:
        return self.actual_direction == self.predicted_direction


def initial_model(instrument_code: str) -> ModelState:
    if instrument_code == "399006":
        market = "CN"
        raw = {
            "valuation": 20,
            "eda": 20,
            "dip": 20,
            "signal_point": 15,
            "trend": 15,
            "market_regime": 10,
        }
    elif instrument_code == "NDX":
        market = "US"
        raw = {
            "valuation": 15,
            "eda": 22,
            "dip": 22,
            "signal_point": 12,
            "trend": 19,
            "market_regime": 10,
        }
    else:
        raise ValueError(f"Unsupported iteration instrument: {instrument_code}")
    return ModelState(
        instrument_code=instrument_code,
        market=market,
        iteration=1,
        version="M001",
        parent_version=None,
        weights={key: Decimal(value) for key, value in raw.items()},
    )


def model_from_dict(payload: Mapping[str, object]) -> ModelState:
    return ModelState(
        instrument_code=str(payload["instrument_code"]),
        market=str(payload["market"]),
        iteration=int(payload["iteration"]),
        version=str(payload["version"]),
        parent_version=(
            str(payload["parent_version"]) if payload.get("parent_version") is not None else None
        ),
        weights={
            str(key): Decimal(str(value))
            for key, value in dict(payload["weights"]).items()  # type: ignore[arg-type]
        },
        confirmation_offset=int(payload.get("confirmation_offset", 0)),
        confidence_bias=Decimal(str(payload.get("confidence_bias", "0"))),
        feedback_count=int(payload.get("feedback_count", 0)),
        recent_errors=tuple(int(value) for value in payload.get("recent_errors", [])),  # type: ignore[arg-type]
    )


def _allocate_integer(total: int, proportions: list[Decimal]) -> list[int]:
    exact = [Decimal(total) * value for value in proportions]
    allocated = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in exact]
    missing = total - sum(allocated)
    order = sorted(
        range(len(exact)),
        key=lambda index: (exact[index] - Decimal(allocated[index]), -index),
        reverse=True,
    )
    for index in order[:missing]:
        allocated[index] += 1
    return allocated


def build_position_plan(
    current_position: int,
    target_position: int,
    *,
    mode: str | None = None,
    confirmed: bool,
) -> PositionPlan:
    current = max(0, min(100, int(current_position)))
    target = max(0, min(100, int(target_position)))
    adjustment = target - current
    resolved_mode = mode or ("buy" if adjustment > 0 else "sell" if adjustment < 0 else "hold")
    magnitude = abs(adjustment)
    if magnitude < 5:
        batches: list[int] = []
    elif resolved_mode == "buy" and magnitude <= 10:
        batches = [magnitude]
    elif resolved_mode == "buy":
        batches = _allocate_integer(
            magnitude,
            [Decimal("0.30"), Decimal("0.30"), Decimal("0.25"), Decimal("0.15")],
        )
    elif magnitude <= 20:
        batches = _allocate_integer(magnitude, [Decimal("0.60"), Decimal("0.40")])
    elif magnitude <= 50:
        batches = _allocate_integer(magnitude, [Decimal("0.40"), Decimal("0.35"), Decimal("0.25")])
    else:
        batches = _allocate_integer(
            magnitude,
            [Decimal("0.40"), Decimal("0.30"), Decimal("0.20"), Decimal("0.10")],
        )
    return PositionPlan(
        current_position=current,
        target_position=target,
        adjustment=adjustment,
        mode=resolved_mode,
        batches=batches,
        conditional_batches=[] if confirmed else batches[1:],
    )


def _component_scores(observation: WeeklyObservation) -> dict[str, Decimal]:
    valuation = (
        _clamp((Decimal("50") - observation.valuation_percentile) * Decimal("2"))
        if observation.valuation_percentile is not None
        else ZERO
    )
    dea = observation.dea or ZERO
    dea_slope = observation.dea_slope or ZERO
    eda = _clamp(
        (Decimal("35") if dea > 0 else Decimal("-35") if dea < 0 else ZERO)
        + _clamp(dea_slope * Decimal("12"), Decimal("-45"), Decimal("45"))
    )
    dif = observation.dif or ZERO
    dif_slope = observation.dif_slope or ZERO
    relative = dif - dea
    dip = _clamp(
        (Decimal("35") if relative > 0 else Decimal("-35") if relative < 0 else ZERO)
        + _clamp(dif_slope * Decimal("12"), Decimal("-45"), Decimal("45"))
    )
    signal_point = _clamp(
        Decimal(observation.golden_strength - observation.black_strength) * Decimal("25")
    )
    trend_parts = []
    if observation.ma_20 is not None:
        trend_parts.append(Decimal("40") if observation.close >= observation.ma_20 else Decimal("-40"))
    if observation.ma_60 is not None:
        trend_parts.append(Decimal("40") if observation.close >= observation.ma_60 else Decimal("-40"))
    trend = sum(trend_parts, ZERO) if trend_parts else ZERO
    regime = ZERO
    if observation.rsi is not None:
        regime += _clamp((observation.rsi - Decimal("50")) * Decimal("2"), Decimal("-35"), Decimal("35"))
    if observation.volume_ratio is not None:
        regime += _clamp((observation.volume_ratio - Decimal("1")) * Decimal("50"), Decimal("-20"), Decimal("20"))
    if observation.drawdown is not None and observation.drawdown < Decimal("-0.20"):
        regime -= Decimal("25")
    return {
        "valuation": valuation,
        "eda": eda,
        "dip": dip,
        "signal_point": signal_point,
        "trend": _clamp(trend),
        "market_regime": _clamp(regime),
    }


def _target_from_score(score: Decimal) -> int:
    for threshold, target in (
        (Decimal("75"), 95),
        (Decimal("55"), 85),
        (Decimal("35"), 70),
        (Decimal("15"), 60),
        (Decimal("-15"), 50),
        (Decimal("-35"), 35),
        (Decimal("-55"), 25),
        (Decimal("-75"), 15),
    ):
        if score >= threshold:
            return target
    return 10


def _ratio_for_score(score: Decimal) -> str:
    if score >= Decimal("35"):
        return "7:3"
    if score >= Decimal("-15"):
        return "6:4"
    return "5:5"


def _split_target(target: int, ratio: str) -> tuple[int, int]:
    fund_units = Decimal(ratio.split(":")[0]) / Decimal("10")
    fund, etf = _allocate_integer(target, [fund_units, Decimal("1") - fund_units])
    return fund, etf


def evaluate_week(
    model: ModelState,
    observation: WeeklyObservation,
    *,
    current_position: int,
) -> WeeklyAdvice:
    scores = _component_scores(observation)
    total = sum(scores[key] * model.weights[key] / ONE_HUNDRED for key in COMPONENTS)
    available = sum(
        value is not None
        for value in (
            observation.dif,
            observation.dea,
            observation.rsi,
            observation.ma_20,
            observation.ma_60,
            observation.valuation_percentile,
            observation.volume,
        )
    )
    confidence = _integer(
        _clamp(
            Decimal("42")
            + abs(total) * Decimal("0.45")
            + Decimal(available * 3)
            + model.confidence_bias,
            Decimal("35"),
            Decimal("92"),
        )
    )
    errors = sum(model.recent_errors[-5:])
    conservative = errors >= 3
    base_target = _target_from_score(total)
    coefficient = (
        Decimal("0.30")
        if confidence < 50
        else Decimal("0.50")
        if confidence < 65
        else Decimal("0.70")
        if confidence < 75
        else Decimal("0.85")
        if confidence < 85
        else Decimal("1")
    )
    target = _integer(Decimal("50") + (Decimal(base_target) - Decimal("50")) * coefficient)
    target = max(10, min(95, target))
    if conservative:
        target = _integer(Decimal("50") + (Decimal(target) - Decimal("50")) * Decimal("0.5"))
    confirmed = observation.golden_strength >= 2 or observation.black_strength >= 2
    plan = build_position_plan(current_position, target, confirmed=confirmed)
    ratio = _ratio_for_score(total)
    fund, etf = _split_target(target, ratio)
    probability = max(5, min(95, _integer(Decimal("50") + total * Decimal("0.42"))))
    if total >= 35:
        state = "周线强势趋势"
    elif total <= -35:
        state = "周线弱势风险"
    elif observation.valuation_percentile is not None and observation.valuation_percentile >= 85 and total > 0:
        state = "上涨趋势中的高估风险"
    else:
        state = "周线震荡观察"
    recommendation = (
        "分批加仓"
        if plan.adjustment >= 5
        else "分批减仓"
        if plan.adjustment <= -5
        else "保持观察"
    )
    return WeeklyAdvice(
        as_of_date=observation.trade_date,
        recommendation=recommendation,
        state_label=state,
        total_score=total.quantize(Decimal("0.01")),
        confidence=confidence,
        probability_up_15d=probability,
        current_position=plan.current_position,
        target_position=plan.target_position,
        adjustment=plan.adjustment,
        batches=plan.batches,
        conditional_batches=plan.conditional_batches,
        fund_etf_ratio=ratio,
        fund_allocation=fund,
        etf_allocation=etf,
        component_scores=scores,
        conservative_mode=conservative,
    )


def update_model(model: ModelState, feedback: IterationFeedback) -> ModelState:
    weights = dict(model.weights)
    signed = {
        key: feedback.component_scores.get(key, ZERO) * Decimal(feedback.actual_direction)
        for key in COMPONENTS
    }
    receiver = max(COMPONENTS, key=lambda key: (signed[key], key))
    donor = min(COMPONENTS, key=lambda key: (signed[key], key))
    if receiver != donor and weights[donor] >= Decimal("2"):
        shift = Decimal("0.5") if feedback.direction_correct else Decimal("1")
        weights[receiver] += shift
        weights[donor] -= shift
    error = 0 if feedback.direction_correct else 1
    recent = (model.recent_errors + (error,))[-10:]
    confidence_delta = Decimal("0.25") if feedback.direction_correct else Decimal("-1")
    timing_delta = 1 if feedback.deviation_days > 0 else -1 if feedback.deviation_days < 0 else 0
    return replace(
        model,
        iteration=model.iteration + 1,
        version=f"M{model.iteration + 1:03d}",
        parent_version=model.version,
        weights=weights,
        confirmation_offset=max(-5, min(5, model.confirmation_offset + timing_delta)),
        confidence_bias=_clamp(
            model.confidence_bias + confidence_delta,
            Decimal("-15"),
            Decimal("10"),
        ),
        feedback_count=model.feedback_count + 1,
        recent_errors=recent,
    )
