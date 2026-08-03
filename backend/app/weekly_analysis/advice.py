"""Immutable, leakage-safe 13-week position advice.

The weekly signal is supplied explicitly and remains the controlling judgment.
This module uses a market-specific future trading calendar only to place
staged batches; it accepts no future price input and performs no persistence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
from threading import Lock
from typing import Literal, Mapping

from .domain import QualityIssue, QualityReport
from .features import (
    FEATURE_SET_VERSION,
    FeatureSnapshot,
    FrozenDict,
    _canonical_json_dumps,
)
from .optimizer import (
    CrossMarketError,
    StepAudit,
    TradingCalendar,
    WorkState,
    seed_model,
    validate_step_audit_acceptance,
    _validate_state_market,
)


ADVICE_VERSION = "weekly-advice-v2"
FORECAST_HORIZON_WEEKS = 13
DEFAULT_TOLERANCE_TRADING_DAYS = 3
MIN_ACTION_CONFIDENCE = Decimal("55")
MIN_ACTION_PROBABILITY = Decimal("55")

Direction = Literal["up", "down", "neutral"]
OperationSide = Literal["buy", "sell", "hold"]
BatchStatus = Literal["ready", "waiting_confirmation"]
Recommendation = Literal[
    "staged_increase",
    "staged_decrease",
    "hold",
    "wait_confirmation",
    "analysis_only",
]

_DIRECTIONS = {"up", "down", "neutral"}
_BATCH_STATUSES = {"ready", "waiting_confirmation"}
_MARKET_STATES = {
    "bearish",
    "bullish",
    "overvalued",
    "overvalued_multi_peak",
    "overvalued_strong_trend",
    "range_bound",
    "undervalued",
    "undervalued_reversal_resonance",
    "undervalued_weak_trend",
}
_MODEL_VALIDATION_CACHE_LOCK = Lock()
_MODEL_VALIDATION_CACHE: dict[str, "_ValidatedModelPrefix"] = {}


@dataclass(frozen=True, slots=True)
class _ValidatedModelPrefix:
    model: WorkState
    expected_model_version: str
    expected_weights: FrozenDict[str, Decimal]
    expected_optimizer_memory: FrozenDict[str, object]
    seen_weeks: frozenset[str]
_MODEL_VERSION_PATTERN = re.compile(r"^M(?P<number>\d{4})$")
_WORK_VERSION_PATTERN = re.compile(r"^W(?P<number>\d{4})$")
_ITERATION_ID_PATTERN = re.compile(r"^I(?P<number>\d{4})$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class AdviceBatch:
    """One calendar-bound five-point-multiple position adjustment."""

    sequence: int
    operation_side: OperationSide
    percent: int
    expected_date: date
    tolerance_trading_days: int
    confirmation_condition: str
    status: BatchStatus

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("Batch sequence must be a positive integer")
        if self.operation_side not in {"buy", "sell"}:
            raise ValueError("A batch operation_side must be buy or sell")
        if (
            isinstance(self.percent, bool)
            or not isinstance(self.percent, int)
            or self.percent <= 0
            or self.percent % 5
        ):
            raise ValueError(
                "Batch percent must be a positive multiple of five"
            )
        if not isinstance(self.expected_date, date):
            raise ValueError("Batch expected_date must be a date")
        if (
            type(self.tolerance_trading_days) is not int
            or self.tolerance_trading_days
            != DEFAULT_TOLERANCE_TRADING_DAYS
        ):
            raise ValueError(
                "Batch tolerance_trading_days must be exactly three"
            )
        if (
            not isinstance(self.confirmation_condition, str)
            or not self.confirmation_condition.strip()
        ):
            raise ValueError("Batch confirmation_condition is required")
        if self.status not in _BATCH_STATUSES:
            raise ValueError("Invalid batch status")

    @property
    def direction(self) -> OperationSide:
        return self.operation_side

    @property
    def planned_date(self) -> date:
        return self.expected_date

    def to_dict(self) -> dict[str, object]:
        return {
            "confirmation_condition": self.confirmation_condition,
            "expected_date": self.expected_date,
            "operation_side": self.operation_side,
            "percent": self.percent,
            "sequence": self.sequence,
            "status": self.status,
            "tolerance_trading_days": self.tolerance_trading_days,
        }


@dataclass(frozen=True, slots=True)
class Advice:
    """A complete immutable current-market advice snapshot."""

    instrument_code: str
    advice_version: str
    model_version: str
    feature_set_version: str
    forecast_horizon_weeks: int
    direction: Direction
    direction_probabilities: FrozenDict[str, Decimal]
    probability: Decimal
    confidence: Decimal
    market_state: str
    operation_side: OperationSide
    recommendation: Recommendation
    current_position: int | None
    target_position: int
    target_position_range: tuple[int, int]
    total_adjustment: int | None
    signed_position_change: int | None
    fund_etf_ratio: str
    batches: tuple[AdviceBatch, ...]
    data_cutoff_date: date
    source_data_max_date: date | None
    audit_notes: tuple[str, ...]
    audit_fields: FrozenDict[str, object]

    def __post_init__(self) -> None:
        if self.instrument_code not in {"399006", "NDX"}:
            raise ValueError("Invalid advice instrument_code")
        if self.advice_version != ADVICE_VERSION:
            raise ValueError("Invalid advice_version")
        if not _is_valid_model_version(self.model_version):
            raise ValueError("model_version must be M0001 or later")
        if self.feature_set_version != FEATURE_SET_VERSION:
            raise ValueError(
                f"feature_set_version must be {FEATURE_SET_VERSION}"
            )
        if (
            type(self.forecast_horizon_weeks) is not int
            or self.forecast_horizon_weeks != FORECAST_HORIZON_WEEKS
        ):
            raise ValueError("Advice horizon must be exactly 13 weeks")
        if self.direction not in _DIRECTIONS:
            raise ValueError("Invalid weekly direction")
        if not isinstance(self.direction_probabilities, Mapping):
            raise ValueError("direction_probabilities must be a mapping")
        probabilities = FrozenDict(
            {
                name: _percentage(value, f"direction_probability:{name}")
                for name, value in self.direction_probabilities.items()
            }
        )
        if set(probabilities) != _DIRECTIONS:
            raise ValueError(
                "direction_probabilities must contain up, down, and neutral"
            )
        if sum(probabilities.values(), Decimal()) != Decimal("100"):
            raise ValueError("direction probabilities must sum to 100")
        probability = _percentage(self.probability, "probability")
        confidence = _percentage(self.confidence, "confidence")
        if probability != probabilities[self.direction]:
            raise ValueError(
                "probability must match the selected direction probability"
            )
        if not isinstance(self.audit_fields, Mapping):
            raise ValueError("audit_fields must be a mapping")
        fields = FrozenDict(self.audit_fields)
        market_inputs = _validated_market_state_inputs(
            fields.get("market_state_inputs")
        )
        expected_market_state = _market_state_from_inputs(
            market_inputs,
            self.direction,
            probability,
            confidence,
        )
        if (
            self.market_state not in _MARKET_STATES
            or self.market_state != expected_market_state
        ):
            raise ValueError(
                "market_state must match audited weekly feature inputs"
            )
        expected_ratio = fund_etf_ratio(
            self.direction,
            probability,
            confidence,
        )
        if self.fund_etf_ratio != expected_ratio:
            raise ValueError(
                "fund_etf_ratio must match direction and confidence"
            )
        feature_publishable = fields.get("feature_publishable")
        if type(feature_publishable) is not bool:
            raise ValueError(
                "audit feature_publishable must be a boolean"
            )
        confirmation_met = fields.get("confirmation_conditions_met")
        if type(confirmation_met) is not bool:
            raise ValueError(
                "audit confirmation_conditions_met must be a boolean"
            )
        batch_count = fields.get("batch_count")
        if type(batch_count) is not int or batch_count < 0:
            raise ValueError(
                "audit batch_count must be a non-negative integer"
            )
        requested_target = fields.get("requested_target_position")
        _validate_quantized_position(
            requested_target,  # type: ignore[arg-type]
            "audit requested_target_position",
        )
        if fields.get("calendar_instrument_code") != self.instrument_code:
            raise ValueError(
                "audit calendar market must match advice market"
            )
        model_work_version = fields.get("model_work_version")
        if (
            not isinstance(model_work_version, str)
            or _WORK_VERSION_PATTERN.fullmatch(model_work_version) is None
        ):
            raise ValueError("audit model_work_version is invalid")
        feature_source_hash = fields.get("feature_source_hash")
        if (
            not isinstance(feature_source_hash, str)
            or _HASH_PATTERN.fullmatch(feature_source_hash) is None
        ):
            raise ValueError("audit feature_source_hash is invalid")
        if fields.get("position_quantization") != "nearest_5_half_up":
            raise ValueError("Invalid position quantization audit")
        if fields.get("weekly_probability_scale") != "0_to_100":
            raise ValueError("Invalid weekly probability audit")
        _validate_quantized_position(
            self.target_position,
            "target_position",
        )
        try:
            target_range = tuple(self.target_position_range)
        except TypeError as error:
            raise ValueError(
                "target_position_range must contain two boundaries"
            ) from error
        if len(target_range) != 2:
            raise ValueError(
                "target_position_range must contain two boundaries"
            )
        lower, upper = target_range
        _validate_quantized_position(lower, "target_position_range lower")
        _validate_quantized_position(upper, "target_position_range upper")
        if lower > upper:
            raise ValueError(
                "target_position_range lower must not exceed upper"
            )
        if lower > self.target_position or upper < self.target_position:
            raise ValueError("target_position must fall in its target range")
        expected_range = (
            max(0, self.target_position - 5),
            min(100, self.target_position + 5),
        )
        if target_range != expected_range:
            raise ValueError(
                "target_position_range must be the audited five-point range"
            )
        batches = tuple(self.batches)
        notes = tuple(self.audit_notes)
        if any(not isinstance(note, str) for note in notes):
            raise ValueError("audit_notes must contain strings")
        if any(not isinstance(batch, AdviceBatch) for batch in batches):
            raise ValueError("batches must contain AdviceBatch values")
        if len(batches) > 4:
            raise ValueError("Advice cannot contain more than four batches")
        if batch_count != len(batches):
            raise ValueError("audit batch_count must match batches")
        actionable = _is_actionable(
            feature_publishable,
            self.direction,
            probability,
            confidence,
        )
        if self.current_position is None:
            if (
                self.total_adjustment is not None
                or self.signed_position_change is not None
                or batches
            ):
                raise ValueError(
                    "Unknown current position cannot have executable batches"
                )
            expected_target = requested_target
            expected_signed = None
            expected_total = None
            expected_side: OperationSide = "hold"
            expected_recommendation: Recommendation = "analysis_only"
        else:
            _validate_quantized_position(
                self.current_position,
                "current_position",
            )
            if (
                type(self.total_adjustment) is not int
                or self.total_adjustment < 0
                or self.total_adjustment > 100
                or self.total_adjustment % 5
            ):
                raise ValueError(
                    "total_adjustment must be a five-point integer"
                )
            if (
                type(self.signed_position_change) is not int
                or not -100 <= self.signed_position_change <= 100
                or self.signed_position_change % 5
            ):
                raise ValueError(
                    "signed_position_change must be a five-point integer"
                )
            expected_target = (
                _risk_limited_target(
                    self.current_position,
                    requested_target,
                    self.market_state,
                )
                if actionable
                else self.current_position
            )
            if self.target_position != expected_target:
                raise ValueError(
                    "target_position must match audited risk controls"
                )
            expected_signed = self.target_position - self.current_position
            expected_total = abs(expected_signed)
            if self.signed_position_change != expected_signed:
                raise ValueError(
                    "signed_position_change must equal target minus current"
                )
            if self.total_adjustment != expected_total:
                raise ValueError(
                    "total_adjustment must equal the absolute position change"
                )
            if sum(batch.percent for batch in batches) != expected_total:
                raise ValueError(
                    "Batch total must conserve the requested position change"
                )
            expected_side: OperationSide = (
                "buy"
                if expected_signed > 0
                else "sell"
                if expected_signed < 0
                else "hold"
            )
            if any(
                batch.operation_side != expected_side
                for batch in batches
            ):
                raise ValueError("Every batch must match the advice side")
            if expected_side == "hold" and batches:
                raise ValueError("Hold advice cannot contain batches")
            running = self.current_position
            for batch in batches:
                running += (
                    batch.percent
                    if batch.operation_side == "buy"
                    else -batch.percent
                )
                if not 0 <= running <= 100:
                    raise ValueError(
                        "Executing batches would leave the 0-100 range"
                    )
            if running != self.target_position:
                raise ValueError("Executing all batches must reach target")
            expected_recommendation = (
                "wait_confirmation"
                if not actionable
                else "staged_increase"
                if expected_side == "buy"
                else "staged_decrease"
                if expected_side == "sell"
                else "hold"
            )
        if self.target_position != expected_target:
            raise ValueError(
                "target_position must match audited requested target"
            )
        if self.operation_side != expected_side:
            raise ValueError(
                "operation_side must match target minus current"
            )
        if self.recommendation != expected_recommendation:
            raise ValueError(
                "recommendation must match operation and confidence"
            )
        expected_percents = (
            ()
            if expected_total is None
            else _split_batches(
                expected_total,
                expected_side,
                self.market_state,
            )
        )
        if tuple(batch.percent for batch in batches) != expected_percents:
            raise ValueError(
                "batches must match audited risk apportionment"
            )
        if batches:
            if tuple(batch.sequence for batch in batches) != tuple(
                range(1, len(batches) + 1)
            ):
                raise ValueError("Batch sequence values must be contiguous")
            dates = tuple(batch.expected_date for batch in batches)
            if dates != tuple(sorted(dates)) or len(set(dates)) != len(dates):
                raise ValueError(
                    "Batch expected dates must be unique and increasing"
                )
            horizon_end = self.data_cutoff_date + timedelta(
                weeks=FORECAST_HORIZON_WEEKS
            )
            if any(
                not self.data_cutoff_date < day <= horizon_end
                for day in dates
            ):
                raise ValueError(
                    "Batch expected dates must be in the future 13-week horizon"
                )
            expected_condition = _confirmation_condition(
                expected_side,
                self.market_state,
            )
            if any(
                batch.confirmation_condition != expected_condition
                for batch in batches
            ):
                raise ValueError(
                    "Batch confirmation condition is inconsistent"
                )
            expected_statuses = (
                ("ready",) * len(batches)
                if confirmation_met
                else ("ready",)
                + ("waiting_confirmation",) * (len(batches) - 1)
            )
            if tuple(batch.status for batch in batches) != expected_statuses:
                raise ValueError(
                    "Batch status vector violates confirmation gate"
                )
        if not isinstance(self.data_cutoff_date, date):
            raise ValueError("data_cutoff_date must be a date")
        if (
            self.source_data_max_date is not None
            and not isinstance(self.source_data_max_date, date)
        ):
            raise ValueError("source_data_max_date must be a date or None")
        if (
            self.source_data_max_date is not None
            and self.source_data_max_date > self.data_cutoff_date
        ):
            raise ValueError(
                "source_data_max_date cannot be after data_cutoff_date"
            )
        object.__setattr__(self, "direction_probabilities", probabilities)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "target_position_range", target_range)
        object.__setattr__(self, "batches", batches)
        object.__setattr__(self, "audit_notes", notes)
        object.__setattr__(self, "audit_fields", fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "advice_version": self.advice_version,
            "audit_fields": dict(self.audit_fields),
            "audit_notes": self.audit_notes,
            "batches": tuple(batch.to_dict() for batch in self.batches),
            "confidence": self.confidence,
            "current_position": self.current_position,
            "data_cutoff_date": self.data_cutoff_date,
            "direction": self.direction,
            "direction_probabilities": dict(self.direction_probabilities),
            "feature_set_version": self.feature_set_version,
            "forecast_horizon_weeks": self.forecast_horizon_weeks,
            "fund_etf_ratio": self.fund_etf_ratio,
            "instrument_code": self.instrument_code,
            "market_state": self.market_state,
            "model_version": self.model_version,
            "operation_side": self.operation_side,
            "probability": self.probability,
            "recommendation": self.recommendation,
            "signed_position_change": self.signed_position_change,
            "source_data_max_date": self.source_data_max_date,
            "target_position": self.target_position,
            "target_position_range": self.target_position_range,
            "total_adjustment": self.total_adjustment,
        }

    def to_json(self) -> str:
        return _canonical_json_dumps(self.to_dict())


def fund_etf_ratio(
    direction: str,
    probability: int | Decimal,
    confidence: int | Decimal,
) -> str:
    """Return the formal five-tier fund/ETF ratio."""

    if direction not in _DIRECTIONS:
        raise ValueError("direction must be up, down, or neutral")
    direction_probability = _percentage(probability, "probability")
    confidence_value = _percentage(confidence, "confidence")
    if direction == "neutral":
        return "5:5"
    if (
        direction_probability < MIN_ACTION_PROBABILITY
        or confidence_value < MIN_ACTION_CONFIDENCE
    ):
        return "5:5"
    if direction == "up":
        if (
            direction_probability >= Decimal("70")
            and confidence_value >= Decimal("70")
        ):
            return "7:3"
        return "6:4"
    if (
        direction_probability >= Decimal("70")
        and confidence_value >= Decimal("70")
    ):
        return "3:7"
    return "4:6"


def build_advice(
    *,
    model: WorkState,
    latest_features: FeatureSnapshot,
    future_trading_calendar: TradingCalendar,
    current_position: int | None,
    target_position: int,
    direction: str,
    probability: int | Decimal,
    confidence: int | Decimal,
    direction_probabilities: Mapping[str, int | Decimal] | None = None,
    market_state: str | None = None,
    confirmation_conditions_met: bool = False,
) -> Advice:
    """Build current 13-week advice without reading any future price.

    ``direction`` and its probability are the weekly judgment. The calendar is
    used only to select future sessions inside the 13-week horizon.
    """

    _validate_boundary_types(
        model,
        latest_features,
        future_trading_calendar,
    )
    _validate_model_contract(model)
    _validate_feature_snapshot_contract(latest_features)
    instrument = model.instrument_code
    if (
        latest_features.instrument_code != instrument
        or future_trading_calendar.instrument_code != instrument
    ):
        raise CrossMarketError(
            "Model, feature snapshot, and trading calendar must match markets"
        )
    if type(confirmation_conditions_met) is not bool:
        raise ValueError(
            "confirmation_conditions_met must be a boolean"
        )
    selected_direction = _direction(direction)
    probability_value = _percentage(probability, "probability")
    confidence_value = _percentage(confidence, "confidence")
    probabilities = _direction_probability_map(
        selected_direction,
        probability_value,
        direction_probabilities,
    )
    requested_target = _quantize_position(
        target_position,
        "target_position",
    )
    quantized_current = (
        None
        if current_position is None
        else _quantize_position(current_position, "current_position")
    )
    market_inputs = _market_state_inputs(latest_features)
    resolved_market_state = _market_state_from_inputs(
        market_inputs,
        selected_direction,
        probability_value,
        confidence_value,
    )
    if market_state is not None:
        asserted_market_state = _nonempty_market_state(market_state)
        if asserted_market_state != resolved_market_state:
            raise ValueError(
                "market_state must match the state derived from weekly inputs"
            )
    ratio = fund_etf_ratio(
        selected_direction,
        probability_value,
        confidence_value,
    )

    notes = [
        "weekly_signal_controls_13_week_judgment",
        "daily_input_not_used_to_change_weekly_judgment",
        "future_prices_not_read",
        "positions_and_batches_quantized_to_nearest_5_percent",
    ]
    actionable = _is_actionable(
        latest_features.quality_report.is_publishable,
        selected_direction,
        probability_value,
        confidence_value,
    )
    if not latest_features.quality_report.is_publishable:
        notes.append("feature_snapshot_not_publishable")
    if (
        selected_direction == "neutral"
        or probability_value < MIN_ACTION_PROBABILITY
        or confidence_value < MIN_ACTION_CONFIDENCE
    ):
        notes.append("weekly_signal_requires_confirmation")
    if quantized_current is None:
        notes.append("current_position_not_set_analysis_only")

    if quantized_current is None:
        effective_target = requested_target
        total_adjustment = None
        signed_change = None
        side: OperationSide = "hold"
        recommendation: Recommendation = "analysis_only"
        batches: tuple[AdviceBatch, ...] = ()
    elif not actionable:
        effective_target = quantized_current
        total_adjustment = 0
        signed_change = 0
        side = "hold"
        recommendation = "wait_confirmation"
        batches = ()
    else:
        effective_target = _risk_limited_target(
            quantized_current,
            requested_target,
            resolved_market_state,
        )
        signed_change = effective_target - quantized_current
        total_adjustment = abs(signed_change)
        side = (
            "buy"
            if signed_change > 0
            else "sell"
            if signed_change < 0
            else "hold"
        )
        recommendation = (
            "staged_increase"
            if side == "buy"
            else "staged_decrease"
            if side == "sell"
            else "hold"
        )
        batch_percents = _split_batches(
            total_adjustment,
            side,
            resolved_market_state,
        )
        dates = _batch_dates(
            latest_features.cutoff_date,
            future_trading_calendar,
            len(batch_percents),
        )
        confirmation_condition = _confirmation_condition(
            side,
            resolved_market_state,
        )
        batches = tuple(
            AdviceBatch(
                sequence=index + 1,
                operation_side=side,  # type: ignore[arg-type]
                percent=percent,
                expected_date=dates[index],
                tolerance_trading_days=(
                    DEFAULT_TOLERANCE_TRADING_DAYS
                ),
                confirmation_condition=confirmation_condition,
                status=(
                    "ready"
                    if index == 0 or confirmation_conditions_met
                    else "waiting_confirmation"
                ),
            )
            for index, percent in enumerate(batch_percents)
        )

    target_range = (
        max(0, effective_target - 5),
        min(100, effective_target + 5),
    )
    audit_fields: FrozenDict[str, object] = FrozenDict(
        {
            "batch_count": len(batches),
            "batch_rule": "risk_front_loaded_nonincreasing_max_4",
            "calendar_instrument_code": (
                future_trading_calendar.instrument_code
            ),
            "calendar_rule": (
                "future_market_sessions_within_13_week_horizon"
            ),
            "confirmation_conditions_met": confirmation_conditions_met,
            "feature_publishable": (
                latest_features.quality_report.is_publishable
            ),
            "feature_source_hash": latest_features.source_hash,
            "market_state_inputs": market_inputs,
            "model_work_version": model.work_version,
            "position_quantization": "nearest_5_half_up",
            "requested_target_position": requested_target,
            "weekly_probability_scale": "0_to_100",
        }
    )
    return Advice(
        instrument_code=instrument,
        advice_version=ADVICE_VERSION,
        model_version=model.current_model_version,
        feature_set_version=latest_features.feature_set_version,
        forecast_horizon_weeks=FORECAST_HORIZON_WEEKS,
        direction=selected_direction,
        direction_probabilities=probabilities,
        probability=probability_value,
        confidence=confidence_value,
        market_state=resolved_market_state,
        operation_side=side,
        recommendation=recommendation,
        current_position=quantized_current,
        target_position=effective_target,
        target_position_range=target_range,
        total_adjustment=total_adjustment,
        signed_position_change=signed_change,
        fund_etf_ratio=ratio,
        batches=batches,
        data_cutoff_date=latest_features.cutoff_date,
        source_data_max_date=latest_features.source_data_max_date,
        audit_notes=tuple(notes),
        audit_fields=audit_fields,
    )


def _validate_boundary_types(
    model: WorkState,
    latest_features: FeatureSnapshot,
    future_trading_calendar: TradingCalendar,
) -> None:
    if not isinstance(model, WorkState):
        raise ValueError("model must be a validated WorkState")
    if not isinstance(latest_features, FeatureSnapshot):
        raise ValueError(
            "latest_features must be a FeatureSnapshot"
        )
    if not isinstance(future_trading_calendar, TradingCalendar):
        raise ValueError(
            "future_trading_calendar must be a TradingCalendar"
        )


def _validate_model_contract(model: WorkState) -> None:
    """Prove the effective model version/weights from its full audit chain."""

    if not _is_valid_model_version(model.current_model_version):
        raise ValueError("model version must be M0001 or later")
    _validate_state_market(model)
    seed = seed_model(model.instrument_code)
    if not model.audits:
        if model != seed:
            raise ValueError(
                "Unaudited model must exactly match its market seed"
            )
        return
    if len(model.week_records) != len(model.audits):
        raise ValueError("Model audit history and week records disagree")

    with _MODEL_VALIDATION_CACHE_LOCK:
        cached = _MODEL_VALIDATION_CACHE.get(model.instrument_code)
    if cached is not None and model is cached.model:
        return
    can_extend = (
        cached is not None
        and len(model.audits) == len(cached.model.audits) + 1
        and all(
            current is previous
            for current, previous in zip(
                model.audits[:-1], cached.model.audits
            )
        )
        and all(
            model.week_records.get(audit.week_key) is audit
            for audit in cached.model.audits
        )
    )
    if can_extend:
        assert cached is not None
        expected_work_version = cached.model.work_version
        expected_iteration_id = cached.model.iteration_id
        expected_model_version = cached.expected_model_version
        expected_weights = cached.expected_weights
        expected_optimizer_memory = cached.expected_optimizer_memory
        seen_weeks = set(cached.seen_weeks)
        audits_to_validate = model.audits[-1:]
    else:
        expected_work_version = seed.work_version
        expected_iteration_id = seed.iteration_id
        expected_model_version = seed.current_model_version
        expected_weights = seed.current_model_weights
        expected_optimizer_memory = seed.optimizer_memory
        seen_weeks = set()
        audits_to_validate = model.audits
    for audit in audits_to_validate:
        if not isinstance(audit, StepAudit):
            raise ValueError("Model audit history contains an invalid record")
        if audit.instrument_code != model.instrument_code:
            raise CrossMarketError("Model audit history crosses markets")
        if type(audit.accepted) is not bool:
            raise ValueError("Model audit accepted flag must be boolean")
        if (
            not isinstance(audit.source_hash, str)
            or _HASH_PATTERN.fullmatch(audit.source_hash) is None
            or not isinstance(audit.input_hash, str)
            or _HASH_PATTERN.fullmatch(audit.input_hash) is None
        ):
            raise ValueError("Model audit hash contract is invalid")
        if (
            type(audit.future_leakage) is not bool
            or audit.future_leakage
            or type(audit.data_integrity) is not bool
        ):
            raise ValueError("Model audit data-integrity contract is invalid")
        if audit.parent_work_version != expected_work_version:
            raise ValueError("Model audit work-version history is broken")
        expected_work_version = _next_version(
            expected_work_version,
            _WORK_VERSION_PATTERN,
            "W",
        )
        if audit.work_version != expected_work_version:
            raise ValueError("Model audit work version is inconsistent")
        expected_weights = validate_step_audit_acceptance(
            audit,
            previous_model_version=expected_model_version,
            previous_model_weights=expected_weights,
            previous_optimizer_memory=expected_optimizer_memory,
        )
        expected_model_version = audit.model_version
        expected_optimizer_memory = audit.optimizer_memory
        expected_iteration_id = _next_version(
            expected_iteration_id,
            _ITERATION_ID_PATTERN,
            "I",
        )
        if audit.iteration_id != expected_iteration_id:
            raise ValueError("Model audit iteration history is inconsistent")
        if audit.week_key in seen_weeks:
            raise ValueError("Model audit history contains duplicate weeks")
        seen_weeks.add(audit.week_key)
        if model.week_records.get(audit.week_key) != audit:
            raise ValueError("Model week record does not match audit history")

    latest = model.audits[-1]
    if (
        model.work_version != latest.work_version
        or model.parent_work_version != latest.parent_work_version
        or model.iteration_id != latest.iteration_id
        or model.current_model_version != expected_model_version
        or model.current_model_weights != expected_weights
        or model.optimizer_memory != latest.optimizer_memory
        or latest.feedback_count != len(model.feedback)
    ):
        raise ValueError(
            "Current model version or weights lack matching audit proof"
        )
    validated = _ValidatedModelPrefix(
        model=model,
        expected_model_version=expected_model_version,
        expected_weights=expected_weights,
        expected_optimizer_memory=expected_optimizer_memory,
        seen_weeks=frozenset(seen_weeks),
    )
    with _MODEL_VALIDATION_CACHE_LOCK:
        existing = _MODEL_VALIDATION_CACHE.get(model.instrument_code)
        if existing is None or len(model.audits) >= len(existing.model.audits):
            _MODEL_VALIDATION_CACHE[model.instrument_code] = validated
def _next_version(
    value: str,
    pattern: re.Pattern[str],
    prefix: str,
) -> str:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid {prefix} audit version")
    return f"{prefix}{int(match.group('number')) + 1:04d}"


def _validate_feature_snapshot_contract(
    snapshot: FeatureSnapshot,
) -> None:
    """Reject snapshots whose Task4 deep-immutability contract was bypassed."""

    if (
        type(snapshot.features) is not FrozenDict
        or type(snapshot.availability) is not FrozenDict
        or type(snapshot.audit_fields) is not FrozenDict
    ):
        raise ValueError(
            "FeatureSnapshot mappings must be immutable FrozenDict values"
        )
    if not isinstance(snapshot.quality_report, QualityReport):
        raise ValueError("FeatureSnapshot quality_report is invalid")
    report = snapshot.quality_report
    if (
        type(report.is_publishable) is not bool
        or type(report.issues) is not tuple
        or any(not isinstance(issue, QualityIssue) for issue in report.issues)
    ):
        raise ValueError(
            "FeatureSnapshot quality report must be deeply immutable"
        )
    if snapshot.instrument_code not in {"399006", "NDX"}:
        raise ValueError("FeatureSnapshot market is unsupported")
    if snapshot.feature_set_version != FEATURE_SET_VERSION:
        raise ValueError(
            f"latest_features must use {FEATURE_SET_VERSION}"
        )
    if (
        not isinstance(snapshot.source_hash, str)
        or _HASH_PATTERN.fullmatch(snapshot.source_hash) is None
    ):
        raise ValueError("FeatureSnapshot source_hash is invalid")
    if not isinstance(snapshot.cutoff_date, date):
        raise ValueError("FeatureSnapshot cutoff_date must be a date")
    if (
        snapshot.source_data_max_date is not None
        and (
            not isinstance(snapshot.source_data_max_date, date)
            or snapshot.source_data_max_date > snapshot.cutoff_date
        )
    ):
        raise ValueError(
            "source_data_max_date cannot be after feature cutoff_date"
        )
    if set(snapshot.features) - set(snapshot.availability):
        raise ValueError(
            "Every feature must carry an immutable availability status"
        )
    for name, value in snapshot.features.items():
        if isinstance(value, Decimal) and not value.is_finite():
            raise ValueError(f"Feature {name} must be finite")
        if isinstance(value, tuple):
            if any(not isinstance(item, str) for item in value):
                raise ValueError(f"Feature {name} tuple must contain strings")
        elif value is not None and not isinstance(
            value,
            (Decimal, str, int, bool),
        ):
            raise ValueError(f"Feature {name} has a mutable or invalid value")
    if any(
        not isinstance(name, str) or not isinstance(status, str)
        for name, status in snapshot.availability.items()
    ):
        raise ValueError("Feature availability entries must be strings")
    if any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, (str, int))
        for name, value in snapshot.audit_fields.items()
    ):
        raise ValueError("Feature audit fields violate Task4 contract")


def _direction(value: str) -> Direction:
    if value not in _DIRECTIONS:
        raise ValueError("direction must be up, down, or neutral")
    return value  # type: ignore[return-value]


def _is_valid_model_version(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = _MODEL_VERSION_PATTERN.fullmatch(value)
    return match is not None and int(match.group("number")) >= 1


def _is_actionable(
    feature_publishable: bool,
    direction: Direction,
    probability: Decimal,
    confidence: Decimal,
) -> bool:
    return (
        feature_publishable
        and direction != "neutral"
        and probability >= MIN_ACTION_PROBABILITY
        and confidence >= MIN_ACTION_CONFIDENCE
    )


def _percentage(
    value: int | Decimal,
    field_name: str,
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"{field_name} must be an integer or Decimal")
    result = Decimal(value)
    if not result.is_finite() or not Decimal() <= result <= Decimal("100"):
        raise ValueError(f"{field_name} must be between 0 and 100")
    return result


def _position_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if not 0 <= value <= 100:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return value


def _quantize_position(value: int, field_name: str) -> int:
    raw = _position_integer(value, field_name)
    units = (
        Decimal(raw) / Decimal("5")
    ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(units * Decimal("5"))


def _validate_quantized_position(value: int, field_name: str) -> None:
    _position_integer(value, field_name)
    if value % 5:
        raise ValueError(f"{field_name} must be a multiple of five")


def _direction_probability_map(
    direction: Direction,
    probability: Decimal,
    supplied: Mapping[str, int | Decimal] | None,
) -> FrozenDict[str, Decimal]:
    if supplied is not None:
        if set(supplied) != _DIRECTIONS:
            raise ValueError(
                "direction_probabilities must contain up, down, and neutral"
            )
        result = FrozenDict(
            {
                name: _percentage(
                    value,
                    f"direction_probability:{name}",
                )
                for name, value in supplied.items()
            }
        )
        if sum(result.values(), Decimal()) != Decimal("100"):
            raise ValueError("direction probabilities must sum to 100")
        if result[direction] != probability:
            raise ValueError(
                "probability must match direction_probabilities"
            )
        return result
    remainder = Decimal("100") - probability
    if direction == "up":
        values = {
            "up": probability,
            "down": Decimal(),
            "neutral": remainder,
        }
    elif direction == "down":
        values = {
            "up": Decimal(),
            "down": probability,
            "neutral": remainder,
        }
    else:
        half = remainder / Decimal("2")
        values = {
            "up": half,
            "down": half,
            "neutral": probability,
        }
    return FrozenDict(values)


def _feature_decimal(
    snapshot: FeatureSnapshot,
    name: str,
) -> Decimal | None:
    raw = snapshot.features.get(name)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, Decimal)):
        return None
    value = Decimal(raw)
    return value if value.is_finite() else None


def _market_state_inputs(
    snapshot: FeatureSnapshot,
) -> FrozenDict[str, object]:
    return FrozenDict(
        {
            "black_point": snapshot.features.get("black_point") is True,
            "black_strength": _feature_decimal(
                snapshot,
                "black_strength",
            ),
            "golden_point": snapshot.features.get("golden_point") is True,
            "golden_strength": _feature_decimal(
                snapshot,
                "golden_strength",
            ),
            "valuation_percentile": _feature_decimal(
                snapshot,
                "valuation_percentile",
            ),
        }
    )


def _validated_market_state_inputs(
    raw: object,
) -> FrozenDict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("audit market_state_inputs must be a mapping")
    required = {
        "black_point",
        "black_strength",
        "golden_point",
        "golden_strength",
        "valuation_percentile",
    }
    if set(raw) != required:
        raise ValueError("audit market_state_inputs keys are incomplete")
    black = raw["black_point"]
    golden = raw["golden_point"]
    if type(black) is not bool or type(golden) is not bool:
        raise ValueError("audited point flags must be booleans")
    valuation = _optional_audit_decimal(
        raw["valuation_percentile"],
        "valuation_percentile",
        maximum=Decimal("100"),
    )
    black_strength = _optional_audit_decimal(
        raw["black_strength"],
        "black_strength",
        maximum=Decimal("5"),
    )
    golden_strength = _optional_audit_decimal(
        raw["golden_strength"],
        "golden_strength",
        maximum=Decimal("5"),
    )
    return FrozenDict(
        {
            "black_point": black,
            "black_strength": black_strength,
            "golden_point": golden,
            "golden_strength": golden_strength,
            "valuation_percentile": valuation,
        }
    )


def _optional_audit_decimal(
    value: object,
    field_name: str,
    *,
    maximum: Decimal,
) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"audited {field_name} must be numeric or None")
    result = Decimal(value)
    if not result.is_finite() or not Decimal() <= result <= maximum:
        raise ValueError(f"audited {field_name} is outside its range")
    return result


def _market_state_from_inputs(
    inputs: Mapping[str, object],
    direction: Direction,
    probability: Decimal,
    confidence: Decimal,
) -> str:
    valuation = inputs["valuation_percentile"]
    golden = inputs["golden_point"]
    black = inputs["black_point"]
    golden_strength = inputs["golden_strength"]
    black_strength = inputs["black_strength"]
    assert valuation is None or isinstance(valuation, Decimal)
    assert type(golden) is bool
    assert type(black) is bool
    assert golden_strength is None or isinstance(
        golden_strength,
        Decimal,
    )
    assert black_strength is None or isinstance(
        black_strength,
        Decimal,
    )
    reversal_resonance = golden and (
        golden_strength is None or golden_strength >= Decimal("5")
    )
    multi_peak = black and (
        black_strength is None or black_strength >= Decimal("5")
    )
    if valuation is not None and valuation >= Decimal("70"):
        if multi_peak:
            return "overvalued_multi_peak"
        if (
            direction == "up"
            and probability >= Decimal("70")
            and confidence >= Decimal("70")
        ):
            return "overvalued_strong_trend"
        return "overvalued"
    if valuation is not None and valuation <= Decimal("30"):
        if reversal_resonance:
            return "undervalued_reversal_resonance"
        if direction == "up":
            return "undervalued_weak_trend"
        return "undervalued"
    if direction == "up":
        return "bullish"
    if direction == "down":
        return "bearish"
    return "range_bound"


def _nonempty_market_state(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("market_state must be a non-empty string")
    return value.strip()


def _risk_limited_target(
    current: int,
    requested_target: int,
    market_state: str,
) -> int:
    if (
        market_state == "undervalued_weak_trend"
        and requested_target > current
    ):
        return min(requested_target, current + 10)
    return requested_target


def _batch_count(
    total_adjustment: int,
    market_state: str,
) -> int:
    units = total_adjustment // 5
    if units == 0:
        return 0
    if market_state in {
        "overvalued_multi_peak",
        "undervalued_reversal_resonance",
    }:
        return min(4, units)
    if market_state == "overvalued_strong_trend":
        return min(2, units)
    if market_state == "undervalued_weak_trend":
        return 1
    if total_adjustment <= 10:
        return 1
    if total_adjustment <= 25:
        return 2
    if total_adjustment <= 45:
        return 3
    return min(4, units)


def _split_batches(
    total_adjustment: int,
    side: OperationSide,
    market_state: str,
) -> tuple[int, ...]:
    if total_adjustment == 0:
        return ()
    if side not in {"buy", "sell"}:
        raise ValueError("A non-zero adjustment requires buy or sell")
    if total_adjustment % 5:
        raise ValueError("Adjustment must be a multiple of five")
    count = _batch_count(total_adjustment, market_state)
    units = total_adjustment // 5
    allocations = [1] * count
    remaining = units - count
    weights = (
        (Decimal("4"), Decimal("3"), Decimal("2"), Decimal("1"))
        if side == "sell"
        else (
            Decimal("6"),
            Decimal("6"),
            Decimal("5"),
            Decimal("3"),
        )
    )[:count]
    if remaining:
        weight_total = sum(weights, Decimal())
        raw = [
            Decimal(remaining) * weight / weight_total
            for weight in weights
        ]
        floors = [
            int(value.to_integral_value(rounding=ROUND_FLOOR))
            for value in raw
        ]
        allocations = [
            base + extra
            for base, extra in zip(allocations, floors, strict=True)
        ]
        leftover = remaining - sum(floors)
        fractional_order = sorted(
            range(count),
            key=lambda index: (
                raw[index] - Decimal(floors[index]),
                -index,
            ),
            reverse=True,
        )
        for index in fractional_order[:leftover]:
            allocations[index] += 1
    allocations.sort(reverse=True)
    result = tuple(allocation * 5 for allocation in allocations)
    if sum(result) != total_adjustment:
        raise AssertionError("Internal batch apportionment failed")
    return result


def _batch_dates(
    cutoff_date: date,
    calendar: TradingCalendar,
    count: int,
) -> tuple[date, ...]:
    if count == 0:
        return ()
    horizon_end = cutoff_date + timedelta(weeks=FORECAST_HORIZON_WEEKS)
    candidates = tuple(
        day
        for day in calendar.expected_trade_dates
        if cutoff_date < day <= horizon_end
    )
    if len(candidates) < count:
        raise ValueError(
            "Future trading calendar has too few sessions in the 13-week horizon"
        )
    indices = tuple((index * len(candidates)) // count for index in range(count))
    dates = tuple(candidates[index] for index in indices)
    if len(set(dates)) != count:
        raise AssertionError("Internal calendar scheduling failed")
    return dates


def _confirmation_condition(
    side: OperationSide,
    market_state: str,
) -> str:
    if market_state == "overvalued_multi_peak":
        return "weekly_black_point_and_peak_signals_remain_confirmed"
    if market_state == "undervalued_reversal_resonance":
        return "weekly_golden_point_and_reversal_signals_remain_confirmed"
    if side == "sell":
        return "weekly_risk_reduction_signal_remains_confirmed"
    if side == "buy":
        return "weekly_uptrend_signal_remains_confirmed"
    return "weekly_direction_and_confidence_require_confirmation"


__all__ = [
    "ADVICE_VERSION",
    "DEFAULT_TOLERANCE_TRADING_DAYS",
    "FORECAST_HORIZON_WEEKS",
    "Advice",
    "AdviceBatch",
    "build_advice",
    "fund_etf_ratio",
]
