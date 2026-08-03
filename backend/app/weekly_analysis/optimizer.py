"""Pure, leakage-safe weekly optimizer state and label evaluation.

This module deliberately has no persistence, task, API, or network boundary.
Every transition is represented by immutable values so callers can persist the
audit trail without giving the optimizer access to external state.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from threading import Lock
from typing import Callable, Iterable, Literal, Mapping, Sequence

from .features import (
    FrozenDict,
    _canonical_json_dumps,
)


InstrumentCode = Literal["399006", "NDX"]
Direction = Literal["up", "down", "neutral"]
LabelStatus = Literal["pending", "partial_1w", "mature_4w", "mature_13w"]
TurnAction = Literal["sell", "buy"]
OperationSide = Literal["buy", "sell", "hold"]
_LABEL_STATUS_RANK = {
    "pending": 0,
    "partial_1w": 1,
    "mature_4w": 2,
    "mature_13w": 3,
}
_LABEL_DIGEST_CACHE_LOCK = Lock()
_LABEL_DIGEST_CACHE: dict[int, tuple["IterationLabel", str]] = {}
_LABEL_DIGEST_CACHE_LIMIT = 8192

ALGORITHM_VERSION = "weekly-optimizer-v2.1-progressive-cursor"
LOSS_VERSION = "weekly-loss-v2-brier"
MAX_TRADE_COUNT = 10
MIN_RELATIVE_IMPROVEMENT = Decimal("0.01")
MAX_OTHER_WINDOW_DETERIORATION = Decimal("0.005")
MAX_DIRECTION_HIT_DROP_POINTS = Decimal("0.02")
MAX_OVERTRADE_RELATIVE_INCREASE = Decimal("0.05")
FEATURE_GROUPS = (
    "valuation",
    "dea_trend",
    "dif_trend",
    "reversal",
    "price_momentum",
    "risk_regime",
)
LOSS_WEIGHTS: FrozenDict[str, Decimal] = FrozenDict(
    {
        "turning_deviation": Decimal("0.45"),
        "direction_calibration": Decimal("0.30"),
        "adverse_excursion": Decimal("0.15"),
        "overtrading": Decimal("0.10"),
    }
)
_SEED_WEIGHTS: FrozenDict[str, FrozenDict[str, Decimal]] = FrozenDict(
    {
        "399006": FrozenDict(
            {
                "valuation": Decimal("30"),
                "dea_trend": Decimal("20"),
                "dif_trend": Decimal("20"),
                "reversal": Decimal("15"),
                "price_momentum": Decimal("5"),
                "risk_regime": Decimal("10"),
            }
        ),
        "NDX": FrozenDict(
            {
                "valuation": Decimal("25"),
                "dea_trend": Decimal("22"),
                "dif_trend": Decimal("22"),
                "reversal": Decimal("12"),
                "price_momentum": Decimal("9"),
                "risk_regime": Decimal("10"),
            }
        ),
    }
)


class CrossMarketError(ValueError):
    """Raised when one market's state or feedback is used for another."""


class ConflictingWeekInput(ValueError):
    """Raised when a processed week is retried with different inputs."""


class ConflictingFeedback(ValueError):
    """Raised when one iteration identity carries conflicting content."""


class FeedbackRegressionError(ValueError):
    """Raised when visible feedback moves backward in maturity or time."""


class FutureLeakageError(ValueError):
    """A blocking source-date violation."""

    code = "SOURCE_DATA_AFTER_CUTOFF"
    severity = "blocking"

    def __init__(self, source_data_max_date: date, cutoff_date: date) -> None:
        self.source_data_max_date = source_data_max_date
        self.cutoff_date = cutoff_date
        super().__init__(
            "source_data_max_date must be at or before cutoff_date "
            f"({source_data_max_date.isoformat()} > {cutoff_date.isoformat()})"
        )


@dataclass(frozen=True, slots=True)
class DailyClose:
    """One strictly positive daily close."""

    trade_date: date
    close: Decimal
    instrument_code: str

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        if not isinstance(self.trade_date, date):
            raise ValueError("trade_date must be a date")
        value = _as_decimal(self.close, "close")
        if value <= 0:
            raise ValueError("Daily close must be finite and positive")
        object.__setattr__(self, "close", value)


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """Expected sessions for one instrument, including holiday decisions."""

    instrument_code: str
    expected_trade_dates: tuple[date, ...]
    coverage_start: date | None = None
    coverage_end: date | None = None

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        values = tuple(self.expected_trade_dates)
        if any(not isinstance(day, date) for day in values):
            raise ValueError("expected_trade_dates must contain dates")
        if len(set(values)) != len(values):
            raise ValueError("expected_trade_dates contains duplicates")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError(
                "expected_trade_dates must be strictly increasing"
            )
        if (self.coverage_start is None) != (self.coverage_end is None):
            raise ValueError(
                "coverage_start and coverage_end must be declared together"
            )
        if self.coverage_start is not None:
            if not isinstance(self.coverage_start, date) or not isinstance(
                self.coverage_end, date
            ):
                raise ValueError("calendar coverage boundaries must be dates")
            if self.coverage_start > self.coverage_end:
                raise ValueError(
                    "calendar coverage_start must not exceed coverage_end"
                )
            if values and (
                values[0] < self.coverage_start
                or values[-1] > self.coverage_end
            ):
                raise ValueError(
                    "expected_trade_dates must fall within calendar coverage"
                )
        object.__setattr__(self, "expected_trade_dates", values)


@dataclass(frozen=True, slots=True)
class IterationPrediction:
    """The immutable prediction emitted at one weekly cutoff."""

    iteration_id: str
    instrument_code: str
    week_key: str
    cutoff_date: date
    predicted_direction: str
    operation_side: str
    probability: Decimal
    predicted_action_date: date
    target_position: Decimal
    trade_count: int

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        if not self.iteration_id:
            raise ValueError("iteration_id is required")
        week_key = _normalize_week_key(self.week_key, self.cutoff_date)
        if self.predicted_direction not in {"up", "down", "neutral"}:
            raise ValueError(
                "predicted_direction must be up, down, or neutral"
            )
        if self.operation_side not in {"buy", "sell", "hold"}:
            raise ValueError("operation_side must be buy, sell, or hold")
        probability = _as_decimal(self.probability, "probability")
        if not Decimal() <= probability <= Decimal("1"):
            raise ValueError("probability must be between 0 and 1")
        position = _as_decimal(self.target_position, "target_position")
        if not Decimal() <= position <= Decimal("1"):
            raise ValueError("target_position must be between 0 and 1")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError("trade_count must be a non-negative integer")
        object.__setattr__(self, "week_key", week_key)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "target_position", position)


@dataclass(frozen=True, slots=True)
class IterationLabel:
    """Visible label state for one prediction at one ``as_of`` date."""

    iteration_id: str
    instrument_code: str
    week_key: str
    cutoff_date: date
    predicted_direction: str
    operation_side: str
    probability: Decimal
    predicted_action_date: date
    target_position: Decimal
    trade_count: int
    status: LabelStatus
    observed_through: date
    visible_row_count: int
    partial_weight: Decimal
    actual_direction: str | None
    actual_turn_date: date | None
    turning_deviation_sessions: int | None
    loss_components: FrozenDict[str, Decimal]
    total_loss: Decimal | None
    direction_hit: bool | None
    calibration_formula: str
    turning_deviation_formula: str = (
        "actual_trading_index-predicted_trading_index"
    )

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        week_key = _normalize_week_key(self.week_key, self.cutoff_date)
        if self.predicted_direction not in {"up", "down", "neutral"}:
            raise ValueError("Invalid predicted direction")
        if self.operation_side not in {"buy", "sell", "hold"}:
            raise ValueError("Invalid operation_side")
        if self.status not in _LABEL_STATUS_RANK:
            raise ValueError("Invalid label status")
        expected_turning_formula = (
            "hold:no_turn;turning_deviation=0"
            if self.operation_side == "hold"
            else "actual_trading_index-predicted_trading_index"
        )
        if self.turning_deviation_formula != expected_turning_formula:
            raise ValueError("Invalid turning deviation formula")
        probability = _as_decimal(self.probability, "probability")
        target_position = _as_decimal(
            self.target_position, "target_position"
        )
        partial_weight = _as_decimal(
            self.partial_weight, "partial_weight"
        )
        if not Decimal() <= probability <= Decimal("1"):
            raise ValueError("probability must be between 0 and 1")
        if not Decimal() <= target_position <= Decimal("1"):
            raise ValueError("target_position must be between 0 and 1")
        if not Decimal() <= partial_weight <= Decimal("1"):
            raise ValueError("partial_weight must be between 0 and 1")
        if self.visible_row_count < 0:
            raise ValueError("visible_row_count must be non-negative")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError("trade_count must be a non-negative integer")
        frozen_components = FrozenDict(
            {
                str(name): _as_decimal(value, f"loss:{name}")
                for name, value in self.loss_components.items()
            }
        )
        total_loss = (
            None
            if self.total_loss is None
            else _as_decimal(self.total_loss, "total_loss")
        )
        if self.status == "pending":
            valid = (
                partial_weight == 0
                and total_loss is None
                and not frozen_components
            )
        elif self.status in {"partial_1w", "mature_4w"}:
            valid = (
                Decimal() < partial_weight < Decimal("1")
                and total_loss is None
                and set(frozen_components) == {"direction_calibration"}
            )
        else:
            if self.operation_side != "hold" and (
                not isinstance(self.actual_turn_date, date)
                or not (
                    self.cutoff_date
                    < self.actual_turn_date
                    <= self.observed_through
                )
            ):
                raise ValueError(
                    "mature actual_turn_date must be after cutoff_date "
                    "and on or before observed_through"
                )
            weighted_total = sum(
                frozen_components[name] * weight
                for name, weight in LOSS_WEIGHTS.items()
            ) if set(frozen_components) == set(LOSS_WEIGHTS) else None
            if total_loss is not None and weighted_total is not None and (
                total_loss != weighted_total
            ):
                raise ValueError(
                    "mature total_loss must equal weighted loss components"
                )
            valid_turn = (
                self.actual_turn_date is None
                and self.turning_deviation_sessions is None
                and frozen_components.get("turning_deviation") == 0
                if self.operation_side == "hold"
                else (
                    self.actual_turn_date is not None
                    and self.turning_deviation_sessions is not None
                )
            )
            valid = (
                partial_weight == 1
                and total_loss is not None
                and set(frozen_components) == set(LOSS_WEIGHTS)
                and self.actual_direction is not None
                and valid_turn
                and self.direction_hit is not None
            )
        if not valid:
            raise ValueError(
                f"Label fields violate status invariant for {self.status}"
            )
        object.__setattr__(self, "week_key", week_key)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "target_position", target_position)
        object.__setattr__(self, "partial_weight", partial_weight)
        object.__setattr__(self, "loss_components", frozen_components)
        object.__setattr__(self, "total_loss", total_loss)

    def to_dict(self) -> dict[str, object]:
        return _label_to_dict(self)

    def to_json(self) -> str:
        return _canonical_json_dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    """Auditable aggregate loss for one rolling evaluation window."""

    instrument_code: str
    window: str
    sample_count: int
    total_loss: Decimal
    direction_hit_rate: Decimal
    calibration_loss: Decimal
    overtrade_penalty: Decimal

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be non-negative")
        for field_name in (
            "total_loss",
            "direction_hit_rate",
            "calibration_loss",
            "overtrade_penalty",
        ):
            value = _as_decimal(getattr(self, field_name), field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        if self.direction_hit_rate > 1:
            raise ValueError("direction_hit_rate must be at most 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument_code": self.instrument_code,
            "window": self.window,
            "sample_count": self.sample_count,
            "total_loss": self.total_loss,
            "direction_hit_rate": self.direction_hit_rate,
            "calibration_loss": self.calibration_loss,
            "overtrade_penalty": self.overtrade_penalty,
        }


@dataclass(frozen=True, slots=True)
class RollingMetricWindows:
    """The loss windows currently enabled by mature sample count."""

    instrument_code: str
    windows: FrozenDict[str, WindowMetrics]
    mature_count: int

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        frozen_windows = FrozenDict(self.windows)
        if self.mature_count < 0:
            raise ValueError("mature_count must be non-negative")
        if any(
            metric.instrument_code != self.instrument_code
            for metric in frozen_windows.values()
        ):
            raise CrossMarketError("Metric windows cannot mix markets")
        object.__setattr__(self, "windows", frozen_windows)

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument_code": self.instrument_code,
            "mature_count": self.mature_count,
            "windows": {
                name: metric.to_dict()
                for name, metric in self.windows.items()
            },
        }


@dataclass(frozen=True, slots=True)
class StepAudit:
    """Complete immutable audit record for one unique weekly transition."""

    instrument_code: str
    iteration_id: str
    work_version: str
    parent_work_version: str
    model_version: str
    parent_model_version: str
    week_key: str
    cutoff_date: date
    source_data_max_date: date | None
    base_weights: FrozenDict[str, Decimal]
    candidate_weights: FrozenDict[str, Decimal]
    accepted: bool
    reasons: tuple[str, ...]
    optimizer_memory: FrozenDict[str, object]
    feedback_count: int
    mature_count: int
    partial_count: int
    pending_count: int
    current_window_metrics: FrozenDict[str, WindowMetrics]
    candidate_window_metrics: FrozenDict[str, WindowMetrics]
    candidate_acceptance_rate: Decimal
    rejected_candidates: tuple[object, ...]
    source_hash: str
    algorithm_version: str
    loss_version: str
    loss_weights: FrozenDict[str, Decimal]
    input_hash: str
    data_integrity: bool
    future_leakage: bool
    active_windows: tuple[str, ...]
    gate_parameters: FrozenDict[str, Decimal]

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        object.__setattr__(self, "base_weights", FrozenDict(self.base_weights))
        object.__setattr__(
            self, "candidate_weights", FrozenDict(self.candidate_weights)
        )
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(
            self, "optimizer_memory", FrozenDict(self.optimizer_memory)
        )
        object.__setattr__(
            self,
            "current_window_metrics",
            FrozenDict(self.current_window_metrics),
        )
        object.__setattr__(
            self,
            "candidate_window_metrics",
            FrozenDict(self.candidate_window_metrics),
        )
        object.__setattr__(
            self,
            "rejected_candidates",
            FrozenDict({"items": self.rejected_candidates})["items"],
        )
        object.__setattr__(self, "loss_weights", FrozenDict(self.loss_weights))
        object.__setattr__(self, "active_windows", tuple(self.active_windows))
        object.__setattr__(
            self, "gate_parameters", FrozenDict(self.gate_parameters)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "active_windows": self.active_windows,
            "algorithm_version": self.algorithm_version,
            "base_weights": dict(self.base_weights),
            "candidate_acceptance_rate": self.candidate_acceptance_rate,
            "candidate_weights": dict(self.candidate_weights),
            "candidate_window_metrics": {
                key: value.to_dict()
                for key, value in self.candidate_window_metrics.items()
            },
            "cutoff_date": self.cutoff_date,
            "data_integrity": self.data_integrity,
            "feedback_count": self.feedback_count,
            "future_leakage": self.future_leakage,
            "gate_parameters": dict(self.gate_parameters),
            "instrument_code": self.instrument_code,
            "input_hash": self.input_hash,
            "iteration_id": self.iteration_id,
            "mature_count": self.mature_count,
            "model_version": self.model_version,
            "loss_version": self.loss_version,
            "loss_weights": dict(self.loss_weights),
            "optimizer_memory": dict(self.optimizer_memory),
            "parent_model_version": self.parent_model_version,
            "parent_work_version": self.parent_work_version,
            "partial_count": self.partial_count,
            "pending_count": self.pending_count,
            "reasons": self.reasons,
            "rejected_candidates": self.rejected_candidates,
            "source_data_max_date": self.source_data_max_date,
            "source_hash": self.source_hash,
            "week_key": self.week_key,
            "work_version": self.work_version,
            "current_window_metrics": {
                key: value.to_dict()
                for key, value in self.current_window_metrics.items()
            },
        }

    def to_json(self) -> str:
        return _canonical_json_dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class WorkState:
    """All state needed to perform the next pure optimizer transition."""

    instrument_code: str
    iteration_id: str
    work_version: str
    parent_work_version: str | None
    current_model_version: str
    current_model_weights: FrozenDict[str, Decimal]
    optimizer_memory: FrozenDict[str, object]
    feedback: tuple[IterationLabel, ...]
    audits: tuple[StepAudit, ...]
    week_records: FrozenDict[str, StepAudit]

    def __post_init__(self) -> None:
        _validate_instrument(self.instrument_code)
        weights = FrozenDict(self.current_model_weights)
        memory = FrozenDict(self.optimizer_memory)
        feedback = tuple(self.feedback)
        audits = tuple(self.audits)
        records = FrozenDict(self.week_records)
        _validate_weights(weights)
        object.__setattr__(self, "current_model_weights", weights)
        object.__setattr__(self, "optimizer_memory", memory)
        object.__setattr__(self, "feedback", feedback)
        object.__setattr__(self, "audits", audits)
        object.__setattr__(self, "week_records", records)

    @property
    def model_version(self) -> str:
        return self.current_model_version

    @property
    def weights(self) -> FrozenDict[str, Decimal]:
        return self.current_model_weights

    def to_dict(self) -> dict[str, object]:
        return {
            "audits": tuple(audit.to_dict() for audit in self.audits),
            "current_model_version": self.current_model_version,
            "current_model_weights": dict(self.current_model_weights),
            "feedback": tuple(label.to_dict() for label in self.feedback),
            "instrument_code": self.instrument_code,
            "iteration_id": self.iteration_id,
            "optimizer_memory": dict(self.optimizer_memory),
            "parent_work_version": self.parent_work_version,
            "week_records": {
                week: audit.to_dict()
                for week, audit in self.week_records.items()
            },
            "work_version": self.work_version,
        }

    def to_json(self) -> str:
        return _canonical_json_dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    """One transition result plus the state to feed into the next week."""

    work_state: WorkState
    audit: StepAudit
    reused: bool = False

    @property
    def instrument_code(self) -> str:
        return self.audit.instrument_code

    @property
    def iteration_id(self) -> str:
        return self.audit.iteration_id

    @property
    def work_version(self) -> str:
        return self.audit.work_version

    @property
    def parent_work_version(self) -> str:
        return self.audit.parent_work_version

    @property
    def current_model_version(self) -> str:
        return self.work_state.current_model_version

    @property
    def current_model_weights(self) -> FrozenDict[str, Decimal]:
        return self.work_state.current_model_weights

    @property
    def model_version(self) -> str:
        return self.current_model_version

    @property
    def base_weights(self) -> FrozenDict[str, Decimal]:
        return self.audit.base_weights

    @property
    def candidate_weights(self) -> FrozenDict[str, Decimal]:
        return self.audit.candidate_weights

    @property
    def accepted(self) -> bool:
        return self.audit.accepted

    @property
    def reasons(self) -> tuple[str, ...]:
        return self.audit.reasons

    @property
    def optimizer_memory(self) -> FrozenDict[str, object]:
        return self.audit.optimizer_memory

    @property
    def source_hash(self) -> str:
        return self.audit.source_hash

    def to_dict(self) -> dict[str, object]:
        return {
            "audit": self.audit.to_dict(),
            "reused": self.reused,
            "work_state": self.work_state.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json_dumps(self.to_dict())


MetricEvaluator = Callable[
    ...,
    Mapping[str, WindowMetrics] | RollingMetricWindows,
]


def seed_model(instrument_code: str) -> WorkState:
    """Return the market-specific M0001 state."""

    instrument = _validate_instrument(instrument_code)
    weights = FrozenDict(_SEED_WEIGHTS[instrument])
    _validate_weights(weights)
    memory: FrozenDict[str, object] = FrozenDict(
        {
            "accepted_candidate_count": 0,
            "candidate_acceptance_rate": Decimal("0"),
            "feedback_count": 0,
            "optimizer_steps": 0,
            "rejected_candidate_count": 0,
            "rejected_candidates": (),
            "seen_feedback_ids": (),
        }
    )
    return WorkState(
        instrument_code=instrument,
        iteration_id="I0000",
        work_version="W0000",
        parent_work_version=None,
        current_model_version="M0001",
        current_model_weights=weights,
        optimizer_memory=memory,
        feedback=(),
        audits=(),
        week_records=FrozenDict({}),
    )


def actual_turn_date(
    action: str,
    daily: Iterable[DailyClose],
) -> date:
    """Return the earliest maximum (sell) or minimum (buy) session."""

    rows = _validate_daily(daily)
    if not rows:
        raise ValueError("At least one daily close is required")
    if action == "sell":
        target = max(row.close for row in rows)
    elif action == "buy":
        target = min(row.close for row in rows)
    else:
        raise ValueError("action must be sell or buy")
    return next(row.trade_date for row in rows if row.close == target)


def update_labels(
    iterations: Iterable[IterationPrediction],
    daily: Iterable[DailyClose],
    as_of: date,
    *,
    expected_trade_dates: TradingCalendar | Iterable[date] | None = None,
    completed_week_ends: Iterable[date] | None = None,
) -> tuple[IterationLabel, ...]:
    """Update labels using only strictly visible closes at ``as_of``."""

    predictions = tuple(iterations)
    rows = _validate_daily(daily)
    if len({item.iteration_id for item in predictions}) != len(predictions):
        raise ValueError("Duplicate iteration_id values are not allowed")
    instruments = {item.instrument_code for item in predictions}
    if len(instruments) > 1:
        raise CrossMarketError("Iterations from different markets cannot mix")
    if predictions and rows and {
        row.instrument_code for row in rows
    } != instruments:
        raise CrossMarketError(
            "Daily closes must match the prediction instrument"
        )
    completed_ends = (
        None
        if completed_week_ends is None
        else tuple(completed_week_ends)
    )
    if isinstance(expected_trade_dates, TradingCalendar):
        if predictions and expected_trade_dates.instrument_code not in instruments:
            raise CrossMarketError(
                "Trading calendar market does not match predictions"
            )
        coverage_start = expected_trade_dates.coverage_start
        coverage_end = expected_trade_dates.coverage_end
    else:
        coverage_start = None
        coverage_end = None
    expected = _calendar_dates(
        expected_trade_dates,
        completed_ends,
    )

    labels: list[IterationLabel] = []
    for prediction in predictions:
        baseline_rows = tuple(
            row
            for row in rows
            if row.trade_date <= min(prediction.cutoff_date, as_of)
        )
        baseline = baseline_rows[-1] if baseline_rows else None
        verified_weeks = _verified_calendar_weeks(
            cutoff_date=prediction.cutoff_date,
            as_of=as_of,
            calendar_dates=expected,
            daily_dates={row.trade_date for row in rows},
            completed_week_ends=completed_ends,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
        )
        verified_count = len(verified_weeks) if baseline is not None else 0
        if verified_count >= 13:
            status: LabelStatus = "mature_13w"
            partial_weight = Decimal("1")
            horizon_weeks = 13
        elif verified_count >= 4:
            status = "mature_4w"
            partial_weight = Decimal("0.50")
            horizon_weeks = 4
        elif verified_count >= 1:
            status = "partial_1w"
            partial_weight = Decimal("0.25")
            horizon_weeks = 1
        else:
            status = "pending"
            partial_weight = Decimal("0")
            horizon_weeks = 0

        if horizon_weeks:
            horizon_dates = {
                day
                for week_dates in verified_weeks[:horizon_weeks]
                for day in week_dates
            }
            visible = tuple(
                row for row in rows if row.trade_date in horizon_dates
            )
        else:
            visible = tuple(
                row
                for row in rows
                if prediction.cutoff_date < row.trade_date <= as_of
            )
        observed_through = (
            visible[-1].trade_date if visible else min(as_of, prediction.cutoff_date)
        )
        count = len(visible)

        actual_direction: str | None = None
        direction_hit: bool | None = None
        calibration = Decimal("0")
        if horizon_weeks and baseline is not None and visible:
            actual_direction = _observed_direction(
                baseline.close, visible[-1].close
            )
            direction_hit = (
                prediction.predicted_direction == actual_direction
            )
            calibration = _brier_loss(
                prediction.probability, direction_hit
            )

        components: FrozenDict[str, Decimal] = FrozenDict({})
        total_loss: Decimal | None = None
        turn_date: date | None = None
        turn_deviation: int | None = None
        if status in {"partial_1w", "mature_4w"}:
            components = FrozenDict(
                {"direction_calibration": calibration}
            )
        elif status == "mature_13w":
            if prediction.operation_side == "hold":
                turning_loss = Decimal("0")
            else:
                action: TurnAction = prediction.operation_side
                turn_date = actual_turn_date(action, visible)
                actual_index = next(
                    index
                    for index, row in enumerate(visible)
                    if row.trade_date == turn_date
                )
                predicted_index = bisect_left(
                    tuple(row.trade_date for row in visible),
                    prediction.predicted_action_date,
                )
                turn_deviation = actual_index - predicted_index
                denominator = Decimal(max(len(visible) - 1, 1))
                turning_loss = min(
                    Decimal(abs(turn_deviation)) / denominator,
                    Decimal("1"),
                )
            adverse = _adverse_excursion(
                prediction.predicted_direction,
                visible,
                baseline.close,
            )
            overtrade = min(
                Decimal(prediction.trade_count) / Decimal(MAX_TRADE_COUNT),
                Decimal("1"),
            )
            components = FrozenDict(
                {
                    "turning_deviation": turning_loss,
                    "direction_calibration": calibration,
                    "adverse_excursion": adverse,
                    "overtrading": overtrade,
                }
            )
            total_loss = sum(
                components[name] * weight
                for name, weight in LOSS_WEIGHTS.items()
            )

        labels.append(
            IterationLabel(
                iteration_id=prediction.iteration_id,
                instrument_code=prediction.instrument_code,
                week_key=prediction.week_key,
                cutoff_date=prediction.cutoff_date,
                predicted_direction=prediction.predicted_direction,
                operation_side=prediction.operation_side,
                probability=prediction.probability,
                predicted_action_date=prediction.predicted_action_date,
                target_position=prediction.target_position,
                trade_count=prediction.trade_count,
                status=status,
                observed_through=observed_through,
                visible_row_count=count,
                partial_weight=partial_weight,
                actual_direction=actual_direction,
                actual_turn_date=turn_date,
                turning_deviation_sessions=turn_deviation,
                loss_components=components,
                total_loss=total_loss,
                direction_hit=direction_hit,
                calibration_formula="brier:(p_predicted-I[correct])^2",
                turning_deviation_formula=(
                    "hold:no_turn;turning_deviation=0"
                    if prediction.operation_side == "hold"
                    else "actual_trading_index-predicted_trading_index"
                ),
            )
        )
    return tuple(labels)


def _calendar_dates(
    expected_trade_dates: TradingCalendar | Iterable[date] | None,
    completed_week_ends: Iterable[date] | None,
) -> tuple[date, ...] | None:
    if isinstance(expected_trade_dates, TradingCalendar):
        return expected_trade_dates.expected_trade_dates
    if expected_trade_dates is not None:
        raise ValueError(
            "expected_trade_dates requires instrument_code via TradingCalendar"
        )
    if completed_week_ends is None:
        return None
    raise ValueError(
        "completed_week_ends requires instrument_code and expected sessions "
        "via TradingCalendar"
    )


def _verified_calendar_weeks(
    *,
    cutoff_date: date,
    as_of: date,
    calendar_dates: tuple[date, ...] | None,
    daily_dates: set[date],
    completed_week_ends: Iterable[date] | None,
    coverage_start: date | None,
    coverage_end: date | None,
) -> tuple[tuple[date, ...], ...]:
    if (
        calendar_dates is None
        or coverage_start is None
        or coverage_end is None
    ):
        return ()
    complete_ends = (
        None
        if completed_week_ends is None
        else frozenset(completed_week_ends)
    )
    grouped: dict[tuple[int, int], list[date]] = {}
    cutoff_week = cutoff_date.isocalendar()[:2]
    for day in calendar_dates:
        if day <= cutoff_date:
            continue
        iso_year, iso_week, _ = day.isocalendar()
        if (iso_year, iso_week) == cutoff_week:
            continue
        grouped.setdefault((iso_year, iso_week), []).append(day)
    verified: list[tuple[date, ...]] = []
    for week_dates_list in grouped.values():
        week_dates = tuple(week_dates_list)
        iso_year, iso_week, _ = week_dates[0].isocalendar()
        civil_week_start = date.fromisocalendar(iso_year, iso_week, 1)
        civil_week_end = date.fromisocalendar(iso_year, iso_week, 7)
        if civil_week_start < coverage_start:
            continue
        if civil_week_end > coverage_end:
            break
        last_session = week_dates[-1]
        if last_session > as_of:
            break
        if complete_ends is not None and last_session not in complete_ends:
            break
        if not set(week_dates).issubset(daily_dates):
            break
        verified.append(week_dates)
    return tuple(verified)


def rolling_metrics(
    labels: Iterable[IterationLabel],
    window: int | None = None,
    *,
    instrument_code: str | None = None,
) -> RollingMetricWindows | WindowMetrics | None:
    """Aggregate only mature 13-week labels.

    With ``window=None`` the acceptance windows are returned: an expanding
    window below 20 samples, a 20-sample window from 20 through 51, and both
    20 and 52 once 52 samples exist.
    """

    label_rows = tuple(labels)
    instruments = {label.instrument_code for label in label_rows}
    if len(instruments) > 1:
        raise CrossMarketError(
            "Feedback from different markets cannot share rolling metrics"
        )
    if instrument_code is None:
        if not instruments:
            raise ValueError(
                "instrument_code is required for empty rolling metrics"
            )
        instrument_code = next(iter(instruments))
    _validate_instrument(instrument_code)
    if instruments and instruments != {instrument_code}:
        raise CrossMarketError(
            "Rolling metric market does not match feedback"
        )
    mature = _sorted_mature(label_rows)
    if window not in {None, 20, 52}:
        raise ValueError("window must be None, 20, or 52")
    if window is not None:
        if not mature:
            return None
        if window == 52 and len(mature) < 52:
            return None
        selected = (
            mature
            if window == 20 and len(mature) < 20
            else mature[-window:]
        )
        label = (
            "expanding"
            if window == 20 and len(mature) < 20
            else str(window)
        )
        return _aggregate_metrics(selected, label)

    if not mature:
        return RollingMetricWindows(instrument_code, FrozenDict({}), 0)
    if len(mature) < 20:
        windows = FrozenDict(
            {"expanding": _aggregate_metrics(mature, "expanding")}
        )
    elif len(mature) < 52:
        windows = FrozenDict(
            {"20": _aggregate_metrics(mature[-20:], "20")}
        )
    else:
        windows = FrozenDict(
            {
                "20": _aggregate_metrics(mature[-20:], "20"),
                "52": _aggregate_metrics(mature[-52:], "52"),
            }
        )
    return RollingMetricWindows(instrument_code, windows, len(mature))


def run_optimizer_step(
    previous_work_state: WorkState | OptimizerStepResult,
    feedback: Iterable[IterationLabel],
    week_key: str,
    cutoff_date: date,
    source_data_max_date: date | None,
    prediction: IterationPrediction | Mapping[str, object] | None = None,
    *,
    evaluator: MetricEvaluator | None = None,
    evaluator_identity: str | None = None,
    current_metrics: Mapping[
        str, WindowMetrics | Mapping[str, object]
    ]
    | RollingMetricWindows
    | None = None,
    candidate_metrics: Mapping[
        str, WindowMetrics | Mapping[str, object]
    ]
    | RollingMetricWindows
    | None = None,
    data_integrity: bool = True,
) -> OptimizerStepResult:
    """Run one deterministic, immutable optimizer transition."""

    state = (
        previous_work_state.work_state
        if isinstance(previous_work_state, OptimizerStepResult)
        else previous_work_state
    )
    _validate_state_market(state)
    if evaluator is not None and (
        not isinstance(evaluator_identity, str)
        or not evaluator_identity.strip()
    ):
        raise ValueError(
            "evaluator_identity is required when evaluator is provided"
        )
    if evaluator is None and evaluator_identity is not None:
        raise ValueError(
            "evaluator_identity requires an evaluator"
        )
    if evaluator_identity is not None:
        evaluator_identity = evaluator_identity.strip()
    week_key = _normalize_week_key(week_key, cutoff_date)
    if (
        source_data_max_date is not None
        and source_data_max_date > cutoff_date
    ):
        raise FutureLeakageError(source_data_max_date, cutoff_date)

    incoming = tuple(feedback)
    _validate_feedback_market(state.instrument_code, incoming)
    visible_feedback = state.feedback + incoming
    future_feedback_dates = tuple(
        feedback_date
        for label in visible_feedback
        for feedback_date in (
            label.cutoff_date,
            label.observed_through,
            label.actual_turn_date,
        )
        if feedback_date is not None and feedback_date > cutoff_date
    )
    if future_feedback_dates:
        raise FutureLeakageError(max(future_feedback_dates), cutoff_date)
    canonical_incoming = _canonicalize_feedback(incoming)
    if isinstance(prediction, IterationPrediction):
        if prediction.instrument_code != state.instrument_code:
            raise CrossMarketError(
                "Prediction market does not match optimizer state"
            )
        if prediction.cutoff_date > cutoff_date:
            raise FutureLeakageError(
                prediction.cutoff_date, cutoff_date
            )
    elif isinstance(prediction, Mapping):
        prediction_market = prediction.get("instrument_code")
        if prediction_market is None:
            raise CrossMarketError(
                "Prediction mapping requires instrument_code"
            )
        if prediction_market != state.instrument_code:
            raise CrossMarketError(
                "Prediction market does not match optimizer state"
            )
        prediction_cutoff = prediction.get("cutoff_date")
        if prediction_cutoff is not None:
            if not isinstance(prediction_cutoff, date):
                raise ValueError(
                    "Prediction mapping cutoff_date must be a date"
                )
            if prediction_cutoff > cutoff_date:
                raise FutureLeakageError(
                    prediction_cutoff, cutoff_date
                )

    metric_feedback_by_id = {
        _feedback_key(label): label for label in state.feedback
    }
    metric_feedback_by_id.update(
        (_feedback_key(label), label) for label in canonical_incoming
    )
    metric_feedback = tuple(metric_feedback_by_id.values())
    request_mature_count = sum(
        label.status == "mature_13w" for label in metric_feedback
    )
    request_bundle = rolling_metrics(
        metric_feedback,
        instrument_code=state.instrument_code,
    )
    assert isinstance(request_bundle, RollingMetricWindows)
    request_active_windows = tuple(request_bundle.windows)

    input_hash = _request_input_hash(
        instrument_code=state.instrument_code,
        feedback=canonical_incoming,
        week_key=week_key,
        cutoff_date=cutoff_date,
        source_data_max_date=source_data_max_date,
        prediction=prediction,
        evaluator_identity=evaluator_identity,
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
        active_windows=request_active_windows,
        mature_count=request_mature_count,
        data_integrity=data_integrity,
    )
    if week_key in state.week_records:
        previous_audit = state.week_records[week_key]
        if previous_audit.input_hash != input_hash:
            raise ConflictingWeekInput(
                f"Week {week_key} was already processed with other inputs"
            )
        return OptimizerStepResult(
            work_state=state,
            audit=previous_audit,
            reused=True,
        )

    merged_feedback = _merge_feedback(
        state.feedback, canonical_incoming
    )
    feedback_count = len(merged_feedback)
    mature_count = sum(
        label.status == "mature_13w" for label in merged_feedback
    )
    partial_count = sum(
        label.status in {"partial_1w", "mature_4w"}
        for label in merged_feedback
    )
    pending_count = sum(
        label.status == "pending" for label in merged_feedback
    )
    default_bundle = rolling_metrics(
        merged_feedback,
        instrument_code=state.instrument_code,
    )
    assert isinstance(default_bundle, RollingMetricWindows)
    active_windows = tuple(default_bundle.windows)

    iteration_id = _increment_version(state.iteration_id, "I")
    work_version = _increment_version(state.work_version, "W")
    base_weights = state.current_model_weights
    if evaluator is not None:
        evaluated_current = _call_evaluator(
            evaluator,
            base_weights,
            merged_feedback,
            tuple(default_bundle.windows),
        )
        current_window_metrics = _coerce_metrics(
            evaluated_current,
            state.instrument_code,
            active_windows,
            mature_count,
        )
        injected_candidate_window_metrics = None
    else:
        current_window_metrics = _coerce_metrics(
            current_metrics
            if current_metrics is not None
            else default_bundle,
            state.instrument_code,
            active_windows,
            mature_count,
        )
        injected_candidate_window_metrics = _coerce_metrics(
            candidate_metrics
            if candidate_metrics is not None
            else default_bundle,
            state.instrument_code,
            active_windows,
            mature_count,
        )
    validation_context_hash = _candidate_validation_context_hash(
        instrument_code=state.instrument_code,
        base_model_version=state.current_model_version,
        base_weights=base_weights,
        feedback=merged_feedback,
        active_windows=active_windows,
        current_metrics=current_window_metrics,
        candidate_metrics=injected_candidate_window_metrics,
        data_integrity=data_integrity,
        evaluator_identity=evaluator_identity,
    )
    candidate_option = _candidate_weights(
        base_weights,
        rejected_candidates=tuple(
            state.optimizer_memory["rejected_candidates"]  # type: ignore[arg-type]
        ),
        validation_context_hash=validation_context_hash,
        candidate_cursor=int(state.optimizer_memory["optimizer_steps"]),
    )
    candidate_exhausted = candidate_option is None
    candidate_weights = (
        base_weights if candidate_option is None else candidate_option
    )

    if evaluator is not None:
        evaluated_candidate = _call_evaluator(
            evaluator,
            candidate_weights,
            merged_feedback,
            tuple(default_bundle.windows),
        )
        candidate_window_metrics = _coerce_metrics(
            evaluated_candidate,
            state.instrument_code,
            active_windows,
            mature_count,
        )
    else:
        assert injected_candidate_window_metrics is not None
        candidate_window_metrics = injected_candidate_window_metrics

    gate_parameters = _gate_parameters()
    reasons = _acceptance_reasons(
        active_windows=active_windows,
        current=current_window_metrics,
        candidate=candidate_window_metrics,
        data_integrity=data_integrity,
    )
    if candidate_exhausted:
        reasons += ("candidate_exhausted",)
    accepted = not reasons
    parent_model_version = state.current_model_version
    model_version = (
        _increment_version(parent_model_version, "M")
        if accepted
        else parent_model_version
    )
    model_weights = (
        candidate_weights if accepted else state.current_model_weights
    )

    rejected_candidates = tuple(
        state.optimizer_memory["rejected_candidates"]  # type: ignore[arg-type]
    )
    candidate_quality_evaluated = data_integrity and bool(active_windows)
    if (
        not accepted
        and not candidate_exhausted
        and candidate_quality_evaluated
    ):
        rejected_candidates += (
            FrozenDict(
                {
                    "base_weights": base_weights,
                    "candidate_weights": candidate_weights,
                    "reasons": reasons,
                    "validation_context_hash": validation_context_hash,
                    "week_key": week_key,
                    "work_version": work_version,
                }
            ),
        )
    optimizer_steps = int(state.optimizer_memory["optimizer_steps"]) + 1
    accepted_count = (
        int(state.optimizer_memory["accepted_candidate_count"])
        + int(accepted)
    )
    acceptance_rate = Decimal(accepted_count) / Decimal(optimizer_steps)
    optimizer_memory: FrozenDict[str, object] = FrozenDict(
        {
            "accepted_candidate_count": accepted_count,
            "candidate_acceptance_rate": acceptance_rate,
            "feedback_count": feedback_count,
            "optimizer_steps": optimizer_steps,
            "rejected_candidate_count": len(rejected_candidates),
            "rejected_candidates": rejected_candidates,
            "seen_feedback_ids": tuple(
                _feedback_key(label) for label in merged_feedback
            ),
        }
    )

    source_material = {
        "algorithm_version": ALGORITHM_VERSION,
        "base_weights": base_weights,
        "candidate_metrics": {
            key: value.to_dict()
            for key, value in candidate_window_metrics.items()
        },
        "candidate_weights": candidate_weights,
        "cutoff_date": cutoff_date,
        "data_integrity": data_integrity,
        "current_metrics": {
            key: value.to_dict()
            for key, value in current_window_metrics.items()
        },
        "feedback_digest": _feedback_digest(merged_feedback),
        "feedback_count": len(merged_feedback),
        "future_leakage": False,
        "gate_parameters": gate_parameters,
        "active_windows": active_windows,
        "input_hash": input_hash,
        "instrument_code": state.instrument_code,
        "loss_version": LOSS_VERSION,
        "loss_weights": LOSS_WEIGHTS,
        "optimizer_memory_before": state.optimizer_memory,
        "parent_work_version": state.work_version,
        "prediction": _prediction_material(prediction),
        "source_data_max_date": source_data_max_date,
        "accepted": accepted,
        "acceptance_reasons": reasons,
        "week_key": week_key,
    }
    source_hash = hashlib.sha256(
        _canonical_json_dumps(source_material).encode("utf-8")
    ).hexdigest()
    audit = StepAudit(
        instrument_code=state.instrument_code,
        iteration_id=iteration_id,
        work_version=work_version,
        parent_work_version=state.work_version,
        model_version=model_version,
        parent_model_version=parent_model_version,
        week_key=week_key,
        cutoff_date=cutoff_date,
        source_data_max_date=source_data_max_date,
        base_weights=base_weights,
        candidate_weights=candidate_weights,
        accepted=accepted,
        reasons=reasons,
        optimizer_memory=optimizer_memory,
        feedback_count=feedback_count,
        mature_count=mature_count,
        partial_count=partial_count,
        pending_count=pending_count,
        current_window_metrics=current_window_metrics,
        candidate_window_metrics=candidate_window_metrics,
        candidate_acceptance_rate=acceptance_rate,
        rejected_candidates=rejected_candidates,
        source_hash=source_hash,
        algorithm_version=ALGORITHM_VERSION,
        loss_version=LOSS_VERSION,
        loss_weights=LOSS_WEIGHTS,
        input_hash=input_hash,
        data_integrity=data_integrity,
        future_leakage=False,
        active_windows=active_windows,
        gate_parameters=gate_parameters,
    )
    validated_model_weights = validate_step_audit_acceptance(
        audit,
        previous_model_version=state.current_model_version,
        previous_model_weights=state.current_model_weights,
        previous_optimizer_memory=state.optimizer_memory,
    )
    if validated_model_weights != model_weights:
        raise ValueError(
            "Optimizer audit current weights are inconsistent"
        )
    new_records = FrozenDict(
        tuple(state.week_records.items()) + ((week_key, audit),)
    )
    new_state = WorkState(
        instrument_code=state.instrument_code,
        iteration_id=iteration_id,
        work_version=work_version,
        parent_work_version=state.work_version,
        current_model_version=model_version,
        current_model_weights=model_weights,
        optimizer_memory=optimizer_memory,
        feedback=merged_feedback,
        audits=state.audits + (audit,),
        week_records=new_records,
    )
    return OptimizerStepResult(
        work_state=new_state,
        audit=audit,
        reused=False,
    )


def validate_step_audit_acceptance(
    audit: StepAudit,
    *,
    previous_model_version: str,
    previous_model_weights: Mapping[str, object],
    previous_optimizer_memory: Mapping[str, object],
) -> FrozenDict[str, Decimal]:
    """Recompute and prove one Task 5 acceptance transition.

    The returned weights are the only model weights made effective by the
    audited gate result.  Callers must compare them with their current state.
    """

    if not isinstance(audit, StepAudit):
        raise ValueError("Task 5 acceptance proof requires a StepAudit")
    if (
        audit.algorithm_version != ALGORITHM_VERSION
        or audit.loss_version != LOSS_VERSION
        or audit.loss_weights != LOSS_WEIGHTS
    ):
        raise ValueError("Task 5 audit algorithm contract is inconsistent")
    if type(audit.accepted) is not bool:
        raise ValueError("Task 5 audit accepted flag must be boolean")
    if type(audit.data_integrity) is not bool:
        raise ValueError("Task 5 audit data_integrity must be boolean")
    if type(audit.future_leakage) is not bool or audit.future_leakage:
        raise ValueError("Task 5 audit cannot contain future leakage")
    for field_name in (
        "feedback_count",
        "mature_count",
        "partial_count",
        "pending_count",
    ):
        value = getattr(audit, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Task 5 audit {field_name} must be a non-negative integer"
            )
    if (
        audit.mature_count + audit.partial_count + audit.pending_count
        != audit.feedback_count
    ):
        raise ValueError("Task 5 audit feedback counts are inconsistent")

    previous_weights = FrozenDict(previous_model_weights)
    _validate_weights(previous_weights)
    _validate_weights(audit.base_weights)
    _validate_weights(audit.candidate_weights)
    if audit.parent_model_version != previous_model_version:
        raise ValueError("Task 5 audit parent model version is inconsistent")
    if audit.base_weights != previous_weights:
        raise ValueError("Task 5 audit base/current weights are inconsistent")
    _validate_audit_candidate_weight_transition(audit)

    expected_windows = _active_windows_for_mature_count(audit.mature_count)
    if audit.active_windows != expected_windows:
        raise ValueError("Task 5 audit active windows are inconsistent")
    if audit.gate_parameters != _gate_parameters():
        raise ValueError("Task 5 audit gate parameters are inconsistent")
    current_metrics = _coerce_metrics(
        audit.current_window_metrics,
        audit.instrument_code,
        audit.active_windows,
        audit.mature_count,
    )
    candidate_metrics = _coerce_metrics(
        audit.candidate_window_metrics,
        audit.instrument_code,
        audit.active_windows,
        audit.mature_count,
    )
    expected_reasons = _acceptance_reasons(
        active_windows=audit.active_windows,
        current=current_metrics,
        candidate=candidate_metrics,
        data_integrity=audit.data_integrity,
    )
    candidate_exhausted = audit.candidate_weights == audit.base_weights
    if candidate_exhausted:
        expected_reasons += ("candidate_exhausted",)
    if audit.reasons != expected_reasons:
        raise ValueError(
            "Task 5 audit acceptance reasons do not match the gate"
        )
    expected_accepted = not expected_reasons
    if audit.accepted is not expected_accepted:
        raise ValueError(
            "Task 5 audit accepted flag does not match the gate"
        )

    expected_model_version = (
        _increment_version(previous_model_version, "M")
        if expected_accepted
        else previous_model_version
    )
    if audit.model_version != expected_model_version:
        raise ValueError(
            "Task 5 audit model version does not match acceptance"
        )
    effective_weights = (
        audit.candidate_weights
        if expected_accepted
        else audit.base_weights
    )
    _validate_audit_optimizer_memory(
        audit,
        previous_optimizer_memory=previous_optimizer_memory,
        expected_accepted=expected_accepted,
        candidate_exhausted=candidate_exhausted,
    )
    return effective_weights


def _active_windows_for_mature_count(mature_count: int) -> tuple[str, ...]:
    if mature_count == 0:
        return ()
    if mature_count < 20:
        return ("expanding",)
    if mature_count < 52:
        return ("20",)
    return ("20", "52")


def _validate_audit_candidate_weight_transition(audit: StepAudit) -> None:
    deltas = tuple(
        audit.candidate_weights[name] - audit.base_weights[name]
        for name in audit.base_weights
        if audit.candidate_weights[name] != audit.base_weights[name]
    )
    if not deltas:
        return
    if (
        len(deltas) != 2
        or sum(deltas, Decimal()) != 0
        or abs(deltas[0]) not in {Decimal("1"), Decimal("2")}
        or abs(deltas[1]) != abs(deltas[0])
    ):
        raise ValueError(
            "Task 5 audit candidate weights violate the optimizer transition"
        )


def _validate_audit_optimizer_memory(
    audit: StepAudit,
    *,
    previous_optimizer_memory: Mapping[str, object],
    expected_accepted: bool,
    candidate_exhausted: bool,
) -> None:
    previous = FrozenDict(previous_optimizer_memory)
    current = audit.optimizer_memory
    integer_fields = (
        "accepted_candidate_count",
        "feedback_count",
        "optimizer_steps",
        "rejected_candidate_count",
    )
    for memory, label in ((previous, "previous"), (current, "current")):
        for field_name in integer_fields:
            value = memory.get(field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"Task 5 {label} optimizer memory {field_name} "
                    "must be a non-negative integer"
                )

    expected_steps = int(previous["optimizer_steps"]) + 1
    expected_accepted_count = (
        int(previous["accepted_candidate_count"])
        + int(expected_accepted)
    )
    expected_rate = (
        Decimal(expected_accepted_count) / Decimal(expected_steps)
    )
    if (
        current["optimizer_steps"] != expected_steps
        or current["accepted_candidate_count"] != expected_accepted_count
        or current.get("candidate_acceptance_rate") != expected_rate
        or audit.candidate_acceptance_rate != expected_rate
        or current["feedback_count"] != audit.feedback_count
    ):
        raise ValueError(
            "Task 5 optimizer memory does not match acceptance"
        )

    previous_rejected = previous.get("rejected_candidates")
    current_rejected = current.get("rejected_candidates")
    if not isinstance(previous_rejected, tuple) or not isinstance(
        current_rejected, tuple
    ):
        raise ValueError(
            "Task 5 optimizer rejected-candidate memory must be tuples"
        )
    should_record_rejection = (
        not expected_accepted
        and not candidate_exhausted
        and audit.data_integrity
        and bool(audit.active_windows)
    )
    expected_rejected_count = (
        len(previous_rejected) + int(should_record_rejection)
    )
    if (
        current_rejected != audit.rejected_candidates
        or current["rejected_candidate_count"] != len(current_rejected)
        or len(current_rejected) != expected_rejected_count
        or current_rejected[: len(previous_rejected)] != previous_rejected
    ):
        raise ValueError(
            "Task 5 optimizer rejection memory is inconsistent"
        )
    if should_record_rejection:
        record = current_rejected[-1]
        if (
            not isinstance(record, Mapping)
            or record.get("base_weights") != audit.base_weights
            or record.get("candidate_weights") != audit.candidate_weights
            or record.get("reasons") != audit.reasons
            or record.get("week_key") != audit.week_key
            or record.get("work_version") != audit.work_version
        ):
            raise ValueError(
                "Task 5 optimizer rejected candidate lacks audit proof"
            )

    seen_feedback_ids = current.get("seen_feedback_ids")
    if (
        not isinstance(seen_feedback_ids, tuple)
        or len(seen_feedback_ids) != audit.feedback_count
    ):
        raise ValueError("Task 5 optimizer feedback memory is inconsistent")


def _validate_instrument(instrument_code: str) -> InstrumentCode:
    if instrument_code not in _SEED_WEIGHTS:
        raise ValueError(f"Unsupported instrument: {instrument_code}")
    return instrument_code  # type: ignore[return-value]


def _normalize_week_key(value: object, cutoff_date: date) -> str:
    if not isinstance(cutoff_date, date):
        raise ValueError("cutoff_date must be a date")
    if not isinstance(value, str):
        raise ValueError("week_key must be an ISO YYYY-Www string")
    normalized = value.strip()
    match = re.fullmatch(r"(\d{4})-W(\d{2})", normalized)
    if match is None:
        raise ValueError("week_key must be an ISO YYYY-Www string")
    iso_year = int(match.group(1))
    iso_week = int(match.group(2))
    try:
        date.fromisocalendar(iso_year, iso_week, 1)
    except ValueError as exc:
        raise ValueError("week_key must name a valid ISO week") from exc
    cutoff_year, cutoff_week, _ = cutoff_date.isocalendar()
    if (iso_year, iso_week) != (cutoff_year, cutoff_week):
        raise ValueError("week_key must match cutoff_date ISO week")
    return f"{iso_year:04d}-W{iso_week:02d}"


def _strict_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _as_decimal(value: object, name: str) -> Decimal:
    try:
        converted = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a finite Decimal") from exc
    if not converted.is_finite():
        if name == "close":
            raise ValueError("Daily close must be finite and positive")
        raise ValueError(f"{name} must be finite")
    return converted


def _validate_weights(weights: Mapping[str, Decimal]) -> None:
    if set(weights) != set(FEATURE_GROUPS):
        raise ValueError("Weights must contain exactly the six feature groups")
    converted = tuple(
        _as_decimal(weights[name], f"weight:{name}") for name in FEATURE_GROUPS
    )
    if any(value < 0 for value in converted):
        raise ValueError("Weights must be non-negative")
    if sum(converted, Decimal()) != Decimal("100"):
        raise ValueError("Weights must sum exactly to 100")


def _validate_daily(
    daily: Iterable[DailyClose],
) -> tuple[DailyClose, ...]:
    rows = tuple(daily)
    instruments = {row.instrument_code for row in rows}
    if len(instruments) > 1:
        raise CrossMarketError("Daily closes cannot mix markets")
    dates = tuple(row.trade_date for row in rows)
    if len(set(dates)) != len(dates):
        raise ValueError("Daily close dates contain duplicate values")
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Daily closes must be strictly increasing")
    return rows


def _observed_direction(start: Decimal, end: Decimal) -> Direction:
    if end > start:
        return "up"
    if end < start:
        return "down"
    return "neutral"


def _brier_loss(probability: Decimal, correct: bool) -> Decimal:
    target = Decimal("1") if correct else Decimal("0")
    return (probability - target) ** 2


def _adverse_excursion(
    direction: str,
    rows: Sequence[DailyClose],
    baseline_close: Decimal,
) -> Decimal:
    start = baseline_close
    if direction == "up":
        raw = max(Decimal(), (start - min(row.close for row in rows)) / start)
    elif direction == "down":
        raw = max(Decimal(), (max(row.close for row in rows) - start) / start)
    else:
        high = (max(row.close for row in rows) - start) / start
        low = (start - min(row.close for row in rows)) / start
        raw = max(high, low, Decimal())
    return min(raw, Decimal("1"))


def _label_to_dict(label: IterationLabel) -> dict[str, object]:
    return {
        "actual_direction": label.actual_direction,
        "actual_turn_date": label.actual_turn_date,
        "calibration_formula": label.calibration_formula,
        "cutoff_date": label.cutoff_date,
        "direction_hit": label.direction_hit,
        "instrument_code": label.instrument_code,
        "iteration_id": label.iteration_id,
        "loss_components": dict(label.loss_components),
        "observed_through": label.observed_through,
        "operation_side": label.operation_side,
        "partial_weight": label.partial_weight,
        "predicted_action_date": label.predicted_action_date,
        "predicted_direction": label.predicted_direction,
        "probability": label.probability,
        "status": label.status,
        "target_position": label.target_position,
        "total_loss": label.total_loss,
        "trade_count": label.trade_count,
        "turning_deviation_formula": label.turning_deviation_formula,
        "turning_deviation_sessions": label.turning_deviation_sessions,
        "visible_row_count": label.visible_row_count,
        "week_key": label.week_key,
    }


def _label_digest(label: IterationLabel) -> str:
    key = id(label)
    with _LABEL_DIGEST_CACHE_LOCK:
        cached = _LABEL_DIGEST_CACHE.get(key)
        if cached is not None and cached[0] is label:
            return cached[1]
    digest = hashlib.sha256(
        _canonical_json_dumps(_label_to_dict(label)).encode("utf-8")
    ).hexdigest()
    with _LABEL_DIGEST_CACHE_LOCK:
        if len(_LABEL_DIGEST_CACHE) >= _LABEL_DIGEST_CACHE_LIMIT:
            _LABEL_DIGEST_CACHE.clear()
        _LABEL_DIGEST_CACHE[key] = (label, digest)
    return digest


def _feedback_digest(feedback: Iterable[IterationLabel]) -> str:
    material = tuple(_label_digest(label) for label in feedback)
    return hashlib.sha256(
        _canonical_json_dumps(material).encode("utf-8")
    ).hexdigest()


def _sorted_mature(
    labels: Iterable[IterationLabel],
) -> tuple[IterationLabel, ...]:
    return tuple(
        sorted(
            (
                label
                for label in labels
                if label.status == "mature_13w"
                and label.total_loss is not None
            ),
            key=lambda label: (
                label.observed_through,
                label.cutoff_date,
                label.iteration_id,
            ),
        )
    )


def _aggregate_metrics(
    labels: Sequence[IterationLabel],
    window: str,
) -> WindowMetrics:
    count = len(labels)
    if count == 0:
        raise ValueError("Cannot aggregate an empty mature-label window")
    denominator = Decimal(count)
    return WindowMetrics(
        instrument_code=labels[0].instrument_code,
        window=window,
        sample_count=count,
        total_loss=sum(
            (label.total_loss for label in labels),
            Decimal(),
        )
        / denominator,  # type: ignore[arg-type]
        direction_hit_rate=sum(
            Decimal(int(bool(label.direction_hit))) for label in labels
        )
        / denominator,
        calibration_loss=sum(
            label.loss_components["direction_calibration"]
            for label in labels
        )
        / denominator,
        overtrade_penalty=sum(
            label.loss_components["overtrading"] for label in labels
        )
        / denominator,
    )


def _feedback_key(label: IterationLabel) -> str:
    return f"{label.instrument_code}:{label.iteration_id}"


def _canonicalize_feedback(
    labels: tuple[IterationLabel, ...],
) -> tuple[IterationLabel, ...]:
    grouped: dict[str, list[IterationLabel]] = {}
    for label in labels:
        grouped.setdefault(_feedback_key(label), []).append(label)
    normalized: tuple[IterationLabel, ...] = ()
    for key in sorted(grouped):
        ordered = tuple(
            sorted(
                grouped[key],
                key=lambda label: (
                    _LABEL_STATUS_RANK[label.status],
                    label.observed_through,
                    _canonical_json_dumps(_label_to_dict(label)),
                ),
            )
        )
        normalized = _merge_feedback(normalized, ordered)
    return normalized


def _merge_feedback(
    previous: tuple[IterationLabel, ...],
    incoming: tuple[IterationLabel, ...],
) -> tuple[IterationLabel, ...]:
    merged = {_feedback_key(label): label for label in previous}
    for label in incoming:
        key = _feedback_key(label)
        existing = merged.get(key)
        if existing is not None:
            if _feedback_prediction_material(existing) != (
                _feedback_prediction_material(label)
            ):
                raise ConflictingFeedback(
                    f"Conflicting prediction content for {key}"
                )
            if (
                _LABEL_STATUS_RANK[label.status]
                < _LABEL_STATUS_RANK[existing.status]
                or label.observed_through < existing.observed_through
            ):
                raise FeedbackRegressionError(
                    f"Feedback regressed for {key}"
                )
            if (
                label.status == existing.status
                and label.observed_through == existing.observed_through
                and label != existing
            ):
                raise ConflictingFeedback(
                    f"Conflicting label content for {key}"
                )
        merged[key] = label
    return tuple(
        merged[key]
        for key in sorted(
            merged,
            key=lambda item: (
                merged[item].cutoff_date,
                merged[item].iteration_id,
                item,
            ),
        )
    )


def _feedback_prediction_material(
    label: IterationLabel,
) -> tuple[object, ...]:
    return (
        label.instrument_code,
        label.iteration_id,
        label.week_key,
        label.cutoff_date,
        label.predicted_direction,
        label.probability,
        label.predicted_action_date,
        label.operation_side,
        label.target_position,
        label.trade_count,
    )


def _validate_feedback_market(
    instrument_code: str,
    labels: tuple[IterationLabel, ...],
) -> None:
    wrong = {
        label.instrument_code
        for label in labels
        if label.instrument_code != instrument_code
    }
    if wrong:
        raise CrossMarketError(
            f"Feedback market {sorted(wrong)!r} does not match "
            f"{instrument_code}"
        )


def _validate_state_market(state: WorkState) -> None:
    instrument = _validate_instrument(state.instrument_code)
    mismatched_feedback = any(
        label.instrument_code != instrument for label in state.feedback
    )
    mismatched_audits = any(
        audit.instrument_code != instrument for audit in state.audits
    ) or any(
        audit.instrument_code != instrument
        for audit in state.week_records.values()
    )
    relabelled_seed = (
        not state.audits
        and (
            state.current_model_version != "M0001"
            or state.current_model_weights != _SEED_WEIGHTS[instrument]
        )
    )
    if mismatched_feedback or mismatched_audits or relabelled_seed:
        raise CrossMarketError(
            "Optimizer work state does not belong to its instrument_code"
        )


def _increment_version(value: str, prefix: str) -> str:
    if not value.startswith(prefix) or not value[1:].isdigit():
        raise ValueError(f"Invalid {prefix} version: {value}")
    return f"{prefix}{int(value[1:]) + 1:04d}"


def _candidate_weights(
    base: FrozenDict[str, Decimal],
    *,
    rejected_candidates: tuple[object, ...],
    validation_context_hash: str,
    candidate_cursor: int = 0,
) -> FrozenDict[str, Decimal] | None:
    _validate_weights(base)
    if candidate_cursor < 0:
        raise ValueError("candidate_cursor must be non-negative")
    rejected_fingerprints: set[tuple[tuple[str, Decimal], ...]] = set()
    for record in rejected_candidates:
        if not isinstance(record, Mapping):
            continue
        weights = record.get("candidate_weights")
        reasons = record.get("reasons")
        if (
            isinstance(weights, Mapping)
            and reasons
            and record.get("validation_context_hash")
            == validation_context_hash
        ):
            rejected_fingerprints.add(_weight_fingerprint(weights))

    options: list[FrozenDict[str, Decimal]] = []
    for step in (Decimal("1"), Decimal("2")):
        for donor in FEATURE_GROUPS:
            if base[donor] < step:
                continue
            for receiver in FEATURE_GROUPS:
                if receiver == donor:
                    continue
                candidate = dict(base)
                candidate[donor] -= step
                candidate[receiver] += step
                frozen = FrozenDict(candidate)
                _validate_weights(frozen)
                options.append(frozen)
    if not options:
        return None
    offset = candidate_cursor % len(options)
    for candidate in options[offset:] + options[:offset]:
        if _weight_fingerprint(candidate) not in rejected_fingerprints:
            return candidate
    return None


def _candidate_validation_context_hash(
    *,
    instrument_code: str,
    base_model_version: str,
    base_weights: FrozenDict[str, Decimal],
    feedback: tuple[IterationLabel, ...],
    active_windows: tuple[str, ...],
    current_metrics: FrozenDict[str, WindowMetrics],
    candidate_metrics: FrozenDict[str, WindowMetrics] | None,
    data_integrity: bool,
    evaluator_identity: str | None,
) -> str:
    material = {
        "active_windows": active_windows,
        "algorithm_version": ALGORITHM_VERSION,
        "base_model_version": base_model_version,
        "base_weights": base_weights,
        "current_metrics": {
            name: metric.to_dict()
            for name, metric in current_metrics.items()
        },
        "candidate_metrics": (
            None
            if candidate_metrics is None
            else {
                name: metric.to_dict()
                for name, metric in candidate_metrics.items()
            }
        ),
        "data_integrity": bool(data_integrity),
        "evaluator_identity": evaluator_identity,
        "feedback_digest": _feedback_digest(feedback),
        "feedback_count": len(feedback),
        "gate_parameters": _gate_parameters(),
        "instrument_code": instrument_code,
        "loss_version": LOSS_VERSION,
        "loss_weights": LOSS_WEIGHTS,
    }
    return hashlib.sha256(
        _canonical_json_dumps(material).encode("utf-8")
    ).hexdigest()


def _weight_fingerprint(
    weights: Mapping[str, object],
) -> tuple[tuple[str, Decimal], ...]:
    return tuple(
        sorted(
            (
                str(name),
                _as_decimal(value, f"candidate_weight:{name}"),
            )
            for name, value in weights.items()
        )
    )


def _coerce_metrics(
    values: Mapping[str, WindowMetrics | Mapping[str, object]]
    | RollingMetricWindows,
    instrument_code: str,
    active_windows: tuple[str, ...],
    mature_count: int,
) -> FrozenDict[str, WindowMetrics]:
    if (
        isinstance(values, RollingMetricWindows)
        and values.instrument_code not in {None, instrument_code}
    ):
        raise CrossMarketError("Metric bundle market does not match state")
    if (
        isinstance(values, RollingMetricWindows)
        and values.mature_count != mature_count
    ):
        raise ValueError(
            "Metric bundle mature_count must match visible feedback"
        )
    raw = values.windows if isinstance(values, RollingMetricWindows) else values
    if set(raw) != set(active_windows):
        raise ValueError(
            "Metric key set must exactly match active windows"
        )
    result: dict[str, WindowMetrics] = {}
    for key, value in raw.items():
        if isinstance(value, WindowMetrics):
            if value.instrument_code != instrument_code:
                raise CrossMarketError(
                    "Window metric market does not match state"
                )
            metric = value
        else:
            metric_market = value.get("instrument_code")
            if metric_market is None:
                raise CrossMarketError(
                    "Metric mapping requires instrument_code"
                )
            if metric_market != instrument_code:
                raise CrossMarketError(
                    "Window metric market does not match state"
                )
            metric = WindowMetrics(
                instrument_code=str(metric_market),
                window=str(value.get("window", key)),
                sample_count=_strict_nonnegative_int(
                    value["sample_count"],
                    "sample_count",
                ),
                total_loss=_as_decimal(value["total_loss"], "total_loss"),
                direction_hit_rate=_as_decimal(
                    value["direction_hit_rate"], "direction_hit_rate"
                ),
                calibration_loss=_as_decimal(
                    value["calibration_loss"], "calibration_loss"
                ),
                overtrade_penalty=_as_decimal(
                    value["overtrade_penalty"], "overtrade_penalty"
                ),
            )
        key_name = str(key)
        if metric.window != key_name:
            raise ValueError("Metric window name must match its key")
        expected_count = (
            mature_count if key_name == "expanding" else int(key_name)
        )
        if metric.sample_count != expected_count:
            raise ValueError(
                f"Metric sample_count for {key_name} must be "
                f"{expected_count}"
            )
        result[key_name] = metric
    return FrozenDict(result)


def _call_evaluator(
    evaluator: MetricEvaluator,
    weights: FrozenDict[str, Decimal],
    feedback: tuple[IterationLabel, ...],
    windows: tuple[str, ...],
) -> Mapping[str, WindowMetrics] | RollingMetricWindows:
    signature = inspect.signature(evaluator)
    parameters = tuple(signature.parameters.values())
    has_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    if has_kwargs or all(
        name in signature.parameters
        for name in ("weights", "feedback", "windows")
    ):
        result = evaluator(
            weights=weights,
            feedback=feedback,
            windows=windows,
        )
    elif len(positional) >= 3:
        result = evaluator(weights, feedback, windows)
    elif len(positional) == 2:
        result = evaluator(weights, feedback)
    else:
        result = evaluator(weights)
    return result


def _relative_improvement(
    current: Decimal,
    candidate: Decimal,
) -> Decimal:
    if current == 0:
        return Decimal("0") if candidate == 0 else Decimal("-Infinity")
    return (current - candidate) / current


def _gate_parameters() -> FrozenDict[str, Decimal]:
    return FrozenDict(
        {
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
            "other_window_max_deterioration": (
                MAX_OTHER_WINDOW_DETERIORATION
            ),
            "direction_hit_max_drop_points": (
                MAX_DIRECTION_HIT_DROP_POINTS
            ),
            "overtrade_max_relative_increase": (
                MAX_OVERTRADE_RELATIVE_INCREASE
            ),
        }
    )


def _acceptance_reasons(
    *,
    active_windows: tuple[str, ...],
    current: FrozenDict[str, WindowMetrics],
    candidate: FrozenDict[str, WindowMetrics],
    data_integrity: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not data_integrity:
        reasons.append("data_integrity_failed")
    if not active_windows:
        reasons.append("insufficient_mature_feedback")
        return tuple(reasons)
    missing = tuple(
        name
        for name in active_windows
        if name not in current or name not in candidate
    )
    if missing:
        reasons.extend(f"missing_metrics:{name}" for name in missing)
        return tuple(reasons)

    improvements = {
        name: _relative_improvement(
            current[name].total_loss,
            candidate[name].total_loss,
        )
        for name in active_windows
    }
    if len(active_windows) == 1:
        if improvements[active_windows[0]] < MIN_RELATIVE_IMPROVEMENT:
            reasons.append("loss_improvement_below_1_percent")
    else:
        if not any(
            improvement >= MIN_RELATIVE_IMPROVEMENT
            for improvement in improvements.values()
        ):
            reasons.append("no_window_improves_1_percent")
        if any(
            improvement < -MAX_OTHER_WINDOW_DETERIORATION
            for improvement in improvements.values()
        ):
            reasons.append("other_window_worsens_over_0_5_percent")

    guard_windows = (
        ("52",) if "52" in active_windows else active_windows
    )
    for name in guard_windows:
        if (
            candidate[name].direction_hit_rate
            < current[name].direction_hit_rate
            - MAX_DIRECTION_HIT_DROP_POINTS
        ):
            if "direction_hit_drop_exceeds_2_points" not in reasons:
                reasons.append("direction_hit_drop_exceeds_2_points")
            reasons.append(
                f"direction_hit_drop_exceeds_2_points:{name}"
            )

    for name in guard_windows:
        current_overtrade = current[name].overtrade_penalty
        candidate_overtrade = candidate[name].overtrade_penalty
        allowed = (
            current_overtrade
            * (Decimal("1") + MAX_OVERTRADE_RELATIVE_INCREASE)
            if current_overtrade > 0
            else Decimal("0")
        )
        if candidate_overtrade > allowed:
            reasons.append("overtrade_increase_exceeds_5_percent")
            break
    return tuple(reasons)


def _prediction_material(
    prediction: IterationPrediction | Mapping[str, object] | None,
) -> object:
    if prediction is None:
        return None
    if isinstance(prediction, Mapping):
        return dict(prediction)
    return {
        "cutoff_date": prediction.cutoff_date,
        "instrument_code": prediction.instrument_code,
        "iteration_id": prediction.iteration_id,
        "operation_side": prediction.operation_side,
        "predicted_action_date": prediction.predicted_action_date,
        "predicted_direction": prediction.predicted_direction,
        "probability": prediction.probability,
        "target_position": prediction.target_position,
        "trade_count": prediction.trade_count,
        "week_key": prediction.week_key,
    }


def _request_input_hash(
    *,
    instrument_code: str,
    feedback: tuple[IterationLabel, ...],
    week_key: str,
    cutoff_date: date,
    source_data_max_date: date | None,
    prediction: IterationPrediction | Mapping[str, object] | None,
    evaluator_identity: str | None,
    current_metrics: Mapping[
        str, WindowMetrics | Mapping[str, object]
    ]
    | RollingMetricWindows
    | None,
    candidate_metrics: Mapping[
        str, WindowMetrics | Mapping[str, object]
    ]
    | RollingMetricWindows
    | None,
    active_windows: tuple[str, ...],
    mature_count: int,
    data_integrity: bool,
) -> str:
    ordered_feedback = tuple(
        _label_to_dict(label)
        for label in sorted(
            feedback,
            key=lambda label: (
                label.instrument_code,
                label.iteration_id,
                label.observed_through,
                _LABEL_STATUS_RANK[label.status],
            ),
        )
    )
    material = {
        "candidate_metrics": _metric_input_material(
            candidate_metrics,
            instrument_code=instrument_code,
            active_windows=active_windows,
            mature_count=mature_count,
        ),
        "current_metrics": _metric_input_material(
            current_metrics,
            instrument_code=instrument_code,
            active_windows=active_windows,
            mature_count=mature_count,
        ),
        "cutoff_date": cutoff_date,
        "data_integrity": bool(data_integrity),
        "evaluator_identity": evaluator_identity,
        "feedback": ordered_feedback,
        "future_leakage": False,
        "instrument_code": instrument_code,
        "prediction": _prediction_material(prediction),
        "source_data_max_date": source_data_max_date,
        "week_key": week_key,
    }
    return hashlib.sha256(
        _canonical_json_dumps(material).encode("utf-8")
    ).hexdigest()


def _metric_input_material(
    values: Mapping[str, WindowMetrics | Mapping[str, object]]
    | RollingMetricWindows
    | None,
    *,
    instrument_code: str,
    active_windows: tuple[str, ...],
    mature_count: int,
) -> object:
    if values is None:
        return None
    try:
        normalized = RollingMetricWindows(
            instrument_code=instrument_code,
            windows=_coerce_metrics(
                values,
                instrument_code,
                active_windows,
                mature_count,
            ),
            mature_count=mature_count,
        )
    except ValueError:
        normalized = None
    if normalized is not None:
        return normalized.to_dict()
    raw = values.windows if isinstance(values, RollingMetricWindows) else values
    if isinstance(values, RollingMetricWindows):
        return values.to_dict()
    return {
        str(name): (
            metric.to_dict()
            if isinstance(metric, WindowMetrics)
            else dict(metric)
        )
        for name, metric in raw.items()
    }


__all__ = [
    "ALGORITHM_VERSION",
    "LOSS_VERSION",
    "LOSS_WEIGHTS",
    "ConflictingFeedback",
    "ConflictingWeekInput",
    "CrossMarketError",
    "DailyClose",
    "FutureLeakageError",
    "FeedbackRegressionError",
    "IterationLabel",
    "IterationPrediction",
    "OptimizerStepResult",
    "RollingMetricWindows",
    "StepAudit",
    "TradingCalendar",
    "WindowMetrics",
    "WorkState",
    "actual_turn_date",
    "rolling_metrics",
    "run_optimizer_step",
    "seed_model",
    "update_labels",
    "validate_step_audit_acceptance",
]
