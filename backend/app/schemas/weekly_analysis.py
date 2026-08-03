"""Public request and response contracts for weekly-analysis V2 APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
import math
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


InstrumentCode = Literal["399006", "NDX"]
TaskId = Annotated[
    str,
    StringConstraints(pattern=r"^weekly-v2-[0-9a-f]{32}$"),
]
AnalysisTaskStatus = Literal[
    "queued",
    "preparing_data",
    "building_features",
    "iterating",
    "validating",
    "generating_advice",
    "completed",
    "failed",
    "recoverable",
]


class AnalysisTaskCreate(BaseModel):
    """A production request intentionally has no caller-supplied cutoff."""

    model_config = ConfigDict(extra="forbid")

    instrument_code: str = Field(min_length=1, max_length=16)


class AnalysisTaskRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: TaskId
    instrument_code: InstrumentCode
    status: AnalysisTaskStatus
    completed_weeks: int = Field(ge=0)
    total_weeks: int | None = Field(default=None, ge=0)
    last_iteration: int = Field(ge=0)
    last_work_version: str
    last_model_version: str
    last_completed_week: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    reused: bool = False

    @field_validator("progress")
    @classmethod
    def progress_must_be_finite(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        _require_finite(value)
        return value


class WindowMetricRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument_code: InstrumentCode
    window: str
    sample_count: int = Field(ge=0)
    total_loss: Decimal
    direction_hit_rate: Decimal
    calibration_loss: Decimal
    overtrade_penalty: Decimal

    @model_validator(mode="after")
    def values_must_be_finite(self) -> WindowMetricRead:
        _require_finite(self.model_dump(mode="python"))
        return self


class IterationMetricRead(BaseModel):
    """One auditable point for the model-improvement chart."""

    model_config = ConfigDict(extra="forbid")

    iteration_number: int = Field(gt=0)
    iteration_id: str
    cutoff_date: date
    model_version: str
    accepted: bool
    deviation_days: int | None = None
    absolute_deviation: Decimal | None = None
    rolling_20_abs_deviation: Decimal | None = None
    rolling_52_abs_deviation: Decimal | None = None
    direction_correct: bool | None = None

    @model_validator(mode="after")
    def values_must_be_finite(self) -> IterationMetricRead:
        _require_finite(self.model_dump(mode="python"))
        return self


class ModelMetricsRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: InstrumentCode
    model_version: str
    iteration_count: int = Field(ge=0)
    last_iteration: int = Field(ge=0)
    accepted_iterations: int = Field(ge=0)
    rejected_iterations: int = Field(ge=0)
    feedback_count: int = Field(ge=0)
    mature_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    candidate_acceptance_rate: Decimal
    windows: dict[str, WindowMetricRead] = Field(default_factory=dict)
    curve: tuple[IterationMetricRead, ...] = ()

    @model_validator(mode="after")
    def values_must_be_finite(self) -> ModelMetricsRead:
        _require_finite(self.model_dump(mode="python"))
        return self


class AdviceBatchRead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_condition: str
    expected_date: date
    operation_side: Literal["buy", "sell"]
    percent: int = Field(gt=0, multiple_of=5)
    sequence: int = Field(gt=0)
    status: Literal["ready", "waiting_confirmation"]
    tolerance_trading_days: Literal[3]


class LatestAdviceRead(BaseModel):
    """Advice payload plus public I/W/M provenance, never a local path."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    instrument_code: InstrumentCode
    iteration_id: str
    work_version: str
    model_version: str
    advice_generation: int = Field(ge=1)
    advice_at: datetime
    advice_version: str
    audit_fields: dict[str, Any]
    audit_notes: tuple[str, ...]
    batches: tuple[AdviceBatchRead, ...]
    confidence: Decimal
    current_position: int | None
    data_cutoff_date: date
    direction: Literal["up", "down", "neutral"]
    direction_probabilities: dict[str, Decimal]
    feature_set_version: str
    forecast_horizon_weeks: Literal[13]
    fund_etf_ratio: Literal["7:3", "6:4", "5:5", "4:6", "3:7"]
    market_state: str
    operation_side: Literal["buy", "sell", "hold"]
    probability: Decimal
    recommendation: Literal[
        "staged_increase",
        "staged_decrease",
        "hold",
        "wait_confirmation",
        "analysis_only",
    ]
    signed_position_change: int | None
    source_data_max_date: date | None
    target_position: int = Field(ge=0, le=100)
    target_position_range: tuple[int, int]
    total_adjustment: int | None

    @field_validator("advice_at")
    @classmethod
    def advice_time_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("advice_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def values_must_be_finite(self) -> LatestAdviceRead:
        _require_finite(self.model_dump(mode="python"))
        return self


def _require_finite(value: object) -> None:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("numeric values must be finite")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("numeric values must be finite")
        return
    if isinstance(value, BaseModel):
        _require_finite(value.model_dump(mode="python"))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite(key)
            _require_finite(item)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _require_finite(item)


__all__ = [
    "AdviceBatchRead",
    "AnalysisTaskCreate",
    "AnalysisTaskRead",
    "AnalysisTaskStatus",
    "InstrumentCode",
    "LatestAdviceRead",
    "IterationMetricRead",
    "ModelMetricsRead",
    "TaskId",
    "WindowMetricRead",
]
