"""Typed contracts for local investment-plan and simulation operations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ExecutionPriceRule = Literal["CLOSE", "OPEN", "PREVIOUS_CLOSE"]
SimulationMode = Literal["REAL_LOT", "FRACTIONAL"]


def _finite_positive(value: Decimal, field: str) -> Decimal:
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field} must be a finite positive decimal")
    return value


class _PlanBase(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    weekly_amount: Decimal = Field(...)
    execution_weekday: int = Field(default=1, ge=0, le=6)
    execution_price_rule: ExecutionPriceRule = "CLOSE"
    start_date: date | None = None
    end_date: date | None = None
    minimum_weekly_amount: Decimal | None = None
    maximum_weekly_amount: Decimal | None = None
    enabled: bool = True
    allow_pause: bool = True
    mode: SimulationMode = "REAL_LOT"
    lot_size: int = Field(default=100, ge=1)

    @field_validator("weekly_amount")
    @classmethod
    def validate_weekly_amount(cls, value: Decimal) -> Decimal:
        return _finite_positive(value, "weekly_amount")

    @field_validator("minimum_weekly_amount", "maximum_weekly_amount")
    @classmethod
    def validate_bound(cls, value: Decimal | None, info: object) -> Decimal | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "weekly amount bound")
        return _finite_positive(value, field_name)

    @model_validator(mode="after")
    def validate_schedule(self) -> "_PlanBase":
        if self.end_date is not None and self.start_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date cannot be earlier than start_date")
        if (
            self.minimum_weekly_amount is not None
            and self.maximum_weekly_amount is not None
            and self.minimum_weekly_amount > self.maximum_weekly_amount
        ):
            raise ValueError("minimum_weekly_amount cannot exceed maximum_weekly_amount")
        if self.minimum_weekly_amount is not None and self.weekly_amount < self.minimum_weekly_amount:
            raise ValueError("weekly_amount cannot be less than minimum_weekly_amount")
        if self.maximum_weekly_amount is not None and self.weekly_amount > self.maximum_weekly_amount:
            raise ValueError("weekly_amount cannot exceed maximum_weekly_amount")
        return self


class InvestmentPlanCreate(_PlanBase):
    instrument_code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)


class InvestmentPlanUpdate(BaseModel):
    """Partial update fields, validated after merging with the persisted plan."""

    model_config = ConfigDict(use_enum_values=True)

    name: str | None = Field(default=None, min_length=1, max_length=128)
    weekly_amount: Decimal | None = None
    execution_weekday: int | None = Field(default=None, ge=0, le=6)
    execution_price_rule: ExecutionPriceRule | None = None
    start_date: date | None = None
    end_date: date | None = None
    minimum_weekly_amount: Decimal | None = None
    maximum_weekly_amount: Decimal | None = None
    enabled: bool | None = None
    allow_pause: bool | None = None
    mode: SimulationMode | None = None
    lot_size: int | None = Field(default=None, ge=1)


class InvestmentPlanRead(_PlanBase):
    id: int
    instrument_id: int
    instrument_code: str
    name: str


class SimulationAccountCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    initial_cash: Decimal = Decimal("0")
    strategy_id: int | None = None
    enabled: bool = True
    notes: str | None = None

    @field_validator("initial_cash")
    @classmethod
    def validate_initial_cash(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value < 0:
            raise ValueError("initial_cash must be a finite nonnegative decimal")
        return value


class SimulationAccountRead(BaseModel):
    id: int
    name: str
    initial_cash: Decimal
    cash_balance: Decimal
    portfolio_value: Decimal
    strategy_id: int | None
    enabled: bool


class SimulationExecutionSettings(BaseModel):
    """Persisted plan settings revalidated immediately before execution."""

    model_config = ConfigDict(use_enum_values=True)

    mode: SimulationMode = "REAL_LOT"
    lot_size: int = Field(default=100, ge=1)
    execution_price_rule: ExecutionPriceRule = "CLOSE"
