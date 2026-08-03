"""Typed local-only contracts for manually recorded real-account activity."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RealTradeSide = Literal["BUY", "SELL"]
MAX_FIXED_POINT_DECIMALS = 8


def _finite_nonnegative(value: Decimal, field_name: str) -> Decimal:
    if not value.is_finite() or value < Decimal("0"):
        raise ValueError(f"{field_name} must be a finite nonnegative decimal")
    return value


def _finite_positive(value: Decimal, field_name: str) -> Decimal:
    if not value.is_finite() or value <= Decimal("0"):
        raise ValueError(f"{field_name} must be a finite positive decimal")
    return value


def _fixed_point(value: Decimal, field_name: str) -> Decimal:
    """Reject values SQLite cannot store exactly before a ledger transaction opens."""
    if value.as_tuple().exponent < -MAX_FIXED_POINT_DECIMALS:
        raise ValueError(f"{field_name} supports at most {MAX_FIXED_POINT_DECIMALS} decimal places")
    return value


class RealAccountCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    broker_name: str | None = Field(default=None, max_length=128)
    initial_cash: Decimal = Decimal("0")
    enabled: bool = True
    notes: str | None = None

    @field_validator("initial_cash")
    @classmethod
    def validate_initial_cash(cls, value: Decimal) -> Decimal:
        return _finite_nonnegative(value, "initial_cash")


class RealAccountRead(BaseModel):
    id: int
    name: str
    broker_name: str | None
    initial_cash: Decimal
    cash_balance: Decimal
    enabled: bool


class RealTransactionCreate(BaseModel):
    """The intentionally small real-trade form.

    Quantity, trade value and fee are ledger outputs.  Keeping them out of the
    public input contract prevents a manually entered value from drifting away
    from the linked investment plan or the chronological holdings replay.
    """

    model_config = ConfigDict(use_enum_values=True)

    instrument_code: str = Field(min_length=1, max_length=32)
    side: RealTradeSide
    transaction_date: date
    price: Decimal
    notes: str | None = None

    @field_validator("instrument_code")
    @classmethod
    def normalize_instrument_code(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("instrument_code cannot be blank")
        return normalized

    @field_validator("price")
    @classmethod
    def validate_positive(cls, value: Decimal, info: object) -> Decimal:
        field_name = getattr(info, "field_name", "value")
        return _fixed_point(_finite_positive(value, field_name), field_name)

class RealTransactionUpdate(BaseModel):
    """Partial transaction payload; the service revalidates it after merge."""

    model_config = ConfigDict(use_enum_values=True)

    instrument_code: str | None = Field(default=None, min_length=1, max_length=32)
    side: RealTradeSide | None = None
    transaction_date: date | None = None
    price: Decimal | None = None
    notes: str | None = None

    @field_validator("price")
    @classmethod
    def validate_positive(cls, value: Decimal | None, info: object) -> Decimal | None:
        if value is None:
            return value
        field_name = getattr(info, "field_name", "value")
        return _fixed_point(_finite_positive(value, field_name), field_name)

class RealTransactionRead(BaseModel):
    id: int
    account_id: int
    instrument_id: int
    instrument_code: str
    side: RealTradeSide
    transaction_date: date
    quantity: Decimal
    price: Decimal
    amount: Decimal
    fee: Decimal
    planned_amount: Decimal
    cash_after: Decimal
    holding_quantity_after: Decimal
    average_cost_after: Decimal
    cost_basis: Decimal
    net_proceeds: Decimal
    realized_pnl: Decimal
    cumulative_realized_pnl: Decimal
    current_unrealized_pnl: Decimal
    total_pnl: Decimal
    total_return: Decimal | None
    market_snapshot: dict[str, object] = Field(default_factory=dict)
    notes: str | None


class RealPositionRead(BaseModel):
    id: int
    account_id: int
    instrument_id: int
    instrument_code: str
    quantity: Decimal
    average_cost: Decimal
    total_cost: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    fees_paid: Decimal


class RealAccountSnapshotRead(BaseModel):
    snapshot_date: date
    total_contribution: Decimal
    cash_balance: Decimal
    market_value: Decimal
    total_assets: Decimal
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    holding_quantity: Decimal
    benchmark_total_assets: Decimal | None
    benchmark_total_pnl: Decimal | None
    benchmark_quantity: Decimal | None
    fees_paid: Decimal
    total_return: Decimal | None
    time_weighted_return: Decimal | None
    money_weighted_return: Decimal | None
    annualized_return: Decimal | None
    max_drawdown: Decimal | None


class CsvImportSummary(BaseModel):
    imported: int
    rejected: int
    transaction_ids: list[int] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
