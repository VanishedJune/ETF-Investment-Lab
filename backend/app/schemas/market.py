"""Portable, explicitly-labelled market-data value objects."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProviderStatus(str, Enum):
    SUCCESS = "success"
    CACHED = "cached"
    DEMO = "demo"
    ERROR = "error"
    INACTIVE = "inactive"


class MarketDataRecord(BaseModel):
    """A normalized daily OHLCV/amount record before it reaches SQLite."""

    trade_date: date
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    adjusted_close_price: Decimal | None = None
    volume: Decimal | None = None
    volume_multiplier: int = Field(default=1, ge=1)
    amount: Decimal | None = None
    source: str | None = None
    volume_source: str | None = None

    @model_validator(mode="after")
    def validate_ohlcv(self) -> "MarketDataRecord":
        values = {
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "volume": self.volume,
            "amount": self.amount,
            "adjusted_close_price": self.adjusted_close_price,
        }
        for name, value in values.items():
            if value is None:
                continue
            if value.as_tuple().exponent < -8:
                raise ValueError(f"{name} supports at most 8 decimal places")
        for name in ("open_price", "high_price", "low_price", "close_price"):
            if values[name] <= 0:
                raise ValueError(f"{name} must be strictly positive")
        if self.adjusted_close_price is not None and self.adjusted_close_price <= 0:
            raise ValueError("adjusted_close_price must be strictly positive")
        if self.volume is not None and self.volume < 0:
            raise ValueError("volume must be nonnegative")
        if self.amount is not None and self.amount < 0:
            raise ValueError("amount must be nonnegative")
        if self.high_price < max(self.open_price, self.close_price, self.low_price):
            raise ValueError("high_price must be at least open_price, close_price, and low_price")
        if self.low_price > min(self.open_price, self.close_price, self.high_price):
            raise ValueError("low_price must be at most open_price, close_price, and high_price")
        return self


class ProviderResult(BaseModel):
    """The common result returned by every local/public market-data provider."""

    source: str
    fetched_at: datetime = Field(default_factory=utc_now)
    cutoff_date: date | None = None
    records: list[MarketDataRecord] = Field(default_factory=list)
    cache_used: bool = False
    demo: bool = False
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    status: ProviderStatus = ProviderStatus.SUCCESS
    volume_availability: Literal[
        "available",
        "partially_unavailable",
        "not_available_for_direct_index",
    ] = "available"

    @classmethod
    def success(
        cls,
        source: str,
        records: list[MarketDataRecord],
        *,
        warnings: list[str] | None = None,
        volume_availability: Literal[
            "available",
            "partially_unavailable",
            "not_available_for_direct_index",
        ] = "available",
    ) -> "ProviderResult":
        return cls(
            source=source,
            records=records,
            cutoff_date=max((record.trade_date for record in records), default=None),
            warnings=warnings or [],
            volume_availability=volume_availability,
        )

    @classmethod
    def failed(
        cls,
        source: str,
        error: str,
        *,
        status: ProviderStatus = ProviderStatus.ERROR,
        volume_availability: Literal[
            "available",
            "partially_unavailable",
            "not_available_for_direct_index",
        ] = "available",
    ) -> "ProviderResult":
        return cls(
            source=source,
            error=error,
            status=status,
            warnings=[error],
            volume_availability=volume_availability,
        )


class DataUpdateResponse(ProviderResult):
    """Storage outcome, including safe cached and demo fallbacks."""

    records_received: int = 0
    records_written: int = 0
    cached_records: int = 0
    records_added: int = 0
    records_updated: int = 0
    records_skipped: int = 0


class CsvRowError(BaseModel):
    row_number: int
    field: str | None = None
    message: str


class CsvPreview(BaseModel):
    headers: list[str] = Field(default_factory=list)
    mapping: dict[str, str] = Field(default_factory=dict)
    sample: list[MarketDataRecord] = Field(default_factory=list)
    valid_rows: int = 0
    invalid_rows: int = 0
    row_errors: list[CsvRowError] = Field(default_factory=list)


class CsvImportResult(BaseModel):
    instrument_code: str
    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    row_errors: list[CsvRowError] = Field(default_factory=list)
