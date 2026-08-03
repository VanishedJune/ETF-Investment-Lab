"""Typed, source-labelled valuation data exchanged inside the local research app."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ValuationDataRecord(BaseModel):
    """One source observation.  It is never a synthetic or ETF-proxy value."""

    valuation_date: date
    pe_ratio: Decimal | None = None
    pb_ratio: Decimal | None = None
    dividend_yield: Decimal | None = None
    source: str
    source_url: str | None = None
    proxy: bool = False
    notes: str | None = None
    raw_values: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def requires_a_real_metric(self) -> "ValuationDataRecord":
        if all(value is None for value in (self.pe_ratio, self.pb_ratio, self.dividend_yield)):
            raise ValueError("A valuation record needs PE, PB, or dividend yield")
        if self.proxy:
            raise ValueError("The direct-index valuation pipeline does not accept proxy values")
        return self


class ValuationProviderResult(BaseModel):
    """Non-throwing outcome from a public valuation source."""

    source: str
    fetched_at: datetime = Field(default_factory=utc_now)
    records: list[ValuationDataRecord] = Field(default_factory=list)
    status: Literal["success", "unsupported", "error"] = "success"
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def success(cls, source: str, records: list[ValuationDataRecord]) -> "ValuationProviderResult":
        return cls(source=source, records=records)

    @classmethod
    def failed(
        cls, source: str, error: str, *, status: Literal["unsupported", "error"] = "error"
    ) -> "ValuationProviderResult":
        return cls(source=source, status=status, error=error, warnings=[error])


class ValuationUpdateResponse(BaseModel):
    source: str
    status: Literal["success", "cached", "unsupported", "error"]
    fetched_at: datetime = Field(default_factory=utc_now)
    records_received: int = 0
    records_written: int = 0
    records_added: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    cached_records: int = 0
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
