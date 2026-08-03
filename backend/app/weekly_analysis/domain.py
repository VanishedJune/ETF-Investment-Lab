"""Immutable value objects for audited weekly/monthly market-data publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal


Timeframe = Literal["weekly", "monthly"]
Severity = Literal["warning", "blocking"]
VolumeAvailability = Literal[
    "available",
    "partially_unavailable",
    "not_available_for_direct_index",
]


@dataclass(frozen=True, slots=True)
class DailyBar:
    """One direct daily observation before period aggregation."""

    trade_date: date
    open_price: Decimal | None
    high_price: Decimal | None
    low_price: Decimal | None
    close_price: Decimal | None
    volume: Decimal | None = None
    source: str | None = None
    adjusted_close_price: Decimal | None = None
    turnover: Decimal | None = None


@dataclass(frozen=True, slots=True)
class PeriodBar:
    """One derived period with separate price and volume provenance."""

    timeframe: Timeframe
    period_start: date
    period_end: date
    open_price: Decimal | None
    high_price: Decimal | None
    low_price: Decimal | None
    close_price: Decimal | None
    volume: Decimal | None
    source: str
    volume_source: str
    price_source: str
    adjusted_close_price: Decimal | None = None
    turnover: Decimal | None = None


@dataclass(frozen=True, slots=True)
class UpstreamWeeklyVolume:
    """A native direct-index weekly volume candidate."""

    trade_date: date
    volume: Decimal | None
    source: str
    unit: str


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    severity: Severity
    start_date: date | None
    end_date: date | None
    message: str


@dataclass(frozen=True, slots=True)
class QualityReport:
    is_publishable: bool
    issues: tuple[QualityIssue, ...]
    volume_availability: VolumeAvailability


@dataclass(frozen=True, slots=True)
class AggregationResult:
    bars: tuple[PeriodBar, ...]
    report: QualityReport
