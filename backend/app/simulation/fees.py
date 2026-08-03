"""Exact, immutable ETF trade-fee calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping


ZERO = Decimal("0")
DEFAULT_COMMISSION_RATE = Decimal("0.00025")
DEFAULT_MINIMUM_COMMISSION = Decimal("5")


def _decimal(value: Decimal | int | float | str, field: str) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    if not result.is_finite() or result < ZERO:
        raise ValueError(f"{field} must be a finite nonnegative decimal")
    return result


@dataclass(frozen=True, slots=True)
class FeeConfiguration:
    """Durable ETF fee settings; values are copied into each trade record."""

    buy_commission_rate: Decimal = DEFAULT_COMMISSION_RATE
    sell_commission_rate: Decimal = DEFAULT_COMMISSION_RATE
    minimum_commission: Decimal = DEFAULT_MINIMUM_COMMISSION
    minimum_commission_enabled: bool = True
    etf_stamp_duty_rate: Decimal = ZERO
    transfer_fee_rate: Decimal = ZERO
    other_fee_rate: Decimal = ZERO

    def __post_init__(self) -> None:
        for name in (
            "buy_commission_rate",
            "sell_commission_rate",
            "minimum_commission",
            "etf_stamp_duty_rate",
            "transfer_fee_rate",
            "other_fee_rate",
        ):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        if not isinstance(self.minimum_commission_enabled, bool):
            raise ValueError("minimum_commission_enabled must be a boolean")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "FeeConfiguration":
        return cls(**dict(values or {}))

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "buy_commission_rate": str(self.buy_commission_rate),
            "sell_commission_rate": str(self.sell_commission_rate),
            "minimum_commission": str(self.minimum_commission),
            "minimum_commission_enabled": self.minimum_commission_enabled,
            "etf_stamp_duty_rate": str(self.etf_stamp_duty_rate),
            "transfer_fee_rate": str(self.transfer_fee_rate),
            "other_fee_rate": str(self.other_fee_rate),
        }


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    """Fees and their frozen configuration inputs for one simulated order."""

    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    other_fee: Decimal
    commission_rate: Decimal
    stamp_duty_rate: Decimal
    transfer_fee_rate: Decimal
    other_fee_rate: Decimal
    minimum_commission: Decimal
    minimum_commission_enabled: bool

    @property
    def total(self) -> Decimal:
        return self.commission + self.stamp_duty + self.transfer_fee + self.other_fee


def calculate_trade_fees(
    gross_amount: Decimal | int | float | str,
    side: str,
    configuration: FeeConfiguration | Mapping[str, object] | None = None,
) -> FeeBreakdown:
    """Calculate fees from a nonnegative gross amount without binary rounding."""
    amount = _decimal(gross_amount, "gross_amount")
    normalized_side = side.upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    config = (
        configuration
        if isinstance(configuration, FeeConfiguration)
        else FeeConfiguration.from_mapping(configuration)
    )
    commission_rate = (
        config.buy_commission_rate if normalized_side == "BUY" else config.sell_commission_rate
    )
    commission = amount * commission_rate
    if config.minimum_commission_enabled and amount > ZERO:
        commission = max(commission, config.minimum_commission)
    stamp_duty = amount * config.etf_stamp_duty_rate
    return FeeBreakdown(
        commission=commission,
        stamp_duty=stamp_duty,
        transfer_fee=amount * config.transfer_fee_rate,
        other_fee=amount * config.other_fee_rate,
        commission_rate=commission_rate,
        stamp_duty_rate=config.etf_stamp_duty_rate,
        transfer_fee_rate=config.transfer_fee_rate,
        other_fee_rate=config.other_fee_rate,
        minimum_commission=config.minimum_commission,
        minimum_commission_enabled=config.minimum_commission_enabled,
    )
