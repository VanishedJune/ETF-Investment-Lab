"""Pure weekly-order sizing rules for the local paper-simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN

from .fees import FeeBreakdown, FeeConfiguration, calculate_trade_fees


ZERO = Decimal("0")
MONEY_QUANTUM = Decimal("0.00000001")


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_EVEN)


def quantity(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True)
class BuyOrder:
    requested_amount: Decimal
    available_cash: Decimal
    quantity: Decimal
    price: Decimal
    gross_amount: Decimal
    fees: FeeBreakdown
    theoretical: bool
    reason: str | None = None

    @property
    def total_debit(self) -> Decimal:
        return money(self.gross_amount + self.fees.total)


def _validate_positive(value: Decimal, name: str) -> Decimal:
    if not value.is_finite() or value <= ZERO:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _fees_for(gross: Decimal, configuration: FeeConfiguration) -> FeeBreakdown:
    raw = calculate_trade_fees(gross, "BUY", configuration)
    return FeeBreakdown(
        commission=money(raw.commission),
        stamp_duty=money(raw.stamp_duty),
        transfer_fee=money(raw.transfer_fee),
        other_fee=money(raw.other_fee),
        commission_rate=raw.commission_rate,
        stamp_duty_rate=raw.stamp_duty_rate,
        transfer_fee_rate=raw.transfer_fee_rate,
        other_fee_rate=raw.other_fee_rate,
        minimum_commission=raw.minimum_commission,
        minimum_commission_enabled=raw.minimum_commission_enabled,
    )


def size_buy_order(
    requested_amount: Decimal,
    carried_cash: Decimal,
    price: Decimal,
    lot_size: int,
    mode: str,
    configuration: FeeConfiguration,
) -> BuyOrder:
    """Fund a buy only from this allocation plus genuinely carried cash.

    The result carries no external quote or future data assumptions.  Real-lot
    sizing walks down from the maximal lot so a minimum commission cannot make
    the final debit exceed the available cash.
    """
    requested = _validate_positive(requested_amount, "requested_amount")
    carry = carried_cash
    if not carry.is_finite() or carry < ZERO:
        raise ValueError("carried_cash must be finite and nonnegative")
    execution_price = _validate_positive(price, "price")
    if lot_size < 1:
        raise ValueError("lot_size must be positive")
    available = money(requested + carry)
    normalized_mode = mode.upper()
    if normalized_mode not in {"REAL_LOT", "FRACTIONAL"}:
        raise ValueError("mode must be REAL_LOT or FRACTIONAL")
    zero_fees = _fees_for(ZERO, configuration)
    if normalized_mode == "REAL_LOT":
        lots = int((available / execution_price / Decimal(lot_size)).to_integral_value(rounding=ROUND_DOWN))
        while lots > 0:
            order_quantity = Decimal(lots * lot_size)
            gross = money(order_quantity * execution_price)
            fees = _fees_for(gross, configuration)
            if money(gross + fees.total) <= available:
                return BuyOrder(requested, available, order_quantity, execution_price, gross, fees, False)
            lots -= 1
        return BuyOrder(requested, available, ZERO, execution_price, ZERO, zero_fees, False, "INSUFFICIENT_FOR_LOT")

    rate_without_minimum = (
        configuration.buy_commission_rate
        + configuration.etf_stamp_duty_rate
        + configuration.transfer_fee_rate
        + configuration.other_fee_rate
    )
    other_rate = (
        configuration.etf_stamp_duty_rate
        + configuration.transfer_fee_rate
        + configuration.other_fee_rate
    )
    candidate = available / (execution_price * (Decimal("1") + rate_without_minimum))
    if configuration.minimum_commission_enabled:
        minimum_candidate = (available - configuration.minimum_commission) / (
            execution_price * (Decimal("1") + other_rate)
        )
        if minimum_candidate > ZERO:
            minimum_gross = minimum_candidate * execution_price
            if minimum_gross * configuration.buy_commission_rate <= configuration.minimum_commission:
                candidate = minimum_candidate
    order_quantity = max(ZERO, quantity(candidate))
    while order_quantity > ZERO:
        gross = money(order_quantity * execution_price)
        fees = _fees_for(gross, configuration)
        if money(gross + fees.total) <= available:
            return BuyOrder(requested, available, order_quantity, execution_price, gross, fees, True)
        order_quantity = quantity(order_quantity - MONEY_QUANTUM)
    return BuyOrder(requested, available, ZERO, execution_price, ZERO, zero_fees, True, "INSUFFICIENT_FUNDS")
