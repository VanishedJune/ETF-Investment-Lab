"""Exact average-cost accounting and snapshot return calculations."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Iterable

from .engine import money


ZERO = Decimal("0")
MINIMUM_ANNUALIZATION_DAYS = 30


@dataclass(frozen=True, slots=True)
class PositionState:
    quantity: Decimal = ZERO
    average_cost: Decimal = ZERO
    total_cost: Decimal = ZERO
    realized_pnl: Decimal = ZERO


@dataclass(frozen=True, slots=True)
class TradeAccountingResult:
    cash_balance: Decimal
    position: PositionState
    realized_pnl: Decimal = ZERO


def apply_buy(
    cash_balance: Decimal, position: PositionState, quantity: Decimal, price: Decimal, fees: Decimal
) -> TradeAccountingResult:
    gross = money(quantity * price)
    debit = money(gross + fees)
    if debit > cash_balance:
        raise ValueError("insufficient cash for simulated buy")
    new_quantity = money(position.quantity + quantity)
    new_total_cost = money(position.total_cost + debit)
    average_cost = money(new_total_cost / new_quantity) if new_quantity else ZERO
    return TradeAccountingResult(
        cash_balance=money(cash_balance - debit),
        position=PositionState(new_quantity, average_cost, new_total_cost, position.realized_pnl),
    )


def apply_sell(
    cash_balance: Decimal, position: PositionState, quantity: Decimal, price: Decimal, fees: Decimal
) -> TradeAccountingResult:
    if quantity <= ZERO:
        raise ValueError("sell quantity must be positive")
    if quantity > position.quantity:
        raise ValueError("sell quantity cannot exceed holding")
    gross = money(quantity * price)
    is_full_sale = quantity == position.quantity
    # The stored average is intentionally rounded to SQLite's fixed-point scale.
    # On a full exit, release the exact accumulated cost basis instead of
    # multiplying that rounded display value back by quantity.
    cost_released = position.total_cost if is_full_sale else money(position.average_cost * quantity)
    proceeds = money(gross - fees)
    realized = money(proceeds - cost_released)
    remaining_quantity = money(position.quantity - quantity)
    remaining_cost = ZERO if is_full_sale else money(position.total_cost - cost_released)
    average_cost = money(remaining_cost / remaining_quantity) if remaining_quantity else ZERO
    return TradeAccountingResult(
        cash_balance=money(cash_balance + proceeds),
        position=PositionState(
            remaining_quantity,
            average_cost,
            remaining_cost,
            money(position.realized_pnl + realized),
        ),
        realized_pnl=realized,
    )


@dataclass(frozen=True, slots=True)
class ReturnMetrics:
    daily_return: Decimal | None
    cumulative_return: Decimal | None
    time_weighted_return: Decimal | None
    money_weighted_return: Decimal | None
    annualized_return: Decimal | None
    max_drawdown: Decimal | None


def calculate_return_metrics(
    *,
    total_assets: Decimal,
    prior_assets: Decimal | None,
    external_cash_flow: Decimal,
    contributed_capital: Decimal,
    historical_assets: Iterable[Decimal],
    prior_time_weighted_return: Decimal | None = None,
    cashflow_history: Iterable[tuple[date, Decimal]] = (),
    valuation_date: date | None = None,
    initial_cash: Decimal = ZERO,
) -> ReturnMetrics:
    cumulative = (
        money((total_assets - contributed_capital) / contributed_capital)
        if contributed_capital > ZERO
        else None
    )
    daily = None
    if prior_assets is not None and prior_assets > ZERO:
        daily = money((total_assets - external_cash_flow) / prior_assets - Decimal("1"))
    time_weighted = None
    if daily is not None:
        previous_chain = prior_time_weighted_return if prior_time_weighted_return is not None else ZERO
        time_weighted = money((Decimal("1") + previous_chain) * (Decimal("1") + daily) - Decimal("1"))
    values = [*historical_assets, total_assets]
    peak = ZERO
    drawdowns: list[Decimal] = []
    for value in values:
        peak = max(peak, value)
        if peak > ZERO:
            drawdowns.append(money(value / peak - Decimal("1")))
    max_drawdown = min(drawdowns) if drawdowns else None
    flows = list(cashflow_history)
    if initial_cash > ZERO and valuation_date is not None:
        first_date = min((flow_date for flow_date, _flow in flows), default=valuation_date)
        flows.append((first_date, initial_cash))
    money_weighted = _money_weighted_return(flows, total_assets, valuation_date)
    annualized = _annualized_return(time_weighted, flows, valuation_date)
    return ReturnMetrics(daily, cumulative, time_weighted, money_weighted, annualized, max_drawdown)


def _money_weighted_return(
    cashflows: list[tuple[date, Decimal]], ending_value: Decimal, valuation_date: date | None
) -> Decimal | None:
    """Return XIRR when a dated contribution and a later valuation exist."""
    if valuation_date is None:
        return None
    nonzero = [(flow_date, value) for flow_date, value in cashflows if value != ZERO]
    if not nonzero:
        return None
    start_date = min(flow_date for flow_date, _value in nonzero)
    if (valuation_date - start_date).days < MINIMUM_ANNUALIZATION_DAYS:
        return None
    dated = [
        (Decimal((flow_date - start_date).days) / Decimal("365"), value)
        for flow_date, value in nonzero
    ]
    ending_years = Decimal((valuation_date - start_date).days) / Decimal("365")

    def net_present_value(rate: Decimal) -> Decimal:
        base = Decimal("1") + rate
        with localcontext() as context:
            context.prec = 48
            logarithm = base.ln()
            return sum(
                (-value / (logarithm * years).exp()) for years, value in dated
            ) + ending_value / (logarithm * ending_years).exp()

    lower, upper = Decimal("-0.999999"), Decimal("1")
    lower_value, upper_value = net_present_value(lower), net_present_value(upper)
    for _ in range(32):
        if lower_value == 0:
            return money(lower)
        if upper_value == 0:
            return money(upper)
        if lower_value * upper_value < 0:
            break
        upper *= Decimal("2")
        upper_value = net_present_value(upper)
    else:
        return None
    for _ in range(96):
        midpoint = (lower + upper) / Decimal("2")
        midpoint_value = net_present_value(midpoint)
        if abs(midpoint_value) < Decimal("1e-20"):
            return money(midpoint)
        if lower_value * midpoint_value <= 0:
            upper, upper_value = midpoint, midpoint_value
        else:
            lower, lower_value = midpoint, midpoint_value
    return money((lower + upper) / Decimal("2"))


def _annualized_return(
    time_weighted_return: Decimal | None,
    cashflows: list[tuple[date, Decimal]],
    valuation_date: date | None,
) -> Decimal | None:
    if time_weighted_return is None or valuation_date is None:
        return None
    dates = [flow_date for flow_date, _value in cashflows]
    if not dates:
        return None
    days = (valuation_date - min(dates)).days
    if days < MINIMUM_ANNUALIZATION_DAYS or time_weighted_return <= Decimal("-1"):
        return None
    with localcontext() as context:
        context.prec = 48
        return money(
            ((Decimal("1") + time_weighted_return).ln() * Decimal("365") / Decimal(days)).exp()
            - Decimal("1")
        )
