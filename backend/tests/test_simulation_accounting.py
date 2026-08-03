from datetime import date
from decimal import Decimal

from backend.app.simulation.accounting import (
    PositionState,
    apply_buy,
    apply_sell,
    calculate_return_metrics,
)


def test_full_sell_releases_exact_stored_total_cost_despite_rounded_average_cost() -> None:
    first_buy = apply_buy(
        Decimal("10"), PositionState(), Decimal("1"), Decimal("1"), Decimal("0")
    )
    second_buy = apply_buy(
        first_buy.cash_balance,
        first_buy.position,
        Decimal("2"),
        Decimal("2"),
        Decimal("0"),
    )

    sold = apply_sell(
        second_buy.cash_balance,
        second_buy.position,
        Decimal("3"),
        Decimal("2"),
        Decimal("0"),
    )

    assert second_buy.position.total_cost == Decimal("5")
    assert sold.realized_pnl == Decimal("1")
    assert sold.position.quantity == Decimal("0")
    assert sold.position.average_cost == Decimal("0")
    assert sold.position.total_cost == Decimal("0")


def test_migrated_cost_basis_supports_exact_partial_then_full_exit_pnl() -> None:
    legacy_position = PositionState(
        quantity=Decimal("100"),
        average_cost=Decimal("1.05"),
        total_cost=Decimal("105"),
    )

    partial = apply_sell(Decimal("0"), legacy_position, Decimal("40"), Decimal("2"), Decimal("0"))
    complete = apply_sell(
        partial.cash_balance, partial.position, Decimal("60"), Decimal("2"), Decimal("0")
    )

    assert partial.realized_pnl == Decimal("38")
    assert partial.position.total_cost == Decimal("63")
    assert complete.realized_pnl == Decimal("57")
    assert complete.position.total_cost == Decimal("0")


def test_time_weighted_return_chain_links_consecutive_snapshot_periods() -> None:
    first_period = calculate_return_metrics(
        total_assets=Decimal("110"),
        prior_assets=Decimal("100"),
        external_cash_flow=Decimal("0"),
        contributed_capital=Decimal("100"),
        historical_assets=[Decimal("100")],
        prior_time_weighted_return=None,
    )
    second_period = calculate_return_metrics(
        total_assets=Decimal("121"),
        prior_assets=Decimal("110"),
        external_cash_flow=Decimal("0"),
        contributed_capital=Decimal("100"),
        historical_assets=[Decimal("100"), Decimal("110")],
        prior_time_weighted_return=first_period.time_weighted_return,
    )

    assert first_period.time_weighted_return == Decimal("0.1")
    assert second_period.time_weighted_return == Decimal("0.21")


def test_dated_cashflow_history_enables_mwr_and_annualized_return() -> None:
    metrics = calculate_return_metrics(
        total_assets=Decimal("121"),
        prior_assets=Decimal("110"),
        external_cash_flow=Decimal("0"),
        contributed_capital=Decimal("100"),
        historical_assets=[Decimal("100"), Decimal("110")],
        prior_time_weighted_return=Decimal("0.1"),
        cashflow_history=[(date(2026, 1, 1), Decimal("100"))],
        valuation_date=date(2027, 1, 1),
    )

    assert metrics.time_weighted_return == Decimal("0.21")
    assert metrics.money_weighted_return == Decimal("0.21")
    assert metrics.annualized_return == Decimal("0.21")
