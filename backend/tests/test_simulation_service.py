from datetime import date
from decimal import Decimal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    Instrument,
    InvestmentPlan,
    MarketPrice,
    SimulationDailySnapshot,
    SimulationAccount,
    SimulationPosition,
    SimulationTransaction,
    StrategyDefinition,
    StrategySignal,
)
from backend.app.schemas.simulation import InvestmentPlanCreate, SimulationAccountCreate
from backend.app.services.plan_service import InvestmentPlanService
from backend.app.services.simulation_service import SimulationService


WEEK = date(2026, 7, 6)


def _service_and_plan(
    tmp_path: Path, *, mode: str = "REAL_LOT", amount: str = "150", initial_cash: str = "0"
) -> tuple[SimulationService, int, int, Path]:
    database = tmp_path / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    factory = create_session_factory(database)
    account = SimulationService(factory).create_account(
        SimulationAccountCreate(name="simulation", initial_cash=Decimal(initial_cash))
    )
    plan = InvestmentPlanService(factory).create(
        InvestmentPlanCreate(
            instrument_code="589850",
            name=f"{mode} weekly",
            weekly_amount=Decimal(amount),
            execution_weekday=0,
            execution_price_rule="CLOSE",
            mode=mode,
            lot_size=100,
        )
    )
    with Session(create_database_engine(database)) as session, session.begin():
        instrument_id = session.scalar(select(Instrument.id).where(Instrument.code == "589850"))
        assert instrument_id is not None
        session.add_all(
            [
                MarketPrice(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 7, 3),
                    timeframe="daily",
                    close_price=Decimal("1"),
                ),
                MarketPrice(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 7, 7),
                    timeframe="daily",
                    close_price=Decimal("100"),
                ),
                MarketPrice(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 7, 10),
                    timeframe="daily",
                    close_price=Decimal("1"),
                ),
            ]
        )
    return SimulationService(factory), account.id, plan.id, database


def test_weekly_real_lot_uses_only_stored_eligible_price_and_carries_cash(tmp_path: Path) -> None:
    service, account_id, plan_id, database = _service_and_plan(tmp_path)

    first = service.run_weekly(account_id, plan_id, WEEK, source="test")
    second = service.run_weekly(account_id, plan_id, date(2026, 7, 13), source="test")
    repeated = service.run_weekly(account_id, plan_id, WEEK, source="test")

    assert first.idempotent is False
    assert first.price_date == date(2026, 7, 3)
    assert first.transaction.requested_amount == Decimal("150")
    assert first.transaction.executed_amount == Decimal("100")
    assert first.transaction.quantity == Decimal("100")
    assert first.transaction.commission == Decimal("5")
    assert first.transaction.cash_after == Decimal("45")
    assert second.transaction.cash_after == Decimal("90")
    assert repeated.idempotent is True
    assert repeated.transaction.id == first.transaction.id
    with Session(create_database_engine(database)) as session:
        assert session.scalar(select(SimulationTransaction).where(SimulationTransaction.account_id == account_id)).quantity == Decimal("100")
        position = session.scalar(select(SimulationPosition).where(SimulationPosition.account_id == account_id))
        assert position is not None and position.quantity == Decimal("200")


def test_fractional_execution_is_explicitly_marked_theoretical(tmp_path: Path) -> None:
    service, account_id, plan_id, _database = _service_and_plan(tmp_path, mode="FRACTIONAL", amount="100")
    service.set_fee_configuration({"minimum_commission_enabled": False})

    result = service.run_weekly(account_id, plan_id, WEEK, source="fractional")

    assert result.theoretical_mode is True
    assert result.transaction.quantity == Decimal("99.97500624")
    assert "THEORETICAL_FRACTIONAL" in (result.transaction.notes or "")


def test_trade_accounting_freezes_fees_and_prevents_oversell(tmp_path: Path) -> None:
    service, account_id, _plan_id, database = _service_and_plan(tmp_path, initial_cash="1000")
    service.set_fee_configuration({"buy_commission_rate": "0", "sell_commission_rate": "0", "minimum_commission_enabled": False})
    with Session(create_database_engine(database)) as session, session.begin():
        instrument_id = session.scalar(select(Instrument.id).where(Instrument.code == "589850"))
        assert instrument_id is not None
        session.add(
            MarketPrice(
                instrument_id=instrument_id,
                trade_date=date(2026, 7, 6),
                timeframe="daily",
                close_price=Decimal("15"),
            )
        )

    bought = service.execute_trade(
        account_id, "589850", "BUY", Decimal("10"), Decimal("10"), date(2026, 7, 3), source="manual-buy"
    )
    service.set_fee_configuration({"buy_commission_rate": "0.2", "minimum_commission_enabled": False})
    sold = service.execute_trade(
        account_id, "589850", "SELL", Decimal("4"), Decimal("15"), date(2026, 7, 6), source="manual-sell"
    )

    assert bought.commission == Decimal("0")
    assert sold.realized_pnl == Decimal("20")
    with pytest.raises(ValueError, match="cannot exceed holding"):
        service.execute_trade(
            account_id, "589850", "SELL", Decimal("7"), Decimal("15"), date(2026, 7, 6), source="oversell"
        )
    with Session(create_database_engine(database)) as session:
        position = session.scalar(select(SimulationPosition).where(SimulationPosition.account_id == account_id))
        snapshot = session.scalar(
            select(SimulationDailySnapshot).where(
                SimulationDailySnapshot.account_id == account_id,
                SimulationDailySnapshot.snapshot_date == date(2026, 7, 6),
            )
        )
        assert position is not None
        assert position.quantity == Decimal("6")
        assert position.average_cost == Decimal("10")
        assert position.total_cost == Decimal("60")
        assert position.realized_pnl == Decimal("20")
        assert snapshot is not None
        assert snapshot.market_value == Decimal("90")
        assert snapshot.total_assets == Decimal("1050")
        frozen_buy = session.scalar(
            select(SimulationTransaction).where(SimulationTransaction.id == bought.id)
        )
        assert frozen_buy is not None
        assert frozen_buy.commission == Decimal("0")
        assert frozen_buy.commission_rate == Decimal("0")


def test_recalculation_rebuilds_account_history_and_chart_series(tmp_path: Path) -> None:
    service, account_id, _plan_id, _database = _service_and_plan(tmp_path, initial_cash="100")
    service.set_fee_configuration({"buy_commission_rate": "0", "sell_commission_rate": "0", "minimum_commission_enabled": False})
    service.execute_trade(
        account_id, "589850", "BUY", Decimal("10"), Decimal("10"), date(2026, 7, 3), source="buy"
    )

    rebuilt = service.recalculate_account(account_id)
    chart = service.chart_series(account_id)

    assert rebuilt.cash_balance == Decimal("0")
    assert chart == [
        {
            "date": date(2026, 7, 3),
            "cash_balance": Decimal("0"),
            "market_value": Decimal("10"),
            "total_assets": Decimal("10"),
            "cumulative_return": Decimal("-0.9"),
            "time_weighted_return": None,
            "money_weighted_return": None,
        }
    ]


def test_strategy_sell_is_lot_sized_and_respects_max_sell_and_min_holding(tmp_path: Path) -> None:
    service, account_id, plan_id, database = _service_and_plan(tmp_path, amount="500")
    service.run_weekly(account_id, plan_id, WEEK, source="initial-buy")
    with Session(create_database_engine(database)) as session, session.begin():
        strategy = StrategyDefinition(
            name="sell-constraint strategy",
            parameters={"maximum_sell_ratio": "0.30", "minimum_holding_ratio": "0.20"},
        )
        session.add(strategy)
        session.flush()
        account = session.get(SimulationAccount, account_id)
        assert account is not None
        account.strategy_id = strategy.id
        instrument_id = session.scalar(select(Instrument.id).where(Instrument.code == "589850"))
        assert instrument_id is not None
        signal = StrategySignal(
            strategy_id=strategy.id,
            instrument_id=instrument_id,
            as_of_date=date(2026, 7, 13),
            strategy_version="1",
            strategy_config_hash="a" * 64,
            source_data_hash="b" * 64,
            recommendation_type="SELL_PARTIAL",
            signal_data={"suggested_sell_ratio": "0.90"},
        )
        session.add(signal)
        session.flush()
        signal_id = signal.id

    result = service.run_weekly(account_id, plan_id, date(2026, 7, 13), source="constrained-sell")

    assert result.transaction.transaction_type == "SELL"
    assert result.transaction.signal_id == signal_id
    assert result.transaction.quantity == Decimal("100")
    with Session(create_database_engine(database)) as session:
        position = session.scalar(select(SimulationPosition).where(SimulationPosition.account_id == account_id))
        assert position is not None
        assert position.quantity == Decimal("300")


def test_snapshot_time_weighted_return_chain_links_100_to_110_to_121(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    factory = create_session_factory(database)
    service = SimulationService(factory)
    account_id = service.create_account(
        SimulationAccountCreate(name="twr account", initial_cash=Decimal("100"))
    ).id

    with factory() as session, session.begin():
        account = session.get(SimulationAccount, account_id)
        assert account is not None
        service._snapshot(session, account, date(2026, 7, 1), external_cash_flow=Decimal("0"))
        account.cash_balance = Decimal("110")
        service._snapshot(session, account, date(2026, 7, 2), external_cash_flow=Decimal("0"))
        account.cash_balance = Decimal("121")
        service._snapshot(session, account, date(2026, 7, 3), external_cash_flow=Decimal("0"))

    with factory() as session:
        snapshots = session.scalars(
            select(SimulationDailySnapshot)
            .where(SimulationDailySnapshot.account_id == account_id)
            .order_by(SimulationDailySnapshot.snapshot_date)
        ).all()
    assert [snapshot.total_assets for snapshot in snapshots] == [
        Decimal("100"), Decimal("110"), Decimal("121")
    ]
    assert snapshots[-1].time_weighted_return == Decimal("0.21")


def test_same_day_snapshots_aggregate_external_cash_flow_before_returns(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    factory = create_session_factory(database)
    service = SimulationService(factory)
    account_id = service.create_account(
        SimulationAccountCreate(name="same-day cashflow", initial_cash=Decimal("0"))
    ).id

    with factory() as session, session.begin():
        account = session.get(SimulationAccount, account_id)
        assert account is not None
        account.cash_balance = Decimal("100")
        service._snapshot(session, account, date(2026, 7, 6), external_cash_flow=Decimal("100"))
        account.cash_balance = Decimal("150")
        service._snapshot(session, account, date(2026, 7, 6), external_cash_flow=Decimal("50"))

    with factory() as session:
        snapshot = session.scalar(
            select(SimulationDailySnapshot).where(SimulationDailySnapshot.account_id == account_id)
        )
    assert snapshot is not None
    assert snapshot.external_cash_flow == Decimal("150")
    assert snapshot.total_assets == Decimal("150")
    assert snapshot.cumulative_return == Decimal("0")


def test_recalculate_with_cutoff_rejects_without_mutating_later_account_state(tmp_path: Path) -> None:
    service, account_id, _plan_id, database = _service_and_plan(tmp_path, initial_cash="100")
    service.set_fee_configuration({"buy_commission_rate": "0", "minimum_commission_enabled": False})
    service.execute_trade(
        account_id, "589850", "BUY", Decimal("10"), Decimal("10"), date(2026, 7, 3), source="first"
    )
    service.execute_trade(
        account_id, "589850", "SELL", Decimal("5"), Decimal("15"), date(2026, 7, 6), source="later"
    )
    with Session(create_database_engine(database)) as session:
        before = session.get(SimulationAccount, account_id)
        assert before is not None
        cash_before = before.cash_balance
        position_before = session.scalar(
            select(SimulationPosition).where(SimulationPosition.account_id == account_id)
        )
        assert position_before is not None
        quantity_before = position_before.quantity

    with pytest.raises(ValueError, match="cutoff"):
        service.recalculate_account(account_id, through_date=date(2026, 7, 3))

    with Session(create_database_engine(database)) as session:
        after = session.get(SimulationAccount, account_id)
        position_after = session.scalar(
            select(SimulationPosition).where(SimulationPosition.account_id == account_id)
        )
    assert after is not None and position_after is not None
    assert after.cash_balance == cash_before
    assert position_after.quantity == quantity_before


def test_invalid_persisted_execution_mode_or_lot_size_is_rejected_before_trade(tmp_path: Path) -> None:
    service, account_id, plan_id, database = _service_and_plan(tmp_path)
    with Session(create_database_engine(database)) as session, session.begin():
        plan = session.get(InvestmentPlan, plan_id)
        assert plan is not None
        plan.rule_parameters = {"mode": "UNLABELLED", "lot_size": 0, "execution_price_rule": "CLOSE"}

    with pytest.raises(ValidationError):
        service.run_weekly(account_id, plan_id, WEEK, source="invalid-persisted-settings")

    with Session(create_database_engine(database)) as session:
        assert session.scalar(
            select(SimulationTransaction).where(SimulationTransaction.account_id == account_id)
        ) is None


def test_invalid_persisted_execution_price_rule_is_rejected_before_trade(tmp_path: Path) -> None:
    service, account_id, plan_id, database = _service_and_plan(tmp_path)
    with Session(create_database_engine(database)) as session, session.begin():
        plan = session.get(InvestmentPlan, plan_id)
        assert plan is not None
        plan.rule_parameters = {"mode": "REAL_LOT", "lot_size": 100, "execution_price_rule": "FUTURE_CLOSE"}

    with pytest.raises(ValidationError):
        service.run_weekly(account_id, plan_id, WEEK, source="invalid-persisted-price-rule")

    with Session(create_database_engine(database)) as session:
        assert session.scalar(
            select(SimulationTransaction).where(SimulationTransaction.account_id == account_id)
        ) is None


def test_concurrent_account_trades_serialize_before_cash_is_read(tmp_path: Path) -> None:
    service, account_id, _plan_id, database = _service_and_plan(tmp_path, initial_cash="100")
    service.set_fee_configuration({"buy_commission_rate": "0", "minimum_commission_enabled": False})

    def buy(source: str) -> object:
        try:
            return service.execute_trade(
                account_id,
                "589850",
                "BUY",
                Decimal("6"),
                Decimal("10"),
                date(2026, 7, 3),
                source=source,
            )
        except Exception as error:  # asserted below; concurrent callers must not hit SQLite lock errors
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(buy, ("parallel-a", "parallel-b")))

    successful = [outcome for outcome in outcomes if isinstance(outcome, SimulationTransaction)]
    rejected = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(successful) == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], ValueError)
    assert "insufficient cash" in str(rejected[0])
    with Session(create_database_engine(database)) as session:
        account = session.get(SimulationAccount, account_id)
        transactions = session.scalars(
            select(SimulationTransaction).where(SimulationTransaction.account_id == account_id)
        ).all()
    assert account is not None
    assert account.cash_balance == Decimal("40")
    assert len(transactions) == 1
