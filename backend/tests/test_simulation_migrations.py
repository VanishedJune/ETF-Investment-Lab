from decimal import Decimal
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.database.migrations import SCHEMA_VERSION, run_migrations
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    SimulationDailySnapshot,
    SimulationPosition,
    SimulationTransaction,
)
from backend.app.services.simulation_service import SimulationService


SCALE = 100_000_000


def _create_v6_simulation_fixture(database: Path) -> None:
    engine = create_database_engine(database)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE simulation_accounts (
                id INTEGER PRIMARY KEY,
                strategy_id INTEGER,
                name VARCHAR(128) NOT NULL,
                initial_cash INTEGER NOT NULL,
                cash_balance INTEGER NOT NULL,
                portfolio_value INTEGER NOT NULL,
                enabled BOOLEAN NOT NULL,
                notes TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE market_prices (
                id INTEGER PRIMARY KEY,
                instrument_id INTEGER NOT NULL,
                trade_date DATE NOT NULL,
                timeframe VARCHAR(16) NOT NULL,
                open_price INTEGER,
                high_price INTEGER,
                low_price INTEGER,
                close_price INTEGER NOT NULL,
                adjusted_close_price INTEGER,
                volume INTEGER,
                turnover INTEGER,
                source VARCHAR(64),
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE simulation_transactions (
                id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                instrument_id INTEGER NOT NULL,
                signal_id INTEGER,
                transaction_at DATETIME NOT NULL,
                transaction_type VARCHAR(32) NOT NULL,
                quantity INTEGER NOT NULL,
                price INTEGER NOT NULL,
                amount INTEGER NOT NULL,
                fee INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE simulation_positions (
                id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                instrument_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                average_cost INTEGER NOT NULL,
                market_value INTEGER NOT NULL,
                unrealized_pnl INTEGER NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE simulation_daily_snapshots (
                id INTEGER PRIMARY KEY,
                account_id INTEGER NOT NULL,
                snapshot_date DATE NOT NULL,
                cash_balance INTEGER NOT NULL,
                market_value INTEGER NOT NULL,
                total_assets INTEGER NOT NULL,
                daily_return INTEGER,
                cumulative_return INTEGER,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.execute(
            text(
                """
                INSERT INTO simulation_transactions
                    (id, account_id, instrument_id, transaction_at, transaction_type,
                     quantity, price, amount, fee, notes, created_at, updated_at)
                VALUES (1, 11, 22, '2026-07-01 00:00:00', 'BUY', :quantity, :price,
                        :amount, :fee, 'v6 history', '2026-07-01 00:00:00', '2026-07-01 00:00:00')
                """
            ),
            {"quantity": 100 * SCALE, "price": SCALE, "amount": 100 * SCALE, "fee": 5 * SCALE},
        )
        connection.execute(
            text(
                """
                INSERT INTO simulation_accounts
                    (id, name, initial_cash, cash_balance, portfolio_value, enabled, created_at, updated_at)
                VALUES (11, 'legacy account', :cash, 0, 0, 1,
                        '2026-07-01 00:00:00', '2026-07-01 00:00:00')
                """
            ),
            {"cash": 105 * SCALE},
        )
        connection.execute(
            text(
                """
                INSERT INTO market_prices
                    (id, instrument_id, trade_date, timeframe, close_price, created_at, updated_at)
                VALUES (1, 22, '2026-07-01', 'daily', :price,
                        '2026-07-01 00:00:00', '2026-07-01 00:00:00')
                """
            ),
            {"price": SCALE},
        )
        connection.execute(
            text(
                """
                INSERT INTO simulation_positions
                    (id, account_id, instrument_id, quantity, average_cost, market_value,
                     unrealized_pnl, created_at, updated_at)
                VALUES (1, 11, 22, :quantity, :average_cost, :market_value, :unrealized,
                        '2026-07-01 00:00:00', '2026-07-01 00:00:00')
                """
            ),
            {
                "quantity": 100 * SCALE,
                "average_cost": 105_000_000,
                "market_value": 100 * SCALE,
                "unrealized": -5 * SCALE,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO simulation_daily_snapshots
                    (id, account_id, snapshot_date, cash_balance, market_value, total_assets,
                     daily_return, cumulative_return, created_at, updated_at)
                VALUES (1, 11, '2026-07-01', :cash, :market, :assets, :daily, :cumulative,
                        '2026-07-01 00:00:00', '2026-07-01 00:00:00')
                """
            ),
            {
                "cash": 50 * SCALE,
                "market": 100 * SCALE,
                "assets": 150 * SCALE,
                "daily": 1_000_000,
                "cumulative": 50_000_000,
            },
        )
        connection.exec_driver_sql("PRAGMA user_version = 6")


def test_v6_simulation_records_upgrade_to_v7_without_losing_history(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    _create_v6_simulation_fixture(database)
    engine = create_database_engine(database)

    assert run_migrations(engine) == SCHEMA_VERSION

    with Session(engine) as session:
        transaction = session.get(SimulationTransaction, 1)
        position = session.get(SimulationPosition, 1)
        snapshot = session.get(SimulationDailySnapshot, 1)
        assert transaction is not None and position is not None and snapshot is not None
        assert transaction.amount == Decimal("100")
        assert transaction.fee == Decimal("5")
        assert transaction.notes == "v6 history"
        assert transaction.source_identity == "legacy-1"
        assert transaction.commission == Decimal("0")
        assert transaction.cash_after == Decimal("0")
        assert transaction.minimum_commission_enabled is True
        assert position.quantity == Decimal("100")
        assert position.average_cost == Decimal("1.05")
        assert position.market_value == Decimal("100")
        assert position.unrealized_pnl == Decimal("-5")
        assert position.total_cost == Decimal("105")
        assert snapshot.cash_balance == Decimal("50")
        assert snapshot.total_assets == Decimal("150")
        assert snapshot.cumulative_return == Decimal("0.5")
        assert snapshot.external_cash_flow == Decimal("0")
        assert snapshot.time_weighted_return is None

    with engine.connect() as connection:
        index_columns = {
            tuple(row[2] for row in connection.exec_driver_sql(f"PRAGMA index_info({index[1]})"))
            for index in connection.exec_driver_sql("PRAGMA index_list(simulation_transactions)")
        }
        assert ("account_id", "plan_id", "execution_date", "source_identity") in index_columns


def test_recalculate_replays_v6_transaction_using_backfilled_cost_basis(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    _create_v6_simulation_fixture(database)
    engine = create_database_engine(database)
    run_migrations(engine)

    rebuilt = SimulationService(create_session_factory(database)).recalculate_account(11)

    assert rebuilt.cash_balance == Decimal("0")
    with Session(engine) as session:
        position = session.get(SimulationPosition, 1)
        transaction = session.get(SimulationTransaction, 1)
    assert position is not None and transaction is not None
    assert position.total_cost == Decimal("105")
    assert position.quantity == Decimal("100")
    assert transaction.cash_after == Decimal("0")


def test_v8_repairs_only_open_v7_positions_with_missing_total_cost(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    _create_v6_simulation_fixture(database)
    engine = create_database_engine(database)
    run_migrations(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("UPDATE simulation_positions SET total_cost = 0 WHERE id = 1")
        connection.exec_driver_sql(
            """
            INSERT INTO simulation_positions
                (id, account_id, instrument_id, quantity, average_cost, total_cost,
                 market_value, unrealized_pnl, realized_pnl, created_at, updated_at)
            VALUES (2, 11, 23, 200000000, 200000000, 400000000,
                    400000000, 0, 0, '2026-07-01 00:00:00', '2026-07-01 00:00:00')
            """
        )
        connection.exec_driver_sql("PRAGMA user_version = 7")

    assert run_migrations(engine) == SCHEMA_VERSION
    with Session(engine) as session:
        repaired = session.get(SimulationPosition, 1)
        preserved = session.get(SimulationPosition, 2)
        assert repaired is not None and preserved is not None
        assert repaired.quantity == Decimal("100")
        assert repaired.total_cost == Decimal("105")
        assert preserved.total_cost == Decimal("4")
    assert run_migrations(engine) == SCHEMA_VERSION
