from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.config import ensure_default_configs
from backend.app.core.paths import database_path, project_root
from backend.app.database.initialize import DatabaseInitializationLock, initialize_database
from backend.app.database.migrations import SCHEMA_VERSION, run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import (
    AppSetting,
    Base,
    DataUpdateLog,
    DecisionLog,
    ForecastPoint,
    ForecastRun,
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    RealAccount,
    RealAccountSnapshot,
    RealTransaction,
    ResearchReport,
    SimulationAccount,
    SimulationDailySnapshot,
    SimulationPosition,
    SimulationTransaction,
    StrategyDefinition,
    StrategySignal,
    ValuationRecord,
)

def test_database_path_is_project_relative() -> None:
    expected_root = Path(__file__).resolve().parents[2]

    assert project_root() == expected_root
    assert database_path() == expected_root / "data" / "investment_lab.db"


def test_initialize_database_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    config_directory = tmp_path / "config"

    initialize_database(database, config_directory)
    initialize_database(database, config_directory)

    engine = create_database_engine(database)
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(Instrument)).scalar_one() == 9
        assert connection.execute(select(func.count()).select_from(InvestmentPlan)).scalar_one() == 3
        assert connection.execute(select(func.count()).select_from(SimulationAccount)).scalar_one() == 4
        assert connection.execute(select(func.count()).select_from(RealAccount)).scalar_one() == 1


def test_initialize_database_preserves_existing_data_and_config(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    config_directory = tmp_path / "config"
    initialize_database(database, config_directory)

    engine = create_database_engine(database)
    with engine.begin() as connection:
        connection.execute(
            Instrument.__table__.insert().values(
                code="510300",
                name="沪深300ETF",
                exchange="SSE",
                category="broad_market",
                is_active=True,
            )
        )
    custom_config = {"custom": "preserve-me"}
    fee_config = config_directory / "fee_config.json"
    fee_config.write_text(json.dumps(custom_config), encoding="utf-8")

    initialize_database(database, config_directory)

    with engine.connect() as connection:
        codes = connection.execute(select(Instrument.code)).scalars().all()
    assert "510300" in codes
    assert json.loads(fee_config.read_text(encoding="utf-8")) == custom_config


def test_market_price_is_unique_per_instrument_date_and_timeframe(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)

    with engine.begin() as connection:
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        connection.execute(
            MarketPrice.__table__.insert().values(
                instrument_id=instrument_id,
                trade_date=date(2026, 7, 1),
                timeframe="daily",
                open_price=Decimal("1.00"),
                high_price=Decimal("1.01"),
                low_price=Decimal("0.99"),
                close_price=Decimal("1.00"),
                volume=Decimal("1000"),
            )
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 7, 1),
                    timeframe="daily",
                    close_price=Decimal("1.01"),
                )
            )


def test_market_price_accepts_weekly_and_monthly_but_rejects_invalid_timeframe(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)

    with engine.begin() as connection:
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        for timeframe in ("weekly", "monthly"):
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 7, 1),
                    timeframe=timeframe,
                    close_price=Decimal("1.00"),
                )
            )
        with pytest.raises(IntegrityError):
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 7, 1),
                    timeframe="hourly",
                    close_price=Decimal("1.00"),
                )
            )


def test_sqlite_rejects_orphan_foreign_key_insertion(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)

    with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=999_999,
                    trade_date=date(2026, 7, 1),
                    timeframe="daily",
                    close_price=Decimal("1.00"),
                )
            )


def test_strategy_signal_rejects_sell_all_recommendation(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)

    with engine.begin() as connection:
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        strategy_id = connection.execute(
            select(StrategyDefinition.id).where(StrategyDefinition.name == "默认规则策略")
        ).scalar_one()
        with pytest.raises(IntegrityError):
            connection.execute(
                StrategySignal.__table__.insert().values(
                    instrument_id=instrument_id,
                    strategy_id=strategy_id,
                    recommendation_type="SELL_ALL",
                )
            )


def test_model_metadata_preserves_all_required_v2_v31_and_v32_tables() -> None:
    legacy_tables = {
        "instruments",
        "market_prices",
        "valuation_records",
        "indicator_records",
        "investment_plans",
        "strategy_definitions",
        "strategy_signals",
        "simulation_accounts",
        "simulation_transactions",
        "simulation_positions",
        "simulation_daily_snapshots",
        "real_accounts",
        "real_transactions",
        "real_positions",
        "real_account_snapshots",
        "research_reports",
        "decision_logs",
        "forecast_runs",
        "forecast_points",
        "app_settings",
        "data_update_logs",
        "investment_calendar_entries",
        "v2_market_data_cache",
        "v2_period_bars",
        "v2_feature_snapshots",
        "v2_week_samples",
        "v2_analysis_iterations",
        "v2_iteration_labels",
        "v2_model_versions",
        "v2_analysis_tasks",
        "v2_position_events",
        "v2_position_snapshots",
        "v2_advice_history",
        "v31_analysis_runs",
        "v31_model_versions",
        "v31_weekly_forecasts",
        "v31_daily_corrections",
        "v31_model_evaluations",
        "v31_instrument_market_mappings",
        "v32_training_runs",
        "v32_training_iterations",
        "v32_training_checkpoints",
        "v32_model_versions",
        "v32_optimizer_states",
        "v32_data_snapshots",
        "v32_weekly_forecasts",
        "v32_daily_corrections",
        "v32_model_evaluations",
        "v32_analysis_runs",
        "v32_app_state",
    }
    assert legacy_tables.issubset(set(Base.metadata.tables))
    assert {
        "v33_instrument_roles",
        "v33_raw_ingestions",
        "v33_source_releases",
        "v33_market_bars",
        "v33_point_in_time_observations",
        "v33_quality_assessments",
        "v33_feature_snapshots",
        "v33_training_runs",
        "v33_training_iterations",
        "v33_training_checkpoints",
        "v33_model_versions",
        "v33_optimizer_states",
        "v33_forecasts",
        "v33_model_evaluations",
        "v33_analysis_runs",
    }.issubset(set(Base.metadata.tables))


def test_money_and_quantity_columns_use_exact_fixed_point_storage() -> None:
    numeric_columns = {
        MarketPrice: (
            "open_price", "high_price", "low_price", "close_price", "adjusted_close_price", "volume", "turnover"
        ),
        ValuationRecord: ("pe_ratio", "pb_ratio", "dividend_yield", "valuation_percentile", "fair_value"),
        IndicatorRecord: ("indicator_value",),
        InvestmentPlan: ("amount",),
        StrategySignal: ("score", "target_allocation", "confidence"),
        SimulationAccount: ("initial_cash", "cash_balance", "portfolio_value"),
        SimulationTransaction: ("quantity", "price", "amount", "fee"),
        SimulationPosition: ("quantity", "average_cost", "market_value", "unrealized_pnl"),
        SimulationDailySnapshot: ("cash_balance", "market_value", "total_assets", "daily_return", "cumulative_return"),
        RealAccount: ("initial_cash", "cash_balance"),
        RealTransaction: ("quantity", "price", "amount", "fee"),
        RealAccountSnapshot: ("cash_balance", "market_value", "total_assets", "total_return"),
        DecisionLog: ("planned_amount", "confidence"),
        ForecastPoint: ("predicted_price", "lower_bound", "upper_bound", "confidence"),
    }

    for model, columns in numeric_columns.items():
        for column in columns:
            assert model.__table__.c[column].type.__class__.__name__ == "FixedPointDecimal", (
                f"{model.__name__}.{column}"
            )


def test_fixed_point_values_round_trip_as_decimal_and_aggregate_exactly(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)

    with engine.begin() as connection:
        account_id = connection.execute(
            select(SimulationAccount.id).where(SimulationAccount.name == "固定定投")
        ).scalar_one()
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        for amount in (Decimal("0.10"), Decimal("0.20")):
            connection.execute(
                SimulationTransaction.__table__.insert().values(
                    account_id=account_id,
                    instrument_id=instrument_id,
                    transaction_type="BUY",
                    quantity=Decimal("1"),
                    price=amount,
                    amount=amount,
                    fee=Decimal("0"),
                )
            )
        storage_type = connection.execute(
            text("SELECT typeof(amount) FROM simulation_transactions LIMIT 1")
        ).scalar_one()
        raw_sum = connection.execute(text("SELECT SUM(amount) FROM simulation_transactions")).scalar_one()

    with Session(engine) as session:
        amounts = session.execute(
            select(SimulationTransaction.amount).order_by(SimulationTransaction.id)
        ).scalars().all()

    assert storage_type == "integer"
    assert raw_sum == 30_000_000
    assert amounts == [Decimal("0.10"), Decimal("0.20")]


def test_fixed_point_query_helpers_return_decimal_correct_results(tmp_path: Path) -> None:
    from backend.app.database.fixed_point import (
        fixed_point_average,
        fixed_point_divide,
        fixed_point_multiply,
    )

    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)

    with engine.begin() as connection:
        account_id = connection.execute(
            select(SimulationAccount.id).where(SimulationAccount.name == "固定定投")
        ).scalar_one()
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        records = (
            ("MULTIPLY", Decimal("1.50"), Decimal("0.10")),
            ("DIVIDE", Decimal("1"), Decimal("3")),
            ("AVERAGE", Decimal("1"), Decimal("1")),
            ("AVERAGE", Decimal("2"), Decimal("2")),
        )
        for transaction_type, price, amount in records:
            connection.execute(
                SimulationTransaction.__table__.insert().values(
                    account_id=account_id,
                    instrument_id=instrument_id,
                    transaction_type=transaction_type,
                    quantity=Decimal("1"),
                    price=price,
                    amount=amount,
                    fee=Decimal("0"),
                )
            )

    with Session(engine) as session:
        product = session.execute(
            select(fixed_point_multiply(SimulationTransaction.price, SimulationTransaction.amount)).where(
                SimulationTransaction.transaction_type == "MULTIPLY"
            )
        ).scalar_one()
        quotient = session.execute(
            select(fixed_point_divide(SimulationTransaction.price, SimulationTransaction.amount)).where(
                SimulationTransaction.transaction_type == "DIVIDE"
            )
        ).scalar_one()
        average = session.execute(
            select(fixed_point_average(SimulationTransaction.price)).where(
                SimulationTransaction.transaction_type == "AVERAGE"
            )
        ).scalar_one()

    assert product == Decimal("0.15")
    assert quotient == Decimal("0.33333333")
    assert average == Decimal("1.5")


def test_fixed_point_query_helpers_coerce_decimal_scalar_operands(tmp_path: Path) -> None:
    from backend.app.database.fixed_point import fixed_point_divide, fixed_point_multiply

    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)

    with engine.begin() as connection:
        account_id = connection.execute(
            select(SimulationAccount.id).where(SimulationAccount.name == "固定定投")
        ).scalar_one()
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        connection.execute(
            SimulationTransaction.__table__.insert().values(
                account_id=account_id,
                instrument_id=instrument_id,
                transaction_type="SCALAR",
                quantity=Decimal("1"),
                price=Decimal("1.50"),
                amount=Decimal("1.50"),
                fee=Decimal("0"),
            )
        )

    with Session(engine) as session:
        product = session.execute(
            select(fixed_point_multiply(SimulationTransaction.price, Decimal("0.10"))).where(
                SimulationTransaction.transaction_type == "SCALAR"
            )
        ).scalar_one()
        quotient = session.execute(
            select(fixed_point_divide(SimulationTransaction.price, Decimal("3"))).where(
                SimulationTransaction.transaction_type == "SCALAR"
            )
        ).scalar_one()
        scalar_product = session.execute(
            select(fixed_point_multiply(Decimal("1.50"), Decimal("0.10"), scale=8))
        ).scalar_one()

    assert product == Decimal("0.15")
    assert quotient == Decimal("0.50")
    assert scalar_product == Decimal("0.15")
    with pytest.raises(ValueError, match="require FixedPointDecimal expressions"):
        fixed_point_multiply(Decimal("1.50"), Decimal("0.10"))


def test_version_one_migration_normalizes_legacy_decimal_artifacts_and_records_diagnostics(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    engine = create_database_engine(database)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE simulation_transactions (id INTEGER PRIMARY KEY, amount NUMERIC)"
        )
        connection.exec_driver_sql(
            "INSERT INTO simulation_transactions (id, amount) VALUES (1, 0.123456789), (2, 0.30000000000000004)"
        )

    run_migrations(engine)

    with engine.connect() as connection:
        stored_values = connection.execute(
            text("SELECT amount FROM simulation_transactions ORDER BY id")
        ).scalars().all()
        diagnostics = connection.execute(
            text("SELECT message FROM data_update_logs WHERE dataset = 'schema_migration' ORDER BY id")
        ).scalars().all()

    assert stored_values == [12_345_679, 30_000_000]
    assert any(
        "table=simulation_transactions column=amount row=1 value=0.123456789" in message
        for message in diagnostics
    )
    assert any(
        "table=simulation_transactions column=amount row=2 value=0.30000000000000004" in message
        for message in diagnostics
    )


def test_initialize_database_is_safe_when_called_concurrently_by_processes(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    configuration_directory = tmp_path / "config"
    start_marker = tmp_path / "start-workers"
    worker_code = (
        "import sys,time;"
        "from pathlib import Path;"
        "from backend.app.database.initialize import initialize_database;"
        "marker=Path(sys.argv[3]);"
        "\nwhile not marker.exists(): time.sleep(0.01);"
        "\ninitialize_database(Path(sys.argv[1]), Path(sys.argv[2]))"
    )
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                worker_code,
                str(database),
                str(configuration_directory),
                str(start_marker),
            ],
            cwd=project_root(),
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]

    try:
        start_marker.touch()
        results = [worker.communicate(timeout=90) for worker in workers]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)

    failures = [
        f"exit={worker.returncode} stdout={stdout!r} stderr={stderr!r}"
        for worker, (stdout, stderr) in zip(workers, results, strict=True)
        if worker.returncode != 0
    ]
    assert failures == []
    assert not (database.parent / ".investment_lab.initialize.lock").exists()

    engine = create_database_engine(database)
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(Instrument)).scalar_one() == 9
        assert connection.execute(select(func.count()).select_from(InvestmentPlan)).scalar_one() == 3
        assert connection.execute(select(func.count()).select_from(SimulationAccount)).scalar_one() == 4


def test_initialize_database_reclaims_a_lock_owned_by_a_nonrunning_process(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    lock_path = database.parent / ".investment_lab.initialize.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps({"pid": 999_999_999, "created_at": time.time()}),
        encoding="utf-8",
    )

    initialize_database(database, tmp_path / "config", lock_timeout_seconds=1)

    assert not lock_path.exists()


def test_initialize_database_does_not_reclaim_a_live_lock(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    lock_path = database.parent / ".investment_lab.initialize.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "created_at": time.time() - 86_400}),
        encoding="utf-8",
    )

    with pytest.raises(TimeoutError, match="Timed out waiting for database initialization lock"):
        initialize_database(database, tmp_path / "config", lock_timeout_seconds=0.1)

    assert lock_path.exists()


def test_recovery_guard_permission_error_propagates_when_guard_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    database.parent.mkdir(parents=True)
    lock = DatabaseInitializationLock(database, timeout_seconds=1)

    def deny_open(_path: object, _flags: int) -> int:
        raise PermissionError("directory is not writable")

    monkeypatch.setattr(os, "open", deny_open)

    with pytest.raises(PermissionError, match="not writable"):
        lock._acquire_recovery_guard()


def test_recovery_guard_permission_error_is_contention_when_guard_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    database.parent.mkdir(parents=True)
    lock = DatabaseInitializationLock(database, timeout_seconds=1)
    lock.recovery_path.write_text("owned", encoding="utf-8")

    def deny_open(_path: object, _flags: int) -> int:
        raise PermissionError("sharing violation")

    monkeypatch.setattr(os, "open", deny_open)

    assert lock._acquire_recovery_guard() is None
    assert lock.recovery_path.exists()


def test_lock_metadata_permission_error_does_not_reclaim_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    lock_path = database.parent / ".investment_lab.initialize.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps({"pid": 999_999_999, "created_at": time.time()}),
        encoding="utf-8",
    )
    old_timestamp = time.time() - 600
    os.utime(lock_path, (old_timestamp, old_timestamp))
    lock = DatabaseInitializationLock(database, timeout_seconds=1)
    original_read_text = Path.read_text

    def deny_lock_metadata(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path == lock_path:
            raise PermissionError("lock metadata is not readable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_lock_metadata)

    with pytest.raises(PermissionError, match="not readable"):
        lock._reclaim_stale_lock_if_safe()

    assert lock_path.exists()


def test_existing_malformed_default_config_raises_without_overwriting(tmp_path: Path) -> None:
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    app_config = config_directory / "app_config.json"
    invalid_contents = '{"locale": '
    app_config.write_text(invalid_contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"Invalid JSON configuration.*app_config\.json"):
        ensure_default_configs(config_directory)

    assert app_config.read_text(encoding="utf-8") == invalid_contents


def test_initialize_database_sets_and_accepts_current_schema_version(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    configuration_directory = tmp_path / "config"

    initialize_database(database, configuration_directory)
    engine = create_database_engine(database)
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA user_version")).scalar_one() == SCHEMA_VERSION

    initialize_database(database, configuration_directory)
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA user_version")).scalar_one() == SCHEMA_VERSION


def test_initialize_database_rejects_unknown_future_schema_version(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    configuration_directory = tmp_path / "config"
    initialize_database(database, configuration_directory)
    engine = create_database_engine(database)
    with engine.begin() as connection:
        connection.exec_driver_sql(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(RuntimeError, match="newer than supported"):
        initialize_database(database, configuration_directory)


def test_default_instruments_and_configuration_are_created(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    config_directory = tmp_path / "config"

    initialize_database(database, config_directory)

    engine = create_database_engine(database)
    with Session(engine) as session:
        instruments = session.execute(
            select(Instrument.code, Instrument.name).order_by(Instrument.code)
        ).all()
        weekly_plans = session.execute(
            select(InvestmentPlan).where(InvestmentPlan.enabled.is_(True))
        ).scalars().all()
        strategy = session.execute(select(StrategyDefinition)).scalar_one()
        settings = session.execute(select(AppSetting)).scalars().all()

    assert instruments == [
        ("000688", "科创50指数"),
        ("159205", "创业板ETF东财"),
        ("159941", "纳指ETF广发"),
        ("399006", "创业板指数"),
        ("512690", "鹏华酒ETF"),
        ("512800", "华宝银行ETF"),
        ("518600", "广发黄金ETF"),
        ("589850", "科创50ETF东财"),
        ("NDX", "纳斯达克100指数"),
    ]
    assert len(weekly_plans) == 3
    assert all(plan.amount == Decimal("150.00") and plan.frequency == "weekly" for plan in weekly_plans)
    assert strategy.enabled is True
    assert settings

    configs = {
        filename: json.loads((config_directory / filename).read_text(encoding="utf-8"))
        for filename in ("app_config.json", "fee_config.json", "strategy_config.json", "watchlist.json")
    }
    assert configs["fee_config.json"] == {
        "buy_commission_rate": 0.00025,
        "sell_commission_rate": 0.00025,
        "minimum_commission": 5,
        "minimum_commission_enabled": True,
        "etf_stamp_duty_rate": 0,
        "transfer_fee_rate": 0,
        "other_fee_rate": 0,
    }
    assert configs["strategy_config.json"]["maximum_sell_ratio"] == 0.3
    assert configs["strategy_config.json"]["minimum_holding_ratio"] == 0.2
    assert [item["code"] for item in configs["watchlist.json"]["instruments"]] == [
        "589850",
        "159205",
        "159941",
    ]


def test_default_config_creation_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    app_config = config_directory / "app_config.json"
    app_config.write_text('{"language": "custom"}', encoding="utf-8")

    ensure_default_configs(config_directory)

    assert app_config.read_text(encoding="utf-8") == '{"language": "custom"}'
