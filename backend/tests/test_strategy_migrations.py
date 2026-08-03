from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database.migrations import SCHEMA_VERSION, run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import Base, Instrument, StrategyDefinition, StrategySignal


def test_true_v3_strategy_signal_migration_preserves_duplicate_legacy_snapshots(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    engine = create_database_engine(database)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            Instrument.__table__.insert().values(
                code="LEGACY-SIGNAL",
                name="Legacy signal ETF",
                exchange="SSE",
                category="etf",
                currency="CNY",
                is_active=True,
                extra_data={},
            )
        )
        connection.execute(
            StrategyDefinition.__table__.insert().values(
                name="legacy strategy",
                strategy_type="rule",
                parameters={},
                version="1.0",
            )
        )
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "LEGACY-SIGNAL")
        ).scalar_one()
        strategy_id = connection.execute(
            select(StrategyDefinition.id).where(StrategyDefinition.name == "legacy strategy")
        ).scalar_one()
        connection.exec_driver_sql("DROP TABLE strategy_signals")
        connection.exec_driver_sql(
            """
            CREATE TABLE strategy_signals (
                id INTEGER NOT NULL PRIMARY KEY,
                strategy_id INTEGER NOT NULL REFERENCES strategy_definitions(id),
                instrument_id INTEGER NOT NULL REFERENCES instruments(id),
                signal_at DATETIME NOT NULL,
                recommendation_type VARCHAR(32) NOT NULL,
                score INTEGER,
                target_allocation INTEGER,
                confidence INTEGER,
                rationale TEXT,
                signal_data JSON NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        legacy_rows = (
            (10, "2026-02-02 00:00:00", "BUY", '{"legacy":"first"}'),
            (11, "2026-02-02 12:00:00", "NONE", '{"legacy":"duplicate"}'),
            (12, "2026-02-03 00:00:00", "REDUCE", '{"legacy":"later"}'),
        )
        for signal_id, signal_at, recommendation, data in legacy_rows:
            connection.exec_driver_sql(
                """
                INSERT INTO strategy_signals
                    (id, strategy_id, instrument_id, signal_at, recommendation_type,
                     score, target_allocation, confidence, rationale, signal_data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?)
                """,
                (
                    signal_id,
                    strategy_id,
                    instrument_id,
                    signal_at,
                    recommendation,
                    data,
                    "2026-02-01 00:00:00",
                    "2026-02-01 00:00:00",
                ),
            )
        connection.exec_driver_sql("PRAGMA user_version = 3")

    assert run_migrations(engine) == SCHEMA_VERSION
    with Session(engine) as session:
        migrated = session.scalars(select(StrategySignal).order_by(StrategySignal.id)).all()

    assert [(row.id, row.as_of_date, row.recommendation_type, row.strategy_version) for row in migrated] == [
        (10, date(2026, 2, 2), "INCREASE", "1.0"),
        (11, date(2026, 2, 2), "HOLD", "1.0-legacy-11"),
        (12, date(2026, 2, 3), "REDUCE", "1.0"),
    ]
    assert all(len(row.strategy_config_hash) == 64 for row in migrated)
    with engine.begin() as connection, pytest.raises(IntegrityError):
        connection.execute(
            StrategySignal.__table__.insert().values(
                strategy_id=migrated[0].strategy_id,
                instrument_id=migrated[0].instrument_id,
                as_of_date=migrated[0].as_of_date,
                strategy_version=migrated[0].strategy_version,
                strategy_config_hash=migrated[0].strategy_config_hash,
                recommendation_type="HOLD",
                signal_data={},
            )
        )
