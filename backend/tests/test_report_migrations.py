from __future__ import annotations

from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.migrations import SCHEMA_VERSION, run_migrations
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import Base, Instrument, ResearchReport, StrategyDefinition, StrategySignal
from backend.app.services.strategy_service import StrategyService


def test_v4_to_v5_report_migration_links_legacy_report_to_signal_identity_and_exports(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    engine = create_database_engine(database)
    Base.metadata.create_all(engine)
    report_date = date(2026, 2, 2)
    with engine.begin() as connection:
        connection.execute(
            Instrument.__table__.insert().values(
                code="LEGACY-REPORT",
                name="Legacy report ETF",
                exchange="SSE",
                category="etf",
                currency="CNY",
                is_active=True,
                extra_data={},
            )
        )
        connection.execute(
            StrategyDefinition.__table__.insert().values(
                name="legacy report strategy",
                strategy_type="rule",
                parameters={},
                version="1.0",
            )
        )
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "LEGACY-REPORT")
        ).scalar_one()
        strategy_id = connection.execute(
            select(StrategyDefinition.id).where(StrategyDefinition.name == "legacy report strategy")
        ).scalar_one()
        connection.exec_driver_sql("DROP TABLE strategy_signals")
        connection.exec_driver_sql("DROP TABLE research_reports")
        connection.exec_driver_sql(
            """
            CREATE TABLE strategy_signals (
                id INTEGER PRIMARY KEY, strategy_id INTEGER NOT NULL, instrument_id INTEGER NOT NULL,
                signal_at DATETIME NOT NULL, as_of_date DATE NOT NULL, strategy_version VARCHAR(32) NOT NULL,
                recommendation_type VARCHAR(32) NOT NULL, score INTEGER, target_allocation INTEGER,
                confidence INTEGER, rationale TEXT, signal_data JSON NOT NULL,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
                CONSTRAINT uq_strategy_signal_snapshot UNIQUE
                    (strategy_id, instrument_id, as_of_date, strategy_version)
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE research_reports (
                id INTEGER PRIMARY KEY, instrument_id INTEGER, title VARCHAR(256) NOT NULL,
                report_date DATE NOT NULL, report_type VARCHAR(64) NOT NULL, content TEXT,
                file_path VARCHAR(512), status VARCHAR(32) NOT NULL,
                created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO strategy_signals
                VALUES (10, ?, ?, '2026-02-02 00:00:00', '2026-02-02', '1.0', 'HOLD',
                        NULL, NULL, NULL, 'legacy', '{}', '2026-02-01', '2026-02-01')
            """,
            (strategy_id, instrument_id),
        )
        connection.exec_driver_sql(
            """
            INSERT INTO research_reports
                VALUES (20, ?, 'legacy report', '2026-02-02', 'strategy_research',
                        'fixed content', NULL, 'generated', '2026-02-01', '2026-02-01')
            """,
            (instrument_id,),
        )
        connection.exec_driver_sql("PRAGMA user_version = 4")

    assert run_migrations(engine) == SCHEMA_VERSION
    with Session(engine) as session:
        signal = session.scalar(select(StrategySignal))
        report = session.scalar(select(ResearchReport))
    assert signal is not None and report is not None
    assert report.strategy_id == signal.strategy_id
    assert report.strategy_version == signal.strategy_version
    assert report.strategy_config_hash == signal.strategy_config_hash
    rows = StrategyService(create_session_factory(database)).export_report_csv_rows(
        etf_code="LEGACY-REPORT", report_date=report_date
    )
    assert rows[0]["strategy_version"] == "1.0"
    assert rows[0]["strategy_config_hash"] == signal.strategy_config_hash
