from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import text

from backend.app.database.initialize import initialize_database
from backend.app.database.migrations import SCHEMA_VERSION, V34_TABLES, run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import Base


def _database(tmp_path: Path):
    path = tmp_path / "data" / "investment_lab.db"
    initialize_database(path, tmp_path / "config")
    return path, create_database_engine(path)


def test_v34_schema_is_isolated_idempotent_and_exactly_13_weeks(tmp_path: Path) -> None:
    _path, engine = _database(tmp_path)
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in V34_TABLES:
                table.drop(connection, checkfirst=True)
        connection.exec_driver_sql("PRAGMA user_version = 19")
        before_v33 = connection.exec_driver_sql(
            "SELECT COUNT(*) FROM v33_model_versions"
        ).scalar_one()

    assert run_migrations(engine) == SCHEMA_VERSION == 22
    assert run_migrations(engine) == 22

    with engine.connect() as connection:
        tables = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert set(V34_TABLES) <= tables
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 22
        assert connection.exec_driver_sql(
            "SELECT COUNT(*) FROM v33_model_versions"
        ).scalar_one() == before_v33
        forecast_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='v34_forecasts'"
        ).scalar_one()
        assert "horizon_weeks = 13" in forecast_sql
        assert "scenario_count >= 1000" in forecast_sql


def test_v34_migration_failure_rolls_back_tables_and_user_version(tmp_path: Path) -> None:
    _path, engine = _database(tmp_path)
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in V34_TABLES:
                table.drop(connection, checkfirst=True)
        connection.exec_driver_sql("PRAGMA user_version = 19")

    def fail_after_first_table(connection):  # type: ignore[no-untyped-def]
        Base.metadata.tables[V34_TABLES[0]].create(connection, checkfirst=True)
        raise RuntimeError("injected V3.4 migration failure")

    with patch(
        "backend.app.database.migrations._upgrade_to_version_twenty",
        side_effect=fail_after_first_table,
    ):
        with pytest.raises(RuntimeError, match="injected V3.4 migration failure"):
            run_migrations(engine)

    with engine.connect() as connection:
        tables = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert connection.execute(text("PRAGMA user_version")).scalar_one() == 19
        assert not (set(V34_TABLES) & tables)
