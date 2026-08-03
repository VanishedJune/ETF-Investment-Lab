from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.database.initialize import initialize_database
from backend.app.database.migrations import SCHEMA_VERSION, V341_TABLES, run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import Base


def _version20_database(tmp_path: Path):
    path = tmp_path / "data" / "investment_lab.db"
    initialize_database(path, tmp_path / "config")
    engine = create_database_engine(path)
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in V341_TABLES:
                table.drop(connection, checkfirst=True)
        connection.exec_driver_sql("PRAGMA user_version = 20")
    return engine


def test_v341_schema_is_append_only_idempotent_and_versioned(tmp_path: Path) -> None:
    assert len(V341_TABLES) == 17
    engine = _version20_database(tmp_path)
    with engine.connect() as connection:
        old_tables = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'v341_%'"
            )
        }
    assert run_migrations(engine) == SCHEMA_VERSION == 22
    assert run_migrations(engine) == 22
    with engine.connect() as connection:
        tables = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert set(V341_TABLES) <= tables
        assert old_tables <= tables
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 22
        forecast_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE name='v341_forecasts'"
        ).scalar_one()
        assert "protocol_version" in forecast_sql
        assert "horizon_weeks = 13" in forecast_sql


def test_v341_migration_failure_rolls_back_without_residue(tmp_path: Path) -> None:
    engine = _version20_database(tmp_path)

    def fail_after_first_table(connection):  # type: ignore[no-untyped-def]
        Base.metadata.tables[V341_TABLES[0]].create(connection, checkfirst=True)
        raise RuntimeError("injected V3.4.1 migration failure")

    with patch(
        "backend.app.database.migrations._upgrade_to_version_twenty_one",
        side_effect=fail_after_first_table,
    ):
        with pytest.raises(RuntimeError, match="injected V3.4.1 migration failure"):
            run_migrations(engine)

    with engine.connect() as connection:
        tables = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 20
        assert not (set(V341_TABLES) & tables)
