from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import hashlib
import json

import pytest
from sqlalchemy.exc import IntegrityError

from backend.app.database.initialize import initialize_database
from backend.app.database.migrations import SCHEMA_VERSION, V342_TABLES, run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import Base


def _version21_database(tmp_path: Path):
    path = tmp_path / "data" / "investment_lab.db"
    initialize_database(path, tmp_path / "config")
    engine = create_database_engine(path)
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in V342_TABLES:
                table.drop(connection, checkfirst=True)
        connection.exec_driver_sql("PRAGMA user_version = 21")
    return engine


def _old_table_fingerprint(connection):  # type: ignore[no-untyped-def]
    output = {}
    for name, sql in connection.exec_driver_sql(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'v342_%' ORDER BY name"
    ):
        columns = list(connection.exec_driver_sql(f'PRAGMA table_info("{name}")'))
        indexes = list(connection.exec_driver_sql(f'PRAGMA index_list("{name}")'))
        foreign_keys = list(connection.exec_driver_sql(f'PRAGMA foreign_key_list("{name}")'))
        primary = [str(row[1]) for row in columns if int(row[5]) > 0]
        order = ", ".join(f'"{item}"' for item in primary) if primary else "rowid"
        digest = hashlib.sha256()
        primary_values = []
        count = 0
        for row in connection.exec_driver_sql(f'SELECT * FROM "{name}" ORDER BY {order}'):
            digest.update(json.dumps(tuple(row), default=str).encode("utf-8"))
            if primary:
                primary_values.append(
                    [row[int(column[0])] for column in columns if str(column[1]) in primary]
                )
            count += 1
        output[str(name)] = {
            "schema": str(sql),
            "columns": [tuple(row) for row in columns],
            "indexes": [tuple(row) for row in indexes],
            "foreign_keys": [tuple(row) for row in foreign_keys],
            "primary_values": primary_values,
            "rows": count,
            "row_hash": digest.hexdigest(),
        }
    return output


def test_v342_schema_is_exact_append_only_idempotent_and_versioned(tmp_path: Path) -> None:
    assert len(V342_TABLES) == 8
    engine = _version21_database(tmp_path)
    with engine.connect() as connection:
        before = _old_table_fingerprint(connection)
    assert run_migrations(engine) == SCHEMA_VERSION == 22
    assert run_migrations(engine) == 22
    with engine.connect() as connection:
        tables = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {name for name in tables if name.startswith("v342_")} == set(V342_TABLES)
        assert _old_table_fingerprint(connection) == before
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 22
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        run_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE name='v342_policy_runs'"
        ).scalar_one()
        assert "next_executable_position_pp" in run_sql
        assert "fund_etf_ratio" in run_sql


def test_v342_migration_failure_rolls_back_without_residue(tmp_path: Path) -> None:
    engine = _version21_database(tmp_path)
    with engine.connect() as connection:
        before = _old_table_fingerprint(connection)

    def fail_after_first_table(connection):  # type: ignore[no-untyped-def]
        Base.metadata.tables[V342_TABLES[0]].create(connection, checkfirst=True)
        raise RuntimeError("injected V3.4.2 migration failure")

    with patch(
        "backend.app.database.migrations._upgrade_to_version_twenty_two",
        side_effect=fail_after_first_table,
    ):
        with pytest.raises(RuntimeError, match="injected V3.4.2 migration failure"):
            run_migrations(engine)

    with engine.connect() as connection:
        tables = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 21
        assert not (set(V342_TABLES) & tables)
        assert _old_table_fingerprint(connection) == before


def test_v342_sqlite_rejects_invalid_grid_fk_conflict_and_duplicate_champion(
    tmp_path: Path,
) -> None:
    engine = _version21_database(tmp_path)
    run_migrations(engine)
    policy_sql = (
        "INSERT INTO v342_policy_versions "
        "(id,protocol_version,model_market,version,parent_policy_id,status,"
        "effective_from_date,max_single_change_pp,minimum_cooldown_sessions,"
        "transaction_cost_bps,degraded_max_position_pp,uncertain_action,"
        "config_json,config_hash,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    with pytest.raises(IntegrityError, match="ck_v342_policy_max_change_grid"):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                policy_sql,
                (
                    "bad-grid", "V3.4.2", "399006", "bad", None, "RETROSPECTIVE",
                    "2026-08-02", 7, 5, 10, 30, "HOLD", "{}", "x" * 64,
                    "2026-08-02 00:00:00",
                ),
            )
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                policy_sql,
                (
                    "bad-fk", "V3.4.2", "399006", "bad-fk", "missing", "RETROSPECTIVE",
                    "2026-08-02", 20, 5, 10, 30, "HOLD", "{}", "y" * 64,
                    "2026-08-02 00:00:00",
                ),
            )
    with engine.begin() as connection:
        for suffix in ("a",):
            connection.exec_driver_sql(
                policy_sql,
                (
                    f"champion-{suffix}", "V3.4.2", "399006", f"champion-{suffix}",
                    None, "CHAMPION", "2026-08-02", 20, 5, 10, 30, "HOLD",
                    "{}", suffix * 64, "2026-08-02 00:00:00",
                ),
            )
    with pytest.raises(IntegrityError, match="UNIQUE"):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                policy_sql,
                (
                    "champion-b", "V3.4.2", "399006", "champion-b", None,
                    "CHAMPION", "2026-08-02", 20, 5, 10, 30, "HOLD", "{}",
                    "b" * 64, "2026-08-02 00:00:00",
                ),
            )
    with pytest.raises(IntegrityError, match="ck_v342_turning_conflict_not_consistent"):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO v342_turning_assessments "
                "(id,protocol_version,model_market,forecast_id,policy_version_id,"
                "forecast_anchor_date,assessment_version,window_extrema_json,"
                "price_turn_status,dif_turn_status,consistency_status,"
                "top_bottom_conflict,criteria_json,input_hash,assessment_hash,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "bad-conflict", "V3.4.2", "399006", -1, "champion-a",
                    "2026-08-02", "bad", "{}", "CONFIRMED", "CONFIRMED",
                    "TEMPORALLY_CONSISTENT", 1, "{}", "c" * 64, "d" * 64,
                    "2026-08-02 00:00:00",
                ),
            )
    with pytest.raises(IntegrityError, match="ck_v342_batch_number"):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO v342_policy_batches "
                "(policy_run_id,batch_number,action,change_pp,target_after_pp,"
                "window_start_date,window_end_date,initial_state,depends_on_batch_id,"
                "trigger_definition_json,invalidation_definition_json,batch_hash,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "missing-run", 5, "BUY", 5, 5, "2026-08-02", "2026-08-09",
                    "WAITING_CONFIRMATION", None, "{}", "{}", "e" * 64,
                    "2026-08-02 00:00:00",
                ),
            )
