from __future__ import annotations

from datetime import date
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import (
    CheckConstraint,
    ForeignKeyConstraint,
    UniqueConstraint,
    event,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

from backend.app.database.initialize import initialize_database
from backend.app.database.migrations import SCHEMA_VERSION, SchemaVersionError, run_migrations
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    Base,
    Instrument,
    MarketPrice,
    V2AdviceHistory,
    V2AnalysisIteration,
    V2AnalysisTask,
    V2IterationLabel,
    V2ModelVersion,
    V2PositionEvent,
    V2PositionSnapshot,
    V2WeekSample,
)


V2_TABLE_NAMES = {
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
}


def test_v2_model_metadata_declares_required_audit_and_domain_contracts() -> None:
    assert V2_TABLE_NAMES <= set(Base.metadata.tables)

    for table_name in V2_TABLE_NAMES:
        columns = set(Base.metadata.tables[table_name].c.keys())
        assert {"id", "status", "audit_data", "created_at", "updated_at"} <= columns

    for table_name in V2_TABLE_NAMES - {"v2_position_snapshots"}:
        assert "instrument_id" in Base.metadata.tables[table_name].c

    feature_snapshot = Base.metadata.tables["v2_feature_snapshots"]
    week_sample = Base.metadata.tables["v2_week_samples"]
    iteration = Base.metadata.tables["v2_analysis_iterations"]
    iteration_label = Base.metadata.tables["v2_iteration_labels"]
    model_version = Base.metadata.tables["v2_model_versions"]
    task = Base.metadata.tables["v2_analysis_tasks"]
    event = Base.metadata.tables["v2_position_events"]
    snapshot = Base.metadata.tables["v2_position_snapshots"]
    advice = Base.metadata.tables["v2_advice_history"]

    unique_column_sets = {
        table.name: {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        for table in (week_sample, iteration, model_version, task)
    }
    assert ("instrument_id", "week_key") in unique_column_sets[week_sample.name]
    assert ("instrument_id", "iteration_number") in unique_column_sets[iteration.name]
    assert ("instrument_id", "week_key") in unique_column_sets[iteration.name]
    assert (
        "instrument_id",
        "iteration_number",
        "work_number",
    ) in unique_column_sets[iteration.name]
    assert ("instrument_id", "model_number") in unique_column_sets[model_version.name]
    assert ("task_key",) in unique_column_sets[task.name]
    assert iteration.c.work_number.nullable is False
    iteration_checks = " ".join(
        str(constraint.sqltext)
        for constraint in iteration.constraints
        if isinstance(constraint, CheckConstraint)
    )
    assert "work_number > 0" in iteration_checks

    assert {
        "direction",
        "operation_date",
        "change_percent",
        "note",
        "sequence",
    } <= set(event.c.keys())
    event_checks = " ".join(
        str(constraint.sqltext)
        for constraint in event.constraints
        if isinstance(constraint, CheckConstraint)
    )
    assert "increase" in event_checks and "decrease" in event_checks
    assert "change_percent" in event_checks
    assert "period_bar_id" not in feature_snapshot.c
    assert "feature_snapshot_id" not in week_sample.c
    assert "week_sample_id" not in iteration.c
    assert "analysis_iteration_id" not in iteration_label.c
    assert "source_iteration_id" not in model_version.c
    assert {"instrument_id", "snapshot_date"}.isdisjoint(snapshot.c.keys())
    assert {"model_version_id", "analysis_iteration_id"}.isdisjoint(advice.c.keys())
    assert "position_percent" in snapshot.c
    assert {
        "model_number",
        "work_number",
        "iteration_number",
        "advice_generation",
        "generation_key",
        "payload",
    } <= set(advice.c.keys())
    assert {"worker_token", "lease_expires_at"} <= set(task.c.keys())
    advice_unique_sets = {
        tuple(column.name for column in constraint.columns)
        for constraint in advice.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    assert (
        "instrument_id",
        "model_number",
        "work_number",
        "iteration_number",
        "advice_generation",
    ) in advice_unique_sets
    assert (
        "instrument_id",
        "model_number",
        "work_number",
        "iteration_number",
        "generation_key",
    ) in advice_unique_sets

    def composite_foreign_keys(table_name: str) -> set[tuple[tuple[str, ...], str, tuple[str, ...]]]:
        constraints = Base.metadata.tables[table_name].constraints
        return {
            (
                tuple(element.parent.name for element in constraint.elements),
                constraint.elements[0].column.table.name,
                tuple(element.column.name for element in constraint.elements),
            )
            for constraint in constraints
            if isinstance(constraint, ForeignKeyConstraint)
        }

    assert (
        ("instrument_id", "week_key"),
        "v2_week_samples",
        ("instrument_id", "week_key"),
    ) in composite_foreign_keys("v2_analysis_iterations")
    assert (
        ("instrument_id", "week_key"),
        "v2_analysis_iterations",
        ("instrument_id", "week_key"),
    ) in composite_foreign_keys("v2_iteration_labels")
    assert (
        ("instrument_id", "model_number"),
        "v2_model_versions",
        ("instrument_id", "model_number"),
    ) in composite_foreign_keys("v2_advice_history")
    assert (
        ("instrument_id", "iteration_number", "work_number"),
        "v2_analysis_iterations",
        ("instrument_id", "iteration_number", "work_number"),
    ) in composite_foreign_keys("v2_advice_history")
    assert (
        ("position_event_id",),
        "v2_position_events",
        ("id",),
    ) in composite_foreign_keys("v2_position_snapshots")


def _schema_fingerprint(connection, table_names: set[str]) -> dict[str, object]:
    fingerprint: dict[str, object] = {}
    for table_name in sorted(table_names):
        quoted_name = f'"{table_name}"'
        create_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :name"),
            {"name": table_name},
        ).scalar_one()
        columns = tuple(
            tuple(row)
            for row in connection.exec_driver_sql(f"PRAGMA table_info({quoted_name})")
        )
        foreign_keys = tuple(
            tuple(row)
            for row in connection.exec_driver_sql(f"PRAGMA foreign_key_list({quoted_name})")
        )
        unique_indexes: list[tuple[object, ...]] = []
        for index_row in connection.exec_driver_sql(f"PRAGMA index_list({quoted_name})"):
            if not index_row[2]:
                continue
            index_name = index_row[1].replace('"', '""')
            index_columns = tuple(
                row[2]
                for row in connection.exec_driver_sql(
                    f'PRAGMA index_info("{index_name}")'
                )
            )
            unique_indexes.append((index_row[1], index_row[3], index_columns))
        fingerprint[table_name] = (
            create_sql,
            columns,
            foreign_keys,
            tuple(sorted(unique_indexes)),
        )
    return fingerprint


def _assert_integrity_error(engine, statement) -> None:
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(statement)


def test_v2_sqlite_rejects_cross_provenance_and_enforces_domain_constraints(
    tmp_path: Path,
) -> None:
    assert "week_sample_id" not in V2AnalysisIteration.__table__.c
    assert "analysis_iteration_id" not in V2IterationLabel.__table__.c
    assert {"model_version_id", "analysis_iteration_id"}.isdisjoint(
        V2AdviceHistory.__table__.c.keys()
    )
    assert {"instrument_id", "snapshot_date"}.isdisjoint(
        V2PositionSnapshot.__table__.c.keys()
    )

    database = tmp_path / "data" / "investment_lab.db"
    engine = create_database_engine(database)
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as connection:
            instrument_one = connection.execute(
                Instrument.__table__.insert().values(
                    code="V2-A", name="V2 A", exchange="TEST"
                )
            ).inserted_primary_key[0]
            instrument_two = connection.execute(
                Instrument.__table__.insert().values(
                    code="V2-B", name="V2 B", exchange="TEST"
                )
            ).inserted_primary_key[0]
            connection.execute(
                V2WeekSample.__table__.insert(),
                [
                    {
                        "instrument_id": instrument_one,
                        "week_key": "2026-W01",
                        "week_start": date(2025, 12, 29),
                        "week_end": date(2026, 1, 2),
                    },
                    {
                        "instrument_id": instrument_two,
                        "week_key": "2026-W02",
                        "week_start": date(2026, 1, 5),
                        "week_end": date(2026, 1, 9),
                    },
                    {
                        "instrument_id": instrument_one,
                        "week_key": "2026-W03",
                        "week_start": date(2026, 1, 12),
                        "week_end": date(2026, 1, 16),
                    },
                    {
                        "instrument_id": instrument_one,
                        "week_key": "2026-W04",
                        "week_start": date(2026, 1, 19),
                        "week_end": date(2026, 1, 23),
                    },
                ],
            )
            connection.execute(
                V2AnalysisIteration.__table__.insert(),
                [
                    {
                        "instrument_id": instrument_one,
                        "week_key": "2026-W01",
                        "iteration_number": 1,
                        "work_number": 7,
                    },
                    {
                        "instrument_id": instrument_two,
                        "week_key": "2026-W02",
                        "iteration_number": 2,
                        "work_number": 2,
                    },
                ],
            )
            connection.execute(
                V2ModelVersion.__table__.insert(),
                [
                    {
                        "instrument_id": instrument_one,
                        "model_number": 1,
                    },
                    {
                        "instrument_id": instrument_two,
                        "model_number": 2,
                    },
                ],
            )
            event_id = connection.execute(
                V2PositionEvent.__table__.insert().values(
                    instrument_id=instrument_one,
                    direction="increase",
                    operation_date=date(2026, 1, 5),
                    change_percent=Decimal("5"),
                    sequence=1,
                )
            ).inserted_primary_key[0]
            connection.execute(
                V2PositionSnapshot.__table__.insert().values(
                    position_event_id=event_id,
                    position_percent=Decimal("5"),
                )
            )
            connection.execute(
                V2PositionEvent.__table__.insert().values(
                    instrument_id=instrument_one,
                    direction="decrease",
                    operation_date=date(2026, 1, 6),
                    change_percent=Decimal("100"),
                    sequence=1,
                )
            )
            connection.execute(
                V2AdviceHistory.__table__.insert().values(
                    instrument_id=instrument_one,
                    model_number=1,
                    work_number=7,
                    iteration_number=1,
                )
            )

        _assert_integrity_error(
            engine,
            V2AdviceHistory.__table__.insert().values(
                instrument_id=instrument_one,
                model_number=1,
                work_number=999,
                iteration_number=1,
            ),
        )
        _assert_integrity_error(
            engine,
            V2AnalysisIteration.__table__.insert().values(
                instrument_id=instrument_one,
                week_key="2026-W03",
                iteration_number=3,
            ),
        )
        _assert_integrity_error(
            engine,
            V2AnalysisIteration.__table__.insert().values(
                instrument_id=instrument_one,
                week_key="2026-W04",
                iteration_number=4,
                work_number=0,
            ),
        )
        _assert_integrity_error(
            engine,
            V2AnalysisIteration.__table__.insert().values(
                instrument_id=instrument_one,
                week_key="2026-W02",
                iteration_number=3,
                work_number=3,
            ),
        )
        _assert_integrity_error(
            engine,
            V2IterationLabel.__table__.insert().values(
                instrument_id=instrument_one,
                week_key="2026-W02",
            ),
        )
        _assert_integrity_error(
            engine,
            V2AdviceHistory.__table__.insert().values(
                instrument_id=instrument_one,
                model_number=2,
                work_number=7,
                iteration_number=1,
            ),
        )
        _assert_integrity_error(
            engine,
            V2AdviceHistory.__table__.insert().values(
                instrument_id=instrument_one,
                model_number=1,
                work_number=2,
                iteration_number=2,
            ),
        )
        _assert_integrity_error(
            engine,
            V2PositionSnapshot.__table__.insert().values(
                position_event_id=999_999,
                position_percent=Decimal("10"),
            ),
        )
        _assert_integrity_error(
            engine,
            V2WeekSample.__table__.insert().values(
                instrument_id=instrument_one,
                week_key="2026-W01",
                week_start=date(2025, 12, 29),
                week_end=date(2026, 1, 2),
            ),
        )
        _assert_integrity_error(
            engine,
            V2PositionEvent.__table__.insert().values(
                instrument_id=instrument_one,
                direction="hold",
                operation_date=date(2026, 1, 7),
                change_percent=Decimal("5"),
                sequence=1,
            ),
        )
        _assert_integrity_error(
            engine,
            V2PositionEvent.__table__.insert().values(
                instrument_id=instrument_one,
                direction="increase",
                operation_date=date(2026, 1, 8),
                change_percent=Decimal("7"),
                sequence=1,
            ),
        )
    finally:
        engine.dispose()


def _table_names(database: Path) -> set[str]:
    engine = create_database_engine(database)
    try:
        with engine.connect() as connection:
            return set(
                connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type = 'table'")
                ).scalars()
            )
    finally:
        engine.dispose()


def test_new_database_uses_current_schema_and_contains_all_v2_tables(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"

    initialize_database(database, tmp_path / "config")

    engine = create_database_engine(database)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA user_version")).scalar_one() == SCHEMA_VERSION
            market_price_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(market_prices)"
                )
            }
    finally:
        engine.dispose()
    assert "volume_source" in market_price_columns
    assert V2_TABLE_NAMES <= _table_names(database)


def _replace_advice_with_v15_table(connection) -> None:
    connection.exec_driver_sql(
        "ALTER TABLE v2_advice_history RENAME TO v2_advice_history_v16"
    )
    connection.exec_driver_sql(
        """
        CREATE TABLE v2_advice_history (
            id INTEGER NOT NULL PRIMARY KEY,
            instrument_id INTEGER NOT NULL,
            model_number INTEGER NOT NULL,
            work_number INTEGER NOT NULL,
            iteration_number INTEGER NOT NULL,
            advice_at DATETIME NOT NULL,
            status VARCHAR(32) NOT NULL,
            payload JSON NOT NULL,
            audit_data JSON NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT uq_v2_advice_history_provenance UNIQUE (
                instrument_id, model_number, work_number, iteration_number
            ),
            CONSTRAINT fk_v2_advice_history_model FOREIGN KEY (
                instrument_id, model_number
            ) REFERENCES v2_model_versions (instrument_id, model_number),
            CONSTRAINT fk_v2_advice_history_iteration FOREIGN KEY (
                instrument_id, iteration_number, work_number
            ) REFERENCES v2_analysis_iterations (
                instrument_id, iteration_number, work_number
            ),
            FOREIGN KEY(instrument_id) REFERENCES instruments (id)
        )
        """
    )
    connection.exec_driver_sql(
        """
        INSERT INTO v2_advice_history (
            id, instrument_id, model_number, work_number, iteration_number,
            advice_at, status, payload, audit_data, created_at, updated_at
        )
        SELECT
            id, instrument_id, model_number, work_number, iteration_number,
            advice_at, status, payload, audit_data, created_at, updated_at
        FROM v2_advice_history_v16
        """
    )
    connection.exec_driver_sql("DROP TABLE v2_advice_history_v16")


def _prepare_nonempty_v15_advice(
    database: Path,
    configuration_directory: Path,
    *,
    malformed_audit: bool = False,
    audit_root: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    initialize_database(database, configuration_directory)
    sessions = create_session_factory(database)
    repository_module = __import__(
        "backend.app.weekly_analysis.repository",
        fromlist=["WeeklyAnalysisRepository"],
    )
    repository_test = __import__(
        "backend.tests.test_weekly_repository_v2",
        fromlist=["_step", "_advice"],
    )
    custom_audit_root = audit_root is not None
    audit_root = (
        database.parent / "weekly_analysis_v2"
        if audit_root is None
        else audit_root
    )
    repository = repository_module.WeeklyAnalysisRepository(
        sessions,
        audit_root=audit_root,
    )
    result = repository_test._step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    advice = repository_test._advice(result.work_state)
    advice_id = repository.save_advice(
        "399006",
        advice,
        generation_key="pre-v16",
    )

    with sessions.begin() as session:
        row = session.get(V2AdviceHistory, advice_id)
        assert row is not None
        audit = dict(row.audit_data)
        snapshot = dict(audit["snapshot"])
        snapshot.pop("advice_generation", None)
        snapshot.pop("generation_key", None)
        snapshot.pop("snapshot_hash", None)
        legacy_snapshot = repository_module._hashed_snapshot(snapshot)
        audit.update(
            {
                "snapshot": legacy_snapshot,
                "snapshot_hash": legacy_snapshot["snapshot_hash"],
            }
        )
        if not custom_audit_root:
            audit.pop("artifact_path", None)
            audit.pop("audit_root", None)
        if malformed_audit:
            audit = {"malformed": True}
        row.audit_data = audit

    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            _replace_advice_with_v15_table(connection)
            connection.exec_driver_sql("PRAGMA user_version = 15")
    finally:
        engine.dispose()

    relative_path = Path(str(audit.get("snapshot_path", "missing.json")))
    artifact = audit_root / relative_path
    if not malformed_audit:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(
            repository_module._snapshot_bytes(legacy_snapshot)
        )
    return artifact, advice.to_dict()


def _prepare_two_nonempty_v15_advices(
    database: Path,
    configuration_directory: Path,
) -> tuple[tuple[Path, ...], tuple[dict[str, object], ...]]:
    initialize_database(database, configuration_directory)
    sessions = create_session_factory(database)
    repository_module = __import__(
        "backend.app.weekly_analysis.repository",
        fromlist=["WeeklyAnalysisRepository"],
    )
    repository_test = __import__(
        "backend.tests.test_weekly_repository_v2",
        fromlist=["_step", "_advice", "_mature_label", "_second_step"],
    )
    audit_root = database.parent / "weekly_analysis_v2"
    repository = repository_module.WeeklyAnalysisRepository(
        sessions,
        audit_root=audit_root,
    )
    first = repository_test._step()
    repository.commit_iteration(
        "399006", first.audit, first.work_state, labels=()
    )
    first_advice = repository_test._advice(first.work_state)
    repository.save_advice(
        "399006",
        first_advice,
        generation_key="pre-v16:first",
    )
    label = repository_test._mature_label()
    second = repository_test._second_step(first, label)
    repository.commit_iteration(
        "399006",
        second.audit,
        second.work_state,
        labels=(label,),
    )
    second_advice = repository_test._advice(
        second.work_state,
        cutoff=second.audit.cutoff_date,
    )
    repository.save_advice(
        "399006",
        second_advice,
        generation_key="pre-v16:second",
    )

    legacy_snapshots: list[dict[str, object]] = []
    artifacts: list[Path] = []
    with sessions.begin() as session:
        rows = tuple(
            session.scalars(
                select(V2AdviceHistory).order_by(V2AdviceHistory.id)
            )
        )
        assert len(rows) == 2
        for row in rows:
            audit = dict(row.audit_data)
            snapshot = dict(audit["snapshot"])
            snapshot.pop("advice_generation", None)
            snapshot.pop("generation_key", None)
            snapshot.pop("snapshot_hash", None)
            legacy_snapshot = repository_module._hashed_snapshot(snapshot)
            audit.update(
                {
                    "snapshot": legacy_snapshot,
                    "snapshot_hash": legacy_snapshot["snapshot_hash"],
                }
            )
            audit.pop("artifact_path", None)
            audit.pop("audit_root", None)
            row.audit_data = audit
            legacy_snapshots.append(legacy_snapshot)
            artifacts.append(
                audit_root / Path(str(audit["snapshot_path"]))
            )

    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            _replace_advice_with_v15_table(connection)
            connection.exec_driver_sql("PRAGMA user_version = 15")
    finally:
        engine.dispose()
    for artifact, snapshot in zip(
        artifacts, legacy_snapshots, strict=True
    ):
        artifact.write_bytes(repository_module._snapshot_bytes(snapshot))
    return (
        tuple(artifacts),
        (first_advice.to_dict(), second_advice.to_dict()),
    )


def _assert_v15_advice_schema(database: Path, *, row_count: int) -> None:
    engine = create_database_engine(database)
    try:
        with engine.connect() as connection:
            assert connection.execute(
                text("PRAGMA user_version")
            ).scalar_one() == 15
            columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(v2_advice_history)"
                )
            }
            assert "advice_generation" not in columns
            assert connection.execute(
                text("SELECT COUNT(*) FROM v2_advice_history")
            ).scalar_one() == row_count
    finally:
        engine.dispose()


def test_v15_two_artifact_second_replace_failure_is_fully_compensated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    artifacts, _payloads = _prepare_two_nonempty_v15_advices(
        database,
        tmp_path / "config",
    )
    original_bytes = tuple(path.read_bytes() for path in artifacts)
    migrations = __import__(
        "backend.app.database.migrations",
        fromlist=["run_migrations"],
    )
    real_replace = migrations.os.replace
    attempts = 0

    def fail_second_staged_replace(source, destination) -> None:
        nonlocal attempts
        if str(source).endswith(".v16.stage"):
            attempts += 1
            if attempts == 2:
                raise OSError("injected second artifact replace failure")
        real_replace(source, destination)

    engine = create_database_engine(database)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                migrations.os,
                "replace",
                fail_second_staged_replace,
            )
            with pytest.raises(
                OSError,
                match="second artifact replace failure",
            ):
                run_migrations(engine)
        _assert_v15_advice_schema(database, row_count=2)
        assert tuple(path.read_bytes() for path in artifacts) == original_bytes
        assert not tuple(database.parent.rglob("*.v16.stage"))
        assert not tuple(database.parent.rglob("*.v15.backup"))

        assert run_migrations(engine) == SCHEMA_VERSION
    finally:
        engine.dispose()
    for generation, artifact in enumerate(artifacts, start=1):
        upgraded = json.loads(artifact.read_text(encoding="utf-8"))
        assert upgraded["advice_generation"] == 1
        assert upgraded["generation_key"] == f"legacy:{generation}"


def test_v15_artifacts_are_compensated_when_database_commit_fails(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    artifacts, _payloads = _prepare_two_nonempty_v15_advices(
        database,
        tmp_path / "config",
    )
    original_bytes = tuple(path.read_bytes() for path in artifacts)
    engine = create_database_engine(database)

    def fail_commit(_connection) -> None:
        raise RuntimeError("injected migration commit failure")

    event.listen(engine, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="commit failure"):
            run_migrations(engine)
    finally:
        event.remove(engine, "commit", fail_commit)
    try:
        _assert_v15_advice_schema(database, row_count=2)
        assert tuple(path.read_bytes() for path in artifacts) == original_bytes
        assert run_migrations(engine) == SCHEMA_VERSION
    finally:
        engine.dispose()


def test_v15_commit_succeeds_then_raises_is_recognized_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    artifacts, payloads = _prepare_two_nonempty_v15_advices(
        database,
        tmp_path / "config",
    )
    engine = create_database_engine(database)
    real_do_commit = engine.dialect.do_commit
    raised = False

    def commit_then_raise(dbapi_connection) -> None:
        nonlocal raised
        real_do_commit(dbapi_connection)
        if not raised:
            raised = True
            raise RuntimeError("injected error after durable commit")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                engine.dialect,
                "do_commit",
                commit_then_raise,
            )
            assert run_migrations(engine) == SCHEMA_VERSION
        assert raised
        assert run_migrations(engine) == SCHEMA_VERSION
    finally:
        engine.dispose()

    repository_module = __import__(
        "backend.app.weekly_analysis.repository",
        fromlist=["WeeklyAnalysisRepository"],
    )
    repository = repository_module.WeeklyAnalysisRepository(
        create_session_factory(database),
        audit_root=database.parent / "weekly_analysis_v2",
    )
    latest = repository.latest_advice("399006")
    assert latest is not None
    assert latest.generation_key == "legacy:2"
    assert dict(latest.payload) == payloads[-1]
    for generation, artifact in enumerate(artifacts, start=1):
        upgraded = json.loads(artifact.read_text(encoding="utf-8"))
        assert upgraded["generation_key"] == f"legacy:{generation}"
    assert not tuple(database.parent.rglob("*.v15.backup"))
    assert not tuple(database.parent.rglob("*.v16.stage"))


def test_v15_custom_audit_root_artifact_is_upgraded_in_place(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    custom_root = tmp_path / "custom-weekly-audits"
    artifact, expected_payload = _prepare_nonempty_v15_advice(
        database,
        tmp_path / "config",
        audit_root=custom_root,
    )
    legacy_default = (
        database.parent
        / "weekly_analysis_v2"
        / artifact.relative_to(custom_root)
    )
    assert artifact.exists()
    assert not legacy_default.exists()

    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            row = connection.exec_driver_sql(
                "SELECT id, audit_data FROM v2_advice_history"
            ).mappings().one()
            audit = json.loads(row["audit_data"])
            audit.pop("artifact_path", None)
            audit.pop("audit_root", None)
            connection.exec_driver_sql(
                "UPDATE v2_advice_history SET audit_data = ? WHERE id = ?",
                (
                    json.dumps(
                        audit,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    row["id"],
                ),
            )
    finally:
        engine.dispose()
    initialize_database(
        database,
        tmp_path / "config",
        legacy_weekly_analysis_audit_root=custom_root,
    )
    upgraded = json.loads(artifact.read_text(encoding="utf-8"))
    assert upgraded["advice_generation"] == 1
    assert upgraded["generation_key"] == "legacy:1"
    assert not legacy_default.exists()

    repository_module = __import__(
        "backend.app.weekly_analysis.repository",
        fromlist=["WeeklyAnalysisRepository"],
    )
    repository = repository_module.WeeklyAnalysisRepository(
        create_session_factory(database),
        audit_root=custom_root,
    )
    latest = repository.latest_advice("399006")
    assert latest is not None
    assert dict(latest.payload) == expected_payload


@pytest.mark.parametrize(
    "tamper",
    ("escape", "cross_symbol", "absolute_outside"),
)
def test_v15_persisted_artifact_path_is_symbol_scoped(
    tmp_path: Path,
    tamper: str,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    custom_root = tmp_path / "custom-weekly-audits"
    artifact, _payload = _prepare_nonempty_v15_advice(
        database,
        tmp_path / "config",
        audit_root=custom_root,
    )
    engine = create_database_engine(database)
    outside_artifact: Path | None = None
    outside_bytes: bytes | None = None
    with engine.begin() as connection:
        row = connection.exec_driver_sql(
            "SELECT id, audit_data FROM v2_advice_history"
        ).mappings().one()
        audit = json.loads(row["audit_data"])
        if tamper == "escape":
            audit["artifact_path"] = (
                Path("..") / "outside" / artifact.name
            ).as_posix()
        else:
            if tamper == "cross_symbol":
                audit["artifact_path"] = str(
                    custom_root / "NDX" / artifact.name
                )
            else:
                outside_artifact = (
                    tmp_path
                    / "untrusted"
                    / artifact.relative_to(custom_root)
                )
                outside_artifact.parent.mkdir(parents=True, exist_ok=True)
                outside_bytes = artifact.read_bytes()
                outside_artifact.write_bytes(outside_bytes)
                audit["artifact_path"] = str(outside_artifact)
        connection.exec_driver_sql(
            "UPDATE v2_advice_history SET audit_data = ? WHERE id = ?",
            (
                json.dumps(
                    audit,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                row["id"],
            ),
        )

    try:
        with pytest.raises(
            SchemaVersionError,
            match="artifact_path|audit root|symbol",
        ):
            run_migrations(engine)
    finally:
        engine.dispose()
    _assert_v15_advice_schema(database, row_count=1)
    assert "advice_generation" not in json.loads(
        artifact.read_text(encoding="utf-8")
    )
    if outside_artifact is not None:
        assert outside_artifact.read_bytes() == outside_bytes


def test_upgrade_from_v15_adds_advice_generation_and_task_claim_columns(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            task_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(v2_analysis_tasks)"
                )
            }
            for column in ("worker_token", "lease_expires_at"):
                if column in task_columns:
                    connection.exec_driver_sql(
                        f"ALTER TABLE v2_analysis_tasks DROP COLUMN {column}"
                    )
            _replace_advice_with_v15_table(connection)
            connection.exec_driver_sql("PRAGMA user_version = 15")

        assert run_migrations(engine) == SCHEMA_VERSION
        with engine.connect() as connection:
            task_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(v2_analysis_tasks)"
                )
            }
            advice_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(v2_advice_history)"
                )
            }
            assert {"worker_token", "lease_expires_at"} <= task_columns
            assert {"advice_generation", "generation_key"} <= advice_columns
    finally:
        engine.dispose()


@pytest.mark.parametrize("artifact_exists", (True, False))
def test_v15_nonempty_advice_upgrade_preserves_readable_history(
    tmp_path: Path,
    artifact_exists: bool,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    artifact, expected_payload = _prepare_nonempty_v15_advice(
        database,
        tmp_path / "config",
    )
    if not artifact_exists:
        artifact.unlink()

    engine = create_database_engine(database)
    try:
        assert run_migrations(engine) == SCHEMA_VERSION
    finally:
        engine.dispose()

    if artifact_exists:
        upgraded_file = json.loads(artifact.read_text(encoding="utf-8"))
        assert upgraded_file["advice_generation"] == 1
        assert upgraded_file["generation_key"] == "legacy:1"
    else:
        assert not artifact.exists()

    repository_module = __import__(
        "backend.app.weekly_analysis.repository",
        fromlist=["WeeklyAnalysisRepository"],
    )
    repository = repository_module.WeeklyAnalysisRepository(
        create_session_factory(database),
        audit_root=database.parent / "weekly_analysis_v2",
    )
    latest = repository.latest_advice("399006")
    assert latest is not None
    assert latest.advice_generation == 1
    assert latest.generation_key == "legacy:1"
    assert dict(latest.payload) == expected_payload
    assert artifact.exists()


def test_v15_nonempty_advice_migration_failure_rolls_back(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    _prepare_nonempty_v15_advice(
        database,
        tmp_path / "config",
        malformed_audit=True,
    )
    engine = create_database_engine(database)
    try:
        with pytest.raises(SchemaVersionError, match="advice|snapshot|audit"):
            run_migrations(engine)
        with engine.connect() as connection:
            assert connection.execute(
                text("PRAGMA user_version")
            ).scalar_one() == 15
            columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(v2_advice_history)"
                )
            }
            assert "advice_generation" not in columns
            assert connection.execute(
                text("SELECT COUNT(*) FROM v2_advice_history")
            ).scalar_one() == 1
    finally:
        engine.dispose()


def test_fake_v16_missing_generation_column_is_rejected(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            _replace_advice_with_v15_table(connection)
            connection.exec_driver_sql("PRAGMA user_version = 16")
        with pytest.raises(SchemaVersionError, match="generation_key|v2_advice"):
            run_migrations(engine)
    finally:
        engine.dispose()


def test_upgrade_from_version_fourteen_adds_volume_provenance_without_rebuilding_v2(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    configuration_directory = tmp_path / "config"
    initialize_database(database, configuration_directory)
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            instrument_id = connection.execute(
                select(Instrument.id).where(Instrument.code == "589850")
            ).scalar_one()
            ndx_id = connection.execute(
                select(Instrument.id).where(Instrument.code == "NDX")
            ).scalar_one()
            connection.execute(
                MarketPrice.__table__.insert(),
                (
                    {
                        "instrument_id": instrument_id,
                        "trade_date": date(2026, 1, 2),
                        "timeframe": "daily",
                        "close_price": Decimal("1.25"),
                        "volume": Decimal("100"),
                        "source": "LEGACY_PROVIDER",
                        "volume_source": "LEGACY_PROVIDER",
                    },
                    {
                        "instrument_id": instrument_id,
                        "trade_date": date(2026, 1, 9),
                        "timeframe": "weekly",
                        "close_price": Decimal("1.30"),
                        "volume": Decimal("500"),
                        "source": "AGGREGATED_DAILY_VOLUME",
                        "volume_source": "AGGREGATED_DAILY_VOLUME",
                    },
                    {
                        "instrument_id": instrument_id,
                        "trade_date": date(2026, 1, 31),
                        "timeframe": "monthly",
                        "close_price": Decimal("1.35"),
                        "volume": Decimal("2000"),
                        "source": "AKSHARE_INDEX_WEEKLY",
                        "volume_source": "AKSHARE_INDEX_WEEKLY",
                    },
                    {
                        "instrument_id": ndx_id,
                        "trade_date": date(2026, 1, 2),
                        "timeframe": "daily",
                        "close_price": Decimal("20000"),
                        "volume": None,
                        "volume_multiplier": 10,
                        "source": "LEGACY_NDX",
                        "volume_source": None,
                    },
                ),
            )
            connection.execute(
                V2AnalysisTask.__table__.insert().values(
                    instrument_id=instrument_id,
                    task_key="v14-provenance-upgrade",
                    task_type="weekly-analysis",
                    request_payload={"preserve": True},
                )
            )
            connection.exec_driver_sql(
                "ALTER TABLE market_prices DROP COLUMN volume_source"
            )
            connection.exec_driver_sql("PRAGMA user_version = 14")
        stable_v2_tables = V2_TABLE_NAMES - {
            "v2_analysis_tasks",
            "v2_advice_history",
        }
        with engine.connect() as connection:
            before_stable_v2_schema = _schema_fingerprint(
                connection, stable_v2_tables
            )
            before_v2_row = connection.execute(
                text(
                    "SELECT task_key, status, request_payload, audit_data "
                    "FROM v2_analysis_tasks "
                    "WHERE task_key = 'v14-provenance-upgrade'"
                )
            ).one()

        assert run_migrations(engine) == SCHEMA_VERSION

        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA user_version")).scalar_one() == SCHEMA_VERSION
            assert (
                _schema_fingerprint(connection, stable_v2_tables)
                == before_stable_v2_schema
            )
            task_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    'PRAGMA table_info("v2_analysis_tasks")'
                )
            }
            advice_columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    'PRAGMA table_info("v2_advice_history")'
                )
            }
            assert {"worker_token", "lease_expires_at"} <= task_columns
            assert {"advice_generation", "generation_key"} <= advice_columns
            assert connection.execute(
                text(
                    "SELECT task_key, status, request_payload, audit_data "
                    "FROM v2_analysis_tasks "
                    "WHERE task_key = 'v14-provenance-upgrade'"
                )
            ).one() == before_v2_row
            legacy_rows = connection.execute(
                text(
                    "SELECT trade_date, timeframe, close_price, volume, "
                    "volume_multiplier, source, volume_source "
                    "FROM market_prices WHERE instrument_id = :instrument_id "
                    "ORDER BY trade_date"
                ),
                {"instrument_id": instrument_id},
            ).all()
            assert legacy_rows == [
                (
                    "2026-01-02",
                    "daily",
                    125000000,
                    10000000000,
                    1,
                    "LEGACY_PROVIDER",
                    "LEGACY_PROVIDER",
                ),
                (
                    "2026-01-09",
                    "weekly",
                    130000000,
                    50000000000,
                    1,
                    "LEGACY_PRICE_SOURCE_UNKNOWN",
                    "AGGREGATED_DAILY_VOLUME",
                ),
                (
                    "2026-01-31",
                    "monthly",
                    135000000,
                    200000000000,
                    1,
                    "LEGACY_PRICE_SOURCE_UNKNOWN",
                    "AKSHARE_INDEX_WEEKLY",
                ),
            ]
            assert connection.execute(
                text(
                    "SELECT volume, volume_multiplier, source, volume_source "
                    "FROM market_prices WHERE instrument_id = :instrument_id"
                ),
                {"instrument_id": ndx_id},
            ).one() == (
                None,
                1,
                "LEGACY_NDX",
                "VOLUME_UNAVAILABLE:DIRECT_INDEX",
            )
        from backend.web import prices

        payload = prices(
            SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(
                        sessions=create_session_factory(database)
                    )
                )
            ),
            "589850",
            "weekly",
        )
        assert payload["rows"][0]["source"] == "LEGACY_PRICE_SOURCE_UNKNOWN"
        assert payload["rows"][0]["volume_source"] == "AGGREGATED_DAILY_VOLUME"
    finally:
        engine.dispose()


def test_version_fifteen_missing_volume_source_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "ALTER TABLE market_prices DROP COLUMN volume_source"
            )
            connection.exec_driver_sql("PRAGMA user_version = 15")

        with pytest.raises(SchemaVersionError, match="volume_source"):
            run_migrations(engine)

        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA user_version")).scalar_one() == 15
    finally:
        engine.dispose()


def test_v2_schema_initialization_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    configuration_directory = tmp_path / "config"

    initialize_database(database, configuration_directory)
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            instrument_id = connection.execute(
                select(Instrument.id).where(Instrument.code == "589850")
            ).scalar_one()
            connection.execute(
                V2AnalysisTask.__table__.insert().values(
                    instrument_id=instrument_id,
                    task_key="idempotence-representative",
                    task_type="weekly-analysis",
                    request_payload={"preserve": True},
                )
            )
        with engine.connect() as connection:
            before_schema = _schema_fingerprint(connection, V2_TABLE_NAMES)
            before_row = connection.execute(
                text(
                    "SELECT task_key, status, request_payload, audit_data "
                    "FROM v2_analysis_tasks WHERE task_key = 'idempotence-representative'"
                )
            ).one()
    finally:
        engine.dispose()

    initialize_database(database, configuration_directory)

    engine = create_database_engine(database)
    try:
        with engine.connect() as connection:
            after_schema = _schema_fingerprint(connection, V2_TABLE_NAMES)
            after_row = connection.execute(
                text(
                    "SELECT task_key, status, request_payload, audit_data "
                    "FROM v2_analysis_tasks WHERE task_key = 'idempotence-representative'"
                )
            ).one()
    finally:
        engine.dispose()
    assert after_schema == before_schema
    assert after_row == before_row


def test_version_thirteen_with_incomplete_v2_table_does_not_upgrade(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE v2_week_samples (id INTEGER PRIMARY KEY)"
            )
            connection.exec_driver_sql("PRAGMA user_version = 13")

        with pytest.raises(SchemaVersionError, match="v2_week_samples"):
            run_migrations(engine)

        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA user_version")).scalar_one() == 13
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "defect",
    (
        "id_affinity",
        "instrument_affinity",
        "primary_key",
        "nullable",
        "ordinary_index",
        "unexpected_default",
        "unexpected_check",
        "unexpected_index",
        "unexpected_unique_index",
        "foreign_key_cascade",
        "partial_index",
        "unexpected_column",
    ),
)
def test_version_thirteen_rejects_v2_table_with_incompatible_physical_schema(
    tmp_path: Path,
    defect: str,
) -> None:
    database = tmp_path / "data" / f"{defect}.db"
    engine = create_database_engine(database)
    try:
        correct_create_sql = str(
            CreateTable(V2AnalysisTask.__table__).compile(dialect=engine.dialect)
        )
        replacements = {
            "id_affinity": ("id INTEGER NOT NULL", "id TEXT NOT NULL"),
            "instrument_affinity": (
                "instrument_id INTEGER NOT NULL",
                "instrument_id TEXT NOT NULL",
            ),
            "primary_key": ("PRIMARY KEY (id)", "PRIMARY KEY (instrument_id)"),
            "nullable": (
                "task_type VARCHAR(64) NOT NULL",
                "task_type VARCHAR(64)",
            ),
            "unexpected_default": (
                "task_type VARCHAR(64) NOT NULL",
                "task_type VARCHAR(64) NOT NULL DEFAULT 'rogue'",
            ),
            "unexpected_check": (
                "FOREIGN KEY(instrument_id) REFERENCES instruments (id)",
                "FOREIGN KEY(instrument_id) REFERENCES instruments (id), "
                "\n\tCONSTRAINT ck_v2_analysis_tasks_rogue "
                "CHECK (task_type = 'only')",
            ),
            "foreign_key_cascade": (
                "FOREIGN KEY(instrument_id) REFERENCES instruments (id)",
                "FOREIGN KEY(instrument_id) REFERENCES instruments (id) "
                "ON DELETE CASCADE",
            ),
            "unexpected_column": (
                "task_type VARCHAR(64) NOT NULL",
                "task_type VARCHAR(64) NOT NULL, "
                "\n\trogue_required TEXT NOT NULL",
            ),
        }
        malformed_create_sql = correct_create_sql
        if defect in replacements:
            old, new = replacements[defect]
            malformed_create_sql = correct_create_sql.replace(old, new, 1)
            assert malformed_create_sql != correct_create_sql

        with engine.begin() as connection:
            connection.exec_driver_sql(malformed_create_sql)
            for index in V2AnalysisTask.__table__.indexes:
                index_sql = str(CreateIndex(index).compile(dialect=engine.dialect))
                indexed_columns = tuple(column.name for column in index.columns)
                if defect == "ordinary_index" and indexed_columns == ("instrument_id",):
                    connection.exec_driver_sql(
                        "CREATE INDEX ix_v2_analysis_tasks_instrument_id "
                        "ON v2_analysis_tasks (task_type)"
                    )
                elif defect == "partial_index" and indexed_columns == ("instrument_id",):
                    connection.exec_driver_sql(
                        "CREATE INDEX ix_v2_analysis_tasks_instrument_id "
                        "ON v2_analysis_tasks (instrument_id) "
                        "WHERE instrument_id > 0"
                    )
                else:
                    connection.exec_driver_sql(index_sql)
            if defect == "unexpected_index":
                connection.exec_driver_sql(
                    "CREATE INDEX ix_v2_analysis_tasks_rogue "
                    "ON v2_analysis_tasks (task_type)"
                )
            if defect == "unexpected_unique_index":
                connection.exec_driver_sql(
                    "CREATE UNIQUE INDEX ix_v2_analysis_tasks_rogue_unique "
                    "ON v2_analysis_tasks (task_type)"
                )
            connection.exec_driver_sql("PRAGMA user_version = 13")

        with pytest.raises(SchemaVersionError, match="v2_analysis_tasks"):
            run_migrations(engine)

        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA user_version")).scalar_one() == 13
    finally:
        engine.dispose()


def test_mixed_named_and_unnamed_foreign_key_mismatch_raises_schema_version_error(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "mixed-foreign-key-names.db"
    engine = create_database_engine(database)
    try:
        correct_create_sql = str(
            CreateTable(V2AnalysisIteration.__table__).compile(
                dialect=engine.dialect
            )
        )
        foreign_key_sql = (
            "CONSTRAINT fk_v2_analysis_iteration_week_sample "
            "FOREIGN KEY(instrument_id, week_key) "
            "REFERENCES v2_week_samples (instrument_id, week_key)"
        )
        malformed_create_sql = correct_create_sql.replace(
            foreign_key_sql,
            f"{foreign_key_sql} ON DELETE CASCADE",
            1,
        )
        assert malformed_create_sql != correct_create_sql

        with engine.begin() as connection:
            connection.exec_driver_sql(malformed_create_sql)
            for index in V2AnalysisIteration.__table__.indexes:
                connection.exec_driver_sql(
                    str(CreateIndex(index).compile(dialect=engine.dialect))
                )
            connection.exec_driver_sql("PRAGMA user_version = 13")

        with pytest.raises(SchemaVersionError, match="v2_analysis_iterations"):
            run_migrations(engine)

        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA user_version")).scalar_one() == 13
    finally:
        engine.dispose()


def test_upgrade_from_version_thirteen_preserves_existing_v1_tables_and_records(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    engine = create_database_engine(database)
    try:
        with engine.begin() as connection:
            Instrument.__table__.create(connection)
            MarketPrice.__table__.create(connection)
            instrument_id = connection.execute(
                Instrument.__table__.insert().values(
                    code="V1-KEEP",
                    name="V1 Representative",
                    exchange="TEST",
                )
            ).inserted_primary_key[0]
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 1, 2),
                    timeframe="daily",
                    close_price=Decimal("1.25"),
                    source="v1-representative",
                )
            )
            connection.exec_driver_sql("PRAGMA user_version = 13")
        with engine.connect() as connection:
            before_schema = _schema_fingerprint(
                connection, {"instruments", "market_prices"}
            )
            before_instruments = connection.execute(
                text(
                    "SELECT id, code, name, exchange FROM instruments ORDER BY id"
                )
            ).all()
            before_prices = connection.execute(
                text(
                    "SELECT instrument_id, trade_date, timeframe, close_price, source "
                    "FROM market_prices ORDER BY id"
                )
            ).all()

        run_migrations(engine)

        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA user_version")).scalar_one() == SCHEMA_VERSION
            after_schema = _schema_fingerprint(
                connection, {"instruments", "market_prices"}
            )
            assert after_schema == before_schema
            assert connection.execute(
                text("SELECT id, code, name, exchange FROM instruments ORDER BY id")
            ).all() == before_instruments
            assert connection.execute(
                text(
                    "SELECT instrument_id, trade_date, timeframe, close_price, source "
                    "FROM market_prices ORDER BY id"
                )
            ).all() == before_prices
    finally:
        engine.dispose()
    assert V2_TABLE_NAMES <= _table_names(database)
