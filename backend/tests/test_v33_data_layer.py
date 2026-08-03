from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import re
from time import perf_counter

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.migrations import SCHEMA_VERSION, run_migrations
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    Base,
    Instrument,
    MarketPrice,
    V32AppState,
    V33Forecast,
    V33InstrumentRole,
    V33MarketBar,
    V33PointInTimeObservation,
    V33QualityAssessment,
    V33RawIngestion,
    V33SourceRelease,
)
from backend.app.schemas.market import MarketDataRecord, ProviderResult
from backend.app.services.v33_data_service import (
    PointInTimeObservationInput,
    V33DataRepository,
    V33DataService,
)


V33_TABLES = {
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
}


def _database(tmp_path: Path):
    path = tmp_path / "data" / "investment_lab.db"
    initialize_database(path, tmp_path / "config")
    return path, create_database_engine(path), create_session_factory(path)


def _record(day: int, *, volume: Decimal | None) -> MarketDataRecord:
    close = Decimal("1.20") + Decimal(day) / Decimal("1000")
    return MarketDataRecord(
        trade_date=date(2026, 7, day),
        open_price=close - Decimal("0.01"),
        high_price=close + Decimal("0.02"),
        low_price=close - Decimal("0.02"),
        close_price=close,
        volume=volume,
        amount=Decimal("1234567.89"),
        source="TEST_PROVIDER",
    )


def _pit_observation(index: int) -> PointInTimeObservationInput:
    available_at = datetime(2018, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
        days=index
    )
    effective_date = available_at.date()
    return PointInTimeObservationInput(
        dataset="NAV",
        series_code="159941.nav",
        observation_date=effective_date,
        effective_date=effective_date,
        published_at=available_at,
        available_at=available_at,
        cutoff_at=available_at,
        vintage=f"test-release-{index:05d}",
        source="NAV_PERFORMANCE_TEST",
        numeric_value=Decimal("1") + Decimal(index) / Decimal("100000"),
        unit="CNY",
        observation_period="daily",
        source_url="https://example.invalid/v33/nav-performance-test",
        instrument_code="159941",
        payload={"source_row": index},
    )


class _Provider:
    def __init__(self, source: str, records: list[MarketDataRecord]) -> None:
        self.source = source
        self.records = records

    def fetch(self, instrument_code, start_date=None, end_date=None):  # type: ignore[no-untyped-def]
        assert instrument_code == "159941"
        return ProviderResult.success(self.source, self.records)


def test_schema_19_migration_is_idempotent_and_does_not_rewrite_v32_rows(
    tmp_path: Path,
) -> None:
    path, engine, _session_factory = _database(tmp_path)
    marker_time = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    with Session(engine) as session, session.begin():
        session.add(
            V32AppState(
                key="immutable-v32-marker",
                value_json={"generation": "V3.2", "value": "preserve-byte-semantics"},
                updated_at=marker_time,
            )
        )

    with engine.begin() as connection:
        before = connection.execute(
            text(
                "SELECT key, value_json, updated_at FROM v32_app_state "
                "WHERE key = 'immutable-v32-marker'"
            )
        ).one()
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in V33_TABLES:
                table.drop(connection, checkfirst=True)
        connection.exec_driver_sql("PRAGMA user_version = 18")

    statements: list[str] = []

    def capture_statement(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ):  # type: ignore[no-untyped-def]
        statements.append(statement)
        return statement, _parameters

    event.listen(engine, "before_cursor_execute", capture_statement, retval=True)
    assert run_migrations(engine) == SCHEMA_VERSION == 22
    assert run_migrations(engine) == 22
    event.remove(engine, "before_cursor_execute", capture_statement)
    with engine.connect() as connection:
        after = connection.execute(
            text(
                "SELECT key, value_json, updated_at FROM v32_app_state "
                "WHERE key = 'immutable-v32-marker'"
            )
        ).one()
        table_names = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert connection.execute(text("PRAGMA user_version")).scalar_one() == 22
    assert after == before
    assert V33_TABLES <= table_names
    assert not [
        statement
        for statement in statements
        if re.search(
            r"\b(?:INSERT|UPDATE|DELETE|ALTER|DROP)\b[^;]*\bv32_",
            statement,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]
    assert path.exists()


def test_initialize_marks_159941_tradable_and_ndx_benchmark_idempotently(
    tmp_path: Path,
) -> None:
    path, engine, _session_factory = _database(tmp_path)
    initialize_database(path, tmp_path / "config")

    with Session(engine) as session:
        etf = session.scalar(select(Instrument).where(Instrument.code == "159941"))
        ndx = session.scalar(select(Instrument).where(Instrument.code == "NDX"))
        roles = list(
            session.scalars(
                select(V33InstrumentRole)
                .where(V33InstrumentRole.model_market == "159941")
                .order_by(V33InstrumentRole.role)
            )
        )

    assert etf is not None and etf.is_active is True
    assert etf.extra_data["benchmark"] == "NDX"
    assert ndx is not None and ndx.extra_data["role"] == "benchmark"
    assert [(role.instrument_id, role.role) for role in roles] == [
        (ndx.id, "benchmark"),
        (etf.id, "tradable"),
    ]
    assert roles[1].benchmark_instrument_id == ndx.id


def test_refresh_159941_rejects_missing_volume_then_uses_next_real_provider(
    tmp_path: Path,
) -> None:
    _path, engine, session_factory = _database(tmp_path)
    service = V33DataService(session_factory)
    good_records = [
        _record(28, volume=Decimal("900000")),
        _record(29, volume=Decimal("1000000")),
    ]
    result = service.refresh_159941(
        start_date=date(2026, 7, 28),
        end_date=date(2026, 7, 29),
        providers=(
            _Provider("AKSHARE", [_record(28, volume=None)]),
            _Provider("TUSHARE", good_records),
        ),
    )

    assert result.status == "success"
    assert result.source == "TUSHARE"
    assert result.records_written == 2
    retry = service.ingest_market_bars(
        "159941",
        good_records,
        source="TUSHARE",
        start_date=date(2026, 7, 28),
        end_date=date(2026, 7, 29),
        require_volume=True,
    )
    assert retry.status == "success"
    assert retry.records_written == 0
    with Session(engine) as session:
        attempts = list(
            session.scalars(
                select(V33RawIngestion).order_by(V33RawIngestion.created_at)
            )
        )
        assessments = list(session.scalars(select(V33QualityAssessment)))
        bars = list(
            session.scalars(
                select(V33MarketBar).order_by(V33MarketBar.trade_date)
            )
        )
        mirrored = list(
            session.scalars(
                select(MarketPrice)
                .join(Instrument)
                .where(
                    Instrument.code == "159941",
                    MarketPrice.timeframe == "daily",
                )
                .order_by(MarketPrice.trade_date)
            )
        )

    assert [attempt.quality_status for attempt in attempts] == [
        "rejected",
        "accepted",
        "accepted",
    ]
    assert any("MISSING_VOLUME" in issue for issue in assessments[0].issues_json)
    assert len(bars) == len(mirrored) == 2
    assert all(bar.available_at <= bar.cutoff_at for bar in bars)
    assert all(bar.retrieved_at >= bar.available_at for bar in bars)
    assert all(row.source == "V33:TUSHARE" for row in mirrored)


def test_point_in_time_repository_blocks_future_vintage_and_never_fills_missing(
    tmp_path: Path,
) -> None:
    _path, _engine, session_factory = _database(tmp_path)
    service = V33DataService(session_factory)
    effective = date(2026, 1, 1)
    first_available = datetime(2026, 1, 10, tzinfo=timezone.utc)
    revised_available = datetime(2026, 2, 10, tzinfo=timezone.utc)

    first = PointInTimeObservationInput(
        dataset="VALUATION",
        series_code="399006.PE",
        observation_date=effective,
        effective_date=effective,
        published_at=first_available,
        available_at=first_available,
        cutoff_at=first_available,
        vintage="initial",
        source="OFFICIAL_TEST",
        numeric_value=Decimal("22.50"),
        unit="ratio",
    )
    revision = PointInTimeObservationInput(
        dataset="VALUATION",
        series_code="399006.PE",
        observation_date=effective,
        effective_date=effective,
        published_at=revised_available,
        available_at=revised_available,
        cutoff_at=revised_available,
        vintage="revision-1",
        source="OFFICIAL_TEST",
        numeric_value=Decimal("23.00"),
        unit="ratio",
    )
    row_ids = service.ingest_point_in_time_observations([first, revision])
    assert service.ingest_point_in_time_observations([first]) == [row_ids[0]]

    early = service.repository.observations_as_of(
        ["399006.PE", "399006.PB"],
        datetime(2026, 1, 31, tzinfo=timezone.utc),
    )
    late = service.repository.observations_as_of(
        ["399006.PE", "399006.PB"],
        datetime(2026, 2, 28, tzinfo=timezone.utc),
    )
    assert [(row.series_code, row.numeric_value) for row in early] == [
        ("399006.PE", Decimal("22.50"))
    ]
    assert [(row.series_code, row.numeric_value) for row in late] == [
        ("399006.PE", Decimal("23.00"))
    ]
    assert all(row.series_code != "399006.PB" for row in (*early, *late))

    with pytest.raises(ValueError, match="must contain a real value"):
        service.store_point_in_time_observation(
            PointInTimeObservationInput(
                dataset="VALUATION",
                series_code="399006.PB",
                observation_date=effective,
                effective_date=effective,
                published_at=first_available,
                available_at=first_available,
                cutoff_at=first_available,
                vintage="missing",
                source="OFFICIAL_TEST",
            )
        )


def test_bulk_pit_ingestion_is_atomic_idempotent_and_fast_for_1200_rows(
    tmp_path: Path,
) -> None:
    _path, engine, _session_factory = _database(tmp_path)
    session_calls = 0
    commit_calls = 0

    def counted_session_factory() -> Session:
        nonlocal session_calls
        session_calls += 1
        return Session(engine, expire_on_commit=False)

    def count_commit(_connection) -> None:  # type: ignore[no-untyped-def]
        nonlocal commit_calls
        commit_calls += 1

    event.listen(engine, "commit", count_commit)
    observations = [_pit_observation(index) for index in range(1200)]
    batch_with_duplicate = [*observations, observations[777]]
    service = V33DataService(counted_session_factory)
    try:
        started_at = perf_counter()
        row_ids = service.ingest_point_in_time_observations(batch_with_duplicate)
        first_elapsed = perf_counter() - started_at

        assert len(row_ids) == 1201
        assert len(set(row_ids)) == 1200
        assert row_ids[-1] == row_ids[777]
        assert session_calls == 1
        assert commit_calls == 1
        # This bound intentionally leaves ample room for a slow CI disk while
        # rejecting a return to thousands of fsync-heavy SQLite transactions.
        assert first_elapsed < 10.0

        started_at = perf_counter()
        retry_ids = service.ingest_point_in_time_observations(batch_with_duplicate)
        retry_elapsed = perf_counter() - started_at

        assert retry_ids == row_ids
        assert session_calls == 2
        assert commit_calls == 2
        assert retry_elapsed < 5.0
    finally:
        event.remove(engine, "commit", count_commit)

    with Session(engine) as session:
        raw_rows = list(session.scalars(select(V33RawIngestion)))
        release_rows = list(session.scalars(select(V33SourceRelease)))
        pit_rows = list(session.scalars(select(V33PointInTimeObservation)))
        quality_rows = list(session.scalars(select(V33QualityAssessment)))

    assert len(raw_rows) == len(release_rows) == len(pit_rows) == len(quality_rows) == 1200
    raw_by_ingestion = {row.id: row for row in raw_rows}
    release_by_ingestion = {row.ingestion_id: row for row in release_rows}
    quality_by_ingestion = {row.ingestion_id: row for row in quality_rows}
    assert set(raw_by_ingestion) == set(release_by_ingestion) == set(quality_by_ingestion)
    assert {row.ingestion_id for row in pit_rows} == set(raw_by_ingestion)
    for row in pit_rows:
        raw = raw_by_ingestion[row.ingestion_id]
        release = release_by_ingestion[row.ingestion_id]
        quality = quality_by_ingestion[row.ingestion_id]
        assert row.source_release_id == release.id
        assert row.raw_payload_hash == raw.raw_payload_hash == release.raw_payload_hash
        assert quality.assessed_payload_hash == row.raw_payload_hash
        assert row.source == raw.provider == release.source == "NAV_PERFORMANCE_TEST"
        assert row.source_url == raw.source_url == release.source_url


def test_bulk_pit_conflict_detects_late_item_and_rolls_back_1000_new_rows(
    tmp_path: Path,
) -> None:
    _path, engine, _session_factory = _database(tmp_path)
    session_calls = 0

    def counted_session_factory() -> Session:
        nonlocal session_calls
        session_calls += 1
        return Session(engine, expire_on_commit=False)

    service = V33DataService(counted_session_factory)
    baseline = _pit_observation(5000)
    baseline_id = service.store_point_in_time_observation(baseline)
    assert service.store_point_in_time_observation(baseline) == baseline_id

    conflicting_revision = replace(
        baseline,
        numeric_value=baseline.numeric_value + Decimal("0.01"),
        payload={"source_row": 5000, "conflicting_payload": True},
    )
    with pytest.raises(ValueError, match="revision identity already exists"):
        service.store_point_in_time_observation(conflicting_revision)

    batch = [
        *(_pit_observation(index) for index in range(1000)),
        conflicting_revision,
    ]
    with pytest.raises(ValueError, match="revision identity already exists"):
        service.ingest_point_in_time_observations(batch)

    # One session each for the initial write, idempotent retry, single-item
    # conflict, and the rolled-back 1001-row conflicting batch.
    assert session_calls == 4
    with Session(engine) as session:
        assert len(list(session.scalars(select(V33RawIngestion)))) == 1
        assert len(list(session.scalars(select(V33SourceRelease)))) == 1
        assert len(list(session.scalars(select(V33PointInTimeObservation)))) == 1
        assert len(list(session.scalars(select(V33QualityAssessment)))) == 1


def test_market_bar_revision_is_not_visible_before_its_retrieval_vintage(
    tmp_path: Path,
) -> None:
    _path, _engine, session_factory = _database(tmp_path)
    service = V33DataService(session_factory)
    original = _record(28, volume=Decimal("1000"))
    revision = original.model_copy(
        update={"close_price": original.close_price + Decimal("0.001")}
    )
    first_seen = datetime(2026, 7, 31, 0, 0, tzinfo=timezone.utc)
    revision_seen = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
    service.ingest_market_bars(
        "159941",
        [original],
        source="AKSHARE",
        fetched_at=first_seen,
        require_volume=True,
    )
    service.ingest_market_bars(
        "159941",
        [revision],
        source="AKSHARE",
        fetched_at=revision_seen,
        require_volume=True,
    )

    before_revision = service.repository.market_bars_as_of(
        "159941",
        datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
    )
    after_revision = service.repository.market_bars_as_of(
        "159941",
        datetime(2026, 8, 1, 1, 0, tzinfo=timezone.utc),
    )
    assert before_revision[-1].close_price == original.close_price
    assert after_revision[-1].close_price == revision.close_price
    assert after_revision[-1].available_at == revision_seen
    assert after_revision[-1].vintage.startswith("revision:")


def test_ndx_alignment_cannot_use_same_natural_date_us_close(tmp_path: Path) -> None:
    _path, engine, session_factory = _database(tmp_path)
    with Session(engine) as session, session.begin():
        ndx_id = session.scalar(select(Instrument.id).where(Instrument.code == "NDX"))
        assert ndx_id is not None
        for trade_date, close in (
            (date(2026, 7, 30), Decimal("23000")),
            (date(2026, 7, 31), Decimal("23100")),
        ):
            session.add(
                MarketPrice(
                    instrument_id=ndx_id,
                    trade_date=trade_date,
                    timeframe="daily",
                    open_price=close,
                    high_price=close,
                    low_price=close,
                    close_price=close,
                    volume=None,
                    volume_multiplier=1,
                    source="DIRECT_NDX_TEST",
                    volume_source="VOLUME_UNAVAILABLE:DIRECT_INDEX",
                )
            )

    alignments = V33DataRepository(session_factory).align_ndx_benchmark(
        [date(2026, 7, 31), date(2026, 8, 3)]
    )
    assert [(item.target_trade_date, item.benchmark_trade_date) for item in alignments] == [
        (date(2026, 7, 31), date(2026, 7, 30)),
        (date(2026, 8, 3), date(2026, 7, 31)),
    ]
    assert all(item.source == "LEGACY_MARKET_PRICE_READ_ONLY" for item in alignments)


def test_pending_v33_forecast_and_missing_observation_have_no_numeric_defaults() -> None:
    for column_name in (
        "confidence",
        "up_probability",
        "sideways_probability",
        "down_probability",
        "expected_max_drawdown",
    ):
        column = V33Forecast.__table__.c[column_name]
        assert column.nullable is True
        assert column.default is None
        assert column.server_default is None
    for column_name in (
        "p10_path_json",
        "p50_path_json",
        "p90_path_json",
        "expected_path_json",
    ):
        assert V33Forecast.__table__.c[column_name].nullable is True
    numeric = V33PointInTimeObservation.__table__.c.numeric_value
    assert numeric.nullable is True
    assert numeric.default is None
    assert numeric.server_default is None


def test_v33_market_bar_rejects_invalid_ohlc_in_database(tmp_path: Path) -> None:
    _path, engine, session_factory = _database(tmp_path)
    service = V33DataService(session_factory)
    accepted = service.persist_market_result(
        "159941",
        ProviderResult.success("AKSHARE", [_record(28, volume=Decimal("100"))]),
    )
    assert accepted.status == "success"
    with Session(engine) as session:
        row = session.scalar(select(V33MarketBar))
        assert row is not None
        values = dict(
            instrument_id=row.instrument_id,
            ingestion_id=row.ingestion_id,
            source_release_id=row.source_release_id,
            trade_date=date(2026, 7, 30),
            timeframe="daily",
            open_price=Decimal("1.0"),
            high_price=Decimal("0.9"),
            low_price=Decimal("0.8"),
            close_price=Decimal("1.0"),
            volume=Decimal("100"),
            volume_multiplier=1,
            effective_date=date(2026, 7, 30),
            published_at=row.published_at,
            available_at=row.available_at,
            retrieved_at=row.retrieved_at,
            cutoff_at=row.cutoff_at,
            vintage="invalid",
            source="AKSHARE",
            raw_payload_hash="f" * 64,
            quality_status="accepted",
            payload_json={},
            created_at=row.created_at,
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                V33MarketBar.__table__.insert().values(
                    **values,
                )
            )


def test_metadata_contains_all_v33_tables() -> None:
    assert V33_TABLES <= set(Base.metadata.tables)
