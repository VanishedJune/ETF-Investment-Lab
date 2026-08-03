from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    Instrument,
    V33PointInTimeObservation,
    V33SourceRelease,
)
from backend.app.services.v34_feature_service import V34FeatureError
from backend.app.services.v341_feature_service import (
    V341FeatureService,
    build_feature_manifest,
)
from backend.tests.test_v341_runtime import _snapshot


def _sessions(tmp_path: Path):
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    return create_session_factory(database)


def _add_observation(session, *, series: str, available: datetime, value: str) -> None:
    instrument_id = session.scalar(select(Instrument.id).where(Instrument.code == "399006"))
    digest = (series.replace(".", "") + available.isoformat()).encode().hex()[:64].ljust(64, "0")
    release_id = f"REL-{series}-{available.date()}"
    session.add(
        V33SourceRelease(
            id=release_id,
            ingestion_id=None,
            instrument_id=instrument_id,
            dataset="VALUATION",
            series_code=series,
            observation_period_start=available.date(),
            observation_period_end=available.date(),
            effective_date=date(2025, 1, 2),
            published_at=available,
            available_at=available,
            vintage="test",
            source="TEST",
            source_url="https://example.invalid",
            raw_payload_hash=digest,
            quality_status="accepted",
            payload_json={},
            created_at=available,
        )
    )
    session.flush()
    session.add(
        V33PointInTimeObservation(
            instrument_id=instrument_id,
            ingestion_id=None,
            source_release_id=release_id,
            dataset="VALUATION",
            series_code=series,
            observation_date=date(2025, 1, 2),
            observation_period="daily",
            effective_date=date(2025, 1, 2),
            published_at=available,
            available_at=available,
            retrieved_at=available,
            cutoff_at=available,
            vintage="test",
            numeric_value=Decimal(value),
            text_value=None,
            unit="ratio",
            source="TEST",
            source_url="https://example.invalid",
            raw_payload_hash=digest,
            quality_status="accepted",
            payload_json={},
            created_at=available,
        )
    )


def test_pit_requires_available_at_and_returns_real_pe_pb(tmp_path: Path) -> None:
    sessions = _sessions(tmp_path)
    with sessions() as session, session.begin():
        _add_observation(
            session,
            series="399006.pe",
            available=datetime(2025, 1, 3, 8, tzinfo=timezone.utc),
            value="31.25",
        )
        _add_observation(
            session,
            series="399006.pb",
            available=datetime(2025, 1, 6, 8, tzinfo=timezone.utc),
            value="4.75",
        )
    with sessions() as session:
        first = V341FeatureService._pit_observations(session, "399006", date(2025, 1, 3))
        later = V341FeatureService._pit_observations(session, "399006", date(2025, 1, 10))
    assert {row.series for row in first} == {"pe"}
    assert {row.series: row.value for row in later} == {"pe": 31.25, "pb": 4.75}


def test_old_snapshot_hash_does_not_change_when_future_data_exists() -> None:
    snapshot = _snapshot(12)
    manifest = build_feature_manifest(snapshot)
    before = V341FeatureService.audit_payload(snapshot, manifest)["snapshot_hash"]
    future_snapshot = _snapshot(13)
    assert future_snapshot.cutoff_date > snapshot.cutoff_date
    after = V341FeatureService.audit_payload(snapshot, manifest)["snapshot_hash"]
    assert before == after


def test_159941_extended_manifest_rejects_non_etf_fields() -> None:
    snapshot = _snapshot(12)
    contaminated = replace(
        snapshot,
        market="159941",
        weekly={**snapshot.weekly, "weekly_ndx_return_20": 0.12},
    )
    with pytest.raises(V34FeatureError, match="forbidden non-ETF fields"):
        build_feature_manifest(contaminated)
