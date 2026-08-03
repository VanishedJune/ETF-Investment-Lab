from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    Instrument,
    MarketPrice,
    ValuationRecord,
    V33AnalysisRun,
    V33FeatureSnapshot,
    V33Forecast,
    V33ModelEvaluation,
    V33ModelVersion,
    V33OptimizerState,
    V33TrainingCheckpoint,
    V33TrainingIteration,
    V33TrainingRun,
)
from backend.app.services.v33_runtime_service import (
    ANALYSIS_FEATURE_VERSION,
    FEATURE_VERSION,
    METHODOLOGY_VERSION,
    SQLiteV33TrainingRepository,
    TRAINING_FEATURE_VERSION,
    V33RuntimeService,
    _china_close,
)
from backend.app.services.v33_feature_service import PointInTimeObservation, PriceBar
from backend.app.services.v33_training_service import state_from_payload


def test_revision_three_model_identity_isolated_from_invalid_revision_two() -> None:
    assert FEATURE_VERSION == "V3.3-20W-FULL100-DCT12-3"
    assert TRAINING_FEATURE_VERSION == f"{FEATURE_VERSION}-SAMPLED"
    assert ANALYSIS_FEATURE_VERSION == f"{FEATURE_VERSION}-LIVE"
    assert METHODOLOGY_VERSION == "V3.3-PROGRESSIVE-PURGED-20W-3"


class _WeekdayCalendar:
    def sessions(self, _market: str, start: date, end: date) -> tuple[date, ...]:
        output: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                output.append(current)
            current += timedelta(days=1)
        return tuple(output)


def _database(tmp_path: Path):
    path = tmp_path / "data" / "investment_lab.db"
    initialize_database(path, tmp_path / "config")
    return path, create_session_factory(path)


def _seed_market(sessions, *, weeks: int = 150) -> date:  # type: ignore[no-untyped-def]
    monday = date(2023, 1, 2)
    with sessions() as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "399006"))
        assert instrument is not None
        for week in range(weeks):
            week_start = monday + timedelta(weeks=week)
            for weekday in range(5):
                trade_date = week_start + timedelta(days=weekday)
                close = (
                    Decimal("2000")
                    + Decimal(week * 4 + weekday)
                    + Decimal(week % 9) * Decimal("0.17")
                )
                session.add(
                    MarketPrice(
                        instrument_id=instrument.id,
                        trade_date=trade_date,
                        timeframe="daily",
                        open_price=close - Decimal("3"),
                        high_price=close + Decimal("10"),
                        low_price=close - Decimal("9"),
                        close_price=close,
                        adjusted_close_price=None,
                        volume=Decimal("100000000") + Decimal(week * 100000 + weekday * 1000),
                        volume_multiplier=1,
                        turnover=Decimal("5000000000") + Decimal(week * 1000000),
                        source="TEST_DIRECT_INDEX",
                        volume_source="TEST_DIRECT_INDEX",
                    )
                )
            valuation_date = week_start + timedelta(days=4)
            session.add(
                ValuationRecord(
                    instrument_id=instrument.id,
                    valuation_date=valuation_date,
                    pe_ratio=Decimal("25") + Decimal(week % 20) / Decimal("10"),
                    pb_ratio=Decimal("4") + Decimal(week % 10) / Decimal("20"),
                    valuation_percentile=None,
                    raw_values={"source": "TEST_PIT_VALUATION", "quality_status": "accepted"},
                )
            )
    return monday + timedelta(weeks=weeks - 1, days=4)


def _counts(sessions) -> tuple[int, int, int, int, int, int, int]:  # type: ignore[no-untyped-def]
    with sessions() as session:
        return (
            int(session.scalar(select(func.count(V33TrainingIteration.id))) or 0),
            int(session.scalar(select(func.count(V33ModelVersion.id))) or 0),
            int(session.scalar(select(func.count(V33OptimizerState.id))) or 0),
            int(session.scalar(select(func.count(V33TrainingCheckpoint.id))) or 0),
            int(session.scalar(select(func.count(V33Forecast.id))) or 0),
            int(session.scalar(select(func.count(V33ModelEvaluation.id))) or 0),
            int(session.scalar(select(func.count(V33FeatureSnapshot.id))) or 0),
        )


def _live_weekday_bars() -> list[PriceBar]:
    rows: list[PriceBar] = []
    current = date(2025, 6, 9)
    while current <= date(2026, 7, 31):
        if current.weekday() < 5:
            index = len(rows)
            close = 100.0 + index * 0.05
            rows.append(
                PriceBar(
                    trade_date=current,
                    open=close - 0.1,
                    high=close + 0.2,
                    low=close - 0.2,
                    close=close,
                    volume=1_000_000.0 + index,
                    turnover=10_000_000.0 + index,
                    source="TEST_LIVE_PRICE",
                )
            )
        current += timedelta(days=1)
    return rows


def test_live_analysis_uses_analysis_clock_and_keeps_same_price_day_vintages(
    tmp_path: Path,
) -> None:
    _path, sessions = _database(tmp_path)
    bars = _live_weekday_bars()
    price_date = bars[-1].trade_date
    first_available = datetime(2026, 8, 1, 1, 0, tzinfo=timezone.utc)
    second_available = datetime(2026, 8, 1, 3, 0, tzinfo=timezone.utc)
    observations = [
        PointInTimeObservation(
            market="399006",
            series="pe",
            value=31.0,
            effective_date=price_date,
            published_at=first_available,
            available_at=first_available,
            source="TEST_WEEKEND_PIT",
            vintage="v1",
        ),
        PointInTimeObservation(
            market="399006",
            series="pe",
            value=32.0,
            effective_date=price_date,
            published_at=second_available,
            available_at=second_available,
            source="TEST_WEEKEND_PIT",
            vintage="v2",
        ),
    ]
    clock = {"now": datetime(2026, 8, 1, 2, 0, tzinfo=timezone.utc)}
    runtime = V33RuntimeService(
        sessions,
        calendar=_WeekdayCalendar(),
        now_provider=lambda: clock["now"],
    )
    runtime._load_source_data = lambda _market: (bars, [], observations)  # type: ignore[method-assign]
    try:
        first, first_id = runtime._build_current_snapshot("399006")
        assert first.weekly["pe"] == 31.0
        assert first.provenance["price_as_of"] == price_date.isoformat()
        assert first.provenance["analysis_as_of"] == clock["now"].isoformat()

        clock["now"] = datetime(2026, 8, 1, 4, 0, tzinfo=timezone.utc)
        second, second_id = runtime._build_current_snapshot("399006")
        assert second.weekly["pe"] == 32.0
        assert second_id != first_id

        historical = runtime.feature_service.build_snapshot(
            "399006",
            bars,
            price_date,
            observations=observations,
            weekly_cutoff_date=price_date,
            cutoff_at=_china_close(price_date),
        )
        assert historical.weekly["pe"] is None
        assert historical.missing_masks["missing_pe"] == 1

        with sessions() as session:
            live_rows = list(
                session.scalars(
                    select(V33FeatureSnapshot).where(
                        V33FeatureSnapshot.model_market == "399006",
                        V33FeatureSnapshot.cutoff_date == price_date,
                        V33FeatureSnapshot.feature_version.like("%-LIVE-%"),
                    )
                )
            )
        assert len(live_rows) == 2
        assert len({row.feature_version for row in live_rows}) == 2
        assert all(row.leakage_audit_json["price_as_of"] == price_date.isoformat() for row in live_rows)
    finally:
        runtime.shutdown()


def test_sqlite_bootstrap_restart_analysis_and_incremental_are_auditable(
    tmp_path: Path,
) -> None:
    path, sessions = _database(tmp_path)
    latest = _seed_market(sessions)
    clock = {"now": datetime(2026, 8, 1, 1, 0, tzinfo=timezone.utc)}
    auxiliary_calls: list[str] = []

    def refresh_auxiliary() -> dict[str, object]:
        auxiliary_calls.append("refresh")
        return {
            "status": "partial",
            "missing_by_market": {"399006": ["399006.real_rate"]},
            "sources": [{"source": "TEST", "status": "partial"}],
        }

    runtime = V33RuntimeService(
        sessions,
        calendar=_WeekdayCalendar(),
        analysis_days=140,
        warmup_weeks=70,
        now_provider=lambda: clock["now"],
        auxiliary_refresh=refresh_auxiliary,
    )
    try:
        result = runtime.bootstrap_sync("399006")
        assert result["status"] == "completed"
        assert result["audited_iteration_count"] == result["expected_iteration_count"]
        assert result["pending_count"] == 20
        assert result["auxiliary_refresh"]["status"] == "partial"
        assert auxiliary_calls == ["refresh"]
        assert len(result["sampled_dates"]) == result["audited_iteration_count"]
        curve = runtime.curve("399006")
        assert curve["iteration_count"] == result["audited_iteration_count"]
        assert curve["pending_count"] == 20
        status = runtime.training_status(result["run_id"])
        assert status["status"] == "completed"
        assert status["created_iteration_count"] == result["audited_iteration_count"]
        assert status["result"]["auxiliary_refresh"]["missing_by_market"] == {
            "399006": ["399006.real_rate"]
        }
    finally:
        runtime.shutdown()

    counts_after_training = _counts(sessions)
    assert counts_after_training[0] == result["audited_iteration_count"]
    assert counts_after_training[1] == result["audited_iteration_count"]
    assert counts_after_training[2] == 1
    assert counts_after_training[3] == 1
    assert counts_after_training[4] == result["audited_iteration_count"]
    assert counts_after_training[5] == result["audited_iteration_count"]

    with sessions() as session:
        root_run = session.get(V33TrainingRun, result["run_id"])
        assert root_run is not None
        anchor = root_run.result_json["warmup_anchor"]
        root_state = state_from_payload(anchor["state_payload"])
        first_iteration = session.scalar(
            select(V33TrainingIteration)
            .where(V33TrainingIteration.model_market == "399006")
            .order_by(V33TrainingIteration.iteration_number)
        )
        assert first_iteration is not None
        first_model = session.scalar(
            select(V33ModelVersion).where(
                V33ModelVersion.model_market == "399006",
                V33ModelVersion.version == first_iteration.champion_model_version,
            )
        )
        assert first_model is not None
        assert root_state.iteration_number == 0
        assert root_state.parent_state_hash is None
        assert anchor["formal_iteration"] is False
        assert anchor["state_hash"] == first_iteration.audit_json["parent_state_hash"]
        assert anchor["state_hash"] == first_model.parameters_json["parent_state_hash"]
        assert anchor["real_matured_sample_count"] == root_state.training_sample_count
        assert anchor["build_snapshot_count"] == len(anchor["build_snapshot_manifest"])
        assert first_iteration.parent_iteration_number is None
        assert first_model.parent_version is None
        assert counts_after_training[0] == counts_after_training[1]

    # A brand-new runtime/repository rehydrates the exact checkpoint bytes.
    repository = SQLiteV33TrainingRepository(sessions)
    restored = repository.latest_state("399006")
    assert restored is not None
    assert restored.iteration_number == result["audited_iteration_count"]
    assert restored.state_hash == runtime.champion("399006")["state_hash"]

    restarted = V33RuntimeService(
        sessions,
        calendar=_WeekdayCalendar(),
        analysis_days=140,
        warmup_weeks=70,
        now_provider=lambda: clock["now"],
    )
    try:
        training_identity_before_analysis = _counts(sessions)[:4]
        training_fingerprint_before_analysis = restarted._training_fingerprint(
            "399006"
        )
        curve_before_analysis = restarted.curve("399006")
        analysis = restarted.run_analysis("399006")
        training_identity_after_analysis = _counts(sessions)[:4]
        training_fingerprint_after_analysis = restarted._training_fingerprint(
            "399006"
        )
        curve_after_analysis = restarted.curve("399006")
        assert analysis["training_mutated"] is False
        assert analysis["training_identity_unchanged"] is True
        assert analysis["training_identity_before"] == analysis["training_identity_after"]
        assert analysis["training_fingerprint_unchanged"] is True
        assert (
            analysis["training_fingerprint_before"]
            == analysis["training_fingerprint_after"]
            == training_fingerprint_before_analysis
            == training_fingerprint_after_analysis
        )
        assert set(analysis["training_fingerprint_before"]) == {
            "iteration_registry_hash",
            "model_registry_hash",
            "optimizer_hash",
            "champion_hash",
            "champion_artifact_hash",
        }
        assert set(analysis["training_identity_before"]) == {
            "iteration_count",
            "model_count",
            "optimizer_count",
            "optimizer_state_hash",
        }
        assert len(analysis["path"]["p10"]) == 20
        assert analysis["position"]["target"] % 5 == 0
        assert training_identity_after_analysis == training_identity_before_analysis
        assert curve_after_analysis["iteration_count"] == curve_before_analysis["iteration_count"]
        assert curve_after_analysis["pending_count"] == curve_before_analysis["pending_count"]
        assert restarted.latest_analysis("399006")["analysis_hash"] == analysis["analysis_hash"]
        with sessions() as session:
            assert int(session.scalar(select(func.count(V33AnalysisRun.id))) or 0) == 1
            assert int(
                session.scalar(
                    select(func.count(V33Forecast.id)).where(
                        V33Forecast.maturity_status == "live"
                    )
                )
                or 0
            ) == 1
            assert int(
                session.scalar(
                    select(func.count(V33Forecast.id)).where(
                        V33Forecast.maturity_status == "pending"
                    )
                )
                or 0
            ) == curve_before_analysis["pending_count"]
            assert int(
                session.scalar(
                    select(func.count(V33TrainingIteration.id)).where(
                        V33TrainingIteration.maturity_status == "pending"
                    )
                )
                or 0
            ) == curve_before_analysis["pending_count"]
            saved_analysis = session.scalar(select(V33AnalysisRun))
            assert saved_analysis is not None
            assert saved_analysis.result_json["training_identity_unchanged"] is True
            assert (
                saved_analysis.result_json["training_identity_before"]
                == saved_analysis.result_json["training_identity_after"]
            )

        duplicate = restarted.bootstrap_sync("399006")
        assert duplicate["status"] == "not_due"
        assert _counts(sessions)[:4] == training_identity_after_analysis

        # Add several complete weeks after eight days.  Incremental training
        # must consume only one (the latest) and report the skipped gap.
        with sessions() as session, session.begin():
            instrument = session.scalar(select(Instrument).where(Instrument.code == "399006"))
            assert instrument is not None
            next_monday = latest + timedelta(days=3)
            for week in range(3):
                for weekday in range(5):
                    trade_date = next_monday + timedelta(weeks=week, days=weekday)
                    close = Decimal("2700") + Decimal(week * 5 + weekday)
                    session.add(
                        MarketPrice(
                            instrument_id=instrument.id,
                            trade_date=trade_date,
                            timeframe="daily",
                            open_price=close - 2,
                            high_price=close + 8,
                            low_price=close - 7,
                            close_price=close,
                            volume=Decimal("130000000") + Decimal(week * 100000),
                            volume_multiplier=1,
                            turnover=Decimal("6000000000"),
                            source="TEST_DIRECT_INDEX",
                            volume_source="TEST_DIRECT_INDEX",
                        )
                    )
        clock["now"] += timedelta(days=8)
        before_incremental = _counts(sessions)[0]
        incremental = restarted.incremental_sync("399006")
        assert incremental["status"] == "completed"
        assert incremental["created_iteration_count"] == 1
        assert incremental["skipped_complete_weeks"] == 2
        assert _counts(sessions)[0] == before_incremental + 1
        # Three calendar weeks matured while the explicit at-most-one rule
        # adds only one new forecast, so pending naturally drops by two.
        assert restarted.curve("399006")["pending_count"] == 18
    finally:
        restarted.shutdown()

    assert path.exists()
