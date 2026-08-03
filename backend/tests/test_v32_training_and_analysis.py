from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    V32AnalysisRun,
    V32DataSnapshot,
    V32OptimizerState,
    V32TrainingIteration,
    V32TrainingRun,
)
from backend.app.services.v32_analysis_service import V32AnalysisService
from backend.app.services.v32_training_service import (
    ForecastResult,
    TrainingPoint,
    V32TrainingService,
    _initial_parameters,
)


def _point(index: int, week: str = "2026-W31") -> TrainingPoint:
    weekly, daily = _initial_parameters("399006")
    del weekly, daily
    return TrainingPoint(
        global_index=index,
        week_key=week,
        cutoff_date=date(2026, 7, 31),
        close=2500.0,
        weekly={
            "return_4w": 0.02,
            "return_8w": 0.03,
            "return_13w": 0.04,
            "volatility_13w": 0.25,
            "ma20_distance": 0.01,
            "macd_spread": 0.001,
            "valuation_percentile": 50.0,
            "drawdown_60w": -0.12,
        },
        daily={
            "return_5d": 0.01,
            "return_20d": 0.02,
            "rsi": 55.0,
            "ma20_distance": 0.01,
            "macd_spread": 0.001,
            "volatility_20d": 0.22,
        },
        future_path=tuple(0.005 * step for step in range(1, 14)),
        price_source="DIRECT_INDEX_TEST",
        volume_source="DAILY_SUM_TEST",
        daily_as_of=date(2026, 7, 31),
        valuation_as_of=date(2026, 7, 31),
    )


def _forecast() -> ForecastResult:
    path = tuple(0.006 * step for step in range(1, 14))
    return ForecastResult(
        p10=tuple(value - 0.03 for value in path),
        p50=path,
        p90=tuple(value + 0.03 for value in path),
        expected=path,
        up_probability=70.0,
        sideways_probability=20.0,
        down_probability=10.0,
        direction_threshold=0.04,
        direction="up",
        confidence=70.0,
        base_target_position=80,
        expected_max_drawdown=-6.0,
        high_week_range="第10—13周",
        low_week_range="第1—3周",
        scenario_count=52,
    )


class _WeekdayCalendar:
    def sessions(self, _market: str, start: date, end: date) -> list[date]:
        days: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)
        return days


def _database(tmp_path: Path):
    path = tmp_path / "data" / "investment_lab.db"
    initialize_database(path, tmp_path / "config")
    return create_session_factory(path)


def test_forecast_only_uses_scenarios_mature_at_the_cutoff(tmp_path: Path) -> None:
    service = V32TrainingService(_database(tmp_path), calendar=_WeekdayCalendar())
    points = []
    for index in range(90):
        point = _point(index, f"2025-W{index + 1:02d}")
        points.append(point)
    weekly, daily = _initial_parameters("399006")
    try:
        result = service._forecast(points, 60, weekly, daily)
    finally:
        service.shutdown()

    # At index 60, only indices 0..47 have a complete 13-week outcome that
    # would have been visible at that cutoff. Future rows cannot enter.
    assert result.scenario_count == 48


def test_same_complete_week_is_idempotent(tmp_path: Path) -> None:
    sessions = _database(tmp_path)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    weekly, daily = _initial_parameters("399006")
    with sessions() as session, session.begin():
        session.add(
            V32OptimizerState(
                market="399006",
                iteration_number=511,
                last_training_week_key="2026-W31",
                champion_weekly_version="CYB_WEEKLY_V3.2.3",
                champion_daily_version="CYB_DAILY_CORRECTOR_V3.2.3",
                weekly_parameters_json=weekly,
                daily_parameters_json=daily,
                optimizer_memory_json={},
                last_successful_training_at=now - timedelta(days=8),
                next_training_eligible_at=now - timedelta(days=1),
                state_hash="test",
            )
        )
    service = V32TrainingService(sessions, now_provider=lambda: now, calendar=_WeekdayCalendar())
    service._load_points = lambda _market: ([_point(600, "2026-W31")], 0)  # type: ignore[method-assign]
    try:
        result = service.train_incremental("399006")
    finally:
        service.shutdown()
    assert result["status"] == "not_due"
    assert result["reason"] == "没有新的完整交易周"


def test_incremental_matures_saved_forecast_without_adding_iteration(tmp_path: Path) -> None:
    sessions = _database(tmp_path)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    weekly, daily = _initial_parameters("399006")
    saved_forecast = _forecast()
    mature_point = _point(587, "2026-W18")
    pending_points = [
        replace(_point(588 + offset, f"2026-W{19 + offset:02d}"), future_path=None)
        for offset in range(13)
    ]
    expected = V32TrainingService._evaluate(mature_point, saved_forecast)
    assert expected is not None

    with sessions() as session, session.begin():
        session.add(
            V32TrainingRun(
                id="V32-TEST-MATURITY",
                market="399006",
                run_type="bootstrap",
                status="completed",
                current_stage="completed",
                requested_through_week="2026-W31",
                created_iteration_count=523,
            )
        )
        session.add(
            V32DataSnapshot(
                id="V32-DS-TEST-MATURITY",
                market="399006",
                cutoff_date=mature_point.cutoff_date,
                daily_data_as_of=mature_point.daily_as_of,
                weekly_data_as_of=mature_point.cutoff_date,
                valuation_data_as_of=mature_point.valuation_as_of,
                source_json={"price": "DIRECT_INDEX_TEST"},
                payload_json={},
                snapshot_hash="test-maturity",
            )
        )
        session.flush()
        for offset, point in enumerate([mature_point, *pending_points]):
            iteration_number = 510 + offset
            session.add(
                V32TrainingIteration(
                    training_run_id="V32-TEST-MATURITY",
                    market="399006",
                    iteration_number=iteration_number,
                    parent_iteration_number=iteration_number - 1,
                    week_key=point.week_key,
                    cutoff_date=point.cutoff_date,
                    status="complete",
                    maturity_status="pending",
                    champion_weekly_version="CYB_WEEKLY_V3.2.31",
                    champion_daily_version="CYB_DAILY_CORRECTOR_V3.2.31",
                    data_snapshot_id="V32-DS-TEST-MATURITY",
                    input_features_json={"weekly": dict(point.weekly), "daily": dict(point.daily)},
                    forecast_json=V32TrainingService._forecast_payload(saved_forecast),
                    evaluation_json={},
                    optimizer_state_json={},
                    audit_json={"future_data_used": False},
                )
            )
        session.add(
            V32OptimizerState(
                market="399006",
                iteration_number=523,
                last_training_week_key="2026-W31",
                champion_weekly_version="CYB_WEEKLY_V3.2.31",
                champion_daily_version="CYB_DAILY_CORRECTOR_V3.2.31",
                weekly_parameters_json=weekly,
                daily_parameters_json=daily,
                optimizer_memory_json={},
                last_successful_training_at=now - timedelta(days=8),
                next_training_eligible_at=now - timedelta(days=1),
                state_hash="test",
            )
        )

    service = V32TrainingService(sessions, now_provider=lambda: now, calendar=_WeekdayCalendar())
    service._load_points = lambda _market: ([mature_point, *pending_points], 0)  # type: ignore[method-assign]
    service._forecast = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("pending maturity must not regenerate a forecast")
    )
    try:
        result = service.train_incremental("399006")
    finally:
        service.shutdown()

    with sessions() as session:
        rows = list(
            session.scalars(
                select(V32TrainingIteration).order_by(V32TrainingIteration.iteration_number)
            )
        )
    assert result["status"] == "not_due"
    assert len(rows) == 14
    assert rows[0].maturity_status == "full"
    assert float(rows[0].composite_loss) == pytest.approx(expected["composite_loss"])
    assert rows[0].evaluation_json["actual_path"] == list(mature_point.future_path or ())
    assert rows[0].audit_json["maturity_evaluation_source"] == "saved_forecast_json"
    assert all(row.maturity_status == "pending" for row in rows[1:])
    assert all(row.composite_loss is None for row in rows[1:])


def test_multiweek_gap_creates_at_most_one_inherited_iteration(tmp_path: Path) -> None:
    sessions = _database(tmp_path)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    weekly, daily = _initial_parameters("399006")
    with sessions() as session, session.begin():
        session.add(
            V32OptimizerState(
                market="399006",
                iteration_number=500,
                last_training_week_key="2026-W27",
                champion_weekly_version="CYB_WEEKLY_V3.2.2",
                champion_daily_version="CYB_DAILY_CORRECTOR_V3.2.2",
                weekly_parameters_json=weekly,
                daily_parameters_json=daily,
                optimizer_memory_json={},
                last_successful_training_at=now - timedelta(days=30),
                next_training_eligible_at=now - timedelta(days=23),
                state_hash="test",
            )
        )
    service = V32TrainingService(sessions, now_provider=lambda: now, calendar=_WeekdayCalendar())
    recorded: list[int] = []
    service._load_points = lambda _market: ([_point(600, "2026-W31")], 0)  # type: ignore[method-assign]
    service._forecast = lambda *_args, **_kwargs: _forecast()  # type: ignore[method-assign]
    service._evaluate = lambda *_args: None  # type: ignore[method-assign]
    service._candidate_parameters = lambda *_args, **_kwargs: (weekly, daily)  # type: ignore[method-assign]
    service._compare_candidate = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "promoted": False,
        "reason": "test rejection",
        "sample_count": 52,
    }
    service._persist_training_step = lambda _run, _market, iteration, *_args: recorded.append(iteration)  # type: ignore[method-assign]
    service._save_optimizer_state = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    try:
        result = service.train_incremental("399006")
    finally:
        service.shutdown()
    assert result["iteration_number"] == 501
    assert recorded == [501]


def test_data_analysis_does_not_change_training_state(tmp_path: Path) -> None:
    sessions = _database(tmp_path)
    point = _point(600)
    weekly, daily = _initial_parameters("399006")

    class TrainingStub:
        def _validate_market(self, market: str) -> None:
            assert market == "399006"

        def champion(self, _market: str):
            return {
                "iteration_number": 511,
                "weekly_version": "CYB_WEEKLY_V3.2.31",
                "daily_version": "CYB_DAILY_CORRECTOR_V3.2.31",
                "weekly_parameters": weekly,
                "daily_parameters": daily,
            }

        def _load_points(self, _market: str):
            return [point], 0

        def _forecast(self, *_args, **_kwargs):
            return _forecast()

        def _persist_snapshot(self, *_args, **_kwargs):
            return "V32-DS-TEST"

    gate = SimpleNamespace(
        complete_week_as_of=point.cutoff_date,
        price_as_of=point.cutoff_date,
        valuation_as_of=point.valuation_as_of,
        degraded=False,
    )

    class PipelineStub:
        def _refresh(self, _market: str, _refresh: bool):
            return {}

        def _data_gate(self, _market: str, _context):
            return gate

        def _gate_payload(self, _gate):
            return {"price_data_as_of": point.cutoff_date.isoformat()}

    class InvestmentStub:
        def current_positions(self):
            return {"399006": 0, "NDX": 0}

        def list_entries(self, _market: str):
            return []

    service = V32AnalysisService(
        sessions,
        training=TrainingStub(),  # type: ignore[arg-type]
        v31_data_pipeline=PipelineStub(),  # type: ignore[arg-type]
        calendar=_WeekdayCalendar(),  # type: ignore[arg-type]
        investment_calendar=InvestmentStub(),  # type: ignore[arg-type]
    )
    with sessions() as session:
        before = session.scalar(select(func.count()).select_from(V32TrainingIteration))
    result = service.run("399006", refresh=False)
    with sessions() as session:
        after = session.scalar(select(func.count()).select_from(V32TrainingIteration))
        run = session.scalar(select(V32AnalysisRun))

    assert before == after == 0
    assert result["training_mutated"] is False
    assert result["model"]["weekly"]["iteration_number"] == 511
    assert result["position"]["defaulted_to_zero"] is True
    assert result["advice"]["final_target_position"] % 5 == 0
    assert all(batch["position_points"] % 5 == 0 for batch in result["advice"]["batches"])
    assert run is not None and run.status == "completed"
