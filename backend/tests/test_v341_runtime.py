from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
from sqlalchemy import func, select

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    V341CandidateTrial,
    V341AnalysisRun,
    V341FeatureSnapshot,
    V341Forecast,
    V341ForecastCalibratorLink,
    V341ForecastEvaluation,
    V341ModelHealthSnapshot,
    V341ModelVersion,
    V341OptimizerState,
    V341OuterEvaluationBlock,
    V341ProbabilityCalibrator,
    V341ResidualRecord,
    V341RandomPlan,
    V341TrainingIteration,
    V341TrainingProfile,
    V341TrainingRun,
)
from backend.app.services.v33_feature_service import FeatureSnapshot
from backend.app.services.v34_training_service import (
    build_training_samples,
    eligible_fully_matured,
    purged_walk_forward_folds,
)
from backend.app.services.v341_feature_service import build_feature_manifest
from backend.app.services.v341_runtime_service import (
    CALIBRATION_VERSION,
    MODEL_FAMILY,
    PROTOCOL_VERSION,
    V341RuntimeService,
    V341RuntimeError,
    count_new_matured_anchors,
    promotion_from_paired_losses,
)
from backend.app.services.v341_scenario_service import (
    RESIDUAL_SCHEMA_VERSION,
    SCENARIO_ADAPTER_VERSION,
    THRESHOLD_FORMULA_VERSION,
    ScenarioRandomPlan,
    generate_return_scenarios,
    generate_scenario_forecast,
    prior_standardized_residual_pool,
)
from backend.app.services.v341_training_service import (
    V341PathForecast,
    conservative_independent_sample_count,
)
from backend.web import (
    V33ModelActionRequest,
    analyze_v341_model,
    train_v341_model,
    v341_iterations,
    v341_model_status,
)


class _Calendar:
    def sessions(self, _instrument_code: str, start_date: date, end_date: date):
        output = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                output.append(current)
            current += timedelta(days=1)
        return output


class _HolidayCalendar(_Calendar):
    def sessions(self, instrument_code: str, start_date: date, end_date: date):
        if start_date <= date(2026, 10, 5) <= end_date:
            return []
        return super().sessions(instrument_code, start_date, end_date)


def _snapshot(index: int) -> FeatureSnapshot:
    anchor = date(2023, 1, 6) + timedelta(days=7 * index)
    base = 100.0 * math.exp(0.001 * index + 0.02 * math.sin(index / 5.0))
    sequence = []
    for offset in range(99, -1, -1):
        day = anchor - timedelta(days=offset)
        close = base * math.exp(-0.0003 * offset)
        sequence.append(
            {
                "trade_date": day.isoformat(),
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0 * (1 + 0.1 * math.sin((index - offset) / 4)),
                "turnover": close * 1_000_000,
                "dif": math.sin((index - offset) / 7) / 1000,
                "dea": math.sin((index - offset - 1) / 7) / 1000,
                "macd_histogram": math.cos((index - offset) / 7) / 2000,
            }
        )
    return FeatureSnapshot(
        market="399006",
        cutoff_date=anchor,
        source_data_max_date=anchor,
        daily_as_of=anchor,
        weekly_as_of=anchor,
        weekly={
            "weekly_dif": math.sin(index / 5) / 1000,
            "weekly_dea": math.sin((index - 1) / 5) / 1000,
            "weekly_macd_histogram": math.cos(index / 5) / 2000,
            "weekly_dif_slope_1": math.cos(index / 5) / 5000,
            "weekly_return_13": math.sin(index / 8) / 30,
        },
        daily={
            "daily_dif": math.sin(index / 3) / 1000,
            "daily_dea": math.sin((index - 1) / 3) / 1000,
            "daily_macd_histogram": math.cos(index / 3) / 2000,
            "daily_dif_slope_3": math.cos(index / 3) / 5000,
            "daily_dea_slope_3": math.cos((index - 1) / 3) / 6000,
            "daily_histogram_speed_3": math.sin(index / 3) / 7000,
            "daily_return_20": math.sin(index / 8) / 50,
        },
        daily_sequence=tuple(sequence),
        missing_masks={},
        provenance={"target": "399006"},
        derivative_turn={},
    )


def _runtime(tmp_path: Path, *, calendar=None) -> V341RuntimeService:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    return V341RuntimeService(
        create_session_factory(database),
        calendar=calendar or _Calendar(),
        now_provider=lambda: datetime(2026, 8, 2, tzinfo=timezone.utc),
        scenario_count=1000,
    )


def _seed_runtime(runtime: V341RuntimeService, snapshots, samples, current_index: int):
    current = samples[current_index]
    manifest = build_feature_manifest(snapshots[0])
    seed_rows = eligible_fully_matured(samples, current.anchor_date)[-520:]
    seed = runtime.training.fit(
        "399006",
        seed_rows,
        version=f"399006-V341-TEST-SEED-{current.anchor_date}",
        parent_version=None,
        trained_through=current.anchor_date,
        effective_from=current.anchor_date,
        alpha=5.0,
        feature_names=manifest.ordered_feature_names,
    )
    with runtime.sessions() as session, session.begin():
        profile = runtime._persist_profile(session, "399006", manifest)
        model = runtime._persist_model(session, seed, profile.id, status="champion")
        runtime._persist_adapter(session, "399006")
        runtime._write_optimizer(
            session,
            market="399006",
            profile=profile,
            anchor=current.anchor_date - timedelta(days=7),
            last_consumed_mature_anchor=max(row.anchor_date for row in seed_rows),
            last_candidate_anchor=None,
            champion_model_id=model.id,
            weekly_iteration_count=0,
            candidate_training_count=0,
            champion_promotion_count=0,
        )
    run_id = runtime._create_run("399006", "bootstrap", current.anchor_date, profile.id)
    return current, manifest, run_id


def test_new_mature_events_trigger_even_when_rolling_length_is_constant() -> None:
    snapshots = tuple(_snapshot(index) for index in range(560))
    matured = eligible_fully_matured(build_training_samples(snapshots), snapshots[-1].cutoff_date)
    rolling = matured[-520:]
    assert len(rolling) == 520
    last = rolling[-5].anchor_date
    assert count_new_matured_anchors(rolling, last) == tuple(
        row.anchor_date for row in rolling[-4:]
    )


def test_promotion_requires_measured_block_bootstrap_not_a_counter() -> None:
    champion = [1.0] * 52
    better = [0.98] * 52
    hard_gates = {
        "champion_coverage": 0.80,
        "candidate_coverage": 0.80,
        "champion_turning_accuracy": 0.60,
        "candidate_turning_accuracy": 0.60,
        "coefficient_norm_ratio": 1.0,
    }
    result = promotion_from_paired_losses(
        champion,
        better,
        seed=11,
        fold_lengths=(52,),
        hard_gate_metrics=hard_gates,
    )
    assert result["promoted"] is True
    noisy = better.copy()
    noisy[-1] = 1.2
    assert promotion_from_paired_losses(
        champion,
        noisy,
        seed=11,
        fold_lengths=(52,),
        hard_gate_metrics=hard_gates,
    )["promoted"] is False
    unstable = {**hard_gates, "coefficient_norm_ratio": 1.30}
    assert promotion_from_paired_losses(
        champion,
        better,
        seed=11,
        fold_lengths=(52,),
        hard_gate_metrics=unstable,
    )["promoted"] is False
    uncovered = {**hard_gates, "candidate_coverage": 0.20}
    assert promotion_from_paired_losses(
        champion,
        better,
        seed=11,
        fold_lengths=(52,),
        hard_gate_metrics=uncovered,
    )["promoted"] is False


def test_candidate_alphas_consume_identical_real_random_plans(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(90))
    rows = eligible_fully_matured(build_training_samples(snapshots), snapshots[-1].cutoff_date)
    training_indices, validation_indices = purged_walk_forward_folds(rows, folds=5)[-1]
    manifest = build_feature_manifest(snapshots[0])
    residuals = prior_standardized_residual_pool(77)
    first_losses, first_plans, first_components, first_payloads = runtime.training.fold_composite_losses(
        "399006",
        rows,
        training_indices,
        validation_indices,
        alpha=5.0,
        feature_names=manifest.ordered_feature_names,
        standardized_residuals=residuals,
        shared_seed=991,
        scenario_count=1000,
        capture_random_plans=True,
    )
    second_losses, second_plans, second_components, _ = runtime.training.fold_composite_losses(
        "399006",
        rows,
        training_indices,
        validation_indices,
        alpha=6.0,
        feature_names=manifest.ordered_feature_names,
        standardized_residuals=residuals,
        shared_seed=991,
        scenario_count=1000,
    )
    assert first_plans == second_plans
    assert tuple(payload["plan_hash"] for payload in first_payloads) == first_plans
    assert all(payload["consumed_fields"] == ["residual_indices"] for payload in first_payloads)
    assert all(set(payload) == {
        "version", "seed", "scenario_count", "residual_pool_size",
        "residual_indices", "consumed_fields", "semantics", "plan_hash",
    } for payload in first_payloads)
    assert len(first_losses) == len(second_losses) == len(validation_indices)
    assert set(first_components[0]) == {
        "path_mae",
        "terminal_mae",
        "brier",
        "wis",
        "drawdown_volatility",
        "recent_stability",
        "coverage",
        "turning_type_correct",
    }
    assert set(second_components[0]) == set(first_components[0])
    runtime.shutdown()


def test_next_complete_anchor_skips_a_closed_holiday_week(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, calendar=_HolidayCalendar())
    assert runtime._next_complete_anchor_after("399006", date(2026, 10, 2)) == date(
        2026, 10, 16
    )
    runtime.shutdown()


def test_history_boundary_uses_exact_520_formal_plus_52_warmup_anchors(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    all_snapshots = tuple(_snapshot(index) for index in range(700))
    by_anchor = {snapshot.cutoff_date: snapshot for snapshot in all_snapshots}
    runtime._complete_anchors = lambda _market: tuple(by_anchor)  # type: ignore[method-assign]
    runtime._snapshots = lambda _market, anchors: tuple(  # type: ignore[method-assign]
        by_anchor[anchor] for anchor in anchors
    )
    anchors, snapshots, _samples = runtime._prepare_history("399006")
    assert len(anchors) == len(snapshots) == 520 + 52
    assert anchors[0] == all_snapshots[-572].cutoff_date
    assert anchors[-520] == all_snapshots[-520].cutoff_date
    structure_samples = runtime._prepare_structure_samples(
        "399006", through=all_snapshots[-1].cutoff_date
    )
    assert len(structure_samples) == len(all_snapshots)
    assert structure_samples[0].anchor_date == all_snapshots[0].cutoff_date
    runtime.shutdown()


def test_formal_anchor_plan_excludes_warmup_and_uses_real_available_count() -> None:
    anchors = tuple(date(2015, 1, 2) + timedelta(days=7 * index) for index in range(565))
    formal = V341RuntimeService._formal_bootstrap_anchors(anchors)
    assert len(formal) == 513
    assert formal[0] == anchors[52]
    assert formal[-1] == anchors[-1]


def test_reconcile_reports_not_started_and_rejects_optimizer_without_iterations(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(580))
    anchors = tuple(snapshot.cutoff_date for snapshot in snapshots)
    runtime._complete_anchors = lambda _market: anchors  # type: ignore[method-assign]
    bounded = snapshots[-572:]
    runtime._prepare_history = lambda _market, through=None: (  # type: ignore[method-assign]
        tuple(snapshot.cutoff_date for snapshot in bounded),
        bounded,
        build_training_samples(bounded),
    )
    report = runtime.reconcile_bootstrap_prefix("399006")
    assert report["state"] == "NOT_STARTED"
    assert report["expected_iteration_count"] == 520
    assert report["feature_warmup_weeks"] == 52

    samples = build_training_samples(snapshots)
    _seed_runtime(runtime, snapshots, samples, 54)
    report = runtime.reconcile_bootstrap_prefix("399006")
    assert report["state"] == "AMBIGUOUS_INVALID"
    assert report["issues"] == ["OPTIMIZER_EXISTS_WITHOUT_FORMAL_ITERATION"]
    runtime.shutdown()


def test_reconcile_uses_recorded_actual_seed_boundary(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(580))
    anchors = tuple(snapshot.cutoff_date for snapshot in snapshots)
    samples = build_training_samples(snapshots)
    runtime._complete_anchors = lambda _market: anchors  # type: ignore[method-assign]
    _seed_runtime(runtime, snapshots, samples, 54)
    first_usable = anchors[80]
    with runtime.sessions() as session, session.begin():
        seed = session.scalar(
            select(V341ModelVersion).where(V341ModelVersion.parent_model_id.is_(None))
        )
        seed.metrics_json = {
            **dict(seed.metrics_json),
            "boundary_audit": {
                "earliest_formal_anchor": first_usable.isoformat(),
                "formal_anchor_count_before_minimum_gate": 520,
            },
        }
    report = runtime.reconcile_bootstrap_prefix("399006")
    assert report["expected_iteration_count"] == 500
    assert report["first_expected_anchor"] == first_usable.isoformat()
    assert report["formal_anchor_count_before_seed_maturity_gate"] == 520
    assert report["seed_maturity_gate_excluded_weeks"] == 20
    assert report["issues"] == ["OPTIMIZER_EXISTS_WITHOUT_FORMAL_ITERATION"]
    runtime.shutdown()


def test_start_incremental_reuses_existing_active_run(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    _current, _manifest, _run_id = _seed_runtime(runtime, snapshots, samples, 54)
    with runtime.sessions() as session:
        profile_id = session.scalar(select(V341OptimizerState)).profile_id
    active_id = runtime._create_run("399006", "incremental", None, profile_id)

    result = runtime.start_incremental("399006")

    assert result == {"run_id": active_id, "status": "queued", "market": "399006"}
    with runtime.sessions() as session:
        assert session.scalar(
            select(func.count()).select_from(V341TrainingRun).where(
                V341TrainingRun.run_type == "incremental"
            )
        ) == 1
    runtime.shutdown()


def test_incremental_not_due_skips_feature_history(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, _manifest, _run_id = _seed_runtime(runtime, snapshots, samples, 54)
    last_anchor = current.anchor_date - timedelta(days=7)
    runtime._complete_anchors = lambda _market: (last_anchor,)  # type: ignore[method-assign]

    def unexpected_history(_market):
        raise AssertionError("NOT_DUE incremental rebuilt feature history")

    runtime._prepare_history = unexpected_history  # type: ignore[method-assign]
    result = runtime.incremental_sync("399006")

    assert result["status"] == "NOT_DUE"
    assert result["last_anchor_date"] == last_anchor.isoformat()
    runtime.shutdown()


def test_analysis_job_is_persisted_reused_and_recovered(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime._analysis_executor.submit = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    created = runtime.start_analysis("399006")
    reused = runtime.start_analysis("399006")

    assert reused == created
    with runtime.sessions() as session:
        row = session.get(V341AnalysisRun, created["run_id"])
        assert row is not None
        assert row.status == "queued"
    assert runtime.recover_interrupted_runs() == 1
    assert runtime.analysis_status(created["run_id"])["status"] == "failed"
    assert runtime.analysis_status(created["run_id"])["error_code"] == "INTERRUPTED_PROCESS"
    runtime.shutdown()


def test_analysis_worker_persists_preflight_failure(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    run_id = runtime._create_analysis_run("399006")

    def fail(_market, _analysis_id):
        raise RuntimeError("injected analysis preflight failure")

    runtime._run_analysis_locked = fail  # type: ignore[method-assign]
    runtime._run_analysis_task(run_id, "399006")

    status = runtime.analysis_status(run_id)
    assert status["status"] == "failed"
    assert status["error_code"] == "RuntimeError"
    assert status["error_message"] == "injected analysis preflight failure"
    runtime.shutdown()


def test_active_calibrator_starts_only_at_its_effective_week(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with runtime.sessions() as session, session.begin():
        session.add(
            V341ProbabilityCalibrator(
                id="CAL-NEXT-WEEK",
                protocol_version=PROTOCOL_VERSION,
                model_market="399006",
                horizon_weeks=13,
                version=f"{CALIBRATION_VERSION}-TEST",
                model_family=MODEL_FAMILY,
                scenario_adapter_version=SCENARIO_ADAPTER_VERSION,
                residual_schema_version=RESIDUAL_SCHEMA_VERSION,
                threshold_formula_version=THRESHOLD_FORMULA_VERSION,
                fit_through_date=date(2026, 7, 31),
                effective_from_date=date(2026, 8, 7),
                raw_sample_count=40,
                effective_sample_count=31,
                status="PRELIMINARY",
                temperature=1.8,
                metrics_json={},
                calibration_hash="a" * 64,
                created_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
            )
        )
    with runtime.sessions() as session:
        assert 13 not in runtime._active_calibrators(session, "399006", date(2026, 7, 31))
        active = runtime._active_calibrators(session, "399006", date(2026, 8, 7))
        assert active[13]["temperature"] == 1.8
    runtime.shutdown()


def test_forecast_reused_by_analysis_without_mutating_training_identity(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 59)
    first = runtime._process_week(
        run_id, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session:
        before = runtime._training_identity(session, "399006")
        forecast_before = session.scalar(select(V341Forecast))
        assert forecast_before is not None
        assert session.scalar(select(func.count()).select_from(V341ModelHealthSnapshot)) == 1
        frozen_hash = forecast_before.forecast_hash
        frozen_seed = forecast_before.scenario_seed
    runtime._complete_anchors = lambda _market: (current.anchor_date,)  # type: ignore[method-assign]

    def unexpected_history(_market):
        raise AssertionError("existing forecast fast path rebuilt feature history")

    runtime._prepare_history = unexpected_history  # type: ignore[method-assign]
    payload = runtime.run_analysis("399006")
    with runtime.sessions() as session:
        after = runtime._training_identity(session, "399006")
        assert session.scalar(select(func.count()).select_from(V341Forecast)) == 1
        assert session.scalar(select(V341Forecast)).forecast_hash == frozen_hash
    assert first["forecast_hash"] == frozen_hash
    assert payload["scenario_seed"] == frozen_seed
    assert before == after
    runtime.shutdown()


def test_analysis_first_uses_the_effective_core_champion_manifest(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(70))
    samples = build_training_samples(snapshots)
    current, _extended_manifest, _run_id = _seed_runtime(runtime, snapshots, samples, 69)
    core_manifest = build_feature_manifest(
        snapshots[-1], feature_set_name="CORE"
    )
    seed_rows = eligible_fully_matured(samples, current.anchor_date)
    core_state = runtime.training.fit(
        "399006",
        seed_rows,
        version=f"399006-V341-TEST-CORE-{current.anchor_date}",
        parent_version=None,
        trained_through=current.anchor_date,
        effective_from=current.anchor_date,
        alpha=5.0,
        feature_names=core_manifest.ordered_feature_names,
    )
    with runtime.sessions() as session, session.begin():
        old_champion = session.scalar(
            select(V341ModelVersion).where(V341ModelVersion.status == "champion")
        )
        old_champion.status = "retired"
        core_profile = runtime._persist_profile(session, "399006", core_manifest)
        core_model = runtime._persist_model(
            session, core_state, core_profile.id, status="champion"
        )
        optimizer = session.scalar(select(V341OptimizerState))
        runtime._write_optimizer(
            session,
            market="399006",
            profile=core_profile,
            anchor=optimizer.last_anchor_date,
            last_consumed_mature_anchor=optimizer.last_consumed_mature_anchor,
            last_candidate_anchor=optimizer.last_candidate_anchor,
            champion_model_id=core_model.id,
            weekly_iteration_count=optimizer.weekly_iteration_count,
            candidate_training_count=optimizer.candidate_training_count,
            champion_promotion_count=optimizer.champion_promotion_count,
            last_structure_audit_anchor=optimizer.last_structure_audit_anchor,
        )
    runtime._prepare_history = lambda _market: (  # type: ignore[method-assign]
        tuple(item.cutoff_date for item in snapshots), snapshots, samples
    )
    with runtime.sessions() as session:
        before = runtime._training_identity(session, "399006")
    payload = runtime.run_analysis("399006")
    with runtime.sessions() as session:
        after = runtime._training_identity(session, "399006")
        optimizer = session.scalar(select(V341OptimizerState))
        model = session.get(V341ModelVersion, optimizer.champion_model_id)
        profile = session.get(V341TrainingProfile, model.profile_id)
        feature_snapshot = session.scalar(
            select(V341FeatureSnapshot).where(
                V341FeatureSnapshot.forecast_anchor_date == current.anchor_date
            )
        )
        assert model.profile_id == optimizer.profile_id == profile.id
        assert profile.feature_set_name == "CORE"
        assert feature_snapshot.feature_manifest_hash == core_manifest.manifest_hash
        assert profile.feature_manifest_hash == core_manifest.manifest_hash
    assert payload["forecast_anchor_date"] == current.anchor_date.isoformat()
    assert before == after
    runtime.shutdown()


def test_analysis_then_training_reuses_forecast_even_after_residual_pool_changes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(70))
    samples = build_training_samples(snapshots)
    old, manifest, old_run = _seed_runtime(runtime, snapshots, samples, 45)
    runtime._process_week(old_run, "399006", old.anchor_date, snapshots, samples, manifest)
    runtime._prepare_history = lambda _market: (  # type: ignore[method-assign]
        tuple(item.cutoff_date for item in snapshots), snapshots, samples
    )
    analysis = runtime.run_analysis("399006")
    with runtime.sessions() as session:
        frozen = session.scalar(
            select(V341Forecast).where(
                V341Forecast.forecast_anchor_date == snapshots[-1].cutoff_date
            )
        )
        frozen_hash = frozen.forecast_hash
        frozen_plan = frozen.random_plan_id
        profile_id = session.scalar(select(V341OptimizerState)).profile_id
    with runtime.sessions() as session, session.begin():
        optimizer = session.scalar(select(V341OptimizerState))
        optimizer.last_consumed_mature_anchor = max(
            row.anchor_date
            for row in eligible_fully_matured(samples, snapshots[-1].cutoff_date)
        )
    training_run = runtime._create_run(
        "399006", "incremental", snapshots[-1].cutoff_date, profile_id
    )
    result = runtime._process_week(
        training_run,
        "399006",
        snapshots[-1].cutoff_date,
        snapshots,
        samples,
        manifest,
    )
    with runtime.sessions() as session:
        same = session.scalar(
            select(V341Forecast).where(
                V341Forecast.forecast_anchor_date == snapshots[-1].cutoff_date
            )
        )
        assert same.forecast_hash == frozen_hash
        assert same.random_plan_id == frozen_plan
        assert session.scalar(select(func.count()).select_from(V341ResidualRecord)) >= 1
    assert analysis["forecast_hash"] == result["forecast_hash"] == frozen_hash
    runtime.shutdown()


@pytest.mark.parametrize(
    "window_mode", ["ROLLING_520W", "EXPANDING_AVAILABLE_HISTORY"]
)
def test_analysis_first_and_training_first_freeze_identical_long_history_forecasts(
    tmp_path: Path,
    window_mode: str,
) -> None:
    all_snapshots = tuple(_snapshot(index) for index in range(620))
    all_samples = build_training_samples(all_snapshots)
    bounded_snapshots = all_snapshots[-585:]
    bounded_samples = build_training_samples(bounded_snapshots)

    def prepared_runtime(root: Path):
        runtime = _runtime(root)
        current, manifest, run_id = _seed_runtime(
            runtime, bounded_snapshots, bounded_samples, 584
        )
        complete_matured = eligible_fully_matured(all_samples, current.anchor_date)
        with runtime.sessions() as session, session.begin():
            optimizer = session.scalar(select(V341OptimizerState))
            active_model = session.get(V341ModelVersion, optimizer.champion_model_id)
            active_profile = session.get(V341TrainingProfile, active_model.profile_id)
            if window_mode == "EXPANDING_AVAILABLE_HISTORY":
                active_model.status = "retired"
                expanding_state = runtime.training.fit(
                    "399006",
                    complete_matured,
                    version=f"399006-V341-ORDER-EXPANDING-{current.anchor_date}",
                    parent_version=None,
                    trained_through=current.anchor_date,
                    effective_from=current.anchor_date,
                    alpha=5.0,
                    feature_names=manifest.ordered_feature_names,
                )
                active_profile = runtime._persist_profile(
                    session,
                    "399006",
                    manifest,
                    training_window_mode=window_mode,
                )
                active_model = runtime._persist_model(
                    session, expanding_state, active_profile.id, status="champion"
                )
            runtime._write_optimizer(
                session,
                market="399006",
                profile=active_profile,
                anchor=optimizer.last_anchor_date,
                last_consumed_mature_anchor=complete_matured[-1].anchor_date,
                last_candidate_anchor=optimizer.last_candidate_anchor,
                champion_model_id=active_model.id,
                weekly_iteration_count=optimizer.weekly_iteration_count,
                candidate_training_count=optimizer.candidate_training_count,
                champion_promotion_count=optimizer.champion_promotion_count,
                last_structure_audit_anchor=current.anchor_date,
            )
        runtime._prepare_history = lambda _market: (  # type: ignore[method-assign]
            tuple(item.cutoff_date for item in bounded_snapshots),
            bounded_snapshots,
            bounded_samples,
        )
        runtime._prepare_structure_samples = (  # type: ignore[method-assign]
            lambda _market, through: all_samples
        )
        return runtime, current, manifest, run_id

    analysis_runtime, analysis_current, analysis_manifest, analysis_run = prepared_runtime(
        tmp_path / "analysis-first"
    )
    analysis_payload = analysis_runtime.run_analysis("399006")
    analysis_training_payload = analysis_runtime._process_week(
        analysis_run,
        "399006",
        analysis_current.anchor_date,
        bounded_snapshots,
        bounded_samples,
        analysis_manifest,
        structure_samples=all_samples,
    )
    training_runtime, training_current, training_manifest, training_run = prepared_runtime(
        tmp_path / "training-first"
    )
    training_payload = training_runtime._process_week(
        training_run,
        "399006",
        training_current.anchor_date,
        bounded_snapshots,
        bounded_samples,
        training_manifest,
        structure_samples=all_samples,
    )
    training_analysis_payload = training_runtime.run_analysis("399006")

    def frozen_identity(runtime: V341RuntimeService):
        with runtime.sessions() as session:
            forecast = session.scalar(select(V341Forecast))
            plan = session.get(V341RandomPlan, forecast.random_plan_id)
            return (
                forecast.forecast_hash,
                forecast.random_plan_hash,
                plan.residual_pool_identity,
                plan.plan_hash,
            )

    analysis_identity = frozen_identity(analysis_runtime)
    training_identity = frozen_identity(training_runtime)
    assert analysis_identity == training_identity
    assert analysis_payload["forecast_hash"] == analysis_training_payload["forecast_hash"]
    assert training_payload["forecast_hash"] == training_analysis_payload["forecast_hash"]
    assert analysis_payload["forecast_hash"] == training_payload["forecast_hash"]
    analysis_runtime.shutdown()
    training_runtime.shutdown()


def test_forecast_freezes_before_same_anchor_maturity_regardless_of_call_order(
    tmp_path: Path,
) -> None:
    snapshots = tuple(_snapshot(index) for index in range(70))
    samples = build_training_samples(snapshots)
    old_index = 56
    current_index = old_index + 13

    def prepared_runtime(root: Path):
        runtime = _runtime(root)
        old, manifest, old_run = _seed_runtime(
            runtime, snapshots, samples, old_index
        )
        runtime._process_week(
            old_run,
            "399006",
            old.anchor_date,
            snapshots,
            samples,
            manifest,
        )
        current = samples[current_index]
        runtime._prepare_history = lambda _market: (  # type: ignore[method-assign]
            tuple(item.cutoff_date for item in snapshots), snapshots, samples
        )
        with runtime.sessions() as session:
            profile_id = session.scalar(select(V341OptimizerState)).profile_id
            old_forecast = session.scalar(
                select(V341Forecast).where(
                    V341Forecast.forecast_anchor_date == old.anchor_date
                )
            )
            assert old_forecast.maturity_status == "IMMATURE"
            assert session.scalar(select(func.count()).select_from(V341ResidualRecord)) == 0
        current_run = runtime._create_run(
            "399006", "incremental", current.anchor_date, profile_id
        )
        return runtime, current, manifest, current_run, old.anchor_date

    def frozen_identity(runtime: V341RuntimeService, anchor: date):
        with runtime.sessions() as session:
            forecast = session.scalar(
                select(V341Forecast).where(
                    V341Forecast.forecast_anchor_date == anchor
                )
            )
            plan = session.get(V341RandomPlan, forecast.random_plan_id)
            calibrator_links = tuple(
                (row.horizon_weeks, row.calibrator_id, row.calibration_hash)
                for row in session.scalars(
                    select(V341ForecastCalibratorLink)
                    .where(V341ForecastCalibratorLink.forecast_id == forecast.id)
                    .order_by(V341ForecastCalibratorLink.horizon_weeks)
                )
            )
            return (
                forecast.forecast_hash,
                forecast.random_plan_hash,
                plan.residual_pool_identity,
                plan.plan_hash,
                calibrator_links,
            )

    def assert_old_forecast_matured(runtime: V341RuntimeService, old_anchor: date):
        with runtime.sessions() as session:
            old_forecast = session.scalar(
                select(V341Forecast).where(
                    V341Forecast.forecast_anchor_date == old_anchor
                )
            )
            assert old_forecast.maturity_status == "FULLY_MATURE_13W"
            assert session.scalar(
                select(func.count())
                .select_from(V341ForecastEvaluation)
                .where(V341ForecastEvaluation.forecast_id == old_forecast.id)
            ) == 3
            assert session.scalar(
                select(func.count())
                .select_from(V341ResidualRecord)
                .where(V341ResidualRecord.forecast_id == old_forecast.id)
            ) == 1

    analysis_runtime, analysis_current, analysis_manifest, analysis_run, analysis_old = (
        prepared_runtime(tmp_path / "same-anchor-analysis-first")
    )
    analysis_payload = analysis_runtime.run_analysis("399006")
    analysis_training_payload = analysis_runtime._process_week(
        analysis_run,
        "399006",
        analysis_current.anchor_date,
        snapshots,
        samples,
        analysis_manifest,
    )

    training_runtime, training_current, training_manifest, training_run, training_old = (
        prepared_runtime(tmp_path / "same-anchor-training-first")
    )
    training_payload = training_runtime._process_week(
        training_run,
        "399006",
        training_current.anchor_date,
        snapshots,
        samples,
        training_manifest,
    )
    training_analysis_payload = training_runtime.run_analysis("399006")

    analysis_identity = frozen_identity(
        analysis_runtime, analysis_current.anchor_date
    )
    training_identity = frozen_identity(
        training_runtime, training_current.anchor_date
    )
    assert analysis_identity == training_identity
    assert analysis_payload["forecast_hash"] == analysis_training_payload["forecast_hash"]
    assert training_payload["forecast_hash"] == training_analysis_payload["forecast_hash"]
    assert analysis_payload["forecast_hash"] == training_payload["forecast_hash"]
    assert_old_forecast_matured(analysis_runtime, analysis_old)
    assert_old_forecast_matured(training_runtime, training_old)
    analysis_runtime.shutdown()
    training_runtime.shutdown()


def test_candidate_consumes_only_one_four_event_batch_and_preserves_remainder(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(95))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 94)
    matured = eligible_fully_matured(samples, current.anchor_date)[-20:]
    with runtime.sessions() as session, session.begin():
        optimizer = session.scalar(select(V341OptimizerState))
        optimizer.last_consumed_mature_anchor = matured[-9].anchor_date
        optimizer.last_candidate_anchor = None
        optimizer.state_json = {
            **optimizer.state_json,
            "last_candidate_effective_count": 0,
        }

    def fake_trial(_session, _market, _anchor, _matured, _state, row, *_args):
        return row, {"promoted": False, "reason": "TEST_CANDIDATE"}

    runtime._candidate_trial = fake_trial  # type: ignore[method-assign]
    runtime._process_week(
        run_id, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session:
        optimizer = session.scalar(select(V341OptimizerState))
        remaining = count_new_matured_anchors(
            eligible_fully_matured(samples, current.anchor_date)[-520:],
            optimizer.last_consumed_mature_anchor,
        )
        assert optimizer.last_consumed_mature_anchor == matured[-5].anchor_date
        assert len(remaining) == 4
        assert optimizer.candidate_training_count == 1
        assert optimizer.last_structure_audit_anchor is None
        assert optimizer.state_json["structure_audit"]["status"] == "NOT_DUE"
    runtime.shutdown()


def test_due_structure_audit_runs_a_real_sealed_feature_comparison(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(180))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 179)
    result = runtime._process_week(
        run_id, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session:
        optimizer = session.scalar(select(V341OptimizerState))
        trial = session.scalar(select(V341CandidateTrial))
        block = session.scalar(select(V341OuterEvaluationBlock))
        assert result["training_triggered"] is False
        assert optimizer.last_structure_audit_anchor == current.anchor_date
        assert optimizer.candidate_training_count == 1
        assert optimizer.state_json["structure_audit"]["status"].startswith(
            "COMPLETED_"
        )
        decision = optimizer.state_json["structure_audit"]["decision"]
        assert decision["trial_type"] == "LOW_FREQUENCY_STRUCTURE_AUDIT"
        assert decision["current_structure"]["feature_set"] == "EXTENDED"
        assert decision["selected_structure"]["feature_set"] == "CORE"
        assert trial.promotion_json["trial_type"] == "LOW_FREQUENCY_STRUCTURE_AUDIT"
        assert len(block.validation_anchors_json) == 13
    runtime.shutdown()


def test_structure_audit_selects_on_inner_folds_and_executes_expanding_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(750))
    samples = build_training_samples(snapshots)
    current, _manifest, run_id = _seed_runtime(runtime, snapshots, samples, 749)
    available = eligible_fully_matured(samples, current.anchor_date)
    folds = purged_walk_forward_folds(available, folds=5)
    core_manifest = build_feature_manifest(snapshots[-1], feature_set_name="CORE")
    outer_calls: list[tuple[str, int]] = []

    def fake_fold_losses(
        _market,
        rows,
        training_indices,
        validation_indices,
        *,
        feature_names,
        capture_random_plans=False,
        **_kwargs,
    ):
        feature_set = "CORE" if tuple(feature_names) == core_manifest.ordered_feature_names else "EXTENDED"
        is_outer = len(validation_indices) == 13
        if is_outer:
            outer_calls.append((feature_set, len(training_indices)))
            score = 0.5 if feature_set == "CORE" and len(training_indices) > 520 else 1.0
        elif feature_set == "CORE" and len(training_indices) > 520:
            score = 0.1
        elif feature_set == "CORE":
            score = 0.4
        elif len(training_indices) > 520:
            score = 0.3
        else:
            score = 0.6
        plans = tuple(f"PLAN-{rows[index].anchor_date}" for index in validation_indices)
        components = tuple(
            {
                "path_mae": score,
                "terminal_mae": score,
                "brier": score,
                "wis": score,
                "drawdown_volatility": score,
                "recent_stability": score,
                "coverage": 0.80,
                "turning_type_correct": 1.0,
            }
            for _ in validation_indices
        )
        payloads = tuple(
            {"plan_hash": plan, "validation_anchor": rows[index].anchor_date.isoformat()}
            for plan, index in zip(plans, validation_indices)
        ) if capture_random_plans else ()
        return (
            np.full(len(validation_indices), score, dtype=float),
            plans,
            components,
            payloads,
        )

    monkeypatch.setattr(runtime.training, "fold_composite_losses", fake_fold_losses)
    monkeypatch.setattr(
        runtime,
        "_shared_standardized_coefficient_norm_ratio",
        lambda _champion, _candidate: (1.0, core_manifest.ordered_feature_names),
    )
    with runtime.sessions() as session, session.begin():
        champion = session.scalar(
            select(V341ModelVersion).where(V341ModelVersion.status == "champion")
        )
        profile = session.get(V341TrainingProfile, champion.profile_id)
        run = session.get(V341TrainingRun, run_id)
        trial = runtime._structure_audit_trial(
            session,
            "399006",
            current.anchor_date,
            available,
            runtime._load_state(session, "399006", current.anchor_date)[0],
            champion,
            profile,
            run,
            1,
        )
        assert trial is not None
        candidate, decision = trial
        assert decision["selected_structure"] == {
            "feature_set": "CORE",
            "window_mode": "EXPANDING_AVAILABLE_HISTORY",
            "daily_mode": "100_SESSION_RIDGE_ONLY",
        }
        assert decision["structure_selection_scope"] == "INNER_PURGED_WALK_FORWARD_ONLY"
        assert decision["outer_block_role"] == "ONE_TIME_LOCKED_CHALLENGER_PROMOTION_ONLY"
        assert session.get(V341TrainingProfile, candidate.profile_id).training_window_mode == "EXPANDING_AVAILABLE_HISTORY"
    with runtime.sessions() as session:
        block = session.scalar(select(V341OuterEvaluationBlock))
        trial_row = session.scalar(select(V341CandidateTrial))
        audit = block.training_anchors_json
        assert len(audit["champion"]["anchors"]) == 520
        assert len(audit["challenger"]["anchors"]) > 520
        inner_validation = {
            anchor
            for fold in trial_row.loss_components_json["inner_fold_audit"]
            for anchor in fold["validation_anchors"]
        }
        assert inner_validation.isdisjoint(set(block.validation_anchors_json))
    assert outer_calls == [("EXTENDED", 520), ("CORE", len(folds[-1][0]))]
    runtime.shutdown()


def test_structure_coefficient_gate_ignores_nonshared_extended_dimensions(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(80))
    samples = build_training_samples(snapshots)
    matured = eligible_fully_matured(samples, samples[-1].anchor_date)
    core_manifest = build_feature_manifest(snapshots[-1], feature_set_name="CORE")
    extended_manifest = build_feature_manifest(
        snapshots[-1], feature_set_name="EXTENDED"
    )
    core = runtime.training.fit(
        "399006",
        matured,
        version="CORE-GATE-TEST",
        parent_version=None,
        trained_through=samples[-1].anchor_date,
        effective_from=samples[-1].anchor_date,
        alpha=5.0,
        feature_names=core_manifest.ordered_feature_names,
    )
    extended = runtime.training.fit(
        "399006",
        matured,
        version="EXTENDED-GATE-TEST",
        parent_version=None,
        trained_through=samples[-1].anchor_date,
        effective_from=samples[-1].anchor_date,
        alpha=5.0,
        feature_names=extended_manifest.ordered_feature_names,
    )
    core_by_name = dict(zip(core.feature_names, core.coefficients))
    adjusted_extended = replace(
        extended,
        coefficients=tuple(
            core_by_name[name]
            if name in core_by_name
            else tuple(1_000_000.0 for _ in extended.intercept)
            for name in extended.feature_names
        ),
    )
    ratio, shared = runtime._shared_standardized_coefficient_norm_ratio(
        adjusted_extended, core
    )
    assert shared == core.feature_names
    assert ratio == pytest.approx(1.0)
    runtime.shutdown()


def test_regular_candidate_after_expanding_promotion_uses_all_available_history(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    all_snapshots = tuple(_snapshot(index) for index in range(620))
    all_samples = build_training_samples(all_snapshots)
    snapshots = all_snapshots[-585:]
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 584)
    all_available = eligible_fully_matured(all_samples, current.anchor_date)
    expanding_state = runtime.training.fit(
        "399006",
        all_available,
        version=f"399006-V341-TEST-EXPANDING-{current.anchor_date}",
        parent_version=None,
        trained_through=current.anchor_date,
        effective_from=current.anchor_date,
        alpha=5.0,
        feature_names=manifest.ordered_feature_names,
    )
    with runtime.sessions() as session, session.begin():
        old_champion = session.scalar(
            select(V341ModelVersion).where(V341ModelVersion.status == "champion")
        )
        old_champion.status = "retired"
        expanding_profile = runtime._persist_profile(
            session,
            "399006",
            manifest,
            training_window_mode="EXPANDING_AVAILABLE_HISTORY",
        )
        expanding_model = runtime._persist_model(
            session, expanding_state, expanding_profile.id, status="champion"
        )
        optimizer = session.scalar(select(V341OptimizerState))
        runtime._write_optimizer(
            session,
            market="399006",
            profile=expanding_profile,
            anchor=optimizer.last_anchor_date,
            last_consumed_mature_anchor=all_available[-5].anchor_date,
            last_candidate_anchor=None,
            champion_model_id=expanding_model.id,
            weekly_iteration_count=optimizer.weekly_iteration_count,
            candidate_training_count=optimizer.candidate_training_count,
            champion_promotion_count=optimizer.champion_promotion_count,
            last_structure_audit_anchor=current.anchor_date,
            state_metadata={"last_candidate_effective_count": 0},
        )
    captured: list[int] = []

    def fake_trial(_session, _market, _anchor, matured, _state, row, *_args):
        captured.append(len(matured))
        return row, {"promoted": False, "reason": "TEST_EXPANDING_CONTINUITY"}

    runtime._candidate_trial = fake_trial  # type: ignore[method-assign]
    result = runtime._process_week(
        run_id,
        "399006",
        current.anchor_date,
        snapshots,
        samples,
        manifest,
        structure_samples=all_samples,
    )
    assert result["training_triggered"] is True
    assert captured == [len(all_available)]
    assert captured[0] > 520
    with runtime.sessions() as session:
        optimizer = session.scalar(select(V341OptimizerState))
        profile = session.get(V341TrainingProfile, optimizer.profile_id)
        assert profile.training_window_mode == "EXPANDING_AVAILABLE_HISTORY"
    runtime.shutdown()


@pytest.mark.parametrize("elapsed_weeks,effective_growth", [(25, 3), (26, 2)])
def test_structure_audit_requires_both_interval_and_effective_growth(
    tmp_path: Path,
    elapsed_weeks: int,
    effective_growth: int,
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(100))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 99)
    matured = eligible_fully_matured(samples, current.anchor_date)[-520:]
    independent = conservative_independent_sample_count(
        tuple(
            (row.anchor_date, row.label_end_date)
            for row in matured
            if row.label_end_date is not None
        )
    )
    previous_anchor = current.anchor_date - timedelta(weeks=elapsed_weeks)
    with runtime.sessions() as session, session.begin():
        optimizer = session.scalar(select(V341OptimizerState))
        optimizer.last_structure_audit_anchor = previous_anchor
        optimizer.state_json = {
            **optimizer.state_json,
            "last_structure_audit_effective_count": independent - effective_growth,
            "structure_candidate_status": "NONE_RUNNING",
        }
    runtime._process_week(
        run_id, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session:
        optimizer = session.scalar(select(V341OptimizerState))
        assert optimizer.last_structure_audit_anchor == previous_anchor
        assert optimizer.state_json["structure_audit"]["status"] == "NOT_DUE"
        assert session.scalar(select(func.count()).select_from(V341CandidateTrial)) == 0
    runtime.shutdown()


def test_regular_candidate_waits_when_four_events_do_not_increase_effective_sample_count(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(110))
    samples = build_training_samples(snapshots)

    def ess_at(index: int) -> int:
        matured = eligible_fully_matured(samples, samples[index].anchor_date)[-520:]
        return conservative_independent_sample_count(
            tuple(
                (row.anchor_date, row.label_end_date)
                for row in matured
                if row.label_end_date is not None
            )
        )

    first_index = next(
        index
        for index in range(80, 105)
        if ess_at(index) == ess_at(index + 4)
    )
    current, manifest, first_run = _seed_runtime(
        runtime, snapshots, samples, first_index
    )
    matured = eligible_fully_matured(samples, current.anchor_date)[-520:]
    with runtime.sessions() as session, session.begin():
        optimizer = session.scalar(select(V341OptimizerState))
        optimizer.last_consumed_mature_anchor = matured[-5].anchor_date
        optimizer.last_candidate_anchor = None
        optimizer.state_json = {
            **optimizer.state_json,
            "last_candidate_effective_count": ess_at(first_index),
        }

    calls: list[date] = []

    def fake_trial(_session, _market, anchor, _matured, _state, row, *_args):
        calls.append(anchor)
        return row, {"promoted": False, "reason": "TEST_FOUR_EVENT_CADENCE"}

    runtime._candidate_trial = fake_trial  # type: ignore[method-assign]
    first_result = runtime._process_week(
        first_run, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session:
        optimizer = session.scalar(select(V341OptimizerState))
        assert optimizer.candidate_training_count == 0
        assert optimizer.last_candidate_anchor is None
        assert optimizer.last_consumed_mature_anchor == matured[-5].anchor_date
    assert ess_at(first_index) == ess_at(first_index + 4)
    assert first_result["training_triggered"] is False
    assert calls == []
    runtime.shutdown()


def test_outer_block_rejects_a_missing_complete_market_week(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    anchors = tuple(date(2026, 1, 9) + timedelta(weeks=index) for index in range(13))
    runtime._assert_consecutive_complete_market_weeks("399006", anchors)
    with pytest.raises(V341RuntimeError, match="consecutive complete market weeks"):
        runtime._assert_consecutive_complete_market_weeks(
            "399006", anchors[:6] + anchors[7:]
        )
    runtime.shutdown()


def test_scenario_medoid_ohlcv_quantiles_and_dif_path_are_continuous(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(65))
    samples = build_training_samples(snapshots)
    snapshot = snapshots[-1]
    matured = eligible_fully_matured(samples, snapshot.cutoff_date)
    varied = []
    for sample in matured:
        rows = []
        for row in sample.realized_ohlcv:
            ordinal = date.fromisoformat(str(row["week_end"])).toordinal()
            close = float(row["close"])
            open_value = close * (0.965 + (ordinal % 11) * 0.004)
            rows.append(
                {
                    **row,
                    "open": open_value,
                    "high": max(open_value, close) * (1.01 + (ordinal % 5) * 0.004),
                    "low": min(open_value, close) * (0.99 - (ordinal % 3) * 0.004),
                    "volume": 800_000.0 + (ordinal % 17) * 75_000.0,
                }
            )
        varied.append(replace(sample, realized_ohlcv=tuple(rows)))
    residuals = np.stack(
        (
            np.linspace(-0.08, 0.04, 13),
            np.linspace(0.03, 0.12, 13),
        )
    )
    path = V341PathForecast(
        market="399006",
        anchor_date=snapshot.cutoff_date,
        model_version="TEST-MEDOID",
        weekly_base=tuple(0.0 for _ in range(13)),
        daily_correction=tuple(0.0 for _ in range(13)),
        expected_path=tuple(0.0 for _ in range(13)),
        source_feature_hash="f" * 64,
        forecast_sigma=0.02,
        ood_diagnostics={"ood_score": 0.0},
    )

    def run(plan: ScenarioRandomPlan | None = None):
        return generate_scenario_forecast(
            path,
            snapshot,
            tuple(varied),
            standardized_residuals=residuals,
            source_sigmas=(1.0, 1.0),
            residual_pool_metadata={"effective_residual_count": 30},
            calibrators={},
            scenario_count=1000,
            seed=None if plan is not None else 341,
            random_plan=plan,
        )

    baseline = run()
    cumulative = generate_return_scenarios(
        path.expected_path,
        residuals,
        (1.0, 1.0),
        path.forecast_sigma,
        baseline.random_plan,
    )
    start_price = float(snapshot.daily_sequence[-1]["close"])
    possible_closes = np.round(start_price * (1.0 + cumulative), 6)
    representative = np.asarray(
        [float(row["close"]) for row in baseline.representative_ohlcv]
    )
    assert any(np.array_equal(representative, row) for row in possible_closes)
    assert len(baseline.representative_ohlcv) == len(baseline.indicators) == 13
    assert all(
        float(row["low"]) <= min(float(row["open"]), float(row["close"]))
        <= max(float(row["open"]), float(row["close"])) <= float(row["high"])
        for row in baseline.representative_ohlcv
    )
    assert all(
        float(row["close_p10"]) <= float(row["close_p50"]) <= float(row["close_p90"])
        for row in baseline.price_quantiles
    )
    assert baseline.indicators[0]["dif_first_change"] is not None
    assert baseline.indicators[0]["dif_second_change"] is not None
    assert baseline.indicators[0]["trend_state"] != "INSUFFICIENT"
    for previous, current_row in zip(baseline.indicators, baseline.indicators[1:]):
        assert float(current_row["dif_first_change"]) == pytest.approx(
            float(current_row["dif"]) - float(previous["dif"]), abs=2e-8
        )

    plan = baseline.random_plan
    residual_changed = run(
        replace(plan, residual_indices=tuple(1 for _ in plan.residual_indices), plan_hash="r" * 64)
    )
    assert residual_changed.price_quantiles != baseline.price_quantiles
    last_index = plan.empirical_pool_size - 1
    gap_changed = run(
        replace(plan, gap_indices=tuple(last_index for _ in plan.gap_indices), plan_hash="g" * 64)
    )
    upper_changed = run(
        replace(plan, upper_wick_indices=tuple(last_index for _ in plan.upper_wick_indices), plan_hash="u" * 64)
    )
    lower_changed = run(
        replace(plan, lower_wick_indices=tuple(last_index for _ in plan.lower_wick_indices), plan_hash="l" * 64)
    )
    volume_changed = run(
        replace(plan, volume_indices=tuple(last_index for _ in plan.volume_indices), plan_hash="v" * 64)
    )
    assert gap_changed.representative_ohlcv != baseline.representative_ohlcv
    assert upper_changed.representative_ohlcv != baseline.representative_ohlcv
    assert lower_changed.representative_ohlcv != baseline.representative_ohlcv
    assert volume_changed.representative_ohlcv != baseline.representative_ohlcv
    runtime.shutdown()


def test_log_gross_residual_scenarios_preserve_positive_price_domain() -> None:
    residuals = np.full((1, 13), -60.0, dtype=float)
    plan = ScenarioRandomPlan.build(
        seed=344,
        scenario_count=1000,
        residual_pool_size=1,
        empirical_pool_size=1,
    )
    scenarios = generate_return_scenarios(
        tuple(0.0 for _ in range(13)),
        residuals,
        (1.0,),
        0.02,
        plan,
    )
    assert scenarios.shape == (1000, 13)
    assert np.all(np.isfinite(scenarios))
    assert np.all(scenarios > -1.0)
    assert scenarios[0, 0] == pytest.approx(np.expm1(-1.2))
    assert not np.isclose(scenarios[0, 0], -0.99)


def test_shrunk_residual_plan_uses_effective_weight_not_raw_path_count(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(65))
    samples = build_training_samples(snapshots)
    snapshot = snapshots[-1]
    matured = eligible_fully_matured(samples, snapshot.cutoff_date)
    path = V341PathForecast(
        market="399006",
        anchor_date=snapshot.cutoff_date,
        model_version="TEST-SHRINKAGE-ALLOCATION",
        weekly_base=tuple(0.0 for _ in range(13)),
        daily_correction=tuple(0.0 for _ in range(13)),
        expected_path=tuple(0.0 for _ in range(13)),
        source_feature_hash="a" * 64,
        forecast_sigma=0.02,
        ood_diagnostics={"ood_score": 0.0},
    )
    empirical = np.stack(
        [np.linspace(-0.04, 0.04, 13) * (index + 1) / 13 for index in range(13)]
    )
    forecast = generate_scenario_forecast(
        path,
        snapshot,
        matured,
        standardized_residuals=empirical,
        source_sigmas=tuple(1.0 for _ in range(13)),
        residual_pool_metadata={"effective_residual_count": 1},
        calibrators={},
        scenario_count=1000,
        seed=771,
    )
    metadata = forecast.residual_pool
    assert metadata["empirical_path_count"] == 13
    assert metadata["target_empirical_weight"] == pytest.approx(1 / 30)
    assert metadata["empirical_draw_count"] == 33
    assert metadata["prior_draw_count"] == 967
    assert metadata["empirical_weight"] == pytest.approx(0.033)
    empirical_group = forecast.random_plan.to_payload()["residual_groups"][0]
    assert empirical_group["start"] == 0
    assert empirical_group["stop"] == 13
    assert empirical_group["draw_count"] == 33
    runtime.shutdown()


def test_random_plan_payload_is_persisted_and_round_trippable(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 59)
    runtime._process_week(
        run_id, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session:
        forecast = session.scalar(select(V341Forecast))
        plan = session.get(V341RandomPlan, forecast.random_plan_id)
        health = session.scalar(select(V341ModelHealthSnapshot))
        payload = json.loads(zlib.decompress(plan.plan_payload_zlib).decode("utf-8"))
        assert payload["plan_hash"] == forecast.random_plan_hash == plan.plan_hash
        assert sum(group["draw_count"] for group in payload["residual_groups"]) == forecast.scenario_count
        assert len(payload["gap_indices"]) == forecast.scenario_count * 13
        assert health.diagnostics_json["multivariate_method"] == "ROBUST_SHRUNK_COVARIANCE_MAHALANOBIS"
    runtime.shutdown()


def test_ood_entry_requires_strictly_consecutive_weeks(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, _manifest, _run_id = _seed_runtime(runtime, snapshots, samples, 59)
    with runtime.sessions() as session:
        model = session.scalar(select(V341OptimizerState)).champion_model_id
    statuses = []
    for offset, score in enumerate((0.60, 0.40, 0.60, 0.60)):
        anchor = current.anchor_date + timedelta(weeks=offset + 1)
        path = V341PathForecast(
            market="399006",
            anchor_date=anchor,
            model_version="TEST",
            weekly_base=tuple(0.0 for _ in range(13)),
            daily_correction=tuple(0.0 for _ in range(13)),
            expected_path=tuple(0.0 for _ in range(13)),
            source_feature_hash="x" * 64,
            forecast_sigma=0.02,
            ood_diagnostics={"ood_score": score},
        )
        with runtime.sessions() as session, session.begin():
            model_row = session.get(V341ModelVersion, model)
            statuses.append(
                runtime._persist_health_snapshot(
                    session,
                    "399006",
                    anchor,
                    model_row,
                    path,
                    {"score": 50.0},
                )
            )
    assert statuses == ["MODEL_NORMAL", "MODEL_NORMAL", "MODEL_NORMAL", "MODEL_OUT_OF_DISTRIBUTION"]
    runtime.shutdown()


def test_incremental_backlog_is_oldest_first_and_stops_on_failure(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, _manifest, _run_id = _seed_runtime(runtime, snapshots, samples, 54)
    anchors = tuple(snapshot.cutoff_date for snapshot in snapshots)
    runtime._prepare_history = lambda _market: (anchors, snapshots, samples)  # type: ignore[method-assign]
    runtime._prepare_structure_samples = lambda _market, through: samples  # type: ignore[method-assign]
    calls = []

    def process(run_id, market, anchor, *_args, **_kwargs):
        calls.append(anchor)
        if len(calls) == 2:
            raise RuntimeError("injected backlog failure")
        with runtime.sessions() as session, session.begin():
            run = session.get(V341TrainingRun, run_id)
            run.result_json = {"first_anchor_committed": anchor.isoformat()}
        return {"status": "PROCESSED", "anchor_date": anchor.isoformat()}

    runtime._process_week = process  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected backlog failure"):
        runtime.incremental_sync("399006")
    assert calls == [snapshots[54].cutoff_date, snapshots[55].cutoff_date]
    with runtime.sessions() as session:
        run = session.scalar(
            select(V341TrainingRun)
            .where(V341TrainingRun.run_type == "incremental")
            .order_by(V341TrainingRun.started_at.desc())
        )
        assert run.status == "failed"
        assert run.current_stage == "failed_stop_backlog"
        assert run.result_json["first_anchor_committed"] == snapshots[54].cutoff_date.isoformat()
    runtime.shutdown()


def test_four_eight_thirteen_maturity_and_single_oos_residual(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 40)
    runtime._process_week(
        run_id, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session, session.begin():
        counts = runtime._mature_forecasts(
            session, "399006", snapshots, snapshots[53].cutoff_date
        )
        assert counts == {"4": 1, "8": 1, "13": 1, "residuals": 1}
    with runtime.sessions() as session:
        evaluations = list(session.scalars(select(V341ForecastEvaluation)))
        residuals = list(session.scalars(select(V341ResidualRecord)))
        assert [row.horizon_weeks for row in evaluations] == [4, 8, 13]
        assert len(residuals) == 1
        assert len(residuals[0].standardized_residual_json) == 13
        original_hash = residuals[0].residual_hash
    with runtime.sessions() as session, session.begin():
        counts = runtime._mature_forecasts(
            session, "399006", snapshots, snapshots[53].cutoff_date
        )
        assert counts == {"4": 0, "8": 0, "13": 0, "residuals": 0}
    with runtime.sessions() as session:
        assert session.scalar(select(func.count()).select_from(V341ResidualRecord)) == 1
        assert session.scalar(select(V341ResidualRecord)).residual_hash == original_hash
        assert session.scalar(select(V341TrainingIteration)).forecast_id == session.scalar(
            select(V341Forecast)
        ).id
    runtime.shutdown()


def test_small_nonempty_oos_pool_is_explicitly_shrunk_with_prior(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 40)
    runtime._process_week(
        run_id, "399006", current.anchor_date, snapshots, samples, manifest
    )
    with runtime.sessions() as session, session.begin():
        runtime._mature_forecasts(
            session, "399006", snapshots, snapshots[53].cutoff_date
        )
        profile_id = session.scalar(select(V341OptimizerState)).profile_id
    next_run = runtime._create_run(
        "399006", "incremental", snapshots[54].cutoff_date, profile_id
    )
    runtime._process_week(
        next_run,
        "399006",
        snapshots[54].cutoff_date,
        snapshots,
        samples,
        manifest,
    )
    with runtime.sessions() as session:
        forecast = session.scalar(
            select(V341Forecast).where(
                V341Forecast.forecast_anchor_date == snapshots[54].cutoff_date
            )
        )
        assert forecast.residual_pool_json["status"] == "SHRUNK_INSUFFICIENT_OOS_RESIDUALS"
        assert 0.0 < forecast.residual_pool_json["empirical_weight"] < 1.0
        assert forecast.residual_pool_json["prior_weight"] > 0.0
        assert forecast.residual_pool_json["allocation_semantics"] == "EXACT_STRATIFIED_FROZEN_DRAW_COUNTS"
        assert forecast.residual_pool_json["empirical_draw_count"] + forecast.residual_pool_json["prior_draw_count"] == forecast.scenario_count
        assert forecast.residual_pool_json["empirical_weight"] == pytest.approx(
            forecast.residual_pool_json["empirical_draw_count"] / forecast.scenario_count
        )
    runtime.shutdown()


def test_v341_web_contract_separates_training_and_analysis() -> None:
    class Runtime:
        def __init__(self):
            self.calls = []
            self.bootstrapped = False

        def champion(self, market):
            self.calls.append(("champion", market))
            return {"version": "V341"} if self.bootstrapped else None

        def start_bootstrap(self, market):
            self.calls.append(("bootstrap", market))
            return {"run_id": "bootstrap"}

        def start_incremental(self, market):
            self.calls.append(("incremental", market))
            return {"run_id": "incremental"}

        def run_analysis(self, market):
            self.calls.append(("analysis", market))
            return {"market": market, "training_mutated": False}

        def curve(self, market):
            return {"market": market, "points": []}

        def model_statuses(self):
            return [{"market": "399006"}, {"market": "159941"}]

    runtime = Runtime()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(v341_runtime=runtime)))
    values = V33ModelActionRequest(instrument_code="399006")
    assert train_v341_model(request, values)["run_id"] == "bootstrap"
    runtime.bootstrapped = True
    assert train_v341_model(request, values)["run_id"] == "incremental"
    assert analyze_v341_model(request, values)["training_mutated"] is False
    assert v341_iterations(request, "399006")["points"] == []
    assert len(v341_model_status(request)) == 2
