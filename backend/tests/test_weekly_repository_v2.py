from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib
import json
from pathlib import Path
from threading import Event, Thread, current_thread
import time
from typing import Any

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app import weekly_analysis
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    Base,
    Instrument,
    V2AdviceHistory,
    V2AnalysisIteration,
    V2AnalysisTask,
    V2IterationLabel,
    V2ModelVersion,
    V2WeekSample,
)
from backend.app.weekly_analysis.advice import Advice, build_advice
from backend.app.weekly_analysis.domain import QualityReport
from backend.app.weekly_analysis.features import (
    FEATURE_SET_VERSION,
    FeatureSnapshot,
    FrozenDict,
)
from backend.app.weekly_analysis.optimizer import (
    IterationLabel,
    IterationPrediction,
    OptimizerStepResult,
    TradingCalendar,
    WindowMetrics,
    run_optimizer_step,
    seed_model,
)


CUTOFF = date(2026, 7, 24)


def _repository_api() -> Any:
    return importlib.import_module("backend.app.weekly_analysis.repository")


def _sessions(tmp_path: Path):
    database = tmp_path / "repository.db"
    sessions = create_session_factory(database)
    engine = sessions.kw["bind"]
    Base.metadata.create_all(
        engine,
        tables=(
            Instrument.__table__,
            V2WeekSample.__table__,
            V2AnalysisIteration.__table__,
            V2IterationLabel.__table__,
            V2ModelVersion.__table__,
            V2AnalysisTask.__table__,
            V2AdviceHistory.__table__,
        ),
    )
    with sessions.begin() as session:
        session.add_all(
            (
                Instrument(
                    code="399006",
                    name="Growth",
                    exchange="SZSE",
                    category="index",
                ),
                Instrument(
                    code="NDX",
                    name="Nasdaq 100",
                    exchange="NASDAQ",
                    category="index",
                ),
            )
        )
    return sessions


def _repository(tmp_path: Path):
    module = _repository_api()
    sessions = _sessions(tmp_path)
    repository = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    return module, sessions, repository


def _prediction(symbol: str, week_key: str, cutoff: date) -> IterationPrediction:
    return IterationPrediction(
        iteration_id="I0001",
        instrument_code=symbol,
        week_key=week_key,
        cutoff_date=cutoff,
        predicted_direction="up",
        operation_side="buy",
        probability=Decimal("0.8"),
        predicted_action_date=cutoff + timedelta(days=3),
        target_position=Decimal("0.75"),
        trade_count=1,
    )


def _step(
    symbol: str = "399006",
    *,
    week_key: str = "2026-W30",
    cutoff: date = CUTOFF,
) -> OptimizerStepResult:
    return run_optimizer_step(
        seed_model(symbol),
        feedback=(),
        week_key=week_key,
        cutoff_date=cutoff,
        source_data_max_date=cutoff,
        prediction=_prediction(symbol, week_key, cutoff),
    )


def _mature_label(
    *, observed_through: date = date(2026, 10, 23)
) -> IterationLabel:
    return IterationLabel(
        iteration_id="I0001",
        instrument_code="399006",
        week_key="2026-W30",
        cutoff_date=CUTOFF,
        predicted_direction="up",
        operation_side="buy",
        probability=Decimal("0.8"),
        predicted_action_date=CUTOFF + timedelta(days=3),
        target_position=Decimal("0.75"),
        trade_count=1,
        status="mature_13w",
        observed_through=observed_through,
        visible_row_count=65,
        partial_weight=Decimal("1"),
        actual_direction="up",
        actual_turn_date=CUTOFF + timedelta(days=3),
        turning_deviation_sessions=0,
        loss_components=FrozenDict(
            {
                "turning_deviation": Decimal("0"),
                "direction_calibration": Decimal("0"),
                "adverse_excursion": Decimal("0"),
                "overtrading": Decimal("0"),
            }
        ),
        total_loss=Decimal("0"),
        direction_hit=True,
        calibration_formula="brier=(p-y)^2",
    )


def _second_step(
    first: OptimizerStepResult,
    label: IterationLabel,
) -> OptimizerStepResult:
    second_cutoff = label.observed_through
    current = WindowMetrics(
        instrument_code="399006",
        window="expanding",
        sample_count=1,
        total_loss=Decimal("0.50"),
        direction_hit_rate=Decimal("0.60"),
        calibration_loss=Decimal("0.20"),
        overtrade_penalty=Decimal("0.20"),
    )
    candidate = replace(current, total_loss=Decimal("0.49"))
    return run_optimizer_step(
        first.work_state,
        feedback=(label,),
        week_key="2026-W43",
        cutoff_date=second_cutoff,
        source_data_max_date=second_cutoff,
        prediction=IterationPrediction(
            iteration_id="I0002",
            instrument_code="399006",
            week_key="2026-W43",
            cutoff_date=second_cutoff,
            predicted_direction="up",
            operation_side="buy",
            probability=Decimal("0.8"),
            predicted_action_date=second_cutoff + timedelta(days=3),
            target_position=Decimal("0.75"),
            trade_count=1,
        ),
        current_metrics={"expanding": current},
        candidate_metrics={"expanding": candidate},
    )


def _weekdays(start: date, count: int) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return tuple(values)


def _advice(model, *, cutoff: date = CUTOFF) -> Advice:
    features = FrozenDict(
        {
            "valuation_percentile": Decimal("20"),
            "golden_point": True,
            "golden_strength": 5,
            "golden_point_reasons": (
                "valuation_percentile_low",
                "dif_crossed_above_dea",
            ),
            "black_point": False,
            "black_strength": 0,
            "black_point_reasons": (),
        }
    )
    snapshot = FeatureSnapshot(
        instrument_code="399006",
        cutoff_date=cutoff,
        source_data_max_date=cutoff,
        feature_set_version=FEATURE_SET_VERSION,
        features=features,
        availability=FrozenDict({name: "available" for name in features}),
        quality_report=QualityReport(
            is_publishable=True,
            issues=(),
            volume_availability="available",
        ),
        audit_fields=FrozenDict({"warmup_weeks": 60}),
        source_hash="a" * 64,
    )
    calendar = TradingCalendar(
        instrument_code="399006",
        expected_trade_dates=_weekdays(cutoff - timedelta(days=7), 70),
    )
    return build_advice(
        model=model,
        latest_features=snapshot,
        future_trading_calendar=calendar,
        current_position=20,
        target_position=80,
        direction="up",
        probability=85,
        confidence=90,
        confirmation_conditions_met=True,
    )


def _row_counts(sessions) -> tuple[int, ...]:
    with sessions() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model)) or 0
            for model in (
                V2WeekSample,
                V2AnalysisIteration,
                V2IterationLabel,
                V2ModelVersion,
                V2AdviceHistory,
            )
        )


def test_commit_survives_reopen_and_recovers_model_metrics_and_checkpoint(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.save_task_checkpoint(
        "weekly-399006",
        {
            "symbol": "399006",
            "status": "iterating",
            "completed_weeks": 0,
            "total_weeks": 10,
        },
    )

    repository.commit_iteration(
        "399006",
        result.audit,
        result.work_state,
        labels=(),
    )

    reopened = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    state = reopened.load_work_state("399006")
    assert state == result.work_state
    assert state.iteration_id == "I0001"
    assert reopened.list_completed_weeks("399006") == {"2026-W30"}
    assert reopened.list_completed_weeks("NDX") == set()
    model = reopened.current_model("399006")
    assert model.version == state.current_model_version
    assert model.work_version == "W0001"
    assert dict(model.weights) == dict(state.current_model_weights)
    metrics = reopened.metrics("399006")
    assert metrics.iteration_count == 1
    assert metrics.last_iteration == 1
    checkpoint = reopened.load_task_checkpoint("weekly-399006")
    assert checkpoint.last_iteration == 1
    assert checkpoint.completed_weeks == 1
    assert checkpoint.last_completed_week == "2026-W30"


def test_same_market_week_is_rejected_by_database_unique_constraint(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    first = _step()
    repository.commit_iteration(
        "399006", first.audit, first.work_state, labels=()
    )

    conflicting = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W30",
        cutoff_date=CUTOFF,
        source_data_max_date=CUTOFF,
        prediction=_prediction("399006", "2026-W30", CUTOFF),
        data_integrity=False,
    )
    with pytest.raises(module.DuplicateWeek, match="2026-W30"):
        repository.commit_iteration(
            "399006",
            conflicting.audit,
            conflicting.work_state,
            labels=(),
        )

    assert _row_counts(sessions)[:2] == (1, 1)


def test_mature_label_and_accepted_model_parent_chain_survive_restart(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    first = _step()
    repository.commit_iteration(
        "399006", first.audit, first.work_state, labels=()
    )
    second_cutoff = date(2026, 10, 23)
    mature = IterationLabel(
        iteration_id="I0001",
        instrument_code="399006",
        week_key="2026-W30",
        cutoff_date=CUTOFF,
        predicted_direction="up",
        operation_side="buy",
        probability=Decimal("0.8"),
        predicted_action_date=CUTOFF + timedelta(days=3),
        target_position=Decimal("0.75"),
        trade_count=1,
        status="mature_13w",
        observed_through=second_cutoff,
        visible_row_count=65,
        partial_weight=Decimal("1"),
        actual_direction="up",
        actual_turn_date=CUTOFF + timedelta(days=3),
        turning_deviation_sessions=0,
        loss_components=FrozenDict(
            {
                "turning_deviation": Decimal("0"),
                "direction_calibration": Decimal("0"),
                "adverse_excursion": Decimal("0"),
                "overtrading": Decimal("0"),
            }
        ),
        total_loss=Decimal("0"),
        direction_hit=True,
        calibration_formula="brier=(p-y)^2",
    )
    current = WindowMetrics(
        instrument_code="399006",
        window="expanding",
        sample_count=1,
        total_loss=Decimal("0.50"),
        direction_hit_rate=Decimal("0.60"),
        calibration_loss=Decimal("0.20"),
        overtrade_penalty=Decimal("0.20"),
    )
    candidate = WindowMetrics(
        instrument_code="399006",
        window="expanding",
        sample_count=1,
        total_loss=Decimal("0.49"),
        direction_hit_rate=Decimal("0.60"),
        calibration_loss=Decimal("0.20"),
        overtrade_penalty=Decimal("0.20"),
    )
    second = run_optimizer_step(
        first.work_state,
        feedback=(mature,),
        week_key="2026-W43",
        cutoff_date=second_cutoff,
        source_data_max_date=second_cutoff,
        prediction=IterationPrediction(
            iteration_id="I0002",
            instrument_code="399006",
            week_key="2026-W43",
            cutoff_date=second_cutoff,
            predicted_direction="up",
            operation_side="buy",
            probability=Decimal("0.8"),
            predicted_action_date=second_cutoff + timedelta(days=3),
            target_position=Decimal("0.75"),
            trade_count=1,
        ),
        current_metrics={"expanding": current},
        candidate_metrics={"expanding": candidate},
    )
    assert second.audit.accepted is True
    assert second.work_state.current_model_version == "M0002"

    repository.commit_iteration(
        "399006",
        second.audit,
        second.work_state,
        labels=(mature,),
    )

    reopened = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    assert reopened.load_work_state("399006") == second.work_state
    model = reopened.current_model("399006")
    assert model.version == "M0002"
    assert model.parent_version == "M0001"
    assert model.iteration_id == "I0002"
    metrics = reopened.metrics("399006")
    assert metrics.iteration_count == 2
    assert metrics.accepted_iterations == 1
    assert metrics.mature_count == 1


@pytest.mark.parametrize(
    "phase",
    ("serialize", "write_temp", "flush", "replace", "commit"),
)
def test_iteration_failure_phases_leave_no_readable_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {phase} failure")

    if phase == "serialize":
        monkeypatch.setattr(module, "_snapshot_bytes", fail)
    elif phase == "write_temp":
        monkeypatch.setattr(module, "_write_snapshot_temp", fail)
    elif phase == "flush":
        monkeypatch.setattr(repository, "_flush_session", fail)
    elif phase == "replace":
        monkeypatch.setattr(module, "_replace_snapshot", fail)
    else:
        monkeypatch.setattr(repository, "_commit_session", fail)

    with pytest.raises(OSError, match=phase):
        repository.commit_iteration(
            "399006", result.audit, result.work_state, labels=()
        )

    assert _row_counts(sessions) == (0, 0, 0, 0, 0)
    assert list((tmp_path / "weekly_analysis_v2").rglob("*.json")) == []
    assert list((tmp_path / "weekly_analysis_v2").rglob("*.tmp")) == []


def test_ambiguous_post_commit_failure_is_deterministically_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()

    def commit_then_fail(session: Session) -> None:
        session.commit()
        raise OSError("injected ambiguous commit failure")

    monkeypatch.setattr(repository, "_commit_session", commit_then_fail)
    with pytest.raises(OSError, match="ambiguous"):
        repository.commit_iteration(
            "399006", result.audit, result.work_state, labels=()
        )

    reopened = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    assert reopened.load_work_state("399006") == result.work_state
    snapshots = list((tmp_path / "weekly_analysis_v2").rglob("*.json"))
    assert len(snapshots) == 1


def test_markets_are_isolated_and_cross_market_commit_is_rejected(
    tmp_path: Path,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    growth = _step("399006")
    ndx = _step("NDX")
    repository.commit_iteration(
        "399006", growth.audit, growth.work_state, labels=()
    )
    repository.commit_iteration("NDX", ndx.audit, ndx.work_state, labels=())

    assert repository.load_work_state("399006") == growth.work_state
    assert repository.load_work_state("NDX") == ndx.work_state
    assert repository.list_completed_weeks("399006") == {"2026-W30"}
    assert repository.list_completed_weeks("NDX") == {"2026-W30"}

    with pytest.raises(module.CrossMarketRepositoryError):
        repository.commit_iteration(
            "NDX", growth.audit, growth.work_state, labels=()
        )


def test_corrupted_or_non_contiguous_persisted_chain_is_rejected(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    with sessions.begin() as session:
        row = session.scalar(
            select(V2AnalysisIteration).where(
                V2AnalysisIteration.iteration_number == 1
            )
        )
        assert row is not None
        corrupted = module._decode_json_mapping(row.output_payload)
        corrupted["work_state"]["instrument_code"] = "NDX"
        row.output_payload = module._json_mapping(corrupted)

    reopened = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    with pytest.raises(module.RepositoryCorruption, match="market|symbol"):
        reopened.load_work_state("399006")


def test_non_boolean_audit_flag_is_rejected_instead_of_coerced(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    with sessions.begin() as session:
        row = session.scalar(select(V2AnalysisIteration))
        assert row is not None
        corrupted = module._decode_json_mapping(row.output_payload)
        work_state = corrupted["work_state"]
        work_state["audits"][0]["data_integrity"] = "true"
        work_state["week_records"]["2026-W30"]["data_integrity"] = "true"
        row.output_payload = module._json_mapping(corrupted)

    reopened = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    with pytest.raises(module.RepositoryCorruption, match="work state"):
        reopened.load_work_state("399006")


def test_unreferenced_post_replace_snapshot_is_quarantined_on_restart(
    tmp_path: Path,
) -> None:
    module, sessions, _repository_instance = _repository(tmp_path)
    orphan = (
        tmp_path
        / "weekly_analysis_v2"
        / "399006"
        / "I0001"
        / "20260731T000000.000000Z-iteration-orphan.json"
    )
    orphan.parent.mkdir(parents=True)
    orphan.write_text('{"complete":"but-uncommitted"}\n', encoding="utf-8")

    module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )

    assert not orphan.exists()
    assert orphan.with_suffix(".json.orphan").exists()


def test_advice_snapshot_is_complete_hashable_and_recoverable(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    advice = _advice(result.work_state)

    advice_id = repository.save_advice("399006", advice)

    reopened = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    record = reopened.latest_advice("399006")
    assert record is not None
    assert record.id == advice_id
    assert record.payload["recommendation"] == advice.recommendation
    raw_payload = json.loads(
        record.snapshot_path.read_text(encoding="utf-8")
    )
    payload = module._decode_json_mapping(raw_payload)
    required = {
        "symbol",
        "analysis_time",
        "data_cutoff",
        "source_data_max_date",
        "source_hash",
        "seed",
        "iteration_id",
        "work_version",
        "model_version",
        "current_metrics",
        "current_position",
        "target_position",
        "direction_probabilities",
        "probability",
        "confidence",
        "market_state",
        "batches",
        "fund_etf_ratio",
        "provenance",
        "integrity",
        "snapshot_hash",
    }
    assert required <= payload.keys()
    assert payload["symbol"] == "399006"
    assert payload["iteration_id"] == "I0001"
    assert payload["work_version"] == "W0001"
    assert payload["model_version"] == result.work_state.current_model_version
    assert payload["source_data_max_date"] <= payload["data_cutoff"]
    # Hashes bind the on-disk tagged representation, not its decoded view.
    expected_hash = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in raw_payload.items()
                if key != "snapshot_hash"
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert payload["snapshot_hash"] == expected_hash
    assert record.snapshot_hash == expected_hash
    assert "399006" in str(record.snapshot_path)
    assert "I0001" in str(record.snapshot_path)


def test_repository_never_writes_legacy_model_iterations(
    tmp_path: Path,
) -> None:
    _module, _sessions_factory, repository = _repository(tmp_path)
    legacy = tmp_path / "model_iterations" / "399006" / "manifest.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"legacy":true}\n', encoding="utf-8")
    before = (
        hashlib.sha256(legacy.read_bytes()).hexdigest(),
        legacy.stat().st_mtime_ns,
    )

    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )

    after = (
        hashlib.sha256(legacy.read_bytes()).hexdigest(),
        legacy.stat().st_mtime_ns,
    )
    assert after == before


def test_missing_symbol_and_unknown_checkpoint_are_explicit(tmp_path: Path) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)

    with pytest.raises(module.UnknownMarket):
        repository.load_work_state("MISSING")
    with pytest.raises(module.CheckpointNotFound):
        repository.load_task_checkpoint("missing-task")
    with pytest.raises(ValueError, match="W version"):
        repository.save_task_checkpoint(
            "bad-version",
            {
                "symbol": "399006",
                "status": "iterating",
                "completed_weeks": 0,
                "last_work_version": "Wbroken",
            },
        )


def test_repository_contract_is_exported_from_weekly_analysis_package() -> None:
    module = _repository_api()

    assert (
        weekly_analysis.WeeklyAnalysisRepository
        is module.WeeklyAnalysisRepository
    )
    assert weekly_analysis.ModelVersion is module.ModelVersion
    assert weekly_analysis.ModelMetrics is module.ModelMetrics
    assert weekly_analysis.TaskCheckpoint is module.TaskCheckpoint


@pytest.mark.parametrize(
    "tamper_target",
    (
        "input_payload",
        "metrics",
        "embedded_snapshot",
        "snapshot_hash",
        "model_audit",
    ),
)
def test_db_tamper_accepted_counterexample_is_rejected_by_every_read_path(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    with sessions.begin() as session:
        iteration_row = session.scalar(select(V2AnalysisIteration))
        model_row = session.scalar(select(V2ModelVersion))
        assert iteration_row is not None and model_row is not None
        if tamper_target == "input_payload":
            value = dict(iteration_row.input_payload)
            value["instrument_code"] = "NDX"
            iteration_row.input_payload = value
        elif tamper_target == "metrics":
            value = dict(iteration_row.metrics)
            value["accepted"] = not result.audit.accepted
            iteration_row.metrics = value
        elif tamper_target == "embedded_snapshot":
            value = json.loads(json.dumps(iteration_row.audit_data))
            value["snapshot"]["work_version"] = "W9999"
            iteration_row.audit_data = value
        elif tamper_target == "snapshot_hash":
            value = dict(iteration_row.audit_data)
            value["snapshot_hash"] = "b" * 64
            iteration_row.audit_data = value
        else:
            value = dict(model_row.audit_data)
            value["work_version"] = "W9999"
            value["iteration_id"] = "I9999"
            model_row.audit_data = value

    reopened = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    readers = (
        lambda: reopened.list_completed_weeks("399006"),
        lambda: reopened.load_work_state("399006"),
        lambda: reopened.current_model("399006"),
    )
    for reader in readers:
        with pytest.raises(module.RepositoryCorruption):
            reader()


def test_checkpoint_tamper_accepted_counterexample_is_rejected(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    repository.save_task_checkpoint(
        "tampered-task",
        {
            "symbol": "399006",
            "status": "iterating",
            "completed_weeks": 0,
            "total_weeks": 2,
        },
    )
    with sessions.begin() as session:
        row = session.scalar(select(V2AnalysisTask))
        assert row is not None
        value = dict(row.result_payload)
        value["symbol"] = "NDX"
        value["status"] = "completed"
        row.result_payload = value

    with pytest.raises(module.RepositoryCorruption, match="checkpoint"):
        repository.load_task_checkpoint("tampered-task")


@pytest.mark.parametrize(
    "progress",
    (
        {
            "symbol": "399006",
            "status": "invented",
            "completed_weeks": 0,
        },
        {
            "symbol": "399006",
            "status": "iterating",
            "completed_weeks": True,
        },
        {
            "symbol": "399006",
            "status": "iterating",
            "completed_weeks": 1,
            "last_iteration": 0,
        },
    ),
)
def test_checkpoint_contract_rejects_invalid_status_bool_and_inconsistent_counts(
    tmp_path: Path,
    progress: dict[str, object],
) -> None:
    _module, _sessions_factory, repository = _repository(tmp_path)

    with pytest.raises(ValueError):
        repository.save_task_checkpoint("invalid-progress", progress)


def test_checkpoint_progress_cannot_regress_after_atomic_iteration(
    tmp_path: Path,
) -> None:
    _module, _sessions_factory, repository = _repository(tmp_path)
    repository.save_task_checkpoint(
        "monotonic-task",
        {
            "symbol": "399006",
            "status": "iterating",
            "completed_weeks": 0,
            "total_weeks": 2,
        },
    )
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )

    with pytest.raises(ValueError, match="regress|monotonic"):
        repository.save_task_checkpoint(
            "monotonic-task",
            {
                "symbol": "399006",
                "status": "iterating",
                "completed_weeks": 0,
                "total_weeks": 2,
            },
        )


def test_advice_provenance_accepted_counterexample_is_rejected(
    tmp_path: Path,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    current_advice = _advice(result.work_state)
    future_advice = replace(
        current_advice,
        data_cutoff_date=CUTOFF + timedelta(days=7),
        batches=tuple(
            replace(
                batch,
                expected_date=batch.expected_date + timedelta(days=7),
            )
            for batch in current_advice.batches
        ),
    )

    with pytest.raises(module.RepositoryCorruption, match="cutoff|provenance"):
        repository.save_advice("399006", future_advice)


def test_latest_advice_rejects_tampered_payload_market_and_provenance(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    repository.save_advice("399006", _advice(result.work_state))
    with sessions.begin() as session:
        row = session.scalar(select(V2AdviceHistory))
        assert row is not None
        payload = dict(row.payload)
        payload["instrument_code"] = "NDX"
        row.payload = payload

    with pytest.raises(module.RepositoryCorruption, match="advice|market"):
        repository.latest_advice("399006")


@pytest.mark.parametrize("failure_phase", ("flush", "commit"))
def test_cleanup_half_json_counterexample_is_quarantined_when_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failure_phase} failure")

    monkeypatch.setattr(module, "_safe_unlink", lambda _path: None)
    if failure_phase == "flush":
        monkeypatch.setattr(repository, "_flush_session", fail)
    else:
        monkeypatch.setattr(repository, "_commit_session", fail)

    with pytest.raises(OSError, match=failure_phase):
        repository.commit_iteration(
            "399006", result.audit, result.work_state, labels=()
        )

    audit_root = tmp_path / "weekly_analysis_v2"
    assert list(audit_root.rglob("*.json")) == []
    assert list(audit_root.rglob("*.tmp")) == []
    assert list(audit_root.rglob("*.orphan"))


def test_save_advice_maps_only_provenance_duplicate_to_duplicate_advice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    advice = _advice(result.work_state)

    def other_integrity_error(_session: Session) -> None:
        raise IntegrityError(
            "INSERT v2_advice_history",
            {},
            RuntimeError("foreign key failure"),
        )

    monkeypatch.setattr(repository, "_flush_session", other_integrity_error)
    with pytest.raises(module.RepositoryError) as captured:
        repository.save_advice("399006", advice)
    assert type(captured.value) is module.RepositoryError
    assert isinstance(captured.value.__cause__, IntegrityError)


def test_same_advice_provenance_is_the_only_duplicate_advice_case(
    tmp_path: Path,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    advice = _advice(result.work_state)
    repository.save_advice("399006", advice)

    with pytest.raises(module.DuplicateAdvice):
        repository.save_advice("399006", advice)


def test_duplicate_labels_accepted_counterexample_is_rejected(
    tmp_path: Path,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    first = _step()
    repository.commit_iteration(
        "399006", first.audit, first.work_state, labels=()
    )
    label = _mature_label()
    second = _second_step(first, label)

    with pytest.raises(module.RepositoryCorruption, match="duplicate label"):
        repository.commit_iteration(
            "399006",
            second.audit,
            second.work_state,
            labels=(label, label),
        )


def test_competing_init_quarantine_counterexample_waits_for_inflight_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    replaced = Event()
    allow_commit = Event()
    init_done = Event()
    errors: list[BaseException] = []
    real_replace = module._replace_snapshot

    def pausing_replace(temp_path: Path, final_path: Path) -> None:
        real_replace(temp_path, final_path)
        replaced.set()
        if not allow_commit.wait(timeout=10):
            raise TimeoutError("test did not release commit")

    monkeypatch.setattr(module, "_replace_snapshot", pausing_replace)

    def commit_worker() -> None:
        try:
            repository.commit_iteration(
                "399006", result.audit, result.work_state, labels=()
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def init_worker() -> None:
        try:
            module.WeeklyAnalysisRepository(
                sessions,
                audit_root=tmp_path / "weekly_analysis_v2",
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            init_done.set()

    commit_thread = Thread(target=commit_worker)
    commit_thread.start()
    assert replaced.wait(timeout=10)
    init_thread = Thread(target=init_worker)
    init_thread.start()
    competing_init_waited = not init_done.wait(timeout=0.25)
    allow_commit.set()
    commit_thread.join(timeout=10)
    init_thread.join(timeout=10)

    assert competing_init_waited
    assert not commit_thread.is_alive()
    assert not init_thread.is_alive()
    assert errors == []
    audit_root = tmp_path / "weekly_analysis_v2"
    assert len(list(audit_root.rglob("*.json"))) == 1
    assert list(audit_root.rglob("*.orphan")) == []


def test_two_repository_instances_same_week_have_one_winner(
    tmp_path: Path,
) -> None:
    module, sessions, first_repository = _repository(tmp_path)
    second_repository = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly_analysis_v2",
    )
    result = _step()

    def commit(repository) -> str:
        try:
            repository.commit_iteration(
                "399006", result.audit, result.work_state, labels=()
            )
        except module.DuplicateWeek:
            return "duplicate"
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            executor.map(commit, (first_repository, second_repository))
        )

    assert sorted(outcomes) == ["duplicate", "ok"]


def test_checkpoint_compare_and_swap_rejects_stale_status_overwrite_across_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, sessions, stale_repository = _repository(tmp_path)
    winning_repository = module.WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "other-audit-root",
    )
    stale_repository.save_task_checkpoint(
        "cas-task",
        {
            "symbol": "399006",
            "status": "queued",
            "completed_weeks": 0,
            "total_weeks": 0,
        },
    )
    stale_read = Event()
    allow_stale_write = Event()
    real_from_mapping = module._checkpoint_from_mapping

    def pause_after_stale_read(*args, **kwargs):
        checkpoint = real_from_mapping(*args, **kwargs)
        if current_thread().name == "stale-checkpoint-writer":
            stale_read.set()
            if not allow_stale_write.wait(timeout=10):
                raise TimeoutError("test did not release stale checkpoint writer")
        return checkpoint

    monkeypatch.setattr(
        module, "_checkpoint_from_mapping", pause_after_stale_read
    )
    stale_errors: list[BaseException] = []

    def write_stale_running() -> None:
        try:
            stale_repository.save_task_checkpoint(
                "cas-task",
                {
                    "symbol": "399006",
                    "status": "preparing_data",
                    "completed_weeks": 0,
                    "total_weeks": 0,
                },
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            stale_errors.append(exc)

    stale_thread = Thread(
        target=write_stale_running,
        name="stale-checkpoint-writer",
    )
    stale_thread.start()
    assert stale_read.wait(timeout=10)
    for status in (
        "preparing_data",
        "building_features",
        "iterating",
        "validating",
        "generating_advice",
        "completed",
    ):
        winning_repository.save_task_checkpoint(
            "cas-task",
            {
                "symbol": "399006",
                "status": status,
                "completed_weeks": 0,
                "total_weeks": 0,
            },
        )
    allow_stale_write.set()
    stale_thread.join(timeout=10)

    assert not stale_thread.is_alive()
    assert len(stale_errors) == 1
    assert isinstance(stale_errors[0], ValueError)
    assert "concurrent" in str(stale_errors[0]).lower()
    assert winning_repository.load_task_checkpoint("cas-task").status == "completed"


def test_commit_preserves_primary_error_when_rollback_and_unlink_also_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()

    def fail_commit(_session: Session) -> None:
        raise OSError("primary commit failure")

    def fail_rollback(_session: Session) -> None:
        raise RuntimeError("secondary rollback failure")

    monkeypatch.setattr(repository, "_commit_session", fail_commit)
    monkeypatch.setattr(module, "_safe_unlink", lambda _path: None)
    with monkeypatch.context() as rollback_patch:
        rollback_patch.setattr(Session, "rollback", fail_rollback)
        with pytest.raises(OSError, match="primary commit") as captured:
            repository.commit_iteration(
                "399006", result.audit, result.work_state, labels=()
            )

    notes = tuple(getattr(captured.value, "__notes__", ()))
    assert any("secondary rollback failure" in note for note in notes)
    audit_root = tmp_path / "weekly_analysis_v2"
    assert list(audit_root.rglob("*.json")) == []
    assert list(audit_root.rglob("*.tmp")) == []
    assert list(audit_root.rglob("*.orphan"))


def test_save_advice_preserves_primary_error_when_rollback_and_unlink_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )

    def fail_commit(_session: Session) -> None:
        raise OSError("primary advice commit failure")

    def fail_rollback(_session: Session) -> None:
        raise RuntimeError("secondary advice rollback failure")

    monkeypatch.setattr(repository, "_commit_session", fail_commit)
    monkeypatch.setattr(module, "_safe_unlink", lambda _path: None)
    with monkeypatch.context() as rollback_patch:
        rollback_patch.setattr(Session, "rollback", fail_rollback)
        with pytest.raises(
            OSError, match="primary advice commit failure"
        ) as captured:
            repository.save_advice(
                "399006", _advice(result.work_state)
            )

    notes = tuple(getattr(captured.value, "__notes__", ()))
    assert any(
        "secondary advice rollback failure" in note for note in notes
    )
    audit_root = tmp_path / "weekly_analysis_v2"
    assert list(audit_root.rglob("*-advice-*.json")) == []
    assert list(audit_root.rglob("*-advice-*.tmp")) == []
    assert list(audit_root.rglob("*-advice-*.orphan"))


def test_save_advice_preserves_primary_error_when_session_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    target_sessions: set[int] = set()
    real_close = Session.close

    def fail_commit(session: Session) -> None:
        target_sessions.add(id(session))
        raise OSError("primary advice close-path failure")

    def close_then_fail(session: Session) -> None:
        real_close(session)
        if id(session) in target_sessions:
            raise RuntimeError("secondary advice close failure")

    monkeypatch.setattr(repository, "_commit_session", fail_commit)
    monkeypatch.setattr(Session, "close", close_then_fail)
    with pytest.raises(
        OSError, match="primary advice close-path failure"
    ) as captured:
        repository.save_advice("399006", _advice(result.work_state))

    notes = tuple(getattr(captured.value, "__notes__", ()))
    assert any("secondary advice close failure" in note for note in notes)
    audit_root = tmp_path / "weekly_analysis_v2"
    assert list(audit_root.rglob("*-advice-*.json")) == []
    assert list(audit_root.rglob("*-advice-*.tmp")) == []


def test_latest_advice_rejects_newest_row_with_corrupt_status(
    tmp_path: Path,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    repository.save_advice("399006", _advice(result.work_state))
    with sessions.begin() as session:
        row = session.scalar(select(V2AdviceHistory))
        assert row is not None
        row.status = "withdrawn"

    with pytest.raises(module.RepositoryCorruption, match="status|advice"):
        repository.latest_advice("399006")


@pytest.mark.parametrize(
    "tamper_target",
    ("target_payload", "activated_at", "artifact_path"),
)
def test_week_sample_and_model_authoritative_mirrors_are_bound(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    with sessions.begin() as session:
        sample = session.scalar(select(V2WeekSample))
        model = session.scalar(select(V2ModelVersion))
        assert sample is not None and model is not None
        if tamper_target == "target_payload":
            sample.target_payload = {"tampered": True}
        elif tamper_target == "activated_at":
            assert model.activated_at is not None
            model.activated_at += timedelta(seconds=1)
        else:
            model.artifact_path = "tampered/model.bin"

    with pytest.raises(module.RepositoryCorruption):
        repository.load_work_state("399006")


@pytest.mark.parametrize(
    "tamper_target",
    (
        "label_date",
        "maturity_date",
        "label_value",
        "label",
        "audit_data",
    ),
)
def test_label_authoritative_mirrors_are_bound(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    first = _step()
    repository.commit_iteration(
        "399006", first.audit, first.work_state, labels=()
    )
    label = _mature_label()
    second = _second_step(first, label)
    repository.commit_iteration(
        "399006", second.audit, second.work_state, labels=(label,)
    )
    with sessions.begin() as session:
        row = session.scalar(select(V2IterationLabel))
        assert row is not None
        if tamper_target == "label_date":
            assert row.label_date is not None
            row.label_date += timedelta(days=1)
        elif tamper_target == "maturity_date":
            assert row.maturity_date is not None
            row.maturity_date += timedelta(days=1)
        elif tamper_target == "label_value":
            row.label_value = Decimal("0.25")
        elif tamper_target == "label":
            row.label = "down"
        else:
            audit = dict(row.audit_data)
            audit["iteration_id"] = "I9999"
            row.audit_data = audit

    with pytest.raises(module.RepositoryCorruption):
        repository.load_work_state("399006")


def test_checkpoint_tagged_codec_roundtrips_types_without_guessing_strings(
    tmp_path: Path,
) -> None:
    _module, _sessions_factory, repository = _repository(tmp_path)
    moment = datetime(2026, 7, 31, 9, 15, tzinfo=timezone.utc)
    repository.save_task_checkpoint(
        "typed-checkpoint",
        {
            "symbol": "399006",
            "status": "iterating",
            "completed_weeks": 0,
            "total_weeks": 0,
            "decimal": Decimal("0.50"),
            "numeric_string": "0.5",
            "day": CUTOFF,
            "moment": moment,
            "frozen": FrozenDict({"ratio": Decimal("0.50")}),
            "tuple_value": (Decimal("0.50"), "0.5"),
        },
    )

    restored = repository.load_task_checkpoint("typed-checkpoint").progress
    assert restored["decimal"] == Decimal("0.50")
    assert isinstance(restored["decimal"], Decimal)
    assert restored["decimal"].as_tuple().exponent == -2
    assert restored["numeric_string"] == "0.5"
    assert isinstance(restored["numeric_string"], str)
    assert restored["day"] == CUTOFF
    assert type(restored["day"]) is date
    assert restored["moment"] == moment
    assert isinstance(restored["moment"], datetime)
    assert isinstance(restored["frozen"], FrozenDict)
    assert restored["frozen"]["ratio"].as_tuple().exponent == -2
    assert restored["tuple_value"] == (Decimal("0.50"), "0.5")
    assert type(restored["tuple_value"]) is tuple


def test_advice_tagged_codec_roundtrips_decimal_date_tuple_and_string(
    tmp_path: Path,
) -> None:
    _module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    advice = _advice(result.work_state)
    probabilities = dict(advice.direction_probabilities)
    probabilities[advice.direction] = Decimal("85.00")
    advice = replace(
        advice,
        direction_probabilities=FrozenDict(probabilities),
        probability=Decimal("85.00"),
        audit_notes=advice.audit_notes + ("0.5",),
    )
    repository.save_advice("399006", advice)

    record = repository.latest_advice("399006")
    assert record is not None
    payload = record.payload
    assert payload["probability"] == Decimal("85.00")
    assert isinstance(payload["probability"], Decimal)
    assert payload["probability"].as_tuple().exponent == -2
    assert payload["audit_notes"][-1] == "0.5"
    assert isinstance(payload["audit_notes"][-1], str)
    assert payload["data_cutoff_date"] == CUTOFF
    assert type(payload["data_cutoff_date"]) is date
    assert type(payload["batches"]) is tuple
    assert type(payload["batches"][0]["expected_date"]) is date


def _commit_synthetic_chain(repository, count: int):
    state = seed_model("399006")
    for number in range(1, count + 1):
        cutoff = CUTOFF + timedelta(days=7 * (number - 1))
        iso_year, iso_week, _weekday = cutoff.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"
        prediction = replace(
            _prediction("399006", week_key, cutoff),
            iteration_id=f"I{number:04d}",
        )
        result = run_optimizer_step(
            state,
            feedback=(),
            week_key=week_key,
            cutoff_date=cutoff,
            source_data_max_date=cutoff,
            prediction=prediction,
        )
        repository.commit_iteration(
            "399006", result.audit, result.work_state, labels=()
        )
        state = result.work_state
    return state


def test_hundred_week_recovery_has_constant_sql_linear_parsing_and_incremental_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    expected = _commit_synthetic_chain(repository, 100)
    snapshot_paths = tuple(
        (tmp_path / "weekly_analysis_v2" / "399006").rglob("*.json")
    )
    assert len(snapshot_paths) == 100
    snapshot_sizes = tuple(path.stat().st_size for path in snapshot_paths)
    assert max(snapshot_sizes) <= min(snapshot_sizes) * 3
    for path in snapshot_paths:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        assert "work_state" not in snapshot

    with sessions() as session:
        rows = tuple(
            session.scalars(
                select(V2AnalysisIteration).order_by(
                    V2AnalysisIteration.iteration_number
                )
            )
        )
        assert sum("work_state" in row.output_payload for row in rows) == 1

    counts = {"queries": 0, "work_state": 0, "step_audit": 0}
    engine = sessions.kw["bind"]
    init_queries = 0

    def count_init_query(*_args, **_kwargs) -> None:
        nonlocal init_queries
        init_queries += 1

    event.listen(engine, "before_cursor_execute", count_init_query)
    try:
        repository = module.WeeklyAnalysisRepository(
            sessions,
            audit_root=tmp_path / "weekly_analysis_v2",
        )
    finally:
        event.remove(engine, "before_cursor_execute", count_init_query)
    assert init_queries <= 4

    real_work_state_parser = module._work_state_from_dict
    real_step_audit_parser = module._step_audit_from_dict

    def count_query(*_args, **_kwargs) -> None:
        counts["queries"] += 1

    def count_work_state(value):
        counts["work_state"] += 1
        return real_work_state_parser(value)

    def count_step_audit(value):
        counts["step_audit"] += 1
        return real_step_audit_parser(value)

    monkeypatch.setattr(module, "_work_state_from_dict", count_work_state)
    monkeypatch.setattr(module, "_step_audit_from_dict", count_step_audit)
    event.listen(engine, "before_cursor_execute", count_query)
    try:
        restored = repository.load_work_state("399006")
    finally:
        event.remove(engine, "before_cursor_execute", count_query)

    assert restored == expected
    assert counts["queries"] <= 6
    assert counts["work_state"] == 1
    assert counts["step_audit"] <= 100


def test_compact_audit_cache_keys_are_stable_and_market_scoped() -> None:
    module = _repository_api()
    growth = _step("399006")
    ndx = _step("NDX")
    cache = {}

    growth_raw = module._work_state_storage_dict(
        growth.work_state,
        audit_cache=cache,
    )
    ndx_raw = module._work_state_storage_dict(
        ndx.work_state,
        audit_cache=cache,
    )

    assert len(cache) == 2
    assert growth_raw["audits"][0]["instrument_code"] == "399006"
    assert ndx_raw["audits"][0]["instrument_code"] == "NDX"
    assert (
        module._audit_storage_cache_key(growth.audit)
        != module._audit_storage_cache_key(ndx.audit)
    )


def test_each_public_read_and_save_advice_validates_the_chain_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    calls = 0
    real_load = repository._load_work_state_locked

    def counted_load(symbol: str):
        nonlocal calls
        calls += 1
        return real_load(symbol)

    monkeypatch.setattr(repository, "_load_work_state_locked", counted_load)
    readers = (
        lambda: repository.list_completed_weeks("399006"),
        lambda: repository.current_model("399006"),
        lambda: repository.metrics("399006"),
    )
    for reader in readers:
        calls = 0
        reader()
        assert calls == 1

    calls = 0
    repository.save_advice("399006", _advice(result.work_state))
    assert calls == 1


def test_historical_completed_task_checkpoint_remains_bound_to_its_own_head(
    tmp_path: Path,
) -> None:
    module, _sessions_factory, repository = _repository(tmp_path)
    first_state = _commit_synthetic_chain(repository, 1)
    first_audit = first_state.audits[-1]
    checkpoint = module.TaskCheckpoint(
        task_id="historical-I0001",
        symbol="399006",
        status="completed",
        completed_weeks=1,
        total_weeks=1,
        last_iteration=1,
        last_work_version="W0001",
        last_model_version=first_state.current_model_version,
        last_completed_week=first_audit.week_key,
        progress=FrozenDict({"stage": "completed"}),
    )
    repository.save_task_checkpoint(checkpoint.task_id, checkpoint)

    cutoff = CUTOFF + timedelta(days=7)
    second = run_optimizer_step(
        first_state,
        feedback=(),
        week_key="2026-W31",
        cutoff_date=cutoff,
        source_data_max_date=cutoff,
        prediction=replace(
            _prediction("399006", "2026-W31", cutoff),
            iteration_id="I0002",
        ),
    )
    repository.commit_iteration(
        "399006",
        second.audit,
        second.work_state,
        labels=(),
    )

    assert repository.load_task_checkpoint(checkpoint.task_id) == checkpoint
    tampered = replace(checkpoint, last_model_version="M9999")
    with pytest.raises(ValueError, match="diverge|chain|checkpoint"):
        repository.save_task_checkpoint(checkpoint.task_id, tampered)


def test_advice_generations_preserve_history_and_deduplicate_request_key(
    tmp_path: Path,
) -> None:
    _module, sessions, repository = _repository(tmp_path)
    result = _step()
    repository.commit_iteration(
        "399006", result.audit, result.work_state, labels=()
    )
    advice = _advice(result.work_state)

    first_id = repository.save_advice(
        "399006", advice, generation_key="weekly:I0001"
    )
    repeated_id = repository.save_advice(
        "399006", advice, generation_key="position:20"
    )
    deduplicated_id = repository.save_advice(
        "399006", advice, generation_key="position:20"
    )
    changed_position_id = repository.save_advice(
        "399006",
        advice,
        generation_key="position:25",
    )

    with sessions() as session:
        rows = tuple(
            session.scalars(
                select(V2AdviceHistory).order_by(
                    V2AdviceHistory.advice_generation
                )
            )
        )
    assert first_id != repeated_id
    assert deduplicated_id == repeated_id
    assert changed_position_id not in {first_id, repeated_id}
    assert [row.advice_generation for row in rows] == [1, 2, 3]
    assert [row.generation_key for row in rows] == [
        "weekly:I0001",
        "position:20",
        "position:25",
    ]
    assert all(row.iteration_number == 1 for row in rows)
    assert all(row.work_number == 1 for row in rows)
    assert all(row.model_number == 1 for row in rows)


@pytest.mark.parametrize("write_kind", ("iteration", "advice"))
def test_stale_task_claim_cannot_commit_iteration_or_advice(
    tmp_path: Path,
    write_kind: str,
) -> None:
    module, sessions, repository = _repository(tmp_path)
    result = _step()
    if write_kind == "advice":
        repository.commit_iteration(
            "399006", result.audit, result.work_state, labels=()
        )

    checkpoint, _created = repository.get_or_create_task(
        "399006",
        f"stale-guard:{write_kind}",
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
    )
    claimed = repository.claim_task_execution(
        checkpoint.task_id,
        worker_token="worker-a",
        lease_seconds=1,
    )
    assert claimed is not None
    with sessions.begin() as session:
        row = session.scalar(
            select(V2AnalysisTask).where(
                V2AnalysisTask.task_key == checkpoint.task_id
            )
        )
        assert row is not None
        row.lease_expires_at = datetime.now(timezone.utc) - timedelta(
            seconds=1
        )
    repository.recover_interrupted_tasks("worker-a lease expired")
    replacement = repository.claim_task_execution(
        checkpoint.task_id,
        worker_token="worker-b",
        lease_seconds=5,
    )
    assert replacement is not None
    guard = module.TaskExecutionGuard(
        task_id=checkpoint.task_id,
        worker_token="worker-a",
    )

    with pytest.raises(
        module.TaskClaimLost,
        match="claim|lease|worker|execution",
    ):
        if write_kind == "iteration":
            repository.commit_iteration(
                "399006",
                result.audit,
                result.work_state,
                labels=(),
                execution_guard=guard,
            )
        else:
            repository.save_advice(
                "399006",
                _advice(result.work_state),
                generation_key="stale-worker",
                execution_guard=guard,
            )

    assert repository.list_completed_weeks("399006") == (
        set() if write_kind == "iteration" else {"2026-W30"}
    )
    if write_kind == "advice":
        with sessions() as session:
            assert session.scalar(
                select(func.count(V2AdviceHistory.id))
            ) == 0
