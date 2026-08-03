"""Append-only weekly runtime for the V3.4.1 model core."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import math
import zlib
from threading import Lock
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.models import (
    Instrument,
    V341AnalysisRun,
    V341CandidateTrial,
    V341FeatureSnapshot as V341FeatureSnapshotRow,
    V341Forecast as V341ForecastRow,
    V341ForecastCalibratorLink,
    V341ForecastEvaluation,
    V341ModelHealthSnapshot,
    V341ModelVersion,
    V341OptimizerState,
    V341OuterEvaluationBlock,
    V341ProbabilityCalibrator,
    V341RandomPlan,
    V341ResidualRecord,
    V341ScenarioAdapter,
    V341TrainingIteration,
    V341TrainingProfile,
    V341TrainingRun,
    utc_now,
)
from backend.app.daemon_executor import DaemonThreadPoolExecutor
from backend.app.services.market_calendar import CalendarProvider
from backend.app.services.v34_training_service import (
    V34TrainingSample,
    build_training_samples,
    eligible_fully_matured,
    purged_walk_forward_folds,
)
from backend.app.services.v341_feature_service import (
    FEATURE_SET_CORE,
    FEATURE_SET_EXTENDED,
    FEATURE_SET_VERSION,
    FeatureManifest,
    V341FeatureService,
    build_feature_manifest,
)
from backend.app.services.v341_scenario_service import (
    RANDOM_PLAN_VERSION,
    RESIDUAL_SCHEMA_VERSION,
    SCENARIO_ADAPTER_VERSION,
    THRESHOLD_FORMULA_VERSION,
    V341ScenarioForecast,
    apply_temperature,
    fit_temperature,
    generate_scenario_forecast,
    prior_standardized_residual_pool,
)
from backend.app.services.v341_training_service import (
    CANDIDATE_RANDOM_PLAN_VERSION,
    LOSS_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    V341ModelState,
    V341PathForecast,
    V341TrainingService,
    candidate_alphas,
    conservative_independent_sample_count,
    multiclass_brier,
    paired_block_bootstrap_interval,
    state_from_payload,
    state_to_payload,
    weighted_interval_score,
)


FORMAL_TRAINING_WEEKS = 520
MINIMUM_FORMAL_BOOTSTRAP_WEEKS = 400
MINIMUM_TRAINING_WEEKS = 30
FEATURE_WARMUP_WEEKS = 52
MIN_NEW_MATURED_FOR_CANDIDATE = 4
STRUCTURE_AUDIT_INTERVAL_WEEKS = 26
MAX_BACKLOG_WEEKS = 8
MODEL_FAMILY = "MULTI_HORIZON_WEIGHTED_RIDGE_V341"
EVALUATION_VERSION = "V3.4.1_EVALUATION_1"
CALIBRATION_VERSION = "V3.4.1_TEMPERATURE_1"


class V341RuntimeError(RuntimeError):
    pass


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(float(value), 8)))


def count_new_matured_anchors(
    matured: Sequence[V34TrainingSample],
    last_consumed_anchor: date | None,
) -> tuple[date, ...]:
    """Count mature events independently from the current rolling-window size."""

    return tuple(
        sample.anchor_date
        for sample in sorted(matured, key=lambda item: item.anchor_date)
        if last_consumed_anchor is None or sample.anchor_date > last_consumed_anchor
    )


def _probability_vector(payload: Mapping[str, Any], key: str) -> np.ndarray:
    values = payload[key]
    return np.asarray(
        [float(values["up"]), float(values["sideways"]), float(values["down"])],
        dtype=float,
    )


def _actual_class(value: float, threshold: float) -> tuple[int, str]:
    if value > threshold:
        return 0, "up"
    if value < -threshold:
        return 2, "down"
    return 1, "sideways"


def promotion_from_paired_losses(
    champion_losses: Sequence[float],
    candidate_losses: Sequence[float],
    *,
    seed: int,
    fold_lengths: Sequence[int],
    hard_gate_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the sealed outer-block gate without editing measured losses."""

    champion = np.asarray(champion_losses, dtype=float)
    candidate = np.asarray(candidate_losses, dtype=float)
    if champion.shape != candidate.shape or champion.ndim != 1 or not len(champion):
        raise ValueError("paired promotion losses must be aligned and non-empty")
    paired = candidate - champion
    relative_improvement = float(
        (np.mean(champion) - np.mean(candidate)) / max(np.mean(champion), 1e-9)
    )
    worst_relative_degradation = float(
        np.max(paired / np.maximum(np.abs(champion), 1e-9))
    )
    interval: tuple[float, float] | None = None
    if len(paired) >= 13:
        interval = paired_block_bootstrap_interval(
            paired,
            block_length=13,
            fold_lengths=fold_lengths,
            seed=seed,
            draws=2000,
        )
    champion_coverage = float(hard_gate_metrics["champion_coverage"])
    candidate_coverage = float(hard_gate_metrics["candidate_coverage"])
    champion_turning = float(hard_gate_metrics["champion_turning_accuracy"])
    candidate_turning = float(hard_gate_metrics["candidate_turning_accuracy"])
    coefficient_norm_ratio = float(hard_gate_metrics["coefficient_norm_ratio"])
    gates = {
        "mean_relative_improvement_at_least_1pct": relative_improvement >= 0.01,
        "worst_anchor_degradation_at_most_0_5pct": worst_relative_degradation <= 0.005,
        "majority_of_outer_anchors_improved": float(np.mean(paired < 0.0)) > 0.5,
        "paired_13w_block_bootstrap_upper_below_zero": (
            interval is not None and interval[1] < 0.0
        ),
        "coverage_not_materially_worse": (
            abs(candidate_coverage - 0.80) <= abs(champion_coverage - 0.80) + 0.03
        ),
        "coverage_not_severely_distorted": 0.50 <= candidate_coverage <= 0.98,
        "turning_type_not_materially_worse": candidate_turning + 0.05 >= champion_turning,
        "coefficient_norm_stable": math.isfinite(coefficient_norm_ratio)
        and coefficient_norm_ratio <= 1.25,
    }
    return {
        "promoted": all(gates.values()),
        "gates": gates,
        "relative_improvement": relative_improvement,
        "worst_relative_degradation": worst_relative_degradation,
        "paired_loss_differences": paired.tolist(),
        "bootstrap_interval_90pct": interval,
        "hard_gate_metrics": dict(hard_gate_metrics),
        "bootstrap_fold_lengths": list(fold_lengths),
    }


class V341RuntimeService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        calendar: CalendarProvider,
        now_provider: Callable[[], datetime] = utc_now,
        scenario_count: int = 1200,
        maximum_backlog_weeks: int = MAX_BACKLOG_WEEKS,
    ) -> None:
        self.sessions = sessions
        self.calendar = calendar
        self.now_provider = now_provider
        self.scenario_count = int(scenario_count)
        self.maximum_backlog_weeks = int(maximum_backlog_weeks)
        self.features = V341FeatureService()
        self.training = V341TrainingService()
        self._executor = DaemonThreadPoolExecutor(
            max_workers=1, thread_name_prefix="v341-model"
        )
        self._analysis_executor = DaemonThreadPoolExecutor(
            max_workers=2, thread_name_prefix="v341-analysis"
        )
        self._locks = {market: Lock() for market in ("399006", "159941")}

    @staticmethod
    def _scenario_with_health(
        scenario: V341ScenarioForecast,
        health_status: str,
    ) -> V341ScenarioForecast:
        reliability = dict(scenario.reliability)
        if health_status in {"MODEL_DEGRADED", "MODEL_OUT_OF_DISTRIBUTION"}:
            reliability["score"] = min(float(reliability["score"]), 50.0)
            reliability["caps"] = sorted(
                set(float(value) for value in reliability.get("caps", ())) | {50.0}
            )
        return replace(
            scenario,
            health_status=health_status,
            reliability=reliability,
        )

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)
        self._analysis_executor.shutdown(wait=wait, cancel_futures=False)

    def recover_interrupted_runs(self) -> int:
        recovered = 0
        with self.sessions() as session, session.begin():
            rows = list(
                session.scalars(
                    select(V341TrainingRun).where(
                        V341TrainingRun.protocol_version == PROTOCOL_VERSION,
                        V341TrainingRun.status.in_(("queued", "running")),
                    )
                )
            )
            for row in rows:
                row.status = "failed"
                row.current_stage = "recovered_after_interruption"
                row.completed_at = self.now_provider()
                row.error_code = "INTERRUPTED_PROCESS"
                row.error_message = "Local process ended before the weekly transaction completed"
                recovered += 1
            analysis_rows = list(
                session.scalars(
                    select(V341AnalysisRun).where(
                        V341AnalysisRun.protocol_version == PROTOCOL_VERSION,
                        V341AnalysisRun.status.in_(("queued", "running")),
                    )
                )
            )
            for row in analysis_rows:
                row.status = "failed"
                row.completed_at = self.now_provider()
                row.error_code = "INTERRUPTED_PROCESS"
                row.error_message = "Local process ended before analysis completed"
                recovered += 1
        return recovered

    @staticmethod
    def validate_market(market: str) -> None:
        if market not in {"399006", "159941"}:
            raise V341RuntimeError("V3.4.1 supports only 399006 and 159941")

    def _instrument_id(self, session: Session, market: str) -> int:
        value = session.scalar(select(Instrument.id).where(Instrument.code == market))
        if value is None:
            raise V341RuntimeError(f"missing instrument {market}")
        return int(value)

    def _complete_anchors(self, market: str) -> tuple[date, ...]:
        today = self.now_provider().date()
        with self.sessions() as session:
            observed = self.features.weekly_anchors(session, market, end_date=today)
        complete: list[date] = []
        for anchor in observed:
            iso_year, iso_week, _ = anchor.isocalendar()
            monday = date.fromisocalendar(iso_year, iso_week, 1)
            expected = tuple(
                self.calendar.sessions(market, monday, monday + timedelta(days=6))
            )
            if expected and max(expected) == anchor and anchor <= today:
                complete.append(anchor)
            elif not expected and anchor.weekday() >= 4 and anchor <= today:
                complete.append(anchor)
        if not complete:
            raise V341RuntimeError(f"no complete weekly anchors for {market}")
        return tuple(complete)

    def _next_complete_anchor_after(self, market: str, anchor: date) -> date:
        monday = anchor - timedelta(days=anchor.weekday()) + timedelta(days=7)
        for offset in range(0, 28, 7):
            start = monday + timedelta(days=offset)
            sessions = tuple(
                self.calendar.sessions(market, start, start + timedelta(days=6))
            )
            if sessions:
                return max(sessions)
        raise V341RuntimeError("cannot resolve the next complete market-week anchor")

    def _assert_consecutive_complete_market_weeks(
        self, market: str, anchors: Sequence[date]
    ) -> None:
        for previous, following in zip(anchors, anchors[1:]):
            expected = self._next_complete_anchor_after(market, previous)
            if following != expected:
                raise V341RuntimeError(
                    "outer block must contain consecutive complete market weeks: "
                    f"expected {expected} after {previous}, got {following}"
                )

    def _snapshots(
        self, market: str, anchors: Sequence[date]
    ) -> tuple[Any, ...]:
        snapshots = []
        with self.sessions() as session:
            for anchor in anchors:
                try:
                    snapshots.append(self.features.load_snapshot(session, market, anchor))
                except Exception as exc:
                    # Only the leading source warm-up may lack 100 daily/35 weekly
                    # bars. Once one snapshot exists, a missing week is a hard data
                    # quality failure and must stop the backlog.
                    if snapshots:
                        raise V341RuntimeError(
                            f"feature snapshot failed at {anchor}: {exc}"
                        ) from exc
        return tuple(snapshots)

    def _persist_profile(
        self,
        session: Session,
        market: str,
        manifest: FeatureManifest,
        *,
        training_window_mode: str = "ROLLING_520W",
    ) -> V341TrainingProfile:
        if training_window_mode not in {
            "ROLLING_520W",
            "EXPANDING_AVAILABLE_HISTORY",
        }:
            raise V341RuntimeError("unsupported V3.4.1 training window mode")
        profile_payload = {
            "protocol_version": PROTOCOL_VERSION,
            "market": market,
            "training_window_mode": training_window_mode,
            "formal_training_weeks": FORMAL_TRAINING_WEEKS,
            "minimum_training_weeks": MINIMUM_TRAINING_WEEKS,
            "feature_warmup_weeks": FEATURE_WARMUP_WEEKS,
            "feature_set_name": manifest.name,
            "feature_set_version": manifest.version,
            "ordered_feature_names": manifest.ordered_feature_names,
            "feature_manifest_hash": manifest.manifest_hash,
            "loss_schema_version": LOSS_SCHEMA_VERSION,
            "random_plan_version": RANDOM_PLAN_VERSION,
            "threshold_formula_version": THRESHOLD_FORMULA_VERSION,
            "regular_candidate_multipliers": [0.50, 0.75, 1.00, 1.25, 1.50],
            "regular_candidate_maximum": 5,
            "structure_audit_interval_weeks": STRUCTURE_AUDIT_INTERVAL_WEEKS,
        }
        profile_hash = _hash(profile_payload)
        profile_id = f"V341-PROFILE-{market}-{profile_hash[:16]}"
        existing = session.get(V341TrainingProfile, profile_id)
        if existing is not None:
            if existing.profile_hash != profile_hash:
                raise V341RuntimeError("immutable V3.4.1 training profile changed")
            return existing
        row = V341TrainingProfile(
            id=profile_id,
            protocol_version=PROTOCOL_VERSION,
            model_market=market,
            training_window_mode=training_window_mode,
            formal_training_weeks=FORMAL_TRAINING_WEEKS,
            minimum_training_weeks=MINIMUM_TRAINING_WEEKS,
            feature_warmup_weeks=FEATURE_WARMUP_WEEKS,
            feature_set_name=manifest.name,
            feature_set_version=manifest.version,
            ordered_feature_names_json=list(manifest.ordered_feature_names),
            feature_manifest_hash=manifest.manifest_hash,
            loss_schema_version=LOSS_SCHEMA_VERSION,
            random_plan_version=RANDOM_PLAN_VERSION,
            threshold_formula_version=THRESHOLD_FORMULA_VERSION,
            profile_json=profile_payload,
            profile_hash=profile_hash,
            created_at=self.now_provider(),
        )
        session.add(row)
        return row

    def _persist_snapshot(
        self,
        session: Session,
        snapshot: Any,
        manifest: FeatureManifest,
    ) -> V341FeatureSnapshotRow:
        payload = self.features.audit_payload(snapshot, manifest)
        snapshot_id = (
            f"V341-FS-{snapshot.market}-{snapshot.cutoff_date}-"
            f"{payload['snapshot_hash'][:12]}"
        )
        existing = session.get(V341FeatureSnapshotRow, snapshot_id)
        if existing is not None:
            if existing.snapshot_hash != payload["snapshot_hash"]:
                raise V341RuntimeError("immutable V3.4.1 feature snapshot changed")
            return existing
        row = V341FeatureSnapshotRow(
            id=snapshot_id,
            protocol_version=PROTOCOL_VERSION,
            model_market=snapshot.market,
            target_instrument_id=self._instrument_id(session, snapshot.market),
            forecast_anchor_date=snapshot.cutoff_date,
            cutoff_at=datetime.fromisoformat(payload["cutoff_at"]),
            source_max_date=snapshot.source_data_max_date,
            feature_manifest_hash=manifest.manifest_hash,
            feature_json=payload["features"],
            daily_sequence_json=payload["daily_sequence"],
            provenance_json=payload["provenance"],
            leakage_audit_json=payload["leakage_audit"],
            snapshot_hash=payload["snapshot_hash"],
            created_at=self.now_provider(),
        )
        session.add(row)
        return row

    def _persist_adapter(self, session: Session, market: str) -> V341ScenarioAdapter:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "market": market,
            "version": SCENARIO_ADAPTER_VERSION,
            "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
            "random_plan_version": RANDOM_PLAN_VERSION,
            "scenario_count": self.scenario_count,
            "formula": (
                "expm1(log1p(Prediction) + "
                "((log1p(Actual)-log1p(Prediction))/SourceSigma)*CurrentSigma)"
            ),
        }
        digest = _hash(payload)
        adapter_id = f"V341-SA-{market}-{SCENARIO_ADAPTER_VERSION}"
        existing = session.get(V341ScenarioAdapter, adapter_id)
        if existing is not None:
            if existing.parameter_hash != digest:
                raise V341RuntimeError("immutable V3.4.1 scenario adapter changed")
            return existing
        row = V341ScenarioAdapter(
            id=adapter_id,
            protocol_version=PROTOCOL_VERSION,
            model_market=market,
            version=SCENARIO_ADAPTER_VERSION,
            residual_schema_version=RESIDUAL_SCHEMA_VERSION,
            random_plan_version=RANDOM_PLAN_VERSION,
            scenario_count=self.scenario_count,
            parameters_json=payload,
            parameter_hash=digest,
            created_at=self.now_provider(),
        )
        session.add(row)
        return row

    def _persist_model(
        self,
        session: Session,
        state: V341ModelState,
        profile_id: str,
        *,
        status: str,
        promotion_gate: Mapping[str, Any] | None = None,
    ) -> V341ModelVersion:
        model_id = f"V341-MODEL-{state.market}-{state.version}"
        existing = session.get(V341ModelVersion, model_id)
        if existing is not None:
            if existing.parameter_hash != state.parameter_hash:
                raise V341RuntimeError("immutable V3.4.1 model version changed")
            return existing
        parent_id = (
            None
            if state.parent_version is None
            else f"V341-MODEL-{state.market}-{state.parent_version}"
        )
        row = V341ModelVersion(
            id=model_id,
            protocol_version=PROTOCOL_VERSION,
            model_market=state.market,
            version=state.version,
            parent_model_id=parent_id,
            profile_id=profile_id,
            status=status,
            trained_through_date=state.trained_through_date,
            effective_from_date=state.effective_from_date,
            feature_anchor_max_date=state.feature_anchor_max_date,
            label_observed_through_date=state.label_observed_through_date,
            raw_matured_sample_count=state.raw_matured_sample_count,
            effective_independent_sample_count=state.effective_independent_sample_count,
            parameters_json=state_to_payload(state),
            metrics_json=dict(state.validation_metrics),
            promotion_gate_json=dict(promotion_gate or {}),
            health_status="MODEL_NORMAL",
            parameter_hash=state.parameter_hash,
            created_at=self.now_provider(),
        )
        session.add(row)
        return row

    @staticmethod
    def _load_state(
        session: Session,
        market: str,
        anchor: date,
    ) -> tuple[V341ModelState, V341ModelVersion] | None:
        row = session.scalar(
            select(V341ModelVersion)
            .where(
                V341ModelVersion.protocol_version == PROTOCOL_VERSION,
                V341ModelVersion.model_market == market,
                V341ModelVersion.status.in_(("champion", "staged")),
                V341ModelVersion.effective_from_date <= anchor,
            )
            .order_by(
                V341ModelVersion.effective_from_date.desc(),
                V341ModelVersion.created_at.desc(),
            )
            .limit(1)
        )
        return None if row is None else (state_from_payload(row.parameters_json), row)

    def _residual_pool(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        rows = list(
            session.scalars(
                select(V341ResidualRecord)
                .where(
                    V341ResidualRecord.model_market == market,
                    V341ResidualRecord.matured_at <= anchor,
                    V341ResidualRecord.model_family == MODEL_FAMILY,
                    V341ResidualRecord.scenario_adapter_version
                    == SCENARIO_ADAPTER_VERSION,
                    V341ResidualRecord.residual_schema_version
                    == RESIDUAL_SCHEMA_VERSION,
                )
                .order_by(V341ResidualRecord.matured_at, V341ResidualRecord.id)
            )
        )
        standardized = np.asarray(
            [row.standardized_residual_json for row in rows], dtype=float
        )
        if not rows:
            standardized = np.empty((0, 13), dtype=float)
        if standardized.ndim != 2 or standardized.shape[1] != 13:
            raise V341RuntimeError("stored V3.4.1 residual pool is not N x 13")
        sigmas = np.asarray([float(row.source_sigma) for row in rows], dtype=float)
        recent_evaluations = list(
            session.scalars(
                select(V341ForecastEvaluation)
                .join(V341ForecastRow, V341ForecastRow.id == V341ForecastEvaluation.forecast_id)
                .where(
                    V341ForecastRow.model_market == market,
                    V341ForecastEvaluation.horizon_weeks == 13,
                    V341ForecastEvaluation.evaluation_available_date <= anchor,
                )
                .order_by(V341ForecastEvaluation.evaluation_available_date.desc())
                .limit(52)
            )
        )
        metadata = {
            "source": "FROZEN_SAMPLE_OUT_RESIDUALS" if rows else "EMPTY_OOS_POOL",
            "status": "AVAILABLE" if rows else "INSUFFICIENT_OOS_RESIDUALS",
            "raw_residual_count": len(rows),
            "effective_residual_count": conservative_independent_sample_count(
                tuple(
                    (
                        session.get(V341ForecastRow, row.forecast_id).forecast_anchor_date,
                        row.matured_at,
                    )
                    for row in rows
                )
            )
            if rows
            else 0,
            "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
            "scenario_adapter_version": SCENARIO_ADAPTER_VERSION,
            "residual_pool_identity": _hash(
                {
                    "market": market,
                    "anchor": anchor,
                    "residual_hashes": [row.residual_hash for row in rows],
                    "schema": RESIDUAL_SCHEMA_VERSION,
                }
            ),
            "recent_brier": float(
                np.mean(
                    [
                        float(row.metrics_json.get("calibrated_brier", 1 / 3))
                        for row in recent_evaluations
                    ]
                )
            )
            if recent_evaluations
            else 1 / 3,
            "recent_wis": float(
                np.mean(
                    [float(row.metrics_json.get("wis", 1.0)) for row in recent_evaluations]
                )
            )
            if recent_evaluations
            else 1.0,
            "recent_coverage": float(
                np.mean(
                    [
                        float(row.metrics_json.get("p10_p90_coverage", 0.8))
                        for row in recent_evaluations
                    ]
                )
            )
            if recent_evaluations
            else 0.8,
            "recent_score": float(
                np.clip(
                    1.0
                    - np.mean(
                        [
                            float(row.metrics_json.get("normalized_abs_error", 1.0))
                            for row in recent_evaluations
                        ]
                    ),
                    0.0,
                    1.0,
                )
            )
            if recent_evaluations
            else 0.5,
        }
        return standardized, sigmas, metadata

    def _active_calibrators(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for horizon in (4, 8, 13):
            row = session.scalar(
                select(V341ProbabilityCalibrator)
                .where(
                    V341ProbabilityCalibrator.protocol_version == PROTOCOL_VERSION,
                    V341ProbabilityCalibrator.model_market == market,
                    V341ProbabilityCalibrator.horizon_weeks == horizon,
                    V341ProbabilityCalibrator.model_family == MODEL_FAMILY,
                    V341ProbabilityCalibrator.scenario_adapter_version
                    == SCENARIO_ADAPTER_VERSION,
                    V341ProbabilityCalibrator.residual_schema_version
                    == RESIDUAL_SCHEMA_VERSION,
                    V341ProbabilityCalibrator.threshold_formula_version
                    == THRESHOLD_FORMULA_VERSION,
                    V341ProbabilityCalibrator.effective_from_date <= anchor,
                )
                .order_by(
                    V341ProbabilityCalibrator.effective_from_date.desc(),
                    V341ProbabilityCalibrator.created_at.desc(),
                )
                .limit(1)
            )
            if row is not None:
                result[horizon] = {
                    "id": row.id,
                    "version": row.version,
                    "temperature": float(row.temperature),
                    "status": row.status,
                    "calibration_hash": row.calibration_hash,
                }
        return result

    @staticmethod
    def _future_label_end(
        calendar: CalendarProvider,
        market: str,
        anchor: date,
    ) -> date:
        current = anchor
        for _ in range(13):
            monday = current - timedelta(days=current.weekday()) + timedelta(days=7)
            resolved: date | None = None
            for offset in range(0, 28, 7):
                start = monday + timedelta(days=offset)
                sessions = tuple(calendar.sessions(market, start, start + timedelta(days=6)))
                if sessions:
                    resolved = max(sessions)
                    break
            if resolved is None:
                raise V341RuntimeError("cannot project the 13-week label end")
            current = resolved
        return current

    def _persist_forecast(
        self,
        session: Session,
        snapshot_row: V341FeatureSnapshotRow,
        model_row: V341ModelVersion,
        adapter_row: V341ScenarioAdapter,
        scenario: V341ScenarioForecast,
        *,
        label_end_date: date,
    ) -> V341ForecastRow:
        existing = session.scalar(
            select(V341ForecastRow).where(
                V341ForecastRow.protocol_version == PROTOCOL_VERSION,
                V341ForecastRow.model_market == scenario.market,
                V341ForecastRow.forecast_anchor_date == scenario.anchor_date,
            )
        )
        random_plan = self._persist_random_plan(
            session,
            market=scenario.market,
            scope="FORECAST",
            plan_payload=scenario.random_plan.to_payload(),
            residual_pool_identity=str(
                scenario.residual_pool.get("residual_pool_identity", "PRIOR_ONLY")
            ),
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "market": scenario.market,
            "forecast_anchor_date": scenario.anchor_date.isoformat(),
            "model_version": scenario.model_version,
            "scenario_seed": scenario.scenario_seed,
            "scenario_count": scenario.scenario_count,
            "random_plan_hash": scenario.random_plan.plan_hash,
            "random_plan_id": random_plan.id,
            "feature_snapshot_id": snapshot_row.id,
            "model_version_id": model_row.id,
            "scenario_adapter_id": adapter_row.id,
            "label_end_date": label_end_date.isoformat(),
            "expected_path": list(scenario.expected_path),
            "representative_ohlcv": list(scenario.representative_ohlcv),
            "indicators": list(scenario.indicators),
            "price_quantiles": list(scenario.price_quantiles),
            "horizon_probabilities": dict(scenario.horizon_probabilities),
            "thresholds": dict(scenario.thresholds),
            "calibrator_versions": dict(scenario.calibrator_versions),
            "residual_pool": dict(scenario.residual_pool),
            "path_probabilities": dict(scenario.path_probabilities),
            "model_reliability": dict(scenario.reliability),
            "health_status": scenario.health_status,
            "scenario_audit": dict(scenario.scenario_audit),
        }
        forecast_hash = _hash(payload)
        if existing is not None:
            if existing.forecast_hash != forecast_hash:
                raise V341RuntimeError("frozen V3.4.1 forecast cannot be overwritten")
            return existing
        row = V341ForecastRow(
            protocol_version=PROTOCOL_VERSION,
            model_market=scenario.market,
            target_instrument_id=self._instrument_id(session, scenario.market),
            feature_snapshot_id=snapshot_row.id,
            model_version_id=model_row.id,
            scenario_adapter_id=adapter_row.id,
            random_plan_id=random_plan.id,
            forecast_anchor_date=scenario.anchor_date,
            label_end_date=label_end_date,
            horizon_weeks=13,
            maturity_status="IMMATURE",
            scenario_seed=scenario.scenario_seed,
            scenario_count=scenario.scenario_count,
            random_plan_hash=scenario.random_plan.plan_hash,
            expected_path_json=list(scenario.expected_path),
            representative_ohlcv_json=list(scenario.representative_ohlcv),
            indicator_path_json=list(scenario.indicators),
            price_quantiles_json=list(scenario.price_quantiles),
            horizon_probabilities_json=dict(scenario.horizon_probabilities),
            thresholds_json=dict(scenario.thresholds),
            calibrator_versions_json=dict(scenario.calibrator_versions),
            residual_pool_json=dict(scenario.residual_pool),
            path_probabilities_json=dict(scenario.path_probabilities),
            model_reliability_score=_decimal(float(scenario.reliability["score"])),
            reliability_components_json=dict(scenario.reliability),
            health_status=scenario.health_status,
            turning_windows_json={},
            consistency_json=dict(
                scenario.scenario_audit.get("indicator_consistency", {})
            ),
            payload_json=payload,
            forecast_hash=forecast_hash,
            created_at=self.now_provider(),
        )
        session.add(row)
        session.flush()
        for horizon, calibrator in scenario.calibrator_versions.items():
            calibrator_id = None
            active_row = session.scalar(
                select(V341ProbabilityCalibrator).where(
                    V341ProbabilityCalibrator.model_market == scenario.market,
                    V341ProbabilityCalibrator.horizon_weeks == int(horizon),
                    V341ProbabilityCalibrator.version == calibrator,
                )
            ) if calibrator is not None else None
            if active_row is not None:
                calibrator_id = active_row.id
            if calibrator_id is not None:
                session.add(
                    V341ForecastCalibratorLink(
                        forecast_id=row.id,
                        calibrator_id=calibrator_id,
                        horizon_weeks=int(horizon),
                        created_at=self.now_provider(),
                    )
                )
        session.flush()
        return row

    def _persist_random_plan(
        self,
        session: Session,
        *,
        market: str,
        scope: str,
        plan_payload: Mapping[str, Any],
        residual_pool_identity: str,
    ) -> V341RandomPlan:
        plan_hash = str(plan_payload["plan_hash"])
        recomputed = _hash(
            {key: value for key, value in plan_payload.items() if key != "plan_hash"}
        )
        if recomputed != plan_hash:
            raise V341RuntimeError("random-plan payload does not match its frozen hash")
        existing = session.scalar(
            select(V341RandomPlan).where(
                V341RandomPlan.protocol_version == PROTOCOL_VERSION,
                V341RandomPlan.model_market == market,
                V341RandomPlan.plan_hash == plan_hash,
            )
        )
        if existing is not None:
            return existing
        canonical = json.dumps(
            dict(plan_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        row = V341RandomPlan(
            id=f"V341-RP-{market}-{plan_hash[:28]}",
            protocol_version=PROTOCOL_VERSION,
            model_market=market,
            plan_scope=scope,
            plan_version=str(plan_payload["version"]),
            seed=int(plan_payload["seed"]),
            scenario_count=int(plan_payload["scenario_count"]),
            residual_pool_identity=residual_pool_identity,
            residual_pool_size=int(plan_payload["residual_pool_size"]),
            empirical_pool_size=int(plan_payload["empirical_pool_size"]),
            payload_codec="JSON_UTF8_ZLIB_9",
            plan_payload_zlib=zlib.compress(canonical, level=9),
            plan_hash=plan_hash,
            created_at=self.now_provider(),
        )
        session.add(row)
        session.flush()
        return row

    def _persist_health_snapshot(
        self,
        session: Session,
        market: str,
        anchor: date,
        model_row: V341ModelVersion,
        path: V341PathForecast,
        reliability: Mapping[str, Any],
    ) -> str:
        existing = session.scalar(
            select(V341ModelHealthSnapshot).where(
                V341ModelHealthSnapshot.protocol_version == PROTOCOL_VERSION,
                V341ModelHealthSnapshot.model_market == market,
                V341ModelHealthSnapshot.anchor_date == anchor,
            )
        )
        if existing is not None:
            return existing.health_status
        previous = session.scalar(
            select(V341ModelHealthSnapshot)
            .where(
                V341ModelHealthSnapshot.protocol_version == PROTOCOL_VERSION,
                V341ModelHealthSnapshot.model_market == market,
                V341ModelHealthSnapshot.anchor_date < anchor,
            )
            .order_by(V341ModelHealthSnapshot.anchor_date.desc())
            .limit(1)
        )
        ood_score = float(path.ood_diagnostics.get("ood_score", 0.0))
        previous_status = "MODEL_NORMAL" if previous is None else previous.health_status
        previous_diagnostics = {} if previous is None else dict(previous.diagnostics_json)
        recent = list(
            session.scalars(
                select(V341ForecastEvaluation)
                .join(V341ForecastRow, V341ForecastRow.id == V341ForecastEvaluation.forecast_id)
                .where(
                    V341ForecastRow.model_market == market,
                    V341ForecastEvaluation.horizon_weeks == 13,
                    V341ForecastEvaluation.evaluation_available_date <= anchor,
                )
                .order_by(V341ForecastEvaluation.evaluation_available_date.desc())
                .limit(8)
            )
        )
        degradation_metrics = {
            "sample_count": len(recent),
            "mean_normalized_error": (
                float(np.mean([row.metrics_json["normalized_abs_error"] for row in recent]))
                if recent else None
            ),
            "mean_brier": (
                float(np.mean([row.metrics_json["calibrated_brier"] for row in recent]))
                if recent else None
            ),
            "coverage": (
                float(np.mean([row.metrics_json["p10_p90_coverage"] for row in recent]))
                if recent else None
            ),
        }
        degradation_signal = bool(
            len(recent) >= 4
            and (
                degradation_metrics["mean_normalized_error"] > 1.50
                or degradation_metrics["mean_brier"] > 0.45
                or abs(degradation_metrics["coverage"] - 0.80) > 0.35
            )
        )
        ood_entry = (
            int(previous_diagnostics.get("ood_entry_count", 0)) + 1
            if ood_score >= 0.50 else 0
        )
        ood_exit = (
            int(previous_diagnostics.get("ood_exit_count", 0)) + 1
            if ood_score <= 0.35 else 0
        )
        degraded_entry = (
            int(previous_diagnostics.get("degraded_entry_count", 0)) + 1
            if degradation_signal else 0
        )
        degraded_exit = (
            int(previous_diagnostics.get("degraded_exit_count", 0)) + 1
            if not degradation_signal else 0
        )
        if ood_entry >= 2:
            health_status = "MODEL_OUT_OF_DISTRIBUTION"
        elif previous_status == "MODEL_OUT_OF_DISTRIBUTION" and ood_exit < 3:
            health_status = previous_status
        elif degraded_entry >= 3:
            health_status = "MODEL_DEGRADED"
        elif previous_status == "MODEL_DEGRADED" and degraded_exit < 3:
            health_status = previous_status
        else:
            health_status = "MODEL_NORMAL"
        entry_count = ood_entry if ood_entry else degraded_entry
        exit_count = ood_exit if previous_status == "MODEL_OUT_OF_DISTRIBUTION" else degraded_exit
        diagnostics = {
            **dict(path.ood_diagnostics),
            "entry_threshold": 0.50,
            "exit_threshold": 0.35,
            "entry_confirmations_required": 2,
            "exit_confirmations_required": 3,
            "previous_status": previous_status,
            "ood_entry_count": ood_entry,
            "ood_exit_count": ood_exit,
            "degraded_entry_count": degraded_entry,
            "degraded_exit_count": degraded_exit,
            "degradation_signal": degradation_signal,
            "degradation_metrics": degradation_metrics,
            "degradation_source": "MATURED_FROZEN_13W_OOS_EVALUATIONS_ONLY",
        }
        payload = {
            "market": market,
            "anchor": anchor,
            "model_id": model_row.id,
            "health_status": health_status,
            "entry_count": entry_count,
            "exit_count": exit_count,
            "diagnostics": diagnostics,
            "reliability_score": reliability["score"],
        }
        session.add(
            V341ModelHealthSnapshot(
                protocol_version=PROTOCOL_VERSION,
                model_market=market,
                anchor_date=anchor,
                model_version_id=model_row.id,
                health_status=health_status,
                consecutive_entry_count=entry_count,
                consecutive_exit_count=exit_count,
                diagnostics_json=diagnostics,
                reliability_score=_decimal(float(reliability["score"])),
                health_hash=_hash(payload),
                created_at=self.now_provider(),
            )
        )
        session.flush()
        return health_status

    def _mature_forecasts(
        self,
        session: Session,
        market: str,
        snapshots: Sequence[Any],
        as_of: date,
    ) -> dict[str, int]:
        ordered = tuple(sorted(snapshots, key=lambda item: item.cutoff_date))
        positions = {snapshot.cutoff_date: index for index, snapshot in enumerate(ordered)}
        closes = {
            snapshot.cutoff_date: float(snapshot.daily_sequence[-1]["close"])
            for snapshot in ordered
        }
        created = {"4": 0, "8": 0, "13": 0, "residuals": 0}
        forecasts = list(
            session.scalars(
                select(V341ForecastRow)
                .where(
                    V341ForecastRow.protocol_version == PROTOCOL_VERSION,
                    V341ForecastRow.model_market == market,
                    V341ForecastRow.forecast_anchor_date < as_of,
                )
                .order_by(V341ForecastRow.forecast_anchor_date, V341ForecastRow.id)
            )
        )
        for forecast in forecasts:
            start = positions.get(forecast.forecast_anchor_date)
            if start is None:
                continue
            available_horizons = [
                horizon
                for horizon in (4, 8, 13)
                if start + horizon < len(ordered)
                and ordered[start + horizon].cutoff_date <= as_of
            ]
            forecast.maturity_status = (
                "FULLY_MATURE_13W"
                if 13 in available_horizons
                else "MATURE_8W"
                if 8 in available_horizons
                else "MATURE_4W"
                if 4 in available_horizons
                else "IMMATURE"
            )
            base = closes[forecast.forecast_anchor_date]
            for horizon in available_horizons:
                existing = session.scalar(
                    select(V341ForecastEvaluation).where(
                        V341ForecastEvaluation.forecast_id == forecast.id,
                        V341ForecastEvaluation.horizon_weeks == horizon,
                        V341ForecastEvaluation.evaluation_version == EVALUATION_VERSION,
                    )
                )
                if existing is not None:
                    continue
                actual_close = closes[ordered[start + horizon].cutoff_date]
                actual_return = actual_close / base - 1.0
                threshold_payload = forecast.thresholds_json[str(horizon)]
                threshold = float(threshold_payload["threshold"])
                class_index, class_name = _actual_class(actual_return, threshold)
                probabilities = forecast.horizon_probabilities_json[str(horizon)]
                raw = _probability_vector(probabilities, "raw")
                calibrated = _probability_vector(probabilities, "calibrated")
                quantile = forecast.price_quantiles_json[horizon - 1]
                p10_return = float(quantile["close_p10"]) / base - 1.0
                p50_return = float(quantile["close_p50"]) / base - 1.0
                p90_return = float(quantile["close_p90"]) / base - 1.0
                expected = float(forecast.expected_path_json[horizon - 1])
                sigma = max(
                    float(threshold_payload["sigma_week_at_forecast"])
                    * np.sqrt(horizon),
                    0.01,
                )
                metrics = {
                    "raw_brier": multiclass_brier(raw, class_index),
                    "calibrated_brier": multiclass_brier(calibrated, class_index),
                    "wis": weighted_interval_score(
                        p10_return, p50_return, p90_return, actual_return
                    )
                    / sigma,
                    "p10_p90_coverage": float(
                        p10_return <= actual_return <= p90_return
                    ),
                    "absolute_error": abs(expected - actual_return),
                    "normalized_abs_error": abs(expected - actual_return) / sigma,
                    "frozen_threshold": threshold,
                    "threshold_formula_version": threshold_payload["formula_version"],
                    "raw_probabilities": raw.tolist(),
                    "calibrated_probabilities": calibrated.tolist(),
                }
                payload = {
                    "forecast_id": forecast.id,
                    "horizon_weeks": horizon,
                    "actual_return": actual_return,
                    "actual_class": class_name,
                    "evaluation_available_date": ordered[start + horizon].cutoff_date.isoformat(),
                    "metrics": metrics,
                }
                session.add(
                    V341ForecastEvaluation(
                        forecast_id=forecast.id,
                        horizon_weeks=horizon,
                        evaluation_version=EVALUATION_VERSION,
                        evaluation_available_date=ordered[start + horizon].cutoff_date,
                        actual_return=_decimal(actual_return),
                        actual_class=class_name,
                        metrics_json=metrics,
                        market_data_version_json={
                            "source": "V341_FEATURE_SNAPSHOT_CLOSES",
                            "anchor_snapshot_date": forecast.forecast_anchor_date.isoformat(),
                            "actual_snapshot_date": ordered[start + horizon].cutoff_date.isoformat(),
                        },
                        evaluation_hash=_hash(payload),
                        created_at=self.now_provider(),
                    )
                )
                created[str(horizon)] += 1

            if 13 not in available_horizons:
                continue
            residual = session.scalar(
                select(V341ResidualRecord).where(
                    V341ResidualRecord.forecast_id == forecast.id
                )
            )
            if residual is not None:
                continue
            actual_path = np.asarray(
                [
                    closes[ordered[start + step].cutoff_date] / base - 1.0
                    for step in range(1, 14)
                ],
                dtype=float,
            )
            prediction = np.asarray(forecast.expected_path_json, dtype=float)
            if np.any(actual_path <= -1.0) or np.any(prediction <= -1.0):
                raise V341RuntimeError(
                    "V3.4.4 residual paths must imply positive gross returns"
                )
            residual_path = np.log1p(actual_path) - np.log1p(prediction)
            source_sigma = max(
                float(forecast.thresholds_json["13"]["sigma_week_at_forecast"]),
                0.005,
            )
            standardized = residual_path / source_sigma
            compatibility = {
                "market": market,
                "model_family": MODEL_FAMILY,
                "scenario_adapter_version": SCENARIO_ADAPTER_VERSION,
                "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
                "return_unit": "LOG_GROSS_CUMULATIVE_RETURN_RESIDUAL",
                "horizon_weeks": 13,
                "source_forecast_adapter_id": forecast.scenario_adapter_id,
            }
            residual_payload = {
                "forecast_id": forecast.id,
                "prediction": prediction.tolist(),
                "actual": actual_path.tolist(),
                "residual": residual_path.tolist(),
                "standardized": standardized.tolist(),
                "source_sigma": source_sigma,
                "compatibility": compatibility,
            }
            session.add(
                V341ResidualRecord(
                    forecast_id=forecast.id,
                    model_market=market,
                    matured_at=ordered[start + 13].cutoff_date,
                    model_family=MODEL_FAMILY,
                    scenario_adapter_version=SCENARIO_ADAPTER_VERSION,
                    residual_schema_version=RESIDUAL_SCHEMA_VERSION,
                    return_unit="LOG_GROSS_CUMULATIVE_RETURN_RESIDUAL",
                    prediction_path_json=prediction.tolist(),
                    actual_path_json=actual_path.tolist(),
                    residual_path_json=residual_path.tolist(),
                    standardized_residual_json=standardized.tolist(),
                    source_sigma=_decimal(source_sigma),
                    compatibility_json=compatibility,
                    residual_hash=_hash(residual_payload),
                    created_at=self.now_provider(),
                )
            )
            created["residuals"] += 1
        session.flush()
        return created

    def _fit_calibrators(
        self,
        session: Session,
        market: str,
        as_of: date,
    ) -> list[str]:
        created: list[str] = []
        effective_from = self._next_complete_anchor_after(market, as_of)
        for horizon in (4, 8, 13):
            evaluations = list(
                session.scalars(
                    select(V341ForecastEvaluation)
                    .join(V341ForecastRow, V341ForecastRow.id == V341ForecastEvaluation.forecast_id)
                    .join(
                        V341ScenarioAdapter,
                        V341ScenarioAdapter.id == V341ForecastRow.scenario_adapter_id,
                    )
                    .where(
                        V341ForecastRow.protocol_version == PROTOCOL_VERSION,
                        V341ForecastRow.model_market == market,
                        V341ScenarioAdapter.version == SCENARIO_ADAPTER_VERSION,
                        V341ForecastEvaluation.horizon_weeks == horizon,
                        V341ForecastEvaluation.evaluation_version == EVALUATION_VERSION,
                        V341ForecastEvaluation.evaluation_available_date <= as_of,
                    )
                    .order_by(
                        V341ForecastEvaluation.evaluation_available_date,
                        V341ForecastEvaluation.id,
                    )
                )
            )
            if not evaluations:
                continue
            forecasts = {
                row.id: row
                for row in session.scalars(
                    select(V341ForecastRow).where(
                        V341ForecastRow.id.in_(
                            tuple(evaluation.forecast_id for evaluation in evaluations)
                        )
                    )
                )
            }
            intervals = tuple(
                (
                    forecasts[evaluation.forecast_id].forecast_anchor_date,
                    evaluation.evaluation_available_date,
                )
                for evaluation in evaluations
            )
            effective_count = conservative_independent_sample_count(intervals)
            raw = np.asarray(
                [evaluation.metrics_json["raw_probabilities"] for evaluation in evaluations],
                dtype=float,
            )
            labels = np.asarray(
                [
                    0 if evaluation.actual_class == "up" else 2 if evaluation.actual_class == "down" else 1
                    for evaluation in evaluations
                ],
                dtype=int,
            )
            if effective_count < 30:
                status = "SHRINKAGE_ONLY"
                temperature = 1.25
            else:
                status = "PRELIMINARY" if effective_count < 50 else "FORMALLY_CALIBRATED"
                temperature = fit_temperature(raw, labels)
            payload = {
                "calibration_version": CALIBRATION_VERSION,
                "market": market,
                "horizon_weeks": horizon,
                "fit_through_date": as_of.isoformat(),
                "effective_from_date": effective_from.isoformat(),
                "evaluation_hashes": [row.evaluation_hash for row in evaluations],
                "raw_sample_count": len(evaluations),
                "effective_sample_count": effective_count,
                "status": status,
                "temperature": temperature,
            }
            digest = _hash(payload)
            version = f"{CALIBRATION_VERSION}-{as_of.isoformat()}-{digest[:10]}"
            row_id = f"V341-CAL-{market}-H{horizon}-{digest[:20]}"
            if session.get(V341ProbabilityCalibrator, row_id) is not None:
                continue
            session.add(
                V341ProbabilityCalibrator(
                    id=row_id,
                    protocol_version=PROTOCOL_VERSION,
                    model_market=market,
                    horizon_weeks=horizon,
                    version=version,
                    model_family=MODEL_FAMILY,
                    scenario_adapter_version=SCENARIO_ADAPTER_VERSION,
                    residual_schema_version=RESIDUAL_SCHEMA_VERSION,
                    threshold_formula_version=THRESHOLD_FORMULA_VERSION,
                    fit_through_date=as_of,
                    effective_from_date=effective_from,
                    raw_sample_count=len(evaluations),
                    effective_sample_count=effective_count,
                    status=status,
                    temperature=_decimal(temperature),
                    metrics_json={
                        "method": "TEMPERATURE_SCALING",
                        "activation": "NEXT_COMPLETE_MARKET_WEEK",
                        "input": "FROZEN_SAMPLE_OUT_PROBABILITIES_ONLY",
                    },
                    calibration_hash=digest,
                    created_at=self.now_provider(),
                )
            )
            created.append(row_id)
        session.flush()
        return created

    def _sealed_outer_fold(
        self,
        session: Session,
        market: str,
        rows: Sequence[V34TrainingSample],
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        folds = purged_walk_forward_folds(rows, folds=5)
        if len(folds) < 2:
            return None
        outer_training, outer_validation = folds[-1]
        consumed_validation_anchors: set[date] = set()
        for block in session.scalars(
            select(V341OuterEvaluationBlock).where(
                V341OuterEvaluationBlock.protocol_version == PROTOCOL_VERSION,
                V341OuterEvaluationBlock.model_market == market,
            )
        ):
            consumed_validation_anchors.update(
                date.fromisoformat(str(value))
                for value in block.validation_anchors_json
            )
        available = tuple(
            index
            for index in outer_validation
            if rows[index].anchor_date not in consumed_validation_anchors
        )
        if len(available) < 13:
            return None
        selected = available[:13]
        self._assert_consecutive_complete_market_weeks(
            market, tuple(rows[index].anchor_date for index in selected)
        )
        return tuple(outer_training), selected

    @staticmethod
    def _windowed_training_indices(
        training_indices: Sequence[int], window_mode: str
    ) -> tuple[int, ...]:
        indices = tuple(int(value) for value in training_indices)
        if window_mode == "ROLLING_520W":
            return indices[-FORMAL_TRAINING_WEEKS:]
        if window_mode == "EXPANDING_AVAILABLE_HISTORY":
            return indices
        raise V341RuntimeError(f"unsupported training window mode: {window_mode}")

    @staticmethod
    def _shared_standardized_coefficient_norm_ratio(
        champion: V341ModelState,
        candidate: V341ModelState,
    ) -> tuple[float, tuple[str, ...]]:
        """Compare only shared standardized feature coefficients.

        Ridge coefficients are learned after per-training-window
        standardization.  Restricting the norm to the ordered intersection
        avoids comparing unrelated CORE/EXTENDED dimensions.
        """

        shared = tuple(name for name in champion.feature_names if name in candidate.feature_names)
        if not shared:
            raise V341RuntimeError("structure models have no shared features")
        champion_by_name = {
            name: np.asarray(champion.coefficients[index], dtype=float)
            for index, name in enumerate(champion.feature_names)
        }
        candidate_by_name = {
            name: np.asarray(candidate.coefficients[index], dtype=float)
            for index, name in enumerate(candidate.feature_names)
        }
        champion_values = np.concatenate([champion_by_name[name] for name in shared])
        candidate_values = np.concatenate([candidate_by_name[name] for name in shared])
        ratio = float(
            np.linalg.norm(candidate_values)
            / max(float(np.linalg.norm(champion_values)), 1e-9)
        )
        return ratio, shared

    def _candidate_trial(
        self,
        session: Session,
        market: str,
        current_anchor: date,
        matured: Sequence[V34TrainingSample],
        champion_state: V341ModelState,
        champion_row: V341ModelVersion,
        profile: V341TrainingProfile,
        run: V341TrainingRun,
        candidate_number: int,
        manifest: FeatureManifest,
    ) -> tuple[V341ModelVersion, dict[str, Any]] | None:
        rows = tuple(sorted(matured, key=lambda item: item.anchor_date))
        folds = purged_walk_forward_folds(rows, folds=5)
        if len(folds) < 2:
            return None
        inner_folds = folds[:-1]
        sealed_outer = self._sealed_outer_fold(session, market, rows)
        if sealed_outer is None:
            return None
        outer_training, outer_validation = sealed_outer
        seed = int.from_bytes(
            hashlib.sha256(
                f"{PROTOCOL_VERSION}:{market}:{current_anchor}:OUTER:{candidate_number}".encode(
                    "utf-8"
                )
            ).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)
        residuals, _, _ = self._residual_pool(session, market, current_anchor)
        if not len(residuals):
            residuals = prior_standardized_residual_pool(seed)
        alpha_scores: dict[float, float] = {}
        alpha_anchor_losses: dict[float, list[float]] = {}
        alpha_plan_hashes: dict[float, list[str]] = {}
        alpha_fold_summaries: dict[float, list[float]] = {}
        for alpha in candidate_alphas(champion_state.ridge_alpha):
            losses: list[float] = []
            plans: list[str] = []
            fold_summaries: list[float] = []
            for fold_number, (training_indices, validation_indices) in enumerate(
                inner_folds, start=1
            ):
                fold_losses, fold_plans, _, _ = self.training.fold_composite_losses(
                        market,
                        rows,
                        training_indices,
                        validation_indices,
                        alpha=alpha,
                        feature_names=manifest.ordered_feature_names,
                        standardized_residuals=residuals,
                        shared_seed=seed + fold_number,
                        scenario_count=max(1000, self.scenario_count),
                        recent_fold_oos_losses=fold_summaries,
                    )
                losses.extend(fold_losses.tolist())
                plans.extend(fold_plans)
                fold_summaries.append(float(np.mean(fold_losses)))
            if losses:
                alpha_scores[alpha] = float(np.mean(losses))
                alpha_anchor_losses[alpha] = losses
                alpha_plan_hashes[alpha] = plans
                alpha_fold_summaries[alpha] = fold_summaries
        if not alpha_scores:
            return None
        selected_alpha = min(alpha_scores, key=lambda value: (alpha_scores[value], value))
        champion_outer, champion_plans, champion_components, outer_plan_payloads = self.training.fold_composite_losses(
            market,
            rows,
            outer_training,
            outer_validation,
            alpha=champion_state.ridge_alpha,
            feature_names=manifest.ordered_feature_names,
            standardized_residuals=residuals,
            shared_seed=seed,
            scenario_count=max(1000, self.scenario_count),
            recent_fold_oos_losses=alpha_fold_summaries.get(
                champion_state.ridge_alpha, ()
            ),
            capture_random_plans=True,
        )
        candidate_outer, candidate_plans, candidate_components, _ = self.training.fold_composite_losses(
            market,
            rows,
            outer_training,
            outer_validation,
            alpha=selected_alpha,
            feature_names=manifest.ordered_feature_names,
            standardized_residuals=residuals,
            shared_seed=seed,
            scenario_count=max(1000, self.scenario_count),
            recent_fold_oos_losses=alpha_fold_summaries.get(selected_alpha, ()),
        )
        if champion_plans != candidate_plans:
            raise V341RuntimeError("champion/candidate common random plans diverged")
        outer_block_id = _hash(
            {
                "market": market,
                "current_anchor": current_anchor,
                "training_anchors": [rows[index].anchor_date for index in outer_training],
                "validation_anchors": [rows[index].anchor_date for index in outer_validation],
            }
        )
        next_anchor = self._next_complete_anchor_after(market, current_anchor)
        candidate_state = self.training.fit(
            market,
            rows,
            version=f"{market}-V341-C{candidate_number:04d}-{current_anchor.isoformat()}",
            parent_version=champion_state.version,
            trained_through=current_anchor,
            effective_from=next_anchor,
            alpha=selected_alpha,
            feature_names=manifest.ordered_feature_names,
        )
        champion_norm = float(np.linalg.norm(np.asarray(champion_state.coefficients)))
        candidate_norm = float(np.linalg.norm(np.asarray(candidate_state.coefficients)))
        hard_gate_metrics = {
            "champion_coverage": float(
                np.mean([value["coverage"] for value in champion_components])
            ),
            "candidate_coverage": float(
                np.mean([value["coverage"] for value in candidate_components])
            ),
            "champion_turning_accuracy": float(
                np.mean([value["turning_type_correct"] for value in champion_components])
            ),
            "candidate_turning_accuracy": float(
                np.mean([value["turning_type_correct"] for value in candidate_components])
            ),
            "coefficient_norm_ratio": candidate_norm / max(champion_norm, 1e-9),
        }
        decision = promotion_from_paired_losses(
            champion_outer,
            candidate_outer,
            seed=seed,
            fold_lengths=(len(outer_validation),),
            hard_gate_metrics=hard_gate_metrics,
        )
        decision.update(
            {
                "loss_schema_version": LOSS_SCHEMA_VERSION,
                "inner_alpha_scores": {str(key): value for key, value in alpha_scores.items()},
                "selected_alpha": selected_alpha,
                "outer_block_id": outer_block_id,
                "outer_block_consumed_once": True,
                "random_plan_semantics": "COMMON_PLAN_ID_FROZEN_FOR_CHAMPION_AND_CANDIDATE",
                "outer_random_plan_hashes": list(champion_plans),
            }
        )
        candidate_state = replace(candidate_state, validation_metrics=decision)
        candidate_row = self._persist_model(
            session,
            candidate_state,
            profile.id,
            status="staged" if decision["promoted"] else "rejected",
            promotion_gate=decision,
        )
        combined_plan_body = {
            "version": CANDIDATE_RANDOM_PLAN_VERSION,
            "seed": seed,
            "scenario_count": max(1000, self.scenario_count),
            "residual_pool_size": len(residuals),
            "empirical_pool_size": len(outer_training),
            "plans": list(outer_plan_payloads),
            "plan_hashes": list(champion_plans),
            "validation_anchors": [rows[index].anchor_date.isoformat() for index in outer_validation],
        }
        combined_plan_payload = {
            **combined_plan_body,
            "plan_hash": _hash(combined_plan_body),
        }
        random_plan_row = self._persist_random_plan(
            session,
            market=market,
            scope="CANDIDATE_OUTER_BLOCK",
            plan_payload=combined_plan_payload,
            residual_pool_identity=_hash(residuals.tolist()),
        )
        outer_block_row = V341OuterEvaluationBlock(
            id=f"V341-OB-{market}-{outer_block_id[:28]}",
            protocol_version=PROTOCOL_VERSION,
            model_market=market,
            training_run_id=run.id,
            anchor_date=current_anchor,
            validation_start_date=rows[outer_validation[0]].anchor_date,
            validation_end_date=rows[outer_validation[-1]].anchor_date,
            training_anchors_json=[rows[index].anchor_date.isoformat() for index in outer_training],
            validation_anchors_json=[rows[index].anchor_date.isoformat() for index in outer_validation],
            block_hash=outer_block_id,
            consumed_at=self.now_provider(),
        )
        session.add(outer_block_row)
        session.flush()
        session.add(
            V341CandidateTrial(
                training_run_id=run.id,
                anchor_date=current_anchor,
                champion_model_id=champion_row.id,
                candidate_model_id=candidate_row.id,
                random_plan_id=random_plan_row.id,
                alpha=_decimal(selected_alpha),
                candidate_rank=1,
                outer_evaluation_block_id=outer_block_row.id,
                random_plan_hash=random_plan_row.plan_hash,
                loss_components_json={
                    "schema_version": LOSS_SCHEMA_VERSION,
                    "champion_outer": champion_outer.tolist(),
                    "candidate_outer": candidate_outer.tolist(),
                    "champion_outer_components": [dict(value) for value in champion_components],
                    "candidate_outer_components": [dict(value) for value in candidate_components],
                    "inner_alpha_anchor_losses": {
                        str(key): values for key, values in alpha_anchor_losses.items()
                    },
                    "inner_alpha_plan_hashes": {
                        str(key): values for key, values in alpha_plan_hashes.items()
                    },
                },
                promotion_json=decision,
                created_at=self.now_provider(),
            )
        )
        return candidate_row, decision

    def _structure_audit_trial(
        self,
        session: Session,
        market: str,
        current_anchor: date,
        available_matured: Sequence[V34TrainingSample],
        champion_state: V341ModelState,
        champion_row: V341ModelVersion,
        champion_profile: V341TrainingProfile,
        run: V341TrainingRun,
        candidate_number: int,
    ) -> tuple[V341ModelVersion, dict[str, Any]] | None:
        """Run a sealed low-frequency feature/window structure comparison."""

        rows = tuple(sorted(available_matured, key=lambda item: item.anchor_date))
        folds = purged_walk_forward_folds(rows, folds=5)
        if len(folds) < 2:
            return None
        inner_folds = folds[:-1]
        sealed_outer = self._sealed_outer_fold(session, market, rows)
        if sealed_outer is None:
            return None
        outer_training, outer_validation = sealed_outer
        snapshot = rows[-1].snapshot
        manifests = {
            FEATURE_SET_EXTENDED: build_feature_manifest(
                snapshot, feature_set_name=FEATURE_SET_EXTENDED
            ),
            FEATURE_SET_CORE: build_feature_manifest(
                snapshot, feature_set_name=FEATURE_SET_CORE
            ),
        }
        current_feature_set = champion_profile.feature_set_name
        current_window_mode = champion_profile.training_window_mode
        if current_feature_set not in manifests or current_window_mode not in {
            "ROLLING_520W",
            "EXPANDING_AVAILABLE_HISTORY",
        }:
            raise V341RuntimeError("champion has an unsupported structure profile")
        window_modes = ["ROLLING_520W"]
        if len(outer_training) > FORMAL_TRAINING_WEEKS:
            window_modes.append("EXPANDING_AVAILABLE_HISTORY")
        configurations: list[tuple[str, str, FeatureManifest]] = []
        for feature_set, candidate_manifest in manifests.items():
            for window_mode in window_modes:
                if (
                    feature_set == current_feature_set
                    and window_mode == current_window_mode
                ):
                    continue
                configurations.append(
                    (feature_set, window_mode, candidate_manifest)
                )
        if not configurations:
            return None
        current_manifest = manifests[current_feature_set]
        current_training = self._windowed_training_indices(
            outer_training, current_window_mode
        )
        seed = int.from_bytes(
            hashlib.sha256(
                f"{PROTOCOL_VERSION}:{market}:{current_anchor}:STRUCTURE:{candidate_number}".encode(
                    "utf-8"
                )
            ).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)
        residuals, _, _ = self._residual_pool(session, market, current_anchor)
        if not len(residuals):
            residuals = prior_standardized_residual_pool(seed)
        inner_plan_reference: dict[int, tuple[str, ...]] = {}
        inner_scored: list[
            tuple[float, str, str, FeatureManifest, tuple[float, ...], tuple[str, ...]]
        ] = []
        for feature_set, window_mode, candidate_manifest in configurations:
            configuration_losses: list[float] = []
            configuration_plans: list[str] = []
            for fold_number, (fold_training, fold_validation) in enumerate(
                inner_folds, start=1
            ):
                training_indices = self._windowed_training_indices(
                    fold_training, window_mode
                )
                losses, plans, _components, _ = self.training.fold_composite_losses(
                    market,
                    rows,
                    training_indices,
                    fold_validation,
                    alpha=champion_state.ridge_alpha,
                    feature_names=candidate_manifest.ordered_feature_names,
                    standardized_residuals=residuals,
                    shared_seed=seed + fold_number,
                    scenario_count=max(1000, self.scenario_count),
                )
                reference = inner_plan_reference.setdefault(fold_number, plans)
                if plans != reference:
                    raise V341RuntimeError(
                        "structure candidates diverged from the inner common plan"
                    )
                configuration_losses.extend(float(value) for value in losses)
                configuration_plans.extend(plans)
            if configuration_losses:
                inner_scored.append(
                    (
                        float(np.mean(configuration_losses)),
                        feature_set,
                        window_mode,
                        candidate_manifest,
                        tuple(configuration_losses),
                        tuple(configuration_plans),
                    )
                )
        if not inner_scored:
            return None
        (
            _selected_inner_score,
            selected_feature_set,
            selected_window_mode,
            selected_manifest,
            selected_inner_losses,
            selected_inner_plans,
        ) = min(inner_scored, key=lambda item: (item[0], item[1], item[2]))
        selected_outer_training = self._windowed_training_indices(
            outer_training, selected_window_mode
        )
        current_losses, current_plans, current_components, plan_payloads = (
            self.training.fold_composite_losses(
                market,
                rows,
                current_training,
                outer_validation,
                alpha=champion_state.ridge_alpha,
                feature_names=current_manifest.ordered_feature_names,
                standardized_residuals=residuals,
                shared_seed=seed,
                scenario_count=max(1000, self.scenario_count),
                capture_random_plans=True,
            )
        )
        selected_losses, selected_plans, selected_components, _ = (
            self.training.fold_composite_losses(
                market,
                rows,
                selected_outer_training,
                outer_validation,
                alpha=champion_state.ridge_alpha,
                feature_names=selected_manifest.ordered_feature_names,
                standardized_residuals=residuals,
                shared_seed=seed,
                scenario_count=max(1000, self.scenario_count),
            )
        )
        if selected_plans != current_plans:
            raise V341RuntimeError(
                "locked structure challenger diverged from the outer common plan"
            )
        outer_effective = rows[outer_validation[0]].anchor_date
        current_gate_rows = tuple(rows[index] for index in current_training)
        selected_gate_rows = tuple(rows[index] for index in selected_outer_training)
        current_gate_state = self.training.fit(
            market,
            current_gate_rows,
            version="STRUCTURE-GATE-CURRENT",
            parent_version=None,
            trained_through=max(
                row.label_end_date for row in current_gate_rows if row.label_end_date
            ),
            effective_from=outer_effective,
            alpha=champion_state.ridge_alpha,
            feature_names=current_manifest.ordered_feature_names,
        )
        selected_gate_state = self.training.fit(
            market,
            selected_gate_rows,
            version="STRUCTURE-GATE-CHALLENGER",
            parent_version=None,
            trained_through=max(
                row.label_end_date for row in selected_gate_rows if row.label_end_date
            ),
            effective_from=outer_effective,
            alpha=champion_state.ridge_alpha,
            feature_names=selected_manifest.ordered_feature_names,
        )
        shared_norm_ratio, shared_coefficient_features = (
            self._shared_standardized_coefficient_norm_ratio(
                current_gate_state, selected_gate_state
            )
        )
        deployment_rows = (
            rows[-FORMAL_TRAINING_WEEKS:]
            if selected_window_mode == "ROLLING_520W"
            else rows
        )
        next_anchor = self._next_complete_anchor_after(market, current_anchor)
        candidate_state = self.training.fit(
            market,
            deployment_rows,
            version=f"{market}-V341-S{candidate_number:04d}-{current_anchor.isoformat()}",
            parent_version=champion_state.version,
            trained_through=current_anchor,
            effective_from=next_anchor,
            alpha=champion_state.ridge_alpha,
            feature_names=selected_manifest.ordered_feature_names,
        )
        hard_gate_metrics = {
            "champion_coverage": float(
                np.mean([value["coverage"] for value in current_components])
            ),
            "candidate_coverage": float(
                np.mean([value["coverage"] for value in selected_components])
            ),
            "champion_turning_accuracy": float(
                np.mean([value["turning_type_correct"] for value in current_components])
            ),
            "candidate_turning_accuracy": float(
                np.mean([value["turning_type_correct"] for value in selected_components])
            ),
            "coefficient_norm_ratio": shared_norm_ratio,
        }
        decision = promotion_from_paired_losses(
            current_losses,
            selected_losses,
            seed=seed,
            fold_lengths=(len(outer_validation),),
            hard_gate_metrics=hard_gate_metrics,
        )
        decision.update(
            {
                "trial_type": "LOW_FREQUENCY_STRUCTURE_AUDIT",
                "current_structure": {
                    "feature_set": current_feature_set,
                    "window_mode": current_window_mode,
                    "daily_mode": "100_SESSION_RIDGE_ONLY",
                },
                "selected_structure": {
                    "feature_set": selected_feature_set,
                    "window_mode": selected_window_mode,
                    "daily_mode": "100_SESSION_RIDGE_ONLY",
                },
                "window_comparison_status": (
                    "EVALUATED"
                    if len(outer_training) > FORMAL_TRAINING_WEEKS
                    else "NOT_IDENTIFIABLE_AVAILABLE_HISTORY_WITHIN_520W"
                ),
                "structure_selection_scope": "INNER_PURGED_WALK_FORWARD_ONLY",
                "outer_block_role": "ONE_TIME_LOCKED_CHALLENGER_PROMOTION_ONLY",
                "configuration_scores": [
                    {
                        "feature_set": feature_set,
                        "window_mode": window_mode,
                        "mean_loss": score,
                    }
                    for score, feature_set, window_mode, *_rest in inner_scored
                ],
                "selected_inner_losses": list(selected_inner_losses),
                "selected_inner_random_plan_hashes": list(selected_inner_plans),
                "coefficient_stability_scope": "SHARED_STANDARDIZED_FEATURES",
                "shared_coefficient_features": list(shared_coefficient_features),
                "random_plan_semantics": "COMMON_RESIDUAL_INDEX_PLAN_SHARED_BY_ALL_STRUCTURES",
                "outer_random_plan_hashes": list(current_plans),
            }
        )
        candidate_state = replace(candidate_state, validation_metrics=decision)
        candidate_profile = self._persist_profile(
            session,
            market,
            selected_manifest,
            training_window_mode=selected_window_mode,
        )
        candidate_row = self._persist_model(
            session,
            candidate_state,
            candidate_profile.id,
            status="staged" if decision["promoted"] else "rejected",
            promotion_gate=decision,
        )
        outer_training_audit = {
            "champion": {
                "feature_set": current_feature_set,
                "window_mode": current_window_mode,
                "anchors": [
                    rows[index].anchor_date.isoformat() for index in current_training
                ],
            },
            "challenger": {
                "feature_set": selected_feature_set,
                "window_mode": selected_window_mode,
                "anchors": [
                    rows[index].anchor_date.isoformat()
                    for index in selected_outer_training
                ],
            },
        }
        outer_block_id = _hash(
            {
                "scope": "STRUCTURE_AUDIT",
                "market": market,
                "current_anchor": current_anchor,
                "training_anchors": outer_training_audit,
                "validation_anchors": [rows[index].anchor_date for index in outer_validation],
            }
        )
        combined_plan_body = {
            "version": CANDIDATE_RANDOM_PLAN_VERSION,
            "scope": "STRUCTURE_AUDIT_OUTER_BLOCK",
            "seed": seed,
            "scenario_count": max(1000, self.scenario_count),
            "residual_pool_size": len(residuals),
            "empirical_pool_size": len(outer_training),
            "plans": list(plan_payloads),
            "plan_hashes": list(current_plans),
            "validation_anchors": [
                rows[index].anchor_date.isoformat() for index in outer_validation
            ],
        }
        combined_plan_payload = {
            **combined_plan_body,
            "plan_hash": _hash(combined_plan_body),
        }
        random_plan_row = self._persist_random_plan(
            session,
            market=market,
            scope="STRUCTURE_AUDIT_OUTER_BLOCK",
            plan_payload=combined_plan_payload,
            residual_pool_identity=_hash(residuals.tolist()),
        )
        outer_block_row = V341OuterEvaluationBlock(
            id=f"V341-SOB-{market}-{outer_block_id[:27]}",
            protocol_version=PROTOCOL_VERSION,
            model_market=market,
            training_run_id=run.id,
            anchor_date=current_anchor,
            validation_start_date=rows[outer_validation[0]].anchor_date,
            validation_end_date=rows[outer_validation[-1]].anchor_date,
            training_anchors_json=outer_training_audit,
            validation_anchors_json=[
                rows[index].anchor_date.isoformat() for index in outer_validation
            ],
            block_hash=outer_block_id,
            consumed_at=self.now_provider(),
        )
        session.add(outer_block_row)
        session.flush()
        session.add(
            V341CandidateTrial(
                training_run_id=run.id,
                anchor_date=current_anchor,
                champion_model_id=champion_row.id,
                candidate_model_id=candidate_row.id,
                random_plan_id=random_plan_row.id,
                alpha=_decimal(champion_state.ridge_alpha),
                candidate_rank=1,
                outer_evaluation_block_id=outer_block_row.id,
                random_plan_hash=random_plan_row.plan_hash,
                loss_components_json={
                    "schema_version": LOSS_SCHEMA_VERSION,
                    "champion_outer": current_losses.tolist(),
                    "candidate_outer": selected_losses.tolist(),
                    "champion_outer_components": [
                        dict(value) for value in current_components
                    ],
                    "candidate_outer_components": [
                        dict(value) for value in selected_components
                    ],
                    "inner_configuration_losses": [
                        {
                            "feature_set": feature_set,
                            "window_mode": window_mode,
                            "mean_loss": score,
                            "losses": list(losses),
                            "random_plan_hashes": list(plans),
                        }
                        for score, feature_set, window_mode, _manifest, losses, plans
                        in inner_scored
                    ],
                    "inner_fold_audit": [
                        {
                            "training_anchors": [
                                rows[index].anchor_date.isoformat()
                                for index in fold_training
                            ],
                            "validation_anchors": [
                                rows[index].anchor_date.isoformat()
                                for index in fold_validation
                            ],
                        }
                        for fold_training, fold_validation in inner_folds
                    ],
                    "outer_training_audit": outer_training_audit,
                },
                promotion_json=decision,
                created_at=self.now_provider(),
            )
        )
        return candidate_row, decision

    def _optimizer_state(
        self,
        session: Session,
        market: str,
    ) -> V341OptimizerState | None:
        return session.scalar(
            select(V341OptimizerState).where(
                V341OptimizerState.protocol_version == PROTOCOL_VERSION,
                V341OptimizerState.model_market == market,
            )
        )

    def _write_optimizer(
        self,
        session: Session,
        *,
        market: str,
        profile: V341TrainingProfile,
        anchor: date,
        last_consumed_mature_anchor: date | None,
        last_candidate_anchor: date | None,
        champion_model_id: str,
        weekly_iteration_count: int,
        candidate_training_count: int,
        champion_promotion_count: int,
        last_structure_audit_anchor: date | None = None,
        state_metadata: Mapping[str, Any] | None = None,
    ) -> V341OptimizerState:
        row = self._optimizer_state(session, market)
        effective_structure_anchor = (
            row.last_structure_audit_anchor
            if last_structure_audit_anchor is None and row is not None
            else last_structure_audit_anchor
        )
        preserved_metadata = (
            {}
            if row is None
            else {
                key: value
                for key, value in dict(row.state_json).items()
                if key in {
                    "last_candidate_effective_count",
                    "last_structure_audit_effective_count",
                    "structure_candidate_status",
                    "structure_audit",
                }
            }
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "market": market,
            "last_anchor_date": anchor.isoformat(),
            "last_consumed_mature_anchor": (
                None
                if last_consumed_mature_anchor is None
                else last_consumed_mature_anchor.isoformat()
            ),
            "last_candidate_anchor": (
                None if last_candidate_anchor is None else last_candidate_anchor.isoformat()
            ),
            "weekly_iteration_count": weekly_iteration_count,
            "candidate_training_count": candidate_training_count,
            "champion_promotion_count": champion_promotion_count,
            "champion_model_id": champion_model_id,
            "trigger": "FOUR_NEW_MATURE_ANCHOR_EVENTS_NOT_ROLLING_LENGTH_DELTA",
            "last_structure_audit_anchor": (
                None
                if effective_structure_anchor is None
                else effective_structure_anchor.isoformat()
            ),
            **preserved_metadata,
            **dict(state_metadata or {}),
        }
        digest = _hash(payload)
        if row is None:
            row = V341OptimizerState(
                protocol_version=PROTOCOL_VERSION,
                model_market=market,
                profile_id=profile.id,
                last_anchor_date=anchor,
                last_consumed_mature_anchor=last_consumed_mature_anchor,
                last_candidate_anchor=last_candidate_anchor,
                last_structure_audit_anchor=effective_structure_anchor,
                weekly_iteration_count=weekly_iteration_count,
                candidate_training_count=candidate_training_count,
                champion_promotion_count=champion_promotion_count,
                champion_model_id=champion_model_id,
                state_json=payload,
                state_hash=digest,
                updated_at=self.now_provider(),
            )
            session.add(row)
        else:
            row.profile_id = profile.id
            row.last_anchor_date = anchor
            row.last_consumed_mature_anchor = last_consumed_mature_anchor
            row.last_candidate_anchor = last_candidate_anchor
            row.last_structure_audit_anchor = effective_structure_anchor
            row.weekly_iteration_count = weekly_iteration_count
            row.candidate_training_count = candidate_training_count
            row.champion_promotion_count = champion_promotion_count
            row.champion_model_id = champion_model_id
            row.state_json = payload
            row.state_hash = digest
            row.updated_at = self.now_provider()
        session.flush()
        return row

    def _create_run(
        self,
        market: str,
        run_type: str,
        through: date | None,
        profile_id: str,
    ) -> str:
        run_id = f"V341-{run_type.upper()}-{market}-{uuid4().hex[:12]}"
        with self.sessions() as session, session.begin():
            session.add(
                V341TrainingRun(
                    id=run_id,
                    protocol_version=PROTOCOL_VERSION,
                    model_market=market,
                    profile_id=profile_id,
                    run_type=run_type,
                    status="queued",
                    current_stage="queued",
                    horizon_weeks=13,
                    requested_through_anchor=through,
                    maximum_backlog_weeks=self.maximum_backlog_weeks,
                    started_at=self.now_provider(),
                    completed_at=None,
                    result_json={},
                    error_code=None,
                    error_message=None,
                )
            )
        return run_id

    def _process_week(
        self,
        run_id: str,
        market: str,
        current_anchor: date,
        snapshots: Sequence[Any],
        samples: Sequence[V34TrainingSample],
        manifest: FeatureManifest,
        *,
        structure_samples: Sequence[V34TrainingSample] | None = None,
    ) -> dict[str, Any]:
        by_anchor = {sample.anchor_date: sample for sample in samples}
        current = by_anchor[current_anchor]
        with self.sessions() as session, session.begin():
            existing_iteration = session.scalar(
                select(V341TrainingIteration).where(
                    V341TrainingIteration.protocol_version == PROTOCOL_VERSION,
                    V341TrainingIteration.model_market == market,
                    V341TrainingIteration.anchor_date == current_anchor,
                )
            )
            if existing_iteration is not None:
                return {
                    "status": "IDEMPOTENT_REUSE",
                    "anchor_date": current_anchor.isoformat(),
                    "iteration": existing_iteration.weekly_iteration_number,
                }
            frozen_forecast = session.scalar(
                select(V341ForecastRow).where(
                    V341ForecastRow.protocol_version == PROTOCOL_VERSION,
                    V341ForecastRow.model_market == market,
                    V341ForecastRow.forecast_anchor_date == current_anchor,
                )
            )
            run = session.get(V341TrainingRun, run_id)
            if run is None:
                raise V341RuntimeError("V3.4.1 training run row disappeared")
            profile = self._persist_profile(session, market, manifest)
            adapter = self._persist_adapter(session, market)
            optimizer = self._optimizer_state(session, market)
            iteration_number = 1 if optimizer is None else optimizer.weekly_iteration_count + 1
            run.status = "running"
            run.current_stage = f"weekly_transaction:{current_anchor.isoformat()}"

            # The current-anchor forecast is issued before any outcome that
            # becomes mature at this anchor is evaluated or admitted to the
            # residual/calibration pools.  This preserves a single forecast
            # identity whether analysis or training is invoked first.
            loaded = self._load_state(session, market, current_anchor)
            if loaded is None:
                raise V341RuntimeError(
                    f"no effective V3.4.1 model at {current_anchor}"
                )
            champion_state, champion_row = loaded
            if frozen_forecast is not None:
                frozen_model = session.get(V341ModelVersion, frozen_forecast.model_version_id)
                if frozen_model is None:
                    raise V341RuntimeError("frozen forecast references a missing model")
                champion_row = frozen_model
                champion_state = state_from_payload(frozen_model.parameters_json)
            champion_profile = session.get(V341TrainingProfile, champion_row.profile_id)
            if champion_profile is None:
                raise V341RuntimeError("champion references a missing training profile")
            active_manifest = build_feature_manifest(
                current.snapshot,
                feature_set_name=champion_profile.feature_set_name,
            )
            complete_available_matured = eligible_fully_matured(
                samples if structure_samples is None else structure_samples,
                current_anchor,
            )
            matured = self._scenario_matured_rows(
                champion_profile,
                samples,
                samples if structure_samples is None else structure_samples,
                current_anchor,
            )
            if frozen_forecast is not None:
                forecast = frozen_forecast
                label_end = frozen_forecast.label_end_date
            else:
                snapshot_row = self._persist_snapshot(
                    session, current.snapshot, active_manifest
                )
                standardized, sigmas, residual_metadata = self._residual_pool(
                    session, market, current_anchor
                )
                active_calibrators = self._active_calibrators(
                    session, market, current_anchor
                )
                path = self.training.predict(champion_state, current.snapshot)
                scenario = generate_scenario_forecast(
                    path,
                    current.snapshot,
                    matured,
                    standardized_residuals=standardized,
                    source_sigmas=sigmas,
                    residual_pool_metadata=residual_metadata,
                    calibrators=active_calibrators,
                    scenario_count=self.scenario_count,
                )
                health_status = self._persist_health_snapshot(
                    session,
                    market,
                    current_anchor,
                    champion_row,
                    path,
                    scenario.reliability,
                )
                scenario = self._scenario_with_health(scenario, health_status)
                label_end = (
                    current.label_end_date
                    if current.label_end_date is not None
                    else self._future_label_end(self.calendar, market, current_anchor)
                )
                forecast = self._persist_forecast(
                    session,
                    snapshot_row,
                    champion_row,
                    adapter,
                    scenario,
                    label_end_date=label_end,
                )

            maturity_counts = self._mature_forecasts(
                session, market, snapshots, current_anchor
            )
            calibrators_created = self._fit_calibrators(
                session, market, current_anchor
            )

            last_consumed = (
                None if optimizer is None else optimizer.last_consumed_mature_anchor
            )
            new_mature_anchors = count_new_matured_anchors(matured, last_consumed)
            previous_candidate_effective_count = (
                0
                if optimizer is None
                else int(optimizer.state_json.get("last_candidate_effective_count", 0))
            )
            independent_count = conservative_independent_sample_count(
                tuple(
                    (sample.anchor_date, sample.label_end_date)
                    for sample in matured
                    if sample.label_end_date is not None
                )
            )
            structure_independent_count = conservative_independent_sample_count(
                tuple(
                    (sample.anchor_date, sample.label_end_date)
                    for sample in complete_available_matured
                    if sample.label_end_date is not None
                )
            )
            cadence_due = (
                optimizer is None
                or optimizer.last_candidate_anchor is None
                or current_anchor >= optimizer.last_candidate_anchor + timedelta(weeks=4)
            )
            training_triggered = (
                len(new_mature_anchors) >= MIN_NEW_MATURED_FOR_CANDIDATE
                and cadence_due
                and independent_count > previous_candidate_effective_count
            )
            promoted = False
            candidate_gate_failures: list[str] = []
            if len(new_mature_anchors) < MIN_NEW_MATURED_FOR_CANDIDATE:
                candidate_gate_failures.append("FEWER_THAN_FOUR_NEW_MATURE_ANCHORS")
            if not cadence_due:
                candidate_gate_failures.append("FOUR_WEEK_CADENCE_NOT_DUE")
            if independent_count <= previous_candidate_effective_count:
                candidate_gate_failures.append("EFFECTIVE_SAMPLE_COUNT_DID_NOT_INCREASE")
            trial_payload: dict[str, Any] = {
                "reason": (
                    candidate_gate_failures[0]
                    if len(candidate_gate_failures) == 1
                    else "MULTIPLE_CANDIDATE_GATES_NOT_MET"
                ),
                "gate_failures": candidate_gate_failures,
                "new_mature_anchors": [value.isoformat() for value in new_mature_anchors],
                "cadence_due": cadence_due,
                "previous_effective_count": previous_candidate_effective_count,
                "current_effective_count": independent_count,
            }
            candidate_count = 0 if optimizer is None else optimizer.candidate_training_count
            promotion_count = 0 if optimizer is None else optimizer.champion_promotion_count
            champion_after = champion_row
            last_candidate_anchor = (
                None if optimizer is None else optimizer.last_candidate_anchor
            )
            if training_triggered:
                trial = self._candidate_trial(
                    session,
                    market,
                    current_anchor,
                    matured,
                    champion_state,
                    champion_row,
                    champion_profile,
                    run,
                    candidate_count + 1,
                    active_manifest,
                )
                if trial is None:
                    trial_payload = {
                        "reason": "INSUFFICIENT_NEW_SEALED_OUTER_BLOCK_OR_PURGED_FOLDS",
                        "new_mature_anchors": [
                            value.isoformat() for value in new_mature_anchors
                        ],
                    }
                else:
                    candidate_count += 1
                    last_consumed = new_mature_anchors[
                        MIN_NEW_MATURED_FOR_CANDIDATE - 1
                    ]
                    last_candidate_anchor = current_anchor
                    previous_candidate_effective_count = independent_count
                    candidate_row, trial_payload = trial
                    promoted = bool(trial_payload["promoted"])
                    if promoted:
                        champion_after = candidate_row
                        promotion_count += 1

            previous_structure_count = (
                0
                if optimizer is None
                else int(
                    optimizer.state_json.get("last_structure_audit_effective_count", 0)
                )
            )
            previous_structure_anchor = (
                None if optimizer is None else optimizer.last_structure_audit_anchor
            )
            structure_due = (
                (previous_structure_anchor is None or current_anchor >= previous_structure_anchor + timedelta(weeks=STRUCTURE_AUDIT_INTERVAL_WEEKS))
                and structure_independent_count >= previous_structure_count + 3
                and not training_triggered
                and (
                    optimizer is None
                    or optimizer.state_json.get("structure_candidate_status")
                    != "STRUCTURE_CANDIDATE_RUNNING"
                )
            )
            structure_audit = {
                "status": (
                    "DUE_AWAITING_SEALED_STRUCTURE_TRIAL"
                    if structure_due else "NOT_DUE"
                ),
                "anchor": current_anchor.isoformat(),
                "interval_weeks_required": STRUCTURE_AUDIT_INTERVAL_WEEKS,
                "effective_sample_increase_required": 3,
                "previous_effective_count": previous_structure_count,
                "current_effective_count": structure_independent_count,
                "candidate_dimensions": {
                    "feature_sets": ["CORE", "EXTENDED"],
                    "window_modes": ["ROLLING_520W", "EXPANDING_AVAILABLE_HISTORY"],
                    "daily_modes": ["100_SESSION_RIDGE_ONLY"],
                },
                "decision": "WAITING_FOR_INTERVAL_AND_EFFECTIVE_SAMPLE_GATE",
            }
            last_structure_anchor = previous_structure_anchor
            if structure_due:
                structure_trial = self._structure_audit_trial(
                    session,
                    market,
                    current_anchor,
                    complete_available_matured,
                    champion_state,
                    champion_row,
                    champion_profile,
                    run,
                    candidate_count + 1,
                )
                if structure_trial is None:
                    structure_audit = {
                        **structure_audit,
                        "status": "DUE_NOT_RUN_NO_NEW_SEALED_OUTER_BLOCK",
                        "decision": "NOT_RUN",
                    }
                else:
                    structure_candidate, structure_decision = structure_trial
                    candidate_count += 1
                    structure_promoted = bool(structure_decision["promoted"])
                    if structure_promoted:
                        champion_after = structure_candidate
                        promotion_count += 1
                        promoted = True
                    last_structure_anchor = current_anchor
                    previous_structure_count = structure_independent_count
                    structure_audit = {
                        **structure_audit,
                        "status": (
                            "COMPLETED_STRUCTURE_PROMOTED"
                            if structure_promoted
                            else "COMPLETED_CURRENT_STRUCTURE_RETAINED"
                        ),
                        "decision": structure_decision,
                    }
            session.add(
                V341TrainingIteration(
                    protocol_version=PROTOCOL_VERSION,
                    training_run_id=run_id,
                    forecast_id=forecast.id,
                    model_market=market,
                    anchor_date=current_anchor,
                    weekly_iteration_number=iteration_number,
                    forecast_model_id=champion_row.id,
                    champion_after_model_id=champion_after.id,
                    training_triggered=training_triggered,
                    promoted=promoted,
                    newly_matured_count=len(new_mature_anchors),
                    raw_matured_sample_count=len(matured),
                    effective_independent_sample_count=independent_count,
                    evaluation_available_date=label_end,
                    validation_json={
                        "protocol_version": PROTOCOL_VERSION,
                        "loss_schema_version": LOSS_SCHEMA_VERSION,
                        "nested_purged_walk_forward": True,
                        "rolling_weeks": FORMAL_TRAINING_WEEKS,
                        "maturity_created": maturity_counts,
                        "calibrators_created": calibrators_created,
                        "candidate": trial_payload,
                        "structure_audit": structure_audit,
                    },
                    completed_at=self.now_provider(),
                )
            )
            optimizer_profile = session.get(V341TrainingProfile, champion_after.profile_id)
            if optimizer_profile is None:
                raise V341RuntimeError("champion profile disappeared before optimizer write")
            self._write_optimizer(
                session,
                market=market,
                profile=optimizer_profile,
                anchor=current_anchor,
                last_consumed_mature_anchor=last_consumed,
                last_candidate_anchor=last_candidate_anchor,
                champion_model_id=champion_after.id,
                weekly_iteration_count=iteration_number,
                candidate_training_count=candidate_count,
                champion_promotion_count=promotion_count,
                last_structure_audit_anchor=last_structure_anchor,
                state_metadata={
                    "last_candidate_effective_count": previous_candidate_effective_count,
                    "last_structure_audit_effective_count": previous_structure_count,
                    "structure_candidate_status": "NONE_RUNNING",
                    "structure_audit": structure_audit,
                },
            )
            result = dict(run.result_json)
            result.update(
                {
                    "market": market,
                    "latest_anchor_date": current_anchor.isoformat(),
                    "weekly_iteration_count": iteration_number,
                    "candidate_training_count": candidate_count,
                    "champion_promotion_count": promotion_count,
                    "champion_model_id": champion_after.id,
                }
            )
            run.result_json = result
            return {
                "status": "PROCESSED",
                "anchor_date": current_anchor.isoformat(),
                "iteration": iteration_number,
                "training_triggered": training_triggered,
                "promoted": promoted,
                "forecast_hash": forecast.forecast_hash,
            }

    def _mark_run_failed(self, run_id: str, exc: BaseException) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V341TrainingRun, run_id)
            if run is not None:
                run.status = "failed"
                run.current_stage = "failed_stop_backlog"
                run.completed_at = self.now_provider()
                run.error_code = type(exc).__name__
                run.error_message = str(exc)

    def _complete_run(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self.sessions() as session, session.begin():
            run = session.get(V341TrainingRun, run_id)
            if run is None:
                raise V341RuntimeError("V3.4.1 training run row disappeared at completion")
            run.status = "completed"
            run.current_stage = str(payload.get("status", "completed")).lower()
            run.completed_at = self.now_provider()
            run.result_json = dict(payload)
        return self.training_status(run_id)

    def _prepare_history(
        self,
        market: str,
        *,
        through: date | None = None,
    ) -> tuple[tuple[date, ...], tuple[Any, ...], tuple[V34TrainingSample, ...]]:
        anchors = self._complete_anchors(market)
        if through is not None:
            anchors = tuple(anchor for anchor in anchors if anchor <= through)
        if not anchors:
            raise V341RuntimeError("no complete anchors at requested cutoff")
        formal_start_index = max(0, len(anchors) - FORMAL_TRAINING_WEEKS)
        build_start_index = max(0, formal_start_index - FEATURE_WARMUP_WEEKS)
        build_anchors = anchors[build_start_index:]
        snapshots = self._snapshots(market, build_anchors)
        samples = build_training_samples(snapshots)
        return build_anchors, snapshots, samples

    @staticmethod
    def _formal_bootstrap_anchors(anchors: Sequence[date]) -> tuple[date, ...]:
        """Return the formal replay anchors after an excluded 52-week warm-up."""

        available = max(0, len(anchors) - FEATURE_WARMUP_WEEKS)
        count = min(FORMAL_TRAINING_WEEKS, available)
        if count < MINIMUM_FORMAL_BOOTSTRAP_WEEKS:
            raise V341RuntimeError(
                "insufficient history for formal V3.4.1 bootstrap: "
                f"{count} anchors after {FEATURE_WARMUP_WEEKS}-week warm-up"
            )
        return tuple(anchors[-count:])

    def reconcile_bootstrap_prefix(
        self,
        market: str,
        *,
        through: date | None = None,
    ) -> dict[str, Any]:
        """Read-only validation of the committed formal replay prefix.

        A valid partial chain must be an exact oldest-to-newest prefix of the
        planned formal anchors.  Ambiguous gaps are reported and never rebuilt
        automatically.
        """

        self.validate_market(market)
        complete = self._complete_anchors(market)
        if through is not None:
            complete = tuple(anchor for anchor in complete if anchor <= through)
        with self.sessions() as session:
            iterations = list(
                session.scalars(
                    select(V341TrainingIteration)
                    .where(
                        V341TrainingIteration.protocol_version == PROTOCOL_VERSION,
                        V341TrainingIteration.model_market == market,
                    )
                    .order_by(V341TrainingIteration.weekly_iteration_number)
                )
            )
            optimizer = self._optimizer_state(session, market)
            seed_model = session.scalar(
                select(V341ModelVersion)
                .where(
                    V341ModelVersion.protocol_version == PROTOCOL_VERSION,
                    V341ModelVersion.model_market == market,
                    V341ModelVersion.parent_model_id.is_(None),
                )
                .order_by(V341ModelVersion.created_at, V341ModelVersion.id)
                .limit(1)
            )

            # The executable bootstrap applies two distinct gates: the explicit
            # 52-week feature warm-up and then the minimum 30 fully matured
            # 13-week labels needed to fit the seed model.  Reconciliation must
            # use the same *actual usable* formal range; subtracting only 52 raw
            # market weeks incorrectly planned 513 anchors for 159941 although
            # its first valid seed anchor is 2017-01-13 and the real run has 489.
            boundary = (
                {}
                if seed_model is None
                else dict(seed_model.metrics_json.get("boundary_audit", {}))
            )
            recorded_start = boundary.get("earliest_formal_anchor")
            if recorded_start is not None:
                first_usable = date.fromisoformat(str(recorded_start))
                expected = tuple(anchor for anchor in complete if anchor >= first_usable)
                expected = expected[-FORMAL_TRAINING_WEEKS:]
                formal_before_seed_gate = int(
                    boundary.get("formal_anchor_count_before_minimum_gate", len(expected))
                )
            else:
                planned = self._formal_bootstrap_anchors(complete)
                planned_set = set(planned)
                _, snapshots, samples = self._prepare_history(market, through=through)
                formal_samples = tuple(
                    sample for sample in samples if sample.anchor_date in planned_set
                )
                first_index = next(
                    (
                        index
                        for index, sample in enumerate(formal_samples)
                        if len(eligible_fully_matured(samples, sample.anchor_date))
                        >= MINIMUM_TRAINING_WEEKS
                    ),
                    None,
                )
                if first_index is None:
                    raise V341RuntimeError(
                        "no anchor has 30 fully matured 13-week seed labels"
                    )
                expected = tuple(
                    sample.anchor_date for sample in formal_samples[first_index:]
                )
                formal_before_seed_gate = len(planned)
            issues: list[str] = []
            actual_anchors = tuple(row.anchor_date for row in iterations)
            expected_prefix = expected[: len(iterations)]
            if len(iterations) > len(expected):
                issues.append("ITERATION_COUNT_EXCEEDS_FORMAL_PLAN")
            elif actual_anchors != expected_prefix:
                first_mismatch = next(
                    (
                        index
                        for index, (actual, planned) in enumerate(
                            zip(actual_anchors, expected_prefix), start=1
                        )
                        if actual != planned
                    ),
                    min(len(actual_anchors), len(expected_prefix)) + 1,
                )
                issues.append(f"NON_PREFIX_ANCHOR_AT_ITERATION_{first_mismatch}")
            numbers = [row.weekly_iteration_number for row in iterations]
            if numbers != list(range(1, len(iterations) + 1)):
                issues.append("NON_CONTIGUOUS_ITERATION_NUMBERS")
            for row in iterations:
                forecast = session.get(V341ForecastRow, row.forecast_id)
                if (
                    forecast is None
                    or forecast.protocol_version != PROTOCOL_VERSION
                    or forecast.model_market != market
                    or forecast.forecast_anchor_date != row.anchor_date
                ):
                    issues.append(
                        f"MISSING_OR_MISMATCHED_FORECAST_AT_{row.anchor_date.isoformat()}"
                    )
                    break
            if iterations:
                last = iterations[-1]
                if optimizer is None:
                    issues.append("MISSING_OPTIMIZER_FOR_COMMITTED_PREFIX")
                elif (
                    optimizer.weekly_iteration_count != len(iterations)
                    or optimizer.last_anchor_date != last.anchor_date
                    or optimizer.champion_model_id != last.champion_after_model_id
                ):
                    issues.append("OPTIMIZER_PREFIX_MISMATCH")
            elif optimizer is not None:
                issues.append("OPTIMIZER_EXISTS_WITHOUT_FORMAL_ITERATION")

        if issues:
            state = "AMBIGUOUS_INVALID"
        elif not iterations:
            state = "NOT_STARTED"
        elif len(iterations) == len(expected):
            state = "COMPLETED"
        else:
            state = "PARTIAL_VALID_PREFIX"
        return {
            "market": market,
            "state": state,
            "expected_iteration_count": len(expected),
            "completed_iteration_count": len(iterations),
            "feature_warmup_weeks": FEATURE_WARMUP_WEEKS,
            "formal_anchor_count_before_seed_maturity_gate": formal_before_seed_gate,
            "seed_maturity_gate_excluded_weeks": formal_before_seed_gate - len(expected),
            "first_expected_anchor": expected[0].isoformat(),
            "last_expected_anchor": expected[-1].isoformat(),
            "first_missing_anchor": (
                expected[len(iterations)].isoformat()
                if not issues and len(iterations) < len(expected)
                else None
            ),
            "issues": issues,
        }

    def _prepare_structure_samples(
        self,
        market: str,
        *,
        through: date,
    ) -> tuple[V34TrainingSample, ...]:
        """Load all point-in-time samples available to a low-frequency audit.

        The regular weekly forecast path deliberately remains bounded to 520
        formal weeks plus feature warm-up.  Structure audits need the complete
        historical prefix so ROLLING_520W and EXPANDING_AVAILABLE_HISTORY are
        both executable candidates without exposing observations after the
        audit anchor.
        """

        anchors = tuple(
            anchor for anchor in self._complete_anchors(market) if anchor <= through
        )
        if not anchors:
            raise V341RuntimeError("no complete anchors for structure audit")
        snapshots = self._snapshots(market, anchors)
        samples = build_training_samples(snapshots)
        return tuple(
            sample
            for sample in samples
            if sample.anchor_date <= through
        )

    @staticmethod
    def _scenario_matured_rows(
        profile: V341TrainingProfile,
        bounded_samples: Sequence[V34TrainingSample],
        complete_samples: Sequence[V34TrainingSample],
        anchor: date,
    ) -> tuple[V34TrainingSample, ...]:
        """Select the single authoritative scenario-history pool."""

        bounded = eligible_fully_matured(bounded_samples, anchor)
        complete = eligible_fully_matured(complete_samples, anchor)
        if profile.training_window_mode == "EXPANDING_AVAILABLE_HISTORY":
            return tuple(complete)
        if profile.training_window_mode == "ROLLING_520W":
            return tuple(bounded[-FORMAL_TRAINING_WEEKS:])
        raise V341RuntimeError(
            f"unsupported scenario training window: {profile.training_window_mode}"
        )

    def bootstrap_sync(self, market: str, *, run_id: str | None = None) -> dict[str, Any]:
        self.validate_market(market)
        with self._locks[market]:
            with self.sessions() as lookup:
                if self._optimizer_state(lookup, market) is not None:
                    raise V341RuntimeError(
                        "V3.4.1 bootstrap already exists; use incremental training"
                    )
            anchors, snapshots, samples = self._prepare_history(market)
            if not snapshots:
                raise V341RuntimeError("no V3.4.1 feature snapshots")
            structure_samples = self._prepare_structure_samples(
                market, through=snapshots[-1].cutoff_date
            )
            manifest = build_feature_manifest(
                snapshots[0], feature_set_name=FEATURE_SET_EXTENDED
            )
            planned_formal = self._formal_bootstrap_anchors(anchors)
            formal_anchor_set = set(planned_formal)
            formal = tuple(
                sample for sample in samples if sample.anchor_date in formal_anchor_set
            )
            first_index = next(
                (
                    index
                    for index, sample in enumerate(formal)
                    if len(eligible_fully_matured(samples, sample.anchor_date))
                    >= MINIMUM_TRAINING_WEEKS
                ),
                None,
            )
            if first_index is None:
                raise V341RuntimeError(
                    "no anchor has 30 fully matured 13-week seed labels"
                )
            formal = formal[first_index:]
            first = formal[0]
            seed_rows = tuple(
                sample
                for sample in eligible_fully_matured(samples, first.anchor_date)
                if sample.anchor_date >= first.anchor_date - timedelta(weeks=FORMAL_TRAINING_WEEKS)
            )
            default_alpha = 5.0 if market == "399006" else 8.0
            effective_from = first.anchor_date
            seed_state = self.training.fit(
                market,
                seed_rows,
                version=f"{market}-V341-SEED-{first.anchor_date.isoformat()}",
                parent_version=None,
                trained_through=first.anchor_date,
                effective_from=effective_from,
                alpha=default_alpha,
                feature_names=manifest.ordered_feature_names,
                validation_metrics={
                    "seed": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "training_window_mode": "ROLLING_520W",
                    "boundary_audit": {
                        "earliest_source_anchor": anchors[0].isoformat(),
                        "earliest_feature_anchor": snapshots[0].cutoff_date.isoformat(),
                        "earliest_supervised_anchor": min(
                            sample.anchor_date
                            for sample in samples
                            if sample.label_end_date is not None
                        ).isoformat(),
                        "earliest_formal_anchor": formal[0].anchor_date.isoformat(),
                        "latest_matured_anchor": max(
                            sample.anchor_date for sample in seed_rows
                        ).isoformat(),
                        "formal_anchor_count_before_minimum_gate": len(formal_anchor_set),
                        "source_warmup_anchor_count": len(anchors) - len(formal_anchor_set),
                    },
                },
            )
            with self.sessions() as session, session.begin():
                profile = self._persist_profile(session, market, manifest)
                seed_row = self._persist_model(
                    session, seed_state, profile.id, status="champion"
                )
                self._persist_adapter(session, market)
                last_seed_anchor = max(sample.anchor_date for sample in seed_rows)
                self._write_optimizer(
                    session,
                    market=market,
                    profile=profile,
                    anchor=first.anchor_date - timedelta(days=1),
                    last_consumed_mature_anchor=last_seed_anchor,
                    last_candidate_anchor=None,
                    champion_model_id=seed_row.id,
                    weekly_iteration_count=0,
                    candidate_training_count=0,
                    champion_promotion_count=0,
                )
            run_id = run_id or self._create_run(
                market, "bootstrap", formal[-1].anchor_date, profile.id
            )
            processed: list[dict[str, Any]] = []
            try:
                for current in formal:
                    processed.append(
                        self._process_week(
                            run_id,
                            market,
                            current.anchor_date,
                            snapshots,
                            samples,
                            manifest,
                            structure_samples=structure_samples,
                        )
                    )
            except BaseException as exc:
                self._mark_run_failed(run_id, exc)
                raise
            return self._complete_run(
                run_id,
                {
                    "status": "COMPLETED",
                    "market": market,
                    "processed_week_count": len(processed),
                    "first_anchor_date": formal[0].anchor_date.isoformat(),
                    "latest_anchor_date": formal[-1].anchor_date.isoformat(),
                    "boundary_audit": seed_state.validation_metrics["boundary_audit"],
                    "no_early_termination": True,
                },
            )

    def incremental_sync(self, market: str, *, run_id: str | None = None) -> dict[str, Any]:
        self.validate_market(market)
        with self._locks[market]:
            snapshots: tuple[Any, ...] = ()
            samples: tuple[V34TrainingSample, ...] = ()
            try:
                anchors = self._complete_anchors(market)
            except V341RuntimeError:
                anchors, snapshots, samples = self._prepare_history(market)
            with self.sessions() as session:
                optimizer = self._optimizer_state(session, market)
                if optimizer is None:
                    raise V341RuntimeError("V3.4.1 has not been bootstrapped")
                profile_id = optimizer.profile_id
                last_anchor = optimizer.last_anchor_date
            backlog = [anchor for anchor in anchors if anchor > last_anchor]
            if not backlog:
                payload = {
                    "status": "NOT_DUE",
                    "market": market,
                    "reason": "NO_NEW_COMPLETE_MARKET_WEEK",
                    "last_anchor_date": last_anchor.isoformat(),
                }
                if run_id is None:
                    return payload
                return self._complete_run(run_id, payload)
            if not snapshots:
                _history_anchors, snapshots, samples = self._prepare_history(market)
            structure_samples = self._prepare_structure_samples(
                market, through=snapshots[-1].cutoff_date
            )
            manifest = build_feature_manifest(
                snapshots[0], feature_set_name=FEATURE_SET_EXTENDED
            )
            selected = tuple(backlog[: self.maximum_backlog_weeks])
            run_id = run_id or self._create_run(
                market, "incremental", selected[-1], profile_id
            )
            processed: list[dict[str, Any]] = []
            try:
                for anchor in selected:
                    processed.append(
                        self._process_week(
                            run_id,
                            market,
                            anchor,
                            snapshots,
                            samples,
                            manifest,
                            structure_samples=structure_samples,
                        )
                    )
            except BaseException as exc:
                self._mark_run_failed(run_id, exc)
                raise
            return self._complete_run(
                run_id,
                {
                    "status": "COMPLETED",
                    "market": market,
                    "processed_week_count": len(processed),
                    "processed_anchors": [value.isoformat() for value in selected],
                    "remaining_backlog_week_count": len(backlog) - len(selected),
                    "backlog_order": "OLDEST_FIRST",
                },
            )

    def start_bootstrap(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        with self._locks[market]:
            with self.sessions() as session:
                if self._optimizer_state(session, market) is not None:
                    raise V341RuntimeError(
                        "V3.4.1 bootstrap already exists; use incremental training"
                    )
                active = session.scalar(
                    select(V341TrainingRun)
                    .where(
                        V341TrainingRun.protocol_version == PROTOCOL_VERSION,
                        V341TrainingRun.model_market == market,
                        V341TrainingRun.run_type == "bootstrap",
                        V341TrainingRun.status.in_(("queued", "running")),
                    )
                    .order_by(V341TrainingRun.started_at.desc())
                    .limit(1)
                )
                if active is not None:
                    return {"run_id": active.id, "status": active.status, "market": market}
            anchors, snapshots, _ = self._prepare_history(market)
            manifest = build_feature_manifest(snapshots[0], feature_set_name=FEATURE_SET_EXTENDED)
            with self.sessions() as session, session.begin():
                profile = self._persist_profile(session, market, manifest)
            run_id = self._create_run(market, "bootstrap", anchors[-1], profile.id)
            self._executor.submit(self._run_task, run_id, market, "bootstrap")
            return {"run_id": run_id, "status": "queued", "market": market}

    def start_incremental(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        with self._locks[market]:
            with self.sessions() as session:
                optimizer = self._optimizer_state(session, market)
                if optimizer is None:
                    raise V341RuntimeError("V3.4.1 has not been bootstrapped")
                active = session.scalar(
                    select(V341TrainingRun)
                    .where(
                        V341TrainingRun.protocol_version == PROTOCOL_VERSION,
                        V341TrainingRun.model_market == market,
                        V341TrainingRun.run_type == "incremental",
                        V341TrainingRun.status.in_(("queued", "running")),
                    )
                    .order_by(V341TrainingRun.started_at.desc())
                    .limit(1)
                )
                if active is not None:
                    return {"run_id": active.id, "status": active.status, "market": market}
                profile_id = optimizer.profile_id
            run_id = self._create_run(market, "incremental", None, profile_id)
            self._executor.submit(self._run_task, run_id, market, "incremental")
            return {"run_id": run_id, "status": "queued", "market": market}

    def _run_task(self, run_id: str, market: str, run_type: str) -> None:
        try:
            self._mark_run_running(run_id)
            if run_type == "bootstrap":
                self.bootstrap_sync(market, run_id=run_id)
            else:
                self.incremental_sync(market, run_id=run_id)
        except BaseException as exc:
            self._mark_run_failed(run_id, exc)

    def incremental_due(self, market: str) -> bool:
        """Return whether a completed market week exists beyond the optimizer cursor."""

        self.validate_market(market)
        anchors = self._complete_anchors(market)
        with self.sessions() as session:
            optimizer = self._optimizer_state(session, market)
            return bool(optimizer is not None and anchors[-1] > optimizer.last_anchor_date)

    def _mark_run_running(self, run_id: str) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V341TrainingRun, run_id)
            if run is not None and run.status == "queued":
                run.status = "running"
                run.current_stage = "preflight"

    def _training_identity(self, session: Session, market: str) -> tuple[int, int, int, str | None]:
        optimizer = self._optimizer_state(session, market)
        return (
            int(
                session.scalar(
                    select(func.count()).select_from(V341TrainingIteration).where(
                        V341TrainingIteration.protocol_version == PROTOCOL_VERSION,
                        V341TrainingIteration.model_market == market,
                    )
                )
                or 0
            ),
            int(
                session.scalar(
                    select(func.count()).select_from(V341CandidateTrial).join(
                        V341TrainingRun,
                        V341TrainingRun.id == V341CandidateTrial.training_run_id,
                    ).where(
                        V341TrainingRun.protocol_version == PROTOCOL_VERSION,
                        V341TrainingRun.model_market == market,
                    )
                )
                or 0
            ),
            0 if optimizer is None else optimizer.weekly_iteration_count,
            None if optimizer is None else optimizer.state_hash,
        )

    def _create_analysis_run(self, market: str) -> str:
        self.validate_market(market)
        analysis_id = f"V341-ANALYSIS-{market}-{uuid4().hex[:12]}"
        with self.sessions() as session, session.begin():
            session.add(
                V341AnalysisRun(
                    id=analysis_id,
                    protocol_version=PROTOCOL_VERSION,
                    model_market=market,
                    forecast_id=None,
                    model_version_id=None,
                    status="queued",
                    forecast_anchor_date=None,
                    started_at=self.now_provider(),
                    completed_at=None,
                    result_json={},
                    error_code=None,
                    error_message=None,
                )
            )
        return analysis_id

    def start_analysis(self, market: str) -> dict[str, Any]:
        """Persist an analysis job before dispatch and return without blocking HTTP."""

        self.validate_market(market)
        with self.sessions() as session:
            active = session.scalar(
                select(V341AnalysisRun)
                .where(
                    V341AnalysisRun.protocol_version == PROTOCOL_VERSION,
                    V341AnalysisRun.model_market == market,
                    V341AnalysisRun.status.in_(("queued", "running")),
                )
                .order_by(V341AnalysisRun.started_at.desc())
                .limit(1)
            )
            if active is not None:
                return {"run_id": active.id, "status": active.status, "market": market}
        analysis_id = self._create_analysis_run(market)
        self._analysis_executor.submit(self._run_analysis_task, analysis_id, market)
        return {"run_id": analysis_id, "status": "queued", "market": market}

    def _run_analysis_task(self, analysis_id: str, market: str) -> None:
        try:
            with self.sessions() as session, session.begin():
                row = session.get(V341AnalysisRun, analysis_id)
                if row is None:
                    raise V341RuntimeError("V3.4.1 analysis run row disappeared")
                row.status = "running"
            with self._locks[market]:
                self._run_analysis_locked(market, analysis_id)
        except BaseException as exc:
            with self.sessions() as session, session.begin():
                row = session.get(V341AnalysisRun, analysis_id)
                if row is not None:
                    row.status = "failed"
                    row.completed_at = self.now_provider()
                    row.error_code = type(exc).__name__
                    row.error_message = str(exc)

    def run_analysis(self, market: str) -> dict[str, Any]:
        """Synchronous compatibility entry point used by deterministic tests."""

        analysis_id = self._create_analysis_run(market)
        with self._locks[market]:
            return self._run_analysis_locked(market, analysis_id)

    def _run_analysis_locked(self, market: str, analysis_id: str) -> dict[str, Any]:
        self.validate_market(market)
        with self.sessions() as session:
            before = self._training_identity(session, market)
        try:
            snapshots: tuple[Any, ...] = ()
            samples: tuple[V34TrainingSample, ...] = ()
            try:
                anchors = self._complete_anchors(market)
                anchor = anchors[-1]
            except V341RuntimeError:
                _history_anchors, snapshots, samples = self._prepare_history(market)
                anchor = snapshots[-1].cutoff_date
            with self.sessions() as session:
                existing = session.scalar(
                    select(V341ForecastRow).where(
                        V341ForecastRow.protocol_version == PROTOCOL_VERSION,
                        V341ForecastRow.model_market == market,
                        V341ForecastRow.forecast_anchor_date == anchor,
                    )
                )
            if existing is None:
                _history_anchors, snapshots, samples = self._prepare_history(market)
                if snapshots[-1].cutoff_date != anchor:
                    raise V341RuntimeError(
                        "analysis history does not end at the current anchor"
                    )
            with self.sessions() as session, session.begin():
                existing = session.scalar(
                    select(V341ForecastRow).where(
                        V341ForecastRow.protocol_version == PROTOCOL_VERSION,
                        V341ForecastRow.model_market == market,
                        V341ForecastRow.forecast_anchor_date == anchor,
                    )
                )
                loaded = self._load_state(session, market, anchor)
                if loaded is None:
                    raise V341RuntimeError(
                        "no effective V3.4.1 model; train the model first"
                    )
                state, model_row = loaded
                model_profile = session.get(V341TrainingProfile, model_row.profile_id)
                if model_profile is None:
                    raise V341RuntimeError(
                        "effective V3.4.1 model references a missing training profile"
                    )
                if existing is None:
                    snapshot = snapshots[-1]
                    manifest = build_feature_manifest(
                        snapshot,
                        feature_set_name=model_profile.feature_set_name,
                    )
                    if manifest.manifest_hash != model_profile.feature_manifest_hash:
                        raise V341RuntimeError(
                            "effective V3.4.1 model profile does not match the active feature manifest"
                        )
                    adapter = self._persist_adapter(session, market)
                    snapshot_row = self._persist_snapshot(session, snapshot, manifest)
                    complete_samples = (
                        self._prepare_structure_samples(market, through=anchor)
                        if model_profile.training_window_mode
                        == "EXPANDING_AVAILABLE_HISTORY"
                        else samples
                    )
                    matured = self._scenario_matured_rows(
                        model_profile, samples, complete_samples, anchor
                    )
                    standardized, sigmas, metadata = self._residual_pool(
                        session, market, anchor
                    )
                    path = self.training.predict(state, snapshot)
                    scenario = generate_scenario_forecast(
                        path,
                        snapshot,
                        matured,
                        standardized_residuals=standardized,
                        source_sigmas=sigmas,
                        residual_pool_metadata=metadata,
                        calibrators=self._active_calibrators(session, market, anchor),
                        scenario_count=self.scenario_count,
                    )
                    health_status = self._persist_health_snapshot(
                        session,
                        market,
                        anchor,
                        model_row,
                        path,
                        scenario.reliability,
                    )
                    scenario = self._scenario_with_health(scenario, health_status)
                    existing = self._persist_forecast(
                        session,
                        snapshot_row,
                        model_row,
                        adapter,
                        scenario,
                        label_end_date=self._future_label_end(
                            self.calendar, market, anchor
                        ),
                    )
                result = dict(existing.payload_json)
                result["forecast_hash"] = existing.forecast_hash
                analysis = session.get(V341AnalysisRun, analysis_id)
                if analysis is None:
                    raise V341RuntimeError("V3.4.1 analysis run row disappeared")
                analysis.forecast_id = existing.id
                analysis.model_version_id = model_row.id
                analysis.status = "completed"
                analysis.forecast_anchor_date = anchor
                analysis.completed_at = self.now_provider()
                analysis.result_json = result
                analysis.error_code = None
                analysis.error_message = None
            with self.sessions() as session:
                after = self._training_identity(session, market)
            if before != after:
                raise V341RuntimeError("data analysis mutated V3.4.1 training identity")
            return result
        except BaseException as exc:
            with self.sessions() as session, session.begin():
                row = session.get(V341AnalysisRun, analysis_id)
                if row is not None:
                    row.status = "failed"
                    row.completed_at = self.now_provider()
                    row.error_code = type(exc).__name__
                    row.error_message = str(exc)
            raise

    def analysis_status(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            row = session.get(V341AnalysisRun, run_id)
            if row is None:
                raise V341RuntimeError("unknown V3.4.1 analysis run")
            return {
                "run_id": row.id,
                "protocol_version": row.protocol_version,
                "market": row.model_market,
                "status": row.status,
                "forecast_anchor_date": (
                    None
                    if row.forecast_anchor_date is None
                    else row.forecast_anchor_date.isoformat()
                ),
                "started_at": row.started_at.isoformat(),
                "completed_at": (
                    None if row.completed_at is None else row.completed_at.isoformat()
                ),
                "result": dict(row.result_json),
                "error_code": row.error_code,
                "error_message": row.error_message,
            }

    def training_status(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            row = session.get(V341TrainingRun, run_id)
            if row is None:
                raise V341RuntimeError("unknown V3.4.1 training run")
            return {
                "run_id": row.id,
                "protocol_version": row.protocol_version,
                "market": row.model_market,
                "run_type": row.run_type,
                "status": row.status,
                "current_stage": row.current_stage,
                "horizon_weeks": row.horizon_weeks,
                "maximum_backlog_weeks": row.maximum_backlog_weeks,
                "started_at": row.started_at.isoformat(),
                "completed_at": (
                    None if row.completed_at is None else row.completed_at.isoformat()
                ),
                "result": dict(row.result_json),
                "error_code": row.error_code,
                "error_message": row.error_message,
            }

    def champion(self, market: str, *, anchor: date | None = None) -> dict[str, Any] | None:
        self.validate_market(market)
        target = anchor or self.now_provider().date()
        with self.sessions() as session:
            loaded = self._load_state(session, market, target)
            if loaded is None:
                return None
            state, row = loaded
            return {
                "market": market,
                "protocol_version": PROTOCOL_VERSION,
                "model_id": row.id,
                "version": state.version,
                "parent_version": state.parent_version,
                "status": row.status,
                "horizon_weeks": 13,
                "trained_through_date": state.trained_through_date.isoformat(),
                "effective_from_date": state.effective_from_date.isoformat(),
                "raw_matured_sample_count": state.raw_matured_sample_count,
                "effective_independent_sample_count": state.effective_independent_sample_count,
                "ridge_alpha": state.ridge_alpha,
                "parameter_hash": state.parameter_hash,
            }

    def model_statuses(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for market in ("399006", "159941"):
            with self.sessions() as session:
                optimizer = self._optimizer_state(session, market)
            output.append(
                {
                    "market": market,
                    "protocol_version": PROTOCOL_VERSION,
                    "version": "V3.4.1_13W",
                    "horizon_weeks": 13,
                    "bootstrapped": optimizer is not None,
                    "champion": self.champion(market),
                    "last_anchor_date": (
                        None if optimizer is None else optimizer.last_anchor_date.isoformat()
                    ),
                    "weekly_iteration_count": (
                        0 if optimizer is None else optimizer.weekly_iteration_count
                    ),
                    "candidate_training_count": (
                        0 if optimizer is None else optimizer.candidate_training_count
                    ),
                    "champion_promotion_count": (
                        0 if optimizer is None else optimizer.champion_promotion_count
                    ),
                }
            )
        return output

    def curve(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        with self.sessions() as session:
            rows = list(
                session.scalars(
                    select(V341TrainingIteration)
                    .where(
                        V341TrainingIteration.protocol_version == PROTOCOL_VERSION,
                        V341TrainingIteration.model_market == market,
                    )
                    .order_by(V341TrainingIteration.weekly_iteration_number)
                )
            )
        return {
            "market": market,
            "protocol_version": PROTOCOL_VERSION,
            "horizon_weeks": 13,
            "points": [
                {
                    "iteration": row.weekly_iteration_number,
                    "anchor_date": row.anchor_date.isoformat(),
                    "newly_matured_count": row.newly_matured_count,
                    "raw_matured_sample_count": row.raw_matured_sample_count,
                    "effective_independent_sample_count": row.effective_independent_sample_count,
                    "training_triggered": row.training_triggered,
                    "promoted": row.promoted,
                    "forecast_model_id": row.forecast_model_id,
                    "champion_after_model_id": row.champion_after_model_id,
                    "candidate": row.validation_json.get("candidate", {}),
                }
                for row in rows
            ],
        }

    def latest_forecast(self, market: str) -> dict[str, Any] | None:
        self.validate_market(market)
        with self.sessions() as session:
            row = session.scalar(
                select(V341ForecastRow)
                .where(
                    V341ForecastRow.protocol_version == PROTOCOL_VERSION,
                    V341ForecastRow.model_market == market,
                )
                .order_by(V341ForecastRow.forecast_anchor_date.desc(), V341ForecastRow.id.desc())
                .limit(1)
            )
            return None if row is None else dict(row.payload_json)
