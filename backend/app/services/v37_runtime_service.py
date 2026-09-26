"""V3.7 runtime: multi-timeframe local model + DeepSeek AI integration.

The V3.7 runtime reuses the V3.5/V3.5.1/V3.6 execution engine (idempotent
weekly transactions, real share/T+1 accounting, pending batches, promotion
gates) under its own ``protocol_version``.  The local feature service is
replaced by the multi-timeframe builder; the training service supports
EARLY_FUSION and LATE_FUSION; the AI layer is an independent challenger that
never modifies local forecasts.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.models import (
    V35FeatureSnapshot as V35FeatureSnapshotRow,
    V35ModelPackage,
    V35ModelVersion,
    V37MultiTimeframeFeature,
)
from backend.app.services.v35_runtime_service import V35RuntimeService
from backend.app.services.v35_training_service import (
    V35ModelState,
    V35TrainingSample,
    state_from_payload,
)
from backend.app.services.v36_config import (
    PROTOCOL_VERSION_36,
)
from backend.app.services.v37_config import (
    FEATURE_SET_EARLY_FUSION,
    FEATURE_SET_LATE_FUSION,
    FEATURE_SET_WEEKLY_ONLY,
    FEATURE_VERSION,
    PROMOTION_RULE_VERSION,
    PROTOCOL_VERSION_37,
)
from backend.app.services.v37_feature_service import (
    V37FeatureService,
)
from backend.app.services.v37_training_service import (
    V37LateFusionModelState,
    V37TrainingService,
    late_fusion_state_from_payload,
    late_fusion_state_to_payload,
)


class V37RuntimeService(V35RuntimeService):
    """V3.7 multi-timeframe + DeepSeek-AI runtime."""

    feature_set_name: str = FEATURE_SET_EARLY_FUSION
    feature_version: str = FEATURE_VERSION
    promotion_rule_version: str = PROMOTION_RULE_VERSION

    def __init__(self, factory: sessionmaker) -> None:
        super().__init__(
            factory,
            protocol_version=PROTOCOL_VERSION_37,
            v351=True,
        )
        # V3.7 keeps V3.6 execution semantics (real shares/T+1, snapshots,
        # alpha set, cadence, activity health) under the V3.7 namespace.
        self.v36 = True
        self._features = V37FeatureService()
        self._training = V37TrainingService()

    def _persist_feature(
        self,
        session: Session,
        market: str,
        anchor: date,
        snapshot: Any,
        manifest: Any,
    ) -> V35FeatureSnapshotRow:
        row = super()._persist_feature(session, market, anchor, snapshot, manifest)
        state = str(snapshot.provenance.get("multi_timeframe_state") or "NEUTRAL_MIXED")
        weekly = {
            key: value
            for key, value in snapshot.features.items()
            if key.startswith("v37_w_") or key.startswith("v35_")
        }
        daily = {
            key: value
            for key, value in snapshot.features.items()
            if key.startswith("v37_d_")
        }
        interactions = {
            key: value
            for key, value in snapshot.features.items()
            if key.startswith("v37_x_")
        }
        feature_hash = self._features.audit_payload(snapshot, manifest)["snapshot_hash"]
        existing = session.scalar(
            select(V37MultiTimeframeFeature).where(
                V37MultiTimeframeFeature.protocol_version == self.protocol_version,
                V37MultiTimeframeFeature.model_market == market,
                V37MultiTimeframeFeature.forecast_anchor_date == anchor,
                V37MultiTimeframeFeature.manifest_version == manifest.version,
            )
        )
        if existing is None:
            from backend.app.models.models import utc_now

            session.add(
                V37MultiTimeframeFeature(
                    protocol_version=self.protocol_version,
                    model_market=market,
                    forecast_anchor_date=anchor,
                    source_max_date=snapshot.source_data_max_date,
                    manifest_version=manifest.version,
                    multi_timeframe_state=state,
                    weekly_branch_json=weekly,
                    daily_branch_json=daily,
                    interaction_json=interactions,
                    feature_hash=feature_hash,
                    created_at=utc_now(),
                )
            )
            session.flush()
        return row

    def _persist_model(
        self,
        session: Session,
        market: str,
        state: V35ModelState | V37LateFusionModelState,
        *,
        status: str,
        metrics: Mapping[str, Any],
        promotion_gate: Mapping[str, Any],
    ) -> V35ModelVersion:
        if isinstance(state, V37LateFusionModelState):
            row = session.get(V35ModelVersion, state.version)
            if row is not None:
                return row
            from backend.app.models.models import utc_now
            from backend.app.services.v35_runtime_service import _pct

            session.add(
                V35ModelVersion(
                    id=state.version,
                    protocol_version=self.protocol_version,
                    model_market=market,
                    version=state.version,
                    parent_model_id=state.parent_version,
                    profile_id=f"{self.protocol_version}:{market}:CORE",
                    status=status,
                    trained_through_date=state.trained_through_date,
                    effective_from_date=state.effective_from_date,
                    feature_anchor_max_date=state.feature_anchor_max_date,
                    label_observed_through_date=state.label_observed_through_date,
                    raw_matured_sample_count=state.raw_matured_sample_count,
                    effective_independent_sample_count=(
                        state.effective_independent_sample_count
                    ),
                    horizon_weeks=8,
                    ridge_alpha=_pct(state.ridge_alpha),
                    parameters_json=late_fusion_state_to_payload(state),
                    metrics_json=dict(metrics),
                    promotion_gate_json=dict(promotion_gate),
                    health_status=status,
                    parameter_hash=state.parameter_hash,
                    created_at=utc_now(),
                )
            )
            session.flush()
            return session.get(V35ModelVersion, state.version)
        return super()._persist_model(
            session,
            market,
            state,
            status=status,
            metrics=metrics,
            promotion_gate=promotion_gate,
        )

    def _load_model_state(
        self,
        session: Session,
        package: V35ModelPackage,
    ) -> V35ModelState | V37LateFusionModelState:
        model = session.get(V35ModelVersion, package.prediction_model_id)
        if model is None:
            from backend.app.services.v35_runtime_service import V35RuntimeError

            raise V35RuntimeError(f"missing model {package.prediction_model_id}")
        payload = dict(model.parameters_json or {})
        if payload.get("model_family") == "LATE_FUSION":
            return late_fusion_state_from_payload(payload)
        return state_from_payload(payload)

    def _v351_prediction_candidates(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
        round_index: int,
    ) -> tuple[V35ModelPackage, ...]:
        from backend.app.services.v35_runtime_service import (
            MAX_PREDICTION_CANDIDATES,
            STRUCTURAL_CHALLENGE_EVERY,
        )
        from backend.app.services.v36_config import (
            CALIBRATION_MODE_VALUES_36,
            MODEL_FAMILY_VALUES_36,
            REGIME_CORRECTION_VALUES_36,
            RESIDUAL_WEIGHTING_VALUES_36,
        )
        from backend.app.services.v351_config import TRAINING_WINDOW_VALUES

        champion_model = session.get(V35ModelVersion, champion.prediction_model_id)
        champion_config = self._v351_model_config(champion)
        alphas: list[float] = []
        alpha_min, alpha_max = self._cfg_alpha_bounds()
        import numpy as np

        for multiplier in self._cfg_alpha_multipliers():
            value = round(
                float(
                    np.clip(
                        float(champion_model.ridge_alpha) * multiplier,
                        alpha_min,
                        alpha_max,
                    )
                ),
                12,
            )
            if value not in alphas:
                alphas.append(value)
        matured = self._matured_samples(session, market, anchor)
        snapshot_rows = self._feature_snapshots(session, market, through=anchor)
        snapshot = replace(
            self._empty_snapshot(market, anchor),
            features=dict(snapshot_rows[-1].feature_json) if snapshot_rows else {},
        )
        base_manifest = self._features.build_manifest(snapshot)
        candidates: list[V35ModelPackage] = []
        attempts: list[dict[str, Any]] = []
        for index, alpha in enumerate(alphas[:MAX_PREDICTION_CANDIDATES]):
            version = f"{self.protocol_version}:{market}:PREDC351:{anchor.isoformat()}:{index}"
            model_config = {
                **champion_config,
                "alpha": alpha,
                "architecture": FEATURE_SET_EARLY_FUSION,
                "training_window_mode": champion_config.get(
                    "training_window_mode", "ROLLING_520W"
                ),
                "feature_set_name": FEATURE_SET_EARLY_FUSION,
                "daily_adjustment_mode": champion_config.get(
                    "daily_adjustment_mode", "DAILY_ADJUSTMENT_OFF"
                ),
                "residual_scale": champion_config.get("residual_scale", 1.0),
            }
            candidate = self._fit_v351_candidate(
                session,
                market,
                anchor,
                champion,
                version,
                model_config,
                matured,
                base_manifest,
                family="PREDICTION",
            )
            if candidate is None:
                continue
            attempts.append({"alpha": alpha, "generated": True})
            candidates.append(candidate)
        if round_index % STRUCTURAL_CHALLENGE_EVERY == 0 and len(candidates) < MAX_PREDICTION_CANDIDATES:
            dimensions = (
                "training_window_mode",
                "architecture",
                "daily_adjustment_mode",
                "residual_scale",
                "residual_weighting",
                "calibration_mode",
                "model_family",
                "regime_correction",
            )
            dimension = dimensions[
                (round_index // STRUCTURAL_CHALLENGE_EVERY) % len(dimensions)
            ]
            model_config = dict(champion_config)
            if dimension == "training_window_mode":
                model_config["training_window_mode"] = next(
                    value
                    for value in TRAINING_WINDOW_VALUES
                    if value != champion_config.get("training_window_mode", "ROLLING_520W")
                )
            elif dimension == "architecture":
                current = champion_config.get("architecture", FEATURE_SET_EARLY_FUSION)
                model_config["architecture"] = next(
                    value
                    for value in (FEATURE_SET_WEEKLY_ONLY, FEATURE_SET_LATE_FUSION)
                    if value != current
                )
            elif dimension == "daily_adjustment_mode":
                from backend.app.services.v351_config import DAILY_ADJUSTMENT_VALUES

                model_config["daily_adjustment_mode"] = next(
                    value
                    for value in DAILY_ADJUSTMENT_VALUES
                    if value
                    != champion_config.get("daily_adjustment_mode", "DAILY_ADJUSTMENT_OFF")
                )
            elif dimension == "residual_scale":
                model_config["residual_scale"] = next(
                    value
                    for value in self._cfg_residual_scale_values()
                    if value != champion_config.get("residual_scale", 1.0)
                )
            elif dimension == "residual_weighting":
                model_config["residual_weighting"] = next(
                    value
                    for value in RESIDUAL_WEIGHTING_VALUES_36
                    if value
                    != champion_config.get("residual_weighting", "UNIFORM")
                )
            elif dimension == "calibration_mode":
                model_config["calibration_mode"] = next(
                    value
                    for value in CALIBRATION_MODE_VALUES_36
                    if value
                    != champion_config.get("calibration_mode", "TEMPERATURE")
                )
            elif dimension == "model_family":
                model_config["model_family"] = next(
                    value
                    for value in MODEL_FAMILY_VALUES_36
                    if value != champion_config.get("model_family", "RIDGE")
                )
            elif dimension == "regime_correction":
                model_config["regime_correction"] = next(
                    value
                    for value in REGIME_CORRECTION_VALUES_36
                    if value
                    != champion_config.get("regime_correction", "OFF")
                )
            version = f"{self.protocol_version}:{market}:STRUCTC351:{anchor.isoformat()}"
            candidate = self._fit_v351_candidate(
                session,
                market,
                anchor,
                champion,
                version,
                model_config,
                matured,
                base_manifest,
                family="PREDICTION_STRUCTURAL",
            )
            if candidate is not None:
                attempts.append(
                    {"structural_dimension": dimension, "generated": True}
                )
                candidates.append(candidate)
        self._pending_v351_attempts = attempts
        return tuple(candidates)

    def _fit_v351_candidate(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
        version: str,
        model_config: Mapping[str, Any],
        matured: Sequence[V35TrainingSample],
        base_manifest: Any,
        *,
        family: str,
    ) -> V35ModelPackage | None:
        from backend.app.services.v35_runtime_service import V35RuntimeError

        existing_model = session.get(V35ModelVersion, version)
        if existing_model is not None:
            payload = dict(existing_model.parameters_json or {})
            state = (
                late_fusion_state_from_payload(payload)
                if payload.get("model_family") == "LATE_FUSION"
                else state_from_payload(payload)
            )
        else:
            training_window = str(
                model_config.get("training_window_mode", "ROLLING_520W")
            )
            limit = 520 if training_window == "ROLLING_520W" else 260
            training_rows = tuple(matured[-limit:] if limit else matured)
            if len(training_rows) < 8:
                return None
            architecture = str(
                model_config.get("architecture", FEATURE_SET_EARLY_FUSION)
            )
            feature_names = tuple(base_manifest.ordered_feature_names)
            if architecture == FEATURE_SET_WEEKLY_ONLY:
                feature_names = tuple(
                    name
                    for name in feature_names
                    if not name.startswith("v37_d_") and not name.startswith("v37_x_")
                )
            weekly_names = tuple(
                name
                for name in feature_names
                if not name.startswith("v37_d_") and not name.startswith("v37_x_")
            )
            daily_names = tuple(
                name
                for name in feature_names
                if name.startswith("v37_d_") or name.startswith("v37_x_")
            )
            validation = self._training.validation_metrics_for(
                market,
                training_rows,
                alpha=float(model_config.get("alpha", 1.0)),
                feature_names=feature_names,
            )
            next_anchor = self._anchor_after(session, market, anchor, steps=1)
            if architecture == FEATURE_SET_LATE_FUSION:
                state = self._training.fit(
                    market,
                    training_rows,
                    version=version,
                    parent_version=champion.prediction_model_id,
                    trained_through=anchor,
                    effective_from=next_anchor or anchor,
                    alpha=float(model_config.get("alpha", 1.0)),
                    feature_names=feature_names,
                    validation_metrics=validation,
                    model_family="LATE_FUSION",
                    weekly_feature_names=weekly_names,
                    daily_feature_names=daily_names,
                )
            else:
                state = self._training.fit(
                    market,
                    training_rows,
                    version=version,
                    parent_version=champion.prediction_model_id,
                    trained_through=anchor,
                    effective_from=next_anchor or anchor,
                    alpha=float(model_config.get("alpha", 1.0)),
                    feature_names=feature_names,
                    validation_metrics=validation,
                )
            self._persist_model(
                session,
                market,
                state,
                status="CANDIDATE",
                metrics=validation,
                promotion_gate={"family": family, "model_config": dict(model_config)},
            )
        package_version = f"{version}:PKG"
        strategy_config = {
            **dict(champion.strategy_config_json),
            "v351_model_config": dict(model_config),
        }
        return self._persist_package(
            session,
            market,
            version=package_version,
            kind=(
                "PREDICTION_CHALLENGER"
                if family == "PREDICTION"
                else "STRUCTURAL_CHALLENGER"
            ),
            prediction_model_id=state.version,
            strategy_config=strategy_config,
            effective_from=anchor,
            parent_package_id=champion.id,
        )

    def incremental_sync(
        self,
        market: str,
        *,
        run_id: str | None = None,
        ai_config: Any | None = None,
        generation: int = 1,
    ) -> dict[str, Any]:
        """Weekly incremental: local week first, then AI + fusion + review.

        The local weekly transaction is committed first (base bootstrap_sync).
        AI analysis runs in a separate transaction so an AI failure never
        rolls back or blocks the local model; failures are recorded as
        ``AI_FALLBACK_LOCAL_ONLY``.  No new complete week -> processed_weeks=0
        and no AI call for a stale anchor.
        """

        result = super().incremental_sync(market, run_id=run_id)
        processed = int(result.get("processed_weeks", 0))
        last_anchor = result.get("last_anchor")
        ai_status: dict[str, Any] = {}
        fusion_status: dict[str, Any] = {}
        if not processed or not last_anchor:
            result["ai"] = {"status": "NO_NEW_WEEK", "fallback": "AI_FALLBACK_LOCAL_ONLY"}
            result["fusion"] = fusion_status
            return result
        anchor = date.fromisoformat(str(last_anchor))
        from backend.app.services.deepseek_client import load_config
        from backend.app.services.v37_ai_service import (
            ai_model_health,
            ensure_ai_forecast,
        )
        from backend.app.services.v37_config import AI_SCOPE_FORWARD_OOS
        from backend.app.services.v37_fusion_service import run_fusion_window_accounts
        from backend.app.services.v37_repair_service import review_due

        config = ai_config if ai_config is not None and getattr(ai_config, "api_key", "") else load_config()
        with self._factory() as session, session.begin():
            ai_status = ensure_ai_forecast(
                session,
                market,
                anchor,
                config,
                screening_scope=AI_SCOPE_FORWARD_OOS,
                generation=generation,
            )
            ai_model_health(
                session,
                market,
                anchor,
                generation=generation,
                model_name=config.model,
            )
            champion = self._champion_package(session, market, anchor)
            if champion is not None:
                fusion_status["windows_run"] = run_fusion_window_accounts(
                    session, market, anchor, champion
                )
            due, trigger = review_due(session, market, anchor)
            ai_status["review_due"] = due
            ai_status["review_trigger"] = trigger
            if ai_status.get("status") not in ("OK", "CACHED"):
                ai_status["fallback"] = "AI_FALLBACK_LOCAL_ONLY"
        result["ai"] = ai_status
        result["fusion"] = fusion_status
        return result
