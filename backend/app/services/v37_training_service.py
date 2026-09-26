"""V3.7 multi-timeframe training: EARLY_FUSION and LATE_FUSION."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from backend.app.services.v35_feature_service import V35FeatureSnapshot
from backend.app.services.v35_training_service import (
    V35ModelState,
    V35PathForecast,
    V35TrainingError,
    V35TrainingSample,
    V35TrainingService,
    state_from_payload,
    state_to_payload,
)
from backend.app.services.v37_config import (
    FEATURE_SET_LATE_FUSION,
    LATE_FUSION_DAILY_WEIGHTS,
    LATE_FUSION_WEEKLY_WEIGHTS,
    PROTOCOL_VERSION_37,
)


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


@dataclass(frozen=True, slots=True)
class V37LateFusionModelState:
    market: str
    version: str
    parent_version: str | None
    trained_through_date: date
    effective_from_date: date
    weekly_state: V35ModelState
    daily_state: V35ModelState
    daily_weights: tuple[float, float, float]
    weekly_weights: tuple[float, float, float]
    raw_matured_sample_count: int
    effective_independent_sample_count: int
    feature_anchor_max_date: date
    label_observed_through_date: date
    validation_metrics: Mapping[str, Any]
    parameter_hash: str

    @property
    def ridge_alpha(self) -> float:
        return self.weekly_state.ridge_alpha


def late_fusion_state_to_payload(state: V37LateFusionModelState) -> dict[str, Any]:
    return {
        "model_family": "LATE_FUSION",
        "protocol_version": PROTOCOL_VERSION_37,
        "market": state.market,
        "version": state.version,
        "parent_version": state.parent_version,
        "trained_through_date": state.trained_through_date.isoformat(),
        "effective_from_date": state.effective_from_date.isoformat(),
        "weekly_state": state_to_payload(state.weekly_state),
        "daily_state": state_to_payload(state.daily_state),
        "daily_weights": list(state.daily_weights),
        "weekly_weights": list(state.weekly_weights),
        "raw_matured_sample_count": state.raw_matured_sample_count,
        "effective_independent_sample_count": state.effective_independent_sample_count,
        "feature_anchor_max_date": state.feature_anchor_max_date.isoformat(),
        "label_observed_through_date": state.label_observed_through_date.isoformat(),
        "validation_metrics": dict(state.validation_metrics),
        "parameter_hash": state.parameter_hash,
    }


def late_fusion_state_from_payload(payload: Mapping[str, Any]) -> V37LateFusionModelState:
    return V37LateFusionModelState(
        market=str(payload["market"]),
        version=str(payload["version"]),
        parent_version=(
            None
            if payload.get("parent_version") is None
            else str(payload["parent_version"])
        ),
        trained_through_date=date.fromisoformat(str(payload["trained_through_date"])),
        effective_from_date=date.fromisoformat(str(payload["effective_from_date"])),
        weekly_state=state_from_payload(payload["weekly_state"]),
        daily_state=state_from_payload(payload["daily_state"]),
        daily_weights=tuple(float(value) for value in payload["daily_weights"]),
        weekly_weights=tuple(float(value) for value in payload["weekly_weights"]),
        raw_matured_sample_count=int(payload["raw_matured_sample_count"]),
        effective_independent_sample_count=int(
            payload["effective_independent_sample_count"]
        ),
        feature_anchor_max_date=date.fromisoformat(
            str(payload["feature_anchor_max_date"])
        ),
        label_observed_through_date=date.fromisoformat(
            str(payload["label_observed_through_date"])
        ),
        validation_metrics=dict(payload.get("validation_metrics", {})),
        parameter_hash=str(payload["parameter_hash"]),
    )


class V37TrainingService(V35TrainingService):
    """V3.7 training service with EARLY_FUSION and LATE_FUSION support."""

    protocol_version: str = PROTOCOL_VERSION_37

    def fit(
        self,
        market: str,
        matured: Sequence[V35TrainingSample],
        *,
        version: str,
        parent_version: str | None,
        trained_through: date,
        effective_from: date,
        alpha: float,
        feature_names: Sequence[str],
        validation_metrics: Mapping[str, Any] | None = None,
        model_family: str = "EARLY_FUSION",
        weekly_feature_names: Sequence[str] | None = None,
        daily_feature_names: Sequence[str] | None = None,
        daily_weights: Sequence[float] = LATE_FUSION_DAILY_WEIGHTS,
        weekly_weights: Sequence[float] = LATE_FUSION_WEEKLY_WEIGHTS,
    ) -> V35ModelState | V37LateFusionModelState:
        if model_family == "LATE_FUSION":
            return self.fit_late_fusion(
                market,
                matured,
                version=version,
                parent_version=parent_version,
                trained_through=trained_through,
                effective_from=effective_from,
                alpha=alpha,
                weekly_feature_names=weekly_feature_names or feature_names,
                daily_feature_names=daily_feature_names or feature_names,
                daily_weights=daily_weights,
                weekly_weights=weekly_weights,
                validation_metrics=validation_metrics,
            )
        return super().fit(
            market,
            matured,
            version=version,
            parent_version=parent_version,
            trained_through=trained_through,
            effective_from=effective_from,
            alpha=alpha,
            feature_names=feature_names,
            validation_metrics=validation_metrics,
        )

    def fit_late_fusion(
        self,
        market: str,
        matured: Sequence[V35TrainingSample],
        *,
        version: str,
        parent_version: str | None,
        trained_through: date,
        effective_from: date,
        alpha: float,
        weekly_feature_names: Sequence[str],
        daily_feature_names: Sequence[str],
        daily_weights: Sequence[float],
        weekly_weights: Sequence[float],
        validation_metrics: Mapping[str, Any] | None,
    ) -> V37LateFusionModelState:
        rows = tuple(sorted(matured, key=lambda item: item.anchor_date))
        if len(rows) < 8:
            raise V35TrainingError("V3.7 LATE_FUSION requires at least eight samples")
        weekly_validation = self.validation_metrics_for(
            market,
            rows,
            alpha=alpha,
            feature_names=weekly_feature_names,
        )
        daily_validation = self.validation_metrics_for(
            market,
            rows,
            alpha=alpha,
            feature_names=daily_feature_names,
        )
        weekly_state = super().fit(
            market,
            rows,
            version=f"{version}:WEEKLY",
            parent_version=parent_version,
            trained_through=trained_through,
            effective_from=effective_from,
            alpha=alpha,
            feature_names=weekly_feature_names,
            validation_metrics=weekly_validation,
        )
        daily_state = super().fit(
            market,
            rows,
            version=f"{version}:DAILY",
            parent_version=parent_version,
            trained_through=trained_through,
            effective_from=effective_from,
            alpha=alpha,
            feature_names=daily_feature_names,
            validation_metrics=daily_validation,
        )
        independent_count = max(
            weekly_state.effective_independent_sample_count,
            daily_state.effective_independent_sample_count,
        )
        payload = {
            "model_family": "LATE_FUSION",
            "protocol_version": PROTOCOL_VERSION_37,
            "market": market,
            "version": version,
            "parent_version": parent_version,
            "trained_through": trained_through.isoformat(),
            "effective_from": effective_from.isoformat(),
            "weekly_state": state_to_payload(weekly_state),
            "daily_state": state_to_payload(daily_state),
            "daily_weights": list(daily_weights),
            "weekly_weights": list(weekly_weights),
            "raw_matured_sample_count": len(rows),
            "effective_independent_sample_count": independent_count,
            "feature_anchor_max_date": rows[-1].anchor_date.isoformat(),
            "label_observed_through_date": max(
                row.label_end_date for row in rows if row.label_end_date
            ).isoformat(),
            "validation_metrics": dict(validation_metrics or {}),
        }
        return V37LateFusionModelState(
            market=market,
            version=version,
            parent_version=parent_version,
            trained_through_date=trained_through,
            effective_from_date=effective_from,
            weekly_state=weekly_state,
            daily_state=daily_state,
            daily_weights=tuple(float(value) for value in daily_weights),
            weekly_weights=tuple(float(value) for value in weekly_weights),
            raw_matured_sample_count=len(rows),
            effective_independent_sample_count=independent_count,
            feature_anchor_max_date=rows[-1].anchor_date,
            label_observed_through_date=max(
                row.label_end_date for row in rows if row.label_end_date
            ),
            validation_metrics=validation_metrics or {},
            parameter_hash=_hash(payload),
        )

    def predict(
        self,
        state: V35ModelState | V37LateFusionModelState,
        snapshot: V35FeatureSnapshot,
    ) -> V35PathForecast:
        if isinstance(state, V37LateFusionModelState):
            weekly = super().predict(state.weekly_state, snapshot)
            daily = super().predict(state.daily_state, snapshot)
            expected = [
                state.weekly_weights[0] * weekly.expected_path[0]
                + state.daily_weights[0] * daily.expected_path[0],
                state.weekly_weights[0] * weekly.expected_path[1]
                + state.daily_weights[0] * daily.expected_path[1],
                state.weekly_weights[1] * weekly.expected_path[2]
                + state.daily_weights[1] * daily.expected_path[2],
                state.weekly_weights[1] * weekly.expected_path[3]
                + state.daily_weights[1] * daily.expected_path[3],
                state.weekly_weights[2] * weekly.expected_path[4]
                + state.daily_weights[2] * daily.expected_path[4],
                state.weekly_weights[2] * weekly.expected_path[5]
                + state.daily_weights[2] * daily.expected_path[5],
                state.weekly_weights[2] * weekly.expected_path[6]
                + state.daily_weights[2] * daily.expected_path[6],
                state.weekly_weights[2] * weekly.expected_path[7]
                + state.daily_weights[2] * daily.expected_path[7],
            ]
            expected = np.clip(np.asarray(expected, dtype=float), -0.75, 1.50)
            increments = np.diff(np.concatenate(([0.0], expected)))
            return V35PathForecast(
                market=state.market,
                anchor_date=snapshot.cutoff_date,
                model_version=state.version,
                expected_path=tuple(float(value) for value in expected),
                source_feature_hash=_hash(
                    {
                        "weekly": weekly.source_feature_hash,
                        "daily": daily.source_feature_hash,
                    }
                ),
                forecast_sigma=max(float(np.std(increments)), 0.01),
                ood_diagnostics={
                    "weekly_ood": weekly.ood_diagnostics,
                    "daily_ood": daily.ood_diagnostics,
                    "model_family": "LATE_FUSION",
                    "ood_score": max(
                        float(weekly.ood_diagnostics.get("ood_score", 0.0)),
                        float(daily.ood_diagnostics.get("ood_score", 0.0)),
                    ),
                },
            )
        return super().predict(state, snapshot)
