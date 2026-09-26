"""V3.5 8-week weighted Ridge model core.

The math reuses the proven V3.4.1 primitives (weighted Ridge, OOD
diagnostics, conservative independent-sample counts) with a strictly 8-week
label horizon and its own frozen manifest/state payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.app.services.v341_training_service import (
    WeightedRidgeFit,
    candidate_alphas,
    conservative_independent_sample_count,
    weighted_ridge_fit,
)
from backend.app.services.v35_config import (
    HORIZON_WEEKS,
    LOSS_SCHEMA_VERSION,
    PROTOCOL_VERSION,
)
from backend.app.services.v35_feature_service import (
    V35FeatureSnapshot,
    V35FeatureError,
)


class V35TrainingError(RuntimeError):
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


@dataclass(frozen=True, slots=True)
class V35TrainingSample:
    market: str
    anchor_date: date
    snapshot: V35FeatureSnapshot
    cumulative_return_path: tuple[float, ...] | None
    label_end_date: date | None

    def __post_init__(self) -> None:
        if self.market != self.snapshot.market:
            raise V35TrainingError("sample market/snapshot mismatch")
        if self.snapshot.source_data_max_date > self.anchor_date:
            raise V35TrainingError("feature snapshot reads data after its anchor")
        if self.cumulative_return_path is None:
            if self.label_end_date is not None:
                raise V35TrainingError("immature sample cannot carry a label end date")
            return
        if len(self.cumulative_return_path) != HORIZON_WEEKS:
            raise V35TrainingError("mature V3.5 sample requires exactly 8 weekly returns")
        if self.label_end_date is None:
            raise V35TrainingError("mature V3.5 sample requires a label end date")


@dataclass(frozen=True, slots=True)
class V35ModelState:
    market: str
    version: str
    parent_version: str | None
    trained_through_date: date
    effective_from_date: date
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    ridge_alpha: float
    feature_quantile_low: tuple[float, ...]
    feature_quantile_high: tuple[float, ...]
    robust_feature_center: tuple[float, ...]
    robust_feature_scale: tuple[float, ...]
    ood_feature_indices: tuple[int, ...]
    ood_precision_matrix: tuple[tuple[float, ...], ...]
    ood_covariance_shrinkage: float
    raw_matured_sample_count: int
    effective_independent_sample_count: int
    feature_anchor_max_date: date
    label_observed_through_date: date
    validation_metrics: Mapping[str, Any]
    parameter_hash: str


@dataclass(frozen=True, slots=True)
class V35PathForecast:
    market: str
    anchor_date: date
    model_version: str
    expected_path: tuple[float, ...]
    source_feature_hash: str
    forecast_sigma: float
    ood_diagnostics: Mapping[str, Any]


def state_to_payload(state: V35ModelState) -> dict[str, Any]:
    return {
        "market": state.market,
        "version": state.version,
        "parent_version": state.parent_version,
        "trained_through_date": state.trained_through_date.isoformat(),
        "effective_from_date": state.effective_from_date.isoformat(),
        "feature_names": list(state.feature_names),
        "medians": list(state.medians),
        "means": list(state.means),
        "scales": list(state.scales),
        "intercept": list(state.intercept),
        "coefficients": [list(row) for row in state.coefficients],
        "ridge_alpha": state.ridge_alpha,
        "feature_quantile_low": list(state.feature_quantile_low),
        "feature_quantile_high": list(state.feature_quantile_high),
        "robust_feature_center": list(state.robust_feature_center),
        "robust_feature_scale": list(state.robust_feature_scale),
        "ood_feature_indices": list(state.ood_feature_indices),
        "ood_precision_matrix": [list(row) for row in state.ood_precision_matrix],
        "ood_covariance_shrinkage": state.ood_covariance_shrinkage,
        "raw_matured_sample_count": state.raw_matured_sample_count,
        "effective_independent_sample_count": state.effective_independent_sample_count,
        "feature_anchor_max_date": state.feature_anchor_max_date.isoformat(),
        "label_observed_through_date": state.label_observed_through_date.isoformat(),
        "validation_metrics": dict(state.validation_metrics),
        "parameter_hash": state.parameter_hash,
    }


def state_from_payload(payload: Mapping[str, Any]) -> V35ModelState:
    return V35ModelState(
        market=str(payload["market"]),
        version=str(payload["version"]),
        parent_version=(
            None if payload.get("parent_version") is None else str(payload["parent_version"])
        ),
        trained_through_date=date.fromisoformat(str(payload["trained_through_date"])),
        effective_from_date=date.fromisoformat(str(payload["effective_from_date"])),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        medians=tuple(float(value) for value in payload["medians"]),
        means=tuple(float(value) for value in payload["means"]),
        scales=tuple(float(value) for value in payload["scales"]),
        intercept=tuple(float(value) for value in payload["intercept"]),
        coefficients=tuple(
            tuple(float(value) for value in row) for row in payload["coefficients"]
        ),
        ridge_alpha=float(payload["ridge_alpha"]),
        feature_quantile_low=tuple(float(value) for value in payload["feature_quantile_low"]),
        feature_quantile_high=tuple(float(value) for value in payload["feature_quantile_high"]),
        robust_feature_center=tuple(float(value) for value in payload["robust_feature_center"]),
        robust_feature_scale=tuple(float(value) for value in payload["robust_feature_scale"]),
        ood_feature_indices=tuple(int(value) for value in payload["ood_feature_indices"]),
        ood_precision_matrix=tuple(
            tuple(float(value) for value in row) for row in payload["ood_precision_matrix"]
        ),
        ood_covariance_shrinkage=float(payload["ood_covariance_shrinkage"]),
        raw_matured_sample_count=int(payload["raw_matured_sample_count"]),
        effective_independent_sample_count=int(payload["effective_independent_sample_count"]),
        feature_anchor_max_date=date.fromisoformat(str(payload["feature_anchor_max_date"])),
        label_observed_through_date=date.fromisoformat(
            str(payload["label_observed_through_date"])
        ),
        validation_metrics=dict(payload.get("validation_metrics", {})),
        parameter_hash=str(payload["parameter_hash"]),
    )


def overlap_weights(samples: Sequence[V35TrainingSample]) -> np.ndarray:
    if not samples:
        return np.asarray([], dtype=float)
    weights: list[float] = []
    for sample in samples:
        if sample.label_end_date is None:
            raise V35TrainingError("overlap weights require fully matured labels")
        concurrent = 1
        for other in samples:
            if other is sample or other.label_end_date is None:
                continue
            if other.anchor_date < sample.label_end_date and sample.anchor_date < other.label_end_date:
                concurrent += 1
        weights.append(1.0 / concurrent)
    return np.asarray(weights, dtype=float)


def purged_walk_forward_folds(
    samples: Sequence[V35TrainingSample],
    *,
    folds: int = 5,
    minimum_training: int = 8,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    ordered = sorted(samples, key=lambda item: item.anchor_date)
    if len(ordered) < 12:
        return ()
    first_validation = max(minimum_training, len(ordered) // 3)
    candidates = np.array_split(np.arange(first_validation, len(ordered)), folds)
    result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for block in candidates:
        if block.size == 0:
            continue
        validation = tuple(int(value) for value in block)
        validation_start = ordered[validation[0]].anchor_date
        training = tuple(
            index
            for index, sample in enumerate(ordered[: validation[0]])
            if sample.label_end_date is not None and sample.label_end_date < validation_start
        )
        if len(training) >= minimum_training:
            result.append((training, validation))
    return tuple(result)


class V35TrainingService:
    """Weighted multi-horizon Ridge for the 8-week V3.5 path."""

    protocol_version: str = PROTOCOL_VERSION

    @staticmethod
    def _feature_row(snapshot: V35FeatureSnapshot, feature_names: Sequence[str]) -> list[float]:
        return [
            np.nan if snapshot.features.get(name) is None else float(snapshot.features[name])
            for name in feature_names
        ]

    def _matrix(
        self,
        samples: Sequence[V35TrainingSample],
        feature_names: Sequence[str],
    ) -> np.ndarray:
        return np.asarray(
            [self._feature_row(sample.snapshot, feature_names) for sample in samples],
            dtype=float,
        )

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
    ) -> V35ModelState:
        rows = tuple(sorted(matured, key=lambda item: item.anchor_date))
        if len(rows) < 8:
            raise V35TrainingError("V3.5 model fit requires at least eight mature samples")
        if any(
            row.market != market
            or row.cumulative_return_path is None
            or row.label_end_date is None
            or row.label_end_date > trained_through
            for row in rows
        ):
            raise V35TrainingError("model fit contains cross-market or immature/future labels")
        names = tuple(str(name) for name in feature_names)
        if not names or len(set(names)) != len(names):
            raise V35TrainingError("feature manifest must be non-empty and unique")
        x_raw = self._matrix(rows, names)
        y = np.asarray([row.cumulative_return_path for row in rows], dtype=float)
        if y.shape[1] != HORIZON_WEEKS:
            raise V35TrainingError("V3.5 labels must be 8-week cumulative return paths")
        weights = overlap_weights(rows)
        fit = weighted_ridge_fit(x_raw, y, weights, alpha=alpha)
        prediction = fit.intercept + fit.transformed_x @ fit.coefficients
        residual = y - prediction
        residual_increments = np.diff(
            np.column_stack((np.zeros(len(residual)), residual)), axis=1
        )
        residual_sigma = max(float(np.std(residual_increments)), 0.005)
        filled = np.where(np.isfinite(x_raw), x_raw, fit.medians)
        low = np.quantile(filled, 0.01, axis=0)
        high = np.quantile(filled, 0.99, axis=0)
        robust_center = np.median(filled, axis=0)
        q25 = np.quantile(filled, 0.25, axis=0)
        q75 = np.quantile(filled, 0.75, axis=0)
        robust_scale = np.maximum((q75 - q25) / 1.349, 1e-9)
        ood_indices = tuple(
            index
            for index, name in enumerate(names)
            if not name.startswith(("seq100_", "missing_"))
            and float(np.std(filled[:, index])) > 1e-12
        )[:32]
        if not ood_indices:
            ood_indices = tuple(range(min(1, len(names))))
        robust_matrix = (
            filled[:, np.asarray(ood_indices)]
            - robust_center[np.asarray(ood_indices)]
        ) / robust_scale[np.asarray(ood_indices)]
        covariance = np.atleast_2d(np.cov(robust_matrix, rowvar=False, ddof=1))
        shrinkage = 0.20
        regularized_covariance = (
            (1.0 - shrinkage) * covariance
            + shrinkage * np.eye(len(ood_indices), dtype=float)
        )
        precision = np.linalg.pinv(regularized_covariance, hermitian=True)
        independent_count = conservative_independent_sample_count(
            tuple((row.anchor_date, row.label_end_date) for row in rows if row.label_end_date)
        )
        metrics = dict(validation_metrics or {})
        metrics.setdefault("training_residual_sigma_diagnostic_only", residual_sigma)
        metrics.setdefault("training_residual_source", "IN_SAMPLE_DIAGNOSTIC_NOT_PRODUCTION")
        payload = {
            "protocol_version": self.protocol_version,
            "market": market,
            "version": version,
            "parent_version": parent_version,
            "trained_through": trained_through.isoformat(),
            "effective_from": effective_from.isoformat(),
            "feature_names": names,
            "medians": fit.medians.tolist(),
            "means": fit.means.tolist(),
            "scales": fit.scales.tolist(),
            "intercept": fit.intercept.tolist(),
            "coefficients": fit.coefficients.tolist(),
            "alpha": float(alpha),
            "feature_quantile_low": low.tolist(),
            "feature_quantile_high": high.tolist(),
            "robust_feature_center": robust_center.tolist(),
            "robust_feature_scale": robust_scale.tolist(),
            "ood_feature_indices": list(ood_indices),
            "ood_precision_matrix": precision.tolist(),
            "ood_covariance_shrinkage": shrinkage,
            "raw_matured_sample_count": len(rows),
            "effective_independent_sample_count": independent_count,
            "feature_anchor_max_date": rows[-1].anchor_date.isoformat(),
            "label_observed_through_date": max(
                row.label_end_date for row in rows if row.label_end_date
            ).isoformat(),
            "horizon_weeks": HORIZON_WEEKS,
        }
        parameter_hash = _hash(payload)
        return V35ModelState(
            market=market,
            version=version,
            parent_version=parent_version,
            trained_through_date=trained_through,
            effective_from_date=effective_from,
            feature_names=names,
            medians=tuple(float(value) for value in fit.medians),
            means=tuple(float(value) for value in fit.means),
            scales=tuple(float(value) for value in fit.scales),
            intercept=tuple(float(value) for value in fit.intercept),
            coefficients=tuple(
                tuple(float(value) for value in row) for row in fit.coefficients
            ),
            ridge_alpha=float(alpha),
            feature_quantile_low=tuple(float(value) for value in low),
            feature_quantile_high=tuple(float(value) for value in high),
            robust_feature_center=tuple(float(value) for value in robust_center),
            robust_feature_scale=tuple(float(value) for value in robust_scale),
            ood_feature_indices=ood_indices,
            ood_precision_matrix=tuple(
                tuple(float(value) for value in row) for row in precision
            ),
            ood_covariance_shrinkage=shrinkage,
            raw_matured_sample_count=len(rows),
            effective_independent_sample_count=independent_count,
            feature_anchor_max_date=rows[-1].anchor_date,
            label_observed_through_date=max(
                row.label_end_date for row in rows if row.label_end_date
            ),
            validation_metrics=metrics,
            parameter_hash=parameter_hash,
        )

    def predict(
        self,
        state: V35ModelState,
        snapshot: V35FeatureSnapshot,
    ) -> V35PathForecast:
        if snapshot.market != state.market:
            raise V35TrainingError("prediction cannot cross market chains")
        if snapshot.cutoff_date < state.effective_from_date:
            raise V35TrainingError("staged model is not effective at the forecast anchor")
        raw = np.asarray(
            [self._feature_row(snapshot, state.feature_names)], dtype=float
        )
        medians = np.asarray(state.medians)
        means = np.asarray(state.means)
        scales = np.asarray(state.scales)
        filled = np.where(np.isfinite(raw), raw, medians)
        transformed = np.clip((filled - means) / scales, -8.0, 8.0)[0]
        expected = np.asarray(state.intercept) + transformed @ np.asarray(
            state.coefficients
        )
        expected = np.clip(expected, -0.75, 1.50)
        lower = np.asarray(state.feature_quantile_low)
        upper = np.asarray(state.feature_quantile_high)
        outside_fraction = float(np.mean((filled[0] < lower) | (filled[0] > upper)))
        robust_z = np.abs(
            (filled[0] - np.asarray(state.robust_feature_center))
            / np.asarray(state.robust_feature_scale)
        )
        robust_z_max = float(np.max(np.clip(robust_z, 0.0, 100.0)))
        ood_indices = np.asarray(state.ood_feature_indices, dtype=int)
        ood_vector = np.clip(
            (
                filled[0, ood_indices]
                - np.asarray(state.robust_feature_center)[ood_indices]
            )
            / np.asarray(state.robust_feature_scale)[ood_indices],
            -12.0,
            12.0,
        )
        precision = np.asarray(state.ood_precision_matrix, dtype=float)
        multivariate_distance = float(
            np.sqrt(max(float(ood_vector @ precision @ ood_vector), 0.0) / len(ood_vector))
        )
        ood_score = float(
            np.clip(
                0.45 * min(1.0, outside_fraction / 0.20)
                + 0.30 * min(1.0, robust_z_max / 8.0)
                + 0.25 * min(1.0, multivariate_distance / 4.0),
                0.0,
                1.0,
            )
        )
        increments = np.diff(np.concatenate(([0.0], expected)))
        forecast_sigma = max(float(np.std(increments)), 0.01)
        source_hash = _hash(
            {
                "market": snapshot.market,
                "anchor": snapshot.cutoff_date.isoformat(),
                "feature_names": state.feature_names,
                "values": filled[0].tolist(),
            }
        )
        return V35PathForecast(
            market=state.market,
            anchor_date=snapshot.cutoff_date,
            model_version=state.version,
            expected_path=tuple(float(value) for value in expected),
            source_feature_hash=source_hash,
            forecast_sigma=forecast_sigma,
            ood_diagnostics={
                "robust_z_max": robust_z_max,
                "outside_training_quantile_fraction": outside_fraction,
                "regularized_multivariate_distance": multivariate_distance,
                "multivariate_method": "ROBUST_SHRUNK_COVARIANCE_MAHALANOBIS",
                "multivariate_feature_count": len(state.ood_feature_indices),
                "covariance_shrinkage": state.ood_covariance_shrinkage,
                "ood_score": ood_score,
            },
        )

    def fold_path_losses(
        self,
        market: str,
        rows: Sequence[V35TrainingSample],
        training_indices: Sequence[int],
        validation_indices: Sequence[int],
        *,
        alpha: float,
        feature_names: Sequence[str],
    ) -> np.ndarray:
        training = tuple(rows[index] for index in training_indices)
        validation = tuple(rows[index] for index in validation_indices)
        state = self.fit(
            market,
            training,
            version="FOLD",
            parent_version=None,
            trained_through=max(row.label_end_date for row in training if row.label_end_date),
            effective_from=validation[0].anchor_date,
            alpha=alpha,
            feature_names=feature_names,
        )
        predicted = np.asarray(
            [
                self.predict(state, row.snapshot).expected_path
                for row in validation
            ],
            dtype=float,
        )
        actual = np.asarray([row.cumulative_return_path for row in validation])
        train_y = np.asarray([row.cumulative_return_path for row in training])
        q25 = np.quantile(train_y, 0.25, axis=0)
        q75 = np.quantile(train_y, 0.75, axis=0)
        scales = np.maximum((q75 - q25) / 1.349, 0.01)
        return np.mean(np.abs(predicted - actual) / scales, axis=1)

    def validation_metrics_for(
        self,
        market: str,
        rows: Sequence[V35TrainingSample],
        *,
        alpha: float,
        feature_names: Sequence[str],
    ) -> dict[str, Any]:
        folds = purged_walk_forward_folds(rows, minimum_training=8)
        if not folds:
            return {
                "schema_version": LOSS_SCHEMA_VERSION,
                "fold_count": 0,
                "note": "insufficient samples for purged walk-forward validation",
            }
        fold_losses: list[float] = []
        for training_indices, validation_indices in folds:
            fold_losses.extend(
                self.fold_path_losses(
                    market,
                    rows,
                    training_indices,
                    validation_indices,
                    alpha=alpha,
                    feature_names=feature_names,
                ).tolist()
            )
        return {
            "schema_version": LOSS_SCHEMA_VERSION,
            "fold_count": len(folds),
            "mean_oos_path_mae": float(np.mean(fold_losses)) if fold_losses else None,
            "median_oos_path_mae": float(np.median(fold_losses)) if fold_losses else None,
            "oos_sample_count": len(fold_losses),
        }
