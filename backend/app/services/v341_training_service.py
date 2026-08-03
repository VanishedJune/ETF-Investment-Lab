"""Leakage-safe statistical primitives for the append-only V3.4.1 model core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.app.services.v33_feature_service import FeatureSnapshot, flatten_snapshot
from backend.app.services.v34_feature_service import assert_159941_self_only
from backend.app.services.v34_training_service import (
    HORIZON_WEEKS,
    V34TrainingSample,
    overlap_weights,
    purged_walk_forward_folds,
)


PROTOCOL_VERSION = "V3.4.1_MODEL_CORE"
LOSS_SCHEMA_VERSION = "LOSS_SCHEMA_V34_1"
CANDIDATE_RANDOM_PLAN_VERSION = "V3.4.4_LOG_GROSS_CANDIDATE_LOSS_PLAN_4"
ALPHA_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)
LOSS_WEIGHTS = {
    "path_mae": 0.35,
    "terminal_mae": 0.15,
    "brier": 0.20,
    "wis": 0.15,
    "drawdown_volatility": 0.10,
    "recent_stability": 0.05,
}


@dataclass(frozen=True, slots=True)
class WeightedRidgeFit:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    transformed_x: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    normalized_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class V341ModelState:
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
class V341PathForecast:
    market: str
    anchor_date: date
    model_version: str
    weekly_base: tuple[float, ...]
    daily_correction: tuple[float, ...]
    expected_path: tuple[float, ...]
    source_feature_hash: str
    forecast_sigma: float
    ood_diagnostics: Mapping[str, Any]


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


def state_to_payload(state: V341ModelState) -> dict[str, Any]:
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


def state_from_payload(payload: Mapping[str, Any]) -> V341ModelState:
    return V341ModelState(
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
            tuple(float(value) for value in row)
            for row in payload["ood_precision_matrix"]
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


def candidate_alphas(champion_alpha: float) -> tuple[float, ...]:
    """Return the complete bounded regular-candidate set in stable order."""

    alpha = float(champion_alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("champion_alpha must be finite and positive")
    values: list[float] = []
    for multiplier in ALPHA_MULTIPLIERS:
        value = round(float(np.clip(alpha * multiplier, 0.1, 100.0)), 12)
        if value not in values:
            values.append(value)
    return tuple(values[:5])


def conservative_independent_sample_count(
    intervals: Sequence[tuple[date, date]],
) -> int:
    """Maximum count of non-overlapping closed label intervals.

    The greedy earliest-finish algorithm is deterministic and intentionally
    conservative for weekly anchors with overlapping 13-week labels.
    """

    normalized = sorted(intervals, key=lambda item: (item[1], item[0]))
    count = 0
    last_end: date | None = None
    for start, end in normalized:
        if end < start:
            raise ValueError("label interval end precedes start")
        if last_end is None or start >= last_end:
            count += 1
            last_end = end
    return count


def weighted_ridge_fit(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> WeightedRidgeFit:
    """Fit Ridge with fold-local imputation and weighted X/Y centering."""

    x_raw = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x_raw.ndim != 2 or y.ndim != 2 or len(x_raw) != len(y) or len(w) != len(y):
        raise ValueError("x, y and weights must share a two-dimensional row axis")
    if not len(w) or np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("weights must be finite and positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")

    medians = np.asarray(
        [
            float(np.median(column[np.isfinite(column)]))
            if np.any(np.isfinite(column))
            else 0.0
            for column in x_raw.T
        ],
        dtype=float,
    )
    filled = np.where(np.isfinite(x_raw), x_raw, medians)
    normalized_w = w / float(np.mean(w))
    total_w = float(np.sum(normalized_w))
    means = np.sum(filled * normalized_w[:, None], axis=0) / total_w
    centered_for_scale = filled - means
    scales = np.sqrt(
        np.sum(centered_for_scale * centered_for_scale * normalized_w[:, None], axis=0)
        / total_w
    )
    scales = np.where(scales < 1e-9, 1.0, scales)
    transformed = np.clip(centered_for_scale / scales, -8.0, 8.0)
    x_mean_w = np.sum(transformed * normalized_w[:, None], axis=0) / total_w
    y_mean_w = np.sum(y * normalized_w[:, None], axis=0) / total_w
    centered_x = transformed - x_mean_w
    centered_y = y - y_mean_w
    root_w = np.sqrt(normalized_w)[:, None]
    weighted_x = centered_x * root_w
    weighted_y = centered_y * root_w
    gram = weighted_x.T @ weighted_x + float(alpha) * np.eye(x_raw.shape[1])
    coefficients = np.linalg.solve(gram, weighted_x.T @ weighted_y)
    intercept = y_mean_w - x_mean_w @ coefficients
    return WeightedRidgeFit(
        medians=medians,
        means=means,
        scales=scales,
        transformed_x=transformed,
        intercept=np.asarray(intercept, dtype=float),
        coefficients=np.asarray(coefficients, dtype=float),
        normalized_weights=normalized_w,
    )


def multiclass_brier(probabilities: Sequence[float], actual_class: int) -> float:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (3,) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("three class probabilities must be within [0, 1]")
    if not np.isclose(np.sum(values), 1.0, atol=1e-9):
        raise ValueError("class probabilities must sum to one")
    if actual_class not in (0, 1, 2):
        raise ValueError("actual_class must be 0, 1 or 2")
    target = np.zeros(3, dtype=float)
    target[actual_class] = 1.0
    return float(0.5 * np.sum((values - target) ** 2))


def weighted_interval_score(p10: float, p50: float, p90: float, actual: float) -> float:
    lower, median, upper, observed = map(float, (p10, p50, p90, actual))
    if not lower <= median <= upper:
        raise ValueError("P10/P50/P90 must be ordered")
    interval_score = upper - lower
    if observed < lower:
        interval_score += 10.0 * (lower - observed)
    elif observed > upper:
        interval_score += 10.0 * (observed - upper)
    return float((0.5 * abs(observed - median) + 0.1 * interval_score) / 1.5)


def true_max_drawdown(cumulative_returns: Sequence[float]) -> float:
    wealth = np.concatenate(([1.0], 1.0 + np.asarray(cumulative_returns, dtype=float)))
    if np.any(wealth <= 0.0):
        raise ValueError("cumulative returns imply non-positive wealth")
    peaks = np.maximum.accumulate(wealth)
    return float(np.max((peaks - wealth) / peaks))


def path_turning_type(cumulative_returns: Sequence[float]) -> str:
    """Classify the dominant 13-week path shape for promotion hard gates."""

    values = np.asarray(cumulative_returns, dtype=float)
    if values.shape != (13,) or np.any(~np.isfinite(values)):
        raise ValueError("turning type requires one finite 13-week path")
    amplitude = float(np.max(values) - np.min(values))
    terminal = float(values[-1])
    scale = max(float(np.std(values)), 0.01)
    if abs(terminal) <= scale and amplitude <= 3.0 * scale:
        return "RANGE"
    if int(np.argmin(values)) < 6 and terminal - float(np.min(values)) > scale:
        return "V_SHAPE"
    if int(np.argmax(values)) < 6 and float(np.max(values)) - terminal > scale:
        return "INVERTED_V"
    return "TREND_UP" if terminal > 0.0 else "TREND_DOWN"


def _realized_volatility(cumulative_returns: np.ndarray) -> float:
    wealth = 1.0 + np.asarray(cumulative_returns, dtype=float)
    if np.any(wealth <= 0.0):
        raise ValueError("cumulative returns imply non-positive wealth")
    log_wealth = np.log(np.concatenate(([1.0], wealth)))
    weekly = np.diff(log_wealth)
    return float(np.std(weekly, ddof=0) * math.sqrt(52.0))


def _actual_class(value: float, threshold: float) -> int:
    return 0 if value > threshold else 2 if value < -threshold else 1


def normalized_loss_components(
    actual_paths: np.ndarray,
    predicted_paths: np.ndarray,
    scenario_paths: np.ndarray,
    probabilities: Mapping[int, np.ndarray],
    thresholds: Mapping[int, float],
    *,
    scales: np.ndarray,
    recent_fold_losses: Sequence[float],
) -> dict[str, object]:
    """Calculate the frozen 35/15/20/15/10/5 evaluation schema."""

    actual = np.asarray(actual_paths, dtype=float)
    predicted = np.asarray(predicted_paths, dtype=float)
    scenarios = np.asarray(scenario_paths, dtype=float)
    scale = np.maximum(np.asarray(scales, dtype=float), 0.01)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 13:
        raise ValueError("actual and predicted paths must be N x 13")
    if scenarios.ndim != 3 or scenarios.shape[0] != len(actual) or scenarios.shape[2] != 13:
        raise ValueError("scenario paths must be N x scenarios x 13")
    if scale.shape != (13,):
        raise ValueError("scales must contain 13 horizons")

    path_mae = float(np.mean(np.abs(predicted - actual) / scale))
    horizon_indices = {4: 3, 8: 7, 13: 12}
    terminal_mae = float(
        np.mean(
            [
                np.mean(np.abs(predicted[:, index] - actual[:, index]) / scale[index])
                for index in horizon_indices.values()
            ]
        )
    )
    brier_values: list[float] = []
    for horizon, index in horizon_indices.items():
        rows = np.asarray(probabilities[horizon], dtype=float)
        if rows.shape != (len(actual), 3):
            raise ValueError(f"probabilities for {horizon}W must be N x 3")
        threshold = float(thresholds[horizon])
        brier_values.extend(
            multiclass_brier(rows[row_index], _actual_class(actual[row_index, index], threshold))
            for row_index in range(len(actual))
        )
    brier = float(np.mean(brier_values))

    p10 = np.quantile(scenarios, 0.10, axis=1)
    p50 = np.quantile(scenarios, 0.50, axis=1)
    p90 = np.quantile(scenarios, 0.90, axis=1)
    wis = float(
        np.mean(
            [
                weighted_interval_score(p10[row, horizon], p50[row, horizon], p90[row, horizon], actual[row, horizon])
                / scale[horizon]
                for row in range(len(actual))
                for horizon in range(13)
            ]
        )
    )

    predicted_drawdown = np.asarray([true_max_drawdown(row) for row in predicted])
    actual_drawdown = np.asarray([true_max_drawdown(row) for row in actual])
    predicted_vol = np.asarray([_realized_volatility(row) for row in predicted])
    actual_vol = np.asarray([_realized_volatility(row) for row in actual])
    drawdown_volatility = float(
        0.5 * np.mean(np.abs(predicted_drawdown - actual_drawdown) / 0.10)
        + 0.5 * np.mean(np.abs(predicted_vol - actual_vol) / 0.20)
    )

    recent = np.asarray(recent_fold_losses, dtype=float)
    if recent.size < 3 or np.any(~np.isfinite(recent)):
        raise ValueError("recent stability requires at least three finite fold losses")
    median = max(float(np.median(np.abs(recent))), 1e-9)
    mad = float(np.median(np.abs(recent - np.median(recent))))
    recent_stability = float(0.5 * recent[-1] + 0.5 * mad / median)
    components = {
        "path_mae": path_mae,
        "terminal_mae": terminal_mae,
        "brier": brier,
        "wis": wis,
        "drawdown_volatility": drawdown_volatility,
        "recent_stability": recent_stability,
    }
    total = float(sum(LOSS_WEIGHTS[name] * value for name, value in components.items()))
    return {
        "schema_version": LOSS_SCHEMA_VERSION,
        "components": components,
        "weights": dict(LOSS_WEIGHTS),
        "total": total,
    }


def paired_block_bootstrap_interval(
    paired_differences: Sequence[float],
    *,
    block_length: int = 13,
    fold_lengths: Sequence[int],
    seed: int,
    draws: int = 2000,
) -> tuple[float, float]:
    values = np.asarray(paired_differences, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("paired differences must be a finite one-dimensional series")
    if block_length <= 0 or block_length > len(values):
        raise ValueError("invalid block length")
    lengths = tuple(int(value) for value in fold_lengths)
    if not lengths or any(value < block_length for value in lengths):
        raise ValueError("each bootstrap fold must contain a complete block")
    if sum(lengths) != len(values):
        raise ValueError("fold lengths must partition paired differences exactly")
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    rng = np.random.default_rng(int(seed))
    fold_ranges: list[tuple[int, int]] = []
    offset = 0
    for length in lengths:
        fold_ranges.append((offset, offset + length))
        offset += length
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled: list[float] = []
        for fold_start, fold_end in fold_ranges:
            fold_sampled: list[float] = []
            possible_starts = np.arange(fold_start, fold_end - block_length + 1)
            target = fold_end - fold_start
            while len(fold_sampled) < target:
                start = int(rng.choice(possible_starts))
                fold_sampled.extend(values[start : start + block_length].tolist())
            sampled.extend(fold_sampled[:target])
        means[draw] = float(np.mean(sampled[: len(values)]))
    return (float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95)))


class V341TrainingService:
    """Weighted multi-horizon Ridge with frozen manifests and OOD metadata."""

    @staticmethod
    def _feature_row(
        snapshot: FeatureSnapshot,
        feature_names: Sequence[str],
    ) -> list[float]:
        values = flatten_snapshot(snapshot)
        if snapshot.market == "159941":
            assert_159941_self_only(values)
        return [
            np.nan if values.get(name) is None else float(values[name])
            for name in feature_names
        ]

    def _matrix(
        self,
        samples: Sequence[V34TrainingSample],
        feature_names: Sequence[str],
    ) -> np.ndarray:
        return np.asarray(
            [self._feature_row(sample.snapshot, feature_names) for sample in samples],
            dtype=float,
        )

    def fit(
        self,
        market: str,
        matured: Sequence[V34TrainingSample],
        *,
        version: str,
        parent_version: str | None,
        trained_through: date,
        effective_from: date,
        alpha: float,
        feature_names: Sequence[str],
        validation_metrics: Mapping[str, Any] | None = None,
    ) -> V341ModelState:
        rows = tuple(sorted(matured, key=lambda item: item.anchor_date))
        if len(rows) < 8:
            raise ValueError("V3.4.1 model fit requires at least eight mature samples")
        if any(
            row.market != market
            or row.cumulative_return_path is None
            or row.label_end_date is None
            or row.label_end_date > trained_through
            for row in rows
        ):
            raise ValueError("model fit contains cross-market or immature/future labels")
        names = tuple(str(name) for name in feature_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("feature manifest must be non-empty and unique")
        x_raw = self._matrix(rows, names)
        y = np.asarray([row.cumulative_return_path for row in rows], dtype=float)
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
            "protocol_version": PROTOCOL_VERSION,
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
            "daily_100_session_mode": "IN_RIDGE_FEATURES_ONLY_NO_POSTHOC_DOUBLE_COUNT",
        }
        parameter_hash = _hash(payload)
        return V341ModelState(
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
        state: V341ModelState,
        snapshot: FeatureSnapshot,
    ) -> V341PathForecast:
        if snapshot.market != state.market:
            raise ValueError("prediction cannot cross market chains")
        if snapshot.cutoff_date < state.effective_from_date:
            raise ValueError("staged model is not effective at the forecast anchor")
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
        return V341PathForecast(
            market=state.market,
            anchor_date=snapshot.cutoff_date,
            model_version=state.version,
            weekly_base=tuple(float(value) for value in expected),
            daily_correction=tuple(0.0 for _ in range(HORIZON_WEEKS)),
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
        rows: Sequence[V34TrainingSample],
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
                self.predict(
                    replace(state, effective_from_date=row.anchor_date), row.snapshot
                ).expected_path
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

    def fold_composite_losses(
        self,
        market: str,
        rows: Sequence[V34TrainingSample],
        training_indices: Sequence[int],
        validation_indices: Sequence[int],
        *,
        alpha: float,
        feature_names: Sequence[str],
        standardized_residuals: np.ndarray,
        shared_seed: int,
        scenario_count: int = 1000,
        recent_fold_oos_losses: Sequence[float] = (),
        capture_random_plans: bool = False,
    ) -> tuple[
        np.ndarray,
        tuple[str, ...],
        tuple[Mapping[str, float], ...],
        tuple[Mapping[str, Any], ...],
    ]:
        """Score one purged fold with exact six-component losses.

        The random indices depend on the frozen fold/anchor seed and residual
        pool only.  They deliberately do not depend on alpha or model version,
        so champion and every candidate consume identical stochastic plans.
        """

        training = tuple(rows[index] for index in training_indices)
        validation = tuple(rows[index] for index in validation_indices)
        residual_pool = np.asarray(standardized_residuals, dtype=float)
        if residual_pool.ndim != 2 or residual_pool.shape[1] != 13 or not len(residual_pool):
            raise ValueError("composite fold requires a non-empty N x 13 residual pool")
        if scenario_count < 1000:
            raise ValueError("composite fold requires at least 1000 scenarios")
        state = self.fit(
            market,
            training,
            version="FOLD-COMPOSITE",
            parent_version=None,
            trained_through=max(row.label_end_date for row in training if row.label_end_date),
            effective_from=validation[0].anchor_date,
            alpha=alpha,
            feature_names=feature_names,
        )
        predictions = np.asarray(
            [
                self.predict(
                    replace(state, effective_from_date=row.anchor_date), row.snapshot
                ).expected_path
                for row in validation
            ],
            dtype=float,
        )
        actual = np.asarray([row.cumulative_return_path for row in validation], dtype=float)
        train_y = np.asarray([row.cumulative_return_path for row in training], dtype=float)
        q25 = np.quantile(train_y, 0.25, axis=0)
        q75 = np.quantile(train_y, 0.75, axis=0)
        scales = np.maximum((q75 - q25) / 1.349, 0.01)
        losses: list[float] = []
        plan_hashes: list[str] = []
        components_out: list[Mapping[str, float]] = []
        plan_payloads: list[Mapping[str, Any]] = []
        fold_history = np.asarray(tuple(float(value) for value in recent_fold_oos_losses), dtype=float)
        if fold_history.size == 0:
            fold_history = np.zeros(3, dtype=float)
        elif fold_history.size < 3:
            fold_history = np.pad(fold_history, (3 - fold_history.size, 0), mode="edge")
        fold_history = fold_history[-3:]
        fold_median = max(float(np.median(np.abs(fold_history))), 1e-9)
        fold_mad = float(np.median(np.abs(fold_history - np.median(fold_history))))
        recent_fold_stability = float(
            0.5 * fold_history[-1] + 0.5 * fold_mad / fold_median
        )
        for row_index, (sample, predicted, observed) in enumerate(
            zip(validation, predictions, actual)
        ):
            anchor_seed = int.from_bytes(
                hashlib.sha256(
                    f"{shared_seed}:{sample.anchor_date.isoformat()}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(anchor_seed)
            indices = rng.integers(0, len(residual_pool), size=scenario_count)
            plan_payload = {
                "version": CANDIDATE_RANDOM_PLAN_VERSION,
                "seed": anchor_seed,
                "scenario_count": scenario_count,
                "residual_pool_size": len(residual_pool),
                "residual_indices": indices.tolist(),
                "consumed_fields": ["residual_indices"],
                "semantics": "RETURN_SCENARIO_LOSS_ONLY",
            }
            plan_hash = hashlib.sha256(
                json.dumps(
                    plan_payload,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            increments = np.diff(np.concatenate(([0.0], predicted)))
            sigma = max(float(np.std(increments)), 0.01)
            if np.any(predicted <= -1.0) or np.any(~np.isfinite(predicted)):
                raise ValueError(
                    "candidate prediction must imply positive gross returns"
                )
            log_gross = np.log1p(predicted) + residual_pool[indices] * sigma
            scenarios = np.expm1(log_gross)
            if np.any(~np.isfinite(scenarios)) or np.any(scenarios <= -1.0):
                raise ValueError(
                    "candidate log-gross scenarios must imply positive prices"
                )
            path_mae = float(np.mean(np.abs(predicted - observed) / scales))
            terminal_mae = float(
                np.mean(
                    [
                        abs(predicted[horizon - 1] - observed[horizon - 1])
                        / scales[horizon - 1]
                        for horizon in (4, 8, 13)
                    ]
                )
            )
            briers = []
            for horizon in (4, 8, 13):
                threshold = max(0.01, sigma * math.sqrt(horizon) * 0.50)
                terminal = scenarios[:, horizon - 1]
                probabilities = np.asarray(
                    [
                        np.mean(terminal > threshold),
                        np.mean((terminal >= -threshold) & (terminal <= threshold)),
                        np.mean(terminal < -threshold),
                    ],
                    dtype=float,
                )
                actual_class = _actual_class(float(observed[horizon - 1]), threshold)
                briers.append(multiclass_brier(probabilities, actual_class))
            brier = float(np.mean(briers))
            p10 = np.quantile(scenarios, 0.10, axis=0)
            p50 = np.quantile(scenarios, 0.50, axis=0)
            p90 = np.quantile(scenarios, 0.90, axis=0)
            wis = float(
                np.mean(
                    [
                        weighted_interval_score(
                            p10[horizon], p50[horizon], p90[horizon], observed[horizon]
                        )
                        / scales[horizon]
                        for horizon in range(13)
                    ]
                )
            )
            coverage = float(np.mean((observed >= p10) & (observed <= p90)))
            turning_type_correct = float(
                path_turning_type(predicted) == path_turning_type(observed)
            )
            drawdown_volatility = float(
                0.5
                * abs(true_max_drawdown(predicted) - true_max_drawdown(observed))
                / 0.10
                + 0.5
                * abs(_realized_volatility(predicted) - _realized_volatility(observed))
                / 0.20
            )
            recent_stability = recent_fold_stability
            components = {
                "path_mae": path_mae,
                "terminal_mae": terminal_mae,
                "brier": brier,
                "wis": wis,
                "drawdown_volatility": drawdown_volatility,
                "recent_stability": recent_stability,
                "coverage": coverage,
                "turning_type_correct": turning_type_correct,
            }
            losses.append(
                float(
                    sum(
                        LOSS_WEIGHTS[name] * float(components[name])
                        for name in LOSS_WEIGHTS
                    )
                )
            )
            plan_hashes.append(plan_hash)
            if capture_random_plans:
                plan_payloads.append({**plan_payload, "plan_hash": plan_hash})
            components_out.append(components)
        return (
            np.asarray(losses, dtype=float),
            tuple(plan_hashes),
            tuple(components_out),
            tuple(plan_payloads),
        )
