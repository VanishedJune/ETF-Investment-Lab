"""Strict rolling V3.4 13-week trainer and promotion gate.

The model family remains the V3.3 weekly ridge/residual ensemble.  V3.4 changes
the label horizon, maturity clock, fold-local preprocessing, overlap weights,
and candidate/champion lifecycle; it does not mutate the frozen V3.3 chain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.app.services.v33_feature_service import FeatureSnapshot, flatten_snapshot
from backend.app.services.v34_feature_service import (
    SCALER_VERSION,
    SUPPORTED_MARKETS,
    assert_159941_self_only,
)


HORIZON_WEEKS = 13
DAILY_CORRECTION_WEEKS = 4
MIN_NEW_MATURED_SAMPLES_FOR_RETRAIN = 4
MIN_TRAIN_SAMPLES = 30
PURGE_WEEKS = HORIZON_WEEKS
MODEL_FAMILY = "weekly_ridge_residual_ensemble"
METHODOLOGY_VERSION = "V3.4_13W_STRICT_ROLLING_1"
MATURITY_STATES = (
    "IMMATURE",
    "MATURE_1W",
    "MATURE_4W",
    "MATURE_8W",
    "FULLY_MATURE_13W",
    "TRAINED",
)


class V34TrainingError(RuntimeError):
    pass


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class V34TrainingSample:
    market: str
    anchor_date: date
    snapshot: FeatureSnapshot
    label_dates: tuple[date, ...]
    cumulative_return_path: tuple[float, ...] | None
    realized_ohlcv: tuple[Mapping[str, float | str | None], ...]
    label_end_date: date | None

    def __post_init__(self) -> None:
        if self.market != self.snapshot.market:
            raise V34TrainingError("sample market/snapshot mismatch")
        if self.snapshot.source_data_max_date > self.anchor_date:
            raise V34TrainingError("feature snapshot reads data after its anchor")
        if self.cumulative_return_path is None:
            if self.label_dates or self.realized_ohlcv or self.label_end_date is not None:
                raise V34TrainingError("immature sample cannot contain partial future labels")
            return
        if len(self.cumulative_return_path) != HORIZON_WEEKS:
            raise V34TrainingError("mature sample requires exactly 13 weekly returns")
        if len(self.label_dates) != HORIZON_WEEKS or len(self.realized_ohlcv) != HORIZON_WEEKS:
            raise V34TrainingError("mature sample requires exactly 13 weekly OHLCV labels")
        if self.label_end_date != self.label_dates[-1]:
            raise V34TrainingError("label_end_date must be the thirteenth weekly anchor")


@dataclass(frozen=True, slots=True)
class V34ModelState:
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
    residual_paths: tuple[tuple[float, ...], ...]
    hyperparameters: Mapping[str, float]
    raw_matured_sample_count: int
    effective_independent_sample_count: float
    validation_metrics: Mapping[str, Any]
    feature_fit_end_date: date
    parameter_hash: str


@dataclass(frozen=True, slots=True)
class V34PathForecast:
    market: str
    anchor_date: date
    model_version: str
    weekly_base: tuple[float, ...]
    daily_correction: tuple[float, ...]
    expected_path: tuple[float, ...]
    residual_paths: tuple[tuple[float, ...], ...]
    source_feature_hash: str


@dataclass(frozen=True, slots=True)
class V34PromotionDecision:
    promoted: bool
    reason: str
    fold_losses_champion: tuple[float, ...]
    fold_losses_candidate: tuple[float, ...]
    majority_improved: bool
    worst_fold_degradation: float
    bootstrap_improvement_interval: tuple[float, float]
    raw_matured_sample_count: int
    effective_independent_sample_count: float


def maturity_status(realized_week_count: int, *, trained: bool = False) -> str:
    if trained:
        return "TRAINED"
    if realized_week_count >= 13:
        return "FULLY_MATURE_13W"
    if realized_week_count >= 8:
        return "MATURE_8W"
    if realized_week_count >= 4:
        return "MATURE_4W"
    if realized_week_count >= 1:
        return "MATURE_1W"
    return "IMMATURE"


def _week_ohlcv(snapshot: FeatureSnapshot, previous_anchor: date) -> dict[str, float | str | None]:
    rows = [
        row
        for row in snapshot.daily_sequence
        if previous_anchor < date.fromisoformat(str(row["trade_date"])) <= snapshot.cutoff_date
    ]
    if not rows:
        rows = [snapshot.daily_sequence[-1]]
    volumes = [float(row["volume"]) for row in rows if row.get("volume") is not None]
    return {
        "week_end": snapshot.cutoff_date.isoformat(),
        "open": float(rows[0]["open"]),
        "high": max(float(row["high"]) for row in rows),
        "low": min(float(row["low"]) for row in rows),
        "close": float(rows[-1]["close"]),
        "volume": None if not volumes else float(sum(volumes)),
    }


def build_training_samples(
    snapshots: Sequence[FeatureSnapshot],
) -> tuple[V34TrainingSample, ...]:
    ordered = sorted(snapshots, key=lambda item: item.cutoff_date)
    if not ordered:
        return ()
    market = ordered[0].market
    if market not in SUPPORTED_MARKETS or any(item.market != market for item in ordered):
        raise V34TrainingError("training snapshots must belong to one supported market")
    if any(left.cutoff_date >= right.cutoff_date for left, right in zip(ordered, ordered[1:])):
        raise V34TrainingError("weekly anchors must be unique and increasing")
    closes = [float(item.daily_sequence[-1]["close"]) for item in ordered]
    result: list[V34TrainingSample] = []
    for index, snapshot in enumerate(ordered):
        if index + HORIZON_WEEKS >= len(ordered):
            labels: tuple[date, ...] = ()
            path = None
            ohlcv: tuple[Mapping[str, float | str | None], ...] = ()
            end = None
        else:
            futures = ordered[index + 1 : index + HORIZON_WEEKS + 1]
            labels = tuple(item.cutoff_date for item in futures)
            path = tuple(closes[index + step] / closes[index] - 1.0 for step in range(1, 14))
            ohlcv = tuple(
                _week_ohlcv(item, ordered[index + step - 1].cutoff_date)
                for step, item in enumerate(futures, start=1)
            )
            end = labels[-1]
        result.append(
            V34TrainingSample(
                market=market,
                anchor_date=snapshot.cutoff_date,
                snapshot=snapshot,
                label_dates=labels,
                cumulative_return_path=path,
                realized_ohlcv=ohlcv,
                label_end_date=end,
            )
        )
    return tuple(result)


def eligible_fully_matured(
    samples: Sequence[V34TrainingSample], as_of: date
) -> tuple[V34TrainingSample, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.anchor_date < as_of
        and sample.cumulative_return_path is not None
        and sample.label_end_date is not None
        and sample.label_end_date <= as_of
    )


def overlap_weights(samples: Sequence[V34TrainingSample]) -> np.ndarray:
    if not samples:
        return np.asarray([], dtype=float)
    weights: list[float] = []
    for sample in samples:
        if sample.label_end_date is None:
            raise V34TrainingError("overlap weights require fully matured labels")
        concurrent = 1
        for other in samples:
            if other is sample or other.label_end_date is None:
                continue
            if other.anchor_date < sample.label_end_date and sample.anchor_date < other.label_end_date:
                concurrent += 1
        weights.append(1.0 / concurrent)
    return np.asarray(weights, dtype=float)


def effective_sample_count(weights: np.ndarray) -> float:
    if weights.size == 0 or float(np.sum(weights * weights)) <= 0.0:
        return 0.0
    return float(np.sum(weights) ** 2 / np.sum(weights * weights))


def purged_walk_forward_folds(
    samples: Sequence[V34TrainingSample], *, folds: int = 5
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    ordered = sorted(samples, key=lambda item: item.anchor_date)
    if len(ordered) < MIN_TRAIN_SAMPLES:
        return ()
    first_validation = max(MIN_TRAIN_SAMPLES // 2, len(ordered) // 3)
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
            if sample.label_end_date is not None
            and sample.label_end_date < validation_start
        )
        if len(training) >= 8:
            if any(ordered[index].label_end_date >= validation_start for index in training):
                raise V34TrainingError("purged fold contains overlapping label intervals")
            result.append((training, validation))
    return tuple(result)


class V34TrainingService:
    def __init__(self, *, random_seed: int = 340013) -> None:
        self.random_seed = random_seed

    @staticmethod
    def _features(snapshot: FeatureSnapshot) -> dict[str, float | None]:
        values = flatten_snapshot(snapshot)
        if snapshot.market == "159941":
            assert_159941_self_only(values)
        return values

    def _matrix(
        self,
        samples: Sequence[V34TrainingSample],
        feature_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        rows = [self._features(sample.snapshot) for sample in samples]
        names = (
            tuple(feature_names)
            if feature_names is not None
            else tuple(sorted({name for row in rows for name in row}))
        )
        return (
            np.asarray(
                [
                    [np.nan if row.get(name) is None else float(row[name]) for name in names]
                    for row in rows
                ],
                dtype=float,
            ),
            names,
        )

    @staticmethod
    def _fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finite = np.where(np.isfinite(values), values, np.nan)
        medians = np.asarray(
            [
                float(np.median(column[np.isfinite(column)]))
                if np.any(np.isfinite(column))
                else 0.0
                for column in finite.T
            ],
            dtype=float,
        )
        filled = np.where(np.isfinite(values), values, medians)
        means = np.mean(filled, axis=0)
        scales = np.std(filled, axis=0)
        scales = np.where(scales < 1e-9, 1.0, scales)
        return medians, means, scales

    @staticmethod
    def _transform(
        values: np.ndarray, medians: np.ndarray, means: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, medians)
        return np.clip((filled - means) / scales, -8.0, 8.0)

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
        validation_metrics: Mapping[str, Any] | None = None,
    ) -> V34ModelState:
        rows = tuple(sorted(matured, key=lambda item: item.anchor_date))
        if market not in SUPPORTED_MARKETS or any(row.market != market for row in rows):
            raise V34TrainingError("cross-market model fitting is forbidden")
        if len(rows) < 8 or any(row.cumulative_return_path is None for row in rows):
            raise V34TrainingError("model fit requires complete 13-week labels")
        x_raw, names = self._matrix(rows)
        medians, means, scales = self._fit_scaler(x_raw)
        x = self._transform(x_raw, medians, means, scales)
        y = np.asarray([row.cumulative_return_path for row in rows], dtype=float)
        weights = overlap_weights(rows)
        root_w = np.sqrt(np.maximum(weights, 1e-9))[:, None]
        intercept = np.sum(y * weights[:, None], axis=0) / np.sum(weights)
        centered = y - intercept
        weighted_x = x * root_w
        weighted_y = centered * root_w
        gram = weighted_x.T @ weighted_x + float(alpha) * np.eye(x.shape[1])
        coefficients = np.linalg.solve(gram, weighted_x.T @ weighted_y)
        predicted = intercept + x @ coefficients
        residuals = y - predicted
        parameters = {
            "model_family": MODEL_FAMILY,
            "methodology_version": METHODOLOGY_VERSION,
            "horizon_weeks": 13,
            "ridge_alpha": float(alpha),
            "daily_correction_weeks": 4,
            "scaler_version": SCALER_VERSION,
            "feature_fit_end_date": trained_through.isoformat(),
            "random_seed": self.random_seed,
        }
        payload = {
            "market": market,
            "version": version,
            "parent_version": parent_version,
            "trained_through": trained_through.isoformat(),
            "effective_from": effective_from.isoformat(),
            "feature_names": names,
            "medians": medians.tolist(),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "intercept": intercept.tolist(),
            "coefficients": coefficients.tolist(),
            "parameters": parameters,
            "raw_matured_sample_count": len(rows),
            "effective_independent_sample_count": effective_sample_count(weights),
        }
        return V34ModelState(
            market=market,
            version=version,
            parent_version=parent_version,
            trained_through_date=trained_through,
            effective_from_date=effective_from,
            feature_names=names,
            medians=tuple(float(value) for value in medians),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
            intercept=tuple(float(value) for value in intercept),
            coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
            residual_paths=tuple(
                tuple(float(value) for value in row) for row in residuals[-260:]
            ),
            hyperparameters=parameters,
            raw_matured_sample_count=len(rows),
            effective_independent_sample_count=effective_sample_count(weights),
            validation_metrics=dict(validation_metrics or {}),
            feature_fit_end_date=trained_through,
            parameter_hash=_hash(payload),
        )

    def predict(self, state: V34ModelState, snapshot: FeatureSnapshot) -> V34PathForecast:
        if state.market != snapshot.market:
            raise V34TrainingError("prediction cannot use another market's champion")
        if state.effective_from_date > snapshot.cutoff_date:
            raise V34TrainingError("candidate is not yet effective at this forecast anchor")
        x_raw, _ = self._matrix(
            (
                V34TrainingSample(
                    market=snapshot.market,
                    anchor_date=snapshot.cutoff_date,
                    snapshot=snapshot,
                    label_dates=(),
                    cumulative_return_path=None,
                    realized_ohlcv=(),
                    label_end_date=None,
                ),
            ),
            state.feature_names,
        )
        x = self._transform(
            x_raw,
            np.asarray(state.medians),
            np.asarray(state.means),
            np.asarray(state.scales),
        )[0]
        weekly_base = np.asarray(state.intercept) + x @ np.asarray(state.coefficients)
        # The daily model applies once to incremental returns for weeks 1-4.
        # Its cumulative effect carries forward, but no new daily correction is
        # introduced after week four.
        daily = snapshot.daily
        raw_signal = (
            0.45 * float(daily.get("daily_dif_slope_3") or 0.0)
            + 0.25 * float(daily.get("daily_dea_slope_3") or 0.0)
            + 0.20 * float(daily.get("daily_histogram_speed_3") or 0.0)
            + 0.10 * float(daily.get("daily_return_20") or 0.0)
        )
        increment_adjustment = np.zeros(HORIZON_WEEKS, dtype=float)
        increment_adjustment[:4] = np.clip(raw_signal, -0.01, 0.01) * np.asarray(
            (1.0, 0.75, 0.5, 0.25)
        )
        daily_cumulative = np.cumsum(increment_adjustment)
        expected = np.clip(weekly_base + daily_cumulative, -0.75, 1.5)
        source_hash = _hash(
            {
                "market": snapshot.market,
                "anchor": snapshot.cutoff_date.isoformat(),
                "features": self._features(snapshot),
            }
        )
        return V34PathForecast(
            market=state.market,
            anchor_date=snapshot.cutoff_date,
            model_version=state.version,
            weekly_base=tuple(float(value) for value in weekly_base),
            daily_correction=tuple(float(value) for value in daily_cumulative),
            expected_path=tuple(float(value) for value in expected),
            residual_paths=state.residual_paths,
            source_feature_hash=source_hash,
        )

    def _fold_loss(
        self,
        market: str,
        rows: Sequence[V34TrainingSample],
        training_indices: Sequence[int],
        validation_indices: Sequence[int],
        *,
        alpha: float,
    ) -> float:
        training = tuple(rows[index] for index in training_indices)
        validation = tuple(rows[index] for index in validation_indices)
        state = self.fit(
            market,
            training,
            version="fold",
            parent_version=None,
            trained_through=training[-1].anchor_date,
            effective_from=validation[0].anchor_date,
            alpha=alpha,
        )
        predictions = np.asarray(
            [self.predict(replace(state, effective_from_date=row.anchor_date), row.snapshot).expected_path for row in validation]
        )
        actual = np.asarray([row.cumulative_return_path for row in validation], dtype=float)
        path_mae = float(np.mean(np.abs(predictions - actual)))
        terminal = float(np.mean(np.abs(predictions[:, -1] - actual[:, -1])))
        direction = float(
            np.mean((np.sign(predictions[:, -1]) - np.sign(actual[:, -1])) ** 2) / 4.0
        )
        drawdown = float(
            np.mean(
                np.abs(
                    np.min(predictions, axis=1) - np.min(actual, axis=1)
                )
            )
        )
        return 0.55 * path_mae + 0.20 * terminal + 0.15 * direction + 0.10 * drawdown

    def compare_candidate(
        self,
        market: str,
        matured: Sequence[V34TrainingSample],
        *,
        champion_alpha: float,
        candidate_alpha: float,
    ) -> V34PromotionDecision:
        rows = tuple(sorted(matured, key=lambda item: item.anchor_date))
        folds = purged_walk_forward_folds(rows)
        weights = overlap_weights(rows)
        if len(folds) < 2:
            return V34PromotionDecision(
                False,
                "insufficient_purged_walk_forward_folds",
                (),
                (),
                False,
                math.inf,
                (-math.inf, math.inf),
                len(rows),
                effective_sample_count(weights),
            )
        incumbent = tuple(
            self._fold_loss(market, rows, train, valid, alpha=champion_alpha)
            for train, valid in folds
        )
        candidate = tuple(
            self._fold_loss(market, rows, train, valid, alpha=candidate_alpha)
            for train, valid in folds
        )
        differences = np.asarray(candidate) - np.asarray(incumbent)
        majority = int(np.sum(differences < 0.0)) > len(differences) / 2
        worst = float(np.max(differences))
        rng = np.random.default_rng(self.random_seed + len(rows))
        bootstrap = np.asarray(
            [
                float(np.mean(differences[rng.integers(0, len(differences), len(differences))]))
                for _ in range(500)
            ]
        )
        interval = (float(np.quantile(bootstrap, 0.05)), float(np.quantile(bootstrap, 0.95)))
        mean_improvement = -float(np.mean(differences))
        promoted = (
            mean_improvement >= 0.002
            and majority
            and worst <= 0.01
            and interval[1] < 0.0
            and effective_sample_count(weights) >= 30.0
        )
        reason = (
            "candidate_significantly_improves_same_out_of_sample_folds"
            if promoted
            else "candidate_improvement_not_beyond_time_series_noise"
        )
        return V34PromotionDecision(
            promoted,
            reason,
            incumbent,
            candidate,
            majority,
            worst,
            interval,
            len(rows),
            effective_sample_count(weights),
        )

    @staticmethod
    def next_candidate_alpha(champion_alpha: float, candidate_number: int) -> float:
        multipliers = (0.75, 1.25, 0.9, 1.1)
        return max(0.1, float(champion_alpha) * multipliers[candidate_number % len(multipliers)])
