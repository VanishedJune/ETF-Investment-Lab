"""Versioned probability, residual, calibration and reliability primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from backend.app.services.v33_feature_service import FeatureSnapshot, flatten_snapshot
from backend.app.services.v34_scenario_service import _macd_future, _trend_state
from backend.app.services.v34_training_service import V34TrainingSample
from backend.app.services.v341_training_service import PROTOCOL_VERSION, V341PathForecast


SCENARIO_ADAPTER_VERSION = "V3.4.4_LOG_GROSS_OOS_RESIDUAL_SCENARIO_2"
RANDOM_PLAN_VERSION = "V3.4.1_COMMON_RANDOM_PLAN_2"
RESIDUAL_SCHEMA_VERSION = "V3.4.4_LOG_GROSS_ACTUAL_MINUS_PREDICTION_2"
THRESHOLD_FORMULA_VERSION = "V3.4.1_SIGMA_HORIZON_THRESHOLD_1"
RELIABILITY_VERSION = "V3.4.1_RELIABILITY_1"
MIN_EFFECTIVE_OOS_RESIDUALS = 30


def canonical_forecast_seed(
    protocol_version: str,
    market: str,
    anchor: date,
    model_version: str,
    adapter_version: str,
) -> int:
    payload = ":".join(
        (protocol_version, market, anchor.isoformat(), model_version, adapter_version)
    )
    # SQLite INTEGER is signed 64-bit.  Keep the deterministic identity in the
    # non-negative 63-bit domain so every valid seed is persistable on every
    # supported Python/SQLite build.
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class ScenarioRandomPlan:
    version: str
    seed: int
    scenario_count: int
    residual_pool_size: int
    empirical_pool_size: int
    residual_indices: tuple[int, ...]
    gap_indices: tuple[int, ...]
    upper_wick_indices: tuple[int, ...]
    lower_wick_indices: tuple[int, ...]
    volume_indices: tuple[int, ...]
    residual_groups: tuple[tuple[int, int, float, int], ...]
    plan_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "scenario_count": self.scenario_count,
            "residual_pool_size": self.residual_pool_size,
            "empirical_pool_size": self.empirical_pool_size,
            "residual_indices": list(self.residual_indices),
            "gap_indices": list(self.gap_indices),
            "upper_wick_indices": list(self.upper_wick_indices),
            "lower_wick_indices": list(self.lower_wick_indices),
            "volume_indices": list(self.volume_indices),
            "residual_groups": [
                {
                    "start": start,
                    "stop": stop,
                    "target_weight": target_weight,
                    "draw_count": draw_count,
                }
                for start, stop, target_weight, draw_count in self.residual_groups
            ],
            "plan_hash": self.plan_hash,
        }

    @classmethod
    def build(
        cls,
        *,
        seed: int,
        scenario_count: int,
        residual_pool_size: int,
        empirical_pool_size: int,
        residual_groups: Sequence[tuple[int, int, float]] | None = None,
    ) -> "ScenarioRandomPlan":
        if scenario_count < 1000:
            raise ValueError("at least 1000 scenarios are required")
        if residual_pool_size <= 0 or empirical_pool_size <= 0:
            raise ValueError("random-plan pools must be non-empty")
        rng = np.random.default_rng(int(seed))
        groups = tuple(residual_groups or ((0, residual_pool_size, 1.0),))
        if (
            not groups
            or any(start < 0 or stop <= start or stop > residual_pool_size for start, stop, _ in groups)
            or any(weight < 0.0 or not math.isfinite(weight) for _, _, weight in groups)
        ):
            raise ValueError("invalid residual sampling groups")
        total_weight = float(sum(weight for _, _, weight in groups))
        if total_weight <= 0.0:
            raise ValueError("residual sampling group weights must be positive")
        normalized = tuple(weight / total_weight for _, _, weight in groups)
        raw_counts = np.asarray(normalized, dtype=float) * scenario_count
        draw_counts = np.floor(raw_counts).astype(int)
        remainder = scenario_count - int(np.sum(draw_counts))
        if remainder:
            order = np.argsort(-(raw_counts - draw_counts), kind="stable")
            for index in order[:remainder]:
                draw_counts[int(index)] += 1
        sampled: list[int] = []
        frozen_groups: list[tuple[int, int, float, int]] = []
        for (start, stop, _), target_weight, draw_count in zip(
            groups, normalized, draw_counts
        ):
            sampled.extend(
                int(value)
                for value in rng.integers(start, stop, size=int(draw_count))
            )
            frozen_groups.append(
                (int(start), int(stop), float(target_weight), int(draw_count))
            )
        rng.shuffle(sampled)
        residual = tuple(sampled)
        matrices = [
            tuple(
                int(value)
                for value in rng.integers(
                    0, empirical_pool_size, size=scenario_count * 13
                )
            )
            for _ in range(4)
        ]
        frozen_group_payload = [
            {
                "start": start,
                "stop": stop,
                "target_weight": target_weight,
                "draw_count": draw_count,
            }
            for start, stop, target_weight, draw_count in frozen_groups
        ]
        payload = {
            "version": RANDOM_PLAN_VERSION,
            "seed": int(seed),
            "scenario_count": int(scenario_count),
            "residual_pool_size": int(residual_pool_size),
            "empirical_pool_size": int(empirical_pool_size),
            "residual_indices": residual,
            "gap_indices": matrices[0],
            "upper_wick_indices": matrices[1],
            "lower_wick_indices": matrices[2],
            "volume_indices": matrices[3],
            "residual_groups": frozen_group_payload,
        }
        digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cls(
            version=RANDOM_PLAN_VERSION,
            seed=int(seed),
            scenario_count=int(scenario_count),
            residual_pool_size=int(residual_pool_size),
            empirical_pool_size=int(empirical_pool_size),
            residual_indices=residual,
            gap_indices=matrices[0],
            upper_wick_indices=matrices[1],
            lower_wick_indices=matrices[2],
            volume_indices=matrices[3],
            residual_groups=tuple(frozen_groups),
            plan_hash=digest,
        )


@dataclass(frozen=True, slots=True)
class V341ScenarioForecast:
    market: str
    anchor_date: date
    model_version: str
    scenario_seed: int
    scenario_count: int
    random_plan: ScenarioRandomPlan
    expected_path: tuple[float, ...]
    representative_ohlcv: tuple[Mapping[str, object], ...]
    indicators: tuple[Mapping[str, object], ...]
    price_quantiles: tuple[Mapping[str, object], ...]
    horizon_probabilities: Mapping[int, Mapping[str, Mapping[str, float]]]
    thresholds: Mapping[int, Mapping[str, object]]
    calibrator_versions: Mapping[int, str | None]
    residual_pool: Mapping[str, object]
    path_probabilities: Mapping[str, float]
    reliability: Mapping[str, object]
    health_status: str
    scenario_audit: Mapping[str, object]
    payload_hash: str


def prior_standardized_residual_pool(seed: int, size: int = 61) -> np.ndarray:
    """Explicit conservative prior used only before OOS residuals are sufficient."""

    if size < 31:
        raise ValueError("prior residual pool must contain at least 31 paths")
    rng = np.random.default_rng(int(seed) ^ 0x3410A55)
    weekly = rng.standard_t(df=5, size=(size, 13)) * 0.55
    cumulative = np.cumsum(weekly, axis=1) / math.sqrt(13.0)
    cumulative -= np.mean(cumulative, axis=0, keepdims=True)
    return cumulative


def _historical_weekly_rows(
    matured: Sequence[V34TrainingSample], anchor: date
) -> tuple[Mapping[str, float | str | None], ...]:
    rows: dict[str, Mapping[str, float | str | None]] = {}
    for sample in matured:
        for row in sample.realized_ohlcv:
            week_end = str(row["week_end"])
            if date.fromisoformat(week_end) <= anchor:
                rows[week_end] = row
    return tuple(rows[key] for key in sorted(rows))


def _historical_dif_tail(
    closes: Sequence[float],
) -> tuple[
    float,
    float,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return the final DIF, final first change and a short continuous tail."""

    if len(closes) < 3:
        raise ValueError("DIF continuity requires at least three historical closes")
    alpha12 = 2.0 / 13.0
    alpha26 = 2.0 / 27.0
    ema12 = float(closes[0])
    ema26 = float(closes[0])
    dea = 0.0
    dif_values: list[float] = []
    dea_values: list[float] = []
    macd_values: list[float] = []
    for close in closes:
        ema12 = alpha12 * float(close) + (1.0 - alpha12) * ema12
        ema26 = alpha26 * float(close) + (1.0 - alpha26) * ema26
        dif = ema12 - ema26
        dea = 0.2 * dif + 0.8 * dea
        dif_values.append(dif)
        dea_values.append(dea)
        macd_values.append(2.0 * (dif - dea))
    return (
        dif_values[-1],
        dif_values[-1] - dif_values[-2],
        tuple(dif_values[-3:]),
        tuple(dea_values[-3:]),
        tuple(macd_values[-3:]),
    )


def _probability_dict(values: Sequence[str], categories: Sequence[str]) -> dict[str, float]:
    counts = np.asarray([sum(value == category for value in values) for category in categories])
    probabilities = counts / max(float(np.sum(counts)), 1.0)
    return {category: float(probability) for category, probability in zip(categories, probabilities)}


def indicator_path_consistency(
    path_forecast: V341PathForecast,
    snapshot: FeatureSnapshot,
) -> dict[str, object]:
    """Compare the forecast direction with independently computed momentum votes."""

    values = flatten_snapshot(snapshot)
    vote_names = (
        "weekly_dif_slope_1",
        "weekly_macd_histogram",
        "daily_dif_slope_3",
        "daily_dea_slope_3",
        "daily_macd_histogram",
    )
    votes: dict[str, int] = {}
    for name in vote_names:
        value = values.get(name)
        if value is not None and math.isfinite(float(value)):
            votes[name] = 1 if float(value) > 0.0 else -1 if float(value) < 0.0 else 0
    terminal = float(path_forecast.expected_path[-1])
    forecast_direction = 1 if terminal > 0.01 else -1 if terminal < -0.01 else 0
    nonzero = tuple(value for value in votes.values() if value)
    vote_balance = 0.0 if not nonzero else float(sum(nonzero) / len(nonzero))
    consistent = (
        True
        if forecast_direction == 0 or len(nonzero) < 3
        else forecast_direction * vote_balance >= -0.20
    )
    return {
        "consistent": consistent,
        "forecast_direction": forecast_direction,
        "terminal_expected_return": terminal,
        "vote_balance": vote_balance,
        "available_vote_count": len(nonzero),
        "votes": votes,
        "version": "V3.4.1_INDICATOR_CONSISTENCY_1",
    }


def generate_scenario_forecast(
    path_forecast: V341PathForecast,
    snapshot: FeatureSnapshot,
    matured: Sequence[V34TrainingSample],
    *,
    standardized_residuals: np.ndarray,
    source_sigmas: Sequence[float],
    residual_pool_metadata: Mapping[str, object],
    calibrators: Mapping[int, Mapping[str, object]],
    scenario_count: int = 1200,
    seed: int | None = None,
    random_plan: ScenarioRandomPlan | None = None,
) -> V341ScenarioForecast:
    if snapshot.market != path_forecast.market:
        raise ValueError("scenario forecast cannot cross market chains")
    if seed is not None and random_plan is not None:
        raise ValueError("provide either a seed or a frozen random plan, not both")
    scenario_seed = (
        random_plan.seed
        if random_plan is not None
        else
        canonical_forecast_seed(
            PROTOCOL_VERSION,
            snapshot.market,
            snapshot.cutoff_date,
            path_forecast.model_version,
            SCENARIO_ADAPTER_VERSION,
        )
        if seed is None
        else int(seed)
    )
    history = _historical_weekly_rows(matured, snapshot.cutoff_date)
    if len(history) < 35:
        raise ValueError("V3.4.1 scenarios require 35 historical weekly rows")
    residuals = np.asarray(standardized_residuals, dtype=float)
    sigmas = np.asarray(source_sigmas, dtype=float)
    effective_residuals = int(residual_pool_metadata.get("effective_residual_count", 0))
    residual_sampling_groups: tuple[tuple[int, int, float], ...] | None = None
    empirical_path_count = 0
    if residuals.size == 0:
        residuals = prior_standardized_residual_pool(scenario_seed)
        sigmas = np.ones(len(residuals), dtype=float)
        residual_sampling_groups = ((0, len(residuals), 1.0),)
        residual_pool_metadata = {
            **dict(residual_pool_metadata),
            "source": "VERSIONED_CONSERVATIVE_PRIOR",
            "status": "INSUFFICIENT_OOS_RESIDUALS",
            "effective_residual_count": 0,
            "prior_size": len(residuals),
            "empirical_weight": 0.0,
            "prior_weight": 1.0,
            "shrinkage_threshold": MIN_EFFECTIVE_OOS_RESIDUALS,
        }
    elif effective_residuals < MIN_EFFECTIVE_OOS_RESIDUALS:
        empirical_count = len(residuals)
        empirical_path_count = empirical_count
        prior_count = max(MIN_EFFECTIVE_OOS_RESIDUALS - effective_residuals, 1)
        prior = prior_standardized_residual_pool(
            scenario_seed, size=max(31, prior_count)
        )[:prior_count]
        residuals = np.vstack((residuals, prior))
        sigmas = np.concatenate((sigmas, np.ones(len(prior), dtype=float)))
        empirical_weight = float(
            np.clip(effective_residuals / MIN_EFFECTIVE_OOS_RESIDUALS, 0.0, 1.0)
        )
        residual_sampling_groups = (
            (0, empirical_count, empirical_weight),
            (empirical_count, len(residuals), 1.0 - empirical_weight),
        )
        residual_pool_metadata = {
            **dict(residual_pool_metadata),
            "source": "FROZEN_OOS_WITH_VERSIONED_CONSERVATIVE_PRIOR",
            "status": "SHRUNK_INSUFFICIENT_OOS_RESIDUALS",
            "prior_size": len(prior),
            "empirical_weight": empirical_weight,
            "prior_weight": 1.0 - empirical_weight,
            "shrinkage_threshold": MIN_EFFECTIVE_OOS_RESIDUALS,
        }
    if random_plan is None:
        plan = ScenarioRandomPlan.build(
            seed=scenario_seed,
            scenario_count=scenario_count,
            residual_pool_size=len(residuals),
            empirical_pool_size=len(history),
            residual_groups=residual_sampling_groups,
        )
    else:
        plan = random_plan
        if (
            plan.scenario_count != scenario_count
            or plan.residual_pool_size != len(residuals)
            or plan.empirical_pool_size != len(history)
        ):
            raise ValueError("frozen random plan does not match scenario pools")
    if residual_sampling_groups is not None:
        empirical_draw_count = (
            0
            if empirical_path_count == 0
            else sum(
                draw_count
                for start, stop, _target, draw_count in plan.residual_groups
                if start == 0 and stop == empirical_path_count
            )
        )
        actual_empirical_weight = empirical_draw_count / scenario_count
        residual_pool_metadata = {
            **dict(residual_pool_metadata),
            "empirical_path_count": empirical_path_count,
            "prior_path_count": len(residuals) - empirical_path_count,
            "empirical_draw_count": empirical_draw_count,
            "prior_draw_count": scenario_count - empirical_draw_count,
            "target_empirical_weight": float(
                residual_pool_metadata.get("empirical_weight", 0.0)
            ),
            "empirical_weight": actual_empirical_weight,
            "prior_weight": 1.0 - actual_empirical_weight,
            "allocation_semantics": "EXACT_STRATIFIED_FROZEN_DRAW_COUNTS",
        }
    cumulative = generate_return_scenarios(
        path_forecast.expected_path,
        residuals,
        sigmas,
        path_forecast.forecast_sigma,
        plan,
    )
    if np.any(~np.isfinite(cumulative)) or np.any(cumulative <= -1.0):
        raise ValueError("residual scenarios imply non-positive prices")

    historical_closes = np.asarray([float(row["close"]) for row in history])
    historical_opens = np.asarray([float(row["open"]) for row in history])
    historical_highs = np.asarray([float(row["high"]) for row in history])
    historical_lows = np.asarray([float(row["low"]) for row in history])
    historical_volumes = np.asarray(
        [max(float(row.get("volume") or 0.0), 1.0) for row in history]
    )
    start_price = float(snapshot.daily_sequence[-1]["close"])
    closes = start_price * (1.0 + cumulative)
    previous_closes = np.column_stack(
        (np.full(scenario_count, start_price), closes[:, :-1])
    )
    gap_history = np.log(
        np.maximum(historical_opens, 1e-9) / np.maximum(historical_closes, 1e-9)
    )
    upper_history = np.maximum(
        historical_highs - np.maximum(historical_opens, historical_closes), 0.0
    ) / np.maximum(historical_closes, 1e-9)
    lower_history = np.maximum(
        np.minimum(historical_opens, historical_closes) - historical_lows, 0.0
    ) / np.maximum(historical_closes, 1e-9)
    volume_change = np.diff(
        np.log(np.concatenate(([historical_volumes[0]], historical_volumes)))
    )
    gap_indices = np.asarray(plan.gap_indices).reshape(scenario_count, 13)
    upper_indices = np.asarray(plan.upper_wick_indices).reshape(scenario_count, 13)
    lower_indices = np.asarray(plan.lower_wick_indices).reshape(scenario_count, 13)
    volume_indices = np.asarray(plan.volume_indices).reshape(scenario_count, 13)
    opens = previous_closes * np.exp(gap_history[gap_indices])
    highs = np.maximum(opens, closes) * (1.0 + upper_history[upper_indices])
    lows = np.minimum(opens, closes) * np.maximum(1.0 - lower_history[lower_indices], 1e-4)
    volumes = historical_volumes[-1] * np.exp(
        np.cumsum(np.clip(volume_change[volume_indices], -2.0, 2.0), axis=1)
    )
    if (
        np.any(lows <= 0.0)
        or np.any(lows > opens)
        or np.any(lows > closes)
        or np.any(highs < opens)
        or np.any(highs < closes)
    ):
        raise ValueError("V3.4.1 generated illegal OHLC")

    p10 = np.quantile(closes, 0.10, axis=0)
    p50 = np.quantile(closes, 0.50, axis=0)
    p90 = np.quantile(closes, 0.90, axis=0)
    if np.any(p10 > p50) or np.any(p50 > p90):
        raise ValueError("P10/P50/P90 crossing detected")
    close_scale = np.maximum(np.std(closes, axis=0), 1e-9)
    representative_index = int(
        np.argmin(np.sum(((closes - p50) / close_scale) ** 2, axis=1))
    )
    dif_all, dea_all, macd_all = _macd_future(historical_closes, closes)
    week_ends = tuple(
        snapshot.cutoff_date + timedelta(days=7 * step) for step in range(1, 14)
    )
    representative: list[Mapping[str, object]] = []
    indicators: list[Mapping[str, object]] = []
    quantiles: list[Mapping[str, object]] = []
    (
        previous_dif,
        previous_slope,
        historical_dif_tail,
        historical_dea_tail,
        historical_macd_tail,
    ) = _historical_dif_tail(historical_closes)
    for index in range(13):
        slope = float(dif_all[representative_index, index] - previous_dif)
        acceleration = float(slope - previous_slope)
        representative.append(
            {
                "week": index + 1,
                "week_start": (week_ends[index] - timedelta(days=6)).isoformat(),
                "week_end": week_ends[index].isoformat(),
                "open": round(float(opens[representative_index, index]), 6),
                "high": round(float(highs[representative_index, index]), 6),
                "low": round(float(lows[representative_index, index]), 6),
                "close": round(float(closes[representative_index, index]), 6),
                "volume_p10": round(float(np.quantile(volumes[:, index], 0.10)), 2),
                "volume_p50": round(float(np.quantile(volumes[:, index], 0.50)), 2),
                "volume_p90": round(float(np.quantile(volumes[:, index], 0.90)), 2),
            }
        )
        indicators.append(
            {
                "week": index + 1,
                "week_end": week_ends[index].isoformat(),
                "dif": round(float(dif_all[representative_index, index]), 8),
                "dea": round(float(dea_all[representative_index, index]), 8),
                "macd": round(float(macd_all[representative_index, index]), 8),
                "dif_first_change": round(slope, 8),
                "dif_second_change": round(acceleration, 8),
                "trend_state": _trend_state(
                    np.concatenate(
                        (
                            historical_closes[-len(historical_dif_tail) :],
                            closes[representative_index, : index + 1],
                        )
                    ),
                    np.concatenate(
                        (historical_dif_tail, dif_all[representative_index, : index + 1])
                    ),
                    np.concatenate(
                        (
                            historical_dea_tail,
                            dea_all[representative_index, : index + 1],
                        )
                    ),
                    np.concatenate(
                        (
                            historical_macd_tail,
                            macd_all[representative_index, : index + 1],
                        )
                    ),
                ),
            }
        )
        quantiles.append(
            {
                "week": index + 1,
                "week_end": week_ends[index].isoformat(),
                "close_p10": round(float(p10[index]), 6),
                "close_p50": round(float(p50[index]), 6),
                "close_p90": round(float(p90[index]), 6),
                "low_p10": round(float(np.quantile(lows[:, index], 0.10)), 6),
                "high_p90": round(float(np.quantile(highs[:, index], 0.90)), 6),
            }
        )
        previous_dif = float(dif_all[representative_index, index])
        previous_slope = slope

    thresholds = {
        horizon: {
            "threshold": max(
                0.01, path_forecast.forecast_sigma * math.sqrt(horizon) * 0.50
            ),
            "sigma_week_at_forecast": path_forecast.forecast_sigma,
            "formula_version": THRESHOLD_FORMULA_VERSION,
        }
        for horizon in (4, 8, 13)
    }
    raw = horizon_probabilities(
        cumulative, {horizon: float(values["threshold"]) for horizon, values in thresholds.items()}
    )
    combined: dict[int, Mapping[str, Mapping[str, float]]] = {}
    calibrator_versions: dict[int, str | None] = {}
    calibration_statuses: list[str] = []
    for horizon in (4, 8, 13):
        raw_vector = np.asarray(
            [raw[horizon]["up"], raw[horizon]["sideways"], raw[horizon]["down"]]
        )
        calibrator = dict(calibrators.get(horizon, {}))
        temperature = float(calibrator.get("temperature", 1.0))
        calibrated = apply_temperature(raw_vector, temperature)
        combined[horizon] = {
            "raw": dict(raw[horizon]),
            "calibrated": {
                "up": float(calibrated[0]),
                "sideways": float(calibrated[1]),
                "down": float(calibrated[2]),
            },
        }
        calibrator_versions[horizon] = (
            None if calibrator.get("version") is None else str(calibrator["version"])
        )
        calibration_statuses.append(str(calibrator.get("status", "SHRINKAGE_ONLY")))

    terminal_threshold = float(thresholds[13]["threshold"])
    path_labels = []
    for row in cumulative:
        terminal = float(row[-1])
        amplitude = float(np.max(row) - np.min(row))
        if abs(terminal) <= terminal_threshold and amplitude <= 3 * terminal_threshold:
            path_labels.append("RANGE")
        elif int(np.argmin(row)) < 6 and terminal - float(np.min(row)) > 2 * terminal_threshold:
            path_labels.append("V_SHAPE")
        elif int(np.argmax(row)) < 6 and float(np.max(row)) - terminal > 2 * terminal_threshold:
            path_labels.append("INVERTED_V")
        else:
            path_labels.append("TREND_UP" if terminal > 0 else "TREND_DOWN")
    path_probabilities = _probability_dict(
        path_labels, ("TREND_UP", "TREND_DOWN", "V_SHAPE", "INVERTED_V", "RANGE")
    )
    health_status = (
        "MODEL_OUT_OF_DISTRIBUTION"
        if float(path_forecast.ood_diagnostics.get("ood_score", 0.0)) >= 0.5
        else "MODEL_NORMAL"
    )
    calibration_status = (
        "FORMALLY_CALIBRATED"
        if calibration_statuses and all(value == "FORMALLY_CALIBRATED" for value in calibration_statuses)
        else "PRELIMINARY"
        if any(value == "PRELIMINARY" for value in calibration_statuses)
        else "SHRINKAGE_ONLY"
    )
    consistency = indicator_path_consistency(path_forecast, snapshot)
    reliability = model_reliability(
        effective_samples=float(residual_pool_metadata.get("effective_residual_count", 0)),
        calibration_status=calibration_status,
        brier=float(residual_pool_metadata.get("recent_brier", 0.45)),
        wis=float(residual_pool_metadata.get("recent_wis", 1.0)),
        coverage=float(residual_pool_metadata.get("recent_coverage", 0.80)),
        recent_score=float(residual_pool_metadata.get("recent_score", 0.50)),
        ood_score=float(path_forecast.ood_diagnostics.get("ood_score", 0.0)),
        consistent=bool(consistency["consistent"]),
        health_status=health_status,
    )
    audit = {
        "scenario_formula": (
            "expm1(log1p(CurrentPrediction) + "
            "StandardizedLogGrossOOSResidual * CurrentSigma)"
        ),
        "residual_definition": "log1p(Actual) - log1p(FrozenPrediction)",
        "residual_schema_version": RESIDUAL_SCHEMA_VERSION,
        "scenario_adapter_version": SCENARIO_ADAPTER_VERSION,
        "scenario_count": scenario_count,
        "random_plan_version": plan.version,
        "random_plan_hash": plan.plan_hash,
        "representative_path_semantics": "one complete scenario medoid",
        "p10_p50_p90_semantics": "marginal weekly close quantiles, not confidence",
        "ohlc_legal_rate": 1.0,
        "ema_initial_state_source": "pre_forecast_real_weekly_close_history",
        "indicator_consistency": consistency,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "market": snapshot.market,
        "anchor": snapshot.cutoff_date.isoformat(),
        "model_version": path_forecast.model_version,
        "scenario_seed": scenario_seed,
        "expected_path": list(path_forecast.expected_path),
        "representative_ohlcv": representative,
        "indicators": indicators,
        "price_quantiles": quantiles,
        "horizon_probabilities": combined,
        "thresholds": thresholds,
        "calibrator_versions": calibrator_versions,
        "residual_pool": dict(residual_pool_metadata),
        "path_probabilities": path_probabilities,
        "reliability": reliability,
        "health_status": health_status,
        "scenario_audit": audit,
    }
    payload_hash = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()
    return V341ScenarioForecast(
        market=snapshot.market,
        anchor_date=snapshot.cutoff_date,
        model_version=path_forecast.model_version,
        scenario_seed=scenario_seed,
        scenario_count=scenario_count,
        random_plan=plan,
        expected_path=path_forecast.expected_path,
        representative_ohlcv=tuple(representative),
        indicators=tuple(indicators),
        price_quantiles=tuple(quantiles),
        horizon_probabilities=combined,
        thresholds=thresholds,
        calibrator_versions=calibrator_versions,
        residual_pool=dict(residual_pool_metadata),
        path_probabilities=path_probabilities,
        reliability=reliability,
        health_status=health_status,
        scenario_audit=audit,
        payload_hash=payload_hash,
    )


def generate_return_scenarios(
    current_prediction: Sequence[float],
    standardized_residuals: np.ndarray,
    source_sigmas: Sequence[float],
    current_sigma: float,
    random_plan: ScenarioRandomPlan,
) -> np.ndarray:
    """Apply OOS residuals in log-gross-return space and return simple returns.

    The V3.4.1 arithmetic adapter could produce cumulative returns below -100%,
    which makes a price path non-positive.  V3.4.4 keeps the same frozen draw
    plan and scenario count, but performs the residual perturbation in the
    natural positive-price domain instead of clipping or discarding tail draws.
    """

    prediction = np.asarray(current_prediction, dtype=float)
    residuals = np.asarray(standardized_residuals, dtype=float)
    source = np.asarray(source_sigmas, dtype=float)
    if prediction.shape != (13,):
        raise ValueError("current prediction must contain 13 cumulative returns")
    if np.any(~np.isfinite(prediction)) or np.any(prediction <= -1.0):
        raise ValueError("current prediction must imply positive gross returns")
    if residuals.ndim != 2 or residuals.shape[1] != 13 or not len(residuals):
        raise ValueError("residual pool must be non-empty and N x 13")
    if np.any(~np.isfinite(residuals)):
        raise ValueError("residual pool must contain only finite values")
    if source.shape != (len(residuals),) or np.any(source <= 0.0):
        raise ValueError("each residual needs a positive source sigma")
    if not math.isfinite(current_sigma) or current_sigma <= 0.0:
        raise ValueError("current sigma must be finite and positive")
    indices = np.asarray(random_plan.residual_indices, dtype=int)
    if len(indices) != random_plan.scenario_count or np.any(indices >= len(residuals)):
        raise ValueError("random plan is incompatible with the residual pool")
    # `residuals` are already
    # `(log1p(Actual)-log1p(Prediction))/source_sigma`; the source sigma is
    # retained and validated for audit but must not be applied twice.
    log_gross = np.log1p(prediction) + residuals[indices] * float(current_sigma)
    scenarios = np.expm1(log_gross)
    if np.any(~np.isfinite(scenarios)) or np.any(scenarios <= -1.0):
        raise ValueError("log-gross residual scenarios must imply positive prices")
    return scenarios


def horizon_probabilities(
    cumulative_scenarios: np.ndarray,
    thresholds: Mapping[int, float],
) -> dict[int, dict[str, float]]:
    scenarios = np.asarray(cumulative_scenarios, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != 13 or not len(scenarios):
        raise ValueError("cumulative scenarios must be N x 13")
    result: dict[int, dict[str, float]] = {}
    for horizon in (4, 8, 13):
        threshold = float(thresholds[horizon])
        if threshold <= 0.0:
            raise ValueError("direction thresholds must be positive")
        values = scenarios[:, horizon - 1]
        counts = np.asarray(
            [
                np.sum(values > threshold),
                np.sum((values >= -threshold) & (values <= threshold)),
                np.sum(values < -threshold),
            ],
            dtype=float,
        )
        probabilities = counts / float(np.sum(counts))
        result[horizon] = {
            "up": float(probabilities[0]),
            "sideways": float(probabilities[1]),
            "down": float(probabilities[2]),
        }
    return result


def apply_temperature(probabilities: Sequence[float], temperature: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (3,) or np.any(values < 0.0) or not np.isclose(np.sum(values), 1.0):
        raise ValueError("temperature calibration requires three normalized probabilities")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(values, 1e-9, 1.0)) / float(temperature)
    logits -= np.max(logits)
    transformed = np.exp(logits)
    return transformed / np.sum(transformed)


def fit_temperature(raw_probabilities: np.ndarray, labels: Sequence[int]) -> float:
    raw = np.asarray(raw_probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) != len(truth):
        raise ValueError("raw probabilities and labels must align")
    if len(raw) < 30 or np.any((truth < 0) | (truth > 2)):
        raise ValueError("temperature fitting requires at least 30 valid samples")
    if np.any(raw < 0.0) or np.any(~np.isclose(np.sum(raw, axis=1), 1.0, atol=1e-9)):
        raise ValueError("each probability row must be normalized")
    candidates = np.linspace(0.25, 4.0, 751)
    losses = []
    for temperature in candidates:
        calibrated = np.asarray(
            [apply_temperature(row, float(temperature)) for row in raw]
        )
        losses.append(
            -float(np.mean(np.log(np.clip(calibrated[np.arange(len(truth)), truth], 1e-12, 1.0))))
        )
    return float(candidates[int(np.argmin(losses))])


def model_reliability(
    *,
    effective_samples: float,
    calibration_status: str,
    brier: float,
    wis: float,
    coverage: float,
    recent_score: float,
    ood_score: float,
    consistent: bool,
    health_status: str,
) -> dict[str, object]:
    sample_component = float(np.clip(effective_samples / 50.0, 0.0, 1.0))
    calibration_component = float(np.clip(1.0 - brier / (2.0 / 3.0), 0.0, 1.0))
    interval_component = float(
        np.clip(0.5 * (1.0 - min(wis, 2.0) / 2.0) + 0.5 * (1.0 - abs(coverage - 0.80) / 0.80), 0.0, 1.0)
    )
    recent_component = float(np.clip(recent_score, 0.0, 1.0))
    ood_component = float(np.clip(1.0 - ood_score, 0.0, 1.0))
    consistency_component = 1.0 if consistent else 0.25
    components = {
        "sample_sufficiency": sample_component,
        "probability_calibration": calibration_component,
        "interval_quality": interval_component,
        "recent_out_of_sample": recent_component,
        "in_distribution": ood_component,
        "indicator_consistency": consistency_component,
    }
    score = 100.0 * (
        0.20 * sample_component
        + 0.20 * calibration_component
        + 0.20 * interval_component
        + 0.15 * recent_component
        + 0.15 * ood_component
        + 0.10 * consistency_component
    )
    caps: list[float] = []
    if effective_samples < 30:
        caps.append(40.0)
    if calibration_status != "FORMALLY_CALIBRATED":
        caps.append(45.0)
    if ood_score >= 0.5:
        caps.append(50.0)
    if health_status in {"MODEL_DEGRADED", "MODEL_OUT_OF_DISTRIBUTION"}:
        caps.append(50.0)
    if caps:
        score = min(score, min(caps))
    if not consistent:
        score = max(0.0, score - 15.0)
    return {
        "version": RELIABILITY_VERSION,
        "score": round(float(np.clip(score, 0.0, 100.0)), 1),
        "components": components,
        "caps": caps,
        "semantics": "model_reliability_not_prediction_probability",
    }
