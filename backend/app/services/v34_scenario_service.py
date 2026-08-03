"""Deterministic 13-week continuous OHLCV scenario adapter for V3.4."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from backend.app.services.v33_feature_service import FeatureSnapshot
from backend.app.services.v34_training_service import (
    HORIZON_WEEKS,
    V34ModelState,
    V34PathForecast,
    V34TrainingSample,
)


SCENARIO_COUNT = 1200
SCENARIO_ADAPTER_VERSION = "V3.4_13W_SCENARIO_1"
PATH_TYPES = ("TREND_UP", "TREND_DOWN", "V_SHAPE", "INVERTED_V", "RANGE")


class V34ScenarioError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class V34ScenarioForecast:
    market: str
    anchor_date: date
    model_version: str
    scenario_adapter_version: str
    scenario_seed: int
    scenario_count: int
    historical_ohlcv: tuple[Mapping[str, Any], ...]
    representative_ohlcv: tuple[Mapping[str, Any], ...]
    indicators: tuple[Mapping[str, Any], ...]
    price_quantiles: tuple[Mapping[str, Any], ...]
    direction_probabilities: Mapping[str, float]
    path_probabilities: Mapping[str, float]
    confidence: float
    calibration: Mapping[str, Any]
    turning_windows: Mapping[str, Any]
    consistency: Mapping[str, Any]
    scenario_audit: Mapping[str, Any]
    payload_hash: str


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _future_week_ends(anchor: date) -> tuple[date, ...]:
    # Forecast timestamps express weekly windows, not guaranteed single-day turns.
    return tuple(anchor + timedelta(days=7 * step) for step in range(1, 14))


def _ema_state(values: Sequence[float], period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    state = float(values[0])
    for value in values[1:]:
        state = alpha * float(value) + (1.0 - alpha) * state
    return state


def _macd_future(
    historical_closes: Sequence[float], future_closes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(historical_closes) < 35:
        raise V34ScenarioError("EMA state requires at least 35 pre-forecast weekly closes")
    ema12 = _ema_state(historical_closes, 12)
    ema26 = _ema_state(historical_closes, 26)
    historical_dif: list[float] = []
    a12 = 2.0 / 13.0
    a26 = 2.0 / 27.0
    a9 = 2.0 / 10.0
    state12 = float(historical_closes[0])
    state26 = float(historical_closes[0])
    dea = 0.0
    for close in historical_closes:
        state12 = a12 * close + (1.0 - a12) * state12
        state26 = a26 * close + (1.0 - a26) * state26
        dif = state12 - state26
        dea = a9 * dif + (1.0 - a9) * dea
        historical_dif.append(dif)
    ema12 = np.full(future_closes.shape[0], ema12, dtype=float)
    ema26 = np.full(future_closes.shape[0], ema26, dtype=float)
    dea_values = np.full(future_closes.shape[0], dea, dtype=float)
    dif_output = np.empty_like(future_closes)
    dea_output = np.empty_like(future_closes)
    histogram = np.empty_like(future_closes)
    for week in range(HORIZON_WEEKS):
        close = future_closes[:, week]
        ema12 = a12 * close + (1.0 - a12) * ema12
        ema26 = a26 * close + (1.0 - a26) * ema26
        dif = ema12 - ema26
        dea_values = a9 * dif + (1.0 - a9) * dea_values
        dif_output[:, week] = dif
        dea_output[:, week] = dea_values
        histogram[:, week] = 2.0 * (dif - dea_values)
    return dif_output, dea_output, histogram


def _trend_state(
    closes: Sequence[float], dif: Sequence[float], dea: Sequence[float], histogram: Sequence[float]
) -> str:
    d1 = np.diff(np.asarray(dif, dtype=float))
    if len(d1) == 0:
        return "INSUFFICIENT"
    d2 = np.diff(d1)
    if d1[-1] < 0.0 and len(d2) and d2[-1] > 0.0:
        return "POTENTIAL_BOTTOMING"
    if len(d1) >= 2 and d1[-2] < 0.0 <= d1[-1] and (not len(d2) or d2[-1] > 0.0):
        return "UPWARD_TURN"
    positive = 0
    for value in reversed(d1):
        if value > 0.0:
            positive += 1
        else:
            break
    if positive >= 3:
        higher_dif = dif[-1] > min(dif[-3:])
        dea_stronger = len(dea) >= 2 and dea[-1] > dea[-2]
        histogram_better = len(histogram) >= 2 and histogram[-1] > histogram[-2]
        price_confirmed = len(closes) >= 3 and closes[-1] > min(closes[-3:])
        if higher_dif and dea_stronger and histogram_better and price_confirmed:
            return "UPTREND_CONFIRMED"
        return "UPTREND_EARLY_CONFIRMED"
    if d1[-1] > 0.0:
        return "UPWARD_TURN_UNCONFIRMED"
    return "WEAK_OR_FALLING"


def _window_from_weeks(weeks: np.ndarray, probability_mass: float = 0.60) -> dict[str, Any]:
    values, counts = np.unique(weeks.astype(int), return_counts=True)
    order = np.argsort(counts)[::-1]
    selected: list[int] = []
    covered = 0
    target = int(math.ceil(len(weeks) * probability_mass))
    for index in order:
        selected.append(int(values[index]))
        covered += int(counts[index])
        if covered >= target:
            break
    return {
        "start_week": min(selected),
        "end_week": max(selected),
        "coverage_probability": round(100.0 * covered / len(weeks), 2),
    }


def _exclusive_path_type(path: np.ndarray, sideways_threshold: float) -> str:
    terminal = float(path[-1])
    low = int(np.argmin(path))
    high = int(np.argmax(path))
    amplitude = float(np.max(path) - np.min(path))
    if abs(terminal) <= sideways_threshold and amplitude <= 3.0 * sideways_threshold:
        return "RANGE"
    if low < len(path) // 2 and terminal - float(path[low]) > 2.0 * sideways_threshold:
        return "V_SHAPE"
    if high < len(path) // 2 and float(path[high]) - terminal > 2.0 * sideways_threshold:
        return "INVERTED_V"
    return "TREND_UP" if terminal > 0.0 else "TREND_DOWN"


def _probabilities(labels: Sequence[str], categories: Sequence[str]) -> dict[str, float]:
    # A one-count symmetric prior prevents misleading exact 0%/100% display.
    counts = np.asarray([sum(label == category for label in labels) + 1 for category in categories])
    values = 100.0 * counts / float(np.sum(counts))
    rounded = [round(float(value), 2) for value in values[:-1]]
    rounded.append(round(100.0 - sum(rounded), 2))
    return dict(zip(categories, rounded))


class V34ScenarioService:
    def __init__(self, *, scenario_count: int = SCENARIO_COUNT) -> None:
        if scenario_count < 1000:
            raise V34ScenarioError("V3.4 requires at least 1000 complete scenarios")
        self.scenario_count = scenario_count

    @staticmethod
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

    def generate(
        self,
        state: V34ModelState,
        path_forecast: V34PathForecast,
        snapshot: FeatureSnapshot,
        matured: Sequence[V34TrainingSample],
        *,
        seed: int | None = None,
    ) -> V34ScenarioForecast:
        if state.market != snapshot.market or path_forecast.market != snapshot.market:
            raise V34ScenarioError("scenario inputs cannot cross market chains")
        scenario_seed = (
            int(seed)
            if seed is not None
            else int.from_bytes(
                hashlib.sha256(
                    f"{snapshot.market}:{snapshot.cutoff_date}:V3.4".encode("utf-8")
                ).digest()[:4],
                "big",
            )
        )
        rng = np.random.default_rng(scenario_seed)
        history = self._historical_weekly_rows(matured, snapshot.cutoff_date)
        if len(history) < 35:
            raise V34ScenarioError("scenario adapter requires at least 35 historical weekly OHLCV rows")
        historical_closes = np.asarray([float(row["close"]) for row in history], dtype=float)
        historical_opens = np.asarray([float(row["open"]) for row in history], dtype=float)
        historical_highs = np.asarray([float(row["high"]) for row in history], dtype=float)
        historical_lows = np.asarray([float(row["low"]) for row in history], dtype=float)
        historical_volumes = np.asarray(
            [float(row["volume"] or 0.0) for row in history], dtype=float
        )
        historical_returns = np.diff(np.log(historical_closes))
        residual_paths = np.asarray(path_forecast.residual_paths, dtype=float)
        if residual_paths.ndim != 2 or residual_paths.shape[1] != 13:
            residual_paths = np.zeros((1, 13), dtype=float)
        residual_increments = np.diff(
            np.column_stack((np.zeros(residual_paths.shape[0]), residual_paths)), axis=1
        )
        expected_cumulative = np.asarray(path_forecast.expected_path, dtype=float)
        expected_log = np.log(np.maximum(1.0 + expected_cumulative, 1e-6))
        expected_increment = np.diff(np.concatenate(([0.0], expected_log)))
        selected = rng.integers(0, residual_increments.shape[0], self.scenario_count)
        innovations = residual_increments[selected].copy()
        phi = float(np.clip(np.corrcoef(historical_returns[:-1], historical_returns[1:])[0, 1], -0.45, 0.45))
        if not math.isfinite(phi):
            phi = 0.0
        for week in range(1, HORIZON_WEEKS):
            innovations[:, week] += phi * innovations[:, week - 1]
        historical_sigma = max(float(np.std(historical_returns[-104:])), 0.005)
        current_sigma = np.std(innovations, axis=0)
        innovations *= historical_sigma / np.maximum(current_sigma, historical_sigma * 0.35)
        weekly_log_returns = expected_increment + 0.70 * innovations

        gap_history = np.log(
            np.maximum(historical_opens[1:], 1e-9) / np.maximum(historical_closes[:-1], 1e-9)
        )
        gap_limit = max(float(np.quantile(np.abs(gap_history), 0.995)), 0.002)
        gaps = np.clip(
            rng.choice(gap_history, size=(self.scenario_count, 13), replace=True),
            -gap_limit,
            gap_limit,
        )
        upper_history = np.maximum(
            historical_highs - np.maximum(historical_opens, historical_closes), 0.0
        ) / np.maximum(historical_closes, 1e-9)
        lower_history = np.maximum(
            np.minimum(historical_opens, historical_closes) - historical_lows, 0.0
        ) / np.maximum(historical_closes, 1e-9)
        upper = rng.choice(upper_history, size=(self.scenario_count, 13), replace=True)
        lower = rng.choice(lower_history, size=(self.scenario_count, 13), replace=True)

        open_values = np.empty((self.scenario_count, 13), dtype=float)
        close_values = np.empty_like(open_values)
        high_values = np.empty_like(open_values)
        low_values = np.empty_like(open_values)
        previous = np.full(self.scenario_count, float(snapshot.daily_sequence[-1]["close"]))
        for week in range(HORIZON_WEEKS):
            open_values[:, week] = previous * np.exp(gaps[:, week])
            close_values[:, week] = previous * np.exp(weekly_log_returns[:, week])
            high_values[:, week] = np.maximum(open_values[:, week], close_values[:, week]) * (
                1.0 + upper[:, week]
            )
            low_values[:, week] = np.minimum(open_values[:, week], close_values[:, week]) * np.maximum(
                1.0 - lower[:, week], 1e-4
            )
            previous = close_values[:, week]

        positive_volume = np.maximum(historical_volumes, 1.0)
        volume_changes = np.diff(np.log(positive_volume))
        volume_phi = 0.0
        if len(volume_changes) > 2:
            volume_phi = float(
                np.clip(np.corrcoef(volume_changes[:-1], volume_changes[1:])[0, 1], -0.5, 0.5)
            )
            if not math.isfinite(volume_phi):
                volume_phi = 0.0
        volume_increments = rng.choice(
            volume_changes, size=(self.scenario_count, 13), replace=True
        )
        for week in range(1, 13):
            volume_increments[:, week] += volume_phi * volume_increments[:, week - 1]
        volumes = np.empty_like(volume_increments)
        previous_volume = np.full(self.scenario_count, positive_volume[-1])
        for week in range(13):
            previous_volume = previous_volume * np.exp(np.clip(volume_increments[:, week], -2.0, 2.0))
            volumes[:, week] = previous_volume

        if (
            np.any(low_values <= 0.0)
            or np.any(low_values > open_values)
            or np.any(low_values > close_values)
            or np.any(high_values < open_values)
            or np.any(high_values < close_values)
        ):
            raise V34ScenarioError("generated scenarios violate OHLC positivity/range constraints")

        close_p10 = np.quantile(close_values, 0.10, axis=0)
        close_p50 = np.quantile(close_values, 0.50, axis=0)
        close_p90 = np.quantile(close_values, 0.90, axis=0)
        if np.any(close_p10 > close_p50) or np.any(close_p50 > close_p90):
            raise V34ScenarioError("P10/P50/P90 crossing detected")
        scale = np.maximum(np.std(close_values, axis=0), 1e-9)
        representative_index = int(
            np.argmin(np.sum(((close_values - close_p50) / scale) ** 2, axis=1))
        )
        dif_all, dea_all, macd_all = _macd_future(historical_closes, close_values)
        week_ends = _future_week_ends(snapshot.cutoff_date)
        representative: list[Mapping[str, Any]] = []
        indicators: list[Mapping[str, Any]] = []
        quantiles: list[Mapping[str, Any]] = []
        previous_dif = None
        previous_slope = None
        rep_closes = close_values[representative_index]
        rep_dif = dif_all[representative_index]
        rep_dea = dea_all[representative_index]
        rep_macd = macd_all[representative_index]
        for week in range(13):
            direction_threshold = max(historical_sigma * 0.35, 0.003)
            scenario_week_returns = weekly_log_returns[:, week]
            direction = _probabilities(
                [
                    "up" if value > direction_threshold else "down" if value < -direction_threshold else "sideways"
                    for value in scenario_week_returns
                ],
                ("up", "sideways", "down"),
            )
            representative.append(
                {
                    "week": week + 1,
                    "week_start": (week_ends[week] - timedelta(days=6)).isoformat(),
                    "week_end": week_ends[week].isoformat(),
                    "open": round(float(open_values[representative_index, week]), 6),
                    "high": round(float(high_values[representative_index, week]), 6),
                    "low": round(float(low_values[representative_index, week]), 6),
                    "close": round(float(rep_closes[week]), 6),
                    "volume_p50": round(float(np.quantile(volumes[:, week], 0.50)), 2),
                    "volume_p10": round(float(np.quantile(volumes[:, week], 0.10)), 2),
                    "volume_p90": round(float(np.quantile(volumes[:, week], 0.90)), 2),
                    "direction_probabilities": direction,
                }
            )
            slope = None if previous_dif is None else float(rep_dif[week] - previous_dif)
            acceleration = (
                None if slope is None or previous_slope is None else float(slope - previous_slope)
            )
            indicators.append(
                {
                    "week": week + 1,
                    "week_end": week_ends[week].isoformat(),
                    "dif": round(float(rep_dif[week]), 8),
                    "dea": round(float(rep_dea[week]), 8),
                    "macd": round(float(rep_macd[week]), 8),
                    "dif_first_change": None if slope is None else round(slope, 8),
                    "dif_second_change": None if acceleration is None else round(acceleration, 8),
                    "trend_state": _trend_state(
                        rep_closes[: week + 1], rep_dif[: week + 1], rep_dea[: week + 1], rep_macd[: week + 1]
                    ),
                }
            )
            previous_dif = float(rep_dif[week])
            previous_slope = slope
            quantiles.append(
                {
                    "week": week + 1,
                    "week_end": week_ends[week].isoformat(),
                    "close_p10": round(float(close_p10[week]), 6),
                    "close_p50": round(float(close_p50[week]), 6),
                    "close_p90": round(float(close_p90[week]), 6),
                    "low_quantile": round(float(np.quantile(low_values[:, week], 0.10)), 6),
                    "high_quantile": round(float(np.quantile(high_values[:, week], 0.90)), 6),
                }
            )

        start_price = float(snapshot.daily_sequence[-1]["close"])
        cumulative = close_values / start_price - 1.0
        sideways_threshold = max(0.02, historical_sigma * math.sqrt(13.0) * 0.25)
        direction_labels = [
            "up" if value > sideways_threshold else "down" if value < -sideways_threshold else "sideways"
            for value in cumulative[:, -1]
        ]
        direction_probabilities = _probabilities(direction_labels, ("up", "sideways", "down"))
        path_labels = [_exclusive_path_type(row, sideways_threshold) for row in cumulative]
        path_probabilities = _probabilities(path_labels, PATH_TYPES)
        low_weeks = np.argmin(close_values, axis=1) + 1
        high_weeks = np.argmax(close_values, axis=1) + 1
        dif_slopes = np.diff(dif_all, axis=1)
        dif_turn_weeks: list[int] = []
        for row in dif_slopes:
            turns = np.where((row[:-1] < 0.0) & (row[1:] >= 0.0))[0]
            dif_turn_weeks.append(int(turns[0] + 2) if turns.size else int(np.argmin(np.abs(row)) + 1))
        terminal_direction = max(direction_probabilities, key=direction_probabilities.get)
        price_weeks = low_weeks if terminal_direction == "up" else high_weeks
        price_window = _window_from_weeks(price_weeks)
        dif_window = _window_from_weeks(np.asarray(dif_turn_weeks))
        separation = max(
            0,
            max(price_window["start_week"], dif_window["start_week"])
            - min(price_window["end_week"], dif_window["end_week"]),
        )
        consistency_status = "CONSISTENT" if separation <= 2 else "FORECAST_INCONSISTENT"
        consistency_reasons: list[str] = []
        if consistency_status != "CONSISTENT":
            consistency_reasons.append(
                f"price and DIF turning windows are separated by {separation} weeks"
            )
        consistency = {
            "status": consistency_status,
            "price_dif_turning_tolerance_weeks": 2,
            "separation_weeks": separation,
            "reasons": consistency_reasons,
            "single_day_turn_suppressed": consistency_status != "CONSISTENT",
            "extreme_position_change_allowed": consistency_status == "CONSISTENT",
        }
        effective = state.effective_independent_sample_count
        calibration_status = (
            "NOT_SUFFICIENTLY_CALIBRATED"
            if effective < 30
            else "PRELIMINARY"
            if effective < 50
            else "FORMALLY_CALIBRATED"
        )
        sample_score = min(1.0, effective / 80.0)
        dispersion_score = max(
            0.0,
            1.0 - float(np.mean((close_p90 - close_p10) / np.maximum(close_p50, 1e-9))) / 0.35,
        )
        confidence = 100.0 * (0.55 * sample_score + 0.25 * dispersion_score + 0.20)
        if effective < 30:
            confidence = min(confidence, 60.0)
        if consistency_status != "CONSISTENT":
            confidence = min(confidence * 0.70, 50.0)
        confidence = round(float(np.clip(confidence, 5.0, 95.0)), 1)
        turning_windows = {
            "price_turning_window": price_window,
            "dif_derivative_zero_window": dif_window,
            "price_turn_kind": "bottom" if terminal_direction == "up" else "top",
            "dif_turn_kind": "bottom",
        }
        audit = {
            "scenario_count": self.scenario_count,
            "ohlc_legal_rate": 1.0,
            "positive_price_rate": 1.0,
            "continuous_process": True,
            "gap_distribution_source": f"{snapshot.market}_weekly_ohlcv_history",
            "gap_absolute_limit": gap_limit,
            "volume_model": "AR1_log_volume_change_self_history",
            "volume_confidence": "FORMALLY_CALIBRATED" if len(volume_changes) >= 50 else "LOW_SAMPLE",
            "ema_initial_state_source": "pre_forecast_real_weekly_close_history",
            "daily_correction_weeks": 4,
            "p10_p50_p90_semantics": "marginal weekly close quantiles, not confidence",
            "representative_path_semantics": "one actual scenario nearest the weekly close medians",
        }
        payload = {
            "market": snapshot.market,
            "anchor_date": snapshot.cutoff_date.isoformat(),
            "model_version": state.version,
            "scenario_adapter_version": SCENARIO_ADAPTER_VERSION,
            "scenario_seed": scenario_seed,
            "historical_ohlcv": [dict(row) for row in history[-26:]],
            "representative_ohlcv": representative,
            "indicators": indicators,
            "price_quantiles": quantiles,
            "direction_probabilities": direction_probabilities,
            "path_probabilities": path_probabilities,
            "confidence": confidence,
            "calibration": {
                "status": calibration_status,
                "raw_matured_sample_count": state.raw_matured_sample_count,
                "effective_independent_sample_count": round(effective, 3),
            },
            "turning_windows": turning_windows,
            "consistency": consistency,
            "scenario_audit": audit,
        }
        return V34ScenarioForecast(
            market=snapshot.market,
            anchor_date=snapshot.cutoff_date,
            model_version=state.version,
            scenario_adapter_version=SCENARIO_ADAPTER_VERSION,
            scenario_seed=scenario_seed,
            scenario_count=self.scenario_count,
            historical_ohlcv=tuple(dict(row) for row in history[-26:]),
            representative_ohlcv=tuple(representative),
            indicators=tuple(indicators),
            price_quantiles=tuple(quantiles),
            direction_probabilities=direction_probabilities,
            path_probabilities=path_probabilities,
            confidence=confidence,
            calibration=payload["calibration"],
            turning_windows=turning_windows,
            consistency=consistency,
            scenario_audit=audit,
            payload_hash=_payload_hash(payload),
        )
