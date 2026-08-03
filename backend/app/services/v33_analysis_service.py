"""Pure V3.3 inference, dated execution windows and position advice.

Training and analysis are intentionally separate.  ``run`` loads an immutable
Champion and calls :meth:`V33TrainingService.predict`; it never calls a train
method or persists optimizer/model state.  A web/repository adapter may store
the returned audit payload as an analysis run after inference.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import json
import math
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

import numpy as np

from .market_calendar import CalendarProvider
from .v33_feature_service import FeatureSnapshot, HORIZON_WEEKS, SUPPORTED_MARKETS
from .v33_training_service import (
    DAILY_CONFIDENCE_ADJUSTMENT_LIMIT,
    DAILY_CORRECTION_DECAY,
    DAILY_CORRECTION_WEEKS,
    DAILY_PATH_CORRECTION_MAX_ABS,
    DAILY_POSITION_ADJUSTMENT_LIMIT,
    FUND_ETF_RATIOS,
    HIGH_CONFIDENCE_GATE,
    V33Forecast,
    V33ModelState,
    V33TrainingError,
    V33TrainingPoint,
    V33TrainingService,
    _snap5,
)


class V33ChampionProvider(Protocol):
    def latest_state(self, market: str) -> V33ModelState | None: ...


class V33AnalysisError(RuntimeError):
    """Analysis cannot produce an auditable recommendation."""


class V33AnalysisService:
    """Use the latest V3.3 Champion without any training side effect.

    Public integration API:

    * ``run(market, snapshot, current_position, calibration_points=())``
    * ``analyze(snapshot, state, current_position, calibration_points=())``

    Suggested web adapters expose ``POST /api/v33/analysis/{market}`` and
    persist this returned JSON as ``latest``/``runs/{run_id}`` without invoking
    the training endpoint.
    """

    def __init__(
        self,
        *,
        training: V33TrainingService,
        champions: V33ChampionProvider,
        calendar: CalendarProvider,
    ) -> None:
        self.training = training
        self.champions = champions
        self.calendar = calendar

    def run(
        self,
        market: str,
        snapshot: FeatureSnapshot,
        *,
        current_position: int = 0,
        calibration_points: Sequence[V33TrainingPoint] = (),
    ) -> dict[str, Any]:
        if market not in SUPPORTED_MARKETS or snapshot.market != market:
            raise V33AnalysisError("V3.3 analysis supports independent 399006/159941 targets only")
        state = self.champions.latest_state(market)
        if state is None:
            raise V33AnalysisError(f"{market} has no trained V3.3 Champion")
        return self.analyze(
            snapshot,
            state,
            current_position=current_position,
            calibration_points=calibration_points,
        )

    def analyze(
        self,
        snapshot: FeatureSnapshot,
        state: V33ModelState,
        *,
        current_position: int = 0,
        calibration_points: Sequence[V33TrainingPoint] = (),
    ) -> dict[str, Any]:
        if current_position != _snap5(current_position):
            raise V33AnalysisError("current position must be on the 5-percentage-point grid")
        if state.market != snapshot.market:
            raise V33AnalysisError("Champion/target market mismatch")
        state_hash_before = state.state_hash
        iteration_before = state.iteration_number
        forecast = self.training.predict(
            state,
            snapshot,
            calibration_points=calibration_points,
        )
        future_sessions = self._future_sessions(snapshot.market, snapshot.daily_as_of)
        weekly_target, policy = self._target_position(
            snapshot, forecast, current_position
        )
        daily = self._daily_adjustments(forecast)
        target = _snap5(weekly_target + daily["position_adjustment"])
        if forecast.confidence >= HIGH_CONFIDENCE_GATE:
            weekly_delta = weekly_target - current_position
            if weekly_delta > 0 and target < current_position:
                target = current_position
            elif weekly_delta < 0 and target > current_position:
                target = current_position
        applied_daily_position = target - weekly_target
        adjusted_confidence = max(
            0.0,
            min(100.0, forecast.confidence + daily["confidence_adjustment"]),
        )
        policy = [
            *policy,
            (
                "100-session daily correction: "
                f"{applied_daily_position:+d} position points, "
                f"{daily['confidence_adjustment']:+.2f} confidence points"
            ),
        ]
        advice = self._advice(
            snapshot,
            forecast,
            current_position,
            target,
            future_sessions,
            policy,
        )
        if state.state_hash != state_hash_before or state.iteration_number != iteration_before:
            raise V33TrainingError("data analysis mutated training state")

        result = {
            "implementation_revision": "V3.3-20W",
            "run_id": f"V33-ANALYSIS-{snapshot.market}-{uuid4().hex[:16]}",
            "market": snapshot.market,
            "benchmark": snapshot.benchmark_market,
            "training_mutated": False,
            "model": {
                "version": state.version,
                "iteration_number": state.iteration_number,
                "state_hash": state.state_hash,
                "trained_through": state.trained_through.isoformat(),
                "training_sample_count": state.training_sample_count,
                "horizon_weeks": HORIZON_WEEKS,
                "daily_window_sessions": len(snapshot.daily_sequence),
                "architecture": "weekly_main_plus_bounded_daily_corrector",
                "daily_correction_weeks": DAILY_CORRECTION_WEEKS,
                "daily_position_adjustment_limit": DAILY_POSITION_ADJUSTMENT_LIMIT,
                "daily_confidence_adjustment_limit": DAILY_CONFIDENCE_ADJUSTMENT_LIMIT,
            },
            "path": self._path_payload(
                forecast,
                adjusted_confidence=adjusted_confidence,
                confidence_adjustment=float(daily["confidence_adjustment"]),
            ),
            "turning_points": self._turning_payload(
                snapshot.market, forecast, future_sessions
            ),
            "position": {
                "current": current_position,
                "weekly_base_target": weekly_target,
                "daily_adjustment": applied_daily_position,
                "target": target,
                "change": target - current_position,
                "source": "local_investment_calendar_or_explicit_zero",
            },
            "advice": advice,
            "daily_corrector": {
                **daily,
                "position_adjustment": applied_daily_position,
                "path_weeks": DAILY_CORRECTION_WEEKS,
                "path_decay": list(DAILY_CORRECTION_DECAY),
                "path_max_abs_return": DAILY_PATH_CORRECTION_MAX_ABS,
                "cannot_reverse_high_confidence_weekly_direction": True,
                "high_confidence_gate": HIGH_CONFIDENCE_GATE,
            },
            "features": {
                "weekly": dict(snapshot.weekly),
                "daily": dict(snapshot.daily),
                "daily_sequence_count": len(snapshot.daily_sequence),
            },
            "data_quality": {
                "source_data_max_date": snapshot.source_data_max_date.isoformat(),
                "cutoff_date": snapshot.cutoff_date.isoformat(),
                "missing_masks": dict(snapshot.missing_masks),
                "missing_series": sorted(
                    name.removeprefix("missing_")
                    for name, value in snapshot.missing_masks.items()
                    if name.startswith("missing_") and int(value) == 1
                ),
                "degraded": any(int(value) == 1 for value in snapshot.missing_masks.values()),
                "provenance": dict(snapshot.provenance),
            },
        }
        result["analysis_hash"] = self._hash_result(result)
        return result

    @staticmethod
    def _target_position(
        snapshot: FeatureSnapshot,
        forecast: V33Forecast,
        current: int,
    ) -> tuple[int, list[str]]:
        terminal = forecast.weekly_base[-1]
        raw = 50.0 + terminal / max(forecast.threshold, 0.01) * 25.0
        weekly_drawdown = V33AnalysisService._drawdown_percent(forecast.weekly_base)
        raw += max(-15.0, min(0.0, weekly_drawdown + 5.0))
        policy: list[str] = ["20-week probability path", "drawdown penalty"]
        pe_percentile = snapshot.weekly.get("pe_expanding_percentile")
        pb_percentile = snapshot.weekly.get("pb_expanding_percentile")
        available_percentiles = [
            float(value) for value in (pe_percentile, pb_percentile) if value is not None
        ]
        valuation = sum(available_percentiles) / len(available_percentiles) if available_percentiles else None
        weekly_dif = snapshot.weekly.get("weekly_dif")
        weekly_dea = snapshot.weekly.get("weekly_dea")
        weekly_dif_slope = snapshot.weekly.get("weekly_dif_slope_3")
        top_confirmed = (
            forecast.dif_turn_kind == "top"
            or (
                weekly_dif is not None
                and weekly_dea is not None
                and weekly_dif_slope is not None
                and float(weekly_dif) < float(weekly_dea)
                and float(weekly_dif_slope) < 0.0
            )
        )
        bottom_confirmed = forecast.dif_turn_kind == "bottom" and forecast.turn_stable
        if valuation is None:
            policy.append("valuation unavailable: no neutral-value substitution")
        elif valuation >= 90.0:
            if top_confirmed and forecast.direction == "down":
                raw = min(raw, 15.0)
                policy.append("high valuation plus confirmed momentum top: risk cap 15%")
            else:
                raw = min(raw, 70.0)
                policy.append("high valuation but trend not fully broken: staged cap 70%")
        elif valuation <= 15.0:
            if bottom_confirmed and forecast.direction == "up":
                raw = max(raw, 80.0)
                policy.append("low valuation plus confirmed DIF bottom: staged recovery cap 80%")
            else:
                raw = min(raw, current + 10.0)
                policy.append("low valuation without reversal confirmation: probe at most 10 points")
        return _snap5(raw), policy

    @staticmethod
    def _daily_adjustments(forecast: V33Forecast) -> dict[str, float | str]:
        correction = np.asarray(
            forecast.daily_correction[:DAILY_CORRECTION_WEEKS], dtype=float
        )
        denominator = DAILY_PATH_CORRECTION_MAX_ABS * sum(DAILY_CORRECTION_DECAY)
        normalized = 0.0 if denominator <= 0 else float(np.sum(correction) / denominator)
        normalized = max(-1.0, min(1.0, normalized))
        position = int(
            round(normalized * DAILY_POSITION_ADJUSTMENT_LIMIT / 5.0) * 5
        )
        position = max(
            -DAILY_POSITION_ADJUSTMENT_LIMIT,
            min(DAILY_POSITION_ADJUSTMENT_LIMIT, position),
        )
        confidence = max(
            -DAILY_CONFIDENCE_ADJUSTMENT_LIMIT,
            min(
                DAILY_CONFIDENCE_ADJUSTMENT_LIMIT,
                normalized * DAILY_CONFIDENCE_ADJUSTMENT_LIMIT,
            ),
        )
        state = "bullish" if normalized > 0.15 else "bearish" if normalized < -0.15 else "neutral"
        return {
            "state": state,
            "normalized_signal": normalized,
            "position_adjustment": position,
            "confidence_adjustment": confidence,
        }

    @staticmethod
    def _drawdown_percent(path: Sequence[float]) -> float:
        wealth = [1.0 + float(value) for value in path]
        peak = wealth[0]
        worst = 0.0
        for value in wealth:
            peak = max(peak, value)
            if peak:
                worst = min(worst, value / peak - 1.0)
        return worst * 100.0

    def _advice(
        self,
        snapshot: FeatureSnapshot,
        forecast: V33Forecast,
        current: int,
        target: int,
        sessions: Sequence[date],
        policy: Sequence[str],
    ) -> dict[str, Any]:
        delta = target - current
        action = "buy" if delta > 0 else "sell" if delta < 0 else "hold"
        amounts = self._split_batches(abs(delta))
        turn_offset = self._turn_offset(forecast)
        batches: list[dict[str, Any]] = []
        for index, amount in enumerate(amounts):
            center_index = min(len(sessions) - 1, max(0, turn_offset + index * 5))
            left_index = max(0, center_index - 1)
            right_index = min(len(sessions) - 1, center_index + 1)
            batches.append(
                {
                    "batch": index + 1,
                    "action": action,
                    "percentage_points": amount,
                    "window_start": sessions[left_index].isoformat(),
                    "expected_date": sessions[center_index].isoformat(),
                    "window_end": sessions[right_index].isoformat(),
                    "condition": self._confirmation_condition(action, index, snapshot, forecast),
                }
            )
        ratio = self._fund_etf_ratio(forecast)
        if ratio not in FUND_ETF_RATIOS:
            raise V33AnalysisError("fund/ETF ratio escaped the approved five-value set")
        return {
            "action": action,
            "summary": (
                "保持当前仓位" if action == "hold" else f"分{len(batches)}批{'增仓' if action == 'buy' else '减仓'}{abs(delta)}个百分点"
            ),
            "batches": batches,
            "fund_etf_ratio": ratio,
            "policy": list(policy),
            "conditional": True,
            "position_grid": 5,
        }

    @staticmethod
    def _split_batches(amount: int) -> list[int]:
        if amount == 0:
            return []
        if amount % 5:
            raise V33AnalysisError("position delta must be a multiple of five")
        batch_count = min(4, max(1, math.ceil(amount / 20.0)))
        units = amount // 5
        # Front-load while reserving one 5-point unit for every later
        # confirmation.  This yields audited integer 5-point batches.
        weights = [4, 3, 2, 1][:batch_count]
        allocation = [1] * batch_count
        remaining = units - batch_count
        for index in range(remaining):
            best = max(range(batch_count), key=lambda item: weights[item] / (allocation[item] + 1))
            allocation[best] += 1
        return [value * 5 for value in allocation]

    @staticmethod
    def _turn_offset(forecast: V33Forecast) -> int:
        if forecast.price_turn_days is not None and forecast.turn_stable:
            return max(0, int(round(forecast.price_turn_days)))
        relevant_week = forecast.predicted_low_week if forecast.direction == "up" else forecast.predicted_high_week
        return max(0, relevant_week * 5 - 1)

    @staticmethod
    def _confirmation_condition(
        action: str,
        index: int,
        snapshot: FeatureSnapshot,
        forecast: V33Forecast,
    ) -> str:
        if action == "buy":
            base = "DIF一阶导数由负转正且二阶导数为正，成交量与价格确认"
            return base if index == 0 else "前一批成立后，DIF不再转弱且价格保持反转结构"
        if action == "sell":
            base = "DIF一阶导数由正转负且二阶导数为负，DEA/量价共同确认"
            return base if index == 0 else "前一批成立后，DIF继续转弱或死叉/放量下破确认"
        return "无需执行；等待20周方向或DIF拐点形成共振"

    @staticmethod
    def _fund_etf_ratio(forecast: V33Forecast) -> str:
        terminal = forecast.p50[-1]
        if terminal >= 0.10:
            return "7:3"
        if terminal >= 0.03:
            return "6:4"
        if terminal > -0.03:
            return "5:5"
        if terminal > -0.10:
            return "4:6"
        return "3:7"

    def _future_sessions(self, market: str, as_of: date) -> tuple[date, ...]:
        # Keep the target instrument identity all the way through the calendar
        # boundary.  In particular, 159941 is a Shenzhen-listed ETF; routing
        # it through 399006 happened to produce the same mainland holidays in
        # the current adapter, but made it impossible to prove that inference
        # asked for the Shenzhen instrument calendar.
        sessions = tuple(
            self.calendar.sessions(
                market,
                as_of + timedelta(days=1),
                as_of + timedelta(days=180),
            )
        )
        if len(sessions) < HORIZON_WEEKS * 5:
            raise V33AnalysisError("exchange calendar has fewer than 100 future sessions")
        return sessions[: HORIZON_WEEKS * 5]

    @staticmethod
    def _path_payload(
        forecast: V33Forecast,
        *,
        adjusted_confidence: float,
        confidence_adjustment: float,
    ) -> dict[str, Any]:
        return {
            "weeks": list(range(1, HORIZON_WEEKS + 1)),
            "p10": list(forecast.p10),
            "p50": list(forecast.p50),
            "p90": list(forecast.p90),
            "expected": list(forecast.expected),
            "weekly_base": list(forecast.weekly_base),
            "daily_correction": list(forecast.daily_correction),
            "direction": forecast.direction,
            "weekly_confidence": forecast.confidence,
            "daily_confidence_adjustment": confidence_adjustment,
            "confidence": adjusted_confidence,
            "up_probability": forecast.up_probability,
            "sideways_probability": forecast.sideways_probability,
            "down_probability": forecast.down_probability,
            "expected_max_drawdown": forecast.expected_max_drawdown,
            "predicted_high_week": forecast.predicted_high_week,
            "predicted_low_week": forecast.predicted_low_week,
            "analogue_role": "residual_interval_calibration_only",
            "analogue_calibration_count": forecast.analogue_calibration_count,
        }

    def _turning_payload(
        self,
        market: str,
        forecast: V33Forecast,
        sessions: Sequence[date],
    ) -> dict[str, Any]:
        def window(days: float | None) -> Mapping[str, str | None]:
            if days is None:
                return {"start": None, "center": None, "end": None}
            center = min(len(sessions) - 1, max(0, int(round(days))))
            return {
                "start": sessions[max(0, center - 2)].isoformat(),
                "center": sessions[center].isoformat(),
                "end": sessions[min(len(sessions) - 1, center + 2)].isoformat(),
            }

        metadata_getter = getattr(self.calendar, "metadata", None)
        if callable(metadata_getter):
            calendar_metadata = dict(metadata_getter(market))
        else:
            calendar_metadata = {
                "target_exchange": "XSHE" if market in {"399006", "159941"} else market,
                "schedule_provider": type(self.calendar).__name__,
                "schedule_name": "caller-provided",
                "equivalent_mainland_schedule_proxy": False,
            }
        target_exchange = str(calendar_metadata["target_exchange"])
        schedule_provider = str(calendar_metadata["schedule_provider"])
        schedule_name = str(calendar_metadata["schedule_name"])
        proxy_note = (
            "; joint mainland schedule proxy"
            if calendar_metadata.get("equivalent_mainland_schedule_proxy")
            else ""
        )

        return {
            "stable": forecast.turn_stable,
            "kind": forecast.dif_turn_kind,
            "dif_derivative_zero": window(forecast.dif_turn_days),
            "price_turn": window(forecast.price_turn_days),
            "dif_axis_zero_is_distinct": True,
            "calendar": calendar_metadata,
            "date_basis": (
                f"actual {target_exchange} target sessions; schedule source "
                f"{schedule_provider}:{schedule_name}{proxy_note}"
            ),
        }

    @staticmethod
    def _hash_result(result: Mapping[str, Any]) -> str:
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
