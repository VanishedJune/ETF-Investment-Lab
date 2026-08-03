"""Leakage-safe V3.2 bootstrap and one-week incremental model training.

The implementation intentionally uses a compact, deterministic weighted
nearest-neighbour path model.  Every forecast at cutoff *t* can only select
historical scenarios whose complete 13-week outcome was already visible at
*t*.  Candidate weights inherit the previous optimizer state and are promoted
only by rolling out-of-sample validation.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import statistics
from threading import Lock
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ..daemon_executor import DaemonThreadPoolExecutor
from ..models.models import (
    IndicatorRecord,
    Instrument,
    MarketPrice,
    ValuationRecord,
    V32DataSnapshot,
    V32ModelVersion,
    V32OptimizerState,
    V32TrainingCheckpoint,
    V32TrainingIteration,
    V32TrainingRun,
    utc_now,
)
from .market_calendar import CalendarProvider, ExchangeCalendarProvider


SUPPORTED_MARKETS = ("399006", "NDX")
FEATURE_VERSION = "V3.2-WEEKLY-DAILY-100D-1"
METHODOLOGY_VERSION = "V3.2-PROGRESSIVE-WALK-FORWARD-1"
ANALYSIS_DAYS = 3653
WARMUP_WEEKS = 60
HORIZON_WEEKS = 13
MIN_ANALOGS = 20
MAX_ANALOGS = 80
MIN_PROMOTION_SAMPLES = 52
# A two-percentage-point coordinate step typically changes the 52-sample
# out-of-sample loss by well below one percent.  Requiring a 3% jump made a
# valid candidate mathematically unreachable in the first audited bootstrap.
# Promotion still requires a *measured* improvement, a non-regressing recent
# window, direction stability, and interval-coverage stability.
PROMOTION_IMPROVEMENT = 0.002
WEEKLY_FEATURES = (
    "return_4w",
    "return_8w",
    "return_13w",
    "volatility_13w",
    "ma20_distance",
    "macd_spread",
    "valuation_percentile",
    "drawdown_60w",
)
DAILY_FEATURES = (
    "return_5d",
    "return_20d",
    "rsi",
    "ma20_distance",
    "macd_spread",
    "volatility_20d",
)
FEATURE_SCALES = {
    "return_4w": 0.12,
    "return_8w": 0.18,
    "return_13w": 0.28,
    "volatility_13w": 0.35,
    "ma20_distance": 0.20,
    "macd_spread": 0.035,
    "valuation_percentile": 45.0,
    "drawdown_60w": 0.35,
    "return_5d": 0.08,
    "return_20d": 0.16,
    "rsi": 25.0,
    "daily_ma20_distance": 0.14,
    "daily_macd_spread": 0.025,
    "volatility_20d": 0.50,
}


class V32TrainingError(RuntimeError):
    """Training cannot continue without violating the V3.2 contract."""


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    global_index: int
    week_key: str
    cutoff_date: date
    close: float
    weekly: Mapping[str, float]
    daily: Mapping[str, float]
    future_path: tuple[float, ...] | None
    price_source: str
    volume_source: str | None
    daily_as_of: date
    valuation_as_of: date | None


@dataclass(frozen=True, slots=True)
class ForecastResult:
    p10: tuple[float, ...]
    p50: tuple[float, ...]
    p90: tuple[float, ...]
    expected: tuple[float, ...]
    up_probability: float
    sideways_probability: float
    down_probability: float
    direction_threshold: float
    direction: str
    confidence: float
    base_target_position: int
    expected_max_drawdown: float
    high_week_range: str
    low_week_range: str
    scenario_count: int


def _float(value: Decimal | float | int | None, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _iso_week(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _window_return(values: Sequence[float], span: int) -> float:
    if len(values) <= span or not values[-span - 1]:
        return 0.0
    return values[-1] / values[-span - 1] - 1.0


def _volatility(values: Sequence[float], span: int, annualization: int) -> float:
    sample = values[-(span + 1) :]
    returns = [right / left - 1.0 for left, right in zip(sample, sample[1:]) if left]
    return statistics.pstdev(returns) * math.sqrt(annualization) if len(returns) >= 2 else 0.0


def _max_drawdown(values: Sequence[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return worst


def _ema(values: Sequence[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    output = [values[0]]
    for value in values[1:]:
        output.append(alpha * value + (1.0 - alpha) * output[-1])
    return output


def _macd(values: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
    fast = _ema(values, 12)
    slow = _ema(values, 26)
    dif = [left - right for left, right in zip(fast, slow)]
    dea = _ema(dif, 9)
    histogram = [(left - right) * 2.0 for left, right in zip(dif, dea)]
    return dif, dea, histogram


def _rsi(values: Sequence[float], period: int = 14) -> float:
    if len(values) <= period:
        return 50.0
    changes = [right - left for left, right in zip(values[-(period + 1) :], values[-period:])]
    gains = _mean([max(change, 0.0) for change in changes])
    losses = _mean([max(-change, 0.0) for change in changes])
    if losses == 0:
        return 100.0 if gains else 50.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def _snap5(value: float) -> int:
    return max(0, min(100, int(round(value / 5.0) * 5)))


def _hash_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalized(weights: Mapping[str, float]) -> dict[str, float]:
    clean = {key: max(0.01, float(value)) for key, value in weights.items()}
    total = sum(clean.values())
    return {key: value / total for key, value in clean.items()}


def _initial_parameters(market: str) -> tuple[dict[str, Any], dict[str, Any]]:
    weekly = _normalized(
        {
            "return_4w": 0.16,
            "return_8w": 0.14,
            "return_13w": 0.16,
            "volatility_13w": 0.12,
            "ma20_distance": 0.14,
            "macd_spread": 0.12,
            "valuation_percentile": 0.10 if market == "399006" else 0.04,
            "drawdown_60w": 0.06 if market == "399006" else 0.12,
        }
    )
    daily = _normalized(
        {
            "return_5d": 0.16,
            "return_20d": 0.18,
            "rsi": 0.14,
            "ma20_distance": 0.20,
            "macd_spread": 0.18,
            "volatility_20d": 0.14,
        }
    )
    return (
        {
            "weights": weekly,
            "daily_contribution": 0.25,
            "base_threshold": 0.045 if market == "399006" else 0.030,
            "volatility_reference": 0.34 if market == "399006" else 0.24,
            "analog_count": MAX_ANALOGS,
        },
        {
            "weights": daily,
            "rsi_overbought": 70.0,
            "rsi_oversold": 30.0,
            "confidence_limit": 10,
            "position_limit": 10,
        },
    )


class V32TrainingService:
    """Run V3.2 bootstrap training and at-most-once weekly increments."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        now_provider: callable | None = None,
        calendar: CalendarProvider | None = None,
    ) -> None:
        self.sessions = sessions
        self.now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self.calendar = calendar or ExchangeCalendarProvider()
        self._matrix_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._executor = DaemonThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="v32-training",
        )
        self._active_lock = Lock()
        self._active: dict[str, Future[dict[str, Any]]] = {}

    def submit_bootstrap(self, market: str) -> dict[str, Any]:
        return self._submit(market, "bootstrap", lambda: self.bootstrap(market))

    def submit_incremental(self, market: str) -> dict[str, Any]:
        return self._submit(market, "incremental", lambda: self.train_incremental(market))

    def submit_due_incrementals(self) -> list[dict[str, Any]]:
        submitted: list[dict[str, Any]] = []
        for market in SUPPORTED_MARKETS:
            if self.status(market)[market]["bootstrapped"]:
                submitted.append(self.submit_incremental(market))
        return submitted

    def _submit(
        self,
        market: str,
        task_type: str,
        operation: callable,
    ) -> dict[str, Any]:
        self._validate_market(market)
        with self._active_lock:
            current = self._active.get(market)
            if current is not None and not current.done():
                return {"market": market, "status": "already_running", "task_type": task_type}
            future = self._executor.submit(operation)
            self._active[market] = future

        def clear(completed: Future[dict[str, Any]]) -> None:
            try:
                completed.result()
            except Exception:
                # bootstrap/train_incremental persist a failed run before the
                # exception reaches this boundary; the API exposes that record.
                pass
            with self._active_lock:
                if self._active.get(market) is completed:
                    self._active.pop(market, None)

        future.add_done_callback(clear)
        return {"market": market, "status": "queued", "task_type": task_type}

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def status(self, market: str | None = None) -> dict[str, Any]:
        markets = (market,) if market else SUPPORTED_MARKETS
        payload: dict[str, Any] = {}
        with self._active_lock:
            active = {
                symbol: future is not None and not future.done()
                for symbol, future in self._active.items()
            }
        with self.sessions() as session:
            for symbol in markets:
                self._validate_market(symbol)
                state = session.scalar(
                    select(V32OptimizerState).where(V32OptimizerState.market == symbol)
                )
                latest_run = session.scalar(
                    select(V32TrainingRun)
                    .where(V32TrainingRun.market == symbol)
                    .order_by(V32TrainingRun.started_at.desc())
                )
                iteration_count = session.scalar(
                    select(func.count()).select_from(V32TrainingIteration).where(
                        V32TrainingIteration.market == symbol
                    )
                ) or 0
                payload[symbol] = {
                    "is_training": active.get(symbol, False),
                    "bootstrapped": state is not None,
                    "iteration_count": iteration_count,
                    "last_training_week_key": None if state is None else state.last_training_week_key,
                    "last_successful_training_at": None if state is None else state.last_successful_training_at.isoformat(),
                    "next_training_eligible_at": None if state is None else state.next_training_eligible_at.isoformat(),
                    "champion_weekly_version": None if state is None else state.champion_weekly_version,
                    "champion_daily_version": None if state is None else state.champion_daily_version,
                    "latest_run": None if latest_run is None else self._run_payload(latest_run),
                }
        return payload

    def bootstrap(self, market: str, *, force: bool = False) -> dict[str, Any]:
        """Create every ten-year progressive iteration for one market."""
        self._validate_market(market)
        points, training_start = self._load_points(market)
        training_points = points[training_start:]
        if not training_points:
            raise V32TrainingError(f"{market}没有可用的十年训练截点")
        with self.sessions() as session:
            existing = session.scalar(
                select(func.count()).select_from(V32TrainingIteration).where(
                    V32TrainingIteration.market == market
                )
            ) or 0
            latest_iteration = session.scalar(
                select(V32TrainingIteration)
                .where(V32TrainingIteration.market == market)
                .order_by(V32TrainingIteration.iteration_number.desc())
            )
            existing_mature = session.scalar(
                select(func.count()).select_from(V32TrainingIteration).where(
                    V32TrainingIteration.market == market,
                    V32TrainingIteration.maturity_status == "full",
                )
            ) or 0
        if existing and force:
            raise V32TrainingError(
                "V3.2训练记录为审计数据，不允许force覆盖；请使用新的空数据库执行重建"
            )
        if existing > len(training_points):
            raise V32TrainingError("已保存的V3.2轮次超过当前十年训练窗口，拒绝静默重写")
        if existing == len(training_points):
            return {
                "market": market,
                "status": "already_bootstrapped",
                "iteration_count": existing,
                "latest_week_key": training_points[-1].week_key,
            }

        if existing:
            if latest_iteration is None:
                raise V32TrainingError("V3.2训练计数与最新轮次不一致")
            saved_state = dict(latest_iteration.optimizer_state_json or {})
            weekly_parameters = dict(saved_state.get("weekly_parameters") or {})
            daily_parameters = dict(saved_state.get("daily_parameters") or {})
            if not weekly_parameters or not daily_parameters:
                raise V32TrainingError("V3.2检查点缺少可恢复模型参数")
            weekly_version = latest_iteration.champion_weekly_version
            daily_version = latest_iteration.champion_daily_version
            weekly_version_number = int(weekly_version.rsplit(".", 1)[1])
            daily_version_number = int(daily_version.rsplit(".", 1)[1])
            promoted_count = int(saved_state.get("accepted_candidate_count", 0))
            rejected_count = int(saved_state.get("rejected_candidate_count", 0))
            run_type = "bootstrap_resume"
            self._mark_interrupted_runs(market)
        else:
            weekly_parameters, daily_parameters = _initial_parameters(market)
            weekly_version_number = 0
            daily_version_number = 0
            weekly_version = self._version_name(market, "weekly", weekly_version_number)
            daily_version = self._version_name(market, "daily", daily_version_number)
            self._create_model_versions(
                market,
                training_points[0].cutoff_date,
                weekly_version,
                daily_version,
                weekly_parameters,
                daily_parameters,
                parent_weekly=None,
                parent_daily=None,
            )
            promoted_count = 0
            rejected_count = 0
            run_type = "bootstrap"

        run_id = f"V32-BOOT-{market}-{uuid4().hex[:16]}"
        self._create_run(
            run_id,
            market,
            run_type,
            training_points[-1].week_key,
            source_parent_iteration=existing or None,
        )
        iteration_metrics: list[dict[str, Any]] = [{} for _ in range(existing_mature)]
        try:
            for offset, point in enumerate(training_points[existing:], start=existing + 1):
                global_position = training_start + offset - 1
                forecast = self._forecast(points, global_position, weekly_parameters, daily_parameters)
                evaluation = self._evaluate(point, forecast)
                candidate: dict[str, Any] | None = None
                promoted = False
                rejection_reason = "等待至少52个完整成熟样本"
                mature_positions = [
                    index
                    for index in range(training_start, global_position + 1)
                    if points[index].future_path is not None
                    and points[index].global_index + HORIZON_WEEKS <= point.global_index
                ]
                if len(mature_positions) >= MIN_PROMOTION_SAMPLES:
                    candidate_weekly, candidate_daily = self._candidate_parameters(
                        weekly_parameters,
                        daily_parameters,
                        iteration_number=offset,
                    )
                    comparison = self._compare_candidate(
                        points,
                        mature_positions,
                        weekly_parameters,
                        daily_parameters,
                        candidate_weekly,
                        candidate_daily,
                    )
                    candidate = {
                        "weekly": candidate_weekly,
                        "daily": candidate_daily,
                        "comparison": comparison,
                    }
                    promoted = comparison["promoted"]
                    rejection_reason = comparison["reason"]
                    if promoted:
                        old_weekly = weekly_version
                        old_daily = daily_version
                        weekly_parameters = candidate_weekly
                        daily_parameters = candidate_daily
                        weekly_version_number += 1
                        daily_version_number += 1
                        weekly_version = self._version_name(market, "weekly", weekly_version_number)
                        daily_version = self._version_name(market, "daily", daily_version_number)
                        self._supersede_and_create_models(
                            market,
                            point.cutoff_date,
                            old_weekly,
                            old_daily,
                            weekly_version,
                            daily_version,
                            weekly_parameters,
                            daily_parameters,
                            comparison,
                        )
                        promoted_count += 1
                    else:
                        rejected_count += 1

                snapshot_id, snapshot_payload, snapshot_hash = self._snapshot_identity(
                    market, point, weekly_version, daily_version
                )
                optimizer_state = {
                    "weekly_parameters": weekly_parameters,
                    "daily_parameters": daily_parameters,
                    "accepted_candidate_count": promoted_count,
                    "rejected_candidate_count": rejected_count,
                    "latest_candidate": candidate,
                    "latest_rejection_reason": None if promoted else rejection_reason,
                }
                self._persist_training_step(
                    run_id,
                    market,
                    offset,
                    point,
                    snapshot_id,
                    snapshot_payload,
                    snapshot_hash,
                    weekly_version,
                    daily_version,
                    candidate,
                    promoted,
                    forecast,
                    evaluation,
                    optimizer_state,
                )
                if evaluation:
                    iteration_metrics.append(evaluation)

            last = training_points[-1]
            now = self.now_provider()
            state_payload = {
                "accepted_candidate_count": promoted_count,
                "rejected_candidate_count": rejected_count,
                "training_window_start": training_points[0].cutoff_date.isoformat(),
                "training_window_end": last.cutoff_date.isoformat(),
                "mature_evaluation_count": len(iteration_metrics),
            }
            self._save_optimizer_state(
                market,
                len(training_points),
                last.week_key,
                weekly_version,
                daily_version,
                weekly_parameters,
                daily_parameters,
                state_payload,
                now,
            )
            result = {
                "market": market,
                "status": "completed",
                "iteration_count": len(training_points),
                "first_week_key": training_points[0].week_key,
                "latest_week_key": last.week_key,
                "mature_evaluation_count": len(iteration_metrics),
                "pending_evaluation_count": len(training_points) - len(iteration_metrics),
                "champion_weekly_version": weekly_version,
                "champion_daily_version": daily_version,
                "promoted_candidate_count": promoted_count,
                "rejected_candidate_count": rejected_count,
            }
            self._complete_run(run_id, result)
            return result
        except Exception as error:
            self._fail_run(run_id, error)
            raise

    def train_incremental(
        self,
        market: str,
        *,
        bootstrap_catchup: bool = False,
    ) -> dict[str, Any]:
        """Add at most one iteration when a new complete week and seven days exist."""
        self._validate_market(market)
        points, training_start = self._load_points(market)
        self.mature_pending(market, points=points)
        latest = points[-1]
        now = self.now_provider()
        with self.sessions() as session:
            state = session.scalar(
                select(V32OptimizerState).where(V32OptimizerState.market == market)
            )
        if state is None:
            raise V32TrainingError("尚未完成V3.2首次历史训练")
        if latest.week_key <= state.last_training_week_key:
            return {"market": market, "status": "not_due", "reason": "没有新的完整交易周"}
        if now < state.next_training_eligible_at and not bootstrap_catchup:
            return {
                "market": market,
                "status": "not_due",
                "reason": "距上次成功训练不足一周",
                "next_training_eligible_at": state.next_training_eligible_at.isoformat(),
            }

        run_id = f"V32-INCR-{market}-{uuid4().hex[:16]}"
        self._create_run(
            run_id,
            market,
            "incremental",
            latest.week_key,
            source_parent_iteration=state.iteration_number,
        )
        weekly_parameters = dict(state.weekly_parameters_json)
        daily_parameters = dict(state.daily_parameters_json)
        global_position = len(points) - 1
        forecast = self._forecast(points, global_position, weekly_parameters, daily_parameters)
        evaluation = self._evaluate(latest, forecast)
        mature_positions = [
            index
            for index in range(training_start, global_position + 1)
            if points[index].future_path is not None
            and points[index].global_index + HORIZON_WEEKS <= latest.global_index
        ]
        candidate_weekly, candidate_daily = self._candidate_parameters(
            weekly_parameters,
            daily_parameters,
            iteration_number=state.iteration_number + 1,
        )
        comparison = self._compare_candidate(
            points,
            mature_positions,
            weekly_parameters,
            daily_parameters,
            candidate_weekly,
            candidate_daily,
        )
        promoted = comparison["promoted"]
        weekly_version = state.champion_weekly_version
        daily_version = state.champion_daily_version
        if promoted:
            weekly_version = self._next_version(weekly_version)
            daily_version = self._next_version(daily_version)
            self._supersede_and_create_models(
                market,
                latest.cutoff_date,
                state.champion_weekly_version,
                state.champion_daily_version,
                weekly_version,
                daily_version,
                candidate_weekly,
                candidate_daily,
                comparison,
            )
            weekly_parameters, daily_parameters = candidate_weekly, candidate_daily

        iteration_number = state.iteration_number + 1
        snapshot_id, snapshot_payload, snapshot_hash = self._snapshot_identity(
            market, latest, weekly_version, daily_version
        )
        memory = dict(state.optimizer_memory_json)
        key = "accepted_candidate_count" if promoted else "rejected_candidate_count"
        memory[key] = int(memory.get(key, 0)) + 1
        memory["latest_candidate"] = comparison
        memory["skipped_week_count"] = self._week_distance(
            state.last_training_week_key, latest.week_key
        ) - 1
        self._persist_training_step(
            run_id,
            market,
            iteration_number,
            latest,
            snapshot_id,
            snapshot_payload,
            snapshot_hash,
            weekly_version,
            daily_version,
            {"weekly": candidate_weekly, "daily": candidate_daily, "comparison": comparison},
            promoted,
            forecast,
            evaluation,
            memory,
        )
        self._save_optimizer_state(
            market,
            iteration_number,
            latest.week_key,
            weekly_version,
            daily_version,
            weekly_parameters,
            daily_parameters,
            memory,
            now,
        )
        result = {
            "market": market,
            "status": "completed",
            "iteration_number": iteration_number,
            "week_key": latest.week_key,
            "promoted": promoted,
            "champion_weekly_version": weekly_version,
            "champion_daily_version": daily_version,
            "comparison": comparison,
        }
        self._complete_run(run_id, result)
        return result

    def mature_pending(
        self,
        market: str,
        *,
        points: Sequence[TrainingPoint] | None = None,
    ) -> int:
        """Evaluate saved pending forecasts once their full 13-week path is known.

        The serialized forecast from the original iteration is immutable. Using
        the current Champion here would leak later model state into the historic
        out-of-sample evaluation curve.
        """
        self._validate_market(market)
        if points is None:
            points, _ = self._load_points(market)
        points_by_week = {point.week_key: point for point in points}

        matured_count = 0
        with self.sessions() as session, session.begin():
            rows = list(
                session.scalars(
                    select(V32TrainingIteration)
                    .where(
                        V32TrainingIteration.market == market,
                        V32TrainingIteration.maturity_status == "pending",
                    )
                    .order_by(V32TrainingIteration.iteration_number)
                )
            )
            for row in rows:
                point = points_by_week.get(row.week_key)
                if point is None or point.future_path is None:
                    continue
                forecast = self._forecast_from_payload(row.forecast_json)
                evaluation = self._evaluate(point, forecast)
                if evaluation is None:
                    continue
                row.maturity_status = "full"
                row.evaluation_json = dict(evaluation)
                row.composite_loss = self._as_decimal(evaluation.get("composite_loss"))
                row.path_error = self._as_decimal(evaluation.get("path_error"))
                row.terminal_return_error = self._as_decimal(
                    evaluation.get("terminal_return_error")
                )
                row.direction_score = self._as_decimal(evaluation.get("direction_score"))
                row.interval_coverage = self._as_decimal(evaluation.get("interval_coverage"))
                row.high_week_error = evaluation.get("high_week_error")
                row.low_week_error = evaluation.get("low_week_error")
                row.audit_json = {
                    **dict(row.audit_json or {}),
                    "maturity_evaluation_source": "saved_forecast_json",
                    "matured_at": utc_now().isoformat(),
                }
                matured_count += 1
        return matured_count

    def iteration_curve(self, market: str) -> dict[str, Any]:
        self._validate_market(market)
        with self.sessions() as session:
            rows = list(
                session.scalars(
                    select(V32TrainingIteration)
                    .where(V32TrainingIteration.market == market)
                    .order_by(V32TrainingIteration.iteration_number)
                )
            )
        points: list[dict[str, Any]] = []
        mature_losses: list[float] = []
        mature_deviations: list[float] = []
        for row in rows:
            loss = self._decimal(row.composite_loss)
            deviation_days = (
                None
                if row.high_week_error is None or row.low_week_error is None
                else (row.high_week_error + row.low_week_error) * 2.5
            )
            if loss is not None:
                mature_losses.append(loss)
            if deviation_days is not None:
                mature_deviations.append(deviation_days)
            points.append(
                {
                    "iteration": row.iteration_number,
                    "parent_iteration": row.parent_iteration_number,
                    "week_key": row.week_key,
                    "cutoff_date": row.cutoff_date.isoformat(),
                    "maturity_status": row.maturity_status,
                    "composite_loss": self._decimal(row.composite_loss),
                    "path_error": self._decimal(row.path_error),
                    "terminal_return_error": self._decimal(row.terminal_return_error),
                    "direction_score": self._decimal(row.direction_score),
                    "interval_coverage": self._decimal(row.interval_coverage),
                    "high_week_error": row.high_week_error,
                    "low_week_error": row.low_week_error,
                    "high_low_deviation_days": deviation_days,
                    "rolling_20_loss": (
                        None if not mature_losses else _mean(mature_losses[-20:])
                    ),
                    "rolling_52_loss": (
                        None if not mature_losses else _mean(mature_losses[-52:])
                    ),
                    "rolling_20_deviation_days": (
                        None if not mature_deviations else _mean(mature_deviations[-20:])
                    ),
                    "rolling_52_deviation_days": (
                        None if not mature_deviations else _mean(mature_deviations[-52:])
                    ),
                    "champion_version": row.champion_weekly_version,
                    "challenger_version": row.challenger_version,
                    "promoted": row.promoted,
                    "rejection_reason": (row.optimizer_state_json or {}).get("latest_rejection_reason")
                    or ((row.optimizer_state_json or {}).get("latest_candidate") or {}).get("reason"),
                }
            )
        return {
            "market": market,
            "points": points,
        }

    def champion(self, market: str) -> dict[str, Any]:
        self._validate_market(market)
        with self.sessions() as session:
            state = session.scalar(
                select(V32OptimizerState).where(V32OptimizerState.market == market)
            )
        if state is None:
            raise V32TrainingError("V3.2 Champion尚未训练")
        return {
            "market": market,
            "iteration_number": state.iteration_number,
            "last_training_week_key": state.last_training_week_key,
            "weekly_version": state.champion_weekly_version,
            "daily_version": state.champion_daily_version,
            "weekly_parameters": state.weekly_parameters_json,
            "daily_parameters": state.daily_parameters_json,
            "optimizer_memory": state.optimizer_memory_json,
            "last_successful_training_at": state.last_successful_training_at.isoformat(),
            "next_training_eligible_at": state.next_training_eligible_at.isoformat(),
        }

    def _load_points(self, market: str) -> tuple[list[TrainingPoint], int]:
        with self.sessions() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == market))
            if instrument is None:
                raise V32TrainingError(f"未知市场：{market}")
            weekly_rows = list(
                session.scalars(
                    select(MarketPrice)
                    .where(
                        MarketPrice.instrument_id == instrument.id,
                        MarketPrice.timeframe == "weekly",
                    )
                    .order_by(MarketPrice.trade_date)
                )
            )
            daily_rows = list(
                session.scalars(
                    select(MarketPrice)
                    .where(
                        MarketPrice.instrument_id == instrument.id,
                        MarketPrice.timeframe == "daily",
                    )
                    .order_by(MarketPrice.trade_date)
                )
            )
            valuations = list(
                session.scalars(
                    select(ValuationRecord)
                    .where(ValuationRecord.instrument_id == instrument.id)
                    .order_by(ValuationRecord.valuation_date)
                )
            )
        if len(weekly_rows) < WARMUP_WEEKS + HORIZON_WEEKS + 20:
            raise V32TrainingError(f"{market}周K历史不足")
        if len(daily_rows) < 100:
            raise V32TrainingError(f"{market}日K历史不足100个交易日")
        weekly_rows = self._complete_weekly_rows(market, weekly_rows, daily_rows)
        self._validate_sources(market, weekly_rows, daily_rows)

        closes = [_float(row.adjusted_close_price or row.close_price) for row in weekly_rows]
        dif, dea, _ = _macd(closes)
        daily_dates = [row.trade_date for row in daily_rows]
        valuation_dates = [row.valuation_date for row in valuations]
        valuation_values = [
            _float(row.pe_ratio if row.pe_ratio is not None else row.pb_ratio, math.nan)
            for row in valuations
        ]
        points: list[TrainingPoint] = []
        for index in range(WARMUP_WEEKS, len(weekly_rows)):
            row = weekly_rows[index]
            history = closes[: index + 1]
            ma20 = _mean(history[-20:])
            daily_end = bisect_right(daily_dates, row.trade_date)
            daily_window = daily_rows[max(0, daily_end - 100) : daily_end]
            if len(daily_window) < 100:
                continue
            daily_closes = [_float(item.adjusted_close_price or item.close_price) for item in daily_window]
            daily_dif, daily_dea, _ = _macd(daily_closes)
            daily_ma20 = _mean(daily_closes[-20:])
            valuation_end = bisect_right(valuation_dates, row.trade_date)
            known_values = [value for value in valuation_values[:valuation_end] if math.isfinite(value)]
            valuation_percentile = (
                sum(value <= known_values[-1] for value in known_values) / len(known_values) * 100.0
                if known_values
                else 50.0
            )
            weekly_features = {
                "return_4w": _window_return(history, 4),
                "return_8w": _window_return(history, 8),
                "return_13w": _window_return(history, 13),
                "volatility_13w": _volatility(history, 13, 52),
                "ma20_distance": history[-1] / ma20 - 1.0 if ma20 else 0.0,
                "macd_spread": (dif[index] - dea[index]) / history[-1] if history[-1] else 0.0,
                "valuation_percentile": valuation_percentile,
                "drawdown_60w": _max_drawdown(history[-60:]),
            }
            daily_features = {
                "return_5d": _window_return(daily_closes, 5),
                "return_20d": _window_return(daily_closes, 20),
                "rsi": _rsi(daily_closes),
                "ma20_distance": daily_closes[-1] / daily_ma20 - 1.0 if daily_ma20 else 0.0,
                "macd_spread": (daily_dif[-1] - daily_dea[-1]) / daily_closes[-1] if daily_closes[-1] else 0.0,
                "volatility_20d": _volatility(daily_closes, 20, 252),
            }
            future_path = None
            if index + HORIZON_WEEKS < len(closes):
                base = closes[index]
                future_path = tuple(
                    closes[index + horizon] / base - 1.0
                    for horizon in range(1, HORIZON_WEEKS + 1)
                )
            points.append(
                TrainingPoint(
                    global_index=index,
                    week_key=_iso_week(row.trade_date),
                    cutoff_date=row.trade_date,
                    close=history[-1],
                    weekly=weekly_features,
                    daily=daily_features,
                    future_path=future_path,
                    price_source=row.source or "UNKNOWN",
                    volume_source=row.volume_source,
                    daily_as_of=daily_window[-1].trade_date,
                    valuation_as_of=None if valuation_end == 0 else valuation_dates[valuation_end - 1],
                )
            )
        if not points:
            raise V32TrainingError(f"{market}无法构建训练特征")
        window_start = points[-1].cutoff_date - timedelta(days=ANALYSIS_DAYS)
        training_start = next(
            (index for index, point in enumerate(points) if point.cutoff_date >= window_start),
            0,
        )
        return points, training_start

    def _complete_weekly_rows(
        self,
        market: str,
        weekly_rows: Sequence[MarketPrice],
        daily_rows: Sequence[MarketPrice],
    ) -> list[MarketPrice]:
        """Exclude every weekly bar whose exchange week has not fully closed."""
        if not weekly_rows or not daily_rows:
            return []
        start = weekly_rows[0].trade_date - timedelta(days=7)
        end = weekly_rows[-1].trade_date + timedelta(days=7)
        sessions = self.calendar.sessions(market, start, end)
        expected_last: dict[tuple[int, int], date] = {}
        for session_day in sessions:
            key = session_day.isocalendar()[:2]
            expected_last[key] = session_day
        daily_dates = {row.trade_date for row in daily_rows}
        latest_daily = daily_rows[-1].trade_date
        complete: list[MarketPrice] = []
        for row in weekly_rows:
            expected = expected_last.get(row.trade_date.isocalendar()[:2])
            if (
                expected is not None
                and expected <= latest_daily
                and expected in daily_dates
                and row.trade_date == expected
            ):
                complete.append(row)
        if len(complete) < WARMUP_WEEKS + HORIZON_WEEKS + 20:
            raise V32TrainingError(f"{market}完整周K历史不足")
        return complete

    @staticmethod
    def _validate_sources(
        market: str,
        weekly_rows: Sequence[MarketPrice],
        daily_rows: Sequence[MarketPrice],
    ) -> None:
        prohibited = ("ETF", "FUND", "QQQ", "159941")
        for row in (*weekly_rows, *daily_rows):
            source = (row.source or "").upper()
            if any(marker in source for marker in prohibited):
                raise V32TrainingError(f"{market}检测到禁止的替代行情来源：{row.source}")
            close = _float(row.adjusted_close_price or row.close_price)
            if close <= 0:
                raise V32TrainingError(f"{market}存在非正收盘价：{row.trade_date}")
        if market == "399006":
            missing = [row.trade_date for row in weekly_rows if row.volume is None]
            if missing:
                raise V32TrainingError(f"创业板周成交量缺失：{missing[-1]}")

    def _forecast(
        self,
        points: Sequence[TrainingPoint],
        position: int,
        weekly_parameters: Mapping[str, Any],
        daily_parameters: Mapping[str, Any],
    ) -> ForecastResult:
        current = points[position]
        weekly_weights = weekly_parameters["weights"]
        daily_weights = daily_parameters["weights"]
        daily_contribution = float(weekly_parameters.get("daily_contribution", 0.25))
        weekly_matrix, daily_matrix, global_indices, path_matrix = self._point_arrays(points)
        weekly_weight_vector = np.asarray(
            [float(weekly_weights[name]) / FEATURE_SCALES[name] for name in WEEKLY_FEATURES],
            dtype=float,
        )
        daily_weight_vector = np.asarray(
            [
                float(daily_weights[name])
                / FEATURE_SCALES[
                    "daily_ma20_distance" if name == "ma20_distance"
                    else "daily_macd_spread" if name == "macd_spread"
                    else name
                ]
                for name in DAILY_FEATURES
            ],
            dtype=float,
        )
        distances = (
            np.abs(weekly_matrix - weekly_matrix[position]) @ weekly_weight_vector
            + daily_contribution
            * (np.abs(daily_matrix - daily_matrix[position]) @ daily_weight_vector)
        )
        valid_mask = (
            (np.arange(len(points)) < position)
            & (global_indices + HORIZON_WEEKS <= current.global_index)
            & np.isfinite(path_matrix[:, 0])
        )
        valid_indices = np.flatnonzero(valid_mask)
        if len(valid_indices) < MIN_ANALOGS:
            raise V32TrainingError(
                f"{current.week_key}只有{len(valid_indices)}个已成熟历史情景，至少需要{MIN_ANALOGS}个"
            )
        ordered = valid_indices[np.argsort(distances[valid_indices], kind="stable")]
        selected = ordered[: int(weekly_parameters.get("analog_count", MAX_ANALOGS))]
        paths_array = path_matrix[selected]
        p10 = tuple(float(value) for value in np.quantile(paths_array, 0.10, axis=0))
        p50 = tuple(float(value) for value in np.quantile(paths_array, 0.50, axis=0))
        p90 = tuple(float(value) for value in np.quantile(paths_array, 0.90, axis=0))
        expected = tuple(float(value) for value in np.mean(paths_array, axis=0))
        volatility = max(float(current.weekly["volatility_13w"]), 0.01)
        threshold = float(weekly_parameters["base_threshold"]) * max(
            0.65,
            min(1.8, volatility / float(weekly_parameters["volatility_reference"])),
        )
        terminals = paths_array[:, -1]
        up = sum(value > threshold for value in terminals) / len(terminals)
        down = sum(value < -threshold for value in terminals) / len(terminals)
        sideways = max(0.0, 1.0 - up - down)
        direction = "up" if up >= max(down, sideways) else "down" if down >= sideways else "sideways"
        confidence = max(up, down, sideways) * 100.0
        expected_terminal = expected[-1]
        base_position = _snap5(50.0 + expected_terminal / max(threshold, 0.01) * 25.0)
        high_weeks = list(np.argmax(paths_array, axis=1) + 1)
        low_weeks = list(np.argmin(paths_array, axis=1) + 1)
        drawdowns = [_max_drawdown([1.0 + float(value) for value in path]) for path in paths_array]
        high_left, high_right = round(_percentile(high_weeks, 0.35)), round(_percentile(high_weeks, 0.65))
        low_left, low_right = round(_percentile(low_weeks, 0.35)), round(_percentile(low_weeks, 0.65))
        return ForecastResult(
            p10=p10,
            p50=p50,
            p90=p90,
            expected=expected,
            up_probability=up * 100.0,
            sideways_probability=sideways * 100.0,
            down_probability=down * 100.0,
            direction_threshold=threshold,
            direction=direction,
            confidence=confidence,
            base_target_position=base_position,
            expected_max_drawdown=_mean(drawdowns) * 100.0,
            high_week_range=f"第{max(1, high_left)}—{min(13, high_right)}周",
            low_week_range=f"第{max(1, low_left)}—{min(13, low_right)}周",
            scenario_count=len(paths_array),
        )

    def _point_arrays(
        self,
        points: Sequence[TrainingPoint],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = id(points)
        cached = self._matrix_cache.get(cache_key)
        if cached is not None:
            return cached
        weekly = np.asarray(
            [[float(point.weekly[name]) for name in WEEKLY_FEATURES] for point in points],
            dtype=float,
        )
        daily = np.asarray(
            [[float(point.daily[name]) for name in DAILY_FEATURES] for point in points],
            dtype=float,
        )
        global_indices = np.asarray([point.global_index for point in points], dtype=int)
        paths = np.full((len(points), HORIZON_WEEKS), np.nan, dtype=float)
        for index, point in enumerate(points):
            if point.future_path is not None:
                paths[index, :] = point.future_path
        cached = (weekly, daily, global_indices, paths)
        self._matrix_cache[cache_key] = cached
        return cached

    @staticmethod
    def _evaluate(point: TrainingPoint, forecast: ForecastResult) -> dict[str, Any] | None:
        actual = point.future_path
        if actual is None:
            return None
        scale = max(float(point.weekly["volatility_13w"]) / math.sqrt(4.0), 0.05)
        rmse = math.sqrt(_mean([(actual[index] - forecast.p50[index]) ** 2 for index in range(HORIZON_WEEKS)]))
        path_error = rmse / scale
        terminal_error = abs(actual[-1] - forecast.p50[-1]) / scale
        outside = _mean([
            max(forecast.p10[index] - actual[index], 0.0)
            + max(actual[index] - forecast.p90[index], 0.0)
            for index in range(HORIZON_WEEKS)
        ]) / scale
        coverage = sum(
            forecast.p10[index] <= actual[index] <= forecast.p90[index]
            for index in range(HORIZON_WEEKS)
        ) / HORIZON_WEEKS
        threshold = forecast.direction_threshold
        actual_direction = "up" if actual[-1] > threshold else "down" if actual[-1] < -threshold else "sideways"
        probabilities = {
            "up": forecast.up_probability / 100.0,
            "sideways": forecast.sideways_probability / 100.0,
            "down": forecast.down_probability / 100.0,
        }
        direction_score = sum(
            (probability - (1.0 if direction == actual_direction else 0.0)) ** 2
            for direction, probability in probabilities.items()
        ) / 3.0
        actual_high = max(range(HORIZON_WEEKS), key=lambda index: actual[index]) + 1
        actual_low = min(range(HORIZON_WEEKS), key=lambda index: actual[index]) + 1
        forecast_high = max(range(HORIZON_WEEKS), key=lambda index: forecast.p50[index]) + 1
        forecast_low = min(range(HORIZON_WEEKS), key=lambda index: forecast.p50[index]) + 1
        high_error = abs(actual_high - forecast_high)
        low_error = abs(actual_low - forecast_low)
        timing_error = (high_error + low_error) / (2.0 * HORIZON_WEEKS)
        actual_target_position = _snap5(
            50.0 + actual[-1] / max(forecast.direction_threshold, 0.01) * 25.0
        )
        position_error = abs(forecast.base_target_position - actual_target_position) / 100.0
        coverage_penalty = abs(coverage - 0.80)
        quantile_component = 0.7 * outside + 0.3 * coverage_penalty
        composite = (
            0.35 * path_error
            + 0.15 * quantile_component
            + 0.15 * terminal_error
            + 0.15 * direction_score
            + 0.10 * timing_error
            + 0.10 * position_error
        )
        return {
            "maturity_status": "full",
            "composite_loss": composite,
            "path_error": path_error,
            "terminal_return_error": terminal_error,
            "direction_score": direction_score,
            "quantile_loss": outside,
            "interval_coverage": coverage * 100.0,
            "high_week_error": high_error,
            "low_week_error": low_error,
            "position_error": position_error,
            "actual_target_position": actual_target_position,
            "actual_path": list(actual),
            "actual_direction": actual_direction,
        }

    def _candidate_parameters(
        self,
        weekly: Mapping[str, Any],
        daily: Mapping[str, Any],
        *,
        iteration_number: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_weekly = json.loads(json.dumps(weekly))
        candidate_daily = json.loads(json.dumps(daily))
        combined = [("weekly", name) for name in WEEKLY_FEATURES] + [
            ("daily", name) for name in DAILY_FEATURES
        ]
        model_type, selected = combined[(iteration_number - 1) % len(combined)]
        direction = 1.0 if ((iteration_number - 1) // len(combined)) % 2 == 0 else -1.0
        target = candidate_weekly if model_type == "weekly" else candidate_daily
        weights = dict(target["weights"])
        weights[selected] = max(0.01, weights[selected] + direction * 0.02)
        target["weights"] = _normalized(weights)
        return candidate_weekly, candidate_daily

    def _compare_candidate(
        self,
        points: Sequence[TrainingPoint],
        mature_positions: Sequence[int],
        current_weekly: Mapping[str, Any],
        current_daily: Mapping[str, Any],
        candidate_weekly: Mapping[str, Any],
        candidate_daily: Mapping[str, Any],
    ) -> dict[str, Any]:
        validation = list(mature_positions[-52:])
        if len(validation) < MIN_PROMOTION_SAMPLES:
            return {
                "promoted": False,
                "reason": "完整成熟样本不足52条",
                "sample_count": len(validation),
            }

        def score(weekly: Mapping[str, Any], daily: Mapping[str, Any], sample: Sequence[int]) -> dict[str, float]:
            evaluations = [
                self._evaluate(points[index], self._forecast(points, index, weekly, daily))
                for index in sample
            ]
            valid = [item for item in evaluations if item is not None]
            return {
                "composite_loss": _mean([item["composite_loss"] for item in valid]),
                "path_error": _mean([item["path_error"] for item in valid]),
                "direction_score": _mean([item["direction_score"] for item in valid]),
                "interval_coverage": _mean([item["interval_coverage"] for item in valid]),
            }

        current_52 = score(current_weekly, current_daily, validation)
        candidate_52 = score(candidate_weekly, candidate_daily, validation)
        recent_20 = validation[-20:]
        current_20 = score(current_weekly, current_daily, recent_20)
        candidate_20 = score(candidate_weekly, candidate_daily, recent_20)
        relative = (
            (current_52["composite_loss"] - candidate_52["composite_loss"])
            / current_52["composite_loss"]
            if current_52["composite_loss"]
            else 0.0
        )
        recent_relative = (
            (current_20["composite_loss"] - candidate_20["composite_loss"])
            / current_20["composite_loss"]
            if current_20["composite_loss"]
            else 0.0
        )
        promoted = (
            relative >= PROMOTION_IMPROVEMENT
            and recent_relative >= -0.005
            and candidate_52["direction_score"] <= current_52["direction_score"] + 0.02
            and abs(candidate_52["interval_coverage"] - 80.0)
            <= abs(current_52["interval_coverage"] - 80.0) + 3.0
        )
        reason = (
            f"52次综合损失改善{relative * 100:.2f}%，最近20次变化{recent_relative * 100:.2f}%，通过晋级门槛"
            if promoted
            else f"未通过：52次改善{relative * 100:.2f}%，最近20次变化{recent_relative * 100:.2f}%"
        )
        return {
            "promoted": promoted,
            "reason": reason,
            "sample_count": len(validation),
            "relative_improvement": relative,
            "recent_20_relative_improvement": recent_relative,
            "current_52": current_52,
            "candidate_52": candidate_52,
            "current_20": current_20,
            "candidate_20": candidate_20,
        }

    def _persist_snapshot(
        self,
        market: str,
        point: TrainingPoint,
        weekly_version: str,
        daily_version: str,
    ) -> str:
        payload = {
            "market": market,
            "cutoff_date": point.cutoff_date.isoformat(),
            "daily_as_of": point.daily_as_of.isoformat(),
            "valuation_as_of": None if point.valuation_as_of is None else point.valuation_as_of.isoformat(),
            "weekly_features": dict(point.weekly),
            "daily_features": dict(point.daily),
            "weekly_version": weekly_version,
            "daily_version": daily_version,
        }
        digest = _hash_json(payload)
        snapshot_id = f"V32-DS-{market}-{digest[:24]}"
        with self.sessions() as session, session.begin():
            if session.get(V32DataSnapshot, snapshot_id) is None:
                session.add(
                    V32DataSnapshot(
                        id=snapshot_id,
                        market=market,
                        cutoff_date=point.cutoff_date,
                        daily_data_as_of=point.daily_as_of,
                        weekly_data_as_of=point.cutoff_date,
                        valuation_data_as_of=point.valuation_as_of,
                        source_json={"price": point.price_source, "volume": point.volume_source},
                        payload_json=payload,
                        snapshot_hash=digest,
                    )
                )
        return snapshot_id

    @staticmethod
    def _snapshot_identity(
        market: str,
        point: TrainingPoint,
        weekly_version: str,
        daily_version: str,
    ) -> tuple[str, dict[str, Any], str]:
        payload = {
            "market": market,
            "cutoff_date": point.cutoff_date.isoformat(),
            "daily_as_of": point.daily_as_of.isoformat(),
            "valuation_as_of": None if point.valuation_as_of is None else point.valuation_as_of.isoformat(),
            "weekly_features": dict(point.weekly),
            "daily_features": dict(point.daily),
            "weekly_version": weekly_version,
            "daily_version": daily_version,
        }
        digest = _hash_json(payload)
        return f"V32-DS-{market}-{digest[:24]}", payload, digest

    def _persist_training_step(
        self,
        run_id: str,
        market: str,
        iteration_number: int,
        point: TrainingPoint,
        snapshot_id: str,
        snapshot_payload: Mapping[str, Any],
        snapshot_hash: str,
        weekly_version: str,
        daily_version: str,
        candidate: Mapping[str, Any] | None,
        promoted: bool,
        forecast: ForecastResult,
        evaluation: Mapping[str, Any] | None,
        optimizer_state: Mapping[str, Any],
    ) -> None:
        """Commit one iteration, snapshot, checkpoint, and progress atomically."""
        values = evaluation or {}
        candidate_version = (
            None
            if candidate is None
            else f"{weekly_version}:CANDIDATE:I{iteration_number:04d}"
        )
        audit = {
            "source_data_max_date": point.cutoff_date.isoformat(),
            "cutoff_date": point.cutoff_date.isoformat(),
            "future_data_used": False,
            "daily_window_sessions": 100,
            "parent_iteration": iteration_number - 1 if iteration_number > 1 else None,
            "methodology_version": METHODOLOGY_VERSION,
        }
        checkpoint_hash = _hash_json(optimizer_state)
        with self.sessions() as session, session.begin():
            if session.get(V32DataSnapshot, snapshot_id) is None:
                session.add(
                    V32DataSnapshot(
                        id=snapshot_id,
                        market=market,
                        cutoff_date=point.cutoff_date,
                        daily_data_as_of=point.daily_as_of,
                        weekly_data_as_of=point.cutoff_date,
                        valuation_data_as_of=point.valuation_as_of,
                        source_json={"price": point.price_source, "volume": point.volume_source},
                        payload_json=dict(snapshot_payload),
                        snapshot_hash=snapshot_hash,
                    )
                )
            session.add(
                V32TrainingIteration(
                    training_run_id=run_id,
                    market=market,
                    iteration_number=iteration_number,
                    parent_iteration_number=iteration_number - 1 if iteration_number > 1 else None,
                    week_key=point.week_key,
                    cutoff_date=point.cutoff_date,
                    status="complete",
                    maturity_status="full" if evaluation else "pending",
                    champion_weekly_version=weekly_version,
                    champion_daily_version=daily_version,
                    challenger_version=candidate_version,
                    promoted=promoted,
                    data_snapshot_id=snapshot_id,
                    input_features_json={"weekly": dict(point.weekly), "daily": dict(point.daily)},
                    forecast_json=self._forecast_payload(forecast),
                    evaluation_json={} if evaluation is None else dict(evaluation),
                    optimizer_state_json=dict(optimizer_state),
                    composite_loss=self._as_decimal(values.get("composite_loss")),
                    path_error=self._as_decimal(values.get("path_error")),
                    terminal_return_error=self._as_decimal(values.get("terminal_return_error")),
                    direction_score=self._as_decimal(values.get("direction_score")),
                    interval_coverage=self._as_decimal(values.get("interval_coverage")),
                    high_week_error=values.get("high_week_error"),
                    low_week_error=values.get("low_week_error"),
                    audit_json=audit,
                )
            )
            checkpoint = session.scalar(
                select(V32TrainingCheckpoint).where(
                    V32TrainingCheckpoint.training_run_id == run_id,
                    V32TrainingCheckpoint.market == market,
                )
            )
            if checkpoint is None:
                session.add(
                    V32TrainingCheckpoint(
                        training_run_id=run_id,
                        market=market,
                        iteration_number=iteration_number,
                        week_key=point.week_key,
                        state_json=dict(optimizer_state),
                        state_hash=checkpoint_hash,
                    )
                )
            else:
                checkpoint.iteration_number = iteration_number
                checkpoint.week_key = point.week_key
                checkpoint.state_json = dict(optimizer_state)
                checkpoint.state_hash = checkpoint_hash
                checkpoint.updated_at = utc_now()
            run = session.get(V32TrainingRun, run_id)
            if run is not None:
                run.created_iteration_count = iteration_number
                run.current_stage = f"训练模型 · {point.week_key}"

    def _persist_iteration(
        self,
        run_id: str,
        market: str,
        iteration_number: int,
        point: TrainingPoint,
        snapshot_id: str,
        weekly_version: str,
        daily_version: str,
        candidate: Mapping[str, Any] | None,
        promoted: bool,
        forecast: ForecastResult,
        evaluation: Mapping[str, Any] | None,
        optimizer_state: Mapping[str, Any],
    ) -> None:
        forecast_payload = self._forecast_payload(forecast)
        evaluation_payload = {} if evaluation is None else dict(evaluation)
        audit = {
            "source_data_max_date": point.cutoff_date.isoformat(),
            "cutoff_date": point.cutoff_date.isoformat(),
            "future_data_used": False,
            "daily_window_sessions": 100,
            "parent_iteration": iteration_number - 1 if iteration_number > 1 else None,
            "methodology_version": METHODOLOGY_VERSION,
        }
        candidate_version = None
        if candidate:
            candidate_version = f"{weekly_version}:CANDIDATE:I{iteration_number:04d}"
        values = evaluation or {}
        with self.sessions() as session, session.begin():
            session.add(
                V32TrainingIteration(
                    training_run_id=run_id,
                    market=market,
                    iteration_number=iteration_number,
                    parent_iteration_number=iteration_number - 1 if iteration_number > 1 else None,
                    week_key=point.week_key,
                    cutoff_date=point.cutoff_date,
                    status="complete",
                    maturity_status="full" if evaluation else "pending",
                    champion_weekly_version=weekly_version,
                    champion_daily_version=daily_version,
                    challenger_version=candidate_version,
                    promoted=promoted,
                    data_snapshot_id=snapshot_id,
                    input_features_json={"weekly": dict(point.weekly), "daily": dict(point.daily)},
                    forecast_json=forecast_payload,
                    evaluation_json=evaluation_payload,
                    optimizer_state_json=dict(optimizer_state),
                    composite_loss=self._as_decimal(values.get("composite_loss")),
                    path_error=self._as_decimal(values.get("path_error")),
                    terminal_return_error=self._as_decimal(values.get("terminal_return_error")),
                    direction_score=self._as_decimal(values.get("direction_score")),
                    interval_coverage=self._as_decimal(values.get("interval_coverage")),
                    high_week_error=values.get("high_week_error"),
                    low_week_error=values.get("low_week_error"),
                    audit_json=audit,
                )
            )

    def _persist_checkpoint(
        self,
        run_id: str,
        market: str,
        iteration_number: int,
        week_key: str,
        state: Mapping[str, Any],
    ) -> None:
        digest = _hash_json(state)
        with self.sessions() as session, session.begin():
            checkpoint = session.scalar(
                select(V32TrainingCheckpoint).where(
                    V32TrainingCheckpoint.training_run_id == run_id,
                    V32TrainingCheckpoint.market == market,
                )
            )
            if checkpoint is None:
                session.add(
                    V32TrainingCheckpoint(
                        training_run_id=run_id,
                        market=market,
                        iteration_number=iteration_number,
                        week_key=week_key,
                        state_json=dict(state),
                        state_hash=digest,
                    )
                )
            else:
                checkpoint.iteration_number = iteration_number
                checkpoint.week_key = week_key
                checkpoint.state_json = dict(state)
                checkpoint.state_hash = digest
                checkpoint.updated_at = utc_now()

    def _save_optimizer_state(
        self,
        market: str,
        iteration_number: int,
        week_key: str,
        weekly_version: str,
        daily_version: str,
        weekly_parameters: Mapping[str, Any],
        daily_parameters: Mapping[str, Any],
        memory: Mapping[str, Any],
        trained_at: datetime,
    ) -> None:
        eligible_at = trained_at + timedelta(days=7)
        payload = {
            "market": market,
            "iteration_number": iteration_number,
            "week_key": week_key,
            "weekly_version": weekly_version,
            "daily_version": daily_version,
            "weekly_parameters": weekly_parameters,
            "daily_parameters": daily_parameters,
            "memory": memory,
        }
        digest = _hash_json(payload)
        with self.sessions() as session, session.begin():
            state = session.scalar(
                select(V32OptimizerState).where(V32OptimizerState.market == market)
            )
            if state is None:
                session.add(
                    V32OptimizerState(
                        market=market,
                        iteration_number=iteration_number,
                        last_training_week_key=week_key,
                        champion_weekly_version=weekly_version,
                        champion_daily_version=daily_version,
                        weekly_parameters_json=dict(weekly_parameters),
                        daily_parameters_json=dict(daily_parameters),
                        optimizer_memory_json=dict(memory),
                        last_successful_training_at=trained_at,
                        next_training_eligible_at=eligible_at,
                        state_hash=digest,
                    )
                )
            else:
                state.iteration_number = iteration_number
                state.last_training_week_key = week_key
                state.champion_weekly_version = weekly_version
                state.champion_daily_version = daily_version
                state.weekly_parameters_json = dict(weekly_parameters)
                state.daily_parameters_json = dict(daily_parameters)
                state.optimizer_memory_json = dict(memory)
                state.last_successful_training_at = trained_at
                state.next_training_eligible_at = eligible_at
                state.state_hash = digest
                state.updated_at = utc_now()

    def _create_model_versions(
        self,
        market: str,
        cutoff: date,
        weekly_version: str,
        daily_version: str,
        weekly_parameters: Mapping[str, Any],
        daily_parameters: Mapping[str, Any],
        *,
        parent_weekly: str | None,
        parent_daily: str | None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        prefix_seed = 39900632 if market == "399006" else 10032
        with self.sessions() as session, session.begin():
            for model_type, version, parent, parameters, seed in (
                ("weekly", weekly_version, parent_weekly, weekly_parameters, prefix_seed),
                ("daily", daily_version, parent_daily, daily_parameters, prefix_seed + 1),
            ):
                digest = _hash_json(parameters)
                session.add(
                    V32ModelVersion(
                        id=f"{market}:{model_type}:{version}",
                        market=market,
                        model_type=model_type,
                        version=version,
                        parent_version=parent,
                        feature_version=FEATURE_VERSION,
                        methodology_version=METHODOLOGY_VERSION,
                        trained_through_date=cutoff,
                        validation_start_date=None,
                        validation_end_date=cutoff,
                        random_seed=seed,
                        parameters_json=dict(parameters),
                        metrics_json=dict(metrics or {"validation_sample_count": 0}),
                        artifact_hash=digest,
                        status="champion",
                    )
                )

    def _supersede_and_create_models(
        self,
        market: str,
        cutoff: date,
        old_weekly: str,
        old_daily: str,
        new_weekly: str,
        new_daily: str,
        weekly_parameters: Mapping[str, Any],
        daily_parameters: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> None:
        with self.sessions() as session, session.begin():
            rows = list(
                session.scalars(
                    select(V32ModelVersion).where(
                        V32ModelVersion.market == market,
                        V32ModelVersion.status == "champion",
                    )
                )
            )
            for row in rows:
                row.status = "superseded"
        self._create_model_versions(
            market,
            cutoff,
            new_weekly,
            new_daily,
            weekly_parameters,
            daily_parameters,
            parent_weekly=old_weekly,
            parent_daily=old_daily,
            metrics=metrics,
        )

    def _create_run(
        self,
        run_id: str,
        market: str,
        run_type: str,
        through_week: str,
        *,
        source_parent_iteration: int | None = None,
    ) -> None:
        with self.sessions() as session, session.begin():
            session.add(
                V32TrainingRun(
                    id=run_id,
                    market=market,
                    run_type=run_type,
                    status="running",
                    current_stage="训练模型",
                    requested_through_week=through_week,
                    source_parent_iteration=source_parent_iteration,
                    stages_json=[{"stage": "训练模型", "status": "running", "at": utc_now().isoformat()}],
                )
            )

    def _mark_interrupted_runs(self, market: str) -> None:
        with self.sessions() as session, session.begin():
            rows = list(
                session.scalars(
                    select(V32TrainingRun).where(
                        V32TrainingRun.market == market,
                        V32TrainingRun.status == "running",
                    )
                )
            )
            for row in rows:
                row.status = "failed"
                row.current_stage = "已由检查点恢复"
                row.completed_at = utc_now()
                row.error_code = "INTERRUPTED_RECOVERED"
                row.error_message = "进程中断；后续训练已从最后完整检查点继续"

    def _update_run_progress(self, run_id: str, count: int, week_key: str) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V32TrainingRun, run_id)
            if run is not None:
                run.created_iteration_count = count
                run.current_stage = f"训练模型 · {week_key}"

    def _complete_run(self, run_id: str, result: Mapping[str, Any]) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V32TrainingRun, run_id)
            if run is None:
                return
            run.status = "completed"
            run.current_stage = "已完成"
            run.completed_at = utc_now()
            run.result_json = dict(result)
            run.stages_json = [
                *list(run.stages_json or []),
                {"stage": "已完成", "status": "completed", "at": utc_now().isoformat()},
            ]

    def _fail_run(self, run_id: str, error: Exception) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V32TrainingRun, run_id)
            if run is None:
                return
            run.status = "failed"
            run.current_stage = "执行失败"
            run.completed_at = utc_now()
            run.error_code = type(error).__name__
            run.error_message = str(error)

    @staticmethod
    def _forecast_payload(forecast: ForecastResult) -> dict[str, Any]:
        return {
            "p10": list(forecast.p10),
            "p50": list(forecast.p50),
            "p90": list(forecast.p90),
            "expected": list(forecast.expected),
            "up_probability": forecast.up_probability,
            "sideways_probability": forecast.sideways_probability,
            "down_probability": forecast.down_probability,
            "direction_threshold": forecast.direction_threshold,
            "direction": forecast.direction,
            "confidence": forecast.confidence,
            "base_target_position": forecast.base_target_position,
            "expected_max_drawdown": forecast.expected_max_drawdown,
            "high_week_range": forecast.high_week_range,
            "low_week_range": forecast.low_week_range,
            "scenario_count": forecast.scenario_count,
        }

    @staticmethod
    def _forecast_from_payload(payload: Mapping[str, Any]) -> ForecastResult:
        """Rehydrate the exact forecast saved by an earlier iteration."""
        try:
            p10 = tuple(float(value) for value in payload["p10"])
            p50 = tuple(float(value) for value in payload["p50"])
            p90 = tuple(float(value) for value in payload["p90"])
            expected = tuple(float(value) for value in payload["expected"])
            if not all(
                len(path) == HORIZON_WEEKS for path in (p10, p50, p90, expected)
            ):
                raise ValueError("forecast paths must each contain 13 weeks")
            return ForecastResult(
                p10=p10,
                p50=p50,
                p90=p90,
                expected=expected,
                up_probability=float(payload["up_probability"]),
                sideways_probability=float(payload["sideways_probability"]),
                down_probability=float(payload["down_probability"]),
                direction_threshold=float(payload["direction_threshold"]),
                direction=str(payload["direction"]),
                confidence=float(payload["confidence"]),
                base_target_position=int(payload["base_target_position"]),
                expected_max_drawdown=float(payload["expected_max_drawdown"]),
                high_week_range=str(payload["high_week_range"]),
                low_week_range=str(payload["low_week_range"]),
                scenario_count=int(payload["scenario_count"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise V32TrainingError(f"Invalid saved V3.2 forecast payload: {error}") from error

    @staticmethod
    def _version_name(market: str, model_type: str, number: int) -> str:
        prefix = "CYB" if market == "399006" else "NDX"
        label = "WEEKLY" if model_type == "weekly" else "DAILY_CORRECTOR"
        return f"{prefix}_{label}_V3.2.{number}"

    @staticmethod
    def _next_version(version: str) -> str:
        prefix, number = version.rsplit(".", 1)
        return f"{prefix}.{int(number) + 1}"

    @staticmethod
    def _week_distance(left: str, right: str) -> int:
        def monday(key: str) -> date:
            year, week = key.split("-W")
            return date.fromisocalendar(int(year), int(week), 1)
        return max(1, (monday(right) - monday(left)).days // 7)

    @staticmethod
    def _validate_market(market: str) -> None:
        if market not in SUPPORTED_MARKETS:
            raise V32TrainingError("V3.2仅支持399006和NDX")

    @staticmethod
    def _as_decimal(value: Any) -> Decimal | None:
        return None if value is None else Decimal(str(round(float(value), 8)))

    @staticmethod
    def _decimal(value: Decimal | None) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _run_payload(run: V32TrainingRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "market": run.market,
            "run_type": run.run_type,
            "status": run.status,
            "current_stage": run.current_stage,
            "requested_through_week": run.requested_through_week,
            "created_iteration_count": run.created_iteration_count,
            "started_at": run.started_at.isoformat(),
            "completed_at": None if run.completed_at is None else run.completed_at.isoformat(),
            "error_code": run.error_code,
            "error_message": run.error_message,
            "result": run.result_json,
        }
