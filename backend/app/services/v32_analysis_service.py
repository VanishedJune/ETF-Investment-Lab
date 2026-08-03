"""Pure-inference V3.2 current-market analysis using the latest Champion."""

from __future__ import annotations

import math
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Mapping
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ..models.models import (
    V32AnalysisRun,
    V32DailyCorrection,
    V32WeeklyForecast,
    utc_now,
)
from .investment_calendar_service import InvestmentCalendarService
from .market_calendar import CalendarProvider
from .v31_analysis_service import DataGateBlocked, V31AnalysisService
from .v32_training_service import (
    ForecastResult,
    TrainingPoint,
    V32TrainingError,
    V32TrainingService,
    _snap5,
)


class V32AnalysisService:
    """Refresh/gate data, then perform inference without mutating model state."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        training: V32TrainingService,
        v31_data_pipeline: V31AnalysisService,
        calendar: CalendarProvider,
        investment_calendar: InvestmentCalendarService,
    ) -> None:
        self.sessions = sessions
        self.training = training
        self.v31_data_pipeline = v31_data_pipeline
        self.calendar = calendar
        self.investment_calendar = investment_calendar

    def run(self, market: str, *, refresh: bool = True) -> dict[str, Any]:
        self.training._validate_market(market)
        run_id = f"V32-ANALYSIS-{market}-{uuid4().hex[:16]}"
        self._create_run(run_id, market)
        context: dict[str, Any] = {}
        try:
            self._stage(run_id, "刷新行情")
            context.update(self.v31_data_pipeline._refresh(market, refresh))
            self._stage(run_id, "校验数据")
            gate = self.v31_data_pipeline._data_gate(market, context)
            champion = self.training.champion(market)
            points, _ = self.training._load_points(market)
            point = points[-1]
            if point.cutoff_date != gate.complete_week_as_of:
                raise DataGateBlocked(
                    "MODEL_WEEK_MISMATCH",
                    f"训练特征最新完整周{point.cutoff_date}与数据门禁{gate.complete_week_as_of}不一致",
                )
            self._stage(run_id, "加载Champion")
            iteration_count_before = champion["iteration_number"]
            self._stage(run_id, "计算周K信号")
            forecast = self.training._forecast(
                points,
                len(points) - 1,
                champion["weekly_parameters"],
                champion["daily_parameters"],
            )
            weekly = self._weekly_payload(point, forecast)
            self._stage(run_id, "计算日K修正")
            daily = self._daily_correction(
                market,
                point,
                forecast,
                champion["daily_parameters"],
            )
            self._stage(run_id, "生成概率路径")
            path = self._path_payload(forecast)
            self._stage(run_id, "读取当前仓位")
            position = self._position(market)
            self._stage(run_id, "生成投资建议")
            advice = self._advice(market, point, forecast, daily, position)
            snapshot_id = self.training._persist_snapshot(
                market,
                point,
                champion["weekly_version"],
                champion["daily_version"],
            )
            forecast_id = self._persist_forecast(
                market,
                point,
                champion,
                forecast,
                daily,
                snapshot_id,
            )
            champion_after = self.training.champion(market)
            if champion_after["iteration_number"] != iteration_count_before:
                # A concurrent training task may promote after inference began.
                # The run remains tied to the captured model and never mixes it.
                context["concurrent_model_update"] = {
                    "captured_iteration": iteration_count_before,
                    "latest_iteration": champion_after["iteration_number"],
                }
            result = {
                "implementation_revision": "V3.2.0",
                "run_id": run_id,
                "market": market,
                "training_mutated": False,
                "model": {
                    "weekly": {
                        "version": champion["weekly_version"],
                        "iteration_number": iteration_count_before,
                    },
                    "daily": {"version": champion["daily_version"], "window": 100},
                },
                "freshness": self.v31_data_pipeline._gate_payload(gate),
                "weekly": weekly,
                "daily": daily,
                "path": path,
                "position": position,
                "advice": advice,
                "data_snapshot_id": snapshot_id,
                "forecast_id": forecast_id,
                "concurrent_model_update": context.get("concurrent_model_update"),
            }
            self._complete(run_id, result, gate, champion["weekly_version"], snapshot_id)
            return result
        except Exception as error:
            self._fail(run_id, error, context)
            raise

    def latest(self, market: str) -> dict[str, Any] | None:
        self.training._validate_market(market)
        with self.sessions() as session:
            row = session.scalar(
                select(V32AnalysisRun)
                .where(V32AnalysisRun.market == market)
                .order_by(V32AnalysisRun.started_at.desc())
            )
        return None if row is None else self._run_payload(row)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            row = session.get(V32AnalysisRun, run_id)
        if row is None:
            raise KeyError(f"未知V3.2分析任务：{run_id}")
        return self._run_payload(row)

    @staticmethod
    def _weekly_payload(point: TrainingPoint, forecast: ForecastResult) -> dict[str, Any]:
        probability = max(
            forecast.up_probability,
            forecast.sideways_probability,
            forecast.down_probability,
        )
        state = (
            "强看多" if forecast.direction == "up" and probability >= 65
            else "中度看多" if forecast.direction == "up"
            else "强看空" if forecast.direction == "down" and probability >= 65
            else "中度看空" if forecast.direction == "down"
            else "中性震荡"
        )
        valuation = point.weekly["valuation_percentile"]
        if valuation >= 90 and forecast.direction == "up":
            market_state = "高估但趋势仍强"
        elif valuation >= 90:
            market_state = "高估且动能减弱"
        elif valuation <= 15 and forecast.direction == "up":
            market_state = "低估反转"
        elif valuation <= 15:
            market_state = "低估但趋势仍弱"
        else:
            market_state = "合理估值上涨" if forecast.direction == "up" else "合理估值震荡" if forecast.direction == "sideways" else "下跌风险"
        return {
            "direction": forecast.direction,
            "confidence": round(forecast.confidence, 1),
            "state": state,
            "market_state": market_state,
            "base_target_position": forecast.base_target_position,
            "features": dict(point.weekly),
            "positive_factors": [
                f"13周上涨情景概率{forecast.up_probability:.1f}%",
                f"历史相似情景{forecast.scenario_count}组",
            ],
            "risk_factors": [
                f"13周下跌情景概率{forecast.down_probability:.1f}%",
                f"估值历史分位{valuation:.1f}%",
            ],
        }

    def _daily_correction(
        self,
        market: str,
        point: TrainingPoint,
        forecast: ForecastResult,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        features = dict(point.daily)
        rsi = float(features["rsi"])
        overheat = rsi >= float(parameters.get("rsi_overbought", 70)) or features["ma20_distance"] >= 0.07
        oversold = rsi <= float(parameters.get("rsi_oversold", 30)) or features["ma20_distance"] <= -0.07
        daily_direction = (
            "up" if features["macd_spread"] > 0 and features["ma20_distance"] > 0
            else "down" if features["macd_spread"] < 0 and features["ma20_distance"] < 0
            else "sideways"
        )
        consistent = daily_direction in (forecast.direction, "sideways")
        confidence_adjustment = 3 if consistent and daily_direction != "sideways" else -4 if not consistent else 0
        position_adjustment = 0
        state = "节奏中性"
        if overheat:
            state, position_adjustment = "短线过热", -5
            confidence_adjustment = min(confidence_adjustment, -3)
        elif oversold:
            state, position_adjustment = "短线超跌", 5
            confidence_adjustment = max(confidence_adjustment, 2) if forecast.direction == "up" else min(confidence_adjustment, -2)
        if forecast.direction == "up" and daily_direction == "down":
            state, position_adjustment = "周线看多、日线等待确认", min(position_adjustment, -5)
        elif forecast.direction == "down" and daily_direction == "up":
            state, position_adjustment = "周线看空、日线超跌反弹", max(position_adjustment, 5)
        confidence_adjustment = max(-10, min(10, int(confidence_adjustment)))
        position_adjustment = max(-10, min(10, int(position_adjustment)))
        sessions = self.calendar.sessions(
            market,
            point.daily_as_of + timedelta(days=1),
            point.daily_as_of + timedelta(days=20),
        )
        window = sessions[:3]
        trigger = "日K收盘站上20日均线且DIF不低于DEA" if forecast.direction == "up" else "日K收盘跌破20日均线且DIF不高于DEA" if forecast.direction == "down" else "周K方向突破震荡区间后再执行"
        invalidation = "日K跌破20日均线且DIF转弱" if forecast.direction == "up" else "日K重新站上20日均线且DIF转强" if forecast.direction == "down" else "周K继续维持震荡"
        return {
            "window_sessions": 100,
            "state": state,
            "direction": daily_direction,
            "overheat": overheat,
            "oversold": oversold,
            "consistent_with_weekly": consistent,
            "confidence_adjustment": confidence_adjustment,
            "position_adjustment": position_adjustment,
            "execution_speed": "slow" if not consistent or overheat or oversold else "normal",
            "first_execution_window": [
                None if not window else window[0].isoformat(),
                None if not window else window[-1].isoformat(),
            ],
            "trigger_conditions": [trigger],
            "invalidation_conditions": [invalidation],
            "features": features,
        }

    @staticmethod
    def _path_payload(forecast: ForecastResult) -> dict[str, Any]:
        return {
            "points": [
                {
                    "horizon_week": index + 1,
                    "p10_cumulative_return": round(forecast.p10[index] * 100, 4),
                    "p50_cumulative_return": round(forecast.p50[index] * 100, 4),
                    "p90_cumulative_return": round(forecast.p90[index] * 100, 4),
                    "expected_cumulative_return": round(forecast.expected[index] * 100, 4),
                }
                for index in range(13)
            ],
            "up_probability": round(forecast.up_probability, 2),
            "sideways_probability": round(forecast.sideways_probability, 2),
            "down_probability": round(forecast.down_probability, 2),
            "direction_threshold": round(forecast.direction_threshold * 100, 4),
            "expected_max_drawdown": round(forecast.expected_max_drawdown, 4),
            "high_week_range": forecast.high_week_range,
            "low_week_range": forecast.low_week_range,
            "scenario_count": forecast.scenario_count,
            "method": "V3.2训练权重历史相似情景路径",
        }

    def _position(self, market: str) -> dict[str, Any]:
        positions = self.investment_calendar.current_positions()
        entries = self.investment_calendar.list_entries(market)
        confirmed = int(positions.get(market, 0))
        return {
            "confirmed_position": confirmed,
            "in_transit_adjustment": 0,
            "settlement_position": confirmed,
            "advice_basis_position": confirmed,
            "defaulted_to_zero": not bool(entries),
        }

    def _advice(
        self,
        market: str,
        point: TrainingPoint,
        forecast: ForecastResult,
        daily: Mapping[str, Any],
        position: Mapping[str, Any],
    ) -> dict[str, Any]:
        current = int(position["advice_basis_position"])
        target = _snap5(forecast.base_target_position + int(daily["position_adjustment"]))
        delta = target - current
        direction = "increase" if delta > 0 else "decrease" if delta < 0 else "hold"
        remaining = abs(delta)
        batch_count = min(4, max(1, math.ceil(remaining / 20))) if remaining else 0
        amounts: list[int] = []
        for index in range(batch_count):
            slots = batch_count - index
            amount = _snap5(remaining / slots)
            amount = max(5, min(remaining, amount))
            amounts.append(amount)
            remaining -= amount
        if remaining and amounts:
            amounts[-1] += remaining
        sessions = list(
            self.calendar.sessions(
                market,
                point.daily_as_of + timedelta(days=1),
                point.daily_as_of + timedelta(days=120),
            )
        )
        batches: list[dict[str, Any]] = []
        running = current
        for index, amount in enumerate(amounts):
            start_index = min(index * 3, max(0, len(sessions) - 1))
            window = sessions[start_index : start_index + 3]
            running += amount if direction == "increase" else -amount
            batches.append(
                {
                    "sequence": index + 1,
                    "direction": direction,
                    "position_points": amount,
                    "execution_window": [
                        None if not window else window[0].isoformat(),
                        None if not window else window[-1].isoformat(),
                    ],
                    "trigger": daily["trigger_conditions"][0],
                    "invalidation": daily["invalidation_conditions"][0],
                    "position_after": running,
                }
            )
        expected_terminal = forecast.expected[-1]
        ratio = "7:3" if expected_terminal >= 0.10 else "6:4" if expected_terminal >= 0.03 else "5:5" if expected_terminal > -0.03 else "4:6" if expected_terminal > -0.10 else "3:7"
        return {
            "current_position": current,
            "weekly_base_target_position": forecast.base_target_position,
            "daily_position_adjustment": daily["position_adjustment"],
            "final_target_position": target,
            "direction": direction,
            "total_change_points": abs(delta),
            "batches": batches,
            "fund_etf_ratio": ratio,
            "position_unit": "占总可投资资金的百分点",
            "review_rule": "下一个完整交易周结束后复核；触发条件未满足则不执行",
        }

    def _persist_forecast(
        self,
        market: str,
        point: TrainingPoint,
        champion: Mapping[str, Any],
        forecast: ForecastResult,
        daily: Mapping[str, Any],
        snapshot_id: str,
    ) -> int:
        with self.sessions() as session, session.begin():
            row = session.scalar(
                select(V32WeeklyForecast).where(
                    V32WeeklyForecast.market == market,
                    V32WeeklyForecast.forecast_date == point.cutoff_date,
                    V32WeeklyForecast.model_version == champion["weekly_version"],
                )
            )
            if row is None:
                row = V32WeeklyForecast(
                    market=market,
                    training_iteration_id=None,
                    forecast_date=point.cutoff_date,
                    model_version=champion["weekly_version"],
                    data_snapshot_id=snapshot_id,
                    weekly_direction=forecast.direction,
                    weekly_confidence=Decimal(str(round(forecast.confidence, 8))),
                    base_target_position=forecast.base_target_position,
                    p10_path_json=list(forecast.p10),
                    p50_path_json=list(forecast.p50),
                    p90_path_json=list(forecast.p90),
                    expected_path_json=list(forecast.expected),
                    up_probability=Decimal(str(round(forecast.up_probability, 8))),
                    sideways_probability=Decimal(str(round(forecast.sideways_probability, 8))),
                    down_probability=Decimal(str(round(forecast.down_probability, 8))),
                    expected_max_drawdown=Decimal(str(round(forecast.expected_max_drawdown, 8))),
                    high_week_range=forecast.high_week_range,
                    low_week_range=forecast.low_week_range,
                    maturity_status="pending",
                    payload_json={"scenario_count": forecast.scenario_count},
                )
                session.add(row)
                session.flush()
            correction = session.scalar(
                select(V32DailyCorrection).where(
                    V32DailyCorrection.weekly_forecast_id == row.id
                )
            )
            window = daily["first_execution_window"]
            values = {
                "correction_date": point.daily_as_of,
                "model_version": champion["daily_version"],
                "daily_state": daily["state"],
                "confidence_adjustment": daily["confidence_adjustment"],
                "position_adjustment": daily["position_adjustment"],
                "execution_window_start": None if window[0] is None else date.fromisoformat(window[0]),
                "execution_window_end": None if window[1] is None else date.fromisoformat(window[1]),
                "trigger_conditions_json": daily["trigger_conditions"],
                "invalidation_conditions_json": daily["invalidation_conditions"],
                "payload_json": daily["features"],
            }
            if correction is None:
                session.add(V32DailyCorrection(weekly_forecast_id=row.id, **values))
            else:
                for key, value in values.items():
                    setattr(correction, key, value)
            return row.id

    def _create_run(self, run_id: str, market: str) -> None:
        with self.sessions() as session, session.begin():
            session.add(
                V32AnalysisRun(
                    id=run_id,
                    market=market,
                    status="running",
                    current_stage="等待执行",
                    data_gate_status="pending",
                    stages_json=[],
                )
            )

    def _stage(self, run_id: str, stage: str) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V32AnalysisRun, run_id)
            if run is not None:
                run.current_stage = stage
                run.stages_json = [
                    *list(run.stages_json or []),
                    {"stage": stage, "status": "completed", "at": utc_now().isoformat()},
                ]

    def _complete(self, run_id: str, result: Mapping[str, Any], gate: Any, model: str, snapshot: str) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V32AnalysisRun, run_id)
            if run is None:
                return
            run.status = "completed"
            run.current_stage = "已完成"
            run.completed_at = utc_now()
            run.model_version = model
            run.data_snapshot_id = snapshot
            run.price_data_as_of = gate.price_as_of
            run.weekly_data_as_of = gate.complete_week_as_of
            run.valuation_data_as_of = gate.valuation_as_of
            run.data_gate_status = "degraded" if gate.degraded else "passed"
            run.result_json = dict(result)

    def _fail(self, run_id: str, error: Exception, context: Mapping[str, Any]) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V32AnalysisRun, run_id)
            if run is None:
                return
            run.status = "failed"
            run.current_stage = "执行失败"
            run.completed_at = utc_now()
            run.data_gate_status = "blocked"
            run.error_code = getattr(error, "code", type(error).__name__)
            run.error_message = str(error)
            run.result_json = {
                "failure_stage": run.current_stage,
                "failure_reason": str(error),
                "data_source": context.get("market_refresh", {}).get("source") if isinstance(context.get("market_refresh"), dict) else None,
                "run_id": run_id,
            }

    @staticmethod
    def _run_payload(run: V32AnalysisRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "market": run.market,
            "status": run.status,
            "current_stage": run.current_stage,
            "started_at": run.started_at.isoformat(),
            "completed_at": None if run.completed_at is None else run.completed_at.isoformat(),
            "model_version": run.model_version,
            "data_snapshot_id": run.data_snapshot_id,
            "price_data_as_of": None if run.price_data_as_of is None else run.price_data_as_of.isoformat(),
            "weekly_data_as_of": None if run.weekly_data_as_of is None else run.weekly_data_as_of.isoformat(),
            "valuation_data_as_of": None if run.valuation_data_as_of is None else run.valuation_data_as_of.isoformat(),
            "data_gate_status": run.data_gate_status,
            "stages": run.stages_json,
            "result": run.result_json,
            "error_code": run.error_code,
            "error_message": run.error_message,
        }
