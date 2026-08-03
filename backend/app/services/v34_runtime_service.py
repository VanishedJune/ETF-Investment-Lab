"""SQLite orchestration for the isolated V3.4 13-week model chain."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from threading import Lock
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.daemon_executor import DaemonThreadPoolExecutor
from backend.app.models.models import (
    Instrument,
    V34AnalysisRun,
    V34FeatureSnapshot as V34FeatureSnapshotRow,
    V34Forecast as V34ForecastRow,
    V34ModelEvaluation,
    V34ModelVersion,
    V34OptimizerState,
    V34ProbabilityCalibrator,
    V34ScenarioAdapter,
    V34TrainingIteration,
    V34TrainingRun,
    utc_now,
)
from backend.app.services.market_calendar import CalendarProvider
from backend.app.services.v33_feature_service import FeatureSnapshot
from backend.app.services.v34_feature_service import (
    FEATURE_VERSION,
    SCALER_VERSION,
    SUPPORTED_MARKETS,
    V34FeatureService,
)
from backend.app.services.v34_scenario_service import (
    SCENARIO_ADAPTER_VERSION,
    V34ScenarioForecast,
    V34ScenarioService,
)
from backend.app.services.v34_training_service import (
    HORIZON_WEEKS,
    MIN_NEW_MATURED_SAMPLES_FOR_RETRAIN,
    METHODOLOGY_VERSION,
    V34ModelState,
    V34PathForecast,
    V34PromotionDecision,
    V34TrainingError,
    V34TrainingSample,
    V34TrainingService,
    build_training_samples,
    effective_sample_count,
    eligible_fully_matured,
    maturity_status,
    overlap_weights,
)


BOOTSTRAP_DAYS = 3653
WARMUP_WEEKS = 65


class V34RuntimeError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(float(value), 8)))


def state_to_payload(state: V34ModelState) -> dict[str, Any]:
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
        "residual_paths": [list(row) for row in state.residual_paths],
        "hyperparameters": dict(state.hyperparameters),
        "raw_matured_sample_count": state.raw_matured_sample_count,
        "effective_independent_sample_count": state.effective_independent_sample_count,
        "validation_metrics": _jsonable(state.validation_metrics),
        "feature_fit_end_date": state.feature_fit_end_date.isoformat(),
        "parameter_hash": state.parameter_hash,
    }


def state_from_payload(payload: Mapping[str, Any]) -> V34ModelState:
    return V34ModelState(
        market=str(payload["market"]),
        version=str(payload["version"]),
        parent_version=None if payload.get("parent_version") is None else str(payload["parent_version"]),
        trained_through_date=date.fromisoformat(str(payload["trained_through_date"])),
        effective_from_date=date.fromisoformat(str(payload["effective_from_date"])),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        medians=tuple(float(value) for value in payload["medians"]),
        means=tuple(float(value) for value in payload["means"]),
        scales=tuple(float(value) for value in payload["scales"]),
        intercept=tuple(float(value) for value in payload["intercept"]),
        coefficients=tuple(tuple(float(value) for value in row) for row in payload["coefficients"]),
        residual_paths=tuple(tuple(float(value) for value in row) for row in payload["residual_paths"]),
        hyperparameters=dict(payload["hyperparameters"]),
        raw_matured_sample_count=int(payload["raw_matured_sample_count"]),
        effective_independent_sample_count=float(payload["effective_independent_sample_count"]),
        validation_metrics=dict(payload.get("validation_metrics") or {}),
        feature_fit_end_date=date.fromisoformat(str(payload["feature_fit_end_date"])),
        parameter_hash=str(payload["parameter_hash"]),
    )


class V34RuntimeService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        calendar: CalendarProvider,
        current_position_provider: Callable[[str], int | None] | None = None,
        now_provider: Callable[[], datetime] = utc_now,
        scenario_count: int = 1200,
    ) -> None:
        self.sessions = sessions
        self.calendar = calendar
        self.current_position_provider = current_position_provider or (lambda _market: 0)
        self.now_provider = now_provider
        self.features = V34FeatureService()
        self.training = V34TrainingService()
        self.scenarios = V34ScenarioService(scenario_count=scenario_count)
        self._executor = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="v34-model")
        self._locks = {market: Lock() for market in SUPPORTED_MARKETS}

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop accepting V3.4 work using the common application lifecycle API."""
        self._executor.shutdown(wait=wait, cancel_futures=False)

    @staticmethod
    def validate_market(market: str) -> None:
        if market not in SUPPORTED_MARKETS:
            raise V34RuntimeError("V3.4 supports only 399006 and 159941")

    def recover_interrupted_runs(self) -> int:
        with self.sessions() as session, session.begin():
            rows = list(
                session.scalars(
                    select(V34TrainingRun).where(V34TrainingRun.status.in_(("queued", "running")))
                )
            )
            for row in rows:
                row.status = "failed"
                row.current_stage = "recovered_after_interruption"
                row.completed_at = self.now_provider()
                row.error_code = "INTERRUPTED"
                row.error_message = "application stopped before the task completed"
            return len(rows)

    def _instrument_id(self, session: Session, market: str) -> int:
        instrument = session.scalar(select(Instrument).where(Instrument.code == market))
        if instrument is None:
            raise V34RuntimeError(f"missing instrument {market}")
        return int(instrument.id)

    def _complete_anchors(self, market: str) -> tuple[date, ...]:
        now = self.now_provider()
        today = now.date()
        with self.sessions() as session:
            anchors = self.features.weekly_anchors(session, market, end_date=today)
        actual = set(anchors)
        complete: list[date] = []
        # weekly_anchors has only each observed week's last session; verify it
        # equals the exchange calendar's final session and that the week ended.
        for observed in anchors:
            year, week, _ = observed.isocalendar()
            monday = date.fromisocalendar(year, week, 1)
            expected = tuple(self.calendar.sessions(market, monday, monday + timedelta(days=6)))
            if expected and max(expected) == observed and observed <= today:
                complete.append(observed)
            elif not expected and observed.weekday() >= 4 and observed <= today:
                complete.append(observed)
        if not complete:
            raise V34RuntimeError(f"no complete weekly anchors for {market}")
        return tuple(complete)

    def _snapshots(self, market: str, anchors: Sequence[date]) -> tuple[FeatureSnapshot, ...]:
        snapshots: list[FeatureSnapshot] = []
        with self.sessions() as session:
            for anchor in anchors:
                try:
                    snapshots.append(self.features.load_snapshot(session, market, anchor))
                except Exception:
                    if anchor >= anchors[-1] - timedelta(days=BOOTSTRAP_DAYS):
                        raise
        return tuple(snapshots)

    def _persist_snapshot(self, snapshot: FeatureSnapshot) -> str:
        payload = self.features.audit_payload(snapshot)
        snapshot_id = f"V34-FS-{snapshot.market}-{snapshot.cutoff_date}-{payload['snapshot_hash'][:10]}"
        with self.sessions() as session, session.begin():
            existing = session.scalar(
                select(V34FeatureSnapshotRow).where(
                    V34FeatureSnapshotRow.model_market == snapshot.market,
                    V34FeatureSnapshotRow.forecast_anchor_date == snapshot.cutoff_date,
                    V34FeatureSnapshotRow.feature_version == FEATURE_VERSION,
                )
            )
            if existing is not None:
                if existing.snapshot_hash != payload["snapshot_hash"]:
                    raise V34RuntimeError("immutable V3.4 feature snapshot changed")
                return existing.id
            session.add(
                V34FeatureSnapshotRow(
                    id=snapshot_id,
                    model_market=snapshot.market,
                    target_instrument_id=self._instrument_id(session, snapshot.market),
                    forecast_anchor_date=snapshot.cutoff_date,
                    source_max_date=snapshot.source_data_max_date,
                    feature_fit_end_date=snapshot.cutoff_date,
                    feature_version=FEATURE_VERSION,
                    scaler_version=SCALER_VERSION,
                    feature_json=payload["features"],
                    daily_sequence_json=payload["daily_sequence"],
                    weekly_state_json=payload["weekly_state"],
                    source_fields_json=payload["source_fields"],
                    snapshot_hash=payload["snapshot_hash"],
                    leakage_audit_json={
                        "source_max_date": snapshot.source_data_max_date.isoformat(),
                        "forecast_anchor_date": snapshot.cutoff_date.isoformat(),
                        "no_future": snapshot.source_data_max_date <= snapshot.cutoff_date,
                        "data_boundary": snapshot.provenance.get("data_boundary", "PIT_ALLOWED"),
                        "forbidden_source_read_count": snapshot.provenance.get(
                            "forbidden_source_read_count", 0
                        ),
                    },
                    created_at=self.now_provider(),
                )
            )
        return snapshot_id

    def _persist_model(
        self,
        state: V34ModelState,
        *,
        status: str,
        decision: V34PromotionDecision | None = None,
    ) -> str:
        model_id = f"V34-MODEL-{state.market}-{state.version}"
        payload = state_to_payload(state)
        gate = {} if decision is None else _jsonable(asdict(decision))
        with self.sessions() as session, session.begin():
            existing = session.get(V34ModelVersion, model_id)
            if existing is not None:
                if existing.parameter_hash != state.parameter_hash:
                    raise V34RuntimeError("immutable V3.4 model version changed")
                return model_id
            session.add(
                V34ModelVersion(
                    id=model_id,
                    model_market=state.market,
                    version=state.version,
                    parent_version=state.parent_version,
                    status=status,
                    trained_through_date=state.trained_through_date,
                    effective_from_date=state.effective_from_date,
                    feature_version=FEATURE_VERSION,
                    scaler_version=SCALER_VERSION,
                    scenario_adapter_version=SCENARIO_ADAPTER_VERSION,
                    raw_matured_sample_count=state.raw_matured_sample_count,
                    effective_independent_sample_count=_decimal(
                        state.effective_independent_sample_count
                    ),
                    parameters_json=payload,
                    metrics_json=_jsonable(state.validation_metrics),
                    promotion_gate_json=gate,
                    parameter_hash=state.parameter_hash,
                    created_at=self.now_provider(),
                )
            )
            adapter_id = f"V34-SA-{state.market}-{state.version}"
            session.add(
                V34ScenarioAdapter(
                    id=adapter_id,
                    model_market=state.market,
                    version=state.version,
                    feature_fit_end_date=state.feature_fit_end_date,
                    scenario_count=self.scenarios.scenario_count,
                    parameters_json={
                        "adapter_version": SCENARIO_ADAPTER_VERSION,
                        "return_process": "weekly_expected_plus_correlated_residual",
                        "gap_process": "self_history_empirical",
                        "volume_process": "self_history_AR1_log_change",
                    },
                    parameter_hash=_hash(
                        {
                            "state": state.parameter_hash,
                            "adapter": SCENARIO_ADAPTER_VERSION,
                            "scenario_count": self.scenarios.scenario_count,
                        }
                    ),
                    created_at=self.now_provider(),
                )
            )
        return model_id

    def _load_state(self, market: str, anchor: date | None = None) -> V34ModelState | None:
        with self.sessions() as session:
            query = select(V34ModelVersion).where(
                V34ModelVersion.model_market == market,
                V34ModelVersion.status == "champion",
            )
            if anchor is not None:
                query = query.where(V34ModelVersion.effective_from_date <= anchor)
            row = session.scalar(
                query.order_by(V34ModelVersion.effective_from_date.desc()).limit(1)
            )
            return None if row is None else state_from_payload(row.parameters_json)

    @staticmethod
    def _split_batches(amount: int) -> list[int]:
        amount = abs(int(amount))
        if amount == 0:
            return []
        raw = [int(round(amount * ratio / 5.0) * 5) for ratio in (0.4, 0.3, 0.2)]
        used = sum(raw)
        batches = [value for value in raw if value > 0]
        remainder = amount - used
        if remainder > 0:
            batches.append(remainder)
        if sum(batches) != amount:
            batches = [amount]
        return batches

    def _advice(self, scenario: V34ScenarioForecast, current_position: int) -> dict[str, Any]:
        up = float(scenario.direction_probabilities["up"])
        down = float(scenario.direction_probabilities["down"])
        consistency = scenario.consistency["status"] == "CONSISTENT"
        edge = up - down
        target = int(round((50.0 + edge * 0.75) / 5.0) * 5)
        target = max(0, min(100, target))
        if scenario.confidence < 60:
            target = int(round((current_position + (target - current_position) * 0.5) / 5.0) * 5)
        if not consistency:
            target = current_position + max(-10, min(10, target - current_position))
            target = int(round(target / 5.0) * 5)
        change = target - current_position
        action = "increase" if change > 0 else "decrease" if change < 0 else "hold"
        batches = self._split_batches(change)
        price_window = scenario.turning_windows["price_turning_window"]
        windows = []
        for index, percentage in enumerate(batches):
            week = min(13, int(price_window["start_week"]) + index)
            start = scenario.anchor_date + timedelta(days=7 * week - 6)
            end = scenario.anchor_date + timedelta(days=7 * week)
            windows.append(
                {
                    "batch": index + 1,
                    "action": action,
                    "position_change_percentage_points": percentage,
                    "window_start": start.isoformat(),
                    "window_end": end.isoformat(),
                    "trigger": (
                        "DIF/DEA and price confirm the scenario direction"
                        if index
                        else "price enters the forecast turning window"
                    ),
                    "invalidation": "price and MACD structure contradict the forecast band",
                }
            )
        if edge >= 30:
            ratio = "7:3"
        elif edge >= 10:
            ratio = "6:4"
        elif edge > -10:
            ratio = "5:5"
        elif edge > -30:
            ratio = "4:6"
        else:
            ratio = "3:7"
        return {
            "current_position": current_position,
            "target_position": target,
            "total_change_percentage_points": abs(change),
            "action": action,
            "batches": windows,
            "fund_etf_ratio": ratio,
            "confidence": scenario.confidence,
            "forecast_consistency_status": scenario.consistency["status"],
            "disclaimer": "概率情景研究结果，不构成保证收益或确定日期承诺。",
        }

    def _persist_forecast(
        self,
        snapshot_id: str,
        model_id: str,
        scenario: V34ScenarioForecast,
        *,
        label_end_date: date,
    ) -> int:
        current = self.current_position_provider(scenario.market)
        current_position = 0 if current is None else int(current)
        advice = self._advice(scenario, current_position)
        payload = {
            "version": "V3.4_13W",
            "forecast_horizon_weeks": 13,
            "market": scenario.market,
            "forecast_anchor_date": scenario.anchor_date.isoformat(),
            "model_version": scenario.model_version,
            "scenario_adapter_version": scenario.scenario_adapter_version,
            "scenario_seed": scenario.scenario_seed,
            "scenario_count": scenario.scenario_count,
            "historical_ohlcv": list(scenario.historical_ohlcv),
            "representative_ohlcv": list(scenario.representative_ohlcv),
            "indicators": list(scenario.indicators),
            "price_quantiles": list(scenario.price_quantiles),
            "direction_probabilities": dict(scenario.direction_probabilities),
            "path_probabilities": dict(scenario.path_probabilities),
            "confidence": scenario.confidence,
            "calibration": dict(scenario.calibration),
            "turning_windows": dict(scenario.turning_windows),
            "consistency": dict(scenario.consistency),
            "scenario_audit": dict(scenario.scenario_audit),
            "advice": advice,
            "semantics": {
                "p10_p50_p90": "每周收盘价边际分位数，不是置信度，也不是累计收益",
                "p50_candles": "P50代表性情景周K，不是唯一确定路径",
                "turning_dates": "周窗口与触发条件，不承诺单日必然反转",
            },
        }
        forecast_hash = _hash(payload)
        with self.sessions() as session, session.begin():
            existing = session.scalar(
                select(V34ForecastRow).where(
                    V34ForecastRow.model_market == scenario.market,
                    V34ForecastRow.forecast_anchor_date == scenario.anchor_date,
                )
            )
            if existing is not None:
                if existing.forecast_hash != forecast_hash:
                    raise V34RuntimeError("frozen weekly forecast cannot be overwritten")
                return int(existing.id)
            row = V34ForecastRow(
                model_market=scenario.market,
                target_instrument_id=self._instrument_id(session, scenario.market),
                feature_snapshot_id=snapshot_id,
                model_version_id=model_id,
                forecast_anchor_date=scenario.anchor_date,
                label_end_date=label_end_date,
                horizon_weeks=13,
                maturity_status="IMMATURE",
                scenario_adapter_version=scenario.scenario_adapter_version,
                scenario_seed=scenario.scenario_seed,
                scenario_count=scenario.scenario_count,
                representative_ohlcv_json=list(scenario.representative_ohlcv),
                indicator_path_json=list(scenario.indicators),
                price_quantiles_json=list(scenario.price_quantiles),
                direction_probabilities_json=dict(scenario.direction_probabilities),
                path_probabilities_json=dict(scenario.path_probabilities),
                calibration_json=dict(scenario.calibration),
                confidence=_decimal(scenario.confidence),
                turning_windows_json=dict(scenario.turning_windows),
                consistency_json=dict(scenario.consistency),
                advice_json=advice,
                payload_json=payload,
                forecast_hash=forecast_hash,
                created_at=self.now_provider(),
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def _update_evaluations(
        self,
        market: str,
        all_snapshots: Sequence[FeatureSnapshot],
    ) -> int:
        by_date = {snapshot.cutoff_date: snapshot for snapshot in all_snapshots}
        dates = sorted(by_date)
        index = {value: idx for idx, value in enumerate(dates)}
        count = 0
        with self.sessions() as session, session.begin():
            forecasts = list(
                session.scalars(
                    select(V34ForecastRow).where(V34ForecastRow.model_market == market)
                )
            )
            for forecast in forecasts:
                anchor_index = index.get(forecast.forecast_anchor_date)
                if anchor_index is None:
                    continue
                realized_dates = dates[anchor_index + 1 : anchor_index + 14]
                realized = [
                    {
                        "week": offset,
                        "week_end": day.isoformat(),
                        "close": float(by_date[day].daily_sequence[-1]["close"]),
                    }
                    for offset, day in enumerate(realized_dates, start=1)
                ]
                status = maturity_status(len(realized_dates))
                forecast.maturity_status = status
                evaluation = session.scalar(
                    select(V34ModelEvaluation).where(
                        V34ModelEvaluation.forecast_id == forecast.id
                    )
                )
                metrics: dict[str, Any] = {}
                if realized:
                    predicted = forecast.representative_ohlcv_json[: len(realized)]
                    base = float(
                        by_date[forecast.forecast_anchor_date].daily_sequence[-1]["close"]
                    )
                    actual_returns = np.asarray([row["close"] / base - 1.0 for row in realized])
                    predicted_returns = np.asarray(
                        [float(row["close"]) / base - 1.0 for row in predicted]
                    )
                    metrics["p50_path_mae"] = float(
                        np.mean(np.abs(actual_returns - predicted_returns))
                    )
                    cover = []
                    for actual, quantile in zip(realized, forecast.price_quantiles_json):
                        cover.append(
                            float(quantile["close_p10"])
                            <= float(actual["close"])
                            <= float(quantile["close_p90"])
                        )
                    metrics["p10_p90_coverage"] = float(np.mean(cover))
                if evaluation is None:
                    evaluation = V34ModelEvaluation(
                        forecast_id=forecast.id,
                        maturity_status=status,
                        realized_week_count=len(realized_dates),
                        realized_labels_json={"weeks": realized},
                        metrics_json=metrics,
                        evaluated_at=self.now_provider() if realized else None,
                    )
                    session.add(evaluation)
                else:
                    evaluation.maturity_status = status
                    evaluation.realized_week_count = len(realized_dates)
                    evaluation.realized_labels_json = {"weeks": realized}
                    evaluation.metrics_json = metrics
                    evaluation.evaluated_at = self.now_provider() if realized else None
                count += 1
        return count

    def _persist_calibrators(
        self,
        market: str,
        candidate_number: int,
        as_of: date,
        forecasts: Sequence[V34TrainingSample],
    ) -> None:
        for horizon in (1, 4, 8, 13):
            eligible = [
                sample
                for sample in forecasts
                if sample.cumulative_return_path is not None
                and len(sample.label_dates) >= horizon
                and sample.label_dates[horizon - 1] <= as_of
            ]
            weights = overlap_weights(eligible) if horizon == 13 else np.ones(len(eligible))
            effective = effective_sample_count(weights)
            status = (
                "NOT_SUFFICIENTLY_CALIBRATED"
                if effective < 30
                else "PRELIMINARY"
                if effective < 50
                else "FORMALLY_CALIBRATED"
            )
            version = f"{market}-H{horizon}-C{candidate_number}"
            calibrator_id = f"V34-CAL-{version}"
            with self.sessions() as session, session.begin():
                if session.get(V34ProbabilityCalibrator, calibrator_id) is not None:
                    continue
                parameters = {
                    "method": "historical_frozen_out_of_sample_identity_shrinkage",
                    "horizon_weeks": horizon,
                    "fit_through_date": as_of.isoformat(),
                }
                session.add(
                    V34ProbabilityCalibrator(
                        id=calibrator_id,
                        model_market=market,
                        horizon_weeks=horizon,
                        version=version,
                        fit_through_date=as_of,
                        raw_sample_count=len(eligible),
                        effective_sample_count=_decimal(effective),
                        status=status,
                        parameters_json=parameters,
                        calibration_hash=_hash(parameters),
                        created_at=self.now_provider(),
                    )
                )

    def _create_run(self, market: str, run_type: str, through: date | None) -> str:
        run_id = f"V34-{run_type.upper()}-{market}-{uuid4().hex[:12]}"
        with self.sessions() as session, session.begin():
            session.add(
                V34TrainingRun(
                    id=run_id,
                    model_market=market,
                    run_type=run_type,
                    status="queued",
                    current_stage="queued",
                    horizon_weeks=13,
                    requested_through_anchor=through,
                    weekly_iteration_count=0,
                    candidate_training_count=0,
                    champion_promotion_count=0,
                    started_at=self.now_provider(),
                    completed_at=None,
                    result_json={},
                )
            )
        return run_id

    def start_bootstrap(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        if self.champion(market) is not None:
            raise V34RuntimeError("V3.4 bootstrap already exists; use incremental training")
        run_id = self._create_run(market, "bootstrap", None)
        self._executor.submit(self._run_task, run_id, market, "bootstrap")
        return {"run_id": run_id, "status": "queued", "market": market}

    def start_incremental(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        if self.champion(market) is None:
            raise V34RuntimeError("V3.4 has not been bootstrapped")
        run_id = self._create_run(market, "incremental", None)
        self._executor.submit(self._run_task, run_id, market, "incremental")
        return {"run_id": run_id, "status": "queued", "market": market}

    def _run_task(self, run_id: str, market: str, run_type: str) -> None:
        try:
            if run_type == "bootstrap":
                self.bootstrap_sync(market, run_id=run_id)
            else:
                self.incremental_sync(market, run_id=run_id)
        except BaseException as exc:
            with self.sessions() as session, session.begin():
                row = session.get(V34TrainingRun, run_id)
                if row is not None:
                    row.status = "failed"
                    row.current_stage = "failed"
                    row.completed_at = self.now_provider()
                    row.error_code = type(exc).__name__
                    row.error_message = str(exc)

    def bootstrap_sync(self, market: str, *, run_id: str | None = None) -> dict[str, Any]:
        self.validate_market(market)
        with self._locks[market]:
            if self.champion(market) is not None:
                raise V34RuntimeError("V3.4 bootstrap already exists")
            anchors = self._complete_anchors(market)
            formal_start = anchors[-1] - timedelta(days=BOOTSTRAP_DAYS)
            build_start = formal_start - timedelta(weeks=WARMUP_WEEKS)
            build_anchors = tuple(anchor for anchor in anchors if anchor >= build_start)
            snapshots = self._snapshots(market, build_anchors)
            samples = build_training_samples(snapshots)
            nominal_formal_samples = tuple(
                sample for sample in samples if sample.anchor_date >= formal_start
            )
            # A recently listed instrument may have enough history for a long
            # replay but not enough pre-formal, fully matured 13-week labels to
            # fit the seed at the nominal ten-year boundary.  Never lower the
            # seed gate and never open future labels.  Instead, advance the
            # first formal anchor until 30 labels have genuinely matured.
            first_seedable_index = next(
                (
                    index
                    for index, sample in enumerate(nominal_formal_samples)
                    if len(eligible_fully_matured(samples, sample.anchor_date)) >= 30
                ),
                None,
            )
            if first_seedable_index is None:
                raise V34RuntimeError(
                    "no point-in-time anchor has 30 fully matured 13-week seed labels"
                )
            formal_samples = nominal_formal_samples[first_seedable_index:]
            if len(formal_samples) < 400:
                raise V34RuntimeError(
                    f"ten-year bootstrap requires at least 400 complete natural weeks, got {len(formal_samples)}"
                )
            first = formal_samples[0]
            seed_rows = eligible_fully_matured(samples, first.anchor_date)
            if len(seed_rows) < 30:
                raise V34RuntimeError("insufficient pre-formal matured labels for V3.4 seed")
            prefix = "CYB" if market == "399006" else "GFNDXETF"
            seed = self.training.fit(
                market,
                seed_rows,
                version=f"{prefix}_HYBRID_13W_V3.4.0",
                parent_version=None,
                trained_through=first.anchor_date,
                effective_from=first.anchor_date,
                alpha=5.0 if market == "399006" else 8.0,
            )
            seed_model_id = self._persist_model(seed, status="champion")
            run_id = run_id or self._create_run(market, "bootstrap", formal_samples[-1].anchor_date)
            return self._progressive_run(
                run_id, market, samples, formal_samples, seed, seed_model_id
            )

    def _progressive_run(
        self,
        run_id: str,
        market: str,
        samples: Sequence[V34TrainingSample],
        formal_samples: Sequence[V34TrainingSample],
        champion_state: V34ModelState,
        champion_model_id: str,
    ) -> dict[str, Any]:
        with self.sessions() as lookup:
            prior_optimizer = lookup.scalar(
                select(V34OptimizerState).where(V34OptimizerState.model_market == market)
            )
        iteration_offset = 0 if prior_optimizer is None else prior_optimizer.weekly_iteration_count
        candidate_count = 0 if prior_optimizer is None else prior_optimizer.candidate_training_count
        promotion_count = 0 if prior_optimizer is None else prior_optimizer.champion_promotion_count
        first_matured_count = len(
            eligible_fully_matured(samples, formal_samples[0].anchor_date)
        )
        matured_at_last_retrain = (
            first_matured_count
            if prior_optimizer is None
            else int(
                prior_optimizer.state_json.get(
                    "last_retrain_matured_count",
                    first_matured_count - prior_optimizer.fully_matured_since_retrain,
                )
            )
        )
        with self.sessions() as session, session.begin():
            run = session.get(V34TrainingRun, run_id)
            if run is None:
                raise V34RuntimeError("training run row disappeared")
            run.status = "running"
            run.current_stage = "strict_weekly_replay"
        for local_iteration, current in enumerate(formal_samples, start=1):
            iteration_number = iteration_offset + local_iteration
            if current.anchor_date < champion_state.effective_from_date:
                raise V34RuntimeError("champion activated before its next-week effective date")
            snapshot_id = self._persist_snapshot(current.snapshot)
            matured = eligible_fully_matured(samples, current.anchor_date)
            path = self.training.predict(champion_state, current.snapshot)
            scenario = self.scenarios.generate(
                champion_state,
                path,
                current.snapshot,
                matured,
                seed=340013 + iteration_number,
            )
            label_end = (
                current.label_end_date
                if current.label_end_date is not None
                else current.anchor_date + timedelta(weeks=13)
            )
            forecast_id = self._persist_forecast(
                snapshot_id, champion_model_id, scenario, label_end_date=label_end
            )
            champion_before = champion_state.version
            champion_after = champion_before
            candidate_version = None
            training_triggered = False
            promoted = False
            decision_payload: dict[str, Any] = {
                "reason": "fewer_than_four_new_fully_matured_samples"
            }
            newly_matured = len(matured) - matured_at_last_retrain
            if newly_matured >= MIN_NEW_MATURED_SAMPLES_FOR_RETRAIN:
                training_triggered = True
                candidate_count += 1
                champion_alpha = float(champion_state.hyperparameters["ridge_alpha"])
                candidate_alpha = self.training.next_candidate_alpha(
                    champion_alpha, candidate_count
                )
                decision = self.training.compare_candidate(
                    market,
                    matured,
                    champion_alpha=champion_alpha,
                    candidate_alpha=candidate_alpha,
                )
                candidate_version = f"{prefix_for(market)}_HYBRID_13W_V3.4.C{candidate_count}"
                next_anchor = (
                    formal_samples[local_iteration].anchor_date
                    if local_iteration < len(formal_samples)
                    else current.anchor_date + timedelta(weeks=1)
                )
                candidate = self.training.fit(
                    market,
                    matured,
                    version=candidate_version,
                    parent_version=champion_state.version,
                    trained_through=current.anchor_date,
                    effective_from=next_anchor,
                    alpha=candidate_alpha,
                    validation_metrics={
                        "champion_fold_losses": decision.fold_losses_champion,
                        "candidate_fold_losses": decision.fold_losses_candidate,
                    },
                )
                candidate_model_id = self._persist_model(
                    candidate,
                    status="champion" if decision.promoted else "rejected",
                    decision=decision,
                )
                decision_payload = _jsonable(asdict(decision))
                matured_at_last_retrain = len(matured)
                self._persist_calibrators(
                    market, candidate_count, current.anchor_date, samples
                )
                if decision.promoted:
                    champion_state = candidate
                    champion_model_id = candidate_model_id
                    champion_after = candidate.version
                    promoted = True
                    promotion_count += 1
            weights = overlap_weights(matured)
            with self.sessions() as session, session.begin():
                session.add(
                    V34TrainingIteration(
                        training_run_id=run_id,
                        forecast_id=forecast_id,
                        model_market=market,
                        anchor_date=current.anchor_date,
                        weekly_iteration_number=iteration_number,
                        champion_before_version=champion_before,
                        candidate_version=candidate_version,
                        champion_after_version=champion_after,
                        training_triggered=training_triggered,
                        promoted=promoted,
                        newly_matured_count=max(0, newly_matured),
                        raw_matured_sample_count=len(matured),
                        effective_independent_sample_count=_decimal(
                            effective_sample_count(weights)
                        ),
                        validation_json={
                            "method": "purged_walk_forward",
                            "purge_horizon_weeks": 13,
                            "scaler_fit_scope": "training_fold_only",
                        },
                        promotion_gate_json=decision_payload,
                        completed_at=self.now_provider(),
                    )
                )
                run = session.get(V34TrainingRun, run_id)
                if run is not None:
                    run.weekly_iteration_count = local_iteration
                    run.candidate_training_count = candidate_count
                    run.champion_promotion_count = promotion_count
        self._update_evaluations(market, [sample.snapshot for sample in samples])
        latest = formal_samples[-1].anchor_date
        with self.sessions() as session, session.begin():
            state_payload = {
                "last_anchor_date": latest.isoformat(),
                "champion_version": champion_state.version,
                "methodology_version": METHODOLOGY_VERSION,
                "horizon_weeks": 13,
                "min_new_matured_samples_for_retrain": 4,
                "last_retrain_matured_count": matured_at_last_retrain,
            }
            optimizer = session.scalar(
                select(V34OptimizerState).where(V34OptimizerState.model_market == market)
            )
            if optimizer is None:
                optimizer = V34OptimizerState(
                    model_market=market,
                    last_anchor_date=latest,
                    weekly_iteration_count=iteration_offset + len(formal_samples),
                    candidate_training_count=candidate_count,
                    champion_promotion_count=promotion_count,
                    fully_matured_since_retrain=(
                        len(eligible_fully_matured(samples, latest)) - matured_at_last_retrain
                    ),
                    champion_version=champion_state.version,
                    state_json=state_payload,
                    state_hash=_hash(state_payload),
                    updated_at=self.now_provider(),
                )
                session.add(optimizer)
            else:
                optimizer.last_anchor_date = latest
                optimizer.weekly_iteration_count = iteration_offset + len(formal_samples)
                optimizer.candidate_training_count = candidate_count
                optimizer.champion_promotion_count = promotion_count
                optimizer.fully_matured_since_retrain = max(
                    0, len(eligible_fully_matured(samples, latest)) - matured_at_last_retrain
                )
                optimizer.champion_version = champion_state.version
                optimizer.state_json = state_payload
                optimizer.state_hash = _hash(state_payload)
                optimizer.updated_at = self.now_provider()
            run = session.get(V34TrainingRun, run_id)
            if run is None:
                raise V34RuntimeError("training run row disappeared at completion")
            run.status = "completed"
            run.current_stage = "completed"
            run.completed_at = self.now_provider()
            run.weekly_iteration_count = len(formal_samples)
            run.candidate_training_count = candidate_count
            run.champion_promotion_count = promotion_count
            run.result_json = {
                "market": market,
                "weekly_iteration_count": len(formal_samples),
                "candidate_training_count": candidate_count,
                "champion_promotion_count": promotion_count,
                "pending_13w_count": sum(
                    sample.cumulative_return_path is None for sample in formal_samples
                ),
                "latest_anchor_date": latest.isoformat(),
                "champion_version": champion_state.version,
            }
        return self.training_status(run_id)

    def incremental_sync(self, market: str, *, run_id: str | None = None) -> dict[str, Any]:
        self.validate_market(market)
        with self._locks[market]:
            champion = self._load_state(market)
            if champion is None:
                raise V34RuntimeError("V3.4 has not been bootstrapped")
            anchors = self._complete_anchors(market)
            with self.sessions() as session:
                last = session.scalar(
                    select(func.max(V34TrainingIteration.anchor_date)).where(
                        V34TrainingIteration.model_market == market
                    )
                )
            available = [anchor for anchor in anchors if last is None or anchor > last]
            if not available:
                result = {
                    "market": market,
                    "status": "not_due",
                    "reason": "no_new_complete_week",
                    "last_anchor_date": None if last is None else last.isoformat(),
                }
                if run_id is not None:
                    with self.sessions() as session, session.begin():
                        run = session.get(V34TrainingRun, run_id)
                        if run is not None:
                            run.status = "completed"
                            run.current_stage = "not_due"
                            run.completed_at = self.now_provider()
                            run.result_json = result
                return result
            # One invocation consumes exactly one new completed week and starts
            # from the existing hundreds-of-weeks champion chain.
            next_anchor = available[0]
            build_start = next_anchor - timedelta(days=BOOTSTRAP_DAYS + 52 * 7)
            build_anchors = tuple(anchor for anchor in anchors if anchor >= build_start and anchor <= next_anchor)
            snapshots = self._snapshots(market, build_anchors)
            samples = build_training_samples(snapshots)
            current = next(sample for sample in samples if sample.anchor_date == next_anchor)
            run_id = run_id or self._create_run(market, "incremental", next_anchor)
            model_id = f"V34-MODEL-{market}-{champion.version}"
            result = self._progressive_run(run_id, market, samples, (current,), champion, model_id)
            return result

    def run_analysis(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        before = self._training_counts(market)
        anchors = self._complete_anchors(market)
        anchor = anchors[-1]
        with self.sessions() as session:
            existing = session.scalar(
                select(V34ForecastRow)
                .where(
                    V34ForecastRow.model_market == market,
                    V34ForecastRow.forecast_anchor_date == anchor,
                )
                .order_by(V34ForecastRow.id.desc())
            )
            if existing is not None:
                result = dict(existing.payload_json)
            else:
                champion = self._load_state(market, anchor)
                if champion is None:
                    raise V34RuntimeError("no effective V3.4 champion; train the model first")
                build_start = anchor - timedelta(days=BOOTSTRAP_DAYS + 52 * 7)
                build_anchors = tuple(value for value in anchors if value >= build_start)
                snapshots = self._snapshots(market, build_anchors)
                samples = build_training_samples(snapshots)
                snapshot = snapshots[-1]
                matured = eligible_fully_matured(samples, anchor)
                snapshot_id = self._persist_snapshot(snapshot)
                path = self.training.predict(champion, snapshot)
                scenario = self.scenarios.generate(champion, path, snapshot, matured)
                forecast_id = self._persist_forecast(
                    snapshot_id,
                    f"V34-MODEL-{market}-{champion.version}",
                    scenario,
                    label_end_date=anchor + timedelta(weeks=13),
                )
                with self.sessions() as lookup:
                    row = lookup.get(V34ForecastRow, forecast_id)
                    if row is None:
                        raise V34RuntimeError("analysis forecast was not persisted")
                    result = dict(row.payload_json)
        after = self._training_counts(market)
        if before != after:
            raise V34RuntimeError("data analysis mutated V3.4 training state")
        analysis_id = f"V34-ANALYSIS-{market}-{uuid4().hex[:12]}"
        with self.sessions() as session, session.begin():
            session.add(
                V34AnalysisRun(
                    id=analysis_id,
                    model_market=market,
                    forecast_id=None,
                    model_version=str(result["model_version"]),
                    status="completed",
                    forecast_anchor_date=anchor,
                    started_at=self.now_provider(),
                    completed_at=self.now_provider(),
                    result_json=result,
                )
            )
        return result

    def _training_counts(self, market: str) -> tuple[int, int, int]:
        with self.sessions() as session:
            return (
                int(
                    session.scalar(
                        select(func.count()).select_from(V34TrainingIteration).where(
                            V34TrainingIteration.model_market == market
                        )
                    )
                    or 0
                ),
                int(
                    session.scalar(
                        select(func.count()).select_from(V34ModelVersion).where(
                            V34ModelVersion.model_market == market
                        )
                    )
                    or 0
                ),
                int(
                    session.scalar(
                        select(func.count()).select_from(V34OptimizerState).where(
                            V34OptimizerState.model_market == market
                        )
                    )
                    or 0
                ),
            )

    def training_status(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            row = session.get(V34TrainingRun, run_id)
            if row is None:
                raise V34RuntimeError("unknown V3.4 training run")
            return {
                "run_id": row.id,
                "market": row.model_market,
                "run_type": row.run_type,
                "status": row.status,
                "current_stage": row.current_stage,
                "horizon_weeks": row.horizon_weeks,
                "weekly_iteration_count": row.weekly_iteration_count,
                "candidate_training_count": row.candidate_training_count,
                "champion_promotion_count": row.champion_promotion_count,
                "started_at": row.started_at.isoformat(),
                "completed_at": None if row.completed_at is None else row.completed_at.isoformat(),
                "result": dict(row.result_json),
                "error_code": row.error_code,
                "error_message": row.error_message,
            }

    def champion(self, market: str) -> dict[str, Any] | None:
        self.validate_market(market)
        state = self._load_state(market)
        if state is None:
            return None
        return {
            "market": market,
            "version": state.version,
            "parent_version": state.parent_version,
            "horizon_weeks": 13,
            "trained_through_date": state.trained_through_date.isoformat(),
            "effective_from_date": state.effective_from_date.isoformat(),
            "raw_matured_sample_count": state.raw_matured_sample_count,
            "effective_independent_sample_count": state.effective_independent_sample_count,
            "parameter_hash": state.parameter_hash,
        }

    def model_statuses(self) -> list[dict[str, Any]]:
        output = []
        for market in SUPPORTED_MARKETS:
            champion = self.champion(market)
            with self.sessions() as session:
                optimizer = session.scalar(
                    select(V34OptimizerState).where(V34OptimizerState.model_market == market)
                )
            output.append(
                {
                    "market": market,
                    "version": "V3.4_13W",
                    "horizon_weeks": 13,
                    "bootstrapped": champion is not None,
                    "champion": champion,
                    "last_anchor_date": (
                        None if optimizer is None else optimizer.last_anchor_date.isoformat()
                    ),
                    "weekly_iteration_count": (
                        0 if optimizer is None else optimizer.weekly_iteration_count
                    ),
                    "candidate_training_count": (
                        0 if optimizer is None else optimizer.candidate_training_count
                    ),
                    "champion_promotion_count": (
                        0 if optimizer is None else optimizer.champion_promotion_count
                    ),
                }
            )
        return output

    def curve(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        with self.sessions() as session:
            rows = list(
                session.scalars(
                    select(V34TrainingIteration)
                    .where(V34TrainingIteration.model_market == market)
                    .order_by(V34TrainingIteration.weekly_iteration_number)
                )
            )
        return {
            "market": market,
            "horizon_weeks": 13,
            "points": [
                {
                    "iteration": row.weekly_iteration_number,
                    "anchor_date": row.anchor_date.isoformat(),
                    "raw_matured_sample_count": row.raw_matured_sample_count,
                    "effective_independent_sample_count": float(
                        row.effective_independent_sample_count
                    ),
                    "training_triggered": row.training_triggered,
                    "promoted": row.promoted,
                    "champion_before": row.champion_before_version,
                    "candidate": row.candidate_version,
                    "champion_after": row.champion_after_version,
                    "champion_loss": row.promotion_gate_json.get(
                        "fold_losses_champion", []
                    ),
                    "candidate_loss": row.promotion_gate_json.get(
                        "fold_losses_candidate", []
                    ),
                }
                for row in rows
            ],
        }

    def latest_forecast(self, market: str) -> dict[str, Any] | None:
        self.validate_market(market)
        with self.sessions() as session:
            row = session.scalar(
                select(V34ForecastRow)
                .where(V34ForecastRow.model_market == market)
                .order_by(V34ForecastRow.forecast_anchor_date.desc(), V34ForecastRow.id.desc())
                .limit(1)
            )
            return None if row is None else dict(row.payload_json)


def prefix_for(market: str) -> str:
    return "CYB" if market == "399006" else "GFNDXETF"
