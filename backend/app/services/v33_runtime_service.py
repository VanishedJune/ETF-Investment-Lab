"""SQLite-backed production orchestration for the V3.3 20-week model.

The runtime owns task status, crash-safe checkpoints, feature/forecast audit
rows and the weekly eligibility rule.  It does not contain model mathematics;
those remain in ``v33_feature_service``, ``v33_training_service`` and
``v33_analysis_service``.

Public service entry points (suitable for a thin FastAPI adapter):

* ``start_bootstrap(market)`` / ``bootstrap_sync(market)``
* ``start_incremental(market)`` / ``incremental_sync(market)``
* ``training_status(run_id)`` / ``latest_training_status(market)``
* ``curve(market)`` / ``champion(market)``
* ``run_analysis(market)`` / ``latest_analysis(market)``

Analysis loads an immutable Champion and verifies that iteration/model counts
are unchanged before/after.  Bootstrap uses one fixed-seed real session from
every complete ISO natural week in the formal ten-year window; earlier history
is warm-up only.  Incremental training adds at most one iteration per market
when both a new complete week and seven elapsed days are present.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from threading import Lock
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ..daemon_executor import DaemonThreadPoolExecutor
from ..models.models import (
    Instrument,
    MarketPrice,
    ValuationRecord,
    V33AnalysisRun,
    V33FeatureSnapshot as V33FeatureSnapshotRow,
    V33Forecast as V33ForecastRow,
    V33MarketBar,
    V33ModelEvaluation,
    V33ModelVersion,
    V33OptimizerState,
    V33PointInTimeObservation,
    V33TrainingCheckpoint,
    V33TrainingIteration,
    V33TrainingRun,
    utc_now,
)
from .investment_calendar_service import InvestmentCalendarService
from .market_calendar import CalendarProvider
from .v33_analysis_service import V33AnalysisService
from .v33_feature_service import (
    DEFAULT_PIT_SERIES_CODES,
    FeatureSnapshot,
    PointInTimeObservation,
    PriceBar,
    V33FeatureError,
    V33FeatureService,
)
from .v33_training_service import (
    HORIZON_WEEKS,
    V33Forecast,
    V33IterationRecord,
    V33ModelState,
    V33ProgressiveResult,
    V33TrainingError,
    V33TrainingPoint,
    V33TrainingRepository,
    V33TrainingService,
    build_training_points,
    deterministic_weekly_sample_dates,
    state_from_payload,
    state_to_payload,
)


FEATURE_VERSION = "V3.3-20W-FULL100-DCT12-3"
TRAINING_FEATURE_VERSION = f"{FEATURE_VERSION}-SAMPLED"
ANALYSIS_FEATURE_VERSION = f"{FEATURE_VERSION}-LIVE"
METHODOLOGY_VERSION = "V3.3-PROGRESSIVE-PURGED-20W-3"
DEFAULT_SAMPLING_SEED = 330020
DEFAULT_ANALYSIS_DAYS = 3653
DEFAULT_WARMUP_WEEKS = 70
SHANGHAI = ZoneInfo("Asia/Shanghai")


class V33RuntimeError(RuntimeError):
    """A production orchestration invariant failed."""


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_warmup_anchor_payload(
    state: V33ModelState,
    *,
    snapshot_manifest: Sequence[Mapping[str, Any]],
    eligible_cutoff_dates: Sequence[date],
    sampling_seed: int,
) -> dict[str, Any]:
    """Return the canonical non-formal root audit artifact for a bootstrap."""

    if state.iteration_number != 0 or state.parent_state_hash is not None:
        raise V33RuntimeError("warm-up anchor must be iteration 0 with no parent")
    manifest = [_jsonable(dict(row)) for row in snapshot_manifest]
    manifest_cutoffs = [str(row.get("cutoff_date") or "") for row in manifest]
    if not manifest or manifest_cutoffs != sorted(set(manifest_cutoffs)):
        raise V33RuntimeError("warm-up snapshot manifest must be unique and ordered")
    eligible = [value.isoformat() for value in eligible_cutoff_dates]
    if eligible != sorted(set(eligible)):
        raise V33RuntimeError("warm-up eligible sample cutoffs must be unique and ordered")
    if state.training_sample_count != len(eligible):
        raise V33RuntimeError("warm-up state sample count differs from eligible cutoffs")
    state_payload = state_to_payload(state)
    body = {
        "kind": "v33_non_formal_warmup_root",
        "formal_iteration": False,
        "market": state.market,
        "iteration_number": 0,
        "parent_state_hash": None,
        "version": state.version,
        "first_formal_cutoff": state.trained_through.isoformat(),
        "real_matured_sample_count": len(eligible),
        "eligible_sample_cutoffs": eligible,
        "build_snapshot_count": len(manifest),
        "build_snapshot_manifest": manifest,
        "manifest_hash": _hash(manifest),
        "feature_version": TRAINING_FEATURE_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "sampling_seed": int(sampling_seed),
        "state_payload": state_payload,
        "state_hash": state.state_hash,
    }
    return {**body, "anchor_hash": _hash(body)}


def _decimal(value: float | int | None) -> Decimal | None:
    return None if value is None else Decimal(str(round(float(value), 8)))


def _week_key(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def _china_close(day: date) -> datetime:
    return datetime.combine(day, time(15, 5), tzinfo=SHANGHAI).astimezone(timezone.utc)


def _forecast_payload(forecast: V33Forecast) -> dict[str, Any]:
    payload = asdict(forecast)
    payload["cutoff_date"] = forecast.cutoff_date.isoformat()
    return _jsonable(payload)


def _forecast_from_payload(payload: Mapping[str, Any]) -> V33Forecast:
    paths = [tuple(float(item) for item in payload[name]) for name in ("p10", "p50", "p90", "expected")]
    if any(len(path) != HORIZON_WEEKS for path in paths):
        raise V33RuntimeError("saved forecast is not an exact 20-week path")
    weekly_base = tuple(
        float(item) for item in payload.get("weekly_base", paths[3])
    )
    daily_correction = tuple(
        float(item)
        for item in payload.get("daily_correction", [0.0] * HORIZON_WEEKS)
    )
    if len(weekly_base) != HORIZON_WEEKS or len(daily_correction) != HORIZON_WEEKS:
        raise V33RuntimeError("saved weekly/daily forecast components are not exact 20-week paths")
    return V33Forecast(
        market=str(payload["market"]),
        cutoff_date=date.fromisoformat(str(payload["cutoff_date"])),
        p10=paths[0],
        p50=paths[1],
        p90=paths[2],
        expected=paths[3],
        weekly_base=weekly_base,
        daily_correction=daily_correction,
        up_probability=float(payload["up_probability"]),
        sideways_probability=float(payload["sideways_probability"]),
        down_probability=float(payload["down_probability"]),
        direction=str(payload["direction"]),
        confidence=float(payload["confidence"]),
        threshold=float(payload["threshold"]),
        expected_max_drawdown=float(payload["expected_max_drawdown"]),
        predicted_high_week=int(payload["predicted_high_week"]),
        predicted_low_week=int(payload["predicted_low_week"]),
        dif_turn_kind=str(payload["dif_turn_kind"]),
        dif_turn_days=None if payload.get("dif_turn_days") is None else float(payload["dif_turn_days"]),
        price_turn_days=None if payload.get("price_turn_days") is None else float(payload["price_turn_days"]),
        turn_stable=bool(payload["turn_stable"]),
        analogue_calibration_count=int(payload["analogue_calibration_count"]),
        model_version=str(payload["model_version"]),
        model_state_hash=str(payload["model_state_hash"]),
    )


def _snapshot_payload(snapshot: FeatureSnapshot) -> dict[str, Any]:
    return _jsonable(
        {
            "market": snapshot.market,
            "cutoff_date": snapshot.cutoff_date.isoformat(),
            "source_data_max_date": snapshot.source_data_max_date.isoformat(),
            "daily_as_of": snapshot.daily_as_of.isoformat(),
            "weekly_as_of": snapshot.weekly_as_of.isoformat(),
            "weekly": dict(snapshot.weekly),
            "daily": dict(snapshot.daily),
            "daily_sequence": list(snapshot.daily_sequence),
            "missing_masks": dict(snapshot.missing_masks),
            "provenance": dict(snapshot.provenance),
            "derivative_turn": dict(snapshot.derivative_turn),
            "benchmark_market": snapshot.benchmark_market,
        }
    )


def _snapshot_from_payload(payload: Mapping[str, Any]) -> FeatureSnapshot:
    return FeatureSnapshot(
        market=str(payload["market"]),
        cutoff_date=date.fromisoformat(str(payload["cutoff_date"])),
        source_data_max_date=date.fromisoformat(str(payload["source_data_max_date"])),
        daily_as_of=date.fromisoformat(str(payload["daily_as_of"])),
        weekly_as_of=date.fromisoformat(str(payload["weekly_as_of"])),
        weekly=dict(payload["weekly"]),
        daily=dict(payload["daily"]),
        daily_sequence=tuple(dict(row) for row in payload["daily_sequence"]),
        missing_masks={str(key): int(value) for key, value in payload["missing_masks"].items()},
        provenance=dict(payload["provenance"]),
        derivative_turn=dict(payload["derivative_turn"]),
        benchmark_market=payload.get("benchmark_market"),
    )


class SQLiteV33TrainingRepository(V33TrainingRepository):
    """Exact model/forecast persistence used by bootstrap and restart resume."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        run_id: str | None = None,
        snapshot_ids: Mapping[date, str] | None = None,
        sampling_seed: int = DEFAULT_SAMPLING_SEED,
        now_provider: Callable[[], datetime] = utc_now,
    ) -> None:
        self.sessions = sessions
        self.run_id = run_id
        self.snapshot_ids = dict(snapshot_ids or {})
        self.sampling_seed = sampling_seed
        self.now_provider = now_provider
        self._states: dict[int, V33ModelState] = {}

    def bind_run(self, run_id: str, snapshot_ids: Mapping[date, str]) -> None:
        self.run_id = run_id
        self.snapshot_ids = dict(snapshot_ids)

    def save_warmup_anchor(
        self,
        state: V33ModelState,
        *,
        build_cutoff_dates: Sequence[date],
        eligible_cutoff_dates: Sequence[date],
    ) -> None:
        if self.run_id is None:
            raise V33RuntimeError("repository run is not bound")
        build_dates = tuple(sorted(set(build_cutoff_dates)))
        eligible_dates = tuple(sorted(set(eligible_cutoff_dates)))
        if not set(eligible_dates).issubset(build_dates):
            raise V33RuntimeError("warm-up eligible samples are absent from the build manifest")
        now = self.now_provider()
        with self.sessions() as session, session.begin():
            run = session.get(V33TrainingRun, self.run_id)
            if run is None:
                raise V33RuntimeError("training run disappeared while saving warm-up anchor")
            if run.run_type != "bootstrap" or run.source_parent_iteration is not None:
                raise V33RuntimeError("warm-up anchor belongs only to a root bootstrap run")
            manifest: list[dict[str, Any]] = []
            for cutoff in build_dates:
                snapshot_id = self.snapshot_ids.get(cutoff)
                if snapshot_id is None:
                    raise V33RuntimeError(
                        f"warm-up build snapshot id is missing for {cutoff.isoformat()}"
                    )
                snapshot = session.get(V33FeatureSnapshotRow, snapshot_id)
                if snapshot is None:
                    raise V33RuntimeError(
                        f"warm-up build snapshot row is missing for {cutoff.isoformat()}"
                    )
                if (
                    snapshot.model_market != state.market
                    or snapshot.cutoff_date != cutoff
                    or snapshot.feature_version != TRAINING_FEATURE_VERSION
                ):
                    raise V33RuntimeError("warm-up build snapshot identity mismatch")
                if _hash(snapshot.feature_json) != snapshot.snapshot_hash:
                    raise V33RuntimeError("warm-up build snapshot content hash mismatch")
                manifest.append(
                    {
                        "snapshot_id": snapshot.id,
                        "cutoff_date": cutoff.isoformat(),
                        "snapshot_hash": snapshot.snapshot_hash,
                    }
                )
            anchor = build_warmup_anchor_payload(
                state,
                snapshot_manifest=manifest,
                eligible_cutoff_dates=eligible_dates,
                sampling_seed=self.sampling_seed,
            )
            existing = dict(run.result_json or {}).get("warmup_anchor")
            if existing is not None:
                if _hash(existing) != _hash(anchor):
                    raise V33RuntimeError(
                        "warm-up anchor changed during an immutable bootstrap"
                    )
                return
            run.result_json = {**dict(run.result_json or {}), "warmup_anchor": anchor}
            run.stages_json = [
                *run.stages_json,
                {
                    "stage": "warmup_anchor_persisted",
                    "state_hash": state.state_hash,
                    "at": now.isoformat(),
                },
            ]

    def latest_state(self, market: str) -> V33ModelState | None:
        with self.sessions() as session:
            completed = session.scalar(
                select(V33TrainingIteration)
                .where(
                    V33TrainingIteration.model_market == market,
                    V33TrainingIteration.status == "completed",
                )
                .order_by(V33TrainingIteration.iteration_number.desc())
            )
            if completed is None:
                return None
            version = session.scalar(
                select(V33ModelVersion).where(
                    V33ModelVersion.model_market == market,
                    V33ModelVersion.version == completed.champion_model_version,
                )
            )
        if version is None:
            raise V33RuntimeError("completed iteration has no model artifact")
        return state_from_payload(version.parameters_json)

    def save_state(self, state: V33ModelState) -> None:
        if self.run_id is None:
            raise V33RuntimeError("repository run is not bound")
        now = self.now_provider()
        payload = state_to_payload(state)
        with self.sessions() as session, session.begin():
            run = session.get(V33TrainingRun, self.run_id)
            if run is None:
                raise V33RuntimeError("training run disappeared while saving checkpoint")
            parent_iteration = int(run.source_parent_iteration or 0)
            run.current_stage = f"iteration_{state.iteration_number}"
            run.requested_through_week = _week_key(state.trained_through)
            run.created_iteration_count = max(0, state.iteration_number - parent_iteration)
            run.stages_json = [
                *run.stages_json,
                {
                    "stage": run.current_stage,
                    "week": run.requested_through_week,
                    "at": now.isoformat(),
                },
            ]
            previous = session.scalar(
                select(V33ModelVersion)
                .where(V33ModelVersion.model_market == state.market)
                .order_by(V33ModelVersion.created_at.desc())
            )
            version_id = f"{state.market}:{state.version}"
            version = session.get(V33ModelVersion, version_id)
            if version is None:
                session.add(
                    V33ModelVersion(
                        id=version_id,
                        model_market=state.market,
                        model_type="hybrid_20w",
                        version=state.version,
                        parent_version=None if previous is None else previous.version,
                        feature_version=TRAINING_FEATURE_VERSION,
                        methodology_version=METHODOLOGY_VERSION,
                        trained_through_date=state.trained_through,
                        validation_start_date=None,
                        validation_end_date=state.trained_through,
                        random_seed=self.sampling_seed,
                        parameters_json=payload,
                        metrics_json=_jsonable(state.validation_metrics),
                        artifact_path=None,
                        artifact_hash=state.state_hash,
                        status="champion",
                        created_at=now,
                    )
                )
                if previous is not None and previous.status == "champion":
                    previous.status = "superseded"
            optimizer = session.scalar(
                select(V33OptimizerState).where(V33OptimizerState.model_market == state.market)
            )
            optimizer_values = {
                "iteration_number": state.iteration_number,
                "last_training_week_key": _week_key(state.trained_through),
                "champion_model_version": state.version,
                "optimizer_memory_json": _jsonable(state.optimizer_memory),
                "last_successful_training_at": now,
                "next_training_eligible_at": now + timedelta(days=7),
                "state_hash": state.state_hash,
                "updated_at": now,
            }
            if optimizer is None:
                session.add(V33OptimizerState(model_market=state.market, **optimizer_values))
            else:
                for key, value in optimizer_values.items():
                    setattr(optimizer, key, value)
            checkpoint = session.scalar(
                select(V33TrainingCheckpoint).where(
                    V33TrainingCheckpoint.training_run_id == self.run_id,
                    V33TrainingCheckpoint.model_market == state.market,
                )
            )
            checkpoint_values = {
                "iteration_number": state.iteration_number,
                "week_key": _week_key(state.trained_through),
                "state_json": payload,
                "state_hash": state.state_hash,
                "updated_at": now,
            }
            if checkpoint is None:
                session.add(
                    V33TrainingCheckpoint(
                        training_run_id=self.run_id,
                        model_market=state.market,
                        **checkpoint_values,
                    )
                )
            else:
                for key, value in checkpoint_values.items():
                    setattr(checkpoint, key, value)
        self._states[state.iteration_number] = state

    def save_iteration(self, iteration: V33IterationRecord) -> None:
        if self.run_id is None:
            raise V33RuntimeError("repository run is not bound")
        now = self.now_provider()
        forecast_payload = _forecast_payload(iteration.forecast)
        forecast_hash = _hash(forecast_payload)
        evaluation = dict(iteration.evaluation or {})
        with self.sessions() as session, session.begin():
            existing = session.scalar(
                select(V33TrainingIteration).where(
                    V33TrainingIteration.model_market == iteration.market,
                    V33TrainingIteration.iteration_number == iteration.iteration_number,
                )
            )
            if existing is None:
                snapshot_id = self.snapshot_ids.get(iteration.cutoff_date)
                if snapshot_id is None:
                    raise V33RuntimeError(f"missing feature snapshot for {iteration.cutoff_date}")
                state = self._states.get(iteration.iteration_number)
                existing = V33TrainingIteration(
                    training_run_id=self.run_id,
                    feature_snapshot_id=snapshot_id,
                    model_market=iteration.market,
                    iteration_number=iteration.iteration_number,
                    parent_iteration_number=(None if iteration.iteration_number <= 1 else iteration.iteration_number - 1),
                    week_key=_week_key(iteration.cutoff_date),
                    cutoff_date=iteration.cutoff_date,
                    status="completed",
                    maturity_status=iteration.status,
                    champion_model_version=iteration.champion_version,
                    challenger_model_version=(iteration.champion_version if iteration.challenger_promoted else None),
                    promoted=iteration.challenger_promoted,
                    forecast_json=forecast_payload,
                    evaluation_json=_jsonable(evaluation),
                    optimizer_state_json={} if state is None else _jsonable(state.optimizer_memory),
                    composite_loss=_decimal(evaluation.get("composite_loss")),
                    wis_loss=_decimal(evaluation.get("wis_pinball_loss")),
                    price_turn_error_days=(None if evaluation.get("price_turn_error_days") is None else int(round(float(evaluation["price_turn_error_days"])))),
                    dif_zero_error_days=(None if evaluation.get("dif_turn_error_days") is None else int(round(float(evaluation["dif_turn_error_days"])))),
                    completed_at=now,
                    audit_json={
                        "state_hash": iteration.state_hash,
                        "parent_state_hash": iteration.parent_state_hash,
                        "promotion_reason": iteration.promotion_reason,
                        "dif_metric_semantics": "d(DIF)/dt zero; not DIF axis zero",
                        "sampling_seed": self.sampling_seed,
                    },
                )
                session.add(existing)
                session.flush()
            else:
                if _hash(existing.forecast_json) != forecast_hash:
                    raise V33RuntimeError("refusing to recompute/overwrite an issued forecast")
                existing.maturity_status = iteration.status
                existing.evaluation_json = _jsonable(evaluation)
                existing.composite_loss = _decimal(evaluation.get("composite_loss"))
                existing.wis_loss = _decimal(evaluation.get("wis_pinball_loss"))
                existing.price_turn_error_days = None if evaluation.get("price_turn_error_days") is None else int(round(float(evaluation["price_turn_error_days"])) )
                existing.dif_zero_error_days = None if evaluation.get("dif_turn_error_days") is None else int(round(float(evaluation["dif_turn_error_days"])) )

            forecast = session.scalar(
                select(V33ForecastRow).where(
                    V33ForecastRow.model_market == iteration.market,
                    V33ForecastRow.forecast_date == iteration.cutoff_date,
                    V33ForecastRow.model_version == iteration.champion_version,
                )
            )
            if forecast is None:
                forecast = V33ForecastRow(
                    model_market=iteration.market,
                    target_instrument_id=self._instrument_id(session, iteration.market),
                    training_iteration_id=existing.id,
                    feature_snapshot_id=existing.feature_snapshot_id,
                    forecast_date=iteration.cutoff_date,
                    model_version=iteration.champion_version,
                    horizon_weeks=HORIZON_WEEKS,
                    maturity_status=iteration.status,
                    weekly_direction=iteration.forecast.direction,
                    confidence=_decimal(iteration.forecast.confidence),
                    p10_path_json=list(iteration.forecast.p10),
                    p50_path_json=list(iteration.forecast.p50),
                    p90_path_json=list(iteration.forecast.p90),
                    expected_path_json=list(iteration.forecast.expected),
                    up_probability=_decimal(iteration.forecast.up_probability),
                    sideways_probability=_decimal(iteration.forecast.sideways_probability),
                    down_probability=_decimal(iteration.forecast.down_probability),
                    expected_max_drawdown=_decimal(iteration.forecast.expected_max_drawdown),
                    target_position=None,
                    payload_json=forecast_payload,
                    forecast_hash=forecast_hash,
                    created_at=now,
                )
                session.add(forecast)
                session.flush()
            else:
                if forecast.forecast_hash != forecast_hash:
                    raise V33RuntimeError("saved forecast hash changed at maturity")
                forecast.maturity_status = iteration.status

            model_evaluation = session.scalar(
                select(V33ModelEvaluation).where(V33ModelEvaluation.forecast_id == forecast.id)
            )
            evaluation_values = self._evaluation_values(iteration.status, evaluation, now)
            if model_evaluation is None:
                session.add(V33ModelEvaluation(forecast_id=forecast.id, **evaluation_values))
            else:
                for key, value in evaluation_values.items():
                    setattr(model_evaluation, key, value)

    @staticmethod
    def _instrument_id(session: Session, market: str) -> int:
        value = session.scalar(select(Instrument.id).where(Instrument.code == market))
        if value is None:
            raise V33RuntimeError(f"missing instrument {market}")
        return int(value)

    @staticmethod
    def _evaluation_values(status: str, metrics: Mapping[str, Any], now: datetime) -> dict[str, Any]:
        full = status == "full"
        turn_error = metrics.get("price_turn_error_days")
        return {
            "maturity_status": status,
            "composite_loss": _decimal(metrics.get("composite_loss")) if full else None,
            "wis_loss": _decimal(metrics.get("wis_pinball_loss")) if full else None,
            "pinball_loss": _decimal(metrics.get("wis_pinball_loss")) if full else None,
            "p50_path_error": _decimal(metrics.get("median_path_loss")) if full else None,
            "terminal_return_error": _decimal(metrics.get("terminal_loss")) if full else None,
            "brier_score": _decimal(metrics.get("brier_score")) if full else None,
            "calibration_error": _decimal(metrics.get("coverage_interval_loss")) if full else None,
            "interval_coverage": _decimal(metrics.get("coverage")) if full else None,
            "interval_width": _decimal(metrics.get("coverage_interval_loss")) if full else None,
            "dif_turn_error_days": (None if not full or metrics.get("dif_turn_error_days") is None else int(round(float(metrics["dif_turn_error_days"])))),
            "price_turn_error_days": (None if not full or turn_error is None else int(round(float(turn_error)))),
            "coverage_3d": _decimal(1 if full and turn_error is not None and float(turn_error) <= 3 else 0) if full and turn_error is not None else None,
            "coverage_5d": _decimal(1 if full and turn_error is not None and float(turn_error) <= 5 else 0) if full and turn_error is not None else None,
            "coverage_10d": _decimal(1 if full and turn_error is not None and float(turn_error) <= 10 else 0) if full and turn_error is not None else None,
            "turning_direction_f1": None,
            "false_turn_alert_rate": None,
            "payload_json": _jsonable(metrics),
            "evaluated_at": now if full else None,
        }


class V33RuntimeService:
    """Crash-safe bootstrap, weekly update and pure current-market analysis."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        calendar: CalendarProvider,
        investment_calendar: InvestmentCalendarService | None = None,
        analysis_days: int = DEFAULT_ANALYSIS_DAYS,
        warmup_weeks: int = DEFAULT_WARMUP_WEEKS,
        sampling_seed: int = DEFAULT_SAMPLING_SEED,
        now_provider: Callable[[], datetime] = utc_now,
        auxiliary_refresh: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.sessions = sessions
        self.calendar = calendar
        self.investment_calendar = investment_calendar or InvestmentCalendarService(sessions)
        self.analysis_days = analysis_days
        self.warmup_weeks = warmup_weeks
        self.sampling_seed = sampling_seed
        self.now_provider = now_provider
        self.auxiliary_refresh = auxiliary_refresh
        self.feature_service = V33FeatureService()
        self._executor = DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="v33-runtime")
        self._lock = Lock()
        self._active: dict[str, str] = {}

    def start_bootstrap(self, market: str) -> dict[str, Any]:
        return self._start(market, "bootstrap", self.bootstrap_sync)

    def start_incremental(self, market: str) -> dict[str, Any]:
        return self._start(market, "incremental", self.incremental_sync)

    def _start(self, market: str, run_type: str, target: Callable[..., dict[str, Any]]) -> dict[str, Any]:
        V33TrainingService.validate_market(market)
        with self._lock:
            if market in self._active:
                return {"status": "running", "run_id": self._active[market], "market": market}
            run_id = self._create_run(market, run_type)
            self._active[market] = run_id

        def execute() -> None:
            try:
                target(market, run_id=run_id)
            finally:
                with self._lock:
                    self._active.pop(market, None)

        self._executor.submit(execute)
        return {"status": "running", "run_id": run_id, "market": market}

    def bootstrap_sync(self, market: str, *, run_id: str | None = None) -> dict[str, Any]:
        V33TrainingService.validate_market(market)
        actual_run_id = run_id or self._create_run(market, "bootstrap")
        try:
            auxiliary_audit = self._refresh_auxiliary_sources(actual_run_id)
            self._stage(actual_run_id, "build_leakage_safe_weekly_samples")
            points, formal_start, expected_count, snapshot_ids = self._build_points(market)
            repository = SQLiteV33TrainingRepository(
                self.sessions,
                run_id=actual_run_id,
                snapshot_ids=snapshot_ids,
                sampling_seed=self.sampling_seed,
                now_provider=self.now_provider,
            )
            previous = repository.latest_state(market)
            if previous is not None:
                self._set_run_parent(actual_run_id, previous.iteration_number)
            with self.sessions() as session:
                last_iteration = session.scalar(
                    select(func.max(V33TrainingIteration.iteration_number)).where(
                        V33TrainingIteration.model_market == market
                    )
                ) or 0
                last_cutoff = session.scalar(
                    select(func.max(V33TrainingIteration.cutoff_date)).where(
                        V33TrainingIteration.model_market == market
                    )
                )
            audit_start = formal_start if last_cutoff is None else next(
                (point.cutoff_date for point in points if point.cutoff_date > last_cutoff),
                None,
            )
            if audit_start is None:
                self._mature_existing(market, points, repository)
                return self._complete_run(actual_run_id, {"status": "not_due", "reason": "baseline_already_complete", "auxiliary_refresh": auxiliary_audit})
            remaining = sum(point.cutoff_date >= audit_start for point in points)
            self._stage(actual_run_id, "progressive_purged_training")
            training = V33TrainingService(repository)
            result = training.train_progressive(
                market,
                points,
                initial_state=previous,
                audit_start_date=audit_start,
                expected_iteration_count=remaining,
                weekly_sampling_seed=self.sampling_seed,
            )
            matured = self._mature_existing(market, points, repository)
            with self.sessions() as session:
                total = int(session.scalar(select(func.count(V33TrainingIteration.id)).where(V33TrainingIteration.model_market == market)) or 0)
            if total != expected_count:
                raise V33RuntimeError(f"audited iteration count {total} != expected natural weeks {expected_count}")
            payload = self._progressive_payload(result)
            payload.update({"status": "completed", "expected_iteration_count": expected_count, "audited_iteration_count": total, "matured_existing_count": matured, "auxiliary_refresh": auxiliary_audit})
            return self._complete_run(actual_run_id, payload)
        except Exception as error:
            self._fail_run(actual_run_id, error)
            raise

    def incremental_sync(self, market: str, *, run_id: str | None = None) -> dict[str, Any]:
        V33TrainingService.validate_market(market)
        actual_run_id = run_id or self._create_run(market, "incremental")
        try:
            repository = SQLiteV33TrainingRepository(
                self.sessions,
                run_id=actual_run_id,
                sampling_seed=self.sampling_seed,
                now_provider=self.now_provider,
            )
            state = repository.latest_state(market)
            if state is None:
                raise V33RuntimeError("bootstrap V3.3 before incremental training")
            self._set_run_parent(actual_run_id, state.iteration_number)
            with self.sessions() as session:
                optimizer = session.scalar(select(V33OptimizerState).where(V33OptimizerState.model_market == market))
            now = self.now_provider()
            if optimizer is not None and now < optimizer.next_training_eligible_at:
                return self._complete_run(actual_run_id, {"status": "not_due", "reason": "seven_day_interval_not_elapsed", "next_eligible_at": optimizer.next_training_eligible_at.isoformat()})
            auxiliary_audit = self._refresh_auxiliary_sources(actual_run_id)
            self._stage(actual_run_id, "build_leakage_safe_weekly_samples")
            points, _formal_start, _expected, snapshot_ids = self._build_points(market)
            repository.bind_run(actual_run_id, snapshot_ids)
            new_points = [point for point in points if _week_key(point.cutoff_date) > _week_key(state.trained_through)]
            if not new_points:
                self._mature_existing(market, points, repository)
                return self._complete_run(actual_run_id, {"status": "not_due", "reason": "no_new_complete_week"})
            current = new_points[-1]  # a multiweek gap still adds at most one
            eligible = [
                point for point in points
                if point.cutoff_date < current.cutoff_date
                and point.future_path is not None
                and point.outcome_available_date is not None
                and point.outcome_available_date <= current.cutoff_date
            ]
            training = V33TrainingService(repository)
            new_state, record = training.train_iteration(current, eligible, state)
            repository.save_state(new_state)
            repository.save_iteration(record)
            matured = self._mature_existing(market, points, repository)
            return self._complete_run(
                actual_run_id,
                {
                    "status": "completed",
                    "created_iteration_count": 1,
                    "iteration_number": new_state.iteration_number,
                    "sampled_date": current.sampled_date.isoformat() if current.sampled_date else None,
                    "skipped_complete_weeks": max(0, len(new_points) - 1),
                    "matured_existing_count": matured,
                    "auxiliary_refresh": auxiliary_audit,
                },
            )
        except Exception as error:
            self._fail_run(actual_run_id, error)
            raise

    def run_analysis(self, market: str) -> dict[str, Any]:
        V33TrainingService.validate_market(market)
        run_id = f"V33-A-{market}-{uuid4().hex[:12]}"
        now = self.now_provider()
        target_id = self._instrument_id(market)
        with self.sessions() as session, session.begin():
            session.add(
                V33AnalysisRun(
                    id=run_id,
                    model_market=market,
                    target_instrument_id=target_id,
                    status="running",
                    current_stage="build_current_snapshot",
                    data_gate_status="pending",
                    started_at=now,
                    stages_json=[{"stage": "build_current_snapshot", "at": now.isoformat()}],
                    result_json={},
                )
            )
        before = self._training_identity(market)
        fingerprint_before = self._training_fingerprint(market)
        try:
            snapshot, snapshot_id = self._build_current_snapshot(market)
            repository = SQLiteV33TrainingRepository(self.sessions, now_provider=self.now_provider)
            training = V33TrainingService(repository)
            analysis = V33AnalysisService(
                training=training,
                champions=repository,
                calendar=self.calendar,
            )
            current_position = int(self.investment_calendar.current_positions().get(market) or 0)
            result = analysis.run(market, snapshot, current_position=current_position)
            forecast_id = self._persist_analysis_forecast(snapshot_id, result)
            after = self._training_identity(market)
            fingerprint_after = self._training_fingerprint(market)
            if before != after or fingerprint_before != fingerprint_after:
                raise V33RuntimeError(
                    "analysis changed iteration/model/optimizer/Champion identity"
                )
            identity_names = (
                "iteration_count",
                "model_count",
                "optimizer_count",
                "optimizer_state_hash",
            )
            result["training_identity_before"] = dict(zip(identity_names, before))
            result["training_identity_after"] = dict(zip(identity_names, after))
            result["training_identity_unchanged"] = True
            result["training_fingerprint_before"] = fingerprint_before
            result["training_fingerprint_after"] = fingerprint_after
            result["training_fingerprint_unchanged"] = True
            with self.sessions() as session, session.begin():
                row = session.get(V33AnalysisRun, run_id)
                if row is None:
                    raise V33RuntimeError("analysis run disappeared")
                row.status = "completed"
                row.current_stage = "completed"
                row.model_version = result["model"]["version"]
                row.feature_snapshot_id = snapshot_id
                row.forecast_id = forecast_id
                row.price_data_as_of = snapshot.daily_as_of
                row.weekly_data_as_of = snapshot.weekly_as_of
                row.valuation_data_as_of = snapshot.cutoff_date if snapshot.weekly.get("pe") is not None else None
                row.macro_data_as_of = snapshot.cutoff_date if not snapshot.missing_masks.get("missing_family_macro", 1) else None
                row.data_gate_status = "degraded" if result["data_quality"]["degraded"] else "passed"
                row.completed_at = self.now_provider()
                row.stages_json = [*row.stages_json, {"stage": "completed", "at": row.completed_at.isoformat()}]
                row.result_json = _jsonable(result)
            return result
        except Exception as error:
            with self.sessions() as session, session.begin():
                row = session.get(V33AnalysisRun, run_id)
                if row is not None:
                    row.status = "failed"
                    row.current_stage = "failed"
                    row.error_code = type(error).__name__
                    row.error_message = str(error)
                    row.completed_at = self.now_provider()
            raise

    def training_status(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            row = session.get(V33TrainingRun, run_id)
        if row is None:
            raise KeyError(f"unknown V3.3 training run {run_id}")
        return {
            "run_id": row.id,
            "market": row.model_market,
            "run_type": row.run_type,
            "status": row.status,
            "current_stage": row.current_stage,
            "created_iteration_count": row.created_iteration_count,
            "started_at": row.started_at.isoformat(),
            "completed_at": None if row.completed_at is None else row.completed_at.isoformat(),
            "error_code": row.error_code,
            "error_message": row.error_message,
            "stages": row.stages_json,
            "result": row.result_json,
        }

    def latest_training_status(self, market: str) -> dict[str, Any] | None:
        V33TrainingService.validate_market(market)
        with self.sessions() as session:
            row = session.scalar(
                select(V33TrainingRun).where(V33TrainingRun.model_market == market).order_by(V33TrainingRun.started_at.desc())
            )
        return None if row is None else self.training_status(row.id)

    def model_status(self, market: str) -> dict[str, Any]:
        """Return the compact polling contract used by the local web UI."""

        V33TrainingService.validate_market(market)
        champion = self.champion(market)
        with self.sessions() as session:
            iteration_count = int(
                session.scalar(
                    select(func.count(V33TrainingIteration.id)).where(
                        V33TrainingIteration.model_market == market
                    )
                )
                or 0
            )
            pending_count = int(
                session.scalar(
                    select(func.count(V33TrainingIteration.id)).where(
                        V33TrainingIteration.model_market == market,
                        V33TrainingIteration.maturity_status == "pending",
                    )
                )
                or 0
            )
            optimizer = session.scalar(
                select(V33OptimizerState).where(
                    V33OptimizerState.model_market == market
                )
            )
            running = session.scalar(
                select(V33TrainingRun)
                .where(
                    V33TrainingRun.model_market == market,
                    V33TrainingRun.status == "running",
                )
                .order_by(V33TrainingRun.started_at.desc())
            )
            latest = session.scalar(
                select(V33TrainingRun)
                .where(V33TrainingRun.model_market == market)
                .order_by(V33TrainingRun.started_at.desc())
            )

        if running is not None:
            message = f"训练进行中：{running.current_stage}"
        elif champion is None:
            message = "尚未完成首次十年逐周训练"
        elif latest is not None and latest.status == "failed":
            message = latest.error_message or "上次训练失败，可从检查点重试"
        elif latest is not None and latest.result_json.get("reason"):
            message = str(latest.result_json["reason"])
        else:
            message = "最新 Champion 已就绪"

        return {
            "instrument_code": market,
            "is_training": running is not None,
            "active_run_id": None if running is None else running.id,
            "bootstrapped": champion is not None,
            "iteration_count": iteration_count,
            "champion_version": None if champion is None else champion["version"],
            "last_training_week_key": (
                None
                if optimizer is None
                else optimizer.last_training_week_key
            ),
            "last_successful_training_at": (
                None
                if optimizer is None or optimizer.last_successful_training_at is None
                else optimizer.last_successful_training_at.isoformat()
            ),
            "next_training_eligible_at": (
                None
                if optimizer is None or optimizer.next_training_eligible_at is None
                else optimizer.next_training_eligible_at.isoformat()
            ),
            "pending_count": pending_count,
            "message": message,
        }

    def model_statuses(self) -> dict[str, Any]:
        return {
            "implementation_revision": "V3.3-20W",
            "markets": {
                market: self.model_status(market)
                for market in ("399006", "159941")
            },
        }

    def champion(self, market: str) -> dict[str, Any] | None:
        V33TrainingService.validate_market(market)
        state = SQLiteV33TrainingRepository(self.sessions).latest_state(market)
        if state is None:
            return None
        return {
            "market": market,
            "version": state.version,
            "iteration_number": state.iteration_number,
            "trained_through": state.trained_through.isoformat(),
            "state_hash": state.state_hash,
            "training_sample_count": state.training_sample_count,
            "validation_metrics": dict(state.validation_metrics),
            "optimizer_memory": dict(state.optimizer_memory),
        }

    def curve(self, market: str) -> dict[str, Any]:
        V33TrainingService.validate_market(market)
        with self.sessions() as session:
            rows = list(
                session.scalars(
                    select(V33TrainingIteration)
                    .where(V33TrainingIteration.model_market == market)
                    .order_by(V33TrainingIteration.iteration_number)
                )
            )
        points = [
            {
                "iteration": row.iteration_number,
                "cutoff_date": row.cutoff_date.isoformat(),
                "status": row.maturity_status,
                "loss": None if row.composite_loss is None else float(row.composite_loss),
                "price_turn_error_days": row.price_turn_error_days,
                "dif_turn_error_days": row.dif_zero_error_days,
                "promoted": row.promoted,
                "champion_version": row.champion_model_version,
            }
            for row in rows
        ]
        iterations = [
            {
                "market": market,
                "iteration_number": row.iteration_number,
                "cutoff_date": row.cutoff_date.isoformat(),
                "parent_state_hash": row.audit_json.get("parent_state_hash"),
                "state_hash": row.audit_json.get("state_hash"),
                "champion_version": row.champion_model_version,
                "challenger_promoted": row.promoted,
                "promotion_reason": row.audit_json.get("promotion_reason"),
                "forecast": dict(row.forecast_json),
                "status": row.maturity_status,
                "evaluation": (
                    dict(row.evaluation_json)
                    if row.maturity_status == "full"
                    else None
                ),
            }
            for row in rows
        ]
        champion = self.champion(market)
        full_count = sum(point["status"] == "full" for point in points)
        pending_count = sum(point["status"] == "pending" for point in points)
        return {
            "implementation_revision": "V3.3-20W",
            "market": market,
            "state": "untrained" if champion is None else str(champion["version"]),
            "iterations": iterations,
            "curve": points,
            "points": points,
            "iteration_count": len(points),
            "expected_iteration_count": len(points),
            "audited_iteration_count": len(points),
            "full_count": full_count,
            "pending_count": pending_count,
            "weekly_sampling_seed": self.sampling_seed,
            "sampled_dates": [point["cutoff_date"] for point in points],
            "x_axis": "iteration",
            "y_axis": "trading_day_error_and_loss",
        }

    def latest_analysis(self, market: str) -> dict[str, Any] | None:
        V33TrainingService.validate_market(market)
        with self.sessions() as session:
            row = session.scalar(
                select(V33AnalysisRun)
                .where(V33AnalysisRun.model_market == market, V33AnalysisRun.status == "completed")
                .order_by(V33AnalysisRun.completed_at.desc())
            )
        return None if row is None else dict(row.result_json)

    def recover_interrupted_runs(self) -> int:
        """Mark orphaned running tasks recoverable; checkpoints remain intact."""

        recovered = 0
        with self.sessions() as session, session.begin():
            rows = list(session.scalars(select(V33TrainingRun).where(V33TrainingRun.status == "running")))
            for row in rows:
                row.status = "failed"
                row.current_stage = "interrupted_recoverable"
                row.error_code = "PROCESS_RESTART"
                row.error_message = "Task interrupted; start bootstrap again to resume from checkpoint"
                row.completed_at = self.now_provider()
                recovered += 1
        return recovered

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _build_points(
        self, market: str
    ) -> tuple[tuple[V33TrainingPoint, ...], date, int, dict[date, str]]:
        bars, benchmark, observations = self._load_source_data(market)
        if not bars:
            raise V33RuntimeError(f"{market} has no daily target data")
        completed = self._complete_week_sessions(market, [bar.trade_date for bar in bars])
        sample_dates = deterministic_weekly_sample_dates(
            [day for sessions in completed.values() for day in sessions],
            market,
            seed=self.sampling_seed,
        )
        if not sample_dates:
            raise V33RuntimeError("no complete natural weeks")
        formal_start = sample_dates[-1] - timedelta(days=self.analysis_days)
        expected_formal_dates = [day for day in sample_dates if day >= formal_start]
        build_start = formal_start - timedelta(weeks=self.warmup_weeks)
        previous_week_end: dict[tuple[int, int], date] = {}
        ordered_weeks = sorted(completed)
        for index in range(1, len(ordered_weeks)):
            previous_week_end[ordered_weeks[index]] = max(completed[ordered_weeks[index - 1]])

        snapshots: list[FeatureSnapshot] = []
        snapshot_ids: dict[date, str] = {}
        for sampled_date in sample_dates:
            if sampled_date < build_start:
                continue
            week = sampled_date.isocalendar()[:2]
            weekly_cutoff = previous_week_end.get(week)
            if weekly_cutoff is None:
                continue
            cutoff_at = _china_close(sampled_date)
            try:
                snapshot = self.feature_service.build_snapshot(
                    market,
                    [bar for bar in bars if bar.trade_date <= sampled_date],
                    sampled_date,
                    observations=self._observations_visible_at(observations, cutoff_at, sampled_date),
                    benchmark_bars=[bar for bar in benchmark if bar.trade_date <= sampled_date],
                    weekly_cutoff_date=weekly_cutoff,
                    cutoff_at=cutoff_at,
                )
            except V33FeatureError:
                if sampled_date >= formal_start:
                    raise
                continue
            snapshots.append(snapshot)
            snapshot_ids[sampled_date] = self._persist_feature_snapshot(
                snapshot, cutoff_at, feature_version=TRAINING_FEATURE_VERSION
            )
        actual_formal_dates = [snapshot.cutoff_date for snapshot in snapshots if snapshot.cutoff_date >= formal_start]
        if actual_formal_dates != expected_formal_dates:
            missing = sorted(set(expected_formal_dates) - set(actual_formal_dates))
            raise V33RuntimeError(f"formal ten-year natural weeks missing feature snapshots: {missing[:5]}")
        points = build_training_points(snapshots, sampling_seed=self.sampling_seed)
        return points, actual_formal_dates[0], len(actual_formal_dates), snapshot_ids

    def _build_current_snapshot(self, market: str) -> tuple[FeatureSnapshot, str]:
        bars, benchmark, observations = self._load_source_data(market)
        completed = self._complete_week_sessions(market, [bar.trade_date for bar in bars])
        if not completed:
            raise V33RuntimeError("no complete weekly bar for current analysis")
        weekly_cutoff = max(max(days) for days in completed.values())
        current_date = bars[-1].trade_date
        analysis_at = self.now_provider()
        if analysis_at.tzinfo is None:
            analysis_at = analysis_at.replace(tzinfo=timezone.utc)
        analysis_at = analysis_at.astimezone(timezone.utc)
        snapshot = self.feature_service.build_snapshot(
            market,
            bars,
            current_date,
            observations=self._observations_visible_at(
                observations, analysis_at, current_date
            ),
            benchmark_bars=benchmark,
            weekly_cutoff_date=weekly_cutoff,
            cutoff_at=analysis_at,
        )
        snapshot = replace(
            snapshot,
            provenance={
                **dict(snapshot.provenance),
                "price_as_of": current_date.isoformat(),
                "weekly_price_as_of": weekly_cutoff.isoformat(),
                "analysis_as_of": analysis_at.isoformat(),
                "pit_cutoff_mode": "live_analysis_clock",
            },
        )
        # A live snapshot can legitimately change while the last price date is
        # unchanged (for example a weekend PE/PB or NAV release).  Give each
        # analysis-time view an immutable identity instead of overwriting the
        # historical sampled snapshot or a previous live view.
        live_feature_version = (
            f"{ANALYSIS_FEATURE_VERSION}-{analysis_at.strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        return snapshot, self._persist_feature_snapshot(
            snapshot,
            analysis_at,
            feature_version=live_feature_version,
        )

    def _load_source_data(
        self, market: str
    ) -> tuple[list[PriceBar], list[PriceBar], list[PointInTimeObservation]]:
        with self.sessions() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == market))
            if instrument is None:
                raise V33RuntimeError(f"missing instrument {market}")
            target_rows = list(
                session.scalars(
                    select(MarketPrice)
                    .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
                    .order_by(MarketPrice.trade_date)
                )
            )
            benchmark_rows: list[MarketPrice] = []
            allowed_observation_instruments = {instrument.id}
            if market == "159941":
                ndx = session.scalar(select(Instrument).where(Instrument.code == "NDX"))
                if ndx is not None:
                    allowed_observation_instruments.add(ndx.id)
                    benchmark_rows = list(
                        session.scalars(
                            select(MarketPrice)
                            .where(MarketPrice.instrument_id == ndx.id, MarketPrice.timeframe == "daily")
                            .order_by(MarketPrice.trade_date)
                        )
                    )
            pit_rows = list(
                session.scalars(
                    select(V33PointInTimeObservation)
                    .where(V33PointInTimeObservation.quality_status.in_(("accepted", "accepted_with_warnings")))
                    .order_by(V33PointInTimeObservation.effective_date, V33PointInTimeObservation.available_at)
                )
            )
            pit_rows = [
                row
                for row in pit_rows
                if row.instrument_id is None
                or row.instrument_id in allowed_observation_instruments
            ]
            valuations = list(
                session.scalars(
                    select(ValuationRecord)
                    .where(ValuationRecord.instrument_id == instrument.id)
                    .order_by(ValuationRecord.valuation_date)
                )
            )
        bars = [self.feature_service._row_to_bar(row) for row in target_rows]
        benchmark = [self.feature_service._row_to_bar(row) for row in benchmark_rows]
        observations = [
            self.feature_service._pit_row_to_observation(row, market)
            for row in pit_rows
            if row.numeric_value is not None
        ]
        for row in valuations:
            stamp = _china_close(row.valuation_date)
            raw = dict(row.raw_values or {})
            for series, value in (("pe", row.pe_ratio), ("pb", row.pb_ratio)):
                if value is not None:
                    observations.append(
                        PointInTimeObservation(
                            market=market,
                            series=series,
                            value=float(value),
                            effective_date=row.valuation_date,
                            published_at=stamp,
                            available_at=stamp,
                            source=str(raw.get("source") or "legacy_valuation_records"),
                            vintage=str(raw.get("vintage") or row.valuation_date.isoformat()),
                            quality_status=str(raw.get("quality_status") or "legacy_date_only"),
                        )
                    )
        return bars, benchmark, observations

    def _complete_week_sessions(
        self, market: str, actual_sessions: Sequence[date]
    ) -> dict[tuple[int, int], tuple[date, ...]]:
        actual_set = set(actual_sessions)
        grouped: dict[tuple[int, int], list[date]] = {}
        for day in sorted(actual_set):
            grouped.setdefault(day.isocalendar()[:2], []).append(day)
        complete: dict[tuple[int, int], tuple[date, ...]] = {}
        for (year, week), days in grouped.items():
            monday = date.fromisocalendar(year, week, 1)
            # 159941 must stay identifiable as the Shenzhen-listed target at
            # the calendar boundary.  The production adapter may share the
            # mainland holiday implementation internally, but callers must
            # never silently relabel it as another instrument.
            expected = tuple(
                self.calendar.sessions(market, monday, monday + timedelta(days=6))
            )
            if expected and set(expected).issubset(actual_set):
                complete[(year, week)] = expected
        return complete

    @staticmethod
    def _observations_visible_at(
        observations: Sequence[PointInTimeObservation],
        cutoff_at: datetime,
        cutoff_date: date,
    ) -> list[PointInTimeObservation]:
        output: list[PointInTimeObservation] = []
        for observation in observations:
            available = observation.available_at
            if available.tzinfo is None:
                available = available.replace(tzinfo=timezone.utc)
            if available <= cutoff_at and observation.effective_date <= cutoff_date:
                output.append(observation)
        return output

    def _persist_feature_snapshot(
        self,
        snapshot: FeatureSnapshot,
        cutoff_at: datetime,
        *,
        feature_version: str,
    ) -> str:
        payload = _snapshot_payload(snapshot)
        snapshot_hash = _hash(payload)
        version_tag = hashlib.sha256(feature_version.encode("utf-8")).hexdigest()[:6]
        snapshot_id = f"V33-FS-{snapshot.market}-{snapshot.cutoff_date.isoformat()}-{version_tag}-{snapshot_hash[:8]}"
        with self.sessions() as session, session.begin():
            existing = session.scalar(
                select(V33FeatureSnapshotRow).where(
                    V33FeatureSnapshotRow.model_market == snapshot.market,
                    V33FeatureSnapshotRow.cutoff_date == snapshot.cutoff_date,
                    V33FeatureSnapshotRow.feature_version == feature_version,
                )
            )
            if existing is not None:
                if existing.snapshot_hash != snapshot_hash:
                    raise V33RuntimeError("feature snapshot changed for an immutable cutoff")
                return existing.id
            session.add(
                V33FeatureSnapshotRow(
                    id=snapshot_id,
                    model_market=snapshot.market,
                    target_instrument_id=self._instrument_id_in_session(session, snapshot.market),
                    cutoff_date=snapshot.cutoff_date,
                    cutoff_available_at=cutoff_at,
                    daily_data_as_of=snapshot.daily_as_of,
                    weekly_data_as_of=snapshot.weekly_as_of,
                    source_max_available_at=cutoff_at,
                    daily_session_count=len(snapshot.daily_sequence),
                    weekly_bar_count=int(snapshot.provenance.get("weekly_bar_count", 0)),
                    feature_version=feature_version,
                    feature_json=payload,
                    missing_features_json=sorted(name for name, value in snapshot.missing_masks.items() if value),
                    source_release_ids_json=[],
                    quality_status="degraded" if any(snapshot.missing_masks.values()) else "accepted",
                    snapshot_hash=snapshot_hash,
                    leakage_audit_json={
                        "source_data_max_date": snapshot.source_data_max_date.isoformat(),
                        "cutoff_date": snapshot.cutoff_date.isoformat(),
                        "price_as_of": snapshot.provenance.get(
                            "price_as_of", snapshot.daily_as_of.isoformat()
                        ),
                        "analysis_as_of": snapshot.provenance.get(
                            "analysis_as_of", cutoff_at.isoformat()
                        ),
                        "no_future": snapshot.source_data_max_date <= snapshot.cutoff_date,
                        "weekly_precedes_sample": snapshot.weekly_as_of.isocalendar()[:2] != snapshot.cutoff_date.isocalendar()[:2],
                    },
                    created_at=self.now_provider(),
                )
            )
        return snapshot_id

    def _mature_existing(
        self,
        market: str,
        points: Sequence[V33TrainingPoint],
        repository: SQLiteV33TrainingRepository,
    ) -> int:
        by_cutoff = {point.cutoff_date: point for point in points}
        with self.sessions() as session:
            rows = list(
                session.scalars(
                    select(V33TrainingIteration).where(
                        V33TrainingIteration.model_market == market,
                        V33TrainingIteration.maturity_status == "pending",
                    )
                )
            )
        count = 0
        for row in rows:
            point = by_cutoff.get(row.cutoff_date)
            if point is None or point.future_path is None:
                continue
            forecast = _forecast_from_payload(row.forecast_json)
            metrics = V33TrainingService.evaluate(point, forecast)
            repository.save_iteration(
                V33IterationRecord(
                    market=market,
                    iteration_number=row.iteration_number,
                    cutoff_date=row.cutoff_date,
                    parent_state_hash=row.audit_json.get("parent_state_hash"),
                    state_hash=str(row.audit_json.get("state_hash") or forecast.model_state_hash),
                    champion_version=row.champion_model_version,
                    challenger_promoted=row.promoted,
                    promotion_reason=str(row.audit_json.get("promotion_reason") or "saved"),
                    forecast=forecast,
                    status="full",
                    evaluation=metrics,
                )
            )
            count += 1
        return count

    def _persist_analysis_forecast(self, snapshot_id: str, result: Mapping[str, Any]) -> int:
        now = self.now_provider()
        path = result["path"]
        turning = result["turning_points"]
        forecast_payload = _jsonable(result)
        issuance_version = f"{result['model']['version']}-A-{str(result['analysis_hash'])[:8]}"
        with self.sessions() as session, session.begin():
            forecast = V33ForecastRow(
                model_market=str(result["market"]),
                target_instrument_id=self._instrument_id_in_session(session, str(result["market"])),
                training_iteration_id=None,
                feature_snapshot_id=snapshot_id,
                forecast_date=date.fromisoformat(str(result["data_quality"]["cutoff_date"])),
                model_version=issuance_version,
                horizon_weeks=HORIZON_WEEKS,
                maturity_status="live",
                weekly_direction=str(path["direction"]),
                confidence=_decimal(path["confidence"]),
                p10_path_json=list(path["p10"]),
                p50_path_json=list(path["p50"]),
                p90_path_json=list(path["p90"]),
                expected_path_json=list(path["expected"]),
                up_probability=_decimal(path["up_probability"]),
                sideways_probability=_decimal(path["sideways_probability"]),
                down_probability=_decimal(path["down_probability"]),
                expected_max_drawdown=_decimal(path["expected_max_drawdown"]),
                dif_turn_start_date=self._date_or_none(turning["dif_derivative_zero"]["start"]),
                dif_turn_end_date=self._date_or_none(turning["dif_derivative_zero"]["end"]),
                price_turn_start_date=self._date_or_none(turning["price_turn"]["start"]),
                price_turn_end_date=self._date_or_none(turning["price_turn"]["end"]),
                target_position=int(result["position"]["target"]),
                execution_batches_json=list(result["advice"]["batches"]),
                qdii_decomposition_json=(
                    {key: value for key, value in result["features"]["weekly"].items() if key.startswith("qdii_")}
                    if result["market"] == "159941"
                    else None
                ),
                payload_json=forecast_payload,
                forecast_hash=_hash(forecast_payload),
                created_at=now,
            )
            session.add(forecast)
            session.flush()
            return int(forecast.id)

    def _training_identity(self, market: str) -> tuple[int, int, int, str | None]:
        with self.sessions() as session:
            return (
                int(session.scalar(select(func.count(V33TrainingIteration.id)).where(V33TrainingIteration.model_market == market)) or 0),
                int(session.scalar(select(func.count(V33ModelVersion.id)).where(V33ModelVersion.model_market == market)) or 0),
                int(session.scalar(select(func.count(V33OptimizerState.id)).where(V33OptimizerState.model_market == market)) or 0),
                session.scalar(select(V33OptimizerState.state_hash).where(V33OptimizerState.model_market == market)),
            )

    def _training_fingerprint(self, market: str) -> dict[str, Any]:
        """Return compact hashes for every mutable training-state boundary.

        Counts alone cannot detect an in-place update.  Analysis therefore
        fingerprints the iteration registry, model registry, complete
        optimizer payload and current Champion artifact before and after live
        inference.  Forecast rows with ``maturity_status='live'`` are
        intentionally outside this fingerprint because they are analysis
        outputs, not training observations or pending labels.
        """

        with self.sessions() as session:
            iteration_rows = session.execute(
                select(
                    V33TrainingIteration.id,
                    V33TrainingIteration.training_run_id,
                    V33TrainingIteration.feature_snapshot_id,
                    V33TrainingIteration.iteration_number,
                    V33TrainingIteration.parent_iteration_number,
                    V33TrainingIteration.week_key,
                    V33TrainingIteration.cutoff_date,
                    V33TrainingIteration.status,
                    V33TrainingIteration.maturity_status,
                    V33TrainingIteration.champion_model_version,
                    V33TrainingIteration.challenger_model_version,
                    V33TrainingIteration.promoted,
                    V33TrainingIteration.forecast_json,
                    V33TrainingIteration.evaluation_json,
                    V33TrainingIteration.optimizer_state_json,
                    V33TrainingIteration.composite_loss,
                    V33TrainingIteration.wis_loss,
                    V33TrainingIteration.price_turn_error_days,
                    V33TrainingIteration.dif_zero_error_days,
                    V33TrainingIteration.completed_at,
                    V33TrainingIteration.audit_json,
                )
                .where(V33TrainingIteration.model_market == market)
                .order_by(V33TrainingIteration.iteration_number)
            ).all()
            model_rows = session.execute(
                select(
                    V33ModelVersion.id,
                    V33ModelVersion.version,
                    V33ModelVersion.parent_version,
                    V33ModelVersion.trained_through_date,
                    V33ModelVersion.artifact_hash,
                    V33ModelVersion.status,
                )
                .where(V33ModelVersion.model_market == market)
                .order_by(V33ModelVersion.created_at, V33ModelVersion.id)
            ).all()
            optimizer = session.scalar(
                select(V33OptimizerState).where(
                    V33OptimizerState.model_market == market
                )
            )
            champion = session.scalar(
                select(V33ModelVersion)
                .where(
                    V33ModelVersion.model_market == market,
                    V33ModelVersion.status == "champion",
                )
                .order_by(V33ModelVersion.created_at.desc())
            )

        iteration_payload = [tuple(row) for row in iteration_rows]
        model_payload = [tuple(row) for row in model_rows]
        optimizer_payload = (
            None
            if optimizer is None
            else {
                "id": optimizer.id,
                "model_market": optimizer.model_market,
                "iteration_number": optimizer.iteration_number,
                "last_training_week_key": optimizer.last_training_week_key,
                "champion_model_version": optimizer.champion_model_version,
                "optimizer_memory": optimizer.optimizer_memory_json,
                "state_hash": optimizer.state_hash,
                "last_successful_training_at": optimizer.last_successful_training_at,
                "next_training_eligible_at": optimizer.next_training_eligible_at,
                "updated_at": optimizer.updated_at,
            }
        )
        champion_payload = (
            None
            if champion is None
            else {
                "version": champion.version,
                "parent_version": champion.parent_version,
                "feature_version": champion.feature_version,
                "methodology_version": champion.methodology_version,
                "trained_through_date": champion.trained_through_date,
                "random_seed": champion.random_seed,
                "artifact_hash": champion.artifact_hash,
                # Validate the actual serialized Champion bytes as well as its
                # declared artifact hash, without hashing all historical model
                # payloads on every user analysis.
                "parameters_hash": _hash(champion.parameters_json),
                "metrics_hash": _hash(champion.metrics_json),
                "status": champion.status,
                "created_at": champion.created_at,
            }
        )
        return {
            "iteration_registry_hash": _hash(iteration_payload),
            "model_registry_hash": _hash(model_payload),
            "optimizer_hash": None if optimizer_payload is None else _hash(optimizer_payload),
            "champion_hash": None if champion_payload is None else _hash(champion_payload),
            "champion_artifact_hash": (
                None if champion is None else champion.artifact_hash
            ),
        }

    def _create_run(self, market: str, run_type: str) -> str:
        run_id = f"V33-T-{market}-{uuid4().hex[:12]}"
        now = self.now_provider()
        with self.sessions() as session, session.begin():
            session.add(
                V33TrainingRun(
                    id=run_id,
                    model_market=market,
                    target_instrument_id=self._instrument_id_in_session(session, market),
                    run_type=run_type,
                    status="running",
                    current_stage="created",
                    horizon_weeks=HORIZON_WEEKS,
                    requested_through_week=None,
                    source_parent_iteration=None,
                    created_iteration_count=0,
                    started_at=now,
                    stages_json=[{"stage": "created", "at": now.isoformat()}],
                    result_json={},
                )
            )
        return run_id

    def _stage(self, run_id: str, stage: str) -> None:
        now = self.now_provider()
        with self.sessions() as session, session.begin():
            run = session.get(V33TrainingRun, run_id)
            if run is None:
                raise V33RuntimeError("training run disappeared")
            run.current_stage = stage
            run.stages_json = [*run.stages_json, {"stage": stage, "at": now.isoformat()}]

    def _set_run_parent(self, run_id: str, iteration_number: int) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V33TrainingRun, run_id)
            if run is None:
                raise V33RuntimeError("training run disappeared")
            run.source_parent_iteration = iteration_number

    def _complete_run(self, run_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        payload = _jsonable(result)
        with self.sessions() as session, session.begin():
            run = session.get(V33TrainingRun, run_id)
            if run is None:
                raise V33RuntimeError("training run disappeared")
            run.status = "completed"
            run.current_stage = "completed"
            run.completed_at = self.now_provider()
            run.created_iteration_count = int(payload.get("created_iteration_count", payload.get("audited_iteration_count", 0)))
            run.result_json = {**dict(run.result_json or {}), **payload}
            run.stages_json = [*run.stages_json, {"stage": "completed", "at": run.completed_at.isoformat()}]
        return {"run_id": run_id, **payload}

    def _fail_run(self, run_id: str, error: Exception) -> None:
        with self.sessions() as session, session.begin():
            run = session.get(V33TrainingRun, run_id)
            if run is not None:
                run.status = "failed"
                run.current_stage = "failed_recoverable"
                run.completed_at = self.now_provider()
                run.error_code = type(error).__name__
                run.error_message = str(error)
                run.result_json = {
                    **dict(run.result_json or {}),
                    "recoverable": True,
                    "resume": "start bootstrap again",
                }

    def _refresh_auxiliary_sources(self, run_id: str) -> dict[str, Any]:
        """Refresh PIT auxiliary sources before fitting and persist every gap.

        Auxiliary failure degrades features but must not fabricate values or
        prevent price-only training.  The returned source-by-source audit is
        stored on the training run before feature construction starts.
        """

        if self.auxiliary_refresh is None:
            payload: dict[str, Any] = {
                "status": "not_configured",
                "missing_preserved": True,
                "sources": [],
            }
        else:
            self._stage(run_id, "refresh_public_auxiliary_sources")
            try:
                payload = _jsonable(dict(self.auxiliary_refresh()))
            except Exception as error:
                payload = {
                    "status": "failed",
                    "error": str(error),
                    "missing_preserved": True,
                    "sources": [],
                }
        with self.sessions() as session, session.begin():
            run = session.get(V33TrainingRun, run_id)
            if run is None:
                raise V33RuntimeError("training run disappeared during auxiliary refresh")
            run.result_json = {
                **dict(run.result_json or {}),
                "auxiliary_refresh": payload,
            }
            run.stages_json = [
                *run.stages_json,
                {
                    "stage": "public_auxiliary_refresh_recorded",
                    "status": payload.get("status"),
                    "at": self.now_provider().isoformat(),
                },
            ]
        return payload

    @staticmethod
    def _progressive_payload(result: V33ProgressiveResult) -> dict[str, Any]:
        return {
            "market": result.market,
            "created_iteration_count": result.audited_iteration_count,
            "pending_count": result.pending_count,
            "full_count": result.full_count,
            "champion_version": result.state.version,
            "champion_state_hash": result.state.state_hash,
            "curve": list(result.curve),
            "weekly_sampling_seed": result.weekly_sampling_seed,
            "sampled_dates": list(result.sampled_dates),
        }

    def _instrument_id(self, market: str) -> int:
        with self.sessions() as session:
            return self._instrument_id_in_session(session, market)

    @staticmethod
    def _instrument_id_in_session(session: Session, market: str) -> int:
        value = session.scalar(select(Instrument.id).where(Instrument.code == market))
        if value is None:
            raise V33RuntimeError(f"missing instrument {market}")
        return int(value)

    @staticmethod
    def _date_or_none(value: Any) -> date | None:
        return None if value in (None, "") else date.fromisoformat(str(value))
