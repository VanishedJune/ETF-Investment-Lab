"""Durable two-worker task orchestration for weekly V2 analysis."""

from __future__ import annotations

from concurrent.futures import Future
from datetime import date
import logging
from threading import Event, Lock, Thread
import time
from typing import Mapping, Protocol
from uuid import uuid4

from sqlalchemy.exc import OperationalError

from ..daemon_executor import DaemonThreadPoolExecutor
from .engine import (
    AnalysisRequest,
    MODEL_LINE,
    ProgressUpdate,
    SUPPORTED_MARKETS,
)
from .repository import (
    MINIMUM_TASK_CLAIM_LEASE_SECONDS,
    TaskCheckpoint,
    TaskClaimLost,
    TaskExecutionGuard,
)


logger = logging.getLogger(__name__)
_HEARTBEAT_READY_TIMEOUT_SECONDS = 5.0
_MAX_REMOTE_POLL_SECONDS = 0.25


class TaskEngine(Protocol):
    def run(
        self,
        symbol: str,
        *,
        as_of: date | None = None,
        progress_callback=None,
        advice_request_key: str | None = None,
        analysis_request: AnalysisRequest | None = None,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> object: ...


class TaskRepository(Protocol):
    def get_or_create_task(
        self,
        symbol: str,
        idempotency_key: str,
        *,
        latest_complete_week: str,
        model_line: str,
        analysis_request: Mapping[str, object] | None = None,
    ) -> tuple[TaskCheckpoint, bool]: ...

    def transition_task_checkpoint(
        self,
        task_id: str,
        *,
        status: str,
        progress: Mapping[str, object] | None = None,
        total_weeks: int | None = None,
        worker_token: str | None = None,
        lease_seconds: float = 300.0,
    ) -> TaskCheckpoint: ...

    def claim_task_execution(
        self,
        task_id: str,
        *,
        worker_token: str,
        lease_seconds: float = 300.0,
    ) -> TaskCheckpoint | None: ...

    def renew_task_execution_claim(
        self,
        task_id: str,
        *,
        worker_token: str,
        lease_seconds: float = 300.0,
    ) -> bool: ...

    def load_task_checkpoint(self, task_id: str) -> TaskCheckpoint: ...

    def recover_interrupted_tasks(
        self, reason: str
    ) -> tuple[TaskCheckpoint, ...]: ...

    def release_task_execution_claim(
        self,
        task_id: str,
        *,
        worker_token: str,
        reason: str,
    ) -> TaskCheckpoint: ...


class WeeklyAnalysisTaskManager:
    """Run at most two markets concurrently and one task per market."""

    def __init__(
        self,
        *,
        engine: TaskEngine,
        repository: TaskRepository,
        claim_lease_seconds: float = 300.0,
        remote_poll_seconds: float = 0.05,
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim_lease_seconds must be positive")
        if remote_poll_seconds <= 0:
            raise ValueError("remote_poll_seconds must be positive")
        self.engine = engine
        self.repository = repository
        self._worker_token = uuid4().hex
        self._claim_lease_seconds = max(
            float(claim_lease_seconds),
            MINIMUM_TASK_CLAIM_LEASE_SECONDS,
        )
        self._remote_poll_seconds = float(remote_poll_seconds)
        self._executor = DaemonThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="weekly-analysis-v2",
        )
        self._market_locks = {symbol: Lock() for symbol in SUPPORTED_MARKETS}
        self._guard = Lock()
        self._futures: dict[str, Future[object]] = {}
        self._active_by_symbol: dict[str, str] = {}
        self._execution_controls: dict[
            str, tuple[Event, Event]
        ] = {}
        self._shutdown = False
        self.recovered_tasks = repository.recover_interrupted_tasks(
            "Weekly analysis worker process restarted"
        )

    def submit(
        self,
        symbol: str,
        *,
        latest_complete_week: str,
        model_line: str = MODEL_LINE,
        as_of: date | None = None,
        advice_request_key: str | None = None,
        analysis_request: AnalysisRequest | None = None,
    ) -> TaskCheckpoint:
        """Queue a test/internal run; production services do not expose as_of."""

        if symbol not in SUPPORTED_MARKETS:
            raise ValueError("Only 399006 and NDX are supported")
        if (
            not isinstance(latest_complete_week, str)
            or not latest_complete_week.strip()
        ):
            raise ValueError("latest_complete_week is required")
        if not isinstance(model_line, str) or not model_line.strip():
            raise ValueError("model_line is required")
        if as_of is not None and not isinstance(as_of, date):
            raise TypeError("as_of must be a date")
        latest = latest_complete_week.strip()
        line = model_line.strip()
        requested_analysis = (
            AnalysisRequest(
                latest_complete_week=latest,
                model_line=line,
                current_position=None,
                position_snapshot_key=(
                    advice_request_key or "unmanaged-position-unset"
                ),
            )
            if analysis_request is None
            else analysis_request
        )
        if not isinstance(requested_analysis, AnalysisRequest):
            raise TypeError("analysis_request must be an AnalysisRequest")
        if (
            requested_analysis.latest_complete_week != latest
            or requested_analysis.model_line != line
            or (
                advice_request_key is not None
                and requested_analysis.position_snapshot_key
                != advice_request_key
            )
        ):
            raise ValueError(
                "analysis_request must match task week/model/position key"
            )
        idempotency_key = f"{symbol}:{latest}:{line}"
        with self._guard:
            if self._shutdown:
                raise RuntimeError("task manager is shut down")
            active_id = self._active_by_symbol.get(symbol)
            if active_id is not None:
                active_checkpoint = self.repository.load_task_checkpoint(
                    active_id
                )
                if active_checkpoint.status not in {"completed", "failed"}:
                    return active_checkpoint
                self._active_by_symbol.pop(symbol, None)
            checkpoint, created = self.repository.get_or_create_task(
                symbol,
                idempotency_key,
                latest_complete_week=latest,
                model_line=line,
                analysis_request=requested_analysis.to_dict(),
            )
            execution_request = _analysis_request_from_checkpoint(
                checkpoint,
                fallback=requested_analysis,
            )
            execution_advice_key: str | None = (
                execution_request.position_snapshot_key
            )
            if checkpoint.status == "completed":
                execution_request = requested_analysis
                execution_advice_key = execution_request.position_snapshot_key
                if not execution_advice_key:
                    raise ValueError("advice_request_key cannot be blank")
                checkpoint, created = self.repository.get_or_create_task(
                    symbol,
                    f"{idempotency_key}:advice:{execution_advice_key}",
                    latest_complete_week=latest,
                    model_line=line,
                    analysis_request=execution_request.to_dict(),
                )
                execution_request = _analysis_request_from_checkpoint(
                    checkpoint,
                    fallback=execution_request,
                )
            if checkpoint.status in {"completed", "failed"}:
                return checkpoint
            existing_future = self._futures.get(checkpoint.task_id)
            if existing_future is not None and not existing_future.done():
                self._active_by_symbol[symbol] = checkpoint.task_id
                return checkpoint
            claimed = self.repository.claim_task_execution(
                checkpoint.task_id,
                worker_token=self._worker_token,
                lease_seconds=self._claim_lease_seconds,
            )
            if claimed is None:
                return self.repository.load_task_checkpoint(checkpoint.task_id)
            checkpoint = claimed
            if checkpoint.status != "preparing_data":
                return checkpoint
            self._active_by_symbol[symbol] = checkpoint.task_id
            self._futures[checkpoint.task_id] = self._executor.submit(
                self._execute,
                checkpoint.task_id,
                symbol,
                as_of,
                execution_advice_key,
                execution_request,
            )
            return checkpoint

    def wait(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> TaskCheckpoint:
        deadline = None if timeout is None else time.monotonic() + timeout
        poll_seconds = self._remote_poll_seconds
        while True:
            checkpoint = self.repository.load_task_checkpoint(task_id)
            if checkpoint.status in {"completed", "failed"}:
                return checkpoint
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for task {task_id}")
            time.sleep(
                poll_seconds
                if deadline is None
                else min(
                    poll_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            )
            poll_seconds = min(
                poll_seconds * 2.0,
                max(self._remote_poll_seconds, _MAX_REMOTE_POLL_SECONDS),
            )

    def get(self, task_id: str) -> TaskCheckpoint:
        return self.repository.load_task_checkpoint(task_id)

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_futures: bool = False,
    ) -> None:
        with self._guard:
            if self._shutdown:
                return
            self._shutdown = True
            active_task_ids = tuple(self._active_by_symbol.values())
            controls = tuple(self._execution_controls.values())
        for heartbeat_stop, claim_lost in controls:
            claim_lost.set()
            heartbeat_stop.set()
        for task_id in active_task_ids:
            try:
                self.repository.release_task_execution_claim(
                    task_id,
                    worker_token=self._worker_token,
                    reason=(
                        "Weekly analysis service shut down "
                        "before completion"
                    ),
                )
            except BaseException:
                logger.exception(
                    "Could not release weekly analysis task claim"
                )
        self._executor.shutdown(
            wait=wait,
            cancel_futures=cancel_futures,
        )

    def _execute(
        self,
        task_id: str,
        symbol: str,
        as_of: date | None,
        advice_request_key: str | None,
        analysis_request: AnalysisRequest,
    ) -> object:
        market_lock = self._market_locks[symbol]
        heartbeat_stop = Event()
        claim_lost = Event()
        heartbeat_ready = Event()
        claim_io_lock = Lock()
        with self._guard:
            if self._shutdown:
                claim_lost.set()
                heartbeat_stop.set()
            else:
                self._execution_controls[task_id] = (
                    heartbeat_stop,
                    claim_lost,
                )
        heartbeat = Thread(
            target=_renew_claim_until_stopped,
            args=(
                self.repository,
                task_id,
                self._worker_token,
                self._claim_lease_seconds,
                heartbeat_stop,
                claim_lost,
                heartbeat_ready,
                claim_io_lock,
            ),
            name=f"weekly-analysis-v2-heartbeat-{task_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            if not heartbeat_ready.wait(
                timeout=_HEARTBEAT_READY_TIMEOUT_SECONDS
            ):
                heartbeat_stop.set()
                claim_lost.set()
                raise _TaskClaimLost(
                    "weekly analysis claim heartbeat did not become ready"
                )
            with self._guard:
                shutting_down = self._shutdown
            if shutting_down:
                heartbeat_stop.set()
                claim_lost.set()
            _raise_if_claim_lost(claim_lost)
            with market_lock:
                def report(value: object) -> None:
                    _raise_if_claim_lost(claim_lost)
                    update = _progress_mapping(value)
                    stage = str(update.get("stage", "iterating"))
                    with claim_io_lock:
                        _raise_if_claim_lost(claim_lost)
                        current = self.repository.load_task_checkpoint(task_id)
                        status = _durable_status(current.status, stage)
                        total_raw = update.get("total_weeks")
                        total = (
                            None
                            if (
                                stage == "preparing_data"
                                and current.total_weeks is None
                            )
                            else (
                                total_raw
                                if type(total_raw) is int
                                and total_raw >= 0
                                else current.total_weeks
                            )
                        )
                        progress = dict(update)
                        _advance_task_stage(
                            self.repository,
                            task_id,
                            current,
                            status,
                            progress,
                            total,
                            worker_token=self._worker_token,
                            lease_seconds=self._claim_lease_seconds,
                        )

                result = self.engine.run(
                    symbol,
                    as_of=as_of,
                    progress_callback=report,
                    advice_request_key=advice_request_key,
                    analysis_request=analysis_request,
                    execution_guard=TaskExecutionGuard(
                        task_id=task_id,
                        worker_token=self._worker_token,
                    ),
                )
                _raise_if_claim_lost(claim_lost)
                with claim_io_lock:
                    try:
                        renewed = self.repository.renew_task_execution_claim(
                            task_id,
                            worker_token=self._worker_token,
                            lease_seconds=self._claim_lease_seconds,
                        )
                    except BaseException as exc:
                        claim_lost.set()
                        raise _TaskClaimLost(
                            "weekly analysis task execution claim could not "
                            "be renewed before completion"
                        ) from exc
                    if not renewed:
                        claim_lost.set()
                _raise_if_claim_lost(claim_lost)
                heartbeat_stop.set()
                heartbeat.join()
                _raise_if_claim_lost(claim_lost)
                with claim_io_lock:
                    current = self.repository.load_task_checkpoint(task_id)
                    current = _advance_task_stage(
                        self.repository,
                        task_id,
                        current,
                        "generating_advice",
                        {"stage": "generating_advice"},
                        current.total_weeks,
                        worker_token=self._worker_token,
                        lease_seconds=self._claim_lease_seconds,
                    )
                    _raise_if_claim_lost(claim_lost)
                    self.repository.transition_task_checkpoint(
                        task_id,
                        status="completed",
                        progress={"stage": "completed"},
                        worker_token=self._worker_token,
                        lease_seconds=self._claim_lease_seconds,
                    )
                return result
        except BaseException as exc:
            if not claim_lost.is_set():
                try:
                    with claim_io_lock:
                        current = self.repository.load_task_checkpoint(task_id)
                        if current.status not in {"completed", "failed"}:
                            self.repository.transition_task_checkpoint(
                                task_id,
                                status="failed",
                                progress={
                                    "stage": "failed",
                                    "failure_reason": str(exc),
                                    "last_checkpoint": (
                                        current.last_completed_week
                                    ),
                                },
                                worker_token=self._worker_token,
                                lease_seconds=self._claim_lease_seconds,
                            )
                except BaseException as checkpoint_error:
                    exc.add_note(
                        "Could not persist failed task checkpoint: "
                        f"{checkpoint_error}"
                    )
            return exc
        finally:
            heartbeat_stop.set()
            heartbeat.join()
            with self._guard:
                self._execution_controls.pop(task_id, None)
                if self._active_by_symbol.get(symbol) == task_id:
                    self._active_by_symbol.pop(symbol, None)


def _progress_mapping(value: object) -> dict[str, object]:
    if isinstance(value, ProgressUpdate):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise TypeError("progress callback values must be mappings")


_TaskClaimLost = TaskClaimLost


def _analysis_request_from_checkpoint(
    checkpoint: TaskCheckpoint,
    *,
    fallback: AnalysisRequest,
) -> AnalysisRequest:
    raw = checkpoint.progress.get("analysis_request")
    if raw is None:
        return fallback
    if not isinstance(raw, Mapping):
        raise ValueError("persisted analysis_request is corrupt")
    try:
        return AnalysisRequest(
            latest_complete_week=str(raw["latest_complete_week"]),
            model_line=str(raw["model_line"]),
            current_position=raw.get("current_position"),  # type: ignore[arg-type]
            position_snapshot_key=str(raw["position_snapshot_key"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("persisted analysis_request is corrupt") from exc


def _raise_if_claim_lost(claim_lost: Event) -> None:
    if claim_lost.is_set():
        raise _TaskClaimLost("weekly analysis task execution claim was lost")


def _claim_heartbeat_interval(lease_seconds: float) -> float:
    if lease_seconds <= 5.0:
        return min(lease_seconds / 3.0, 0.1)
    return min(
        lease_seconds / 3.0,
        max(0.01, min(lease_seconds / 4.0, 30.0)),
    )


def _renew_claim_until_stopped(
    repository: TaskRepository,
    task_id: str,
    worker_token: str,
    lease_seconds: float,
    heartbeat_stop: Event,
    claim_lost: Event,
    heartbeat_ready: Event,
    claim_io_lock=None,
) -> None:
    interval = _claim_heartbeat_interval(lease_seconds)
    lease_deadline = time.monotonic() + lease_seconds
    retry_delay = min(0.05, interval)
    try:
        while not heartbeat_stop.is_set():
            try:
                if claim_io_lock is None:
                    renewed = repository.renew_task_execution_claim(
                        task_id,
                        worker_token=worker_token,
                        lease_seconds=lease_seconds,
                    )
                else:
                    with claim_io_lock:
                        if heartbeat_stop.is_set():
                            return
                        renewed = repository.renew_task_execution_claim(
                            task_id,
                            worker_token=worker_token,
                            lease_seconds=lease_seconds,
                        )
            except OperationalError:
                remaining = lease_deadline - time.monotonic()
                if remaining > 0 and not heartbeat_stop.wait(
                    min(retry_delay, remaining)
                ):
                    continue
                claim_lost.set()
                return
            except BaseException:
                claim_lost.set()
                return
            if not renewed:
                claim_lost.set()
                return
            lease_deadline = time.monotonic() + lease_seconds
            heartbeat_ready.set()
            if heartbeat_stop.wait(interval):
                return
    finally:
        heartbeat_ready.set()


def _durable_status(current_status: str, stage: str) -> str:
    if stage == "preparing_data":
        return (
            "preparing_data"
            if current_status in {"queued", "preparing_data"}
            else current_status
        )
    if stage == "building_features":
        return (
            "building_features"
            if current_status in {"queued", "preparing_data", "building_features"}
            else current_status
        )
    if stage == "iterating":
        return (
            "iterating"
            if current_status
            in {"queued", "preparing_data", "building_features", "iterating"}
            else current_status
        )
    if stage == "validating":
        return (
            "validating"
            if current_status
            in {
                "queued",
                "preparing_data",
                "building_features",
                "iterating",
                "validating",
            }
            else current_status
        )
    if stage == "generating_advice":
        return "generating_advice"
    if stage == "completed":
        return current_status
    return current_status


def _advance_task_stage(
    repository: TaskRepository,
    task_id: str,
    current: TaskCheckpoint,
    target_status: str,
    progress: Mapping[str, object],
    total_weeks: int | None,
    *,
    worker_token: str,
    lease_seconds: float,
) -> TaskCheckpoint:
    order = (
        "preparing_data",
        "building_features",
        "iterating",
        "validating",
        "generating_advice",
    )
    if target_status not in order:
        target_status = current.status
    if current.status not in order:
        return current
    start = order.index(current.status)
    end = order.index(target_status)
    if end < start:
        return current
    value = current
    for status in order[start : end + 1]:
        value = repository.transition_task_checkpoint(
            task_id,
            status=status,
            progress=(
                progress
                if status == target_status
                else {"stage": status}
            ),
            total_weeks=total_weeks,
            worker_token=worker_token,
            lease_seconds=lease_seconds,
        )
    return value


__all__ = [
    "AnalysisRequest",
    "DaemonThreadPoolExecutor",
    "TaskEngine",
    "TaskRepository",
    "WeeklyAnalysisTaskManager",
]
