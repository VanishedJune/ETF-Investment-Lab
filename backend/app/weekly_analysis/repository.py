"""Transactional V2 weekly-analysis persistence and JSON audit snapshots.

SQLite is authoritative.  A complete snapshot is staged and fsynced before
the database transaction is flushed, atomically renamed before commit, and
removed on a failed commit.  Because SQLite and the filesystem cannot share a
physical transaction, every database row also carries the canonical snapshot
and its digest: a repository can deterministically restore a *missing* file
after an ambiguous commit outcome.  Existing corrupt files are never silently
replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Lock, RLock, local
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ..models.models import (
    Instrument,
    V2AdviceHistory,
    V2AnalysisIteration,
    V2AnalysisTask,
    V2IterationLabel,
    V2ModelVersion,
    V2WeekSample,
)
from .advice import Advice
from .features import FrozenDict
from .optimizer import (
    CrossMarketError,
    IterationLabel,
    OptimizerStepResult,
    StepAudit,
    WindowMetrics,
    WorkState,
    seed_model,
    validate_step_audit_acceptance,
)


_MARKETS = frozenset({"399006", "NDX"})
_VERSION_PATTERN = re.compile(r"^(?P<prefix>[IWM])(?P<number>\d{4,})$")
_DECIMAL_PATTERN = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
_FIXED_POINT_QUANTUM = Decimal("0.00000001")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_SCHEMA = "weekly-analysis-repository-v2"
_SNAPSHOT_FORMAT = "incremental-checkpoint-v1"
_WORK_STATE_STORAGE_FORMAT = "compact-work-state-v1"
_STEP_AUDIT_STORAGE_FORMAT = "step-audit-delta-v1"
_CODEC_VERSION = "weekly-analysis-tagged-json-v1"
_TYPE_TAG = "__weekly_analysis_type__"
_TYPE_VALUE = "value"
_TASK_STATUSES = frozenset(
    {
        "queued",
        "preparing_data",
        "building_features",
        "iterating",
        "validating",
        "generating_advice",
        "completed",
        "failed",
        "recoverable",
    }
)
_LEGACY_TASK_STATUSES = frozenset({"pending", "running", "cancelled"})
_TASK_TERMINAL_STATUSES = frozenset({"completed", "failed"})
# Windows can pause a Python worker for longer than one second while many
# SQLite-heavy test or recovery tasks contend for I/O.  A five-second floor
# prevents a live worker from being reclaimed solely because of scheduler
# jitter; the production default remains 300 seconds.
MINIMUM_TASK_CLAIM_LEASE_SECONDS = 5.0
_TASK_STATUS_TRANSITIONS = {
    "queued": frozenset(
        {
            "queued",
            "preparing_data",
            "failed",
            "recoverable",
        }
    ),
    "preparing_data": frozenset(
        {
            "preparing_data",
            "building_features",
            "failed",
            "recoverable",
        }
    ),
    "building_features": frozenset(
        {
            "building_features",
            "iterating",
            "failed",
            "recoverable",
        }
    ),
    "iterating": frozenset(
        {
            "iterating",
            "validating",
            "failed",
            "recoverable",
        }
    ),
    "validating": frozenset(
        {
            "validating",
            "generating_advice",
            "failed",
            "recoverable",
        }
    ),
    "generating_advice": frozenset(
        {
            "generating_advice",
            "completed",
            "failed",
            "recoverable",
        }
    ),
    "completed": frozenset({"completed"}),
    "failed": frozenset({"failed"}),
    "recoverable": frozenset(
        {"recoverable", "preparing_data", "failed"}
    ),
}
_AUDIT_LOCK_TIMEOUT_SECONDS = 30.0
_AUDIT_LOCK_POLL_SECONDS = 0.05
_AUDIT_LOCKS_GUARD = Lock()
_AUDIT_THREAD_LOCKS: dict[str, RLock] = {}
_AUDIT_LOCK_LOCAL = local()


class RepositoryError(RuntimeError):
    """Base class for V2 persistence failures."""


class TaskClaimLost(RepositoryError):
    """A managed worker no longer owns its durable execution claim."""


@dataclass(frozen=True, slots=True)
class TaskExecutionGuard:
    """Token identity required for managed task-owned database writes."""

    task_id: str
    worker_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("execution guard task_id is required")
        if (
            not isinstance(self.worker_token, str)
            or not self.worker_token.strip()
        ):
            raise ValueError("execution guard worker_token is required")


class UnknownMarket(RepositoryError):
    """The requested market is not one of the two V2 direct indexes."""


class DuplicateWeek(RepositoryError):
    """A completed week already exists for one market."""


class DuplicateAdvice(RepositoryError):
    """Advice with the same I/W/M provenance already exists."""


class CheckpointNotFound(RepositoryError):
    """No task checkpoint exists for the requested task id."""


class CrossMarketRepositoryError(RepositoryError):
    """A value from one market was offered to another market repository."""


class RepositoryCorruption(RepositoryError):
    """Persisted V2 state is corrupt, incomplete, or discontinuous."""


@dataclass(frozen=True, slots=True)
class TaskCheckpoint:
    """Immutable task progress recovered from ``v2_analysis_tasks``."""

    task_id: str
    symbol: str
    status: str
    completed_weeks: int
    total_weeks: int | None
    last_iteration: int
    last_work_version: str
    last_model_version: str
    last_completed_week: str | None
    progress: FrozenDict[str, object]

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        if not self.task_id:
            raise ValueError("task_id is required")
        if self.status not in _TASK_STATUSES:
            raise ValueError("checkpoint status is invalid")
        if (
            type(self.completed_weeks) is not int
            or self.completed_weeks < 0
            or type(self.last_iteration) is not int
            or self.last_iteration < 0
        ):
            raise ValueError("checkpoint counters must be non-negative integers")
        if self.total_weeks is not None and (
            type(self.total_weeks) is not int
            or self.total_weeks < self.completed_weeks
        ):
            raise ValueError("total_weeks cannot be below completed_weeks")
        work_number = _version_number(
            self.last_work_version, "W", allow_zero=True
        )
        _version_number(
            self.last_model_version, "M", allow_zero=False
        )
        if work_number != self.last_iteration:
            raise ValueError(
                "last_work_version must match last_iteration"
            )
        if self.completed_weeks != self.last_iteration:
            raise ValueError(
                "completed_weeks must match last_iteration"
            )
        if (self.last_iteration == 0) != (
            self.last_completed_week is None
        ):
            raise ValueError(
                "last_completed_week must match checkpoint progress"
            )
        object.__setattr__(self, "progress", FrozenDict(self.progress))

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "symbol": self.symbol,
            "status": self.status,
            "completed_weeks": self.completed_weeks,
            "total_weeks": self.total_weeks,
            "last_iteration": self.last_iteration,
            "last_work_version": self.last_work_version,
            "last_model_version": self.last_model_version,
            "last_completed_week": self.last_completed_week,
            "progress": dict(self.progress),
        }


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Stable immutable repository view of the current model."""

    symbol: str
    version: str
    parent_version: str | None
    work_version: str
    iteration_id: str
    trained_through_week: str | None
    weights: FrozenDict[str, Decimal]
    metrics: FrozenDict[str, object]

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _version_number(self.version, "M", allow_zero=False)
        _version_number(self.work_version, "W", allow_zero=True)
        _version_number(self.iteration_id, "I", allow_zero=True)
        if self.parent_version is not None:
            _version_number(self.parent_version, "M", allow_zero=False)
        object.__setattr__(
            self,
            "weights",
            FrozenDict(
                {
                    str(name): _decimal(value, f"weight:{name}")
                    for name, value in self.weights.items()
                }
            ),
        )
        object.__setattr__(self, "metrics", FrozenDict(self.metrics))

    @property
    def model_number(self) -> int:
        return _version_number(self.version, "M", allow_zero=False)


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Minimal durable metrics contract for Task 8/9 consumers."""

    symbol: str
    model_version: str
    iteration_count: int
    last_iteration: int
    accepted_iterations: int
    rejected_iterations: int
    feedback_count: int
    mature_count: int
    partial_count: int
    pending_count: int
    candidate_acceptance_rate: Decimal
    windows: FrozenDict[str, object]
    curve: tuple[FrozenDict[str, object], ...] = ()

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _version_number(self.model_version, "M", allow_zero=False)
        for field_name in (
            "iteration_count",
            "last_iteration",
            "accepted_iterations",
            "rejected_iterations",
            "feedback_count",
            "mature_count",
            "partial_count",
            "pending_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        object.__setattr__(
            self,
            "candidate_acceptance_rate",
            _decimal(
                self.candidate_acceptance_rate,
                "candidate_acceptance_rate",
            ),
        )
        object.__setattr__(self, "windows", FrozenDict(self.windows))
        frozen_curve = tuple(FrozenDict(point) for point in self.curve)
        if any(
            point.get("iteration_number") != expected
            for expected, point in enumerate(frozen_curve, start=1)
        ):
            raise ValueError("metrics curve must use contiguous iterations")
        object.__setattr__(self, "curve", frozen_curve)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "model_version": self.model_version,
            "iteration_count": self.iteration_count,
            "last_iteration": self.last_iteration,
            "accepted_iterations": self.accepted_iterations,
            "rejected_iterations": self.rejected_iterations,
            "feedback_count": self.feedback_count,
            "mature_count": self.mature_count,
            "partial_count": self.partial_count,
            "pending_count": self.pending_count,
            "candidate_acceptance_rate": self.candidate_acceptance_rate,
            "windows": dict(self.windows),
            "curve": tuple(dict(point) for point in self.curve),
        }


@dataclass(frozen=True, slots=True)
class AdviceRecord:
    """Immutable persisted advice plus its verified JSON snapshot."""

    id: int
    symbol: str
    iteration_id: str
    work_version: str
    model_version: str
    advice_generation: int
    generation_key: str
    advice_at: datetime
    payload: FrozenDict[str, object]
    snapshot_path: Path
    snapshot_hash: str

    def __post_init__(self) -> None:
        _validate_symbol(self.symbol)
        _version_number(self.iteration_id, "I", allow_zero=False)
        _version_number(self.work_version, "W", allow_zero=False)
        _version_number(self.model_version, "M", allow_zero=False)
        if self.advice_generation < 1 or not self.generation_key:
            raise ValueError("advice generation identity is invalid")
        if not _HASH_PATTERN.fullmatch(self.snapshot_hash):
            raise ValueError("snapshot_hash must be SHA-256")
        object.__setattr__(self, "payload", FrozenDict(self.payload))


def _thread_lock_for(root: Path) -> RLock:
    key = str(root)
    with _AUDIT_LOCKS_GUARD:
        return _AUDIT_THREAD_LOCKS.setdefault(key, RLock())


class _AuditStoreLock:
    """Process/thread lock plus a cross-process lease for one audit root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / ".weekly-analysis-v2.audit.lock"
        self.thread_lock = _thread_lock_for(root)
        self.token = uuid4().hex
        self._owns_file = False

    def __enter__(self) -> _AuditStoreLock:
        self.thread_lock.acquire()
        depths = getattr(_AUDIT_LOCK_LOCAL, "depths", None)
        if depths is None:
            depths = {}
            _AUDIT_LOCK_LOCAL.depths = depths
        key = str(self.root)
        current_depth = int(depths.get(key, 0))
        depths[key] = current_depth + 1
        if current_depth:
            return self
        try:
            self._acquire_file_lease()
            return self
        except Exception:
            depths.pop(key, None)
            self.thread_lock.release()
            raise

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        depths = getattr(_AUDIT_LOCK_LOCAL, "depths", {})
        key = str(self.root)
        depth = int(depths.get(key, 1)) - 1
        if depth:
            depths[key] = depth
        else:
            depths.pop(key, None)
            self._release_file_lease()
        self.thread_lock.release()

    def _acquire_file_lease(self) -> None:
        deadline = time.monotonic() + _AUDIT_LOCK_TIMEOUT_SECONDS
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "token": self.token,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if self._reclaim_dead_owner():
                    continue
                if time.monotonic() >= deadline:
                    raise RepositoryError(
                        f"Timed out waiting for audit lock {self.path}"
                    )
                time.sleep(_AUDIT_LOCK_POLL_SECONDS)
                continue
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._owns_file = True
            return

    def _reclaim_dead_owner(self) -> bool:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            pid = raw.get("pid") if isinstance(raw, Mapping) else None
        except (OSError, json.JSONDecodeError):
            pid = None
        if type(pid) is int and _pid_is_alive(pid):
            return False
        quarantine = self.path.with_name(
            f"{self.path.name}.{uuid4().hex}.stale.orphan"
        )
        try:
            os.replace(self.path, quarantine)
        except FileNotFoundError:
            pass
        except OSError:
            return False
        return True

    def _release_file_lease(self) -> None:
        if not self._owns_file:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, Mapping) and raw.get("token") != self.token:
            self._owns_file = False
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            quarantine = self.path.with_name(
                f"{self.path.name}.{self.token}.release.orphan"
            )
            try:
                os.replace(self.path, quarantine)
            except OSError:
                pass
        self._owns_file = False


class WeeklyAnalysisRepository:
    """Repository for V2 weekly optimizer state, checkpoints, and advice."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        audit_root: Path | str = Path("data") / "weekly_analysis_v2",
    ) -> None:
        if not callable(sessions):
            raise TypeError("sessions must be a SQLAlchemy session factory")
        self._sessions = sessions
        self.audit_root = Path(audit_root).resolve()
        self._validated_heads: dict[
            str, tuple[WorkState, str | None]
        ] = {}
        self._audit_storage_cache: dict[
            tuple[str, str, str, str, str],
            dict[str, object],
        ] = {}
        self.audit_root.mkdir(parents=True, exist_ok=True)
        with self._audit_store_lock():
            for symbol in sorted(_MARKETS):
                (self.audit_root / symbol).mkdir(parents=True, exist_ok=True)
            self._discard_abandoned_temps()
            self._repair_missing_snapshots()

    def _audit_store_lock(self) -> _AuditStoreLock:
        return _AuditStoreLock(self.audit_root)

    def list_completed_weeks(self, symbol: str) -> set[str]:
        """Return completed week keys for exactly one market."""

        return set(self.load_work_state(symbol).week_records)

    def get_or_create_task(
        self,
        symbol: str,
        idempotency_key: str,
        *,
        latest_complete_week: str,
        model_line: str,
        analysis_request: Mapping[str, object] | None = None,
    ) -> tuple[TaskCheckpoint, bool]:
        """Atomically return the one task bound to a canonical run identity."""

        symbol = _validate_symbol(symbol)
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        if (
            not isinstance(latest_complete_week, str)
            or not latest_complete_week.strip()
        ):
            raise ValueError("latest_complete_week is required")
        if not isinstance(model_line, str) or not model_line.strip():
            raise ValueError("model_line is required")
        if analysis_request is not None and not isinstance(
            analysis_request, Mapping
        ):
            raise TypeError("analysis_request must be a mapping")
        captured_request = (
            None
            if analysis_request is None
            else _json_mapping(dict(analysis_request))
        )
        canonical_key = idempotency_key.strip()
        task_id = (
            "weekly-v2-"
            + hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:32]
        )
        with self._audit_store_lock():
            try:
                existing = self.load_task_checkpoint(task_id)
            except CheckpointNotFound:
                existing = None
            if existing is not None:
                if (
                    existing.symbol != symbol
                    or existing.progress.get("idempotency_key")
                    != canonical_key
                    or existing.progress.get("latest_complete_week")
                    != latest_complete_week.strip()
                    or existing.progress.get("model_line")
                    != model_line.strip()
                ):
                    raise RepositoryCorruption(
                        "task idempotency identity conflicts with persisted task"
                    )
                return existing, False

            state = self._load_work_state_locked(symbol)
            completed = len(state.audits)
            checkpoint = TaskCheckpoint(
                task_id=task_id,
                symbol=symbol,
                status="queued",
                completed_weeks=completed,
                total_weeks=None,
                last_iteration=completed,
                last_work_version=state.work_version,
                last_model_version=state.current_model_version,
                last_completed_week=(
                    None if not state.audits else state.audits[-1].week_key
                ),
                progress=FrozenDict(
                    {
                        "idempotency_key": canonical_key,
                        "latest_complete_week": latest_complete_week.strip(),
                        "model_line": model_line.strip(),
                        **(
                            {}
                            if captured_request is None
                            else {"analysis_request": captured_request}
                        ),
                        "stage": "queued",
                    }
                ),
            )
            self._save_task_checkpoint_locked(
                task_id,
                checkpoint,
                task_request={
                    "symbol": symbol,
                    **(
                        {}
                        if captured_request is None
                        else {"analysis_request": captured_request}
                    ),
                },
            )
            return checkpoint, True

    def transition_task_checkpoint(
        self,
        task_id: str,
        *,
        status: str,
        progress: Mapping[str, object] | None = None,
        total_weeks: int | None = None,
        worker_token: str | None = None,
        lease_seconds: float = 300.0,
    ) -> TaskCheckpoint:
        """Advance one task with Task 7 chain validation and CAS persistence."""

        if status not in _TASK_STATUSES:
            raise ValueError("checkpoint status is invalid")
        if progress is not None and not isinstance(progress, Mapping):
            raise TypeError("progress must be a mapping")
        if worker_token is not None and (
            not isinstance(worker_token, str) or not worker_token.strip()
        ):
            raise ValueError("worker_token must be a non-empty string")
        if not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        effective_lease_seconds = max(
            float(lease_seconds), MINIMUM_TASK_CLAIM_LEASE_SECONDS
        )
        with self._audit_store_lock():
            current = self.load_task_checkpoint(task_id)
            if status not in _TASK_STATUS_TRANSITIONS[current.status]:
                raise ValueError(
                    f"task status transition {current.status!r} -> "
                    f"{status!r} is not adjacent"
                )
            # Progress reporting must not replace the exact validated
            # WorkState object that the running engine will use for its next
            # incremental commit. The trusted head is already updated after
            # every successful commit; only hydrate from storage when this
            # repository instance has not validated a head yet.
            trusted_head = self._validated_heads.get(current.symbol)
            state = (
                trusted_head[0]
                if trusted_head is not None
                else self._load_work_state_locked(current.symbol)
            )
            completed = len(state.audits)
            resolved_total = (
                total_weeks
                if total_weeks is not None
                else current.total_weeks
            )
            if resolved_total is not None:
                resolved_total = max(resolved_total, completed)
            merged_progress = dict(current.progress)
            if progress is not None:
                merged_progress.update(progress)
            merged_progress["stage"] = str(
                merged_progress.get("stage", status)
            )
            checkpoint = TaskCheckpoint(
                task_id=current.task_id,
                symbol=current.symbol,
                status=status,
                completed_weeks=completed,
                total_weeks=resolved_total,
                last_iteration=completed,
                last_work_version=state.work_version,
                last_model_version=state.current_model_version,
                last_completed_week=(
                    None if not state.audits else state.audits[-1].week_key
                ),
                progress=FrozenDict(merged_progress),
            )
            self._save_task_checkpoint_locked(
                task_id,
                checkpoint,
                expected_worker_token=worker_token,
                lease_expires_at=(
                    datetime.now(timezone.utc)
                    + timedelta(seconds=effective_lease_seconds)
                    if worker_token is not None
                    and status not in _TASK_TERMINAL_STATUSES
                    else None
                ),
                clear_claim=status in _TASK_TERMINAL_STATUSES,
            )
            return checkpoint

    def claim_task_execution(
        self,
        task_id: str,
        *,
        worker_token: str,
        lease_seconds: float = 300.0,
    ) -> TaskCheckpoint | None:
        """Atomically claim a queued/recoverable task for one worker."""

        if not isinstance(worker_token, str) or not worker_token.strip():
            raise ValueError("worker_token is required")
        if not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        effective_lease_seconds = max(
            float(lease_seconds), MINIMUM_TASK_CLAIM_LEASE_SECONDS
        )
        token = worker_token.strip()
        with self._audit_store_lock():
            current = self.load_task_checkpoint(task_id)
            if current.status not in {"queued", "recoverable"}:
                return None
            state = self._load_work_state_locked(current.symbol)
            completed = len(state.audits)
            progress = dict(current.progress)
            progress.update(
                {
                    "stage": "preparing_data",
                    "worker_token": token,
                    "resume_from": (
                        None
                        if not state.audits
                        else state.audits[-1].week_key
                    ),
                }
            )
            claimed = TaskCheckpoint(
                task_id=current.task_id,
                symbol=current.symbol,
                status="preparing_data",
                completed_weeks=completed,
                total_weeks=(
                    None
                    if current.total_weeks is None
                    else max(current.total_weeks, completed)
                ),
                last_iteration=completed,
                last_work_version=state.work_version,
                last_model_version=state.current_model_version,
                last_completed_week=(
                    None if not state.audits else state.audits[-1].week_key
                ),
                progress=FrozenDict(progress),
            )
            session = self._sessions()
            try:
                row = session.scalar(
                    select(V2AnalysisTask).where(
                        V2AnalysisTask.task_key == task_id
                    )
                )
                if row is None:
                    raise CheckpointNotFound(
                        f"No checkpoint exists for task {task_id!r}"
                    )
                if row.status not in {"queued", "recoverable"}:
                    return None
                old_status = row.status
                old_payload = row.result_payload
                old_updated_at = row.updated_at
                result = session.execute(
                    update(V2AnalysisTask)
                    .where(
                        V2AnalysisTask.id == row.id,
                        V2AnalysisTask.status == old_status,
                        V2AnalysisTask.result_payload == old_payload,
                        V2AnalysisTask.updated_at == old_updated_at,
                        V2AnalysisTask.worker_token.is_(None),
                    )
                    .values(
                        status="preparing_data",
                        result_payload=_json_mapping(claimed.to_dict()),
                        worker_token=token,
                        lease_expires_at=(
                            datetime.now(timezone.utc)
                            + timedelta(seconds=effective_lease_seconds)
                        ),
                        started_at=row.started_at
                        or datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    session.rollback()
                    return None
                session.commit()
                return claimed
            finally:
                session.close()

    def renew_task_execution_claim(
        self,
        task_id: str,
        *,
        worker_token: str,
        lease_seconds: float = 300.0,
    ) -> bool:
        """Extend one unexpired executing claim through a token-fenced CAS."""

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id is required")
        if not isinstance(worker_token, str) or not worker_token.strip():
            raise ValueError("worker_token is required")
        if not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        effective_lease_seconds = max(
            float(lease_seconds), MINIMUM_TASK_CLAIM_LEASE_SECONDS
        )
        now = datetime.now(timezone.utc)
        with self._sessions() as session:
            result = session.execute(
                update(V2AnalysisTask)
                .where(
                    V2AnalysisTask.task_key == task_id.strip(),
                    V2AnalysisTask.worker_token == worker_token.strip(),
                    V2AnalysisTask.status.in_(
                        (
                            "preparing_data",
                            "building_features",
                            "iterating",
                            "validating",
                            "generating_advice",
                        )
                    ),
                    V2AnalysisTask.lease_expires_at.is_not(None),
                    V2AnalysisTask.lease_expires_at > now,
                )
                .values(
                    lease_expires_at=now
                    + timedelta(seconds=effective_lease_seconds)
                )
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return result.rowcount == 1

    def release_task_execution_claim(
        self,
        task_id: str,
        *,
        worker_token: str,
        reason: str,
    ) -> TaskCheckpoint:
        """Fence one owned worker and persist a resumable checkpoint."""

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id is required")
        if not isinstance(worker_token, str) or not worker_token.strip():
            raise ValueError("worker_token is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("recoverable reason is required")
        task_key = task_id.strip()
        token = worker_token.strip()
        recovery_reason = reason.strip()
        with self._audit_store_lock():
            with self._sessions() as session:
                row = session.scalar(
                    select(V2AnalysisTask).where(
                        V2AnalysisTask.task_key == task_key
                    )
                )
                if row is None:
                    raise CheckpointNotFound(
                        f"No checkpoint exists for task {task_key!r}"
                    )
                instrument = session.get(Instrument, row.instrument_id)
                if instrument is None:
                    raise RepositoryCorruption(
                        f"Task {task_key!r} references a missing instrument"
                    )
                if (
                    row.status in _TASK_TERMINAL_STATUSES
                    or row.status == "recoverable"
                    or row.worker_token != token
                ):
                    return self.load_task_checkpoint(task_key)
                raw_payload = _decode_json_mapping(row.result_payload)
                payload_progress = raw_payload.get("progress", {})
                progress = (
                    dict(_mapping(payload_progress, "progress"))
                    if isinstance(payload_progress, Mapping)
                    else {}
                )
                progress.update(
                    {
                        "stage": "recoverable",
                        "recoverable_reason": recovery_reason,
                    }
                )
                state = self._load_work_state_locked(instrument.code)
                completed = len(state.audits)
                total_raw = raw_payload.get("total_weeks")
                total = (
                    None
                    if total_raw is None
                    else max(
                        _strict_int(total_raw, "total_weeks"),
                        completed,
                    )
                )
                checkpoint = TaskCheckpoint(
                    task_id=task_key,
                    symbol=instrument.code,
                    status="recoverable",
                    completed_weeks=completed,
                    total_weeks=total,
                    last_iteration=completed,
                    last_work_version=state.work_version,
                    last_model_version=state.current_model_version,
                    last_completed_week=(
                        None
                        if not state.audits
                        else state.audits[-1].week_key
                    ),
                    progress=FrozenDict(progress),
                )
                encoded = _json_mapping(checkpoint.to_dict())
                result = session.execute(
                    update(V2AnalysisTask)
                    .where(
                        V2AnalysisTask.id == row.id,
                        V2AnalysisTask.status == row.status,
                        V2AnalysisTask.updated_at == row.updated_at,
                        V2AnalysisTask.result_payload
                        == row.result_payload,
                        V2AnalysisTask.worker_token == token,
                    )
                    .values(
                        status="recoverable",
                        result_payload=encoded,
                        error_message=recovery_reason,
                        worker_token=None,
                        lease_expires_at=None,
                        updated_at=datetime.now(timezone.utc),
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    session.rollback()
                    actual = self.load_task_checkpoint(task_key)
                    if actual.status in (
                        "recoverable",
                        "completed",
                        "failed",
                    ):
                        return actual
                    raise TaskClaimLost(
                        "task shutdown fencing lost its worker claim"
                    )
                session.commit()
                return checkpoint

    def list_interrupted_task_checkpoints(self) -> tuple[TaskCheckpoint, ...]:
        """List formal non-terminal tasks; legacy rows require recovery first."""

        with self._sessions() as session:
            rows = tuple(
                session.scalars(
                    select(V2AnalysisTask)
                    .where(
                        V2AnalysisTask.task_type == "weekly-analysis-v2",
                        V2AnalysisTask.status.not_in(
                            tuple(_TASK_TERMINAL_STATUSES | {"recoverable"})
                        ),
                    )
                    .order_by(V2AnalysisTask.queued_at, V2AnalysisTask.id)
                )
            )
            checkpoints: list[TaskCheckpoint] = []
            for row in rows:
                instrument = session.get(Instrument, row.instrument_id)
                if instrument is None:
                    raise RepositoryCorruption(
                        f"Task {row.task_key!r} references a missing instrument"
                    )
                if row.status in _LEGACY_TASK_STATUSES:
                    raise RepositoryCorruption(
                        "legacy task status must be migrated through "
                        "recover_interrupted_tasks"
                    )
                checkpoints.append(
                    _checkpoint_from_mapping(
                        row.task_key,
                        instrument.code,
                        row.status,
                        _decode_json_mapping(row.result_payload),
                    )
                )
            return tuple(checkpoints)

    def recover_interrupted_tasks(
        self,
        reason: str,
    ) -> tuple[TaskCheckpoint, ...]:
        """CAS-migrate every interrupted formal/legacy task to recoverable."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("recoverable reason is required")
        recovery_reason = reason.strip()
        recovered: list[TaskCheckpoint] = []
        with self._audit_store_lock():
            with self._sessions() as session:
                task_ids = tuple(
                    session.scalars(
                        select(V2AnalysisTask.task_key)
                        .where(
                            V2AnalysisTask.task_type
                            == "weekly-analysis-v2",
                            V2AnalysisTask.status.not_in(
                                (
                                    "completed",
                                    "failed",
                                    "recoverable",
                                    "cancelled",
                                )
                            ),
                        )
                        .order_by(
                            V2AnalysisTask.queued_at,
                            V2AnalysisTask.id,
                        )
                    )
                )
            for task_id in task_ids:
                checkpoint = self._recover_task_checkpoint(
                    task_id, recovery_reason
                )
                recovered.append(checkpoint)
        return tuple(recovered)

    def _recover_task_checkpoint(
        self,
        task_id: str,
        reason: str,
    ) -> TaskCheckpoint:
        session = self._sessions()
        try:
            row = session.scalar(
                select(V2AnalysisTask).where(
                    V2AnalysisTask.task_key == task_id
                )
            )
            if row is None:
                raise CheckpointNotFound(
                    f"No checkpoint exists for task {task_id!r}"
                )
            now = datetime.now(timezone.utc)
            if (
                row.worker_token is not None
                and row.lease_expires_at is not None
                and row.lease_expires_at > now
            ):
                return self.load_task_checkpoint(task_id)
            instrument = session.get(Instrument, row.instrument_id)
            if instrument is None:
                raise RepositoryCorruption(
                    f"Task {task_id!r} references a missing instrument"
                )
            old_status = row.status
            if old_status in _TASK_TERMINAL_STATUSES | {"recoverable"}:
                return self.load_task_checkpoint(task_id)
            raw_payload = _decode_json_mapping(row.result_payload)
            payload_progress = raw_payload.get("progress", {})
            existing_progress = (
                dict(_mapping(payload_progress, "progress"))
                if isinstance(payload_progress, Mapping)
                else {}
            )
            existing_progress.update(
                {
                    "stage": "recoverable",
                    "recoverable_reason": reason,
                    **(
                        {"legacy_status": old_status}
                        if old_status in _LEGACY_TASK_STATUSES
                        else {}
                    ),
                }
            )
            state = self._load_work_state_locked(instrument.code)
            completed = len(state.audits)
            total_raw = raw_payload.get("total_weeks")
            total = (
                None
                if total_raw is None
                else max(_strict_int(total_raw, "total_weeks"), completed)
            )
            checkpoint = TaskCheckpoint(
                task_id=task_id,
                symbol=instrument.code,
                status="recoverable",
                completed_weeks=completed,
                total_weeks=total,
                last_iteration=completed,
                last_work_version=state.work_version,
                last_model_version=state.current_model_version,
                last_completed_week=(
                    None if not state.audits else state.audits[-1].week_key
                ),
                progress=FrozenDict(existing_progress),
            )
            encoded = _json_mapping(checkpoint.to_dict())
            old_updated_at = row.updated_at
            old_payload = row.result_payload
            old_lease_expires_at = row.lease_expires_at
            result = session.execute(
                update(V2AnalysisTask)
                .where(
                    V2AnalysisTask.id == row.id,
                    V2AnalysisTask.updated_at == old_updated_at,
                    V2AnalysisTask.status == old_status,
                    V2AnalysisTask.result_payload == old_payload,
                    V2AnalysisTask.worker_token == row.worker_token,
                    V2AnalysisTask.lease_expires_at
                    == old_lease_expires_at,
                )
                .values(
                    status="recoverable",
                    result_payload=encoded,
                    error_message=reason,
                    worker_token=None,
                    lease_expires_at=None,
                    updated_at=datetime.now(timezone.utc),
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                session.rollback()
                actual = self.load_task_checkpoint(task_id)
                if actual.status == "recoverable":
                    return actual
                with self._sessions() as verification_session:
                    actual_row = verification_session.scalar(
                        select(V2AnalysisTask).where(
                            V2AnalysisTask.task_key == task_id
                        )
                    )
                    if (
                        actual_row is not None
                        and actual_row.worker_token is not None
                        and actual_row.lease_expires_at is not None
                        and actual_row.lease_expires_at
                        > datetime.now(timezone.utc)
                    ):
                        return actual
                raise ValueError(
                    "task recovery lost a concurrent checkpoint CAS"
                )
            session.commit()
            return checkpoint
        finally:
            session.close()

    def load_work_state(self, symbol: str) -> WorkState:
        """Rebuild and validate the latest immutable optimizer work state."""

        with self._audit_store_lock():
            symbol = _validate_symbol(symbol)
            # Public reads must validate the durable mirrors exactly once.
            # Other local sessions (imports, migrations, audit tools) can
            # legitimately alter SQLite without going through this repository,
            # so an in-memory head alone is not an integrity boundary.
            return self._load_work_state_locked(symbol)

    def _load_work_state_locked(self, symbol: str) -> WorkState:
        symbol = _validate_symbol(symbol)
        with self._sessions() as session:
            instrument_id = self._instrument_id(session, symbol)
            rows = tuple(
                session.scalars(
                    select(V2AnalysisIteration)
                    .where(V2AnalysisIteration.instrument_id == instrument_id)
                    .order_by(V2AnalysisIteration.iteration_number)
                )
            )
            if not rows:
                state = seed_model(symbol)
                trusted = self._validated_heads.get(symbol)
                if trusted is not None and trusted[1] is None and trusted[0] == state:
                    return trusted[0]
                self._validated_heads[symbol] = (state, None)
                return state

            sample_rows = tuple(
                session.scalars(
                    select(V2WeekSample).where(
                        V2WeekSample.instrument_id == instrument_id
                    )
                )
            )
            samples_by_week = {row.week_key: row for row in sample_rows}
            if len(samples_by_week) != len(sample_rows):
                raise RepositoryCorruption(
                    f"{symbol} week sample identities are duplicated"
                )
            try:
                latest_output = _decode_json_mapping(
                    rows[-1].output_payload
                )
                state_raw = _mapping(
                    latest_output.get("work_state"),
                    "latest work_state checkpoint",
                )
                raw_audits = tuple(
                    _mapping(item, "work_state audit")
                    for item in _sequence(state_raw.get("audits"))
                )
            except (TypeError, ValueError) as exc:
                raise RepositoryCorruption(
                    f"{symbol} latest work state checkpoint is corrupt"
                ) from exc
            if len(raw_audits) != len(rows):
                raise RepositoryCorruption(
                    f"{symbol} latest work state history is incomplete"
                )

            previous_state_hash: str | None = None
            analysis_times_by_model: dict[int, datetime] = {}
            for expected, (row, state_audit_raw) in enumerate(
                zip(rows, raw_audits), start=1
            ):
                if (
                    row.status != "complete"
                    or row.iteration_number != expected
                    or row.work_number != expected
                ):
                    raise RepositoryCorruption(
                        f"{symbol} iteration/work chain is not contiguous"
                    )
                _path, _digest, snapshot = self._verified_snapshot(
                    row.audit_data,
                    expected_symbol=symbol,
                    expected_kind="iteration",
                )
                try:
                    candidate_raw = _mapping(
                        snapshot.get("candidate_audit"),
                        "candidate_audit",
                    )
                    analysis_time = _parse_datetime(
                        snapshot.get("analysis_time"),
                        "analysis_time",
                    )
                    cutoff = _parse_date(
                        candidate_raw.get("cutoff_date"), "cutoff_date"
                    )
                    source_max = _parse_optional_date(
                        candidate_raw.get("source_data_max_date"),
                        "source_data_max_date",
                    )
                    state_hash = snapshot.get("state_hash")
                    row_input = _decode_json_mapping(row.input_payload)
                    row_output = _decode_json_mapping(row.output_payload)
                    row_metrics = _decode_json_mapping(row.metrics)
                except (TypeError, ValueError) as exc:
                    raise RepositoryCorruption(
                        f"{symbol} incremental iteration snapshot is corrupt"
                    ) from exc
                expected_iteration = f"I{expected:04d}"
                expected_work = f"W{expected:04d}"
                expected_metrics = {
                    "accepted": candidate_raw.get("accepted"),
                    "current": snapshot.get("current_metrics"),
                    "candidate": snapshot.get("candidate_metrics"),
                }
                expected_output: dict[str, object] = {
                    "state_hash": state_hash
                }
                if expected == len(rows):
                    expected_output["work_state"] = state_raw
                if (
                    snapshot.get("schema_version") != _SNAPSHOT_SCHEMA
                    or snapshot.get("snapshot_format")
                    != _SNAPSHOT_FORMAT
                    or snapshot.get("codec_version") != _CODEC_VERSION
                    or not isinstance(state_hash, str)
                    or _HASH_PATTERN.fullmatch(state_hash) is None
                    or snapshot.get("parent_state_hash")
                    != previous_state_hash
                    or not _stored_audits_equivalent(
                        candidate_raw, state_audit_raw
                    )
                    or not _canonical_equal(row_input, candidate_raw)
                    or not _canonical_equal(row_output, expected_output)
                    or not _canonical_equal(row_metrics, expected_metrics)
                    or candidate_raw.get("instrument_code") != symbol
                    or candidate_raw.get("iteration_id")
                    != expected_iteration
                    or candidate_raw.get("work_version") != expected_work
                    or candidate_raw.get("week_key") != row.week_key
                    or snapshot.get("symbol") != symbol
                    or snapshot.get("iteration_id")
                    != expected_iteration
                    or snapshot.get("work_version") != expected_work
                    or snapshot.get("model_version")
                    != candidate_raw.get("model_version")
                    or snapshot.get("data_cutoff") != cutoff
                    or snapshot.get("source_data_max_date")
                    != source_max
                    or snapshot.get("source_hash")
                    != candidate_raw.get("source_hash")
                    or row.started_at != analysis_time
                    or row.completed_at != analysis_time
                ):
                    raise RepositoryCorruption(
                        f"{symbol} incremental I/W/work state market/symbol "
                        "chain is invalid"
                    )
                sample = samples_by_week.get(row.week_key)
                monday = cutoff - timedelta(days=cutoff.weekday())
                expected_sample_input = {
                    "input_hash": candidate_raw.get("input_hash"),
                    "source_hash": candidate_raw.get("source_hash"),
                    "source_data_max_date": source_max,
                }
                expected_sample_audit = {
                    "cutoff_date": cutoff,
                    "data_integrity": candidate_raw.get(
                        "data_integrity"
                    ),
                }
                if (
                    sample is None
                    or sample.status != "complete"
                    or sample.week_start != monday
                    or sample.week_end != cutoff
                    or not _canonical_equal(
                        _decode_json_mapping(sample.input_payload),
                        expected_sample_input,
                    )
                    or not _canonical_equal(
                        _decode_json_mapping(sample.target_payload), {}
                    )
                    or not _canonical_equal(
                        _decode_json_mapping(sample.audit_data),
                        expected_sample_audit,
                    )
                ):
                    raise RepositoryCorruption(
                        f"{symbol} week sample is not bound to iteration"
                    )
                model_value = candidate_raw.get("model_version")
                if not isinstance(model_value, str):
                    raise RepositoryCorruption(
                        f"{symbol} iteration model version is invalid"
                    )
                analysis_times_by_model[
                    _version_number(
                        model_value, "M", allow_zero=False
                    )
                ] = analysis_time
                previous_state_hash = state_hash

            if (
                latest_output.get("state_hash") != previous_state_hash
                or _value_digest(state_raw) != previous_state_hash
                or len(samples_by_week) != len(rows)
            ):
                raise RepositoryCorruption(
                    f"{symbol} latest work state market/symbol "
                    "hash/sample chain is invalid"
                )
            try:
                latest_state = _work_state_from_dict(state_raw)
            except (
                KeyError,
                TypeError,
                ValueError,
                CrossMarketError,
            ) as exc:
                raise RepositoryCorruption(
                    f"{symbol} latest work state is corrupt"
                ) from exc
            if (
                state_raw.get("_storage_format")
                == _WORK_STATE_STORAGE_FORMAT
            ):
                for audit, raw in zip(latest_state.audits, raw_audits):
                    self._audit_storage_cache[
                        _audit_storage_cache_key(audit)
                    ] = dict(raw)
            self._validate_work_state_chain(latest_state)
            self._validate_label_rows(
                session,
                instrument_id=instrument_id,
                symbol=symbol,
                state=latest_state,
            )
            self._validate_model_rows(
                session,
                instrument_id=instrument_id,
                symbol=symbol,
                state=latest_state,
                analysis_times_by_model=analysis_times_by_model,
            )
            trusted = self._validated_heads.get(symbol)
            if (
                trusted is not None
                and trusted[1] == previous_state_hash
                and trusted[0] == latest_state
            ):
                # A concurrent status/UI read must not replace the exact
                # validated object held by the running incremental engine.
                return trusted[0]
            self._validated_heads[symbol] = (
                latest_state,
                previous_state_hash,
            )
            return latest_state

    def current_model(self, symbol: str) -> ModelVersion:
        """Return the current durable model, validating its parent chain."""

        state = self.load_work_state(symbol)
        symbol = state.instrument_code
        model_number = _version_number(
            state.current_model_version, "M", allow_zero=False
        )
        with self._sessions() as session:
            instrument_id = self._instrument_id(session, symbol)
            row = session.scalar(
                select(V2ModelVersion).where(
                    V2ModelVersion.instrument_id == instrument_id,
                    V2ModelVersion.model_number == model_number,
                )
            )
            if row is None:
                if state.audits:
                    raise RepositoryCorruption(
                        f"{symbol} current model row is missing"
                    )
                return ModelVersion(
                    symbol=symbol,
                    version="M0001",
                    parent_version=None,
                    work_version="W0000",
                    iteration_id="I0000",
                    trained_through_week=None,
                    weights=state.current_model_weights,
                    metrics=FrozenDict({}),
                )
            try:
                audit = _decode_json_mapping(row.audit_data)
                parameters = _decode_json_mapping(row.parameters)
                metrics = _decode_json_mapping(row.metrics)
            except (TypeError, ValueError) as exc:
                raise RepositoryCorruption(
                    f"{symbol} current model payload is corrupt"
                ) from exc
            return ModelVersion(
                symbol=symbol,
                version=f"M{row.model_number:04d}",
                parent_version=_optional_string(
                    audit.get("parent_model_version")
                ),
                work_version=str(
                    audit.get("work_version", state.work_version)
                ),
                iteration_id=str(
                    audit.get("iteration_id", state.iteration_id)
                ),
                trained_through_week=row.trained_through_week_key,
                weights=FrozenDict(
                    _decimal_mapping(parameters.get("weights", {}))
                ),
                metrics=FrozenDict(metrics),
            )

    def commit_iteration(
        self,
        symbol: str,
        iteration: StepAudit,
        work_state: WorkState,
        labels: Iterable[IterationLabel],
        *,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> None:
        """Commit one complete week and its JSON audit as one logical unit."""

        with self._audit_store_lock():
            self._commit_iteration_locked(
                symbol,
                iteration,
                work_state,
                labels,
                execution_guard=execution_guard,
            )

    def commit_iteration_incremental(
        self,
        symbol: str,
        previous_work_state: WorkState,
        result: OptimizerStepResult,
        labels: Iterable[IterationLabel],
        *,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> None:
        """Commit one in-process successor using a validated Task 7 head."""

        if not isinstance(previous_work_state, WorkState):
            raise TypeError("previous_work_state must be a WorkState")
        if not isinstance(result, OptimizerStepResult):
            raise TypeError("result must be an OptimizerStepResult")
        symbol = _validate_symbol(symbol)
        with self._audit_store_lock():
            trusted = self._validated_heads.get(symbol)
            if trusted is None or trusted[0] is not previous_work_state:
                raise RepositoryCorruption(
                    "incremental commit requires the repository's "
                    "last validated WorkState object"
                )
            self._commit_iteration_locked(
                symbol,
                result.audit,
                result.work_state,
                labels,
                trusted_previous=previous_work_state,
                trusted_parent_state_hash=trusted[1],
                execution_guard=execution_guard,
            )

    def _commit_iteration_locked(
        self,
        symbol: str,
        iteration: StepAudit,
        work_state: WorkState,
        labels: Iterable[IterationLabel],
        *,
        trusted_previous: WorkState | None = None,
        trusted_parent_state_hash: str | None = None,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> None:
        symbol = _validate_symbol(symbol)
        label_rows = tuple(labels)
        if trusted_previous is None:
            self._validate_iteration_input(
                symbol, iteration, work_state, label_rows
            )
            changed_label_rows = label_rows
        else:
            self._validate_incremental_iteration_input(
                symbol,
                trusted_previous,
                iteration,
                work_state,
                label_rows,
            )
            previous_labels = {
                (label.iteration_id, label.week_key): label
                for label in trusted_previous.feedback
            }
            changed_label_rows = tuple(
                label
                for label in label_rows
                if previous_labels.get(
                    (label.iteration_id, label.week_key)
                )
                != label
            )
        generated_at = datetime.now(timezone.utc)
        relative_path = _snapshot_relative_path(
            symbol,
            iteration.iteration_id,
            generated_at,
            "iteration",
        )
        final_path = self.audit_root / relative_path
        temp_path: Path | None = None
        replaced = False
        session = self._sessions()
        failure: BaseException | None = None
        try:
            instrument_id = self._instrument_id(session, symbol)
            self._fence_managed_write(
                session,
                execution_guard,
                instrument_id=instrument_id,
                allowed_statuses=("validating",),
            )
            duplicate = session.scalar(
                select(V2AnalysisIteration.id).where(
                    V2AnalysisIteration.instrument_id == instrument_id,
                    V2AnalysisIteration.week_key == iteration.week_key,
                )
            )
            if duplicate is not None:
                raise DuplicateWeek(
                    f"{symbol} week {iteration.week_key} is already complete"
                )
            last_row = session.scalar(
                select(V2AnalysisIteration)
                .where(
                    V2AnalysisIteration.instrument_id == instrument_id
                )
                .order_by(V2AnalysisIteration.iteration_number.desc())
            )
            expected = (
                1 if last_row is None else last_row.iteration_number + 1
            )
            iteration_number = _version_number(
                iteration.iteration_id, "I", allow_zero=False
            )
            work_number = _version_number(
                iteration.work_version, "W", allow_zero=False
            )
            if iteration_number != expected or work_number != expected:
                raise RepositoryCorruption(
                    f"{symbol} next I/W must both be {expected}"
                )
            parent_state_hash: str | None = None
            if last_row is not None:
                embedded = _mapping(
                    last_row.audit_data.get("snapshot"),
                    "previous iteration snapshot",
                )
                parent_state_hash = embedded.get("state_hash")  # type: ignore[assignment]
                previous_output = _decode_json_mapping(
                    last_row.output_payload
                )
                if (
                    not isinstance(parent_state_hash, str)
                    or _HASH_PATTERN.fullmatch(parent_state_hash) is None
                    or previous_output.get("state_hash")
                    != parent_state_hash
                    or "work_state" not in previous_output
                ):
                    raise RepositoryCorruption(
                        f"{symbol} previous state checkpoint is corrupt"
                    )
            if (
                trusted_previous is not None
                and parent_state_hash != trusted_parent_state_hash
            ):
                raise RepositoryCorruption(
                    f"{symbol} trusted incremental parent hash diverged"
                )

            stored_work_state = _work_state_storage_dict(
                work_state,
                audit_cache=self._audit_storage_cache,
            )
            stored_audit = self._audit_storage_cache[
                _audit_storage_cache_key(iteration)
            ]
            snapshot = _hashed_snapshot(
                self._iteration_snapshot(
                    symbol,
                    iteration,
                    work_state,
                    generated_at,
                    parent_state_hash=parent_state_hash,
                    stored_work_state=stored_work_state,
                    stored_audit=stored_audit,
                )
            )
            snapshot_bytes = _snapshot_bytes(snapshot)
            temp_path = _write_snapshot_temp(final_path, snapshot_bytes)

            monday = iteration.cutoff_date - timedelta(
                days=iteration.cutoff_date.weekday()
            )
            audit_reference = {
                "snapshot_path": relative_path.as_posix(),
                "snapshot_hash": snapshot["snapshot_hash"],
                "snapshot": snapshot,
                "symbol": symbol,
                "kind": "iteration",
                "codec_version": _CODEC_VERSION,
            }
            week_sample = V2WeekSample(
                instrument_id=instrument_id,
                week_key=iteration.week_key,
                week_start=monday,
                week_end=iteration.cutoff_date,
                status="complete",
                input_payload=_json_mapping(
                    {
                        "input_hash": iteration.input_hash,
                        "source_hash": iteration.source_hash,
                        "source_data_max_date": (
                            iteration.source_data_max_date
                        ),
                    }
                ),
                target_payload=_json_mapping({}),
                audit_data=_json_mapping(
                    {
                        "cutoff_date": iteration.cutoff_date,
                        "data_integrity": iteration.data_integrity,
                    }
                ),
            )
            session.add(week_sample)
            # The iteration FK targets the week sample by a composite key.
            # An explicit in-transaction flush makes the dependency ordering
            # unambiguous even though these ORM models expose no relationship.
            session.flush((week_sample,))
            session.add(
                V2AnalysisIteration(
                    instrument_id=instrument_id,
                    week_key=iteration.week_key,
                    iteration_number=iteration_number,
                    work_number=work_number,
                    started_at=generated_at,
                    completed_at=generated_at,
                    status="complete",
                    input_payload=_json_mapping(stored_audit),
                    output_payload=_json_mapping(
                        {
                            "state_hash": snapshot["state_hash"],
                            "work_state": stored_work_state,
                        }
                    ),
                    metrics=_json_mapping(
                        _iteration_metrics_payload(iteration)
                    ),
                    audit_data=_json_mapping(audit_reference),
                )
            )
            if last_row is not None:
                last_row.output_payload = _json_mapping(
                    {"state_hash": parent_state_hash}
                )
            self._upsert_labels(
                session,
                instrument_id=instrument_id,
                labels=changed_label_rows,
            )
            self._upsert_current_model(
                session,
                instrument_id=instrument_id,
                symbol=symbol,
                iteration=iteration,
                work_state=work_state,
                generated_at=generated_at,
            )
            self._advance_running_tasks(
                session,
                instrument_id=instrument_id,
                iteration=iteration,
                work_state=work_state,
            )
            self._flush_session(session)
            _replace_snapshot(temp_path, final_path)
            replaced = True
            temp_path = None
            self._commit_session(session)
        except BaseException as exc:
            failure = exc
            _attempt_secondary(failure, "rollback", session.rollback)
            if isinstance(exc, IntegrityError):
                try:
                    duplicate_exists = self._week_exists(
                        symbol, iteration.week_key
                    )
                except BaseException as secondary:
                    _add_secondary_note(
                        failure, "duplicate check", secondary
                    )
                    duplicate_exists = False
                if duplicate_exists:
                    mapped = DuplicateWeek(
                        f"{symbol} week {iteration.week_key} "
                        "is already complete"
                    )
                    mapped.__cause__ = exc
                    failure = mapped
        finally:
            if failure is not None:
                _attempt_secondary(
                    failure,
                    "temporary audit cleanup",
                    lambda: _cleanup_artifact(temp_path),
                )
                if replaced:
                    _attempt_secondary(
                        failure,
                        "final audit cleanup",
                        lambda: _cleanup_artifact(final_path),
                    )
            try:
                session.close()
            except BaseException as close_error:
                if failure is None:
                    raise
                _add_secondary_note(failure, "session close", close_error)
        if failure is not None:
            raise failure
        self._validated_heads[symbol] = (
            work_state,
            str(snapshot["state_hash"]),
        )

    def save_task_checkpoint(
        self,
        task_id: str,
        progress: TaskCheckpoint | Mapping[str, object],
    ) -> None:
        """Create or replace one durable task checkpoint."""

        with self._audit_store_lock():
            self._save_task_checkpoint_locked(task_id, progress)

    def _save_task_checkpoint_locked(
        self,
        task_id: str,
        progress: TaskCheckpoint | Mapping[str, object],
        *,
        expected_worker_token: str | None = None,
        lease_expires_at: datetime | None = None,
        clear_claim: bool = False,
        task_request: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id is required")
        checkpoint = _coerce_checkpoint(task_id.strip(), progress)
        session = self._sessions()
        try:
            instrument_id = self._instrument_id(session, checkpoint.symbol)
            row = session.scalar(
                select(V2AnalysisTask).where(
                    V2AnalysisTask.task_key == checkpoint.task_id
                )
            )
            (
                persisted_week,
                persisted_work,
                persisted_model,
            ) = self._checkpoint_chain_reference(
                session,
                instrument_id=instrument_id,
                symbol=checkpoint.symbol,
                iteration_number=checkpoint.last_iteration,
            )
            if (
                checkpoint.last_work_version != persisted_work
                or checkpoint.last_model_version != persisted_model
                or checkpoint.last_completed_week != persisted_week
            ):
                raise ValueError(
                    "checkpoint progress would regress or diverge "
                    "from the persisted iteration chain"
                )
            payload = _json_mapping(checkpoint.to_dict())
            checkpoint_audit = _json_mapping(
                {
                    "checkpoint_version": _SNAPSHOT_SCHEMA,
                    "codec_version": _CODEC_VERSION,
                    **(
                        {}
                        if checkpoint.last_iteration == 0
                        else {
                            "last_atomic_iteration": (
                                checkpoint.last_work_version.replace(
                                    "W", "I", 1
                                )
                            )
                        }
                    ),
                }
            )
            if row is None:
                error_message = _checkpoint_error_message(checkpoint)
                session.add(
                    V2AnalysisTask(
                        instrument_id=instrument_id,
                        task_key=checkpoint.task_id,
                        task_type="weekly-analysis-v2",
                        status=checkpoint.status,
                        request_payload=_json_mapping(
                            (
                                {"symbol": checkpoint.symbol}
                                if task_request is None
                                else dict(task_request)
                            )
                        ),
                        result_payload=payload,
                        error_message=error_message,
                        audit_data=checkpoint_audit,
                    )
                )
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    actual = self.load_task_checkpoint(
                        checkpoint.task_id
                    )
                    if actual == checkpoint:
                        return
                    raise ValueError(
                        "checkpoint was concurrently created with "
                        "different progress"
                    )
            else:
                if row.instrument_id != instrument_id:
                    raise CrossMarketRepositoryError(
                        "task checkpoint cannot change markets"
                    )
                if row.worker_token != expected_worker_token:
                    raise ValueError(
                        "task checkpoint worker claim token does not match"
                    )
                if expected_worker_token is not None and (
                    row.lease_expires_at is None
                    or row.lease_expires_at <= datetime.now(timezone.utc)
                ):
                    raise ValueError(
                        "task checkpoint worker claim lease expired"
                    )
                try:
                    previous = _checkpoint_from_mapping(
                        row.task_key,
                        checkpoint.symbol,
                        row.status,
                        _decode_json_mapping(row.result_payload),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RepositoryCorruption(
                        f"Task {task_id!r} checkpoint is corrupt"
                    ) from exc
                if (
                    checkpoint.completed_weeks
                    < previous.completed_weeks
                    or (
                        row.status in _LEGACY_TASK_STATUSES
                        and checkpoint.status != "recoverable"
                    )
                    or (
                        row.status not in _LEGACY_TASK_STATUSES
                        and checkpoint.status
                        not in _TASK_STATUS_TRANSITIONS[previous.status]
                    )
                    or (
                        previous.total_weeks is not None
                        and checkpoint.total_weeks
                        != previous.total_weeks
                    )
                ):
                    raise ValueError(
                        "checkpoint progress/status must be monotonic"
                    )
                old_status = row.status
                old_payload = row.result_payload
                values: dict[str, object] = {
                    "status": checkpoint.status,
                    "result_payload": payload,
                    "error_message": _checkpoint_error_message(checkpoint),
                    "audit_data": checkpoint_audit,
                    "updated_at": datetime.now(timezone.utc),
                }
                if clear_claim:
                    values.update(
                        {
                            "worker_token": None,
                            "lease_expires_at": None,
                            "completed_at": datetime.now(timezone.utc),
                        }
                    )
                elif expected_worker_token is not None:
                    values.update(
                        {
                            "worker_token": expected_worker_token,
                            "lease_expires_at": lease_expires_at,
                        }
                    )
                result = session.execute(
                    update(V2AnalysisTask)
                    .where(
                        V2AnalysisTask.id == row.id,
                        V2AnalysisTask.status == old_status,
                        V2AnalysisTask.result_payload == old_payload,
                        V2AnalysisTask.worker_token
                        == expected_worker_token,
                    )
                    .values(**values)
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount != 1:
                    session.rollback()
                    actual = self.load_task_checkpoint(
                        checkpoint.task_id
                    )
                    if actual == checkpoint:
                        return
                    raise ValueError(
                        "checkpoint was concurrently advanced; stale "
                        "progress was rejected"
                    )
                session.commit()
        finally:
            session.close()

    def _checkpoint_chain_reference(
        self,
        session: Session,
        *,
        instrument_id: int,
        symbol: str,
        iteration_number: int,
    ) -> tuple[str | None, str, str]:
        if iteration_number == 0:
            return None, "W0000", "M0001"
        row = session.scalar(
            select(V2AnalysisIteration).where(
                V2AnalysisIteration.instrument_id == instrument_id,
                V2AnalysisIteration.iteration_number == iteration_number,
                V2AnalysisIteration.status == "complete",
            )
        )
        if row is None or row.work_number != iteration_number:
            raise ValueError(
                "checkpoint references a missing or non-contiguous iteration"
            )
        _path, _digest, snapshot = self._verified_snapshot(
            row.audit_data,
            expected_symbol=symbol,
            expected_kind="iteration",
        )
        expected_iteration = f"I{iteration_number:04d}"
        expected_work = f"W{iteration_number:04d}"
        model_version = snapshot.get("model_version")
        if (
            snapshot.get("iteration_id") != expected_iteration
            or snapshot.get("work_version") != expected_work
            or not isinstance(model_version, str)
        ):
            raise ValueError("checkpoint iteration snapshot diverges from chain")
        return row.week_key, expected_work, model_version

    def load_task_checkpoint(self, task_id: str) -> TaskCheckpoint:
        """Load a task checkpoint after a service restart."""

        with self._sessions() as session:
            row = session.scalar(
                select(V2AnalysisTask).where(
                    V2AnalysisTask.task_key == task_id
                )
            )
            if row is None:
                raise CheckpointNotFound(
                    f"No checkpoint exists for task {task_id!r}"
                )
            instrument = session.get(Instrument, row.instrument_id)
            if instrument is None:
                raise RepositoryCorruption(
                    f"Task {task_id!r} references a missing instrument"
                )
            try:
                payload = _decode_json_mapping(row.result_payload)
                request_payload = _decode_json_mapping(
                    row.request_payload
                )
                audit_data = _decode_json_mapping(row.audit_data)
                checkpoint = _checkpoint_from_mapping(
                    task_id,
                    instrument.code,
                    row.status,
                    payload,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RepositoryCorruption(
                    f"Task {task_id!r} checkpoint is corrupt"
                ) from exc
            expected_request_payload: dict[str, object] = {
                "symbol": instrument.code
            }
            captured_request = checkpoint.progress.get("analysis_request")
            if captured_request is not None:
                expected_request_payload["analysis_request"] = (
                    captured_request
                )
            if (
                row.task_type != "weekly-analysis-v2"
                or not _canonical_equal(
                    request_payload, expected_request_payload
                )
                or not _canonical_equal(
                    audit_data,
                    {
                        "checkpoint_version": _SNAPSHOT_SCHEMA,
                        "codec_version": _CODEC_VERSION,
                        **(
                            {}
                            if checkpoint.last_iteration == 0
                            else {
                                "last_atomic_iteration": (
                                    checkpoint.last_work_version.replace(
                                        "W", "I", 1
                                    )
                                )
                            }
                        ),
                    },
                )
            ):
                raise RepositoryCorruption(
                    f"Task {task_id!r} checkpoint metadata is corrupt"
                )
            (
                persisted_week,
                persisted_work,
                persisted_model,
            ) = self._checkpoint_chain_reference(
                session,
                instrument_id=row.instrument_id,
                symbol=instrument.code,
                iteration_number=checkpoint.last_iteration,
            )
            if (
                checkpoint.last_work_version != persisted_work
                or checkpoint.last_completed_week != persisted_week
                or checkpoint.last_model_version != persisted_model
            ):
                raise RepositoryCorruption(
                    f"Task {task_id!r} checkpoint diverges from chain"
                )
            return checkpoint

    def save_advice(
        self,
        symbol: str,
        advice: Advice,
        *,
        generation_key: str | None = None,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> int:
        """Persist one advice generation and deduplicate explicit requests."""

        with self._audit_store_lock():
            return self._save_advice_locked(
                symbol,
                advice,
                generation_key=generation_key,
                execution_guard=execution_guard,
            )

    def _save_advice_locked(
        self,
        symbol: str,
        advice: Advice,
        *,
        generation_key: str | None = None,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> int:
        symbol = _validate_symbol(symbol)
        if not isinstance(advice, Advice):
            raise TypeError("advice must be a validated Advice")
        if advice.instrument_code != symbol:
            raise CrossMarketRepositoryError(
                "advice market does not match repository symbol"
            )
        explicit_generation_key = generation_key is not None
        resolved_generation_key = (
            "default"
            if generation_key is None
            else str(generation_key).strip()
        )
        if (
            not resolved_generation_key
            or len(resolved_generation_key) > 128
        ):
            raise ValueError(
                "generation_key must contain 1..128 characters"
            )
        state = self.load_work_state(symbol)
        if not state.audits:
            raise RepositoryError(
                "advice requires at least one committed iteration"
            )
        latest = state.audits[-1]
        current_metrics = _metrics_from_state(state)
        advice_work_version = advice.audit_fields.get("model_work_version")
        feature_source_hash = advice.audit_fields.get(
            "feature_source_hash"
        )
        if (
            advice.model_version != state.current_model_version
            or advice_work_version != state.work_version
            or advice.data_cutoff_date != latest.cutoff_date
            or (
                advice.source_data_max_date is not None
                and advice.source_data_max_date > latest.cutoff_date
            )
            or not isinstance(feature_source_hash, str)
            or _HASH_PATTERN.fullmatch(feature_source_hash) is None
            or advice.audit_fields.get("calendar_instrument_code")
            != symbol
        ):
            raise RepositoryCorruption(
                "advice cutoff/provenance does not match current I/W/M"
            )
        generated_at = datetime.now(timezone.utc)
        final_path: Path | None = None
        temp_path: Path | None = None
        replaced = False
        session = self._sessions()
        failure: BaseException | None = None
        advice_id: int | None = None
        try:
            instrument_id = self._instrument_id(session, symbol)
            self._fence_managed_write(
                session,
                execution_guard,
                instrument_id=instrument_id,
                allowed_statuses=("generating_advice",),
            )
            model_number = _version_number(
                state.current_model_version, "M", allow_zero=False
            )
            iteration_number = _version_number(
                state.iteration_id, "I", allow_zero=False
            )
            work_number = _version_number(
                state.work_version, "W", allow_zero=False
            )
            duplicate = session.scalar(
                select(V2AdviceHistory).where(
                    V2AdviceHistory.instrument_id == instrument_id,
                    V2AdviceHistory.model_number == model_number,
                    V2AdviceHistory.work_number == work_number,
                    V2AdviceHistory.iteration_number
                    == iteration_number,
                    V2AdviceHistory.generation_key
                    == resolved_generation_key,
                )
            )
            if duplicate is not None:
                if (
                    explicit_generation_key
                    and _canonical_equal(
                        _decode_json_mapping(duplicate.payload),
                        advice.to_dict(),
                    )
                ):
                    return duplicate.id
                raise DuplicateAdvice(
                    f"{symbol} already has advice for "
                    f"{state.iteration_id}/{state.work_version}/"
                    f"{state.current_model_version}/"
                    f"{resolved_generation_key}"
                )
            advice_generation = (
                session.scalar(
                    select(func.max(V2AdviceHistory.advice_generation)).where(
                        V2AdviceHistory.instrument_id == instrument_id,
                        V2AdviceHistory.model_number == model_number,
                        V2AdviceHistory.work_number == work_number,
                        V2AdviceHistory.iteration_number
                        == iteration_number,
                    )
                )
                or 0
            ) + 1
            snapshot = _hashed_snapshot(
                self._advice_snapshot(
                    symbol,
                    advice,
                    state,
                    current_metrics,
                    generated_at,
                    advice_generation=advice_generation,
                    generation_key=resolved_generation_key,
                )
            )
            snapshot_bytes = _snapshot_bytes(snapshot)
            relative_path = _snapshot_relative_path(
                symbol,
                latest.iteration_id,
                generated_at,
                "advice",
            )
            final_path = self.audit_root / relative_path
            temp_path = _write_snapshot_temp(final_path, snapshot_bytes)
            row = V2AdviceHistory(
                instrument_id=instrument_id,
                model_number=model_number,
                work_number=work_number,
                iteration_number=iteration_number,
                advice_generation=advice_generation,
                generation_key=resolved_generation_key,
                advice_at=generated_at,
                status="published",
                payload=_json_mapping(advice.to_dict()),
                audit_data=_json_mapping(
                    {
                        "snapshot_path": relative_path.as_posix(),
                        "artifact_path": str(final_path.resolve()),
                        "audit_root": str(self.audit_root),
                        "snapshot_hash": snapshot["snapshot_hash"],
                        "snapshot": snapshot,
                        "symbol": symbol,
                        "kind": "advice",
                        "codec_version": _CODEC_VERSION,
                    }
                ),
            )
            session.add(row)
            self._flush_session(session)
            advice_id = row.id
            _replace_snapshot(temp_path, final_path)
            replaced = True
            temp_path = None
            self._commit_session(session)
        except BaseException as exc:
            failure = exc
            _attempt_secondary(failure, "rollback", session.rollback)
            if isinstance(exc, IntegrityError):
                try:
                    provenance_exists = self._advice_provenance_exists(
                        symbol,
                        iteration_number=iteration_number,
                        work_number=work_number,
                        model_number=model_number,
                    )
                except BaseException as secondary:
                    _add_secondary_note(
                        failure, "advice provenance check", secondary
                    )
                    provenance_exists = False
                if provenance_exists:
                    mapped = DuplicateAdvice(
                        f"{symbol} already has advice for "
                        f"{state.iteration_id}/{state.work_version}/"
                        f"{state.current_model_version}"
                    )
                else:
                    mapped = RepositoryError(
                        f"Failed to persist advice for {symbol}"
                    )
                mapped.__cause__ = exc
                failure = mapped
        finally:
            if failure is not None:
                _attempt_secondary(
                    failure,
                    "temporary advice audit cleanup",
                    lambda: _cleanup_artifact(temp_path),
                )
                if replaced:
                    _attempt_secondary(
                        failure,
                        "final advice audit cleanup",
                    lambda: _cleanup_artifact(final_path),
                    )
            try:
                session.close()
            except BaseException as close_error:
                if failure is None:
                    raise
                _add_secondary_note(failure, "session close", close_error)
        if failure is not None:
            raise failure
        assert advice_id is not None
        return advice_id

    def latest_advice(self, symbol: str) -> AdviceRecord | None:
        """Return the newest verified advice record for one market."""

        symbol = _validate_symbol(symbol)
        state = self.load_work_state(symbol)
        with self._sessions() as session:
            instrument_id = self._instrument_id(session, symbol)
            row = session.scalar(
                select(V2AdviceHistory)
                .where(V2AdviceHistory.instrument_id == instrument_id)
                .order_by(
                    V2AdviceHistory.advice_at.desc(),
                    V2AdviceHistory.id.desc(),
                )
            )
            if row is None:
                return None
            if (
                row.iteration_number < 1
                or row.iteration_number > len(state.audits)
                or row.work_number != row.iteration_number
            ):
                raise RepositoryCorruption(
                    f"{symbol} advice I/W provenance is invalid"
                )
            canonical_audit = state.audits[row.iteration_number - 1]
            if row.model_number != _version_number(
                canonical_audit.model_version, "M", allow_zero=False
            ):
                raise RepositoryCorruption(
                    f"{symbol} advice model provenance is invalid"
                )
            path, digest, snapshot = self._verified_snapshot(
                row.audit_data,
                expected_symbol=symbol,
                expected_kind="advice",
            )
            advice_payload = _mapping(
                snapshot.get("advice"), "snapshot advice"
            )
            audit_fields = _mapping(
                advice_payload.get("audit_fields"),
                "advice audit_fields",
            )
            expected_iteration = f"I{row.iteration_number:04d}"
            expected_work = f"W{row.work_number:04d}"
            expected_model = f"M{row.model_number:04d}"
            source_max = _parse_optional_date(
                advice_payload.get("source_data_max_date"),
                "source_data_max_date",
            )
            cutoff = _parse_date(
                advice_payload.get("data_cutoff_date"),
                "data_cutoff_date",
            )
            if (
                row.status != "published"
                or snapshot.get("status") != "published"
                or not _canonical_equal(
                    _decode_json_mapping(row.payload), advice_payload
                )
                or advice_payload.get("instrument_code") != symbol
                or advice_payload.get("model_version") != expected_model
                or audit_fields.get("model_work_version")
                != expected_work
                or audit_fields.get("calendar_instrument_code")
                != symbol
                or snapshot.get("iteration_id") != expected_iteration
                or snapshot.get("work_version") != expected_work
                or snapshot.get("model_version") != expected_model
                or snapshot.get("advice_generation")
                != row.advice_generation
                or snapshot.get("generation_key") != row.generation_key
                or row.advice_generation < 1
                or not row.generation_key
                or snapshot.get("data_cutoff") != cutoff
                or snapshot.get("source_data_max_date")
                != source_max
                or snapshot.get("source_hash")
                != audit_fields.get("feature_source_hash")
                or not isinstance(snapshot.get("source_hash"), str)
                or _HASH_PATTERN.fullmatch(
                    str(snapshot.get("source_hash"))
                )
                is None
                or cutoff != canonical_audit.cutoff_date
                or (source_max is not None and source_max > cutoff)
                or _parse_datetime(
                    snapshot.get("analysis_time"), "analysis_time"
                )
                != row.advice_at
            ):
                raise RepositoryCorruption(
                    f"{symbol} advice payload/audit/market is corrupt"
                )
            return AdviceRecord(
                id=row.id,
                symbol=symbol,
                iteration_id=f"I{row.iteration_number:04d}",
                work_version=f"W{row.work_number:04d}",
                model_version=f"M{row.model_number:04d}",
                advice_generation=row.advice_generation,
                generation_key=row.generation_key,
                advice_at=row.advice_at,
                payload=FrozenDict(
                    _decode_json_mapping(row.payload)
                ),
                snapshot_path=path,
                snapshot_hash=digest,
            )

    def metrics(self, symbol: str) -> ModelMetrics:
        """Return model/optimizer metrics recoverable after restart."""

        return _metrics_from_state(self.load_work_state(symbol))

    def _fence_managed_write(
        self,
        session: Session,
        execution_guard: TaskExecutionGuard | None,
        *,
        instrument_id: int,
        allowed_statuses: tuple[str, ...],
    ) -> None:
        if execution_guard is None:
            return
        if not isinstance(execution_guard, TaskExecutionGuard):
            raise TypeError(
                "execution_guard must be a TaskExecutionGuard or None"
            )
        now = datetime.now(timezone.utc)
        result = session.execute(
            update(V2AnalysisTask)
            .where(
                V2AnalysisTask.task_key == execution_guard.task_id,
                V2AnalysisTask.instrument_id == instrument_id,
                V2AnalysisTask.worker_token
                == execution_guard.worker_token,
                V2AnalysisTask.lease_expires_at.is_not(None),
                V2AnalysisTask.lease_expires_at > now,
                V2AnalysisTask.status.in_(allowed_statuses),
            )
            .values(
                lease_expires_at=V2AnalysisTask.lease_expires_at
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            current = session.scalar(
                select(V2AnalysisTask).where(
                    V2AnalysisTask.task_key == execution_guard.task_id
                )
            )
            diagnostic = (
                "task_missing"
                if current is None
                else (
                    f"status={current.status},"
                    f"token_match={current.worker_token == execution_guard.worker_token},"
                    f"lease_active={current.lease_expires_at is not None and current.lease_expires_at > now}"
                )
            )
            raise TaskClaimLost(
                "managed task execution claim is missing, expired, "
                f"or in the wrong durable stage ({diagnostic})"
            )

    def _instrument_id(self, session: Session, symbol: str) -> int:
        symbol = _validate_symbol(symbol)
        instrument_id = session.scalar(
            select(Instrument.id).where(Instrument.code == symbol)
        )
        if instrument_id is None:
            raise UnknownMarket(
                f"V2 market {symbol!r} is not present in instruments"
            )
        return instrument_id

    def _validate_iteration_input(
        self,
        symbol: str,
        iteration: StepAudit,
        work_state: WorkState,
        labels: tuple[IterationLabel, ...],
    ) -> None:
        if not isinstance(iteration, StepAudit):
            raise TypeError("iteration must be a StepAudit")
        if not isinstance(work_state, WorkState):
            raise TypeError("work_state must be a WorkState")
        markets = {
            iteration.instrument_code,
            work_state.instrument_code,
            *(label.instrument_code for label in labels),
        }
        if markets != {symbol}:
            raise CrossMarketRepositoryError(
                f"commit values do not all belong to {symbol}"
            )
        if (
            not work_state.audits
            or work_state.audits[-1] != iteration
            or work_state.iteration_id != iteration.iteration_id
            or work_state.work_version != iteration.work_version
            or work_state.current_model_version != iteration.model_version
        ):
            raise RepositoryCorruption(
                "iteration is not the terminal audit of work_state"
            )
        if (
            iteration.source_data_max_date is not None
            and iteration.source_data_max_date > iteration.cutoff_date
        ):
            raise RepositoryCorruption(
                "source_data_max_date exceeds cutoff_date"
            )
        supplied_identities = tuple(
            (label.iteration_id, label.week_key) for label in labels
        )
        if len(set(supplied_identities)) != len(supplied_identities):
            raise RepositoryCorruption(
                "duplicate label identity is not allowed"
            )
        supplied = {
            identity: label
            for identity, label in zip(supplied_identities, labels)
        }
        expected = {
            (label.iteration_id, label.week_key): label
            for label in work_state.feedback
        }
        if supplied != expected:
            raise RepositoryCorruption(
                "labels must exactly match work_state feedback"
            )
        self._validate_work_state_chain(work_state)

    def _validate_incremental_iteration_input(
        self,
        symbol: str,
        previous: WorkState,
        iteration: StepAudit,
        work_state: WorkState,
        labels: tuple[IterationLabel, ...],
    ) -> None:
        """Validate only the successor edge after the prefix was fully proven."""

        if {
            previous.instrument_code,
            iteration.instrument_code,
            work_state.instrument_code,
            *(label.instrument_code for label in labels),
        } != {symbol}:
            raise CrossMarketRepositoryError(
                "incremental commit values do not all belong to one market"
            )
        expected_number = len(previous.audits) + 1
        if (
            len(work_state.audits) != expected_number
            or len(work_state.week_records) != expected_number
            or work_state.audits[:-1] != previous.audits
            or work_state.audits[-1] != iteration
            or iteration.parent_work_version != previous.work_version
            or iteration.work_version != f"W{expected_number:04d}"
            or iteration.iteration_id != f"I{expected_number:04d}"
            or work_state.work_version != iteration.work_version
            or work_state.iteration_id != iteration.iteration_id
            or work_state.parent_work_version != previous.work_version
            or iteration.parent_model_version
            != previous.current_model_version
            or work_state.current_model_version
            != iteration.model_version
            or iteration.week_key in previous.week_records
            or work_state.week_records.get(iteration.week_key) != iteration
            or any(
                work_state.week_records.get(audit.week_key) != audit
                for audit in previous.audits
            )
        ):
            raise RepositoryCorruption(
                "incremental I/W/M successor edge is invalid"
            )
        supplied = {
            (label.iteration_id, label.week_key): label
            for label in labels
        }
        expected_labels = {
            (label.iteration_id, label.week_key): label
            for label in work_state.feedback
        }
        if len(supplied) != len(labels) or supplied != expected_labels:
            raise RepositoryCorruption(
                "labels must exactly match incremental work_state feedback"
            )
        if (
            iteration.source_data_max_date is not None
            and iteration.source_data_max_date > iteration.cutoff_date
        ):
            raise RepositoryCorruption(
                "source_data_max_date exceeds cutoff_date"
            )
        effective_weights = validate_step_audit_acceptance(
            iteration,
            previous_model_version=previous.current_model_version,
            previous_model_weights=previous.current_model_weights,
            previous_optimizer_memory=previous.optimizer_memory,
        )
        if (
            effective_weights != work_state.current_model_weights
            or work_state.optimizer_memory != iteration.optimizer_memory
            or iteration.feedback_count != len(work_state.feedback)
        ):
            raise RepositoryCorruption(
                "incremental model/optimizer successor is inconsistent"
            )

    def _validate_iteration_row(
        self,
        session: Session,
        *,
        row: V2AnalysisIteration,
        instrument_id: int,
        symbol: str,
        expected: int,
    ) -> WorkState:
        if (
            row.status != "complete"
            or row.iteration_number != expected
            or row.work_number != expected
        ):
            raise RepositoryCorruption(
                f"{symbol} iteration/work chain is not contiguous"
            )
        _path, _digest, snapshot = self._verified_snapshot(
            row.audit_data,
            expected_symbol=symbol,
            expected_kind="iteration",
        )
        candidate_raw = _mapping(
            snapshot.get("candidate_audit"), "candidate_audit"
        )
        state_raw = _mapping(snapshot.get("work_state"), "work_state")
        expected_metrics = {
            "accepted": candidate_raw.get("accepted"),
            "current": snapshot.get("current_metrics"),
            "candidate": snapshot.get("candidate_metrics"),
        }
        if (
            not _canonical_equal(row.input_payload, candidate_raw)
            or not _canonical_equal(row.output_payload, state_raw)
            or not _canonical_equal(row.metrics, expected_metrics)
        ):
            raise RepositoryCorruption(
                f"{symbol} DB work state/market payload or metrics "
                "are not bound to snapshot"
            )
        try:
            state = _work_state_from_dict(state_raw)
            audit = _step_audit_from_dict(candidate_raw)
            analysis_time = _parse_datetime(
                snapshot.get("analysis_time"), "analysis_time"
            )
        except (KeyError, TypeError, ValueError, CrossMarketError) as exc:
            raise RepositoryCorruption(
                f"{symbol} canonical iteration snapshot is invalid"
            ) from exc
        expected_iteration = f"I{expected:04d}"
        expected_work = f"W{expected:04d}"
        if (
            not state.audits
            or state.instrument_code != symbol
            or audit.instrument_code != symbol
            or state.audits[-1] != audit
            or state.iteration_id != expected_iteration
            or state.work_version != expected_work
            or audit.iteration_id != expected_iteration
            or audit.work_version != expected_work
            or audit.week_key != row.week_key
            or snapshot.get("symbol") != symbol
            or snapshot.get("iteration_id") != expected_iteration
            or snapshot.get("work_version") != expected_work
            or snapshot.get("model_version") != audit.model_version
            or snapshot.get("data_cutoff")
            != audit.cutoff_date.isoformat()
            or snapshot.get("source_data_max_date")
            != (
                None
                if audit.source_data_max_date is None
                else audit.source_data_max_date.isoformat()
            )
            or snapshot.get("source_hash") != audit.source_hash
            or row.completed_at != analysis_time
            or len(state.audits) != expected
        ):
            raise RepositoryCorruption(
                f"{symbol} snapshot market/version/date chain is invalid"
            )
        sample = session.scalar(
            select(V2WeekSample).where(
                V2WeekSample.instrument_id == instrument_id,
                V2WeekSample.week_key == row.week_key,
            )
        )
        monday = audit.cutoff_date - timedelta(
            days=audit.cutoff_date.weekday()
        )
        expected_sample_input = {
            "input_hash": audit.input_hash,
            "source_hash": audit.source_hash,
            "source_data_max_date": audit.source_data_max_date,
        }
        expected_sample_audit = {
            "cutoff_date": audit.cutoff_date,
            "data_integrity": audit.data_integrity,
        }
        if (
            sample is None
            or sample.status != "complete"
            or sample.week_start != monday
            or sample.week_end != audit.cutoff_date
            or not _canonical_equal(
                sample.input_payload, expected_sample_input
            )
            or not _canonical_equal(
                sample.audit_data, expected_sample_audit
            )
        ):
            raise RepositoryCorruption(
                f"{symbol} week sample is not bound to iteration snapshot"
            )
        return state

    def _validate_work_state_chain(self, state: WorkState) -> None:
        seed = seed_model(state.instrument_code)
        previous_model_version = seed.current_model_version
        previous_weights = seed.current_model_weights
        previous_memory = seed.optimizer_memory
        previous_work = seed.work_version
        seen_weeks: set[str] = set()
        for number, audit in enumerate(state.audits, start=1):
            if (
                audit.instrument_code != state.instrument_code
                or audit.iteration_id != f"I{number:04d}"
                or audit.work_version != f"W{number:04d}"
                or audit.parent_work_version != previous_work
                or audit.week_key in seen_weeks
                or _HASH_PATTERN.fullmatch(audit.source_hash) is None
                or _HASH_PATTERN.fullmatch(audit.input_hash) is None
                or (
                    audit.source_data_max_date is not None
                    and audit.source_data_max_date > audit.cutoff_date
                )
            ):
                raise RepositoryCorruption(
                    f"{state.instrument_code} I/W/week chain is invalid"
                )
            try:
                effective_weights = validate_step_audit_acceptance(
                    audit,
                    previous_model_version=previous_model_version,
                    previous_model_weights=previous_weights,
                    previous_optimizer_memory=previous_memory,
                )
            except (ValueError, CrossMarketError) as exc:
                raise RepositoryCorruption(
                    f"{state.instrument_code} acceptance audit is invalid"
                ) from exc
            previous_model_version = audit.model_version
            previous_weights = effective_weights
            previous_memory = audit.optimizer_memory
            previous_work = audit.work_version
            seen_weeks.add(audit.week_key)
        if (
            state.iteration_id != f"I{len(state.audits):04d}"
            or state.work_version != f"W{len(state.audits):04d}"
            or state.current_model_version != previous_model_version
            or state.current_model_weights != previous_weights
            or state.optimizer_memory != previous_memory
            or set(state.week_records) != seen_weeks
            or any(
                state.week_records.get(audit.week_key) != audit
                for audit in state.audits
            )
        ):
            raise RepositoryCorruption(
                f"{state.instrument_code} terminal work state is inconsistent"
            )

    def _validate_label_rows(
        self,
        session: Session,
        *,
        instrument_id: int,
        symbol: str,
        state: WorkState,
    ) -> None:
        rows = tuple(
            session.scalars(
                select(V2IterationLabel)
                .where(V2IterationLabel.instrument_id == instrument_id)
                .order_by(V2IterationLabel.week_key)
            )
        )
        try:
            payloads = tuple(
                _decode_json_mapping(row.payload) for row in rows
            )
            labels = tuple(
                _label_from_dict(payload)
                for payload in payloads
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RepositoryCorruption(
                f"{symbol} persisted labels are corrupt"
            ) from exc
        if (
            any(
                row.week_key != label.week_key
                or row.label_date != label.observed_through
                or row.maturity_date
                != (
                    label.observed_through
                    if label.status == "mature_13w"
                    else None
                )
                or row.label_value
                != (
                    None
                    if label.total_loss is None
                    else _fixed_point_decimal(label.total_loss)
                )
                or row.label != label.actual_direction
                or row.status != label.status
                or not _canonical_equal(
                    _decode_json_mapping(row.audit_data),
                    {
                        "iteration_id": label.iteration_id,
                        "cutoff_date": label.cutoff_date,
                        "observed_through": label.observed_through,
                    },
                )
                for row, label in zip(rows, labels)
            )
            or set(labels) != set(state.feedback)
        ):
            raise RepositoryCorruption(
                f"{symbol} label state disagrees with work state"
            )

    def _validate_model_rows(
        self,
        session: Session,
        *,
        instrument_id: int,
        symbol: str,
        state: WorkState,
        analysis_times_by_model: Mapping[int, datetime],
    ) -> None:
        rows = tuple(
            session.scalars(
                select(V2ModelVersion)
                .where(V2ModelVersion.instrument_id == instrument_id)
                .order_by(V2ModelVersion.model_number)
            )
        )
        expected_current = _version_number(
            state.current_model_version, "M", allow_zero=False
        )
        if tuple(row.model_number for row in rows) != tuple(
            range(1, expected_current + 1)
        ):
            raise RepositoryCorruption(
                f"{symbol} model version chain is not contiguous"
            )
        weights_by_number: dict[int, Mapping[str, Decimal]] = {
            1: seed_model(symbol).current_model_weights
        }
        last_audit_by_number: dict[int, StepAudit] = {}
        for audit in state.audits:
            model_number = _version_number(
                audit.model_version, "M", allow_zero=False
            )
            if audit.accepted:
                weights_by_number[model_number] = audit.candidate_weights
            last_audit_by_number[model_number] = audit
        for row in rows:
            expected_parent = (
                None
                if row.model_number == 1
                else f"M{row.model_number - 1:04d}"
            )
            audit = last_audit_by_number.get(row.model_number)
            if audit is None:
                raise RepositoryCorruption(
                    f"{symbol} model row has no canonical iteration audit"
                )
            expected_parameters = {
                "weights": dict(weights_by_number[row.model_number]),
                "optimizer_memory": dict(audit.optimizer_memory),
            }
            expected_status = (
                "active"
                if row.model_number == expected_current
                else "superseded"
            )
            expected_analysis_time = analysis_times_by_model.get(
                row.model_number
            )
            try:
                parameters = _decode_json_mapping(row.parameters)
                metrics = _decode_json_mapping(row.metrics)
                audit_data = _decode_json_mapping(row.audit_data)
            except (TypeError, ValueError) as exc:
                raise RepositoryCorruption(
                    f"{symbol} model tagged payload is corrupt"
                ) from exc
            if (
                row.status != expected_status
                or row.trained_through_week_key != audit.week_key
                or expected_analysis_time is None
                or row.training_started_at != expected_analysis_time
                or row.training_completed_at != expected_analysis_time
                or row.activated_at != expected_analysis_time
                or row.artifact_path is not None
                or not _canonical_equal(
                    parameters, expected_parameters
                )
                or not _canonical_equal(
                    metrics, _model_metrics_payload(audit)
                )
                or not _canonical_equal(
                    audit_data,
                    _model_audit_payload(
                        audit,
                        parent_model_version=expected_parent,
                    ),
                )
            ):
                raise RepositoryCorruption(
                    f"{symbol} model audit W/I/metrics chain is invalid"
                )
        current = rows[-1]
        if _decimal_mapping(
            _decode_json_mapping(current.parameters).get("weights", {})
        ) != dict(state.current_model_weights):
            raise RepositoryCorruption(
                f"{symbol} current model weights are corrupt"
            )

    def _upsert_labels(
        self,
        session: Session,
        *,
        instrument_id: int,
        labels: tuple[IterationLabel, ...],
    ) -> None:
        week_keys = tuple(label.week_key for label in labels)
        rows_by_week: dict[str, V2IterationLabel] = {}
        for offset in range(0, len(week_keys), 500):
            key_batch = week_keys[offset : offset + 500]
            rows_by_week.update(
                {
                    row.week_key: row
                    for row in session.scalars(
                        select(V2IterationLabel).where(
                            V2IterationLabel.instrument_id
                            == instrument_id,
                            V2IterationLabel.week_key.in_(key_batch),
                        )
                    )
                }
            )
        for label in labels:
            row = rows_by_week.get(label.week_key)
            values = {
                "label_date": label.observed_through,
                "maturity_date": (
                    label.observed_through
                    if label.status == "mature_13w"
                    else None
                ),
                "label_value": (
                    None
                    if label.total_loss is None
                    else _fixed_point_decimal(label.total_loss)
                ),
                "label": label.actual_direction,
                "status": label.status,
                "payload": _json_mapping(label.to_dict()),
                "audit_data": _json_mapping(
                    {
                        "iteration_id": label.iteration_id,
                        "cutoff_date": label.cutoff_date,
                        "observed_through": label.observed_through,
                    }
                ),
            }
            if row is None:
                session.add(
                    V2IterationLabel(
                        instrument_id=instrument_id,
                        week_key=label.week_key,
                        **values,
                    )
                )
            else:
                for field_name, value in values.items():
                    setattr(row, field_name, value)

    def _upsert_current_model(
        self,
        session: Session,
        *,
        instrument_id: int,
        symbol: str,
        iteration: StepAudit,
        work_state: WorkState,
        generated_at: datetime,
    ) -> None:
        seed = seed_model(symbol)
        seed_row = session.scalar(
            select(V2ModelVersion).where(
                V2ModelVersion.instrument_id == instrument_id,
                V2ModelVersion.model_number == 1,
            )
        )
        if seed_row is None:
            seed_row = V2ModelVersion(
                instrument_id=instrument_id,
                model_number=1,
                status="active",
                parameters=_json_mapping(
                    {"weights": dict(seed.current_model_weights)}
                ),
                metrics={},
                audit_data={
                    "parent_model_version": None,
                    "work_version": "W0000",
                    "iteration_id": "I0000",
                    "seed": True,
                },
            )
            session.add(seed_row)
        current_number = _version_number(
            work_state.current_model_version, "M", allow_zero=False
        )
        current = (
            seed_row
            if current_number == 1
            else session.scalar(
                select(V2ModelVersion).where(
                    V2ModelVersion.instrument_id == instrument_id,
                    V2ModelVersion.model_number == current_number,
                )
            )
        )
        if current is None:
            previous_number = current_number - 1
            previous = session.scalar(
                select(V2ModelVersion.id).where(
                    V2ModelVersion.instrument_id == instrument_id,
                    V2ModelVersion.model_number == previous_number,
                )
            )
            if previous is None:
                raise RepositoryCorruption(
                    f"{symbol} model parent M{previous_number:04d} is missing"
                )
            current = V2ModelVersion(
                instrument_id=instrument_id,
                model_number=current_number,
            )
            session.add(current)
        parent = (
            None
            if current_number == 1
            else f"M{current_number - 1:04d}"
        )
        current.trained_through_week_key = iteration.week_key
        current.training_started_at = generated_at
        current.training_completed_at = generated_at
        current.activated_at = generated_at
        current.status = "active"
        current.parameters = _json_mapping(
            {
                "weights": dict(work_state.current_model_weights),
                "optimizer_memory": dict(work_state.optimizer_memory),
            }
        )
        current.metrics = _json_mapping(
            _model_metrics_payload(iteration)
        )
        current.audit_data = _json_mapping(
            _model_audit_payload(
                iteration,
                parent_model_version=parent,
            )
        )
        for row in session.scalars(
            select(V2ModelVersion).where(
                V2ModelVersion.instrument_id == instrument_id,
                V2ModelVersion.model_number != current_number,
                V2ModelVersion.status == "active",
            )
        ):
            row.status = "superseded"

    def _advance_running_tasks(
        self,
        session: Session,
        *,
        instrument_id: int,
        iteration: StepAudit,
        work_state: WorkState,
    ) -> None:
        for task in session.scalars(
            select(V2AnalysisTask).where(
                V2AnalysisTask.instrument_id == instrument_id,
                V2AnalysisTask.status.in_(
                    (
                        "preparing_data",
                        "building_features",
                        "iterating",
                        "validating",
                        "generating_advice",
                    )
                ),
            )
        ):
            payload = _decode_json_mapping(task.result_payload)
            payload.update(
                {
                    "completed_weeks": _version_number(
                        work_state.iteration_id, "I", allow_zero=False
                    ),
                    "last_iteration": _version_number(
                        work_state.iteration_id, "I", allow_zero=False
                    ),
                    "last_work_version": work_state.work_version,
                    "last_model_version": work_state.current_model_version,
                    "last_completed_week": iteration.week_key,
                    "status": task.status,
                    "symbol": iteration.instrument_code,
                    "task_id": task.task_key,
                }
            )
            encoded_payload = _json_mapping(payload)
            encoded_audit = _json_mapping(
                _decode_json_mapping(task.audit_data)
                | {
                    "last_atomic_iteration": work_state.iteration_id,
                }
            )
            session.execute(
                update(V2AnalysisTask)
                .where(
                    V2AnalysisTask.id == task.id,
                    V2AnalysisTask.updated_at == task.updated_at,
                    V2AnalysisTask.status == task.status,
                    V2AnalysisTask.result_payload
                    == task.result_payload,
                )
                .values(
                    status=task.status,
                    result_payload=encoded_payload,
                    audit_data=encoded_audit,
                    updated_at=datetime.now(timezone.utc),
                )
                .execution_options(synchronize_session=False)
            )

    def _iteration_snapshot(
        self,
        symbol: str,
        iteration: StepAudit,
        state: WorkState,
        generated_at: datetime,
        *,
        parent_state_hash: str | None,
        stored_work_state: Mapping[str, object] | None = None,
        stored_audit: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        persisted_state = (
            _work_state_storage_dict(state)
            if stored_work_state is None
            else dict(stored_work_state)
        )
        persisted_audit = (
            _step_audit_storage_dict(
                iteration,
                None if len(state.audits) < 2 else state.audits[-2],
            )
            if stored_audit is None
            else dict(stored_audit)
        )
        state_hash = _value_digest(persisted_state)
        return {
            "schema_version": _SNAPSHOT_SCHEMA,
            "snapshot_format": _SNAPSHOT_FORMAT,
            "kind": "iteration",
            "symbol": symbol,
            "analysis_time": generated_at,
            "data_cutoff": iteration.cutoff_date,
            "source_data_max_date": iteration.source_data_max_date,
            "source_hash": iteration.source_hash,
            "seed": {
                "model_version": "M0001",
                "weights": dict(seed_model(symbol).current_model_weights),
            },
            "iteration_id": iteration.iteration_id,
            "work_version": iteration.work_version,
            "model_version": iteration.model_version,
            "current_metrics": {
                name: metric.to_dict()
                for name, metric in iteration.current_window_metrics.items()
            },
            "candidate_metrics": {
                name: metric.to_dict()
                for name, metric in iteration.candidate_window_metrics.items()
            },
            "current_position": None,
            "target_position": None,
            "direction_probabilities": {},
            "probability": None,
            "confidence": None,
            "market_state": None,
            "batches": [],
            "fund_etf_ratio": None,
            "candidate_audit": persisted_audit,
            "state_hash": state_hash,
            "parent_state_hash": parent_state_hash,
            "provenance": {
                "algorithm_version": iteration.algorithm_version,
                "loss_version": iteration.loss_version,
                "input_hash": iteration.input_hash,
                "parent_work_version": iteration.parent_work_version,
                "parent_model_version": iteration.parent_model_version,
            },
            "integrity": {
                "data_integrity": iteration.data_integrity,
                "future_leakage": iteration.future_leakage,
                "source_not_after_cutoff": (
                    iteration.source_data_max_date is None
                    or iteration.source_data_max_date
                    <= iteration.cutoff_date
                ),
                "all_candidates_audited": True,
            },
        }

    def _advice_snapshot(
        self,
        symbol: str,
        advice: Advice,
        state: WorkState,
        metrics: ModelMetrics,
        generated_at: datetime,
        *,
        advice_generation: int,
        generation_key: str,
    ) -> dict[str, object]:
        latest = state.audits[-1]
        return {
            "schema_version": _SNAPSHOT_SCHEMA,
            "kind": "advice",
            "status": "published",
            "symbol": symbol,
            "analysis_time": generated_at,
            "data_cutoff": advice.data_cutoff_date,
            "source_data_max_date": advice.source_data_max_date,
            "source_hash": advice.audit_fields["feature_source_hash"],
            "seed": {
                "model_version": "M0001",
                "weights": dict(seed_model(symbol).current_model_weights),
            },
            "iteration_id": state.iteration_id,
            "work_version": state.work_version,
            "model_version": state.current_model_version,
            "advice_generation": advice_generation,
            "generation_key": generation_key,
            "current_metrics": metrics.to_dict(),
            "current_position": advice.current_position,
            "target_position": advice.target_position,
            "direction_probabilities": dict(
                advice.direction_probabilities
            ),
            "probability": advice.probability,
            "confidence": advice.confidence,
            "market_state": advice.market_state,
            "batches": tuple(batch.to_dict() for batch in advice.batches),
            "fund_etf_ratio": advice.fund_etf_ratio,
            "advice": advice.to_dict(),
            "provenance": {
                "advice_version": advice.advice_version,
                "feature_set_version": advice.feature_set_version,
                "forecast_horizon_weeks": (
                    advice.forecast_horizon_weeks
                ),
                "iteration_input_hash": latest.input_hash,
                "iteration_source_hash": latest.source_hash,
            },
            "integrity": {
                "source_not_after_cutoff": (
                    advice.source_data_max_date is None
                    or advice.source_data_max_date
                    <= advice.data_cutoff_date
                ),
                "feature_publishable": advice.audit_fields[
                    "feature_publishable"
                ],
                "position_quantization": advice.audit_fields[
                    "position_quantization"
                ],
            },
        }

    def _verified_snapshot(
        self,
        audit_data: Mapping[str, object],
        *,
        expected_symbol: str,
        expected_kind: str,
    ) -> tuple[Path, str, dict[str, object]]:
        relative_value = audit_data.get("snapshot_path")
        digest = audit_data.get("snapshot_hash")
        embedded = audit_data.get("snapshot")
        if (
            not isinstance(relative_value, str)
            or not isinstance(digest, str)
            or _HASH_PATTERN.fullmatch(digest) is None
            or not isinstance(embedded, Mapping)
            or audit_data.get("symbol") != expected_symbol
            or audit_data.get("kind") != expected_kind
        ):
            raise RepositoryCorruption(
                f"{expected_symbol} snapshot reference is incomplete"
            )
        path = self._safe_snapshot_path(relative_value, expected_symbol)
        embedded_dict = dict(embedded)
        if (
            embedded_dict.get("snapshot_hash") != digest
            or _snapshot_digest(embedded_dict) != digest
        ):
            raise RepositoryCorruption(
                f"{expected_symbol} embedded snapshot/hash is corrupt"
            )
        if not path.exists():
            temp = _write_snapshot_temp(
                path, _snapshot_bytes(embedded_dict)
            )
            try:
                _replace_snapshot(temp, path)
            except Exception:
                _cleanup_artifact(temp)
                raise
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryCorruption(
                f"{expected_symbol} JSON snapshot is unreadable"
            ) from exc
        if (
            not isinstance(parsed, dict)
            or parsed.get("symbol") != expected_symbol
            or parsed.get("kind") != expected_kind
            or parsed.get("codec_version") != _CODEC_VERSION
            or parsed.get("snapshot_hash") != digest
            or _snapshot_digest(parsed) != digest
            or not _canonical_equal(parsed, embedded_dict)
        ):
            raise RepositoryCorruption(
                f"{expected_symbol} JSON snapshot hash/market is corrupt"
            )
        decoded = _decode_json_mapping(parsed)
        return path, digest, decoded

    def _safe_snapshot_path(
        self, relative_value: str, expected_symbol: str
    ) -> Path:
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RepositoryCorruption("snapshot path escapes audit root")
        path = (self.audit_root / relative).resolve()
        try:
            path.relative_to(self.audit_root)
        except ValueError as exc:
            raise RepositoryCorruption(
                "snapshot path escapes audit root"
            ) from exc
        if not relative.parts or relative.parts[0] != expected_symbol:
            raise RepositoryCorruption(
                "snapshot path does not match its market"
            )
        return path

    def _discard_abandoned_temps(self) -> None:
        for path in self.audit_root.rglob("*.tmp"):
            _cleanup_artifact(path)

    def _repair_missing_snapshots(self) -> None:
        """Repair only missing files; validation rejects existing corruption."""

        try:
            with self._sessions() as session:
                references: list[tuple[Mapping[str, object], str, str]] = []
                for row, symbol in session.execute(
                    select(V2AnalysisIteration, Instrument.code)
                    .join(
                        Instrument,
                        Instrument.id
                        == V2AnalysisIteration.instrument_id,
                    )
                    .where(V2AnalysisIteration.status == "complete")
                ):
                    if symbol not in _MARKETS:
                        raise RepositoryCorruption(
                            "V2 iteration references an invalid market"
                        )
                    references.append((row.audit_data, symbol, "iteration"))
                for row, symbol in session.execute(
                    select(V2AdviceHistory, Instrument.code)
                    .join(
                        Instrument,
                        Instrument.id == V2AdviceHistory.instrument_id,
                    )
                    .where(V2AdviceHistory.status == "published")
                ):
                    if symbol not in _MARKETS:
                        raise RepositoryCorruption(
                            "V2 advice references an invalid market"
                        )
                    references.append((row.audit_data, symbol, "advice"))
                referenced_paths: set[Path] = set()
                for audit, symbol, kind in references:
                    relative = audit.get("snapshot_path")
                    if not isinstance(relative, str):
                        continue
                    path = self._safe_snapshot_path(relative, symbol)
                    referenced_paths.add(path)
                    if not path.exists():
                        self._verified_snapshot(
                            audit,
                            expected_symbol=symbol,
                            expected_kind=kind,
                        )
                # A crash after os.replace but before SQLite commit can leave
                # a complete yet unreferenced final JSON.  Keep it auditable,
                # but move it out of the normal ``*.json`` reader namespace.
                for path in self.audit_root.rglob("*.json"):
                    resolved = path.resolve()
                    if resolved in referenced_paths:
                        continue
                    quarantine = path.with_suffix(path.suffix + ".orphan")
                    if quarantine.exists():
                        quarantine = path.with_suffix(
                            path.suffix + f".{uuid4().hex}.orphan"
                        )
                    os.replace(path, quarantine)
        except (
            RepositoryCorruption,
            UnknownMarket,
        ):
            raise
        except Exception:
            # A caller may construct the repository before migrations/tables.
            # Ordinary repository methods will surface any real schema error.
            return

    def _symbol_for_instrument(
        self, session: Session, instrument_id: int
    ) -> str:
        symbol = session.scalar(
            select(Instrument.code).where(Instrument.id == instrument_id)
        )
        if symbol not in _MARKETS:
            raise RepositoryCorruption(
                "V2 row references a non-V2 or missing market"
            )
        return symbol

    def _week_exists(self, symbol: str, week_key: str) -> bool:
        with self._sessions() as session:
            instrument_id = self._instrument_id(session, symbol)
            return (
                session.scalar(
                    select(V2AnalysisIteration.id).where(
                        V2AnalysisIteration.instrument_id == instrument_id,
                        V2AnalysisIteration.week_key == week_key,
                    )
                )
                is not None
            )

    def _advice_provenance_exists(
        self,
        symbol: str,
        *,
        iteration_number: int,
        work_number: int,
        model_number: int,
    ) -> bool:
        with self._sessions() as session:
            instrument_id = self._instrument_id(session, symbol)
            return (
                session.scalar(
                    select(V2AdviceHistory.id).where(
                        V2AdviceHistory.instrument_id == instrument_id,
                        V2AdviceHistory.iteration_number
                        == iteration_number,
                        V2AdviceHistory.work_number == work_number,
                        V2AdviceHistory.model_number == model_number,
                    )
                )
                is not None
            )

    def _flush_session(self, session: Session) -> None:
        session.flush()

    def _commit_session(self, session: Session) -> None:
        session.commit()


def _validate_symbol(symbol: str) -> str:
    if symbol not in _MARKETS:
        raise UnknownMarket(
            f"V2 weekly analysis supports only {sorted(_MARKETS)}"
        )
    return symbol


def _version_number(
    value: str, prefix: str, *, allow_zero: bool
) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{prefix} version must be a string")
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None or match.group("prefix") != prefix:
        raise ValueError(f"Invalid {prefix} version: {value!r}")
    number = int(match.group("number"))
    if number < (0 if allow_zero else 1):
        raise ValueError(f"Invalid {prefix} version: {value!r}")
    return number


def _decimal(value: object, field_name: str) -> Decimal:
    try:
        result = (
            value if isinstance(value, Decimal) else Decimal(str(value))
        )
    except Exception as exc:
        raise ValueError(f"{field_name} must be Decimal-compatible") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _fixed_point_decimal(value: Decimal) -> Decimal:
    """Round a derived metric to the database's explicit 8-place scale."""

    return value.quantize(
        _FIXED_POINT_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _decimal_mapping(values: Mapping[str, object]) -> dict[str, Decimal]:
    return {
        str(name): _decimal(value, f"value:{name}")
        for name, value in values.items()
    }


def _strict_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _strict_int(
    value: object, field_name: str, *, minimum: int = 0
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{field_name} must be an integer >= {minimum}"
        )
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RepositoryCorruption("expected string or null")
    return value


def _json_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    encoded = _encode_typed(value)
    if not isinstance(encoded, dict):
        raise TypeError("Tagged JSON mapping did not produce an object")
    return encoded


def _decode_json_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    decoded = _decode_typed(value)
    if not isinstance(decoded, dict):
        raise TypeError("Tagged JSON mapping did not decode to an object")
    return decoded


def _encode_typed(value: object) -> Any:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Decimal JSON values must be finite")
        return {_TYPE_TAG: "decimal", _TYPE_VALUE: str(value)}
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetime JSON values must be timezone-aware")
        return {
            _TYPE_TAG: "datetime",
            _TYPE_VALUE: value.astimezone(timezone.utc).isoformat(),
        }
    if isinstance(value, date):
        return {_TYPE_TAG: "date", _TYPE_VALUE: value.isoformat()}
    if isinstance(value, FrozenDict):
        return {
            _TYPE_TAG: "frozen_dict",
            _TYPE_VALUE: {
                str(name): _encode_typed(item)
                for name, item in value.items()
            },
        }
    if isinstance(value, tuple):
        return {
            _TYPE_TAG: "tuple",
            _TYPE_VALUE: [_encode_typed(item) for item in value],
        }
    if isinstance(value, frozenset):
        encoded_items = [_encode_typed(item) for item in value]
        encoded_items.sort(
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return {_TYPE_TAG: "frozenset", _TYPE_VALUE: encoded_items}
    if isinstance(value, Mapping):
        return {
            str(name): _encode_typed(item)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_encode_typed(item) for item in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"Unsupported tagged JSON value: {type(value).__name__}")


def _decode_typed(value: object) -> Any:
    if isinstance(value, list):
        return [_decode_typed(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    if set(value) == {_TYPE_TAG, _TYPE_VALUE}:
        tag = value.get(_TYPE_TAG)
        raw = value.get(_TYPE_VALUE)
        if tag == "decimal" and isinstance(raw, str):
            parsed = Decimal(raw)
            if not parsed.is_finite():
                raise ValueError("Decimal JSON values must be finite")
            return parsed
        if tag == "date" and isinstance(raw, str):
            return date.fromisoformat(raw)
        if tag == "datetime" and isinstance(raw, str):
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                raise ValueError("datetime JSON values must be timezone-aware")
            return parsed.astimezone(timezone.utc)
        if tag in {"tuple", "frozenset"} and isinstance(raw, list):
            items = tuple(_decode_typed(item) for item in raw)
            return items if tag == "tuple" else frozenset(items)
        if tag == "frozen_dict" and isinstance(raw, Mapping):
            return FrozenDict(
                {
                    str(name): _decode_typed(item)
                    for name, item in raw.items()
                }
            )
        raise ValueError("Tagged JSON value has an invalid type payload")
    return {
        str(name): _decode_typed(item)
        for name, item in value.items()
    }


def _canonical_typed_json(value: object) -> str:
    return json.dumps(
        _encode_typed(_decode_typed(value)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_equal(left: object, right: object) -> bool:
    try:
        return _canonical_typed_json(left) == _canonical_typed_json(right)
    except (TypeError, ValueError):
        return False


def _value_digest(value: object) -> str:
    return hashlib.sha256(
        _canonical_typed_json(value).encode("utf-8")
    ).hexdigest()


def _iteration_metrics_payload(
    iteration: StepAudit,
) -> dict[str, object]:
    return {
        "accepted": iteration.accepted,
        "current": {
            name: metric.to_dict()
            for name, metric in iteration.current_window_metrics.items()
        },
        "candidate": {
            name: metric.to_dict()
            for name, metric in iteration.candidate_window_metrics.items()
        },
    }


def _metrics_from_state(state: WorkState) -> ModelMetrics:
    accepted = sum(audit.accepted for audit in state.audits)
    latest = state.audits[-1] if state.audits else None
    windows: dict[str, object] = (
        {}
        if latest is None
        else {
            name: metric.to_dict()
            for name, metric in latest.current_window_metrics.items()
        }
    )
    return ModelMetrics(
        symbol=state.instrument_code,
        model_version=state.current_model_version,
        iteration_count=len(state.audits),
        last_iteration=_version_number(
            state.iteration_id, "I", allow_zero=True
        ),
        accepted_iterations=accepted,
        rejected_iterations=len(state.audits) - accepted,
        feedback_count=0 if latest is None else latest.feedback_count,
        mature_count=0 if latest is None else latest.mature_count,
        partial_count=0 if latest is None else latest.partial_count,
        pending_count=0 if latest is None else latest.pending_count,
        candidate_acceptance_rate=(
            Decimal("0")
            if latest is None
            else latest.candidate_acceptance_rate
        ),
        windows=FrozenDict(windows),
        curve=_metrics_curve_from_state(state),
    )


def _metrics_curve_from_state(
    state: WorkState,
) -> tuple[FrozenDict[str, object], ...]:
    labels = {label.iteration_id: label for label in state.feedback}
    absolute_deviations: list[Decimal] = []
    curve: list[FrozenDict[str, object]] = []
    for iteration_number, audit in enumerate(state.audits, start=1):
        label = labels.get(audit.iteration_id)
        deviation = (
            None
            if label is None or label.status != "mature_13w"
            else label.turning_deviation_sessions
        )
        absolute = (
            None if deviation is None else Decimal(abs(deviation))
        )
        if absolute is not None:
            absolute_deviations.append(absolute)
        curve.append(
            FrozenDict(
                {
                    "iteration_number": iteration_number,
                    "iteration_id": audit.iteration_id,
                    "cutoff_date": audit.cutoff_date,
                    "model_version": audit.model_version,
                    "accepted": audit.accepted,
                    "deviation_days": deviation,
                    "absolute_deviation": absolute,
                    "rolling_20_abs_deviation": _rolling_mean(
                        absolute_deviations, 20
                    ),
                    "rolling_52_abs_deviation": _rolling_mean(
                        absolute_deviations, 52
                    ),
                    "direction_correct": (
                        None if label is None else label.direction_hit
                    ),
                }
            )
        )
    return tuple(curve)


def _rolling_mean(
    values: list[Decimal],
    window: int,
) -> Decimal | None:
    if len(values) < window:
        return None
    selected = values[-window:]
    return sum(selected, Decimal()) / Decimal(window)


def _model_metrics_payload(iteration: StepAudit) -> dict[str, object]:
    return {
        "accepted": iteration.accepted,
        "candidate_acceptance_rate": (
            iteration.candidate_acceptance_rate
        ),
        "feedback_count": iteration.feedback_count,
        "mature_count": iteration.mature_count,
        "partial_count": iteration.partial_count,
        "pending_count": iteration.pending_count,
        "windows": {
            name: metric.to_dict()
            for name, metric in iteration.current_window_metrics.items()
        },
    }


def _model_audit_payload(
    iteration: StepAudit,
    *,
    parent_model_version: str | None,
) -> dict[str, object]:
    return {
        "parent_model_version": parent_model_version,
        "work_version": iteration.work_version,
        "iteration_id": iteration.iteration_id,
        "accepted": iteration.accepted,
        "source_hash": iteration.source_hash,
    }


def _hashed_snapshot(value: Mapping[str, object]) -> dict[str, object]:
    source = dict(value)
    declared_codec = source.setdefault("codec_version", _CODEC_VERSION)
    if declared_codec != _CODEC_VERSION:
        raise ValueError("snapshot codec_version is invalid")
    body = _json_mapping(source)
    if "snapshot_hash" in body:
        raise ValueError("snapshot body must not predeclare snapshot_hash")
    body["snapshot_hash"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return body


def _snapshot_digest(value: Mapping[str, object]) -> str:
    body = {
        key: item for key, item in value.items() if key != "snapshot_hash"
    }
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _snapshot_bytes(snapshot: Mapping[str, object]) -> bytes:
    digest = snapshot.get("snapshot_hash")
    if (
        not isinstance(digest, str)
        or _HASH_PATTERN.fullmatch(digest) is None
        or _snapshot_digest(snapshot) != digest
    ):
        raise ValueError("snapshot hash is missing or invalid")
    return (
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _snapshot_relative_path(
    symbol: str,
    iteration_id: str,
    generated_at: datetime,
    kind: str,
) -> Path:
    _validate_symbol(symbol)
    _version_number(iteration_id, "I", allow_zero=False)
    timestamp = generated_at.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    return (
        Path(symbol)
        / iteration_id
        / f"{timestamp}-{kind}-{uuid4().hex}.json"
    )


def _write_snapshot_temp(final_path: Path, payload: bytes) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    )
    temp_path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return temp_path
    except Exception:
        _cleanup_artifact(temp_path)
        raise


def _replace_snapshot(temp_path: Path, final_path: Path) -> None:
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite {final_path}")
    os.replace(temp_path, final_path)


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_artifact(path: Path | None) -> Path | None:
    """Remove an artifact or atomically move it outside reader namespaces."""

    if path is None:
        return None
    _safe_unlink(path)
    if not path.exists():
        return None
    quarantine = path.with_suffix(path.suffix + ".orphan")
    if quarantine.exists():
        quarantine = path.with_suffix(
            path.suffix + f".{uuid4().hex}.orphan"
        )
    try:
        os.replace(path, quarantine)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RepositoryError(
            f"Could not remove or quarantine audit artifact {path}"
        ) from exc
    return quarantine


def _add_secondary_note(
    primary: BaseException,
    operation: str,
    secondary: BaseException,
) -> None:
    primary.add_note(
        f"Secondary {operation} failure: "
        f"{type(secondary).__name__}: {secondary}"
    )


def _attempt_secondary(
    primary: BaseException,
    operation: str,
    action: Any,
) -> None:
    try:
        action()
    except BaseException as secondary:
        _add_secondary_note(primary, operation, secondary)


def _restore_numbers(value: object) -> Any:
    """Decode explicitly tagged values without guessing numeric strings."""

    return _decode_typed(value)


def _parse_date(value: object, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date")
    return date.fromisoformat(value)


def _parse_datetime(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{field_name} must be timezone-aware")
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO datetime")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _parse_optional_date(
    value: object, field_name: str
) -> date | None:
    return None if value is None else _parse_date(value, field_name)


def _pid_is_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _window_metric_from_dict(value: Mapping[str, object]) -> WindowMetrics:
    return WindowMetrics(
        instrument_code=str(value["instrument_code"]),
        window=str(value["window"]),
        sample_count=_strict_int(value["sample_count"], "sample_count"),
        total_loss=_decimal(value["total_loss"], "total_loss"),
        direction_hit_rate=_decimal(
            value["direction_hit_rate"], "direction_hit_rate"
        ),
        calibration_loss=_decimal(
            value["calibration_loss"], "calibration_loss"
        ),
        overtrade_penalty=_decimal(
            value["overtrade_penalty"], "overtrade_penalty"
        ),
    )


def _label_from_dict(value: Mapping[str, object]) -> IterationLabel:
    return IterationLabel(
        iteration_id=str(value["iteration_id"]),
        instrument_code=str(value["instrument_code"]),
        week_key=str(value["week_key"]),
        cutoff_date=_parse_date(value["cutoff_date"], "cutoff_date"),
        predicted_direction=str(value["predicted_direction"]),
        operation_side=str(value["operation_side"]),
        probability=_decimal(value["probability"], "probability"),
        predicted_action_date=_parse_date(
            value["predicted_action_date"], "predicted_action_date"
        ),
        target_position=_decimal(
            value["target_position"], "target_position"
        ),
        trade_count=_strict_int(value["trade_count"], "trade_count"),
        status=str(value["status"]),  # type: ignore[arg-type]
        observed_through=_parse_date(
            value["observed_through"], "observed_through"
        ),
        visible_row_count=_strict_int(
            value["visible_row_count"], "visible_row_count"
        ),
        partial_weight=_decimal(
            value["partial_weight"], "partial_weight"
        ),
        actual_direction=_optional_string(value.get("actual_direction")),
        actual_turn_date=_parse_optional_date(
            value.get("actual_turn_date"), "actual_turn_date"
        ),
        turning_deviation_sessions=(
            None
            if value.get("turning_deviation_sessions") is None
            else _strict_int(
                value["turning_deviation_sessions"],
                "turning_deviation_sessions",
                minimum=-10_000,
            )
        ),
        loss_components=FrozenDict(
            _decimal_mapping(
                _mapping(value["loss_components"], "loss_components")
            )
        ),
        total_loss=(
            None
            if value.get("total_loss") is None
            else _decimal(value["total_loss"], "total_loss")
        ),
        direction_hit=(
            None
            if value.get("direction_hit") is None
            else _strict_bool(value["direction_hit"], "direction_hit")
        ),
        calibration_formula=str(value["calibration_formula"]),
        turning_deviation_formula=str(
            value["turning_deviation_formula"]
        ),
    )


def _step_audit_from_dict(value: Mapping[str, object]) -> StepAudit:
    return StepAudit(
        instrument_code=str(value["instrument_code"]),
        iteration_id=str(value["iteration_id"]),
        work_version=str(value["work_version"]),
        parent_work_version=str(value["parent_work_version"]),
        model_version=str(value["model_version"]),
        parent_model_version=str(value["parent_model_version"]),
        week_key=str(value["week_key"]),
        cutoff_date=_parse_date(value["cutoff_date"], "cutoff_date"),
        source_data_max_date=_parse_optional_date(
            value.get("source_data_max_date"), "source_data_max_date"
        ),
        base_weights=FrozenDict(
            _decimal_mapping(_mapping(value["base_weights"], "base_weights"))
        ),
        candidate_weights=FrozenDict(
            _decimal_mapping(
                _mapping(value["candidate_weights"], "candidate_weights")
            )
        ),
        accepted=_strict_bool(value["accepted"], "accepted"),
        reasons=tuple(str(item) for item in _sequence(value["reasons"])),
        optimizer_memory=FrozenDict(
            _restore_numbers(
                _mapping(value["optimizer_memory"], "optimizer_memory")
            )
        ),
        feedback_count=_strict_int(
            value["feedback_count"], "feedback_count"
        ),
        mature_count=_strict_int(value["mature_count"], "mature_count"),
        partial_count=_strict_int(value["partial_count"], "partial_count"),
        pending_count=_strict_int(value["pending_count"], "pending_count"),
        current_window_metrics=FrozenDict(
            {
                str(name): _window_metric_from_dict(
                    _mapping(item, f"current_window_metrics:{name}")
                )
                for name, item in _mapping(
                    value["current_window_metrics"],
                    "current_window_metrics",
                ).items()
            }
        ),
        candidate_window_metrics=FrozenDict(
            {
                str(name): _window_metric_from_dict(
                    _mapping(item, f"candidate_window_metrics:{name}")
                )
                for name, item in _mapping(
                    value["candidate_window_metrics"],
                    "candidate_window_metrics",
                ).items()
            }
        ),
        candidate_acceptance_rate=_decimal(
            value["candidate_acceptance_rate"],
            "candidate_acceptance_rate",
        ),
        rejected_candidates=tuple(
            _restore_numbers(item)
            for item in _sequence(value["rejected_candidates"])
        ),
        source_hash=str(value["source_hash"]),
        algorithm_version=str(value["algorithm_version"]),
        loss_version=str(value["loss_version"]),
        loss_weights=FrozenDict(
            _decimal_mapping(
                _mapping(value["loss_weights"], "loss_weights")
            )
        ),
        input_hash=str(value["input_hash"]),
        data_integrity=_strict_bool(
            value["data_integrity"], "data_integrity"
        ),
        future_leakage=_strict_bool(
            value["future_leakage"], "future_leakage"
        ),
        active_windows=tuple(
            str(item) for item in _sequence(value["active_windows"])
        ),
        gate_parameters=FrozenDict(
            _decimal_mapping(
                _mapping(value["gate_parameters"], "gate_parameters")
            )
        ),
    )


def _stored_audit_full_hash(value: Mapping[str, object]) -> str | None:
    if value.get("_storage_format") == _STEP_AUDIT_STORAGE_FORMAT:
        digest = value.get("_full_audit_hash")
        return (
            digest
            if isinstance(digest, str)
            and _HASH_PATTERN.fullmatch(digest) is not None
            else None
        )
    return _value_digest(value)


def _stored_audits_equivalent(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    if _canonical_equal(left, right):
        return True
    left_hash = _stored_audit_full_hash(left)
    right_hash = _stored_audit_full_hash(right)
    return left_hash is not None and left_hash == right_hash


def _audit_storage_cache_key(
    audit: StepAudit,
) -> tuple[str, str, str, str, str]:
    return (
        audit.instrument_code,
        audit.iteration_id,
        audit.work_version,
        audit.source_hash,
        audit.input_hash,
    )


def _step_audit_storage_dict(
    audit: StepAudit,
    previous_audit: StepAudit | None,
) -> dict[str, object]:
    """Encode one audit without repeating cumulative optimizer sequences."""

    raw = audit.to_dict()
    memory = dict(audit.optimizer_memory)
    previous_memory = (
        {} if previous_audit is None else previous_audit.optimizer_memory
    )
    sequence_deltas: dict[str, object] = {}
    for field_name in ("rejected_candidates", "seen_feedback_ids"):
        current = tuple(memory.pop(field_name, ()))
        previous = tuple(previous_memory.get(field_name, ()))
        extends_previous = (
            len(current) >= len(previous)
            and current[: len(previous)] == previous
        )
        sequence_deltas[field_name] = {
            "reset": not extends_previous,
            "items": current if not extends_previous else current[len(previous) :],
        }
    raw["optimizer_memory"] = memory
    raw.pop("rejected_candidates", None)
    raw["_storage_format"] = _STEP_AUDIT_STORAGE_FORMAT
    raw["_optimizer_memory_sequences"] = sequence_deltas
    raw["_full_audit_hash"] = _value_digest(audit.to_dict())
    return raw


def _step_audit_from_storage_dict(
    value: Mapping[str, object],
    previous_audit: StepAudit | None,
) -> StepAudit:
    """Decode either a legacy full audit or the compact delta format."""

    if value.get("_storage_format") != _STEP_AUDIT_STORAGE_FORMAT:
        return _step_audit_from_dict(value)
    full_hash = value.get("_full_audit_hash")
    if not isinstance(full_hash, str) or _HASH_PATTERN.fullmatch(full_hash) is None:
        raise ValueError("compact audit full hash is invalid")
    sequences = _mapping(
        value.get("_optimizer_memory_sequences"),
        "_optimizer_memory_sequences",
    )
    memory = dict(
        _restore_numbers(
            _mapping(value.get("optimizer_memory"), "optimizer_memory")
        )
    )
    previous_memory = (
        {} if previous_audit is None else previous_audit.optimizer_memory
    )
    for field_name in ("rejected_candidates", "seen_feedback_ids"):
        delta = _mapping(sequences.get(field_name), field_name)
        reset = _strict_bool(delta.get("reset"), f"{field_name}.reset")
        items = tuple(_restore_numbers(item) for item in _sequence(delta.get("items")))
        prefix = () if reset else tuple(previous_memory.get(field_name, ()))
        memory[field_name] = prefix + items
    expanded = {
        key: item
        for key, item in value.items()
        if key
        not in {
            "_storage_format",
            "_optimizer_memory_sequences",
            "_full_audit_hash",
        }
    }
    expanded["optimizer_memory"] = memory
    expanded["rejected_candidates"] = memory["rejected_candidates"]
    audit = _step_audit_from_dict(expanded)
    if _value_digest(audit.to_dict()) != full_hash:
        raise ValueError("compact audit does not match its full audit hash")
    return audit


def _work_state_storage_dict(
    state: WorkState,
    *,
    audit_cache: dict[
        tuple[str, str, str, str, str],
        dict[str, object],
    ]
    | None = None,
) -> dict[str, object]:
    """Persist a complete W state with incremental audit history."""

    audits: list[dict[str, object]] = []
    week_records: dict[str, object] = {}
    previous: StepAudit | None = None
    for index, audit in enumerate(state.audits):
        stored = (
            None
            if audit_cache is None
            else audit_cache.get(_audit_storage_cache_key(audit))
        )
        if stored is None:
            stored = _step_audit_storage_dict(audit, previous)
            if audit_cache is not None:
                audit_cache[_audit_storage_cache_key(audit)] = stored
        audits.append(stored)
        week_records[audit.week_key] = {
            "audit_index": index,
            "iteration_id": audit.iteration_id,
            "data_integrity": audit.data_integrity,
        }
        previous = audit
    return {
        "_storage_format": _WORK_STATE_STORAGE_FORMAT,
        "audits": tuple(audits),
        "current_model_version": state.current_model_version,
        "current_model_weights": dict(state.current_model_weights),
        "feedback": tuple(label.to_dict() for label in state.feedback),
        "instrument_code": state.instrument_code,
        "iteration_id": state.iteration_id,
        # The final memory is retained in full so W remains independently
        # inspectable; only its repeated copies inside historical audits are
        # delta encoded.
        "optimizer_memory": dict(state.optimizer_memory),
        "parent_work_version": state.parent_work_version,
        "week_records": week_records,
        "work_version": state.work_version,
    }


def _work_state_from_dict(value: Mapping[str, object]) -> WorkState:
    audit_values = tuple(
        _mapping(item, "audit")
        for item in _sequence(value["audits"])
    )
    compact = value.get("_storage_format") == _WORK_STATE_STORAGE_FORMAT
    if compact:
        decoded_audits: list[StepAudit] = []
        previous: StepAudit | None = None
        for item in audit_values:
            audit = _step_audit_from_storage_dict(item, previous)
            decoded_audits.append(audit)
            previous = audit
        audits = tuple(decoded_audits)
    else:
        audits = tuple(
            _step_audit_from_dict(item) for item in audit_values
        )
    feedback = tuple(
        _label_from_dict(_mapping(item, "feedback"))
        for item in _sequence(value["feedback"])
    )
    records_raw = _mapping(value["week_records"], "week_records")
    audits_by_week = {
        audit.week_key: (audit, raw)
        for audit, raw in zip(audits, audit_values)
    }
    audit_positions = {
        audit.week_key: index for index, audit in enumerate(audits)
    }
    if set(records_raw) != set(audits_by_week):
        raise ValueError("week_records must match audit week identities")
    if compact:
        for week, raw in records_raw.items():
            reference = _mapping(raw, f"week_record:{week}")
            audit = audits_by_week[str(week)][0]
            if (
                _strict_int(
                    reference.get("audit_index"),
                    f"week_record:{week}.audit_index",
                )
                != audit_positions[audit.week_key]
                or reference.get("iteration_id") != audit.iteration_id
                or _strict_bool(
                    reference.get("data_integrity"),
                    f"week_record:{week}.data_integrity",
                )
                != audit.data_integrity
            ):
                raise ValueError("week_records must mirror canonical audits")
    else:
        for week, raw in records_raw.items():
            if not _canonical_equal(
                _mapping(raw, f"week_record:{week}"),
                audits_by_week[str(week)][1],
            ):
                raise ValueError("week_records must mirror canonical audits")
    records = FrozenDict(
        {
            str(week): audits_by_week[str(week)][0]
            for week in records_raw
        }
    )
    state = WorkState(
        instrument_code=str(value["instrument_code"]),
        iteration_id=str(value["iteration_id"]),
        work_version=str(value["work_version"]),
        parent_work_version=_optional_string(
            value.get("parent_work_version")
        ),
        current_model_version=str(value["current_model_version"]),
        current_model_weights=FrozenDict(
            _decimal_mapping(
                _mapping(
                    value["current_model_weights"],
                    "current_model_weights",
                )
            )
        ),
        optimizer_memory=FrozenDict(
            _restore_numbers(
                _mapping(value["optimizer_memory"], "optimizer_memory")
            )
        ),
        feedback=feedback,
        audits=audits,
        week_records=records,
    )
    if compact and audits and state.optimizer_memory != audits[-1].optimizer_memory:
        raise ValueError("compact work state optimizer memory is inconsistent")
    return state


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value  # type: ignore[return-value]


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("value must be a sequence")
    return tuple(value)


def _coerce_checkpoint(
    task_id: str,
    progress: TaskCheckpoint | Mapping[str, object],
) -> TaskCheckpoint:
    if isinstance(progress, TaskCheckpoint):
        if progress.task_id != task_id:
            raise ValueError("task_id does not match checkpoint")
        return progress
    if not isinstance(progress, Mapping):
        raise TypeError("progress must be a mapping or TaskCheckpoint")
    symbol_value = progress.get("symbol", progress.get("instrument_code"))
    if not isinstance(symbol_value, str):
        raise ValueError("checkpoint progress requires symbol")
    symbol = _validate_symbol(symbol_value)
    completed = _strict_int(
        progress.get("completed_weeks", 0), "completed_weeks"
    )
    last_iteration = _strict_int(
        progress.get("last_iteration", completed), "last_iteration"
    )
    status = progress.get("status", "queued")
    if not isinstance(status, str):
        raise ValueError("checkpoint status must be a string")
    total_value = progress.get("total_weeks")
    total_weeks = (
        None
        if total_value is None
        else _strict_int(total_value, "total_weeks")
    )
    last_work_version = progress.get(
        "last_work_version", f"W{last_iteration:04d}"
    )
    last_model_version = progress.get(
        "last_model_version", "M0001"
    )
    if not isinstance(last_work_version, str) or not isinstance(
        last_model_version, str
    ):
        raise ValueError("checkpoint W/M versions must be strings")
    return TaskCheckpoint(
        task_id=task_id,
        symbol=symbol,
        status=status,
        completed_weeks=completed,
        total_weeks=total_weeks,
        last_iteration=last_iteration,
        last_work_version=last_work_version,
        last_model_version=last_model_version,
        last_completed_week=_optional_string(
            progress.get("last_completed_week")
        ),
        progress=FrozenDict(
            {
                str(name): _restore_numbers(value)
                for name, value in progress.items()
                if name
                not in {
                    "task_id",
                    "symbol",
                    "instrument_code",
                    "status",
                    "completed_weeks",
                    "total_weeks",
                    "last_iteration",
                    "last_work_version",
                    "last_model_version",
                    "last_completed_week",
                    "progress",
                }
            }
            | (
                dict(
                    _mapping(progress["progress"], "progress")
                )
                if "progress" in progress
                else {}
            )
        ),
    )


def _checkpoint_from_mapping(
    task_id: str,
    symbol: str,
    status: str,
    payload: Mapping[str, object],
) -> TaskCheckpoint:
    if (
        payload.get("task_id") != task_id
        or payload.get("symbol") != symbol
        or payload.get("status") != status
    ):
        raise ValueError(
            "checkpoint payload identity/status disagrees with DB row"
        )
    return _coerce_checkpoint(task_id, payload)


def _checkpoint_error_message(checkpoint: TaskCheckpoint) -> str | None:
    value = (
        checkpoint.progress.get("failure_reason")
        if checkpoint.status == "failed"
        else checkpoint.progress.get("recoverable_reason")
        if checkpoint.status == "recoverable"
        else None
    )
    return value if isinstance(value, str) and value else None


__all__ = [
    "AdviceRecord",
    "CheckpointNotFound",
    "CrossMarketRepositoryError",
    "DuplicateAdvice",
    "DuplicateWeek",
    "ModelMetrics",
    "ModelVersion",
    "RepositoryCorruption",
    "RepositoryError",
    "TaskCheckpoint",
    "UnknownMarket",
    "WeeklyAnalysisRepository",
]
