from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib
from pathlib import Path
import subprocess
import shutil
import sys
from threading import Barrier, Event, Lock, Thread, enumerate as enumerate_threads
import time
from typing import Any

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError

from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    Base,
    Instrument,
    V2AdviceHistory,
    V2AnalysisIteration,
    V2AnalysisTask,
    V2IterationLabel,
    V2ModelVersion,
    V2WeekSample,
)
from backend.app.weekly_analysis.optimizer import (
    IterationPrediction,
    run_optimizer_step,
    seed_model,
)


def _repository_api() -> Any:
    return importlib.import_module("backend.app.weekly_analysis.repository")


def _tasks_api() -> Any:
    return importlib.import_module("backend.app.weekly_analysis.tasks")


def _repository(tmp_path: Path):
    module = _repository_api()
    sessions = create_session_factory(tmp_path / "tasks.db")
    Base.metadata.create_all(
        sessions.kw["bind"],
        tables=(
            Instrument.__table__,
            V2WeekSample.__table__,
            V2AnalysisIteration.__table__,
            V2IterationLabel.__table__,
            V2ModelVersion.__table__,
            V2AnalysisTask.__table__,
            V2AdviceHistory.__table__,
        ),
    )
    with sessions.begin() as session:
        session.add_all(
            (
                Instrument(
                    code="399006",
                    name="Growth",
                    exchange="SZSE",
                    category="index",
                ),
                Instrument(
                    code="NDX",
                    name="Nasdaq 100",
                    exchange="NASDAQ",
                    category="index",
                ),
            )
        )
    return (
        sessions,
        module.WeeklyAnalysisRepository(
            sessions,
            audit_root=tmp_path / "weekly-analysis-v2",
        ),
    )


class _BlockingEngine:
    def __init__(self, participants: int = 1) -> None:
        self.calls: list[tuple[str, date | None]] = []
        self.started = Event()
        self.both_started = Event()
        self.release = Event()
        self.barrier = Barrier(participants) if participants > 1 else None
        self.lock = Lock()
        self.active: set[str] = set()
        self.parallel_markets = False

    def run(
        self,
        symbol: str,
        *,
        as_of=None,
        progress_callback=None,
        advice_request_key=None,
        analysis_request=None,
        execution_guard=None,
    ):
        with self.lock:
            self.calls.append((symbol, as_of))
            self.active.add(symbol)
            self.parallel_markets = len(self.active) > 1
            if self.parallel_markets:
                self.both_started.set()
        self.started.set()
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "iterating",
                    "completed_weeks": 0,
                    "total_weeks": 1,
                    "last_checkpoint": None,
                    "elapsed_seconds": 0.01,
                }
            )
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if not self.release.wait(timeout=10):
            raise TimeoutError("test did not release engine")
        with self.lock:
            self.active.remove(symbol)
        return {"symbol": symbol}


class _SilentBlockingEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = Event()
        self.release = Event()
        self.lock = Lock()

    def run(
        self,
        symbol: str,
        *,
        as_of=None,
        progress_callback=None,
        advice_request_key=None,
        analysis_request=None,
        execution_guard=None,
    ):
        with self.lock:
            self.calls.append(symbol)
        self.started.set()
        if not self.release.wait(timeout=10):
            raise TimeoutError("test did not release silent engine")
        return {"symbol": symbol}


@pytest.mark.parametrize(
    "python_command",
    ((sys.executable,), ("py", "-3.14")),
    ids=("project-venv", "system-python-3.14"),
)
def test_daemon_executor_submit_shutdown_smoke(
    python_command: tuple[str, ...],
) -> None:
    if python_command[0] == "py" and shutil.which("py") is None:
        pytest.skip("Windows Python launcher is not installed")
    script = """
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread
from backend.app.daemon_executor import DaemonThreadPoolExecutor

executor = DaemonThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="weekly-analysis-v2-smoke",
)
assert isinstance(executor, ThreadPoolExecutor)
assert executor._max_workers == 2
daemon, name = executor.submit(
    lambda: (current_thread().daemon, current_thread().name)
).result(timeout=5)
workers = tuple(executor._threads)
executor.shutdown(wait=True, cancel_futures=True)
assert daemon is True
assert name.startswith("weekly-analysis-v2-smoke")
assert workers
assert all(not worker.is_alive() for worker in workers)
print("OK")
"""
    completed = subprocess.run(
        [*python_command, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "OK"


def test_daemon_executor_does_not_join_silent_worker_at_process_exit() -> None:
    script = """
import sys
import time
from backend.app.weekly_analysis.tasks import DaemonThreadPoolExecutor

executor = DaemonThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="weekly-analysis-v2-subprocess",
)
executor.submit(time.sleep, 2.5)
time.sleep(0.05)
print("READY", flush=True)
executor.shutdown(wait=False, cancel_futures=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    started = time.perf_counter()
    try:
        returncode = process.wait(timeout=1)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    elapsed = time.perf_counter() - started
    stderr = "" if process.stderr is None else process.stderr.read()

    assert returncode == 0, stderr
    assert elapsed < 1.0


def test_shutdown_without_wait_releases_claim_and_fences_old_worker(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)
    engine = _SilentBlockingEngine()
    manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
        claim_lease_seconds=5,
    )
    submitted = manager.submit(
        "399006",
        latest_complete_week="2026-W30",
        as_of=date(2026, 7, 30),
    )
    assert engine.started.wait(timeout=10)

    started = time.perf_counter()
    manager.shutdown(wait=False)
    elapsed = time.perf_counter() - started
    released = repository.load_task_checkpoint(submitted.task_id)
    with repository._sessions() as session:
        row = session.scalar(
            select(V2AnalysisTask).where(
                V2AnalysisTask.task_key == submitted.task_id
            )
        )
        assert row is not None
        assert row.worker_token is None
        assert row.lease_expires_at is None

    assert elapsed < 1.0
    assert released.status == "recoverable"
    assert released.progress["recoverable_reason"] == (
        "Weekly analysis service shut down before completion"
    )

    engine.release.set()
    future = manager._futures[submitted.task_id]
    future.result(timeout=10)
    assert repository.load_task_checkpoint(submitted.task_id).status == (
        "recoverable"
    )


def test_pre_stopped_heartbeat_always_signals_ready() -> None:
    tasks = _tasks_api()
    heartbeat_stop = Event()
    claim_lost = Event()
    heartbeat_ready = Event()
    heartbeat_stop.set()

    class NoRenewRepository:
        def renew_task_execution_claim(self, *args, **kwargs):
            raise AssertionError("pre-stopped heartbeat must not renew")

    tasks._renew_claim_until_stopped(
        NoRenewRepository(),
        "weekly-v2-" + "a" * 32,
        "worker",
        1.0,
        heartbeat_stop,
        claim_lost,
        heartbeat_ready,
    )

    assert heartbeat_ready.is_set()


def test_execute_heartbeat_ready_wait_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = _tasks_api()
    _sessions, repository = _repository(tmp_path)
    engine = _SilentBlockingEngine()
    heartbeat_returned = Event()
    captured_events: dict[str, Event] = {}

    def exit_without_signalling_ready(
        _repository,
        _task_id,
        _worker_token,
        _lease_seconds,
        heartbeat_stop,
        claim_lost,
        heartbeat_ready,
        claim_io_lock=None,
    ) -> None:
        captured_events["stop"] = heartbeat_stop
        captured_events["lost"] = claim_lost
        captured_events["ready"] = heartbeat_ready
        heartbeat_returned.set()

    monkeypatch.setattr(
        tasks,
        "_renew_claim_until_stopped",
        exit_without_signalling_ready,
    )
    monkeypatch.setattr(
        tasks,
        "_HEARTBEAT_READY_TIMEOUT_SECONDS",
        0.05,
        raising=False,
    )
    manager = tasks.WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
    )
    submitted = manager.submit(
        "399006",
        latest_complete_week="2026-W30",
        as_of=date(2026, 7, 30),
    )
    future = manager._futures[submitted.task_id]
    try:
        assert heartbeat_returned.wait(timeout=5)
        result = future.result(timeout=0.5)
        assert isinstance(result, tasks._TaskClaimLost)
        assert engine.calls == []
    finally:
        captured_events.get("lost", Event()).set()
        captured_events.get("ready", Event()).set()
        manager.shutdown(wait=True)


@pytest.mark.parametrize("wait", (False, True))
def test_shutdown_before_execute_respects_queue_policy_and_leaks_nothing(
    tmp_path: Path,
    wait: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = _tasks_api()
    _sessions, repository = _repository(tmp_path)
    engine = _SilentBlockingEngine()
    manager = tasks.WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
    )
    blockers_ready = Barrier(3)
    release_blockers = Event()
    heartbeat_called = Event()
    captured_events: dict[str, Event] = {}
    real_heartbeat = tasks._renew_claim_until_stopped

    def capture_heartbeat_events(
        heartbeat_repository,
        task_id,
        worker_token,
        lease_seconds,
        heartbeat_stop,
        claim_lost,
        heartbeat_ready,
        claim_io_lock=None,
    ) -> None:
        captured_events["lost"] = claim_lost
        captured_events["ready"] = heartbeat_ready
        heartbeat_called.set()
        real_heartbeat(
            heartbeat_repository,
            task_id,
            worker_token,
            lease_seconds,
            heartbeat_stop,
            claim_lost,
            heartbeat_ready,
            claim_io_lock,
        )

    monkeypatch.setattr(
        tasks,
        "_renew_claim_until_stopped",
        capture_heartbeat_events,
    )

    def occupy_worker() -> None:
        blockers_ready.wait(timeout=5)
        if not release_blockers.wait(timeout=5):
            raise TimeoutError("test did not release executor blocker")

    blockers = tuple(
        manager._executor.submit(occupy_worker)
        for _index in range(2)
    )
    blockers_ready.wait(timeout=5)
    submitted = manager.submit(
        "399006",
        latest_complete_week="2026-W30",
        as_of=date(2026, 7, 30),
    )
    future = manager._futures[submitted.task_id]
    shutdown_errors: list[BaseException] = []
    shutdown_calls: list[tuple[bool, bool]] = []
    shutdown_called = Event()
    real_executor_shutdown = manager._executor.shutdown

    def observed_executor_shutdown(
        *,
        wait: bool,
        cancel_futures: bool = False,
    ) -> None:
        shutdown_calls.append((wait, cancel_futures))
        shutdown_called.set()
        real_executor_shutdown(
            wait=wait,
            cancel_futures=cancel_futures,
        )

    monkeypatch.setattr(
        manager._executor,
        "shutdown",
        observed_executor_shutdown,
    )

    def shut_down() -> None:
        try:
            manager.shutdown(
                wait=wait,
                cancel_futures=False,
            )
        except BaseException as error:
            shutdown_errors.append(error)

    shutdown_thread = Thread(target=shut_down)
    shutdown_thread.start()
    assert shutdown_called.wait(timeout=5)
    if not wait:
        shutdown_thread.join(timeout=5)
    release_blockers.set()
    if wait:
        shutdown_thread.join(timeout=5)
    for blocker in blockers:
        blocker.result(timeout=5)
    completed_without_rescue = future.done()
    if not completed_without_rescue and heartbeat_called.wait(timeout=1):
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and not future.done():
            time.sleep(0.01)
        completed_without_rescue = future.done()
    if not future.done():
        captured_events["lost"].set()
        captured_events["ready"].set()
    shutdown_thread.join(timeout=5)
    workers = tuple(manager._executor._threads)
    for worker in workers:
        worker.join(timeout=5)

    assert not shutdown_thread.is_alive()
    assert shutdown_errors == []
    assert shutdown_calls == [(wait, False)]
    assert completed_without_rescue
    assert not future.cancelled()
    result = future.result(timeout=5)
    assert isinstance(result, tasks._TaskClaimLost)
    assert engine.calls == []
    assert all(not worker.is_alive() for worker in workers)
    heartbeat_name = (
        f"weekly-analysis-v2-heartbeat-{submitted.task_id}"
    )
    assert not any(
        thread.name == heartbeat_name and thread.is_alive()
        for thread in enumerate_threads()
    )
    assert repository.load_task_checkpoint(submitted.task_id).status == (
        "recoverable"
    )


def test_same_worker_heartbeat_between_read_and_progress_cas_is_allowed(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)
    worker_token = "same-worker"
    checkpoint, _created = repository.get_or_create_task(
        "399006",
        "same-worker-heartbeat-cas",
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
    )
    claimed = repository.claim_task_execution(
        checkpoint.task_id,
        worker_token=worker_token,
        lease_seconds=5,
    )
    assert claimed is not None

    transition_update_ready = Event()
    allow_transition_update = Event()
    transition_results: list[object] = []
    transition_errors: list[BaseException] = []
    engine = _sessions.kw["bind"]

    def pause_progress_update(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = statement.lower()
        if (
            normalized.lstrip().startswith("update v2_analysis_tasks")
            and "result_payload" in normalized
            and not transition_update_ready.is_set()
        ):
            transition_update_ready.set()
            if not allow_transition_update.wait(timeout=5):
                raise TimeoutError(
                    "test did not release progress checkpoint update"
                )

    def transition() -> None:
        try:
            transition_results.append(
                repository.transition_task_checkpoint(
                    checkpoint.task_id,
                    status="building_features",
                    progress={"stage": "building_features"},
                    worker_token=worker_token,
                    lease_seconds=5,
                )
            )
        except BaseException as error:
            transition_errors.append(error)

    event.listen(engine, "before_cursor_execute", pause_progress_update)
    transition_thread = Thread(target=transition)
    transition_thread.start()
    try:
        assert transition_update_ready.wait(timeout=5)
        assert repository.renew_task_execution_claim(
            checkpoint.task_id,
            worker_token=worker_token,
            lease_seconds=5,
        )
    finally:
        allow_transition_update.set()
        transition_thread.join(timeout=5)
        event.remove(
            engine,
            "before_cursor_execute",
            pause_progress_update,
        )

    assert not transition_thread.is_alive()
    assert transition_errors == []
    assert len(transition_results) == 1
    assert transition_results[0].status == "building_features"
    with pytest.raises(ValueError, match="worker claim token"):
        repository.transition_task_checkpoint(
            checkpoint.task_id,
            status="iterating",
            progress={"stage": "iterating"},
            worker_token="replacement-worker",
            lease_seconds=5,
        )


def test_two_managers_share_one_database_execution_claim_and_remote_wait(
    tmp_path: Path,
) -> None:
    sessions, first_repository = _repository(tmp_path)
    second_repository = _repository_api().WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly-analysis-v2-second-manager",
    )
    engine = _BlockingEngine()
    first_manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=first_repository,
        claim_lease_seconds=5,
    )
    second_manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=second_repository,
        claim_lease_seconds=5,
    )
    submit_barrier = Barrier(2)
    submitted: list[tuple[Any, Any]] = []
    submit_lock = Lock()

    def submit(manager) -> None:
        submit_barrier.wait(timeout=10)
        checkpoint = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            as_of=date(2026, 7, 30),
        )
        with submit_lock:
            submitted.append((manager, checkpoint))

    threads = (
        Thread(target=submit, args=(first_manager,)),
        Thread(target=submit, args=(second_manager,)),
    )
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert len(submitted) == 2
        assert engine.started.wait(timeout=10)
        assert submitted[0][1].task_id == submitted[1][1].task_id
        assert len(engine.calls) == 1

        owner = next(
            manager
            for manager, checkpoint in submitted
            if checkpoint.task_id in manager._futures
        )
        observer = (
            second_manager if owner is first_manager else first_manager
        )
        observed: list[Any] = []
        waiter = Thread(
            target=lambda: observed.append(
                observer.wait(submitted[0][1].task_id, timeout=10)
            )
        )
        waiter.start()
        time.sleep(0.1)
        assert waiter.is_alive()
        engine.release.set()
        waiter.join(timeout=10)
        assert observed[0].status == "completed"
        assert len(engine.calls) == 1
    finally:
        engine.release.set()
        first_manager.shutdown(wait=True)
        second_manager.shutdown(wait=True)


def test_short_claim_lease_is_renewed_while_engine_is_silent(
    tmp_path: Path,
) -> None:
    sessions, first_repository = _repository(tmp_path)
    second_repository = _repository_api().WeeklyAnalysisRepository(
        sessions,
        audit_root=tmp_path / "weekly-analysis-v2-second-manager",
    )
    engine = _SilentBlockingEngine()
    first_manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=first_repository,
        claim_lease_seconds=0.05,
        remote_poll_seconds=0.01,
    )
    second_manager = None
    try:
        submitted = first_manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            as_of=date(2026, 7, 30),
        )
        assert engine.started.wait(timeout=10)
        time.sleep(0.2)

        second_manager = _tasks_api().WeeklyAnalysisTaskManager(
            engine=engine,
            repository=second_repository,
            claim_lease_seconds=0.05,
            remote_poll_seconds=0.01,
        )
        observed = second_manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            as_of=date(2026, 7, 30),
        )
        time.sleep(0.05)

        assert observed.task_id == submitted.task_id
        with engine.lock:
            assert engine.calls == ["399006"]

        engine.release.set()
        assert second_manager.wait(observed.task_id, timeout=10).status == "completed"
    finally:
        engine.release.set()
        first_manager.shutdown(wait=True)
        if second_manager is not None:
            second_manager.shutdown(wait=True)

    heartbeat_name = f"weekly-analysis-v2-heartbeat-{submitted.task_id}"
    assert not any(
        thread.name == heartbeat_name and thread.is_alive()
        for thread in enumerate_threads()
    )


def test_short_lease_completion_is_stable_under_high_frequency_remote_wait(
    tmp_path: Path,
) -> None:
    for round_index in range(5):
        round_root = tmp_path / f"round-{round_index}"
        sessions, first_repository = _repository(round_root)
        second_repository = _repository_api().WeeklyAnalysisRepository(
            sessions,
            audit_root=round_root / "remote-weekly-analysis-v2",
        )
        engine = _SilentBlockingEngine()
        first_manager = _tasks_api().WeeklyAnalysisTaskManager(
            engine=engine,
            repository=first_repository,
            claim_lease_seconds=0.05,
            remote_poll_seconds=0.001,
        )
        second_manager = None
        wait_results: list[object] = []
        wait_errors: list[BaseException] = []
        try:
            submitted = first_manager.submit(
                "399006",
                latest_complete_week="2026-W30",
                as_of=date(2026, 7, 30),
            )
            assert engine.started.wait(timeout=10)
            time.sleep(0.2)
            second_manager = _tasks_api().WeeklyAnalysisTaskManager(
                engine=engine,
                repository=second_repository,
                claim_lease_seconds=0.05,
                remote_poll_seconds=0.001,
            )
            observed = second_manager.submit(
                "399006",
                latest_complete_week="2026-W30",
                as_of=date(2026, 7, 30),
            )
            assert observed.task_id == submitted.task_id

            def wait_remotely() -> None:
                try:
                    wait_results.append(
                        second_manager.wait(
                            observed.task_id,
                            timeout=10,
                        )
                    )
                except BaseException as error:
                    wait_errors.append(error)

            remote_waiter = Thread(target=wait_remotely)
            remote_waiter.start()
            time.sleep(0.05)
            engine.release.set()
            remote_waiter.join(timeout=10)

            assert not remote_waiter.is_alive()
            assert wait_errors == []
            assert len(wait_results) == 1
            assert wait_results[0].status == "completed"
            assert engine.calls == ["399006"]
        finally:
            engine.release.set()
            first_manager.shutdown(wait=True)
            if second_manager is not None:
                second_manager.shutdown(wait=True)


def test_lost_claim_heartbeat_fences_stale_worker_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions, repository = _repository(tmp_path)
    engine = _SilentBlockingEngine()
    claim_lost = Event()
    renewal_attempts = 0

    def lose_claim(
        task_id: str,
        *,
        worker_token: str,
        lease_seconds: float,
    ) -> bool:
        nonlocal renewal_attempts
        renewal_attempts += 1
        if renewal_attempts == 1:
            return True
        with sessions.begin() as session:
            row = session.scalar(
                select(V2AnalysisTask).where(
                    V2AnalysisTask.task_key == task_id
                )
            )
            assert row is not None
            row.worker_token = "replacement-worker"
            row.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=1
            )
        claim_lost.set()
        return False

    monkeypatch.setattr(
        repository,
        "renew_task_execution_claim",
        lose_claim,
        raising=False,
    )
    manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
        claim_lease_seconds=0.05,
        remote_poll_seconds=0.01,
    )
    try:
        submitted = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            as_of=date(2026, 7, 30),
        )
        assert engine.started.wait(timeout=10)
        assert claim_lost.wait(timeout=1)
        engine.release.set()

        outcome = manager._futures[submitted.task_id].result(timeout=10)
        assert isinstance(outcome, RuntimeError)
        assert "claim" in str(outcome).lower()
        assert repository.load_task_checkpoint(submitted.task_id).status == (
            "preparing_data"
        )
    finally:
        engine.release.set()
        manager.shutdown(wait=True)

    heartbeat_name = f"weekly-analysis-v2-heartbeat-{submitted.task_id}"
    assert not any(
        thread.name == heartbeat_name and thread.is_alive()
        for thread in enumerate_threads()
    )


def test_transient_heartbeat_operational_error_retries_within_live_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sessions, repository = _repository(tmp_path)
    engine = _SilentBlockingEngine()
    real_renew = repository.renew_task_execution_claim
    attempts = 0

    def transient_then_renew(*args, **kwargs) -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OperationalError(
                "UPDATE v2_analysis_tasks",
                {},
                RuntimeError("temporary sqlite busy"),
            )
        return real_renew(*args, **kwargs)

    monkeypatch.setattr(
        repository,
        "renew_task_execution_claim",
        transient_then_renew,
    )
    manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
        claim_lease_seconds=0.05,
        remote_poll_seconds=0.01,
    )
    try:
        submitted = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            as_of=date(2026, 7, 30),
        )
        assert engine.started.wait(timeout=10)
        assert attempts >= 2
        engine.release.set()
        assert manager.wait(submitted.task_id, timeout=10).status == "completed"
    finally:
        engine.release.set()
        manager.shutdown(wait=True)


def test_service_captures_each_position_once_and_reuses_the_same_request() -> None:
    tasks = _tasks_api()
    service_module = importlib.import_module(
        "backend.app.services.weekly_analysis_service"
    )

    class CapturingProvider:
        def __init__(self) -> None:
            self.positions = iter((20, 25, 20))
            self.reads = 0

        def capture_analysis_request(self, symbol: str, *, model_line: str):
            self.reads += 1
            position = next(self.positions)
            payload = str(position)
            return tasks.AnalysisRequest(
                latest_complete_week="2026-W30",
                model_line=model_line,
                current_position=position,
                position_snapshot_key=hashlib.sha256(
                    payload.encode("ascii")
                ).hexdigest(),
            )

    class CapturingManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def submit(self, _symbol: str, **kwargs):
            self.calls.append(kwargs)
            return kwargs

    provider = CapturingProvider()
    manager = CapturingManager()
    service = service_module.WeeklyAnalysisService(
        provider=provider,
        task_manager=manager,
    )

    first = service.start("399006")
    second = service.start("399006")
    third = service.start("399006")

    assert provider.reads == 3
    requests = [call["analysis_request"] for call in manager.calls]
    assert [request.current_position for request in requests] == [20, 25, 20]
    assert first["advice_request_key"] == requests[0].position_snapshot_key
    assert second["advice_request_key"] == requests[1].position_snapshot_key
    assert third["advice_request_key"] == requests[2].position_snapshot_key
    assert first["advice_request_key"] == third["advice_request_key"]


def test_analysis_request_is_persisted_and_passed_unchanged_to_engine(
    tmp_path: Path,
) -> None:
    tasks = _tasks_api()
    sessions, repository = _repository(tmp_path)
    captured: list[object] = []

    class CapturingEngine:
        def run(
            self,
            _symbol: str,
            *,
            as_of=None,
            progress_callback=None,
            advice_request_key=None,
            analysis_request=None,
            execution_guard=None,
        ):
            captured.append((analysis_request, execution_guard))
            return {"position": analysis_request.current_position}

    request = tasks.AnalysisRequest(
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
        current_position=20,
        position_snapshot_key=hashlib.sha256(b"20").hexdigest(),
    )
    manager = tasks.WeeklyAnalysisTaskManager(
        engine=CapturingEngine(),
        repository=repository,
    )
    try:
        submitted = manager.submit(
            "399006",
            latest_complete_week=request.latest_complete_week,
            model_line=request.model_line,
            advice_request_key=request.position_snapshot_key,
            analysis_request=request,
        )
        assert manager.wait(submitted.task_id, timeout=10).status == "completed"
    finally:
        manager.shutdown(wait=True)

    with sessions() as session:
        row = session.scalar(
            select(V2AnalysisTask).where(
                V2AnalysisTask.task_key == submitted.task_id
            )
        )
        assert row is not None
        persisted = row.request_payload["analysis_request"]
    assert persisted["current_position"] == 20
    assert persisted["position_snapshot_key"] == request.position_snapshot_key
    assert captured[0][0] == request
    assert captured[0][1].task_id == submitted.task_id


@pytest.mark.parametrize(
    "illegal_status",
    (
        "building_features",
        "iterating",
        "validating",
        "generating_advice",
        "completed",
    ),
)
def test_repository_rejects_nonadjacent_task_status_transitions(
    tmp_path: Path,
    illegal_status: str,
) -> None:
    _sessions, repository = _repository(tmp_path)
    queued, _created = repository.get_or_create_task(
        "399006",
        f"399006:2026-W30:weekly-v2:{illegal_status}",
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
    )

    with pytest.raises(ValueError, match="adjacent|transition|monotonic"):
        repository.transition_task_checkpoint(
            queued.task_id,
            status=illegal_status,
        )


def test_validating_cannot_skip_generating_advice(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)
    checkpoint, _created = repository.get_or_create_task(
        "399006",
        "399006:2026-W30:strict-validating-transition",
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
    )
    for status in (
        "preparing_data",
        "building_features",
        "iterating",
        "validating",
    ):
        checkpoint = repository.transition_task_checkpoint(
            checkpoint.task_id,
            status=status,
            progress={"stage": status},
        )

    with pytest.raises(ValueError, match="adjacent|transition|monotonic"):
        repository.transition_task_checkpoint(
            checkpoint.task_id,
            status="completed",
        )

    generating = repository.transition_task_checkpoint(
        checkpoint.task_id,
        status="generating_advice",
    )
    completed = repository.transition_task_checkpoint(
        checkpoint.task_id,
        status="completed",
    )
    assert generating.status == "generating_advice"
    assert completed.status == "completed"


def test_completed_weekly_task_creates_one_idempotent_advice_only_generation(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)

    class ImmediateEngine:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def run(
            self,
            symbol,
            *,
            as_of=None,
            progress_callback=None,
            advice_request_key=None,
            analysis_request=None,
            execution_guard=None,
        ):
            self.calls.append(advice_request_key)
            if progress_callback is not None:
                progress_callback({"stage": "building_features"})
                progress_callback({"stage": "generating_advice"})
            return {"symbol": symbol}

    engine = ImmediateEngine()
    manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
    )
    try:
        weekly = manager.submit(
            "399006", latest_complete_week="2026-W30"
        )
        weekly_done = manager.wait(weekly.task_id, timeout=10)
        advice_only = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            advice_request_key="position:20",
        )
        advice_done = manager.wait(advice_only.task_id, timeout=10)
        same = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            advice_request_key="position:20",
        )
    finally:
        manager.shutdown(wait=True)

    assert weekly_done.status == "completed"
    assert advice_done.status == "completed"
    assert advice_done.last_iteration == weekly_done.last_iteration == 0
    assert advice_only.task_id != weekly.task_id
    assert same.task_id == advice_only.task_id
    assert engine.calls == ["unmanaged-position-unset", "position:20"]


def test_repository_task_idempotency_and_formal_state_transitions(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)

    first, created = repository.get_or_create_task(
        "399006",
        "399006:2026-W30:weekly-v2",
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
    )
    same, created_again = repository.get_or_create_task(
        "399006",
        "399006:2026-W30:weekly-v2",
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
    )
    preparing = repository.transition_task_checkpoint(
        first.task_id,
        status="preparing_data",
        progress={"stage": "preparing_data"},
    )
    repository.transition_task_checkpoint(
        first.task_id,
        status="building_features",
        progress={"stage": "building_features"},
    )
    iterating = repository.transition_task_checkpoint(
        first.task_id,
        status="iterating",
        progress={"stage": "iterating"},
    )

    assert created is True
    assert created_again is False
    assert same.task_id == first.task_id
    assert first.status == "queued"
    assert preparing.status == "preparing_data"
    assert iterating.status == "iterating"
    with pytest.raises(ValueError, match="transition|monotonic"):
        repository.transition_task_checkpoint(
            first.task_id,
            status="queued",
        )


def test_repository_recovers_all_legacy_and_nonterminal_tasks(
    tmp_path: Path,
) -> None:
    sessions, repository = _repository(tmp_path)
    queued, _created = repository.get_or_create_task(
        "399006",
        "399006:2026-W30:weekly-v2",
        latest_complete_week="2026-W30",
        model_line="weekly-v2",
    )
    repository.transition_task_checkpoint(
        queued.task_id,
        status="preparing_data",
    )
    with sessions.begin() as session:
        instrument_id = session.scalar(
            select(Instrument.id).where(Instrument.code == "NDX")
        )
        session.add(
            V2AnalysisTask(
                instrument_id=instrument_id,
                task_key="legacy-running-task",
                task_type="weekly-analysis-v2",
                status="running",
                request_payload={"symbol": "NDX"},
                result_payload={
                    "task_id": "legacy-running-task",
                    "symbol": "NDX",
                    "status": "running",
                    "completed_weeks": 0,
                    "total_weeks": 0,
                    "last_iteration": 0,
                    "last_work_version": "W0000",
                    "last_model_version": "M0001",
                    "last_completed_week": None,
                    "progress": {},
                },
                audit_data={
                    "checkpoint_version": "weekly-analysis-repository-v2",
                    "codec_version": "weekly-analysis-tagged-json-v1",
                },
            )
        )

    recovered = repository.recover_interrupted_tasks(
        "FastAPI process restarted"
    )

    assert {task.task_id for task in recovered} == {
        queued.task_id,
        "legacy-running-task",
    }
    assert all(task.status == "recoverable" for task in recovered)
    legacy = repository.load_task_checkpoint("legacy-running-task")
    assert legacy.progress["legacy_status"] == "running"
    assert legacy.progress["recoverable_reason"] == "FastAPI process restarted"


def test_manager_is_idempotent_and_same_market_never_runs_twice(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)
    engine = _BlockingEngine()
    manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
    )
    try:
        first = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            as_of=date(2026, 7, 30),
        )
        assert engine.started.wait(timeout=10)
        second = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
            as_of=date(2026, 7, 30),
        )
        assert second.task_id == first.task_id
        assert len(engine.calls) == 1
        engine.release.set()
        completed = manager.wait(first.task_id, timeout=10)
        assert completed.status == "completed"
    finally:
        engine.release.set()
        manager.shutdown(wait=True)


def test_two_markets_run_in_parallel_with_two_workers(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)
    engine = _BlockingEngine(participants=2)
    manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=engine,
        repository=repository,
    )
    try:
        growth = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
        )
        nasdaq = manager.submit(
            "NDX",
            latest_complete_week="2026-W30",
        )
        assert engine.both_started.wait(timeout=10)
        assert engine.parallel_markets is True
        engine.release.set()
        assert manager.wait(growth.task_id, timeout=10).status == "completed"
        assert manager.wait(nasdaq.task_id, timeout=10).status == "completed"
    finally:
        engine.release.set()
        manager.shutdown(wait=True)


def test_manager_failure_preserves_message_and_last_atomic_checkpoint(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)

    class FailingEngine:
        def run(
            self,
            symbol: str,
            *,
            as_of=None,
            progress_callback=None,
            advice_request_key=None,
            analysis_request=None,
            execution_guard=None,
        ):
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "iterating",
                        "completed_weeks": 0,
                        "total_weeks": 3,
                        "last_checkpoint": None,
                        "elapsed_seconds": 0.1,
                    }
                )
            raise RuntimeError("verified dataset unavailable")

    manager = _tasks_api().WeeklyAnalysisTaskManager(
        engine=FailingEngine(),
        repository=repository,
    )
    try:
        submitted = manager.submit(
            "399006",
            latest_complete_week="2026-W30",
        )
        failed = manager.wait(submitted.task_id, timeout=10)
    finally:
        manager.shutdown(wait=True)

    assert failed.status == "failed"
    assert failed.progress["failure_reason"] == "verified dataset unavailable"
    assert failed.last_completed_week is None
    assert failed.last_iteration == 0


def test_task7_incremental_commit_recovers_five_week_chain(
    tmp_path: Path,
) -> None:
    _sessions, repository = _repository(tmp_path)
    state = repository.load_work_state("399006")
    start = date(2026, 1, 9)
    for number in range(1, 6):
        cutoff = start + timedelta(weeks=number - 1)
        week_key = _week_key(cutoff)
        result = run_optimizer_step(
            state,
            feedback=(),
            week_key=week_key,
            cutoff_date=cutoff,
            source_data_max_date=cutoff,
            prediction=IterationPrediction(
                iteration_id=f"I{number:04d}",
                instrument_code="399006",
                week_key=week_key,
                cutoff_date=cutoff,
                predicted_direction="up",
                operation_side="hold",
                probability=Decimal("0.60"),
                predicted_action_date=cutoff,
                target_position=Decimal("0.50"),
                trade_count=0,
            ),
        )
        repository.commit_iteration_incremental(
            "399006", state, result, labels=result.work_state.feedback
        )
        state = result.work_state

    reopened = repository.load_work_state("399006")
    assert reopened == state
    assert reopened.iteration_id == "I0005"
    assert tuple(reopened.week_records) == tuple(
        _week_key(start + timedelta(weeks=index))
        for index in range(5)
    )
    with pytest.raises(Exception, match="validated WorkState|incremental"):
        repository.commit_iteration_incremental(
            "399006",
            seed_model("399006"),
            result,
            labels=result.work_state.feedback,
        )


def _week_key(day: date) -> str:
    iso_year, iso_week, _weekday = day.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"
