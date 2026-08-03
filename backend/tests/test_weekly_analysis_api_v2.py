from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from pydantic import ValidationError

from backend.app.weekly_analysis.features import FrozenDict
from backend.app.weekly_analysis.repository import (
    CheckpointNotFound,
    RepositoryCorruption,
    TaskCheckpoint,
)
from backend import web


TASK_399006 = "weekly-v2-" + "1" * 32
TASK_NDX = "weekly-v2-" + "2" * 32
TASK_RECOVERABLE = "weekly-v2-" + "3" * 32
TASK_COMPLETED = "weekly-v2-" + "4" * 32
TASK_FAILED = "weekly-v2-" + "5" * 32
TASK_MISSING = "weekly-v2-" + "6" * 32


class _Response:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def json(self):
        return json.loads(self._body)


class _AsgiClient:
    def __init__(self, application) -> None:
        self.application = application

    def request(
        self,
        method: str,
        path: str,
        *,
        params=None,
        json_body=None,
    ) -> _Response:
        async def invoke() -> _Response:
            body = (
                b""
                if json_body is None
                else json.dumps(json_body).encode("utf-8")
            )
            headers = []
            if json_body is not None:
                headers.append((b"content-type", b"application/json"))
            messages = []
            delivered = False

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {
                        "type": "http.request",
                        "body": body,
                        "more_body": False,
                    }
                return {"type": "http.disconnect"}

            async def send(message):
                messages.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": urlencode(params or {}).encode("ascii"),
                "headers": headers,
                "client": ("test", 50000),
                "server": ("testserver", 80),
                "root_path": "",
                "app": self.application,
            }
            await self.application(scope, receive, send)
            status = next(
                message["status"]
                for message in messages
                if message["type"] == "http.response.start"
            )
            response_body = b"".join(
                message.get("body", b"")
                for message in messages
                if message["type"] == "http.response.body"
            )
            return _Response(status, response_body)

        return asyncio.run(invoke())

    def get(self, path: str, *, params=None) -> _Response:
        return self.request("GET", path, params=params)

    def post(self, path: str, *, json=None) -> _Response:
        return self.request("POST", path, json_body=json)

    def patch(self, path: str, *, json=None) -> _Response:
        return self.request("PATCH", path, json_body=json)

    def delete(self, path: str, *, params=None) -> _Response:
        return self.request("DELETE", path, params=params)


def _checkpoint(
    *,
    task_id: str = TASK_399006,
    symbol: str = "399006",
    status: str = "queued",
    progress: dict[str, object] | None = None,
) -> TaskCheckpoint:
    return TaskCheckpoint(
        task_id=task_id,
        symbol=symbol,
        status=status,
        completed_weeks=0,
        total_weeks=10,
        last_iteration=0,
        last_work_version="W0000",
        last_model_version="M0001",
        last_completed_week=None,
        progress=FrozenDict(
            {
                "latest_complete_week": "2026-W30",
                "model_line": "weekly-analysis-v2",
                "stage": status,
                **(progress or {}),
            }
        ),
    )


class _FakeWeeklyAnalysisService:
    def __init__(self, manager) -> None:
        self.task_manager = manager
        self._guard = Lock()
        self._tasks: dict[str, TaskCheckpoint] = {}
        self.start_calls = 0

    def start(self, symbol: str) -> TaskCheckpoint:
        with self._guard:
            self.start_calls += 1
            return self._tasks.setdefault(
                symbol,
                _checkpoint(
                    task_id=(
                        TASK_399006 if symbol == "399006" else TASK_NDX
                    ),
                    symbol=symbol,
                ),
            )


class _FakeTaskManager:
    def __init__(self) -> None:
        self.tasks: dict[str, TaskCheckpoint] = {}
        self.submit_calls: list[dict[str, object]] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def get(self, task_id: str) -> TaskCheckpoint:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise CheckpointNotFound("private missing checkpoint path") from None

    def submit(self, symbol: str, **kwargs) -> TaskCheckpoint:
        self.submit_calls.append({"symbol": symbol, **kwargs})
        task_id = next(
            key
            for key, checkpoint in self.tasks.items()
            if checkpoint.symbol == symbol
            and checkpoint.status == "recoverable"
        )
        resumed = replace(
            self.tasks[task_id],
            status="preparing_data",
            progress=FrozenDict(
                {**dict(self.tasks[task_id].progress), "stage": "preparing_data"}
            ),
        )
        self.tasks[task_id] = resumed
        return resumed

    def shutdown(
        self,
        *,
        wait: bool,
        cancel_futures: bool = False,
    ) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class _FakeRepository:
    def __init__(self) -> None:
        self.metrics_result = None
        self.advice_result = None

    def metrics(self, symbol: str):
        if self.metrics_result is not None:
            if isinstance(self.metrics_result, Exception):
                raise self.metrics_result
            return self.metrics_result
        return SimpleNamespace(
            to_dict=lambda: {
                "symbol": symbol,
                "model_version": "M0007",
                "iteration_count": 7,
                "last_iteration": 7,
                "accepted_iterations": 5,
                "rejected_iterations": 2,
                "feedback_count": 6,
                "mature_count": 4,
                "partial_count": 1,
                "pending_count": 1,
                "candidate_acceptance_rate": Decimal("62.50"),
                "curve": [{
                    "iteration_number": 1,
                    "iteration_id": "I0001",
                    "cutoff_date": date(2026, 1, 9),
                    "model_version": "M0001",
                    "accepted": True,
                    "deviation_days": -2,
                    "absolute_deviation": Decimal("2"),
                    "rolling_20_abs_deviation": None,
                    "rolling_52_abs_deviation": None,
                    "direction_correct": True,
                }],
                "windows": {
                    "single": {
                        "instrument_code": symbol,
                        "window": "single",
                        "sample_count": 1,
                        "total_loss": Decimal("0.25"),
                        "direction_hit_rate": Decimal("75"),
                        "calibration_loss": Decimal("0.125"),
                        "overtrade_penalty": Decimal("0"),
                    }
                },
            }
        )

    def latest_advice(self, symbol: str):
        if self.advice_result is not None:
            if isinstance(self.advice_result, Exception):
                raise self.advice_result
            return self.advice_result
        return SimpleNamespace(
            id=8,
            symbol=symbol,
            iteration_id="I0007",
            work_version="W0007",
            model_version="M0007",
            advice_generation=2,
            generation_key="safe-key",
            advice_at=datetime(2026, 7, 31, 2, 3, tzinfo=timezone.utc),
            payload=FrozenDict(
                {
                    "advice_version": "weekly-advice-v2",
                    "audit_fields": {
                        "feature_source_hash": "a" * 64,
                        "nested_score": Decimal("0.5"),
                        "market_state_inputs": FrozenDict(
                            {
                                "black_point": False,
                                "valuation_percentile": None,
                            }
                        ),
                    },
                    "audit_notes": ("verified",),
                    "batches": (
                        {
                            "confirmation_condition": "DIF crosses DEA",
                            "expected_date": date(2026, 8, 3),
                            "operation_side": "buy",
                            "percent": 10,
                            "sequence": 1,
                            "status": "ready",
                            "tolerance_trading_days": 3,
                        },
                    ),
                    "confidence": Decimal("75"),
                    "current_position": 60,
                    "instrument_code": symbol,
                    "direction": "up",
                    "direction_probabilities": {
                        "up": Decimal("71.25"),
                        "down": Decimal("18.75"),
                        "neutral": Decimal("10"),
                    },
                    "feature_set_version": "weekly-features-v2",
                    "forecast_horizon_weeks": 13,
                    "fund_etf_ratio": "7:3",
                    "market_state": "bullish",
                    "model_version": "M0007",
                    "operation_side": "buy",
                    "probability": Decimal("71.25"),
                    "recommendation": "staged_increase",
                    "signed_position_change": 10,
                    "data_cutoff_date": date(2026, 7, 24),
                    "source_data_max_date": date(2026, 7, 24),
                    "target_position": 70,
                    "target_position_range": (65, 75),
                    "total_adjustment": 10,
                }
            ),
            snapshot_path=Path(r"C:\private\audit\advice.json"),
            snapshot_hash="a" * 64,
        )


class _FakeCalendar:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def list_entries(self, instrument_code=None):
        return [
            row
            for row in self.rows
            if instrument_code is None or row["index"] == instrument_code
        ]

    def create(self, values):
        row = {
            "id": 1,
            "index": values.instrument_code,
            "direction": values.direction,
            "date": values.operation_date,
            "change_percent": values.change_percent,
            "position_after": values.change_percent,
            "note": values.note,
        }
        self.rows.append(row)
        return row

    def update(self, entry_id, values):
        row = next(row for row in self.rows if row["id"] == entry_id)
        changes = values.model_dump(exclude_unset=True)
        if "instrument_code" in changes:
            row["index"] = changes.pop("instrument_code")
        if "operation_date" in changes:
            row["date"] = changes.pop("operation_date")
        row.update(changes)
        return row

    def delete(self, entry_id):
        self.rows = [row for row in self.rows if row["id"] != entry_id]


@pytest.fixture
def api_client():
    manager = _FakeTaskManager()
    service = _FakeWeeklyAnalysisService(manager)
    repository = _FakeRepository()
    calendar = _FakeCalendar()
    web.app.state.weekly_analysis = service
    web.app.state.analysis_tasks = manager
    web.app.state.weekly_analysis_repository = repository
    web.app.state.investment_calendar = calendar
    return _AsgiClient(web.app), service, manager, repository, calendar


def test_create_task_poll_and_completed_idempotent_reuse(api_client) -> None:
    client, service, manager, _repository, _calendar = api_client

    created = client.post(
        "/api/v2/analysis/tasks",
        json={"instrument_code": "399006"},
    )

    assert created.status_code == 202
    task = created.json()
    assert task["id"] == TASK_399006
    assert task["instrument_code"] == "399006"
    assert task["status"] == "queued"
    manager.tasks[task["id"]] = service._tasks["399006"]
    polled = client.get(f"/api/v2/analysis/tasks/{task['id']}")
    assert polled.status_code == 200
    assert polled.json()["status"] == "queued"

    service._tasks["399006"] = replace(
        service._tasks["399006"],
        status="completed",
    )
    reused = client.post(
        "/api/v2/analysis/tasks",
        json={"instrument_code": "399006"},
    )
    assert reused.status_code == 200
    assert reused.json()["id"] == task["id"]
    assert reused.json()["reused"] is True


def test_create_rejects_unknown_market_and_all_client_cutoff_fields(
    api_client,
) -> None:
    client, _service, _manager, _repository, _calendar = api_client

    unknown = client.post(
        "/api/v2/analysis/tasks",
        json={"instrument_code": "000688"},
    )
    with_as_of = client.post(
        "/api/v2/analysis/tasks",
        json={"instrument_code": "399006", "as_of": "2026-07-24"},
    )
    with_cutoff = client.post(
        "/api/v2/analysis/tasks",
        json={
            "instrument_code": "399006",
            "data_cutoff_date": "2026-07-24",
        },
    )

    assert unknown.status_code == 400
    assert with_as_of.status_code == 422
    assert with_cutoff.status_code == 422


def test_metrics_and_latest_advice_json_encode_decimal_dates_without_paths(
    api_client,
) -> None:
    client, _service, _manager, _repository, _calendar = api_client

    metrics = client.get("/api/v2/models/NDX/metrics")
    advice = client.get("/api/v2/advice/NDX/latest")

    assert metrics.status_code == 200
    assert metrics.json()["candidate_acceptance_rate"] == "62.50"
    assert metrics.json()["curve"][0] == {
        "iteration_number": 1,
        "iteration_id": "I0001",
        "cutoff_date": "2026-01-09",
        "model_version": "M0001",
        "accepted": True,
        "deviation_days": -2,
        "absolute_deviation": "2",
        "rolling_20_abs_deviation": None,
        "rolling_52_abs_deviation": None,
        "direction_correct": True,
    }
    assert metrics.json()["windows"]["single"] == {
        "instrument_code": "NDX",
        "window": "single",
        "sample_count": 1,
        "total_loss": "0.25",
        "direction_hit_rate": "75",
        "calibration_loss": "0.125",
        "overtrade_penalty": "0",
    }
    assert advice.status_code == 200
    assert advice.json()["probability"] == "71.25"
    assert advice.json()["data_cutoff_date"] == "2026-07-24"
    assert advice.json()["advice_at"] == "2026-07-31T02:03:00Z"
    assert advice.json()["audit_fields"]["market_state_inputs"] == {
        "black_point": False,
        "valuation_percentile": None,
    }
    assert "snapshot_path" not in advice.json()
    assert "private" not in json.dumps(advice.json())


def test_unknown_model_and_corrupt_repository_errors_are_safe(
    api_client,
) -> None:
    client, _service, _manager, repository, _calendar = api_client

    unknown = client.get("/api/v2/models/000688/metrics")
    repository.metrics = lambda _symbol: (_ for _ in ()).throw(
        RepositoryCorruption(r"corrupt C:\private\weekly\snapshot.json")
    )
    corrupt = client.get("/api/v2/models/399006/metrics")

    assert unknown.status_code == 400
    assert corrupt.status_code == 503
    rendered = json.dumps(corrupt.json())
    assert "private" not in rendered
    assert "snapshot" not in rendered
    assert "traceback" not in rendered.lower()


def test_unknown_value_error_is_logged_and_returns_fixed_safe_500(
    api_client,
    caplog,
) -> None:
    client, _service, _manager, repository, _calendar = api_client
    repository.metrics_result = ValueError(
        r"secret worker_token=abc C:\private\state.json"
    )

    response = client.get("/api/v2/models/399006/metrics")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    rendered = json.dumps(response.json())
    assert "worker_token" not in rendered
    assert "private" not in rendered
    assert any(
        "weekly analysis" in record.getMessage().lower()
        for record in caplog.records
    )


def test_nonfinite_metrics_are_rejected_without_stringifying_nan(
    api_client,
) -> None:
    client, _service, _manager, repository, _calendar = api_client
    payload = repository.metrics("399006").to_dict()
    payload["windows"]["single"]["calibration_loss"] = Decimal("NaN")
    repository.metrics_result = payload

    response = client.get("/api/v2/models/399006/metrics")

    assert response.status_code in {500, 503}
    rendered = json.dumps(response.json())
    assert "NaN" not in rendered
    assert "nan" not in rendered


@pytest.mark.parametrize("endpoint", ("create", "status", "resume"))
@pytest.mark.parametrize(
    "mutation",
    ("malformed_task_id", "nan_progress"),
)
def test_task_endpoints_hide_response_validation_details(
    api_client,
    caplog,
    endpoint,
    mutation,
) -> None:
    client, service, manager, _repository, _calendar = api_client
    checkpoint = _checkpoint(
        task_id=TASK_RECOVERABLE if endpoint == "resume" else TASK_399006,
        status="recoverable" if endpoint == "resume" else "queued",
    )
    if mutation == "malformed_task_id":
        checkpoint = replace(
            checkpoint,
            task_id=r"C:\private\fake-task-id",
        )
    else:
        checkpoint = replace(
            checkpoint,
            progress=FrozenDict(
                {
                    **dict(checkpoint.progress),
                    "elapsed_seconds": float("nan"),
                }
            ),
        )

    if endpoint == "create":
        service._tasks["399006"] = checkpoint
        response = client.post(
            "/api/v2/analysis/tasks",
            json={"instrument_code": "399006"},
        )
    elif endpoint == "status":
        manager.tasks[TASK_399006] = checkpoint
        response = client.get(f"/api/v2/analysis/tasks/{TASK_399006}")
    else:
        manager.tasks[TASK_RECOVERABLE] = checkpoint
        response = client.post(
            f"/api/v2/analysis/tasks/{TASK_RECOVERABLE}/resume"
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    rendered = json.dumps(response.json())
    assert "private" not in rendered.lower()
    assert "fake-task-id" not in rendered
    assert "nan" not in rendered.lower()
    assert "input_value" not in rendered
    assert any(
        "weekly analysis" in record.getMessage().lower()
        for record in caplog.records
    )


def test_global_value_error_handler_hides_pydantic_validation_details(
    caplog,
) -> None:
    malformed = _checkpoint(task_id=r"C:\private\fake-task-id")
    with pytest.raises(ValidationError) as captured:
        web._analysis_task_payload(malformed)

    response = asyncio.run(
        web.value_error_handler(None, captured.value)
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {"detail": "Internal server error"}
    assert any(
        "validation" in record.getMessage().lower()
        for record in caplog.records
    )


def test_latest_advice_none_is_404(api_client) -> None:
    client, _service, _manager, repository, _calendar = api_client
    repository.latest_advice = lambda _symbol: None

    response = client.get("/api/v2/advice/399006/latest")

    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found"}


@pytest.mark.parametrize(
    "mutation",
    ("unexpected_extra", "naive_datetime", "nested_nan"),
)
def test_latest_advice_schema_is_closed_timezone_aware_and_finite(
    api_client,
    mutation,
) -> None:
    client, _service, _manager, repository, _calendar = api_client
    record = repository.latest_advice("399006")
    if mutation == "unexpected_extra":
        record.payload = FrozenDict(
            {**dict(record.payload), "unexpected_private": "secret"}
        )
    elif mutation == "naive_datetime":
        record.advice_at = datetime(2026, 7, 31, 2, 3)
    else:
        record.payload = FrozenDict(
            {
                **dict(record.payload),
                "audit_fields": {
                    "nested": {"loss": Decimal("Infinity")}
                },
            }
        )
    repository.advice_result = record

    response = client.get("/api/v2/advice/399006/latest")

    assert response.status_code in {500, 503}
    rendered = json.dumps(response.json())
    assert "secret" not in rendered
    assert "Infinity" not in rendered


@pytest.mark.parametrize(
    "task_id",
    (
        "recover-me",
        "weekly-v2-" + "g" * 32,
        "weekly-v2-" + "a" * 31,
        "weekly-v2-" + "a" * 33,
        "weekly-v2-" + "a" * 32 + r"\..\secret",
    ),
)
def test_task_id_path_rejects_malformed_values_before_repository_lookup(
    api_client,
    task_id,
) -> None:
    client, _service, manager, _repository, _calendar = api_client

    status = client.get(f"/api/v2/analysis/tasks/{task_id}")
    resumed = client.post(f"/api/v2/analysis/tasks/{task_id}/resume")

    assert status.status_code in {400, 422}
    assert resumed.status_code in {400, 422}
    assert manager.tasks == {}


def test_resume_only_recoverable_and_missing_task_is_404(api_client) -> None:
    client, _service, manager, _repository, _calendar = api_client
    recoverable = _checkpoint(
        task_id=TASK_RECOVERABLE,
        status="recoverable",
        progress={"recoverable_reason": "planned maintenance pause"},
    )
    manager.tasks[recoverable.task_id] = recoverable
    manager.tasks[TASK_COMPLETED] = _checkpoint(
        task_id=TASK_COMPLETED,
        status="completed",
    )

    resumed = client.post(
        f"/api/v2/analysis/tasks/{TASK_RECOVERABLE}/resume"
    )
    refused = client.post(
        f"/api/v2/analysis/tasks/{TASK_COMPLETED}/resume"
    )
    missing = client.post(
        f"/api/v2/analysis/tasks/{TASK_MISSING}/resume"
    )

    assert resumed.status_code == 202
    assert resumed.json()["status"] == "preparing_data"
    assert resumed.json()["message"] is None
    assert manager.submit_calls[0]["latest_complete_week"] == "2026-W30"
    assert refused.status_code == 409
    assert missing.status_code == 404


def test_failed_task_exposes_auditable_message_without_internal_progress(
    api_client,
) -> None:
    client, _service, manager, _repository, _calendar = api_client
    manager.tasks[TASK_FAILED] = _checkpoint(
        task_id=TASK_FAILED,
        status="failed",
        progress={
            "failure_reason": (
                "authorization=Bearer private-api-key "
                r"C:\private\snapshot.json"
            ),
            "stage": "failed",
            "elapsed_seconds": 1.25,
            "api_key": "must-not-leak",
            "authorization": "must-not-leak",
            "password": "must-not-leak",
            "cookie": "must-not-leak",
            "secret": "must-not-leak",
            "worker_token": "must-not-leak",
            "nested": {
                "api_key": "nested-must-not-leak",
                "authorization": "nested-must-not-leak",
            },
        },
    )

    response = client.get(f"/api/v2/analysis/tasks/{TASK_FAILED}")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Weekly analysis failed; inspect local audit logs"
    )
    assert response.json()["progress"] == {
        "stage": "failed",
        "elapsed_seconds": 1.25,
    }
    rendered = json.dumps(response.json())
    for private_value in (
        "api_key",
        "authorization",
        "password",
        "cookie",
        "secret",
        "worker_token",
        "must-not-leak",
        "private-api-key",
        "snapshot",
    ):
        assert private_value not in rendered


def test_concurrent_requests_share_one_task_and_single_app_state_instances(
    api_client,
) -> None:
    client, service, manager, repository, _calendar = api_client
    identities = (
        id(web.app.state.weekly_analysis),
        id(web.app.state.analysis_tasks),
        id(web.app.state.weekly_analysis_repository),
    )

    def create():
        response = client.post(
            "/api/v2/analysis/tasks",
            json={"instrument_code": "NDX"},
        )
        return response.status_code, response.json()["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _index: create(), range(16)))

    assert {status for status, _task_id in results} == {202}
    assert {task_id for _status, task_id in results} == {TASK_NDX}
    assert identities == (
        id(web.app.state.weekly_analysis),
        id(manager),
        id(repository),
    )
    assert service.task_manager is manager


def test_v2_position_event_routes_use_v2_field_names(api_client) -> None:
    client, _service, _manager, _repository, _calendar = api_client

    created = client.post(
        "/api/v2/position-events",
        json={
            "instrument_code": "399006",
            "direction": "increase",
            "operation_date": "2026-07-31",
            "change_percent": 25,
            "note": "V2",
        },
    )
    listed = client.get(
        "/api/v2/position-events",
        params={"instrument_code": "399006"},
    )
    patched = client.patch(
        "/api/v2/position-events/1",
        json={"operation_date": "2026-08-03", "note": "moved"},
    )
    deleted = client.delete(
        "/api/v2/position-events/1",
        params={"confirmed": "true"},
    )

    assert created.status_code == 200
    assert created.json()["instrument_code"] == "399006"
    assert created.json()["operation_date"] == "2026-07-31"
    assert "index" not in created.json()
    assert "date" not in created.json()
    assert listed.json()[0]["instrument_code"] == "399006"
    assert patched.json()["operation_date"] == "2026-08-03"
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "id": 1}


def test_nested_lifespans_share_one_runtime_and_last_exit_shuts_it_down(
    tmp_path,
    monkeypatch,
) -> None:
    manager = _FakeTaskManager()
    manager.recovered_tasks = (_checkpoint(status="recoverable"),)
    service = _FakeWeeklyAnalysisService(manager)
    build_calls = []

    def fake_builder(sessions, **kwargs):
        build_calls.append((sessions, kwargs))
        return service

    class FakeV32Training:
        shutdown_calls: list[bool] = []

        def __init__(self, *_args, **_kwargs):
            pass

        def submit_due_incrementals(self):
            return []

        def shutdown(self, *, wait: bool = False):
            self.shutdown_calls.append(wait)

    class FakeV33Runtime:
        shutdown_calls: list[bool] = []

        def __init__(self, *_args, **_kwargs):
            pass

        def recover_interrupted_runs(self):
            return 0

        def champion(self, _market):
            return None

        def start_incremental(self, _market):
            raise AssertionError("unbootstrapped fake must not queue training")

        def shutdown(self, *, wait: bool = False):
            self.shutdown_calls.append(wait)

    class FakeV34Runtime(FakeV33Runtime):
        shutdown_calls: list[bool] = []

    class FakeV341Runtime(FakeV33Runtime):
        shutdown_calls: list[bool] = []

    class FakeIndicatorService:
        def __init__(self, *_args, **_kwargs):
            pass

        def recalculate_stale_timeframes(self, _instrument):
            return ()

    sessions = lambda: None
    monkeypatch.setattr(web, "project_root", lambda: tmp_path)
    monkeypatch.setattr(web, "initialize_database", lambda **_kwargs: None)
    monkeypatch.setattr(web, "create_session_factory", lambda: sessions)
    monkeypatch.setattr(web, "IndicatorService", FakeIndicatorService)
    monkeypatch.setattr(web, "build_weekly_analysis_service", fake_builder)
    monkeypatch.setattr(web, "V32TrainingService", FakeV32Training)
    monkeypatch.setattr(web, "V33RuntimeService", FakeV33Runtime)
    monkeypatch.setattr(web, "V34RuntimeService", FakeV34Runtime)
    monkeypatch.setattr(web, "V341RuntimeService", FakeV341Runtime)

    async def exercise() -> None:
        async with web.lifespan(web.app):
            first_service = web.app.state.weekly_analysis
            first_manager = web.app.state.analysis_tasks
            assert web.app.state.weekly_analysis is service
            assert web.app.state.analysis_tasks is manager
            assert (
                web.app.state.weekly_analysis_repository
                is manager.repository
            )
            async with web.lifespan(web.app):
                assert web.app.state.weekly_analysis is first_service
                assert web.app.state.analysis_tasks is first_manager
                assert manager.shutdown_calls == []
            assert web.app.state.weekly_analysis is first_service
            assert web.app.state.analysis_tasks is first_manager
            assert manager.shutdown_calls == []

    manager.repository = SimpleNamespace()
    asyncio.run(exercise())

    assert len(build_calls) == 1
    assert build_calls[0][0] is sessions
    assert build_calls[0][1]["audit_root"] == (
        tmp_path / "data" / "weekly_analysis_v2"
    )
    assert callable(build_calls[0][1]["current_position_provider"])
    assert manager.shutdown_calls == [(False, False)]
    assert FakeV32Training.shutdown_calls == [False]
    assert FakeV33Runtime.shutdown_calls == [False]
    assert FakeV34Runtime.shutdown_calls == [False]
    assert FakeV341Runtime.shutdown_calls == [False]
    assert not hasattr(web.app.state, "weekly_analysis")
    assert not hasattr(web.app.state, "analysis_tasks")
    assert not hasattr(web.app.state, "weekly_analysis_repository")
