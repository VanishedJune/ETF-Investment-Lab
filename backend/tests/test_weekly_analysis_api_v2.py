from __future__ import annotations

import asyncio
import json

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend import web


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 10000),
            "server": ("127.0.0.1", 8765),
        }
    )


@pytest.mark.parametrize(
    "path",
    (
        "/api/v2/analysis/tasks",
        "/api/v3.1/analysis/399006",
        "/api/v3.2/training/399006",
        "/api/v33/models/399006",
        "/api/v34/analysis/399006",
        "/api/v341/training/399006",
        "/api/v342/policy/399006",
        "/api/v343/analysis/399006",
        "/api/v37/analysis/399006",
        "/api/deepseek/test",
        "/api/v351/forecast/512010",
    ),
)
def test_data_only_guard_blocks_legacy_training_analysis_and_ai(path: str) -> None:
    async def exercise():
        async def forbidden_call_next(_request: Request):
            raise AssertionError("blocked legacy request reached a route handler")

        return await web.data_only_endpoint_guard(
            _request(path),
            forbidden_call_next,
        )

    response = asyncio.run(exercise())
    payload = json.loads(response.body)

    assert response.status_code == 404
    assert payload == {
        "detail": "V3.7 data-only build: AI、训练、推理与预测功能已移除。"
    }


@pytest.mark.parametrize(
    "path",
    (
        "/api/health",
        "/api/instruments",
        "/api/market/159941/prices",
        "/api/market/159941/refresh",
        "/api/indicators/159941",
        "/api/v2/position-events",
        "/api/investment-calendar/current-positions",
        "/api/v351/instrument-slots",
    ),
)
def test_data_only_guard_keeps_market_calendar_and_etf_replacement_active(
    path: str,
) -> None:
    calls: list[str] = []

    async def exercise():
        async def call_next(request: Request):
            calls.append(request.url.path)
            return JSONResponse({"active": True})

        return await web.data_only_endpoint_guard(_request(path), call_next)

    response = asyncio.run(exercise())

    assert response.status_code == 200
    assert json.loads(response.body) == {"active": True}
    assert calls == [path]


def test_blocked_prefixes_do_not_accidentally_block_etf_slot_replacement() -> None:
    assert not any(
        "/api/v351/instrument-slots".startswith(prefix)
        for prefix in (
            *web._DATA_ONLY_BLOCKED_PREFIXES,
            *web._DATA_ONLY_BLOCKED_V351_PREFIXES,
        )
    )


def test_data_only_health_contract_declares_incremental_online_refresh(tmp_path) -> None:
    request = type(
        "HealthRequest",
        (),
        {"app": type("App", (), {"state": type("State", (), {"root": tmp_path})()})()},
    )()

    payload = web.health(request)  # type: ignore[arg-type]

    assert payload["status"] == "ok"
    assert payload["mode"] == "local_only"
    assert payload["market_refresh_mode"] == (
        "ONLINE_INCREMENTAL_WITH_LOCAL_CACHE_FALLBACK"
    )
