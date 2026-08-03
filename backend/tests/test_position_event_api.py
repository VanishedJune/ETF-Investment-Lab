from __future__ import annotations

import asyncio
import json
from urllib.parse import urlencode

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.models.models import Base, Instrument
from backend.app.services.investment_calendar_service import InvestmentCalendarService
from backend.web import app


class _Response:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def json(self):
        return json.loads(self._body)


class _AsgiClient:
    def __init__(self, application) -> None:
        self.application = application

    def request(self, method: str, path: str, *, params=None, json_body=None) -> _Response:
        async def invoke() -> _Response:
            body = b"" if json_body is None else json.dumps(json_body).encode("utf-8")
            headers = []
            if json_body is not None:
                headers.append((b"content-type", b"application/json"))
            messages = []
            delivered = False

            async def receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.disconnect"}

            async def send(message):
                messages.append(message)

            query_string = urlencode(params or {}).encode("ascii")
            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": query_string,
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

    def post(self, path: str, *, json) -> _Response:
        return self.request("POST", path, json_body=json)

    def patch(self, path: str, *, json) -> _Response:
        return self.request("PATCH", path, json_body=json)

    def delete(self, path: str, *, params=None) -> _Response:
        return self.request("DELETE", path, params=params)


def _client(tmp_path) -> _AsgiClient:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Instrument(
                    code="399006",
                    name="创业板指数",
                    exchange="SZSE",
                    category="index",
                    currency="CNY",
                ),
                Instrument(
                    code="159941",
                    name="纳斯达克100指数",
                    exchange="NASDAQ",
                    category="index",
                    currency="USD",
                ),
            ]
        )
    app.state.root = tmp_path
    app.state.investment_calendar = InvestmentCalendarService(
        sessionmaker(bind=engine, expire_on_commit=False)
    )
    return _AsgiClient(app)


def test_position_event_api_crud_and_current_positions(tmp_path) -> None:
    client = _client(tmp_path)

    assert client.get("/api/investment-calendar/current-positions").json() == {
        "399006": 0,
        "159941": 0,
    }
    created = client.post(
        "/api/investment-calendar",
        json={
            "instrument_code": "399006",
            "direction": "increase",
            "operation_date": "2026-07-01",
            "change_percent": 25,
            "note": "API 建仓",
        },
    )
    assert created.status_code == 200
    event = created.json()
    assert event["index"] == "399006"
    assert event["date"] == "2026-07-01"
    assert event["change_percent"] == 25
    assert event["position_after"] == 25
    assert "product_name" not in event
    assert "quantity" not in event

    patched = client.patch(
        f"/api/investment-calendar/{event['id']}",
        json={"change_percent": 35, "note": "API 调整"},
    )
    assert patched.status_code == 200
    assert patched.json()["position_after"] == 35
    assert client.get("/api/investment-calendar/current-positions").json() == {
        "399006": 35,
        "159941": 0,
    }
    assert client.get(
        "/api/investment-calendar", params={"instrument_code": "399006"}
    ).json()[0]["note"] == "API 调整"

    deleted = client.delete(
        f"/api/investment-calendar/{event['id']}",
        params={"confirmed": "true"},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "id": event["id"]}
    assert client.get("/api/investment-calendar/current-positions").json() == {
        "399006": 0,
        "159941": 0,
    }


def test_position_event_api_validation_and_business_errors_have_expected_statuses(
    tmp_path,
) -> None:
    client = _client(tmp_path)
    base = {
        "instrument_code": "399006",
        "direction": "increase",
        "operation_date": "2026-07-01",
        "change_percent": 20,
    }

    for changes in (
        {"instrument_code": "000688"},
        {"direction": "hold"},
        {"change_percent": 7},
    ):
        response = client.post(
            "/api/investment-calendar",
            json={**base, **changes},
        )
        assert response.status_code == 422

    conflict = client.post(
        "/api/investment-calendar",
        json={**base, "direction": "decrease"},
    )
    assert conflict.status_code == 400
    assert conflict.json() == {"detail": "Invalid request"}

    missing_update = client.patch(
        "/api/investment-calendar/999999",
        json={"note": "missing"},
    )
    missing_delete = client.delete(
        "/api/investment-calendar/999999",
        params={"confirmed": "true"},
    )
    assert missing_update.status_code == 400
    assert missing_delete.status_code == 400
    assert missing_update.json() == {"detail": "Invalid request"}
    assert missing_delete.json() == {"detail": "Invalid request"}


def test_position_event_api_create_rejects_legacy_share_fields(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/investment-calendar",
        json={
            "instrument_code": "399006",
            "direction": "increase",
            "operation_date": "2026-07-01",
            "change_percent": 20,
            "product_name": "旧基金字段",
            "quantity": 100,
            "actual_price": 1.25,
        },
    )

    assert response.status_code == 422
    assert client.get("/api/investment-calendar").json() == []


def test_position_event_api_update_rejects_legacy_only_and_mixed_fields(tmp_path) -> None:
    client = _client(tmp_path)
    created = client.post(
        "/api/investment-calendar",
        json={
            "instrument_code": "399006",
            "direction": "increase",
            "operation_date": "2026-07-01",
            "change_percent": 20,
            "note": "原始备注",
        },
    ).json()

    legacy_only = client.patch(
        f"/api/investment-calendar/{created['id']}",
        json={"quantity": 100},
    )
    mixed = client.patch(
        f"/api/investment-calendar/{created['id']}",
        json={"note": "不应保存", "actual_price": 1.25},
    )

    assert legacy_only.status_code == 422
    assert mixed.status_code == 422
    persisted = client.get("/api/investment-calendar").json()[0]
    assert persisted["note"] == "原始备注"
    assert persisted["change_percent"] == 20


def test_v2_position_patch_and_delete_missing_event_are_404(tmp_path) -> None:
    client = _client(tmp_path)

    patched = client.patch(
        "/api/v2/position-events/999999",
        json={"note": "missing"},
    )
    deleted = client.delete(
        "/api/v2/position-events/999999",
        params={"confirmed": "true"},
    )

    assert patched.status_code == 404
    assert deleted.status_code == 404
    assert patched.json() == {"detail": "Resource not found"}


def test_v2_position_replay_conflict_is_fixed_safe_409(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/api/v2/position-events",
        json={
            "instrument_code": "399006",
            "direction": "decrease",
            "operation_date": "2026-07-01",
            "change_percent": 20,
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Request conflicts with current state"
    }


def test_v2_position_events_accepts_the_active_159941_calendar(tmp_path) -> None:
    client = _client(tmp_path)

    created = client.post(
        "/api/v2/position-events",
        json={
            "instrument_code": "159941",
            "direction": "increase",
            "operation_date": "2026-07-31",
            "change_percent": 10,
            "note": "159941 calendar regression",
        },
    )
    assert created.status_code == 200

    filtered = client.get(
        "/api/v2/position-events",
        params={"instrument_code": "159941"},
    )
    assert filtered.status_code == 200
    assert [row["instrument_code"] for row in filtered.json()] == ["159941"]

    benchmark = client.get(
        "/api/v2/position-events",
        params={"instrument_code": "NDX"},
    )
    assert benchmark.status_code == 400
