"""Unit tests for the DeepSeek AI settings/test integration."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import requests
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from backend.app.database.migrations import run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import (
    IndicatorRecord,
    Instrument,
    MarketPrice,
)
from backend.app.services import deepseek_client
from backend.app.services.deepseek_client import (
    ERROR_API_KEY_INVALID,
    ERROR_MODEL_NOT_AVAILABLE,
    ERROR_NETWORK,
    ERROR_RATE_LIMITED,
    ERROR_TIMEOUT,
    DeepSeekConfig,
    DeepSeekError,
    chat_messages,
    chat_json,
    load_config,
    mask_key,
    save_config,
    test_connection as deepseek_test_connection,
)
from backend.app.services.deepseek_market_payload import (
    build_market_context,
    validate_analysis,
)


def _write_config(monkeypatch, tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "deepseek.local.json"
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(deepseek_client, "config_path", lambda: path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    return path


def test_mask_key_never_exposes_full_secret() -> None:
    assert mask_key("sk-abcdef1234567890") == "sk-****7890"
    assert mask_key("short") == "****"


def test_load_config_file_and_env_override(monkeypatch, tmp_path) -> None:
    _write_config(
        monkeypatch,
        tmp_path,
        {"api_key": "file-key", "base_url": "https://example.com", "model": "m1"},
    )
    config = load_config()
    assert config.api_key == "file-key"
    assert config.base_url == "https://example.com"
    monkeypatch.setenv("DEEPSEEK_MODEL", "env-model")
    assert load_config().model == "env-model"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")
    assert load_config().api_key == "env-key"


def test_save_config_persists_and_masks(monkeypatch, tmp_path) -> None:
    path = _write_config(monkeypatch, tmp_path, {})
    config = save_config(api_key="sk-save-key", base_url="https://x", model="m")
    assert config.api_key == "sk-save-key"
    assert mask_key(config.api_key) == "sk-****-key"
    assert path.is_file()


def test_error_mapping(monkeypatch) -> None:
    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")

    def raise_timeout(*_args, **_kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(deepseek_client.requests, "post", raise_timeout)
    with pytest.raises(DeepSeekError) as exc:
        deepseek_test_connection(config)
    assert exc.value.code == ERROR_TIMEOUT

    def status_401(*_args, **_kwargs):
        return requests.Response()

    monkeypatch.setattr(deepseek_client.requests, "post", status_401)

    def fake_response(status: int):
        def _inner(*_args, **_kwargs):
            response = requests.Response()
            response.status_code = status
            response._content = b"{}"
            return response

        return _inner

    monkeypatch.setattr(deepseek_client.requests, "post", fake_response(401))
    with pytest.raises(DeepSeekError) as exc:
        deepseek_test_connection(config)
    assert exc.value.code == ERROR_API_KEY_INVALID

    monkeypatch.setattr(deepseek_client.requests, "post", fake_response(429))
    with pytest.raises(DeepSeekError) as exc:
        deepseek_test_connection(config)
    assert exc.value.code == ERROR_RATE_LIMITED

    monkeypatch.setattr(deepseek_client.requests, "post", fake_response(404))
    with pytest.raises(DeepSeekError) as exc:
        deepseek_test_connection(config)
    assert exc.value.code == ERROR_MODEL_NOT_AVAILABLE

    monkeypatch.setattr(deepseek_client.requests, "post", fake_response(500))
    with pytest.raises(DeepSeekError) as exc:
        deepseek_test_connection(config)
    assert exc.value.code == ERROR_NETWORK


def test_test_connection_ok(monkeypatch) -> None:
    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")

    def ok_response(*_args, **_kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = (
            '{"choices": [{"message": {"content": "DEEPSEEK_API_OK"}}]}'
        ).encode()
        return response

    monkeypatch.setattr(deepseek_client.requests, "post", ok_response)
    result = deepseek_test_connection(config)
    assert result["ok"] is True
    assert result["response"] == "DEEPSEEK_API_OK"


def test_chat_messages_maps_401_and_429(monkeypatch) -> None:
    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")

    def fake_response(status: int):
        def _inner(*_args, **_kwargs):
            response = requests.Response()
            response.status_code = status
            response._content = b"{}"
            return response

        return _inner

    monkeypatch.setattr(deepseek_client.requests, "post", fake_response(401))
    with pytest.raises(DeepSeekError) as exc:
        chat_messages(
            config,
            system="s",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert exc.value.code == ERROR_API_KEY_INVALID

    monkeypatch.setattr(deepseek_client.requests, "post", fake_response(429))
    with pytest.raises(DeepSeekError) as exc:
        chat_messages(
            config,
            system="s",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert exc.value.code == ERROR_RATE_LIMITED


def test_chat_json_extracts_fenced_response(monkeypatch) -> None:
    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")

    def fenced(*_args, **_kwargs):
        return (
            '```json\n{"trend_1_2w": "BULLISH", "decision": "HOLD"}\n```',
            12,
            {},
        )

    monkeypatch.setattr(deepseek_client, "_chat_messages_once", fenced)
    parsed, elapsed = chat_json(config, system="s", user="u")
    assert parsed["decision"] == "HOLD"
    assert elapsed == 12


def test_chat_json_detailed_returns_token_usage(monkeypatch) -> None:
    from backend.app.services.deepseek_client import chat_json_detailed

    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")

    def ok_response(*_args, **_kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = (
            '{"choices": [{"message": {"content": "{\\"ok\\": true}"}}],'
            '"usage": {"prompt_tokens": 12, "completion_tokens": 3,'
            '"total_tokens": 15}}'
        ).encode()
        return response

    monkeypatch.setattr(deepseek_client.requests, "post", ok_response)
    parsed, elapsed, usage = chat_json_detailed(config, system="s", user="u")
    assert parsed == {"ok": True}
    assert usage == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    assert elapsed >= 0


def test_validate_analysis_strict() -> None:
    valid = validate_analysis(
        {
            "trend_1_2w": "BULLISH",
            "trend_4w": "NEUTRAL",
            "trend_8w": "BEARISH",
            "confidence": 0.7,
            "risk_level": "MEDIUM",
            "summary": "短期偏强，中期震荡。",
            "decision": "HOLD",
            "decision_reason": "短期反弹但中期偏弱。",
        }
    )
    assert valid["confidence"] == 0.7
    assert valid["decision"] == "HOLD"
    with pytest.raises(DeepSeekError):
        validate_analysis(
            {
                "trend_1_2w": "UP",
                "trend_4w": "NEUTRAL",
                "trend_8w": "BEARISH",
                "confidence": 0.7,
                "risk_level": "MEDIUM",
                "summary": "x",
                "decision": "HOLD",
                "decision_reason": "x",
            }
        )
    with pytest.raises(DeepSeekError):
        validate_analysis(
            {
                "trend_1_2w": "BULLISH",
                "trend_4w": "NEUTRAL",
                "trend_8w": "BEARISH",
                "confidence": 1.2,
                "risk_level": "MEDIUM",
                "summary": "x",
                "decision": "HOLD",
                "decision_reason": "x",
            }
        )
    with pytest.raises(DeepSeekError):
        validate_analysis(
            {
                "trend_1_2w": "BULLISH",
                "trend_4w": "NEUTRAL",
                "trend_8w": "BEARISH",
                "confidence": 0.7,
                "risk_level": "MEDIUM",
                "summary": "x",
                "decision": "BUY_NOW",
                "decision_reason": "x",
            }
        )


def _seed_market(engine) -> None:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        instrument = session.scalar(
            select(Instrument).where(Instrument.code == "399006")
        )
        if instrument is None:
            instrument = Instrument(
                code="399006",
                name="创业板指数",
                exchange="SZSE",
                category="index",
                currency="CNY",
                is_active=True,
                extra_data={},
            )
            session.add(instrument)
            session.flush()
        for index in range(10):
            session.add(
                MarketPrice(
                    instrument_id=instrument.id,
                    timeframe="daily",
                    trade_date=date(2026, 7, 20 + index),
                    open_price=Decimal("100.00") + index,
                    high_price=Decimal("102.00") + index,
                    low_price=Decimal("99.00") + index,
                    close_price=Decimal("101.00") + index,
                    volume=Decimal("1000"),
                    source="TEST",
                    volume_multiplier=1,
                )
            )
            session.add(
                MarketPrice(
                    instrument_id=instrument.id,
                    timeframe="weekly",
                    trade_date=date(2026, 7, 24) + timedelta(days=7 * index),
                    open_price=Decimal("100.00") + index,
                    high_price=Decimal("102.00") + index,
                    low_price=Decimal("99.00") + index,
                    close_price=Decimal("101.00") + index,
                    volume=Decimal("5000"),
                    source="TEST",
                    volume_multiplier=1,
                )
            )
        session.add(
            IndicatorRecord(
                instrument_id=instrument.id,
                timeframe="weekly",
                indicator_name="technical_indicators/v2_partial_window",
                indicator_date=date(2026, 8, 2),
                indicator_values={
                    "values": {
                        "ma5": 105.0,
                        "ma10": 104.0,
                        "ma20": 103.0,
                        "ma60": 102.0,
                        "dif": 0.5,
                        "dea": 0.4,
                        "macd_histogram": 0.2,
                    }
                },
            )
        )


def test_build_market_context_reads_existing_data(tmp_path) -> None:
    engine = create_database_engine(tmp_path / "ai.db")
    run_migrations(engine)
    _seed_market(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        context = build_market_context(session, "399006")
    assert context["latest_close"] is not None
    assert context["ma"]["ma20"] == 103.0
    assert context["dif"] == 0.5
    assert context["daily"]["closes"]
    assert "ma20" in context["daily"]["ma"]
    assert context["forecast"] is None
    engine.dispose()
