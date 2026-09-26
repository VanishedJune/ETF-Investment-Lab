"""V3.7 Phase-1 tests: multi-timeframe features, AI packet/cache, scope, chat."""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal

import pytest
import requests
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.migrations import run_migrations, SCHEMA_VERSION
from backend.app.database.session import create_database_engine
from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V37AiEvaluation,
    V37AiForecast,
    V37AiModelHealth,
    V37AiRequest,
    utc_now,
)
from backend.app.services import deepseek_client
from backend.app.services.deepseek_client import (
    DeepSeekConfig,
    DeepSeekError,
    chat_messages,
)
from backend.app.services.v37_ai_service import (
    build_ai_packet,
    ensure_ai_forecast,
    validate_ai_forecast,
)
from backend.app.services.v37_config import (
    AI_CALIBRATION_UNAVAILABLE,
    AI_SCOPE_FORWARD_OOS,
    AI_SCOPE_HISTORICAL_SCREENING,
    PROTOCOL_VERSION_37,
    REPAIR_PARAMETER_WHITELIST,
    ai_weight_cap,
)
from backend.app.services.v37_feature_service import V37FeatureService


def _session_dates(start: date, count: int) -> list[date]:
    result: list[date] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def _seed_market(engine, code: str = "399006", days: int = 320) -> None:
    rng = random.Random(11)
    price = 1000.0
    start = date(2023, 1, 2)
    with Session(engine) as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == code))
        if instrument is None:
            instrument = Instrument(
                code=code,
                name=code,
                exchange="SZSE",
                category="index" if code == "399006" else "etf",
                currency="CNY",
                is_active=True,
            )
            session.add(instrument)
            session.flush()
        for trade_date in _session_dates(start, days):
            previous = price
            price = max(100.0, price * (1.0 + rng.gauss(0.001, 0.012)))
            session.add(
                MarketPrice(
                    instrument_id=instrument.id,
                    timeframe="daily",
                    trade_date=trade_date,
                    open_price=Decimal(str(round(previous, 4))),
                    high_price=Decimal(str(round(max(previous, price) * 1.001, 4))),
                    low_price=Decimal(str(round(min(previous, price) * 0.999, 4))),
                    close_price=Decimal(str(round(price, 4))),
                    volume=Decimal(str(int(1000 + rng.random() * 500))),
                    source="TEST",
                    volume_multiplier=1,
                )
            )


@pytest.fixture()
def engine(tmp_path):
    engine = create_database_engine(tmp_path / "v37.db")
    run_migrations(engine)
    yield engine
    engine.dispose()


def test_schema_29_migration_and_idempotency(engine) -> None:
    assert SCHEMA_VERSION == 29
    with Session(engine) as session:
        rows = session.scalar(
            select(func.count()).select_from(V37AiRequest)
        )
        assert int(rows or 0) == 0
    run_migrations(engine)  # idempotent
    assert SCHEMA_VERSION == 29


def test_v37_feature_snapshot_builds_multi_timeframe(engine) -> None:
    _seed_market(engine)
    anchor = _session_dates(date(2023, 1, 2), 320)[-1]
    with Session(engine) as session:
        snapshot = V37FeatureService().load_snapshot(session, "399006", anchor)
        v37_features = {
            key: value for key, value in snapshot.features.items() if key.startswith("v37_")
        }
        assert len(v37_features) >= 55
        assert snapshot.provenance["multi_timeframe_state"] in {
            "BOTH_BULLISH",
            "BOTH_BEARISH",
            "DAILY_BULLISH_WEEKLY_BEARISH",
            "DAILY_BEARISH_WEEKLY_BULLISH",
            "DAILY_RECOVERY_WEEKLY_UNCONFIRMED",
            "WEEKLY_UPTREND_DAILY_PULLBACK",
            "NEUTRAL_MIXED",
        }
        manifest = V37FeatureService().build_manifest(snapshot)
        assert manifest.name == "EARLY_FUSION"
        assert "v37_d_ret_5" in manifest.ordered_feature_names
        assert "v37_x_direction_agreement" in manifest.ordered_feature_names


def test_ai_packet_anonymization(engine) -> None:
    _seed_market(engine)
    anchor = _session_dates(date(2023, 1, 2), 320)[-1]
    with Session(engine) as session:
        packet = build_ai_packet(session, "399006", anchor, anonymize=True)
        assert packet["market"] == "MARKET_A"
        assert packet["anchor"] == "T"
        assert packet["anchor_close"] == 1.0
        assert packet["daily"][-1]["date_relative"] == "T-0"
        assert packet["weekly"][-1]["week_relative"] == "T-0"
        for row in packet["daily"]:
            assert "399006" not in str(row)
            assert "2023" not in str(row)


def test_validate_ai_forecast_strict() -> None:
    valid = {
        "trend_1w": "BULLISH",
        "trend_2w": "NEUTRAL_BULLISH",
        "trend_4w": "NEUTRAL",
        "trend_8w": "NEUTRAL_BEARISH",
        "direction_score_1w": 0.72,
        "direction_score_2w": 0.68,
        "direction_score_4w": 0.61,
        "direction_score_8w": 0.53,
        "daily_trend": "BULLISH",
        "weekly_trend": "NEUTRAL",
        "multi_timeframe_state": "DAILY_RECOVERY_WEEKLY_UNCONFIRMED",
        "risk_level": "MEDIUM",
        "confidence_raw": 0.67,
        "expected_return_4w": 0.035,
        "expected_return_8w": 0.052,
        "support_distance_pct": -0.05,
        "resistance_distance_pct": 0.06,
        "reason_codes": ["DAILY_MACD_RECOVERY"],
    }
    result = validate_ai_forecast(valid)
    assert result["direction_scores"]["direction_score_8w"] == 0.53
    with pytest.raises(DeepSeekError):
        validate_ai_forecast({**valid, "trend_8w": "UP"})
    with pytest.raises(DeepSeekError):
        validate_ai_forecast({**valid, "confidence_raw": 1.5})
    with pytest.raises(DeepSeekError):
        validate_ai_forecast({**valid, "reason_codes": "not-a-list"})


def test_ai_weight_cap_tiers() -> None:
    assert ai_weight_cap(10) == 0
    assert ai_weight_cap(29) == 0
    assert ai_weight_cap(30) == 10
    assert ai_weight_cap(49) == 10
    assert ai_weight_cap(50) == 30
    assert ai_weight_cap(99) == 30
    assert ai_weight_cap(100) == 40


def test_repair_whitelist_has_14_items() -> None:
    assert len(REPAIR_PARAMETER_WHITELIST) == 14


def test_ensure_ai_forecast_retry_then_dedupe(engine, monkeypatch) -> None:
    import backend.app.services.v37_ai_service as ai_service

    _seed_market(engine)
    anchor = _session_dates(date(2023, 1, 2), 320)[-1]
    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")
    calls = {"count": 0}
    valid = {
        "trend_1w": "BULLISH",
        "trend_2w": "NEUTRAL",
        "trend_4w": "NEUTRAL",
        "trend_8w": "BEARISH",
        "direction_score_1w": 0.7,
        "direction_score_2w": 0.6,
        "direction_score_4w": 0.5,
        "direction_score_8w": 0.4,
        "daily_trend": "BULLISH",
        "weekly_trend": "NEUTRAL",
        "multi_timeframe_state": "NEUTRAL_MIXED",
        "risk_level": "HIGH",
        "confidence_raw": 0.5,
        "expected_return_4w": 0.0,
        "expected_return_8w": -0.02,
        "support_distance_pct": -0.05,
        "resistance_distance_pct": 0.05,
        "reason_codes": ["TEST"],
    }

    def fake_chat_json(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise DeepSeekError("NETWORK_ERROR", "transient")
        return valid, 12, {"prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(ai_service, "chat_json_detailed", fake_chat_json)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        first = ensure_ai_forecast(
            session,
            "399006",
            anchor,
            config,
            screening_scope=AI_SCOPE_FORWARD_OOS,
        )
        assert first["status"] == "OK"
        assert first["attempt"] == 3
        request_count = int(
            session.scalar(select(func.count()).select_from(V37AiRequest)) or 0
        )
        assert request_count == 3
        forecast_count = int(
            session.scalar(select(func.count()).select_from(V37AiForecast)) or 0
        )
        assert forecast_count == 1
        ok_request = session.scalar(
            select(V37AiRequest).where(
                V37AiRequest.response_status == "OK"
            )
        )
        assert ok_request is not None
        assert ok_request.token_usage_json.get("prompt_tokens") == 10
        cached = ensure_ai_forecast(
            session,
            "399006",
            anchor,
            config,
            screening_scope=AI_SCOPE_FORWARD_OOS,
        )
        assert cached["cached"] is True
        assert calls["count"] == 3


def test_ai_scope_isolation(engine) -> None:
    from backend.app.services.v37_ai_service import ai_calibration_status

    _seed_market(engine)
    anchor = _session_dates(date(2023, 1, 2), 320)[-1]
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        for scope, count in (
            (AI_SCOPE_HISTORICAL_SCREENING, 20),
            (AI_SCOPE_FORWARD_OOS, 5),
        ):
            for index in range(count):
                forecast_anchor = anchor - timedelta(days=7 * index)
                request = V37AiRequest(
                    protocol_version=PROTOCOL_VERSION_37,
                    model_market="399006",
                    forecast_anchor_date=forecast_anchor,
                    provider="deepseek",
                    model_name="m",
                    prompt_version="v1",
                    schema_version="v1",
                    ai_generation_version=1,
                    attempt_number=1,
                    screening_scope=scope,
                    input_hash=f"{scope}-{index}",
                    response_status="OK",
                    request_timestamp=utc_now(),
                    latency_ms=1,
                    token_usage_json={},
                    point_in_time_pass=True,
                    created_at=utc_now(),
                )
                session.add(request)
                session.flush()
                forecast = V37AiForecast(
                    protocol_version=PROTOCOL_VERSION_37,
                    model_market="399006",
                    forecast_anchor_date=forecast_anchor,
                    ai_request_id=request.id,
                    ai_generation_version=1,
                    model_name="m",
                    prompt_version="v1",
                    schema_version="v1",
                    screening_scope=scope,
                    input_hash=f"{scope}-{index}",
                    trend_1w="NEUTRAL",
                    trend_2w="NEUTRAL",
                    trend_4w="NEUTRAL",
                    trend_8w="NEUTRAL",
                    direction_scores_json={
                        "direction_score_1w": 0.5,
                        "direction_score_2w": 0.5,
                        "direction_score_4w": 0.5,
                        "direction_score_8w": 0.5,
                    },
                    daily_trend="NEUTRAL",
                    weekly_trend="NEUTRAL",
                    multi_timeframe_state="NEUTRAL_MIXED",
                    risk_level="LOW",
                    confidence_raw=Decimal("0.5"),
                    expected_return_4w=Decimal("0"),
                    expected_return_8w=Decimal("0"),
                    support_distance_pct=Decimal("0"),
                    resistance_distance_pct=Decimal("0"),
                    reason_codes_json=[],
                    status="PENDING",
                    structured_result_json={},
                    output_hash=f"h-{scope}-{index}",
                    created_at=utc_now(),
                )
                session.add(forecast)
                session.flush()
                if scope == AI_SCOPE_FORWARD_OOS:
                    session.add(
                        V37AiEvaluation(
                            ai_forecast_id=forecast.id,
                            protocol_version=PROTOCOL_VERSION_37,
                            model_market="399006",
                            forecast_anchor_date=forecast_anchor,
                            evaluation_available_date=anchor,
                            direction_hits_json={},
                            mae_4w=Decimal("0.01"),
                            mae_8w=Decimal("0.01"),
                            brier_4w=Decimal("0.25"),
                            brier_8w=Decimal("0.25"),
                            strong_up_recognized=False,
                            strong_down_recognized=False,
                            up_missed=False,
                            down_false_alarm=False,
                            metrics_json={"actual_return_8w": 0.0},
                            evaluation_hash=f"e-{scope}-{index}",
                            created_at=utc_now(),
                        )
                    )
                session.flush()
        status, count, _calibrator = ai_calibration_status(
            session, "399006", anchor
        )
        assert status == AI_CALIBRATION_UNAVAILABLE
        assert count == 5  # only FORWARD_OOS evaluations count


def test_chat_messages_validates_roles_and_errors(monkeypatch) -> None:
    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")
    with pytest.raises(DeepSeekError):
        chat_messages(
            config,
            system="s",
            messages=[{"role": "system", "content": "x"}],
        )
    with pytest.raises(DeepSeekError):
        chat_messages(config, system="s", messages=[{"role": "user", "content": ""}])

    def ok_response(*_args, **_kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = (
            '{"choices": [{"message": {"content": "hello"}}]}'
        ).encode()
        return response

    monkeypatch.setattr(deepseek_client.requests, "post", ok_response)
    reply, elapsed = chat_messages(
        config,
        system="s",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert reply == "hello"
    assert elapsed >= 0


def test_chat_messages_timeout_maps(monkeypatch) -> None:
    config = DeepSeekConfig(api_key="k", base_url="https://x", model="m")

    def raise_timeout(*_args, **_kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(deepseek_client.requests, "post", raise_timeout)
    with pytest.raises(DeepSeekError) as exc:
        chat_messages(
            config,
            system="s",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert exc.value.code == "TIMEOUT"


def test_validate_chat_messages_strict() -> None:
    from backend.app.api.deepseek import validate_chat_messages

    assert validate_chat_messages(
        [{"role": "user", "content": " hi "}]
    ) == [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError):
        validate_chat_messages([])
    with pytest.raises(ValueError):
        validate_chat_messages([{"role": "system", "content": "x"}])
    with pytest.raises(ValueError):
        validate_chat_messages([{"role": "user", "content": ""}])
    with pytest.raises(ValueError):
        validate_chat_messages(["not-a-dict"])


def test_fusion_position_grid_and_local_only_fallback(engine) -> None:
    from backend.app.services.v37_fusion_service import (
        V37FusionConfig,
        ensure_fusion_configs,
        fusion_position,
    )
    from backend.app.services.v35_config import default_strategy_config

    with Session(engine) as session:
        configs = ensure_fusion_configs(session, "399006")
        assert len(configs) == 4
        config = configs[0]  # LOCAL 90 / AI 10
        champion_config = default_strategy_config("399006")
        position = fusion_position(
            local_position_pp=30,
            local_score=60.0,
            ai_up_probability=None,
            config=config,
            champion_strategy_config=champion_config,
        )
        assert position % 5 == 0
        assert 0 <= position <= 80
        position_with_ai = fusion_position(
            local_position_pp=30,
            local_score=60.0,
            ai_up_probability=0.9,
            config=config,
            champion_strategy_config=champion_config,
            conflict_state="HIGH_MODEL_CONFLICT",
        )
        assert position_with_ai % 5 == 0


def test_validate_repair_proposal_whitelist() -> None:
    from backend.app.services.v37_repair_service import validate_repair_proposal

    valid = {
        "diagnosis": "DAILY_REVERSAL_UNDERWEIGHTED",
        "target_layer": "MULTI_TIMEFRAME_FUSION",
        "severity": "HIGH",
        "proposed_change": "INCREASE_DAILY_BRANCH_WEIGHT",
        "parameter_changes": {"daily_weekly_fusion_weight": 0.40},
        "expected_effect": "improve participation",
        "primary_risk": "more false entries",
        "requires_full_replay": True,
    }
    result = validate_repair_proposal(valid)
    assert result["parameter_changes"]["daily_weekly_fusion_weight"] == 0.40
    with pytest.raises(DeepSeekError):
        validate_repair_proposal(
            {**valid, "parameter_changes": {"not_allowed": 1.0}}
        )
    with pytest.raises(DeepSeekError):
        validate_repair_proposal(
            {**valid, "parameter_changes": {"ridge_alpha": 999.0}}
        )
    with pytest.raises(DeepSeekError):
        validate_repair_proposal({**valid, "parameter_changes": {}})


def test_repair_eligible_windows_purge(engine) -> None:
    from backend.app.services.v37_repair_service import repair_eligible_windows
    from backend.app.services.v37_feature_service import V37FeatureService

    _seed_market(engine)
    with Session(engine) as session:
        anchors = V37FeatureService().weekly_anchors(session, "399006")
        assert len(anchors) >= 30
        proposal_anchor = anchors[10]
        eligible = repair_eligible_windows(session, "399006", proposal_anchor)
        if eligible:
            assert eligible[0] > proposal_anchor
        # All eligible windows must start after the purge gap.
        assert all(anchor >= anchors[18] for anchor in eligible)


def test_ai_model_health_upsert_and_empty_status(engine) -> None:
    from backend.app.services.v37_ai_service import ai_model_health
    from backend.app.services.v37_config import AI_CALIBRATION_UNAVAILABLE

    _seed_market(engine)
    anchor = _session_dates(date(2023, 1, 2), 320)[-1]
    with Session(engine) as session, session.begin():
        row = ai_model_health(session, "399006", anchor)
        assert row.status == AI_CALIBRATION_UNAVAILABLE
        assert row.weight_cap_pp == 0
        row_id = row.id
        row_again = ai_model_health(session, "399006", anchor)
        assert row_again.id == row_id
        count = session.scalar(
            select(func.count()).select_from(V37AiModelHealth)
        )
        assert int(count or 0) == 1


def test_evaluate_ai_challenger_shadow_without_champion(engine) -> None:
    from backend.app.services.v37_ai_service import evaluate_ai_challenger

    _seed_market(engine)
    anchor = _session_dates(date(2023, 1, 2), 320)[-1]
    with Session(engine) as session, session.begin():
        result = evaluate_ai_challenger(session, "399006", anchor)
        assert result["decision"] == "AI_SHADOW"


def test_evaluate_repair_challenger_missing_package(engine) -> None:
    from backend.app.models.models import V37ModelRepairProposal, V37ModelRepairRun
    from backend.app.services.v37_repair_service import evaluate_repair_challenger

    _seed_market(engine)
    anchor = _session_dates(date(2023, 1, 2), 320)[-1]
    with Session(engine) as session, session.begin():
        proposal = V37ModelRepairProposal(
            protocol_version=PROTOCOL_VERSION_37,
            model_market="399006",
            anchor_date=anchor,
            trigger="TEST",
            diagnosis="d",
            target_layer="MULTI_TIMEFRAME_FUSION",
            severity="MEDIUM",
            proposed_change="c",
            parameter_changes_json={},
            expected_effect="e",
            primary_risk="r",
            requires_full_replay=True,
            architecture_change_proposal_json={},
            status="PENDING",
            proposal_hash="h",
            created_at=utc_now(),
        )
        session.add(proposal)
        session.flush()
        repair_run = V37ModelRepairRun(
            protocol_version=PROTOCOL_VERSION_37,
            model_market="399006",
            proposal_id=proposal.id,
            challenger_package_id=None,
            training_run_id=None,
            status="CREATED",
            validation_json={},
            created_at=utc_now(),
        )
        session.add(repair_run)
        session.flush()
        result = evaluate_repair_challenger(session, "399006", repair_run, None, anchor)
        assert result["decision"] == "FAILED"


def test_incremental_sync_no_new_week(engine) -> None:
    from backend.app.services.v37_runtime_service import V37RuntimeService

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    runtime = V37RuntimeService(factory)
    result = runtime.incremental_sync("399006")
    assert int(result["processed_weeks"]) == 0
    assert result["ai"]["status"] == "NO_NEW_WEEK"


def test_desktop_schema_self_check(tmp_path) -> None:
    import sqlite3

    import desktop_launcher

    home = tmp_path / "home"
    data = home / "data"
    data.mkdir(parents=True)
    db = data / "investment_lab.db"
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    assert (
        desktop_launcher._check_schema_compatibility(home, notify=False)
        is False
    )
    status = (data / "startup-status.json").read_text(encoding="utf-8")
    assert "SCHEMA_MISMATCH" in status

    connection = sqlite3.connect(db)
    connection.execute("PRAGMA user_version = 1")
    connection.close()
    assert (
        desktop_launcher._check_schema_compatibility(home, notify=False)
        is True
    )
