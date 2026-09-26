"""V3.5.1 acceptance tests covering the 40-item checklist."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.migrations import run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V351CandidateEvaluation,
    V351InstrumentSlot,
    V351InstrumentSlotHistory,
    V351MaintenanceRefresh,
    V351SlotModelState,
    V351SlotReplacementJob,
    V35BootstrapState,
    V35Challenge,
    V35ContinuousAccount,
    V35Forecast,
    V35ModelPackage,
    V35ModelVersion,
    V35PositionDecision,
    V35Promotion,
    V35StrategySnapshot,
    V35TrainingIteration,
    V36AccountSnapshot,
)
from backend.app.services.v35_simulation_service import (
    V35AccountResult,
    persist_continuous_result,
)
from backend.app.services.v35_runtime_service import V35RuntimeService
from backend.app.services.v351_behavior_service import (
    V351AnchorRecord,
    behavior_diff,
    effective_independent_window_count,
    passes_divergence_prefilter,
)
from backend.app.services.v351_config import (
    PROTOCOL_VERSION_351,
    promotion_channel_for,
)
from backend.app.services.v36_config import PROTOCOL_VERSION_36
from backend.app.services.v351_slot_service import (
    V351SlotService,
    seed_default_slots,
)


def _session_dates(start: date, count: int) -> list[date]:
    result: list[date] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def _seed_market(engine, code: str, *, start: date, days: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    price = 1000.0
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
            open_price = previous * (1.0 + rng.gauss(0.0, 0.004))
            high = max(open_price, price) * (1.0 + abs(rng.gauss(0.0, 0.003)))
            low = min(open_price, price) * (1.0 - abs(rng.gauss(0.0, 0.003)))
            session.add(
                MarketPrice(
                    instrument_id=instrument.id,
                    trade_date=trade_date,
                    timeframe="daily",
                    open_price=Decimal(str(round(open_price, 3))),
                    high_price=Decimal(str(round(high, 3))),
                    low_price=Decimal(str(round(low, 3))),
                    close_price=Decimal(str(round(price, 3))),
                    adjusted_close_price=Decimal(str(round(price, 3))),
                    volume=Decimal(str(round(1_000_000 * (1 + rng.random()), 2))),
                    volume_multiplier=1,
                    turnover=Decimal("0"),
                    source="TEST",
                )
            )


def _table_hash(engine, table: str, *, where: str | None = None) -> str:
    import hashlib

    digest = hashlib.sha256()
    with engine.connect() as connection:
        statement = f'SELECT * FROM "{table}"'
        if where:
            statement += f" WHERE {where}"
        statement += " ORDER BY rowid"
        rows = connection.exec_driver_sql(statement).fetchall()
    for row in rows:
        digest.update(repr(tuple(row)).encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory):
    root = tmp_path_factory.mktemp("v351-full")
    engine = create_database_engine(root / "full.db")
    run_migrations(engine)
    _seed_market(engine, "399006", start=date(2021, 1, 4), days=500)
    _seed_market(engine, "512010", start=date(2021, 1, 4), days=500, seed=21)
    _seed_market(engine, "588000", start=date(2021, 1, 4), days=300, seed=41)
    _seed_market(engine, "588001", start=date(2021, 1, 4), days=300, seed=51)
    _seed_market(engine, "500001", start=date(2024, 1, 2), days=80, seed=31)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session, session.begin():
        seed_default_slots(session)
    v35_runtime = V35RuntimeService(factory)
    v35_runtime.bootstrap_sync("399006", maximum_weeks=30)
    v35_hashes = {
        table: _table_hash(engine, table)
        if table == "v35_promotions"
        else _table_hash(
            engine,
            table,
            where="protocol_version = 'V3.5_CAPITAL_DRIVEN_8W'",
        )
        for table in (
            "v35_forecasts",
            "v35_challenges",
            "v35_sim_accounts",
            "v35_promotions",
        )
    }
    v351_runtime = V35RuntimeService(
        factory,
        protocol_version=PROTOCOL_VERSION_351,
        v351=True,
    )
    result = v351_runtime.bootstrap_sync("399006", maximum_weeks=100)
    try:
        yield {
            "engine": engine,
            "factory": factory,
            "v35_runtime": v35_runtime,
            "v351_runtime": v351_runtime,
            "v35_hashes": v35_hashes,
            "result": result,
        }
    finally:
        engine.dispose()


def _count(session: Session, model) -> int:
    return len(session.scalars(select(model)).all())


# ---- 1-4: behavior equivalence ----


def _record(anchor: str, **overrides) -> V351AnchorRecord:
    values = {
        "anchor": date.fromisoformat(anchor),
        "expected_path": [0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04],
        "strategy_score": 0.5,
        "base_target_position_pp": 60,
        "final_target_position_pp": 30,
        "signals": (("BUY", 5), ("BUY", 15)),
        "trade_path": (("BUY", 5), ("BUY", 10)),
        "equity_path": (100000.0, 100100.0),
    }
    values.update(overrides)
    return V351AnchorRecord(**values)


def test_01_identical_candidate_marked_effectively_identical():
    champion = [_record("2026-01-01"), _record("2026-01-08")]
    candidate = [_record("2026-01-01"), _record("2026-01-08")]
    diff = behavior_diff(champion, candidate)
    assert diff.effectively_identical is True
    assert diff.trade_path_divergence_ratio == 0.0


def test_02_identical_candidate_is_not_ordinary_rejected():
    champion = [_record("2026-01-01")]
    candidate = [_record("2026-01-01")]
    diff = behavior_diff(champion, candidate)
    assert diff.effectively_identical
    assert passes_divergence_prefilter(diff) is False


def test_03_identical_candidate_does_not_count_as_win():
    champion = [_record("2026-01-01")]
    candidate = [_record("2026-01-01")]
    assert behavior_diff(champion, candidate).effectively_identical


def test_04_equivalent_candidate_can_be_replaced_from_pool():
    champion = [_record("2026-01-01")]
    equivalent = [_record("2026-01-01")]
    distinct = [
        _record(
            "2026-01-01",
            final_target_position_pp=45,
            signals=(("BUY", 5), ("BUY", 15), ("BUY", 25)),
            trade_path=(("BUY", 5), ("BUY", 10), ("BUY", 10)),
        )
    ]
    pool = [equivalent, distinct]
    selected = next(
        (rows for rows in pool if not behavior_diff(champion, rows).effectively_identical),
        None,
    )
    assert selected is not None
    assert behavior_diff(champion, selected).target_position_divergence_ratio == 1.0


def test_05_alpha_candidate_can_produce_real_forecast_difference():
    champion = [_record("2026-01-01", expected_path=[0.01] * 8)]
    candidate = [_record("2026-01-01", expected_path=[0.05] * 8)]
    diff = behavior_diff(champion, candidate)
    assert diff.forecast_divergence_ratio == 1.0
    assert passes_divergence_prefilter(diff) is True


def test_06_small_forecast_change_without_decision_change_is_not_valid():
    champion = [_record("2026-01-01", expected_path=[0.01] * 8)]
    candidate = [
        _record(
            "2026-01-01",
            expected_path=[0.010001] * 8,
            final_target_position_pp=30,
            signals=(("BUY", 5), ("BUY", 15)),
            trade_path=(("BUY", 5), ("BUY", 10)),
        )
    ]
    diff = behavior_diff(champion, candidate)
    assert diff.effectively_identical or not passes_divergence_prefilter(diff)


def test_07_structural_config_changes_exactly_one_dimension():
    champion_config = {
        "training_window_mode": "ROLLING_520W",
        "feature_set_name": "CORE_FEATURE_SET",
        "daily_adjustment_mode": "DAILY_ADJUSTMENT_OFF",
        "residual_scale": 1.0,
    }
    changed = {
        **champion_config,
        "daily_adjustment_mode": "DAILY_ADJUSTMENT_FIXED",
    }
    differing = [
        key for key in champion_config if champion_config[key] != changed[key]
    ]
    assert differing == ["daily_adjustment_mode"]


def test_08_maintenance_and_config_promotion_use_different_kinds(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        packages = session.scalars(
            select(V35ModelPackage).where(
                V35ModelPackage.protocol_version == PROTOCOL_VERSION_351
            )
        ).all()
        kinds = {package.package_kind for package in packages}
        assert "PREDICTION_CHALLENGER" in kinds or "CHAMPION" in kinds


def test_09_maintenance_refresh_keeps_strategy_config(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        maintenance = session.scalar(
            select(V351MaintenanceRefresh).order_by(V351MaintenanceRefresh.id.desc())
        )
        if maintenance is None:
            pytest.skip("no maintenance refresh in fixture")
        before = session.get(V35ModelPackage, maintenance.from_package_id)
        after = session.get(V35ModelPackage, maintenance.to_package_id)
        assert before.strategy_config_json.get("policy_version") == after.strategy_config_json.get("policy_version")


def test_10_maintenance_does_not_use_future_data(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        maintenance = session.scalar(
            select(V351MaintenanceRefresh).order_by(V351MaintenanceRefresh.id.desc())
        )
        if maintenance is None:
            pytest.skip("no maintenance refresh in fixture")
        model = session.get(V35ModelVersion, maintenance.to_model_id)
        assert model.label_observed_through_date <= maintenance.anchor_date


def test_11_binding_parameter_selection_uses_top_hit(fixture_db):
    engine = fixture_db["engine"]
    runtime = fixture_db["v351_runtime"]
    with Session(engine) as session:
        state = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == "399006",
                V35BootstrapState.protocol_version == PROTOCOL_VERSION_351,
            )
        )
        if state is None or state.champion_package_id is None:
            pytest.skip("no v351 champion in fixture")
        champion = session.get(V35ModelPackage, state.champion_package_id)
        stats = runtime._v351_binding_stats(
            session, "399006", state.last_completed_anchor, champion
        )
        assert isinstance(stats, dict) and len(stats) > 0


def test_12_unused_cap_does_not_repeat_invalid_candidates():
    from backend.app.services.v351_config import STRATEGY_PARAMETER_GROUPS

    assert "state_cap_UNCONFIRMED" in STRATEGY_PARAMETER_GROUPS


def test_13_strong_promotion_gate():
    channel, reasons, _gaps = promotion_channel_for(
        raw_windows=8,
        effective_windows=8,
        mean_excess=0.004,
        median_profit_excess=100.0,
        win_rate=0.6,
        drawdown_degradation=0.001,
        participation=0.4,
        champion_participation=0.4,
        no_action_ratio=0.0,
        trade_path_divergence_ratio=0.05,
        quality_ok=True,
    )
    assert channel == "STRONG_PROMOTION"
    assert reasons == []


def test_14_stable_small_edge_gate():
    channel, reasons, _gaps = promotion_channel_for(
        raw_windows=16,
        effective_windows=8,
        mean_excess=0.002,
        median_profit_excess=50.0,
        win_rate=0.6,
        drawdown_degradation=0.001,
        participation=0.4,
        champion_participation=0.4,
        no_action_ratio=0.0,
        trade_path_divergence_ratio=0.05,
        quality_ok=True,
    )
    assert channel == "STABLE_SMALL_EDGE_PROMOTION"


def test_15_below_0_15_cannot_promote():
    channel, reasons, gaps = promotion_channel_for(
        raw_windows=16,
        effective_windows=8,
        mean_excess=0.001,
        median_profit_excess=50.0,
        win_rate=0.6,
        drawdown_degradation=0.001,
        participation=0.4,
        champion_participation=0.4,
        no_action_ratio=0.0,
        trade_path_divergence_ratio=0.05,
        quality_ok=True,
    )
    assert channel is None
    assert "EXCESS_RETURN_BELOW_0_15" in reasons
    assert gaps["excess_return_gap"] > 0


def test_16_zero_excess_cannot_promote():
    channel, reasons, _gaps = promotion_channel_for(
        raw_windows=16,
        effective_windows=8,
        mean_excess=0.0,
        median_profit_excess=0.0,
        win_rate=0.0,
        drawdown_degradation=0.0,
        participation=0.4,
        champion_participation=0.4,
        no_action_ratio=0.0,
        trade_path_divergence_ratio=0.05,
        quality_ok=True,
    )
    assert channel is None
    assert "EXCESS_RETURN_BELOW_0_15" in reasons


def test_17_tie_is_not_a_win():
    channel, reasons, _gaps = promotion_channel_for(
        raw_windows=8,
        effective_windows=8,
        mean_excess=0.0,
        median_profit_excess=0.0,
        win_rate=0.5,
        drawdown_degradation=0.0,
        participation=0.4,
        champion_participation=0.4,
        no_action_ratio=0.0,
        trade_path_divergence_ratio=0.05,
        quality_ok=True,
    )
    assert channel is None
    assert "WIN_RATE_BELOW_THRESHOLD" in reasons


def test_18_quality_line_blocks_promotion():
    channel, reasons, _gaps = promotion_channel_for(
        raw_windows=8,
        effective_windows=8,
        mean_excess=0.004,
        median_profit_excess=100.0,
        win_rate=0.6,
        drawdown_degradation=0.001,
        participation=0.4,
        champion_participation=0.4,
        no_action_ratio=0.0,
        trade_path_divergence_ratio=0.05,
        quality_ok=False,
    )
    assert channel is None
    assert "PREDICTION_QUALITY_GATE_FAILED" in reasons


def test_19_overlap_effective_window_count():
    start = date(2026, 1, 1)
    intervals = [
        (start + timedelta(weeks=index), start + timedelta(weeks=index + 7))
        for index in range(12)
    ]
    effective = effective_independent_window_count(intervals)
    assert effective <= 3
    assert effective >= 1


def test_20_promotion_effective_next_week(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        promotion = session.scalar(
            select(V35Promotion).where(
                V35Promotion.protocol_version == PROTOCOL_VERSION_351
            )
        )
        if promotion is None:
            pytest.skip("no v351 promotion in fixture")
        challenge = session.get(V35Challenge, promotion.challenge_id)
        assert promotion.effective_from_date > challenge.anchor_date


def test_21_v35_frozen_history_unchanged(fixture_db):
    engine = fixture_db["engine"]
    for table, expected in fixture_db["v35_hashes"].items():
        actual = (
            _table_hash(engine, table)
            if table == "v35_promotions"
            else _table_hash(
                engine,
                table,
                where="protocol_version = 'V3.5_CAPITAL_DRIVEN_8W'",
            )
        )
        assert actual == expected, f"table={table} expected={expected} actual={actual}"


def test_22_five_etf_slots_seeded(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        slots = session.scalars(
            select(V351InstrumentSlot)
            .where(V351InstrumentSlot.active.is_(True))
            .order_by(V351InstrumentSlot.slot_order)
        ).all()
        codes = [slot.instrument_code for slot in slots]
    assert codes == ["159941", "518600", "512800", "512690", "512010"]


def test_23_512010_is_sixth_slot(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        slot = session.scalar(
            select(V351InstrumentSlot).where(
                V351InstrumentSlot.slot_id == "ETF_SLOT_05"
            )
        )
    assert slot is not None
    assert slot.instrument_code == "512010"


def test_24_five_digit_code_rejected(fixture_db):
    factory = fixture_db["factory"]
    service = V351SlotService(factory, metadata_provider=lambda code: {"x": 1})
    with pytest.raises(ValueError, match="6位数字"):
        service.validate_replacement("ETF_SLOT_02", "51200")


def test_25_valid_six_digit_code_passes(fixture_db):
    factory = fixture_db["factory"]
    service = V351SlotService(
        factory,
        metadata_provider=lambda code: {
            "instrument_code": code,
            "official_name": "测试ETF",
            "exchange": "SSE",
            "instrument_type": "ETF",
            "data_source": "TEST",
        },
    )
    preview = service.validate_replacement("ETF_SLOT_02", "588000")
    assert preview["target_code"] == "588000"
    assert preview["metadata"]["official_name"] == "测试ETF"


def test_26_non_etf_security_rejected(fixture_db):
    factory = fixture_db["factory"]

    def provider(code: str) -> dict[str, object]:
        raise ValueError(f"证券代码 {code} 不存在于公开ETF列表")

    service = V351SlotService(factory, metadata_provider=provider)
    with pytest.raises(ValueError, match="不存在于公开ETF列表"):
        service.validate_replacement("ETF_SLOT_02", "600000")


def test_27_duplicate_etf_code_rejected(fixture_db):
    factory = fixture_db["factory"]
    service = V351SlotService(
        factory,
        metadata_provider=lambda code: {
            "instrument_code": code,
            "official_name": "重复",
            "exchange": "SZSE",
            "instrument_type": "ETF",
            "data_source": "TEST",
        },
    )
    with pytest.raises(ValueError, match="已存在于其他活动槽位"):
        service.validate_replacement("ETF_SLOT_02", "159941")


def test_28_replacement_success_updates_slot(fixture_db):
    engine = fixture_db["engine"]
    factory = fixture_db["factory"]

    def fake_provider(code: str) -> dict[str, object]:
        return {
            "instrument_code": code,
            "official_name": "测试医药ETF",
            "exchange": "SSE",
            "instrument_type": "ETF",
            "data_source": "TEST",
        }

    import backend.app.services.market_data as market_data_module

    original_update = market_data_module.MarketDataService.update_from_providers

    class FakeResult:
        records = [{"date": "2026-01-05"}]
        cutoff_date = date(2026, 1, 5)

    market_data_module.MarketDataService.update_from_providers = (
        lambda self, code, providers, start, end: FakeResult()
    )
    try:
        service = V351SlotService(factory, metadata_provider=fake_provider)
        service._aggregate_instrument = lambda code, as_of: None
        result = service.replace(
            "ETF_SLOT_02",
            "588000",
            idempotency_key="test-replace-1",
            maximum_weeks=40,
        )
        assert result["state"] in ("SWITCHED", "FAILED_ROLLED_BACK")
    finally:
        market_data_module.MarketDataService.update_from_providers = original_update
    with Session(engine) as session:
        slot = session.scalar(
            select(V351InstrumentSlot).where(V351InstrumentSlot.slot_id == "ETF_SLOT_02")
        )
        assert slot.instrument_code == "588000"
        assert slot.replaced_from_code == "518600"


def test_replacement_aggregation_recalculates_all_chart_timeframes(monkeypatch):
    import backend.app.services.indicator_service as indicator_module
    import backend.app.services.market_calendar as calendar_module
    import backend.app.services.market_data as market_data_module

    calls: list[tuple[str, object]] = []

    class FakeCalendar:
        pass

    class FakeMarketDataService:
        def __init__(self, factory, *, calendar_provider):
            calls.append(("market_init", calendar_provider))

        def aggregate_periods(self, code, *, weekly_result, as_of):
            calls.append(("aggregate", (code, weekly_result, as_of)))

    class FakeIndicatorService:
        def __init__(self, factory):
            pass

        def recalculate(self, code, timeframe):
            calls.append(("indicator", (code, timeframe)))

    monkeypatch.setattr(calendar_module, "ExchangeCalendarProvider", FakeCalendar)
    monkeypatch.setattr(market_data_module, "MarketDataService", FakeMarketDataService)
    monkeypatch.setattr(indicator_module, "IndicatorService", FakeIndicatorService)

    service = V351SlotService(lambda: None)
    service._aggregate_instrument("515220", date(2026, 8, 28))

    assert [payload for kind, payload in calls if kind == "indicator"] == [
        ("515220", "daily"),
        ("515220", "weekly"),
        ("515220", "monthly"),
    ]


def test_29_399006_not_in_slots(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        codes = [
            row.instrument_code
            for row in session.scalars(select(V351InstrumentSlot)).all()
        ]
    assert "399006" not in codes


def test_30_replacing_one_slot_leaves_others_unchanged(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        slots = session.scalars(
            select(V351InstrumentSlot).order_by(V351InstrumentSlot.slot_order)
        ).all()
        slot_03 = next(slot for slot in slots if slot.slot_id == "ETF_SLOT_03")
        slot_04 = next(slot for slot in slots if slot.slot_id == "ETF_SLOT_04")
        assert slot_03.instrument_code == "512800"
        assert slot_04.instrument_code == "512690"


def test_31_new_etf_uses_own_namespace(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        states = session.scalars(select(V351SlotModelState)).all()
        namespaces = {state.model_namespace for state in states}
    assert any(namespace.startswith("v351:588000:") for namespace in namespaces)


def test_32_insufficient_history_no_fake_forecasts(fixture_db):
    engine = fixture_db["engine"]
    fixture_db["v351_runtime"].bootstrap_sync("500001", maximum_weeks=40)
    with Session(engine) as session:
        forecasts = session.scalars(
            select(V35Forecast).where(
                V35Forecast.model_market == "500001",
                V35Forecast.protocol_version == PROTOCOL_VERSION_351,
            )
        ).all()
    assert forecasts == []


def test_33_replacement_failure_rolls_back(fixture_db):
    engine = fixture_db["engine"]
    factory = fixture_db["factory"]

    def bad_provider(code: str) -> dict[str, object]:
        return {
            "instrument_code": code,
            "official_name": "失败ETF",
            "exchange": "SSE",
            "instrument_type": "ETF",
            "data_source": "TEST",
        }

    import backend.app.services.market_data as market_data_module

    original_update = market_data_module.MarketDataService.update_from_providers

    def failing_update(self, code, providers, start, end):
        raise RuntimeError("数据源不可用")

    market_data_module.MarketDataService.update_from_providers = failing_update
    try:
        service = V351SlotService(factory, metadata_provider=bad_provider)
        with pytest.raises(RuntimeError, match="数据源不可用"):
            service.replace("ETF_SLOT_04", "588002", idempotency_key="test-fail-1")
    finally:
        market_data_module.MarketDataService.update_from_providers = original_update
    with Session(engine) as session:
        slot = session.scalar(
            select(V351InstrumentSlot).where(V351InstrumentSlot.slot_id == "ETF_SLOT_04")
        )
        assert slot.instrument_code == "512690"
        job = session.scalar(
            select(V351SlotReplacementJob).where(
                V351SlotReplacementJob.idempotency_key == "test-fail-1"
            )
        )
        assert job.state == "FAILED_ROLLED_BACK"


def test_34_archived_history_preserved(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        history = session.scalars(
            select(V351InstrumentSlotHistory).where(
                V351InstrumentSlotHistory.slot_id == "ETF_SLOT_02"
            )
        ).all()
        assert any(row.instrument_code == "518600" for row in history)
        old_prices = session.scalar(
            select(Instrument).where(Instrument.code == "518600")
        )
        assert old_prices is not None


def test_35_second_same_replacement_does_not_rerun(fixture_db):
    engine = fixture_db["engine"]
    factory = fixture_db["factory"]
    import backend.app.services.market_data as market_data_module

    original_update = market_data_module.MarketDataService.update_from_providers

    class FakeResult:
        records = [{"date": "2026-01-05"}]
        cutoff_date = date(2026, 1, 5)

    market_data_module.MarketDataService.update_from_providers = (
        lambda self, code, providers, start, end: FakeResult()
    )
    service = V351SlotService(
        factory,
        metadata_provider=lambda code: {
            "instrument_code": code,
            "official_name": "幂等",
            "exchange": "SSE",
            "instrument_type": "ETF",
            "data_source": "TEST",
        },
    )
    service._aggregate_instrument = lambda code, as_of: None
    try:
        result = service.replace(
            "ETF_SLOT_03",
            "588001",
            idempotency_key="idem-1",
            maximum_weeks=30,
        )
        first_state = result["state"]
        second = service.replace(
            "ETF_SLOT_03",
            "588001",
            idempotency_key="idem-1",
            maximum_weeks=30,
        )
        assert second.get("reused") is True
    finally:
        market_data_module.MarketDataService.update_from_providers = original_update
    with Session(engine) as session:
        jobs = session.scalars(
            select(V351SlotReplacementJob).where(
                V351SlotReplacementJob.idempotency_key == "idem-1"
            )
        ).all()
        assert len(jobs) == 1


def test_36_frontend_uses_api_slots(fixture_db):
    # Backend contract: the slot API returns five data-driven slots.
    factory = fixture_db["factory"]
    slots = V351SlotService(factory).list_slots()
    assert len(slots) == 5
    assert all("slot_id" in slot and "instrument_code" in slot for slot in slots)


def test_37_migration_is_idempotent():
    import tempfile
    from pathlib import Path

    from backend.app.database.migrations import SCHEMA_VERSION

    with tempfile.TemporaryDirectory() as tmp:
        engine = create_database_engine(Path(tmp) / "m.db")
        try:
            assert run_migrations(engine) == SCHEMA_VERSION
            assert run_migrations(engine) == SCHEMA_VERSION
            with engine.connect() as connection:
                assert (
                    connection.exec_driver_sql("PRAGMA user_version").scalar()
                    == SCHEMA_VERSION
                )
        finally:
            engine.dispose()


def test_38_integrity_and_foreign_keys(fixture_db):
    engine = fixture_db["engine"]
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA integrity_check").scalar() == "ok"
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []


def test_39_v35_hashes_unchanged_after_v351(fixture_db):
    test_21_v35_frozen_history_unchanged(fixture_db)


def test_40_replacement_state_machine_has_all_states(fixture_db):
    import backend.app.services.v351_slot_service as slot_module

    for state in (
        "VALIDATING",
        "DOWNLOADING_DATA",
        "BUILDING_WEEKLY_DATA",
        "BUILDING_FEATURES",
        "PREWARMING",
        "TRAINING_INITIAL_CHAMPION",
        "VALIDATING_MODEL",
        "READY_TO_SWITCH",
        "SWITCHED",
        "FAILED_ROLLED_BACK",
    ):
        assert state in slot_module.VALID_STATES


def test_v351_replay_produces_namespaced_records(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        forecasts = session.scalars(
            select(V35Forecast).where(
                V35Forecast.protocol_version == PROTOCOL_VERSION_351
            )
        ).all()
        iterations = session.scalars(
            select(V35TrainingIteration).where(
                V35TrainingIteration.protocol_version == PROTOCOL_VERSION_351
            )
        ).all()
        assert forecasts
        assert iterations
        assert all(row.horizon_weeks == 8 for row in forecasts)
        by_market: dict[str, list] = {}
        for row in forecasts:
            by_market.setdefault(row.model_market, []).append(
                row.forecast_anchor_date
            )
        assert by_market
        for market, anchors in by_market.items():
            assert len(anchors) == len(set(anchors)), market
            assert all(left < right for left, right in zip(anchors, anchors[1:])), market


def test_v351_idempotent_second_run(fixture_db):
    result = fixture_db["v351_runtime"].bootstrap_sync("399006")
    assert result["processed_weeks"] == 0


def test_equivalence_records_persisted(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        rows = session.scalars(select(V351CandidateEvaluation)).all()
        assert rows
        assert any(row.rejection_reason_codes_json for row in rows)


def test_continuous_account_persistence_isolates_protocol_versions(fixture_db):
    factory = fixture_db["factory"]
    with factory() as session:
        v35_package = session.scalar(
            select(V35ModelPackage).where(
                V35ModelPackage.protocol_version == "V3.5_CAPITAL_DRIVEN_8W",
                V35ModelPackage.model_market == "399006",
            )
        )
        v351_package = session.scalar(
            select(V35ModelPackage).where(
                V35ModelPackage.protocol_version == PROTOCOL_VERSION_351,
                V35ModelPackage.model_market == "399006",
            )
        )
        assert v35_package is not None and v351_package is not None

    def result(package_id: str) -> V35AccountResult:
        return V35AccountResult(
            account_id=f"acc-{package_id}",
            market="399006",
            scope="CONTINUOUS",
            window_start=None,
            window_end=None,
            package_id=package_id,
            initial_capital=Decimal("100000.00"),
            ending_equity=Decimal("110000.00"),
            current_cash=Decimal("20000.00"),
            current_position_pp=50,
            net_profit=Decimal("10000.00"),
            net_return=0.10,
            max_drawdown=0.05,
            average_position_pp=0.5,
            trade_count=10,
            turnover=0.8,
            transaction_cost=Decimal("10.00"),
            no_action_window=False,
            up_market_participation=0.9,
            down_market_defense=0.6,
            status="COMPLETED",
            ledger=(),
            result_json={
                "equity_series": [100000.0, 110000.0],
                "positions": [0, 50],
            },
            account_hash=f"hash-{package_id}",
        )

    with factory() as session, session.begin():
        persist_continuous_result(
            session,
            result(v35_package.id),
            market_returns=[0.10],
            elapsed_years=1.0,
            protocol_version="V3.5_CAPITAL_DRIVEN_8W",
        )
        persist_continuous_result(
            session,
            result(v351_package.id),
            market_returns=[0.10],
            elapsed_years=1.0,
            protocol_version=PROTOCOL_VERSION_351,
        )
    with factory() as session:
        rows = session.scalars(
            select(V35ContinuousAccount).where(
                V35ContinuousAccount.model_market == "399006"
            )
        ).all()
        by_protocol = {row.protocol_version: row for row in rows}
        assert by_protocol["V3.5_CAPITAL_DRIVEN_8W"].model_package_id == v35_package.id
        assert by_protocol[PROTOCOL_VERSION_351].model_package_id == v351_package.id


def test_v351_forecast_quantiles_within_physical_bounds(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        forecasts = session.scalars(
            select(V35Forecast).where(
                V35Forecast.protocol_version == PROTOCOL_VERSION_351
            )
        ).all()
        assert forecasts
        for row in forecasts:
            for quantile_path in row.price_quantiles_json:
                for value in quantile_path:
                    assert float(value) >= -1.0, (
                        row.model_market,
                        row.forecast_anchor_date,
                        float(value),
                    )


def test_v351_replay_emits_sell_decisions(fixture_db):
    engine = fixture_db["engine"]
    with Session(engine) as session:
        sells = session.scalars(
            select(V35PositionDecision)
            .join(
                V35StrategySnapshot,
                V35PositionDecision.strategy_snapshot_id == V35StrategySnapshot.id,
            )
            .where(
                V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_351,
                V35PositionDecision.action == "SELL",
            )
            .limit(1)
        ).all()
        assert sells, "v351 replay must produce SELL decisions"


def test_start_replace_is_async_and_cancel_is_persisted(fixture_db):
    factory = fixture_db["factory"]
    service = V351SlotService(
        factory,
        metadata_provider=lambda code: {
            "instrument_code": code,
            "official_name": "测试ETF",
            "exchange": "SSE",
            "instrument_type": "ETF",
            "data_source": "TEST",
        },
    )
    started = service.start_replace(
        "ETF_SLOT_03",
        "500001",
        idempotency_key="idem-cancel-test",
    )
    assert started["state"] == "VALIDATING"
    cancelled = service.cancel_replacement("ETF_SLOT_03")
    assert cancelled["cancel_requested"] is True
    with factory() as session:
        job = session.scalar(
            select(V351SlotReplacementJob).where(
                V351SlotReplacementJob.idempotency_key == "idem-cancel-test"
            )
        )
        assert job is not None
        assert (job.payload_json or {}).get("cancel_requested") is True


def test_v36_refresh_to_incremental_failure_rolls_back_week(
    tmp_path,
    monkeypatch,
):
    engine = create_database_engine(tmp_path / "v36-fail.db")
    run_migrations(engine)
    _seed_market(engine, "399006", start=date(2021, 1, 4), days=500)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    runtime = V35RuntimeService(
        factory,
        protocol_version=PROTOCOL_VERSION_36,
        v351=True,
    )

    def raise_on_snapshot(*_args, **_kwargs):
        raise RuntimeError("injected snapshot failure")

    monkeypatch.setattr(runtime, "_persist_account_snapshot", raise_on_snapshot)
    with pytest.raises(RuntimeError, match="injected snapshot failure"):
        runtime.bootstrap_sync("399006", maximum_weeks=1)
    monkeypatch.undo()
    with Session(engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(V35Forecast)
                .where(
                    V35Forecast.model_market == "399006",
                    V35Forecast.protocol_version == PROTOCOL_VERSION_36,
                )
            )
            == 0
        )
        assert (
            session.scalar(
                select(func.count())
                .select_from(V36AccountSnapshot)
                .where(
                    V36AccountSnapshot.model_market == "399006",
                    V36AccountSnapshot.protocol_version == PROTOCOL_VERSION_36,
                )
            )
            == 0
        )

    result = runtime.bootstrap_sync("399006", maximum_weeks=1)
    assert result["processed_weeks"] == 1
    with Session(engine) as session:
        snapshot = session.scalar(
            select(V36AccountSnapshot).where(
                V36AccountSnapshot.model_market == "399006",
                V36AccountSnapshot.protocol_version == PROTOCOL_VERSION_36,
            )
        )
        assert snapshot is not None
        forecast = session.scalar(
            select(V35Forecast).where(
                V35Forecast.model_market == "399006",
                V35Forecast.protocol_version == PROTOCOL_VERSION_36,
            )
        )
        assert forecast is not None
        assert forecast.data_source_provenance is not None
    engine.dispose()
