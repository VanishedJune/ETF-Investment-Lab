from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from time import sleep

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import Base, Instrument, MarketPrice
from backend.app.services.investment_calendar_service import InvestmentCalendarService
from backend.app.services.v31_analysis_service import (
    GateSnapshot,
    V31AnalysisService,
    _call_with_timeout,
)


class WeekdayCalendar:
    def sessions(self, _symbol: str, start: date, end: date) -> tuple[date, ...]:
        return tuple(
            start + timedelta(days=offset)
            for offset in range((end - start).days + 1)
            if (start + timedelta(days=offset)).weekday() < 5
        )


def _service(tmp_path):
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    factory = create_session_factory(database)
    calendar = WeekdayCalendar()
    service = V31AnalysisService(
        factory,
        market=object(),  # type: ignore[arg-type]
        valuations=object(),  # type: ignore[arg-type]
        indicators=object(),  # type: ignore[arg-type]
        calendar=calendar,
        investment_calendar=InvestmentCalendarService(factory),
    )
    return service, factory


def _seed_curves(factory) -> tuple[date, date]:
    start = date(2021, 1, 4)
    trading_days = [
        start + timedelta(days=offset)
        for offset in range(1600)
        if (start + timedelta(days=offset)).weekday() < 5
    ][:1100]
    with factory() as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "399006"))
        assert instrument is not None
        for index, trade_day in enumerate(trading_days):
            close = Decimal("1000") + Decimal(index) * Decimal("0.8") + Decimal(index % 17 - 8)
            session.add(MarketPrice(
                instrument_id=instrument.id,
                trade_date=trade_day,
                timeframe="daily",
                open_price=close - Decimal("2"),
                high_price=close + Decimal("5"),
                low_price=close - Decimal("5"),
                close_price=close,
                adjusted_close_price=close,
                volume=Decimal("1000000") + Decimal(index * 100),
                source="AKSHARE_INDEX_SINA",
                volume_source="AKSHARE_INDEX_SINA",
            ))
        for week_index, offset in enumerate(range(4, len(trading_days), 5)):
            trade_day = trading_days[offset]
            week_days = trading_days[offset - 4 : offset + 1]
            closes = [Decimal("1000") + Decimal(trading_days.index(day)) * Decimal("0.8") + Decimal(trading_days.index(day) % 17 - 8) for day in week_days]
            session.add(MarketPrice(
                instrument_id=instrument.id,
                trade_date=trade_day,
                timeframe="weekly",
                open_price=closes[0],
                high_price=max(closes) + Decimal("5"),
                low_price=min(closes) - Decimal("5"),
                close_price=closes[-1],
                adjusted_close_price=closes[-1],
                volume=Decimal("5000000") + Decimal(week_index * 1000),
                source="AGGREGATED_DAILY_VOLUME:AKSHARE_INDEX_SINA",
                volume_source="AGGREGATED_DAILY_VOLUME:AKSHARE_INDEX_SINA",
            ))
    return trading_days[-1], trading_days[(len(trading_days) // 5) * 5 - 1]


def test_v31_models_use_daily_100_window_and_continuous_13_week_quantiles(tmp_path) -> None:
    service, factory = _service(tmp_path)
    daily_cutoff, weekly_cutoff = _seed_curves(factory)
    gate = GateSnapshot(
        symbol="399006",
        latest_available=daily_cutoff,
        price_as_of=daily_cutoff,
        valuation_as_of=None,
        complete_week_as_of=weekly_cutoff,
        incomplete_week_as_of=daily_cutoff,
        volume_as_of=weekly_cutoff,
        source="AKSHARE_INDEX_SINA",
        volume_source="AGGREGATED_DAILY_VOLUME:AKSHARE_INDEX_SINA",
        degraded=("PE/PB估值不可用",),
    )
    models = service._ensure_models("399006", weekly_cutoff)
    weekly = service._weekly_signal("399006", gate, models["weekly"])
    daily = service._daily_correction("399006", gate, weekly)
    path = service._probability_path("399006", gate, weekly, models["weekly"])
    advice = service._advice(
        "399006", gate, weekly, daily, path,
        {"confirmed_position": 0, "in_transit_adjustment": 0, "settlement_position": 0, "advice_basis_position": 0},
    )

    assert daily["window_sessions"] == 100
    assert -10 <= daily["confidence_adjustment"] <= 10
    assert -10 <= daily["position_adjustment"] <= 10
    assert len(path["points"]) == 13
    assert round(path["up_probability"] + path["sideways_probability"] + path["down_probability"], 5) == 100
    assert all(
        point["p10_cumulative_return"] <= point["p50_cumulative_return"] <= point["p90_cumulative_return"]
        for point in path["points"]
    )
    assert advice["current_position"] == 0
    assert advice["target_position"] % 5 == 0
    assert advice["fund_etf"]["target_ratio"] in {"7:3", "6:4", "5:5", "4:6", "3:7"}
    assert all(batch["position_points"] % 5 == 0 for batch in advice["batches"])


def test_schema_keeps_v2_and_adds_all_v31_audit_tables() -> None:
    assert "v2_advice_history" in Base.metadata.tables
    assert {
        "v31_analysis_runs",
        "v31_model_versions",
        "v31_weekly_forecasts",
        "v31_daily_corrections",
        "v31_model_evaluations",
        "v31_instrument_market_mappings",
    } <= set(Base.metadata.tables)


def test_public_provider_calls_have_a_hard_timeout() -> None:
    result = _call_with_timeout(
        lambda: (sleep(0.05), "late")[1],
        timeout_seconds=0.001,
        timeout_value=lambda: "timed-out",
    )

    assert result == "timed-out"


def test_latest_run_persists_and_returns_a_failed_gate_result(tmp_path) -> None:
    service, _factory = _service(tmp_path)
    run_id = "V31-399006-failed-gate"
    service._create_run(run_id, "399006")
    service._update_run(
        run_id,
        status="failed",
        current_stage="执行失败",
        error_code="STALE_DAILY_DATA",
        error_message="行情未更新",
    )

    latest = service.latest("399006")

    assert latest is not None
    assert latest["id"] == run_id
    assert latest["status"] == "failed"
    assert latest["error_code"] == "STALE_DAILY_DATA"
