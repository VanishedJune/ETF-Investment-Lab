from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    V33InstrumentRole,
)
from backend.app.services.instrument_universe import DISPLAY_ONLY_CODES, MODEL_MARKETS
from backend.app.services.indicator_service import IndicatorService
from backend.app.services.market_calendar import ExchangeCalendarProvider
from backend.app.services.v343_market_ui_service import (
    V343MarketUIError,
    V343MarketUIService,
    _adjusted_ohlcv,
    _with_derivatives,
)
from backend.web import (
    V33ModelActionRequest,
    _provisional_indicator_row,
    _provisional_period_row,
    analyze_v343_model,
    v343_analysis_run,
)


class _Calendar:
    def sessions(self, _market: str, start: date, end: date):
        return [start] if start == end else [start, end]


def test_data_only_etfs_are_seeded_without_new_plans_or_new_model_roles(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    sessions = create_session_factory(database)

    with sessions() as session:
        instruments = list(
            session.scalars(
                select(Instrument).where(Instrument.code.in_(DISPLAY_ONLY_CODES))
            )
        )
        instrument_ids = [row.id for row in instruments]
        plan_count = session.scalar(
            select(func.count())
            .select_from(InvestmentPlan)
            .where(InvestmentPlan.instrument_id.in_(instrument_ids))
        )
        role_codes = set(
            session.scalars(
                select(Instrument.code)
                .join(V33InstrumentRole, V33InstrumentRole.instrument_id == Instrument.id)
                .where(Instrument.id.in_(instrument_ids))
            )
        )

    assert {row.code for row in instruments} == DISPLAY_ONLY_CODES
    assert all((row.extra_data or {}).get("role") == "display_only" for row in instruments)
    assert all((row.extra_data or {}).get("model_eligible") is False for row in instruments)
    assert plan_count == 0
    # 159941 keeps its frozen V3.3 audit role, but the expanded data-only ETF
    # catalog must not create any new training/model role.
    assert role_codes == (DISPLAY_ONLY_CODES & MODEL_MARKETS)


def test_v343_model_adapter_rejects_nonlegacy_display_only_codes(tmp_path: Path) -> None:
    service = V343MarketUIService(lambda: None, calendar=_Calendar())  # type: ignore[arg-type]
    for code in DISPLAY_ONLY_CODES - MODEL_MARKETS:
        with pytest.raises(V343MarketUIError):
            service.validate_market(code)


def test_display_only_etfs_have_real_exchange_calendar_bindings() -> None:
    calendar = ExchangeCalendarProvider()

    for code in DISPLAY_ONLY_CODES:
        metadata = calendar.metadata(code)
        assert metadata["schedule_name"] == "XSHG"
        if code.startswith("1"):
            assert metadata["target_exchange"] == "XSHE"
            assert metadata["equivalent_mainland_schedule_proxy"] is True
        else:
            assert metadata["target_exchange"] == "XSHG"
            assert metadata["equivalent_mainland_schedule_proxy"] is False


def test_historical_dif_derivatives_do_not_bridge_missing_periods() -> None:
    rows = [
        SimpleNamespace(
            indicator_date=date(2026, 1, 2),
            indicator_values={"values": {"dif": 1.0, "dea": 0.8, "macd_histogram": 0.4}},
        ),
        SimpleNamespace(
            indicator_date=date(2026, 1, 9),
            indicator_values={"values": {"dif": None, "dea": None, "macd_histogram": None}},
        ),
        SimpleNamespace(
            indicator_date=date(2026, 1, 16),
            indicator_values={"values": {"dif": 1.2, "dea": 1.0, "macd_histogram": 0.4}},
        ),
        SimpleNamespace(
            indicator_date=date(2026, 1, 23),
            indicator_values={"values": {"dif": 1.5, "dea": 1.1, "macd_histogram": 0.8}},
        ),
    ]

    output = _with_derivatives(rows)  # type: ignore[arg-type]

    assert output[date(2026, 1, 16)]["dif_first_change"] is None
    assert output[date(2026, 1, 16)]["dif_second_change"] is None
    assert output[date(2026, 1, 23)]["dif_first_change"] == pytest.approx(0.3)


def test_forecast_history_keeps_indicator_slots_aligned_by_week(tmp_path: Path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    sessions = create_session_factory(database)
    weeks = [date(2026, 1, day) for day in (2, 9, 16, 23)]
    with sessions() as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "399006"))
        assert instrument is not None
        session.add_all(
            [
                MarketPrice(
                    instrument_id=instrument.id,
                    trade_date=when,
                    timeframe="weekly",
                    open_price=Decimal("100"),
                    high_price=Decimal("110"),
                    low_price=Decimal("90"),
                    close_price=Decimal("100"),
                    volume=Decimal("1000"),
                    source="TEST",
                )
                for when in weeks
            ]
        )
        session.add_all(
            [
                IndicatorRecord(
                    instrument_id=instrument.id,
                    indicator_date=when,
                    timeframe="weekly",
                    indicator_name="technical_indicators",
                    indicator_values={"values": {
                        "dif": None if when == weeks[1] else str(index + 1),
                        "dea": None if when == weeks[1] else str(index + 0.5),
                        "macd_histogram": None if when == weeks[1] else "1",
                    }},
                )
                for index, when in enumerate(weeks)
            ]
        )

    result = V343MarketUIService(sessions, calendar=_Calendar()).enrich_forecast(
        "399006", {"forecast_anchor_date": weeks[-1].isoformat()}
    )

    assert [row["week_end"] for row in result["historical_indicators"]] == [
        when.isoformat() for when in weeks
    ]
    assert result["historical_indicators"][1]["dif"] is None
    assert result["historical_indicators"][2]["dif"] == pytest.approx(3.0)
    assert result["historical_indicators"][2]["dif_first_change"] is None


def test_adjusted_history_rejects_non_positive_close() -> None:
    row = SimpleNamespace(
        trade_date=date(2026, 1, 2),
        open_price=Decimal("0"),
        high_price=Decimal("0"),
        low_price=Decimal("0"),
        close_price=Decimal("0"),
        adjusted_close_price=None,
        volume=Decimal("1"),
        volume_multiplier=1,
        source="TEST",
    )
    with pytest.raises(V343MarketUIError, match="non-positive close"):
        _adjusted_ohlcv(row)  # type: ignore[arg-type]


def test_current_week_overlay_is_read_only_and_sums_daily_volume() -> None:
    daily = [
        SimpleNamespace(
            trade_date=date(2026, 8, day),
            open_price=Decimal(str(10 + day)),
            high_price=Decimal(str(11 + day)),
            low_price=Decimal(str(9 + day)),
            close_price=Decimal(str(10.5 + day)),
            volume=Decimal("100"),
            volume_multiplier=100,
            turnover=Decimal("200"),
        )
        for day in (3, 4, 5)
    ]

    overlay = _provisional_period_row(
        daily,  # type: ignore[arg-type]
        [],
        "weekly",
        volume_available=True,
    )

    assert overlay is not None
    assert overlay["date"] == date(2026, 8, 5)
    assert overlay["volume"] == Decimal("30000")
    assert overlay["is_complete"] is False
    assert overlay["period_status"] == "INCOMPLETE_CURRENT_PERIOD"


def test_current_week_overlay_has_matching_indicator_row() -> None:
    def market_row(day: date, close: str, volume: str = "100") -> SimpleNamespace:
        value = Decimal(close)
        return SimpleNamespace(
            trade_date=day,
            open_price=value,
            high_price=value + Decimal("1"),
            low_price=value - Decimal("1"),
            close_price=value,
            adjusted_close_price=None,
            volume=Decimal(volume),
            volume_multiplier=1,
            turnover=Decimal("10"),
            source="TEST",
        )

    completed = [market_row(date(2026, 7, 31), "100")]
    daily = [
        market_row(date(2026, 8, 3), "101"),
        market_row(date(2026, 8, 4), "102"),
        market_row(date(2026, 8, 5), "103"),
    ]
    indicator = _provisional_indicator_row(
        IndicatorService(lambda: None),  # type: ignore[arg-type]
        "518600",
        "weekly",
        daily,  # type: ignore[arg-type]
        completed,  # type: ignore[arg-type]
    )

    assert indicator is not None
    assert indicator["date"] == date(2026, 8, 5)
    assert all(
        field in indicator
        for field in ("dif", "dea", "macd_histogram", "dif_first_change")
    )


def test_v343_analysis_queues_then_composes_frozen_forecast_without_training() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def start_analysis(self, market: str):
            self.calls.append(market)
            return {"run_id": "analysis-1", "status": "queued", "market": market}

        def analysis_status(self, run_id: str):
            assert run_id == "analysis-1"
            return {
                "run_id": run_id,
                "status": "completed",
                "market": "399006",
                "result": {
                    "forecast_anchor_date": "2026-07-31",
                    "market": "399006",
                },
            }

    class Policy:
        def analyze(self, market: str):
            return {"market": market, "decision": {"action": "HOLD"}}

    class Adapter:
        def enrich_forecast(self, market: str, forecast, *, policy=None):
            return {**forecast, "display_version": "V3.4.3_MARKET_UI", "policy": policy}

    runtime = Runtime()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                v341_runtime=runtime,
                v342_policy=Policy(),
                v343_market_ui=Adapter(),
            )
        )
    )

    created = analyze_v343_model(
        request,  # type: ignore[arg-type]
        V33ModelActionRequest(instrument_code="399006"),
    )
    payload = v343_analysis_run(request, created["run_id"])  # type: ignore[arg-type]

    assert created == {"run_id": "analysis-1", "status": "queued", "market": "399006"}
    assert payload["result"]["display_version"] == "V3.4.3_MARKET_UI"
    assert payload["result"]["policy"]["decision"]["action"] == "HOLD"
    assert runtime.calls == ["399006"]
