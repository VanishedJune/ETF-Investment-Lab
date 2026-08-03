from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.forecasting import HistoricalSimilarityForecaster
from backend.app.models.models import ForecastRun, Instrument, MarketPrice, ValuationRecord
from backend.app.schemas.market import MarketDataRecord, ProviderResult
from backend.app.services.market_data import MarketDataService


def _service(tmp_path):
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    return create_session_factory(database)


def _weekday_records(count: int) -> list[MarketDataRecord]:
    day = date(2025, 1, 2)
    records: list[MarketDataRecord] = []
    for index in range(count):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        close = Decimal("1000") + Decimal(index) * Decimal("1.25")
        records.append(
            MarketDataRecord(
                trade_date=day,
                open_price=close - Decimal("1"),
                high_price=close + Decimal("2"),
                low_price=close - Decimal("2"),
                close_price=close,
                volume=Decimal("100000"),
                source="TEST_INDEX",
            )
        )
        day += timedelta(days=1)
    return records


def test_13_week_scenario_reports_the_true_non_overlapping_history_requirement(tmp_path) -> None:
    sessions = _service(tmp_path)
    MarketDataService(sessions).store_result(
        "000688", ProviderResult.success("TEST_INDEX", _weekday_records(160))
    )

    with pytest.raises(ValueError, match="at least 189 daily rows; local cache has 160"):
        HistoricalSimilarityForecaster(sessions).run("000688")


def test_13_week_scenario_produces_all_weekly_points_once_history_is_sufficient(tmp_path) -> None:
    sessions = _service(tmp_path)
    MarketDataService(sessions).store_result(
        "000688", ProviderResult.success("TEST_INDEX", _weekday_records(189))
    )

    result = HistoricalSimilarityForecaster(sessions).run("000688")

    assert len(result["points"]) == 13
    assert len(result["matched_windows"]) == 5


def test_13_week_scenario_persists_a_valuation_path_for_each_matched_window(tmp_path) -> None:
    sessions = _service(tmp_path)
    MarketDataService(sessions).store_result(
        "000688", ProviderResult.success("TEST_INDEX", _weekday_records(189))
    )
    with sessions() as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "000688"))
        assert instrument is not None
        prices = session.scalars(
            select(MarketPrice)
            .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
            .order_by(MarketPrice.trade_date)
        ).all()
        for index, price in enumerate(prices):
            session.add(
                ValuationRecord(
                    instrument_id=instrument.id,
                    valuation_date=price.trade_date,
                    pe_ratio=Decimal("10") + Decimal(index) / Decimal("10"),
                    pb_ratio=Decimal("1") + Decimal(index) / Decimal("100"),
                    valuation_percentile=Decimal(index) / Decimal("200"),
                )
            )

    result = HistoricalSimilarityForecaster(sessions).run("000688")

    assert len(result["matched_windows"]) == 5
    for window in result["matched_windows"]:
        path = window["valuation_path"]
        assert len(path) == 14
        assert [point["week"] for point in path] == list(range(14))
        assert path[0]["date"] == window["end_date"]
        assert all(point["pe_ratio"] is not None for point in path)
        assert path[0]["pe_ratio"] != path[-1]["pe_ratio"]

    with sessions() as session:
        run = session.scalar(select(ForecastRun).order_by(ForecastRun.id.desc()))
    assert run is not None
    assert run.summary["matched_windows"] == result["matched_windows"]


def test_availability_exposes_13_week_data_readiness_without_running_a_scenario(tmp_path) -> None:
    sessions = _service(tmp_path)
    MarketDataService(sessions).store_result(
        "000688", ProviderResult.success("TEST_INDEX", _weekday_records(160))
    )

    availability = HistoricalSimilarityForecaster(sessions).availability("000688")

    assert availability["daily_rows"] == 160
    assert availability["required_rows"] == 189
    assert availability["can_run"] is False
