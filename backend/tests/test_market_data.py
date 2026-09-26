from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
import sys
import threading

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    DataUpdateLog,
    IndicatorRecord,
    Instrument,
    MarketPrice,
)
from backend.app.schemas.market import (
    CsvImportResult,
    DataUpdateResponse,
    MarketDataRecord,
    ProviderResult,
    ProviderStatus,
)
from backend.app.services.csv_import import CsvImportService
from backend.app.services.market_data import MarketDataService
from backend.app.services.market_calendar import ExchangeCalendarProvider
from backend.app.services.providers import (
    AkShareEtfResearchProvider,
    AkShareIndexProvider,
    AkShareProvider,
    CsvProvider,
    TushareProvider,
    YahooDirectIndexProvider,
)


class FailingProvider:
    source = "AKSHARE"

    def fetch(self, instrument_code: str, start_date: date | None = None, end_date: date | None = None) -> ProviderResult:
        return ProviderResult.failed(self.source, "AkShare is unavailable for this test")


def test_yahoo_direct_index_provider_reads_ndx_without_fabricating_volume() -> None:
    calls: list[tuple[str, date | None, date | None]] = []
    timestamp = int(
        datetime(2019, 11, 8, 21, 0, tzinfo=timezone.utc).timestamp()
    )

    def loader(symbol, start_date, end_date):  # type: ignore[no-untyped-def]
        calls.append((symbol, start_date, end_date))
        return {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [timestamp],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [8208.3701171875],
                                    "high": [8256.2900390625],
                                    "low": [8186.31005859375],
                                    "close": [8255.8896484375],
                                    "volume": [None],
                                }
                            ]
                        },
                    }
                ],
            }
        }

    result = YahooDirectIndexProvider(loader=loader).fetch(
        "NDX", date(2019, 11, 8), date(2019, 11, 8)
    )

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "YAHOO_DIRECT_INDEX"
    assert calls == [("^NDX", date(2019, 11, 8), date(2019, 11, 8))]
    assert result.volume_availability == "not_available_for_direct_index"
    assert result.records[0].trade_date == date(2019, 11, 8)
    assert result.records[0].close_price == Decimal("8255.88964844")
    assert result.records[0].volume is None
    assert result.records[0].volume_source == "VOLUME_UNAVAILABLE:DIRECT_INDEX"


def test_direct_index_provider_uses_index_endpoints_not_etf_adapter() -> None:
    china_calls: list[dict[str, object]] = []
    us_calls: list[dict[str, object]] = []

    def china_loader(**kwargs):  # type: ignore[no-untyped-def]
        china_calls.append(kwargs)
        return [{"date": "2026-07-01", "open": "1000", "high": "1010", "low": "995", "close": "1005", "volume": "12345", "amount": "67890"}]

    def us_loader(**kwargs):  # type: ignore[no-untyped-def]
        us_calls.append(kwargs)
        return [{"date": "2026-07-02", "open": "20000", "high": "20120", "low": "19980", "close": "20080", "volume": "4567", "amount": "89012"}]

    provider = AkShareIndexProvider(china_loader=china_loader, us_loader=us_loader)
    china = provider.fetch("000688", date(2026, 7, 1), date(2026, 7, 1))
    us = provider.fetch("NDX")

    assert china.status is ProviderStatus.SUCCESS
    assert china.records[0].close_price == Decimal("1005")
    assert china.records[0].amount is None
    assert china_calls == [{"symbol": "000688", "period": "daily", "start_date": "20260701", "end_date": "20260701"}]
    assert us.status is ProviderStatus.SUCCESS
    assert us.records[0].close_price == Decimal("20080")
    assert us.records[0].volume is None
    assert us.records[0].amount is None
    assert us.volume_availability == "not_available_for_direct_index"
    assert us_calls == [{"symbol": ".NDX"}]


def test_direct_china_index_provider_requests_native_weekly_volume_before_local_aggregation() -> None:
    calls: list[dict[str, object]] = []

    def china_loader(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return [
            {
                "date": "2026-07-24",
                "open": "2300",
                "high": "2350",
                "low": "2280",
                "close": "2330",
                "volume": "987654321",
            }
        ]

    result = AkShareIndexProvider(china_loader=china_loader).fetch_weekly(
        "399006", date(2026, 7, 20), date(2026, 7, 24)
    )

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_INDEX_WEEKLY"
    assert result.records[0].volume == Decimal("987654321")
    assert calls == [
        {
            "symbol": "399006",
            "period": "weekly",
            "start_date": "20260720",
            "end_date": "20260724",
        }
    ]


def test_etf_provider_requests_native_weekly_ohlcv_for_159941() -> None:
    calls: list[dict[str, object]] = []

    def loader(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return [
            {
                "date": "2026-07-31",
                "open": "1.571",
                "high": "1.587",
                "low": "1.568",
                "close": "1.574",
                "volume": "11789918",
                "amount": "1858237595.505",
            }
        ]

    result = AkShareProvider(history_loader=loader).fetch_weekly(
        "159941", date(2026, 7, 27), date(2026, 7, 31)
    )

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_ETF_WEEKLY"
    assert result.records[0].volume == Decimal("11789918")
    assert result.records[0].amount == Decimal("1858237595.505")
    assert calls == [
        {
            "symbol": "159941",
            "period": "weekly",
            "start_date": "20260727",
            "end_date": "20260731",
            "adjust": "",
        }
    ]


def test_etf_research_provider_keeps_trade_close_and_stores_cny_qfq_close() -> None:
    calls: list[str] = []

    def loader(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(str(kwargs["adjust"]))
        close = "2.468" if kwargs["adjust"] == "qfq" else "1.574"
        return [
            {
                "date": "2026-07-31",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": "11789918",
                "amount": "1858237595.505",
            }
        ]

    result = AkShareEtfResearchProvider(history_loader=loader).fetch("159941")

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_ETF"
    assert result.records[0].close_price == Decimal("1.574")
    assert result.records[0].adjusted_close_price == Decimal("2.468")
    assert result.records[0].volume_multiplier == 100
    assert result.records[0].volume_source == "AKSHARE_ETF:LOTS_X100"
    assert result.records[0].source == "AKSHARE_ETF"
    assert calls == ["", "qfq"]


def test_etf_research_provider_falls_back_to_sina_and_tencent_without_unit_drift() -> None:
    def unavailable_primary(**_kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("EastMoney unavailable")

    def sina(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"symbol": "sz159941"}
        return [
            {
                "date": "2026-07-31",
                "open": "1.571",
                "high": "1.587",
                "low": "1.568",
                "close": "1.574",
                "volume": "1178991823",
                "amount": "1858237596",
            }
        ]

    result = AkShareEtfResearchProvider(
        history_loader=unavailable_primary,
        sina_loader=sina,
        tencent_loader=lambda *_args: [
            ["2026-07-31", "1.571", "1.574", "1.587", "1.568", "11789918"]
        ],
    ).fetch("159941")

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_ETF_SINA"
    assert result.records[0].volume == Decimal("1178991823")
    assert result.records[0].volume_multiplier == 1
    assert result.records[0].adjusted_close_price == Decimal("1.574")
    assert any("Sina public history used" in warning for warning in result.warnings)


def test_direct_sina_etf_fallback_parses_plain_jsonp_without_native_runtime(
    monkeypatch,
) -> None:
    body = b'''/* redirect guard */\nvar _etf_sz159941=([{"day":"2026-08-10","open":"1.700","high":"1.704","low":"1.683","close":"1.684","volume":"588611661"}]);'''
    requests: list[object] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return body

    def opener(request, *, timeout):  # type: ignore[no-untyped-def]
        requests.append(request)
        assert timeout == 30
        return Response()

    monkeypatch.setattr("backend.app.services.providers.urlopen", opener)

    rows = AkShareEtfResearchProvider._download_sina_unadjusted(symbol="sz159941")

    assert rows == [
        {
            "day": "2026-08-10",
            "date": "2026-08-10",
            "open": "1.700",
            "high": "1.704",
            "low": "1.683",
            "close": "1.684",
            "volume": "588611661",
        }
    ]
    assert "CN_MarketDataService.getKLineData" in requests[0].full_url
    assert "symbol=sz159941" in requests[0].full_url


def test_default_sina_fallback_does_not_access_akshare_native_decoder(
    monkeypatch,
) -> None:
    def unavailable_primary(**_kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("EastMoney unavailable")

    class AkShareWithoutSina:
        fund_etf_hist_em = staticmethod(unavailable_primary)

        def __getattr__(self, name: str):
            if name == "fund_etf_hist_sina":
                raise AssertionError("native AkShare Sina decoder must not be accessed")
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "akshare", AkShareWithoutSina())
    monkeypatch.setattr(
        AkShareEtfResearchProvider,
        "_download_sina_unadjusted",
        staticmethod(
            lambda **_kwargs: [
                {
                    "date": "2026-08-10",
                    "open": "1.700",
                    "high": "1.704",
                    "low": "1.683",
                    "close": "1.684",
                    "volume": "588611661",
                }
            ]
        ),
    )

    result = AkShareEtfResearchProvider(
        tencent_loader=lambda *_args: [
            ["2026-08-10", "1.700", "1.684", "1.704", "1.683", "5886116"]
        ]
    ).fetch("159941", date(2026, 8, 8))

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_ETF_SINA"
    assert result.cutoff_date == date(2026, 8, 10)
    assert result.records[0].close_price == Decimal("1.684")
    assert result.records[0].adjusted_close_price == Decimal("1.684")


def test_etf_research_provider_treats_empty_eastmoney_result_as_fallback_condition() -> None:
    primary_calls: list[str] = []

    def empty_primary(**kwargs):  # type: ignore[no-untyped-def]
        primary_calls.append(str(kwargs["adjust"]))
        return []

    result = AkShareEtfResearchProvider(
        history_loader=empty_primary,
        sina_loader=lambda **_kwargs: [
            {
                "date": "2026-08-07",
                "open": "1.500",
                "high": "1.530",
                "low": "1.490",
                "close": "1.520",
                "volume": "1200000",
                "amount": "1820000",
            }
        ],
        tencent_loader=lambda *_args: [],
    ).fetch("159941")

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_ETF_SINA"
    assert result.cutoff_date == date(2026, 8, 7)
    assert primary_calls == ["", "qfq"]
    assert any("returned no usable rows" in warning for warning in result.warnings)


def test_etf_research_provider_replaces_only_invalid_sina_zero_volume_from_crosscheck() -> None:
    def unavailable_primary(**_kwargs):  # type: ignore[no-untyped-def]
        raise ConnectionError("EastMoney unavailable")

    result = AkShareEtfResearchProvider(
        history_loader=unavailable_primary,
        sina_loader=lambda **_kwargs: [
            {
                "date": "2024-11-01",
                "open": "1.081",
                "high": "1.091",
                "low": "1.080",
                "close": "1.085",
                "volume": "0",
                "amount": "0",
            }
        ],
        tencent_loader=lambda *_args: [
            ["2024-11-01", "1.081", "1.085", "1.091", "1.080", "9916642"]
        ],
    ).fetch("159941")

    assert result.status is ProviderStatus.SUCCESS
    assert result.records[0].volume == Decimal("9916642")
    assert result.records[0].volume_multiplier == 100
    assert result.records[0].volume_source == "TENCENT_QFQ:LOTS_X100"


def test_ndx_provider_failures_retain_explicit_volume_unavailability() -> None:
    def failed_us_loader(**_kwargs):
        raise RuntimeError("network unavailable")

    provider = AkShareIndexProvider(us_loader=failed_us_loader)

    daily = provider.fetch("NDX")
    weekly = provider.fetch_weekly("NDX")

    assert daily.status is ProviderStatus.ERROR
    assert weekly.status is ProviderStatus.ERROR
    assert (
        daily.volume_availability
        == weekly.volume_availability
        == "not_available_for_direct_index"
    )


def test_direct_china_index_uses_akshare_sina_history_when_primary_endpoint_is_unavailable() -> None:
    fallback_calls: list[dict[str, object]] = []

    def failed_primary(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("primary public endpoint unavailable")

    def fallback(**kwargs):  # type: ignore[no-untyped-def]
        fallback_calls.append(kwargs)
        return [{"date": "2026-07-29", "open": "1670", "high": "1690", "low": "1660", "close": "1678", "volume": "1200000"}]

    result = AkShareIndexProvider(china_loader=failed_primary, china_fallback_loader=fallback).fetch("000688")

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_INDEX_SINA"
    assert result.records[0].close_price == Decimal("1678")
    assert fallback_calls == [{"symbol": "sh000688"}]


def test_direct_china_index_uses_fresh_tencent_history_when_primary_is_unavailable() -> None:
    def failed_primary(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("EastMoney unavailable")

    def tencent(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {
            "symbol": "sz399006",
            "start_date": "20260801",
            "end_date": "20260807",
        }
        return [
            {
                "date": "2026-08-05",
                "open": "3372.08",
                "high": "3584.06",
                "low": "3372.04",
                "close": "3535.14",
                "amount": "226495353",
            },
            {
                "date": "2026-08-06",
                "open": "3472.15",
                "high": "3574.41",
                "low": "3444.36",
                "close": "3515.56",
                "amount": "218807247",
            },
            {
                "date": "2026-08-07",
                "open": "3537.44",
                "high": "3613.87",
                "low": "3507.62",
                "close": "3563.12",
                "amount": "229977435",
            },
        ]

    result = AkShareIndexProvider(
        china_loader=failed_primary,
        china_tencent_loader=tencent,
    ).fetch("399006", date(2026, 8, 1), date(2026, 8, 7))

    assert result.status is ProviderStatus.SUCCESS
    assert result.source == "AKSHARE_INDEX_TENCENT"
    assert result.cutoff_date == date(2026, 8, 7)
    assert result.records[-1].close_price == Decimal("3563.12")
    assert result.records[-1].volume == Decimal("229977435")
    assert result.records[-1].volume_multiplier == 100
    assert result.records[-1].volume_source == "AKSHARE_INDEX_TENCENT:LOTS_X100"


def test_direct_china_index_appends_newer_tencent_sessions_to_stale_primary() -> None:
    def primary(**_kwargs):  # type: ignore[no-untyped-def]
        return [
            {
                "date": "2026-08-05",
                "open": "3372.08",
                "high": "3584.06",
                "low": "3372.04",
                "close": "3535.143",
                "volume": "22649535325",
            }
        ]

    def tencent(**_kwargs):  # type: ignore[no-untyped-def]
        return [
            {
                "date": "2026-08-05",
                "open": "3372.08",
                "high": "3584.06",
                "low": "3372.04",
                "close": "3535.14",
                "amount": "226495353",
            },
            {
                "date": "2026-08-06",
                "open": "3472.15",
                "high": "3574.41",
                "low": "3444.36",
                "close": "3515.56",
                "amount": "218807247",
            },
        ]

    result = AkShareIndexProvider(
        china_loader=primary,
        china_tencent_loader=tencent,
    ).fetch("399006", date(2026, 8, 1), date(2026, 8, 7))

    assert result.status is ProviderStatus.SUCCESS
    assert [row.trade_date for row in result.records] == [
        date(2026, 8, 5),
        date(2026, 8, 6),
    ]
    assert result.records[0].close_price == Decimal("3535.143")
    assert result.records[1].source == "AKSHARE_INDEX_TENCENT"


def test_direct_index_demo_fallback_is_ohlcv_valid_and_explicitly_labelled(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)

    result = service.update_from_providers("000688", [FailingProvider()])

    assert result.status is ProviderStatus.DEMO
    assert result.demo is True
    assert result.source == "DEMO"
    assert len(result.records) == 160
    assert all(record.source == "DEMO_INDEX" for record in result.records)
    service.aggregate_timeframes(
        "000688",
        expected_trade_dates=tuple(record.trade_date for record in result.records),
        as_of=result.records[-1].trade_date,
    )
    with Session(engine) as session:
        index_rows = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(Instrument.code == "000688", MarketPrice.timeframe == "daily")
        ).all()
    assert len(index_rows) == 160
    assert all(row.turnover is None for row in index_rows)


def test_cached_fallback_returns_when_failure_audit_log_cannot_be_written(
    caplog,
    tmp_path,
) -> None:
    class FaultInjectingMarketDataService(MarketDataService):
        def _add_log(self, session, **values):
            if values.get("status") == "error":
                raise RuntimeError("injected failure audit log failure")
            return super()._add_log(session, **values)

    session_factory, _engine = _service(tmp_path)
    MarketDataService(session_factory).store_result(
        "589850",
        ProviderResult.success(
            "CSV",
            [_record(date(2026, 7, 1), "1.00")],
        ),
    )

    response = FaultInjectingMarketDataService(
        session_factory
    ).update_from_provider("589850", FailingProvider())

    assert response.status is ProviderStatus.CACHED
    assert response.records
    assert "Could not persist market-data audit log" in caplog.text


def test_demo_fallback_returns_when_separate_failure_log_cannot_be_written(
    caplog,
    tmp_path,
) -> None:
    class FaultInjectingMarketDataService(MarketDataService):
        def _add_log(self, session, **values):
            if values.get("status") == "error":
                raise RuntimeError("injected failure audit log failure")
            return super()._add_log(session, **values)

    session_factory, engine = _service(tmp_path)

    response = FaultInjectingMarketDataService(
        session_factory
    ).update_from_provider("000688", FailingProvider())

    assert response.status is ProviderStatus.DEMO
    assert len(response.records) == 160
    assert "Could not persist market-data audit log" in caplog.text
    with Session(engine) as session:
        assert session.scalar(
            select(func.count())
            .select_from(DataUpdateLog)
            .where(DataUpdateLog.status == "demo")
        ) == 1


@pytest.mark.parametrize("instrument_code", ["589850", "000688"])
def test_demo_data_rolls_back_when_its_atomic_audit_log_fails(
    caplog,
    tmp_path,
    instrument_code: str,
) -> None:
    class FaultInjectingMarketDataService(MarketDataService):
        def _add_log(self, session, **values):
            if values.get("status") == "demo":
                raise RuntimeError("injected demo audit log failure")
            return super()._add_log(session, **values)

    session_factory, engine = _service(tmp_path)

    response = FaultInjectingMarketDataService(
        session_factory
    ).update_from_provider(instrument_code, FailingProvider())

    assert response.status is ProviderStatus.ERROR
    assert response.records == []
    assert "Could not persist DEMO market data and audit log" in caplog.text
    with Session(engine) as session:
        assert session.scalar(
            select(func.count()).select_from(MarketPrice)
        ) == 0


def test_real_direct_index_refresh_replaces_explicit_demo_rows(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    service.update_from_providers("000688", [FailingProvider()])

    service.store_result(
        "000688",
        ProviderResult.success(
            "AKSHARE_INDEX_SINA",
            [_record(date(2026, 7, 29), "1678", source="AKSHARE_INDEX_SINA")],
        ),
    )

    with Session(engine) as session:
        rows = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(Instrument.code == "000688", MarketPrice.timeframe == "daily")
        ).all()
    assert len(rows) == 1
    assert rows[0].source == "AKSHARE_INDEX_SINA"


def test_store_result_rolls_back_daily_data_when_success_log_fails(tmp_path) -> None:
    class FaultInjectingMarketDataService(MarketDataService):
        def _add_log(self, session, **values):
            if values.get("timeframe") == "daily" and values.get("status") == "success":
                raise RuntimeError("injected daily success log failure")
            return super()._add_log(session, **values)

    session_factory, engine = _service(tmp_path)
    service = FaultInjectingMarketDataService(session_factory)

    with pytest.raises(RuntimeError, match="daily success log failure"):
        service.store_result(
            "399006",
            ProviderResult.success(
                "DIRECT_INDEX_DAILY",
                [_record(date(2026, 7, 1), "100")],
            ),
        )

    with Session(engine) as session:
        count = session.scalar(
            select(func.count())
            .select_from(MarketPrice)
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe == "daily",
            )
        )
    assert count == 0


def _service(tmp_path):
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    return create_session_factory(database), create_database_engine(database)


def _aggregate(service, instrument_code: str, trade_dates, *, weekly_result=None):
    expected = tuple(trade_dates)
    return service.aggregate_timeframes(
        instrument_code,
        expected_trade_dates=expected,
        as_of=max(expected),
        weekly_result=weekly_result,
    )


def _record(day: date, close: str, *, source: str = "CSV") -> MarketDataRecord:
    close_value = Decimal(close)
    return MarketDataRecord(
        trade_date=day,
        open_price=close_value - Decimal("0.02"),
        high_price=close_value + Decimal("0.03"),
        low_price=close_value - Decimal("0.04"),
        close_price=close_value,
        volume=Decimal("100"),
        amount=Decimal("200"),
        source=source,
    )


def test_daily_upsert_replaces_matching_instrument_date_and_timeframe(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    result = ProviderResult.success("CSV", [_record(date(2026, 7, 1), "1.00")])

    service.store_result("589850", result)
    service.store_result("589850", ProviderResult.success("CSV", [_record(date(2026, 7, 1), "1.20")]))

    with Session(engine) as session:
        rows = session.scalars(
            select(MarketPrice).where(MarketPrice.timeframe == "daily")
        ).all()
    assert len(rows) == 1
    assert rows[0].close_price == Decimal("1.20")


def test_daily_store_preloads_once_and_batches_final_upserts(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)
    engine = session_factory.kw["bind"]
    service = MarketDataService(session_factory)
    service.store_result(
        "589850",
        ProviderResult.success(
            "CSV",
            [_record(date(2026, 7, 1), "1.00")],
        ),
    )
    statements: list[tuple[str, bool]] = []

    def capture(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        executemany,
    ):
        normalized = " ".join(statement.upper().split())
        if "MARKET_PRICES" in normalized:
            statements.append((normalized, executemany))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = service.store_result(
            "589850",
            ProviderResult.success(
                "CSV",
                [
                    _record(date(2026, 7, 1), "1.20"),
                    _record(date(2026, 7, 2), "1.30"),
                    _record(date(2026, 7, 2), "1.30"),
                ],
            ),
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    price_selects = [
        statement
        for statement, _executemany in statements
        if statement.startswith("SELECT")
    ]
    price_inserts = [
        (statement, executemany)
        for statement, executemany in statements
        if statement.startswith("INSERT")
    ]
    assert len(price_selects) == 1
    assert len(price_inserts) == 1
    assert "ON CONFLICT" in price_inserts[0][0]
    assert (
        response.records_added,
        response.records_updated,
        response.records_skipped,
    ) == (1, 1, 1)


def test_daily_store_chunks_full_history_below_sqlite_variable_limit(
    tmp_path,
) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    start = date(2010, 1, 1)
    records = [
        _record(start + timedelta(days=index), "1.00")
        for index in range(2300)
    ]

    response = service.store_result(
        "589850",
        ProviderResult.success("YAHOO_DIRECT_INDEX", records),
    )

    assert response.status is ProviderStatus.SUCCESS
    assert response.records_added == 2300
    with Session(engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(MarketPrice)
                .where(MarketPrice.timeframe == "daily")
            )
            == 2300
        )


def test_weekly_and_monthly_aggregation_are_ohlcv_and_idempotent(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    service.store_result(
        "589850",
        ProviderResult.success(
            "CSV",
            [
                _record(date(2026, 6, 26), "1.00"),
                _record(date(2026, 6, 30), "1.10"),
                _record(date(2026, 7, 1), "1.20"),
                _record(date(2026, 7, 3), "1.30"),
            ],
        ),
    )

    expected_dates = (
        date(2026, 6, 26),
        date(2026, 6, 30),
        date(2026, 7, 1),
        date(2026, 7, 3),
    )
    _aggregate(service, "589850", expected_dates)
    _aggregate(service, "589850", expected_dates)

    with Session(engine) as session:
        weekly = session.scalars(select(MarketPrice).where(MarketPrice.timeframe == "weekly")).all()
        monthly = session.scalars(select(MarketPrice).where(MarketPrice.timeframe == "monthly")).all()
    assert [(row.trade_date, row.open_price, row.high_price, row.low_price, row.close_price, row.volume, row.turnover) for row in weekly] == [
        (date(2026, 6, 26), Decimal("0.98"), Decimal("1.03"), Decimal("0.96"), Decimal("1.00"), Decimal("100"), Decimal("200")),
        (date(2026, 7, 3), Decimal("1.08"), Decimal("1.33"), Decimal("1.06"), Decimal("1.30"), Decimal("300"), Decimal("600")),
    ]
    assert [(row.trade_date, row.close_price, row.volume) for row in monthly] == [
        (date(2026, 6, 30), Decimal("1.10"), Decimal("200")),
        (date(2026, 7, 3), Decimal("1.30"), Decimal("200")),
    ]


def test_invalid_upstream_weekly_volume_batch_falls_back_entirely_to_daily_sum(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    daily = [
        _record(date(2026, 7, 6), "100"),
        _record(date(2026, 7, 7), "101"),
        _record(date(2026, 7, 8), "102"),
        _record(date(2026, 7, 13), "103"),
        _record(date(2026, 7, 14), "104"),
    ]
    service.store_result("399006", ProviderResult.success("AKSHARE_INDEX_SINA", daily))
    upstream = ProviderResult.success(
        "AKSHARE_INDEX_WEEKLY",
        [
            _record(date(2026, 7, 10), "102", source="AKSHARE_INDEX_WEEKLY").model_copy(
                update={"volume": Decimal("900")}
            ),
            _record(date(2026, 7, 17), "104", source="AKSHARE_INDEX_WEEKLY").model_copy(
                update={"volume": Decimal("0")}
            ),
        ],
    )

    _aggregate(
        service,
        "399006",
        (record.trade_date for record in daily),
        weekly_result=upstream,
    )

    with Session(engine) as session:
        weekly = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(Instrument.code == "399006", MarketPrice.timeframe == "weekly")
            .order_by(MarketPrice.trade_date)
        ).all()

    assert [row.volume * row.volume_multiplier for row in weekly] == [Decimal("300"), Decimal("200")]
    assert {row.source for row in weekly} == {"CSV"}
    assert {row.volume_source for row in weekly} == {
        "AGGREGATED_DAILY_VOLUME"
    }


def test_aggregation_preserves_high_period_volume_with_an_explicit_storage_multiplier(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    records = [
        _record(date(2026, 7, day), str(100 + day)).model_copy(update={"volume": Decimal("30000000000")})
        for day in (6, 7, 8, 9)
    ]
    service.store_result("399006", ProviderResult.success("AKSHARE_INDEX_SINA", records))

    _aggregate(service, "399006", (record.trade_date for record in records))

    with Session(engine) as session:
        daily = session.scalars(select(MarketPrice).where(MarketPrice.timeframe == "daily")).all()
        weekly = session.scalars(select(MarketPrice).where(MarketPrice.timeframe == "weekly")).all()
    assert daily[0].volume == Decimal("30000000000")
    assert weekly[0].volume == Decimal("12000000000")
    assert weekly[0].volume_multiplier == 10
    assert weekly[0].volume * weekly[0].volume_multiplier == Decimal("120000000000")
    assert "VOLUME_UNAVAILABLE" not in (weekly[0].source or "")


def test_failed_provider_returns_typed_cached_data_without_deleting_prices(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    service.store_result("589850", ProviderResult.success("CSV", [_record(date(2026, 7, 1), "1.00")]))

    response = service.update_from_provider("589850", FailingProvider())

    assert response.status is ProviderStatus.CACHED
    assert response.cache_used is True
    assert "unavailable" in response.error
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketPrice).where(MarketPrice.timeframe == "daily")) == 1
        assert session.scalar(select(DataUpdateLog.status).order_by(DataUpdateLog.id.desc()).limit(1)) == "error"


def test_csv_preview_aliases_and_confirm_import_keeps_valid_rows(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    importer = CsvImportService(session_factory)
    csv_text = "日期,开盘,最高,最低,收盘,成交量,成交额\n2026-07-01,1.00,1.10,0.90,1.05,1000,1050\nnot-a-date,1,1,1,nope,10,10\n"

    preview = importer.preview(csv_text)
    result = importer.confirm_import("589850", csv_text)

    assert preview.mapping["trade_date"] == "日期"
    assert preview.mapping["close_price"] == "收盘"
    assert preview.valid_rows == 1
    assert preview.invalid_rows == 1
    assert result.added == 1
    assert result.failed == 1
    assert result.row_errors[0].row_number == 3
    with Session(engine) as session:
        price = session.scalar(select(MarketPrice).where(MarketPrice.timeframe == "daily"))
    assert price is not None
    assert price.source == "CSV"
    assert price.volume_source == "CSV"
    assert price.close_price == Decimal("1.05")


def test_ndx_csv_import_discards_unverified_volume_before_storage(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    csv_text = (
        "date,open,high,low,close,volume,amount\n"
        "2026-07-01,20000,20100,19900,20050,123456,999999\n"
    )

    result = CsvImportService(session_factory).confirm_import("NDX", csv_text)

    assert result.added == 1
    with Session(engine) as session:
        price = session.scalar(
            select(MarketPrice)
            .join(Instrument)
            .where(Instrument.code == "NDX", MarketPrice.timeframe == "daily")
        )
    assert price is not None
    assert price.volume is None
    assert price.source == "CSV"
    assert price.volume_source == "VOLUME_UNAVAILABLE:DIRECT_INDEX"


def test_ndx_store_result_normalizes_response_and_all_daily_storage_volume(
    tmp_path,
) -> None:
    session_factory, engine = _service(tmp_path)
    with Session(engine) as session:
        instrument_id = session.scalar(
            select(Instrument.id).where(Instrument.code == "NDX")
        )
        session.add(
            MarketPrice(
                instrument_id=instrument_id,
                trade_date=date(2026, 6, 30),
                timeframe="daily",
                open_price=Decimal("19900"),
                high_price=Decimal("20100"),
                low_price=Decimal("19800"),
                close_price=Decimal("20000"),
                volume=Decimal("999"),
                volume_multiplier=1,
                source="LEGACY_NDX",
            )
        )
        session.commit()
    service = MarketDataService(session_factory)

    response = service.store_result(
        "NDX",
        ProviderResult.success(
            "NDX_DIRECT_INDEX_VOLUME",
            [
                MarketDataRecord(
                    trade_date=date(2026, 7, 1),
                    open_price=Decimal("20000"),
                    high_price=Decimal("20200"),
                    low_price=Decimal("19900"),
                    close_price=Decimal("20100"),
                    volume=Decimal("123456"),
                    source="NDX_DIRECT_INDEX_VOLUME",
                )
            ],
        ),
    )

    assert response.volume_availability == "not_available_for_direct_index"
    assert response.records[0].volume is None
    with Session(engine) as session:
        rows = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(Instrument.code == "NDX", MarketPrice.timeframe == "daily")
            .order_by(MarketPrice.trade_date)
        ).all()
    assert len(rows) == 2
    assert all(row.volume is None for row in rows)
    assert rows[-1].source == "NDX_DIRECT_INDEX_VOLUME"
    assert rows[-1].volume_source == "VOLUME_UNAVAILABLE:DIRECT_INDEX"


def test_ndx_prices_api_suppresses_legacy_volume_and_declares_unavailability(
    tmp_path,
) -> None:
    from backend.web import prices

    session_factory, engine = _service(tmp_path)
    with Session(engine) as session:
        instrument_id = session.scalar(
            select(Instrument.id).where(Instrument.code == "NDX")
        )
        session.add(
            MarketPrice(
                instrument_id=instrument_id,
                trade_date=date(2026, 7, 1),
                timeframe="daily",
                open_price=Decimal("20000"),
                high_price=Decimal("20200"),
                low_price=Decimal("19900"),
                close_price=Decimal("20100"),
                volume=Decimal("123456"),
                volume_multiplier=1,
                source="LEGACY_NDX",
            )
        )
        session.commit()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(sessions=session_factory))
    )

    payload = prices(request, "NDX")

    assert payload["volume_availability"] == "not_available_for_direct_index"
    assert payload["rows"][0]["volume"] is None
    assert "volume_source" in payload["rows"][0]


def test_v33_dashboard_exposes_two_active_targets_and_hides_ndx_benchmark(
    tmp_path,
) -> None:
    from backend.web import dashboard

    session_factory, engine = _service(tmp_path)
    with Session(engine) as session:
        instrument_id = session.scalar(
            select(Instrument.id).where(Instrument.code == "NDX")
        )
        session.add(
            MarketPrice(
                instrument_id=instrument_id,
                trade_date=date(2026, 7, 1),
                timeframe="daily",
                open_price=Decimal("20000"),
                high_price=Decimal("20200"),
                low_price=Decimal("19900"),
                close_price=Decimal("20100"),
                volume=Decimal("123456"),
                volume_multiplier=1,
                source="LEGACY_NDX",
            )
        )
        session.commit()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(sessions=session_factory))
    )

    payload = dashboard(request)
    by_code = {card["code"]: card for card in payload["instruments"]}

    assert set(by_code) == {"399006", "159941"}
    assert "NDX" not in by_code
    assert by_code["159941"]["volume_availability"] == "available"


def test_database_initialization_idempotently_clears_all_ndx_volume_artifacts(
    tmp_path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    config = tmp_path / "config"
    initialize_database(database, config)
    engine = create_database_engine(database)
    with engine.begin() as connection:
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "NDX")
        ).scalar_one()
        for timeframe in ("daily", "weekly", "monthly"):
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=instrument_id,
                    trade_date=date(2026, 7, 1),
                    timeframe=timeframe,
                    open_price=Decimal("20000"),
                    high_price=Decimal("20200"),
                    low_price=Decimal("19900"),
                    close_price=Decimal("20100"),
                    volume=(
                        None
                        if timeframe == "monthly"
                        else Decimal("123456")
                    ),
                    volume_multiplier=10,
                    source="LEGACY_NDX",
                )
            )
            connection.execute(
                IndicatorRecord.__table__.insert().values(
                    instrument_id=instrument_id,
                    indicator_date=date(2026, 7, 1),
                    timeframe=timeframe,
                    indicator_name="technical_indicators",
                    indicator_values={
                        "values": {
                            "volume_ma_20": "123456",
                            "rsi_14": "55",
                        },
                        "metadata": {
                            "formula_label": "preserve-me",
                            "data_cutoff": "2026-07-01",
                        },
                    },
                )
            )

    initialize_database(database, config)

    with Session(engine) as session:
        rows = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(Instrument.code == "NDX")
            .order_by(MarketPrice.timeframe)
        ).all()
        indicator_payloads = [
            record.indicator_values
            for record in session.scalars(
                select(IndicatorRecord)
                .join(Instrument)
                .where(Instrument.code == "NDX")
                .order_by(IndicatorRecord.timeframe)
            )
        ]
    assert {row.timeframe for row in rows} == {"daily", "weekly", "monthly"}
    assert all(row.volume is None for row in rows)
    assert all(row.volume_multiplier == 1 for row in rows)
    assert all(
        row.volume_source == "VOLUME_UNAVAILABLE:DIRECT_INDEX"
        for row in rows
    )
    assert all(row.close_price == Decimal("20100") for row in rows)
    assert all(
        payload["values"]["volume_ma_20"] is None
        for payload in indicator_payloads
    )
    assert all(
        payload["values"]["rsi_14"] == "55"
        and payload["metadata"]["formula_label"] == "preserve-me"
        for payload in indicator_payloads
    )

    initialize_database(database, config)
    with Session(engine) as session:
        after_second_initialize = [
            record.indicator_values
            for record in session.scalars(
                select(IndicatorRecord)
                .join(Instrument)
                .where(Instrument.code == "NDX")
                .order_by(IndicatorRecord.timeframe)
            )
        ]
    assert after_second_initialize == indicator_payloads


def test_refresh_api_uses_shared_calendar_backed_aggregation(
    monkeypatch,
) -> None:
    from backend import web

    record = _record(date(2026, 7, 1), "1.00", source="DIRECT_INDEX_DAILY")
    update = DataUpdateResponse(
        source="DIRECT_INDEX_DAILY",
        records=[record],
        cutoff_date=record.trade_date,
        records_received=1,
        records_written=1,
    )
    update_calls: list[tuple[str, date | None, date | None]] = []

    def update_from_providers(
        code: str,
        _providers,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataUpdateResponse:
        update_calls.append((code, start_date, end_date))
        return update

    market = SimpleNamespace(
        latest_daily_date=lambda _code: None,
        update_from_providers=update_from_providers,
    )
    recalculated: list[tuple[str, str]] = []

    def recalculate(code: str, timeframe: str) -> dict[str, str]:
        recalculated.append((code, timeframe))
        return {"status": "success", "timeframe": timeframe}

    state = SimpleNamespace(
        market=market,
        indicators=SimpleNamespace(
            recalculate=recalculate
        ),
        real_accounts=SimpleNamespace(recalculate_all_accounts=lambda: 0),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    class Provider:
        def fetch_weekly(
            self,
            _instrument_code,
            start_date: date | None = None,
            end_date: date | None = None,
        ):
            raise AssertionError("refresh must derive weekly volume from canonical daily rows")

    calls: list[tuple[str, ProviderResult | None, date | None, date | None]] = []
    aggregation_result = {
        "weekly": 1,
        "monthly": 0,
        "timeframe_status": {
            "weekly": {"status": "success"},
            "monthly": {"status": "error"},
        },
    }
    monkeypatch.setattr(web, "AkShareIndexProvider", Provider)
    monkeypatch.setattr(
        web,
        "_aggregate_market_timeframes",
        lambda _request, code, *, weekly_result=None, as_of=None, changed_start_date=None: (
            calls.append((code, weekly_result, as_of, changed_start_date))
            or aggregation_result
        ),
    )

    payload = web.refresh_market(request, "399006")

    assert calls == [("399006", None, record.trade_date, None)]
    assert update_calls == [("399006", None, None)]
    assert recalculated == [("399006", "daily"), ("399006", "weekly")]
    assert payload["aggregation"] == aggregation_result
    assert payload["aggregation_status"] == "partial_failure"
    assert payload["indicator_recalculation"] == {
        "daily": {"status": "success", "timeframe": "daily"},
        "weekly": {"status": "success", "timeframe": "weekly"},
    }


def test_exchange_calendar_freshness_uses_close_buffer_and_weekend() -> None:
    provider = ExchangeCalendarProvider()

    assert provider.latest_completed_session(
        "159941",
        as_of=datetime(2026, 8, 7, 7, 29, tzinfo=timezone.utc),
    ) == date(2026, 8, 6)
    assert provider.latest_completed_session(
        "159941",
        as_of=datetime(2026, 8, 7, 7, 30, tzinfo=timezone.utc),
    ) == date(2026, 8, 7)
    assert provider.latest_completed_session(
        "159941",
        as_of=datetime(2026, 8, 9, 4, 0, tzinfo=timezone.utc),
    ) == date(2026, 8, 7)


def test_exchange_calendar_freshness_honors_us_daylight_saving_close() -> None:
    provider = ExchangeCalendarProvider()

    assert provider.latest_completed_session(
        "NDX",
        as_of=datetime(2026, 8, 7, 20, 29, tzinfo=timezone.utc),
    ) == date(2026, 8, 6)
    assert provider.latest_completed_session(
        "NDX",
        as_of=datetime(2026, 8, 7, 20, 30, tzinfo=timezone.utc),
    ) == date(2026, 8, 7)


def test_refresh_skips_network_when_local_daily_data_is_already_fresh(
    monkeypatch,
) -> None:
    from backend import web

    cutoff = date(2026, 8, 7)

    class Market:
        def latest_daily_date(self, _code: str) -> date:
            return cutoff

        def expected_latest_daily_date(self, _code: str, *, as_of: datetime) -> date:
            assert as_of.tzinfo is not None
            return cutoff

        def update_from_providers(self, *_args, **_kwargs):
            raise AssertionError("an already-fresh refresh must not call the network")

    monkeypatch.setattr(web, "AkShareIndexProvider", lambda: object())
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(market=Market())))

    payload = web.refresh_market(request, "399006")

    assert payload["refresh_outcome"] == "ALREADY_FRESH"
    assert payload["network_requested"] is False
    assert payload["local_cutoff_before"] == "2026-08-07"
    assert payload["expected_cutoff"] == "2026-08-07"
    assert payload["local_cutoff_after"] == "2026-08-07"


def test_refresh_updates_stale_local_data_and_verifies_expected_session(
    monkeypatch,
) -> None:
    from backend import web

    expected = date(2026, 8, 7)

    class Market:
        cutoff = date(2026, 7, 31)

        def latest_daily_date(self, _code: str) -> date:
            return self.cutoff

        def expected_latest_daily_date(self, _code: str, *, as_of: datetime) -> date:
            return expected

        def update_from_providers(
            self,
            _code: str,
            _providers,
            start_date: date | None,
            end_date: date | None,
        ) -> DataUpdateResponse:
            assert start_date == date(2026, 7, 17)
            assert end_date is None
            self.cutoff = expected
            record = _record(expected, "1.52", source="AKSHARE_INDEX")
            return DataUpdateResponse(
                source="AKSHARE_INDEX",
                records=[record],
                cutoff_date=expected,
                records_received=1,
                records_written=1,
                records_added=1,
            )

    aggregate_calls: list[date | None] = []
    monkeypatch.setattr(web, "AkShareIndexProvider", lambda: object())
    monkeypatch.setattr(
        web,
        "_aggregate_market_timeframes",
        lambda *_args, changed_start_date=None, **_kwargs: (
            aggregate_calls.append(changed_start_date)
            or {"timeframe_status": {}}
        ),
    )
    state = SimpleNamespace(
        market=Market(),
        indicators=SimpleNamespace(
            recalculate=lambda _code, timeframe: {"status": "success", "timeframe": timeframe}
        ),
    )

    payload = web.refresh_market(
        SimpleNamespace(app=SimpleNamespace(state=state)),
        "399006",
    )

    assert payload["refresh_outcome"] == "UPDATED"
    assert payload["network_requested"] is True
    assert payload["verified_fresh"] is True
    assert payload["local_cutoff_before"] == "2026-07-31"
    assert payload["local_cutoff_after"] == "2026-08-07"
    assert payload["real_accounts_recalculated"] == 0
    assert aggregate_calls == [date(2026, 7, 17)]


def test_refresh_reports_still_stale_when_upstream_stops_before_expected_session(
    monkeypatch,
) -> None:
    from backend import web

    class Market:
        cutoff = date(2026, 7, 31)

        def latest_daily_date(self, _code: str) -> date:
            return self.cutoff

        def expected_latest_daily_date(self, _code: str, *, as_of: datetime) -> date:
            return date(2026, 8, 7)

        def update_from_providers(self, *_args, **_kwargs) -> DataUpdateResponse:
            self.cutoff = date(2026, 8, 5)
            record = _record(self.cutoff, "1.50", source="AKSHARE_INDEX")
            return DataUpdateResponse(
                source="AKSHARE_INDEX",
                records=[record],
                cutoff_date=self.cutoff,
                records_received=1,
                records_written=0,
                records_skipped=1,
            )

    monkeypatch.setattr(web, "AkShareIndexProvider", lambda: object())
    state = SimpleNamespace(
        market=Market(),
        indicators=SimpleNamespace(recalculate=lambda *_args: None),
        real_accounts=SimpleNamespace(recalculate_all_accounts=lambda: 0),
    )

    payload = web.refresh_market(
        SimpleNamespace(app=SimpleNamespace(state=state)),
        "399006",
    )

    assert payload["refresh_outcome"] == "STILL_STALE"
    assert payload["verified_fresh"] is False
    assert payload["expected_cutoff"] == "2026-08-07"
    assert payload["local_cutoff_after"] == "2026-08-05"
    assert "仍未达到应有交易日" in payload["refresh_error"]


def test_refresh_rejects_duplicate_request_for_same_instrument() -> None:
    from backend import web

    lock = web._market_refresh_lock("518600")
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as captured:
            web.refresh_market(
                SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
                "518600",
            )
        assert captured.value.status_code == 409
        assert "正在刷新" in str(captured.value.detail)
    finally:
        lock.release()


def test_refresh_api_keeps_daily_overlap_and_skips_aggregation_for_cached_failure(
    monkeypatch,
) -> None:
    from backend import web

    latest = date(2026, 7, 30)
    expected_start = latest - timedelta(days=14)
    update_calls: list[tuple[str, date | None, date | None]] = []
    weekly_calls: list[tuple[str, date | None, date | None]] = []
    update = DataUpdateResponse(
        source="CACHE",
        error="unchanged",
        status=ProviderStatus.CACHED,
    )

    class Market:
        def latest_daily_date(self, code: str) -> date:
            assert code == "399006"
            return latest

        def update_from_providers(
            self,
            code: str,
            _providers,
            start_date: date | None = None,
            end_date: date | None = None,
        ) -> DataUpdateResponse:
            update_calls.append((code, start_date, end_date))
            return update

    class Provider:
        def fetch_weekly(
            self,
            code: str,
            start_date: date | None = None,
            end_date: date | None = None,
        ) -> ProviderResult:
            weekly_calls.append((code, start_date, end_date))
            return ProviderResult.failed("AKSHARE_INDEX_WEEKLY", "unavailable")

    state = SimpleNamespace(
        market=Market(),
        indicators=SimpleNamespace(recalculate=lambda *_args: None),
        real_accounts=SimpleNamespace(recalculate_all_accounts=lambda: 0),
    )
    monkeypatch.setattr(web, "AkShareIndexProvider", Provider)

    payload = web.refresh_market(
        SimpleNamespace(app=SimpleNamespace(state=state)),
        "399006",
    )

    # A cached response is not a successful refresh.  Do not make a second
    # full-history weekly request or recalculate indicators/accounts when the
    # daily provider was unavailable.
    assert weekly_calls == []
    assert update_calls == [("399006", expected_start, None)]
    assert payload["aggregation_status"] == "not_attempted"
    assert payload["refresh_outcome"] == "UPSTREAM_FAILED"


def test_incremental_refresh_preserves_old_periods_and_uses_daily_volume(
    monkeypatch,
    tmp_path,
) -> None:
    from backend import web

    session_factory, engine = _service(tmp_path)
    old_week = tuple(date(2025, 1, day) for day in range(6, 11))
    current_week = tuple(date(2026, 7, day) for day in range(20, 25))
    expected_sessions = (*old_week, *current_week)

    class Calendar:
        def sessions(
            self,
            _instrument_code: str,
            start_date: date,
            end_date: date,
        ) -> tuple[date, ...]:
            return tuple(
                day
                for day in expected_sessions
                if start_date <= day <= end_date
            )

    market = MarketDataService(
        session_factory,
        calendar_provider=Calendar(),
    )
    market.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                *[
                    _record(
                        day,
                        str(100 + index),
                        source="DIRECT_INDEX_DAILY",
                    )
                    for index, day in enumerate(old_week)
                ],
            ],
        ),
    )
    _aggregate(market, "399006", old_week)
    market.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [_record(current_week[0], "200", source="DIRECT_INDEX_DAILY")],
        ),
    )
    with Session(engine) as session:
        old_row_before = session.scalar(
            select(MarketPrice)
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe == "weekly",
                MarketPrice.trade_date == old_week[-1],
            )
        )
        assert old_row_before is not None
        old_row_identity = (
            old_row_before.id,
            old_row_before.open_price,
            old_row_before.high_price,
            old_row_before.low_price,
            old_row_before.close_price,
            old_row_before.volume,
            old_row_before.source,
            old_row_before.updated_at,
        )
    expected_overlap = current_week[0] - timedelta(days=14)
    daily_calls: list[tuple[date | None, date | None]] = []
    weekly_calls: list[tuple[date | None, date | None]] = []

    class Provider:
        source = "AKSHARE_INDEX"

        def fetch(
            self,
            _instrument_code: str,
            start_date: date | None = None,
            end_date: date | None = None,
        ) -> ProviderResult:
            daily_calls.append((start_date, end_date))
            return ProviderResult.success(
                "DIRECT_INDEX_DAILY",
                [
                    _record(
                        day,
                        str(200 + index),
                        source="DIRECT_INDEX_DAILY",
                    )
                    for index, day in enumerate(current_week)
                ],
            )

        def fetch_weekly(
            self,
            _instrument_code: str,
            start_date: date | None = None,
            end_date: date | None = None,
        ) -> ProviderResult:
            weekly_calls.append((start_date, end_date))
            return ProviderResult.success(
                "AKSHARE_INDEX_WEEKLY",
                [
                    _record(
                        old_week[-1],
                        "104",
                        source="AKSHARE_INDEX_WEEKLY",
                    ).model_copy(update={"volume": Decimal("500")}),
                    _record(
                        current_week[-1],
                        "204",
                        source="AKSHARE_INDEX_WEEKLY",
                    ).model_copy(update={"volume": Decimal("500")}),
                ],
            )

    state = SimpleNamespace(
        sessions=session_factory,
        market=market,
        indicators=SimpleNamespace(
            recalculate=lambda _code, timeframe: {
                "status": "success",
                "timeframe": timeframe,
            }
        ),
        real_accounts=SimpleNamespace(recalculate_all_accounts=lambda: 0),
    )
    monkeypatch.setattr(web, "AkShareIndexProvider", Provider)

    payload = web.refresh_market(
        SimpleNamespace(app=SimpleNamespace(state=state)),
        "399006",
    )

    assert daily_calls == [(expected_overlap, None)]
    assert weekly_calls == []
    assert payload["aggregation"]["timeframe_status"]["weekly"]["status"] == "success"
    with Session(engine) as session:
        weekly_rows = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe == "weekly",
            )
            .order_by(MarketPrice.trade_date)
        ).all()
    assert [row.trade_date for row in weekly_rows] == [
        old_week[-1],
        current_week[-1],
    ]
    assert {row.volume_source for row in weekly_rows} == {"AGGREGATED_DAILY_VOLUME"}
    assert (
        weekly_rows[0].id,
        weekly_rows[0].open_price,
        weekly_rows[0].high_price,
        weekly_rows[0].low_price,
        weekly_rows[0].close_price,
        weekly_rows[0].volume,
        weekly_rows[0].source,
        weekly_rows[0].updated_at,
    ) == old_row_identity


def test_manual_aggregate_api_uses_shared_calendar_backed_aggregation(
    monkeypatch,
) -> None:
    from backend import web

    recalculated: list[tuple[str, str]] = []

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                indicators=SimpleNamespace(
                    recalculate=lambda code, timeframe: recalculated.append(
                        (code, timeframe)
                    )
                )
            )
        )
    )
    calls: list[str] = []
    result = {
        "weekly": 1,
        "monthly": 0,
        "timeframe_status": {
            "weekly": {"status": "success", "records_written": 1},
            "monthly": {"status": "error", "records_written": 0},
        },
    }
    monkeypatch.setattr(
        web,
        "_aggregate_market_timeframes",
        lambda _request, code, *, weekly_result=None: (
            calls.append(code) or result
        ),
    )

    payload = web.aggregate_market(request, "399006")

    assert payload == result
    assert calls == ["399006"]
    assert recalculated == [("399006", "weekly")]


def test_manual_aggregate_api_skips_indicators_when_no_timeframe_published(
    monkeypatch,
) -> None:
    from backend import web

    recalculated: list[tuple[str, str]] = []
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                indicators=SimpleNamespace(
                    chart_timeframes=("daily", "weekly", "monthly"),
                    recalculate=lambda code, timeframe: recalculated.append(
                        (code, timeframe)
                    ),
                )
            )
        )
    )
    result = {
        "weekly": 0,
        "monthly": 0,
        "timeframe_status": {
            "weekly": {"status": "error"},
            "monthly": {"status": "blocked"},
        },
    }
    monkeypatch.setattr(
        web,
        "_aggregate_market_timeframes",
        lambda _request, code, *, weekly_result=None: result,
    )

    web.aggregate_market(request, "399006")

    assert recalculated == []


def test_csv_import_api_uses_shared_calendar_backed_aggregation(
    monkeypatch,
) -> None:
    from backend import web

    recalculated: list[tuple[str, str]] = []

    def recalculate(code: str, timeframe: str) -> dict[str, str]:
        recalculated.append((code, timeframe))
        return {"status": "success", "timeframe": timeframe}

    class Request:
        def __init__(self) -> None:
            self.app = SimpleNamespace(
                state=SimpleNamespace(
                    sessions=object(),
                    indicators=SimpleNamespace(
                        recalculate=recalculate
                    ),
                    real_accounts=SimpleNamespace(
                        recalculate_all_accounts=lambda: 0
                    ),
                )
            )

        async def body(self) -> bytes:
            return b"csv"

    class Importer:
        def __init__(self, _sessions) -> None:
            pass

        def confirm_import(self, _instrument_code, _contents) -> CsvImportResult:
            return CsvImportResult(instrument_code="399006", added=1)

    calls: list[str] = []
    aggregation_result = {
        "weekly": 0,
        "monthly": 1,
        "timeframe_status": {
            "weekly": {"status": "error"},
            "monthly": {"status": "success"},
        },
    }
    monkeypatch.setattr(web, "CsvImportService", Importer)
    monkeypatch.setattr(
        web,
        "_aggregate_market_timeframes",
        lambda _request, code, *, weekly_result=None: (
            calls.append(code) or aggregation_result
        ),
    )

    payload = asyncio.run(web.import_market_csv(Request(), "399006"))

    assert calls == ["399006"]
    assert recalculated == [("399006", "daily"), ("399006", "monthly")]
    assert payload["aggregation"] == aggregation_result
    assert payload["aggregation_status"] == "partial_failure"
    assert payload["indicator_recalculation"] == {
        "daily": {"status": "success", "timeframe": "daily"},
        "monthly": {"status": "success", "timeframe": "monthly"},
    }


def test_csv_confirm_reports_duplicate_as_skipped_not_added(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)
    importer = CsvImportService(session_factory)
    csv_text = "date,open,high,low,close,volume,amount\n2026-07-01,1,1.1,0.9,1,100,100\n2026-07-01,1,1.1,0.9,1,100,100\n"

    result = importer.confirm_import("589850", csv_text)

    assert result.added == 1
    assert result.updated == 0
    assert result.skipped == 1
    assert result.failed == 0


def test_demo_data_is_labelled_and_created_only_for_empty_price_cache(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)

    response = service.update_from_provider("589850", FailingProvider())

    assert response.status is ProviderStatus.DEMO
    assert response.demo is True
    assert response.source == "DEMO"
    with Session(engine) as session:
        demo_count = session.scalar(
            select(func.count()).select_from(MarketPrice).where(MarketPrice.source == "DEMO")
        )
        instruments_with_demo = session.scalar(
            select(func.count(func.distinct(MarketPrice.instrument_id))).where(MarketPrice.source == "DEMO")
        )
        session.add(
            MarketPrice(
                instrument_id=session.scalar(select(Instrument.id).where(Instrument.code == "589850")),
                trade_date=date(2026, 8, 1),
                timeframe="daily",
                close_price=Decimal("2.00"),
                source="CSV",
            )
        )
        session.commit()

    second = service.update_from_provider("589850", FailingProvider())
    with Session(engine) as session:
        demo_count_after = session.scalar(
            select(func.count()).select_from(MarketPrice).where(MarketPrice.source == "DEMO")
        )
    assert demo_count == 15
    assert instruments_with_demo == 3
    assert second.status is ProviderStatus.CACHED
    assert demo_count_after == demo_count


def test_failure_for_instrument_without_cache_returns_typed_error_when_other_cache_exists(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    service.store_result("589850", ProviderResult.success("CSV", [_record(date(2026, 7, 1), "1.00")]))

    response = service.update_from_provider("159915", FailingProvider())

    assert response.status is ProviderStatus.ERROR
    assert response.cache_used is False
    assert response.demo is False
    assert response.records == []


def test_incomplete_week_and_month_do_not_replace_last_complete_endpoints(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    completed = tuple(date(2026, 1, day) for day in range(26, 31))
    service.store_result(
        "589850",
        ProviderResult.success(
            "CSV",
            [_record(day, str(Decimal("1.00") + Decimal(index) / Decimal("100"))) for index, day in enumerate(completed)],
        ),
    )
    _aggregate(service, "589850", completed)
    partial = (date(2026, 2, 2), date(2026, 2, 3))
    service.store_result(
        "589850",
        ProviderResult.success(
            "CSV",
            [_record(day, str(Decimal("1.10") + Decimal(index) / Decimal("100"))) for index, day in enumerate(partial)],
        ),
    )
    service.aggregate_timeframes(
        "589850",
        expected_trade_dates=completed + partial,
        as_of=partial[-1],
    )
    expected = completed + tuple(date(2026, 2, day) for day in range(2, 7))

    service.aggregate_timeframes(
        "589850",
        expected_trade_dates=expected,
        as_of=partial[-1],
    )

    with Session(engine) as session:
        weekly = session.scalars(select(MarketPrice).where(MarketPrice.timeframe == "weekly")).all()
        monthly = session.scalars(select(MarketPrice).where(MarketPrice.timeframe == "monthly")).all()
    assert [(row.trade_date, row.close_price) for row in weekly] == [
        (date(2026, 1, 30), Decimal("1.04")),
        (date(2026, 2, 3), Decimal("1.11")),
    ]
    assert [(row.trade_date, row.close_price) for row in monthly] == [
        (date(2026, 1, 30), Decimal("1.04")),
        (date(2026, 2, 3), Decimal("1.11")),
    ]


def test_tokenless_tushare_is_explicitly_inactive_without_prompting(monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    result = TushareProvider().fetch("589850")

    assert result.status is ProviderStatus.INACTIVE
    assert result.records == []
    assert "not configured" in result.error


def test_configured_tushare_imports_client_and_normalizes_daily_response(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        def fund_daily(self, **kwargs):
            calls["request"] = kwargs
            return [
                {
                    "trade_date": "20260701",
                    "open": "1.00",
                    "high": "1.10",
                    "low": "0.90",
                    "close": "1.05",
                    "vol": "100",
                    "amount": "105",
                }
            ]

    def pro_api(token: str) -> FakeClient:
        calls["token"] = token
        return FakeClient()

    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(pro_api=pro_api))
    result = TushareProvider(token="configured-token").fetch("589850", date(2026, 7, 1), date(2026, 7, 1))

    assert result.status is ProviderStatus.SUCCESS
    assert result.records[0].close_price == Decimal("1.05")
    assert calls["token"] == "configured-token"
    assert calls["request"] == {"ts_code": "589850.SH", "start_date": "20260701", "end_date": "20260701"}


def test_csv_provider_accepts_standard_english_aliases() -> None:
    provider = CsvProvider(
        "trade_date,open,high,low,close,volume,amount\n20260701,1,2,0.5,1.5,100,150\n"
    )

    result = provider.fetch("589850")

    assert result.status is ProviderStatus.SUCCESS
    assert result.records[0].trade_date == date(2026, 7, 1)
    assert result.records[0].source == "CSV"


@pytest.mark.parametrize(
    "values",
    [
        {"close_price": Decimal("1.000000001")},
        {"open_price": Decimal("0")},
        {"high_price": Decimal("0.90"), "close_price": Decimal("1.00")},
        {"low_price": Decimal("1.10"), "close_price": Decimal("1.00")},
        {"volume": Decimal("-1")},
        {"amount": Decimal("-1")},
    ],
)
def test_market_record_rejects_invalid_ohlcv_before_database_write(values) -> None:
    payload = _record(date(2026, 7, 1), "1.00").model_dump()
    payload.update(values)

    with pytest.raises(ValidationError):
        MarketDataRecord(**payload)


def test_csv_preview_and_confirm_mark_invalid_price_row_without_writing(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    importer = CsvImportService(session_factory)
    contents = "date,open,high,low,close,volume,amount\n2026-07-01,1,0.9,0.8,1,10,10\n"

    preview = importer.preview(contents)
    result = importer.confirm_import("589850", contents)

    assert preview.valid_rows == 0
    assert preview.invalid_rows == 1
    assert result.failed == 1
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketPrice)) == 0


def test_concurrent_same_daily_upsert_has_one_row_and_no_integrity_error(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    barrier = threading.Barrier(2)
    responses = []
    errors = []

    def store() -> None:
        try:
            barrier.wait()
            responses.append(service.store_result("589850", ProviderResult.success("CSV", [_record(date(2026, 7, 1), "1.00")])) )
        except Exception as error:  # regression assertion is made on this collection
            errors.append(error)

    threads = [threading.Thread(target=store) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(response.records_added for response in responses) == [0, 1]
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketPrice).where(MarketPrice.timeframe == "daily")) == 1


def test_cached_demo_rows_remain_explicitly_labelled_after_provider_failure(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    service.update_from_provider("589850", FailingProvider())

    response = service.update_from_provider("589850", FailingProvider())

    assert response.source == "DEMO"
    assert response.demo is True
    assert response.status is ProviderStatus.DEMO
    assert response.cache_used is True


def test_tushare_amount_is_normalized_from_thousand_cny_to_cny() -> None:
    class FakeClient:
        def fund_daily(self, **_kwargs):
            return [{"trade_date": "20260701", "open": "1", "high": "1.1", "low": "0.9", "close": "1", "vol": "100", "amount": "12.34"}]

    result = TushareProvider(token="token", client=FakeClient()).fetch("589850")

    assert result.records[0].amount == Decimal("12340")


def test_akshare_and_tushare_preserve_zero_volume_and_amount() -> None:
    akshare = AkShareProvider(
        history_loader=lambda **_kwargs: [
            {"日期": "2026-07-01", "开盘": 1, "最高": 1.1, "最低": 0.9, "收盘": 1, "成交量": 0, "成交额": 0}
        ]
    ).fetch("589850")

    class TushareZeroClient:
        def fund_daily(self, **_kwargs):
            return [{"trade_date": "20260701", "open": 1, "high": 1.1, "low": 0.9, "close": 1, "vol": 0, "amount": 0}]

    tushare = TushareProvider(token="token", client=TushareZeroClient()).fetch("589850")

    assert (akshare.records[0].volume, akshare.records[0].amount) == (Decimal("0"), Decimal("0"))
    assert (tushare.records[0].volume, tushare.records[0].amount) == (Decimal("0"), Decimal("0"))


def test_unchanged_aggregation_writes_zero_rows_and_log_counts(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    service = MarketDataService(session_factory)
    service.store_result("589850", ProviderResult.success("CSV", [_record(date(2026, 7, 1), "1.00")]))
    _aggregate(service, "589850", (date(2026, 7, 1),))

    outcome = _aggregate(service, "589850", (date(2026, 7, 1),))

    assert outcome["weekly"] == 0
    assert outcome["monthly"] == 0
    assert outcome["timeframe_status"]["weekly"]["status"] == "success"
    assert outcome["timeframe_status"]["monthly"]["status"] == "success"
    with Session(engine) as session:
        logs = session.scalars(
            select(DataUpdateLog).where(DataUpdateLog.timeframe.in_(("weekly", "monthly"))).order_by(DataUpdateLog.id.desc()).limit(2)
        ).all()
    assert [log.records_written for log in logs] == [0, 0]


def test_csv_provider_handles_missing_path_and_keeps_partial_row_warnings(tmp_path) -> None:
    missing = CsvProvider(tmp_path / "missing.csv").fetch("589850")
    partial = CsvProvider("date,open,high,low,close\n2026-07-01,1,1.1,0.9,1\nbad,1,1.1,0.9,1\n").fetch("589850")

    assert missing.status is ProviderStatus.ERROR
    assert missing.warnings
    assert partial.status is ProviderStatus.SUCCESS
    assert len(partial.records) == 1
    assert partial.warnings


def test_csv_confirm_handles_unreadable_path_as_failed_result(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)

    result = CsvImportService(session_factory).confirm_import("589850", tmp_path / "missing.csv")

    assert result.failed == 1
    assert result.row_errors


def test_csv_preview_handles_unreadable_path_as_row_zero_error(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)

    preview = CsvImportService(session_factory).preview(tmp_path / "missing.csv")

    assert (preview.valid_rows, preview.invalid_rows) == (0, 1)
    assert preview.row_errors[0].row_number == 0


def test_short_csv_row_is_a_row_error_in_preview_and_confirm(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    importer = CsvImportService(session_factory)
    contents = (
        "date,open,high,low,close,volume,amount\n"
        "2026-07-01,1,1.1,0.9,1,10,10\n"
        "2026-07-02,1,1.1\n"
    )

    preview = importer.preview(contents)
    result = importer.confirm_import("589850", contents)

    assert (preview.valid_rows, preview.invalid_rows) == (1, 1)
    assert (result.added, result.failed) == (1, 1)
    assert result.row_errors[0].row_number == 3
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketPrice).where(MarketPrice.timeframe == "daily")) == 1


def test_aggregation_blocks_legacy_close_only_daily_row_instead_of_fabricating_ohlc(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    with Session(engine) as session:
        instrument_id = session.scalar(select(Instrument.id).where(Instrument.code == "589850"))
        session.add(
            MarketPrice(
                instrument_id=instrument_id,
                trade_date=date(2026, 7, 1),
                timeframe="daily",
                close_price=Decimal("1.23"),
                source="LEGACY",
            )
        )
        session.commit()

    with pytest.raises(ValueError, match="INVALID_DAILY_OHLC"):
        _aggregate(
            MarketDataService(session_factory),
            "589850",
            (date(2026, 7, 1),),
        )

    with Session(engine) as session:
        rows = session.scalars(select(MarketPrice).where(MarketPrice.timeframe.in_(("weekly", "monthly")))).all()
    assert rows == []


def test_csv_confirm_unknown_instrument_returns_global_failure(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)
    result = CsvImportService(session_factory).confirm_import(
        "unknown-etf", "date,open,high,low,close\n2026-07-01,1,1.1,0.9,1\n"
    )

    assert result.added == 0
    assert result.failed == 1
    assert result.row_errors[0].row_number == 0
    assert "Unknown instrument" in result.row_errors[0].message


def test_long_csv_row_is_rejected_in_preview_and_confirm(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)
    importer = CsvImportService(session_factory)
    contents = "date,open,high,low,close\n2026-07-01,1,1.1,0.9,1,EXTRA\n"

    preview = importer.preview(contents)
    result = importer.confirm_import("589850", contents)

    assert (preview.valid_rows, preview.invalid_rows) == (0, 1)
    assert (result.added, result.failed) == (0, 1)
    assert result.row_errors[0].row_number == 2
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(MarketPrice).where(MarketPrice.timeframe == "daily")) == 0


def test_unknown_instrument_returns_typed_validation_outcome(tmp_path) -> None:
    session_factory, _engine = _service(tmp_path)
    response = MarketDataService(session_factory).store_result(
        "not-a-watchlist-code", ProviderResult.success("CSV", [_record(date(2026, 7, 1), "1.00")])
    )

    assert response.status is ProviderStatus.ERROR
    assert "Unknown instrument" in response.error


def test_update_log_records_provider_timing_and_actual_outcomes(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)

    class TimedProvider:
        source = "TEST_PROVIDER"
        called_at = None

        def fetch(self, instrument_code, start_date=None, end_date=None):
            self.called_at = datetime.now(timezone.utc)
            return ProviderResult.success(self.source, [_record(date(2026, 7, 1), "1.00")])

    provider = TimedProvider()
    response = MarketDataService(session_factory).update_from_provider("589850", provider)

    with Session(engine) as session:
        log = session.scalar(select(DataUpdateLog).order_by(DataUpdateLog.id.desc()))
    assert response.records_added == 1
    assert log.source == "TEST_PROVIDER"
    assert log.records_added == 1 and log.records_updated == 0 and log.records_skipped == 0
    assert log.started_at <= provider.called_at <= log.completed_at


def test_multi_provider_success_log_keeps_pre_fetch_start_time(tmp_path) -> None:
    session_factory, engine = _service(tmp_path)

    class FailedThenTimedProvider:
        source = "SECOND_PROVIDER"
        called_at = None

        def fetch(self, instrument_code, start_date=None, end_date=None):
            self.called_at = datetime.now(timezone.utc)
            return ProviderResult.success(self.source, [_record(date(2026, 7, 1), "1.00")])

    provider = FailedThenTimedProvider()
    response = MarketDataService(session_factory).update_from_providers("589850", [FailingProvider(), provider])

    with Session(engine) as session:
        log = session.scalar(select(DataUpdateLog).order_by(DataUpdateLog.id.desc()))
    assert response.status is ProviderStatus.SUCCESS
    assert log.source == "SECOND_PROVIDER"
    assert log.started_at <= provider.called_at <= log.completed_at
