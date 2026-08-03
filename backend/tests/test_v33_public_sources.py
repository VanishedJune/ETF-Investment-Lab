from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import V33PointInTimeObservation
from backend.app.services.v33_data_service import V33DataService
from backend.app.services.v33_public_sources import V33PublicSourcesService


def _database(tmp_path: Path):
    database_path = tmp_path / "data" / "investment_lab.db"
    initialize_database(database_path, tmp_path / "config")
    engine = create_database_engine(database_path)
    return engine, create_session_factory(database_path)


def _eastmoney_payload() -> dict[str, object]:
    # The production endpoint currently exposes these fourteen LSJZ fields.
    record = {
        "FSRQ": "2026-07-01",
        "DWJZ": "1.4251",
        "LJJZ": "5.7004",
        "SDATE": None,
        "ACTUALSYI": "",
        "NAVTYPE": "1",
        "JZZZL": "3.34",
        "SGZT": "场内买入",
        "SHZT": "场内卖出",
        "FHFCZ": "",
        "FHFCZ10": "",
        "FHFCBZ": "",
        "DTYPE": None,
        "FHSP": "",
    }
    assert len(record) == 14
    return {
        "Data": {"LSJZList": [record], "FundType": "007"},
        "ErrCode": 0,
        "ErrMsg": None,
        "TotalCount": 1,
        "PageSize": 500,
        "PageIndex": 1,
    }


def _haoetf_html() -> str:
    return """
    <html><body>
      <table><thead><tr><th>代码</th><th>名称</th></tr></thead>
      <tbody><tr><td>159941</td><td>纳指ETF</td></tr></tbody></table>
      <h5>历史数据</h5>
      <table>
        <thead><tr>
          <th>日期</th><th>收盘价</th><th>T-1日净值</th><th>T-1日估值</th>
          <th>估值误差</th><th>T-1日溢价率</th><th>份额(万份)</th>
          <th>份额涨幅</th><th>T-1日指数涨跌</th><th>估算仓位</th>
        </tr></thead>
        <tbody>
          <tr><td>2026-07-02</td><td>1.50</td><td>1.40</td><td>1.41</td>
              <td>0.1%</td><td>6.38%</td><td>1050</td><td>5%</td><td>1.2%</td><td>95%</td></tr>
          <tr><td>2026-07-01</td><td>1.45</td><td>1.38</td><td>1.39</td>
              <td>0.2%</td><td>4.32%</td><td>1000</td><td>1%</td><td>-0.4%</td><td>94%</td></tr>
        </tbody>
      </table>
    </body></html>
    """


def _yahoo_payload() -> dict[str, object]:
    london = ZoneInfo("Europe/London")
    timestamp = int(datetime(2026, 7, 1, 0, 0, tzinfo=london).timestamp())
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "USDCNY=X",
                        "exchangeTimezoneName": "Europe/London",
                    },
                    "timestamp": [timestamp],
                    "indicators": {"quote": [{"close": [6.789412345]}]},
                }
            ],
            "error": None,
        }
    }


def _valuation_loaders():
    return {
        "cyb_pe": lambda: [{"日期": date(2026, 7, 1), "指数": 2210, "平均市盈率": 35.25}],
        "cyb_pb": lambda: [{"日期": date(2026, 7, 1), "指数": 2210, "市净率": 4.75}],
    }


def _macro_loaders():
    return {
        "china_lpr": lambda: [{"TRADE_DATE": date(2026, 7, 20), "LPR1Y": 3.0, "LPR5Y": 3.5}],
        "china_pmi": lambda: [{"月份": "2026年06月份", "制造业-指数": 51.3}],
        "china_m2": lambda: [{"月份": "2026年06月份", "货币和准货币(M2)-同比增长": 8.3}],
        "china_cpi": lambda: [{"月份": "2026年06月份", "全国-同比增长": 0.7}],
        "us_cpi": lambda: [{"日期": date(2026, 7, 14), "今值": 0.2}],
        "us_unemployment": lambda: [{"日期": date(2026, 7, 3), "今值": 4.1}],
        "us_pmi": lambda: [{"日期": date(2026, 7, 24), "今值": 52.2}],
        "us_rate": lambda: [{"日期": date(2026, 7, 29), "今值": 4.25}],
        "bond_rates": lambda: [
            {
                "日期": date(2026, 7, 1),
                "中国国债收益率10年": 1.72,
                "中国国债收益率10年-2年": 0.31,
                "美国国债收益率10年": 4.36,
                "美国国债收益率10年-2年": 0.48,
            }
        ],
    }


def test_refresh_all_persists_namespaced_real_values_and_reports_gaps(tmp_path: Path) -> None:
    engine, session_factory = _database(tmp_path)
    fixed_now = datetime(2026, 8, 1, 6, 30, tzinfo=timezone.utc)
    service = V33PublicSourcesService(
        V33DataService(session_factory),
        eastmoney_nav_loader=lambda _url, _params, _headers: _eastmoney_payload(),
        haoetf_loader=lambda _url, _headers: _haoetf_html(),
        yahoo_loader=lambda _url, _params, _headers: _yahoo_payload(),
        valuation_loaders=_valuation_loaders(),
        macro_loaders=_macro_loaders(),
        now_provider=lambda: fixed_now,
    )

    result = service.refresh_all(date(2026, 6, 1), date(2026, 7, 31))

    assert result.status == "partial"  # unavailable fields remain explicit gaps
    assert all(item.status != "failed" for item in result.sources)
    assert "159941.aum" in result.missing_by_market["159941"]
    assert "159941.real_rate" in result.missing_by_market["159941"]
    assert "399006.real_rate" in result.missing_by_market["399006"]
    assert "399006.earnings_growth" in result.missing_by_market["399006"]
    assert "159941.earnings_growth" in result.missing_by_market["159941"]
    assert "399006.pe" in result.available_series
    assert "159941.fx_usdcny" in result.available_series

    with Session(engine) as session:
        rows = list(
            session.scalars(
                select(V33PointInTimeObservation).order_by(
                    V33PointInTimeObservation.series_code,
                    V33PointInTimeObservation.effective_date,
                )
            )
        )

    assert rows
    assert all("." in row.series_code for row in rows)
    assert all(row.series_code.split(".", 1)[1].islower() for row in rows)
    assert all(row.numeric_value is not None for row in rows)
    assert all(row.payload_json["not_imputed"] is True for row in rows)

    nav = next(row for row in rows if row.series_code == "159941.nav")
    assert nav.numeric_value == Decimal("1.42510000")
    assert nav.available_at == datetime(2026, 7, 3, 15, 59, tzinfo=timezone.utc)
    assert nav.payload_json["upstream_field_count"] == 14
    assert nav.payload_json["daily_growth_pct"] == "3.34"

    hao_rows = [row for row in rows if row.source == "HAOETF_159941_HISTORY"]
    assert hao_rows and all(row.available_at == fixed_now for row in hao_rows)
    subscription = next(
        row
        for row in hao_rows
        if row.series_code == "159941.subscriptions"
        and row.observation_date == date(2026, 7, 2)
    )
    assert subscription.numeric_value == Decimal("500000.00000000")
    assert subscription.payload_json["share_change_derived_from_adjacent_totals"] is True

    fx = next(row for row in rows if row.series_code == "159941.fx_usdcny")
    assert fx.numeric_value == Decimal("6.78941234")
    assert fx.payload_json["symbol_returned"] == "USDCNY=X"


def test_source_failure_does_not_block_other_sources_or_write_defaults(tmp_path: Path) -> None:
    engine, session_factory = _database(tmp_path)

    def fail_eastmoney(_url, _params, _headers):
        raise OSError("network denied")

    def fail_pe():
        raise RuntimeError("valuation upstream unavailable")

    def fail_macro():
        raise TimeoutError("macro timeout")

    service = V33PublicSourcesService(
        V33DataService(session_factory),
        eastmoney_nav_loader=fail_eastmoney,
        haoetf_loader=lambda _url, _headers: "<html><body>no table</body></html>",
        yahoo_loader=lambda _url, _params, _headers: _yahoo_payload(),
        valuation_loaders={
            "cyb_pe": fail_pe,
            "cyb_pb": lambda: [{"日期": date(2026, 7, 1), "市净率": 4.75}],
        },
        macro_loaders={
            "china_lpr": fail_macro,
            "china_pmi": fail_macro,
            "china_m2": fail_macro,
            "china_cpi": fail_macro,
            "us_cpi": fail_macro,
            "us_unemployment": fail_macro,
            "us_pmi": lambda: [{"日期": date(2026, 7, 24), "今值": 52.2}],
            "us_rate": fail_macro,
            "bond_rates": fail_macro,
        },
        now_provider=lambda: datetime(2026, 8, 1, tzinfo=timezone.utc),
    )

    result = service.refresh_all(date(2026, 7, 1), date(2026, 7, 31))

    by_source = {item.source: item for item in result.sources}
    assert result.status == "partial"
    assert by_source["EASTMONEY_159941_NAV"].status == "failed"
    assert "network denied" in (by_source["EASTMONEY_159941_NAV"].error or "")
    assert by_source["YAHOO_CHART_USDCNY"].status == "success"
    assert by_source["AKSHARE_LEGULEGU_CYB_VALUATION"].status == "partial"
    assert "cyb_pe" in (by_source["AKSHARE_LEGULEGU_CYB_VALUATION"].error or "")
    assert "159941.nav" in result.missing_by_market["159941"]
    assert "399006.pe" in result.missing_by_market["399006"]

    with Session(engine) as session:
        rows = list(session.scalars(select(V33PointInTimeObservation)))
    assert {row.series_code for row in rows} == {
        "159941.fx_usdcny",
        "399006.pb",
        "159941.pmi",
    }
    assert all(row.numeric_value not in {Decimal("0"), Decimal("50")} for row in rows)


def test_refresh_all_rejects_inverted_range_before_calling_loaders(tmp_path: Path) -> None:
    _engine, session_factory = _database(tmp_path)
    service = V33PublicSourcesService(V33DataService(session_factory))

    try:
        service.refresh_all(date(2026, 8, 1), date(2026, 7, 1))
    except ValueError as error:
        assert str(error) == "start_date cannot be after end_date"
    else:
        raise AssertionError("Expected inverted refresh range to fail")


def test_eastmoney_nav_uses_total_count_beyond_one_hundred_pages() -> None:
    total_count = 2_021
    page_size = 20
    requested_pages: list[int] = []

    class CaptureService:
        def __init__(self) -> None:
            self.observations = []

        def ingest_point_in_time_observations(self, observations):
            self.observations.extend(observations)
            return list(range(1, len(observations) + 1))

    def loader(_url, params, _headers):
        page = int(params["pageIndex"])
        requested_pages.append(page)
        first = (page - 1) * page_size
        stop = min(first + page_size, total_count)
        rows = [
            {
                "FSRQ": (date(2018, 1, 1) + timedelta(days=index)).isoformat(),
                "DWJZ": f"{1 + index / 100_000:.5f}",
                "LJJZ": f"{1 + index / 50_000:.5f}",
                "JZZZL": "0.01",
            }
            for index in range(first, stop)
        ]
        return {
            "Data": {"LSJZList": rows},
            "ErrCode": 0,
            "TotalCount": total_count,
            "PageSize": page_size,
            "PageIndex": page,
        }

    capture = CaptureService()
    service = V33PublicSourcesService(capture, eastmoney_nav_loader=loader)  # type: ignore[arg-type]

    result = service.refresh_eastmoney_nav()

    assert result.status == "success"
    assert result.observations_received == total_count
    assert result.observations_persisted == total_count
    assert len(capture.observations) == total_count
    assert requested_pages == list(range(1, 103))


def test_eastmoney_nav_rejects_implausible_total_count_before_persistence() -> None:
    class CaptureService:
        def __init__(self) -> None:
            self.called = False

        def ingest_point_in_time_observations(self, _observations):
            self.called = True
            raise AssertionError("unsafe oversized response must not be persisted")

    def loader(_url, _params, _headers):
        return {
            "Data": {"LSJZList": [{"FSRQ": "2026-07-01", "DWJZ": "1.4"}]},
            "ErrCode": 0,
            "TotalCount": 20_001,
            "PageSize": 20,
            "PageIndex": 1,
        }

    capture = CaptureService()
    service = V33PublicSourcesService(capture, eastmoney_nav_loader=loader)  # type: ignore[arg-type]

    result = service.refresh_eastmoney_nav()

    assert result.status == "failed"
    assert "20000-record safety cap" in (result.error or "")
    assert capture.called is False


def test_cyb_valuation_does_not_write_implied_earnings_growth_proxy() -> None:
    class CaptureService:
        def __init__(self) -> None:
            self.observations = []

        def ingest_point_in_time_observations(self, observations):
            self.observations.extend(observations)
            return list(range(1, len(observations) + 1))

    capture = CaptureService()
    service = V33PublicSourcesService(
        capture,  # type: ignore[arg-type]
        valuation_loaders={
            "cyb_pe": lambda: [
                {"日期": date(2025, 7, 1), "指数": 1_800, "平均市盈率": 30},
                {"日期": date(2026, 7, 1), "指数": 2_400, "平均市盈率": 30},
            ],
            "cyb_pb": lambda: [{"日期": date(2026, 7, 1), "市净率": 4.5}],
        },
    )

    result = service.refresh_akshare_valuation(
        start_date=date(2026, 7, 1), end_date=date(2026, 7, 1)
    )

    assert result.status == "partial"
    assert result.series_codes == ("399006.pb", "399006.pe")
    assert result.missing_series == ("399006.earnings_growth",)
    assert any("left missing by design" in warning for warning in result.warnings)
    assert {observation.series_code for observation in capture.observations} == {
        "399006.pb",
        "399006.pe",
    }
