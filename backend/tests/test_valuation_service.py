from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import ValuationRecord
from backend.app.schemas.valuation import ValuationDataRecord, ValuationProviderResult
from backend.app.services.valuation_service import ValuationService
from backend.app.services.providers import AkShareValuationProvider


class _StaticProvider:
    source = "AKSHARE_CSINDEX"

    def fetch(self, instrument_code: str) -> ValuationProviderResult:
        assert instrument_code == "000688"
        return ValuationProviderResult.success(
            self.source,
            [
                ValuationDataRecord(
                    valuation_date=date(2026, 7, 1),
                    pe_ratio=Decimal("20"),
                    pb_ratio=Decimal("2"),
                    dividend_yield=Decimal("1.1"),
                    source=self.source,
                    source_url="https://www.csindex.com.cn/",
                ),
                ValuationDataRecord(
                    valuation_date=date(2026, 7, 8),
                    pe_ratio=Decimal("40"),
                    pb_ratio=Decimal("3"),
                    dividend_yield=Decimal("0.9"),
                    source=self.source,
                    source_url="https://www.csindex.com.cn/",
                ),
            ],
        )


def _service(tmp_path):
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    return create_session_factory(database), create_database_engine(database)


def test_refresh_persists_real_valuation_history_with_auditable_source_and_percentile(tmp_path) -> None:
    sessions, engine = _service(tmp_path)

    result = ValuationService(sessions).refresh("000688", _StaticProvider())

    assert result.status == "success"
    assert result.records_written == 2
    with Session(engine) as session:
        rows = session.scalars(select(ValuationRecord).order_by(ValuationRecord.valuation_date)).all()
    assert [row.valuation_percentile for row in rows] == [Decimal("0.5"), Decimal("1")]
    assert rows[-1].raw_values["source"] == "AKSHARE_CSINDEX"
    assert rows[-1].raw_values["source_url"] == "https://www.csindex.com.cn/"
    assert rows[-1].raw_values["proxy"] is False


def test_valuation_read_returns_cached_history_without_fabricating_missing_values(tmp_path) -> None:
    sessions, _engine = _service(tmp_path)
    service = ValuationService(sessions)
    service.refresh("000688", _StaticProvider())

    result = service.read("000688")

    assert result["latest"]["pe_ratio"] == Decimal("40")
    assert result["latest"]["source"] == "AKSHARE_CSINDEX"
    assert result["source_status"] == "cached"


def test_akshare_valuation_provider_uses_only_direct_index_identifiers() -> None:
    china_calls: list[dict[str, object]] = []
    spot_calls: list[dict[str, object]] = []
    us_calls: list[dict[str, object]] = []

    def china_loader(**kwargs):  # type: ignore[no-untyped-def]
        china_calls.append(kwargs)
        return [{"日期": "2026-07-29", "市盈率1": "96.04", "股息率1": "0.24"}]

    def spot_loader(**kwargs):  # type: ignore[no-untyped-def]
        spot_calls.append(kwargs)
        return [{"代码": "399006", "市盈率-动态": "42.1", "市净率": "5.2"}]

    def us_loader(**kwargs):  # type: ignore[no-untyped-def]
        us_calls.append(kwargs)
        if kwargs["indicator"] == "市盈率(TTM)":
            return [{"日期": "2026-07-29", "数值": "31.5"}]
        return [{"日期": "2026-07-29", "数值": "7.1"}]

    provider = AkShareValuationProvider(
        china_index_loader=china_loader,
        china_spot_loader=spot_loader,
        us_loader=us_loader,
    )

    china = provider.fetch("000688")
    growth = provider.fetch("399006")
    nasdaq = provider.fetch("NDX")

    assert china.records[0].pe_ratio == Decimal("96.04")
    assert growth.records[0].pb_ratio == Decimal("5.2")
    assert nasdaq.records[0].pe_ratio == Decimal("31.5")
    assert china_calls == [{"symbol": "000688"}]
    assert spot_calls == [{"symbol": "深证系列指数"}]
    assert all(call["symbol"] == "NDX" for call in us_calls)
    assert all(record.proxy is False for record in (*china.records, *growth.records, *nasdaq.records))
