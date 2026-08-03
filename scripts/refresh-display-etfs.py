"""Refresh and validate the three display-only ETF chart datasets.

This maintenance command deliberately touches only market_prices,
indicator_records, and data_update_logs for 518600/512800/512690.  These
instruments are excluded from every model/training universe.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
import sys

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.database.session import create_session_factory
from backend.app.models.models import IndicatorRecord, Instrument, MarketPrice
from backend.app.schemas.market import ProviderResult
from backend.app.services.indicator_service import IndicatorService
from backend.app.services.market_calendar import ExchangeCalendarProvider
from backend.app.services.market_data import MarketDataService
from backend.app.services.providers import AkShareEtfResearchProvider


CODES = ("518600", "512800", "512690")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="aggregate and calculate from already stored daily ETF bars",
    )
    parser.add_argument(
        "--source-database",
        type=Path,
        help="reuse verified real daily bars from another local SQLite database",
    )
    args = parser.parse_args()
    database = args.database.resolve()
    if not database.is_file():
        raise SystemExit(f"database does not exist: {database}")

    sessions = create_session_factory(database)
    market = MarketDataService(
        sessions,
        calendar_provider=ExchangeCalendarProvider(),
    )
    indicators = IndicatorService(sessions)
    source_sessions = (
        create_session_factory(args.source_database.resolve())
        if args.source_database is not None
        else None
    )
    source_market = (
        MarketDataService(
            source_sessions,
            calendar_provider=ExchangeCalendarProvider(),
        )
        if source_sessions is not None
        else None
    )
    report: dict[str, object] = {
        "database": str(database),
        "reuse_existing": bool(args.reuse_existing),
        "instruments": {},
    }

    for code in CODES:
        provider = AkShareEtfResearchProvider()
        refresh: dict[str, object] | None = None
        weekly_result = None
        if source_sessions is not None and source_market is not None:
            with source_sessions() as source_session:
                source_instrument = source_session.scalar(
                    select(Instrument).where(Instrument.code == code)
                )
                source_rows = (
                    list(
                        source_session.scalars(
                            select(MarketPrice)
                            .where(
                                MarketPrice.instrument_id == source_instrument.id,
                                MarketPrice.timeframe == "daily",
                            )
                            .order_by(MarketPrice.trade_date)
                        )
                    )
                    if source_instrument is not None
                    else []
                )
                source_records = [
                    source_market._as_record(row) for row in source_rows
                ]
            if not source_records:
                raise RuntimeError(f"{code} source database has no daily records")
            result = market.store_result(
                code,
                ProviderResult.success("LOCAL_VERIFIED_ETF_CACHE", source_records),
                requested_source="LOCAL_VERIFIED_ETF_CACHE",
            )
            refresh = result.model_dump(mode="json")
        elif not args.reuse_existing:
            result = market.update_from_providers(code, [provider], None, None)
            refresh = result.model_dump(mode="json")
            if not result.records:
                raise RuntimeError(f"{code} refresh returned no usable daily records")
            weekly_result = provider.fetch_weekly(code, None, None)
        aggregation = market.aggregate_periods(
            code,
            as_of=None,
            weekly_result=weekly_result,
        )
        calculations = indicators.recalculate_all_timeframes(code)
        with sessions() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == code))
            if instrument is None:
                raise RuntimeError(f"missing instrument seed: {code}")
            counts = {
                timeframe: int(
                    session.scalar(
                        select(func.count()).select_from(MarketPrice).where(
                            MarketPrice.instrument_id == instrument.id,
                            MarketPrice.timeframe == timeframe,
                        )
                    )
                    or 0
                )
                for timeframe in ("daily", "weekly", "monthly")
            }
            indicator_counts = {
                timeframe: int(
                    session.scalar(
                        select(func.count()).select_from(IndicatorRecord).where(
                            IndicatorRecord.instrument_id == instrument.id,
                            IndicatorRecord.timeframe == timeframe,
                        )
                    )
                    or 0
                )
                for timeframe in ("daily", "weekly", "monthly")
            }
            latest = session.scalar(
                select(MarketPrice.trade_date)
                .where(
                    MarketPrice.instrument_id == instrument.id,
                    MarketPrice.timeframe == "daily",
                )
                .order_by(MarketPrice.trade_date.desc())
                .limit(1)
            )
            prices = list(
                session.scalars(
                    select(MarketPrice)
                    .where(MarketPrice.instrument_id == instrument.id)
                    .order_by(MarketPrice.timeframe, MarketPrice.trade_date)
                )
            )
            invalid_ohlc = [
                row.id
                for row in prices
                if row.open_price <= 0
                or row.high_price < max(row.open_price, row.close_price)
                or row.low_price > min(row.open_price, row.close_price)
                or row.low_price <= 0
                or row.volume is None
                or row.volume < 0
            ]
            demo_source_count = sum(
                1 for row in prices if (row.source or "").upper().startswith("DEMO")
            )
            indicator_rows = list(
                session.scalars(
                    select(IndicatorRecord)
                    .where(IndicatorRecord.instrument_id == instrument.id)
                    .order_by(
                        IndicatorRecord.timeframe,
                        IndicatorRecord.indicator_date,
                    )
                )
            )
            missing_derivative_after_first: dict[str, list[str]] = {}
            for timeframe in ("daily", "weekly", "monthly"):
                timeframe_rows = [
                    row for row in indicator_rows if row.timeframe == timeframe
                ]
                missing_derivative_after_first[timeframe] = [
                    row.indicator_date.isoformat()
                    for row in timeframe_rows[1:]
                    if (row.indicator_values or {})
                    .get("values", {})
                    .get("dif_first_change")
                    is None
                ]
            omitted_suspension_session_present = bool(
                code == "512690"
                and any(
                    row.timeframe == "daily"
                    and row.trade_date == date(2021, 5, 14)
                    for row in prices
                )
            )
        if any(value <= 0 for value in counts.values()):
            raise RuntimeError(f"{code} has an empty chart timeframe: {counts}")
        if indicator_counts != counts:
            raise RuntimeError(
                f"{code} indicator/price coverage mismatch: {indicator_counts} != {counts}"
            )
        if invalid_ohlc:
            raise RuntimeError(f"{code} has invalid OHLCV rows: {invalid_ohlc[:10]}")
        if demo_source_count:
            raise RuntimeError(f"{code} contains {demo_source_count} DEMO price rows")
        if any(missing_derivative_after_first.values()):
            raise RuntimeError(
                f"{code} has missing DIF first changes after the first row: "
                f"{missing_derivative_after_first}"
            )
        if omitted_suspension_session_present:
            raise RuntimeError("512690 contains a fabricated 2021-05-14 daily bar")
        report["instruments"][code] = {
            "refresh": refresh,
            "aggregation": aggregation,
            "indicator_calculations": [asdict(item) for item in calculations],
            "price_counts": counts,
            "indicator_counts": indicator_counts,
            "latest_daily_date": None if latest is None else latest.isoformat(),
            "invalid_ohlcv_count": len(invalid_ohlc),
            "demo_source_count": demo_source_count,
            "missing_dif_first_change_after_first": missing_derivative_after_first,
            "fabricated_2021_05_14_present": omitted_suspension_session_present,
        }

    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
