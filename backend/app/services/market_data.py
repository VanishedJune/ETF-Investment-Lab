"""Local, incremental market-price storage with cache and demo safety nets."""

from __future__ import annotations

from collections import defaultdict
import calendar
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import logging
from typing import Callable, Iterable, Sequence

from sqlalchemy import bindparam, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..models.models import DataUpdateLog, Instrument, MarketPrice
from ..schemas.market import (
    DataUpdateResponse,
    MarketDataRecord,
    ProviderResult,
    ProviderStatus,
)
from ..weekly_analysis.aggregation import (
    aggregate_bars,
    apply_upstream_weekly_volume,
)
from ..weekly_analysis.domain import (
    AggregationResult,
    DailyBar,
    PeriodBar,
    UpstreamWeeklyVolume,
)
from .providers import MarketDataProvider
from .market_calendar import CalendarProvider, ExchangeCalendarProvider
from .instrument_universe import DISPLAY_ONLY_CODES, is_etf_code


DEMO_DATES = (
    date(2026, 1, 5),
    date(2026, 1, 6),
    date(2026, 1, 7),
    date(2026, 1, 8),
    date(2026, 1, 9),
)
DEMO_BASE_PRICES = {"589850": Decimal("1.0000"), "159915": Decimal("2.0000"), "159941": Decimal("1.1000")}
DIRECT_INDEX_DEMO_BASE_PRICES = {
    "000688": Decimal("1000"),
    "399006": Decimal("2000"),
    "NDX": Decimal("20000"),
}
MAX_FIXED_POINT_VALUE = Decimal(2**63 - 1) / Decimal(10**8)
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MarketDataService:
    """Writes only additive/upserted local data and never clears cached prices."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        calendar_provider: CalendarProvider | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.calendar_provider = calendar_provider or ExchangeCalendarProvider()

    @staticmethod
    def _instrument(session: Session, instrument_code: str) -> Instrument:
        instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
        if instrument is None:
            raise ValueError(f"Unknown instrument code: {instrument_code}")
        return instrument

    def _known_instrument(self, instrument_code: str) -> bool:
        with self.session_factory() as session:
            return session.scalar(select(Instrument.id).where(Instrument.code == instrument_code)) is not None

    def latest_daily_date(self, instrument_code: str) -> date | None:
        with self.session_factory() as session:
            return session.scalar(
                select(func.max(MarketPrice.trade_date))
                .join(Instrument)
                .where(
                    Instrument.code == instrument_code,
                    MarketPrice.timeframe == "daily",
                )
            )

    def expected_latest_daily_date(
        self,
        instrument_code: str,
        *,
        as_of: datetime | None = None,
    ) -> date | None:
        """Return the latest exchange session expected from public vendors."""

        current = as_of or _now()
        resolver = getattr(self.calendar_provider, "latest_completed_session", None)
        if callable(resolver):
            return resolver(instrument_code, as_of=current)

        # Lightweight injected calendars used by tests and offline adapters
        # may only implement ``sessions``.  This fallback keeps those adapters
        # usable, while production uses the close-time-aware method above.
        sessions = self.calendar_provider.sessions(
            instrument_code,
            current.date() - timedelta(days=31),
            current.date(),
        )
        return sessions[-1] if sessions else None

    @staticmethod
    def _aggregation_rebuild_start(changed_start_date: date | None) -> date | None:
        if changed_start_date is None:
            return None
        week_start = changed_start_date - timedelta(days=changed_start_date.weekday())
        # If the affected ISO week crosses a month boundary, include the full
        # earlier month so a partial slice can never overwrite a monthly bar.
        return min(
            changed_start_date.replace(day=1),
            week_start.replace(day=1),
        )

    @staticmethod
    def _unknown_instrument_response(instrument_code: str, result: ProviderResult) -> DataUpdateResponse:
        return DataUpdateResponse(
            source=result.source,
            fetched_at=result.fetched_at,
            error=f"Unknown instrument code: {instrument_code}",
            status=ProviderStatus.ERROR,
        )

    @staticmethod
    def _price_values(record: MarketDataRecord, default_source: str) -> dict[str, object]:
        price_source = record.source or default_source
        return {
            "open_price": record.open_price,
            "high_price": record.high_price,
            "low_price": record.low_price,
            "close_price": record.close_price,
            "adjusted_close_price": record.adjusted_close_price,
            "volume": record.volume,
            "volume_multiplier": record.volume_multiplier,
            "turnover": record.amount,
            "source": price_source,
            "volume_source": (
                record.volume_source
                or (price_source if record.volume is not None else None)
            ),
        }

    @staticmethod
    def _normalize_ndx_result(
        instrument_code: str,
        result: ProviderResult,
    ) -> ProviderResult:
        if instrument_code != "NDX":
            return result
        return result.model_copy(
            update={
                "records": [
                    record.model_copy(
                        update={
                            "volume": None,
                            "volume_multiplier": 1,
                            "volume_source": "VOLUME_UNAVAILABLE:DIRECT_INDEX",
                        }
                    )
                    for record in result.records
                ],
                "volume_availability": "not_available_for_direct_index",
            }
        )

    @staticmethod
    def _clear_ndx_volume(session: Session, instrument_id: int) -> None:
        rows = session.scalars(
            select(MarketPrice).where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.volume.is_not(None),
            )
        ).all()
        for row in rows:
            row.volume = None
            row.volume_multiplier = 1
            row.volume_source = "VOLUME_UNAVAILABLE:DIRECT_INDEX"

    @staticmethod
    def _upsert_price(
        session: Session,
        instrument_id: int,
        record: MarketDataRecord,
        timeframe: str,
        default_source: str,
    ) -> str:
        values = MarketDataService._price_values(record, default_source)
        identity = {
            "instrument_id": instrument_id,
            "trade_date": record.trade_date,
            "timeframe": timeframe,
        }
        insert = sqlite_insert(MarketPrice).values(**identity, **values)
        inserted = session.execute(
            insert.on_conflict_do_nothing(index_elements=("instrument_id", "trade_date", "timeframe"))
        )
        if inserted.rowcount:
            return "added"
        price = session.scalar(
            select(MarketPrice).where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.trade_date == record.trade_date,
                MarketPrice.timeframe == timeframe,
            )
        )
        if price is None:
            # SQLite's conflict handling is atomic. This guard is defensive for
            # a connection interrupted between DO NOTHING and its readback.
            return "skipped"
        if all(getattr(price, field) == value for field, value in values.items()):
            return "skipped"
        session.execute(
            insert.on_conflict_do_update(
                index_elements=("instrument_id", "trade_date", "timeframe"),
                set_=values,
            )
        )
        return "updated"

    @staticmethod
    def _batch_upsert_daily(
        session: Session,
        instrument_id: int,
        records: Sequence[MarketDataRecord],
        default_source: str,
    ) -> dict[str, int]:
        """Classify one daily batch with one preload and batched DML.

        The initial conflict-safe insert runs before the preload so concurrent
        same-date writers serialize without a read-then-write lock upgrade.
        Rows that already exist are then read in one query and changed through
        one executemany UPDATE.
        """
        if not records:
            return defaultdict(int)
        value_fields = tuple(
            MarketDataService._price_values(records[0], default_source)
        )
        final_values: dict[date, dict[str, object]] = {}
        ordered_values: list[tuple[date, dict[str, object]]] = []
        for record in records:
            values = MarketDataService._price_values(record, default_source)
            ordered_values.append((record.trade_date, values))
            final_values[record.trade_date] = values

        insert_payloads = [
            {
                "instrument_id": instrument_id,
                "trade_date": trade_date,
                "timeframe": "daily",
                **values,
            }
            for trade_date, values in final_values.items()
        ]
        # Keep every statement below SQLite's conservative 999-variable
        # ceiling. One MarketPrice INSERT binds 15 values, so 50 rows remain
        # portable when a provider returns complete multi-decade history.
        insert_batch_size = 50
        insert = sqlite_insert(MarketPrice)
        inserted_dates: set[date] = set()
        for offset in range(0, len(insert_payloads), insert_batch_size):
            payload_batch = insert_payloads[
                offset : offset + insert_batch_size
            ]
            inserted_dates.update(
                session.scalars(
                    insert.values(payload_batch)
                    .on_conflict_do_nothing(
                        index_elements=(
                            "instrument_id",
                            "trade_date",
                            "timeframe",
                        )
                    )
                    .returning(MarketPrice.trade_date)
                ).all()
            )

        stored_rows: list[MarketPrice] = []
        stored_dates = tuple(final_values)
        select_batch_size = 500
        for offset in range(0, len(stored_dates), select_batch_size):
            date_batch = stored_dates[offset : offset + select_batch_size]
            stored_rows.extend(
                session.scalars(
                    select(MarketPrice).where(
                        MarketPrice.instrument_id == instrument_id,
                        MarketPrice.timeframe == "daily",
                        MarketPrice.trade_date.in_(date_batch),
                    )
                ).all()
            )
        stored_values = {
            row.trade_date: {
                field: getattr(row, field)
                for field in value_fields
            }
            for row in stored_rows
        }
        state = {
            trade_date: values
            for trade_date, values in stored_values.items()
            if trade_date not in inserted_dates
        }
        outcomes: dict[str, int] = defaultdict(int)
        for trade_date, values in ordered_values:
            current = state.get(trade_date)
            if current is None:
                outcomes["added"] += 1
            elif current == values:
                outcomes["skipped"] += 1
            else:
                outcomes["updated"] += 1
            state[trade_date] = values

        update_payloads = [
            {
                "_instrument_id": instrument_id,
                "_trade_date": trade_date,
                "_timeframe": "daily",
                **values,
            }
            for trade_date, values in final_values.items()
            if trade_date not in inserted_dates
            and stored_values.get(trade_date) != values
        ]
        if update_payloads:
            update = (
                MarketPrice.__table__.update()
                .where(
                    MarketPrice.instrument_id == bindparam("_instrument_id"),
                    MarketPrice.trade_date == bindparam("_trade_date"),
                    MarketPrice.timeframe == bindparam("_timeframe"),
                )
                .values(
                    {
                        field: bindparam(field)
                        for field in value_fields
                    }
                )
            )
            session.execute(update, update_payloads)
        return outcomes

    def _add_log(
        self,
        session: Session,
        *,
        instrument_id: int | None,
        source: str,
        status: str,
        records_received: int,
        records_written: int,
        records_added: int = 0,
        records_updated: int = 0,
        records_skipped: int = 0,
        started_at: datetime | None = None,
        message: str | None = None,
        timeframe: str = "daily",
    ) -> None:
        completed_at = _now()
        session.add(
            DataUpdateLog(
                instrument_id=instrument_id,
                dataset="market_prices",
                source=source,
                timeframe=timeframe,
                status=status,
                started_at=started_at or completed_at,
                completed_at=completed_at,
                records_received=records_received,
                records_written=records_written,
                records_added=records_added,
                records_updated=records_updated,
                records_skipped=records_skipped,
                message=message,
            )
        )

    def _record_log(
        self,
        instrument_id: int | None,
        *,
        source: str,
        status: str,
        records_received: int,
        records_written: int,
        records_added: int = 0,
        records_updated: int = 0,
        records_skipped: int = 0,
        started_at: datetime | None = None,
        message: str | None = None,
        timeframe: str = "daily",
    ) -> None:
        """Persist a standalone audit entry when no success transaction exists."""
        with self.session_factory() as session, session.begin():
            self._add_log(
                session,
                instrument_id=instrument_id,
                source=source,
                status=status,
                records_received=records_received,
                records_written=records_written,
                records_added=records_added,
                records_updated=records_updated,
                records_skipped=records_skipped,
                started_at=started_at,
                message=message,
                timeframe=timeframe,
            )

    def _record_log_best_effort(self, *args, **kwargs) -> None:
        try:
            self._record_log(*args, **kwargs)
        except Exception as error:
            logger.warning(
                "Could not persist market-data audit log: %s",
                error,
                exc_info=True,
            )

    @staticmethod
    def _as_record(price: MarketPrice) -> MarketDataRecord:
        # Legacy database rows can predate the normalized OHLC contract.
        # Preserve their usable close rather than letting a cache fallback fail.
        open_price = price.open_price or price.close_price
        high_price = price.high_price or max(open_price, price.close_price)
        low_price = price.low_price or min(open_price, price.close_price)
        return MarketDataRecord(
            trade_date=price.trade_date,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=price.close_price,
            adjusted_close_price=price.adjusted_close_price,
            volume=(price.volume * Decimal(price.volume_multiplier) if price.volume is not None else None),
            volume_multiplier=1,
            amount=price.turnover,
            source=price.source,
            volume_source=price.volume_source,
        )

    def store_result(
        self,
        instrument_code: str,
        result: ProviderResult,
        *,
        requested_source: str | None = None,
        started_at: datetime | None = None,
    ) -> DataUpdateResponse:
        """Incrementally upsert a successful result without deleting real price rows.

        A successful direct-index response is the one deliberate exception for
        explicit DEMO_INDEX placeholders: retaining fabricated holiday rows
        beside actual history would contaminate technical indicators and the
        historical-similarity sample.  Only those labelled demo rows are
        removed, and the update log records the replacement.
        """
        started_at = started_at or _now()
        requested_source = requested_source or result.source
        result = self._normalize_ndx_result(instrument_code, result)
        if not self._known_instrument(instrument_code):
            return self._unknown_instrument_response(instrument_code, result)
        if result.error or not result.records:
            return self._handle_failure(
                instrument_code, result, requested_source=requested_source, started_at=started_at
            )
        with self.session_factory() as session, session.begin():
            instrument = self._instrument(session, instrument_code)
            if instrument_code == "NDX":
                self._clear_ndx_volume(session, instrument.id)
            outcomes = defaultdict(int)
            replacing_direct_demo = (
                instrument_code in DIRECT_INDEX_DEMO_BASE_PRICES
                and any(not (record.source or result.source).startswith("DEMO") for record in result.records)
            )
            removed_demo_rows = 0
            if replacing_direct_demo:
                demo_rows = session.scalars(
                    select(MarketPrice).where(
                        MarketPrice.instrument_id == instrument.id,
                        MarketPrice.timeframe == "daily",
                        MarketPrice.source.like("DEMO%"),
                    )
                ).all()
                removed_demo_rows = len(demo_rows)
                for row in demo_rows:
                    session.delete(row)
                session.flush()
            outcomes = self._batch_upsert_daily(
                session,
                instrument.id,
                result.records,
                result.source,
            )
            instrument_id = instrument.id
            self._add_log(
                session,
                instrument_id=instrument_id,
                source=requested_source,
                status="success",
                records_received=len(result.records),
                records_written=outcomes["added"] + outcomes["updated"],
                records_added=outcomes["added"],
                records_updated=outcomes["updated"],
                records_skipped=outcomes["skipped"],
                started_at=started_at,
                timeframe="daily",
                message=(
                    f"Replaced {removed_demo_rows} explicit DEMO_INDEX rows with direct-index source {result.source}."
                    if removed_demo_rows
                    else None
                ),
            )
        response_values = result.model_dump()
        response_values["status"] = ProviderStatus.SUCCESS
        response_values["records_received"] = len(result.records)
        response_values["records_written"] = outcomes["added"] + outcomes["updated"]
        response_values["records_added"] = outcomes["added"]
        response_values["records_updated"] = outcomes["updated"]
        response_values["records_skipped"] = outcomes["skipped"]
        return DataUpdateResponse(**response_values)

    def update_from_provider(
        self,
        instrument_code: str,
        provider: MarketDataProvider,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataUpdateResponse:
        """Fetch one provider safely; an outage returns cached or labelled demo data."""
        started_at = _now()
        requested_source = getattr(provider, "source", "UNKNOWN")
        try:
            result = provider.fetch(instrument_code, start_date, end_date)
        except Exception as error:
            result = ProviderResult.failed(requested_source, f"Provider fetch failed: {error}")
        return self.store_result(
            instrument_code, result, requested_source=requested_source, started_at=started_at
        )

    def update_from_providers(
        self,
        instrument_code: str,
        providers: Sequence[MarketDataProvider],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> DataUpdateResponse:
        """Try configured providers in order, then use local cache/demo only if all fail."""
        errors: list[str] = []
        last_source = "MULTI_PROVIDER"
        last_started_at = _now()
        for provider in providers:
            provider_started_at = _now()
            requested_source = getattr(provider, "source", "UNKNOWN")
            try:
                result = provider.fetch(instrument_code, start_date, end_date)
            except Exception as error:
                result = ProviderResult.failed(requested_source, str(error))
            if result.records and not result.error:
                return self.store_result(
                    instrument_code,
                    result,
                    requested_source=requested_source,
                    started_at=provider_started_at,
                )
            errors.append(result.error or f"{result.source} returned no records")
            last_source = requested_source
            last_started_at = provider_started_at
        return self.store_result(
            instrument_code,
            ProviderResult.failed(last_source, "; ".join(errors) or "No providers configured"),
            requested_source=last_source,
            started_at=last_started_at,
        )

    update_instrument = update_from_provider

    def _cached_records(self, session: Session, instrument_id: int) -> list[MarketDataRecord]:
        prices = session.scalars(
            select(MarketPrice)
            .where(MarketPrice.instrument_id == instrument_id, MarketPrice.timeframe == "daily")
            .order_by(MarketPrice.trade_date)
        ).all()
        return [self._as_record(price) for price in prices]

    def _handle_failure(
        self,
        instrument_code: str,
        result: ProviderResult,
        *,
        requested_source: str,
        started_at: datetime,
    ) -> DataUpdateResponse:
        error = result.error or "Provider returned no records"
        with self.session_factory() as session, session.begin():
            instrument = self._instrument(session, instrument_code)
            if instrument_code == "NDX":
                self._clear_ndx_volume(session, instrument.id)
            cached = self._cached_records(session, instrument.id)
            instrument_id = instrument.id
        if cached:
            is_demo_cache = all((record.source or "").startswith("DEMO") for record in cached)
            cached_sources = sorted({record.source for record in cached if record.source})
            self._record_log_best_effort(
                instrument_id,
                source=requested_source,
                status="error",
                records_received=0,
                records_written=0,
                started_at=started_at,
                message=error,
            )
            return DataUpdateResponse(
                source="DEMO" if is_demo_cache else (cached_sources[0] if len(cached_sources) == 1 else "CACHE"),
                fetched_at=result.fetched_at,
                cutoff_date=cached[-1].trade_date,
                records=cached,
                cache_used=True,
                error=error,
                demo=is_demo_cache,
                status=ProviderStatus.DEMO if is_demo_cache else ProviderStatus.CACHED,
                cached_records=len(cached),
                volume_availability=(
                    "not_available_for_direct_index"
                    if instrument_code == "NDX"
                    else "available"
                ),
            )

        # Direct index research must never fall through to the legacy ETF demo
        # generator.  Each requested index receives its own clearly labelled
        # curve, even when no other local market cache exists yet.
        try:
            if instrument_code in DIRECT_INDEX_DEMO_BASE_PRICES:
                demo_created = self._create_demo_index_data_if_instrument_empty(
                    instrument_code
                )
            else:
                demo_created = self._create_demo_data_if_price_cache_is_empty()
        except Exception as demo_error:
            logger.warning(
                "Could not persist DEMO market data and audit log: %s",
                demo_error,
                exc_info=True,
            )
            demo_created = False
        self._record_log_best_effort(
            instrument_id,
            source=requested_source,
            status="error",
            records_received=0,
            records_written=0,
            started_at=started_at,
            message=error,
        )
        if not demo_created:
            return DataUpdateResponse(
                source=result.source,
                fetched_at=result.fetched_at,
                records=[],
                cache_used=False,
                demo=False,
                error=error,
                status=ProviderStatus.ERROR,
                volume_availability=(
                    "not_available_for_direct_index"
                    if instrument_code == "NDX"
                    else "available"
                ),
            )
        with self.session_factory() as session:
            instrument = self._instrument(session, instrument_code)
            demo_records = self._cached_records(session, instrument.id)
        return DataUpdateResponse(
            source="DEMO",
            cutoff_date=demo_records[-1].trade_date if demo_records else None,
            records=demo_records,
            demo=True,
            error=error,
            status=ProviderStatus.DEMO,
            records_received=len(demo_records),
            records_written=len(demo_records),
            volume_availability=(
                "not_available_for_direct_index"
                if instrument_code == "NDX"
                else "available"
            ),
        )

    def _create_demo_data_if_price_cache_is_empty(self) -> bool:
        """Generate fixed, clearly DEMO-labelled rows only in an entirely empty price store."""
        with self.session_factory() as session, session.begin():
            if session.scalar(select(func.count()).select_from(MarketPrice)):
                return False
            instruments = session.scalars(select(Instrument).where(Instrument.code.in_(DEMO_BASE_PRICES))).all()
            for instrument in instruments:
                base = DEMO_BASE_PRICES[instrument.code]
                for index, trade_day in enumerate(DEMO_DATES):
                    close = base + Decimal(index) * Decimal("0.005")
                    self._upsert_price(
                        session,
                        instrument.id,
                        MarketDataRecord(
                            trade_date=trade_day,
                            open_price=close - Decimal("0.003"),
                            high_price=close + Decimal("0.006"),
                            low_price=close - Decimal("0.007"),
                            close_price=close,
                            volume=Decimal("100000") + Decimal(index * 1000),
                            amount=(Decimal("100000") + Decimal(index * 1000)) * close,
                            source="DEMO",
                        ),
                        "daily",
                        "DEMO",
                    )
            records_written = len(instruments) * len(DEMO_DATES)
            self._add_log(
                session,
                instrument_id=None,
                source="DEMO",
                status="demo",
                records_received=0,
                records_written=records_written,
                records_added=records_written,
                message=(
                    "Created deterministic DEMO market data because "
                    "no local price cache existed."
                ),
            )
        return True

    def _create_demo_index_data_if_instrument_empty(self, instrument_code: str) -> bool:
        """Keep index curves usable offline without ever disguising demo data.

        Existing ETF cache must not prevent a new direct-index research series
        from receiving its own labelled deterministic fallback.
        """
        with self.session_factory() as session, session.begin():
            instrument = self._instrument(session, instrument_code)
            existing = session.scalar(
                select(func.count()).select_from(MarketPrice).where(
                    MarketPrice.instrument_id == instrument.id,
                    MarketPrice.timeframe == "daily",
                )
            )
            if existing:
                return False
            base = DIRECT_INDEX_DEMO_BASE_PRICES[instrument_code]
            trade_day = date(2025, 10, 1)
            written = 0
            price_quantum = Decimal("0.00000001")
            for sequence in range(160):
                while trade_day.weekday() >= 5:
                    trade_day = trade_day.fromordinal(trade_day.toordinal() + 1)
                drift = Decimal(sequence) * Decimal("0.0012")
                cycle = Decimal((sequence * 7) % 13 - 6) * Decimal("0.0009")
                close = (base * (Decimal("1") + drift + cycle)).quantize(price_quantum)
                open_price = (close * Decimal("0.998")).quantize(price_quantum)
                high_price = (close * Decimal("1.006")).quantize(price_quantum)
                low_price = (close * Decimal("0.994")).quantize(price_quantum)
                self._upsert_price(
                    session,
                    instrument.id,
                    MarketDataRecord(
                        trade_date=trade_day,
                        open_price=open_price,
                        high_price=high_price,
                        low_price=low_price,
                        close_price=close,
                        volume=(
                            None
                            if instrument_code == "NDX"
                            else Decimal("1000000") + Decimal(sequence * 7500)
                        ),
                        # Direct-index turnover is intentionally unavailable:
                        # its source units vary and monthly totals can exceed
                        # the SQLite fixed-point storage range.
                        amount=None,
                        source="DEMO_INDEX",
                    ),
                    "daily",
                    "DEMO_INDEX",
                )
                written += 1
                trade_day = trade_day.fromordinal(trade_day.toordinal() + 1)
            self._add_log(
                session,
                instrument_id=instrument.id,
                source="DEMO_INDEX",
                status="demo",
                records_received=0,
                records_written=written,
                records_added=written,
                message=(
                    "Created deterministic direct-index DEMO data because "
                    "no local index cache existed."
                ),
            )
        return True

    def aggregate_timeframes(
        self,
        instrument_code: str,
        *,
        expected_trade_dates: Iterable[date] | None,
        as_of: date | None,
        weekly_result: ProviderResult | None = None,
        changed_start_date: date | None = None,
    ) -> dict[str, object] | DataUpdateResponse:
        """Publish complete weekly/monthly rows using an explicit trading calendar."""
        if not self._known_instrument(instrument_code):
            return self._unknown_instrument_response(
                instrument_code, ProviderResult.failed("LOCAL", "Invalid aggregation instrument")
            )
        started_at = _now()
        rebuild_start = self._aggregation_rebuild_start(changed_start_date)
        with self.session_factory() as session:
            instrument = self._instrument(session, instrument_code)
            daily_query = select(MarketPrice).where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
            )
            if rebuild_start is not None:
                daily_query = daily_query.where(MarketPrice.trade_date >= rebuild_start)
            daily = session.scalars(
                daily_query.order_by(MarketPrice.trade_date)
            ).all()
            instrument_id = instrument.id
        daily_bars = tuple(self._as_daily_bar(price) for price in daily)
        if rebuild_start is not None and expected_trade_dates is not None:
            expected_trade_dates = tuple(
                trade_day
                for trade_day in expected_trade_dates
                if trade_day >= rebuild_start
            )
        if instrument_code in DIRECT_INDEX_DEMO_BASE_PRICES:
            # Direct-index turnover is vendor-specific and excluded from the
            # immutable read snapshot before either timeframe is evaluated.
            daily_bars = tuple(replace(bar, turnover=None) for bar in daily_bars)

        aggregations: dict[str, AggregationResult] = {}
        quality_errors: dict[str, str] = {}
        for timeframe in ("weekly", "monthly"):
            aggregation = aggregate_bars(
                daily_bars,
                timeframe,  # type: ignore[arg-type]
                instrument_code,
                expected_trade_dates=expected_trade_dates,
                as_of=as_of,
            )
            if (
                timeframe == "weekly"
                and weekly_result is not None
                and weekly_result.records
                and not weekly_result.error
            ):
                aggregation = self._apply_upstream_weekly_volume(
                    aggregation,
                    weekly_result,
                    instrument_code,
                )
            aggregations[timeframe] = aggregation
            try:
                self._require_publishable(aggregation)
            except ValueError as error:
                quality_errors[timeframe] = str(error)

        if len(quality_errors) == 2:
            raise ValueError("; ".join(quality_errors.values()))

        outcomes: dict[str, dict[str, int]] = {}
        timeframe_status: dict[str, dict[str, object]] = {}
        for timeframe in ("weekly", "monthly"):
            if timeframe in quality_errors:
                error = quality_errors[timeframe]
                counts = {"added": 0, "updated": 0, "skipped": 0}
                status = "blocked"
            else:
                error = None
                try:
                    with self.session_factory() as session, session.begin():
                        aggregate_options: dict[str, object] = {
                            "aggregation": aggregations[timeframe],
                        }
                        # Preserve compatibility with audit/test subclasses
                        # that override the historical full-aggregation
                        # signature.  The new keyword is needed only for a
                        # real incremental rebuild.
                        if rebuild_start is not None:
                            aggregate_options["replace_from"] = rebuild_start
                        counts = self._aggregate(
                            session,
                            instrument_id,
                            daily_bars,
                            timeframe,
                            instrument_code,
                            **aggregate_options,
                        )
                        records_written = counts["added"] + counts["updated"]
                        self._add_log(
                            session,
                            instrument_id=instrument_id,
                            source="AGGREGATION",
                            status="success",
                            records_received=len(daily),
                            records_written=records_written,
                            records_added=counts["added"],
                            records_updated=counts["updated"],
                            records_skipped=counts["skipped"],
                            started_at=started_at,
                            timeframe=timeframe,
                        )
                except Exception as persistence_error:
                    error = str(persistence_error)
                    counts = {"added": 0, "updated": 0, "skipped": 0}
                    status = "error"
                else:
                    status = "success"
            outcomes[timeframe] = counts
            records_written = counts["added"] + counts["updated"]
            timeframe_status[timeframe] = {
                "status": status,
                "records_written": records_written,
                "records_added": counts["added"],
                "records_updated": counts["updated"],
                "records_skipped": counts["skipped"],
                "error": error,
            }
            if status != "success":
                self._record_log_best_effort(
                    instrument_id,
                    source="AGGREGATION",
                    status=status,
                    records_received=len(daily),
                    records_written=records_written,
                    records_added=counts["added"],
                    records_updated=counts["updated"],
                    records_skipped=counts["skipped"],
                    started_at=started_at,
                    message=error,
                    timeframe=timeframe,
                )

        if (
            instrument_code in DIRECT_INDEX_DEMO_BASE_PRICES
            and any(
                details["status"] == "success"
                for details in timeframe_status.values()
            )
        ):
            # Legacy direct-index turnover cleanup is a third, isolated and
            # idempotent transaction; it cannot roll back either period commit.
            with self.session_factory() as session, session.begin():
                direct_daily = session.scalars(
                    select(MarketPrice).where(
                        MarketPrice.instrument_id == instrument_id,
                        MarketPrice.timeframe == "daily",
                        MarketPrice.turnover.is_not(None),
                    )
                ).all()
                for price in direct_daily:
                    price.turnover = None

        return {
            "weekly": (
                outcomes["weekly"]["added"] + outcomes["weekly"]["updated"]
            ),
            "monthly": (
                outcomes["monthly"]["added"] + outcomes["monthly"]["updated"]
            ),
            "timeframe_status": timeframe_status,
            "recalculated_from": rebuild_start,
        }

    def aggregate_periods(
        self,
        instrument_code: str,
        *,
        expected_trade_dates: Iterable[date] | None = None,
        as_of: date | None = None,
        weekly_result: ProviderResult | None = None,
        changed_start_date: date | None = None,
    ) -> dict[str, object] | DataUpdateResponse:
        """Aggregate with explicit context or derive it from the exchange calendar."""
        if expected_trade_dates is None:
            with self.session_factory() as session:
                instrument = session.scalar(
                    select(Instrument).where(Instrument.code == instrument_code)
                )
                daily_bounds = (
                    session.execute(
                        select(MarketPrice.trade_date)
                        .where(
                            MarketPrice.instrument_id == instrument.id,
                            MarketPrice.timeframe == "daily",
                        )
                        .order_by(MarketPrice.trade_date)
                    ).scalars().all()
                    if instrument is not None
                    else []
                )
            earliest_daily = daily_bounds[0] if daily_bounds else None
            latest_daily = daily_bounds[-1] if daily_bounds else None
            rebuild_start = self._aggregation_rebuild_start(changed_start_date)
            resolved_as_of = (
                min(as_of, latest_daily)
                if as_of is not None and latest_daily is not None
                else latest_daily
            )
            if earliest_daily is not None and resolved_as_of is not None:
                calendar_start = (
                    max(earliest_daily, rebuild_start)
                    if rebuild_start is not None
                    else earliest_daily
                )
                week_end = resolved_as_of + timedelta(
                    days=6 - resolved_as_of.weekday()
                )
                month_end = resolved_as_of.replace(
                    day=calendar.monthrange(
                        resolved_as_of.year,
                        resolved_as_of.month,
                    )[1]
                )
                calendar_dates = self.calendar_provider.sessions(
                    instrument_code,
                    calendar_start,
                    max(week_end, month_end),
                )
                # ETFs can legitimately suspend trading on a mainland session
                # (or a public source omit a zero-trade day).  Never block the
                # weekly/monthly chart or model pipeline on those sessions and
                # never synthesize an OHLC row or carry a price forward:
                # aggregate the real sessions that actually exist locally.
                expected_trade_dates = (
                    tuple(
                        trade_day
                        for trade_day in daily_bounds
                        if trade_day >= calendar_start
                    )
                    if instrument_code in DISPLAY_ONLY_CODES
                    or is_etf_code(instrument_code)
                    else calendar_dates
                )
            else:
                expected_trade_dates = ()
            as_of = resolved_as_of
        return self.aggregate_timeframes(
            instrument_code,
            expected_trade_dates=expected_trade_dates,
            as_of=as_of,
            weekly_result=weekly_result,
            changed_start_date=changed_start_date,
        )

    def _apply_upstream_weekly_volume(
        self,
        aggregation: AggregationResult,
        result: ProviderResult,
        instrument_code: str,
    ) -> AggregationResult:
        """Return an audited native-volume overlay or the untouched daily fallback."""
        upstream = tuple(
            UpstreamWeeklyVolume(
                trade_date=record.trade_date,
                volume=(
                    record.volume * Decimal(record.volume_multiplier)
                    if record.volume is not None
                    else None
                ),
                source=record.source or result.source,
                unit="shares",
            )
            for record in result.records
        )
        return apply_upstream_weekly_volume(
            aggregation,
            upstream,
            instrument_code=instrument_code,
            upstream_source=result.source,
        )

    def _aggregate(
        self,
        session: Session,
        instrument_id: int,
        daily: Iterable[DailyBar],
        timeframe: str,
        instrument_code: str,
        *,
        aggregation: AggregationResult | None = None,
        replace_from: date | None = None,
    ) -> dict[str, int]:
        if timeframe not in ("weekly", "monthly"):
            raise ValueError(f"Unsupported aggregation timeframe: {timeframe}")
        result = aggregation or aggregate_bars(
            tuple(daily),
            timeframe,
            instrument_code,
        )
        self._require_publishable(result)
        deferred_keys = {
            self._period_key(issue.start_date, timeframe)
            for issue in result.report.issues
            if issue.code == "INCOMPLETE_CURRENT_PERIOD"
            and issue.start_date is not None
        }
        targets: dict[tuple[int, int], MarketDataRecord] = {}
        for period in result.bars:
            period_volume, volume_multiplier = self._encode_volume(
                period.volume
            )
            assert period.open_price is not None
            assert period.high_price is not None
            assert period.low_price is not None
            assert period.close_price is not None
            targets[self._period_key(period.period_end, timeframe)] = MarketDataRecord(
                trade_date=period.period_end,
                open_price=period.open_price,
                high_price=period.high_price,
                low_price=period.low_price,
                close_price=period.close_price,
                adjusted_close_price=period.adjusted_close_price,
                volume=period_volume,
                volume_multiplier=volume_multiplier,
                amount=period.turnover,
                source=period.price_source,
                volume_source=period.volume_source,
            )
        existing_query = select(MarketPrice).where(
            MarketPrice.instrument_id == instrument_id,
            MarketPrice.timeframe == timeframe,
        )
        if replace_from is not None:
            existing_query = existing_query.where(
                MarketPrice.trade_date >= replace_from
            )
        existing = session.scalars(existing_query).all()
        by_period: dict[tuple[int, int], list[MarketPrice]] = defaultdict(list)
        for row in existing:
            by_period[self._period_key(row.trade_date, timeframe)].append(row)
        outcomes = defaultdict(int)
        for key, rows in by_period.items():
            target = targets.get(key)
            if target is None:
                if key in deferred_keys:
                    outcomes["skipped"] += len(rows)
                    continue
                for row in rows:
                    session.delete(row)
                    outcomes["updated"] += 1
                continue
            target_values = self._price_values(target, target.source or "AGGREGATION")
            matching = next(
                (
                    row
                    for row in rows
                    if row.trade_date == target.trade_date
                    and all(getattr(row, field) == value for field, value in target_values.items())
                ),
                None,
            )
            stale = [row for row in rows if row is not matching]
            for row in stale:
                session.delete(row)
            if stale and matching is not None:
                outcomes["updated"] += len(stale)
        session.flush()
        for key, target in targets.items():
            rows = by_period.get(key, [])
            target_values = self._price_values(target, target.source or "AGGREGATION")
            unchanged = any(
                row.trade_date == target.trade_date
                and all(getattr(row, field) == value for field, value in target_values.items())
                for row in rows
            )
            if unchanged:
                # A matching endpoint needs no DML. Any stale sibling has
                # already been removed and counted as an update above.
                outcomes["skipped"] += 1
                continue
            outcome = self._upsert_price(session, instrument_id, target, timeframe, target.source or "AGGREGATION")
            outcomes[outcome] += 1
        return outcomes

    @staticmethod
    def _as_daily_bar(price: MarketPrice) -> DailyBar:
        return DailyBar(
            trade_date=price.trade_date,
            open_price=price.open_price,
            high_price=price.high_price,
            low_price=price.low_price,
            close_price=price.close_price,
            adjusted_close_price=price.adjusted_close_price,
            volume=(
                price.volume * Decimal(price.volume_multiplier)
                if price.volume is not None
                else None
            ),
            turnover=price.turnover,
            source=price.source,
        )

    @staticmethod
    def _require_publishable(aggregation: AggregationResult) -> None:
        if aggregation.report.is_publishable:
            return
        blocking = [
            f"{issue.code}: {issue.message}"
            for issue in aggregation.report.issues
            if issue.severity == "blocking"
        ]
        raise ValueError("Aggregation quality gate blocked publication: " + "; ".join(blocking))

    @staticmethod
    def _period_key(trade_date: date, timeframe: str) -> tuple[int, int]:
        if timeframe == "weekly":
            iso_year, iso_week, _ = trade_date.isocalendar()
            return iso_year, iso_week
        return trade_date.year, trade_date.month

    @staticmethod
    def _sum_optional(values: Iterable[Decimal | None]) -> Decimal | None:
        present = [value for value in values if value is not None]
        return sum(present, Decimal("0")) if present else None

    @staticmethod
    def _encode_volume(volume: Decimal | None) -> tuple[Decimal | None, int]:
        """Fit a period's real volume into SQLite without truncating it.

        The persisted multiplier is part of the row contract.  API readers
        multiply it back before rendering, so charts and exported values keep
        the source's original unit.
        """
        if volume is None:
            return None, 1
        multiplier = 1
        stored = volume
        while abs(stored) > MAX_FIXED_POINT_VALUE:
            stored /= Decimal("10")
            multiplier *= 10
        return stored, multiplier
