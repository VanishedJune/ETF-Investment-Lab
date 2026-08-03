"""V3.3 point-in-time market data, provenance, and leakage-safe reads.

The V3.3 store is intentionally independent from the V3.2 model tables.  This
module may mirror accepted 159941 OHLCV into the shared ``market_prices`` table
for the existing chart API, but it never writes a ``v32_*`` row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from typing import Any, Callable, Iterable, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import select, tuple_
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..models.models import (
    Instrument,
    MarketPrice,
    V33MarketBar,
    V33PointInTimeObservation,
    V33QualityAssessment,
    V33RawIngestion,
    V33SourceRelease,
)
from ..schemas.market import MarketDataRecord, ProviderResult, ProviderStatus
from .providers import AkShareEtfResearchProvider, MarketDataProvider, TushareProvider


V33_MARKET_GATE_VERSION = "V3.3-MARKET-GATE-1"
V33_OBSERVATION_GATE_VERSION = "V3.3-PIT-GATE-1"
V33_PIT_IDENTITY_QUERY_CHUNK = 150
SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")

SOURCE_URLS = {
    "AKSHARE": "https://akshare.akfamily.xyz/data/fund/fund_public.html",
    "TUSHARE": "https://tushare.pro/document/2?doc_id=127",
    "AKSHARE_INDEX": "https://akshare.akfamily.xyz/data/index/index.html",
    "YAHOO_DIRECT_INDEX": "https://finance.yahoo.com/quote/%5ENDX/history/",
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        default=str,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _source_url(source: str) -> str | None:
    upper = source.upper()
    for prefix, url in SOURCE_URLS.items():
        if upper.startswith(prefix):
            return url
    return None


def _record_payload(record: MarketDataRecord) -> dict[str, object]:
    return {
        "trade_date": record.trade_date.isoformat(),
        "open_price": str(record.open_price),
        "high_price": str(record.high_price),
        "low_price": str(record.low_price),
        "close_price": str(record.close_price),
        "adjusted_close_price": (
            str(record.adjusted_close_price)
            if record.adjusted_close_price is not None
            else None
        ),
        "volume": str(record.volume) if record.volume is not None else None,
        "volume_multiplier": record.volume_multiplier,
        "amount": str(record.amount) if record.amount is not None else None,
        "source": record.source,
        "volume_source": record.volume_source,
    }


@dataclass(frozen=True)
class DataQualityGate:
    status: str
    accepted: bool
    records: tuple[MarketDataRecord, ...]
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class V33RefreshResult:
    instrument_code: str
    source: str
    status: str
    quality_status: str
    ingestion_id: str
    quality_assessment_id: str
    records_received: int
    records_written: int
    data_as_of: date | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkAlignment:
    target_trade_date: date
    benchmark_trade_date: date
    benchmark_close: Decimal
    benchmark_available_at: datetime
    source: str
    raw_payload_hash: str


@dataclass(frozen=True)
class PointInTimeObservationInput:
    dataset: str
    series_code: str
    observation_date: date
    effective_date: date
    published_at: datetime
    available_at: datetime
    cutoff_at: datetime
    vintage: str
    source: str
    numeric_value: Decimal | None = None
    text_value: str | None = None
    unit: str | None = None
    observation_period: str | None = None
    source_url: str | None = None
    instrument_code: str | None = None
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PreparedPointInTimeObservation:
    """Validated PIT input plus the immutable identities used for persistence."""

    observation: PointInTimeObservationInput
    published_at: datetime
    available_at: datetime
    cutoff_at: datetime
    retrieved_at: datetime
    raw_payload: dict[str, Any]
    raw_hash: str
    ingestion_id: str
    release_id: str
    quality_assessment_id: str
    source_url: str | None

    @property
    def revision_identity(self) -> tuple[str, date, datetime, str, str]:
        return (
            self.observation.series_code,
            self.observation.effective_date,
            self.available_at,
            self.observation.vintage,
            self.observation.source,
        )


class V33DataRepository:
    """Read/query API that enforces the V3.3 availability cutoff."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    def market_bars_as_of(
        self,
        instrument_code: str,
        cutoff_at: datetime,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        timeframe: str = "daily",
    ) -> list[V33MarketBar]:
        """Return the latest revision that was available at ``cutoff_at``.

        ``retrieved_at`` remains audit metadata.  Eligibility is based on the
        real-world publication/availability timestamp; later revisions receive
        a later ``available_at`` when first seen and therefore cannot leak into
        earlier cutoffs.
        """

        cutoff = _utc(cutoff_at)
        with self.session_factory() as session:
            instrument_id = session.scalar(
                select(Instrument.id).where(Instrument.code == instrument_code)
            )
            if instrument_id is None:
                raise ValueError(f"Unknown instrument code: {instrument_code}")
            statement = (
                select(V33MarketBar)
                .where(
                    V33MarketBar.instrument_id == instrument_id,
                    V33MarketBar.timeframe == timeframe,
                    V33MarketBar.available_at <= cutoff,
                    V33MarketBar.cutoff_at <= cutoff,
                    V33MarketBar.quality_status.in_(
                        ("accepted", "accepted_with_warnings")
                    ),
                )
                .order_by(
                    V33MarketBar.trade_date,
                    V33MarketBar.available_at,
                    V33MarketBar.retrieved_at,
                    V33MarketBar.id,
                )
            )
            if start_date is not None:
                statement = statement.where(V33MarketBar.trade_date >= start_date)
            if end_date is not None:
                statement = statement.where(V33MarketBar.trade_date <= end_date)
            rows = list(session.scalars(statement))

        latest: dict[date, V33MarketBar] = {}
        for row in rows:
            latest[row.trade_date] = row
        return [latest[key] for key in sorted(latest)]

    def observations_as_of(
        self,
        series_codes: Sequence[str],
        cutoff_at: datetime,
    ) -> list[V33PointInTimeObservation]:
        """Return latest known vintages; missing series stay absent, never 0/50."""

        if not series_codes:
            return []
        cutoff = _utc(cutoff_at)
        with self.session_factory() as session:
            rows = list(
                session.scalars(
                    select(V33PointInTimeObservation)
                    .where(
                        V33PointInTimeObservation.series_code.in_(tuple(series_codes)),
                        V33PointInTimeObservation.available_at <= cutoff,
                        V33PointInTimeObservation.cutoff_at <= cutoff,
                        V33PointInTimeObservation.quality_status.in_(
                            ("accepted", "accepted_with_warnings")
                        ),
                    )
                    .order_by(
                        V33PointInTimeObservation.series_code,
                        V33PointInTimeObservation.effective_date,
                        V33PointInTimeObservation.available_at,
                        V33PointInTimeObservation.retrieved_at,
                        V33PointInTimeObservation.id,
                    )
                )
            )
        latest: dict[tuple[str, date], V33PointInTimeObservation] = {}
        for row in rows:
            latest[(row.series_code, row.effective_date)] = row
        return [latest[key] for key in sorted(latest)]

    @staticmethod
    def _target_close_at(trade_date: date) -> datetime:
        return datetime.combine(trade_date, time(15, 5), tzinfo=SHANGHAI).astimezone(
            timezone.utc
        )

    @staticmethod
    def _legacy_ndx_available_at(trade_date: date) -> datetime:
        # 16:05 New York is deliberately conservative.  It also prevents the
        # same natural-date US close from leaking into the earlier SZSE close.
        return datetime.combine(trade_date, time(16, 5), tzinfo=NEW_YORK).astimezone(
            timezone.utc
        )

    def align_ndx_benchmark(
        self,
        target_trade_dates: Iterable[date],
        *,
        global_cutoff_at: datetime | None = None,
        allow_read_only_legacy: bool = True,
    ) -> list[BenchmarkAlignment]:
        """Align each 159941 close only to NDX data already available then."""

        target_dates = sorted(set(target_trade_dates))
        if not target_dates:
            return []
        global_cutoff = _utc(global_cutoff_at) if global_cutoff_at is not None else None
        maximum_cutoff = self._target_close_at(target_dates[-1])
        if global_cutoff is not None:
            maximum_cutoff = min(maximum_cutoff, global_cutoff)
        v33_rows = self.market_bars_as_of(
            "NDX",
            maximum_cutoff,
            end_date=target_dates[-1],
        )

        legacy_rows: list[MarketPrice] = []
        if allow_read_only_legacy:
            with self.session_factory() as session:
                legacy_rows = list(
                    session.scalars(
                        select(MarketPrice)
                        .join(Instrument)
                        .where(
                            Instrument.code == "NDX",
                            MarketPrice.timeframe == "daily",
                            MarketPrice.trade_date <= target_dates[-1],
                        )
                        .order_by(MarketPrice.trade_date)
                    )
                )

        aligned: list[BenchmarkAlignment] = []
        for target_date in target_dates:
            cutoff = self._target_close_at(target_date)
            if global_cutoff is not None:
                cutoff = min(cutoff, global_cutoff)
            eligible_v33 = [
                row
                for row in v33_rows
                if row.trade_date <= target_date and row.available_at <= cutoff
            ]
            if eligible_v33:
                row = eligible_v33[-1]
                aligned.append(
                    BenchmarkAlignment(
                        target_trade_date=target_date,
                        benchmark_trade_date=row.trade_date,
                        benchmark_close=row.close_price,
                        benchmark_available_at=row.available_at,
                        source=row.source,
                        raw_payload_hash=row.raw_payload_hash,
                    )
                )
                continue
            eligible_legacy = [
                row
                for row in legacy_rows
                if row.trade_date <= target_date
                and self._legacy_ndx_available_at(row.trade_date) <= cutoff
            ]
            if not eligible_legacy:
                continue
            row = eligible_legacy[-1]
            available_at = self._legacy_ndx_available_at(row.trade_date)
            legacy_payload = {
                "instrument": "NDX",
                "trade_date": row.trade_date.isoformat(),
                "close_price": str(row.close_price),
                "source": row.source,
                "availability_policy": "US_CLOSE_1605_AMERICA_NEW_YORK",
            }
            aligned.append(
                BenchmarkAlignment(
                    target_trade_date=target_date,
                    benchmark_trade_date=row.trade_date,
                    benchmark_close=row.close_price,
                    benchmark_available_at=available_at,
                    source="LEGACY_MARKET_PRICE_READ_ONLY",
                    raw_payload_hash=_sha256(legacy_payload),
                )
            )
        return aligned


class V33DataService:
    """Write-side API for verified 159941 and point-in-time observations."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory
        self.repository = V33DataRepository(session_factory)

    @staticmethod
    def _instrument(session: Session, instrument_code: str) -> Instrument:
        instrument = session.scalar(
            select(Instrument).where(Instrument.code == instrument_code)
        )
        if instrument is None:
            raise ValueError(f"Unknown instrument code: {instrument_code}")
        return instrument

    @staticmethod
    def _quality_gate(
        result: ProviderResult,
        *,
        instrument_code: str,
        start_date: date | None,
        end_date: date | None,
        require_volume: bool,
    ) -> DataQualityGate:
        issues: list[str] = []
        warnings: list[str] = list(result.warnings)
        if result.status is not ProviderStatus.SUCCESS:
            issues.append(f"PROVIDER_STATUS:{result.status.value}")
        if result.demo or "DEMO" in result.source.upper():
            issues.append("DEMO_DATA_FORBIDDEN")
        if not result.records:
            issues.append("NO_USABLE_RECORDS")

        by_date: dict[date, MarketDataRecord] = {}
        duplicate_payloads: dict[date, str] = {}
        for record in result.records:
            payload_hash = _sha256(_record_payload(record))
            prior_hash = duplicate_payloads.get(record.trade_date)
            if prior_hash is not None and prior_hash != payload_hash:
                issues.append(f"CONFLICTING_DUPLICATE:{record.trade_date.isoformat()}")
                continue
            if prior_hash is not None:
                warnings.append(f"IDENTICAL_DUPLICATE:{record.trade_date.isoformat()}")
            duplicate_payloads[record.trade_date] = payload_hash
            by_date[record.trade_date] = record

            if start_date is not None and record.trade_date < start_date:
                issues.append(f"DATE_BEFORE_REQUEST:{record.trade_date.isoformat()}")
            if end_date is not None and record.trade_date > end_date:
                issues.append(f"DATE_AFTER_REQUEST:{record.trade_date.isoformat()}")
            if record.trade_date > date.today():
                issues.append(f"FUTURE_TRADE_DATE:{record.trade_date.isoformat()}")
            if (
                record.trade_date == date.today()
                and _utc(result.fetched_at)
                < V33DataService._market_close_available_at(
                    instrument_code, record.trade_date
                )
            ):
                issues.append(f"INCOMPLETE_SESSION:{record.trade_date.isoformat()}")
            if require_volume and record.volume is None:
                issues.append(f"MISSING_VOLUME:{record.trade_date.isoformat()}")
            if require_volume and record.volume is not None and record.volume <= 0:
                issues.append(f"NONPOSITIVE_VOLUME:{record.trade_date.isoformat()}")

        records = tuple(by_date[key] for key in sorted(by_date))
        if records and end_date is not None and end_date <= date.today():
            lag = (end_date - records[-1].trade_date).days
            if lag > 14:
                issues.append(f"STALE_END_DATE:{lag}D")
            elif lag > 5:
                warnings.append(f"POSSIBLE_HOLIDAY_OR_STALE_END_DATE:{lag}D")
        if instrument_code == "159941" and not require_volume:
            issues.append("159941_VOLUME_GATE_MUST_BE_ENABLED")

        unique_issues = tuple(dict.fromkeys(issues))
        unique_warnings = tuple(dict.fromkeys(warnings))
        accepted = not unique_issues
        status = (
            "accepted_with_warnings"
            if accepted and unique_warnings
            else "accepted"
            if accepted
            else "rejected"
        )
        return DataQualityGate(
            status=status,
            accepted=accepted,
            records=records,
            issues=unique_issues,
            warnings=unique_warnings,
        )

    @staticmethod
    def _market_close_available_at(instrument_code: str, trade_date: date) -> datetime:
        if instrument_code == "NDX":
            return datetime.combine(
                trade_date, time(16, 5), tzinfo=NEW_YORK
            ).astimezone(timezone.utc)
        return datetime.combine(
            trade_date, time(15, 5), tzinfo=SHANGHAI
        ).astimezone(timezone.utc)

    @staticmethod
    def _mirror_market_price(
        session: Session,
        instrument_id: int,
        record: MarketDataRecord,
        source: str,
    ) -> None:
        values = {
            "open_price": record.open_price,
            "high_price": record.high_price,
            "low_price": record.low_price,
            "close_price": record.close_price,
            "adjusted_close_price": record.adjusted_close_price,
            "volume": record.volume,
            "volume_multiplier": record.volume_multiplier,
            "turnover": record.amount,
            "source": f"V33:{source}",
            "volume_source": (
                f"V33:{record.volume_source or source}"
                if record.volume is not None
                else None
            ),
        }
        insert = sqlite_insert(MarketPrice).values(
            instrument_id=instrument_id,
            trade_date=record.trade_date,
            timeframe="daily",
            **values,
        )
        session.execute(
            insert.on_conflict_do_update(
                index_elements=("instrument_id", "trade_date", "timeframe"),
                set_=values,
            )
        )

    def _persist_attempt(
        self,
        instrument_code: str,
        result: ProviderResult,
        gate: DataQualityGate,
        *,
        start_date: date | None,
        end_date: date | None,
    ) -> V33RefreshResult:
        fetched_at = _utc(result.fetched_at)
        records_payload = [_record_payload(record) for record in gate.records]
        raw_payload = {
            "instrument_code": instrument_code,
            "provider_status": result.status.value,
            "provider_error": result.error,
            "provider_warnings": result.warnings,
            "records": records_payload,
        }
        batch_hash = _sha256(raw_payload)
        ingestion_id = f"ing-{uuid4().hex}"
        assessment_id = f"qa-{uuid4().hex}"
        source_url = _source_url(result.source)
        vintage = f"{result.source}:{fetched_at.isoformat()}"
        records_written = 0

        with self.session_factory() as session, session.begin():
            instrument = self._instrument(session, instrument_code)
            ingestion = V33RawIngestion(
                id=ingestion_id,
                instrument_id=instrument.id,
                dataset="MARKET_OHLCV",
                provider=result.source,
                source_url=source_url,
                requested_start_date=start_date,
                requested_end_date=end_date,
                fetched_at=fetched_at,
                available_at=fetched_at,
                vintage=vintage,
                raw_payload_hash=batch_hash,
                raw_payload_json=raw_payload,
                request_json={
                    "instrument_code": instrument_code,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "timeframe": "daily",
                },
                status="accepted" if gate.accepted else "rejected",
                quality_status=gate.status,
                records_received=len(result.records),
                records_written=0,
                error_message=("; ".join(gate.issues) or result.error),
                created_at=fetched_at,
            )
            session.add(ingestion)
            # These models intentionally have no ORM relationships: the store
            # is append-only and callers use explicit repository methods.
            # Flush the parent before rows that reference its audit identity.
            session.flush()
            session.add(
                V33QualityAssessment(
                    id=assessment_id,
                    ingestion_id=ingestion_id,
                    instrument_id=instrument.id,
                    dataset="MARKET_OHLCV",
                    gate_version=V33_MARKET_GATE_VERSION,
                    assessed_at=fetched_at,
                    quality_status=gate.status,
                    coverage_start_date=(gate.records[0].trade_date if gate.records else None),
                    coverage_end_date=(gate.records[-1].trade_date if gate.records else None),
                    records_received=len(result.records),
                    records_valid=len(gate.records) if gate.accepted else 0,
                    records_rejected=(0 if gate.accepted else len(result.records)),
                    required_fields_json=[
                        *(
                            "trade_date",
                            "open_price",
                            "high_price",
                            "low_price",
                            "close_price",
                        ),
                        *(("volume",) if instrument_code == "159941" else ()),
                    ],
                    missing_fields_json=[
                        issue
                        for issue in gate.issues
                        if issue.startswith("MISSING_")
                    ],
                    issues_json=list(gate.issues),
                    warnings_json=list(gate.warnings),
                    assessed_payload_hash=batch_hash,
                    details_json={
                        "provider": result.source,
                        "demo": result.demo,
                        "volume_required": instrument_code == "159941",
                    },
                    created_at=fetched_at,
                )
            )

            if gate.accepted:
                existing_rows = list(
                    session.scalars(
                        select(V33MarketBar).where(
                            V33MarketBar.instrument_id == instrument.id,
                            V33MarketBar.timeframe == "daily",
                            V33MarketBar.trade_date.in_(
                                tuple(record.trade_date for record in gate.records)
                            ),
                        )
                    )
                )
                existing_by_date: dict[date, list[V33MarketBar]] = {}
                for row in existing_rows:
                    existing_by_date.setdefault(row.trade_date, []).append(row)

                for record in gate.records:
                    record_payload = _record_payload(record)
                    record_hash = _sha256(record_payload)
                    prior_rows = existing_by_date.get(record.trade_date, [])
                    if any(row.raw_payload_hash == record_hash for row in prior_rows):
                        self._mirror_market_price(
                            session, instrument.id, record, result.source
                        )
                        continue

                    original_release_at = self._market_close_available_at(
                        instrument_code, record.trade_date
                    )
                    is_revision = bool(prior_rows)
                    available_at = fetched_at if is_revision else original_release_at
                    availability_policy = (
                        "FIRST_SEEN_REVISION_AVAILABLE_AT_RETRIEVAL"
                        if is_revision
                        else "UNADJUSTED_OFFICIAL_BAR_AVAILABLE_AFTER_CLOSE"
                    )
                    release_vintage = (
                        f"revision:{fetched_at.isoformat()}"
                        if is_revision
                        else f"original:{record.trade_date.isoformat()}"
                    )
                    release_identity = {
                        "dataset": "MARKET_OHLCV",
                        "series_code": instrument_code,
                        "effective_date": record.trade_date.isoformat(),
                        "source": result.source,
                        "raw_payload_hash": record_hash,
                    }
                    release_id = f"rel-{_sha256(release_identity)[:60]}"
                    release_insert = sqlite_insert(V33SourceRelease).values(
                        id=release_id,
                        ingestion_id=ingestion_id,
                        instrument_id=instrument.id,
                        dataset="MARKET_OHLCV",
                        series_code=instrument_code,
                        observation_period_start=record.trade_date,
                        observation_period_end=record.trade_date,
                        effective_date=record.trade_date,
                        published_at=available_at,
                        available_at=available_at,
                        vintage=release_vintage,
                        source=result.source,
                        source_url=source_url,
                        raw_payload_hash=record_hash,
                        quality_status=gate.status,
                        payload_json={
                            "availability_policy": availability_policy,
                            "retrieved_at": fetched_at.isoformat(),
                        },
                        created_at=fetched_at,
                    )
                    session.execute(release_insert.on_conflict_do_nothing())
                    bar_insert = sqlite_insert(V33MarketBar).values(
                        instrument_id=instrument.id,
                        ingestion_id=ingestion_id,
                        source_release_id=release_id,
                        trade_date=record.trade_date,
                        timeframe="daily",
                        open_price=record.open_price,
                        high_price=record.high_price,
                        low_price=record.low_price,
                        close_price=record.close_price,
                        adjusted_close_price=record.adjusted_close_price,
                        volume=record.volume,
                        volume_multiplier=record.volume_multiplier,
                        turnover=record.amount,
                        effective_date=record.trade_date,
                        published_at=available_at,
                        available_at=available_at,
                        retrieved_at=fetched_at,
                        cutoff_at=available_at,
                        vintage=release_vintage,
                        source=result.source,
                        source_url=source_url,
                        raw_payload_hash=record_hash,
                        quality_status=gate.status,
                        payload_json={
                            "availability_policy": availability_policy,
                            "provider_record_source": record.source,
                        },
                        created_at=fetched_at,
                    )
                    outcome = session.execute(
                        bar_insert.on_conflict_do_nothing(
                            index_elements=(
                                "instrument_id",
                                "trade_date",
                                "timeframe",
                                "source",
                                "raw_payload_hash",
                            )
                        )
                    )
                    records_written += int(outcome.rowcount or 0)
                    self._mirror_market_price(session, instrument.id, record, result.source)
                ingestion.records_written = records_written

        return V33RefreshResult(
            instrument_code=instrument_code,
            source=result.source,
            status="success" if gate.accepted else "rejected",
            quality_status=gate.status,
            ingestion_id=ingestion_id,
            quality_assessment_id=assessment_id,
            records_received=len(result.records),
            records_written=records_written,
            data_as_of=gate.records[-1].trade_date if gate.records else None,
            issues=gate.issues,
            warnings=gate.warnings,
        )

    def persist_market_result(
        self,
        instrument_code: str,
        result: ProviderResult,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        require_volume: bool | None = None,
    ) -> V33RefreshResult:
        """Validate and persist an injected result (also used by deterministic tests)."""

        volume_required = instrument_code == "159941" if require_volume is None else require_volume
        gate = self._quality_gate(
            result,
            instrument_code=instrument_code,
            start_date=start_date,
            end_date=end_date,
            require_volume=volume_required,
        )
        return self._persist_attempt(
            instrument_code,
            result,
            gate,
            start_date=start_date,
            end_date=end_date,
        )

    def ingest_market_bars(
        self,
        instrument_code: str,
        records: Sequence[MarketDataRecord],
        *,
        source: str,
        fetched_at: datetime | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        require_volume: bool | None = None,
        warnings: Sequence[str] = (),
    ) -> V33RefreshResult:
        """Bulk ingestion API for verified connectors and offline importers."""

        provider_result = ProviderResult.success(
            source,
            list(records),
            warnings=list(warnings),
        )
        if fetched_at is not None:
            provider_result = provider_result.model_copy(
                update={"fetched_at": _utc(fetched_at)}
            )
        return self.persist_market_result(
            instrument_code,
            provider_result,
            start_date=start_date,
            end_date=end_date,
            require_volume=require_volume,
        )

    def refresh_159941(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
        providers: Sequence[MarketDataProvider] | None = None,
    ) -> V33RefreshResult:
        """Try AkShare then Tushare; only a volume-complete real batch can win."""

        chain: Sequence[MarketDataProvider] = providers or (
            AkShareEtfResearchProvider(),
            TushareProvider(),
        )
        last_result: V33RefreshResult | None = None
        for provider in chain:
            try:
                provider_result = provider.fetch("159941", start_date, end_date)
            except Exception as error:  # protocol adapters should not throw, but audit it if one does
                provider_result = ProviderResult.failed(
                    getattr(provider, "source", type(provider).__name__),
                    f"Provider raised unexpectedly: {error}",
                )
            last_result = self.persist_market_result(
                "159941",
                provider_result,
                start_date=start_date,
                end_date=end_date,
                require_volume=True,
            )
            if last_result.status == "success":
                return last_result
        if last_result is None:
            raise ValueError("At least one 159941 provider is required")
        return last_result

    def store_point_in_time_observation(
        self,
        observation: PointInTimeObservationInput,
    ) -> int:
        """Persist one explicitly versioned value without any neutral default.

        The single-item API deliberately shares the same atomic implementation
        as connector batches so its validation, hashes, provenance, idempotency,
        and conflict behaviour remain identical.
        """

        return self.ingest_point_in_time_observations((observation,))[0]

    @staticmethod
    def _prepare_point_in_time_observation(
        observation: PointInTimeObservationInput,
    ) -> _PreparedPointInTimeObservation:
        if (
            observation.numeric_value is None
            and observation.text_value is None
            and observation.payload is None
        ):
            raise ValueError("Point-in-time observation must contain a real value")
        published_at = _utc(observation.published_at)
        available_at = _utc(observation.available_at)
        cutoff_at = _utc(observation.cutoff_at)
        if available_at < published_at:
            raise ValueError("available_at cannot precede published_at")
        if cutoff_at < available_at:
            raise ValueError("cutoff_at cannot precede available_at")
        retrieved_at = datetime.now(timezone.utc)
        raw_payload = {
            "dataset": observation.dataset,
            "series_code": observation.series_code,
            "observation_date": observation.observation_date.isoformat(),
            "effective_date": observation.effective_date.isoformat(),
            "published_at": published_at.isoformat(),
            "available_at": available_at.isoformat(),
            "cutoff_at": cutoff_at.isoformat(),
            "vintage": observation.vintage,
            "source": observation.source,
            "numeric_value": (
                str(observation.numeric_value)
                if observation.numeric_value is not None
                else None
            ),
            "text_value": observation.text_value,
            "unit": observation.unit,
            "payload": observation.payload,
        }
        raw_hash = _sha256(raw_payload)
        return _PreparedPointInTimeObservation(
            observation=observation,
            published_at=published_at,
            available_at=available_at,
            cutoff_at=cutoff_at,
            retrieved_at=retrieved_at,
            raw_payload=raw_payload,
            raw_hash=raw_hash,
            ingestion_id=f"ing-{uuid4().hex}",
            release_id=(
                f"rel-{_sha256({**raw_payload, 'raw_hash': raw_hash})[:60]}"
            ),
            quality_assessment_id=f"qa-{uuid4().hex}",
            source_url=observation.source_url or _source_url(observation.source),
        )

    @staticmethod
    def _raise_pit_payload_conflict() -> None:
        raise ValueError(
            "Observation revision identity already exists with a different "
            "payload; provide its true new available_at or vintage"
        )

    @staticmethod
    def _existing_point_in_time_observations(
        session: Session,
        identities: Sequence[tuple[str, date, datetime, str, str]],
    ) -> dict[tuple[str, date, datetime, str, str], V33PointInTimeObservation]:
        """Load existing revisions in bounded queries below SQLite's bind limit."""

        existing_by_identity: dict[
            tuple[str, date, datetime, str, str], V33PointInTimeObservation
        ] = {}
        identity_columns = tuple_(
            V33PointInTimeObservation.series_code,
            V33PointInTimeObservation.effective_date,
            V33PointInTimeObservation.available_at,
            V33PointInTimeObservation.vintage,
            V33PointInTimeObservation.source,
        )
        for offset in range(0, len(identities), V33_PIT_IDENTITY_QUERY_CHUNK):
            chunk = identities[offset : offset + V33_PIT_IDENTITY_QUERY_CHUNK]
            rows = session.scalars(
                select(V33PointInTimeObservation).where(identity_columns.in_(chunk))
            )
            for row in rows:
                identity = (
                    row.series_code,
                    row.effective_date,
                    _utc(row.available_at),
                    row.vintage,
                    row.source,
                )
                existing_by_identity[identity] = row
        return existing_by_identity

    def ingest_point_in_time_observations(
        self,
        observations: Sequence[PointInTimeObservationInput],
    ) -> list[int]:
        """Bulk public API preserving each release/vintage audit identity.

        A connector may contain multiple published releases in one response.
        Each release remains individually idempotent so a retry cannot create
        a second value or silently replace a historical vintage.
        """

        if not observations:
            return []

        # Validate and hash every item before opening the write transaction.  A
        # bad late item therefore cannot leave a partially committed connector
        # response, unlike the former one-session-per-row implementation.
        prepared = [
            self._prepare_point_in_time_observation(observation)
            for observation in observations
        ]
        unique_identities = list(
            dict.fromkeys(item.revision_identity for item in prepared)
        )

        result_ids: list[int]
        with self.session_factory() as session, session.begin():
            instrument_codes = list(
                dict.fromkeys(
                    item.observation.instrument_code
                    for item in prepared
                    if item.observation.instrument_code is not None
                )
            )
            instrument_ids: dict[str, int] = {}
            for offset in range(
                0, len(instrument_codes), V33_PIT_IDENTITY_QUERY_CHUNK
            ):
                code_chunk = instrument_codes[
                    offset : offset + V33_PIT_IDENTITY_QUERY_CHUNK
                ]
                instrument_ids.update(
                    dict(
                        session.execute(
                            select(Instrument.code, Instrument.id).where(
                                Instrument.code.in_(code_chunk)
                            )
                        ).all()
                    )
                )
            for instrument_code in instrument_codes:
                if instrument_code not in instrument_ids:
                    raise ValueError(f"Unknown instrument code: {instrument_code}")

            existing_by_identity = self._existing_point_in_time_observations(
                session, unique_identities
            )
            new_by_identity: dict[
                tuple[str, date, datetime, str, str],
                tuple[_PreparedPointInTimeObservation, V33PointInTimeObservation],
            ] = {}
            result_references: list[int | V33PointInTimeObservation] = []
            raw_ingestions: list[V33RawIngestion] = []
            source_releases: list[V33SourceRelease] = []
            observation_rows: list[V33PointInTimeObservation] = []
            quality_assessments: list[V33QualityAssessment] = []

            for item in prepared:
                observation = item.observation
                identity = item.revision_identity
                existing = existing_by_identity.get(identity)
                if existing is not None:
                    if existing.raw_payload_hash != item.raw_hash:
                        self._raise_pit_payload_conflict()
                    result_references.append(existing.id)
                    continue

                pending = new_by_identity.get(identity)
                if pending is not None:
                    prior, row = pending
                    if prior.raw_hash != item.raw_hash:
                        self._raise_pit_payload_conflict()
                    result_references.append(row)
                    continue

                instrument_id = (
                    instrument_ids[observation.instrument_code]
                    if observation.instrument_code is not None
                    else None
                )
                raw_ingestions.append(
                    V33RawIngestion(
                        id=item.ingestion_id,
                        instrument_id=instrument_id,
                        dataset=observation.dataset,
                        provider=observation.source,
                        source_url=item.source_url,
                        requested_start_date=observation.effective_date,
                        requested_end_date=observation.effective_date,
                        fetched_at=item.retrieved_at,
                        available_at=item.available_at,
                        vintage=observation.vintage,
                        raw_payload_hash=item.raw_hash,
                        raw_payload_json=item.raw_payload,
                        request_json={"series_code": observation.series_code},
                        status="accepted",
                        quality_status="accepted",
                        records_received=1,
                        records_written=1,
                        error_message=None,
                        created_at=item.retrieved_at,
                    )
                )
                source_releases.append(
                    V33SourceRelease(
                        id=item.release_id,
                        ingestion_id=item.ingestion_id,
                        instrument_id=instrument_id,
                        dataset=observation.dataset,
                        series_code=observation.series_code,
                        observation_period_start=observation.observation_date,
                        observation_period_end=observation.observation_date,
                        effective_date=observation.effective_date,
                        published_at=item.published_at,
                        available_at=item.available_at,
                        vintage=observation.vintage,
                        source=observation.source,
                        source_url=item.source_url,
                        raw_payload_hash=item.raw_hash,
                        quality_status="accepted",
                        payload_json={"cutoff_at": item.cutoff_at.isoformat()},
                        created_at=item.retrieved_at,
                    )
                )
                row = V33PointInTimeObservation(
                    instrument_id=instrument_id,
                    ingestion_id=item.ingestion_id,
                    source_release_id=item.release_id,
                    dataset=observation.dataset,
                    series_code=observation.series_code,
                    observation_date=observation.observation_date,
                    observation_period=observation.observation_period,
                    effective_date=observation.effective_date,
                    published_at=item.published_at,
                    available_at=item.available_at,
                    retrieved_at=item.retrieved_at,
                    cutoff_at=item.cutoff_at,
                    vintage=observation.vintage,
                    numeric_value=observation.numeric_value,
                    text_value=observation.text_value,
                    unit=observation.unit,
                    source=observation.source,
                    source_url=item.source_url,
                    raw_payload_hash=item.raw_hash,
                    quality_status="accepted",
                    payload_json=observation.payload,
                    created_at=item.retrieved_at,
                )
                observation_rows.append(row)
                quality_assessments.append(
                    V33QualityAssessment(
                        id=item.quality_assessment_id,
                        ingestion_id=item.ingestion_id,
                        instrument_id=instrument_id,
                        dataset=observation.dataset,
                        gate_version=V33_OBSERVATION_GATE_VERSION,
                        assessed_at=item.retrieved_at,
                        quality_status="accepted",
                        coverage_start_date=observation.effective_date,
                        coverage_end_date=observation.effective_date,
                        records_received=1,
                        records_valid=1,
                        records_rejected=0,
                        required_fields_json=[
                            "published_at",
                            "available_at",
                            "vintage",
                            "source",
                            "raw_payload_hash",
                        ],
                        missing_fields_json=[],
                        issues_json=[],
                        warnings_json=[],
                        assessed_payload_hash=item.raw_hash,
                        details_json={"cutoff_at": item.cutoff_at.isoformat()},
                        created_at=item.retrieved_at,
                    )
                )
                new_by_identity[identity] = (item, row)
                result_references.append(row)

            # Four bounded flush phases preserve explicit foreign-key ordering
            # while letting SQLAlchemy batch each table.  There is still only
            # one SQLite transaction and one final commit for the whole input.
            if raw_ingestions:
                session.add_all(raw_ingestions)
                session.flush()
                session.add_all(source_releases)
                session.flush()
                session.add_all(observation_rows)
                session.flush()
                session.add_all(quality_assessments)
                session.flush()

            result_ids = [
                reference if isinstance(reference, int) else reference.id
                for reference in result_references
            ]

        return result_ids


__all__ = [
    "BenchmarkAlignment",
    "DataQualityGate",
    "PointInTimeObservationInput",
    "V33DataRepository",
    "V33DataService",
    "V33RefreshResult",
]
