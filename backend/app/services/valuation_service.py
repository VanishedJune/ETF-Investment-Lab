"""Persistent, auditable valuation storage for direct market indexes only."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.models import DataUpdateLog, Instrument, ValuationRecord
from ..schemas.valuation import (
    ValuationDataRecord,
    ValuationProviderResult,
    ValuationUpdateResponse,
)


class ValuationDataProvider(Protocol):
    source: str

    def fetch(self, instrument_code: str) -> ValuationProviderResult: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ValuationService:
    """Upsert observations, preserve historical dates, and derive PE percentiles."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    @staticmethod
    def _instrument(session: Session, instrument_code: str) -> Instrument:
        instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
        if instrument is None:
            raise ValueError(f"Unknown instrument code: {instrument_code}")
        return instrument

    @staticmethod
    def _raw_values(record: ValuationDataRecord, fetched_at: datetime) -> dict[str, object]:
        return {
            "source": record.source,
            "source_url": record.source_url,
            "proxy": False,
            "notes": record.notes,
            "fetched_at": fetched_at.isoformat(),
            "source_values": record.raw_values,
        }

    @staticmethod
    def _same_values(row: ValuationRecord, record: ValuationDataRecord, raw_values: dict[str, object]) -> bool:
        return (
            row.pe_ratio == record.pe_ratio
            and row.pb_ratio == record.pb_ratio
            and row.dividend_yield == record.dividend_yield
            and row.raw_values == raw_values
        )

    @staticmethod
    def _recompute_percentiles(session: Session, instrument_id: int) -> None:
        rows = session.scalars(
            select(ValuationRecord)
            .where(ValuationRecord.instrument_id == instrument_id)
            .order_by(ValuationRecord.valuation_date)
        ).all()
        pe_values = sorted(row.pe_ratio for row in rows if row.pe_ratio is not None and row.pe_ratio.is_finite())
        if not pe_values:
            for row in rows:
                row.valuation_percentile = None
            return
        for row in rows:
            if row.pe_ratio is None or not row.pe_ratio.is_finite():
                row.valuation_percentile = None
                continue
            rank = sum(value <= row.pe_ratio for value in pe_values)
            row.valuation_percentile = Decimal(rank) / Decimal(len(pe_values))

    def _record_log(
        self,
        instrument_id: int | None,
        result: ValuationProviderResult,
        response: ValuationUpdateResponse,
        started_at: datetime,
    ) -> None:
        with self.session_factory() as session, session.begin():
            session.add(
                DataUpdateLog(
                    instrument_id=instrument_id,
                    dataset="valuations",
                    source=result.source,
                    timeframe="daily",
                    status=response.status,
                    started_at=started_at,
                    completed_at=_now(),
                    records_received=response.records_received,
                    records_written=response.records_written,
                    records_added=response.records_added,
                    records_updated=response.records_updated,
                    records_skipped=response.records_skipped,
                    message=response.error,
                )
            )

    def refresh(self, instrument_code: str, provider: ValuationDataProvider) -> ValuationUpdateResponse:
        """Fetch one permitted direct-index source and persist every returned date."""
        started_at = _now()
        try:
            result = provider.fetch(instrument_code)
        except Exception as error:  # Providers should not throw, but cache safety is mandatory.
            result = ValuationProviderResult.failed(provider.source, f"Valuation provider failed: {error}")

        with self.session_factory() as session:
            try:
                instrument_id = self._instrument(session, instrument_code).id
            except ValueError:
                return ValuationUpdateResponse(
                    source=result.source,
                    status="error",
                    fetched_at=result.fetched_at,
                    error=f"Unknown instrument code: {instrument_code}",
                )

        if result.status != "success" or not result.records:
            cached = self.read(instrument_code)["series"]
            status = "cached" if cached else result.status
            response = ValuationUpdateResponse(
                source=result.source,
                status=status,
                fetched_at=result.fetched_at,
                cached_records=len(cached),
                error=result.error or "Provider returned no usable direct-index valuation records",
                warnings=result.warnings,
            )
            self._record_log(instrument_id, result, response, started_at)
            return response

        outcomes = {"added": 0, "updated": 0, "skipped": 0}
        with self.session_factory() as session, session.begin():
            instrument = self._instrument(session, instrument_code)
            for record in result.records:
                raw_values = self._raw_values(record, result.fetched_at)
                row = session.scalar(
                    select(ValuationRecord).where(
                        ValuationRecord.instrument_id == instrument.id,
                        ValuationRecord.valuation_date == record.valuation_date,
                    )
                )
                if row is None:
                    session.add(
                        ValuationRecord(
                            instrument_id=instrument.id,
                            valuation_date=record.valuation_date,
                            pe_ratio=record.pe_ratio,
                            pb_ratio=record.pb_ratio,
                            dividend_yield=record.dividend_yield,
                            raw_values=raw_values,
                        )
                    )
                    outcomes["added"] += 1
                elif self._same_values(row, record, raw_values):
                    outcomes["skipped"] += 1
                else:
                    row.pe_ratio = record.pe_ratio
                    row.pb_ratio = record.pb_ratio
                    row.dividend_yield = record.dividend_yield
                    row.raw_values = raw_values
                    outcomes["updated"] += 1
            session.flush()
            self._recompute_percentiles(session, instrument.id)
            instrument_id = instrument.id

        response = ValuationUpdateResponse(
            source=result.source,
            status="success",
            fetched_at=result.fetched_at,
            records_received=len(result.records),
            records_written=outcomes["added"] + outcomes["updated"],
            records_added=outcomes["added"],
            records_updated=outcomes["updated"],
            records_skipped=outcomes["skipped"],
            warnings=result.warnings,
        )
        self._record_log(instrument_id, result, response, started_at)
        return response

    def read(self, instrument_code: str) -> dict[str, object]:
        with self.session_factory() as session:
            instrument = self._instrument(session, instrument_code)
            rows = session.scalars(
                select(ValuationRecord)
                .where(ValuationRecord.instrument_id == instrument.id)
                .order_by(ValuationRecord.valuation_date)
            ).all()

        series = [
            {
                "date": row.valuation_date,
                "pe_ratio": row.pe_ratio,
                "pb_ratio": row.pb_ratio,
                "dividend_yield": row.dividend_yield,
                "valuation_percentile": row.valuation_percentile,
                "source": row.raw_values.get("source"),
                "source_url": row.raw_values.get("source_url"),
                "proxy": bool(row.raw_values.get("proxy", False)),
                "notes": row.raw_values.get("notes"),
                "updated_at": row.updated_at,
            }
            for row in rows
        ]
        latest = series[-1] if series else None
        return {
            "instrument_code": instrument_code,
            "source_status": "cached" if latest else "unavailable",
            "latest": latest,
            "series": series,
            "record_count": len(series),
        }
