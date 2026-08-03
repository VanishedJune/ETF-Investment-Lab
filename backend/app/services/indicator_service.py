"""Persistence boundary for backend-computed technical indicators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Callable, Mapping

from sqlalchemy import or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..indicators.calculator import (
    IndicatorCalculator,
    IndicatorParameters,
    InvalidIndicatorParametersError,
    InvalidPriceDataError,
    PricePoint,
    TIMEFRAME_PERIODS_PER_YEAR,
)
from ..models.models import IndicatorRecord, Instrument, MarketPrice, utc_now


class IndicatorCalculationStatus(str, Enum):
    """Typed outcomes that callers can handle without exception parsing."""

    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    NO_DATA = "no_data"


@dataclass(frozen=True, slots=True)
class IndicatorCalculationResult:
    """The persisted-calculation outcome and reproducibility metadata."""

    status: IndicatorCalculationStatus
    instrument_code: str
    timeframe: str
    formula_label: str
    parameters: Mapping[str, object]
    data_cutoff: date | None
    completeness: Mapping[str, bool]
    price_rows: int = 0
    records_added: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    error: str | None = None


class IndicatorService:
    """Read ordered local prices, calculate indicators, and idempotently persist them."""

    indicator_name = "technical_indicators"
    payload_schema_version = 3
    required_chart_fields = frozenset({"dif", "dea", "macd_histogram", "dif_first_change"})
    chart_timeframes = ("daily", "weekly", "monthly")

    def __init__(
        self,
        session_factory: Callable[[], Session],
        parameters: IndicatorParameters | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.calculator = IndicatorCalculator(parameters)

    def recalculate(self, instrument_code: str, timeframe: str = "daily") -> IndicatorCalculationResult:
        """Recalculate every stored date for one instrument/timeframe without future leakage."""
        try:
            self.calculator.parameters.validate()
        except InvalidIndicatorParametersError as error:
            return self._validation_result(
                instrument_code,
                timeframe,
                f"Invalid indicator input: {error}",
            )
        if timeframe not in TIMEFRAME_PERIODS_PER_YEAR:
            return self._validation_result(
                instrument_code, timeframe, f"Unsupported timeframe: {timeframe}"
            )
        with self.session_factory() as session, session.begin():
            instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
            if instrument is None:
                return self._validation_result(
                    instrument_code, timeframe, f"Unknown instrument code: {instrument_code}"
                )
            prices = session.scalars(
                select(MarketPrice)
                .where(
                    MarketPrice.instrument_id == instrument.id,
                    MarketPrice.timeframe == timeframe,
                )
                .order_by(MarketPrice.trade_date, MarketPrice.id)
            ).all()
            if not prices:
                return IndicatorCalculationResult(
                    status=IndicatorCalculationStatus.NO_DATA,
                    instrument_code=instrument_code,
                    timeframe=timeframe,
                    formula_label=self.calculator.formula_label,
                    parameters=self.calculator.parameters.as_dict(),
                    data_cutoff=None,
                    completeness=self.calculator.calculate((), timeframe).completeness,
                    error="No stored prices for instrument/timeframe",
                )
            try:
                computation = self.calculator.calculate(
                    tuple(
                        PricePoint(
                            trade_date=price.trade_date,
                            close=price.adjusted_close_price or price.close_price,
                            volume=(
                                price.volume * Decimal(price.volume_multiplier)
                                if instrument_code != "NDX"
                                and price.volume is not None
                                else None
                            ),
                        )
                        for price in prices
                    ),
                    timeframe,
                )
            except (InvalidIndicatorParametersError, InvalidPriceDataError) as error:
                return self._validation_result(
                    instrument_code,
                    timeframe,
                    f"Invalid indicator input: {error}",
                )
            expected_dates = {snapshot.trade_date for snapshot in computation.snapshots}
            stale_records = session.scalars(
                select(IndicatorRecord).where(
                    IndicatorRecord.instrument_id == instrument.id,
                    IndicatorRecord.timeframe == timeframe,
                    IndicatorRecord.indicator_date.not_in(expected_dates),
                )
            ).all()
            for record in stale_records:
                session.delete(record)
            added = updated = skipped = 0
            for snapshot in computation.snapshots:
                payload = self._payload(snapshot.values, snapshot.metadata)
                outcome = self._upsert_snapshot(
                    session,
                    instrument_id=instrument.id,
                    indicator_date=snapshot.trade_date,
                    timeframe=timeframe,
                    payload=payload,
                )
                if outcome == "added":
                    added += 1
                elif outcome == "updated":
                    updated += 1
                else:
                    skipped += 1
            return IndicatorCalculationResult(
                status=IndicatorCalculationStatus.SUCCESS,
                instrument_code=instrument_code,
                timeframe=timeframe,
                formula_label=computation.formula_label,
                parameters=computation.parameters,
                data_cutoff=computation.data_cutoff,
                completeness=computation.completeness,
                price_rows=len(prices),
                records_added=added,
                records_updated=updated,
                records_skipped=skipped,
            )

    def recalculate_all_timeframes(
        self, instrument_code: str
    ) -> tuple[IndicatorCalculationResult, ...]:
        """Synchronize daily, weekly, and monthly indicator dates with stored prices."""
        return tuple(self.recalculate(instrument_code, timeframe) for timeframe in self.chart_timeframes)

    def stale_timeframes(self, instrument_code: str) -> tuple[str, ...]:
        """Return stored chart periods whose indicator payload is absent or obsolete.

        This compatibility check intentionally keys off required fields, not just a
        version label.  Older payloads that already contain the complete chart
        contract remain untouched, while the pre-DIF-derivative 159941 rows are
        deterministically rebuilt from their existing local prices.
        """

        with self.session_factory() as session:
            instrument = session.scalar(
                select(Instrument).where(Instrument.code == instrument_code)
            )
            if instrument is None:
                return ()
            stale: list[str] = []
            for timeframe in self.chart_timeframes:
                price_dates = tuple(
                    session.scalars(
                        select(MarketPrice.trade_date)
                        .where(
                            MarketPrice.instrument_id == instrument.id,
                            MarketPrice.timeframe == timeframe,
                        )
                        .order_by(MarketPrice.trade_date)
                    )
                )
                if not price_dates:
                    continue
                rows = tuple(
                    session.scalars(
                        select(IndicatorRecord)
                        .where(
                            IndicatorRecord.instrument_id == instrument.id,
                            IndicatorRecord.timeframe == timeframe,
                        )
                        .order_by(IndicatorRecord.indicator_date)
                    )
                )
                if tuple(row.indicator_date for row in rows) != price_dates:
                    stale.append(timeframe)
                    continue
                if any(
                    not self.required_chart_fields.issubset(
                        dict((row.indicator_values or {}).get("values", {}))
                    )
                    for row in rows
                ):
                    stale.append(timeframe)
            return tuple(stale)

    def recalculate_stale_timeframes(
        self, instrument_code: str
    ) -> tuple[IndicatorCalculationResult, ...]:
        """Rebuild only periods that fail the persisted chart-field contract."""

        return tuple(
            self.recalculate(instrument_code, timeframe)
            for timeframe in self.stale_timeframes(instrument_code)
        )

    calculate_and_persist = recalculate

    def _validation_result(
        self, instrument_code: str, timeframe: str, error: str
    ) -> IndicatorCalculationResult:
        return IndicatorCalculationResult(
            status=IndicatorCalculationStatus.VALIDATION_ERROR,
            instrument_code=instrument_code,
            timeframe=timeframe,
            formula_label=self.calculator.formula_label,
            parameters=self.calculator.parameters.as_dict(),
            data_cutoff=None,
            completeness=self.calculator._empty_completeness(),
            error=error,
        )

    def _upsert_snapshot(
        self,
        session: Session,
        *,
        instrument_id: int,
        indicator_date: date,
        timeframe: str,
        payload: dict[str, object],
    ) -> str:
        """Atomically add, change, or retain one calculated period record."""
        identity = {
            "instrument_id": instrument_id,
            "indicator_date": indicator_date,
            "timeframe": timeframe,
        }
        values = {
            "indicator_name": self.indicator_name,
            "indicator_value": None,
            "indicator_values": payload,
        }
        insert = sqlite_insert(IndicatorRecord).values(**identity, **values)
        inserted = session.execute(
            insert.on_conflict_do_nothing(
                index_elements=("instrument_id", "indicator_date", "timeframe")
            )
        )
        if inserted.rowcount:
            return "added"
        changed = or_(
            IndicatorRecord.indicator_name != self.indicator_name,
            IndicatorRecord.indicator_value.is_not(None),
            IndicatorRecord.indicator_values != payload,
        )
        updated = session.execute(
            insert.on_conflict_do_update(
                index_elements=("instrument_id", "indicator_date", "timeframe"),
                set_={**values, "updated_at": utc_now()},
                where=changed,
            )
        )
        return "updated" if updated.rowcount else "skipped"

    def _payload(
        self,
        values: Mapping[str, Decimal | None],
        metadata: Mapping[str, Decimal | None],
    ) -> dict[str, object]:
        # JSON cannot faithfully transport Decimal. Persist canonical Decimal
        # strings, never binary floats; the nullable scalar column is not used
        # because this aggregate record contains multiple named values.
        return {
            "schema_version": self.payload_schema_version,
            "formula_label": self.calculator.formula_label,
            "parameters": self.calculator.parameters.as_dict(),
            "values": {name: self._decimal_string(value) for name, value in values.items()},
            "metadata": {name: self._decimal_string(value) for name, value in metadata.items()},
        }

    @staticmethod
    def _decimal_string(value: Decimal | None) -> str | None:
        if value is None:
            return None
        rendered = format(value, "f").rstrip("0").rstrip(".")
        return rendered or "0"
