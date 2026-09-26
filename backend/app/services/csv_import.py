"""CSV preview, validation, and transaction-safe import for local prices."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Callable

from ..schemas.market import (
    CsvImportResult,
    CsvPreview,
    CsvRowError,
    MarketDataRecord,
    ProviderResult,
    ProviderStatus,
)


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "trade_date": ("date", "trade_date", "日期"),
    "open_price": ("raw_open", "open", "open_price", "开盘"),
    "high_price": ("raw_high", "high", "high_price", "最高"),
    "low_price": ("raw_low", "low", "low_price", "最低"),
    "close_price": ("raw_close", "close", "close_price", "收盘"),
    "adjusted_close_price": ("adj_close", "adjusted_close", "adjusted_close_price"),
    "volume": ("volume", "成交量"),
    "amount": ("amount", "成交额"),
}


class ParsedCsv:
    def __init__(
        self,
        headers: list[str],
        mapping: dict[str, str],
        records: list[MarketDataRecord],
        errors: list[CsvRowError],
        total_rows: int,
    ) -> None:
        self.headers = headers
        self.mapping = mapping
        self.records = records
        self.errors = errors
        self.total_rows = total_rows


def _normalise_header(value: str) -> str:
    return value.strip().lstrip("\ufeff").lower().replace(" ", "").replace("_", "")


def _decode_csv(contents: str | bytes | Path) -> str:
    if isinstance(contents, Path):
        contents = contents.read_bytes()
    if isinstance(contents, str):
        return contents
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return contents.decode(encoding)
        except UnicodeDecodeError:
            continue
    return contents.decode("utf-8", errors="replace")


def _find_mapping(headers: list[str]) -> dict[str, str]:
    normalized_headers = {_normalise_header(header): header for header in headers}
    mapping: dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            header = normalized_headers.get(_normalise_header(alias))
            if header is not None:
                mapping[field] = header
                break
    return mapping


def _cell_text(value: object | None) -> str:
    """Normalize DictReader's None for omitted trailing cells before parsing."""
    return "" if value is None else str(value).strip()


def _parse_date(value: object | None) -> date:
    cleaned = _cell_text(value)
    for parser in (date.fromisoformat,):
        try:
            return parser(cleaned)
        except ValueError:
            pass
    if len(cleaned) == 8 and cleaned.isdigit():
        return date(int(cleaned[:4]), int(cleaned[4:6]), int(cleaned[6:]))
    raise ValueError("must be an ISO date (YYYY-MM-DD) or YYYYMMDD")


def _parse_decimal(value: object | None, field: str) -> Decimal:
    cleaned = _cell_text(value).replace(",", "")
    if not cleaned:
        raise ValueError(f"{field} is required")
    try:
        return Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError(f"{field} must be a number") from error


def parse_csv_market_data(contents: str | bytes | Path) -> ParsedCsv:
    """Normalize standard English/Chinese headers and retain row-level errors."""
    reader = csv.DictReader(StringIO(_decode_csv(contents)), skipinitialspace=True)
    headers = [header for header in (reader.fieldnames or []) if header is not None]
    mapping = _find_mapping(headers)
    errors: list[CsvRowError] = []
    records: list[MarketDataRecord] = []
    required = ("trade_date", "open_price", "high_price", "low_price", "close_price")
    for field in required:
        if field not in mapping:
            errors.append(CsvRowError(row_number=1, field=field, message=f"Missing required column: {field}"))

    total_rows = 0
    if errors:
        return ParsedCsv(headers, mapping, records, errors, total_rows)

    for row_number, row in enumerate(reader, start=2):
        total_rows += 1
        try:
            surplus_cells = row.get(None, [])
            if any(_cell_text(value) for value in surplus_cells):
                raise ValueError("Row contains surplus cells without headers")
            trade_date = _parse_date(row.get(mapping["trade_date"], ""))
            close_price = _parse_decimal(row.get(mapping["close_price"], ""), "close_price")
            numeric: dict[str, Decimal | None] = {}
            for field in ("open_price", "high_price", "low_price", "volume", "amount"):
                header = mapping.get(field)
                value = _cell_text(row.get(header, "") if header else "")
                if field in ("open_price", "high_price", "low_price"):
                    numeric[field] = _parse_decimal(value, field)
                else:
                    numeric[field] = _parse_decimal(value, field) if value else None
            adjusted_header = mapping.get("adjusted_close_price")
            adjusted_text = _cell_text(row.get(adjusted_header, "") if adjusted_header else "")
            adjusted_close = (
                _parse_decimal(adjusted_text, "adjusted_close_price").quantize(Decimal("0.00000001"))
                if adjusted_text
                else None
            )
            records.append(
                MarketDataRecord(
                    trade_date=trade_date,
                    close_price=close_price,
                    adjusted_close_price=adjusted_close,
                    source="CSV",
                    **numeric,
                )
            )
        except (AttributeError, TypeError, ValueError) as error:
            errors.append(CsvRowError(row_number=row_number, message=str(error)))
    return ParsedCsv(headers, mapping, records, errors, total_rows)


class CsvImportService:
    """Import only validated CSV records; invalid rows never enter a transaction."""

    def __init__(self, session_factory: Callable) -> None:
        self.session_factory = session_factory

    def preview(self, contents: str | bytes | Path, sample_size: int = 20) -> CsvPreview:
        try:
            parsed = parse_csv_market_data(contents)
        except (OSError, UnicodeError, ValueError) as error:
            return CsvPreview(
                invalid_rows=1,
                row_errors=[CsvRowError(row_number=0, message=f"CSV cannot be read: {error}")],
            )
        return CsvPreview(
            headers=parsed.headers,
            mapping=parsed.mapping,
            sample=parsed.records[:sample_size],
            valid_rows=len(parsed.records),
            invalid_rows=len(parsed.errors),
            row_errors=parsed.errors,
        )

    def validate(self, contents: str | bytes | Path) -> CsvPreview:
        return self.preview(contents)

    def confirm_import(self, instrument_code: str, contents: str | bytes | Path) -> CsvImportResult:
        try:
            parsed = parse_csv_market_data(contents)
        except (OSError, UnicodeError, ValueError) as error:
            return CsvImportResult(
                instrument_code=instrument_code,
                failed=1,
                row_errors=[CsvRowError(row_number=0, message=f"CSV cannot be read: {error}")],
            )
        if not parsed.records:
            return CsvImportResult(
                instrument_code=instrument_code,
                failed=len(parsed.errors),
                row_errors=parsed.errors,
            )
        from .market_data import MarketDataService

        response = MarketDataService(self.session_factory).store_result(
            instrument_code,
            ProviderResult.success("CSV", parsed.records),
        )
        if response.status is ProviderStatus.ERROR:
            return CsvImportResult(
                instrument_code=instrument_code,
                failed=len(parsed.errors) + 1,
                row_errors=[
                    *parsed.errors,
                    CsvRowError(row_number=0, message=response.error or "CSV import was rejected"),
                ],
            )
        return CsvImportResult(
            instrument_code=instrument_code,
            added=response.records_added,
            updated=response.records_updated,
            skipped=response.records_skipped,
            failed=len(parsed.errors),
            row_errors=parsed.errors,
        )
