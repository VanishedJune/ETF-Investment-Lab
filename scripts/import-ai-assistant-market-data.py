"""Explicit, one-way import of eight verified monthly-report ETF CSV files.

The command never invokes a model, forecast, strategy, report, or portfolio
writer.  Formal daily rows are validated as one bundle and then imported with
backup/restore protection.  Weekly/monthly bars and indicators are rebuilt by
InvestmentLab's existing deterministic services only after all daily rows are
stored.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT.parent / "AI-Investment-Assistant"
DEFAULT_DATABASE = ROOT / "data" / "investment_lab.db"
ETF_CODES = ("159915", "517520", "516150", "159622", "159611", "515220", "512800", "512690")
SOURCE_NAME = "AI_ASSISTANT_CSV"
AUDIT_SCHEMA = "ai-assistant-import-audit-v1"
STORAGE_SCALE = Decimal("100000000")
REQUIRED_COLUMNS = {
    "code",
    "date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "adj_close",
    "volume",
    "amount",
    "run_id",
}


@dataclass(frozen=True)
class SourceRow:
    code: str
    trade_date: str
    raw_open: Decimal
    raw_high: Decimal
    raw_low: Decimal
    raw_close: Decimal
    adj_close: Decimal
    volume: Decimal | None
    amount: Decimal | None
    run_id: str


@dataclass(frozen=True)
class SourceFile:
    code: str
    path: Path
    relative_path: str
    rows: tuple[SourceRow, ...]


@dataclass(frozen=True)
class SourceBundle:
    source_root: Path
    run_id: str
    as_of: str
    files: tuple[SourceFile, ...]
    price_ticks: dict[str, Decimal]


def _decimal(value: object, field: str, *, optional: bool = False) -> Decimal | None:
    text = "" if value is None else str(value).strip().replace(",", "")
    if not text:
        if optional:
            return None
        raise ValueError(f"{field} is required")
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{field} is not numeric") from error
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return parsed


def _price(value: object, field: str) -> Decimal:
    parsed = _decimal(value, field)
    assert parsed is not None
    return parsed.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def _load_ticks(source_root: Path) -> dict[str, Decimal]:
    config_path = source_root / "config" / "instruments.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    instruments = config.get("instruments")
    if not isinstance(instruments, list):
        raise ValueError("instrument configuration is missing instruments")
    ticks: dict[str, Decimal] = {}
    for item in instruments:
        if not isinstance(item, dict) or str(item.get("code")) not in ETF_CODES:
            continue
        code = str(item["code"])
        value = _decimal(item.get("price_tick"), f"{code}.price_tick")
        assert value is not None
        if value <= 0:
            raise ValueError(f"{code}.price_tick must be positive")
        ticks[code] = value
    if set(ticks) != set(ETF_CODES):
        raise ValueError(f"price_tick configuration is incomplete: {sorted(set(ETF_CODES) - set(ticks))}")
    return ticks


def _parse_source_file(path: Path, *, code: str, run_id: str) -> tuple[SourceRow, ...]:
    rows: list[SourceRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - headers)
        if missing:
            raise ValueError(f"{path.name}: missing V2 columns {missing}")
        previous_date = ""
        seen: set[str] = set()
        for row_number, raw in enumerate(reader, start=2):
            row_code = str(raw.get("code", "")).strip()
            row_run_id = str(raw.get("run_id", "")).strip()
            trade_date = str(raw.get("date", "")).strip()
            if row_code != code:
                raise ValueError(f"{path.name}:{row_number}: code {row_code!r} != {code}")
            if row_run_id != run_id:
                raise ValueError(f"{path.name}:{row_number}: run_id mismatch")
            try:
                datetime.strptime(trade_date, "%Y-%m-%d")
            except ValueError as error:
                raise ValueError(f"{path.name}:{row_number}: invalid date") from error
            if trade_date in seen or (previous_date and trade_date <= previous_date):
                raise ValueError(f"{path.name}:{row_number}: dates are not strictly increasing and unique")
            seen.add(trade_date)
            previous_date = trade_date
            opened = _price(raw.get("raw_open"), "raw_open")
            high = _price(raw.get("raw_high"), "raw_high")
            low = _price(raw.get("raw_low"), "raw_low")
            close = _price(raw.get("raw_close"), "raw_close")
            adjusted = _price(raw.get("adj_close"), "adj_close")
            volume = _decimal(raw.get("volume"), "volume", optional=True)
            amount = _decimal(raw.get("amount"), "amount", optional=True)
            if min(opened, high, low, close, adjusted) <= 0:
                raise ValueError(f"{path.name}:{row_number}: prices must be positive")
            if low > min(opened, close) or high < max(opened, close) or low > high:
                raise ValueError(f"{path.name}:{row_number}: invalid OHLC")
            if volume is not None and volume < 0 or amount is not None and amount < 0:
                raise ValueError(f"{path.name}:{row_number}: volume/amount must be nonnegative")
            rows.append(
                SourceRow(
                    code=code,
                    trade_date=trade_date,
                    raw_open=opened,
                    raw_high=high,
                    raw_low=low,
                    raw_close=close,
                    adj_close=adjusted,
                    volume=volume,
                    amount=amount,
                    run_id=run_id,
                )
            )
    if not rows:
        raise ValueError(f"{path.name}: no data rows")
    return tuple(rows)


def load_bundle(source_root: Path) -> SourceBundle:
    source_root = source_root.resolve()
    manifest_path = source_root / "data_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("schema_version") != "2.0.0":
        raise ValueError("unsupported data manifest schema")
    if manifest.get("market_status") != "closed":
        raise ValueError("formal market data is not closed")
    if manifest.get("quality_status") not in {"passed", "passed_with_warnings"}:
        raise ValueError("formal market quality gate did not pass")
    run_id = str(manifest.get("run_id", "")).strip()
    as_of = str(manifest.get("as_of", "")).strip()
    if not run_id or not as_of:
        raise ValueError("manifest run_id/as_of is missing")
    latest_dates = manifest.get("latest_dates")
    if not isinstance(latest_dates, dict) or {str(latest_dates.get(code)) for code in ETF_CODES} != {as_of}:
        raise ValueError("eight ETFs do not share the manifest cutoff")
    file_entries = manifest.get("files")
    if not isinstance(file_entries, list):
        raise ValueError("manifest files list is missing")

    files: list[SourceFile] = []
    for code in ETF_CODES:
        matches = [
            item
            for item in file_entries
            if isinstance(item, dict)
            and Path(str(item.get("path", ""))).name.startswith(f"{code}_")
            and Path(str(item.get("path", ""))).name.endswith("_日线.csv")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one manifest daily file for {code}, found {len(matches)}")
        entry = matches[0]
        relative = str(entry["path"])
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"manifest path is missing or unsafe: {relative}")
        if str(entry.get("run_id")) != run_id or str(entry.get("end")) != as_of:
            raise ValueError(f"manifest file run_id/end mismatch: {relative}")
        rows = _parse_source_file(path, code=code, run_id=run_id)
        if len(rows) != int(entry.get("rows", -1)):
            raise ValueError(f"manifest row count mismatch: {relative}")
        if rows[0].trade_date != str(entry.get("start")) or rows[-1].trade_date != as_of:
            raise ValueError(f"manifest date range mismatch: {relative}")
        files.append(SourceFile(code, path, relative, rows))

    return SourceBundle(
        source_root=source_root,
        run_id=run_id,
        as_of=as_of,
        files=tuple(files),
        price_ticks=_load_ticks(source_root),
    )


def _audit_path(database: Path) -> Path:
    return database.parent / "ai-assistant-import-runs.json"


def _read_audit(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"schema_version": AUDIT_SCHEMA, "runs": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != AUDIT_SCHEMA or not isinstance(value.get("runs"), list):
        raise RuntimeError(f"invalid import audit file: {path}")
    return value


def _existing_audit(bundle: SourceBundle, audit: dict[str, object]) -> dict[str, object] | None:
    for item in audit["runs"]:
        if not isinstance(item, dict) or item.get("run_id") != bundle.run_id:
            continue
        if item.get("status") == "success":
            return item
    return None


def _db_decimal(value: object | None) -> Decimal | None:
    return None if value is None else Decimal(int(value)) / STORAGE_SCALE


def _storage(value: Decimal | None) -> int | None:
    if value is None:
        return None
    scaled = value * STORAGE_SCALE
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(f"{value} cannot be stored at 8-decimal fixed precision")
    return int(integral)


def preflight(bundle: SourceBundle, database: Path) -> dict[str, object]:
    database = database.resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database does not exist: {database}")
    audit = _read_audit(_audit_path(database))
    prior = _existing_audit(bundle, audit)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick != "ok":
            raise RuntimeError(f"database quick_check failed: {quick}")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(market_prices)")}
        required = {
            "instrument_id", "trade_date", "timeframe", "open_price", "high_price",
            "low_price", "close_price", "adjusted_close_price", "volume", "turnover",
            "source", "volume_source", "volume_multiplier",
        }
        if not required.issubset(columns):
            raise RuntimeError(f"market_prices schema is missing {sorted(required - columns)}")
        instruments = {
            str(row["code"]): int(row["id"])
            for row in connection.execute(
                "SELECT id, code FROM instruments WHERE code IN ({})".format(",".join("?" for _ in ETF_CODES)),
                ETF_CODES,
            )
        }
        if set(instruments) != set(ETF_CODES):
            raise RuntimeError(f"database instrument universe is incomplete: {sorted(set(ETF_CODES) - set(instruments))}")
        operations: dict[str, dict[str, object]] = {}
        conflicts: list[str] = []
        for source_file in bundle.files:
            instrument_id = instruments[source_file.code]
            existing = {
                str(row["trade_date"]): row
                for row in connection.execute(
                    "SELECT * FROM market_prices WHERE instrument_id=? AND timeframe='daily'",
                    (instrument_id,),
                )
            }
            added = updated = skipped = 0
            changes: list[tuple[str, SourceRow, int | None]] = []
            tick = bundle.price_ticks[source_file.code]
            for row in source_file.rows:
                current = existing.get(row.trade_date)
                if current is None:
                    added += 1
                    changes.append(("insert", row, None))
                    continue
                for field, source_value in (
                    ("open_price", row.raw_open),
                    ("high_price", row.raw_high),
                    ("low_price", row.raw_low),
                    ("close_price", row.raw_close),
                ):
                    current_value = _db_decimal(current[field])
                    if current_value is None or abs(current_value - source_value) > tick:
                        conflicts.append(
                            f"{source_file.code} {row.trade_date} {field}: database={current_value} source={source_value} tick={tick}"
                        )
                desired = {
                    "open_price": row.raw_open,
                    "high_price": row.raw_high,
                    "low_price": row.raw_low,
                    "close_price": row.raw_close,
                    "adjusted_close_price": row.adj_close,
                    "volume": row.volume,
                    "turnover": row.amount if row.amount is not None else _db_decimal(current["turnover"]),
                }
                changed = any(_db_decimal(current[field]) != value for field, value in desired.items())
                if changed:
                    updated += 1
                    changes.append(("update", row, int(current["id"])))
                else:
                    skipped += 1
            operations[source_file.code] = {
                "instrument_id": instrument_id,
                "added": added,
                "updated": updated,
                "skipped": skipped,
                "changes": changes,
                "source_rows": len(source_file.rows),
            }
        if conflicts:
            raise RuntimeError("price conflicts exceed one tick:\n" + "\n".join(conflicts[:20]))
        return {
            "database": str(database),
            "quick_check": quick,
            "already_synced": prior is not None and all(not op["changes"] for op in operations.values()),
            "operations": operations,
        }
    finally:
        connection.close()


class SyncLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "SyncLock":
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"sync is already running: {self.path}") from error
        os.write(self.fd, f"pid={os.getpid()}\n".encode("ascii"))
        return self

    def __exit__(self, *_args: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _desktop_running(database: Path) -> bool:
    port_file = database.parent / "desktop-port.json"
    if not port_file.is_file():
        return False
    try:
        pid = int(json.loads(port_file.read_text(encoding="utf-8"))["pid"])
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _backup_database(database: Path, backup: Path) -> None:
    source = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
        if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("temporary database backup failed quick_check")
    finally:
        destination.close()
        source.close()


def _restore_database(database: Path, backup: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(str(database) + suffix).unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{backup.as_posix()}?mode=ro", uri=True)
    destination = sqlite3.connect(database)
    try:
        source.backup(destination)
        if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("backup cannot be restored: quick_check failed")
    finally:
        destination.close()
        source.close()


def _apply_daily(bundle: SourceBundle, database: Path, plan: dict[str, object]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            for source_file in bundle.files:
                operations = plan["operations"][source_file.code]
                instrument_id = int(operations["instrument_id"])
                for action, row, row_id in operations["changes"]:
                    amount = _storage(row.amount)
                    values = (
                        _storage(row.raw_open),
                        _storage(row.raw_high),
                        _storage(row.raw_low),
                        _storage(row.raw_close),
                        _storage(row.adj_close),
                        _storage(row.volume),
                    )
                    if action == "insert":
                        connection.execute(
                            """INSERT INTO market_prices
                            (instrument_id,trade_date,timeframe,open_price,high_price,low_price,close_price,
                             adjusted_close_price,volume,volume_multiplier,turnover,source,volume_source,created_at,updated_at)
                            VALUES (?,?, 'daily', ?,?,?,?,?,?,1,?,?,?,?,?)""",
                            (
                                instrument_id,
                                row.trade_date,
                                *values,
                                amount,
                                SOURCE_NAME,
                                SOURCE_NAME if row.volume is not None else None,
                                now,
                                now,
                            ),
                        )
                    else:
                        if amount is None:
                            connection.execute(
                                """UPDATE market_prices SET open_price=?,high_price=?,low_price=?,close_price=?,
                                adjusted_close_price=?,volume=?,source=?,volume_source=?,updated_at=? WHERE id=?""",
                                (*values, SOURCE_NAME, SOURCE_NAME if row.volume is not None else None, now, row_id),
                            )
                        else:
                            connection.execute(
                                """UPDATE market_prices SET open_price=?,high_price=?,low_price=?,close_price=?,
                                adjusted_close_price=?,volume=?,turnover=?,source=?,volume_source=?,updated_at=? WHERE id=?""",
                                (*values, amount, SOURCE_NAME, SOURCE_NAME if row.volume is not None else None, now, row_id),
                            )
    finally:
        connection.close()


def _rebuild_derived(database: Path, codes: tuple[str, ...]) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.app.database.session import create_session_factory
    from backend.app.services.indicator_service import IndicatorService
    from backend.app.services.market_calendar import ExchangeCalendarProvider
    from backend.app.services.market_data import MarketDataService

    sessions = create_session_factory(database)
    market = MarketDataService(sessions, calendar_provider=ExchangeCalendarProvider())
    indicators = IndicatorService(sessions)
    for code in codes:
        market.aggregate_periods(code, as_of=None)
        indicators.recalculate_all_timeframes(code)


def _verify_database(bundle: SourceBundle, database: Path) -> None:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("post-import database quick_check failed")
        for source_file in bundle.files:
            instrument = connection.execute("SELECT id FROM instruments WHERE code=?", (source_file.code,)).fetchone()
            assert instrument is not None
            stored = connection.execute(
                "SELECT trade_date,adjusted_close_price FROM market_prices WHERE instrument_id=? AND timeframe='daily' ORDER BY trade_date",
                (instrument[0],),
            ).fetchall()
            by_date = {str(row["trade_date"]): _db_decimal(row["adjusted_close_price"]) for row in stored}
            for source in source_file.rows:
                if by_date.get(source.trade_date) != source.adj_close:
                    raise RuntimeError(f"post-import adjusted close mismatch: {source_file.code} {source.trade_date}")
    finally:
        connection.close()


def sync_database(
    bundle: SourceBundle,
    database: Path,
    *,
    rebuild_derived: Callable[[Path, tuple[str, ...]], None] = _rebuild_derived,
) -> dict[str, object]:
    database = database.resolve()
    if _desktop_running(database):
        raise RuntimeError("InvestmentLab is running; close it before syncing")
    lock_path = database.parent / ".ai-assistant-sync.lock"
    with SyncLock(lock_path):
        plan = preflight(bundle, database)
        if plan["already_synced"]:
            return {
                "status": "already_synced",
                "run_id": bundle.run_id,
                "database": str(database),
                "database_changes": 0,
            }
        required_free = database.stat().st_size + 256 * 1024 * 1024
        available = shutil.disk_usage(database.parent).free
        if available < required_free:
            raise RuntimeError(f"insufficient free space for verified SQLite backup: need {required_free}, have {available}")
        fd, backup_name = tempfile.mkstemp(prefix=".ai-sync-backup-", suffix=".db", dir=database.parent)
        os.close(fd)
        backup = Path(backup_name)
        backup.unlink()
        try:
            _backup_database(database, backup)
            try:
                _apply_daily(bundle, database, plan)
                rebuild_derived(database, ETF_CODES)
                _verify_database(bundle, database)
                audit_path = _audit_path(database)
                audit = _read_audit(audit_path)
                entry = {
                    "run_id": bundle.run_id,
                    "validation": "row-by-row OHLCV, dates, counts and adjustment fields",
                    "data_as_of": bundle.as_of,
                    "imported_at": datetime.now(timezone.utc).isoformat(),
                    "status": "success",
                    "counts": {
                        code: {
                            key: int(plan["operations"][code][key])
                            for key in ("added", "updated", "skipped", "source_rows")
                        }
                        for code in ETF_CODES
                    },
                }
                audit["runs"].append(entry)
                temporary = audit_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, audit_path)
            except Exception:
                _restore_database(database, backup)
                raise
        finally:
            backup.unlink(missing_ok=True)
        return {
            "status": "success",
            "run_id": bundle.run_id,
            "database": str(database),
            "operations": {
                code: {key: int(plan["operations"][code][key]) for key in ("added", "updated", "skipped")}
                for code in ETF_CODES
            },
        }


def check_report(bundle: SourceBundle, database: Path) -> dict[str, object]:
    plan = preflight(bundle, database)
    return {
        "status": "already_synced" if plan["already_synced"] else "ready",
        "mode": "check",
        "source_root": str(bundle.source_root),
        "database": str(database.resolve()),
        "run_id": bundle.run_id,
        "as_of": bundle.as_of,
        "operations": {
            code: {key: int(plan["operations"][code][key]) for key in ("added", "updated", "skipped", "source_rows")}
            for code in ETF_CODES
        },
    }


def main() -> int:
    global ETF_CODES
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--db", "--database", dest="database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    try:
        with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
            codes = tuple(row[0] for row in connection.execute(
                'SELECT instrument_code FROM v351_instrument_slots WHERE active=1 ORDER BY slot_order,id'
            ))
        if not codes or len(codes) != len(set(codes)):
            raise ValueError('active slot directory is missing or has duplicate instruments')
        ETF_CODES = codes
        bundle = load_bundle(args.source_root)
        report = check_report(bundle, args.database) if args.check else sync_database(bundle, args.database)
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
