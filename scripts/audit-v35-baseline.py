"""Read-only V3.5 baseline audit for the production database.

Verifies the current schema version, the V3.4.1 formal iteration counts, the
champion rows, market-data spans and the pre-V3.5 table namespace, then
creates an online SQLite backup under ``audit/v35/backups``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = "V3.5_CAPITAL_DRIVEN_8W"


def _report_row_count(connection: sqlite3.Connection, table: str) -> int:
    try:
        return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    except sqlite3.OperationalError:
        return -1


def audit(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        report: dict[str, object] = {
            "protocol": PROTOCOL,
            "database": str(database),
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }
        report["schema_version"] = int(connection.execute("PRAGMA user_version").fetchone()[0])
        report["integrity_check"] = connection.execute("PRAGMA integrity_check").fetchone()[0]
        report["foreign_key_check"] = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        report["table_count"] = len(tables)
        report["v35_table_namespace"] = sorted(name for name in tables if name.startswith("v35_"))
        for market in ("399006", "159941"):
            report[f"{market}_v341_iterations"] = connection.execute(
                "SELECT COUNT(*) FROM v341_training_iterations WHERE model_market = ?",
                (market,),
            ).fetchone()[0]
            report[f"{market}_v341_forecasts"] = connection.execute(
                "SELECT COUNT(*) FROM v341_forecasts WHERE model_market = ?",
                (market,),
            ).fetchone()[0]
            report[f"{market}_v341_mature_evaluations"] = connection.execute(
                "SELECT COUNT(*) FROM v341_forecast_evaluations "
                "JOIN v341_forecasts ON v341_forecast_evaluations.forecast_id = v341_forecasts.id "
                "WHERE v341_forecasts.model_market = ? "
                "AND v341_forecast_evaluations.horizon_weeks = 13",
                (market,),
            ).fetchone()[0]
        report["v341_model_versions"] = _report_row_count(connection, "v341_model_versions")
        report["v341_optimizer_states"] = _report_row_count(
            connection, "v341_optimizer_states"
        )
        spans: dict[str, object] = {}
        instrument_ids = {
            code: connection.execute(
                "SELECT id FROM instruments WHERE code = ?", (code,)
            ).fetchone()
            for code in ("399006", "159941")
        }
        for code, row in instrument_ids.items():
            if row is None:
                spans[code] = None
                continue
            values = connection.execute(
                "SELECT MIN(trade_date), MAX(trade_date), COUNT(*) "
                "FROM market_prices WHERE instrument_id = ? AND timeframe = 'daily'",
                (row[0],),
            ).fetchone()
            spans[code] = {
                "min_date": values[0],
                "max_date": values[1],
                "daily_rows": values[2],
            }
        report["market_data_spans"] = spans
        frozen_prefixes = tuple(
            name
            for name in tables
            if name.startswith(("v2_", "v31_", "v32_", "v33_", "v34", "v341_", "v342_"))
        )
        report["frozen_legacy_table_count"] = len(frozen_prefixes)
        return report
    finally:
        connection.close()


def online_backup(database: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=30)
    target = sqlite3.connect(str(output), timeout=60)
    try:
        source.backup(target)
        target.execute("PRAGMA integrity_check")
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        size = output.stat().st_size
        return {"backup": str(output), "bytes": size, "integrity_check": integrity}
    finally:
        target.close()
        source.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=Path("data/investment_lab.db"),
        type=Path,
    )
    parser.add_argument("--report", type=Path, default=Path("audit/v35/baseline.json"))
    parser.add_argument("--backup-dir", type=Path, default=Path("audit/v35/backups"))
    parser.add_argument("--skip-backup", action="store_true")
    args = parser.parse_args()
    database = args.database.resolve()
    report = audit(database)
    if not args.skip_backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = args.backup_dir / f"investment_lab_pre_v35_{stamp}.db"
        report["backup"] = online_backup(database, backup)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
