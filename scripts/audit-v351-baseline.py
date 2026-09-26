"""Read-only V3.5.1 baseline audit and pre-migration online backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_351 = "V3.5.1_EFFECTIVE_CHALLENGER_AND_DYNAMIC_ETF_SLOTS"


def _table_hash(connection: sqlite3.Connection, table: str) -> str:
    """Deterministic row-level SHA-256 over all rows of a table."""
    digest = hashlib.sha256()
    try:
        rows = connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
    except sqlite3.OperationalError:
        return "TABLE_MISSING"
    for row in rows:
        digest.update(repr(tuple(row)).encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit(database: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        report: dict[str, object] = {
            "protocol": PROTOCOL_351,
            "database": str(database),
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            "foreign_key_check": len(connection.execute("PRAGMA foreign_key_check").fetchall()),
        }
        instruments = connection.execute(
            "SELECT code, name, category, is_active FROM instruments ORDER BY id"
        ).fetchall()
        report["instruments"] = [
            {"code": r[0], "name": r[1], "category": r[2], "is_active": bool(r[3])}
            for r in instruments
        ]
        chart_etfs = [
            code
            for code in ("159941", "518600", "512800", "512690")
            if any(r[0] == code for r in instruments)
        ]
        report["current_chart_etf_slots"] = chart_etfs
        report["index_markets"] = [r[0] for r in instruments if r[2] == "index"]

        v35_tables = [
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v35_%' ORDER BY name"
            ).fetchall()
        ]
        report["v35_table_count"] = len(v35_tables)
        report["v35_table_hashes"] = {t: _table_hash(connection, t) for t in v35_tables}

        challenge_summary: dict[str, object] = {}
        for market in ("399006", "159941", *chart_etfs):
            challenges = connection.execute(
                "SELECT challenger_family, COUNT(*) FROM v35_challenges "
                "WHERE model_market=? GROUP BY challenger_family",
                (market,),
            ).fetchall()
            promotions = connection.execute(
                "SELECT COUNT(*) FROM v35_promotions WHERE model_market=?", (market,)
            ).fetchone()[0]
            candidates = connection.execute(
                "SELECT COALESCE(SUM(candidate_count),0) FROM v35_challenges WHERE model_market=?",
                (market,),
            ).fetchone()[0]
            champion = connection.execute(
                "SELECT champion_package_id FROM v35_bootstrap_state WHERE model_market=?",
                (market,),
            ).fetchone()
            challenge_summary[market] = {
                "challenges_by_family": dict(challenges),
                "promotion_count": promotions,
                "challenger_total": candidates,
                "champion_package_id": champion[0] if champion else None,
            }
        report["v35_challenge_summary"] = challenge_summary

        hash_columns = {
            "v35_forecasts": "forecast_hash",
            "v35_strategy_snapshots": "strategy_hash",
            "v35_sim_evaluations": "evaluation_hash",
        }
        digest_rows: dict[str, object] = {}
        for table, column in hash_columns.items():
            values = connection.execute(
                f"SELECT {column} FROM {table} ORDER BY rowid"
            ).fetchall()
            combined = hashlib.sha256()
            for (value,) in values:
                combined.update(str(value).encode("utf-8"))
                combined.update(b"\n")
            digest_rows[table] = {
                "row_count": len(values),
                "combined_sha256": combined.hexdigest(),
            }
        report["hash_columns"] = digest_rows
        return report
    finally:
        connection.close()


def online_backup(database: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=30)
    target = sqlite3.connect(str(output), timeout=60)
    try:
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "backup": str(output),
            "bytes": output.stat().st_size,
            "integrity_check": integrity,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    finally:
        target.close()
        source.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=Path("data/investment_lab.db"), type=Path)
    parser.add_argument("--report", default=Path("audit/v351/baseline.json"), type=Path)
    parser.add_argument("--backup-dir", default=Path("audit/v351/backups"), type=Path)
    parser.add_argument("--skip-backup", action="store_true")
    args = parser.parse_args()
    database = args.database.resolve()
    report = audit(database)
    if not args.skip_backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = args.backup_dir / f"investment_lab_pre_v351_{stamp}.db"
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
