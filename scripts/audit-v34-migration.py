from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.migrations import SCHEMA_VERSION, V34_TABLES, run_migrations
from backend.app.database.session import create_database_engine


FROZEN_TABLES = (
    "v32_training_iterations",
    "v32_model_versions",
    "v32_weekly_forecasts",
    "v32_model_evaluations",
    "v33_training_iterations",
    "v33_model_versions",
    "v33_forecasts",
    "v33_model_evaluations",
)


def online_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst, pages=2048)


def table_digest(connection: sqlite3.Connection, table: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    cursor = connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
    for row in cursor:
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def frozen_hashes(database: Path) -> dict[str, dict[str, object]]:
    with sqlite3.connect(database) as connection:
        return {
            table: {"rows": count, "sha256": digest}
            for table in FROZEN_TABLES
            for count, digest in (table_digest(connection, table),)
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    if not source.is_file():
        raise SystemExit(f"source database does not exist: {source}")
    online_backup(source, target)
    before = frozen_hashes(target)
    engine = create_database_engine(target)
    migrated_version = run_migrations(engine)
    engine.dispose()
    after = frozen_hashes(target)

    with sqlite3.connect(target) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        v34_counts = {
            table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in V34_TABLES
        }

    report = {
        "source": str(source),
        "target": str(target),
        "source_size": source.stat().st_size,
        "target_size": target.stat().st_size,
        "schema_version": user_version,
        "migration_return": migrated_version,
        "integrity_check": integrity,
        "v34_tables_present": sorted(set(V34_TABLES) & tables),
        "v34_row_counts": v34_counts,
        "frozen_hashes_before": before,
        "frozen_hashes_after": after,
        "frozen_hashes_unchanged": before == after,
        "status": "PASS"
        if (
            migrated_version == SCHEMA_VERSION == user_version == 20
            and integrity == "ok"
            and set(V34_TABLES) <= tables
            and before == after
        )
        else "FAIL",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
