"""Verify that V3.3 did not mutate the frozen V3.2/legacy audit baseline.

The baseline values are intentionally embedded here as well as recorded in
``reports/V33_BASELINE_FREEZE.md``.  The script first verifies that the report
still declares those exact values, then opens SQLite read-only and recomputes
every table and directory fingerprint.

Canonical table fingerprint (the algorithm used by the freeze):

* read rows in SQLite ``rowid`` order;
* encode each row as compact UTF-8 JSON with keys sorted;
* append one LF byte after every row;
* SHA-256 the resulting byte stream.

Canonical directory fingerprint:

* sort files by their POSIX-style path relative to the audited directory;
* for each file, append the UTF-8 relative path and the 32 raw bytes of that
  file's SHA-256 digest, without separators;
* SHA-256 the resulting byte stream.

The default mode only reads and prints a Markdown report.  Pass ``--report``
to persist the same report at an explicit path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Iterable, Mapping
from urllib.parse import quote


TABLE_BASELINE: Mapping[str, tuple[int, str]] = {
    "v32_analysis_runs": (
        6,
        "cea6b69cfdd284de73de77b68cd09733737c7e684868925cbf44f23458639aa1",
    ),
    "v32_app_state": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "v32_daily_corrections": (
        2,
        "816209ad1feae2d19244b65fa6422b5b620dd776b809030329ed1ce7eb9b2d77",
    ),
    "v32_data_snapshots": (
        1034,
        "5e52fcbe762f0649c1e0108f449379bb277805be44d642dd91bb701e484a1c03",
    ),
    "v32_model_evaluations": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "v32_model_versions": (
        92,
        "ce80cab2ed4a006e31c5a9d0c8fdf1fb2270c2270c58762ce290780d99e50733",
    ),
    "v32_optimizer_states": (
        2,
        "3f76700c3373b761e39d86a32b18098395919086a02555bd5e15dab18c5f284d",
    ),
    "v32_training_checkpoints": (
        3,
        "6b8e8a23f2dbd033c52f0ff37ff5179fcce7a3e35d724bb164734f7a149d67e9",
    ),
    "v32_training_iterations": (
        1034,
        "7ce34955ab218c1f1b77edbb342ebe22e17111966177545c78990904b58adbc7",
    ),
    "v32_training_runs": (
        3,
        "b3f8505f22c8adad2c6bd6138aaf1ae59185e001aa672172ddcded79ad33d09d",
    ),
    "v32_weekly_forecasts": (
        2,
        "2d8d457697f5328b9bed2c71462bc37b6bbf64f77527a0dbeb7be3887feeba59",
    ),
}

TREE_BASELINE: Mapping[str, tuple[int, str]] = {
    "data/model_iterations/399006": (
        2,
        "8c226d5f229f90edc075b383ecc617f54460d7cff6d8b45b7debd7a88380e9e0",
    ),
    "data/model_iterations/NDX": (
        2,
        "4f3d87aeafa684ba2a8bb8fe97f4465d2e69b9ce02a22638e387d2f20595476d",
    ),
    "data/weekly_analysis_v2": (
        1057,
        "aee33cd9670fb26114d7a44436d1f0caf128b9326addcfaa38bbe96801d56a1b",
    ),
}


@dataclass(frozen=True, slots=True)
class AuditResult:
    name: str
    expected_count: int
    actual_count: int
    expected_sha256: str
    actual_sha256: str

    @property
    def passed(self) -> bool:
        return (
            self.actual_count == self.expected_count
            and self.actual_sha256 == self.expected_sha256
        )


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sqlite_readonly_uri(path: Path) -> str:
    absolute = path.resolve().as_posix()
    return f"file:{quote(absolute, safe='/:')}?mode=ro"


def table_fingerprints(database: Path) -> tuple[AuditResult, ...]:
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    connection = sqlite3.connect(_sqlite_readonly_uri(database), uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        existing = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = sorted(set(TABLE_BASELINE) - existing)
        if missing:
            raise RuntimeError(f"Frozen V3.2 tables are missing: {missing}")

        results: list[AuditResult] = []
        for table, (expected_rows, expected_hash) in TABLE_BASELINE.items():
            digest = hashlib.sha256()
            actual_rows = 0
            statement = (
                f"SELECT * FROM {_quoted_identifier(table)} ORDER BY rowid"
            )
            for row in connection.execute(statement):
                canonical = json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                digest.update(canonical.encode("utf-8"))
                digest.update(b"\n")
                actual_rows += 1
            results.append(
                AuditResult(
                    name=table,
                    expected_count=expected_rows,
                    actual_count=actual_rows,
                    expected_sha256=expected_hash,
                    actual_sha256=digest.hexdigest(),
                )
            )
        return tuple(results)
    finally:
        connection.close()


def tree_fingerprints(project_root: Path) -> tuple[AuditResult, ...]:
    results: list[AuditResult] = []
    for relative_root, (expected_files, expected_hash) in TREE_BASELINE.items():
        root = project_root / Path(relative_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Frozen audit directory not found: {root}")
        files = sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        digest = hashlib.sha256()
        for path in files:
            relative_path = path.relative_to(root).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        results.append(
            AuditResult(
                name=relative_root,
                expected_count=expected_files,
                actual_count=len(files),
                expected_sha256=expected_hash,
                actual_sha256=digest.hexdigest(),
            )
        )
    return tuple(results)


_FREEZE_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`\s*\|\s*$"
)


def verify_freeze_declaration(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"freeze declaration missing: {path}"
    declared: dict[str, tuple[int, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _FREEZE_ROW.match(line)
        if match:
            declared[match.group(1)] = (int(match.group(2)), match.group(3))
    expected = {**TABLE_BASELINE, **TREE_BASELINE}
    if declared != expected:
        missing = sorted(set(expected) - set(declared))
        extra = sorted(set(declared) - set(expected))
        changed = sorted(
            name
            for name in set(expected) & set(declared)
            if expected[name] != declared[name]
        )
        return (
            False,
            f"freeze declaration differs (missing={missing}, extra={extra}, changed={changed})",
        )
    return True, "freeze declaration matches embedded independent constants"


def render_report(
    *,
    database: Path,
    freeze_path: Path,
    declaration_ok: bool,
    declaration_message: str,
    table_results: Iterable[AuditResult],
    tree_results: Iterable[AuditResult],
) -> tuple[str, bool]:
    tables = tuple(table_results)
    trees = tuple(tree_results)
    passed = declaration_ok and all(result.passed for result in (*tables, *trees))
    lines = [
        "# V3.3 read-only baseline audit",
        "",
        f"- Audited at (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Database: `{database}` (opened read-only)",
        f"- Freeze declaration: `{freeze_path}`",
        f"- Declaration check: `{'PASS' if declaration_ok else 'FAIL'}`: {declaration_message}",
        f"- Overall result: `{'PASS' if passed else 'FAIL'}`",
        "",
        "## Frozen V3.2 tables",
        "",
        "| Table | Expected rows | Actual rows | Expected SHA-256 | Actual SHA-256 | Result |",
        "|---|---:|---:|---|---|---|",
    ]
    for result in tables:
        lines.append(
            f"| `{result.name}` | {result.expected_count} | {result.actual_count} | "
            f"`{result.expected_sha256}` | `{result.actual_sha256}` | "
            f"`{'PASS' if result.passed else 'FAIL'}` |"
        )
    lines.extend(
        [
            "",
            "## Frozen legacy audit directories",
            "",
            "| Directory | Expected files | Actual files | Expected SHA-256 | Actual SHA-256 | Result |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for result in trees:
        lines.append(
            f"| `{result.name}` | {result.expected_count} | {result.actual_count} | "
            f"`{result.expected_sha256}` | `{result.actual_sha256}` | "
            f"`{'PASS' if result.passed else 'FAIL'}` |"
        )
    lines.extend(
        [
            "",
            "The audit intentionally ignores V3.3 tables and other mutable application data. "
            "A larger SQLite file is therefore not evidence of a V3.2 mutation; the canonical "
            "row streams above are the authoritative check.",
            "",
        ]
    )
    return "\n".join(lines), passed


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify the frozen V3.2 tables and legacy audit directories."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional explicit output path for the generated Markdown report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    project_root = args.project_root.resolve()
    database = (args.database or project_root / "data" / "investment_lab.db").resolve()
    freeze_path = (
        args.freeze or project_root / "reports" / "V33_BASELINE_FREEZE.md"
    ).resolve()
    declaration_ok, declaration_message = verify_freeze_declaration(freeze_path)
    table_results = table_fingerprints(database)
    tree_results = tree_fingerprints(project_root)
    report, passed = render_report(
        database=database,
        freeze_path=freeze_path,
        declaration_ok=declaration_ok,
        declaration_message=declaration_message,
        table_results=table_results,
        tree_results=tree_results,
    )
    print(report)
    if args.report is not None:
        output = args.report.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
