from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "import-ai-assistant-market-data.py"
SPEC = importlib.util.spec_from_file_location("ai_assistant_import", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _fixture(tmp_path: Path, *, conflict: bool = False) -> tuple[Path, Path]:
    source = tmp_path / "AI Investment Assistant"
    data = source / "数据"
    config = source / "config"
    data.mkdir(parents=True)
    config.mkdir()
    instruments = [{"code": code, "price_tick": 0.001} for code in module.ETF_CODES]
    (config / "instruments.json").write_text(
        json.dumps({"schema_version": "2.0.0", "instruments": instruments}), encoding="utf-8"
    )
    entries = []
    for index, code in enumerate(module.ETF_CODES):
        path = data / f"{code}_测试ETF_日线.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(module.REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerow(
                {
                    "code": code,
                    "date": "2026-08-07",
                    "raw_open": "1.000",
                    "raw_high": "1.020",
                    "raw_low": "0.990",
                    "raw_close": "1.010",
                    "adj_close": "1.0100000000001",
                    "volume": "1000",
                    "amount": "" if index == 0 else "1010",
                    "run_id": "RUN1",
                }
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": digest,
                "rows": 1,
                "start": "2026-08-07",
                "end": "2026-08-07",
                "run_id": "RUN1",
            }
        )
    manifest = {
        "schema_version": "2.0.0",
        "run_id": "RUN1",
        "market_status": "closed",
        "quality_status": "passed",
        "as_of": "2026-08-07",
        "latest_dates": {code: "2026-08-07" for code in module.ETF_CODES},
        "files": entries,
    }
    (source / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    database = tmp_path / "investment_lab.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE instruments (id INTEGER PRIMARY KEY, code TEXT UNIQUE NOT NULL);
            CREATE TABLE market_prices (
                id INTEGER PRIMARY KEY, instrument_id INTEGER NOT NULL, trade_date TEXT NOT NULL,
                timeframe TEXT NOT NULL, open_price NUMERIC, high_price NUMERIC, low_price NUMERIC,
                close_price NUMERIC NOT NULL, adjusted_close_price NUMERIC, volume NUMERIC,
                volume_multiplier INTEGER NOT NULL DEFAULT 1, turnover NUMERIC, source TEXT,
                volume_source TEXT, created_at TEXT, updated_at TEXT,
                UNIQUE(instrument_id, trade_date, timeframe)
            );
            """
        )
        for index, code in enumerate(module.ETF_CODES, start=1):
            connection.execute("INSERT INTO instruments(id,code) VALUES (?,?)", (index, code))
            open_price = 110000000 if conflict and index == 1 else 100000000
            connection.execute(
                """INSERT INTO market_prices
                (instrument_id,trade_date,timeframe,open_price,high_price,low_price,close_price,
                 adjusted_close_price,volume,turnover,source,volume_source,created_at,updated_at)
                VALUES (?,?, 'daily', ?,102000000,99000000,101000000,NULL,100000000000,99900000000,
                        'AKSHARE','AKSHARE','x','x')""",
                (index, "2026-08-07", open_price),
            )
    return source, database


def test_v2_sync_preserves_blank_turnover_and_is_idempotent(tmp_path: Path) -> None:
    source, database = _fixture(tmp_path)
    bundle = module.load_bundle(source)
    result = module.sync_database(bundle, database, rebuild_derived=lambda *_args: None)
    assert result["status"] == "success"
    with sqlite3.connect(database) as connection:
        first = connection.execute(
            "SELECT adjusted_close_price,turnover FROM market_prices WHERE instrument_id=1"
        ).fetchone()
    assert first == (101000000, 99900000000)
    second = module.sync_database(bundle, database, rebuild_derived=lambda *_args: None)
    assert second["status"] == "already_synced"
    assert second["database_changes"] == 0


def test_conflict_over_one_tick_blocks_entire_bundle(tmp_path: Path) -> None:
    source, database = _fixture(tmp_path, conflict=True)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="price conflicts exceed one tick"):
        module.preflight(module.load_bundle(source), database)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_derived_failure_restores_original_database(tmp_path: Path) -> None:
    source, database = _fixture(tmp_path)
    connection = sqlite3.connect(database)
    try:
        before = connection.execute(
            "SELECT instrument_id,trade_date,open_price,adjusted_close_price,turnover "
            "FROM market_prices ORDER BY instrument_id,trade_date"
        ).fetchall()
    finally:
        connection.close()

    def fail(*_args: object) -> None:
        raise RuntimeError("injected derived failure")

    with pytest.raises(RuntimeError, match="injected derived failure"):
        module.sync_database(module.load_bundle(source), database, rebuild_derived=fail)
    connection = sqlite3.connect(database)
    try:
        after = connection.execute(
            "SELECT instrument_id,trade_date,open_price,adjusted_close_price,turnover "
            "FROM market_prices ORDER BY instrument_id,trade_date"
        ).fetchall()
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
    finally:
        connection.close()
    assert after == before
    assert not module._audit_path(database).exists()
