"""Small, versioned SQLite migration runner for the local database."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING, Mapping
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    ForeignKeyConstraint,
    UniqueConstraint,
    inspect,
    text,
)

from ..models.models import Base, DataUpdateLog, FixedPointDecimal
from .fixed_point import _rounded_integer_quotient

if TYPE_CHECKING:
    from sqlalchemy import Connection, Engine


SCHEMA_VERSION = 22

V2_TABLE_NAMES = (
    "v2_market_data_cache",
    "v2_period_bars",
    "v2_feature_snapshots",
    "v2_week_samples",
    "v2_analysis_iterations",
    "v2_iteration_labels",
    "v2_model_versions",
    "v2_analysis_tasks",
    "v2_position_events",
    "v2_position_snapshots",
    "v2_advice_history",
)


class SchemaVersionError(RuntimeError):
    """Raised when a database needs a newer application version."""


def _quoted(identifier: str) -> str:
    return f'"{identifier.replace("\"", "\"\"")}"'


def _normalize_legacy_value(value: object, scale: int) -> tuple[Decimal, Decimal]:
    """Normalize pre-fixed-point values deterministically before integer storage."""
    original = Decimal(str(value))
    normalized = original.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_EVEN)
    return original, normalized


def _record_normalization(
    connection: Connection,
    table_name: str,
    column_name: str,
    rowid: int,
    original: Decimal,
    normalized: Decimal,
) -> None:
    connection.execute(
        DataUpdateLog.__table__.insert().values(
            dataset="schema_migration",
            status="normalized",
            records_received=1,
            records_written=1,
            message=(
                "Normalized legacy fixed-point value: "
                f"table={table_name} column={column_name} row={rowid} "
                f"value={original} normalized={normalized}"
            ),
        )
    )


def _upgrade_to_version_one(connection: Connection) -> None:
    """Create the schema and convert pre-version fixed-point values if needed."""
    Base.metadata.create_all(connection)

    for table in Base.metadata.sorted_tables:
        declared_types = {
            row[1]: (row[2] or "").upper()
            for row in connection.exec_driver_sql(f"PRAGMA table_info({_quoted(table.name)})")
        }
        for column in table.columns:
            if not isinstance(column.type, FixedPointDecimal):
                continue
            declared_type = declared_types.get(column.name)
            if declared_type is None or declared_type.startswith("INTEGER"):
                continue
            values = connection.execute(
                text(
                    f"SELECT rowid, {_quoted(column.name)} AS value "
                    f"FROM {_quoted(table.name)} WHERE {_quoted(column.name)} IS NOT NULL"
                )
            ).all()
            for rowid, value in values:
                original, normalized = _normalize_legacy_value(value, column.type.scale)
                scaled_value = column.type.to_storage(normalized)
                connection.exec_driver_sql(
                    f"UPDATE {_quoted(table.name)} SET {_quoted(column.name)} = ? WHERE rowid = ?",
                    (scaled_value, rowid),
                )
                if normalized != original:
                    _record_normalization(
                        connection,
                        table.name,
                        column.name,
                        rowid,
                        original,
                        normalized,
                    )


def _upgrade_to_version_two(connection: Connection) -> None:
    """Add auditable provider and precise row-outcome fields to update logs."""
    declared_columns = {
        row[1]
        for row in connection.exec_driver_sql("PRAGMA table_info(data_update_logs)")
    }
    additions = (
        ("source", "VARCHAR(64)"),
        ("records_added", "INTEGER NOT NULL DEFAULT 0"),
        ("records_updated", "INTEGER NOT NULL DEFAULT 0"),
        ("records_skipped", "INTEGER NOT NULL DEFAULT 0"),
    )
    for name, sql_type in additions:
        if name not in declared_columns:
            connection.exec_driver_sql(f"ALTER TABLE data_update_logs ADD COLUMN {_quoted(name)} {sql_type}")


def _indicator_records_have_period_unique_constraint(connection: Connection) -> bool:
    """Return whether the current SQLite table is unique by one price period."""
    for index in connection.exec_driver_sql("PRAGMA index_list(indicator_records)"):
        if not index[2]:
            continue
        index_name = index[1]
        columns = tuple(
            row[2]
            for row in connection.exec_driver_sql(f"PRAGMA index_info({_quoted(index_name)})")
        )
        if columns == ("instrument_id", "indicator_date", "timeframe"):
            return True
    return False


def _legacy_indicator_payload(rows: list[tuple[object, ...]]) -> str:
    """Retain every old indicator row when collapsing a legacy period duplicate."""
    legacy_records: list[dict[str, object]] = []
    for row in rows:
        raw_values = row[6]
        try:
            values = json.loads(raw_values) if isinstance(raw_values, str) else raw_values
        except json.JSONDecodeError:
            values = {"unparseable_legacy_json": str(raw_values)}
        raw_value = row[5]
        legacy_records.append(
            {
                "id": row[0],
                "indicator_name": row[4],
                "indicator_value": (
                    None
                    if raw_value is None
                    else format(Decimal(raw_value) / Decimal(10**8), "f")
                ),
                "indicator_values": values,
                "created_at": str(row[7]),
                "updated_at": str(row[8]),
            }
        )
    return json.dumps({"legacy_records": legacy_records}, ensure_ascii=False, sort_keys=True)


def _upgrade_to_version_three(connection: Connection) -> None:
    """Guarantee one aggregate indicator row per instrument/date/timeframe.

    SQLite cannot add a UNIQUE constraint in place. The rebuild uses the
    lowest legacy ``id`` as the stable physical row for a duplicated period
    and places every legacy value in that row's lossless JSON payload.
    """
    if _indicator_records_have_period_unique_constraint(connection):
        return
    rows = connection.exec_driver_sql(
        """
        SELECT id, instrument_id, indicator_date, timeframe, indicator_name,
               indicator_value, indicator_values, created_at, updated_at
        FROM indicator_records
        ORDER BY instrument_id, indicator_date, timeframe, id
        """
    ).all()
    grouped: dict[tuple[object, object, object], list[tuple[object, ...]]] = {}
    for row in rows:
        grouped.setdefault((row[1], row[2], row[3]), []).append(tuple(row))
    connection.exec_driver_sql(
        """
        CREATE TABLE indicator_records_v3 (
            id INTEGER NOT NULL PRIMARY KEY,
            instrument_id INTEGER NOT NULL REFERENCES instruments(id),
            indicator_date DATE NOT NULL,
            timeframe VARCHAR(16) NOT NULL,
            indicator_name VARCHAR(64) NOT NULL,
            indicator_value INTEGER,
            indicator_values JSON NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT uq_indicator_record UNIQUE (instrument_id, indicator_date, timeframe)
        )
        """
    )
    for period_rows in grouped.values():
        first = period_rows[0]
        if len(period_rows) == 1:
            row = first
            values = row[6]
            name = row[4]
            value = row[5]
        else:
            row = first
            values = _legacy_indicator_payload(period_rows)
            name = "technical_indicators"
            value = None
        connection.exec_driver_sql(
            """
            INSERT INTO indicator_records_v3
                (id, instrument_id, indicator_date, timeframe, indicator_name,
                 indicator_value, indicator_values, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (row[0], row[1], row[2], row[3], name, value, values, row[7], row[8]),
        )
    connection.exec_driver_sql("DROP TABLE indicator_records")
    connection.exec_driver_sql("ALTER TABLE indicator_records_v3 RENAME TO indicator_records")
    connection.exec_driver_sql(
        "CREATE INDEX ix_indicator_records_instrument_id ON indicator_records (instrument_id)"
    )


def _strategy_signals_have_snapshot_identity(connection: Connection) -> bool:
    """Return whether strategy signals have the v4 date/version identity."""
    columns = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(strategy_signals)")
    }
    if not {"as_of_date", "strategy_version"}.issubset(columns):
        return False
    for index in connection.exec_driver_sql("PRAGMA index_list(strategy_signals)"):
        if not index[2]:
            continue
        index_name = index[1]
        names = tuple(
            row[2]
            for row in connection.exec_driver_sql(f"PRAGMA index_info({_quoted(index_name)})")
        )
        if names in (
            ("strategy_id", "instrument_id", "as_of_date", "strategy_version"),
            (
                "strategy_id",
                "instrument_id",
                "as_of_date",
                "strategy_version",
                "strategy_config_hash",
            ),
            (
                "strategy_id",
                "instrument_id",
                "as_of_date",
                "strategy_version",
                "strategy_config_hash",
                "source_data_hash",
            ),
        ):
            return True
    return False


def _upgrade_to_version_four(connection: Connection) -> None:
    """Make strategy snapshots immutable by strategy/instrument/date/version.

    The rebuild also maps the legacy BUY/NONE recommendation labels into the
    research-only whitelist.  Duplicate historic snapshots are retained by
    assigning a stable ``-legacy-<id>`` version suffix rather than dropping
    data during the uniqueness migration.
    """
    if _strategy_signals_have_snapshot_identity(connection):
        return
    rows = connection.exec_driver_sql(
        """
        SELECT s.id, s.strategy_id, s.instrument_id, s.signal_at,
               s.recommendation_type, s.score, s.target_allocation,
               s.confidence, s.rationale, s.signal_data, s.created_at,
               s.updated_at, d.version AS definition_version
        FROM strategy_signals AS s
        LEFT JOIN strategy_definitions AS d ON d.id = s.strategy_id
        ORDER BY s.id
        """
    ).mappings().all()
    connection.exec_driver_sql(
        """
        CREATE TABLE strategy_signals_v4 (
            id INTEGER NOT NULL PRIMARY KEY,
            strategy_id INTEGER NOT NULL REFERENCES strategy_definitions(id),
            instrument_id INTEGER NOT NULL REFERENCES instruments(id),
            signal_at DATETIME NOT NULL,
            as_of_date DATE NOT NULL,
            strategy_version VARCHAR(32) NOT NULL,
            recommendation_type VARCHAR(32) NOT NULL,
            score INTEGER,
            target_allocation INTEGER,
            confidence INTEGER,
            rationale TEXT,
            signal_data JSON NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT ck_strategy_signal_recommendation CHECK
                (recommendation_type IN ('INCREASE', 'NORMAL', 'REDUCE', 'PAUSE', 'HOLD', 'SELL_PARTIAL')),
            CONSTRAINT uq_strategy_signal_snapshot UNIQUE
                (strategy_id, instrument_id, as_of_date, strategy_version)
        )
        """
    )
    used: set[tuple[object, object, str, str]] = set()
    labels = {"INCREASE", "NORMAL", "REDUCE", "PAUSE", "HOLD", "SELL_PARTIAL"}
    for row in rows:
        as_of = str(row["signal_at"] or "1970-01-01")[:10]
        version = str(row["definition_version"] or "1.0")
        key = (row["strategy_id"], row["instrument_id"], as_of, version)
        if key in used:
            base = version[:20]
            version = f"{base}-legacy-{row['id']}"
        used.add((row["strategy_id"], row["instrument_id"], as_of, version))
        raw_recommendation = str(row["recommendation_type"])
        recommendation = {"BUY": "INCREASE", "NONE": "HOLD"}.get(raw_recommendation, raw_recommendation)
        if recommendation not in labels:
            recommendation = "HOLD"
        connection.exec_driver_sql(
            """
            INSERT INTO strategy_signals_v4
                (id, strategy_id, instrument_id, signal_at, as_of_date, strategy_version,
                 recommendation_type, score, target_allocation, confidence, rationale,
                 signal_data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"], row["strategy_id"], row["instrument_id"], row["signal_at"], as_of,
                version, recommendation, row["score"], row["target_allocation"], row["confidence"],
                row["rationale"], row["signal_data"], row["created_at"], row["updated_at"],
            ),
        )
    connection.exec_driver_sql("DROP TABLE strategy_signals")
    connection.exec_driver_sql("ALTER TABLE strategy_signals_v4 RENAME TO strategy_signals")
    connection.exec_driver_sql(
        "CREATE INDEX ix_strategy_signals_strategy_id ON strategy_signals (strategy_id)"
    )
    connection.exec_driver_sql(
        "CREATE INDEX ix_strategy_signals_instrument_id ON strategy_signals (instrument_id)"
    )
    connection.exec_driver_sql(
        "CREATE INDEX ix_strategy_signals_as_of_date ON strategy_signals (as_of_date)"
    )


def _has_unique_index(connection: Connection, table: str, columns: tuple[str, ...]) -> bool:
    for index in connection.exec_driver_sql(f"PRAGMA index_list({_quoted(table)})"):
        if not index[2]:
            continue
        index_name = index[1]
        names = tuple(
            row[2] for row in connection.exec_driver_sql(f"PRAGMA index_info({_quoted(index_name)})")
        )
        if names == columns:
            return True
    return False


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _legacy_signal_config_snapshot(row: Mapping[str, object]) -> tuple[str, str, dict[str, object]]:
    raw_data = row["signal_data"]
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else dict(raw_data or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {"unparseable_legacy_signal_data": str(raw_data)}
    raw_json = data.get("strategy_config_json")
    raw_hash = data.get("strategy_config_hash")
    if isinstance(raw_json, str) and isinstance(raw_hash, str) and len(raw_hash) == 64:
        config_json = raw_json
        config_hash = raw_hash
    else:
        config = data.get("strategy_config")
        if not isinstance(config, dict):
            config = {
                "legacy_signal_id": row["id"],
                "strategy_version": row["strategy_version"],
            }
        config_json = _canonical_json(config)
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    data["strategy_config"] = json.loads(config_json)
    data["strategy_config_json"] = config_json
    data["strategy_config_hash"] = config_hash
    return config_json, config_hash, data


def _strategy_signals_have_hashed_identity(connection: Connection) -> bool:
    columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(strategy_signals)")}
    if "strategy_config_hash" not in columns:
        return False
    return any(
        _has_unique_index(connection, "strategy_signals", identity)
        for identity in (
            ("strategy_id", "instrument_id", "as_of_date", "strategy_version", "strategy_config_hash"),
            (
                "strategy_id",
                "instrument_id",
                "as_of_date",
                "strategy_version",
                "strategy_config_hash",
                "source_data_hash",
            ),
        )
    )


def _upgrade_strategy_signals_to_version_five(connection: Connection) -> None:
    if _strategy_signals_have_hashed_identity(connection):
        return
    rows = connection.exec_driver_sql(
        """
        SELECT id, strategy_id, instrument_id, signal_at, as_of_date, strategy_version,
               recommendation_type, score, target_allocation, confidence, rationale,
               signal_data, created_at, updated_at
        FROM strategy_signals ORDER BY id
        """
    ).mappings().all()
    connection.exec_driver_sql(
        """
        CREATE TABLE strategy_signals_v5 (
            id INTEGER NOT NULL PRIMARY KEY,
            strategy_id INTEGER NOT NULL REFERENCES strategy_definitions(id),
            instrument_id INTEGER NOT NULL REFERENCES instruments(id),
            signal_at DATETIME NOT NULL,
            as_of_date DATE NOT NULL,
            strategy_version VARCHAR(32) NOT NULL,
            strategy_config_hash VARCHAR(64) NOT NULL,
            recommendation_type VARCHAR(32) NOT NULL,
            score INTEGER,
            target_allocation INTEGER,
            confidence INTEGER,
            rationale TEXT,
            signal_data JSON NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT ck_strategy_signal_recommendation CHECK
                (recommendation_type IN ('INCREASE', 'NORMAL', 'REDUCE', 'PAUSE', 'HOLD', 'SELL_PARTIAL')),
            CONSTRAINT uq_strategy_signal_snapshot UNIQUE
                (strategy_id, instrument_id, as_of_date, strategy_version, strategy_config_hash)
        )
        """
    )
    for row in rows:
        _config_json, config_hash, data = _legacy_signal_config_snapshot(row)
        rationale = str(row["rationale"] or "")
        hash_reason = f"strategy_config_sha256={config_hash}"
        if hash_reason not in rationale:
            rationale = f"{rationale}\n{hash_reason}".strip()
        connection.exec_driver_sql(
            """
            INSERT INTO strategy_signals_v5
                (id, strategy_id, instrument_id, signal_at, as_of_date, strategy_version,
                 strategy_config_hash, recommendation_type, score, target_allocation,
                 confidence, rationale, signal_data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"], row["strategy_id"], row["instrument_id"], row["signal_at"],
                row["as_of_date"], row["strategy_version"], config_hash, row["recommendation_type"],
                row["score"], row["target_allocation"], row["confidence"], rationale,
                _canonical_json(data), row["created_at"], row["updated_at"],
            ),
        )
    connection.exec_driver_sql("DROP TABLE strategy_signals")
    connection.exec_driver_sql("ALTER TABLE strategy_signals_v5 RENAME TO strategy_signals")
    connection.exec_driver_sql("CREATE INDEX ix_strategy_signals_strategy_id ON strategy_signals (strategy_id)")
    connection.exec_driver_sql("CREATE INDEX ix_strategy_signals_instrument_id ON strategy_signals (instrument_id)")
    connection.exec_driver_sql("CREATE INDEX ix_strategy_signals_as_of_date ON strategy_signals (as_of_date)")


def _research_reports_have_strategy_identity(connection: Connection) -> bool:
    columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(research_reports)")}
    if not {"strategy_id", "strategy_version", "strategy_config_hash"}.issubset(columns):
        return False
    return any(
        _has_unique_index(connection, "research_reports", identity)
        for identity in (
            (
                "instrument_id",
                "report_date",
                "report_type",
                "strategy_id",
                "strategy_version",
                "strategy_config_hash",
            ),
            (
                "instrument_id",
                "report_date",
                "report_type",
                "strategy_id",
                "strategy_version",
                "strategy_config_hash",
                "source_data_hash",
            ),
        )
    )


def _upgrade_research_reports_to_version_five(connection: Connection) -> None:
    if _research_reports_have_strategy_identity(connection):
        return
    rows = connection.exec_driver_sql(
        """
        SELECT id, instrument_id, title, report_date, report_type, content, file_path,
               status, created_at, updated_at
        FROM research_reports ORDER BY id
        """
    ).mappings().all()
    connection.exec_driver_sql(
        """
        CREATE TABLE research_reports_v5 (
            id INTEGER NOT NULL PRIMARY KEY,
            instrument_id INTEGER REFERENCES instruments(id),
            strategy_id INTEGER REFERENCES strategy_definitions(id),
            strategy_version VARCHAR(32),
            strategy_config_hash VARCHAR(64),
            title VARCHAR(256) NOT NULL,
            report_date DATE NOT NULL,
            report_type VARCHAR(64) NOT NULL,
            content TEXT,
            file_path VARCHAR(512),
            status VARCHAR(32) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT uq_research_report_strategy_snapshot UNIQUE
                (instrument_id, report_date, report_type, strategy_id, strategy_version, strategy_config_hash)
        )
        """
    )
    for row in rows:
        signal = connection.exec_driver_sql(
            """
            SELECT strategy_id, strategy_version, strategy_config_hash
            FROM strategy_signals
            WHERE instrument_id = ? AND as_of_date = ?
            ORDER BY id LIMIT 1
            """,
            (row["instrument_id"], row["report_date"]),
        ).mappings().first()
        connection.exec_driver_sql(
            """
            INSERT INTO research_reports_v5
                (id, instrument_id, strategy_id, strategy_version, strategy_config_hash,
                 title, report_date, report_type, content, file_path, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"], row["instrument_id"],
                signal["strategy_id"] if signal else None,
                signal["strategy_version"] if signal else None,
                signal["strategy_config_hash"] if signal else None,
                row["title"], row["report_date"], row["report_type"], row["content"],
                row["file_path"], row["status"], row["created_at"], row["updated_at"],
            ),
        )
    connection.exec_driver_sql("DROP TABLE research_reports")
    connection.exec_driver_sql("ALTER TABLE research_reports_v5 RENAME TO research_reports")
    connection.exec_driver_sql("CREATE INDEX ix_research_reports_instrument_id ON research_reports (instrument_id)")
    connection.exec_driver_sql("CREATE INDEX ix_research_reports_strategy_id ON research_reports (strategy_id)")


def _upgrade_to_version_five(connection: Connection) -> None:
    """Snapshot configuration hashes and report strategy identities without data loss."""
    _upgrade_strategy_signals_to_version_five(connection)
    _upgrade_research_reports_to_version_five(connection)


def _legacy_signal_source_snapshot(row: Mapping[str, object]) -> tuple[str, str, dict[str, object]]:
    raw_data = row["signal_data"]
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else dict(raw_data or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {"unparseable_legacy_signal_data": str(raw_data)}
    raw_json = data.get("source_snapshot_json")
    raw_hash = data.get("source_data_hash")
    if isinstance(raw_json, str) and isinstance(raw_hash, str) and len(raw_hash) == 64:
        source_json = raw_json
        source_hash = raw_hash
    else:
        source = data.get("source_snapshot")
        if not isinstance(source, dict):
            source = {
                "legacy_signal_id": row["id"],
                "as_of_date": str(row["as_of_date"]),
                "data_cutoff": data.get("data_cutoff"),
                "close_price": data.get("close_price"),
                "volume": data.get("volume"),
                "valuation": data.get("valuation"),
                "indicators": data.get("indicators"),
                "observations": data.get("observations"),
            }
        source_json = _canonical_json(source)
        source_hash = hashlib.sha256(source_json.encode("utf-8")).hexdigest()
    data["source_snapshot"] = json.loads(source_json)
    data["source_snapshot_json"] = source_json
    data["source_data_hash"] = source_hash
    return source_json, source_hash, data


def _strategy_signals_have_source_identity(connection: Connection) -> bool:
    columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(strategy_signals)")}
    return "source_data_hash" in columns and _has_unique_index(
        connection,
        "strategy_signals",
        (
            "strategy_id",
            "instrument_id",
            "as_of_date",
            "strategy_version",
            "strategy_config_hash",
            "source_data_hash",
        ),
    )


def _upgrade_strategy_signals_to_version_six(connection: Connection) -> None:
    if _strategy_signals_have_source_identity(connection):
        return
    rows = connection.exec_driver_sql(
        """
        SELECT id, strategy_id, instrument_id, signal_at, as_of_date, strategy_version,
               strategy_config_hash, recommendation_type, score, target_allocation,
               confidence, rationale, signal_data, created_at, updated_at
        FROM strategy_signals ORDER BY id
        """
    ).mappings().all()
    connection.exec_driver_sql(
        """
        CREATE TABLE strategy_signals_v6 (
            id INTEGER NOT NULL PRIMARY KEY,
            strategy_id INTEGER NOT NULL REFERENCES strategy_definitions(id),
            instrument_id INTEGER NOT NULL REFERENCES instruments(id),
            signal_at DATETIME NOT NULL,
            as_of_date DATE NOT NULL,
            strategy_version VARCHAR(32) NOT NULL,
            strategy_config_hash VARCHAR(64) NOT NULL,
            source_data_hash VARCHAR(64) NOT NULL,
            recommendation_type VARCHAR(32) NOT NULL,
            score INTEGER,
            target_allocation INTEGER,
            confidence INTEGER,
            rationale TEXT,
            signal_data JSON NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT ck_strategy_signal_recommendation CHECK
                (recommendation_type IN ('INCREASE', 'NORMAL', 'REDUCE', 'PAUSE', 'HOLD', 'SELL_PARTIAL')),
            CONSTRAINT uq_strategy_signal_snapshot UNIQUE
                (strategy_id, instrument_id, as_of_date, strategy_version, strategy_config_hash, source_data_hash)
        )
        """
    )
    for row in rows:
        _source_json, source_hash, data = _legacy_signal_source_snapshot(row)
        rationale = str(row["rationale"] or "")
        hash_reason = f"source_data_sha256={source_hash}"
        if hash_reason not in rationale:
            rationale = f"{rationale}\n{hash_reason}".strip()
        connection.exec_driver_sql(
            """
            INSERT INTO strategy_signals_v6
                (id, strategy_id, instrument_id, signal_at, as_of_date, strategy_version,
                 strategy_config_hash, source_data_hash, recommendation_type, score,
                 target_allocation, confidence, rationale, signal_data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"], row["strategy_id"], row["instrument_id"], row["signal_at"],
                row["as_of_date"], row["strategy_version"], row["strategy_config_hash"],
                source_hash, row["recommendation_type"], row["score"], row["target_allocation"],
                row["confidence"], rationale, _canonical_json(data), row["created_at"], row["updated_at"],
            ),
        )
    connection.exec_driver_sql("DROP TABLE strategy_signals")
    connection.exec_driver_sql("ALTER TABLE strategy_signals_v6 RENAME TO strategy_signals")
    connection.exec_driver_sql("CREATE INDEX ix_strategy_signals_strategy_id ON strategy_signals (strategy_id)")
    connection.exec_driver_sql("CREATE INDEX ix_strategy_signals_instrument_id ON strategy_signals (instrument_id)")
    connection.exec_driver_sql("CREATE INDEX ix_strategy_signals_as_of_date ON strategy_signals (as_of_date)")


def _research_reports_have_source_identity(connection: Connection) -> bool:
    columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(research_reports)")}
    return "source_data_hash" in columns and _has_unique_index(
        connection,
        "research_reports",
        (
            "instrument_id",
            "report_date",
            "report_type",
            "strategy_id",
            "strategy_version",
            "strategy_config_hash",
            "source_data_hash",
        ),
    )


def _upgrade_research_reports_to_version_six(connection: Connection) -> None:
    if _research_reports_have_source_identity(connection):
        return
    rows = connection.exec_driver_sql(
        """
        SELECT id, instrument_id, strategy_id, strategy_version, strategy_config_hash,
               title, report_date, report_type, content, file_path, status, created_at, updated_at
        FROM research_reports ORDER BY id
        """
    ).mappings().all()
    connection.exec_driver_sql(
        """
        CREATE TABLE research_reports_v6 (
            id INTEGER NOT NULL PRIMARY KEY,
            instrument_id INTEGER REFERENCES instruments(id),
            strategy_id INTEGER REFERENCES strategy_definitions(id),
            strategy_version VARCHAR(32),
            strategy_config_hash VARCHAR(64),
            source_data_hash VARCHAR(64),
            title VARCHAR(256) NOT NULL,
            report_date DATE NOT NULL,
            report_type VARCHAR(64) NOT NULL,
            content TEXT,
            file_path VARCHAR(512),
            status VARCHAR(32) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT uq_research_report_strategy_snapshot UNIQUE
                (instrument_id, report_date, report_type, strategy_id, strategy_version,
                 strategy_config_hash, source_data_hash)
        )
        """
    )
    for row in rows:
        source_hash = None
        if row["strategy_id"] is not None and row["strategy_config_hash"] is not None:
            source_hash = connection.exec_driver_sql(
                """
                SELECT source_data_hash FROM strategy_signals
                WHERE strategy_id = ? AND instrument_id = ? AND as_of_date = ?
                      AND strategy_version = ? AND strategy_config_hash = ?
                ORDER BY id LIMIT 1
                """,
                (
                    row["strategy_id"], row["instrument_id"], row["report_date"],
                    row["strategy_version"], row["strategy_config_hash"],
                ),
            ).scalar_one_or_none()
        if source_hash is None:
            source_hash = hashlib.sha256(
                _canonical_json({"legacy_report_id": row["id"]}).encode("utf-8")
            ).hexdigest()
        connection.exec_driver_sql(
            """
            INSERT INTO research_reports_v6
                (id, instrument_id, strategy_id, strategy_version, strategy_config_hash,
                 source_data_hash, title, report_date, report_type, content, file_path,
                 status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"], row["instrument_id"], row["strategy_id"], row["strategy_version"],
                row["strategy_config_hash"], source_hash, row["title"], row["report_date"],
                row["report_type"], row["content"], row["file_path"], row["status"],
                row["created_at"], row["updated_at"],
            ),
        )
    connection.exec_driver_sql("DROP TABLE research_reports")
    connection.exec_driver_sql("ALTER TABLE research_reports_v6 RENAME TO research_reports")
    connection.exec_driver_sql("CREATE INDEX ix_research_reports_instrument_id ON research_reports (instrument_id)")
    connection.exec_driver_sql("CREATE INDEX ix_research_reports_strategy_id ON research_reports (strategy_id)")


def _upgrade_to_version_six(connection: Connection) -> None:
    """Append immutable source-data revisions to strategy signals and reports."""
    _upgrade_strategy_signals_to_version_six(connection)
    _upgrade_research_reports_to_version_six(connection)


def _add_column_if_missing(connection: Connection, table: str, name: str, declaration: str) -> bool:
    columns = {row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({_quoted(table)})")}
    if name not in columns:
        connection.exec_driver_sql(
            f"ALTER TABLE {_quoted(table)} ADD COLUMN {_quoted(name)} {declaration}"
        )
        return True
    return False


def _upgrade_to_version_seven(connection: Connection) -> None:
    """Add auditable simulation inputs and return fields without rewriting history."""
    transaction_columns = (
        ("plan_id", "INTEGER REFERENCES investment_plans(id)"),
        ("execution_date", "DATE"),
        ("source_identity", "VARCHAR(128) NOT NULL DEFAULT 'legacy'"),
        ("requested_amount", "INTEGER NOT NULL DEFAULT 0"),
        ("executed_amount", "INTEGER NOT NULL DEFAULT 0"),
        ("commission", "INTEGER NOT NULL DEFAULT 0"),
        ("stamp_duty", "INTEGER NOT NULL DEFAULT 0"),
        ("transfer_fee", "INTEGER NOT NULL DEFAULT 0"),
        ("other_fee", "INTEGER NOT NULL DEFAULT 0"),
        ("commission_rate", "INTEGER NOT NULL DEFAULT 0"),
        ("stamp_duty_rate", "INTEGER NOT NULL DEFAULT 0"),
        ("transfer_fee_rate", "INTEGER NOT NULL DEFAULT 0"),
        ("other_fee_rate", "INTEGER NOT NULL DEFAULT 0"),
        ("minimum_commission", "INTEGER NOT NULL DEFAULT 0"),
        ("minimum_commission_enabled", "BOOLEAN NOT NULL DEFAULT 1"),
        ("cash_after", "INTEGER NOT NULL DEFAULT 0"),
        ("realized_pnl", "INTEGER NOT NULL DEFAULT 0"),
        ("is_theoretical", "BOOLEAN NOT NULL DEFAULT 0"),
        ("price_date", "DATE"),
    )
    for name, declaration in transaction_columns:
        _add_column_if_missing(connection, "simulation_transactions", name, declaration)
    transaction_column_names = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(simulation_transactions)")
    }
    if "transaction_at" in transaction_column_names:
        connection.exec_driver_sql(
            "UPDATE simulation_transactions SET execution_date = DATE(transaction_at) "
            "WHERE execution_date IS NULL AND transaction_at IS NOT NULL"
        )
    connection.exec_driver_sql(
        "UPDATE simulation_transactions SET source_identity = 'legacy-' || id "
        "WHERE source_identity = 'legacy'"
    )
    if {"account_id", "plan_id", "execution_date", "source_identity"}.issubset(
        transaction_column_names
    ):
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_simulation_transaction_week_source "
            "ON simulation_transactions (account_id, plan_id, execution_date, source_identity)"
        )
    total_cost_was_added = _add_column_if_missing(
        connection, "simulation_positions", "total_cost", "INTEGER NOT NULL DEFAULT 0"
    )
    _add_column_if_missing(connection, "simulation_positions", "realized_pnl", "INTEGER NOT NULL DEFAULT 0")
    if total_cost_was_added:
        scale = 10**8
        legacy_positions = connection.exec_driver_sql(
            "SELECT id, quantity, average_cost FROM simulation_positions"
        ).all()
        for row_id, quantity, average_cost in legacy_positions:
            if quantity is None or average_cost is None:
                continue
            scaled_cost = _rounded_integer_quotient(int(quantity) * int(average_cost), scale)
            connection.exec_driver_sql(
                "UPDATE simulation_positions SET total_cost = ? WHERE id = ?",
                (scaled_cost, row_id),
            )
    for name, declaration in (
        ("external_cash_flow", "INTEGER NOT NULL DEFAULT 0"),
        ("time_weighted_return", "INTEGER"),
        ("money_weighted_return", "INTEGER"),
        ("annualized_return", "INTEGER"),
        ("max_drawdown", "INTEGER"),
    ):
        _add_column_if_missing(connection, "simulation_daily_snapshots", name, declaration)


def _upgrade_to_version_eight(connection: Connection) -> None:
    """Repair v7 open positions whose newly-added cost basis was left at zero."""
    columns = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(simulation_positions)")
    }
    required = {"id", "quantity", "average_cost", "total_cost"}
    if not required.issubset(columns):
        return
    scale = 10**8
    affected_rows = connection.exec_driver_sql(
        """
        SELECT id, quantity, average_cost
        FROM simulation_positions
        WHERE total_cost = 0 AND quantity != 0 AND average_cost != 0
        """
    ).all()
    for row_id, quantity, average_cost in affected_rows:
        repaired_cost = _rounded_integer_quotient(int(quantity) * int(average_cost), scale)
        connection.exec_driver_sql(
            "UPDATE simulation_positions SET total_cost = ? WHERE id = ? AND total_cost = 0",
            (repaired_cost, row_id),
        )


def _upgrade_to_version_nine(connection: Connection) -> None:
    """Add exact real-account ledger projections and comparable return metrics."""
    # Some minimal legacy fixtures (and early app builds that disabled the real
    # module) have no real-account tables at all.  Create their current table
    # definitions before attempting additive upgrades.
    for table_name in (
        "real_accounts",
        "real_transactions",
        "real_account_snapshots",
        "real_positions",
    ):
        Base.metadata.tables[table_name].create(connection, checkfirst=True)
    for name, declaration in (
        ("cost_basis", "INTEGER NOT NULL DEFAULT 0"),
        ("net_proceeds", "INTEGER NOT NULL DEFAULT 0"),
        ("realized_pnl", "INTEGER NOT NULL DEFAULT 0"),
    ):
        _add_column_if_missing(connection, "real_transactions", name, declaration)
    for name, declaration in (
        ("total_contribution", "INTEGER NOT NULL DEFAULT 0"),
        ("external_cash_flow", "INTEGER NOT NULL DEFAULT 0"),
        ("total_pnl", "INTEGER NOT NULL DEFAULT 0"),
        ("fees_paid", "INTEGER NOT NULL DEFAULT 0"),
        ("daily_return", "INTEGER"),
        ("time_weighted_return", "INTEGER"),
        ("money_weighted_return", "INTEGER"),
        ("annualized_return", "INTEGER"),
        ("max_drawdown", "INTEGER"),
    ):
        _add_column_if_missing(connection, "real_account_snapshots", name, declaration)


def _upgrade_to_version_ten(connection: Connection) -> None:
    """Reserve the index-only research universe without rewriting ETF records."""
    # Minimal historical migration fixtures can intentionally omit the market
    # tables.  They still need their schema version advanced, but an index seed
    # cannot be written until the normal application schema is present.
    instruments_table = connection.execute(
        text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'instruments'")
    ).scalar_one_or_none()
    if instruments_table is None:
        return
    from ..database.initialize import DEFAULT_INDEX_INSTRUMENTS
    from ..models.models import Instrument

    for defaults in DEFAULT_INDEX_INSTRUMENTS:
        exists = connection.execute(
            text("SELECT 1 FROM instruments WHERE code = :code"), {"code": defaults["code"]}
        ).scalar_one_or_none()
        if exists is None:
            connection.execute(
                Instrument.__table__.insert().values(
                    **defaults,
                    category="index",
                    currency="USD" if defaults["code"] == "NDX" else "CNY",
                    is_active=True,
                    extra_data={"research_kind": "index", "data_source": "AKSHARE_INDEX"},
                )
            )


def _upgrade_to_version_eleven(connection: Connection) -> None:
    """Store an explicit lossless scale for high-volume aggregated periods."""
    market_prices_table = connection.execute(
        text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'market_prices'")
    ).scalar_one_or_none()
    if market_prices_table is not None:
        _add_column_if_missing(connection, "market_prices", "volume_multiplier", "INTEGER NOT NULL DEFAULT 1")


def _upgrade_to_version_twelve(connection: Connection) -> None:
    """Persist automatic real-account calculations and benchmark snapshots."""
    for table_name in ("real_transactions", "real_account_snapshots"):
        Base.metadata.tables[table_name].create(connection, checkfirst=True)
    for name, declaration in (
        ("planned_amount", "INTEGER NOT NULL DEFAULT 0"),
        ("cash_after", "INTEGER NOT NULL DEFAULT 0"),
        ("holding_quantity_after", "INTEGER NOT NULL DEFAULT 0"),
        ("average_cost_after", "INTEGER NOT NULL DEFAULT 0"),
        ("market_snapshot", "JSON NOT NULL DEFAULT '{}'") ,
    ):
        _add_column_if_missing(connection, "real_transactions", name, declaration)
    for name, declaration in (
        ("realized_pnl", "INTEGER NOT NULL DEFAULT 0"),
        ("unrealized_pnl", "INTEGER NOT NULL DEFAULT 0"),
        ("holding_quantity", "INTEGER NOT NULL DEFAULT 0"),
        ("benchmark_total_assets", "INTEGER"),
        ("benchmark_total_pnl", "INTEGER"),
        ("benchmark_quantity", "INTEGER"),
    ):
        _add_column_if_missing(connection, "real_account_snapshots", name, declaration)


def _upgrade_to_version_thirteen(connection: Connection) -> None:
    """Add the durable, non-simulated investment calendar."""
    Base.metadata.tables["investment_calendar_entries"].create(connection, checkfirst=True)


def _normalized_check_sql(sql: object) -> str:
    return "".join(str(sql).lower().split()).strip("()")


def _sqlite_type_affinity(declared_type: object) -> str:
    type_name = str(declared_type).upper()
    if "INT" in type_name:
        return "INTEGER"
    if any(token in type_name for token in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if not type_name or "BLOB" in type_name:
        return "BLOB"
    if any(token in type_name for token in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def _normalized_server_default(default: object) -> tuple[str, str]:
    value = str(default).strip()
    while value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        quote = value[0]
        return ("literal", value[1:-1].replace(quote * 2, quote))
    return ("expression", "".join(value.lower().split()))


def _normalized_optional_sql(sql: object | None) -> str | None:
    return None if sql is None else _normalized_check_sql(sql)


def _normalized_foreign_key_option(value: object | None) -> object | None:
    if value is None:
        return None
    if isinstance(value, str):
        return " ".join(value.upper().split())
    return value


def _normalized_foreign_key_options(
    options: Mapping[str, object | None],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted(
            (
                key,
                normalized_value,
            )
            for key, value in options.items()
            if (normalized_value := _normalized_foreign_key_option(value)) is not None
        )
    )


def _validate_v2_schema(connection: Connection) -> None:
    """Reject same-name V2 tables that do not satisfy the declared metadata."""
    inspector = inspect(connection)
    for table_name in V2_TABLE_NAMES:
        metadata_table = Base.metadata.tables[table_name]
        inspected_columns = inspector.get_columns(table_name)
        actual_columns = {column["name"]: column for column in inspected_columns}
        required_columns = set(metadata_table.c.keys())
        problems: list[str] = []
        actual_column_names = set(actual_columns)
        if actual_column_names != required_columns:
            problems.append(
                f"columns {sorted(actual_column_names)} "
                f"do not match {sorted(required_columns)}"
            )

        for column_name in required_columns & actual_column_names:
            metadata_column = metadata_table.c[column_name]
            actual_column = actual_columns[column_name]
            required_affinity = _sqlite_type_affinity(
                metadata_column.type.compile(dialect=connection.dialect)
            )
            actual_affinity = _sqlite_type_affinity(
                actual_column["type"].compile(dialect=connection.dialect)
            )
            if actual_affinity != required_affinity:
                problems.append(
                    f"column {column_name} affinity {actual_affinity} "
                    f"does not match {required_affinity}"
                )
            if bool(actual_column["nullable"]) != bool(metadata_column.nullable):
                problems.append(
                    f"column {column_name} nullable={actual_column['nullable']} "
                    f"does not match nullable={metadata_column.nullable}"
                )
            if metadata_column.server_default is None:
                required_default = None
            else:
                default_argument = metadata_column.server_default.arg
                if isinstance(default_argument, str):
                    required_default = ("literal", default_argument)
                elif hasattr(default_argument, "compile"):
                    compiled_default = default_argument.compile(
                        dialect=connection.dialect,
                        compile_kwargs={"literal_binds": True},
                    )
                    required_default = _normalized_server_default(compiled_default)
                else:
                    required_default = _normalized_server_default(default_argument)
            actual_default_value = actual_column["default"]
            actual_default = (
                None
                if actual_default_value is None
                else _normalized_server_default(actual_default_value)
            )
            if actual_default != required_default:
                problems.append(
                    f"column {column_name} server default {actual_default} "
                    f"does not match {required_default}"
                )

        required_primary_key = tuple(
            column.name for column in metadata_table.primary_key.columns
        )
        actual_primary_key = tuple(
            inspector.get_pk_constraint(table_name)["constrained_columns"]
        )
        if actual_primary_key != required_primary_key:
            problems.append(
                f"primary key {actual_primary_key} does not match {required_primary_key}"
            )

        required_unique_constraints = {
            tuple(column.name for column in constraint.columns)
            for constraint in metadata_table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        actual_unique_constraints = {
            tuple(constraint["column_names"])
            for constraint in inspector.get_unique_constraints(table_name)
        }
        if actual_unique_constraints != required_unique_constraints:
            problems.append(
                f"unique constraints {sorted(actual_unique_constraints, key=repr)} "
                f"do not match {sorted(required_unique_constraints, key=repr)}"
            )

        required_foreign_keys = {
            (
                constraint.name,
                tuple(element.parent.name for element in constraint.elements),
                constraint.elements[0].column.table.schema,
                constraint.elements[0].column.table.name,
                tuple(element.column.name for element in constraint.elements),
                _normalized_foreign_key_option(constraint.ondelete),
                _normalized_foreign_key_option(constraint.onupdate),
                _normalized_foreign_key_options(
                    {
                        "deferrable": constraint.deferrable,
                        "initially": constraint.initially,
                        "match": constraint.match,
                    }
                ),
            )
            for constraint in metadata_table.constraints
            if isinstance(constraint, ForeignKeyConstraint)
        }
        actual_foreign_keys = set()
        for constraint in inspector.get_foreign_keys(table_name):
            options = dict(constraint.get("options") or {})
            actual_foreign_keys.add(
                (
                    constraint.get("name"),
                    tuple(constraint["constrained_columns"]),
                    constraint.get("referred_schema"),
                    constraint["referred_table"],
                    tuple(constraint["referred_columns"]),
                    _normalized_foreign_key_option(options.pop("ondelete", None)),
                    _normalized_foreign_key_option(options.pop("onupdate", None)),
                    _normalized_foreign_key_options(options),
                )
            )
        if actual_foreign_keys != required_foreign_keys:
            problems.append(
                f"foreign keys {sorted(actual_foreign_keys, key=repr)} "
                f"do not match {sorted(required_foreign_keys, key=repr)}"
            )

        required_checks = {
            (constraint.name, _normalized_check_sql(constraint.sqltext))
            for constraint in metadata_table.constraints
            if isinstance(constraint, CheckConstraint)
        }
        actual_checks = {
            (constraint["name"], _normalized_check_sql(constraint["sqltext"]))
            for constraint in inspector.get_check_constraints(table_name)
        }
        if actual_checks != required_checks:
            problems.append(
                f"check constraints {sorted(actual_checks, key=repr)} "
                f"do not match {sorted(required_checks, key=repr)}"
            )

        required_indexes = {
            (
                index.name,
                bool(index.unique),
                tuple(column.name for column in index.columns),
                _normalized_optional_sql(
                    index.dialect_options["sqlite"].get("where")
                ),
            )
            for index in metadata_table.indexes
        }
        actual_indexes = {
            (
                index["name"],
                bool(index["unique"]),
                tuple(index["column_names"]),
                _normalized_optional_sql(
                    (index.get("dialect_options") or {}).get("sqlite_where")
                ),
            )
            for index in inspector.get_indexes(table_name)
        }
        if actual_indexes != required_indexes:
            problems.append(
                f"indexes {sorted(actual_indexes, key=repr)} "
                f"do not match {sorted(required_indexes, key=repr)}"
            )

        if problems:
            raise SchemaVersionError(
                f"V2 table {table_name} does not satisfy schema metadata: "
                + "; ".join(problems)
            )


def _upgrade_to_version_fourteen(connection: Connection) -> None:
    """Create isolated V2 weekly-analysis persistence without altering V1 data."""
    for table_name in V2_TABLE_NAMES:
        Base.metadata.tables[table_name].create(connection, checkfirst=True)
    _validate_v2_schema(connection)


def _validate_market_price_volume_source(connection: Connection) -> None:
    columns = {
        row[1]: row
        for row in connection.exec_driver_sql("PRAGMA table_info(market_prices)")
    }
    column = columns.get("volume_source")
    if column is None:
        raise SchemaVersionError(
            "market_prices does not satisfy schema version 15: "
            "missing volume_source"
        )
    declared_type = (column[2] or "").upper()
    if not declared_type.startswith("VARCHAR") or column[3] or column[4] is not None:
        raise SchemaVersionError(
            "market_prices does not satisfy schema version 15: "
            "volume_source must be nullable VARCHAR without a server default"
        )


def _upgrade_to_version_fifteen(connection: Connection) -> None:
    """Separate legacy volume provenance from the OHLC price source.

    Version-14 daily rows carried one provider label that supplied both price
    and volume. Legacy period rows, however, stored the volume derivation in
    ``source``; their OHLC provider is therefore explicitly unknown.
    """
    market_prices_table = connection.execute(
        text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'market_prices'"
        )
    ).scalar_one_or_none()
    if market_prices_table is None:
        return
    columns = {
        row[1]
        for row in connection.exec_driver_sql("PRAGMA table_info(market_prices)")
    }
    if "volume_source" not in columns:
        connection.exec_driver_sql(
            "ALTER TABLE market_prices ADD COLUMN volume_source VARCHAR(64)"
        )
    connection.exec_driver_sql(
        "UPDATE market_prices "
        "SET volume_source = source "
        "WHERE volume_source IS NULL AND source IS NOT NULL"
    )
    connection.exec_driver_sql(
        "UPDATE market_prices "
        "SET source = 'LEGACY_PRICE_SOURCE_UNKNOWN' "
        "WHERE timeframe IN ('weekly', 'monthly')"
    )
    instruments_table = connection.execute(
        text(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'instruments'"
        )
    ).scalar_one_or_none()
    if instruments_table is not None:
        connection.exec_driver_sql(
            "UPDATE market_prices "
            "SET volume = NULL, volume_multiplier = 1, "
            "volume_source = 'VOLUME_UNAVAILABLE:DIRECT_INDEX' "
            "WHERE instrument_id IN ("
            "SELECT id FROM instruments WHERE code = 'NDX'"
            ")"
        )
    _validate_market_price_volume_source(connection)


_WEEKLY_ANALYSIS_CODEC = "weekly-analysis-tagged-json-v1"
_WEEKLY_ANALYSIS_SCHEMA = "weekly-analysis-repository-v2"


def _migration_json_mapping(
    value: object,
    *,
    context: str,
) -> dict[str, Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise SchemaVersionError(f"{context} is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise SchemaVersionError(f"{context} must be a JSON object")
    return {str(name): item for name, item in parsed.items()}


def _migration_snapshot_digest(snapshot: Mapping[str, object]) -> str:
    body = {
        name: item
        for name, item in snapshot.items()
        if name != "snapshot_hash"
    }
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _migration_database_directory(connection: Connection) -> Path | None:
    for row in connection.exec_driver_sql("PRAGMA database_list"):
        if row[1] == "main" and row[2]:
            return Path(row[2]).resolve().parent
    return None


def _migration_artifact_path(
    connection: Connection,
    audit: Mapping[str, object],
    *,
    relative_path: Path,
    symbol: str,
    context: str,
    legacy_weekly_analysis_audit_root: Path | str | None,
) -> Path | None:
    """Resolve the persisted artifact authority without assuming one root."""

    database_directory = _migration_database_directory(connection)
    persisted_value = audit.get("artifact_path")
    root_value = audit.get("audit_root")
    root: Path | None
    if root_value is not None:
        if not isinstance(root_value, str) or not root_value.strip():
            raise SchemaVersionError(
                f"{context} audit_root must be a non-empty string"
            )
        root_path = Path(root_value)
        if root_path.is_absolute():
            root = root_path.resolve()
        elif database_directory is not None:
            root = (database_directory / root_path).resolve()
        else:
            raise SchemaVersionError(
                f"{context} relative audit_root needs a database path"
            )
    elif legacy_weekly_analysis_audit_root is not None:
        root = Path(legacy_weekly_analysis_audit_root).resolve()
    elif database_directory is not None:
        root = (database_directory / "weekly_analysis_v2").resolve()
    else:
        root = None

    if persisted_value is None:
        if root is None:
            return None
        candidate = (root / relative_path).resolve()
    else:
        if (
            not isinstance(persisted_value, str)
            or not persisted_value.strip()
        ):
            raise SchemaVersionError(
                f"{context} artifact_path must be a non-empty string"
            )
        persisted_path = Path(persisted_value)
        if persisted_path.is_absolute():
            candidate = persisted_path.resolve()
        else:
            if root is None:
                raise SchemaVersionError(
                    f"{context} relative artifact_path has no trusted root"
                )
            candidate = (root / persisted_path).resolve()

    if root is None:
        raise SchemaVersionError(
            f"{context} absolute artifact_path has no trusted audit root"
        )
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SchemaVersionError(
            f"{context} artifact_path escapes its audit root"
        ) from exc
    relative_parts = tuple(part.casefold() for part in relative_path.parts)
    candidate_parts = tuple(part.casefold() for part in candidate.parts)
    if (
        not relative_parts
        or len(candidate_parts) < len(relative_parts)
        or candidate_parts[-len(relative_parts) :] != relative_parts
        or relative_path.parts[0] != symbol
    ):
        raise SchemaVersionError(
            f"{context} artifact_path crosses its symbol snapshot path"
        )
    return candidate


def _write_fsynced_sibling(
    path: Path,
    content: bytes,
    *,
    suffix: str,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{uuid4().hex}.",
        suffix=suffix,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


@dataclass(slots=True)
class _MigrationArtifactEntry:
    path: Path
    backup_path: Path
    staged_path: Path
    upgraded_bytes: bytes
    applied: bool = False


class _MigrationArtifactBatch:
    """Compensating filesystem transaction spanning the SQLite commit."""

    def __init__(self, recovery_root: Path) -> None:
        self.recovery_root = recovery_root
        self.entries: list[_MigrationArtifactEntry] = []
        self.initial_user_version: int | None = None
        self.initial_advice_columns: frozenset[str] = frozenset()
        self.initial_task_columns: frozenset[str] = frozenset()
        self.expected_advice_rows: dict[
            int, tuple[dict[str, Any], dict[str, Any], str]
        ] = {}

    def record_initial_database(
        self,
        connection: Connection,
        *,
        user_version: int,
    ) -> None:
        self.initial_user_version = user_version
        self.initial_advice_columns = frozenset(
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(v2_advice_history)"
            )
        )
        self.initial_task_columns = frozenset(
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(v2_analysis_tasks)"
            )
        )

    def record_expected_advice_rows(
        self,
        rows: list[dict[str, object]],
    ) -> None:
        expected: dict[
            int, tuple[dict[str, Any], dict[str, Any], str]
        ] = {}
        for row in rows:
            advice_id = row.get("id")
            legacy_audit = row.get("legacy_audit")
            upgraded_audit = row.get("upgraded_audit")
            generation_key = row.get("generation_key")
            if (
                type(advice_id) is not int
                or not isinstance(legacy_audit, Mapping)
                or not isinstance(upgraded_audit, str)
                or not isinstance(generation_key, str)
            ):
                raise SchemaVersionError(
                    "version-16 advice inspection metadata is invalid"
                )
            expected[advice_id] = (
                dict(legacy_audit),
                _migration_json_mapping(
                    upgraded_audit,
                    context=f"v16 advice {advice_id} expected audit",
                ),
                generation_key,
            )
        self.expected_advice_rows = expected

    def prepare(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            path = row.get("artifact_path")
            if not isinstance(path, Path):
                continue
            original = row.get("original_artifact_bytes")
            upgraded = row.get("artifact_bytes")
            if not isinstance(original, bytes) or not isinstance(
                upgraded, bytes
            ):
                raise SchemaVersionError(
                    f"v15 advice {row.get('id')} artifact staging is invalid"
                )
            backup_path: Path | None = None
            staged_path: Path | None = None
            try:
                backup_path = _write_fsynced_sibling(
                    path,
                    original,
                    suffix=".v15.backup",
                )
                staged_path = _write_fsynced_sibling(
                    path,
                    upgraded,
                    suffix=".v16.stage",
                )
                self.entries.append(
                    _MigrationArtifactEntry(
                        path=path,
                        backup_path=backup_path,
                        staged_path=staged_path,
                        upgraded_bytes=upgraded,
                    )
                )
            except BaseException:
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)
                if backup_path is not None:
                    backup_path.unlink(missing_ok=True)
                raise

    def apply_all(self) -> None:
        for entry in self.entries:
            # Mark before the syscall so an ambiguous OSError raised after a
            # successful rename is still compensated from the backup.
            entry.applied = True
            os.replace(entry.staged_path, entry.path)

    def compensate(
        self,
        primary: BaseException,
        *,
        extra_errors: tuple[str, ...] = (),
    ) -> None:
        errors = list(extra_errors)
        for entry in reversed(self.entries):
            if not entry.applied:
                continue
            try:
                os.replace(entry.backup_path, entry.path)
                entry.applied = False
            except BaseException as exc:
                errors.append(
                    f"restore {entry.path}: {type(exc).__name__}: {exc}"
                )
        if not errors:
            for entry in self.entries:
                for temporary_path in (
                    entry.staged_path,
                    entry.backup_path,
                ):
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except BaseException as exc:
                        errors.append(
                            f"cleanup {temporary_path}: "
                            f"{type(exc).__name__}: {exc}"
                        )
        if errors:
            self._raise_recovery_error(primary, errors)

    def finalize(self) -> None:
        errors: list[str] = []
        for entry in self.entries:
            for temporary_path in (
                entry.staged_path,
                entry.backup_path,
            ):
                try:
                    temporary_path.unlink(missing_ok=True)
                except BaseException as exc:
                    errors.append(
                        f"cleanup {temporary_path}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if errors:
            primary = RuntimeError(
                "version-16 migration committed but temporary cleanup failed"
            )
            self._raise_recovery_error(primary, errors)

    def raise_mixed_state(
        self,
        primary: BaseException,
        detail: str,
    ) -> None:
        self._raise_recovery_error(
            primary,
            [f"durable database state is mixed: {detail}"],
        )

    def _raise_recovery_error(
        self,
        primary: BaseException,
        errors: list[str],
    ) -> None:
        manifest_path = (
            self.recovery_root
            / f"weekly_analysis_v16_recovery_{uuid4().hex}.json"
        )
        manifest = {
            "status": "recoverable",
            "primary_error": f"{type(primary).__name__}: {primary}",
            "compensation_errors": errors,
            "artifacts": [
                {
                    "path": str(entry.path),
                    "backup_path": str(entry.backup_path),
                    "staged_path": str(entry.staged_path),
                    "applied": entry.applied,
                }
                for entry in self.entries
            ],
        }
        manifest_error: BaseException | None = None
        try:
            self.recovery_root.mkdir(parents=True, exist_ok=True)
            manifest_bytes = (
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            temporary = _write_fsynced_sibling(
                manifest_path,
                manifest_bytes,
                suffix=".manifest.stage",
            )
            os.replace(temporary, manifest_path)
        except BaseException as exc:
            manifest_error = exc
        message = (
            "version-16 migration compensation failed; recovery manifest: "
            f"{manifest_path}"
        )
        recovery = SchemaVersionError(message)
        for error in errors:
            recovery.add_note(error)
        if manifest_error is not None:
            recovery.add_note(
                "recovery manifest write failed: "
                f"{type(manifest_error).__name__}: {manifest_error}"
            )
        raise recovery from primary


def _validated_v15_advice_rows(
    connection: Connection,
    *,
    legacy_weekly_analysis_audit_root: Path | str | None,
) -> list[dict[str, object]]:
    """Validate and prepare every legacy advice before any version-16 DDL."""

    rows = connection.exec_driver_sql(
        "SELECT advice.id, advice.instrument_id, advice.model_number, "
        "advice.work_number, advice.iteration_number, advice.advice_at, "
        "advice.status, advice.payload, advice.audit_data, "
        "advice.created_at, advice.updated_at, instruments.code AS symbol "
        "FROM v2_advice_history AS advice "
        "LEFT JOIN instruments ON instruments.id = advice.instrument_id "
        "ORDER BY advice.id"
    ).mappings()
    prepared: list[dict[str, object]] = []
    for raw_row in rows:
        row = dict(raw_row)
        advice_id = row["id"]
        symbol = row["symbol"]
        context = f"v15 advice {advice_id}"
        if not isinstance(symbol, str) or symbol not in {"399006", "NDX"}:
            raise SchemaVersionError(
                f"{context} references an invalid instrument"
            )
        audit = _migration_json_mapping(
            row["audit_data"],
            context=f"{context} audit",
        )
        snapshot = _migration_json_mapping(
            audit.get("snapshot"),
            context=f"{context} embedded snapshot",
        )
        relative_value = audit.get("snapshot_path")
        digest = audit.get("snapshot_hash")
        relative_path = (
            Path(relative_value)
            if isinstance(relative_value, str)
            else None
        )
        if (
            relative_path is None
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative_path.parts[0] != symbol
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or audit.get("symbol") != symbol
            or audit.get("kind") != "advice"
            or audit.get("codec_version") != _WEEKLY_ANALYSIS_CODEC
            or snapshot.get("schema_version") != _WEEKLY_ANALYSIS_SCHEMA
            or snapshot.get("kind") != "advice"
            or snapshot.get("symbol") != symbol
            or snapshot.get("codec_version") != _WEEKLY_ANALYSIS_CODEC
            or snapshot.get("snapshot_hash") != digest
            or _migration_snapshot_digest(snapshot) != digest
        ):
            raise SchemaVersionError(
                f"{context} audit snapshot reference is invalid"
            )

        artifact_path = _migration_artifact_path(
            connection,
            audit,
            relative_path=relative_path,
            symbol=symbol,
            context=context,
            legacy_weekly_analysis_audit_root=(
                legacy_weekly_analysis_audit_root
            ),
        )
        original_artifact_bytes: bytes | None = None
        if artifact_path is not None:
            candidate = artifact_path
            if candidate.exists():
                if not candidate.is_file():
                    raise SchemaVersionError(
                        f"{context} snapshot artifact is not a file"
                    )
                try:
                    original_artifact_bytes = candidate.read_bytes()
                    artifact_snapshot = _migration_json_mapping(
                        original_artifact_bytes.decode("utf-8"),
                        context=f"{context} snapshot artifact",
                    )
                except (OSError, UnicodeDecodeError) as exc:
                    raise SchemaVersionError(
                        f"{context} snapshot artifact is unreadable"
                    ) from exc
                if artifact_snapshot != snapshot:
                    raise SchemaVersionError(
                        f"{context} snapshot artifact diverges from its audit"
                    )
            else:
                artifact_path = None

        generation_key = f"legacy:{advice_id}"
        upgraded_snapshot = dict(snapshot)
        upgraded_snapshot.pop("snapshot_hash", None)
        upgraded_snapshot["advice_generation"] = 1
        upgraded_snapshot["generation_key"] = generation_key
        upgraded_digest = _migration_snapshot_digest(upgraded_snapshot)
        upgraded_snapshot["snapshot_hash"] = upgraded_digest
        upgraded_audit = dict(audit)
        upgraded_audit.update(
            {
                "snapshot": upgraded_snapshot,
                "snapshot_hash": upgraded_digest,
                "symbol": symbol,
                "kind": "advice",
                "codec_version": _WEEKLY_ANALYSIS_CODEC,
            }
        )
        row.update(
            {
                "generation_key": generation_key,
                "legacy_audit": audit,
                "upgraded_audit": json.dumps(
                    upgraded_audit,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "artifact_path": artifact_path,
                "original_artifact_bytes": original_artifact_bytes,
                "artifact_bytes": (
                    json.dumps(
                        upgraded_snapshot,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8"),
            }
        )
        prepared.append(row)
    return prepared


def _upgrade_to_version_sixteen(
    connection: Connection,
    artifact_batch: _MigrationArtifactBatch,
    *,
    legacy_weekly_analysis_audit_root: Path | str | None,
) -> None:
    """Add durable worker leases and versioned, idempotent advice history."""

    advice_columns = {
        row[1]
        for row in connection.exec_driver_sql(
            "PRAGMA table_info(v2_advice_history)"
        )
    }
    needs_advice_upgrade = bool(
        {"advice_generation", "generation_key"} - advice_columns
    )
    prepared_advice = (
        _validated_v15_advice_rows(
            connection,
            legacy_weekly_analysis_audit_root=(
                legacy_weekly_analysis_audit_root
            ),
        )
        if needs_advice_upgrade
        else []
    )
    artifact_batch.record_expected_advice_rows(prepared_advice)
    artifact_batch.prepare(prepared_advice)

    task_columns = {
        row[1]
        for row in connection.exec_driver_sql(
            "PRAGMA table_info(v2_analysis_tasks)"
        )
    }
    if "worker_token" not in task_columns:
        connection.exec_driver_sql(
            "ALTER TABLE v2_analysis_tasks ADD COLUMN worker_token VARCHAR(128)"
        )
    if "lease_expires_at" not in task_columns:
        connection.exec_driver_sql(
            "ALTER TABLE v2_analysis_tasks ADD COLUMN lease_expires_at DATETIME"
        )

    if needs_advice_upgrade:
        connection.exec_driver_sql(
            "ALTER TABLE v2_advice_history RENAME TO v2_advice_history_v15"
        )
        connection.exec_driver_sql(
            "DROP INDEX IF EXISTS ix_v2_advice_history_instrument_id"
        )
        Base.metadata.tables["v2_advice_history"].create(
            connection, checkfirst=False
        )
        for row in prepared_advice:
            connection.exec_driver_sql(
                "INSERT INTO v2_advice_history ("
                "id, instrument_id, model_number, work_number, "
                "iteration_number, advice_generation, generation_key, "
                "advice_at, status, payload, audit_data, created_at, "
                "updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    row["instrument_id"],
                    row["model_number"],
                    row["work_number"],
                    row["iteration_number"],
                    1,
                    row["generation_key"],
                    row["advice_at"],
                    row["status"],
                    row["payload"],
                    row["upgraded_audit"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )
        connection.exec_driver_sql("DROP TABLE v2_advice_history_v15")
    _validate_v2_schema(connection)
    artifact_batch.apply_all()


def _upgrade_to_version_seventeen(connection: Connection) -> None:
    """Add the append-only V3.1 prediction and evaluation store.

    V2 tables and audit artifacts are deliberately left untouched so the two
    model generations can be compared over identical cutoffs.
    """

    for table_name in (
        "v31_analysis_runs",
        "v31_model_versions",
        "v31_weekly_forecasts",
        "v31_daily_corrections",
        "v31_model_evaluations",
        "v31_instrument_market_mappings",
    ):
        Base.metadata.tables[table_name].create(connection, checkfirst=True)


def _upgrade_to_version_eighteen(connection: Connection) -> None:
    """Add the independent V3.2 training, inference, and audit store."""

    for table_name in (
        "v32_training_runs",
        "v32_training_iterations",
        "v32_training_checkpoints",
        "v32_model_versions",
        "v32_optimizer_states",
        "v32_data_snapshots",
        "v32_weekly_forecasts",
        "v32_daily_corrections",
        "v32_model_evaluations",
        "v32_analysis_runs",
        "v32_app_state",
    ):
        Base.metadata.tables[table_name].create(connection, checkfirst=True)


def _upgrade_to_version_nineteen(connection: Connection) -> None:
    """Add the isolated V3.3 point-in-time, training, and inference store.

    This migration only creates new ``v33_*`` tables.  It intentionally does
    not select from, update, rename, or rebuild any V1/V2/V3.1/V3.2 table, so
    the previously delivered model generations remain immutable audit data.
    """

    for table_name in (
        "v33_instrument_roles",
        "v33_raw_ingestions",
        "v33_source_releases",
        "v33_market_bars",
        "v33_point_in_time_observations",
        "v33_quality_assessments",
        "v33_feature_snapshots",
        "v33_training_runs",
        "v33_training_iterations",
        "v33_training_checkpoints",
        "v33_model_versions",
        "v33_optimizer_states",
        "v33_forecasts",
        "v33_model_evaluations",
        "v33_analysis_runs",
    ):
        Base.metadata.tables[table_name].create(connection, checkfirst=True)


V34_TABLES = (
    "v34_feature_snapshots",
    "v34_training_runs",
    "v34_model_versions",
    "v34_optimizer_states",
    "v34_scenario_adapters",
    "v34_probability_calibrators",
    "v34_forecasts",
    "v34_model_evaluations",
    "v34_training_iterations",
    "v34_analysis_runs",
)


def _validate_v34_schema(connection: Connection) -> None:
    """Verify the isolated V3.4 store without reading or mutating frozen rows."""

    existing = {
        str(row[0])
        for row in connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = set(V34_TABLES) - existing
    if missing:
        raise SchemaVersionError(
            "V3.4 schema is incomplete: " + ", ".join(sorted(missing))
        )
    forecast_columns = {
        str(row[1])
        for row in connection.exec_driver_sql("PRAGMA table_info(v34_forecasts)")
    }
    required_forecast_columns = {
        "forecast_anchor_date",
        "label_end_date",
        "horizon_weeks",
        "scenario_count",
        "representative_ohlcv_json",
        "indicator_path_json",
        "price_quantiles_json",
        "direction_probabilities_json",
        "path_probabilities_json",
        "consistency_json",
        "forecast_hash",
    }
    if not required_forecast_columns.issubset(forecast_columns):
        raise SchemaVersionError("V3.4 forecast schema is missing required audit fields")


def _upgrade_to_version_twenty(connection: Connection) -> None:
    """Add the isolated V3.4 13-week training, scenario, and audit store.

    No existing table is selected, updated, renamed, or rebuilt.  The V3.2 and
    V3.3 chains therefore retain byte-identical records and hashes.
    """

    for table_name in V34_TABLES:
        Base.metadata.tables[table_name].create(connection, checkfirst=True)
    _validate_v34_schema(connection)


V341_TABLES = (
    "v341_training_profiles",
    "v341_feature_snapshots",
    "v341_training_runs",
    "v341_model_versions",
    "v341_optimizer_states",
    "v341_scenario_adapters",
    "v341_probability_calibrators",
    "v341_random_plans",
    "v341_outer_evaluation_blocks",
    "v341_forecasts",
    "v341_forecast_calibrators",
    "v341_forecast_evaluations",
    "v341_residual_records",
    "v341_candidate_trials",
    "v341_model_health_snapshots",
    "v341_training_iterations",
    "v341_analysis_runs",
)


def _validate_v341_schema(connection: Connection) -> None:
    """Validate the isolated append-only V3.4.1 namespace."""

    existing = {
        str(row[0])
        for row in connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = set(V341_TABLES) - existing
    if missing:
        raise SchemaVersionError(
            "V3.4.1 schema is incomplete: " + ", ".join(sorted(missing))
        )
    forecast_columns = {
        str(row[1])
        for row in connection.exec_driver_sql("PRAGMA table_info(v341_forecasts)")
    }
    required = {
        "protocol_version",
        "forecast_anchor_date",
        "horizon_weeks",
        "scenario_count",
        "random_plan_hash",
        "random_plan_id",
        "expected_path_json",
        "horizon_probabilities_json",
        "thresholds_json",
        "calibrator_versions_json",
        "residual_pool_json",
        "model_reliability_score",
        "health_status",
        "forecast_hash",
    }
    if not required.issubset(forecast_columns):
        raise SchemaVersionError("V3.4.1 forecast schema is missing required audit fields")
    plan_columns = {
        str(row[1])
        for row in connection.exec_driver_sql("PRAGMA table_info(v341_random_plans)")
    }
    if not {
        "plan_version",
        "seed",
        "scenario_count",
        "residual_pool_identity",
        "plan_payload_zlib",
        "plan_hash",
    }.issubset(plan_columns):
        raise SchemaVersionError("V3.4.1 random-plan schema is incomplete")
    residual_columns = {
        str(row[1])
        for row in connection.exec_driver_sql("PRAGMA table_info(v341_residual_records)")
    }
    if not {
        "forecast_id",
        "prediction_path_json",
        "actual_path_json",
        "residual_path_json",
        "standardized_residual_json",
        "source_sigma",
        "residual_hash",
    }.issubset(residual_columns):
        raise SchemaVersionError("V3.4.1 residual schema is incomplete")


def _upgrade_to_version_twenty_one(connection: Connection) -> None:
    """Create V3.4.1 without altering any earlier-generation table."""

    for table_name in V341_TABLES:
        Base.metadata.tables[table_name].create(connection, checkfirst=True)
    _validate_v341_schema(connection)


V342_TABLES = (
    "v342_policy_versions",
    "v342_turning_assessments",
    "v342_turning_candidates",
    "v342_policy_runs",
    "v342_policy_batches",
    "v342_policy_batch_events",
    "v342_strategy_backtest_runs",
    "v342_strategy_backtest_points",
)


def _validate_v342_schema(connection: Connection) -> None:
    """Validate the append-only V3.4.2 turning and policy namespace."""

    existing = {
        str(row[0])
        for row in connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    namespace = {name for name in existing if name.startswith("v342_")}
    if namespace != set(V342_TABLES):
        raise SchemaVersionError(
            "V3.4.2 table namespace differs: "
            f"expected={sorted(V342_TABLES)} actual={sorted(namespace)}"
        )
    required_columns = {
        "v342_policy_versions": {
            "protocol_version", "model_market", "version", "config_hash",
            "max_single_change_pp", "minimum_cooldown_sessions",
            "transaction_cost_bps", "degraded_max_position_pp",
        },
        "v342_turning_assessments": {
            "forecast_id", "policy_version_id", "window_extrema_json",
            "consistency_status", "top_bottom_conflict", "assessment_hash",
        },
        "v342_turning_candidates": {
            "assessment_id", "signal_kind", "turn_kind", "candidate_week",
            "classification", "confirmation_status", "candidate_hash",
        },
        "v342_policy_runs": {
            "forecast_id", "policy_version_id", "turning_assessment_id",
            "position_source_event_id", "current_position_pp",
            "uncapped_target_pp", "target_position_pp",
            "next_executable_position_pp", "input_identity_hash",
        },
        "v342_policy_batches": {
            "policy_run_id", "batch_number", "change_pp", "initial_state",
            "depends_on_batch_id", "batch_hash",
        },
        "v342_policy_batch_events": {
            "batch_id", "from_state", "to_state", "event_hash",
        },
        "v342_strategy_backtest_runs": {
            "policy_version_id", "forecast_set_hash", "realized_return",
            "maximum_drawdown", "risk_adjusted_return", "turnover",
            "transaction_cost", "buy_hold_return", "fixed_dca_return",
            "evaluation_available_through", "price_set_hash",
        },
        "v342_strategy_backtest_points": {
            "backtest_run_id", "point_date", "position_pp", "net_return",
            "strategy_equity", "buy_hold_equity", "fixed_dca_equity",
        },
    }
    for table_name, required in required_columns.items():
        table = Base.metadata.tables[table_name]
        actual_column_rows = list(
            connection.exec_driver_sql(f"PRAGMA table_info({_quoted(table_name)})")
        )
        actual_columns = {str(row[1]): row for row in actual_column_rows}
        expected_names = {column.name for column in table.columns}
        if set(actual_columns) != expected_names or not required.issubset(actual_columns):
            raise SchemaVersionError(
                f"V3.4.2 table {table_name} has an inexact column set"
            )
        for column in table.columns:
            actual = actual_columns[column.name]
            expected_type = connection.dialect.type_compiler.process(column.type).upper()
            actual_type = str(actual[2]).upper()
            if actual_type != expected_type:
                raise SchemaVersionError(
                    f"V3.4.2 {table_name}.{column.name} type differs: "
                    f"expected={expected_type} actual={actual_type}"
                )
            if not column.primary_key and bool(actual[3]) != (not column.nullable):
                raise SchemaVersionError(
                    f"V3.4.2 {table_name}.{column.name} nullability differs"
                )
        expected_pk = tuple(
            column.name for column in sorted(table.primary_key.columns, key=lambda item: item.name)
        )
        actual_pk = tuple(
            str(row[1])
            for row in sorted(
                (row for row in actual_column_rows if int(row[5]) > 0),
                key=lambda row: int(row[5]),
            )
        )
        if actual_pk != expected_pk:
            raise SchemaVersionError(f"V3.4.2 {table_name} primary key differs")
        expected_fks = {
            (column.name, fk.column.table.name, fk.column.name)
            for column in table.columns
            for fk in column.foreign_keys
        }
        actual_fks = {
            (str(row[3]), str(row[2]), str(row[4]))
            for row in connection.exec_driver_sql(
                f"PRAGMA foreign_key_list({_quoted(table_name)})"
            )
        }
        if actual_fks != expected_fks:
            raise SchemaVersionError(f"V3.4.2 {table_name} foreign keys differ")
        index_rows = list(
            connection.exec_driver_sql(f"PRAGMA index_list({_quoted(table_name)})")
        )
        actual_index_names = {str(row[1]) for row in index_rows}
        expected_index_names = {index.name for index in table.indexes}
        if not expected_index_names.issubset(actual_index_names):
            raise SchemaVersionError(f"V3.4.2 {table_name} indexes are incomplete")
        actual_unique_columns = {
            tuple(
                str(info[2])
                for info in connection.exec_driver_sql(
                    f"PRAGMA index_info({_quoted(str(row[1]))})"
                )
            )
            for row in index_rows
            if bool(row[2])
        }
        expected_unique_columns = {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        }
        expected_unique_columns.update(
            tuple(column.name for column in index.columns)
            for index in table.indexes
            if index.unique
        )
        if not expected_unique_columns.issubset(actual_unique_columns):
            raise SchemaVersionError(f"V3.4.2 {table_name} unique constraints differ")
        create_sql = str(
            connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).scalar_one()
        )
        check_names = {
            constraint.name
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint) and constraint.name
        }
        if any(name not in create_sql for name in check_names):
            raise SchemaVersionError(f"V3.4.2 {table_name} check constraints differ")


def _upgrade_to_version_twenty_two(connection: Connection) -> None:
    """Create V3.4.2 without selecting or altering any earlier table."""

    for table_name in V342_TABLES:
        Base.metadata.tables[table_name].create(connection, checkfirst=True)
    _validate_v342_schema(connection)


def _durable_migration_state(
    engine: Engine,
    artifact_batch: _MigrationArtifactBatch,
) -> tuple[str, str]:
    """Classify an uncertain COMMIT using a separate durable connection."""

    try:
        with engine.connect() as inspection:
            user_version = int(
                inspection.execute(
                    text("PRAGMA user_version")
                ).scalar_one()
            )
            advice_columns = frozenset(
                row[1]
                for row in inspection.exec_driver_sql(
                    "PRAGMA table_info(v2_advice_history)"
                )
            )
            task_columns = frozenset(
                row[1]
                for row in inspection.exec_driver_sql(
                    "PRAGMA table_info(v2_analysis_tasks)"
                )
            )
            if user_version == SCHEMA_VERSION:
                try:
                    _validate_market_price_volume_source(inspection)
                    _validate_v2_schema(inspection)
                    _validate_v34_schema(inspection)
                    _validate_v341_schema(inspection)
                    _validate_v342_schema(inspection)
                except SchemaVersionError as exc:
                    return "mixed", f"schema validation failed: {exc}"
                for advice_id, (
                    _legacy_audit,
                    upgraded_audit,
                    generation_key,
                ) in artifact_batch.expected_advice_rows.items():
                    row = inspection.exec_driver_sql(
                        "SELECT advice_generation, generation_key, "
                        "audit_data FROM v2_advice_history WHERE id = ?",
                        (advice_id,),
                    ).mappings().one_or_none()
                    if row is None:
                        return (
                            "mixed",
                            f"v16 advice {advice_id} is missing",
                        )
                    actual_audit = _migration_json_mapping(
                        row["audit_data"],
                        context=f"v16 advice {advice_id} durable audit",
                    )
                    if (
                        row["advice_generation"] != 1
                        or row["generation_key"] != generation_key
                        or actual_audit != upgraded_audit
                    ):
                        return (
                            "mixed",
                            f"v16 advice {advice_id} hash/provenance differs",
                        )
                for entry in artifact_batch.entries:
                    try:
                        artifact_bytes = entry.path.read_bytes()
                    except OSError as exc:
                        return (
                            "mixed",
                            f"v16 artifact {entry.path} is unreadable: {exc}",
                        )
                    if artifact_bytes != entry.upgraded_bytes:
                        return (
                            "mixed",
                            f"v16 artifact {entry.path} differs from its row",
                        )
                return "committed", "complete version-16 state is durable"

            if (
                artifact_batch.initial_user_version is not None
                and user_version == artifact_batch.initial_user_version
                and advice_columns
                == artifact_batch.initial_advice_columns
                and task_columns == artifact_batch.initial_task_columns
            ):
                for advice_id, (
                    legacy_audit,
                    _upgraded_audit,
                    _generation_key,
                ) in artifact_batch.expected_advice_rows.items():
                    row = inspection.exec_driver_sql(
                        "SELECT audit_data FROM v2_advice_history "
                        "WHERE id = ?",
                        (advice_id,),
                    ).mappings().one_or_none()
                    if row is None:
                        return (
                            "mixed",
                            f"pre-migration advice {advice_id} is missing",
                        )
                    actual_audit = _migration_json_mapping(
                        row["audit_data"],
                        context=(
                            f"pre-migration advice {advice_id} durable audit"
                        ),
                    )
                    if actual_audit != legacy_audit:
                        return (
                            "mixed",
                            f"pre-migration advice {advice_id} changed",
                        )
                return "rolled_back", "pre-migration state is durable"

            return (
                "mixed",
                "user_version/table columns match neither recorded state",
            )
    except BaseException as exc:
        return (
            "mixed",
            f"independent durable inspection failed: "
            f"{type(exc).__name__}: {exc}",
        )


def run_migrations(
    engine: Engine,
    *,
    legacy_weekly_analysis_audit_root: Path | str | None = None,
) -> int:
    """Upgrade supported database versions and reject unknown future versions."""
    with engine.connect() as connection:
        # Python's sqlite3 legacy transaction mode does not start a physical
        # transaction for DDL.  An explicit BEGIN keeps schema changes
        # rollback-capable if either artifact replacement or COMMIT fails.
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        recovery_root = _migration_database_directory(connection) or Path.cwd()
        artifact_batch = _MigrationArtifactBatch(recovery_root)
        try:
            current_version = int(
                connection.execute(text("PRAGMA user_version")).scalar_one()
            )
            artifact_batch.record_initial_database(
                connection,
                user_version=current_version,
            )
            if current_version > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"Database schema version {current_version} is newer "
                    f"than supported version {SCHEMA_VERSION}."
                )
            if current_version < 1:
                _upgrade_to_version_one(connection)
                current_version = 1
            if current_version < 2:
                _upgrade_to_version_two(connection)
                current_version = 2
            if current_version < 3:
                _upgrade_to_version_three(connection)
                current_version = 3
            if current_version < 4:
                _upgrade_to_version_four(connection)
                current_version = 4
            if current_version < 5:
                _upgrade_to_version_five(connection)
                current_version = 5
            if current_version < 6:
                _upgrade_to_version_six(connection)
                current_version = 6
            if current_version < 7:
                _upgrade_to_version_seven(connection)
                current_version = 7
            if current_version < 8:
                _upgrade_to_version_eight(connection)
                current_version = 8
            if current_version < 9:
                _upgrade_to_version_nine(connection)
                current_version = 9
            if current_version < 10:
                _upgrade_to_version_ten(connection)
                current_version = 10
            if current_version < 11:
                _upgrade_to_version_eleven(connection)
                current_version = 11
            if current_version < 12:
                _upgrade_to_version_twelve(connection)
                current_version = 12
            if current_version < 13:
                _upgrade_to_version_thirteen(connection)
                current_version = 13
            if current_version < 14:
                _upgrade_to_version_fourteen(connection)
                current_version = 14
            if current_version < 15:
                _upgrade_to_version_fifteen(connection)
                current_version = 15
            if current_version < 16:
                _upgrade_to_version_sixteen(
                    connection,
                    artifact_batch,
                    legacy_weekly_analysis_audit_root=(
                        legacy_weekly_analysis_audit_root
                    ),
                )
                current_version = 16
            if current_version < 17:
                _upgrade_to_version_seventeen(connection)
                current_version = 17
            if current_version < 18:
                _upgrade_to_version_eighteen(connection)
                current_version = 18
            if current_version < 19:
                _upgrade_to_version_nineteen(connection)
                current_version = 19
            if current_version < 20:
                _upgrade_to_version_twenty(connection)
                current_version = 20
            if current_version < 21:
                _upgrade_to_version_twenty_one(connection)
                current_version = 21
            if current_version < 22:
                _upgrade_to_version_twenty_two(connection)
                current_version = 22
            _validate_market_price_volume_source(connection)
            _validate_v2_schema(connection)
            _validate_v34_schema(connection)
            _validate_v341_schema(connection)
            _validate_v342_schema(connection)
            connection.exec_driver_sql(
                f"PRAGMA user_version = {current_version}"
            )
            connection.commit()
        except BaseException as primary:
            rollback_errors: list[str] = []
            try:
                if connection.in_transaction():
                    connection.rollback()
                # A commit-event/driver failure can mark SQLAlchemy's logical
                # transaction inactive while sqlite3 still owns BEGIN
                # IMMEDIATE.  Always clear that physical transaction too.
                connection.connection.rollback()
            except BaseException as exc:
                rollback_errors.append(
                    f"database rollback: {type(exc).__name__}: {exc}"
                )
            durable_state, detail = _durable_migration_state(
                engine,
                artifact_batch,
            )
            if durable_state == "committed":
                artifact_batch.finalize()
                return SCHEMA_VERSION
            if durable_state == "rolled_back":
                artifact_batch.compensate(
                    primary,
                    extra_errors=tuple(rollback_errors),
                )
                raise
            artifact_batch.raise_mixed_state(
                primary,
                "; ".join((*rollback_errors, detail)),
            )
        artifact_batch.finalize()
    return SCHEMA_VERSION
