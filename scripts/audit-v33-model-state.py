#!/usr/bin/env python3
"""Independent, read-only release audit for persisted V3.3 model state.

This script intentionally uses sqlite3 rather than the application ORM.  It
opens the database with SQLite URI ``mode=ro`` and additionally enables
``PRAGMA query_only`` so running the audit cannot advance training or analysis
state.  Only an explicitly requested Markdown report may be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "investment_lab.db"
ACTIVE_MARKETS = ("399006", "159941")
ALLOWED_MARKETS = frozenset(ACTIVE_MARKETS)
BENCHMARK_MARKET = "NDX"
SAMPLING_SEED = 330020
ANALYSIS_DAYS = 3653
HORIZON_WEEKS = 20
DAILY_CORRECTION_DECAY = (1.0, 0.75, 0.5, 0.25)
DAILY_CORRECTION_MAX_ABS = 0.03
EXPECTED_FEATURE_VERSION = "V3.3-20W-FULL100-DCT12-3-SAMPLED"
EXPECTED_METHODOLOGY_VERSION = "V3.3-PROGRESSIVE-PURGED-20W-3"
MODEL_STATE_HASH_FIELDS = (
    "market",
    "version",
    "iteration_number",
    "trained_through",
    "parent_state_hash",
    "feature_names",
    "medians",
    "means",
    "scales",
    "intercept",
    "coefficients",
    "stumps",
    "residual_p10",
    "residual_p50",
    "residual_p90",
    "hyperparameters",
    "optimizer_memory",
    "turn_lag_days",
    "training_sample_count",
)
EVALUATION_METRIC_COLUMNS = (
    "composite_loss",
    "wis_loss",
    "pinball_loss",
    "p50_path_error",
    "terminal_return_error",
    "brier_score",
    "calibration_error",
    "interval_coverage",
    "interval_width",
    "dif_turn_error_days",
    "price_turn_error_days",
    "coverage_3d",
    "coverage_5d",
    "coverage_10d",
    "turning_direction_f1",
    "false_turn_alert_rate",
)
TRAINING_IDENTITY_KEYS = {
    "iteration_count",
    "model_count",
    "optimizer_count",
    "optimizer_state_hash",
}
FULL_REQUIRED_METRICS = (
    "composite_loss",
    "wis_loss",
    "pinball_loss",
    "p50_path_error",
    "terminal_return_error",
    "brier_score",
    "calibration_error",
    "interval_coverage",
    "interval_width",
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def model_state_hash(parameters: Mapping[str, Any]) -> str:
    missing = [name for name in MODEL_STATE_HASH_FIELDS if name not in parameters]
    if missing:
        raise ValueError(f"model artifact is missing state fields: {missing}")
    return canonical_hash({name: parameters[name] for name in MODEL_STATE_HASH_FIELDS})


def parse_json(value: Any, *, context: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        raise ValueError(f"{context} is SQL NULL")
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON: {error}") from error


def parse_date(value: Any, *, context: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} is not an ISO date: {value!r}") from error


def parse_datetime(value: Any, *, context: str) -> datetime:
    text = str(value).replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{context} is not an ISO timestamp: {value!r}") from error
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def week_key(day: date) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def deterministic_sample(candidates: Sequence[date], market: str) -> date:
    if not candidates:
        raise ValueError("cannot sample an empty natural week")
    year, week, _ = candidates[0].isocalendar()
    digest = hashlib.sha256(
        f"{SAMPLING_SEED}:{market}:{year}-W{week:02d}".encode("utf-8")
    ).digest()
    return tuple(sorted(candidates))[int.from_bytes(digest[:8], "big") % len(candidates)]


def numbers(value: Any, *, context: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{context} is not a JSON array")
    if len(value) != HORIZON_WEEKS:
        raise ValueError(f"{context} has {len(value)} points, expected 20")
    result: list[float] = []
    for index, item in enumerate(value, start=1):
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context}[{index}] is not numeric") from error
        if not math.isfinite(number):
            raise ValueError(f"{context}[{index}] is not finite")
        result.append(number)
    return result


def json_equal(left: Any, right: Any) -> bool:
    return canonical_hash(left) == canonical_hash(right)


def open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    uri = f"file:{quote(resolved.as_posix(), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
        connection.close()
        raise RuntimeError("SQLite query_only could not be enabled")
    # Keep every table check on one coherent snapshot even if another process
    # is still preparing a database.  A release audit must not mix rows from
    # different commits.
    connection.execute("BEGIN")
    return connection


def reconstruct_warmup_anchor(
    connection: sqlite3.Connection,
    market: str,
    first_cutoff: date,
    random_seed: int,
) -> dict[str, Any]:
    """Independently rebuild the non-formal root from persisted snapshots."""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from backend.app.services.v33_runtime_service import (
        _snapshot_from_payload,
        build_warmup_anchor_payload,
    )
    from backend.app.services.v33_training_service import (
        V33TrainingService,
        build_training_points,
    )

    rows = list(
        connection.execute(
            "SELECT * FROM v33_feature_snapshots "
            "WHERE model_market=? AND feature_version=? AND cutoff_date<=? "
            "ORDER BY cutoff_date",
            (market, EXPECTED_FEATURE_VERSION, first_cutoff.isoformat()),
        )
    )
    if not rows:
        raise ValueError("warm-up build snapshots are missing")
    snapshots = []
    manifest: list[dict[str, Any]] = []
    for row in rows:
        payload = parse_json(row["feature_json"], context="warm-up feature_json")
        if canonical_hash(payload) != str(row["snapshot_hash"]):
            raise ValueError(f"warm-up snapshot {row['id']} content hash mismatch")
        snapshots.append(_snapshot_from_payload(payload))
        manifest.append(
            {
                "snapshot_id": str(row["id"]),
                "cutoff_date": str(row["cutoff_date"]),
                "snapshot_hash": str(row["snapshot_hash"]),
            }
        )
    points = build_training_points(snapshots, sampling_seed=random_seed)
    service = V33TrainingService()
    state, eligible = service.fit_warmup_anchor(
        market,
        points,
        first_formal_cutoff=first_cutoff,
    )
    return build_warmup_anchor_payload(
        state,
        snapshot_manifest=manifest,
        eligible_cutoff_dates=tuple(point.cutoff_date for point in eligible),
        sampling_seed=random_seed,
    )


def warmup_anchor_envelope_errors(
    anchor: Any,
    *,
    market: str,
    first_cutoff: date,
    sampling_seed: int,
) -> tuple[list[str], str | None]:
    """Validate the complete non-formal root envelope without trusting SQL links."""

    if not isinstance(anchor, dict):
        return ["root bootstrap warmup_anchor is missing"], None
    errors: list[str] = []
    body = {key: value for key, value in anchor.items() if key != "anchor_hash"}
    if anchor.get("anchor_hash") != canonical_hash(body):
        errors.append("anchor_hash mismatch")

    manifest = anchor.get("build_snapshot_manifest")
    manifest_cutoffs: list[str] = []
    if not isinstance(manifest, list) or not manifest:
        errors.append("build snapshot manifest is empty")
    else:
        if not all(isinstance(row, dict) for row in manifest):
            errors.append("build snapshot manifest contains a non-object row")
        else:
            manifest_cutoffs = [str(row.get("cutoff_date") or "") for row in manifest]
            if manifest_cutoffs != sorted(set(manifest_cutoffs)):
                errors.append("build snapshot manifest cutoffs are not unique and ordered")
            if any(
                not row.get("snapshot_id")
                or len(str(row.get("snapshot_hash") or "")) != 64
                for row in manifest
            ):
                errors.append("build snapshot manifest identity/hash is incomplete")
        if anchor.get("build_snapshot_count") != len(manifest):
            errors.append("build_snapshot_count mismatch")
        if anchor.get("manifest_hash") != canonical_hash(manifest):
            errors.append("manifest_hash mismatch")

    eligible = anchor.get("eligible_sample_cutoffs")
    if not isinstance(eligible, list):
        errors.append("eligible_sample_cutoffs is not an array")
        eligible = []
    else:
        eligible = [str(value) for value in eligible]
        if eligible != sorted(set(eligible)):
            errors.append("eligible sample cutoffs are not unique and ordered")
        if manifest_cutoffs and not set(eligible).issubset(manifest_cutoffs):
            errors.append("eligible sample cutoff is absent from build manifest")
    if anchor.get("real_matured_sample_count") != len(eligible):
        errors.append("real_matured_sample_count mismatch")

    state_payload = anchor.get("state_payload")
    anchor_state_hash: str | None = None
    if not isinstance(state_payload, dict):
        errors.append("complete state_payload is missing")
    else:
        try:
            recomputed_state_hash = model_state_hash(state_payload)
        except Exception as error:
            errors.append(str(error))
        else:
            anchor_state_hash = str(state_payload.get("state_hash") or "")
            if recomputed_state_hash != anchor_state_hash:
                errors.append("warm-up state_hash does not recompute")
            if anchor.get("state_hash") != anchor_state_hash:
                errors.append("anchor state_hash differs from state_payload")
        if int(state_payload.get("iteration_number", -1)) != 0:
            errors.append("warm-up state iteration_number is not 0")
        if state_payload.get("parent_state_hash") is not None:
            errors.append("warm-up state has a parent")
        if state_payload.get("market") != market:
            errors.append("warm-up state market mismatch")
        if state_payload.get("version") != anchor.get("version"):
            errors.append("warm-up state version mismatch")
        if str(state_payload.get("trained_through")) != first_cutoff.isoformat():
            errors.append("warm-up state trained_through mismatch")
        if int(state_payload.get("training_sample_count", -1)) != len(eligible):
            errors.append("warm-up state sample count mismatch")

    if anchor.get("kind") != "v33_non_formal_warmup_root":
        errors.append("warm-up anchor kind mismatch")
    if anchor.get("formal_iteration") is not False:
        errors.append("warm-up anchor is incorrectly marked formal")
    if anchor.get("market") != market or anchor.get("iteration_number") != 0:
        errors.append("warm-up anchor market/iteration identity mismatch")
    if anchor.get("parent_state_hash") is not None:
        errors.append("warm-up anchor parent must be NULL")
    if anchor.get("first_formal_cutoff") != first_cutoff.isoformat():
        errors.append("warm-up first_formal_cutoff mismatch")
    if anchor.get("feature_version") != EXPECTED_FEATURE_VERSION:
        errors.append("warm-up feature_version mismatch")
    if anchor.get("methodology_version") != EXPECTED_METHODOLOGY_VERSION:
        errors.append("warm-up methodology_version mismatch")
    if anchor.get("sampling_seed") != sampling_seed:
        errors.append("warm-up sampling_seed mismatch")
    return errors, anchor_state_hash


@dataclass(frozen=True)
class Finding:
    status: str
    check: str
    detail: str


class Audit:
    def __init__(self, database: Path) -> None:
        self.database = database
        self.findings: list[Finding] = []
        self.market_summary: dict[str, dict[str, Any]] = {}

    def add(self, ok: bool, check: str, detail: str) -> None:
        self.findings.append(Finding("PASS" if ok else "FAIL", check, detail))

    def fail(self, check: str, detail: str) -> None:
        self.add(False, check, detail)

    @property
    def passed(self) -> bool:
        return bool(self.findings) and not any(row.status == "FAIL" for row in self.findings)

    def run(self) -> None:
        try:
            connection = open_read_only(self.database)
        except Exception as error:  # report an operational failure as an audit failure
            self.fail("database.open_read_only", str(error))
            return
        try:
            self.add(True, "database.open_read_only", "SQLite URI mode=ro; PRAGMA query_only=1")
            required = {
                "instruments",
                "market_prices",
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
            }
            present = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            missing = sorted(required - present)
            self.add(not missing, "database.required_tables", "all present" if not missing else f"missing: {missing}")
            if missing:
                return
            self._audit_instrument_roles(connection)
            self._audit_market_names(connection)
            for market in ACTIVE_MARKETS:
                self._audit_market(connection, market)
            self._audit_analysis(connection)
        finally:
            connection.close()

    def _audit_instrument_roles(self, connection: sqlite3.Connection) -> None:
        rows = list(
            connection.execute(
                "SELECT roles.*, target.code AS instrument_code, "
                "benchmark.code AS benchmark_code "
                "FROM v33_instrument_roles AS roles "
                "JOIN instruments AS target ON target.id=roles.instrument_id "
                "LEFT JOIN instruments AS benchmark ON benchmark.id=roles.benchmark_instrument_id "
                "WHERE roles.is_active=1"
            )
        )
        errors: list[str] = []
        if any(str(row["model_market"]) == BENCHMARK_MARKET for row in rows):
            errors.append("NDX appears as an active model_market")
        qdii = [row for row in rows if str(row["model_market"]) == "159941"]
        tradable = [
            row
            for row in qdii
            if str(row["role"]) == "tradable"
            and str(row["instrument_code"]) == "159941"
            and str(row["benchmark_code"]) == BENCHMARK_MARKET
        ]
        benchmark = [
            row
            for row in qdii
            if str(row["role"]) == "benchmark"
            and str(row["instrument_code"]) == BENCHMARK_MARKET
            and row["benchmark_code"] is None
        ]
        if len(tradable) != 1:
            errors.append(
                "159941 must have exactly one active tradable role whose benchmark is NDX"
            )
        if len(benchmark) != 1 or len(qdii) != 2:
            errors.append(
                "159941 must have exactly one active NDX benchmark role and no extra active roles"
            )
        if any(
            str(row["role"]) == "tradable"
            and str(row["instrument_code"]) != str(row["model_market"])
            for row in rows
        ):
            errors.append("an active tradable role uses another instrument")
        self.add(
            not errors,
            "markets.instrument_roles",
            "159941 is the sole active QDII tradable and NDX is its benchmark only"
            if not errors
            else "; ".join(errors),
        )

    def _audit_market_names(self, connection: sqlite3.Connection) -> None:
        tables = (
            "v33_feature_snapshots",
            "v33_training_runs",
            "v33_training_iterations",
            "v33_training_checkpoints",
            "v33_model_versions",
            "v33_optimizer_states",
            "v33_forecasts",
            "v33_analysis_runs",
        )
        unexpected: dict[str, list[str]] = {}
        observed: dict[str, list[str]] = {}
        for table in tables:
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is None:
                continue
            values = [
                str(row[0])
                for row in connection.execute(
                    f'SELECT DISTINCT model_market FROM "{table}" WHERE model_market IS NOT NULL'
                )
            ]
            observed[table] = sorted(values)
            bad = sorted(set(values) - ALLOWED_MARKETS)
            if bad:
                unexpected[table] = bad
        self.add(
            not unexpected,
            "markets.active_namespaces",
            f"active namespaces by table: {observed}"
            if not unexpected
            else f"unexpected active namespaces (NDX is forbidden): {unexpected}",
        )

    def _instrument_id(self, connection: sqlite3.Connection, market: str) -> int | None:
        row = connection.execute("SELECT id FROM instruments WHERE code=?", (market,)).fetchone()
        return None if row is None else int(row[0])

    def _expected_formal_samples(
        self, connection: sqlite3.Connection, market: str, instrument_id: int
    ) -> list[date]:
        actual = {
            parse_date(row[0], context=f"{market} market_prices.trade_date")
            for row in connection.execute(
                "SELECT trade_date FROM market_prices "
                "WHERE instrument_id=? AND timeframe='daily' ORDER BY trade_date",
                (instrument_id,),
            )
        }
        if not actual:
            raise ValueError("no real daily target bars")
        try:
            import exchange_calendars
        except ImportError as error:
            raise ValueError("exchange_calendars is required for the XSHG audit") from error
        calendar = exchange_calendars.get_calendar("XSHG")
        calendar_start = min(actual) - timedelta(days=min(actual).isoweekday() - 1)
        calendar_end = max(actual) + timedelta(days=7 - max(actual).isoweekday())
        expected_sessions = [
            session.date()
            for session in calendar.sessions_in_range(calendar_start, calendar_end)
        ]
        grouped: dict[tuple[int, int], list[date]] = {}
        for session in expected_sessions:
            grouped.setdefault(session.isocalendar()[:2], []).append(session)
        complete = [
            sessions
            for _, sessions in sorted(grouped.items())
            if sessions and set(sessions).issubset(actual)
        ]
        if not complete:
            raise ValueError("no actual XSHG-complete natural weeks")
        samples = [deterministic_sample(sessions, market) for sessions in complete]
        formal_start = samples[-1] - timedelta(days=ANALYSIS_DAYS)
        return [sample for sample in samples if sample >= formal_start]

    def _audit_warmup_anchor(
        self,
        connection: sqlite3.Connection,
        market: str,
        first_iteration: sqlite3.Row,
        first_model: sqlite3.Row | None,
    ) -> str | None:
        errors: list[str] = []
        anchor_hash: str | None = None
        try:
            if first_model is None:
                raise ValueError("iteration-1 model artifact is missing")
            if first_iteration["parent_iteration_number"] is not None:
                errors.append("iteration 1 has a formal parent_iteration_number")
            if first_model["parent_version"] is not None:
                errors.append("iteration-1 model has a formal parent_version")
            run = connection.execute(
                "SELECT * FROM v33_training_runs WHERE id=?",
                (first_iteration["training_run_id"],),
            ).fetchone()
            if run is None:
                raise ValueError("iteration-1 bootstrap run is missing")
            if str(run["run_type"]) != "bootstrap" or run["source_parent_iteration"] is not None:
                errors.append("iteration 1 is not owned by a root bootstrap run")
            result = parse_json(run["result_json"], context="root bootstrap result_json")
            anchor = result.get("warmup_anchor")
            if not isinstance(anchor, dict):
                raise ValueError("root bootstrap warmup_anchor is missing")
            first_cutoff = parse_date(first_iteration["cutoff_date"], context="first cutoff")
            audit_json = parse_json(first_iteration["audit_json"], context="iteration-1 audit_json")
            parameters = parse_json(first_model["parameters_json"], context="iteration-1 parameters_json")
            envelope_errors, anchor_hash = warmup_anchor_envelope_errors(
                anchor,
                market=market,
                first_cutoff=first_cutoff,
                sampling_seed=int(first_model["random_seed"]),
            )
            errors.extend(envelope_errors)
            if anchor_hash != audit_json.get("parent_state_hash"):
                errors.append("iteration-1 audit does not link to warm-up root")
            if anchor_hash != parameters.get("parent_state_hash"):
                errors.append("iteration-1 model does not link to warm-up root")
            registered_anchor = connection.execute(
                "SELECT COUNT(*) FROM v33_model_versions WHERE model_market=? AND version=?",
                (market, str(anchor.get("version") or "")),
            ).fetchone()[0]
            if int(registered_anchor) != 0:
                errors.append("non-formal warm-up root was registered as a formal model version")
            reconstructed = reconstruct_warmup_anchor(
                connection,
                market,
                first_cutoff,
                int(first_model["random_seed"]),
            )
            if not json_equal(anchor, reconstructed):
                errors.append("persisted root differs from deterministic snapshot reconstruction")
        except Exception as error:
            errors.append(str(error))
        self.add(
            not errors,
            f"{market}.warmup_anchor",
            (
                "non-formal iteration 0 was deterministically rebuilt, fully persisted, and linked to iteration 1"
                if not errors
                else "; ".join(errors[:12])
            ),
        )
        return anchor_hash if not errors else None

    def _audit_market(self, connection: sqlite3.Connection, market: str) -> None:
        prefix = f"{market}."
        instrument_id = self._instrument_id(connection, market)
        self.add(instrument_id is not None, prefix + "instrument", f"instrument_id={instrument_id}")
        if instrument_id is None:
            return

        iterations = list(
            connection.execute(
                "SELECT * FROM v33_training_iterations WHERE model_market=? "
                "ORDER BY iteration_number",
                (market,),
            )
        )
        if not iterations:
            self.fail(prefix + "trained", "no persisted training iterations; production model is untrained")
            self.market_summary[market] = {"iterations": 0}
            return
        self.add(True, prefix + "trained", f"{len(iterations)} persisted iterations")

        run_mismatches = int(
            connection.execute(
                "SELECT COUNT(*) FROM v33_training_runs "
                "WHERE model_market=? AND target_instrument_id<>?",
                (market, instrument_id),
            ).fetchone()[0]
        )
        self.add(
            run_mismatches == 0,
            prefix + "training_targets",
            "all training runs target the market's own instrument"
            if run_mismatches == 0
            else f"{run_mismatches} training run(s) target another instrument",
        )

        expected_numbers = list(range(1, len(iterations) + 1))
        actual_numbers = [int(row["iteration_number"]) for row in iterations]
        parents = [row["parent_iteration_number"] for row in iterations]
        expected_parents: list[int | None] = [None, *range(1, len(iterations))]
        self.add(
            actual_numbers == expected_numbers and parents == expected_parents,
            prefix + "iteration_chain",
            f"iterations={actual_numbers[0]}..{actual_numbers[-1]}; parent_iteration_number contiguous"
            if actual_numbers == expected_numbers and parents == expected_parents
            else f"numbers/parents are not contiguous: numbers={actual_numbers[:5]}.., parents={parents[:5]}..",
        )

        try:
            expected_samples = self._expected_formal_samples(connection, market, instrument_id)
        except Exception as error:
            self.fail(prefix + "formal_weeks", str(error))
            expected_samples = []
        actual_dates = [parse_date(row["cutoff_date"], context="iteration cutoff_date") for row in iterations]
        actual_weeks = [str(row["week_key"]) for row in iterations]
        date_match = actual_dates == expected_samples
        week_match = actual_weeks == [week_key(day) for day in expected_samples]
        self.add(
            bool(expected_samples) and date_match and week_match,
            prefix + "formal_weeks",
            (
                f"{len(actual_dates)} iterations exactly match {len(expected_samples)} XSHG-complete "
                f"natural weeks in the latest {ANALYSIS_DAYS}-day window; seed={SAMPLING_SEED}; "
                f"range={actual_dates[0]}..{actual_dates[-1]}"
                if expected_samples and date_match and week_match
                else f"stored={len(actual_dates)}, expected={len(expected_samples)}, "
                f"first mismatch={self._first_mismatch(actual_dates, expected_samples)}"
            ),
        )

        models = list(
            connection.execute(
                "SELECT * FROM v33_model_versions WHERE model_market=? ORDER BY trained_through_date, created_at, id",
                (market,),
            )
        )
        models_by_version = {str(row["version"]): row for row in models}
        self.add(
            len(models) == len(iterations) and len(models_by_version) == len(models),
            prefix + "model_count",
            f"models={len(models)}, iterations={len(iterations)}, unique_versions={len(models_by_version)}",
        )
        first_model = models_by_version.get(str(iterations[0]["champion_model_version"]))
        warmup_state_hash = self._audit_warmup_anchor(
            connection,
            market,
            iterations[0],
            first_model,
        )

        snapshots_ok = True
        snapshot_errors: list[str] = []
        chain_ok = True
        chain_errors: list[str] = []
        model_hash_ok = True
        model_hash_errors: list[str] = []
        optimizer_history_ok = True
        optimizer_history_errors: list[str] = []
        forecast_ok = True
        forecast_errors: list[str] = []
        previous_hash: str | None = warmup_state_hash
        previous_version: str | None = None

        for index, iteration in enumerate(iterations, start=1):
            label = f"iteration {index}"
            if str(iteration["status"]) != "completed":
                chain_ok = False
                chain_errors.append(f"{label}: status={iteration['status']!r}")
            audit_json = parse_json(iteration["audit_json"], context=f"{label} audit_json")
            if audit_json.get("sampling_seed") != SAMPLING_SEED:
                chain_ok = False
                chain_errors.append(f"{label}: sampling_seed={audit_json.get('sampling_seed')!r}")
            if audit_json.get("parent_state_hash") != previous_hash:
                chain_ok = False
                chain_errors.append(f"{label}: parent_state_hash does not match predecessor")

            snapshot = connection.execute(
                "SELECT * FROM v33_feature_snapshots WHERE id=?",
                (iteration["feature_snapshot_id"],),
            ).fetchone()
            if snapshot is None:
                snapshots_ok = False
                snapshot_errors.append(f"{label}: missing feature snapshot")
            else:
                try:
                    snapshot_payload = parse_json(snapshot["feature_json"], context=f"{label} feature_json")
                    leakage = parse_json(snapshot["leakage_audit_json"], context=f"{label} leakage_audit_json")
                    cutoff = parse_date(iteration["cutoff_date"], context=f"{label} cutoff")
                    source_max = parse_date(
                        leakage.get("source_data_max_date", snapshot_payload.get("source_data_max_date")),
                        context=f"{label} source_data_max_date",
                    )
                    weekly_as_of = parse_date(snapshot["weekly_data_as_of"], context=f"{label} weekly_data_as_of")
                    monday = cutoff - timedelta(days=cutoff.isoweekday() - 1)
                    daily_sequence = snapshot_payload.get("daily_sequence")
                    reasons: list[str] = []
                    if str(snapshot["model_market"]) != market:
                        reasons.append("snapshot market mismatch")
                    if int(snapshot["target_instrument_id"]) != instrument_id:
                        reasons.append("target instrument mismatch")
                    if parse_date(snapshot["cutoff_date"], context="snapshot cutoff") != cutoff:
                        reasons.append("snapshot cutoff mismatch")
                    if source_max > cutoff:
                        reasons.append("source_data_max_date exceeds cutoff")
                    if weekly_as_of >= monday:
                        reasons.append("weekly cutoff is not before sampled natural week")
                    if int(snapshot["daily_session_count"]) != 100:
                        reasons.append(f"daily_session_count={snapshot['daily_session_count']}")
                    if str(snapshot["feature_version"]) != EXPECTED_FEATURE_VERSION:
                        reasons.append(
                            f"feature_version={snapshot['feature_version']!r}, "
                            f"expected {EXPECTED_FEATURE_VERSION!r}"
                        )
                    if not isinstance(daily_sequence, list) or len(daily_sequence) != 100:
                        reasons.append("daily_sequence is not exactly 100 sessions")
                    if canonical_hash(snapshot_payload) != str(snapshot["snapshot_hash"]):
                        reasons.append("snapshot_hash mismatch")
                    if parse_datetime(snapshot["source_max_available_at"], context="source max available") > parse_datetime(snapshot["cutoff_available_at"], context="cutoff available"):
                        reasons.append("source availability exceeds cutoff")
                    if reasons:
                        snapshots_ok = False
                        snapshot_errors.append(f"{label}: {', '.join(reasons)}")
                except Exception as error:
                    snapshots_ok = False
                    snapshot_errors.append(f"{label}: {error}")

            version = str(iteration["champion_model_version"])
            model = models_by_version.get(version)
            if model is None:
                model_hash_ok = False
                model_hash_errors.append(f"{label}: missing model {version}")
                current_hash = str(audit_json.get("state_hash") or "")
            else:
                try:
                    parameters = parse_json(model["parameters_json"], context=f"{label} parameters_json")
                    recomputed = model_state_hash(parameters)
                    current_hash = str(parameters.get("state_hash") or "")
                    reasons = []
                    if recomputed != current_hash:
                        reasons.append("recomputed state_hash mismatch")
                    if str(model["artifact_hash"]) != current_hash:
                        reasons.append("artifact_hash mismatch")
                    if str(audit_json.get("state_hash")) != current_hash:
                        reasons.append("iteration audit state_hash mismatch")
                    if parameters.get("parent_state_hash") != previous_hash:
                        reasons.append("model parent_state_hash mismatch")
                    if model["parent_version"] != previous_version:
                        reasons.append("model parent_version mismatch")
                    if int(parameters.get("iteration_number", -1)) != index:
                        reasons.append("model iteration_number mismatch")
                    if parse_date(parameters.get("trained_through"), context="trained_through") != actual_dates[index - 1]:
                        reasons.append("model trained_through mismatch")
                    if int(model["random_seed"]) != SAMPLING_SEED:
                        reasons.append("model random_seed mismatch")
                    if str(model["feature_version"]) != EXPECTED_FEATURE_VERSION:
                        reasons.append(
                            f"model feature_version={model['feature_version']!r}, "
                            f"expected {EXPECTED_FEATURE_VERSION!r}"
                        )
                    if str(model["methodology_version"]) != EXPECTED_METHODOLOGY_VERSION:
                        reasons.append(
                            f"model methodology_version={model['methodology_version']!r}, "
                            f"expected {EXPECTED_METHODOLOGY_VERSION!r}"
                        )
                    iteration_optimizer = parse_json(
                        iteration["optimizer_state_json"], context=f"{label} optimizer_state_json"
                    )
                    if not json_equal(iteration_optimizer, parameters.get("optimizer_memory")):
                        optimizer_history_ok = False
                        optimizer_history_errors.append(f"{label}: optimizer memory differs from model artifact")
                    if reasons:
                        model_hash_ok = False
                        model_hash_errors.append(f"{label}: {', '.join(reasons)}")
                except Exception as error:
                    model_hash_ok = False
                    model_hash_errors.append(f"{label}: {error}")
                    current_hash = str(audit_json.get("state_hash") or "")
            if previous_hash is not None and not previous_hash:
                chain_ok = False
            previous_hash = current_hash
            previous_version = version

            row = connection.execute(
                "SELECT * FROM v33_forecasts WHERE training_iteration_id=?",
                (iteration["id"],),
            ).fetchall()
            if len(row) != 1:
                forecast_ok = False
                forecast_errors.append(f"{label}: training forecast rows={len(row)}, expected 1")
            else:
                errors = self._training_forecast_errors(row[0], iteration, current_hash)
                if str(row[0]["model_market"]) != market:
                    errors.append("forecast market mismatch")
                if int(row[0]["target_instrument_id"]) != instrument_id:
                    errors.append("forecast target instrument mismatch")
                if parse_date(row[0]["forecast_date"], context="forecast date") != actual_dates[index - 1]:
                    errors.append("forecast date differs from iteration cutoff")
                if str(row[0]["model_version"]) != version:
                    errors.append("forecast model version differs from iteration Champion")
                if errors:
                    forecast_ok = False
                    forecast_errors.append(f"{label}: {', '.join(errors)}")

        self.add(snapshots_ok, prefix + "feature_snapshots", "all snapshots are PIT-safe, prior-week, and exactly 100 sessions" if snapshots_ok else "; ".join(snapshot_errors[:8]))
        self.add(chain_ok, prefix + "state_parent_chain", f"warm-up root plus {len(iterations)} formal state hashes form one continuous chain" if chain_ok else "; ".join(chain_errors[:8]))
        self.add(model_hash_ok, prefix + "model_artifacts", "all artifact hashes and model state hashes recompute" if model_hash_ok else "; ".join(model_hash_errors[:8]))
        self.add(optimizer_history_ok, prefix + "optimizer_history", "every iteration optimizer payload matches its immutable model artifact" if optimizer_history_ok else "; ".join(optimizer_history_errors[:8]))
        self.add(forecast_ok, prefix + "forecasts", "all issued training forecasts pass 20-week/hash/component constraints" if forecast_ok else "; ".join(forecast_errors[:8]))

        self._audit_final_optimizer(connection, market, iterations[-1], models_by_version)
        pending, full = self._audit_maturity(connection, market, iterations)
        champion_count = sum(str(row["status"]) == "champion" for row in models)
        latest_model = models_by_version.get(str(iterations[-1]["champion_model_version"]))
        champion_ok = (
            champion_count == 1
            and latest_model is not None
            and str(latest_model["status"]) == "champion"
        )
        self.add(champion_ok, prefix + "champion", f"champion_rows={champion_count}; latest iteration owns Champion")
        self.market_summary[market] = {
            "iterations": len(iterations),
            "expected_iterations": len(expected_samples),
            "first_week": actual_weeks[0],
            "last_week": actual_weeks[-1],
            "full": full,
            "pending": pending,
            "last_state_hash": previous_hash,
        }

    @staticmethod
    def _first_mismatch(left: Sequence[Any], right: Sequence[Any]) -> str:
        for index, (a, b) in enumerate(zip(left, right), start=1):
            if a != b:
                return f"index {index}: stored={a}, expected={b}"
        if len(left) != len(right):
            return f"common prefix {min(len(left), len(right))}; lengths differ"
        return "none"

    def _training_forecast_errors(
        self, forecast: sqlite3.Row, iteration: sqlite3.Row, state_hash: str
    ) -> list[str]:
        errors: list[str] = []
        try:
            payload = parse_json(forecast["payload_json"], context="forecast payload_json")
            iteration_payload = parse_json(iteration["forecast_json"], context="iteration forecast_json")
            if canonical_hash(payload) != str(forecast["forecast_hash"]):
                errors.append("forecast_hash mismatch")
            if not json_equal(payload, iteration_payload):
                errors.append("forecast payload differs from immutable iteration payload")
            if str(payload.get("model_state_hash")) != state_hash:
                errors.append("forecast model_state_hash is not traceable to iteration artifact")
            paths = {
                name: numbers(payload.get(name), context=name)
                for name in ("p10", "p50", "p90", "expected", "weekly_base", "daily_correction")
            }
            for index, (low, mid, high) in enumerate(
                zip(paths["p10"], paths["p50"], paths["p90"]), start=1
            ):
                if low > mid + 1e-12 or mid > high + 1e-12:
                    errors.append(f"crossed quantiles at week {index}")
                    break
            for index, (base, correction, expected) in enumerate(
                zip(paths["weekly_base"], paths["daily_correction"], paths["expected"]), start=1
            ):
                if not math.isclose(base + correction, expected, rel_tol=0.0, abs_tol=1e-10):
                    errors.append(f"weekly_base + daily_correction != expected at week {index}")
                    break
            for index, correction in enumerate(paths["daily_correction"]):
                if index < 4:
                    limit = DAILY_CORRECTION_MAX_ABS * DAILY_CORRECTION_DECAY[index]
                    if abs(correction) > limit + 1e-12:
                        errors.append(f"daily correction exceeds week {index + 1} limit {limit}")
                elif correction != 0.0:
                    errors.append(f"daily correction is not exactly zero at week {index + 1}")
                    break
            table_paths = {
                "p10": forecast["p10_path_json"],
                "p50": forecast["p50_path_json"],
                "p90": forecast["p90_path_json"],
                "expected": forecast["expected_path_json"],
            }
            for name, raw in table_paths.items():
                if not json_equal(parse_json(raw, context=f"forecast {name} column"), payload.get(name)):
                    errors.append(f"{name} table column differs from payload")
            if int(forecast["horizon_weeks"]) != HORIZON_WEEKS:
                errors.append("horizon_weeks is not 20")
        except Exception as error:
            errors.append(str(error))
        return errors

    def _audit_final_optimizer(
        self,
        connection: sqlite3.Connection,
        market: str,
        last_iteration: sqlite3.Row,
        models_by_version: Mapping[str, sqlite3.Row],
    ) -> None:
        rows = list(
            connection.execute(
                "SELECT * FROM v33_optimizer_states WHERE model_market=?", (market,)
            )
        )
        errors: list[str] = []
        if len(rows) != 1:
            errors.append(f"optimizer rows={len(rows)}, expected 1")
        else:
            optimizer = rows[0]
            version = str(last_iteration["champion_model_version"])
            model = models_by_version.get(version)
            if model is None:
                errors.append("latest model artifact is missing")
            else:
                parameters = parse_json(model["parameters_json"], context="latest model parameters")
                if int(optimizer["iteration_number"]) != int(last_iteration["iteration_number"]):
                    errors.append("optimizer iteration differs from latest iteration")
                if str(optimizer["last_training_week_key"]) != str(last_iteration["week_key"]):
                    errors.append("optimizer week differs from latest iteration")
                if str(optimizer["champion_model_version"]) != version:
                    errors.append("optimizer Champion differs from latest iteration")
                if str(optimizer["state_hash"]) != str(model["artifact_hash"]):
                    errors.append("optimizer state_hash differs from latest model")
                memory = parse_json(optimizer["optimizer_memory_json"], context="optimizer memory")
                if not json_equal(memory, parameters.get("optimizer_memory")):
                    errors.append("final optimizer memory differs from latest model")
        self.add(
            not errors,
            f"{market}.final_optimizer",
            "final optimizer, Champion, week, model hash, and memory agree"
            if not errors
            else "; ".join(errors),
        )

    def _audit_maturity(
        self, connection: sqlite3.Connection, market: str, iterations: Sequence[sqlite3.Row]
    ) -> tuple[int, int]:
        errors: list[str] = []
        statuses = [str(row["maturity_status"]) for row in iterations]
        pending = statuses.count("pending")
        full = statuses.count("full")
        if len(iterations) < HORIZON_WEEKS:
            errors.append("fewer than 20 iterations")
        if statuses[-HORIZON_WEEKS:] != ["pending"] * HORIZON_WEEKS:
            errors.append("latest 20 iterations are not all pending")
        if any(status != "full" for status in statuses[:-HORIZON_WEEKS]):
            errors.append("an iteration before the latest 20 is not full")
        if pending != HORIZON_WEEKS:
            errors.append(f"pending count={pending}, expected 20")
        if full != max(0, len(iterations) - HORIZON_WEEKS):
            errors.append(f"full count={full}, expected {len(iterations) - HORIZON_WEEKS}")

        for iteration in iterations:
            forecast = connection.execute(
                "SELECT * FROM v33_forecasts WHERE training_iteration_id=?",
                (iteration["id"],),
            ).fetchone()
            if forecast is None:
                continue
            evaluation_rows = list(
                connection.execute(
                    "SELECT * FROM v33_model_evaluations WHERE forecast_id=?", (forecast["id"],)
                )
            )
            if len(evaluation_rows) != 1:
                errors.append(f"iteration {iteration['iteration_number']}: evaluation rows={len(evaluation_rows)}")
                continue
            evaluation = evaluation_rows[0]
            status = str(iteration["maturity_status"])
            if str(forecast["maturity_status"]) != status or str(evaluation["maturity_status"]) != status:
                errors.append(f"iteration {iteration['iteration_number']}: maturity status disagreement")
            if status == "pending":
                iteration_metrics = (
                    iteration["composite_loss"],
                    iteration["wis_loss"],
                    iteration["price_turn_error_days"],
                    iteration["dif_zero_error_days"],
                )
                if any(value is not None for value in iteration_metrics):
                    errors.append(f"iteration {iteration['iteration_number']}: pending iteration SQL metric is non-NULL")
                if any(evaluation[name] is not None for name in EVALUATION_METRIC_COLUMNS):
                    errors.append(f"iteration {iteration['iteration_number']}: pending evaluation SQL metric is non-NULL")
                if evaluation["evaluated_at"] is not None:
                    errors.append(f"iteration {iteration['iteration_number']}: pending evaluated_at is non-NULL")
                payload = parse_json(evaluation["payload_json"], context="pending evaluation payload")
                if isinstance(payload, dict) and any(value is not None for value in payload.values()):
                    errors.append(f"iteration {iteration['iteration_number']}: pending payload contains evaluation values")
            elif status == "full":
                if iteration["composite_loss"] is None or iteration["wis_loss"] is None:
                    errors.append(f"iteration {iteration['iteration_number']}: full iteration lacks measured loss")
                missing = [name for name in FULL_REQUIRED_METRICS if evaluation[name] is None]
                if missing or evaluation["evaluated_at"] is None:
                    errors.append(f"iteration {iteration['iteration_number']}: full evaluation missing {missing or ['evaluated_at']}")
                payload = parse_json(evaluation["payload_json"], context="full evaluation payload")
                if not isinstance(payload, dict) or payload.get("composite_loss") is None:
                    errors.append(f"iteration {iteration['iteration_number']}: full payload lacks real composite_loss")

        self.add(
            not errors,
            f"{market}.maturity",
            f"full={full}; latest pending={pending}; pending SQL metrics are all NULL"
            if not errors
            else "; ".join(errors[:12]),
        )
        return pending, full

    def _audit_analysis(self, connection: sqlite3.Connection) -> None:
        for market in ACTIVE_MARKETS:
            rows = list(
                connection.execute(
                    "SELECT * FROM v33_analysis_runs WHERE model_market=? AND status='completed' "
                    "ORDER BY completed_at",
                    (market,),
                )
            )
            if not rows:
                self.fail(
                    f"{market}.analysis",
                    "no completed persisted analysis; live forecast and before/after training identity are unverified",
                )
                continue
            errors: list[str] = []
            for row in rows:
                label = f"analysis {row['id']}"
                try:
                    result = parse_json(row["result_json"], context=f"{label} result_json")
                    before = result.get("training_identity_before")
                    after = result.get("training_identity_after")
                    if not isinstance(before, dict) or not isinstance(after, dict):
                        errors.append(f"{label}: training identity before/after field missing")
                    elif set(before) != TRAINING_IDENTITY_KEYS or set(after) != TRAINING_IDENTITY_KEYS:
                        errors.append(
                            f"{label}: training identity keys must be exactly {sorted(TRAINING_IDENTITY_KEYS)}"
                        )
                    elif before != after or result.get("training_identity_unchanged") is not True:
                        errors.append(f"{label}: training identity changed")
                    forecast = connection.execute(
                        "SELECT * FROM v33_forecasts WHERE id=?", (row["forecast_id"],)
                    ).fetchone()
                    if forecast is None:
                        errors.append(f"{label}: persisted forecast missing")
                        continue
                    if str(forecast["maturity_status"]) != "live":
                        errors.append(f"{label}: forecast is not live")
                    instrument_id = self._instrument_id(connection, market)
                    if instrument_id is None or int(row["target_instrument_id"]) != instrument_id:
                        errors.append(f"{label}: analysis targets another instrument")
                    if instrument_id is None or int(forecast["target_instrument_id"]) != instrument_id:
                        errors.append(f"{label}: live forecast targets another instrument")
                    if forecast["training_iteration_id"] is not None:
                        errors.append(f"{label}: live forecast has training_iteration_id")
                    forecast_payload = parse_json(forecast["payload_json"], context=f"{label} forecast payload")
                    if canonical_hash(forecast_payload) != str(forecast["forecast_hash"]):
                        errors.append(f"{label}: forecast_hash mismatch")
                    model = result.get("model")
                    if not isinstance(model, dict) or not model.get("version") or not model.get("state_hash"):
                        errors.append(f"{label}: model trace fields missing")
                        continue
                    artifact = connection.execute(
                        "SELECT * FROM v33_model_versions WHERE model_market=? AND version=?",
                        (market, str(model["version"])),
                    ).fetchone()
                    if artifact is None or str(artifact["artifact_hash"]) != str(model["state_hash"]):
                        errors.append(f"{label}: result model hash is not traceable to an artifact")
                    if str(row["model_version"]) != str(model["version"]):
                        errors.append(f"{label}: analysis row model version mismatch")
                    if not str(forecast["model_version"]).startswith(str(model["version"]) + "-A-"):
                        errors.append(f"{label}: live issuance model version lacks analysis suffix")
                    path = result.get("path")
                    if not isinstance(path, dict):
                        errors.append(f"{label}: path payload missing")
                    else:
                        synthetic_iteration = {"forecast_json": json.dumps(result)}
                        component_errors = self._live_path_errors(path)
                        errors.extend(f"{label}: {message}" for message in component_errors)
                        for name, column in (
                            ("p10", "p10_path_json"),
                            ("p50", "p50_path_json"),
                            ("p90", "p90_path_json"),
                            ("expected", "expected_path_json"),
                        ):
                            if not json_equal(
                                parse_json(forecast[column], context=f"{label} {column}"),
                                path.get(name),
                            ):
                                errors.append(f"{label}: forecast {column} differs from analysis path.{name}")
                    if not result.get("data_quality", {}).get("analysis_as_of"):
                        # Current runtime also stores analysis_as_of in the feature payload;
                        # accepting either location avoids relying on display-only nesting.
                        snapshot = connection.execute(
                            "SELECT * FROM v33_feature_snapshots WHERE id=?",
                            (row["feature_snapshot_id"],),
                        ).fetchone()
                        snapshot_payload = (
                            {} if snapshot is None else parse_json(snapshot["feature_json"], context="live snapshot")
                        )
                        provenance = snapshot_payload.get("provenance", {}) if isinstance(snapshot_payload, dict) else {}
                        if not provenance.get("analysis_as_of"):
                            errors.append(f"{label}: analysis_as_of missing from persisted live snapshot/result")
                except Exception as error:
                    errors.append(f"{label}: {error}")
            self.add(
                not errors,
                f"{market}.analysis",
                f"{len(rows)} completed analyses preserve training identity and trace live forecasts"
                if not errors
                else "; ".join(errors[:12]),
            )

    @staticmethod
    def _live_path_errors(path: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        try:
            paths = {
                name: numbers(path.get(name), context=f"analysis path.{name}")
                for name in ("p10", "p50", "p90", "expected", "weekly_base", "daily_correction")
            }
            for index, (low, mid, high) in enumerate(zip(paths["p10"], paths["p50"], paths["p90"]), start=1):
                if low > mid + 1e-12 or mid > high + 1e-12:
                    errors.append(f"crossed live quantiles at week {index}")
                    break
            for index, (base, correction, expected) in enumerate(
                zip(paths["weekly_base"], paths["daily_correction"], paths["expected"]), start=1
            ):
                if not math.isclose(base + correction, expected, rel_tol=0.0, abs_tol=1e-10):
                    errors.append(f"live component sum mismatch at week {index}")
                    break
            for index, correction in enumerate(paths["daily_correction"]):
                limit = DAILY_CORRECTION_MAX_ABS * DAILY_CORRECTION_DECAY[index] if index < 4 else 0.0
                if index >= 4 and correction != 0.0:
                    errors.append(f"live daily correction is not zero at week {index + 1}")
                    break
                if index < 4 and abs(correction) > limit + 1e-12:
                    errors.append(f"live daily correction exceeds week {index + 1} limit")
        except Exception as error:
            errors.append(str(error))
        return errors

    def markdown(self) -> str:
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        status = "PASS" if self.passed else "FAIL"
        lines = [
            "# V3.3 Model State Read-only Audit",
            "",
            f"- Result: **{status}**",
            f"- Database: `{self.database.resolve()}`",
            f"- Audited at (UTC): `{timestamp}`",
            "- Access: SQLite URI `mode=ro` + `PRAGMA query_only=ON`",
            f"- Active targets: `{ACTIVE_MARKETS[0]}`, `{ACTIVE_MARKETS[1]}`; `NDX` is benchmark/read-only only",
            f"- Forecast horizon: `{HORIZON_WEEKS}` weeks; sampling seed: `{SAMPLING_SEED}`",
            f"- Required feature identity: `{EXPECTED_FEATURE_VERSION}`",
            f"- Required methodology identity: `{EXPECTED_METHODOLOGY_VERSION}`",
            "",
            "## Market summary",
            "",
            "| Market | Iterations | Expected | Full | Pending | Formal range |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for market in ACTIVE_MARKETS:
            summary = self.market_summary.get(market, {})
            formal_range = (
                f"{summary.get('first_week')}..{summary.get('last_week')}"
                if summary.get("first_week")
                else "not available"
            )
            lines.append(
                f"| {market} | {summary.get('iterations', 0)} | "
                f"{summary.get('expected_iterations', 'n/a')} | {summary.get('full', 'n/a')} | "
                f"{summary.get('pending', 'n/a')} | {formal_range} |"
            )
        lines.extend(
            [
                "",
                "## Detailed checks",
                "",
                "| Status | Check | Detail |",
                "|---|---|---|",
            ]
        )
        for row in self.findings:
            detail = row.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| **{row.status}** | `{row.check}` | {detail} |")
        failures = [row for row in self.findings if row.status == "FAIL"]
        lines.extend(["", "## Release decision", ""])
        if failures:
            lines.append(
                f"**FAIL — desktop release is blocked by {len(failures)} failed check(s).** "
                "No training or analysis state was modified by this audit."
            )
        else:
            lines.append(
                "**PASS — persisted V3.3 model state satisfies this release audit.** "
                "This result does not replace the separate frozen V3.2 baseline audit or UI/EXE tests."
            )
        lines.append("")
        return "\n".join(lines)


def synthetic_persistence_self_test() -> None:
    """Exercise the green path for hashes, maturity NULLs, and analysis identity."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    metric_ddl = ",\n".join(f"{name} REAL" for name in EVALUATION_METRIC_COLUMNS)
    connection.executescript(
        f"""
        CREATE TABLE instruments (id INTEGER PRIMARY KEY, code TEXT NOT NULL);
        CREATE TABLE v33_training_iterations (
            id INTEGER PRIMARY KEY, iteration_number INTEGER NOT NULL,
            maturity_status TEXT NOT NULL, composite_loss REAL, wis_loss REAL,
            price_turn_error_days INTEGER, dif_zero_error_days INTEGER
        );
        CREATE TABLE v33_forecasts (
            id INTEGER PRIMARY KEY, model_market TEXT, target_instrument_id INTEGER,
            training_iteration_id INTEGER, maturity_status TEXT, payload_json TEXT,
            forecast_hash TEXT, model_version TEXT, p10_path_json TEXT,
            p50_path_json TEXT, p90_path_json TEXT, expected_path_json TEXT,
            horizon_weeks INTEGER
        );
        CREATE TABLE v33_model_evaluations (
            id INTEGER PRIMARY KEY, forecast_id INTEGER, maturity_status TEXT,
            {metric_ddl}, payload_json TEXT, evaluated_at TEXT
        );
        CREATE TABLE v33_model_versions (
            id INTEGER PRIMARY KEY, model_market TEXT, version TEXT, artifact_hash TEXT
        );
        CREATE TABLE v33_analysis_runs (
            id TEXT PRIMARY KEY, model_market TEXT, target_instrument_id INTEGER,
            status TEXT, completed_at TEXT, result_json TEXT, forecast_id INTEGER,
            model_version TEXT, feature_snapshot_id TEXT
        );
        CREATE TABLE v33_feature_snapshots (
            id TEXT PRIMARY KEY, feature_json TEXT
        );
        CREATE TABLE v33_instrument_roles (
            id INTEGER PRIMARY KEY, instrument_id INTEGER, model_market TEXT,
            role TEXT, benchmark_instrument_id INTEGER, is_active INTEGER
        );
        """
    )
    zero_path = [0.0] * HORIZON_WEEKS
    p10 = [-0.01] * HORIZON_WEEKS
    p90 = [0.01] * HORIZON_WEEKS
    for iteration_number in range(1, 22):
        status = "full" if iteration_number == 1 else "pending"
        measured = 0.1 if status == "full" else None
        connection.execute(
            "INSERT INTO v33_training_iterations VALUES (?,?,?,?,?,?,?)",
            (
                iteration_number,
                iteration_number,
                status,
                measured,
                measured,
                1 if status == "full" else None,
                1 if status == "full" else None,
            ),
        )
        connection.execute(
            "INSERT INTO v33_forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                iteration_number,
                "399006",
                1,
                iteration_number,
                status,
                "{}",
                canonical_hash({}),
                f"synthetic-{iteration_number}",
                json.dumps(p10),
                json.dumps(zero_path),
                json.dumps(p90),
                json.dumps(zero_path),
                HORIZON_WEEKS,
            ),
        )
        metric_values = [measured] * len(EVALUATION_METRIC_COLUMNS)
        placeholders = ",".join("?" for _ in range(5 + len(metric_values)))
        connection.execute(
            f"INSERT INTO v33_model_evaluations VALUES ({placeholders})",
            (
                iteration_number,
                iteration_number,
                status,
                *metric_values,
                json.dumps({"composite_loss": measured}) if status == "full" else "{}",
                "2026-01-01T00:00:00+00:00" if status == "full" else None,
            ),
        )

    audit = Audit(Path("synthetic.db"))
    iteration_rows = list(
        connection.execute(
            "SELECT * FROM v33_training_iterations ORDER BY iteration_number"
        )
    )
    pending, full = audit._audit_maturity(connection, "399006", iteration_rows)
    assert pending == 20 and full == 1 and audit.findings[-1].status == "PASS"

    identity = {
        "iteration_count": 21,
        "model_count": 21,
        "optimizer_count": 1,
        "optimizer_state_hash": "state-hash",
    }
    for instrument_id, market in enumerate(ACTIVE_MARKETS, start=1):
        connection.execute(
            "INSERT INTO instruments VALUES (?,?)", (instrument_id, market)
        )
        model_hash = f"{market}-state-hash"
        model_version = f"{market}-champion"
        connection.execute(
            "INSERT INTO v33_model_versions VALUES (?,?,?,?)",
            (instrument_id, market, model_version, model_hash),
        )
        path = {
            "p10": p10,
            "p50": zero_path,
            "p90": p90,
            "expected": zero_path,
            "weekly_base": zero_path,
            "daily_correction": zero_path,
        }
        result = {
            "model": {"version": model_version, "state_hash": model_hash},
            "path": path,
            "data_quality": {"analysis_as_of": "2026-08-01T00:00:00+00:00"},
            "training_identity_before": identity,
            "training_identity_after": identity,
            "training_identity_unchanged": True,
        }
        forecast_payload = {"synthetic_analysis": market}
        forecast_id = 100 + instrument_id
        connection.execute(
            "INSERT INTO v33_forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                forecast_id,
                market,
                instrument_id,
                None,
                "live",
                json.dumps(forecast_payload),
                canonical_hash(forecast_payload),
                f"{model_version}-A-deadbeef",
                json.dumps(p10),
                json.dumps(zero_path),
                json.dumps(p90),
                json.dumps(zero_path),
                HORIZON_WEEKS,
            ),
        )
        connection.execute(
            "INSERT INTO v33_analysis_runs VALUES (?,?,?,?,?,?,?,?,?)",
            (
                f"analysis-{market}",
                market,
                instrument_id,
                "completed",
                "2026-08-01T00:00:00+00:00",
                json.dumps(result),
                forecast_id,
                model_version,
                None,
            ),
        )
    connection.execute("INSERT INTO instruments VALUES (3,'NDX')")
    connection.execute(
        "INSERT INTO v33_instrument_roles VALUES (1,2,'159941','tradable',3,1)"
    )
    connection.execute(
        "INSERT INTO v33_instrument_roles VALUES (2,3,'159941','benchmark',NULL,1)"
    )
    audit._audit_instrument_roles(connection)
    assert audit.findings[-1].status == "PASS"
    audit._audit_analysis(connection)
    analysis_findings = [
        row for row in audit.findings if row.check.endswith(".analysis")
    ]
    assert len(analysis_findings) == 2
    assert all(row.status == "PASS" for row in analysis_findings)
    connection.close()
    print(
        "PASS: synthetic green path covers model/evaluation NULLs, roles, and analysis identity"
    )


def self_test(database: Path | None = None) -> None:
    candidates = tuple(date(2026, 7, day) for day in range(27, 32))
    first = deterministic_sample(candidates, "399006")
    second = deterministic_sample(candidates, "399006")
    assert first == second and first in candidates
    payload = {name: None for name in MODEL_STATE_HASH_FIELDS}
    payload.update(
        {
            "market": "399006",
            "version": "self-test",
            "iteration_number": 1,
            "trained_through": "2026-07-31",
            "feature_names": [],
            "medians": [],
            "means": [],
            "scales": [],
            "intercept": [0.0] * HORIZON_WEEKS,
            "coefficients": [],
            "stumps": [],
            "residual_p10": [0.0] * HORIZON_WEEKS,
            "residual_p50": [0.0] * HORIZON_WEEKS,
            "residual_p90": [0.0] * HORIZON_WEEKS,
            "hyperparameters": {},
            "optimizer_memory": {},
            "turn_lag_days": {},
            "training_sample_count": 24,
        }
    )
    expected_state_hash = model_state_hash(payload)
    assert expected_state_hash == model_state_hash(json.loads(json.dumps(payload)))
    payload["state_hash"] = expected_state_hash
    payload["terminal_residuals"] = [0.0]
    payload["validation_metrics"] = {}
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from backend.app.services.v33_runtime_service import build_warmup_anchor_payload
    from backend.app.services.v33_training_service import state_from_payload

    restored = state_from_payload(payload)
    assert restored.state_hash == expected_state_hash
    print("PASS: independent state hash matches runtime state_from_payload algorithm")

    root_payload = json.loads(json.dumps(payload))
    root_payload.update(
        {
            "version": "self-test-root",
            "iteration_number": 0,
            "trained_through": "2026-07-01",
            "parent_state_hash": None,
            "training_sample_count": 2,
        }
    )
    root_payload["state_hash"] = model_state_hash(root_payload)
    root_state = state_from_payload(root_payload)
    anchor = build_warmup_anchor_payload(
        root_state,
        snapshot_manifest=(
            {
                "snapshot_id": "self-test-1",
                "cutoff_date": "2026-06-01",
                "snapshot_hash": "a" * 64,
            },
            {
                "snapshot_id": "self-test-2",
                "cutoff_date": "2026-06-08",
                "snapshot_hash": "b" * 64,
            },
        ),
        eligible_cutoff_dates=(date(2026, 6, 1), date(2026, 6, 8)),
        sampling_seed=SAMPLING_SEED,
    )
    errors, anchor_state_hash = warmup_anchor_envelope_errors(
        anchor,
        market="399006",
        first_cutoff=date(2026, 7, 1),
        sampling_seed=SAMPLING_SEED,
    )
    assert not errors and anchor_state_hash == root_state.state_hash
    for mutation in ("state_payload", "manifest", "anchor_hash"):
        tampered = json.loads(json.dumps(anchor))
        if mutation == "state_payload":
            tampered["state_payload"]["intercept"][0] = 1.0
        elif mutation == "manifest":
            tampered["build_snapshot_manifest"][0]["snapshot_hash"] = "c" * 64
        else:
            tampered["anchor_hash"] = "0" * 64
        errors, _ = warmup_anchor_envelope_errors(
            tampered,
            market="399006",
            first_cutoff=date(2026, 7, 1),
            sampling_seed=SAMPLING_SEED,
        )
        assert errors, f"tampered {mutation} unexpectedly passed"
    print("PASS: warm-up anchor envelope accepts the canonical root and rejects state/manifest/hash tampering")
    synthetic_persistence_self_test()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "readonly.db"
        writable = sqlite3.connect(path)
        writable.execute("CREATE TABLE proof (value INTEGER)")
        writable.commit()
        writable.close()
        readonly = open_read_only(path)
        try:
            try:
                readonly.execute("INSERT INTO proof VALUES (1)")
            except sqlite3.OperationalError:
                pass
            else:
                raise AssertionError("read-only connection unexpectedly allowed a write")
        finally:
            readonly.close()
    if database is not None and database.exists():
        production = open_read_only(database)
        try:
            schedule_audit = Audit(database)
            schedules: list[str] = []
            for market in ACTIVE_MARKETS:
                instrument_id = schedule_audit._instrument_id(production, market)
                if instrument_id is None:
                    raise AssertionError(f"missing self-test instrument {market}")
                expected = schedule_audit._expected_formal_samples(
                    production, market, instrument_id
                )
                repeated = schedule_audit._expected_formal_samples(
                    production, market, instrument_id
                )
                assert expected and expected == repeated
                schedules.append(
                    f"{market}={len(expected)} ({expected[0]}..{expected[-1]})"
                )
        finally:
            production.close()
        print("PASS: deterministic XSHG schedules: " + "; ".join(schedules))
    print("PASS: audit-v33-model-state self-test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database (default: data/investment_lab.db)")
    parser.add_argument("--report", type=Path, help="write the detailed Markdown report to this explicit path")
    parser.add_argument("--self-test", action="store_true", help="run deterministic/hash/read-only smoke checks")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(args.db)
        return 0
    audit = Audit(args.db)
    audit.run()
    report = audit.markdown()
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    print(report)
    return 0 if audit.passed else 1


if __name__ == "__main__":
    sys.exit(main())
