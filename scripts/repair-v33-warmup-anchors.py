#!/usr/bin/env python3
"""Rebuild and persist the two non-formal V3.3 warm-up root anchors.

The script never creates a formal iteration or model version.  It reconstructs
iteration 0 from the immutable training feature snapshots at/before the first
formal cutoff, requires an exact match with the already-issued iteration-1
``parent_state_hash``, takes a SQLite online backup, and only then adds the full
root audit artifact to the original bootstrap run's ``result_json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "investment_lab.db"
ACTIVE_MARKETS = ("399006", "159941")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.v33_runtime_service import (  # noqa: E402
    METHODOLOGY_VERSION,
    TRAINING_FEATURE_VERSION,
    _snapshot_from_payload,
    build_warmup_anchor_payload,
)
from backend.app.services.v33_training_service import (  # noqa: E402
    V33TrainingService,
    build_training_points,
)


class WarmupAnchorRepairError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RebuiltAnchor:
    market: str
    training_run_id: str
    anchor: Mapping[str, Any]
    already_persisted: bool


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_json(value: Any, *, context: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise WarmupAnchorRepairError(f"{context} is not valid JSON: {error}") from error


def open_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    uri = f"file:{quote(resolved.as_posix(), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _formal_identity(connection: sqlite3.Connection) -> tuple[int, ...]:
    return tuple(
        int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "v33_training_iterations",
            "v33_model_versions",
            "v33_forecasts",
            "v33_model_evaluations",
        )
    )


def rebuild_anchor(connection: sqlite3.Connection, market: str) -> RebuiltAnchor:
    if market not in ACTIVE_MARKETS:
        raise WarmupAnchorRepairError(f"unsupported market {market!r}")
    first = connection.execute(
        """
        SELECT * FROM v33_training_iterations
        WHERE model_market=? AND iteration_number=1
        """,
        (market,),
    ).fetchone()
    if first is None:
        raise WarmupAnchorRepairError(f"{market}: formal iteration 1 is missing")
    if first["parent_iteration_number"] is not None:
        raise WarmupAnchorRepairError(f"{market}: iteration 1 must have no formal parent iteration")
    run = connection.execute(
        "SELECT * FROM v33_training_runs WHERE id=?",
        (first["training_run_id"],),
    ).fetchone()
    if run is None or run["run_type"] != "bootstrap" or run["source_parent_iteration"] is not None:
        raise WarmupAnchorRepairError(f"{market}: iteration 1 is not owned by a root bootstrap run")
    model = connection.execute(
        """
        SELECT * FROM v33_model_versions
        WHERE model_market=? AND version=?
        """,
        (market, first["champion_model_version"]),
    ).fetchone()
    if model is None:
        raise WarmupAnchorRepairError(f"{market}: iteration-1 model artifact is missing")
    if model["parent_version"] is not None:
        raise WarmupAnchorRepairError(
            f"{market}: iteration-1 parent_version must be NULL for a non-formal root"
        )
    audit = parse_json(first["audit_json"], context=f"{market} iteration-1 audit_json")
    parameters = parse_json(
        model["parameters_json"], context=f"{market} iteration-1 parameters_json"
    )
    parent_hash = audit.get("parent_state_hash")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise WarmupAnchorRepairError(f"{market}: iteration-1 parent_state_hash is invalid")
    if parameters.get("parent_state_hash") != parent_hash:
        raise WarmupAnchorRepairError(
            f"{market}: iteration audit and model artifact disagree on root hash"
        )
    first_cutoff = date.fromisoformat(str(first["cutoff_date"]))
    snapshot_rows = list(
        connection.execute(
            """
            SELECT * FROM v33_feature_snapshots
            WHERE model_market=? AND feature_version=? AND cutoff_date<=?
            ORDER BY cutoff_date
            """,
            (market, TRAINING_FEATURE_VERSION, first_cutoff.isoformat()),
        )
    )
    if not snapshot_rows:
        raise WarmupAnchorRepairError(f"{market}: build-period feature snapshots are missing")
    snapshots = []
    manifest: list[dict[str, Any]] = []
    for row in snapshot_rows:
        feature_payload = parse_json(
            row["feature_json"], context=f"{market} snapshot {row['id']} feature_json"
        )
        content_hash = canonical_hash(feature_payload)
        if content_hash != str(row["snapshot_hash"]):
            raise WarmupAnchorRepairError(
                f"{market}: snapshot {row['id']} content hash mismatch"
            )
        snapshots.append(_snapshot_from_payload(feature_payload))
        manifest.append(
            {
                "snapshot_id": str(row["id"]),
                "cutoff_date": str(row["cutoff_date"]),
                "snapshot_hash": str(row["snapshot_hash"]),
            }
        )
    points = build_training_points(snapshots, sampling_seed=int(model["random_seed"]))
    service = V33TrainingService()
    state, eligible = service.fit_warmup_anchor(
        market,
        points,
        first_formal_cutoff=first_cutoff,
    )
    if state.state_hash != parent_hash:
        raise WarmupAnchorRepairError(
            f"{market}: reconstructed root {state.state_hash} != issued parent {parent_hash}"
        )
    if str(model["feature_version"]) != TRAINING_FEATURE_VERSION:
        raise WarmupAnchorRepairError(f"{market}: iteration-1 feature version changed")
    if str(model["methodology_version"]) != METHODOLOGY_VERSION:
        raise WarmupAnchorRepairError(f"{market}: iteration-1 methodology version changed")
    anchor = build_warmup_anchor_payload(
        state,
        snapshot_manifest=manifest,
        eligible_cutoff_dates=tuple(point.cutoff_date for point in eligible),
        sampling_seed=int(model["random_seed"]),
    )
    result_json = parse_json(run["result_json"], context=f"{market} bootstrap result_json")
    existing = result_json.get("warmup_anchor")
    if existing is not None and canonical_hash(existing) != canonical_hash(anchor):
        raise WarmupAnchorRepairError(f"{market}: persisted warm-up anchor is different")
    return RebuiltAnchor(
        market=market,
        training_run_id=str(run["id"]),
        anchor=anchor,
        already_persisted=existing is not None,
    )


def _backup_online(source_path: Path, backup_root: Path) -> tuple[Path, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = backup_root / f"pre-v33-warmup-anchor-{stamp}"
    suffix = 1
    while folder.exists():
        folder = backup_root / f"pre-v33-warmup-anchor-{stamp}-{suffix}"
        suffix += 1
    folder.mkdir(parents=True, exist_ok=False)
    target = folder / "investment_lab.db"
    source = open_read_only(source_path)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
        check = destination.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise WarmupAnchorRepairError(f"backup integrity_check failed: {check}")
    finally:
        destination.close()
        source.close()
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    (folder / "backup.json").write_text(
        json.dumps(
            {
                "source": str(source_path.resolve()),
                "backup": str(target.resolve()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sha256": checksum,
                "purpose": "before V3.3 non-formal warm-up root anchor persistence",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target, checksum


def repair_database(
    database: Path,
    *,
    apply: bool,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    database = database.expanduser().resolve(strict=True)
    readonly = open_read_only(database)
    try:
        preview = [rebuild_anchor(readonly, market) for market in ACTIVE_MARKETS]
        identity_before = _formal_identity(readonly)
    finally:
        readonly.close()
    pending = [item for item in preview if not item.already_persisted]
    result: dict[str, Any] = {
        "database": str(database),
        "mode": "apply" if apply else "dry_run",
        "markets": {
            item.market: {
                "state_hash": item.anchor["state_hash"],
                "build_snapshot_count": item.anchor["build_snapshot_count"],
                "real_matured_sample_count": item.anchor["real_matured_sample_count"],
                "already_persisted": item.already_persisted,
            }
            for item in preview
        },
        "formal_identity_before": identity_before,
    }
    if not apply or not pending:
        result["status"] = "verified" if not pending else "ready_to_apply"
        result["changed_markets"] = []
        return result

    backup_base = (
        backup_root.expanduser().resolve()
        if backup_root is not None
        else database.parent / "backups"
    )
    writer = sqlite3.connect(database, timeout=60, isolation_level=None)
    writer.row_factory = sqlite3.Row
    writer.execute("PRAGMA foreign_keys=ON")
    writer.execute("PRAGMA busy_timeout=60000")
    backup_path: Path | None = None
    try:
        writer.execute("BEGIN IMMEDIATE")
        # The reserved write lock prevents another writer from changing the
        # verified source between this online backup and the anchor updates.
        backup_path, backup_sha256 = _backup_online(database, backup_base)
        rebuilt = [rebuild_anchor(writer, market) for market in ACTIVE_MARKETS]
        changed: list[str] = []
        repair_at = datetime.now(timezone.utc).isoformat()
        for item in rebuilt:
            if item.already_persisted:
                continue
            row = writer.execute(
                "SELECT result_json, stages_json FROM v33_training_runs WHERE id=?",
                (item.training_run_id,),
            ).fetchone()
            if row is None:
                raise WarmupAnchorRepairError(f"{item.market}: bootstrap run disappeared")
            run_result = parse_json(row["result_json"], context="bootstrap result_json")
            stages = parse_json(row["stages_json"], context="bootstrap stages_json")
            run_result["warmup_anchor"] = item.anchor
            stages.append(
                {
                    "stage": "warmup_anchor_reconstructed",
                    "state_hash": item.anchor["state_hash"],
                    "at": repair_at,
                }
            )
            cursor = writer.execute(
                "UPDATE v33_training_runs SET result_json=?, stages_json=? WHERE id=?",
                (
                    json.dumps(run_result, ensure_ascii=False, sort_keys=True),
                    json.dumps(stages, ensure_ascii=False, sort_keys=True),
                    item.training_run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise WarmupAnchorRepairError(f"{item.market}: bootstrap run update failed")
            changed.append(item.market)
        identity_after = _formal_identity(writer)
        if identity_after != identity_before:
            raise WarmupAnchorRepairError("formal iteration/model/forecast identity changed")
        writer.commit()
    except Exception:
        writer.rollback()
        raise
    finally:
        writer.close()

    verification = open_read_only(database)
    try:
        final = [rebuild_anchor(verification, market) for market in ACTIVE_MARKETS]
        identity_verified = _formal_identity(verification)
    finally:
        verification.close()
    if not all(item.already_persisted for item in final):
        raise WarmupAnchorRepairError("post-commit warm-up anchor verification failed")
    if identity_verified != identity_before:
        raise WarmupAnchorRepairError("post-commit formal identity changed")
    result.update(
        {
            "status": "completed",
            "changed_markets": changed,
            "backup": str(backup_path.resolve()) if backup_path is not None else None,
            "backup_sha256": backup_sha256,
            "formal_identity_after": identity_verified,
        }
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="persist verified anchors; without this flag the command is read-only",
    )
    parser.add_argument("--backup-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = repair_database(args.db, apply=args.apply, backup_root=args.backup_root)
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
