from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


MARKETS = ("399006", "159941")
FORBIDDEN_159941 = (
    "ndx",
    "nasdaq_index",
    "usd",
    "fx",
    "exchange_rate",
    "nav",
    "premium",
    "discount",
    "overseas",
    "futures",
)
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
# Schema 20 introduced V3.4.0.  Schemas 21 and 22 only append the V3.4.1
# training and V3.4.2 turning-policy tables; they do not rewrite frozen v34_* rows.
SUPPORTED_SCHEMA_VERSIONS = frozenset({20, 21, 22})


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _forbidden_hits(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            token = str(key).lower()
            key_path = f"{path}.{key}" if path else str(key)
            if any(forbidden in token for forbidden in FORBIDDEN_159941):
                hits.append(key_path)
            hits.extend(_forbidden_hits(item, key_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_forbidden_hits(item, f"{path}[{index}]"))
    return hits


def audit(database: Path) -> dict[str, Any]:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type=?", ("table",)
            )
        }
        table_counts = {
            table: int(connection.execute(f'SELECT COUNT(1) FROM "{table}"').fetchone()[0])
            for table in V34_TABLES
            if table in tables
        }
        markets: dict[str, Any] = {}
        for market in MARKETS:
            iterations = list(
                connection.execute(
                    "SELECT weekly_iteration_number, anchor_date, forecast_id "
                    "FROM v34_training_iterations WHERE model_market=? "
                    "ORDER BY weekly_iteration_number",
                    (market,),
                )
            )
            iteration_numbers = [int(row[0]) for row in iterations]
            forecasts = list(
                connection.execute(
                    "SELECT id, forecast_anchor_date, horizon_weeks, scenario_count, "
                    "maturity_status FROM v34_forecasts WHERE model_market=?",
                    (market,),
                )
            )
            forecast_by_id = {str(row[0]): row for row in forecasts}
            training_forecasts = [forecast_by_id[str(row[2])] for row in iterations]
            pending = [row for row in training_forecasts if str(row[4]) != "FULLY_MATURE_13W"]
            bad_anchor_links = sum(
                str(row[1]) != str(forecast_by_id[str(row[2])][1]) for row in iterations
            )
            optimizer = connection.execute(
                "SELECT last_anchor_date, weekly_iteration_count, candidate_training_count, "
                "champion_promotion_count, champion_version FROM v34_optimizer_states "
                "WHERE model_market=?",
                (market,),
            ).fetchone()
            latest_run = connection.execute(
                "SELECT id, status, weekly_iteration_count, candidate_training_count, "
                "champion_promotion_count, result_json, error_code, error_message "
                "FROM v34_training_runs WHERE model_market=? ORDER BY started_at DESC LIMIT 1",
                (market,),
            ).fetchone()
            leakage_rows = int(
                connection.execute(
                    "SELECT COUNT(1) FROM v34_feature_snapshots WHERE model_market=? "
                    "AND (source_max_date > forecast_anchor_date "
                    "OR feature_fit_end_date > forecast_anchor_date)",
                    (market,),
                ).fetchone()[0]
            )
            markets[market] = {
                "iteration_count": len(iterations),
                "iteration_numbers_contiguous": iteration_numbers
                == list(range(1, len(iterations) + 1)),
                "first_anchor": str(iterations[0][1]) if iterations else None,
                "last_anchor": str(iterations[-1][1]) if iterations else None,
                "training_forecast_count": len(training_forecasts),
                "horizon_violations": sum(int(row[2]) != 13 for row in training_forecasts),
                "scenario_count_violations": sum(
                    int(row[3]) < 1000 for row in training_forecasts
                ),
                "anchor_link_violations": bad_anchor_links,
                "pending_13w_count": len(pending),
                "maturity_counts": {
                    str(status): int(count)
                    for status, count in connection.execute(
                        "SELECT maturity_status, COUNT(1) FROM v34_forecasts "
                        "WHERE model_market=? GROUP BY maturity_status",
                        (market,),
                    )
                },
                "point_in_time_leakage_rows": leakage_rows,
                "optimizer": list(optimizer) if optimizer else None,
                "latest_run": {
                    "id": latest_run[0],
                    "status": latest_run[1],
                    "weekly_iteration_count": latest_run[2],
                    "candidate_training_count": latest_run[3],
                    "champion_promotion_count": latest_run[4],
                    "result": _json(latest_run[5]),
                    "error_code": latest_run[6],
                    "error_message": latest_run[7],
                }
                if latest_run
                else None,
            }

        forbidden_hits: list[str] = []
        for snapshot_id, source_fields, feature_json, daily_json, weekly_json in connection.execute(
            "SELECT id, source_fields_json, feature_json, daily_sequence_json, weekly_state_json "
            "FROM v34_feature_snapshots WHERE model_market=?",
            ("159941",),
        ):
            for column, value in (
                ("source_fields_json", source_fields),
                ("feature_json", feature_json),
                ("daily_sequence_json", daily_json),
                ("weekly_state_json", weekly_json),
            ):
                forbidden_hits.extend(
                    f"{snapshot_id}:{column}:{hit}"
                    for hit in _forbidden_hits(_json(value))
                )

    checks = {
        "schema_version_supported": schema_version in SUPPORTED_SCHEMA_VERSIONS,
        "integrity_ok": integrity == "ok",
        "all_v34_tables_present": set(V34_TABLES) <= tables,
        "only_expected_markets": set(markets) == set(MARKETS),
        "at_least_400_iterations_each": all(
            value["iteration_count"] >= 400 for value in markets.values()
        ),
        "iterations_contiguous": all(
            value["iteration_numbers_contiguous"] for value in markets.values()
        ),
        "training_forecasts_match_iterations": all(
            value["training_forecast_count"] == value["iteration_count"]
            for value in markets.values()
        ),
        "horizon_is_13": all(value["horizon_violations"] == 0 for value in markets.values()),
        "scenario_count_at_least_1000": all(
            value["scenario_count_violations"] == 0 for value in markets.values()
        ),
        "forecast_anchor_links_match": all(
            value["anchor_link_violations"] == 0 for value in markets.values()
        ),
        "latest_13_are_pending": all(
            value["pending_13w_count"] == 13 for value in markets.values()
        ),
        "point_in_time_dates_safe": all(
            value["point_in_time_leakage_rows"] == 0 for value in markets.values()
        ),
        "latest_runs_completed": all(
            value["latest_run"] and value["latest_run"]["status"] == "completed"
            for value in markets.values()
        ),
        "159941_forbidden_feature_hits_zero": not forbidden_hits,
    }
    return {
        "database": str(database.resolve()),
        "schema_version": schema_version,
        "integrity_check": integrity,
        "table_counts": table_counts,
        "markets": markets,
        "forbidden_159941_hits": forbidden_hits[:100],
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V3.4 production-state audit")
    parser.add_argument("--db", type=Path, default=Path("data/investment_lab.db"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.db)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
