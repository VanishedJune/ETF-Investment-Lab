"""Reset the v351 namespace for a clean re-run (V3.5 records untouched)."""

from __future__ import annotations

import argparse
import sqlite3


PROTO = "V3.5.1_EFFECTIVE_CHALLENGER_AND_DYNAMIC_ETF_SLOTS"


def reset(path: str) -> dict[str, int]:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=OFF")
    with con:
        # 1. Children of forecasts / strategy snapshots / accounts / challenges.
        con.execute(
            "DELETE FROM v35_forecast_calibrators WHERE forecast_id IN "
            "(SELECT id FROM v35_forecasts WHERE protocol_version=?)",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_forecast_evaluations WHERE forecast_id IN "
            "(SELECT id FROM v35_forecasts WHERE protocol_version=?)",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_residual_records WHERE forecast_id IN "
            "(SELECT id FROM v35_forecasts WHERE protocol_version=?)",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_position_decisions WHERE strategy_snapshot_id IN "
            "(SELECT id FROM v35_strategy_snapshots WHERE protocol_version=?)",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_sim_ledger WHERE account_id IN "
            "(SELECT id FROM v35_sim_accounts WHERE protocol_version=?)",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_sim_evaluations WHERE account_id IN "
            "(SELECT id FROM v35_sim_accounts WHERE protocol_version=?)",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v351_candidate_evaluations WHERE protocol_version=?",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_challenger_windows WHERE challenge_id IN "
            "(SELECT id FROM v35_challenges WHERE protocol_version=?)",
            (PROTO,),
        )
        con.execute("DELETE FROM v35_promotions WHERE protocol_version=?", (PROTO,))
        # 2. Parents of the above children.
        con.execute(
            "DELETE FROM v35_strategy_snapshots WHERE protocol_version=?",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_sim_accounts WHERE protocol_version=?",
            (PROTO,),
        )
        con.execute(
            "DELETE FROM v35_continuous_accounts WHERE protocol_version=?",
            (PROTO,),
        )
        con.execute("DELETE FROM v35_challenges WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_health_snapshots WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_market_state WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_forecasts WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_training_iterations WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_training_runs WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_training_profiles WHERE protocol_version=?", (PROTO,))
        con.execute(
            "DELETE FROM v35_probability_calibrators WHERE protocol_version=?",
            (PROTO,),
        )
        con.execute("DELETE FROM v35_random_plans WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_model_packages WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_model_versions WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_feature_snapshots WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v35_bootstrap_state WHERE protocol_version=?", (PROTO,))
        con.execute("DELETE FROM v351_maintenance_refreshes")
        con.execute("DELETE FROM v351_slot_replacement_jobs")
        con.execute("DELETE FROM v351_slot_model_state")
    con.commit()
    violations = con.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"foreign key violations after reset: {violations[:5]}")
    checks = {
        "v35_forecasts": "SELECT COUNT(*) FROM v35_forecasts WHERE protocol_version=?",
        "v35_bootstrap_state": "SELECT COUNT(*) FROM v35_bootstrap_state WHERE protocol_version=?",
        "v351_candidate_evaluations": "SELECT COUNT(*) FROM v351_candidate_evaluations WHERE protocol_version=?",
    }
    counts = {
        table: int(con.execute(query, (PROTO,)).fetchone()[0])
        for table, query in checks.items()
    }
    con.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=str)
    args = parser.parse_args()
    counts = reset(args.database)
    print(counts)
    if any(counts.values()):
        raise SystemExit("v351 namespace reset incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
