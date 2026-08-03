"""Exercise the persisted asynchronous V3.4.3 analysis API on a real database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time

import requests


def training_identity(database: Path) -> dict[str, object]:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return {
            "iterations": connection.execute(
                "SELECT COUNT(*) FROM v341_training_iterations"
            ).fetchone()[0],
            "candidates": connection.execute(
                "SELECT COUNT(*) FROM v341_candidate_trials"
            ).fetchone()[0],
            "optimizers": connection.execute(
                "SELECT protocol_version, model_market, last_anchor_date, "
                "weekly_iteration_count, state_hash "
                "FROM v341_optimizer_states ORDER BY model_market"
            ).fetchall(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    root = args.project_root.resolve()
    os.environ["INVESTMENT_LAB_HOME"] = str(root)
    os.environ["INVESTMENT_LAB_RESOURCE_ROOT"] = str(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    database = root / "data" / "investment_lab.db"
    before = training_identity(database)
    results: dict[str, object] = {}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    environment = dict(os.environ)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.web:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        startup_deadline = time.perf_counter() + 60.0
        while True:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=5)
                raise RuntimeError(
                    f"analysis server exited during startup: {stdout} {stderr}"
                )
            try:
                health = requests.get(f"{base_url}/api/health", timeout=2)
                if health.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.perf_counter() >= startup_deadline:
                raise TimeoutError("analysis server did not become healthy within 60 seconds")
            time.sleep(0.25)
        for market in ("399006", "159941"):
            started = time.perf_counter()
            response = requests.post(
                f"{base_url}/api/v343/model/analysis",
                json={"instrument_code": market},
                timeout=10,
            )
            enqueue_seconds = time.perf_counter() - started
            if response.status_code != 202:
                raise RuntimeError(
                    f"{market} enqueue failed: {response.status_code} {response.text}"
                )
            created = response.json()
            deadline = time.perf_counter() + args.timeout_seconds
            polls = 0
            while True:
                polls += 1
                current = requests.get(
                    f"{base_url}/api/v343/model/analysis/runs/{created['run_id']}",
                    timeout=10,
                )
                if current.status_code != 200:
                    raise RuntimeError(
                        f"{market} poll failed: {current.status_code} {current.text}"
                    )
                payload = current.json()
                if payload["status"] == "failed":
                    raise RuntimeError(
                        f"{market} analysis failed: {payload.get('error_message')}"
                    )
                if payload["status"] == "completed":
                    result = payload.get("result") or {}
                    if result.get("market") != market:
                        raise RuntimeError(
                            f"{market} result market mismatch: {result.get('market')}"
                        )
                    if result.get("display_version") != "V3.4.3_MARKET_UI":
                        raise RuntimeError(
                            f"{market} result was not enriched for the current UI"
                        )
                    results[market] = {
                        "enqueue_seconds": enqueue_seconds,
                        "total_seconds": time.perf_counter() - started,
                        "polls": polls,
                        "run_id": created["run_id"],
                        "forecast_anchor_date": result.get("forecast_anchor_date"),
                        "display_version": result.get("display_version"),
                    }
                    break
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        f"{market} analysis exceeded {args.timeout_seconds} seconds"
                    )
                time.sleep(0.25)
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    after = training_identity(database)
    if before != after:
        raise RuntimeError("data analysis mutated the V3.4.1 training identity")
    print(
        json.dumps(
            {
                "database": str(database),
                "training_identity_unchanged": True,
                "training_identity": before,
                "markets": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
