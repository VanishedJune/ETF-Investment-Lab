"""Run or resume the V3.5 production Bootstrap for one market.

The runner is idempotent: completed weeks are skipped and an interrupted run
continues from the first missing anchor.  It writes only ``v35_*`` rows in
the production database.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.app.database.session import create_session_factory
from backend.app.services.v35_runtime_service import V35RuntimeService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("market", choices=("399006", "159941"))
    parser.add_argument("--database", default=Path("data/investment_lab.db"), type=Path)
    parser.add_argument("--max-weeks", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    factory = create_session_factory(args.database)
    runtime = V35RuntimeService(factory)
    started = time.monotonic()
    before = runtime.status(args.market)
    print(
        json.dumps(
            {
                "market": args.market,
                "action": "before",
                "status": before,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    result = runtime.bootstrap_sync(args.market, maximum_weeks=args.max_weeks)
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    result["status_after"] = runtime.status(args.market)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
