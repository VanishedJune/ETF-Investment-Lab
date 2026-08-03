"""Rebuild obsolete local chart indicator payloads without touching models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.database.session import create_session_factory
from backend.app.models.models import IndicatorRecord, Instrument
from backend.app.services.indicator_service import IndicatorService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--instrument", required=True)
    args = parser.parse_args()

    database = args.database.resolve()
    if not database.is_file():
        raise SystemExit(f"database does not exist: {database}")
    sessions = create_session_factory(database)
    service = IndicatorService(sessions)
    stale_before = service.stale_timeframes(args.instrument)
    results = service.recalculate_stale_timeframes(args.instrument)
    stale_after = service.stale_timeframes(args.instrument)
    if stale_after:
        raise RuntimeError(f"indicator repair remained stale: {stale_after}")

    coverage: dict[str, dict[str, object]] = {}
    with sessions() as session:
        instrument = session.scalar(
            select(Instrument).where(Instrument.code == args.instrument)
        )
        if instrument is None:
            raise RuntimeError(f"unknown instrument: {args.instrument}")
        for timeframe in service.chart_timeframes:
            rows = list(
                session.scalars(
                    select(IndicatorRecord)
                    .where(
                        IndicatorRecord.instrument_id == instrument.id,
                        IndicatorRecord.timeframe == timeframe,
                    )
                    .order_by(IndicatorRecord.indicator_date)
                )
            )
            missing = [
                row.indicator_date.isoformat()
                for row in rows[1:]
                if (row.indicator_values or {})
                .get("values", {})
                .get("dif_first_change")
                is None
            ]
            if missing:
                raise RuntimeError(
                    f"{args.instrument} {timeframe} missing DIF first changes: {missing[:10]}"
                )
            coverage[timeframe] = {
                "rows": len(rows),
                "missing_dif_first_change_after_first": missing,
            }
    print(
        json.dumps(
            {
                "database": str(database),
                "instrument": args.instrument,
                "stale_before": stale_before,
                "results": [asdict(item) for item in results],
                "stale_after": stale_after,
                "coverage": coverage,
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
