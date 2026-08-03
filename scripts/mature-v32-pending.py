"""Backfill newly observable V3.2 outcomes without creating training rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sqlalchemy import func, select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.database.session import create_session_factory  # noqa: E402
from backend.app.models.models import V32TrainingIteration  # noqa: E402
from backend.app.services.v32_training_service import (  # noqa: E402
    SUPPORTED_MARKETS,
    V32TrainingService,
)


def _counts(sessions, market: str) -> dict[str, int]:
    with sessions() as session:
        total = session.scalar(
            select(func.count()).select_from(V32TrainingIteration).where(
                V32TrainingIteration.market == market
            )
        ) or 0
        full = session.scalar(
            select(func.count()).select_from(V32TrainingIteration).where(
                V32TrainingIteration.market == market,
                V32TrainingIteration.maturity_status == "full",
            )
        ) or 0
        pending = session.scalar(
            select(func.count()).select_from(V32TrainingIteration).where(
                V32TrainingIteration.market == market,
                V32TrainingIteration.maturity_status == "pending",
            )
        ) or 0
    return {"total": total, "full": full, "pending": pending}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "investment_lab.db",
    )
    args = parser.parse_args()
    sessions = create_session_factory(args.database)
    service = V32TrainingService(sessions)
    result: dict[str, object] = {"database": str(args.database.resolve()), "markets": {}}
    try:
        for market in SUPPORTED_MARKETS:
            before = _counts(sessions, market)
            matured = service.mature_pending(market)
            after = _counts(sessions, market)
            result["markets"][market] = {  # type: ignore[index]
                "before": before,
                "matured": matured,
                "after": after,
                "iteration_count_unchanged": before["total"] == after["total"],
            }
    finally:
        service.shutdown()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
