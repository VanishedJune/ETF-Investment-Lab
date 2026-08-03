from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import sys

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.agent_iterations.runner import (
    InsufficientMatureData,
    PriceBar,
    append_due_iterations,
    run_progressive_baseline,
)
from backend.app.agent_iterations.storage import AgentIterationStorage
from backend.app.database.session import create_session_factory
from backend.app.models.models import Instrument, MarketPrice, ValuationRecord
from backend.app.services.indicator_service import IndicatorService
from backend.app.services.market_data import MarketDataService
from backend.app.services.providers import AkShareIndexProvider


LEGACY_V1_BASELINE_ITERATIONS = 100
LEGACY_V1_MONTHLY_INCREMENT = 2


def _policy() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / "config" / "agent_iteration_policy.json").read_text(encoding="utf-8"))


def _legacy_limits(policy: dict[str, object]) -> tuple[int, int]:
    """Read an old test/config shape or use the frozen V1 audit constants."""

    return (
        int(
            policy.get(
                "baseline_iterations_per_market",
                LEGACY_V1_BASELINE_ITERATIONS,
            )
        ),
        int(
            policy.get(
                "monthly_iterations_per_market",
                LEGACY_V1_MONTHLY_INCREMENT,
            )
        ),
    )


def _assert_legacy_write_allowed(policy: dict[str, object]) -> None:
    legacy = policy.get("legacy_v1")
    if (
        isinstance(legacy, dict)
        and legacy.get("status") == "read_only_audit"
    ):
        raise RuntimeError(
            "V1 is read-only. Run Weekly Analysis V2 from the web page."
        )


def _load_series(code: str) -> tuple[list[PriceBar], list[PriceBar], dict[date, Decimal]]:
    sessions = create_session_factory(PROJECT_ROOT / "data" / "investment_lab.db")
    with sessions() as session:
        instrument = session.scalar(select(Instrument).where(Instrument.code == code))
        if instrument is None:
            raise ValueError(f"Instrument {code} is not initialized")
        rows = session.scalars(
            select(MarketPrice)
            .where(MarketPrice.instrument_id == instrument.id)
            .order_by(MarketPrice.trade_date)
        ).all()
        valuations = session.scalars(
            select(ValuationRecord)
            .where(ValuationRecord.instrument_id == instrument.id)
            .order_by(ValuationRecord.valuation_date)
        ).all()
    by_timeframe: dict[str, list[PriceBar]] = {"daily": [], "weekly": []}
    for row in rows:
        if row.timeframe not in by_timeframe:
            continue
        close = row.close_price
        by_timeframe[row.timeframe].append(
            PriceBar(
                trade_date=row.trade_date,
                open=row.open_price or close,
                high=row.high_price or close,
                low=row.low_price or close,
                close=close,
                volume=(row.volume * Decimal(row.volume_multiplier) if row.volume is not None else None),
                source=row.source or "UNKNOWN",
            )
        )
    percentile_map = {
        row.valuation_date: row.valuation_percentile
        for row in valuations
        if row.valuation_percentile is not None
    }
    return by_timeframe["weekly"], by_timeframe["daily"], percentile_map


def _repair_periods(code: str) -> None:
    sessions = create_session_factory(PROJECT_ROOT / "data" / "investment_lab.db")
    market = MarketDataService(sessions)
    provider = AkShareIndexProvider()
    upstream = provider.fetch_weekly(code)
    market.aggregate_periods(code, weekly_result=upstream)
    IndicatorService(sessions).recalculate_all_timeframes(code)


def status() -> list[dict[str, object]]:
    policy = _policy()
    baseline_count, monthly_increment = _legacy_limits(policy)
    storage = AgentIterationStorage(PROJECT_ROOT / "data" / "model_iterations")
    return [
        storage.status(
            code,
            required_baseline=baseline_count,
            monthly_increment=monthly_increment,
        )
        for code in policy["instruments"]
    ]


def baseline(executed_by: str) -> list[dict[str, object]]:
    policy = _policy()
    _assert_legacy_write_allowed(policy)
    baseline_count, monthly_increment = _legacy_limits(policy)
    storage = AgentIterationStorage(PROJECT_ROOT / "data" / "model_iterations")
    manifests = []
    for code in policy["instruments"]:
        current = storage.status(
            code,
            required_baseline=baseline_count,
            monthly_increment=monthly_increment,
        )
        if current["baseline_status"] == "unchanged":
            manifests.append(
                {
                    **current,
                    "operation_status": "unchanged",
                    "message": "Baseline artifacts are already complete and were not rewritten.",
                }
            )
            continue
        _repair_periods(code)
        weekly, daily, valuations = _load_series(code)
        manifests.append(
            run_progressive_baseline(
                instrument_code=code,
                weekly_bars=weekly,
                daily_bars=daily,
                valuation_percentiles=valuations,
                storage=storage,
                count=baseline_count,
                seed=int(policy["seed"]),
                executed_by=executed_by,
            )
        )
    return manifests


def due(executed_by: str, as_of: date | None = None) -> list[dict[str, object]]:
    policy = _policy()
    _assert_legacy_write_allowed(policy)
    _baseline_count, monthly_limit = _legacy_limits(policy)
    current = status()
    if any(int(item["baseline_remaining"]) > 0 for item in current):
        raise RuntimeError("Baseline is incomplete; run -Mode Baseline before monthly Due iterations")
    storage = AgentIterationStorage(PROJECT_ROOT / "data" / "model_iterations")
    results: list[dict[str, object]] = []
    effective_as_of = as_of or date.today()
    period = effective_as_of.strftime("%Y-%m")
    for code in policy["instruments"]:
        manifest, _records = storage.read_consistent_snapshot(code)
        period_entry = (
            dict(manifest.get("maintenance_ledger") or {}).get(period, {})
            if manifest is not None
            else {}
        )
        if int(dict(period_entry).get("count", 0)) >= monthly_limit:
            results.append(
                {
                    **next(item for item in current if item["instrument_code"] == code),
                    "operation_status": "unchanged",
                    "message": f"Monthly maintenance quota for {period} is already exhausted.",
                }
            )
            continue
        _repair_periods(code)
        weekly, daily, valuations = _load_series(code)
        try:
            results.append(
                append_due_iterations(
                    instrument_code=code,
                    weekly_bars=weekly,
                    daily_bars=daily,
                    valuation_percentiles=valuations,
                    storage=storage,
                    count=monthly_limit,
                    executed_by=executed_by,
                    as_of=effective_as_of,
                )
            )
        except InsufficientMatureData as error:
            results.append(
                {
                    **next(item for item in current if item["instrument_code"] == code),
                    "operation_status": "unchanged",
                    "message": str(error),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent-owned progressive weekly model iteration")
    parser.add_argument("--mode", choices=("status", "baseline", "due"), required=True)
    parser.add_argument("--executed-by", default="Codex")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    if args.mode == "status":
        payload = status()
    elif args.mode == "baseline":
        payload = baseline(args.executed_by)
    else:
        payload = due(args.executed_by, args.as_of)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
