"""Bounded anonymized DeepSeek historical screening for V3.7 research.

Per market up to ``--per-market`` anchors are sampled (recent 52 weeks are
always included; the rest are evenly stratified).  Calls are anonymized
(MARKET_A / T-n / anchor_close=1.0) and frozen as HISTORICAL_SCREENING; they
never enter formal AI calibration, weight unlocks or promotions.

Rules: transient failures retry with 1s/2s/4s backoff; a market stops when
its failure rate exceeds 20%; the whole run stops when the token budget is
exceeded; ``--only-missing`` resumes by skipping anchors that already have a
successful HISTORICAL_SCREENING forecast for the generation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.session import create_session_factory
from backend.app.models.models import V37AiForecast
from backend.app.services.deepseek_client import load_config
from backend.app.services.v37_ai_service import ensure_ai_forecast
from backend.app.services.v37_config import (
    AI_SCOPE_HISTORICAL_SCREENING,
    PROTOCOL_VERSION_37,
)
from backend.app.services.v37_feature_service import V37FeatureService


ALL_MARKETS = ("399006", "159941", "518600", "512800", "512690", "512010")
DEFAULT_TOKEN_BUDGET = 3_000_000
FAILURE_RATE_LIMIT = 0.20
RETRY_DELAYS = (1.0, 2.0, 4.0)


def sample_anchors(anchors: tuple, per_market: int) -> list:
    if len(anchors) <= per_market:
        return list(anchors)
    recent = list(anchors[-52:])
    earlier = list(anchors[:-52])
    remaining = per_market - len(recent)
    if remaining <= 0:
        return recent[:per_market]
    step = max(1, len(earlier) / max(remaining, 1))
    sampled = [
        earlier[int(index * step)]
        for index in range(remaining)
        if int(index * step) < len(earlier)
    ]
    return sorted(set(sampled + recent))


def _existing_missing(
    factory,
    market: str,
    anchors: list,
    generation: int,
) -> list:
    if not anchors:
        return []
    with Session(factory) as session:
        existing = set(
            session.scalars(
                select(V37AiForecast.forecast_anchor_date)
                .where(
                    V37AiForecast.protocol_version == PROTOCOL_VERSION_37,
                    V37AiForecast.model_market == market,
                    V37AiForecast.screening_scope
                    == AI_SCOPE_HISTORICAL_SCREENING,
                    V37AiForecast.ai_generation_version == generation,
                    V37AiForecast.status == "PENDING",
                )
            ).all()
        )
    return [anchor for anchor in anchors if anchor not in existing]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=Path("data/investment_lab.db"), type=Path)
    parser.add_argument("--market", choices=ALL_MARKETS, default=None)
    parser.add_argument("--per-market", type=int, default=120)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--delay-seconds", type=float, default=0.3)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--max-total-tokens", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--price-per-1m-tokens", type=float, default=0.0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    if not config.api_key:
        print("DEEPSEEK_API_KEY is not configured")
        return 2
    factory = create_session_factory(args.database)
    markets = (args.market,) if args.market else ALL_MARKETS
    summary: dict[str, object] = {}
    total_started = time.monotonic()
    total_tokens = 0
    for market in markets:
        with Session(factory) as session:
            anchors = V37FeatureService().weekly_anchors(session, market)
        selected = sample_anchors(anchors, args.per_market)
        if args.only_missing:
            selected = _existing_missing(factory, market, selected, args.generation)
        counts = {
            "selected": len(selected),
            "new": 0,
            "cached": 0,
            "failed": 0,
            "failed_codes": {},
            "tokens": 0,
        }
        attempted = 0
        stopped = None
        for index, anchor in enumerate(selected, start=1):
            attempted += 1
            with Session(factory) as session, session.begin():
                result = ensure_ai_forecast(
                    session,
                    market,
                    anchor,
                    config,
                    anonymize=True,
                    generation=args.generation,
                    screening_scope=AI_SCOPE_HISTORICAL_SCREENING,
                    retry_delays=RETRY_DELAYS,
                )
            if result.get("cached"):
                counts["cached"] += 1
            elif result.get("status") == "OK":
                counts["new"] += 1
                usage = result.get("token_usage") or {}
                tokens = int(usage.get("total_tokens") or 0)
                counts["tokens"] += tokens
                total_tokens += tokens
            else:
                counts["failed"] += 1
                code = str(result.get("error_code") or result.get("status") or "UNKNOWN")
                counts["failed_codes"][code] = (
                    counts["failed_codes"].get(code, 0) + 1
                )
            if index % 10 == 0:
                print(f"{market} {index}/{len(selected)}", counts)
            if attempted and counts["failed"] / attempted > FAILURE_RATE_LIMIT:
                stopped = f"FAILURE_RATE_{counts['failed'] / attempted:.2f}"
                break
            if total_tokens > args.max_total_tokens:
                stopped = "TOKEN_BUDGET_EXCEEDED"
                break
            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)
        counts["stopped_reason"] = stopped
        summary[market] = counts
        if stopped == "TOKEN_BUDGET_EXCEEDED":
            break
    summary["elapsed_seconds"] = round(time.monotonic() - total_started, 2)
    summary["total_tokens"] = total_tokens
    summary["estimated_cost"] = round(
        total_tokens / 1_000_000.0 * args.price_per_1m_tokens, 6
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
