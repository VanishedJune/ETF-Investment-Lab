"""Leakage-safe V3.4 feature snapshots with a strict 159941 self-data boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import Instrument, MarketPrice
from backend.app.services.v33_feature_service import (
    FeatureSnapshot,
    PriceBar,
    V33FeatureError,
    V33FeatureService,
    flatten_snapshot,
)


SUPPORTED_MARKETS = ("399006", "159941")
FEATURE_VERSION = "V3.4_13W_FEATURES_1"
SCALER_VERSION = "V3.4_TRAIN_FOLD_ONLY_1"
FORBIDDEN_159941_TOKENS = (
    "ndx",
    "nasdaq_index",
    "benchmark",
    "fx",
    "usd",
    "nav",
    "premium",
    "discount",
    "qdii",
    "futures",
    "macro",
    "interest_rate",
    "aum",
    "fund_shares",
    "subscriptions",
    "pe",
    "pb",
)
FORBIDDEN_159941_EXACT_SEGMENTS = {"ndx", "fx", "usd", "nav", "pe", "pb", "aum"}


class V34FeatureError(RuntimeError):
    pass


def _hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_forbidden_159941_field(name: str) -> bool:
    lowered = name.lower()
    segments = {segment for segment in re.split(r"[^a-z0-9]+|_", lowered) if segment}
    if segments & FORBIDDEN_159941_EXACT_SEGMENTS:
        return True
    return any(
        token in lowered
        for token in FORBIDDEN_159941_TOKENS
        if token not in FORBIDDEN_159941_EXACT_SEGMENTS
    )


def assert_159941_self_only(fields: Iterable[str]) -> None:
    forbidden = sorted({str(field) for field in fields if is_forbidden_159941_field(str(field))})
    if forbidden:
        raise V34FeatureError(
            "159941 V3.4 attempted to use forbidden non-ETF fields: "
            + ", ".join(forbidden)
        )


class V34FeatureService:
    """Builds the existing weekly+100-day feature family under V3.4 rules."""

    def __init__(self) -> None:
        self._base = V33FeatureService(require_weekly_bars=35, daily_window=100)

    @staticmethod
    def validate_market(market: str) -> None:
        if market not in SUPPORTED_MARKETS:
            raise V34FeatureError("V3.4 supports only 399006 and 159941")

    def weekly_anchors(
        self,
        session: Session,
        market: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[date, ...]:
        """Return each natural week's final actual trading session."""

        self.validate_market(market)
        instrument = session.scalar(select(Instrument).where(Instrument.code == market))
        if instrument is None:
            raise V34FeatureError(f"missing instrument {market}")
        query = select(MarketPrice.trade_date).where(
            MarketPrice.instrument_id == instrument.id,
            MarketPrice.timeframe == "daily",
        )
        if start_date is not None:
            query = query.where(MarketPrice.trade_date >= start_date)
        if end_date is not None:
            query = query.where(MarketPrice.trade_date <= end_date)
        sessions = tuple(session.scalars(query.order_by(MarketPrice.trade_date)).all())
        by_week: dict[tuple[int, int], date] = {}
        for trade_date in sessions:
            year, week, _ = trade_date.isocalendar()
            by_week[(year, week)] = trade_date
        return tuple(by_week[key] for key in sorted(by_week))

    def load_snapshot(self, session: Session, market: str, anchor: date) -> FeatureSnapshot:
        self.validate_market(market)
        if market == "399006":
            try:
                return self._base.load_snapshot(
                    session, market, anchor, weekly_cutoff_date=anchor
                )
            except V33FeatureError as exc:
                raise V34FeatureError(str(exc)) from exc

        instrument = session.scalar(select(Instrument).where(Instrument.code == market))
        if instrument is None:
            raise V34FeatureError("missing instrument 159941")
        rows = session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= anchor,
            )
            .order_by(MarketPrice.trade_date)
        ).all()
        bars = tuple(
            PriceBar(
                trade_date=row.trade_date,
                open=float(row.open_price),
                high=float(row.high_price),
                low=float(row.low_price),
                close=float(row.close_price),
                raw_close=(
                    float(row.raw_close_price)
                    if getattr(row, "raw_close_price", None) is not None
                    else float(row.close_price)
                ),
                volume=None if row.volume is None else float(row.volume),
                turnover=None if row.turnover is None else float(row.turnover),
                source=row.source,
            )
            for row in rows
        )
        try:
            raw = self._base.build_snapshot(
                market,
                bars,
                anchor,
                observations=(),
                benchmark_bars=(),
                weekly_cutoff_date=anchor,
            )
        except V33FeatureError as exc:
            raise V34FeatureError(str(exc)) from exc
        weekly = {
            key: value for key, value in raw.weekly.items() if not is_forbidden_159941_field(key)
        }
        daily = {
            key: value for key, value in raw.daily.items() if not is_forbidden_159941_field(key)
        }
        missing = {
            key: value
            for key, value in raw.missing_masks.items()
            if not is_forbidden_159941_field(key)
        }
        fields = (*weekly.keys(), *daily.keys(), *missing.keys())
        assert_159941_self_only(fields)
        return replace(
            raw,
            weekly=weekly,
            daily=daily,
            missing_masks=missing,
            provenance={
                "target": "159941",
                "data_boundary": "SELF_OHLCV_ONLY",
                "target_price_sources": raw.provenance.get("target_price_sources", []),
                "daily_session_count": raw.provenance.get("daily_session_count"),
                "weekly_bar_count": raw.provenance.get("weekly_bar_count"),
                "weekly_cutoff_date": anchor.isoformat(),
                "daily_window_used": 100,
                "forbidden_source_read_count": 0,
            },
            benchmark_market=None,
        )

    def audit_payload(self, snapshot: FeatureSnapshot) -> dict[str, Any]:
        features = flatten_snapshot(snapshot)
        if snapshot.market == "159941":
            assert_159941_self_only(features)
        source_fields = sorted(features)
        payload = {
            "market": snapshot.market,
            "forecast_anchor_date": snapshot.cutoff_date.isoformat(),
            "source_max_date": snapshot.source_data_max_date.isoformat(),
            "feature_fit_end_date": snapshot.cutoff_date.isoformat(),
            "feature_version": FEATURE_VERSION,
            "scaler_version": SCALER_VERSION,
            "features": features,
            "daily_sequence": list(snapshot.daily_sequence),
            "weekly_state": dict(snapshot.weekly),
            "source_fields": source_fields,
            "provenance": dict(snapshot.provenance),
            "derivative_turn": dict(snapshot.derivative_turn),
        }
        payload["snapshot_hash"] = _hash(payload)
        return payload
