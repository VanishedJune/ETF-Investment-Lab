"""Point-in-time V3.4.1 features with explicit versioned manifests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
import hashlib
import json
from typing import Any, Mapping

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.models.models import V33PointInTimeObservation
from backend.app.services.v33_feature_service import (
    DEFAULT_PIT_SERIES_CODES,
    FeatureSnapshot,
    PointInTimeObservation,
    V33FeatureError,
    V33FeatureService,
    flatten_snapshot,
)
from backend.app.services.v34_feature_service import (
    SUPPORTED_MARKETS,
    V34FeatureError,
    V34FeatureService,
    assert_159941_self_only,
)
from backend.app.services.v341_training_service import PROTOCOL_VERSION


FEATURE_SET_CORE = "CORE"
FEATURE_SET_EXTENDED = "EXTENDED"
FEATURE_SET_VERSION = "V3.4.1_FEATURE_MANIFEST_1"


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    name: str
    version: str
    ordered_feature_names: tuple[str, ...]
    group_counts: Mapping[str, int]
    manifest_hash: str


def build_feature_manifest(
    snapshot: FeatureSnapshot,
    *,
    feature_set_name: str = FEATURE_SET_EXTENDED,
) -> FeatureManifest:
    if feature_set_name not in {FEATURE_SET_CORE, FEATURE_SET_EXTENDED}:
        raise ValueError("unknown V3.4.1 feature set")
    flattened = flatten_snapshot(snapshot)
    if snapshot.market == "159941":
        assert_159941_self_only(flattened)
    names = tuple(
        sorted(
            name
            for name in flattened
            if feature_set_name == FEATURE_SET_EXTENDED or not name.startswith("seq100_")
        )
    )
    group_counts = {
        "weekly": sum(name.startswith("weekly_") for name in names),
        "daily_scalar": sum(name.startswith("daily_") for name in names),
        "daily_sequence": sum(name.startswith("seq100_") for name in names),
        "missing_masks": sum(name.startswith("missing_") for name in names),
        "point_in_time": sum(
            name.startswith(
                (
                    "pe",
                    "pb",
                    "earnings_",
                    "short_rate",
                    "long_rate",
                    "real_rate",
                    "term_spread",
                    "fund_flow",
                    "market_breadth",
                    "liquidity",
                    "inflation",
                    "pmi",
                    "employment",
                    "volatility_index",
                )
            )
            for name in names
        ),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "market": snapshot.market,
        "name": feature_set_name,
        "version": FEATURE_SET_VERSION,
        "ordered_feature_names": names,
        "group_counts": group_counts,
    }
    return FeatureManifest(
        name=feature_set_name,
        version=FEATURE_SET_VERSION,
        ordered_feature_names=names,
        group_counts=group_counts,
        manifest_hash=_hash(payload),
    )


def manifest_values(snapshot: FeatureSnapshot, manifest: FeatureManifest) -> dict[str, float | None]:
    values = flatten_snapshot(snapshot)
    if snapshot.market == "159941":
        assert_159941_self_only(values)
    return {name: values.get(name) for name in manifest.ordered_feature_names}


class V341FeatureService:
    """Build corrected PIT snapshots without changing frozen V3.4 semantics."""

    def __init__(self) -> None:
        self._base = V33FeatureService(require_weekly_bars=35, daily_window=100)
        self._v34 = V34FeatureService()

    @staticmethod
    def validate_market(market: str) -> None:
        if market not in SUPPORTED_MARKETS:
            raise V34FeatureError("V3.4.1 supports only 399006 and 159941")

    def weekly_anchors(
        self,
        session: Session,
        market: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[date, ...]:
        return self._v34.weekly_anchors(
            session, market, start_date=start_date, end_date=end_date
        )

    @staticmethod
    def _pit_observations(
        session: Session,
        market: str,
        anchor: date,
    ) -> tuple[PointInTimeObservation, ...]:
        cutoff_at = datetime.combine(anchor, time.max, tzinfo=timezone.utc)
        statement = (
            select(V33PointInTimeObservation)
            .where(
                V33PointInTimeObservation.effective_date <= anchor,
                V33PointInTimeObservation.available_at <= cutoff_at,
                V33PointInTimeObservation.cutoff_at <= cutoff_at,
                V33PointInTimeObservation.quality_status.in_(
                    ("accepted", "accepted_with_warnings")
                ),
                or_(
                    V33PointInTimeObservation.series_code.like(f"{market}.%"),
                    V33PointInTimeObservation.series_code.in_(
                        tuple(DEFAULT_PIT_SERIES_CODES)
                    ),
                ),
            )
            .order_by(
                V33PointInTimeObservation.series_code,
                V33PointInTimeObservation.effective_date,
                V33PointInTimeObservation.available_at,
                V33PointInTimeObservation.id,
            )
        )
        latest: dict[tuple[str, date], V33PointInTimeObservation] = {}
        for row in session.scalars(statement):
            latest[(str(row.series_code), row.effective_date)] = row
        observations: list[PointInTimeObservation] = []
        for row in latest.values():
            try:
                observations.append(V33FeatureService._pit_row_to_observation(row, market))
            except V33FeatureError:
                continue
        return tuple(observations)

    def load_snapshot(self, session: Session, market: str, anchor: date) -> FeatureSnapshot:
        self.validate_market(market)
        if market == "159941":
            snapshot = self._v34.load_snapshot(session, market, anchor)
            return replace(
                snapshot,
                provenance={
                    **snapshot.provenance,
                    "protocol_version": PROTOCOL_VERSION,
                    "pit_observation_count": 0,
                },
            )
        observations = self._pit_observations(session, market, anchor)
        try:
            snapshot = self._base.load_snapshot(
                session,
                market,
                anchor,
                observations=observations,
                weekly_cutoff_date=anchor,
            )
        except V33FeatureError as exc:
            raise V34FeatureError(str(exc)) from exc
        return replace(
            snapshot,
            provenance={
                **snapshot.provenance,
                "protocol_version": PROTOCOL_VERSION,
                "pit_observation_count": len(observations),
                "pit_cutoff_at": datetime.combine(
                    anchor, time.max, tzinfo=timezone.utc
                ).isoformat(),
            },
        )

    @staticmethod
    def audit_payload(
        snapshot: FeatureSnapshot,
        manifest: FeatureManifest,
    ) -> dict[str, Any]:
        values = manifest_values(snapshot, manifest)
        cutoff_at = datetime.combine(
            snapshot.cutoff_date, time.max, tzinfo=timezone.utc
        )
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "market": snapshot.market,
            "forecast_anchor_date": snapshot.cutoff_date.isoformat(),
            "cutoff_at": cutoff_at.isoformat(),
            "source_max_date": snapshot.source_data_max_date.isoformat(),
            "feature_manifest_hash": manifest.manifest_hash,
            "feature_set_name": manifest.name,
            "feature_set_version": manifest.version,
            "features": values,
            "daily_sequence": list(snapshot.daily_sequence),
            "provenance": dict(snapshot.provenance),
            "leakage_audit": {
                "effective_date_at_or_before_anchor": True,
                "available_at_at_or_before_cutoff": True,
                "source_max_date_at_or_before_anchor": (
                    snapshot.source_data_max_date <= snapshot.cutoff_date
                ),
                "159941_boundary": snapshot.provenance.get("data_boundary"),
            },
        }
        payload["snapshot_hash"] = _hash(payload)
        return payload

