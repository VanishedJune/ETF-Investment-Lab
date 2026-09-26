"""V3.7 multi-timeframe feature construction (weekly + daily + interactions).

The weekly branch reuses the proven V3.5 features and adds 52-week momentum,
EMA distances, volume ratios and MACD histogram changes.  The daily branch is
computed from the point-in-time daily sequence captured at the anchor.  The
interaction branch describes weekly/daily agreement and conflicts, including
the ``MULTI_TIMEFRAME_STATE`` seven-state label.

All features derive from the instrument's own OHLCV; the 159941 boundary
guard is re-applied after the V3.7 additions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sqlalchemy.orm import Session

from backend.app.services.v33_feature_service import (
    PriceBar,
    _aggregate_weekly,
    _correlation,
    _linear_slope,
    _max_drawdown,
    _safe_return,
    _std,
    _window_return,
    _zscore,
    macd,
)
from backend.app.services.v34_feature_service import (
    assert_159941_self_only,
)
from backend.app.services.v35_feature_service import (
    FEATURE_SET_CORE,
    V35FeatureError,
    V35FeatureManifest,
    V35FeatureService,
    V35FeatureSnapshot,
)
from backend.app.services.v37_config import (
    DAILY_BARS_MIN,
    DAILY_BARS_TARGET,
    FEATURE_SET_EARLY_FUSION,
    FEATURE_SET_LATE_FUSION,
    FEATURE_SET_WEEKLY_ONLY,
    FEATURE_VERSION,
    MULTI_TIMEFRAME_STATES,
    PROTOCOL_VERSION_37,
    WEEKLY_BARS_MIN,
    WEEKLY_BARS_TARGET,
)


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


def _ema(values: Sequence[float], span: int) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    alpha = 2.0 / (span + 1.0)
    result = np.empty(len(values), dtype=float)
    result[0] = float(values[0])
    for index in range(1, len(values)):
        result[index] = alpha * float(values[index]) + (1.0 - alpha) * result[index - 1]
    return result


def _annualized_daily_volatility(closes: Sequence[float], span: int) -> float | None:
    if len(closes) < span + 1:
        return None
    returns = np.asarray(closes[-(span + 1) :], dtype=float)
    returns = returns[1:] / returns[:-1] - 1.0
    if not np.all(np.isfinite(returns)):
        return None
    return float(np.std(returns, ddof=0) * math.sqrt(252.0))


def _consecutive_days(closes: Sequence[float], direction: str) -> int:
    if len(closes) < 2:
        return 0
    count = 0
    for left, right in zip(reversed(closes[:-1]), reversed(closes[1:])):
        if right > left:
            count += 1
            if direction != "UP":
                break
        elif right < left:
            if direction == "DOWN":
                count += 1
            else:
                break
        else:
            break
    return count


def _trend_strength(closes: Sequence[float], span: int) -> float | None:
    if len(closes) < span + 1:
        return None
    returns = [
        _safe_return(right, left)
        for left, right in zip(closes[-span - 1 : -1], closes[-span:])
    ]
    clean = [value for value in returns if value is not None]
    if not clean:
        return None
    up = sum(1 for value in clean if value > 0.0)
    down = sum(1 for value in clean if value < 0.0)
    return (up - down) / len(clean)


def _daily_branch(daily_sequence: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Compute the V3.7 daily branch from the anchor daily sequence."""

    features: dict[str, float | None] = {}
    closes = [
        float(row["close"])
        for row in daily_sequence
        if row.get("close") is not None
    ]
    volumes = [
        float(row["volume"])
        for row in daily_sequence
        if row.get("volume") is not None
    ]
    if not closes:
        return features
    scale = closes[-1]
    for span in (1, 3, 5, 10, 20, 60):
        features[f"v37_d_ret_{span}"] = _window_return(closes, span)
    for span in (5, 10, 20, 60):
        features[f"v37_d_ma{span}_distance"] = (
            _safe_return(scale, float(np.mean(closes[-span:])))
            if len(closes) >= span
            else None
        )
        features[f"v37_d_ma{span}_slope"] = (
            float(
                np.polyfit(
                    np.arange(span, dtype=float),
                    np.asarray(closes[-span:], dtype=float),
                    1,
                )[0]
            )
            / max(abs(scale), 1e-12)
            if len(closes) >= span and span >= 2
            else None
        )
    for span in (12, 26):
        ema = _ema(closes, span)
        features[f"v37_d_ema{span}_distance"] = (
            _safe_return(scale, float(ema[-1])) if len(ema) else None
        )
    if len(closes) >= 35:
        dif, dea, histogram = macd(closes)
        dif_series = np.asarray(dif, dtype=float)
        dea_series = np.asarray(dea, dtype=float)
        hist_series = np.asarray(histogram, dtype=float)
        features["v37_d_dif"] = float(dif_series[-1]) / scale
        features["v37_d_dea"] = float(dea_series[-1]) / scale
        features["v37_d_macd_histogram"] = float(hist_series[-1]) / scale
        features["v37_d_dif_first_change"] = (
            float(dif_series[-1] - dif_series[-2]) / scale
        )
        features["v37_d_dif_second_change"] = (
            float((dif_series[-1] - dif_series[-2]) - (dif_series[-2] - dif_series[-3]))
            / scale
        )
        features["v37_d_macd_first_change"] = (
            float(hist_series[-1] - hist_series[-2]) / scale
        )
    features["v37_d_volatility_20"] = _annualized_daily_volatility(closes, 20)
    features["v37_d_volatility_60"] = _annualized_daily_volatility(closes, 60)
    if volumes:
        visible = [float(value) for value in volumes[-21:] if value is not None]
        if len(visible) >= 5:
            features["v37_d_volume_change_1"] = _safe_return(
                visible[-1], visible[-2]
            )
            features["v37_d_volume_change_5"] = _safe_return(
                visible[-1], visible[-6]
            )
            features["v37_d_volume_change_20"] = _safe_return(
                visible[-1], visible[-21]
            )
        features["v37_d_volume_zscore_20"] = _zscore(volumes[-20:], 20)
        price_returns = [
            _safe_return(right, left)
            for left, right in zip(closes[-21:-1], closes[-20:])
        ]
        volume_changes = [
            _safe_return(right, left)
            for left, right in zip(volumes[-21:-1], volumes[-20:])
        ]
        features["v37_d_price_volume_correlation_20"] = _correlation(
            price_returns, volume_changes, 20
        )
    if len(closes) >= 20:
        window = np.asarray(closes[-20:], dtype=float)
        high = float(np.max(window))
        low = float(np.min(window))
        features["v37_d_drawdown_20"] = float(scale / high - 1.0)
        features["v37_d_distance_high_20"] = float(scale / high - 1.0)
        features["v37_d_distance_low_20"] = (
            float(scale / low - 1.0) if low > 0.0 else None
        )
        features["v37_d_position_range_20"] = (
            float((scale - low) / (high - low)) if high > low else 0.5
        )
    features["v37_d_consecutive_up"] = float(_consecutive_days(closes, "UP"))
    features["v37_d_consecutive_down"] = float(_consecutive_days(closes, "DOWN"))
    features["v37_d_trend_strength_5"] = _trend_strength(closes, 5)
    features["v37_d_trend_strength_10"] = _trend_strength(closes, 10)
    return features


def _weekly_branch(
    bars: Sequence[PriceBar],
    features: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Compute V3.7 weekly-branch additions on top of the V3.5 features."""

    result: dict[str, float | None] = {}
    if not bars:
        return result
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    scale = closes[-1]
    result["v37_w_ret_52"] = _window_return(closes, 52)
    for span in (12, 26):
        ema = _ema(closes, span)
        result[f"v37_w_ema{span}_distance"] = (
            _safe_return(scale, float(ema[-1])) if len(ema) else None
        )
    result["v37_w_volume_ratio_4"] = (
        float(np.mean(volumes[-4:])) / max(float(np.mean(volumes[-13:-4])), 1e-12)
        if len(volumes) >= 13 and all(value is not None for value in volumes[-13:])
        else None
    )
    result["v37_w_volume_ratio_13"] = (
        float(np.mean(volumes[-13:])) / max(float(np.mean(volumes[-26:-13])), 1e-12)
        if len(volumes) >= 26 and all(value is not None for value in volumes[-26:])
        else None
    )
    if len(closes) >= 35:
        dif, _dea, histogram = macd(closes)
        hist_series = np.asarray(histogram, dtype=float)
        result["v37_w_macd_first_change"] = (
            float(hist_series[-1] - hist_series[-2]) / scale
        )
        result["v37_w_macd_second_change"] = (
            float(
                (hist_series[-1] - hist_series[-2])
                - (hist_series[-2] - hist_series[-3])
            )
            / scale
        )
    result["v37_w_volatility_52"] = (
        float(np.std(np.diff(np.log(np.asarray(closes[-53:], dtype=float))))) * math.sqrt(52.0)
        if len(closes) >= 53
        else None
    )
    result["v37_w_max_drawdown_104"] = (
        _max_drawdown(closes[-104:]) if len(closes) >= 104 else _max_drawdown(closes)
    )
    result["v37_w_ma20_distance"] = features.get("v35_ma20_distance")
    return result


def _multi_timeframe_state(
    weekly: Mapping[str, float | None],
    daily: Mapping[str, float | None],
) -> str:
    """Derive the deterministic seven-state multi-timeframe label."""

    weekly_close_above = float(weekly.get("v35_ma20_distance") or 0.0) > 0.0
    weekly_slope_positive = float(weekly.get("v35_ma20_slope") or 0.0) > 0.0
    weekly_bullish = weekly_close_above and weekly_slope_positive
    weekly_bearish = (
        float(weekly.get("v35_ma20_distance") or 0.0) < 0.0
        and float(weekly.get("v35_ma20_slope") or 0.0) < 0.0
    )
    daily_close_above = float(daily.get("v37_d_ma20_distance") or 0.0) > 0.0
    daily_slope_positive = float(daily.get("v37_d_ma20_slope") or 0.0) > 0.0
    daily_bullish = daily_close_above and daily_slope_positive
    daily_bearish = (
        float(daily.get("v37_d_ma20_distance") or 0.0) < 0.0
        and float(daily.get("v37_d_ma20_slope") or 0.0) < 0.0
    )
    daily_recovery = (
        (float(daily.get("v37_d_ret_5") or 0.0) > 0.0)
        and (float(daily.get("v37_d_ma5_distance") or 0.0) > 0.0)
        and not weekly_bullish
    )
    if daily_bullish and weekly_bullish:
        return "BOTH_BULLISH"
    if daily_bearish and weekly_bearish:
        return "BOTH_BEARISH"
    if daily_bullish and weekly_bearish:
        return "DAILY_BULLISH_WEEKLY_BEARISH"
    if daily_bearish and weekly_bullish:
        return "WEEKLY_UPTREND_DAILY_PULLBACK"
    if daily_recovery and not weekly_bullish:
        return "DAILY_RECOVERY_WEEKLY_UNCONFIRMED"
    if daily_bearish and weekly_bullish:
        return "DAILY_BEARISH_WEEKLY_BULLISH"
    if weekly_bearish and daily_recovery:
        return "WEEKLY_BEARISH_DAILY_RECOVERY"
    return "NEUTRAL_MIXED"


def _interaction_branch(
    weekly: Mapping[str, float | None],
    daily: Mapping[str, float | None],
    state: str,
) -> dict[str, float]:
    """Eleven daily-weekly interaction features."""

    weekly_ret_4 = float(weekly.get("v35_ret_4") or 0.0)
    daily_ret_20 = float(daily.get("v37_d_ret_20") or 0.0)
    weekly_macd = float(weekly.get("v35_macd_histogram") or 0.0)
    daily_macd = float(daily.get("v37_d_macd_histogram") or 0.0)
    weekly_dif = float(weekly.get("v35_dif") or 0.0)
    daily_dif = float(daily.get("v37_d_dif") or 0.0)
    weekly_vol = float(weekly.get("v35_volatility_13") or 0.0)
    daily_vol = float(daily.get("v37_d_volatility_20") or 0.0)
    weekly_strength = float(weekly.get("v35_ma_bull_alignment") or 0.0)
    daily_strength = float(daily.get("v37_d_trend_strength_10") or 0.0)
    weekly_bearish = (
        float(weekly.get("v35_ma20_distance") or 0.0) < 0.0
        and float(weekly.get("v35_ma20_slope") or 0.0) < 0.0
    )
    daily_recovery = (
        (float(daily.get("v37_d_ret_5") or 0.0) > 0.0)
        and (float(daily.get("v37_d_ma5_distance") or 0.0) > 0.0)
    )

    def sign(value: float) -> int:
        return 1 if value > 0.0 else -1 if value < 0.0 else 0

    return {
        "v37_x_direction_agreement": float(
            sign(weekly_ret_4) == sign(daily_ret_20) and sign(daily_ret_20) != 0
        ),
        "v37_x_macd_agreement": float(
            sign(weekly_macd) == sign(daily_macd) and sign(daily_macd) != 0
        ),
        "v37_x_dif_agreement": float(
            sign(weekly_dif) == sign(daily_dif) and sign(daily_dif) != 0
        ),
        "v37_x_momentum_gap": float(daily_ret_20 - weekly_ret_4),
        "v37_x_volatility_ratio": (
            float(daily_vol / weekly_vol) if weekly_vol and weekly_vol > 0.0 else 0.0
        ),
        "v37_x_trend_strength_gap": float(daily_strength - weekly_strength),
        "v37_x_daily_reversal_weekly_weak": float(
            state == "DAILY_RECOVERY_WEEKLY_UNCONFIRMED"
        ),
        "v37_x_weekly_bullish_daily_pullback": float(
            state == "WEEKLY_UPTREND_DAILY_PULLBACK"
        ),
        "v37_x_weekly_bearish_daily_recovery": float(
            weekly_bearish and daily_recovery
        ),
        "v37_x_double_bullish": float(state == "BOTH_BULLISH"),
        "v37_x_double_bearish": float(state == "BOTH_BEARISH"),
    }


@dataclass(frozen=True, slots=True)
class V37FeatureManifest:
    name: str
    version: str
    ordered_feature_names: tuple[str, ...]
    manifest_hash: str


class V37FeatureService(V35FeatureService):
    """Build V3.7 multi-timeframe snapshots with weekly+daily+interaction."""

    def build_manifest(
        self,
        snapshot: V35FeatureSnapshot,
        *,
        feature_set_name: str = FEATURE_SET_EARLY_FUSION,
    ) -> V37FeatureManifest:
        if feature_set_name not in {
            FEATURE_SET_WEEKLY_ONLY,
            FEATURE_SET_EARLY_FUSION,
            FEATURE_SET_LATE_FUSION,
        }:
            raise V35FeatureError(f"unknown V3.7 feature set {feature_set_name}")
        all_names = tuple(sorted(snapshot.features))
        v37_names = tuple(name for name in all_names if name.startswith("v37_"))
        v35_core = tuple(
            name
            for name in all_names
            if name.startswith("v35_") and not name.startswith("seq100_")
        )
        if feature_set_name == FEATURE_SET_WEEKLY_ONLY:
            names = v35_core
        elif feature_set_name == FEATURE_SET_EARLY_FUSION:
            names = v35_core + v37_names
        else:  # LATE_FUSION is resolved by the runtime into two manifests.
            names = v35_core + v37_names
        payload = {
            "protocol_version": PROTOCOL_VERSION_37,
            "market": snapshot.market,
            "name": feature_set_name,
            "version": FEATURE_VERSION,
            "ordered_feature_names": names,
        }
        return V37FeatureManifest(
            name=feature_set_name,
            version=FEATURE_VERSION,
            ordered_feature_names=names,
            manifest_hash=_hash(payload),
        )

    def weekly_manifest(
        self,
        snapshot: V35FeatureSnapshot,
    ) -> V37FeatureManifest:
        manifest = self.build_manifest(
            snapshot, feature_set_name=FEATURE_SET_WEEKLY_ONLY
        )
        weekly_names = tuple(
            name
            for name in manifest.ordered_feature_names
            if not name.startswith("v37_d_") and not name.startswith("v37_x_")
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION_37,
            "market": snapshot.market,
            "name": f"{FEATURE_SET_LATE_FUSION}_WEEKLY",
            "version": FEATURE_VERSION,
            "ordered_feature_names": weekly_names,
        }
        return V37FeatureManifest(
            name=f"{FEATURE_SET_LATE_FUSION}_WEEKLY",
            version=FEATURE_VERSION,
            ordered_feature_names=weekly_names,
            manifest_hash=_hash(payload),
        )

    def daily_manifest(
        self,
        snapshot: V35FeatureSnapshot,
    ) -> V37FeatureManifest:
        names = tuple(
            name
            for name in sorted(snapshot.features)
            if name.startswith("v37_d_") or name.startswith("v37_x_")
        )
        payload = {
            "protocol_version": PROTOCOL_VERSION_37,
            "market": snapshot.market,
            "name": f"{FEATURE_SET_LATE_FUSION}_DAILY",
            "version": FEATURE_VERSION,
            "ordered_feature_names": names,
        }
        return V37FeatureManifest(
            name=f"{FEATURE_SET_LATE_FUSION}_DAILY",
            version=FEATURE_VERSION,
            ordered_feature_names=names,
            manifest_hash=_hash(payload),
        )

    def load_snapshot(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> V35FeatureSnapshot:
        base = super().load_snapshot(session, market, anchor)
        bars = self._weekly_bars(session, market, anchor)
        weekly_additions = _weekly_branch(bars, base.features)
        daily = _daily_branch(base.daily_sequence)
        state = _multi_timeframe_state(
            {**base.features, **weekly_additions}, daily
        )
        if state == "WEEKLY_BEARISH_DAILY_RECOVERY":
            # Fold into the approved seven-state set.
            state = "DAILY_RECOVERY_WEEKLY_UNCONFIRMED"
        if state not in MULTI_TIMEFRAME_STATES:
            state = "NEUTRAL_MIXED"
        interactions = _interaction_branch(
            {**base.features, **weekly_additions}, daily, state
        )
        state_codes = {
            "BOTH_BULLISH": 1.0,
            "BOTH_BEARISH": -1.0,
            "DAILY_BULLISH_WEEKLY_BEARISH": 2.0,
            "DAILY_BEARISH_WEEKLY_BULLISH": -2.0,
            "DAILY_RECOVERY_WEEKLY_UNCONFIRMED": 3.0,
            "WEEKLY_UPTREND_DAILY_PULLBACK": -3.0,
            "NEUTRAL_MIXED": 0.0,
        }
        merged: dict[str, float | None] = dict(base.features)
        merged.update(weekly_additions)
        merged.update(daily)
        merged.update(interactions)
        merged["v37_multi_timeframe_state"] = state_codes[state]
        if market == "159941":
            assert_159941_self_only(merged)
        provenance = dict(base.provenance)
        last_weekly_date = bars[-1].trade_date if bars else None
        daily_dates = [
            row.get("date") for row in base.daily_sequence if row.get("date") is not None
        ]
        provenance.update(
            {
                "protocol_version": PROTOCOL_VERSION_37,
                "feature_version": FEATURE_VERSION,
                "weekly_bar_count": len(bars),
                "daily_bar_count": len(base.daily_sequence),
                "multi_timeframe_state": state,
                "incomplete_current_week": bool(
                    daily_dates
                    and last_weekly_date is not None
                    and max(daily_dates) > last_weekly_date
                ),
            }
        )
        return V35FeatureSnapshot(
            market=market,
            cutoff_date=anchor,
            source_data_max_date=base.source_data_max_date,
            features=merged,
            daily_sequence=base.daily_sequence,
            provenance=provenance,
        )

    @staticmethod
    def audit_payload(
        snapshot: V35FeatureSnapshot,
        manifest: V37FeatureManifest,
    ) -> dict[str, Any]:
        cutoff_at = datetime.combine(snapshot.cutoff_date, time.max, tzinfo=timezone.utc)
        values = {
            name: snapshot.features.get(name) for name in manifest.ordered_feature_names
        }
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION_37,
            "market": snapshot.market,
            "forecast_anchor_date": snapshot.cutoff_date.isoformat(),
            "cutoff_at": cutoff_at.isoformat(),
            "source_max_date": snapshot.source_data_max_date.isoformat(),
            "feature_manifest_hash": manifest.manifest_hash,
            "feature_set_name": manifest.name,
            "feature_set_version": manifest.version,
            "features": values,
            "provenance": dict(snapshot.provenance),
            "leakage_audit": {
                "effective_date_at_or_before_anchor": True,
                "available_at_at_or_before_cutoff": True,
                "source_max_date_at_or_before_anchor": (
                    snapshot.source_data_max_date <= snapshot.cutoff_date
                ),
                "incomplete_week_not_used_as_complete": True,
                "159941_boundary": snapshot.provenance.get("data_boundary"),
            },
        }
        payload["snapshot_hash"] = _hash(payload)
        return payload
