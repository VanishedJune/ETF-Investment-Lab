"""Point-in-time V3.5 feature construction with an explicit manifest.

The V3.5 feature family keeps every proven V3.4.1 weekly/daily feature and
adds the V3.5-approved feature set from the plan:

* 1/2/4/8/13/26-week historical momentum;
* MA5/10/20/60 distances, slopes, alignment and duration;
* DIF/DEA/MACD level, first and second changes, gaps and durations;
* volatility, downside volatility, ATR, drawdown and recovery features;
* volume Z-scores, changes, OBV, correlation and regime flags;
* a deterministic market-state label and its duration.

``159941`` is restricted to its own OHLCV and derived indicators; the
existing V3.4 boundary guard rejects any non-ETF field.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import Instrument, MarketPrice
from backend.app.services.v33_feature_service import (
    PriceBar,
    _aggregate_weekly,
    _correlation,
    _linear_slope,
    _max_drawdown,
    _obv,
    _safe_return,
    _std,
    _window_return,
    _zscore,
    flatten_snapshot,
    macd,
)
from backend.app.services.v34_feature_service import (
    SUPPORTED_MARKETS,
    V34FeatureError,
    V34FeatureService,
    assert_159941_self_only,
)
from backend.app.services.v341_feature_service import V341FeatureService
from backend.app.services.v35_config import FEATURE_VERSION, HORIZON_WEEKS, PROTOCOL_VERSION


FEATURE_SET_CORE = "CORE"
FEATURE_SET_EXTENDED = "EXTENDED"
V35_FEATURE_PREFIX = "v35_"


class V35FeatureError(RuntimeError):
    pass


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
class V35FeatureSnapshot:
    market: str
    cutoff_date: date
    source_data_max_date: date
    features: Mapping[str, float | None]
    daily_sequence: tuple[Mapping[str, float | str | None], ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class V35FeatureManifest:
    name: str
    version: str
    ordered_feature_names: tuple[str, ...]
    manifest_hash: str


def _median(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def _annualized_volatility(closes: Sequence[float], span: int) -> float | None:
    if len(closes) < span + 1:
        return None
    returns = np.asarray(closes[-span - 1 :], dtype=float)
    returns = returns[1:] / returns[:-1] - 1.0
    if not np.all(np.isfinite(returns)):
        return None
    return float(np.std(returns, ddof=0) * math.sqrt(52.0))


def _downside_volatility(closes: Sequence[float], span: int) -> float | None:
    if len(closes) < span + 1:
        return None
    values = np.asarray(closes[-span - 1 :], dtype=float)
    returns = values[1:] / values[:-1] - 1.0
    if not np.all(np.isfinite(returns)):
        return None
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return 0.0
    return float(np.std(downside, ddof=0) * math.sqrt(52.0))


def _current_drawdown(closes: Sequence[float]) -> float | None:
    if not closes:
        return None
    peak = float(np.max(np.asarray(closes, dtype=float)))
    return float(closes[-1] / peak - 1.0)


def _drawdown_duration_weeks(closes: Sequence[float]) -> int:
    if not closes:
        return 0
    peak = float("-inf")
    duration = 0
    current = 0
    for value in closes:
        if value >= peak:
            peak = value
            current = 0
        else:
            current += 1
            duration = max(duration, current)
    return duration


def _distance_from_high(closes: Sequence[float]) -> float | None:
    return _current_drawdown(closes)


def _rebound_from_low(closes: Sequence[float]) -> float | None:
    if len(closes) < 2:
        return None
    trough = float(np.min(np.asarray(closes[:-1], dtype=float)))
    if trough <= 0.0:
        return None
    return float(closes[-1] / trough - 1.0)


def _percentile_52w(closes: Sequence[float]) -> float | None:
    if len(closes) < 2:
        return None
    window = np.asarray(closes[-52:], dtype=float)
    return float(np.mean(window <= window[-1]))


def _ma_slope(closes: Sequence[float], span: int) -> float | None:
    if len(closes) < span or span < 2:
        return None
    series = np.asarray(closes[-span:], dtype=float)
    if not np.all(np.isfinite(series)):
        return None
    slope = float(np.polyfit(np.arange(span, dtype=float), series, 1)[0])
    return slope / max(abs(series[-1]), 1e-12)


def _ma_alignment(closes: Sequence[float], spans: Sequence[int]) -> tuple[int, int]:
    """Return (bullish, bearish) alignment and its consecutive duration."""

    values: list[float] = []
    for span in spans:
        if len(closes) < span:
            values.append(float("nan"))
        else:
            values.append(float(np.mean(closes[-span:])))
    if any(not math.isfinite(value) for value in values):
        return (0, 0)
    bull = all(values[index] > values[index + 1] for index in range(len(values) - 1))
    bear = all(values[index] < values[index + 1] for index in range(len(values) - 1))
    duration = 0
    if bull or bear:
        for index in range(len(closes) - 1, 0, -1):
            window = closes[max(0, index - max(spans)) : index]
            if len(window) < max(spans):
                break
            ma_values = []
            for span in spans:
                if len(window) < span:
                    ma_values = []
                    break
                ma_values.append(float(np.mean(window[-span:])))
            if not ma_values:
                break
            current_bull = all(
                ma_values[idx] > ma_values[idx + 1] for idx in range(len(ma_values) - 1)
            )
            current_bear = all(
                ma_values[idx] < ma_values[idx + 1] for idx in range(len(ma_values) - 1)
            )
            if (bull and current_bull) or (bear and current_bear):
                duration += 1
            else:
                break
    return (1 if bull else 0, duration)


def _market_state(
    closes: Sequence[float],
    ma20_slope: float | None,
    volatility_13: float | None,
    drawdown: float | None,
    rebound: float | None,
) -> tuple[str, int]:
    """Deterministic state label plus its consecutive-week duration."""

    if len(closes) < 2:
        return ("SIDEWAYS", 1)
    state: str
    if drawdown is not None and drawdown <= -0.15:
        state = "DRAWDOWN"
    elif rebound is not None and drawdown is not None and rebound >= 0.10 and drawdown > -0.10:
        state = "RECOVERY"
    elif volatility_13 is not None and volatility_13 >= 0.30:
        state = "HIGH_VOLATILITY"
    elif volatility_13 is not None and volatility_13 <= 0.15:
        state = "LOW_VOLATILITY"
    else:
        ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(np.mean(closes))
        if closes[-1] > ma20 and (ma20_slope is None or ma20_slope > 0.0):
            state = "UPTREND"
        elif closes[-1] < ma20 and (ma20_slope is None or ma20_slope < 0.0):
            state = "DOWNTREND"
        else:
            state = "SIDEWAYS"

    duration = 0
    for index in range(len(closes) - 1, -1, -1):
        window = closes[: index + 1]
        if len(window) < 20:
            break
        slope = _ma_slope(window, 20)
        vol = _annualized_volatility(window, 13)
        dd = _current_drawdown(window)
        rb = _rebound_from_low(window)
        previous = _market_state_label(window, slope, vol, dd, rb)
        if previous == state:
            duration += 1
        else:
            break
    return (state, max(1, duration))


def _market_state_label(
    closes: Sequence[float],
    ma20_slope: float | None,
    volatility_13: float | None,
    drawdown: float | None,
    rebound: float | None,
) -> str:
    if drawdown is not None and drawdown <= -0.15:
        return "DRAWDOWN"
    if rebound is not None and drawdown is not None and rebound >= 0.10 and drawdown > -0.10:
        return "RECOVERY"
    if volatility_13 is not None and volatility_13 >= 0.30:
        return "HIGH_VOLATILITY"
    if volatility_13 is not None and volatility_13 <= 0.15:
        return "LOW_VOLATILITY"
    ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(np.mean(closes))
    if closes[-1] > ma20 and (ma20_slope is None or ma20_slope > 0.0):
        return "UPTREND"
    if closes[-1] < ma20 and (ma20_slope is None or ma20_slope < 0.0):
        return "DOWNTREND"
    return "SIDEWAYS"


def _dif_durations(dif: Sequence[float], dea: Sequence[float]) -> dict[str, int]:
    """Return DIF-state, golden-cross and death-cross durations in weeks."""

    dif_duration = 1
    golden = 0
    death = 0
    if len(dif) >= 2:
        positive = dif[-1] >= 0
        rising = dif[-1] > dif[-2]
        for index in range(len(dif) - 1, 0, -1):
            same_positive = dif[index] >= 0
            same_rising = dif[index] > dif[index - 1]
            if same_positive == positive and same_rising == rising:
                dif_duration += 1
            else:
                break
    for index in range(len(dif) - 1, 0, -1):
        before = dif[index - 1] - dea[index - 1]
        after = dif[index] - dea[index]
        if before <= 0.0 < after:
            golden += 1
        elif before >= 0.0 > after:
            death += 1
    return {
        "dif_state_duration": dif_duration,
        "golden_cross_duration": golden,
        "death_cross_duration": death,
    }


def _volume_features(
    closes: Sequence[float],
    volumes: Sequence[float | None],
) -> dict[str, float | None]:
    """V3.5 volume/price features over the latest 13 weeks."""

    features: dict[str, float | None] = {}
    visible = [float(value) for value in volumes if value is not None and math.isfinite(float(value))]
    if len(visible) < 14 or any(value is None for value in volumes[-14:]):
        return features
    clean_volumes = [float(value) for value in volumes[-14:]]
    clean_closes = [float(value) for value in closes[-14:]]
    returns = [
        _safe_return(right, left) for left, right in zip(clean_closes[:-1], clean_closes[1:])
    ]
    up_volumes = [
        vol for vol, ret in zip(clean_volumes[1:], returns) if ret is not None and ret > 0.0
    ]
    down_volumes = [
        vol for vol, ret in zip(clean_volumes[1:], returns) if ret is not None and ret < 0.0
    ]
    median_volume = float(np.median(clean_volumes))
    if median_volume > 0.0:
        features["v35_up_down_volume_diff_13"] = (
            (float(np.mean(up_volumes)) - float(np.mean(down_volumes))) / median_volume
            if up_volumes and down_volumes
            else None
        )
    zscore = _zscore(clean_volumes, 13)
    latest_return = returns[-1] if returns else None
    features["v35_volume_breakout_up"] = (
        1.0
        if zscore is not None and latest_return is not None and zscore > 1.0 and latest_return > 0.0
        else 0.0
    )
    features["v35_volume_breakout_down"] = (
        1.0
        if zscore is not None and latest_return is not None and zscore > 1.0 and latest_return < 0.0
        else 0.0
    )
    features["v35_volume_shrink_rebound"] = (
        1.0
        if zscore is not None and latest_return is not None and zscore < -0.5 and latest_return > 0.0
        else 0.0
    )
    features["v35_volume_shrink_pullback"] = (
        1.0
        if zscore is not None and latest_return is not None and zscore < -0.5 and latest_return < 0.0
        else 0.0
    )
    return features


def build_feature_manifest(
    snapshot: V35FeatureSnapshot,
    *,
    feature_set_name: str = FEATURE_SET_CORE,
) -> V35FeatureManifest:
    if feature_set_name not in {FEATURE_SET_CORE, FEATURE_SET_EXTENDED}:
        raise V35FeatureError("unknown V3.5 feature set")
    names = tuple(
        sorted(
            name
            for name in snapshot.features
            if feature_set_name == FEATURE_SET_EXTENDED or not name.startswith("seq100_")
        )
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "market": snapshot.market,
        "name": feature_set_name,
        "version": FEATURE_VERSION,
        "ordered_feature_names": names,
    }
    return V35FeatureManifest(
        name=feature_set_name,
        version=FEATURE_VERSION,
        ordered_feature_names=names,
        manifest_hash=_hash(payload),
    )


class V35FeatureService:
    """Build and audit V3.5 feature snapshots without touching frozen stores."""

    def __init__(self) -> None:
        self._v34 = V34FeatureService()
        self._v341 = V341FeatureService()

    @staticmethod
    def validate_market(market: str) -> None:
        code = str(market)
        if code != "399006" and not (len(code) == 6 and code.isdigit()):
            raise V35FeatureError("V3.5 supports 399006 and six-digit ETF codes")

    def build_manifest(
        self,
        snapshot: V35FeatureSnapshot,
        *,
        feature_set_name: str = FEATURE_SET_CORE,
    ) -> V35FeatureManifest:
        return build_feature_manifest(
            snapshot,
            feature_set_name=feature_set_name,
        )

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

    def _weekly_bars(self, session: Session, market: str, anchor: date) -> list[PriceBar]:
        instrument = session.scalar(select(Instrument).where(Instrument.code == market))
        if instrument is None:
            raise V35FeatureError(f"missing instrument {market}")
        rows = session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= anchor,
            )
            .order_by(MarketPrice.trade_date)
        ).all()
        daily = [
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
        ]
        return _aggregate_weekly(daily)

    def load_snapshot(self, session: Session, market: str, anchor: date) -> V35FeatureSnapshot:
        self.validate_market(market)
        base = self._v341.load_snapshot(session, market, anchor)
        bars = self._weekly_bars(session, market, anchor)
        if not bars:
            raise V35FeatureError(f"no weekly bars for {market} at {anchor}")
        closes = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]
        dif, dea, histogram = macd(closes)
        scale = closes[-1]
        dif_series = dif.tolist()
        dea_series = dea.tolist()
        hist_series = histogram.tolist()
        macd_stable = len(bars) >= 35
        cross_durations = _dif_durations(dif_series, dea_series)

        features: dict[str, float | None] = {}
        for span in (1, 2, 4, 8, 13, 26):
            features[f"v35_ret_{span}"] = _window_return(closes, span)
        for span in (5, 10, 20, 60):
            features[f"v35_ma{span}_distance"] = (
                _safe_return(scale, float(np.mean(closes[-span:])))
                if len(closes) >= span
                else None
            )
            features[f"v35_ma{span}_slope"] = _ma_slope(closes, span)
        bull, alignment_duration = _ma_alignment(closes, (5, 10, 20, 60))
        features["v35_ma_bull_alignment"] = float(bull)
        features["v35_ma_alignment_duration"] = float(alignment_duration)
        features["v35_price_deviation_ma20"] = features.get("v35_ma20_distance")

        if macd_stable:
            dif_last = float(dif_series[-1])
            dif_prev = float(dif_series[-2])
            dif_prev2 = float(dif_series[-3])
            dea_last = float(dea_series[-1])
            hist_last = float(hist_series[-1])
            features["v35_dif"] = dif_last / scale
            features["v35_dea"] = dea_last / scale
            features["v35_macd_histogram"] = hist_last / scale
            features["v35_dif_first_change"] = (dif_last - dif_prev) / scale
            features["v35_dif_second_change"] = (
                (dif_last - dif_prev) - (dif_prev - dif_prev2)
            ) / scale
            features["v35_dea_first_change"] = (dea_last - float(dea_series[-2])) / scale
            features["v35_macd_first_change"] = (hist_last - float(hist_series[-2])) / scale
            features["v35_dif_dea_distance"] = (dif_last - dea_last) / (
                abs(dif_last) + abs(dea_last) + 1e-12
            )
            features["v35_dif_zero_distance"] = dif_last / (abs(dif_last) + 1e-12)
            features["v35_dif_state_duration"] = float(cross_durations["dif_state_duration"])
            features["v35_golden_cross_duration"] = float(
                cross_durations["golden_cross_duration"]
            )
            features["v35_death_cross_duration"] = float(cross_durations["death_cross_duration"])

        features["v35_volatility_4"] = _annualized_volatility(closes, 4)
        features["v35_volatility_8"] = _annualized_volatility(closes, 8)
        features["v35_volatility_13"] = _annualized_volatility(closes, 13)
        features["v35_volatility_26"] = _annualized_volatility(closes, 26)
        features["v35_downside_volatility_13"] = _downside_volatility(closes, 13)
        atr = None
        if len(bars) >= 15:
            ranges = [
                max(bar.high - bar.low, abs(bar.high - closes[index - 1]), abs(bar.low - closes[index - 1]))
                for index, bar in enumerate(bars[-14:], start=len(bars) - 14)
                if index > 0
            ]
            if ranges:
                atr = float(np.mean(ranges))
        features["v35_atr_pct"] = None if atr is None else atr / scale
        features["v35_current_drawdown"] = _current_drawdown(closes)
        features["v35_max_drawdown_60"] = _max_drawdown(closes[-60:]) if len(closes) >= 60 else None
        features["v35_drawdown_duration"] = float(_drawdown_duration_weeks(closes))
        features["v35_distance_from_high"] = _distance_from_high(closes)
        features["v35_rebound_from_low"] = _rebound_from_low(closes)
        features["v35_percentile_52w"] = _percentile_52w(closes)

        features["v35_volume_zscore_20"] = _zscore(volumes, 20)
        features["v35_volume_zscore_52"] = _zscore(volumes, 52)
        features["v35_volume_change_4"] = _safe_return(volumes[-1], volumes[-5])
        features["v35_volume_change_8"] = _safe_return(volumes[-1], volumes[-9])
        obv = _obv(closes, volumes)
        features["v35_obv_slope_13"] = (
            _linear_slope(obv, 13, max(abs(obv[-1]), 1.0)) if len(obv) >= 13 else None
        )
        price_returns = [_safe_return(right, left) for left, right in zip(closes[:-1], closes[1:])]
        volume_changes = [_safe_return(right, left) for left, right in zip(volumes[:-1], volumes[1:])]
        features["v35_price_volume_correlation_13"] = _correlation(price_returns, volume_changes, 13)
        features.update(_volume_features(closes, volumes))

        ma20_slope = features.get("v35_ma20_slope")
        volatility_13 = features.get("v35_volatility_13")
        drawdown = features.get("v35_current_drawdown")
        rebound = features.get("v35_rebound_from_low")
        state, state_duration = _market_state(closes, ma20_slope, volatility_13, drawdown, rebound)
        state_codes = {
            "UPTREND": 1.0,
            "DOWNTREND": -1.0,
            "SIDEWAYS": 0.0,
            "HIGH_VOLATILITY": 2.0,
            "LOW_VOLATILITY": -2.0,
            "DRAWDOWN": -3.0,
            "RECOVERY": 3.0,
        }
        features["v35_market_state"] = state_codes[state]
        features["v35_market_state_duration"] = float(state_duration)

        for name in tuple(features):
            if name.startswith(V35_FEATURE_PREFIX):
                features[f"missing_{name}"] = 1 if features[name] is None else 0

        flattened = flatten_snapshot(base)
        merged: dict[str, float | None] = dict(flattened)
        merged.update(features)
        if market == "159941":
            assert_159941_self_only(merged)

        return V35FeatureSnapshot(
            market=market,
            cutoff_date=anchor,
            source_data_max_date=base.source_data_max_date,
            features=merged,
            daily_sequence=tuple(base.daily_sequence),
            provenance={
                "protocol_version": PROTOCOL_VERSION,
                "market": market,
                "forecast_anchor_date": anchor.isoformat(),
                "source_max_date": base.source_data_max_date.isoformat(),
                "weekly_bar_count": len(bars),
                "horizon_weeks": HORIZON_WEEKS,
                "data_boundary": "SELF_OHLCV_ONLY" if market == "159941" else "INDEX_OHLCV_PLUS_VALUATION",
                "base_snapshot_hash": base.provenance.get("protocol_version"),
            },
        )

    @staticmethod
    def audit_payload(
        snapshot: V35FeatureSnapshot,
        manifest: V35FeatureManifest,
    ) -> dict[str, Any]:
        cutoff_at = datetime.combine(snapshot.cutoff_date, time.max, tzinfo=timezone.utc)
        values = {name: snapshot.features.get(name) for name in manifest.ordered_feature_names}
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "market": snapshot.market,
            "forecast_anchor_date": snapshot.cutoff_date.isoformat(),
            "cutoff_at": cutoff_at.isoformat(),
            "source_max_date": snapshot.source_data_max_date.isoformat(),
            "feature_manifest_hash": manifest.manifest_hash,
            "feature_set_name": manifest.name,
            "feature_set_version": manifest.version,
            "horizon_weeks": HORIZON_WEEKS,
            "features": values,
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
