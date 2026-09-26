"""V3.5.1 analysis payloads: 8W forecast K-lines, strategy and turning checks."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    IndicatorRecord,
    MarketPrice,
    V35Forecast,
    V35PositionDecision,
    V35StrategySnapshot,
)
from backend.app.services.v351_config import PROTOCOL_VERSION_351


DISPLAY_VERSION = "V3.5.1_ANALYSIS"
PREDICTED_WEEKS = 8
HISTORY_LIMIT = 52


def _ema(values: Sequence[float], span: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    alpha = 2.0 / (span + 1.0)
    output = np.empty_like(array)
    output[0] = array[0]
    for index in range(1, len(array)):
        output[index] = alpha * array[index] + (1.0 - alpha) * output[index - 1]
    return output


def predicted_series_and_indicators(
    *,
    anchor_close: float,
    expected_path: Sequence[float],
    p10_path: Sequence[float],
    p90_path: Sequence[float],
    history_closes: Sequence[float],
    last_volume: float | None,
) -> dict[str, Any]:
    """Build 8 representative candles, P10/P90 bands and predicted MACD."""

    expected = [float(value) for value in expected_path]
    p10 = [float(value) for value in p10_path]
    p90 = [float(value) for value in p90_path]
    candles: list[dict[str, float | None]] = []
    previous_close = anchor_close
    volume = float(last_volume or 0.0)
    for week in range(PREDICTED_WEEKS):
        close = anchor_close * (1.0 + expected[week])
        open_price = previous_close
        high = anchor_close * (1.0 + max(p90[week], p10[week]))
        low = anchor_close * (1.0 + min(p10[week], p90[week]))
        candles.append(
            {
                "week": week + 1,
                "open": round(open_price, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": round(volume * (0.98**week), 0) if volume else None,
            }
        )
        previous_close = close
    combined_closes = [float(value) for value in history_closes] + [
        float(candle["close"]) for candle in candles
    ]
    ema12 = _ema(combined_closes, 12)
    ema26 = _ema(combined_closes, 26)
    dif = ema12 - ema26
    dea = _ema(dif.tolist(), 9)
    macd = 2.0 * (dif - dea)
    dif_first = np.concatenate(([0.0], np.diff(dif)))
    indicators = [
        {
            "week": week + 1,
            "dif": round(float(dif[-(PREDICTED_WEEKS - week)]), 6),
            "dea": round(float(dea[-(PREDICTED_WEEKS - week)]), 6),
            "macd": round(float(macd[-(PREDICTED_WEEKS - week)]), 6),
            "dif_first_change": round(
                float(dif_first[-(PREDICTED_WEEKS - week)]), 6
            ),
        }
        for week in range(PREDICTED_WEEKS)
    ]
    return {
        "candles": candles,
        "indicators": indicators,
        "predicted_dif": [float(value) for value in dif[-PREDICTED_WEEKS:]],
        "predicted_dif_first": [
            float(value) for value in dif_first[-PREDICTED_WEEKS:]
        ],
    }


def assess_turning_points(
    *,
    expected_path: Sequence[float],
    predicted_dif: Sequence[float],
    predicted_dif_first: Sequence[float],
    ma20_distance: float | None,
    ma20_slope: float | None,
    probabilities_8: Sequence[float],
) -> dict[str, Any]:
    """Deterministic 8W price/DIF turning windows and temporal consistency."""

    expected = [float(value) for value in expected_path]
    weekly = [expected[0]] + [
        expected[index] - expected[index - 1] for index in range(1, len(expected))
    ]
    probabilities = list(probabilities_8)
    up_edge = (
        float(probabilities[0]) - float(probabilities[2])
        if len(probabilities) == 3
        else 0.0
    )
    price_candidates: list[dict[str, Any]] = []
    for index in range(1, len(weekly) - 1):
        if weekly[index] > 0 and weekly[index - 1] < 0 and weekly[index + 1] < 0:
            kind = "top"
        elif weekly[index] < 0 and weekly[index - 1] > 0 and weekly[index + 1] > 0:
            kind = "bottom"
        else:
            continue
        confirmed = (
            (kind == "bottom" and up_edge > 0.05)
            or (kind == "top" and up_edge < -0.05)
            or (
                ma20_distance is not None
                and ma20_slope is not None
                and ((kind == "bottom" and ma20_slope > 0) or (kind == "top" and ma20_slope < 0))
            )
        )
        price_candidates.append(
            {
                "signal_kind": "PRICE",
                "turn_kind": kind,
                "classification": "VALID_TURN" if confirmed else "CANDIDATE",
                "confirmation_status": "CONFIRMED" if confirmed else "PENDING",
                "window_start_date": f"W{index + 1}",
                "window_end_date": f"W{index + 2}",
                "week": index + 1,
            }
        )
    dif_candidates: list[dict[str, Any]] = []
    for index in range(1, len(predicted_dif_first)):
        previous = predicted_dif_first[index - 1]
        current = predicted_dif_first[index]
        if (previous <= 0 < current) or (previous >= 0 > current):
            dif_candidates.append(
                {
                    "signal_kind": "DIF",
                    "turn_kind": "bottom" if current > 0 else "top",
                    "classification": "VALID_TURN",
                    "confirmation_status": "CONFIRMED",
                    "window_start_date": f"W{index + 1}",
                    "window_end_date": f"W{index + 2}",
                    "week": index + 1,
                }
            )
    consistent = any(
        abs(price_candidate["week"] - dif_candidate["week"]) <= 1
        for price_candidate in price_candidates
        if price_candidate["confirmation_status"] == "CONFIRMED"
        for dif_candidate in dif_candidates
    )
    confirmed_price = [
        candidate for candidate in price_candidates if candidate["confirmation_status"] == "CONFIRMED"
    ]
    confirmed_dif = [
        candidate for candidate in dif_candidates if candidate["confirmation_status"] == "CONFIRMED"
    ]
    return {
        "price_turn_status": (
            "CONFIRMED" if confirmed_price else "PENDING"
        ),
        "dif_turn_status": "CONFIRMED" if confirmed_dif else "PENDING",
        "consistency_status": (
            "TEMPORALLY_CONSISTENT"
            if consistent
            else "INCONSISTENT"
            if confirmed_price and confirmed_dif
            else "PENDING"
        ),
        "candidates": price_candidates + dif_candidates,
    }


def load_history(session: Session, market: str, anchor: date) -> dict[str, Any]:
    from backend.app.models.models import Instrument

    instrument = session.scalar(select(Instrument).where(Instrument.code == market))
    if instrument is None:
        return {"ohlcv": [], "indicators": [], "closes": [], "last_volume": None}
    prices = list(
        session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "weekly",
                MarketPrice.trade_date <= anchor,
            )
            .order_by(MarketPrice.trade_date)
        )
    )[-HISTORY_LIMIT:]
    indicator_rows = list(
        session.scalars(
            select(IndicatorRecord)
            .where(
                IndicatorRecord.instrument_id == instrument.id,
                IndicatorRecord.timeframe == "weekly",
                IndicatorRecord.indicator_date <= anchor,
            )
            .order_by(IndicatorRecord.indicator_date)
        )
    )
    indicator_by_date = {
        row.indicator_date: (row.indicator_values or {}).get("values", {})
        for row in indicator_rows
    }
    ohlcv = [
        {
            "week_end": row.trade_date.isoformat(),
            "open": float(row.open_price),
            "high": float(row.high_price),
            "low": float(row.low_price),
            "close": float(row.close_price),
            "volume": (
                float(row.volume * row.volume_multiplier)
                if row.volume is not None
                else None
            ),
        }
        for row in prices
    ]
    indicators = [
        {
            "week_end": row.trade_date.isoformat(),
            "dif": values.get("dif"),
            "dea": values.get("dea"),
            "macd": values.get("macd_histogram"),
            "dif_first_change": values.get("dif_first_change"),
        }
        for row in prices
        for values in [indicator_by_date.get(row.trade_date, {})]
    ]
    closes = [float(row.close_price) for row in prices]
    last_volume = (
        float(prices[-1].volume * prices[-1].volume_multiplier)
        if prices and prices[-1].volume is not None
        else None
    )
    return {
        "ohlcv": ohlcv,
        "indicators": indicators,
        "closes": closes,
        "last_volume": last_volume,
        "anchor_close": closes[-1] if closes else None,
    }


def build_forecast_payload(
    session: Session,
    *,
    market: str,
    forecast: V35Forecast,
    strategy_snapshot: V35StrategySnapshot | None,
    batches: Sequence[V35PositionDecision],
) -> dict[str, Any]:
    history = load_history(session, market, forecast.forecast_anchor_date)
    anchor_close = history["anchor_close"]
    if anchor_close is None:
        raise ValueError(f"no weekly close before {forecast.forecast_anchor_date}")
    p10_path = [float(value) for value in forecast.price_quantiles_json[0]]
    p50_path = [float(value) for value in forecast.price_quantiles_json[1]]
    p90_path = [float(value) for value in forecast.price_quantiles_json[2]]
    predicted = predicted_series_and_indicators(
        anchor_close=anchor_close,
        expected_path=forecast.expected_path_json,
        p10_path=p10_path,
        p90_path=p90_path,
        history_closes=history["closes"],
        last_volume=history["last_volume"],
    )
    strategy_payload = None
    if strategy_snapshot is not None:
        strategy_payload = build_strategy_payload(
            market,
            strategy_snapshot,
            batches,
        )
    assessment = (
        (strategy_snapshot.strategy_json or {}).get("turning_assessment")
        if strategy_snapshot is not None
        else None
    )
    reliability = float(forecast.model_reliability_score)
    return {
        "protocol_version": PROTOCOL_VERSION_351,
        "display_version": DISPLAY_VERSION,
        "market": market,
        "forecast_anchor_date": forecast.forecast_anchor_date.isoformat(),
        "model_version": forecast.model_version_id,
        "scenario_seed": forecast.scenario_seed,
        "scenario_count": forecast.scenario_count,
        "historical_ohlcv": history["ohlcv"],
        "historical_indicators": history["indicators"],
        "representative_ohlcv": predicted["candles"],
        "indicators": predicted["indicators"],
        "expected_path": [float(value) for value in forecast.expected_path_json],
        "price_quantiles": {
            "p10": p10_path,
            "p50": p50_path,
            "p90": p90_path,
        },
        "horizon_probabilities": forecast.horizon_probabilities_json,
        "model_reliability": {
            "score": reliability,
            "semantics": "V3.5.1 8W reliability score",
        },
        "health_status": forecast.health_status,
        "strategy": strategy_payload,
        "policy": (
            {
                "turning_assessment": assessment,
                "decision": {
                    "current_position": None,
                    "target_position": strategy_payload["final_target_position_pp"]
                    if strategy_payload
                    else None,
                    "batches": strategy_payload["batches"]
                    if strategy_payload
                    else [],
                    "empty_reason": strategy_payload["empty_reason"]
                    if strategy_payload
                    else None,
                },
            }
            if strategy_payload is not None
            else None
        ),
        "chart_semantics": {
            "history_length": len(history["ohlcv"]),
            "predicted_weeks": PREDICTED_WEEKS,
            "representative_path": "P50 cumulative-return candles",
            "frozen_forecast_unchanged": True,
        },
        "forecast_hash": forecast.forecast_hash,
    }


def build_strategy_payload(
    market: str,
    snapshot: V35StrategySnapshot,
    batches: Sequence[V35PositionDecision],
) -> dict[str, Any]:
    batch_rows = []
    for batch in batches:
        condition = dict(batch.condition_json or {})
        batch_rows.append(
            {
                "batch_number": batch.batch_number,
                "action": batch.action,
                "position_pp": batch.position_pp,
                "batch_change_pp": batch.batch_change_pp,
                "target_position_pp": batch.target_position_pp,
                "cooldown_trading_days": int(
                    condition.get("cooldown_trading_days", 5)
                ),
                "condition": condition.get("condition", ""),
                "earliest_execution_week": 1,
                "execution_window_start": snapshot.forecast_anchor_date.isoformat(),
                "execution_window_end": snapshot.forecast_anchor_date.isoformat(),
            }
        )
    empty_reason = None
    if not batch_rows:
        final_target = snapshot.final_target_position_pp
        if final_target <= 0:
            empty_reason = "NO_TARGET_POSITION"
        elif snapshot.confirmation_status in (
            "TOP_CONFIRMED",
            "BEARISH_CONFIRMED",
        ):
            empty_reason = "HARD_CONDITION_BLOCK"
        else:
            empty_reason = "WAITING_CONFIRMATION"
    return {
        "market": market,
        "forecast_anchor_date": snapshot.forecast_anchor_date.isoformat(),
        "dif_trend_state": snapshot.dif_trend_state,
        "confirmation_status": snapshot.confirmation_status,
        "strategy_score": float(snapshot.strategy_score),
        "base_target_position_pp": snapshot.base_target_position_pp,
        "state_position_cap_pp": snapshot.state_position_cap_pp,
        "final_target_position_pp": snapshot.final_target_position_pp,
        "batches": batch_rows,
        "empty_reason": empty_reason,
        "reasons": (snapshot.strategy_json or {}).get("reasons", []),
        "turning_assessment": (snapshot.strategy_json or {}).get(
            "turning_assessment"
        ),
        "strategy_hash": snapshot.strategy_hash,
    }
