"""V3.7 DeepSeek AI service: packet, independent analyst, cache, evaluation.

The AI layer is deliberately isolated from the local model: the independent
analyst packet never contains local predictions, StrategyScore, positions or
funnels.  Every call is frozen in ``v37_ai_requests`` / ``v37_ai_forecasts``;
the same input is never re-called or overwritten.  Historical inputs are
anonymized (MARKET_A / T-n / anchor_close=1.0).  AI accounts run through the
same execution engine as Local/Fusion accounts.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V35Forecast,
    V35BootstrapState,
    V35ForecastEvaluation,
    V35ModelPackage,
    V35Promotion,
    V35SimAccount,
    V35SimEvaluation,
    V35SimLedger,
    V36AccountSnapshot,
    V37AiCalibrator,
    V37AiEvaluation,
    V37AiForecast,
    V37AiModelHealth,
    V37AiRequest,
    V37ModelConflict,
    utc_now,
)
from backend.app.services.deepseek_client import (
    DeepSeekConfig,
    DeepSeekError,
    ERROR_INVALID_RESPONSE,
    chat_json_detailed,
)
from backend.app.services.v33_feature_service import (
    PriceBar,
    _aggregate_weekly,
    macd,
)
from backend.app.services.v35_simulation_service import (
    INITIAL_CAPITAL,
    load_weekly_bars,
    persist_window_result,
    run_standard_window,
)
from backend.app.services.v35_strategy_service import V35StrategyDecision
from backend.app.services.v36_calibration import (
    apply_method,
    fit_method,
)
from backend.app.services.v351_behavior_service import (
    effective_independent_window_count,
)
from backend.app.services.v37_config import (
    AI_CALIBRATION_FULL,
    AI_CALIBRATION_PRELIMINARY,
    AI_CALIBRATION_UNAVAILABLE,
    AI_CALIBRATION_PRELIMINARY_SAMPLES,
    AI_CALIBRATION_UNAVAILABLE_SAMPLES,
    AI_DIRECTION_TREND_VALUES,
    AI_MAX_ATTEMPTS,
    AI_PACKET_SCHEMA_VERSION,
    AI_PROMPT_VERSION_INDEPENDENT,
    AI_PROVIDER,
    AI_RETRYABLE_ERROR_CODES,
    AI_RESPONSE_CACHED,
    AI_RESPONSE_OK,
    AI_RISK_VALUES,
    AI_SCOPE_FORWARD_OOS,
    AI_SCOPE_HISTORICAL_SCREENING,
    AI_TREND_VALUES,
    AI_WEIGHT_DEGRADED,
    MULTI_TIMEFRAME_STATES,
    PROTOCOL_VERSION_37,
    ai_weight_cap,
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


def _daily_rows(session: Session, market: str, anchor: date) -> list[dict[str, Any]]:
    instrument = session.scalar(select(Instrument).where(Instrument.code == market))
    if instrument is None:
        raise ValueError(f"unknown instrument {market}")
    rows = session.scalars(
        select(MarketPrice)
        .where(
            MarketPrice.instrument_id == instrument.id,
            MarketPrice.timeframe == "daily",
            MarketPrice.trade_date <= anchor,
        )
        .order_by(MarketPrice.trade_date)
    ).all()
    return [
        {
            "date": row.trade_date,
            "open": float(row.open_price),
            "high": float(row.high_price),
            "low": float(row.low_price),
            "close": float(row.close_price),
            "volume": (
                None if row.volume is None else float(row.volume)
            ),
        }
        for row in rows
    ]


def _weekly_bars(session: Session, market: str, anchor: date) -> list[PriceBar]:
    instrument = session.scalar(select(Instrument).where(Instrument.code == market))
    if instrument is None:
        raise ValueError(f"unknown instrument {market}")
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
            raw_close=float(row.close_price),
            volume=None if row.volume is None else float(row.volume),
            turnover=None if row.turnover is None else float(row.turnover),
            source=row.source,
        )
        for row in rows
    ]
    return _aggregate_weekly(daily)


def _indicator_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    weekly: bool,
) -> list[dict[str, Any]]:
    closes = [float(row["close"]) for row in rows if row.get("close") is not None]
    volumes = [float(row["volume"]) for row in rows if row.get("volume") is not None]
    scale = closes[-1] if closes else 1.0
    dif = dea = histogram = None
    if len(closes) >= 35:
        dif_raw, dea_raw, hist_raw = macd(closes)
        dif = np.asarray(dif_raw, dtype=float)
        dea = np.asarray(dea_raw, dtype=float)
        histogram = np.asarray(hist_raw, dtype=float)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        close = float(row["close"])
        item: dict[str, Any] = {
            "date": row["date"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": close,
        }
        if index > 0:
            item["return"] = round(close / float(rows[index - 1]["close"]) - 1.0, 8)
        else:
            item["return"] = None
        for span in (5, 10, 20, 60):
            if len(closes) >= span and index + 1 >= span:
                item[f"ma{span}"] = round(
                    float(np.mean(closes[index - span + 1 : index + 1])), 8
                )
            else:
                item[f"ma{span}"] = None
        if dif is not None and index >= 2:
            item["dif"] = round(float(dif[index]) / scale, 8)
            item["dea"] = round(float(dea[index]) / scale, 8)
            item["macd"] = round(float(histogram[index]) / scale, 8)
            item["dif_first_change"] = round(
                float(dif[index] - dif[index - 1]) / scale, 8
            )
            if weekly:
                item["dif_second_change"] = round(
                    float((dif[index] - dif[index - 1]) - (dif[index - 1] - dif[index - 2]))
                    / scale,
                    8,
                )
        if index > 0 and len(volumes) > 1:
            visible = volumes[: index + 1]
            mean_volume = float(np.mean(visible[-21:])) if len(visible) >= 5 else None
            item["volume_ratio"] = (
                round(float(row["volume"]) / mean_volume, 6)
                if row.get("volume") is not None and mean_volume
                else None
            )
        else:
            item["volume_ratio"] = None
        result.append(item)
    # Drawdown for the weekly branch.
    if weekly and closes:
        peak = float("-inf")
        drawdowns: list[float] = []
        for value in closes:
            peak = max(peak, value)
            drawdowns.append(value / peak - 1.0)
        for item, dd in zip(result, drawdowns):
            item["drawdown"] = round(dd, 8)
    return result


def build_ai_packet(
    session: Session,
    market: str,
    anchor: date,
    *,
    anonymize: bool = False,
) -> dict[str, Any]:
    """Build MULTI_TIMEFRAME_AI_PACKET_V1 (daily + weekly + deterministic summary)."""

    daily = _daily_rows(session, market, anchor)
    if len(daily) > 60:
        daily = daily[-60:]
    weekly_bars = _weekly_bars(session, market, anchor)
    if len(weekly_bars) > 52:
        weekly_bars = weekly_bars[-52:]
    weekly_rows = [
        {
            "date": bar.trade_date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in weekly_bars
    ]
    daily_indicators = _indicator_rows(daily, weekly=False)
    weekly_indicators = _indicator_rows(weekly_rows, weekly=True)
    daily_closes = [float(row["close"]) for row in daily]
    weekly_closes = [float(row["close"]) for row in weekly_rows]
    daily_vol = (
        float(np.std(np.diff(np.log(np.asarray(daily_closes[-21:], dtype=float)))))
        * math.sqrt(252.0)
        if len(daily_closes) >= 21
        else None
    )
    weekly_vol = (
        float(np.std(np.diff(np.log(np.asarray(weekly_closes[-14:], dtype=float)))))
        * math.sqrt(52.0)
        if len(weekly_closes) >= 14
        else None
    )
    weekly_drawdown = (
        float(weekly_closes[-1] / max(weekly_closes) - 1.0)
        if weekly_closes
        else None
    )
    daily_trend = "BULLISH" if len(daily_closes) >= 20 and daily_closes[-1] > float(np.mean(daily_closes[-20:])) else "BEARISH" if len(daily_closes) >= 20 else "NEUTRAL"
    weekly_trend = "BULLISH" if len(weekly_closes) >= 20 and weekly_closes[-1] > float(np.mean(weekly_closes[-20:])) else "BEARISH" if len(weekly_closes) >= 20 else "NEUTRAL"
    summary = {
        "daily_trend": daily_trend,
        "weekly_trend": weekly_trend,
        "direction_agreement": daily_trend == weekly_trend,
        "daily_volatility_20d": daily_vol,
        "weekly_volatility_13w": weekly_vol,
        "current_drawdown": weekly_drawdown,
        "distance_from_52w_high": (
            round(float(weekly_closes[-1] / max(weekly_closes) - 1.0), 8)
            if weekly_closes
            else None
        ),
        "distance_from_52w_low": (
            round(float(weekly_closes[-1] / min(weekly_closes) - 1.0), 8)
            if weekly_closes and min(weekly_closes) > 0
            else None
        ),
        "current_week_incomplete": anchor.weekday() != 4,
    }
    account_position = None
    snapshot = session.scalar(
        select(V36AccountSnapshot)
        .where(
            V36AccountSnapshot.protocol_version == PROTOCOL_VERSION_37,
            V36AccountSnapshot.model_market == market,
            V36AccountSnapshot.anchor_date <= anchor,
        )
        .order_by(V36AccountSnapshot.anchor_date.desc())
    )
    if snapshot is not None:
        account_position = int(snapshot.position_pp)
    packet: dict[str, Any] = {
        "schema_version": AI_PACKET_SCHEMA_VERSION,
        "market": market if not anonymize else "MARKET_A",
        "anchor": anchor.isoformat() if not anonymize else "T",
        "anchor_close": (
            round(float(daily[-1]["close"]), 8) if daily else None
        ),
        "daily": [
            {
                **row,
                "date_relative": (
                    row["date"].isoformat()
                    if not anonymize
                    else f"T-{len(daily) - index - 1}"
                ),
            }
            for index, row in enumerate(daily_indicators)
        ],
        "weekly": [
            {
                **row,
                "week_relative": (
                    row["date"].isoformat()
                    if not anonymize
                    else f"T-{len(weekly_indicators) - index - 1}"
                ),
            }
            for index, row in enumerate(weekly_indicators)
        ],
        "summary": summary,
        "account_position_pp": account_position,
    }
    return anonymize_packet(packet) if anonymize else packet


def anonymize_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return an anonymized copy: MARKET_A, T-n, normalized prices, ratios."""

    anchor_close = float(packet.get("anchor_close") or 1.0)
    result = dict(packet)
    result["market"] = "MARKET_A"
    result["anchor"] = "T"
    result["anchor_close"] = 1.0
    result["daily"] = [
        {
            key: (
                round(float(value) / anchor_close, 8)
                if key in ("open", "high", "low", "close")
                and value is not None
                else value
            )
            for key, value in row.items()
        }
        for row in packet["daily"]
    ]
    result["weekly"] = [
        {
            key: (
                round(float(value) / anchor_close, 8)
                if key in ("open", "high", "low", "close")
                and value is not None
                else value
            )
            for key, value in row.items()
        }
        for row in packet["weekly"]
    ]
    return result


def packet_hash(packet: Mapping[str, Any]) -> str:
    return _hash(packet)


def validate_ai_forecast(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate the INDEPENDENT_ANALYST JSON output."""

    def enum_value(key: str, allowed: set[str]) -> str:
        value = parsed.get(key)
        if str(value) not in allowed:
            raise DeepSeekError(
                ERROR_INVALID_RESPONSE,
                f"{key} must be one of {sorted(allowed)}",
            )
        return str(value)

    def numeric(key: str, *, lower: float | None = None, upper: float | None = None) -> float:
        try:
            value = float(parsed.get(key))
        except (TypeError, ValueError) as exc:
            raise DeepSeekError(
                ERROR_INVALID_RESPONSE, f"{key} must be numeric"
            ) from exc
        if math.isnan(value) or math.isinf(value):
            raise DeepSeekError(ERROR_INVALID_RESPONSE, f"{key} must be finite")
        if lower is not None and value < lower:
            raise DeepSeekError(ERROR_INVALID_RESPONSE, f"{key} below {lower}")
        if upper is not None and value > upper:
            raise DeepSeekError(ERROR_INVALID_RESPONSE, f"{key} above {upper}")
        return value

    result: dict[str, Any] = {
        "trend_1w": enum_value("trend_1w", AI_TREND_VALUES),
        "trend_2w": enum_value("trend_2w", AI_TREND_VALUES),
        "trend_4w": enum_value("trend_4w", AI_TREND_VALUES),
        "trend_8w": enum_value("trend_8w", AI_TREND_VALUES),
        "daily_trend": enum_value("daily_trend", AI_DIRECTION_TREND_VALUES),
        "weekly_trend": enum_value("weekly_trend", AI_DIRECTION_TREND_VALUES),
        "multi_timeframe_state": enum_value(
            "multi_timeframe_state", set(MULTI_TIMEFRAME_STATES)
        ),
        "risk_level": enum_value("risk_level", AI_RISK_VALUES),
        "confidence_raw": numeric("confidence_raw", lower=0.0, upper=1.0),
        "expected_return_4w": numeric("expected_return_4w", lower=-1.0, upper=2.0),
        "expected_return_8w": numeric("expected_return_8w", lower=-1.0, upper=2.0),
        "support_distance_pct": numeric(
            "support_distance_pct", lower=-1.0, upper=1.0
        ),
        "resistance_distance_pct": numeric(
            "resistance_distance_pct", lower=-1.0, upper=1.0
        ),
    }
    direction_scores: dict[str, float] = {}
    for horizon in ("1w", "2w", "4w", "8w"):
        direction_scores[f"direction_score_{horizon}"] = numeric(
            f"direction_score_{horizon}", lower=0.0, upper=1.0
        )
    result["direction_scores"] = direction_scores
    reason_codes = parsed.get("reason_codes")
    if not isinstance(reason_codes, list) or not all(
        isinstance(code, str) and code.strip() for code in reason_codes
    ):
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "reason_codes must be a non-empty string list"
        )
    result["reason_codes"] = [str(code).strip() for code in reason_codes]
    return result


def build_independent_prompt(
    packet: Mapping[str, Any],
) -> tuple[str, str]:
    system = (
        "You are an independent multi-timeframe market analyst for an 8-week "
        "forecasting system. You must decide ONLY from the provided data; do "
        "not guess the asset identity; do not use any external historical "
        "knowledge or news. Output exactly one JSON object with no extra text. "
        "Fields: trend_1w/trend_2w/trend_4w/trend_8w in "
        "BULLISH|NEUTRAL_BULLISH|NEUTRAL|NEUTRAL_BEARISH|BEARISH; "
        "direction_score_1w..direction_score_8w numbers in [0,1]; "
        "daily_trend and weekly_trend in BULLISH|NEUTRAL|BEARISH; "
        "multi_timeframe_state in "
        "BOTH_BULLISH|BOTH_BEARISH|DAILY_BULLISH_WEEKLY_BEARISH|"
        "DAILY_BEARISH_WEEKLY_BULLISH|DAILY_RECOVERY_WEEKLY_UNCONFIRMED|"
        "WEEKLY_UPTREND_DAILY_PULLBACK|NEUTRAL_MIXED; risk_level in "
        "LOW|MEDIUM|HIGH; confidence_raw in [0,1]; expected_return_4w and "
        "expected_return_8w as decimal returns; support_distance_pct and "
        "resistance_distance_pct as decimals; reason_codes as a string array."
    )
    user = (
        "Analyze the following anonymized daily + weekly market packet and "
        "forecast the next 8 weeks.\n"
        + json.dumps(packet, ensure_ascii=False, default=str)
    )
    return system, user


def _existing_ai_forecast(
    session: Session,
    market: str,
    anchor: date,
    *,
    model_name: str,
    prompt_version: str,
    generation: int,
    screening_scope: str,
) -> V37AiForecast | None:
    return session.scalar(
        select(V37AiForecast)
        .where(
            V37AiForecast.protocol_version == PROTOCOL_VERSION_37,
            V37AiForecast.model_market == market,
            V37AiForecast.forecast_anchor_date == anchor,
            V37AiForecast.model_name == model_name,
            V37AiForecast.prompt_version == prompt_version,
            V37AiForecast.ai_generation_version == generation,
            V37AiForecast.screening_scope == screening_scope,
        )
    )


def ensure_ai_forecast(
    session: Session,
    market: str,
    anchor: date,
    config: DeepSeekConfig,
    *,
    anonymize: bool = False,
    generation: int = 1,
    screening_scope: str = AI_SCOPE_FORWARD_OOS,
    max_attempts: int = AI_MAX_ATTEMPTS,
    retry_delays: Sequence[float] = (),
) -> dict[str, Any]:
    """Frozen independent-analyst call with audit, dedupe and bounded retry.

    Dedupe applies only to successful frozen forecasts: once a success exists
    for the same input, it is never re-called or overwritten.  Transient
    failures (TIMEOUT / RATE_LIMITED / NETWORK_ERROR / MODEL_NOT_AVAILABLE)
    allow up to ``max_attempts`` total attempts; every failed attempt keeps
    its own immutable request row.
    """

    packet = build_ai_packet(
        session, market, anchor, anonymize=anonymize
    )
    input_hash = packet_hash(packet)
    existing = _existing_ai_forecast(
        session,
        market,
        anchor,
        model_name=config.model,
        prompt_version=AI_PROMPT_VERSION_INDEPENDENT,
        generation=generation,
        screening_scope=screening_scope,
    )
    if existing is not None:
        return {
            "cached": True,
            "status": AI_RESPONSE_CACHED,
            "ai_forecast_id": existing.id,
            "input_hash": input_hash,
        }
    system, user = build_independent_prompt(packet)
    last_error: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        request_timestamp = utc_now()
        try:
            parsed, elapsed_ms, usage = chat_json_detailed(
                config, system=system, user=user
            )
            validated = validate_ai_forecast(parsed)
        except DeepSeekError as exc:
            session.add(
                V37AiRequest(
                    protocol_version=PROTOCOL_VERSION_37,
                    model_market=market,
                    forecast_anchor_date=anchor,
                    provider=AI_PROVIDER,
                    model_name=config.model,
                    model_version=None,
                    prompt_version=AI_PROMPT_VERSION_INDEPENDENT,
                    schema_version=AI_PACKET_SCHEMA_VERSION,
                    ai_generation_version=generation,
                    attempt_number=attempt,
                    screening_scope=screening_scope,
                    input_hash=input_hash,
                    output_hash=None,
                    raw_response_hash=None,
                    response_status=exc.code,
                    request_timestamp=request_timestamp,
                    latency_ms=0,
                    token_usage_json={},
                    error_code=exc.code,
                    error_detail=exc.detail,
                    point_in_time_pass=True,
                    created_at=utc_now(),
                )
            )
            session.flush()
            last_error = {
                "status": exc.code,
                "error_code": exc.code,
                "error_detail": exc.detail,
                "attempt": attempt,
            }
            if exc.code not in AI_RETRYABLE_ERROR_CODES or attempt >= max_attempts:
                return {"cached": False, **last_error}
            if attempt - 1 < len(retry_delays) and retry_delays[attempt - 1] > 0:
                time.sleep(retry_delays[attempt - 1])
            continue
        output_hash = _hash(validated)
        request_row = V37AiRequest(
            protocol_version=PROTOCOL_VERSION_37,
            model_market=market,
            forecast_anchor_date=anchor,
            provider=AI_PROVIDER,
            model_name=config.model,
            model_version=None,
            prompt_version=AI_PROMPT_VERSION_INDEPENDENT,
            schema_version=AI_PACKET_SCHEMA_VERSION,
            ai_generation_version=generation,
            attempt_number=attempt,
            screening_scope=screening_scope,
            input_hash=input_hash,
            output_hash=output_hash,
            raw_response_hash=output_hash,
            response_status=AI_RESPONSE_OK,
            request_timestamp=request_timestamp,
            latency_ms=elapsed_ms,
            token_usage_json=usage,
            error_code=None,
            error_detail=None,
            point_in_time_pass=True,
            created_at=utc_now(),
        )
        session.add(request_row)
        session.flush()
        forecast_row = V37AiForecast(
            protocol_version=PROTOCOL_VERSION_37,
            model_market=market,
            forecast_anchor_date=anchor,
            ai_request_id=request_row.id,
            ai_generation_version=generation,
            model_name=config.model,
            prompt_version=AI_PROMPT_VERSION_INDEPENDENT,
            schema_version=AI_PACKET_SCHEMA_VERSION,
            screening_scope=screening_scope,
            input_hash=input_hash,
            trend_1w=validated["trend_1w"],
            trend_2w=validated["trend_2w"],
            trend_4w=validated["trend_4w"],
            trend_8w=validated["trend_8w"],
            direction_scores_json=validated["direction_scores"],
            daily_trend=validated["daily_trend"],
            weekly_trend=validated["weekly_trend"],
            multi_timeframe_state=validated["multi_timeframe_state"],
            risk_level=validated["risk_level"],
            confidence_raw=Decimal(str(round(validated["confidence_raw"], 8))),
            expected_return_4w=Decimal(
                str(round(validated["expected_return_4w"], 8))
            ),
            expected_return_8w=Decimal(
                str(round(validated["expected_return_8w"], 8))
            ),
            support_distance_pct=Decimal(
                str(round(validated["support_distance_pct"], 8))
            ),
            resistance_distance_pct=Decimal(
                str(round(validated["resistance_distance_pct"], 8))
            ),
            reason_codes_json=validated["reason_codes"],
            status="PENDING",
            structured_result_json=validated,
            output_hash=output_hash,
            created_at=utc_now(),
        )
        session.add(forecast_row)
        session.flush()
        return {
            "cached": False,
            "status": AI_RESPONSE_OK,
            "ai_forecast_id": forecast_row.id,
            "input_hash": input_hash,
            "elapsed_ms": elapsed_ms,
            "attempt": attempt,
            "token_usage": usage,
        }
    return {"cached": False, **(last_error or {})}


def _anchor_after(
    session: Session,
    market: str,
    anchor: date,
    *,
    steps: int,
) -> date | None:
    from backend.app.services.v37_feature_service import V37FeatureService

    anchors = V37FeatureService().weekly_anchors(session, market)
    try:
        index = anchors.index(anchor)
    except ValueError:
        return None
    target = index + steps
    return anchors[target] if target < len(anchors) else None


def _weekly_closes(session: Session, market: str) -> dict[date, float]:
    bars = _weekly_bars(session, market, date.max)
    return {bar.trade_date: float(bar.close) for bar in bars}


def evaluate_matured_ai_forecasts(
    session: Session,
    market: str,
    as_of: date,
) -> list[int]:
    """Evaluate matured (8-week) AI forecasts and persist V37AiEvaluation rows."""

    rows = session.scalars(
        select(V37AiForecast)
        .where(
            V37AiForecast.protocol_version == PROTOCOL_VERSION_37,
            V37AiForecast.model_market == market,
            V37AiForecast.screening_scope == AI_SCOPE_FORWARD_OOS,
            V37AiForecast.status == "PENDING",
            V37AiForecast.forecast_anchor_date < as_of,
        )
        .order_by(V37AiForecast.forecast_anchor_date)
    ).all()
    closes = _weekly_closes(session, market)
    created_ids: list[int] = []
    for row in rows:
        end_anchor = _anchor_after(
            session, market, row.forecast_anchor_date, steps=8
        )
        if end_anchor is None or end_anchor > as_of:
            continue
        start_close = closes.get(row.forecast_anchor_date)
        end_close = closes.get(end_anchor)
        if start_close is None or end_close is None or start_close <= 0.0:
            continue
        if session.scalar(
            select(V37AiEvaluation).where(
                V37AiEvaluation.ai_forecast_id == row.id
            )
        ) is not None:
            continue
        actual = end_close / start_close - 1.0
        direction_scores = dict(row.direction_scores_json)

        def direction_hit(trend: str, horizon: str) -> bool | None:
            if trend in ("BULLISH", "NEUTRAL_BULLISH"):
                return actual > 0.0
            if trend in ("BEARISH", "NEUTRAL_BEARISH"):
                return actual < 0.0
            return None

        hits = {
            f"{horizon}w": direction_hit(getattr(row, f"trend_{horizon}"), f"{horizon}w")
            for horizon in ("1", "2", "4", "8")
        }
        mae_4 = abs(float(row.expected_return_4w) - actual)
        mae_8 = abs(float(row.expected_return_8w) - actual)
        up_score_4 = float(direction_scores.get("direction_score_4w", 0.5))
        up_score_8 = float(direction_scores.get("direction_score_8w", 0.5))
        brier_4 = (up_score_4 - (1.0 if actual > 0.0 else 0.0)) ** 2
        brier_8 = (up_score_8 - (1.0 if actual > 0.0 else 0.0)) ** 2
        strong_up = actual >= (0.05 if market == "399006" else 0.08)
        strong_down = actual <= -0.08
        strong_up_recognized = bool(
            strong_up and row.trend_8w in ("BULLISH", "NEUTRAL_BULLISH")
        )
        strong_down_recognized = bool(
            strong_down and row.trend_8w in ("BEARISH", "NEUTRAL_BEARISH")
        )
        up_missed = bool(strong_up and not strong_up_recognized)
        down_false_alarm = bool(
            actual >= -0.03 and row.trend_8w in ("BEARISH", "NEUTRAL_BEARISH")
        )
        metrics = {
            "actual_return_8w": actual,
            "direction_hits": hits,
            "mae_4w": mae_4,
            "mae_8w": mae_8,
            "brier_4w": brier_4,
            "brier_8w": brier_8,
        }
        evaluation = V37AiEvaluation(
            ai_forecast_id=row.id,
            protocol_version=PROTOCOL_VERSION_37,
            model_market=market,
            forecast_anchor_date=row.forecast_anchor_date,
            evaluation_available_date=end_anchor,
            direction_hits_json=hits,
            mae_4w=Decimal(str(round(mae_4, 8))),
            mae_8w=Decimal(str(round(mae_8, 8))),
            brier_4w=Decimal(str(round(brier_4, 8))),
            brier_8w=Decimal(str(round(brier_8, 8))),
            strong_up_recognized=strong_up_recognized,
            strong_down_recognized=strong_down_recognized,
            up_missed=up_missed,
            down_false_alarm=down_false_alarm,
            metrics_json=metrics,
            evaluation_hash=_hash(metrics),
            created_at=utc_now(),
        )
        session.add(evaluation)
        row.status = "FULLY_MATURE_8W"
        created_ids.append(row.id)
    session.flush()
    return created_ids


def ai_calibration_status(
    session: Session,
    market: str,
    as_of: date,
    *,
    generation: int = 1,
    model_name: str | None = None,
) -> tuple[str, int, V37AiCalibrator | None]:
    """Return AI calibration status using only matured Forward-OOS samples."""

    statement = (
        select(V37AiEvaluation)
        .join(V37AiForecast, V37AiEvaluation.ai_forecast_id == V37AiForecast.id)
        .where(
            V37AiEvaluation.protocol_version == PROTOCOL_VERSION_37,
            V37AiEvaluation.model_market == market,
            V37AiEvaluation.evaluation_available_date <= as_of,
            V37AiForecast.screening_scope == AI_SCOPE_FORWARD_OOS,
            V37AiForecast.ai_generation_version == generation,
        )
        .order_by(V37AiEvaluation.evaluation_available_date)
    )
    if model_name is not None:
        statement = statement.where(V37AiForecast.model_name == model_name)
    evaluations = session.scalars(statement).all()
    count = len(evaluations)
    if count < AI_CALIBRATION_UNAVAILABLE_SAMPLES:
        return AI_CALIBRATION_UNAVAILABLE, count, None
    status = (
        AI_CALIBRATION_FULL
        if count >= AI_CALIBRATION_PRELIMINARY_SAMPLES
        else AI_CALIBRATION_PRELIMINARY
    )
    rows: list[tuple[float, int]] = []
    for evaluation in evaluations:
        forecast = session.get(V37AiForecast, evaluation.ai_forecast_id)
        if forecast is None:
            continue
        up_score = float(
            forecast.direction_scores_json.get("direction_score_8w", 0.5)
        )
        actual_up = int(float(evaluation.metrics_json["actual_return_8w"]) > 0.0)
        rows.append((up_score, actual_up))
    if not rows:
        return AI_CALIBRATION_UNAVAILABLE, count, None
    probabilities = [
        [score, (1.0 - score) / 2.0, (1.0 - score) / 2.0] for score, _actual in rows
    ]
    actuals = [0 if actual == 1 else 2 for _score, actual in rows]
    method, params = select_method(probabilities, actuals)
    version = f"AI8_{as_of.isoformat()}_{method}"
    calibrator_id = f"{PROTOCOL_VERSION_37}:{market}:8:{version}"
    calibrator = session.get(V37AiCalibrator, calibrator_id)
    if calibrator is None:
        payload = {
            "method": method,
            "sample_count": count,
            "fit_through": as_of.isoformat(),
        }
        calibrator = V37AiCalibrator(
            id=calibrator_id,
            protocol_version=PROTOCOL_VERSION_37,
            model_market=market,
            horizon_weeks=8,
            version=version,
            status=status,
            raw_sample_count=count,
            effective_sample_count=count,
            fit_through_date=as_of,
            effective_from_date=as_of,
            calibrator_params_json=params,
            metrics_json=payload,
            calibration_hash=_hash(payload),
            created_at=utc_now(),
        )
        session.add(calibrator)
        session.flush()
    return status, count, calibrator


def ai_weight_status(
    session: Session,
    market: str,
    anchor: date,
    *,
    generation: int = 1,
    model_name: str | None = None,
) -> dict[str, Any]:
    """AI weight gates + degradation detection from Forward-OOS evidence."""

    statement = (
        select(V37AiEvaluation)
        .join(V37AiForecast, V37AiEvaluation.ai_forecast_id == V37AiForecast.id)
        .where(
            V37AiEvaluation.protocol_version == PROTOCOL_VERSION_37,
            V37AiEvaluation.model_market == market,
            V37AiEvaluation.evaluation_available_date <= anchor,
            V37AiForecast.screening_scope == AI_SCOPE_FORWARD_OOS,
            V37AiForecast.ai_generation_version == generation,
        )
        .order_by(V37AiEvaluation.evaluation_available_date)
    )
    if model_name is not None:
        statement = statement.where(V37AiForecast.model_name == model_name)
    evaluations = session.scalars(statement).all()
    forward_count = len(evaluations)
    calibration_status, raw_count, _calibrator = ai_calibration_status(
        session, market, anchor, generation=generation, model_name=model_name
    )
    cap = ai_weight_cap(forward_count)

    champion = session.scalar(
        select(V35BootstrapState).where(
            V35BootstrapState.protocol_version == PROTOCOL_VERSION_37,
            V35BootstrapState.model_market == market,
        )
    )
    champion_package_id = champion.champion_package_id if champion else None
    ai_package_id = f"{PROTOCOL_VERSION_37}:{market}:DEEPSEEK_AI"

    account_rows = session.execute(
        select(
            V35SimAccount.model_package_id,
            V35SimEvaluation.window_end_date,
            V35SimEvaluation.net_return,
            V35SimEvaluation.max_drawdown,
            V35SimEvaluation.average_position_pp,
            V35SimEvaluation.no_action_window,
        )
        .join(V35SimAccount, V35SimEvaluation.account_id == V35SimAccount.id)
        .where(
            V35SimAccount.protocol_version == PROTOCOL_VERSION_37,
            V35SimAccount.model_market == market,
            V35SimAccount.model_package_id.in_(
                [
                    package_id
                    for package_id in (champion_package_id, ai_package_id)
                    if package_id is not None
                ]
            ),
            V35SimEvaluation.window_end_date <= anchor,
        )
    ).all()
    local_by_end: dict[date, Any] = {}
    ai_by_end: dict[date, Any] = {}
    for package_id, end_date, net_return, max_drawdown, avg_position, no_action in account_rows:
        target = local_by_end if package_id == champion_package_id else ai_by_end
        target[end_date] = {
            "net_return": float(net_return),
            "max_drawdown": float(max_drawdown),
            "average_position_pp": float(avg_position),
            "no_action_window": bool(no_action),
        }

    matched = [
        (end, ai_by_end[end], local_by_end[end])
        for end in sorted(ai_by_end)
        if end in local_by_end
    ]
    excesses = [ai["net_return"] - local["net_return"] for _end, ai, local in matched]
    drawdown_diffs = [
        ai["max_drawdown"] - local["max_drawdown"] for _end, ai, local in matched
    ]
    participation_diffs = [
        ai["average_position_pp"] - local["average_position_pp"]
        for _end, ai, local in matched
    ]
    no_action_ratio = (
        sum(1 for _end, ai, _local in matched if ai["no_action_window"])
        / len(matched)
        if matched
        else 1.0
    )
    avg_excess = float(np.mean(excesses)) if excesses else 0.0
    avg_drawdown_diff = float(np.mean(drawdown_diffs)) if drawdown_diffs else 0.0
    avg_participation_diff = (
        float(np.mean(participation_diffs)) if participation_diffs else 0.0
    )

    ai_brier = (
        float(np.mean([float(row.brier_8w) for row in evaluations]))
        if evaluations
        else 1.0
    )
    local_brier_rows = session.scalars(
        select(V35ForecastEvaluation)
        .join(V35Forecast, V35ForecastEvaluation.forecast_id == V35Forecast.id)
        .where(
            V35Forecast.protocol_version == PROTOCOL_VERSION_37,
            V35Forecast.model_market == market,
            V35ForecastEvaluation.horizon_weeks == 8,
            V35ForecastEvaluation.evaluation_available_date <= anchor,
        )
    ).all()
    local_brier = (
        float(
            np.mean(
                [
                    float(row.metrics_json.get("brier", 0.25))
                    for row in local_brier_rows
                ]
            )
        )
        if local_brier_rows
        else None
    )
    ai_mae = (
        float(np.mean([float(row.mae_8w) for row in evaluations]))
        if evaluations
        else None
    )
    local_mae_values = [
        float(row.metrics_json["mae_8w"])
        for row in local_brier_rows
        if isinstance(row.metrics_json.get("mae_8w"), (int, float))
    ]
    local_mae = float(np.mean(local_mae_values)) if local_mae_values else None

    brier_pass = local_brier is None or ai_brier <= local_brier + 0.02
    mae_pass = local_mae is None or ai_mae is None or ai_mae <= local_mae * 1.15
    calibration_valid = calibration_status in (
        AI_CALIBRATION_PRELIMINARY,
        AI_CALIBRATION_FULL,
    )
    account_not_worse = avg_excess >= -0.0005
    drawdown_ok = avg_drawdown_diff <= 0.02
    participation_ok = avg_participation_diff >= -0.05
    no_action_ok = no_action_ratio <= 0.60

    trailing_negative = 0
    for excess in reversed(excesses[-4:]):
        if excess < 0.0:
            trailing_negative += 1
        else:
            break
    degraded = bool(
        (matched and trailing_negative >= 4)
        or (matched and avg_participation_diff <= -0.05)
    )
    status = (
        AI_WEIGHT_DEGRADED
        if degraded
        else calibration_status
        if forward_count
        else AI_CALIBRATION_UNAVAILABLE
    )
    effective_cap = min(cap, 10) if degraded else cap
    payload = {
        "forward_oos_windows": forward_count,
        "raw_samples": raw_count,
        "weight_cap_pp": effective_cap,
        "matched_window_count": len(matched),
        "avg_excess_return": round(avg_excess, 6),
        "avg_drawdown_degradation_pp": round(avg_drawdown_diff * 100.0, 4),
        "avg_participation_diff_pp": round(avg_participation_diff * 100.0, 4),
        "no_action_window_ratio": round(no_action_ratio, 4),
        "ai_brier_8w": round(ai_brier, 6),
        "local_brier_8w": round(local_brier, 6) if local_brier is not None else None,
        "ai_mae_8w": round(ai_mae, 6) if ai_mae is not None else None,
        "local_mae_8w": round(local_mae, 6) if local_mae is not None else None,
        "degraded": degraded,
        "gates": {
            "calibration_valid": calibration_valid,
            "quality_ok": mae_pass and brier_pass,
            "mae_pass": mae_pass,
            "brier_pass": brier_pass,
            "account_not_worse": account_not_worse,
            "drawdown_ok": drawdown_ok,
            "participation_ok": participation_ok,
            "no_action_ok": no_action_ok,
        },
    }
    return {
        "status": status,
        "calibration_status": calibration_status,
        "forward_oos_window_count": forward_count,
        "weight_cap_pp": effective_cap,
        "metrics": payload,
    }


def ai_model_health(
    session: Session,
    market: str,
    anchor: date,
    *,
    generation: int = 1,
    model_name: str | None = None,
) -> V37AiModelHealth:
    """Upsert per-anchor AI health with gates and degradation state."""

    result = ai_weight_status(
        session, market, anchor, generation=generation, model_name=model_name
    )
    existing = session.scalar(
        select(V37AiModelHealth).where(
            V37AiModelHealth.protocol_version == PROTOCOL_VERSION_37,
            V37AiModelHealth.model_market == market,
            V37AiModelHealth.anchor_date == anchor,
        )
    )
    payload = dict(result["metrics"])
    health_hash = _hash(payload)
    if existing is None:
        existing = V37AiModelHealth(
            protocol_version=PROTOCOL_VERSION_37,
            model_market=market,
            anchor_date=anchor,
            status=result["status"],
            forward_oos_window_count=result["forward_oos_window_count"],
            weight_cap_pp=result["weight_cap_pp"],
            ai_calibration_status=result["calibration_status"],
            metrics_json=payload,
            health_hash=health_hash,
            created_at=utc_now(),
        )
        session.add(existing)
    else:
        existing.status = result["status"]
        existing.forward_oos_window_count = result["forward_oos_window_count"]
        existing.weight_cap_pp = result["weight_cap_pp"]
        existing.ai_calibration_status = result["calibration_status"]
        existing.metrics_json = payload
        existing.health_hash = health_hash
    session.flush()
    return existing


def evaluate_ai_challenger(
    session: Session,
    market: str,
    as_of: date,
    *,
    generation: int = 1,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate the DEEPSEEK_AI account vs Local Champion (promotion gates)."""

    champion = session.scalar(
        select(V35BootstrapState).where(
            V35BootstrapState.protocol_version == PROTOCOL_VERSION_37,
            V35BootstrapState.model_market == market,
        )
    )
    if champion is None or champion.champion_package_id is None:
        return {"decision": "AI_SHADOW", "reason": "NO_CHAMPION"}
    ai_package_id = f"{PROTOCOL_VERSION_37}:{market}:DEEPSEEK_AI"
    health = ai_weight_status(
        session, market, as_of, generation=generation, model_name=model_name
    )
    gates = dict(health["metrics"]["gates"])
    metrics = dict(health["metrics"])
    matched_count = int(metrics["matched_window_count"])
    if (
        matched_count < 8
        or health["calibration_status"] == AI_CALIBRATION_UNAVAILABLE
        or not gates["calibration_valid"]
    ):
        return {
            "decision": "AI_SHADOW",
            "reason": "INSUFFICIENT_OR_UNCALIBRATED",
            "matched_window_count": matched_count,
            "calibration_status": health["calibration_status"],
        }

    account_rows = session.execute(
        select(
            V35SimAccount.id,
            V35SimAccount.model_package_id,
            V35SimEvaluation.window_start_date,
            V35SimEvaluation.window_end_date,
            V35SimEvaluation.net_return,
            V35SimEvaluation.max_drawdown,
            V35SimEvaluation.average_position_pp,
            V35SimEvaluation.no_action_window,
        )
        .join(V35SimAccount, V35SimEvaluation.account_id == V35SimAccount.id)
        .where(
            V35SimAccount.protocol_version == PROTOCOL_VERSION_37,
            V35SimAccount.model_market == market,
            V35SimAccount.model_package_id.in_(
                [champion.champion_package_id, ai_package_id]
            ),
            V35SimEvaluation.window_end_date <= as_of,
        )
    ).all()
    local_by_end: dict[date, Any] = {}
    ai_by_end: dict[date, Any] = {}
    for (
        account_id,
        package_id,
        start,
        end,
        net_return,
        max_drawdown,
        avg_position,
        no_action,
    ) in account_rows:
        target = local_by_end if package_id == champion.champion_package_id else ai_by_end
        target[end] = {
            "account_id": account_id,
            "start": start,
            "net_return": float(net_return),
            "max_drawdown": float(max_drawdown),
            "average_position_pp": float(avg_position),
            "no_action_window": bool(no_action),
        }
    ends = sorted(end for end in ai_by_end if end in local_by_end)
    intervals = [
        (ai_by_end[end]["start"], end)
        for end in ends
        if ai_by_end[end]["start"] is not None
    ]
    independent = effective_independent_window_count(intervals)
    excesses = [
        ai_by_end[end]["net_return"] - local_by_end[end]["net_return"] for end in ends
    ]
    mean_excess = float(np.mean(excesses)) if excesses else 0.0
    win_rate = float(np.mean([1.0 if value > 0.0 else 0.0 for value in excesses])) if excesses else 0.0
    drawdown_degradation = float(
        np.mean(
            [
                ai_by_end[end]["max_drawdown"] - local_by_end[end]["max_drawdown"]
                for end in ends
            ]
        )
    ) if ends else 0.0
    participation_diff = float(
        np.mean(
            [
                ai_by_end[end]["average_position_pp"]
                - local_by_end[end]["average_position_pp"]
                for end in ends
            ]
        )
    ) if ends else 0.0
    no_action_ratio = (
        sum(1 for end in ends if ai_by_end[end]["no_action_window"]) / len(ends)
        if ends
        else 1.0
    )

    # Trade-path divergence from the shared ledger (same anchors).
    diverged = 0
    compared = 0
    for end in ends:
        local_account_id = local_by_end[end]["account_id"]
        ai_account_id = ai_by_end[end]["account_id"]
        local_actions = dict(
            session.execute(
                select(V35SimLedger.anchor_date, V35SimLedger.trade_action).where(
                    V35SimLedger.account_id == local_account_id
                )
            ).all()
        )
        ai_actions = dict(
            session.execute(
                select(V35SimLedger.anchor_date, V35SimLedger.trade_action).where(
                    V35SimLedger.account_id == ai_account_id
                )
            ).all()
        )
        for anchor in local_actions:
            if anchor in ai_actions:
                compared += 1
                if local_actions[anchor] != ai_actions[anchor]:
                    diverged += 1
    trade_path_divergence = float(diverged / compared) if compared else 0.0

    reason_codes: list[str] = []
    if mean_excess < 0.0030:
        reason_codes.append("EXCESS_RETURN_BELOW_0_30")
    if win_rate < 0.55:
        reason_codes.append("WIN_RATE_BELOW_THRESHOLD")
    if drawdown_degradation > 0.02:
        reason_codes.append("DRAWDOWN_GATE_FAILED")
    if participation_diff < -0.05:
        reason_codes.append("PARTICIPATION_GATE_FAILED")
    if no_action_ratio > 0.60:
        reason_codes.append("NO_ACTION_DEPENDENCY")
    if not gates["quality_ok"]:
        reason_codes.append("PREDICTION_QUALITY_GATE_FAILED")
    if trade_path_divergence < 0.03:
        reason_codes.append("INSUFFICIENT_BEHAVIOR_DIVERGENCE")
    if independent < 8:
        reason_codes.append("INSUFFICIENT_INDEPENDENT_WINDOWS")

    evaluation_metrics = {
        "matched_window_count": matched_count,
        "independent_window_count": independent,
        "mean_excess_return": round(mean_excess, 6),
        "win_rate": round(win_rate, 4),
        "drawdown_degradation_pp": round(drawdown_degradation * 100.0, 4),
        "participation_diff_pp": round(participation_diff * 100.0, 4),
        "no_action_window_ratio": round(no_action_ratio, 4),
        "trade_path_divergence_ratio": round(trade_path_divergence, 4),
        "ai_brier_8w": metrics.get("ai_brier_8w"),
        "local_brier_8w": metrics.get("local_brier_8w"),
    }
    if reason_codes:
        return {
            "decision": "REJECTED",
            "reason_codes": reason_codes,
            "metrics": evaluation_metrics,
        }

    from backend.app.services.v37_feature_service import V37FeatureService

    anchors = V37FeatureService().weekly_anchors(session, market)
    try:
        index = anchors.index(as_of)
        effective_from = anchors[index + 1] if index + 1 < len(anchors) else as_of
    except ValueError:
        effective_from = as_of
    challenge_id = f"{PROTOCOL_VERSION_37}:{market}:AI_CHALLENGE:{as_of.isoformat()}"
    existing_promotion = session.scalar(
        select(V35Promotion).where(V35Promotion.challenge_id == challenge_id)
    )
    if existing_promotion is None:
        payload = {
            "decision": "PROMOTED",
            "channel": "DEEPSEEK_AI_PROMOTION",
            **evaluation_metrics,
        }
        session.add(
            V35Promotion(
                protocol_version=PROTOCOL_VERSION_37,
                model_market=market,
                challenge_id=challenge_id,
                champion_package_id=champion.champion_package_id,
                candidate_package_id=ai_package_id,
                challenger_family="DEEPSEEK_AI",
                evaluation_window_count=matched_count,
                promotion_decision="PROMOTED",
                promotion_reason="DEEPSEEK_AI_INDEPENDENT_PROMOTION",
                effective_from_date=effective_from,
                excess_profit=Decimal("0"),
                excess_return=Decimal(str(round(mean_excess, 8))),
                metrics_json=payload,
                promotion_hash=_hash(payload),
                created_at=utc_now(),
            )
        )
        champion.promotion_count = (champion.promotion_count or 0) + 1
        session.flush()
    return {
        "decision": "PROMOTED",
        "channel": "DEEPSEEK_AI_PROMOTION",
        "effective_from": effective_from.isoformat(),
        "metrics": evaluation_metrics,
    }


def ai_target_position_pp(
    forecast: V37AiForecast,
    calibrated_up_probability: float | None,
) -> tuple[int, tuple[Mapping[str, Any], ...]]:
    """Deterministic AI position mapping (research/account comparison only)."""

    if calibrated_up_probability is None:
        return 0, ()
    probability = float(np.clip(calibrated_up_probability, 0.0, 1.0))
    risk_multiplier = 1.0 if forecast.risk_level in ("LOW", "MEDIUM") else 0.5
    if forecast.trend_8w in ("BEARISH", "NEUTRAL_BEARISH"):
        target = 0
    elif probability >= 0.60:
        target = 40
    elif probability >= 0.50:
        target = 20
    elif probability >= 0.40:
        target = 10
    else:
        target = 0
    target = int(round(target * risk_multiplier / 5.0) * 5)
    target = max(0, min(80, target))
    if target <= 0:
        return 0, ()
    first = min(10, target)
    batches: list[Mapping[str, Any]] = []
    remaining = target
    for batch_number in range(1, 5):
        if remaining <= 0:
            break
        change = first if batch_number == 1 else min(10, remaining)
        change = max(5, int(round(change / 5.0) * 5))
        batches.append(
            {
                "action": "BUY",
                "target_position_pp": target,
                "batch_change_pp": change,
                "cooldown_trading_days": 5,
            }
        )
        remaining -= change
    return target, tuple(batches)


def ai_decision(
    market: str,
    anchor: date,
    forecast: V37AiForecast,
    calibrated_up_probability: float | None,
) -> V35StrategyDecision:
    target, batches = ai_target_position_pp(forecast, calibrated_up_probability)
    return V35StrategyDecision(
        market=market,
        forecast_anchor_date=anchor.isoformat(),
        dif_trend_state="UNCONFIRMED",
        confirmation_status="UNCONFIRMED",
        market_state=forecast.multi_timeframe_state,
        strategy_score=float(forecast.confidence_raw) * 100.0,
        base_target_position_pp=target,
        state_position_cap_pp=80,
        final_target_position_pp=target,
        batches=batches,
        components={
            "path": "DEEPSEEK_AI",
            "calibrated_up_probability": calibrated_up_probability,
            "risk_level": forecast.risk_level,
        },
        reasons=("DEEPSEEK_AI_INDEPENDENT",),
    )


def ensure_ai_package(
    session: Session,
    market: str,
    champion_package: V35ModelPackage,
) -> V35ModelPackage:
    """Create/return a pseudo-package that scopes AI account rows."""

    version = f"{PROTOCOL_VERSION_37}:{market}:DEEPSEEK_AI"
    package = session.get(V35ModelPackage, version)
    if package is not None:
        return package
    strategy_config = {
        **dict(champion_package.strategy_config_json),
        "v351_model_config": {
            "path": "DEEPSEEK_AI",
            "protocol_version": PROTOCOL_VERSION_37,
        },
        "policy_version": "POLICY_V37_AI_1",
    }
    from backend.app.services.v35_runtime_service import _hash

    package = V35ModelPackage(
        id=version,
        protocol_version=PROTOCOL_VERSION_37,
        model_market=market,
        version=version,
        package_kind="DEEPSEEK_AI_ACCOUNT",
        parent_package_id=champion_package.id,
        prediction_model_id=champion_package.prediction_model_id,
        policy_version="POLICY_V37_AI_1",
        feature_version="V3.7_FEATURE_MANIFEST_1",
        promotion_rule_version="PROMOTION_RULE_V37_1",
        strategy_config_json=strategy_config,
        effective_from_date=champion_package.effective_from_date,
        package_hash=_hash(strategy_config),
        created_at=utc_now(),
    )
    session.add(package)
    session.flush()
    return package


def run_ai_window_accounts(
    session: Session,
    market: str,
    as_of: date,
    champion_package: V35ModelPackage,
) -> int:
    """Run 8-week AI accounts through the shared execution engine."""

    from backend.app.services.v35_runtime_service import _hash as rt_hash
    from backend.app.services.v37_feature_service import V37FeatureService

    ai_package = ensure_ai_package(session, market, champion_package)
    anchors = V37FeatureService().weekly_anchors(session, market)
    try:
        end_index = anchors.index(as_of)
    except ValueError:
        return 0
    window_index = end_index - 8 + 1
    if window_index < 0:
        return 0
    window = anchors[window_index : window_index + 8]
    if len(window) < 8 or window[-1] > as_of:
        return 0
    existing = session.scalar(
        select(V35SimAccount).where(
            V35SimAccount.model_market == market,
            V35SimAccount.scope == "STANDARD_8W",
            V35SimAccount.window_start_date == window[0],
            V35SimAccount.model_package_id == ai_package.id,
        )
    )
    if existing is not None and existing.status == "COMPLETED":
        return 0
    decisions: dict[date, V35StrategyDecision] = {}
    for anchor in window:
        forecast = session.scalar(
            select(V37AiForecast).where(
                V37AiForecast.protocol_version == PROTOCOL_VERSION_37,
                V37AiForecast.model_market == market,
                V37AiForecast.forecast_anchor_date == anchor,
                V37AiForecast.screening_scope == AI_SCOPE_FORWARD_OOS,
            )
        )
        if forecast is None:
            return 0
        _status, _count, calibrator = ai_calibration_status(
            session, market, anchor
        )
        calibrated = None
        if calibrator is not None:
            up_score = float(
                forecast.direction_scores_json.get("direction_score_8w", 0.5)
            )
            calibrated = apply_method(
                calibrator.calibrator_params_json, [up_score, 0.5, 0.5]
            )[0]
        decisions[anchor] = ai_decision(
            market, anchor, forecast, calibrated
        )
    bars = load_weekly_bars(session, market, window)
    initial_snapshot = None
    if window_index > 0:
        initial_snapshot = session.scalar(
            select(V36AccountSnapshot)
            .where(
                V36AccountSnapshot.protocol_version == PROTOCOL_VERSION_37,
                V36AccountSnapshot.model_market == market,
                V36AccountSnapshot.anchor_date == anchors[window_index - 1],
            )
        )
        if initial_snapshot is not None:
            initial_snapshot = {
                "equity": float(initial_snapshot.equity),
                "cash": float(initial_snapshot.cash),
                "position_pp": int(initial_snapshot.position_pp),
                "held_shares": float(initial_snapshot.held_shares),
                "average_cost": (
                    float(initial_snapshot.average_cost)
                    if initial_snapshot.average_cost is not None
                    else 0.0
                ),
                "sellable_shares": float(initial_snapshot.sellable_shares),
                "pending_sellable": [],
                "pending_batches": list(initial_snapshot.pending_batches_json),
            }
    result = run_standard_window(
        session,
        market=market,
        package_id=ai_package.id,
        anchors=window,
        decisions=decisions,
        bars=bars,
        config=dict(champion_package.strategy_config_json),
        accounting_mode="SHARES",
        initial_snapshot=initial_snapshot,
    )
    result_json = dict(result.result_json)
    result_json["path"] = "DEEPSEEK_AI"
    result = result.__class__(
        account_id=result.account_id,
        market=result.market,
        scope=result.scope,
        window_start=result.window_start,
        window_end=result.window_end,
        package_id=result.package_id,
        initial_capital=result.initial_capital,
        ending_equity=result.ending_equity,
        current_cash=result.current_cash,
        current_position_pp=result.current_position_pp,
        net_profit=result.net_profit,
        net_return=result.net_return,
        max_drawdown=result.max_drawdown,
        average_position_pp=result.average_position_pp,
        trade_count=result.trade_count,
        turnover=result.turnover,
        transaction_cost=result.transaction_cost,
        no_action_window=result.no_action_window,
        up_market_participation=result.up_market_participation,
        down_market_defense=result.down_market_defense,
        status=result.status,
        ledger=result.ledger,
        result_json=result_json,
        account_hash=result.account_hash,
    )
    persist_window_result(
        session,
        result,
        protocol_version=PROTOCOL_VERSION_37,
    )
    return 1


def record_model_conflict(
    session: Session,
    market: str,
    anchor: date,
    local_forecast: V35Forecast | None,
    ai_forecast: V37AiForecast | None,
) -> V37ModelConflict | None:
    """Persist Local vs AI consensus/conflict state for one anchor."""

    if ai_forecast is None:
        return None
    local_up = (
        float(local_forecast.horizon_probabilities_json.get("8", [0.5, 0.0, 0.5])[0])
        if local_forecast is not None
        else 0.5
    )
    ai_up = float(ai_forecast.direction_scores_json.get("direction_score_8w", 0.5))
    local_bull = local_up >= 0.5
    ai_bull = ai_up >= 0.5
    conflict_score = float(abs(local_up - ai_up))
    if local_bull and ai_bull:
        consensus = "CONSENSUS_STRONG_BULL" if min(local_up, ai_up) >= 0.65 else "CONSENSUS_BULL"
    elif not local_bull and not ai_bull:
        consensus = "CONSENSUS_STRONG_BEAR" if max(local_up, ai_up) <= 0.35 else "CONSENSUS_BEAR"
    elif abs(local_up - ai_up) < 0.10:
        consensus = "CONSENSUS_NEUTRAL"
    else:
        consensus = "CONSENSUS_NEUTRAL"
    if local_bull != ai_bull:
        conflict_state = "LOCAL_BULL_AI_BEAR" if local_bull else "LOCAL_BEAR_AI_BULL"
    elif conflict_score >= 0.25:
        conflict_state = "HIGH_MODEL_CONFLICT"
    else:
        conflict_state = "NONE"
    existing = session.scalar(
        select(V37ModelConflict).where(
            V37ModelConflict.protocol_version == PROTOCOL_VERSION_37,
            V37ModelConflict.model_market == market,
            V37ModelConflict.forecast_anchor_date == anchor,
        )
    )
    payload = {
        "consensus_state": consensus,
        "conflict_state": conflict_state,
        "local_up_probability_8w": local_up,
        "ai_up_probability_8w": ai_up,
        "conflict_score": conflict_score,
    }
    if existing is not None:
        return existing
    row = V37ModelConflict(
        protocol_version=PROTOCOL_VERSION_37,
        model_market=market,
        forecast_anchor_date=anchor,
        local_forecast_id=local_forecast.id if local_forecast is not None else None,
        ai_forecast_id=ai_forecast.id,
        consensus_state=consensus,
        conflict_state=conflict_state,
        local_up_probability_8w=Decimal(str(round(local_up, 6))),
        ai_up_probability_8w=Decimal(str(round(ai_up, 6))),
        conflict_score=Decimal(str(round(conflict_score, 6))),
        resolution="NONE",
        metrics_json=payload,
        conflict_hash=_hash(payload),
        created_at=utc_now(),
    )
    session.add(row)
    session.flush()
    return row
