"""Read-only market context + strict JSON validation for the AI analysis test."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    IndicatorRecord,
    Instrument,
    MarketPrice,
    V35ContinuousAccount,
    V35Forecast,
    V35StrategySnapshot,
)
from backend.app.services.deepseek_client import (
    DeepSeekConfig,
    DeepSeekError,
    ERROR_INVALID_RESPONSE,
    chat_json,
)


TREND_VALUES = {"BULLISH", "NEUTRAL", "BEARISH"}
RISK_VALUES = {"LOW", "MEDIUM", "HIGH"}
DECISION_VALUES = {"BUY", "HOLD", "SELL", "WAIT"}


def _latest_forecast(session: Session, market: str, protocol: str) -> V35Forecast | None:
    return session.scalar(
        select(V35Forecast)
        .where(
            V35Forecast.model_market == market,
            V35Forecast.protocol_version == protocol,
        )
        .order_by(V35Forecast.forecast_anchor_date.desc())
    )


def build_market_context(session: Session, market: str) -> dict[str, Any]:
    instrument = session.scalar(select(Instrument).where(Instrument.code == market))
    if instrument is None:
        raise ValueError(f"unknown instrument: {market}")
    daily = list(
        session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
            )
            .order_by(MarketPrice.trade_date)
        )
    )
    latest = daily[-1] if daily else None
    previous = daily[-2] if len(daily) >= 2 else None
    latest_close = float(latest.close_price) if latest else None
    change_pct = (
        (float(latest.close_price) / float(previous.close_price) - 1.0)
        if latest is not None and previous is not None and previous.close_price
        else None
    )
    recent_daily = [
        {
            "date": row.trade_date.isoformat(),
            "close": float(row.close_price),
        }
        for row in daily[-10:]
    ]

    def daily_return(days: int) -> float | None:
        if len(daily) < days + 1:
            return None
        base = float(daily[-days - 1].close_price)
        return latest_close / base - 1.0 if latest_close and base else None

    daily_indicator = session.scalar(
        select(IndicatorRecord)
        .where(
            IndicatorRecord.instrument_id == instrument.id,
            IndicatorRecord.timeframe == "daily",
        )
        .order_by(IndicatorRecord.indicator_date.desc())
    )
    daily_values = (
        (daily_indicator.indicator_values or {}).get("values", {})
        if daily_indicator
        else {}
    )
    daily_ma = {
        f"ma{span}": daily_values.get(f"ma{span}") for span in (5, 10, 20, 60)
    }
    weekly_closes = [
        float(row.close_price)
        for row in session.scalars(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "weekly",
            )
            .order_by(MarketPrice.trade_date)
        )
    ]
    last_weekly_date = (
        session.scalar(
            select(MarketPrice.trade_date)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "weekly",
            )
            .order_by(MarketPrice.trade_date.desc())
        )
        if weekly_closes
        else None
    )
    current_week_incomplete = bool(
        latest is not None
        and (last_weekly_date is None or latest.trade_date > last_weekly_date)
    )

    def window_return(span: int) -> float | None:
        if len(weekly_closes) < span + 1:
            return None
        base = weekly_closes[-span - 1]
        return weekly_closes[-1] / base - 1.0 if base else None

    indicator = session.scalar(
        select(IndicatorRecord)
        .where(
            IndicatorRecord.instrument_id == instrument.id,
            IndicatorRecord.timeframe == "weekly",
        )
        .order_by(IndicatorRecord.indicator_date.desc())
    )
    values = (indicator.indicator_values or {}).get("values", {}) if indicator else {}
    ma = {span: values.get(f"ma{span}") for span in (5, 10, 20, 60)}

    forecast_v36 = _latest_forecast(session, market, "V3.6_CAPITAL_DRIVEN_8W")
    forecast_v351 = _latest_forecast(
        session, market, "V3.5.1_EFFECTIVE_CHALLENGER_AND_DYNAMIC_ETF_SLOTS"
    )
    forecast = forecast_v36 or forecast_v351
    source_protocol = None
    forecast_payload: dict[str, Any] | None = None
    target_position = None
    if forecast is not None:
        source_protocol = forecast.protocol_version
        snapshot = session.scalar(
            select(V35StrategySnapshot)
            .where(
                V35StrategySnapshot.model_market == market,
                V35StrategySnapshot.protocol_version == forecast.protocol_version,
                V35StrategySnapshot.forecast_anchor_date
                == forecast.forecast_anchor_date,
            )
        )
        target_position = (
            snapshot.final_target_position_pp if snapshot is not None else None
        )
        forecast_payload = {
            "anchor": forecast.forecast_anchor_date.isoformat(),
            "expected_path": [float(v) for v in forecast.expected_path_json],
            "p50_8w": float(forecast.price_quantiles_json[1][-1])
            if forecast.price_quantiles_json
            else None,
            "horizon_probabilities": forecast.horizon_probabilities_json,
            "final_target_position_pp": target_position,
        }
    continuous = session.scalar(
        select(V35ContinuousAccount)
        .where(V35ContinuousAccount.model_market == market)
        .order_by(V35ContinuousAccount.updated_at.desc())
    )
    context = {
        "instrument_code": market,
        "latest_close": latest_close,
        "latest_trade_date": latest.trade_date.isoformat() if latest else None,
        "daily_change_pct": change_pct,
        "daily": {
            "closes": recent_daily,
            "return_1d": daily_return(1),
            "return_3d": daily_return(3),
            "return_5d": daily_return(5),
            "ma": {
                k: (float(v) if v is not None else None)
                for k, v in daily_ma.items()
            },
            "dif": (
                float(daily_values["dif"])
                if daily_values.get("dif") is not None
                else None
            ),
            "dea": (
                float(daily_values["dea"])
                if daily_values.get("dea") is not None
                else None
            ),
            "macd": (
                float(daily_values["macd_histogram"])
                if daily_values.get("macd_histogram") is not None
                else None
            ),
        },
        "current_week_incomplete": current_week_incomplete,
        "ma": {f"ma{k}": (float(v) if v is not None else None) for k, v in ma.items()},
        "dif": float(values["dif"]) if values.get("dif") is not None else None,
        "dea": float(values["dea"]) if values.get("dea") is not None else None,
        "macd": float(values["macd_histogram"])
        if values.get("macd_histogram") is not None
        else None,
        "return_1w": window_return(1),
        "return_4w": window_return(4),
        "return_8w": window_return(8),
        "forecast": forecast_payload,
        "forecast_source_protocol": source_protocol,
        "current_position_pp": (
            float(continuous.average_position_pp)
            if continuous is not None
            else None
        ),
    }
    return context


def context_hash(context: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(context, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()[:16]


def validate_analysis(parsed: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("trend_1_2w", "trend_4w", "trend_8w"):
        if str(parsed.get(key)) not in TREND_VALUES:
            raise DeepSeekError(
                ERROR_INVALID_RESPONSE,
                f"{key} must be one of {sorted(TREND_VALUES)}",
            )
    confidence = parsed.get("confidence")
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError) as exc:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "confidence must be numeric"
        ) from exc
    if not 0.0 <= confidence_float <= 1.0:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "confidence must be within [0, 1]"
        )
    if str(parsed.get("risk_level")) not in RISK_VALUES:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            f"risk_level must be one of {sorted(RISK_VALUES)}",
        )
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise DeepSeekError(ERROR_INVALID_RESPONSE, "summary must be non-empty string")
    decision = parsed.get("decision")
    if str(decision) not in DECISION_VALUES:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            f"decision must be one of {sorted(DECISION_VALUES)}",
        )
    decision_reason = parsed.get("decision_reason")
    if not isinstance(decision_reason, str) or not decision_reason.strip():
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            "decision_reason must be non-empty string",
        )
    return {
        "trend_1_2w": str(parsed["trend_1_2w"]),
        "trend_4w": str(parsed["trend_4w"]),
        "trend_8w": str(parsed["trend_8w"]),
        "confidence": confidence_float,
        "risk_level": str(parsed["risk_level"]),
        "summary": summary.strip(),
        "decision": str(decision),
        "decision_reason": decision_reason.strip(),
    }


def build_prompt(context: Mapping[str, Any]) -> tuple[str, str]:
    system = (
        "你是行情分析助手。请同时结合日K与周K两个周期综合分析，只输出一个 JSON 对象，"
        "不要输出任何其他文字。"
        "字段必须为 trend_1_2w/trend_4w/trend_8w（BULLISH|NEUTRAL|BEARISH）、"
        "confidence（0到1数字）、risk_level（LOW|MEDIUM|HIGH）、summary（简短中文分析）。"
        "额外必须包含 decision（BUY|HOLD|SELL|WAIT）与 decision_reason（简短中文理由）。"
        "若 current_week_incomplete=true，说明当前周尚未收完，需以最新日K为主判断短期方向。"
    )
    user = (
        "请基于以下只读行情数据给出未来1-2周/4周/8周方向判断：\n"
        + json.dumps(context, ensure_ascii=False, default=str)
    )
    return system, user


def analyze_market(
    session: Session,
    market: str,
    config: DeepSeekConfig,
) -> dict[str, Any]:
    context = build_market_context(session, market)
    system, user = build_prompt(context)
    parsed, elapsed_ms = chat_json(config, system=system, user=user)
    analysis = validate_analysis(parsed)
    return {
        "ok": True,
        "analysis": analysis,
        "forecast_source_protocol": context.get("forecast_source_protocol"),
        "context_hash": context_hash(context),
        "elapsed_ms": elapsed_ms,
    }
