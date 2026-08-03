"""Read-only V3.4.3 chart adapter over frozen V3.4.1/V3.4.2 records."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    IndicatorRecord,
    Instrument,
    MarketPrice,
    V341Forecast,
    V341ForecastEvaluation,
    V341TrainingIteration,
)
from backend.app.services.instrument_universe import MODEL_MARKETS
from backend.app.services.market_calendar import CalendarProvider
from backend.app.services.v341_runtime_service import EVALUATION_VERSION
from backend.app.services.v341_training_service import PROTOCOL_VERSION
from backend.app.services.v342_turning_policy_service import (
    TurningCandidate,
    V342PolicyError,
    assess_turning_points,
)


class V343MarketUIError(RuntimeError):
    pass


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _adjusted_ohlcv(row: MarketPrice) -> dict[str, object]:
    raw_close = float(row.close_price)
    if raw_close <= 0:
        raise V343MarketUIError(
            f"invalid non-positive close for {row.trade_date}: {raw_close}"
        )
    adjusted_close = float(row.adjusted_close_price or row.close_price)
    factor = adjusted_close / raw_close
    volume = (
        None
        if row.volume is None
        else float(row.volume * Decimal(row.volume_multiplier))
    )
    return {
        "week_end": row.trade_date.isoformat(),
        "open": float(row.open_price or row.close_price) * factor,
        "high": float(row.high_price or row.close_price) * factor,
        "low": float(row.low_price or row.close_price) * factor,
        "close": adjusted_close,
        "volume": volume,
        "source": row.source,
        "price_adjustment": "qfq" if row.adjusted_close_price is not None else "raw",
    }


def _indicator_payload(row: IndicatorRecord) -> Mapping[str, object]:
    return dict((row.indicator_values or {}).get("values", {}))


def _with_derivatives(
    rows: Sequence[IndicatorRecord],
) -> dict[date, dict[str, object]]:
    output: dict[date, dict[str, object]] = {}
    previous_dif: float | None = None
    previous_change: float | None = None
    for row in rows:
        values = _indicator_payload(row)
        dif = _finite(values.get("dif"))
        dea = _finite(values.get("dea"))
        macd = _finite(values.get("macd_histogram"))
        change = (
            None if dif is None or previous_dif is None else dif - previous_dif
        )
        second = (
            None if change is None or previous_change is None else change - previous_change
        )
        output[row.indicator_date] = {
            "week_end": row.indicator_date.isoformat(),
            "dif": dif,
            "dea": dea,
            "macd": macd,
            "dif_first_change": change,
            "dif_second_change": second,
        }
        if dif is None:
            previous_dif = None
            previous_change = None
        else:
            previous_dif = dif
            previous_change = change
    return output


def _confirmed_candidate(
    candidates: Sequence[TurningCandidate], signal_kind: str
) -> TurningCandidate | None:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.signal_kind == signal_kind
        and candidate.classification == "VALID_TURN"
        and candidate.confirmation_status == "CONFIRMED"
    ]
    return max(eligible, key=lambda item: item.prominence_value, default=None)


class V343MarketUIService:
    """Build display payloads without mutating any forecast or policy row."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        calendar: CalendarProvider,
    ) -> None:
        self.sessions = session_factory
        self.calendar = calendar

    @staticmethod
    def validate_market(market: str) -> None:
        if market not in MODEL_MARKETS:
            raise V343MarketUIError(
                f"V3.4.3 model display supports only 399006 and 159941, got {market!r}"
            )

    def enrich_forecast(
        self,
        market: str,
        payload: Mapping[str, Any],
        *,
        policy: Mapping[str, Any] | None = None,
        history_limit: int = 52,
    ) -> dict[str, Any]:
        """Attach aligned real weekly history to one immutable forecast payload."""

        self.validate_market(market)
        anchor = date.fromisoformat(str(payload["forecast_anchor_date"]))
        with self.sessions() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == market))
            if instrument is None:
                raise V343MarketUIError(f"unknown instrument: {market}")
            price_rows = list(
                session.scalars(
                    select(MarketPrice)
                    .where(
                        MarketPrice.instrument_id == instrument.id,
                        MarketPrice.timeframe == "weekly",
                        MarketPrice.trade_date <= anchor,
                    )
                    .order_by(MarketPrice.trade_date)
                )
            )
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
        indicator_by_date = _with_derivatives(indicator_rows)
        selected_prices = price_rows[-history_limit:]
        historical_ohlcv = [_adjusted_ohlcv(row) for row in selected_prices]
        # Preserve one indicator slot for every historical candle.  Filtering
        # missing rows would shift all later indicators onto the wrong dates in
        # the browser.  Missing values remain explicit chart gaps.
        historical_indicators = []
        for row in selected_prices:
            values = indicator_by_date.get(row.trade_date)
            historical_indicators.append(
                values
                if values is not None
                else {
                    "week_end": row.trade_date.isoformat(),
                    "dif": None,
                    "dea": None,
                    "macd": None,
                    "dif_first_change": None,
                    "dif_second_change": None,
                }
            )
        enriched = dict(payload)
        enriched.update(
            {
                "display_version": "V3.4.3_MARKET_UI",
                "historical_ohlcv": historical_ohlcv,
                "historical_indicators": historical_indicators,
                "policy": None if policy is None else dict(policy),
                "chart_semantics": {
                    "historical_price_adjustment": "qfq where available",
                    "display_only_join": True,
                    "frozen_forecast_unchanged": True,
                    "representative_path": "one complete scenario medoid",
                    "dif_first_change": "DIF_t - DIF_(t-1)",
                },
            }
        )
        return enriched

    def iteration_curve(self, market: str) -> dict[str, Any]:
        """Return mature evaluation points and real-session turning deviations."""

        self.validate_market(market)
        with self.sessions() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == market))
            if instrument is None:
                raise V343MarketUIError(f"unknown instrument: {market}")
            joined = list(
                session.execute(
                    select(V341TrainingIteration, V341Forecast, V341ForecastEvaluation)
                    .join(V341Forecast, V341Forecast.id == V341TrainingIteration.forecast_id)
                    .outerjoin(
                        V341ForecastEvaluation,
                        (V341ForecastEvaluation.forecast_id == V341Forecast.id)
                        & (V341ForecastEvaluation.horizon_weeks == 13)
                        & (
                            V341ForecastEvaluation.evaluation_version
                            == EVALUATION_VERSION
                        ),
                    )
                    .where(
                        V341TrainingIteration.protocol_version == PROTOCOL_VERSION,
                        V341TrainingIteration.model_market == market,
                        V341Forecast.protocol_version == PROTOCOL_VERSION,
                        V341Forecast.model_market == market,
                    )
                    .order_by(V341TrainingIteration.weekly_iteration_number)
                )
            )
            weekly_prices = list(
                session.scalars(
                    select(MarketPrice)
                    .where(
                        MarketPrice.instrument_id == instrument.id,
                        MarketPrice.timeframe == "weekly",
                    )
                    .order_by(MarketPrice.trade_date)
                )
            )
            weekly_indicators = list(
                session.scalars(
                    select(IndicatorRecord)
                    .where(
                        IndicatorRecord.instrument_id == instrument.id,
                        IndicatorRecord.timeframe == "weekly",
                    )
                    .order_by(IndicatorRecord.indicator_date)
                )
            )
        weekly_dates = [row.trade_date for row in weekly_prices]
        indicator_by_date = _with_derivatives(weekly_indicators)
        points: list[dict[str, object]] = []
        for iteration, forecast, evaluation in joined:
            maturity = str(forecast.maturity_status)
            mature = evaluation is not None and maturity == "FULLY_MATURE_13W"
            price_days = dif_days = None
            if mature:
                price_days, dif_days = self._turning_deviations(
                    market,
                    forecast,
                    weekly_prices,
                    weekly_dates,
                    indicator_by_date,
                )
            metrics = {} if evaluation is None else dict(evaluation.metrics_json or {})
            available = (
                evaluation.evaluation_available_date
                if mature and evaluation is not None
                else iteration.evaluation_available_date
            )
            deviations = [value for value in (price_days, dif_days) if value is not None]
            points.append(
                {
                    "iteration": iteration.weekly_iteration_number,
                    "weekly_iteration_index": iteration.weekly_iteration_number,
                    "anchor_date": iteration.anchor_date.isoformat(),
                    "forecast_anchor_date": iteration.anchor_date.isoformat(),
                    "evaluation_available_date": (
                        None if available is None else available.isoformat()
                    ),
                    "maturity_status": maturity,
                    "pending": not mature,
                    "normalized_endpoint_error": (
                        _finite(metrics.get("normalized_abs_error")) if mature else None
                    ),
                    "price_turn_deviation_trading_days": price_days if mature else None,
                    "dif_turn_deviation_trading_days": dif_days if mature else None,
                    "mean_turn_deviation_trading_days": (
                        sum(deviations) / len(deviations) if deviations else None
                    ),
                    "turn_deviation_component_count": len(deviations),
                    "training_triggered": bool(iteration.training_triggered),
                    "promoted": bool(iteration.promoted),
                    "forecast_model_id": iteration.forecast_model_id,
                    "champion_after_model_id": iteration.champion_after_model_id,
                    "candidate": dict(iteration.validation_json or {}).get("candidate", {}),
                }
            )
        return {
            "market": market,
            "version": "V3.4.3_MARKET_UI",
            "horizon_weeks": 13,
            "x_axis": [
                "forecast_anchor_date",
                "weekly_iteration_index",
                "evaluation_available_date",
            ],
            "points": points,
        }

    def _turning_deviations(
        self,
        market: str,
        forecast: V341Forecast,
        weekly_prices: Sequence[MarketPrice],
        weekly_dates: Sequence[date],
        indicator_by_date: Mapping[date, Mapping[str, object]],
    ) -> tuple[int | None, int | None]:
        anchor = forecast.forecast_anchor_date
        end = forecast.label_end_date
        history_end = bisect_right(weekly_dates, anchor)
        future_end = bisect_right(weekly_dates, end)
        historical_rows = weekly_prices[max(0, history_end - 60) : history_end]
        future_rows = weekly_prices[history_end:future_end][:13]
        if len(historical_rows) < 20 or len(future_rows) != 13:
            return None, None
        historical = [_adjusted_ohlcv(row) for row in historical_rows]
        actual_ohlcv = [_adjusted_ohlcv(row) for row in future_rows]
        actual_indicators = [indicator_by_date.get(row.trade_date) for row in future_rows]
        if any(
            values is None
            or values.get("dif") is None
            or values.get("dea") is None
            or values.get("macd") is None
            or values.get("dif_first_change") is None
            or values.get("dif_second_change") is None
            for values in actual_indicators
        ):
            return None, None
        try:
            predicted = assess_turning_points(
                forecast.representative_ohlcv_json,
                forecast.indicator_path_json,
                historical,
            )
            actual = assess_turning_points(
                actual_ohlcv,
                actual_indicators,  # type: ignore[arg-type]
                historical,
            )
        except (TypeError, ValueError, V342PolicyError):
            return None, None
        return (
            self._candidate_deviation(market, predicted.candidates, actual.candidates, "PRICE"),
            self._candidate_deviation(market, predicted.candidates, actual.candidates, "DIF"),
        )

    def _candidate_deviation(
        self,
        market: str,
        predicted: Sequence[TurningCandidate],
        actual: Sequence[TurningCandidate],
        signal_kind: str,
    ) -> int | None:
        forecast_candidate = _confirmed_candidate(predicted, signal_kind)
        if forecast_candidate is None:
            return None
        matches = [
            candidate
            for candidate in actual
            if candidate.signal_kind == signal_kind
            and candidate.turn_kind == forecast_candidate.turn_kind
            and candidate.classification == "VALID_TURN"
            and candidate.confirmation_status == "CONFIRMED"
        ]
        if not matches:
            return None
        forecast_date = forecast_candidate.window_start_date + (
            forecast_candidate.window_end_date - forecast_candidate.window_start_date
        ) / 2
        actual_candidate = min(
            matches,
            key=lambda candidate: abs(
                (
                    candidate.window_start_date
                    + (candidate.window_end_date - candidate.window_start_date) / 2
                    - forecast_date
                ).days
            ),
        )
        actual_date = actual_candidate.window_start_date + (
            actual_candidate.window_end_date - actual_candidate.window_start_date
        ) / 2
        start, end = sorted((forecast_date, actual_date))
        sessions = tuple(self.calendar.sessions(market, start, end))
        return max(0, len(sessions) - 1)
