"""Historical pattern matching used for explicitly labelled 13-week scenarios."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from math import sqrt
from statistics import fmean, pstdev
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.models import ForecastPoint, ForecastRun, Instrument, MarketPrice, ValuationRecord


class HistoricalSimilarityForecaster:
    """Compare only prior completed windows, then average their realised paths.

    This is a historical-statistics scenario, not a price promise or an AI
    prediction.  Candidate windows end before the current observed window and
    need their realised continuation already present in SQLite.
    """

    MODEL_NAME = "historical_similarity_v1"

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    @staticmethod
    def required_daily_rows(window_days: int, horizon_weeks: int, candidate_count: int) -> int:
        """Return the exact sample length needed for non-overlapping candidates."""
        return (2 * window_days) + (horizon_weeks * 5) + candidate_count - 1

    def availability(
        self,
        instrument_code: str,
        *,
        window_days: int = 60,
        horizon_weeks: int = 13,
        candidate_count: int = 5,
    ) -> dict[str, object]:
        """Describe whether a scenario can run, without creating a forecast run."""
        if window_days < 20 or horizon_weeks < 1 or candidate_count < 1:
            raise ValueError("window_days >= 20, horizon_weeks >= 1, and candidate_count >= 1 are required")
        with self.session_factory() as session:
            instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
            if instrument is None:
                raise ValueError(f"Unknown instrument code: {instrument_code}")
            rows = session.scalars(
                select(MarketPrice)
                .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
                .order_by(MarketPrice.trade_date)
            ).all()
        required_rows = self.required_daily_rows(window_days, horizon_weeks, candidate_count)
        sources = sorted({row.source for row in rows if row.source})
        return {
            "instrument_code": instrument_code,
            "daily_rows": len(rows),
            "required_rows": required_rows,
            "can_run": len(rows) >= required_rows,
            "data_cutoff": rows[-1].trade_date if rows else None,
            "sources": sources,
            "demo": bool(rows) and all((row.source or "").startswith("DEMO") for row in rows),
        }

    @staticmethod
    def _pattern(values: list[Decimal]) -> list[float]:
        first = float(values[0])
        return [float(value) / first - 1.0 for value in values]

    @staticmethod
    def _distance(left: list[float], right: list[float]) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left))

    def run(
        self,
        instrument_code: str,
        *,
        window_days: int = 60,
        horizon_weeks: int = 13,
        candidate_count: int = 5,
    ) -> dict[str, object]:
        if window_days < 20 or horizon_weeks < 1 or candidate_count < 1:
            raise ValueError("window_days >= 20, horizon_weeks >= 1, and candidate_count >= 1 are required")
        horizon_days = horizon_weeks * 5
        with self.session_factory() as session, session.begin():
            instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
            if instrument is None:
                raise ValueError(f"Unknown instrument code: {instrument_code}")
            rows = session.scalars(
                select(MarketPrice)
                .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
                .order_by(MarketPrice.trade_date)
            ).all()
            closes = [row.adjusted_close_price or row.close_price for row in rows]
            # Each candidate needs its own lookback window and known realised
            # horizon, and must end strictly before the current lookback.  The
            # last term supplies the requested number of distinct end points.
            minimum_rows = self.required_daily_rows(window_days, horizon_weeks, candidate_count)
            if len(rows) < minimum_rows:
                raise ValueError(
                    f"Historical scenario needs at least {minimum_rows} daily rows; local cache has {len(rows)}"
                )
            current = closes[-window_days:]
            target = self._pattern(current)
            candidates: list[tuple[float, int]] = []
            # Candidate end must leave a known continuation and must not overlap
            # the current analysis window.
            for end_index in range(window_days - 1, len(rows) - window_days - horizon_days):
                start_index = end_index - window_days + 1
                distance = self._distance(self._pattern(closes[start_index : end_index + 1]), target)
                candidates.append((distance, end_index))
            selected = sorted(candidates, key=lambda item: item[0])[:candidate_count]
            if not selected:
                raise ValueError("No non-overlapping historical comparison window is available")
            valuation_by_date = {
                row.valuation_date: row
                for row in session.scalars(
                    select(ValuationRecord)
                    .where(ValuationRecord.instrument_id == instrument.id)
                    .order_by(ValuationRecord.valuation_date)
                ).all()
            }

            def valuation_path(end_index: int) -> list[dict[str, object]]:
                """Return only recorded valuation observations for the realised path.

                A missing date intentionally remains a chart gap: projecting,
                forward-filling, or substituting ETF valuation would make a
                historical comparison look more complete than the local source.
                """
                path: list[dict[str, object]] = []
                for week in range(horizon_weeks + 1):
                    price = rows[end_index + (week * 5)]
                    valuation = valuation_by_date.get(price.trade_date)
                    path.append(
                        {
                            "week": week,
                            "date": price.trade_date.isoformat(),
                            "pe_ratio": format(valuation.pe_ratio, "f") if valuation and valuation.pe_ratio is not None else None,
                            "pb_ratio": format(valuation.pb_ratio, "f") if valuation and valuation.pb_ratio is not None else None,
                            "valuation_percentile": format(valuation.valuation_percentile, "f") if valuation and valuation.valuation_percentile is not None else None,
                            "source": valuation.raw_values.get("source") if valuation else None,
                        }
                    )
                return path

            matched_windows = [
                {
                    "end_date": rows[end_index].trade_date.isoformat(),
                    "distance": round(distance, 8),
                    "valuation_path": valuation_path(end_index),
                }
                for distance, end_index in selected
            ]
            base_price = closes[-1]
            dates = [rows[-1].trade_date + timedelta(days=7 * week) for week in range(1, horizon_weeks + 1)]
            points: list[dict[str, object]] = []
            for week, forecast_date in enumerate(dates, start=1):
                day_offset = min(week * 5, horizon_days)
                ratios = [
                    float(closes[end_index + day_offset]) / float(closes[end_index])
                    for _distance, end_index in selected
                ]
                mean_ratio = fmean(ratios)
                deviation = pstdev(ratios) if len(ratios) > 1 else 0.0
                points.append(
                    {
                        "date": forecast_date,
                        "predicted_price": Decimal(str(round(float(base_price) * mean_ratio, 8))),
                        "lower_bound": Decimal(str(round(float(base_price) * max(mean_ratio - deviation, 0), 8))),
                        "upper_bound": Decimal(str(round(float(base_price) * (mean_ratio + deviation), 8))),
                    }
                )
            run = ForecastRun(
                instrument_id=instrument.id,
                model_name=self.MODEL_NAME,
                horizon_days=horizon_days,
                status="completed",
                parameters={"window_days": window_days, "horizon_weeks": horizon_weeks, "candidate_count": candidate_count},
                summary={
                    "label": "历史相似情景推演（非投资建议）",
                    "data_cutoff": rows[-1].trade_date.isoformat(),
                    "matched_windows": matched_windows,
                },
            )
            session.add(run)
            session.flush()
            for point in points:
                session.add(ForecastPoint(forecast_run_id=run.id, forecast_date=point["date"], predicted_price=point["predicted_price"], lower_bound=point["lower_bound"], upper_bound=point["upper_bound"], confidence=Decimal("0")))
            return {
                "id": run.id,
                "instrument_code": instrument_code,
                "label": run.summary["label"],
                "data_cutoff": run.summary["data_cutoff"],
                "matched_windows": run.summary["matched_windows"],
                "points": points,
            }
