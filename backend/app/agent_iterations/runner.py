from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
import math
import random
from typing import Sequence

from backend.app.agent_iterations.model import (
    IterationFeedback,
    WeeklyObservation,
    evaluate_week,
    initial_model,
    model_from_dict,
    update_model,
)
from backend.app.agent_iterations.storage import (
    AgentIterationStorage,
    ArtifactConsistencyError,
)


class InsufficientMatureData(ValueError):
    """Raised when a maintenance period lacks the requested mature cutoffs."""


@dataclass(frozen=True)
class PriceBar:
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None
    source: str


def _label_maturity(cutoff: date, daily_dates: Sequence[date]) -> date | None:
    index = bisect_right(daily_dates, cutoff)
    future = daily_dates[index : index + 20]
    return future[-1] if len(future) == 20 else None


def select_progressive_cutoffs(
    weekly_dates: Sequence[date],
    daily_dates: Sequence[date],
    *,
    count: int,
    seed: int,
    warmup_weeks: int = 52,
) -> list[date]:
    """Choose deterministic, chronological cutoffs with mature prior labels."""
    if count <= 0:
        return []
    ordered_weekly = sorted(set(weekly_dates))
    ordered_daily = sorted(set(daily_dates))
    eligible = [
        item
        for item in ordered_weekly[warmup_weeks:]
        if _label_maturity(item, ordered_daily) is not None
    ]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} mature weekly cutoffs are available; {count} required")
    rng = random.Random(seed)
    step = (len(eligible) - 1) / max(1, count - 1)
    selected: list[date] = []
    previous_maturity: date | None = None
    cursor = 0
    for iteration in range(count):
        target = int(round(iteration * step)) + rng.choice((-1, 0, 1))
        target = max(cursor, min(len(eligible) - 1, target))
        while target < len(eligible):
            candidate = eligible[target]
            if previous_maturity is None or candidate > previous_maturity:
                break
            target += 1
        if target >= len(eligible):
            raise ValueError("Not enough non-overlapping mature cutoffs for progressive iteration")
        candidate = eligible[target]
        selected.append(candidate)
        maturity = _label_maturity(candidate, ordered_daily)
        if maturity is None:
            raise ValueError("Selected cutoff does not have a complete 20-trading-day label")
        previous_maturity = maturity
        cursor = target + 1
    return selected


def _ema(values: Sequence[Decimal], period: int) -> list[Decimal]:
    if not values:
        return []
    alpha = Decimal("2") / Decimal(period + 1)
    output = [values[0]]
    for value in values[1:]:
        output.append(value * alpha + output[-1] * (Decimal("1") - alpha))
    return output


def _mean(values: Sequence[Decimal]) -> Decimal | None:
    return sum(values, Decimal("0")) / Decimal(len(values)) if values else None


def _rsi(values: Sequence[Decimal], period: int = 6) -> Decimal | None:
    if len(values) <= period:
        return None
    changes = [values[index] - values[index - 1] for index in range(len(values) - period, len(values))]
    gains = sum((max(change, Decimal("0")) for change in changes), Decimal("0")) / Decimal(period)
    losses = sum((max(-change, Decimal("0")) for change in changes), Decimal("0")) / Decimal(period)
    if losses == 0:
        return Decimal("100")
    return Decimal("100") - Decimal("100") / (Decimal("1") + gains / losses)


def _volatility(values: Sequence[Decimal], period: int = 20) -> Decimal | None:
    if len(values) <= 2:
        return None
    subset = values[-(period + 1) :]
    returns = [float(subset[index] / subset[index - 1] - Decimal("1")) for index in range(1, len(subset))]
    if len(returns) < 2:
        return None
    average = sum(returns) / len(returns)
    variance = sum((value - average) ** 2 for value in returns) / (len(returns) - 1)
    return Decimal(str(math.sqrt(variance) * math.sqrt(52)))


def _nearest_valuation(cutoff: date, valuations: dict[date, Decimal]) -> Decimal | None:
    candidates = [item for item in valuations if item <= cutoff]
    return valuations[max(candidates)] if candidates else None


def _observation(
    weekly_bars: Sequence[PriceBar],
    index: int,
    valuations: dict[date, Decimal],
) -> WeeklyObservation:
    prefix = weekly_bars[: index + 1]
    closes = [item.close for item in prefix]
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    difs = [short - long for short, long in zip(ema12, ema26)]
    deas = _ema(difs, 9)
    current_dif = difs[-1]
    current_dea = deas[-1]
    previous_dif = difs[-2] if len(difs) > 1 else current_dif
    previous_dea = deas[-2] if len(deas) > 1 else current_dea
    crossed_up = previous_dif <= previous_dea and current_dif > current_dea
    crossed_down = previous_dif >= previous_dea and current_dif < current_dea
    ma20 = _mean(closes[-20:])
    ma60 = _mean(closes[-60:])
    valuation = _nearest_valuation(prefix[-1].trade_date, valuations)
    golden = 0
    black = 0
    if crossed_up:
        golden = 1
        golden += int(ma20 is not None and closes[-1] >= ma20)
        golden += int(valuation is not None and valuation <= Decimal("20"))
    if crossed_down:
        black = 1
        black += int(ma20 is not None and closes[-1] < ma20)
        black += int(valuation is not None and valuation >= Decimal("80"))
    volumes = [item.volume for item in prefix[-20:] if item.volume is not None]
    volume_average = _mean([value for value in volumes if value is not None])
    current_volume = prefix[-1].volume
    volume_ratio = (
        current_volume / volume_average
        if current_volume is not None and volume_average not in {None, Decimal("0")}
        else None
    )
    running_high = max(closes)
    return WeeklyObservation(
        trade_date=prefix[-1].trade_date,
        close=closes[-1],
        volume=current_volume,
        volume_ratio=volume_ratio,
        dif=current_dif,
        dea=current_dea,
        dif_slope=current_dif - previous_dif,
        dea_slope=current_dea - previous_dea,
        macd_histogram=Decimal("2") * (current_dif - current_dea),
        rsi=_rsi(closes),
        ma_20=ma20,
        ma_60=ma60,
        valuation_percentile=valuation,
        volatility=_volatility(closes),
        drawdown=closes[-1] / running_high - Decimal("1"),
        golden_strength=golden,
        black_strength=black,
    )


def _hash_payload(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trading_deviation(predicted_index: int, actual_index: int) -> int:
    return actual_index - predicted_index


def _is_completed_week(bar_date: date, as_of: date) -> bool:
    if bar_date > as_of:
        return False
    if bar_date.isocalendar()[:2] != as_of.isocalendar()[:2]:
        return True
    return as_of.weekday() >= 5


def _current_advice(
    *,
    model,
    current_position: int,
    weekly: Sequence[PriceBar],
    valuation_percentiles: dict[date, Decimal],
    as_of: date,
) -> dict[str, object]:
    complete = [
        item
        for item in weekly
        if _is_completed_week(item.trade_date, as_of)
    ]
    if not complete:
        raise ValueError("No completed weekly bar is available for current advice")
    cutoff = complete[-1].trade_date
    full_index = next(index for index, item in enumerate(weekly) if item.trade_date == cutoff)
    observation = _observation(weekly, full_index, valuation_percentiles)
    advice = evaluate_week(model, observation, current_position=current_position)
    return {
        "status": "pending",
        "cutoff_date": cutoff.isoformat(),
        "model_version": model.version,
        "recommendation": advice.recommendation,
        "state_label": advice.state_label,
        "score": str(advice.total_score),
        "confidence": advice.confidence,
        "probability_up_15d": advice.probability_up_15d,
        "current_position": advice.current_position,
        "target_position": advice.target_position,
        "batches": advice.batches,
        "conditional_batches": advice.conditional_batches,
        "fund_etf_ratio": advice.fund_etf_ratio,
        "fund_allocation": advice.fund_allocation,
        "etf_allocation": advice.etf_allocation,
        "component_scores": {key: str(value) for key, value in advice.component_scores.items()},
        "note": "当前建议由Agent使用最新完整周生成，未来结果尚未揭晓，不计入已完成迭代。",
    }


def _evaluate_iteration(
    *,
    instrument_code: str,
    iteration_number: int,
    model,
    current_position: int,
    cutoff: date,
    weekly: Sequence[PriceBar],
    daily: Sequence[PriceBar],
    valuation_percentiles: dict[date, Decimal],
    executed_by: str,
) -> tuple[dict[str, object], IterationFeedback, int]:
    weekly_index = {item.trade_date: index for index, item in enumerate(weekly)}
    daily_dates = [item.trade_date for item in daily]
    observation = _observation(weekly, weekly_index[cutoff], valuation_percentiles)
    advice = evaluate_week(model, observation, current_position=current_position)
    daily_start = bisect_right(daily_dates, cutoff)
    future = daily[daily_start : daily_start + 20]
    if len(future) != 20:
        raise ValueError(f"Cutoff {cutoff} has no complete 20-trading-day label")
    cutoff_daily = daily[daily_start - 1]
    day15 = future[14]
    actual_direction = 1 if day15.close >= cutoff_daily.close else -1
    predicted_direction = 1 if advice.probability_up_15d >= 50 else -1
    predicted_index = max(0, min(5, model.confirmation_offset))
    if advice.adjustment < 0:
        actual_index = max(range(len(future)), key=lambda index: future[index].close)
    elif advice.adjustment > 0:
        actual_index = min(range(len(future)), key=lambda index: future[index].close)
    elif predicted_direction > 0:
        actual_index = min(range(len(future)), key=lambda index: future[index].close)
    else:
        actual_index = max(range(len(future)), key=lambda index: future[index].close)
    deviation = _trading_deviation(predicted_index, actual_index)
    feedback = IterationFeedback(
        actual_direction=actual_direction,
        predicted_direction=predicted_direction,
        deviation_days=deviation,
        probability=Decimal(advice.probability_up_15d) / Decimal("100"),
        component_scores=advice.component_scores,
    )
    source_rows = weekly[: weekly_index[cutoff] + 1]
    source_payload = [
        {
            "date": item.trade_date.isoformat(),
            "close": str(item.close),
            "volume": str(item.volume) if item.volume is not None else None,
            "source": item.source,
        }
        for item in source_rows
    ]
    record: dict[str, object] = {
        "iteration": iteration_number,
        "instrument_code": instrument_code,
        "cutoff_date": cutoff.isoformat(),
        "source_data_max_date": source_rows[-1].trade_date.isoformat(),
        "future_label_max_date": future[-1].trade_date.isoformat(),
        "model": model.as_dict(),
        "recommendation": advice.recommendation,
        "state_label": advice.state_label,
        "score": str(advice.total_score),
        "confidence": advice.confidence,
        "probability_up_15d": advice.probability_up_15d,
        "current_position": advice.current_position,
        "target_position": advice.target_position,
        "batches": advice.batches,
        "conditional_batches": advice.conditional_batches,
        "fund_etf_ratio": advice.fund_etf_ratio,
        "fund_allocation": advice.fund_allocation,
        "etf_allocation": advice.etf_allocation,
        "predicted_action_date": future[predicted_index].trade_date.isoformat(),
        "actual_turn_date": future[actual_index].trade_date.isoformat(),
        "deviation_days": deviation,
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "direction_correct": feedback.direction_correct,
        "component_scores": {key: str(value) for key, value in advice.component_scores.items()},
        "source_data_hash": _hash_payload(source_payload),
        "executed_by": executed_by,
        "execution_mode": "agent_offline",
    }
    return record, feedback, advice.target_position


def run_progressive_baseline(
    *,
    instrument_code: str,
    weekly_bars: Sequence[PriceBar],
    daily_bars: Sequence[PriceBar],
    valuation_percentiles: dict[date, Decimal],
    storage: AgentIterationStorage,
    count: int,
    seed: int,
    executed_by: str,
) -> dict[str, object]:
    existing_manifest, existing_records = storage.read_consistent_snapshot(instrument_code)
    if existing_manifest is not None:
        if len(existing_records) >= count:
            return {
                **existing_manifest,
                "operation_status": "unchanged",
                "message": "Baseline artifacts are already complete and were not rewritten.",
            }
        raise ArtifactConsistencyError(
            f"Existing baseline artifacts are incomplete "
            f"({len(existing_records)}/{count}); refusing to overwrite non-empty history"
        )
    weekly = sorted(weekly_bars, key=lambda item: item.trade_date)
    daily = sorted(daily_bars, key=lambda item: item.trade_date)
    cutoffs = select_progressive_cutoffs(
        [item.trade_date for item in weekly],
        [item.trade_date for item in daily],
        count=count,
        seed=seed,
    )
    model = initial_model(instrument_code)
    current_position = 50
    records: list[dict[str, object]] = []
    for iteration_number, cutoff in enumerate(cutoffs, start=1):
        record, feedback, current_position = _evaluate_iteration(
            instrument_code=instrument_code,
            iteration_number=iteration_number,
            model=model,
            current_position=current_position,
            cutoff=cutoff,
            weekly=weekly,
            daily=daily,
            valuation_percentiles=valuation_percentiles,
            executed_by=executed_by,
        )
        records.append(record)
        if iteration_number < count:
            model = update_model(model, feedback)
    deviations = [abs(int(item["deviation_days"])) for item in records]
    direction_hits = sum(bool(item["direction_correct"]) for item in records)
    sorted_deviations = sorted(deviations)
    median = (
        sorted_deviations[len(sorted_deviations) // 2]
        if sorted_deviations
        else 0
    )
    manifest: dict[str, object] = {
        "instrument_code": instrument_code,
        "completed_iterations": len(records),
        "latest_model_version": model.version,
        "latest_model": model.as_dict(),
        "baseline_seed": seed,
        "baseline_date": "2026-07-30",
        "execution_mode": "agent_offline",
        "executed_by": executed_by,
        "direction_accuracy": round(direction_hits / len(records), 4) if records else 0,
        "mean_absolute_deviation": round(sum(deviations) / len(deviations), 2) if deviations else 0,
        "median_absolute_deviation": median,
        "source_data_max_date": records[-1]["source_data_max_date"] if records else None,
        "monthly_increment": 2,
        "maintenance_ledger": {},
        "operation_status": "updated",
        "current_advice": _current_advice(
            model=model,
            current_position=current_position,
            weekly=weekly,
            valuation_percentiles=valuation_percentiles,
            as_of=date(2026, 7, 30),
        ),
    }
    storage.write(instrument_code, manifest=manifest, iterations=records)
    return manifest


def append_due_iterations(
    *,
    instrument_code: str,
    weekly_bars: Sequence[PriceBar],
    daily_bars: Sequence[PriceBar],
    valuation_percentiles: dict[date, Decimal],
    storage: AgentIterationStorage,
    count: int,
    executed_by: str,
    as_of: date | None = None,
) -> dict[str, object]:
    """Append mature monthly iterations without rewriting baseline artifacts."""
    manifest, records = storage.read_consistent_snapshot(instrument_code)
    if manifest is None or not records:
        raise ValueError("Progressive baseline is missing")
    effective_as_of = as_of or date.today()
    period = effective_as_of.strftime("%Y-%m")
    ledger = {
        str(key): dict(value)
        for key, value in dict(manifest.get("maintenance_ledger") or {}).items()
    }
    period_entry = dict(ledger.get(period) or {"count": 0, "runs": []})
    used = int(period_entry.get("count", 0))
    remaining = max(0, count - used)
    if remaining == 0:
        return {
            **manifest,
            "operation_status": "unchanged",
            "message": f"Monthly maintenance quota for {period} is already exhausted.",
        }
    weekly = sorted(weekly_bars, key=lambda item: item.trade_date)
    daily = sorted(daily_bars, key=lambda item: item.trade_date)
    daily_dates = [
        item.trade_date
        for item in daily
        if item.trade_date <= effective_as_of
    ]
    last_label = date.fromisoformat(str(records[-1]["future_label_max_date"]))
    candidates = [
        item.trade_date
        for item in weekly
        if (
            item.trade_date > last_label
            and item.trade_date <= effective_as_of
            and _label_maturity(item.trade_date, daily_dates) is not None
        )
    ]
    selected: list[date] = []
    previous_maturity = last_label
    for candidate in candidates:
        if candidate <= previous_maturity:
            continue
        selected.append(candidate)
        maturity = _label_maturity(candidate, daily_dates)
        if maturity is None:
            break
        previous_maturity = maturity
        if len(selected) == remaining:
            break
    if len(selected) < remaining:
        raise InsufficientMatureData(
            f"Only {len(selected)} mature post-baseline cutoffs are available as of "
            f"{effective_as_of.isoformat()}; {remaining} required"
        )
    model = model_from_dict(dict(manifest["latest_model"]))  # type: ignore[arg-type]
    last = records[-1]
    model = update_model(
        model,
        IterationFeedback(
            actual_direction=int(last["actual_direction"]),
            predicted_direction=int(last["predicted_direction"]),
            deviation_days=int(last["deviation_days"]),
            probability=Decimal(str(last["probability_up_15d"])) / Decimal("100"),
            component_scores={
                key: Decimal(str(value))
                for key, value in dict(last["component_scores"]).items()  # type: ignore[arg-type]
            },
        ),
    )
    current_position = int(last["target_position"])
    for offset, cutoff in enumerate(selected):
        number = len(records) + 1
        record, feedback, current_position = _evaluate_iteration(
            instrument_code=instrument_code,
            iteration_number=number,
            model=model,
            current_position=current_position,
            cutoff=cutoff,
            weekly=weekly,
            daily=daily,
            valuation_percentiles=valuation_percentiles,
            executed_by=executed_by,
        )
        records.append(record)
        if offset < len(selected) - 1:
            model = update_model(model, feedback)
    deviations = [abs(int(item["deviation_days"])) for item in records]
    hits = sum(bool(item["direction_correct"]) for item in records)
    runs = list(period_entry.get("runs") or [])
    runs.append(
        {
            "as_of": effective_as_of.isoformat(),
            "executed_by": executed_by,
            "added_iterations": len(selected),
            "status": "updated",
        }
    )
    ledger[period] = {
        "count": used + len(selected),
        "runs": runs,
    }
    updated = {
        **manifest,
        "completed_iterations": len(records),
        "latest_model_version": model.version,
        "latest_model": model.as_dict(),
        "executed_by": executed_by,
        "direction_accuracy": round(hits / len(records), 4),
        "mean_absolute_deviation": round(sum(deviations) / len(deviations), 2),
        "median_absolute_deviation": sorted(deviations)[len(deviations) // 2],
        "source_data_max_date": records[-1]["source_data_max_date"],
        "maintenance_ledger": ledger,
        "operation_status": "updated",
        "current_advice": _current_advice(
            model=model,
            current_position=current_position,
            weekly=weekly,
            valuation_percentiles=valuation_percentiles,
            as_of=effective_as_of,
        ),
    }
    storage.write(instrument_code, manifest=updated, iterations=records)
    return updated
