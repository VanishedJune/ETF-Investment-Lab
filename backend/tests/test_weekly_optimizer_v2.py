from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from datetime import date, timedelta
from decimal import Decimal

import pytest

from backend.app import weekly_analysis
from backend.app.weekly_analysis import optimizer as optimizer_module
from backend.app.weekly_analysis.optimizer import (
    LOSS_WEIGHTS,
    ConflictingFeedback,
    ConflictingWeekInput,
    CrossMarketError,
    DailyClose,
    FeedbackRegressionError,
    FutureLeakageError,
    IterationLabel,
    IterationPrediction,
    RollingMetricWindows,
    WindowMetrics,
    actual_turn_date,
    rolling_metrics,
    run_optimizer_step,
    seed_model,
    update_labels,
)


def _weekdays(start: date, count: int) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return tuple(values)


def _iso_week_key(day: date) -> str:
    iso_year, iso_week, _ = day.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _daily(
    count: int = 70,
    *,
    start: date = date(2026, 4, 27),
    instrument_code: str = "399006",
    baseline_close: Decimal = Decimal("99"),
) -> tuple[DailyClose, ...]:
    dates = _weekdays(start, count)
    return (DailyClose(date(2026, 4, 24), baseline_close, instrument_code),) + tuple(
        DailyClose(day, Decimal(100 + index), instrument_code)
        for index, day in enumerate(dates)
    )


def _prediction(
    *,
    iteration_id: str = "I0001",
    instrument_code: str = "399006",
    week_key: str | None = None,
    cutoff_date: date = date(2026, 4, 24),
    probability: Decimal = Decimal("0.8"),
    direction: str = "up",
    operation_side: str | None = None,
    action_date: date = date(2026, 4, 27),
    trade_count: int = 2,
) -> IterationPrediction:
    return IterationPrediction(
        iteration_id=iteration_id,
        instrument_code=instrument_code,
        week_key=(
            week_key
            if week_key is not None
            else _iso_week_key(cutoff_date)
        ),
        cutoff_date=cutoff_date,
        predicted_direction=direction,
        operation_side=(
            operation_side
            if operation_side is not None
            else {"up": "buy", "down": "sell", "neutral": "hold"}[direction]
        ),
        probability=probability,
        predicted_action_date=action_date,
        target_position=Decimal("0.75"),
        trade_count=trade_count,
    )


def _expected_trade_dates(daily: tuple[DailyClose, ...]) -> tuple[date, ...]:
    return tuple(
        row.trade_date
        for row in daily
        if row.trade_date > date(2026, 4, 24)
    )


def _market_calendar(
    instrument_code: str,
    dates: tuple[date, ...],
):
    calendar_type = getattr(optimizer_module, "TradingCalendar", None)
    if calendar_type is None:
        return dates
    return calendar_type(
        instrument_code=instrument_code,
        expected_trade_dates=dates,
        coverage_start=(
            dates[0] - timedelta(days=dates[0].weekday())
            if dates
            else None
        ),
        coverage_end=(
            dates[-1] + timedelta(days=6 - dates[-1].weekday())
            if dates
            else None
        ),
    )


def _update_labels(
    iterations: tuple[IterationPrediction, ...],
    daily: tuple[DailyClose, ...],
    *,
    as_of: date,
    expected_trade_dates: tuple[date, ...] | None = None,
) -> tuple[IterationLabel, ...]:
    calendar = (
        expected_trade_dates
        if hasattr(expected_trade_dates, "instrument_code")
        else _market_calendar(
            iterations[0].instrument_code,
            (
                expected_trade_dates
                if expected_trade_dates is not None
                else _expected_trade_dates(daily)
            ),
        )
    )
    return update_labels(
        iterations,
        daily,
        as_of=as_of,
        expected_trade_dates=calendar,
    )


def _mature_labels(
    count: int,
    *,
    instrument_code: str = "399006",
) -> tuple:
    daily = _daily(instrument_code=instrument_code)
    base = _update_labels(
        (_prediction(instrument_code=instrument_code),),
        daily,
        as_of=date(2026, 7, 24),
    )[0]
    assert base.status == "mature_13w"
    return tuple(
        replace(
            base,
            iteration_id=f"I{index + 1:04d}",
            direction_hit=(index % 2 == 0),
        )
        for index in range(count)
    )


def _metrics(
    window: str,
    loss: str,
    *,
    hit: str = "0.60",
    calibration: str = "0.20",
    overtrade: str = "0.20",
    count: int = 20,
    instrument_code: str = "399006",
) -> WindowMetrics:
    return WindowMetrics(
        instrument_code=instrument_code,
        window=window,
        sample_count=count,
        total_loss=Decimal(loss),
        direction_hit_rate=Decimal(hit),
        calibration_loss=Decimal(calibration),
        overtrade_penalty=Decimal(overtrade),
    )


def test_market_specific_seed_models_are_auditable_and_deeply_immutable() -> None:
    growth = seed_model("399006")
    ndx = seed_model("NDX")

    assert growth.instrument_code == "399006"
    assert growth.current_model_version == "M0001"
    assert dict(growth.current_model_weights) == {
        "valuation": Decimal("30"),
        "dea_trend": Decimal("20"),
        "dif_trend": Decimal("20"),
        "reversal": Decimal("15"),
        "price_momentum": Decimal("5"),
        "risk_regime": Decimal("10"),
    }
    assert dict(ndx.current_model_weights) == {
        "valuation": Decimal("25"),
        "dea_trend": Decimal("22"),
        "dif_trend": Decimal("22"),
        "reversal": Decimal("12"),
        "price_momentum": Decimal("9"),
        "risk_regime": Decimal("10"),
    }
    assert sum(growth.current_model_weights.values(), Decimal()) == Decimal("100")
    assert sum(ndx.current_model_weights.values(), Decimal()) == Decimal("100")
    with pytest.raises(TypeError):
        growth.current_model_weights["valuation"] = Decimal("0")
    with pytest.raises(FrozenInstanceError):
        growth.instrument_code = "NDX"
    with pytest.raises(ValueError, match="Unsupported instrument"):
        seed_model("SPX")
    assert sum(LOSS_WEIGHTS.values(), Decimal()) == Decimal("1")
    assert dict(LOSS_WEIGHTS) == {
        "turning_deviation": Decimal("0.45"),
        "direction_calibration": Decimal("0.30"),
        "adverse_excursion": Decimal("0.15"),
        "overtrading": Decimal("0.10"),
    }


def test_label_progression_uses_only_rows_visible_as_of_and_never_fakes_maturity() -> None:
    prediction = _prediction()
    daily = _daily()

    pending = _update_labels((prediction,), daily, as_of=date(2026, 4, 30))[0]
    partial = _update_labels((prediction,), daily, as_of=date(2026, 5, 1))[0]
    mature_4w = _update_labels(
        (prediction,), daily, as_of=daily[20].trade_date
    )[0]
    mature_13w = _update_labels(
        (prediction,), daily, as_of=daily[65].trade_date
    )[0]

    assert pending.status == "pending"
    assert pending.visible_row_count == 4
    assert pending.partial_weight == Decimal("0")
    assert pending.observed_through <= date(2026, 4, 30)
    assert partial.status == "partial_1w"
    assert partial.visible_row_count == 5
    assert Decimal("0") < partial.partial_weight < Decimal("1")
    assert mature_4w.status == "mature_4w"
    assert mature_4w.partial_weight < Decimal("1")
    assert mature_4w.total_loss is None
    assert set(mature_4w.loss_components) == {"direction_calibration"}
    assert mature_13w.status == "mature_13w"
    assert mature_13w.visible_row_count == 65
    assert mature_13w.partial_weight == Decimal("1")
    assert mature_13w.total_loss is not None
    assert mature_13w.observed_through == daily[65].trade_date

    no_future_rows = _update_labels(
        (prediction,), daily, as_of=date(2026, 4, 23)
    )[0]
    assert no_future_rows.status == "pending"
    assert no_future_rows.visible_row_count == 0
    assert no_future_rows.observed_through <= date(2026, 4, 23)


def test_thirteen_complete_trading_weeks_mature_despite_holiday_row_count() -> None:
    daily = _daily(65)
    holiday_adjusted = tuple(
        row for index, row in enumerate(daily) if index != 30
    )

    label = _update_labels(
        (_prediction(),),
        holiday_adjusted,
        as_of=daily[65].trade_date,
        expected_trade_dates=_market_calendar(
            "399006", _expected_trade_dates(holiday_adjusted)
        ),
    )[0]

    assert label.status == "mature_13w"
    assert label.visible_row_count == 64
    assert label.observed_through == daily[65].trade_date
    assert label.total_loss is not None


def test_turning_points_use_earliest_tie_and_signed_trading_session_deviation() -> None:
    dates = _weekdays(date(2026, 4, 27), 65)
    closes = [Decimal("100")] * 65
    closes[1] = Decimal("120")
    closes[2] = Decimal("120")
    closes[6] = Decimal("80")
    closes[7] = Decimal("80")
    daily = tuple(
        DailyClose(day, close, "399006")
        for day, close in zip(dates, closes, strict=True)
    )

    assert actual_turn_date("sell", daily) == dates[1]
    assert actual_turn_date("buy", daily) == dates[6]
    label = _update_labels(
        (
            _prediction(
                direction="down",
                action_date=dates[4],
                probability=Decimal("0.7"),
            ),
        ),
        (DailyClose(date(2026, 4, 24), Decimal("100"), "399006"),) + daily,
        as_of=dates[-1],
    )[0]

    assert label.actual_turn_date == dates[1]
    assert label.turning_deviation_sessions == -3
    assert (
        label.turning_deviation_formula
        == "actual_trading_index-predicted_trading_index"
    )
    assert label.actual_direction == "neutral"
    assert label.direction_hit is False


def test_daily_series_rejects_invalid_duplicate_and_non_increasing_rows() -> None:
    valid = _daily(6)
    duplicate = valid[:2] + (valid[1],) + valid[2:]
    reversed_rows = tuple(reversed(valid))

    with pytest.raises(ValueError, match="duplicate"):
        _update_labels((_prediction(),), duplicate, as_of=valid[-1].trade_date)
    with pytest.raises(ValueError, match="strictly increasing"):
        _update_labels(
            (_prediction(),),
            reversed_rows,
            as_of=valid[-1].trade_date,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        DailyClose(date(2026, 1, 1), Decimal("NaN"), "399006")
    with pytest.raises(ValueError, match="finite and positive"):
        DailyClose(date(2026, 1, 1), Decimal("0"), "399006")


def test_mature_loss_has_auditable_components_and_weighted_total() -> None:
    daily = _daily()
    label = _update_labels(
        (_prediction(direction="down", probability=Decimal("0.9"), trade_count=7),),
        daily,
        as_of=daily[65].trade_date,
    )[0]

    assert set(label.loss_components) == set(LOSS_WEIGHTS)
    expected = sum(
        label.loss_components[name] * weight
        for name, weight in LOSS_WEIGHTS.items()
    )
    assert label.total_loss == expected
    assert label.calibration_formula == "brier:(p_predicted-I[correct])^2"
    assert label.direction_hit is False
    assert Decimal("0") <= label.loss_components["adverse_excursion"] <= Decimal(
        "1"
    )
    assert label.loss_components["overtrading"] == Decimal("0.7")


def test_hold_operation_has_no_fabricated_turn_and_preserves_weighted_loss() -> None:
    assert "operation_side" in {
        field.name for field in fields(IterationPrediction)
    }
    prediction = replace(
        _prediction(direction="neutral", trade_count=0),
        operation_side="hold",
    )
    label = _update_labels(
        (prediction,),
        _daily(),
        as_of=date(2026, 7, 24),
    )[0]

    assert label.operation_side == "hold"
    assert label.actual_turn_date is None
    assert label.turning_deviation_sessions is None
    assert label.loss_components["turning_deviation"] == Decimal("0")
    assert (
        label.turning_deviation_formula
        == "hold:no_turn;turning_deviation=0"
    )
    assert label.total_loss == sum(
        label.loss_components[name] * weight
        for name, weight in LOSS_WEIGHTS.items()
    )
    assert label.to_dict()["operation_side"] == "hold"

    buy = replace(prediction, operation_side="buy")
    hold_result = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        prediction=prediction,
    )
    buy_result = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        prediction=buy,
    )
    assert hold_result.audit.input_hash != buy_result.audit.input_hash


def test_mature_label_rejects_total_loss_not_equal_to_weighted_components() -> None:
    mature = _mature_labels(1)[0]

    with pytest.raises(ValueError, match="weighted"):
        replace(
            mature,
            total_loss=mature.total_loss + Decimal("0.0001"),
        )


def test_rolling_metrics_exclude_pending_and_expose_expanding_20_and_52_windows() -> None:
    pending = _update_labels(
        (_prediction(iteration_id="PENDING"),),
        _daily(4),
        as_of=date(2026, 4, 30),
    )[0]
    labels_19 = _mature_labels(19) + (pending,)
    labels_20 = _mature_labels(20) + (pending,)
    labels_52 = _mature_labels(52) + (pending,)

    expanding = rolling_metrics(labels_19)
    twenty = rolling_metrics(labels_20)
    fifty_two = rolling_metrics(labels_52)

    assert set(expanding.windows) == {"expanding"}
    assert expanding.windows["expanding"].sample_count == 19
    assert set(twenty.windows) == {"20"}
    assert twenty.windows["20"].sample_count == 20
    assert set(fifty_two.windows) == {"20", "52"}
    assert fifty_two.windows["20"].sample_count == 20
    assert fifty_two.windows["52"].sample_count == 52
    assert rolling_metrics(labels_19, window=20).sample_count == 19
    assert rolling_metrics(labels_52, window=20).sample_count == 20
    assert rolling_metrics(labels_52, window=52).sample_count == 52


def test_rolling_metrics_reject_cross_market_feedback() -> None:
    with pytest.raises(CrossMarketError):
        rolling_metrics(
            _mature_labels(1, instrument_code="399006")
            + _mature_labels(1, instrument_code="NDX")
        )


def test_new_weeks_increment_i_and_w_and_preserve_rejection_audit() -> None:
    first_feedback = _mature_labels(1)[0]
    second_feedback = replace(
        first_feedback,
        iteration_id="I0002",
        week_key="2026-W18",
        cutoff_date=date(2026, 5, 1),
        actual_turn_date=date(2026, 5, 4),
    )
    seed = seed_model("399006")

    first = run_optimizer_step(
        seed,
        feedback=(first_feedback,),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )
    repeated = run_optimizer_step(
        first.work_state,
        feedback=(first_feedback,),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )
    second = run_optimizer_step(
        first.work_state,
        feedback=(second_feedback,),
        week_key="2026-W32",
        cutoff_date=date(2026, 8, 6),
        source_data_max_date=date(2026, 8, 6),
    )

    assert (first.iteration_id, first.work_version) == ("I0001", "W0001")
    assert first.parent_work_version == seed.work_version
    assert first.accepted is False
    assert first.current_model_version == "M0001"
    assert first.current_model_weights == seed.current_model_weights
    assert repeated.reused is True
    assert repeated.work_version == first.work_version
    assert repeated.current_model_version == first.current_model_version
    assert second.parent_work_version == first.work_version
    assert (second.iteration_id, second.work_version) == ("I0002", "W0002")
    assert second.optimizer_memory["feedback_count"] == 2
    assert second.optimizer_memory["rejected_candidate_count"] == 2
    assert second.audit.feedback_count == 2
    assert second.audit.rejected_candidates


def test_candidate_is_conservative_sum_preserving_and_reads_prior_memory() -> None:
    feedback = _mature_labels(20)
    metrics = {"20": _metrics("20", "0.5")}
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=feedback,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=metrics,
        candidate_metrics=metrics,
    )
    second = run_optimizer_step(
        first.work_state,
        feedback=feedback,
        week_key="2026-W32",
        cutoff_date=date(2026, 8, 6),
        source_data_max_date=date(2026, 8, 6),
        current_metrics=metrics,
        candidate_metrics=metrics,
    )

    assert sum(first.candidate_weights.values(), Decimal()) == Decimal("100")
    assert all(value >= 0 for value in first.candidate_weights.values())
    assert all(
        abs(first.candidate_weights[name] - first.base_weights[name])
        <= Decimal("2")
        for name in first.base_weights
    )
    assert sum(
        first.candidate_weights[name] != first.base_weights[name]
        for name in first.base_weights
    ) == 2
    assert second.candidate_weights != first.candidate_weights


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "feedback": (),
        },
        {
            "feedback": _mature_labels(20),
            "current_metrics": {"20": _metrics("20", "0.5")},
            "candidate_metrics": {"20": _metrics("20", "0.495")},
            "data_integrity": False,
        },
    ],
)
def test_non_quality_rejections_do_not_consume_candidates(arguments) -> None:
    result = run_optimizer_step(
        seed_model("399006"),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        **arguments,
    )

    assert result.optimizer_memory["rejected_candidate_count"] == 0
    assert result.audit.rejected_candidates == ()


def test_progressive_cursor_explores_changed_validation_context() -> None:
    feedback = _mature_labels(20)
    metrics = {"20": _metrics("20", "0.5")}
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=feedback,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=metrics,
        candidate_metrics=metrics,
    )
    same_context = run_optimizer_step(
        first.work_state,
        feedback=feedback,
        week_key="2026-W32",
        cutoff_date=date(2026, 8, 6),
        source_data_max_date=date(2026, 8, 6),
        current_metrics=metrics,
        candidate_metrics=metrics,
    )
    changed_context = run_optimizer_step(
        first.work_state,
        feedback=_mature_labels(21),
        week_key="2026-W32",
        cutoff_date=date(2026, 8, 6),
        source_data_max_date=date(2026, 8, 6),
        current_metrics=metrics,
        candidate_metrics=metrics,
    )

    assert same_context.candidate_weights != first.candidate_weights
    assert changed_context.candidate_weights != first.candidate_weights
    assert first.audit.rejected_candidates[0]["validation_context_hash"]
    assert (
        changed_context.audit.rejected_candidates[-1]["validation_context_hash"]
        != first.audit.rejected_candidates[0]["validation_context_hash"]
    )


def test_changed_candidate_gate_metrics_keep_progressive_cursor() -> None:
    feedback = _mature_labels(20)
    current = {"20": _metrics("20", "0.5")}
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=feedback,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=current,
        candidate_metrics=current,
    )
    changed_gate_input = run_optimizer_step(
        first.work_state,
        feedback=feedback,
        week_key="2026-W32",
        cutoff_date=date(2026, 8, 6),
        source_data_max_date=date(2026, 8, 6),
        current_metrics=current,
        candidate_metrics={"20": _metrics("20", "0.49")},
    )

    assert changed_gate_input.candidate_weights != first.candidate_weights
    assert changed_gate_input.accepted is True


@pytest.mark.parametrize(
    ("mature_count", "current", "candidate", "expected"),
    [
        (
            19,
            {"expanding": _metrics("expanding", "0.5", count=19)},
            {"expanding": _metrics("expanding", "0.495", count=19)},
            True,
        ),
        (
            20,
            {"20": _metrics("20", "0.5")},
            {"20": _metrics("20", "0.4951")},
            False,
        ),
        (
            20,
            {"20": _metrics("20", "0.5")},
            {"20": _metrics("20", "0.495")},
            True,
        ),
        (
            52,
            {
                "20": _metrics("20", "0.5"),
                "52": _metrics("52", "0.5", count=52),
            },
            {
                "20": _metrics("20", "0.494"),
                "52": _metrics("52", "0.502", count=52),
            },
            True,
        ),
        (
            52,
            {
                "20": _metrics("20", "0.5"),
                "52": _metrics("52", "0.5", count=52),
            },
            {
                "20": _metrics("20", "0.494"),
                "52": _metrics("52", "0.503", count=52),
            },
            False,
        ),
    ],
)
def test_acceptance_loss_gate_uses_expanding_20_and_52_windows(
    mature_count: int,
    current: dict[str, WindowMetrics],
    candidate: dict[str, WindowMetrics],
    expected: bool,
) -> None:
    result = run_optimizer_step(
        seed_model("399006"),
        feedback=_mature_labels(mature_count),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=current,
        candidate_metrics=candidate,
    )

    assert result.accepted is expected
    assert result.current_model_version == ("M0002" if expected else "M0001")


def test_direction_hit_drop_is_at_most_two_percentage_points_and_overtrade_five_percent() -> None:
    current = {
        "20": _metrics("20", "0.5", hit="0.60", overtrade="0.20"),
        "52": _metrics(
            "52", "0.5", hit="0.60", overtrade="0.20", count=52
        ),
    }
    boundary = {
        "20": _metrics("20", "0.494", hit="0.58", overtrade="0.21"),
        "52": _metrics(
            "52", "0.502", hit="0.58", overtrade="0.21", count=52
        ),
    }
    too_low_hit = {
        "20": replace(boundary["20"], direction_hit_rate=Decimal("0.579")),
        "52": replace(boundary["52"], direction_hit_rate=Decimal("0.579")),
    }
    too_much_trading = {
        "20": replace(boundary["20"], overtrade_penalty=Decimal("0.2101")),
        "52": replace(boundary["52"], overtrade_penalty=Decimal("0.2101")),
    }

    accepted = run_optimizer_step(
        seed_model("399006"),
        feedback=_mature_labels(52),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=current,
        candidate_metrics=boundary,
    )
    hit_rejected = run_optimizer_step(
        seed_model("399006"),
        feedback=_mature_labels(52),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=current,
        candidate_metrics=too_low_hit,
    )
    trading_rejected = run_optimizer_step(
        seed_model("399006"),
        feedback=_mature_labels(52),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=current,
        candidate_metrics=too_much_trading,
    )

    assert accepted.accepted is True
    assert hit_rejected.accepted is False
    assert "direction_hit_drop_exceeds_2_points" in hit_rejected.reasons
    assert trading_rejected.accepted is False
    assert "overtrade_increase_exceeds_5_percent" in trading_rejected.reasons


def test_rejected_step_keeps_model_but_feedback_and_audit_reach_next_work() -> None:
    feedback = _mature_labels(20)
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=feedback[:1],
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )
    second = run_optimizer_step(
        first.work_state,
        feedback=feedback[1:2],
        week_key="2026-W32",
        cutoff_date=date(2026, 8, 6),
        source_data_max_date=date(2026, 8, 6),
    )

    assert first.current_model_version == second.current_model_version == "M0001"
    assert second.current_model_weights == first.current_model_weights
    assert second.optimizer_memory["feedback_count"] == 2
    assert second.audit.parent_work_version == first.work_version
    assert second.audit.rejected_candidates
    assert second.audit.candidate_acceptance_rate == Decimal("0")
    assert (
        second.audit.algorithm_version
        == "weekly-optimizer-v2.1-progressive-cursor"
    )
    assert second.audit.loss_version == "weekly-loss-v2-brier"
    assert second.audit.loss_weights == LOSS_WEIGHTS


def test_cross_market_inputs_and_future_source_dates_are_blocked() -> None:
    ndx_feedback = _mature_labels(1, instrument_code="NDX")
    with pytest.raises(CrossMarketError):
        run_optimizer_step(
            seed_model("399006"),
            feedback=ndx_feedback,
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )
    with pytest.raises(CrossMarketError):
        update_labels(
            (_prediction(), _prediction(iteration_id="I0002", instrument_code="NDX")),
            _daily(),
            as_of=date(2026, 7, 30),
            expected_trade_dates=_market_calendar(
                "399006", _expected_trade_dates(_daily())
            ),
        )
    with pytest.raises(FutureLeakageError) as captured:
        run_optimizer_step(
            seed_model("NDX"),
            feedback=(),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 31),
        )
    assert captured.value.severity == "blocking"


def test_feedback_observed_after_optimizer_cutoff_is_future_leakage() -> None:
    future_feedback = replace(
        _mature_labels(1)[0],
        observed_through=date(2026, 7, 31),
    )

    with pytest.raises(FutureLeakageError) as captured:
        run_optimizer_step(
            seed_model("399006"),
            feedback=(future_feedback,),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )

    assert captured.value.severity == "blocking"


@pytest.mark.parametrize(
    "actual_turn_date",
    (
        date(2026, 4, 24),
        date(2026, 7, 25),
    ),
)
def test_mature_trade_turn_date_must_be_inside_its_visible_window(
    actual_turn_date: date,
) -> None:
    mature = _mature_labels(1)[0]

    with pytest.raises(
        ValueError,
        match="actual_turn_date.*cutoff_date.*observed_through",
    ):
        replace(mature, actual_turn_date=actual_turn_date)


def test_optimizer_defensively_blocks_future_actual_turn_date() -> None:
    future_feedback = _mature_labels(1)[0]
    object.__setattr__(
        future_feedback,
        "actual_turn_date",
        date(2027, 1, 1),
    )

    with pytest.raises(FutureLeakageError) as captured:
        run_optimizer_step(
            seed_model("399006"),
            feedback=(future_feedback,),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )

    assert captured.value.source_data_max_date == date(2027, 1, 1)
    assert captured.value.severity == "blocking"


def test_pending_feedback_prediction_cutoff_after_optimizer_cutoff_is_future_leakage() -> None:
    pending = _update_labels(
        (_prediction(),),
        _daily(4),
        as_of=date(2026, 4, 30),
    )[0]
    future_pending = replace(
        pending,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 31),
    )

    with pytest.raises(FutureLeakageError):
        run_optimizer_step(
            seed_model("399006"),
            feedback=(future_pending,),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )


def test_prediction_cutoff_after_optimizer_cutoff_is_future_leakage() -> None:
    with pytest.raises(FutureLeakageError):
        run_optimizer_step(
            seed_model("399006"),
            feedback=(),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            prediction=_prediction(cutoff_date=date(2026, 7, 31)),
        )
    with pytest.raises(FutureLeakageError):
        run_optimizer_step(
            seed_model("399006"),
            feedback=(),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            prediction={
                "instrument_code": "399006",
                "cutoff_date": date(2026, 7, 31),
            },
        )


def test_model_state_cannot_be_relabelled_as_the_other_market() -> None:
    relabelled = replace(seed_model("399006"), instrument_code="NDX")

    with pytest.raises(CrossMarketError):
        run_optimizer_step(
            relabelled,
            feedback=(),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )


def test_source_hash_is_canonical_stable_and_binds_visible_feedback_and_parameters() -> None:
    feedback = _mature_labels(1)
    equivalent_feedback = (
        replace(
            feedback[0],
            probability=Decimal("0.800"),
            target_position=Decimal("0.7500"),
        ),
    )
    base_kwargs = {
        "week_key": "2026-W31",
        "cutoff_date": date(2026, 7, 30),
        "source_data_max_date": date(2026, 7, 30),
    }

    first = run_optimizer_step(
        seed_model("399006"), feedback=feedback, **base_kwargs
    )
    same = run_optimizer_step(
        seed_model("399006"), feedback=equivalent_feedback, **base_kwargs
    )
    changed_feedback = run_optimizer_step(
        seed_model("399006"),
        feedback=(replace(feedback[0], probability=Decimal("0.81")),),
        **base_kwargs,
    )
    changed_cutoff = run_optimizer_step(
        seed_model("399006"),
        feedback=feedback,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 31),
        source_data_max_date=date(2026, 7, 30),
    )

    assert first.source_hash == same.source_hash
    assert first.source_hash != changed_feedback.source_hash
    assert first.source_hash != changed_cutoff.source_hash
    assert first.to_json() == first.to_json()


def test_label_maturity_requires_explicit_complete_market_calendar() -> None:
    daily = _daily(65)
    prediction = _prediction()

    without_calendar = update_labels(
        (prediction,),
        daily,
        as_of=daily[-1].trade_date,
    )[0]
    expected = _expected_trade_dates(daily)
    sparse = (daily[0],) + tuple(
        row for row in daily[1:] if row.trade_date == max(
            day
            for day in expected
            if day.isocalendar()[:2] == row.trade_date.isocalendar()[:2]
        )
    )
    sparse_label = update_labels(
        (prediction,),
        sparse,
        as_of=daily[-1].trade_date,
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]

    assert without_calendar.status == "pending"
    assert without_calendar.total_loss is None
    assert sparse_label.status == "pending"
    assert sparse_label.visible_row_count == 13


def test_cutoff_natural_week_remainder_is_not_a_complete_trading_week() -> None:
    cutoff = date(2026, 4, 28)
    dates = (
        date(2026, 4, 29),
        date(2026, 4, 30),
        date(2026, 5, 1),
    ) + _weekdays(date(2026, 5, 4), 5)
    daily = (
        DailyClose(cutoff, Decimal("100"), "399006"),
    ) + tuple(
        DailyClose(day, Decimal("101"), "399006") for day in dates
    )
    prediction = _prediction(cutoff_date=cutoff)
    calendar = _market_calendar("399006", dates)

    remainder_only = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 5, 1),
        expected_trade_dates=calendar,
    )[0]
    first_complete = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 5, 8),
        expected_trade_dates=calendar,
    )[0]

    assert remainder_only.status == "pending"
    assert first_complete.status == "partial_1w"
    assert first_complete.observed_through == date(2026, 5, 8)


def test_truncated_calendar_cannot_prove_current_week_is_complete() -> None:
    cutoff = date(2026, 4, 24)
    dates = (
        date(2026, 4, 27),
        date(2026, 4, 28),
        date(2026, 4, 29),
    )
    daily = (DailyClose(cutoff, Decimal("100"), "399006"),) + tuple(
        DailyClose(day, Decimal("101"), "399006") for day in dates
    )
    calendar = optimizer_module.TradingCalendar(
        instrument_code="399006",
        expected_trade_dates=dates,
    )

    label = update_labels(
        (_prediction(),),
        daily,
        as_of=dates[-1],
        expected_trade_dates=calendar,
    )[0]

    assert label.status == "pending"
    assert label.total_loss is None


def test_trading_calendar_rejects_invalid_coverage_boundaries() -> None:
    with pytest.raises(ValueError, match="coverage"):
        optimizer_module.TradingCalendar(
            instrument_code="399006",
            expected_trade_dates=(date(2026, 4, 27),),
            coverage_start=date(2026, 4, 29),
            coverage_end=date(2026, 4, 27),
        )


def test_trading_calendar_carries_market_and_bare_dates_are_rejected() -> None:
    daily = _daily(5)
    dates = _expected_trade_dates(daily)

    with pytest.raises(CrossMarketError):
        update_labels(
            (_prediction(),),
            daily,
            as_of=dates[-1],
            expected_trade_dates=_market_calendar("NDX", dates),
        )
    with pytest.raises(ValueError, match="instrument_code"):
        update_labels(
            (_prediction(),),
            daily,
            as_of=dates[-1],
            expected_trade_dates=dates,
        )


def test_calendar_maturity_uses_last_actual_session_at_exact_1_4_13_week_boundaries() -> None:
    daily = _daily(65)
    expected = _expected_trade_dates(daily)
    prediction = _prediction()

    before_one = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 4, 30),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]
    one = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 5, 1),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]
    before_four = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 5, 21),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]
    four = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 5, 22),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]
    before_thirteen = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 7, 23),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]
    thirteen = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 7, 24),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]

    assert before_one.status == "pending"
    assert one.status == "partial_1w"
    assert before_four.status == "partial_1w"
    assert four.status == "mature_4w"
    assert before_thirteen.status == "mature_4w"
    assert thirteen.status == "mature_13w"
    assert thirteen.observed_through == date(2026, 7, 24)


def test_full_market_holiday_week_delays_thirteenth_trading_week() -> None:
    full = _daily(70)
    holiday_week = full[21].trade_date.isocalendar()[:2]
    daily = (full[0],) + tuple(
        row
        for row in full[1:]
        if row.trade_date.isocalendar()[:2] != holiday_week
    )
    expected = _expected_trade_dates(daily)

    label = update_labels(
        (_prediction(),),
        daily,
        as_of=expected[-1],
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]

    assert len({day.isocalendar()[:2] for day in expected}) == 13
    assert expected[-1] == date(2026, 7, 31)
    assert label.status == "mature_13w"
    assert label.observed_through == expected[-1]


def test_completed_week_ends_alone_never_guess_holiday_sessions() -> None:
    daily = _daily(65)
    completed_ends = tuple(
        max(
            row.trade_date
            for row in daily[1:]
            if row.trade_date.isocalendar()[:2] == week
        )
        for week in dict.fromkeys(
            row.trade_date.isocalendar()[:2] for row in daily[1:]
        )
    )

    with pytest.raises(ValueError, match="instrument_code"):
        update_labels(
            (_prediction(),),
            daily,
            as_of=completed_ends[-1],
            completed_week_ends=(day for day in completed_ends),
        )


def test_actual_direction_uses_last_close_at_or_before_cutoff_not_post_cutoff_gap() -> None:
    dates = _weekdays(date(2026, 4, 27), 65)
    closes = [Decimal("150")] + [Decimal("130")] * 63 + [Decimal("120")]
    daily = (
        DailyClose(date(2026, 4, 24), Decimal("100"), "399006"),
    ) + tuple(
        DailyClose(day, close, "399006")
        for day, close in zip(dates, closes, strict=True)
    )

    label = update_labels(
        (_prediction(direction="down"),),
        daily,
        as_of=dates[-1],
        expected_trade_dates=_market_calendar("399006", dates),
    )[0]

    assert label.actual_direction == "up"
    assert label.direction_hit is False
    assert label.actual_turn_date == dates[0]
    assert label.loss_components["adverse_excursion"] == Decimal("0.5")


def test_same_week_reuses_only_identical_canonical_request_input() -> None:
    feedback = _mature_labels(1)
    prediction = _prediction(iteration_id="REQUEST")
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=feedback,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        prediction=prediction,
    )
    repeated = run_optimizer_step(
        first.work_state,
        feedback=feedback,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        prediction=replace(
            prediction,
            probability=Decimal("0.800"),
            target_position=Decimal("0.7500"),
        ),
    )

    assert repeated.reused is True
    assert repeated.audit.input_hash == first.audit.input_hash

    conflicting_calls = (
        {"cutoff_date": date(2026, 7, 31)},
        {"source_data_max_date": date(2026, 7, 29)},
        {"feedback": (replace(feedback[0], probability=Decimal("0.81")),)},
        {"prediction": replace(prediction, probability=Decimal("0.81"))},
        {"data_integrity": False},
        {
            "current_metrics": {
                "expanding": _metrics("expanding", "0.5", count=1)
            }
        },
    )
    for change in conflicting_calls:
        arguments = {
            "feedback": feedback,
            "week_key": "2026-W31",
            "cutoff_date": date(2026, 7, 30),
            "source_data_max_date": date(2026, 7, 30),
            "prediction": prediction,
        }
        arguments.update(change)
        with pytest.raises(ConflictingWeekInput):
            run_optimizer_step(first.work_state, **arguments)


def test_week_key_is_canonicalized_before_same_week_idempotency() -> None:
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key=" 2026-W31 ",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )
    repeated = run_optimizer_step(
        first.work_state,
        feedback=(),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )

    assert first.audit.week_key == "2026-W31"
    assert repeated.reused is True
    assert repeated.audit.input_hash == first.audit.input_hash


@pytest.mark.parametrize(
    "week_key",
    ("2026-W1", "2026-w31", "2026-W54", "week-31"),
)
def test_optimizer_week_key_rejects_noncanonical_iso_values(
    week_key: str,
) -> None:
    with pytest.raises(ValueError, match="week_key"):
        run_optimizer_step(
            seed_model("399006"),
            feedback=(),
            week_key=week_key,
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )


def test_week_key_must_match_cutoff_iso_week_for_requests_and_predictions() -> None:
    canonical_prediction = replace(
        _prediction(),
        week_key=" 2026-W17 ",
    )
    assert canonical_prediction.week_key == "2026-W17"

    with pytest.raises(ValueError, match="week_key"):
        replace(_prediction(), week_key="2026-W18")
    with pytest.raises(ValueError, match="week_key"):
        run_optimizer_step(
            seed_model("399006"),
            feedback=(),
            week_key="2026-W30",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )


def test_same_week_feedback_hash_deduplicates_stable_iteration_identity() -> None:
    label = _mature_labels(1)[0]
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=(label,),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )
    repeated = run_optimizer_step(
        first.work_state,
        feedback=(label, label),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )

    assert repeated.reused is True
    assert repeated.audit.input_hash == first.audit.input_hash


def test_evaluator_requires_explicit_behavior_identity_for_idempotency() -> None:
    feedback = _mature_labels(20)

    def make_evaluator(loss: str):
        def evaluate(weights, labels, windows):
            return {"20": _metrics("20", loss)}

        return evaluate

    evaluator = make_evaluator("0.5")
    with pytest.raises(ValueError, match="evaluator_identity"):
        run_optimizer_step(
            seed_model("399006"),
            feedback=feedback,
                week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            evaluator=evaluator,
        )

    first = run_optimizer_step(
        seed_model("399006"),
        feedback=feedback,
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        evaluator=evaluator,
        evaluator_identity="loss=0.5",
    )
    with pytest.raises(ConflictingWeekInput):
        run_optimizer_step(
            first.work_state,
            feedback=feedback,
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            evaluator=make_evaluator("0.4"),
            evaluator_identity="loss=0.4",
        )


def test_feedback_merge_is_monotonic_deduplicated_and_rejects_conflicts() -> None:
    daily = _daily(65)
    expected = _expected_trade_dates(daily)
    prediction = _prediction(iteration_id="STABLE")
    partial = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 5, 1),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]
    mature = update_labels(
        (prediction,),
        daily,
        as_of=date(2026, 7, 24),
        expected_trade_dates=_market_calendar("399006", expected),
    )[0]
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=(partial,),
        week_key="2026-W18",
        cutoff_date=date(2026, 5, 1),
        source_data_max_date=date(2026, 5, 1),
    )
    upgraded = run_optimizer_step(
        first.work_state,
        feedback=(mature,),
        week_key="2026-W30",
        cutoff_date=date(2026, 7, 24),
        source_data_max_date=date(2026, 7, 24),
    )
    duplicate = run_optimizer_step(
        upgraded.work_state,
        feedback=(mature,),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )

    assert upgraded.optimizer_memory["feedback_count"] == 1
    assert upgraded.work_state.feedback[0].status == "mature_13w"
    assert duplicate.optimizer_memory["feedback_count"] == 1

    with pytest.raises(FeedbackRegressionError):
        run_optimizer_step(
            upgraded.work_state,
            feedback=(partial,),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )
    with pytest.raises(ConflictingFeedback):
        run_optimizer_step(
            upgraded.work_state,
            feedback=(replace(mature, probability=Decimal("0.81")),),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )
    with pytest.raises(ConflictingFeedback):
        run_optimizer_step(
            upgraded.work_state,
            feedback=(replace(mature, cutoff_date=date(2026, 4, 23)),),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
        )


def test_source_hash_binds_integrity_active_windows_and_all_gate_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        data_integrity=True,
    )
    integrity_changed = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        data_integrity=False,
    )
    monkeypatch.setattr(
        optimizer_module,
        "MIN_RELATIVE_IMPROVEMENT",
        Decimal("0.011"),
    )
    gate_changed = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        data_integrity=True,
    )

    assert baseline.source_hash != integrity_changed.source_hash
    assert baseline.source_hash != gate_changed.source_hash
    assert baseline.audit.data_integrity is True
    assert baseline.audit.future_leakage is False
    assert baseline.audit.active_windows == ()
    assert baseline.audit.gate_parameters["direction_hit_max_drop_points"] == Decimal(
        "0.02"
    )


def test_direction_hit_and_overtrade_gates_use_only_52_after_52_mature() -> None:
    current = {
        "20": _metrics("20", "0.5", hit="0.60"),
        "52": _metrics(
            "52", "0.5", hit="0.60", overtrade="0.20", count=52
        ),
    }
    candidate = {
        "20": _metrics(
            "20", "0.494", hit="0.50", overtrade="0.50"
        ),
        "52": _metrics(
            "52", "0.502", hit="0.58", overtrade="0.21", count=52
        ),
    }

    result = run_optimizer_step(
        seed_model("399006"),
        feedback=_mature_labels(52),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=current,
        candidate_metrics=candidate,
    )

    assert result.accepted is True
    assert "direction_hit_drop_exceeds_2_points:20" not in result.reasons
    assert "overtrade_increase_exceeds_5_percent" not in result.reasons


@pytest.mark.parametrize(
    ("current", "candidate"),
    [
        (
            {"20": _metrics("52", "0.5", count=20)},
            {"20": _metrics("20", "0.49", count=20)},
        ),
        (
            {"20": _metrics("20", "0.5", count=1)},
            {"20": _metrics("20", "0.49", count=1)},
        ),
        (
            {"52": _metrics("52", "0.5", count=52)},
            {"52": _metrics("52", "0.49", count=52)},
        ),
    ],
)
def test_injected_metrics_must_match_active_keys_names_and_sample_counts(
    current: dict[str, WindowMetrics],
    candidate: dict[str, WindowMetrics],
) -> None:
    with pytest.raises(ValueError, match="metric|window|sample"):
        run_optimizer_step(
            seed_model("399006"),
            feedback=_mature_labels(20),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            current_metrics=current,
            candidate_metrics=candidate,
        )


@pytest.mark.parametrize(
    "sample_count",
    ("20", True, Decimal("20.9")),
)
def test_mapping_metric_sample_count_requires_a_nonbool_integer(
    sample_count,
) -> None:
    raw = _metrics("20", "0.5").to_dict()
    raw["sample_count"] = sample_count

    with pytest.raises(ValueError, match="sample_count.*integer"):
        run_optimizer_step(
            seed_model("399006"),
            feedback=_mature_labels(20),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            current_metrics={"20": raw},
            candidate_metrics={"20": _metrics("20", "0.49")},
        )


def test_injected_metric_bundle_mature_count_matches_visible_feedback() -> None:
    inconsistent = RollingMetricWindows(
        instrument_code="399006",
        windows={"20": _metrics("20", "0.5")},
        mature_count=999,
    )

    with pytest.raises(ValueError, match="mature_count"):
        run_optimizer_step(
            seed_model("399006"),
            feedback=_mature_labels(20),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            current_metrics=inconsistent,
            candidate_metrics=inconsistent,
        )


def test_evaluator_metric_bundle_mature_count_matches_visible_feedback() -> None:
    inconsistent = RollingMetricWindows(
        instrument_code="399006",
        windows={"20": _metrics("20", "0.5")},
        mature_count=999,
    )

    def evaluator(weights, feedback, windows):
        return inconsistent

    with pytest.raises(ValueError, match="mature_count"):
        run_optimizer_step(
            seed_model("399006"),
            feedback=_mature_labels(20),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            evaluator=evaluator,
            evaluator_identity="invalid-bundle-count",
        )


def test_flat_and_bundle_metrics_have_the_same_canonical_hashes() -> None:
    feedback = _mature_labels(20)
    flat = {"20": _metrics("20", "0.5")}
    bundle = RollingMetricWindows(
        instrument_code="399006",
        windows=flat,
        mature_count=20,
    )
    arguments = {
        "feedback": feedback,
        "week_key": "2026-W31",
        "cutoff_date": date(2026, 7, 30),
        "source_data_max_date": date(2026, 7, 30),
    }

    flat_result = run_optimizer_step(
        seed_model("399006"),
        current_metrics=flat,
        candidate_metrics=flat,
        **arguments,
    )
    bundle_result = run_optimizer_step(
        seed_model("399006"),
        current_metrics=bundle,
        candidate_metrics=bundle,
        **arguments,
    )

    assert bundle_result.accepted == flat_result.accepted
    assert bundle_result.reasons == flat_result.reasons
    assert bundle_result.audit.input_hash == flat_result.audit.input_hash
    assert bundle_result.source_hash == flat_result.source_hash


def test_same_week_flat_metrics_replay_as_an_equivalent_bundle() -> None:
    feedback = _mature_labels(20)
    flat = {"20": _metrics("20", "0.5")}
    bundle = RollingMetricWindows(
        instrument_code="399006",
        windows=flat,
        mature_count=20,
    )
    arguments = {
        "feedback": feedback,
        "week_key": "2026-W31",
        "cutoff_date": date(2026, 7, 30),
        "source_data_max_date": date(2026, 7, 30),
    }
    first = run_optimizer_step(
        seed_model("399006"),
        current_metrics=flat,
        candidate_metrics=flat,
        **arguments,
    )

    repeated = run_optimizer_step(
        first.work_state,
        current_metrics=bundle,
        candidate_metrics=bundle,
        **arguments,
    )

    assert repeated.reused is True
    assert repeated.audit.input_hash == first.audit.input_hash


def test_same_week_metric_bundle_mature_count_change_is_a_conflict() -> None:
    valid = RollingMetricWindows(
        instrument_code="399006",
        windows={"20": _metrics("20", "0.5")},
        mature_count=20,
    )
    first = run_optimizer_step(
        seed_model("399006"),
        feedback=_mature_labels(20),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
        current_metrics=valid,
        candidate_metrics=valid,
    )
    inconsistent = replace(valid, mature_count=999)

    with pytest.raises(ConflictingWeekInput):
        run_optimizer_step(
            first.work_state,
            feedback=_mature_labels(20),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            current_metrics=inconsistent,
            candidate_metrics=inconsistent,
        )


def test_market_identity_is_required_on_daily_prediction_mapping_and_metrics() -> None:
    mixed_daily = _daily(10) + (
        DailyClose(date(2026, 8, 1), Decimal("101"), "NDX"),
    )
    with pytest.raises(CrossMarketError):
        _update_labels(
            (_prediction(),),
            mixed_daily,
            as_of=date(2026, 8, 1),
        )
    with pytest.raises(CrossMarketError):
        run_optimizer_step(
            seed_model("399006"),
            feedback=(),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            prediction={"feature": "value"},
        )
    with pytest.raises(CrossMarketError):
        run_optimizer_step(
            seed_model("399006"),
            feedback=_mature_labels(20),
            week_key="2026-W31",
            cutoff_date=date(2026, 7, 30),
            source_data_max_date=date(2026, 7, 30),
            current_metrics={
                "20": _metrics("20", "0.5", instrument_code="NDX")
            },
            candidate_metrics={"20": _metrics("20", "0.49")},
        )
    assert rolling_metrics(_mature_labels(1)).instrument_code == "399006"


def test_empty_rolling_metrics_require_and_preserve_explicit_market() -> None:
    with pytest.raises(ValueError, match="instrument_code"):
        rolling_metrics(())

    empty = rolling_metrics((), instrument_code="399006")
    assert empty.instrument_code == "399006"
    assert empty.windows == {}


def test_frozen_dataclasses_copy_nested_inputs_and_enforce_state_invariants() -> None:
    partial = _update_labels(
        (_prediction(),),
        _daily(5),
        as_of=date(2026, 5, 1),
    )[0]
    source_components = {"direction_calibration": Decimal("0.04")}
    copied = replace(partial, loss_components=source_components)
    source_components["direction_calibration"] = Decimal("0.99")
    with pytest.raises(TypeError):
        copied.loss_components["direction_calibration"] = Decimal("0")
    assert copied.loss_components["direction_calibration"] == Decimal("0.04")

    source_windows = {
        "expanding": _metrics("expanding", "0.2", count=1)
    }
    bundle = RollingMetricWindows(
        instrument_code="399006",
        windows=source_windows,
        mature_count=1,
    )
    source_windows.clear()
    assert bundle.windows["expanding"].sample_count == 1

    source_memory = dict(seed_model("399006").optimizer_memory)
    copied_state = replace(
        seed_model("399006"),
        optimizer_memory=source_memory,
    )
    source_memory["feedback_count"] = 99
    assert copied_state.optimizer_memory["feedback_count"] == 0

    mature = _mature_labels(1)[0]
    with pytest.raises(ValueError, match="status"):
        replace(mature, status="pending")

    result = run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W31",
        cutoff_date=date(2026, 7, 30),
        source_data_max_date=date(2026, 7, 30),
    )
    rejected_source = [
        {
            "candidate_weights": dict(seed_model("399006").weights),
            "reasons": ["review"],
        }
    ]
    copied_audit = replace(
        result.audit,
        rejected_candidates=rejected_source,
    )
    rejected_source[0]["candidate_weights"]["valuation"] = Decimal("0")
    assert (
        copied_audit.rejected_candidates[0]["candidate_weights"]["valuation"]
        == Decimal("30")
    )
    with pytest.raises(TypeError):
        copied_audit.rejected_candidates[0]["reasons"][0] = "changed"


def test_value_objects_reject_non_integer_counts_and_non_date_daily_rows() -> None:
    with pytest.raises(ValueError, match="trade_count"):
        replace(_prediction(), trade_count=Decimal("1.5"))
    with pytest.raises(ValueError, match="trade_date"):
        DailyClose("2026-04-24", Decimal("100"), "399006")


def test_same_context_rejected_candidate_memory_never_cycles_weights() -> None:
    state = seed_model("399006")
    feedback = _mature_labels(20)
    metrics = {"20": _metrics("20", "0.5")}
    fingerprints: set[tuple[tuple[str, Decimal], ...]] = set()
    for index in range(7):
        cutoff = date(2026, 7, 30) + timedelta(weeks=index)
        result = run_optimizer_step(
            state,
            feedback=feedback,
            week_key=_iso_week_key(cutoff),
            cutoff_date=cutoff,
            source_data_max_date=cutoff,
            current_metrics=metrics,
            candidate_metrics=metrics,
        )
        fingerprint = tuple(sorted(result.candidate_weights.items()))
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        state = result.work_state


def test_candidate_space_exhaustion_is_explicit_and_does_not_store_duplicate() -> None:
    state = seed_model("399006")
    feedback = _mature_labels(20)
    metrics = {"20": _metrics("20", "0.5")}
    for index in range(60):
        cutoff = date(2026, 7, 30) + timedelta(weeks=index)
        result = run_optimizer_step(
            state,
            feedback=feedback,
            week_key=_iso_week_key(cutoff),
            cutoff_date=cutoff,
            source_data_max_date=cutoff,
            current_metrics=metrics,
            candidate_metrics=metrics,
        )
        state = result.work_state
    exhausted_cutoff = date(2026, 7, 30) + timedelta(weeks=60)
    exhausted = run_optimizer_step(
        state,
        feedback=feedback,
        week_key=_iso_week_key(exhausted_cutoff),
        cutoff_date=exhausted_cutoff,
        source_data_max_date=exhausted_cutoff,
        current_metrics=metrics,
        candidate_metrics=metrics,
    )

    assert exhausted.accepted is False
    assert "candidate_exhausted" in exhausted.reasons
    assert exhausted.optimizer_memory["rejected_candidate_count"] == 60


def test_optimizer_public_boundary_exports_only_pure_task5_api() -> None:
    assert weekly_analysis.seed_model is seed_model
    assert weekly_analysis.run_optimizer_step is run_optimizer_step
    assert weekly_analysis.update_labels is update_labels
    assert weekly_analysis.ConflictingWeekInput is ConflictingWeekInput
    assert {
        "ConflictingFeedback",
        "ConflictingWeekInput",
        "FeedbackRegressionError",
    }.issubset(optimizer_module.__all__)
