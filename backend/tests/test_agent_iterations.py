from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from backend.app.agent_iterations.model import (
    IterationFeedback,
    ModelState,
    WeeklyObservation,
    build_position_plan,
    evaluate_week,
    initial_model,
    update_model,
)
from backend.app.agent_iterations.runner import select_progressive_cutoffs


def _observation(**changes: object) -> WeeklyObservation:
    values: dict[str, object] = {
        "trade_date": date(2024, 6, 28),
        "close": Decimal("2000"),
        "volume": Decimal("1000000"),
        "volume_ratio": Decimal("1.2"),
        "dif": Decimal("20"),
        "dea": Decimal("12"),
        "dif_slope": Decimal("3"),
        "dea_slope": Decimal("2"),
        "macd_histogram": Decimal("16"),
        "rsi": Decimal("62"),
        "ma_20": Decimal("1850"),
        "ma_60": Decimal("1750"),
        "valuation_percentile": Decimal("35"),
        "volatility": Decimal("0.22"),
        "drawdown": Decimal("-0.08"),
        "golden_strength": 2,
        "black_strength": 0,
    }
    values.update(changes)
    return WeeklyObservation(**values)  # type: ignore[arg-type]


def test_fund_etf_advice_uses_only_three_integer_ratios_and_preserves_target() -> None:
    model = initial_model("399006")

    strong = evaluate_week(model, _observation(), current_position=40)
    neutral = evaluate_week(
        model,
        _observation(
            dif=Decimal("0"),
            dea=Decimal("0"),
            dif_slope=Decimal("0"),
            dea_slope=Decimal("0"),
            macd_histogram=Decimal("0"),
            close=Decimal("1800"),
            ma_20=Decimal("1800"),
            ma_60=Decimal("1800"),
            valuation_percentile=Decimal("50"),
            golden_strength=0,
        ),
        current_position=50,
    )
    weak = evaluate_week(
        model,
        _observation(
            dif=Decimal("-30"),
            dea=Decimal("-10"),
            dif_slope=Decimal("-4"),
            dea_slope=Decimal("-2"),
            macd_histogram=Decimal("-40"),
            rsi=Decimal("34"),
            close=Decimal("1500"),
            ma_20=Decimal("1750"),
            ma_60=Decimal("1850"),
            valuation_percentile=Decimal("92"),
            golden_strength=0,
            black_strength=3,
        ),
        current_position=90,
    )

    assert {strong.fund_etf_ratio, neutral.fund_etf_ratio, weak.fund_etf_ratio} == {
        "7:3",
        "6:4",
        "5:5",
    }
    for advice in (strong, neutral, weak):
        assert advice.fund_allocation + advice.etf_allocation == advice.target_position
        assert isinstance(advice.fund_allocation, int)
        assert isinstance(advice.etf_allocation, int)


def test_position_batches_match_the_four_approved_safety_examples() -> None:
    high_but_strong = build_position_plan(90, 70, mode="sell", confirmed=False)
    high_and_topping = build_position_plan(95, 15, mode="sell", confirmed=True)
    low_but_weak = build_position_plan(10, 20, mode="buy", confirmed=False)
    low_and_reversing = build_position_plan(20, 80, mode="buy", confirmed=True)

    assert high_but_strong.batches == [12, 8]
    assert high_but_strong.conditional_batches == [8]
    assert high_and_topping.batches == [32, 24, 16, 8]
    assert low_but_weak.batches == [10]
    assert low_and_reversing.batches == [18, 18, 15, 9]
    assert max(high_and_topping.batches) < 90


def test_first_feedback_is_applied_to_the_second_model_with_bounded_weights() -> None:
    first = initial_model("399006")
    feedback = IterationFeedback(
        actual_direction=-1,
        predicted_direction=1,
        deviation_days=3,
        probability=Decimal("0.63"),
        component_scores={
            "valuation": Decimal("-88"),
            "eda": Decimal("70"),
            "dip": Decimal("45"),
            "signal_point": Decimal("0"),
            "trend": Decimal("80"),
            "market_regime": Decimal("40"),
        },
    )

    second = update_model(first, feedback)

    assert second.iteration == 2
    assert second.version == "M002"
    assert second.parent_version == "M001"
    assert sum(second.weights.values()) == Decimal("100")
    assert max(abs(second.weights[key] - first.weights[key]) for key in first.weights) <= Decimal("2")
    assert second.feedback_count == 1


def test_china_and_us_models_start_with_independent_parameters() -> None:
    china = initial_model("399006")
    us = initial_model("NDX")

    assert china.weights != us.weights
    assert china.market == "CN"
    assert us.market == "US"


def test_progressive_cutoffs_are_deterministic_chronological_and_labels_mature_before_next() -> None:
    weekly_dates = [date(2014, 1, 3) + timedelta(days=7 * index) for index in range(650)]
    daily_dates = [date(2014, 1, 2) + timedelta(days=index) for index in range(4600)]
    daily_dates = [item for item in daily_dates if item.weekday() < 5]

    first = select_progressive_cutoffs(weekly_dates, daily_dates, count=100, seed=20260730)
    second = select_progressive_cutoffs(weekly_dates, daily_dates, count=100, seed=20260730)

    assert first == second
    assert len(first) == 100
    assert first == sorted(first)
    for current, following in zip(first, first[1:]):
        future_days = [item for item in daily_dates if item > current][:20]
        assert len(future_days) == 20
        assert future_days[-1] < following


def test_model_state_serialization_keeps_explicit_eda_dip_aliases() -> None:
    state: ModelState = initial_model("NDX")
    payload = state.as_dict()

    assert payload["indicator_aliases"] == {"DIP": "DIF", "EDA": "DEA"}
