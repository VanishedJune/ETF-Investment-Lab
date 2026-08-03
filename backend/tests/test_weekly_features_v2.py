from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import pytest

from backend.app.indicators.calculator import (
    IndicatorCalculator,
    IndicatorParameters,
    PricePoint,
)
from backend.app.weekly_analysis import features as features_module
from backend.app.weekly_analysis.domain import PeriodBar
from backend.app.weekly_analysis.features import (
    FutureDataError,
    ValuationPoint,
    build_feature_snapshot,
)


def _weekly_bars(
    closes: list[int | str],
    *,
    instrument_code: str = "399006",
    start: date = date(2024, 1, 1),
) -> tuple[PeriodBar, ...]:
    bars: list[PeriodBar] = []
    for index, raw_close in enumerate(closes):
        period_start = start + timedelta(days=index * 7)
        period_end = period_start + timedelta(days=4)
        close = Decimal(str(raw_close))
        volume = (
            None
            if instrument_code == "NDX"
            else Decimal(1_000 + index * 10)
        )
        bars.append(
            PeriodBar(
                timeframe="weekly",
                period_start=period_start,
                period_end=period_end,
                open_price=close,
                high_price=close,
                low_price=close,
                close_price=close,
                volume=volume,
                source="DIRECT_INDEX",
                volume_source=(
                    "VOLUME_UNAVAILABLE:DIRECT_INDEX"
                    if instrument_code == "NDX"
                    else "AGGREGATED_DAILY_VOLUME"
                ),
                price_source="DIRECT_INDEX",
            )
        )
    return tuple(bars)


def _valuations(
    bars: tuple[PeriodBar, ...],
    values: list[int | str] | None = None,
) -> tuple[ValuationPoint, ...]:
    raw_values = values or list(range(1, len(bars) + 1))
    return tuple(
        ValuationPoint(bar.period_end, Decimal(str(value)))
        for bar, value in zip(bars, raw_values, strict=True)
    )


def _issue_codes(snapshot) -> set[str]:
    return {issue.code for issue in snapshot.quality_report.issues}


def _calendar_sessions(bars: tuple[PeriodBar, ...]) -> tuple[date, ...]:
    sessions: list[date] = []
    for bar in bars:
        current = bar.period_start
        while current <= bar.period_end:
            if current.weekday() < 5:
                sessions.append(current)
            current += timedelta(days=1)
    return tuple(sorted(set(sessions)))


def test_feature_snapshot_matches_indicator_calculator_and_audits_aliases() -> None:
    closes = [100 + ((index * 7) % 19) for index in range(70)]
    bars = _weekly_bars(closes)
    cutoff = bars[-1].period_end
    valuations = _valuations(bars)

    snapshot = build_feature_snapshot(
        bars,
        valuations,
        cutoff,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )
    reference = IndicatorCalculator().calculate(
        [
            PricePoint(bar.period_end, bar.close_price, bar.volume)
            for bar in bars
            if bar.close_price is not None
        ],
        "weekly",
    ).snapshots[-1]

    assert snapshot.instrument_code == "399006"
    assert snapshot.cutoff_date == cutoff
    assert snapshot.source_data_max_date == cutoff
    assert snapshot.source_data_max_date <= cutoff
    assert snapshot.feature_set_version == "weekly-features-v2"
    for name in (
        "ema_12",
        "ema_26",
        "dif",
        "dea",
        "macd_histogram",
        "rsi_6",
        "ma_20",
        "ma_60",
        "volatility_20",
        "current_drawdown",
        "running_drawdown",
    ):
        assert snapshot.features[name] == reference.values[name]
    assert snapshot.features["dip"] == snapshot.features["dif"]
    assert snapshot.features["eda"] == snapshot.features["dea"]
    assert snapshot.features["macd_histogram"] == Decimal("2") * (
        snapshot.features["dif"] - snapshot.features["dea"]
    )
    assert snapshot.audit_fields["dip_alias_of"] == "dif"
    assert snapshot.audit_fields["eda_alias_of"] == "dea"
    assert snapshot.audit_fields["indicator_formula_label"] == (
        "technical_indicators/v2_partial_window"
    )
    assert snapshot.quality_report.is_publishable is True


def test_snapshot_is_immutable_and_has_stable_json_decimal_serialization() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    with pytest.raises(TypeError):
        snapshot.features["dif"] = Decimal("0")
    assert not isinstance(snapshot.features, dict)
    with pytest.raises(TypeError):
        dict.__setitem__(snapshot.features, "dif", Decimal("0"))
    local_features = snapshot.features
    with pytest.raises(TypeError):
        local_features |= {"dif": Decimal("0")}
    first = snapshot.to_json()
    second = snapshot.to_json()
    payload = json.loads(first)

    assert first == second
    assert Decimal(payload["features"]["dif"]) == snapshot.features["dif"]
    assert payload["features"]["dif"] == "7"
    assert payload["cutoff_date"] == bars[-1].period_end.isoformat()
    assert payload["source_hash"] == snapshot.source_hash


def test_frozen_mapping_hash_matches_order_independent_equality_contract() -> None:
    nested_forward = features_module.FrozenDict(
        [("x", Decimal("1")), ("y", ("a", "b"))]
    )
    nested_reverse = features_module.FrozenDict(
        [("y", ("a", "b")), ("x", Decimal("1.0"))]
    )
    forward = features_module.FrozenDict(
        [("alpha", nested_forward), ("beta", 2)]
    )
    reverse = features_module.FrozenDict(
        [("beta", 2), ("alpha", nested_reverse)]
    )

    assert forward == reverse
    assert hash(forward) == hash(reverse)
    assert len({forward, reverse}) == 1
    lookup = {forward: "found"}
    assert lookup[reverse] == "found"


def test_decimal_equivalents_share_canonical_json_and_source_hash() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    equivalent_bars = tuple(
        replace(
            bar,
            open_price=(
                Decimal("1E+2")
                if index == 0
                else Decimal(f"{bar.open_price}.00")
            ),
            high_price=(
                Decimal("1E+2")
                if index == 0
                else Decimal(f"{bar.high_price}.00")
            ),
            low_price=(
                Decimal("1E+2")
                if index == 0
                else Decimal(f"{bar.low_price}.00")
            ),
            close_price=(
                Decimal("1E+2")
                if index == 0
                else Decimal(f"{bar.close_price}.00")
            ),
            volume=Decimal(f"{bar.volume}.00"),
        )
        for index, bar in enumerate(bars)
    )
    baseline_valuations = tuple(
        ValuationPoint(bar.period_end, Decimal("0.0"))
        for bar in bars
    )
    equivalent_valuations = tuple(
        ValuationPoint(bar.period_end, Decimal("-0E+7"))
        for bar in equivalent_bars
    )
    expected = tuple(bar.period_end for bar in bars)

    baseline = build_feature_snapshot(
        bars,
        baseline_valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    equivalent = build_feature_snapshot(
        equivalent_bars,
        equivalent_valuations,
        equivalent_bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )

    assert baseline.source_hash == equivalent.source_hash
    assert baseline.to_json() == equivalent.to_json()
    assert json.loads(equivalent.to_json())["features"]["valuation_value"] == "0"


def test_decimal_canonicalizer_never_rounds_distinct_high_precision_values() -> None:
    first = Decimal("12345678901234567890123456781")
    second = Decimal("12345678901234567890123456782")

    assert features_module._canonical_decimal(first) == str(first)
    assert features_module._canonical_decimal(second) == str(second)
    assert (
        features_module._canonical_decimal(first)
        != features_module._canonical_decimal(second)
    )


def test_sixty_week_warmup_blocks_publication_without_fabricating_zeroes() -> None:
    bars = _weekly_bars(list(range(100, 159)))

    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert snapshot.quality_report.is_publishable is False
    assert "INSUFFICIENT_WARMUP" in _issue_codes(snapshot)
    assert snapshot.availability["ma_60"] == "insufficient_warmup"
    assert snapshot.features["ma_60"] is not None
    assert snapshot.features["valuation_percentile"] is not None
    assert snapshot.features["golden_point"] is None
    assert snapshot.features["golden_strength"] is None
    assert snapshot.features["golden_point_reasons"] is None
    assert snapshot.availability["golden_point"] == "insufficient_warmup"


@pytest.mark.parametrize("future_kind", ["price", "valuation"])
def test_future_rows_are_rejected_with_a_blocking_quality_issue(
    future_kind: str,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    cutoff = bars[-2].period_end
    valuations = _valuations(bars[:-1])
    input_bars = bars if future_kind == "price" else bars[:-1]
    input_valuations = (
        valuations
        if future_kind == "price"
        else valuations
        + (ValuationPoint(bars[-1].period_end, Decimal("999999")),)
    )

    with pytest.raises(FutureDataError) as captured:
        build_feature_snapshot(
            input_bars,
            input_valuations,
            cutoff,
            "399006",
            expected_week_ends=tuple(
                bar.period_end for bar in bars if bar.period_end <= cutoff
            ),
        )

    assert captured.value.quality_issue.severity == "blocking"
    assert captured.value.quality_issue.code in {
        "WEEKLY_BAR_AFTER_CUTOFF",
        "VALUATION_AFTER_CUTOFF",
    }
    assert captured.value.quality_issue.start_date > cutoff


def test_multiple_cutoff_prefixes_are_invariant_to_extreme_future_source_rows() -> None:
    bars = _weekly_bars([100 + (index % 11) for index in range(72)])
    valuations = _valuations(bars)
    extreme_bar = replace(
        bars[-1],
        period_start=bars[-1].period_start + timedelta(days=7),
        period_end=bars[-1].period_end + timedelta(days=7),
        open_price=Decimal("999999999"),
        high_price=Decimal("999999999"),
        low_price=Decimal("999999999"),
        close_price=Decimal("999999999"),
        volume=Decimal("999999999"),
    )
    extended_bars = bars + (extreme_bar,)
    extended_valuations = valuations + (
        ValuationPoint(extreme_bar.period_end, Decimal("-999999999")),
    )

    def visible_snapshot(
        all_bars: tuple[PeriodBar, ...],
        all_valuations: tuple[ValuationPoint, ...],
        cutoff: date,
    ):
        visible_bars = tuple(
            bar for bar in all_bars if bar.period_end <= cutoff
        )
        visible_valuations = tuple(
            point
            for point in all_valuations
            if point.valuation_date <= cutoff
        )
        return build_feature_snapshot(
            visible_bars,
            visible_valuations,
            cutoff,
            "399006",
            expected_week_ends=tuple(bar.period_end for bar in visible_bars),
        )

    for index in (59, 63, 68):
        cutoff = bars[index].period_end
        baseline = visible_snapshot(bars, valuations, cutoff)
        mutated = visible_snapshot(
            extended_bars,
            extended_valuations,
            cutoff,
        )
        assert mutated == baseline
        assert mutated.to_json() == baseline.to_json()
        assert mutated.source_data_max_date <= cutoff


def test_valuation_percentile_uses_only_values_visible_at_cutoff() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    visible_values = [50] * 64 + [1]
    valuations = _valuations(bars, visible_values)

    snapshot = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert snapshot.features["valuation_value"] == Decimal("1")
    assert snapshot.features["valuation_percentile"] == (
        Decimal("50") / Decimal("65")
    )
    assert snapshot.features["valuation_observation_count"] == 65
    assert snapshot.availability["valuation_percentile"] == "available"


@pytest.mark.parametrize(
    ("observation_count", "expected_availability"),
    [
        (1, "insufficient_history"),
        (19, "insufficient_history"),
        (20, "available"),
    ],
)
def test_valuation_percentile_requires_twenty_visible_observations(
    observation_count: int,
    expected_availability: str,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuation_bars = bars[-observation_count:]

    snapshot = build_feature_snapshot(
        bars,
        _valuations(valuation_bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert (
        snapshot.features["valuation_observation_count"]
        == observation_count
    )
    assert (
        snapshot.availability["valuation_percentile"]
        == expected_availability
    )
    if observation_count < 20:
        assert snapshot.features["valuation_percentile"] is None
        assert "INSUFFICIENT_VALUATION_HISTORY" in _issue_codes(snapshot)
        assert snapshot.features["golden_point"] is None
        assert (
            snapshot.availability["golden_point"]
            == "insufficient_valuation_history"
        )
    else:
        assert snapshot.features["valuation_percentile"] is not None


def test_valuation_percentile_uses_audited_midrank_tie_policy() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuation_bars = bars[-20:]
    valuations = _valuations(
        valuation_bars,
        [1] * 5 + [2] * 15,
    )

    snapshot = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert snapshot.features["valuation_percentile"] == Decimal("62.5")
    assert snapshot.audit_fields["valuation_tie_policy"] == "midrank"
    assert snapshot.audit_fields["valuation_percentile_formula"] == (
        "100*(count_less+count_equal/2)/observation_count"
    )
    assert snapshot.audit_fields["valuation_min_observations"] == 20


def test_valuation_minimum_history_is_part_of_feature_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuations = _valuations(bars[-20:])
    expected = tuple(bar.period_end for bar in bars)
    baseline = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )

    monkeypatch.setattr(
        features_module,
        "_VALUATION_MIN_OBSERVATIONS",
        21,
        raising=False,
    )
    changed = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )

    assert baseline.features["valuation_percentile"] is not None
    assert changed.features["valuation_percentile"] is None
    assert baseline.source_hash != changed.source_hash


def test_missing_valuation_is_none_and_reported_without_zero_or_forward_fill() -> None:
    bars = _weekly_bars(list(range(100, 165)))

    snapshot = build_feature_snapshot(
        bars,
        (),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert snapshot.features["valuation_value"] is None
    assert snapshot.features["valuation_percentile"] is None
    assert snapshot.features["valuation_observation_count"] == 0
    assert snapshot.availability["valuation_percentile"] == "missing"
    assert snapshot.features["golden_point"] is None
    assert snapshot.features["golden_strength"] is None
    assert snapshot.features["golden_point_reasons"] is None
    assert snapshot.availability["golden_point"] == "missing_valuation"
    assert "VALUATION_UNAVAILABLE" in _issue_codes(snapshot)


def test_399006_incomplete_twenty_week_volume_window_is_blocking() -> None:
    bars = list(_weekly_bars(list(range(100, 165))))
    bars[-10] = replace(bars[-10], volume=None)

    snapshot = build_feature_snapshot(
        tuple(bars),
        _valuations(tuple(bars)),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert snapshot.features["volume_ratio"] is None
    assert snapshot.availability["volume_ratio"] == "missing_required_volume"
    assert snapshot.quality_report.is_publishable is False
    assert "INCOMPLETE_VOLUME_WINDOW" in _issue_codes(snapshot)


def test_ndx_direct_index_volume_is_unavailable_without_blocking_price_features() -> None:
    bars = _weekly_bars(list(range(100, 165)), instrument_code="NDX")

    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "NDX",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert snapshot.features["volume_ratio"] is None
    assert (
        snapshot.availability["volume_ratio"]
        == "not_available_for_direct_index"
    )
    assert (
        snapshot.quality_report.volume_availability
        == "not_available_for_direct_index"
    )
    assert snapshot.quality_report.is_publishable is True
    assert snapshot.features["ma_60"] is not None
    assert "DIRECT_INDEX_VOLUME_UNAVAILABLE" in _issue_codes(snapshot)


def test_missing_expected_week_uses_real_two_vs_three_session_threshold() -> None:
    all_bars = _weekly_bars(list(range(100, 166)))
    missing_one = all_bars[:30] + all_bars[31:]
    expected = tuple(bar.period_end for bar in all_bars)
    missing_period = all_bars[30]
    missing_week = missing_period.period_end.isocalendar()[:2]
    other_sessions = tuple(
        day
        for day in _calendar_sessions(all_bars)
        if day.isocalendar()[:2] != missing_week
    )
    two_sessions = (
        *other_sessions,
        missing_period.period_start,
        missing_period.period_end,
    )
    three_sessions = (
        *other_sessions,
        missing_period.period_start,
        missing_period.period_start + timedelta(days=1),
        missing_period.period_end,
    )

    short_gap = build_feature_snapshot(
        missing_one,
        _valuations(missing_one),
        missing_one[-1].period_end,
        "399006",
        expected_week_ends=expected,
        expected_trade_dates=two_sessions,
    )
    long_gap = build_feature_snapshot(
        missing_one,
        _valuations(missing_one),
        missing_one[-1].period_end,
        "399006",
        expected_week_ends=expected,
        expected_trade_dates=three_sessions,
    )

    assert "SHORT_EXPECTED_WEEK_GAP" in _issue_codes(short_gap)
    assert short_gap.quality_report.is_publishable is True
    assert "MISSING_COMPLETED_WEEK" in _issue_codes(long_gap)
    assert long_gap.quality_report.is_publishable is False
    assert short_gap.source_hash != long_gap.source_hash


def test_expected_week_without_any_calendar_session_is_blocking_inconsistency() -> None:
    all_bars = _weekly_bars(list(range(100, 166)))
    missing_one = all_bars[:30] + all_bars[31:]
    missing_week = all_bars[30].period_end.isocalendar()[:2]
    sessions_without_missing_week = tuple(
        day
        for day in _calendar_sessions(all_bars)
        if day.isocalendar()[:2] != missing_week
    )

    snapshot = build_feature_snapshot(
        missing_one,
        _valuations(missing_one),
        missing_one[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in all_bars),
        expected_trade_dates=sessions_without_missing_week,
    )

    assert "CALENDAR_WEEK_WITHOUT_SESSIONS" in _issue_codes(snapshot)
    assert snapshot.quality_report.is_publishable is False


def test_actual_week_end_later_than_expected_is_always_blocking() -> None:
    original = _weekly_bars(list(range(100, 165)))
    target = original[30]
    late = replace(target, period_end=target.period_end + timedelta(days=1))
    bars = original[:30] + (late,) + original[31:]

    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in original),
        expected_trade_dates=_calendar_sessions(original),
    )

    assert "WEEK_END_AFTER_EXPECTED" in _issue_codes(snapshot)
    assert snapshot.quality_report.is_publishable is False


def test_weekday_holiday_uses_calendar_confirmed_endpoint_without_fake_gap() -> None:
    original = _weekly_bars(list(range(100, 165)))
    target = original[30]
    holiday_bar = replace(
        target,
        period_end=target.period_start + timedelta(days=3),
    )
    bars = original[:30] + (holiday_bar,) + original[31:]
    expected = tuple(bar.period_end for bar in bars)
    holiday_sessions = _calendar_sessions(bars)

    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
        expected_trade_dates=holiday_sessions,
    )

    assert "MISSING_COMPLETED_WEEK" not in _issue_codes(snapshot)
    assert "WEEK_END_MISMATCH" not in _issue_codes(snapshot)
    assert snapshot.quality_report.is_publishable is True


def test_expected_week_end_must_be_latest_calendar_session_even_when_actual_matches() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    target = bars[30]
    target_week = target.period_end.isocalendar()[:2]
    sessions_through_thursday = tuple(
        day
        for day in _calendar_sessions(bars)
        if not (
            day.isocalendar()[:2] == target_week
            and day == target.period_end
        )
    )

    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
        expected_trade_dates=sessions_through_thursday,
    )

    assert "EXPECTED_WEEK_END_MISMATCH" in _issue_codes(snapshot)
    assert snapshot.quality_report.is_publishable is False


def test_multiple_expected_endpoints_in_one_iso_week_are_blocking() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    target = bars[30]
    expected_with_duplicate_week = (
        tuple(bar.period_end for bar in bars)
        + (target.period_end - timedelta(days=1),)
    )

    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected_with_duplicate_week,
        expected_trade_dates=_calendar_sessions(bars),
    )

    assert "MULTIPLE_EXPECTED_WEEK_ENDS" in _issue_codes(snapshot)
    assert snapshot.quality_report.is_publishable is False


@pytest.mark.parametrize(
    (
        "closes",
        "valuation_values",
        "point_name",
        "strength_name",
        "expected_reasons",
    ),
    [
        (
            [100] * 60 + [95, 90, 85, 80, 82, 85, 90, 95, 100],
            [50] * 68 + [1],
            "golden_point",
            "golden_strength",
            (
                "dif_crossed_above_dea",
                "dif_at_or_below_zero",
                "macd_histogram_rising",
                "valuation_percentile_low",
                "price_at_or_above_ma20",
            ),
        ),
        (
            [100] * 60 + [105, 110, 115, 120, 118, 115, 110, 105, 100],
            [50] * 68 + [99],
            "black_point",
            "black_strength",
            (
                "dif_crossed_below_dea",
                "dif_at_or_above_zero",
                "macd_histogram_falling",
                "valuation_percentile_high",
                "price_below_ma20",
            ),
        ),
    ],
)
def test_golden_and_black_points_are_deterministic_explainable_weekly_signals(
    closes: list[int],
    valuation_values: list[int],
    point_name: str,
    strength_name: str,
    expected_reasons: tuple[str, ...],
) -> None:
    bars = _weekly_bars(closes)
    valuations = _valuations(bars, valuation_values)

    first = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )
    second = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert first == second
    assert first.features[point_name] is True
    assert first.features[strength_name] == 5
    assert first.features[f"{point_name}_reasons"] == expected_reasons
    assert first.availability[point_name] == "available"
    assert first.availability[strength_name] == "available"
    assert first.availability[f"{point_name}_reasons"] == "available"
    assert first.audit_fields["signal_engine"] == "deterministic_weekly_v2"


@pytest.mark.parametrize(
    ("closes", "valuation_values", "expected_strength", "excluded_reason"),
    [
        (
            [100] * 60 + [95, 90, 85, 80, 82, 85, 90, 95, 100],
            [50] * 69,
            4,
            "valuation_percentile_low",
        ),
        (
            list(range(100, 165)),
            [50] * 64 + [1],
            0,
            "dif_crossed_above_dea",
        ),
    ],
)
def test_golden_signal_counterexamples_remain_available_but_false(
    closes: list[int],
    valuation_values: list[int],
    expected_strength: int,
    excluded_reason: str,
) -> None:
    bars = _weekly_bars(closes)
    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars, valuation_values),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )

    assert snapshot.availability["golden_point"] == "available"
    assert snapshot.features["golden_point"] is False
    assert snapshot.features["golden_strength"] == expected_strength
    assert (
        excluded_reason
        not in snapshot.features["golden_point_reasons"]
    )


def test_visible_source_changes_change_hash_without_touching_model_artifacts() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    expected = tuple(bar.period_end for bar in bars)
    first = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    changed = bars[:-1] + (
        replace(
            bars[-1],
            open_price=Decimal("999"),
            high_price=Decimal("999"),
            low_price=Decimal("999"),
            close_price=Decimal("999"),
        ),
    )
    second = build_feature_snapshot(
        changed,
        _valuations(changed),
        changed[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )

    assert first.source_hash != second.source_hash

    changed_high_only = bars[:-1] + (
        replace(bars[-1], high_price=bars[-1].high_price + Decimal("1")),
    )
    third = build_feature_snapshot(
        changed_high_only,
        _valuations(changed_high_only),
        changed_high_only[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    assert first.source_hash != third.source_hash


def test_complete_indicator_parameters_are_audited() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    snapshot = build_feature_snapshot(
        bars,
        _valuations(bars),
        bars[-1].period_end,
        "399006",
        expected_week_ends=tuple(bar.period_end for bar in bars),
    )
    assert json.loads(
        snapshot.audit_fields["indicator_parameters_json"]
    ) == IndicatorParameters().as_dict()


def test_feature_hash_binds_indicator_formula_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuations = _valuations(bars)
    expected = tuple(bar.period_end for bar in bars)
    baseline = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    monkeypatch.setattr(
        IndicatorCalculator,
        "formula_label",
        "technical_indicators/review_formula",
    )
    formula_changed = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    assert baseline.source_hash != formula_changed.source_hash


def test_feature_hash_binds_complete_indicator_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuations = _valuations(bars)
    expected = tuple(bar.period_end for bar in bars)
    baseline = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    monkeypatch.setattr(
        features_module,
        "IndicatorParameters",
        lambda: IndicatorParameters(macd_histogram_multiplier=1),
    )
    parameters_changed = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    assert baseline.source_hash != parameters_changed.source_hash


def test_volume_ratio_uses_hashed_indicator_volume_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuations = _valuations(bars)
    expected = tuple(bar.period_end for bar in bars)
    baseline = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )

    monkeypatch.setattr(
        features_module,
        "IndicatorParameters",
        lambda: IndicatorParameters(volume_ma_period=19),
    )
    changed = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )

    assert changed.features["volume_ratio"] != baseline.features["volume_ratio"]
    assert changed.source_hash != baseline.source_hash


def test_feature_hash_binds_signal_rule_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuations = _valuations(bars)
    expected = tuple(bar.period_end for bar in bars)
    baseline = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )

    monkeypatch.setattr(
        features_module,
        "_SIGNAL_RULE_VERSION",
        "deterministic_weekly_review",
        raising=False,
    )
    version_changed = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    assert baseline.source_hash != version_changed.source_hash


def test_feature_hash_binds_signal_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuations = _valuations(bars)
    expected = tuple(bar.period_end for bar in bars)
    baseline = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    monkeypatch.setattr(
        features_module,
        "_GOLDEN_VALUATION_MAX",
        Decimal("31"),
        raising=False,
    )
    threshold_changed = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
    )
    assert baseline.source_hash != threshold_changed.source_hash


def test_expected_trade_dates_are_normalized_at_or_before_cutoff_in_hash() -> None:
    bars = _weekly_bars(list(range(100, 165)))
    valuations = _valuations(bars)
    expected = tuple(bar.period_end for bar in bars)
    sessions = _calendar_sessions(bars)
    baseline = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
        expected_trade_dates=sessions,
    )
    noisy_equivalent = build_feature_snapshot(
        bars,
        valuations,
        bars[-1].period_end,
        "399006",
        expected_week_ends=expected,
        expected_trade_dates=(
            tuple(reversed(sessions))
            + (sessions[0], bars[-1].period_end + timedelta(days=7))
        ),
    )

    assert baseline.source_hash == noisy_equivalent.source_hash
