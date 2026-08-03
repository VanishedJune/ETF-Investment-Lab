from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.services.v33_analysis_service import V33AnalysisService
from backend.app.services.v33_feature_service import (
    FeatureSnapshot,
    FutureLeakageError,
    PointInTimeObservation,
    PriceBar,
    V33FeatureError,
    V33FeatureService,
    WEEKLY_MACD_STABLE_BARS,
    flatten_snapshot,
)
from backend.app.services.market_calendar import ExchangeCalendarProvider
from backend.app.services.v33_training_service import (
    DAILY_CONFIDENCE_ADJUSTMENT_LIMIT,
    DAILY_CORRECTION_DECAY,
    DAILY_CORRECTION_WEEKS,
    DAILY_PATH_CORRECTION_MAX_ABS,
    DAILY_POSITION_ADJUSTMENT_LIMIT,
    HORIZON_WEEKS,
    MIN_159941_COLD_START_SAMPLES,
    MIN_TRAIN_SAMPLES,
    InMemoryV33TrainingRepository,
    V33Forecast,
    V33TrainingError,
    V33TrainingPoint,
    V33TrainingService,
    build_training_points,
    deterministic_weekly_sample_dates,
    state_from_payload,
    state_to_payload,
)


def _weekday_bars(count: int = 620, *, start: date = date(2023, 1, 2), base: float = 100.0) -> list[PriceBar]:
    rows: list[PriceBar] = []
    current = start
    index = 0
    while len(rows) < count:
        if current.weekday() < 5:
            trend = base * (1.0 + 0.0007 * index)
            wave = 0.02 * base * np.sin(index / 17.0)
            close = float(trend + wave)
            open_price = close * (1.0 - 0.002 * np.cos(index / 9.0))
            rows.append(
                PriceBar(
                    trade_date=current,
                    open=float(open_price),
                    high=float(max(open_price, close) * 1.01),
                    low=float(min(open_price, close) * 0.99),
                    close=close,
                    volume=float(1_000_000 + 1000 * index + 50000 * np.sin(index / 11.0)),
                    turnover=float(100_000_000 + index * 10000),
                    source="TEST_DIRECT",
                )
            )
            index += 1
        current += timedelta(days=1)
    return rows


def _observation(
    market: str,
    series: str,
    value: float,
    effective: date,
    *,
    available: date | None = None,
) -> PointInTimeObservation:
    available_date = available or effective
    stamp = datetime.combine(available_date, datetime.min.time(), tzinfo=timezone.utc)
    return PointInTimeObservation(
        market=market,
        series=series,
        value=value,
        effective_date=effective,
        published_at=stamp,
        available_at=stamp,
        source="TEST_PIT",
    )


def test_features_use_full_100_day_curve_and_keep_missing_masks() -> None:
    bars = _weekday_bars()
    cutoff = bars[-1].trade_date
    observations = []
    for week in range(30):
        observed = cutoff - timedelta(weeks=29 - week)
        observations.extend(
            [
                _observation("399006", "pe", 20.0 + week * 0.2, observed),
                _observation("399006", "pb", 3.0 + week * 0.02, observed),
            ]
        )
    observations.append(
        _observation(
            "399006",
            "pe",
            999.0,
            cutoff + timedelta(days=1),
            available=cutoff + timedelta(days=1),
        )
    )
    snapshot = V33FeatureService().build_snapshot(
        "399006", bars, cutoff, observations=observations
    )

    assert len(snapshot.daily_sequence) == 100
    assert snapshot.weekly["weekly_dif"] is not None
    assert snapshot.weekly["weekly_dea"] is not None
    assert snapshot.weekly["weekly_macd_histogram"] is not None
    assert snapshot.daily["daily_dif_slope_5"] is not None
    assert snapshot.daily["daily_dif_curvature_5"] is not None
    for span in (9, 13, 21):
        stem = f"daily_dif_turn_{span:02d}"
        assert snapshot.daily[f"{stem}_a"] is not None
        assert snapshot.daily[f"{stem}_b"] is not None
        assert snapshot.daily[f"{stem}_c"] is not None
        assert snapshot.daily[f"{stem}_second_derivative"] is not None
        assert snapshot.daily[f"{stem}_kind"] in (-1.0, 0.0, 1.0)
    for span in (5, 9, 13):
        stem = f"weekly_dif_turn_{span:02d}"
        assert snapshot.weekly[f"{stem}_a"] is not None
        assert snapshot.weekly[f"{stem}_b"] is not None
        assert snapshot.weekly[f"{stem}_c"] is not None
        assert snapshot.weekly[f"{stem}_second_derivative"] is not None
        assert snapshot.weekly[f"{stem}_kind"] in (-1.0, 0.0, 1.0)
    assert snapshot.weekly["pe"] == pytest.approx(25.8)
    assert snapshot.weekly["pe_change_13"] is not None
    assert snapshot.weekly["pb_change_26"] is not None
    assert snapshot.missing_masks["missing_pe"] == 0
    assert snapshot.missing_masks["missing_earnings_growth"] == 1
    assert snapshot.provenance["excluded_future_observations"] == 1
    flattened = flatten_snapshot(snapshot)
    assert "seq100_dct_dif_00" in flattened
    assert "seq100_dct_macd_histogram_11" in flattened
    assert flattened["missing_earnings_growth"] == 1.0
    assert flattened["weekly_dif"] == pytest.approx(snapshot.weekly["weekly_dif"])
    assert flattened["daily_dif"] == pytest.approx(snapshot.daily["daily_dif"])
    assert flattened["pe"] == pytest.approx(snapshot.weekly["pe"])
    assert flattened["missing_weekly_dif"] == 0.0
    assert flattened["missing_daily_dif"] == 0.0
    assert flattened["daily_dif_turn_09_a"] == pytest.approx(
        snapshot.daily["daily_dif_turn_09_a"]
    )
    assert flattened["weekly_dif_turn_05_a"] == pytest.approx(
        snapshot.weekly["weekly_dif_turn_05_a"]
    )
    assert flattened["missing_daily_dif_turn_09_a"] == 0.0
    assert flattened["missing_weekly_dif_turn_05_a"] == 0.0


def test_future_price_bar_is_hard_blocked() -> None:
    bars = _weekday_bars(150)
    with pytest.raises(FutureLeakageError):
        V33FeatureService(require_weekly_bars=20).build_snapshot(
            "399006", bars, bars[-2].trade_date
        )


def test_qdii_first_formal_week_accepts_54_weeks_and_masks_60_week_feature() -> None:
    # 159941 starts in July 2015, leaving only about 54 complete ETF weeks at
    # the first date of the ten-year formal window.  NDX must not be used to
    # manufacture the missing ETF history.
    first_window = _weekday_bars(54 * 5, start=date(2015, 7, 13), base=1.0)
    first = V33FeatureService().build_snapshot(
        "159941", first_window, first_window[-1].trade_date
    )

    assert first.provenance["weekly_bar_count"] == 54
    assert first.weekly["weekly_dif"] is not None
    assert first.weekly["weekly_dea"] is not None
    assert first.weekly["weekly_macd_histogram"] is not None
    assert first.weekly["weekly_drawdown_60"] is None
    assert first.missing_masks["missing_weekly_drawdown_60"] == 1
    assert first.weekly["weekly_volume_zscore_52"] is not None

    mature_window = _weekday_bars(60 * 5, start=date(2015, 7, 13), base=1.0)
    mature = V33FeatureService().build_snapshot(
        "159941", mature_window, mature_window[-1].trade_date
    )
    assert mature.provenance["weekly_bar_count"] == 60
    assert mature.weekly["weekly_drawdown_60"] is not None
    assert mature.missing_masks["missing_weekly_drawdown_60"] == 0


def test_weekly_macd_stability_window_is_missing_not_fabricated() -> None:
    early_bars = _weekday_bars(20 * 5, start=date(2015, 7, 13), base=1.0)
    early = V33FeatureService().build_snapshot(
        "159941", early_bars, early_bars[-1].trade_date
    )
    assert early.provenance["weekly_bar_count"] == 20
    assert early.weekly["weekly_dif"] is None
    assert early.weekly["weekly_dea"] is None
    assert early.weekly["weekly_macd_histogram"] is None
    assert early.missing_masks["missing_weekly_dif"] == 1
    assert early.weekly["weekly_dif_turn_05_a"] is None
    assert early.weekly["weekly_dif_turn_consensus_stable"] is None
    assert early.missing_masks["missing_weekly_dif_turn_05_a"] == 1
    assert early.missing_masks["missing_weekly_dif_turn_consensus_stable"] == 1

    stable_bars = _weekday_bars(
        WEEKLY_MACD_STABLE_BARS * 5, start=date(2015, 7, 13), base=1.0
    )
    stable = V33FeatureService().build_snapshot(
        "159941", stable_bars, stable_bars[-1].trade_date
    )
    assert stable.provenance["weekly_bar_count"] == WEEKLY_MACD_STABLE_BARS
    assert stable.weekly["weekly_dif"] is not None
    assert stable.missing_masks["missing_weekly_dif"] == 0
    assert stable.weekly["weekly_dif_turn_05_a"] is not None
    assert stable.missing_masks["missing_weekly_dif_turn_05_a"] == 0


def test_adjusted_close_scales_all_ohlc_and_preserves_candle_shape() -> None:
    row = SimpleNamespace(
        trade_date=date(2026, 7, 31),
        open_price=10.0,
        high_price=12.0,
        low_price=9.0,
        close_price=11.0,
        adjusted_close_price=22.0,
        volume=1000,
        volume_multiplier=100,
        turnover=2000,
        source="AKSHARE_ETF_QFQ",
    )
    bar = V33FeatureService._row_to_bar(row)
    assert (bar.open, bar.high, bar.low, bar.close) == (20.0, 24.0, 18.0, 22.0)
    assert bar.raw_close == 11.0
    assert bar.volume == 100_000.0
    raw_body_ratio = (11.0 - 10.0) / (12.0 - 9.0)
    adjusted_body_ratio = (bar.close - bar.open) / (bar.high - bar.low)
    assert adjusted_body_ratio == pytest.approx(raw_body_ratio)


def test_ohlc_validation_tolerates_only_qfq_machine_rounding() -> None:
    bars = _weekday_bars(220)
    large_close = 23_200_000.0
    rounded_high = np.nextafter(large_close, 0.0)
    bars[-1] = replace(
        bars[-1],
        open=large_close - 10.0,
        high=float(rounded_high),
        low=large_close - 20.0,
        close=large_close,
    )
    snapshot = V33FeatureService().build_snapshot(
        "159941", bars, bars[-1].trade_date
    )
    assert snapshot.daily_as_of == bars[-1].trade_date

    bars[-1] = replace(bars[-1], high=large_close - 1.0)
    with pytest.raises(V33FeatureError, match="invalid OHLC"):
        V33FeatureService().build_snapshot("159941", bars, bars[-1].trade_date)


def test_159941_uses_direct_target_and_lagged_ndx_qdii_features() -> None:
    etf = _weekday_bars(base=1.2)
    ndx = _weekday_bars(base=15_000.0)
    cutoff = etf[-1].trade_date
    observations = [
        _observation("159941", "nav", 1.35, cutoff),
        _observation("159941", "estimated_nav", 1.36, cutoff),
        _observation("159941", "fx_usdcny", 7.1, cutoff),
    ]
    snapshot = V33FeatureService().build_snapshot(
        "159941",
        etf,
        cutoff,
        observations=observations,
        benchmark_bars=ndx,
    )
    assert snapshot.market == "159941"
    assert snapshot.benchmark_market == "NDX"
    assert snapshot.daily_sequence[-1]["close"] == etf[-1].close
    assert snapshot.weekly["qdii_ndx_return_20"] is not None
    assert snapshot.weekly["qdii_nav_premium"] == pytest.approx(etf[-1].close / 1.35 - 1.0)
    assert snapshot.missing_masks["missing_ndx_benchmark"] == 0
    assert snapshot.missing_masks["missing_fund_shares"] == 1


def test_qdii_premium_uses_raw_close_while_technicals_keep_qfq_close() -> None:
    etf = _weekday_bars(base=1.2)
    latest = etf[-1]
    etf[-1] = replace(
        latest,
        open=latest.open * 2.0,
        high=latest.high * 2.0,
        low=latest.low * 2.0,
        close=latest.close * 2.0,
        raw_close=latest.close,
    )
    cutoff = etf[-1].trade_date
    snapshot = V33FeatureService().build_snapshot(
        "159941",
        etf,
        cutoff,
        observations=[
            _observation("159941", "nav", latest.close, cutoff),
            _observation("159941", "estimated_nav", latest.close, cutoff),
        ],
        benchmark_bars=_weekday_bars(base=15_000.0),
    )

    assert snapshot.daily_sequence[-1]["close"] == pytest.approx(latest.close * 2.0)
    assert snapshot.weekly["qdii_raw_close"] == pytest.approx(latest.close)
    assert snapshot.weekly["qdii_nav_premium"] == pytest.approx(0.0)
    assert snapshot.weekly["qdii_estimated_nav_premium"] == pytest.approx(0.0)
    assert snapshot.provenance["qdii_nav_premium_price_basis"] == "unadjusted_exchange_close"


def _simple_snapshot(market: str, index: int) -> FeatureSnapshot:
    cutoff = date(2020, 1, 3) + timedelta(weeks=index)
    sequence = tuple(
        {
            "trade_date": (cutoff - timedelta(days=99 - day)).isoformat(),
            "open": 100.0 + index + day * 0.01,
            "high": 101.0 + index + day * 0.01,
            "low": 99.0 + index + day * 0.01,
            "close": 100.5 + index + day * 0.01,
            "volume": 1_000_000.0 + day,
            "turnover": 10_000_000.0 + day,
            "dif": 0.1 + index * 0.001 + day * 0.0001,
            "dea": 0.08 + index * 0.001 + day * 0.00008,
            "macd_histogram": 0.04 + day * 0.00004,
            "dif_slope_1": 0.001,
            "dea_slope_1": 0.0008,
            "body_ratio": 0.25,
            "upper_shadow_ratio": 0.25,
            "lower_shadow_ratio": 0.25,
        }
        for day in range(100)
    )
    return FeatureSnapshot(
        market=market,
        cutoff_date=cutoff,
        source_data_max_date=cutoff,
        daily_as_of=cutoff,
        weekly_as_of=cutoff - timedelta(days=7),
        weekly={
            "weekly_return_4": 0.01 * np.sin(index / 4),
            "weekly_dif": 0.001 * np.sin(index / 8),
            "weekly_dea": 0.0008 * np.sin(index / 8),
            "weekly_dif_slope_1": 0.001 * np.cos(index / 8),
            "weekly_dif_slope_3": 0.001 * np.cos(index / 8),
            "pe_expanding_percentile": 50.0,
            "pb_expanding_percentile": 50.0,
        },
        daily={
            "daily_return_5": 0.005 * np.sin(index / 3),
            "daily_dif": 0.001 * np.sin(index / 5),
            "daily_dea": 0.0008 * np.sin(index / 5),
        },
        daily_sequence=sequence,
        missing_masks={"missing_family_macro": 1, "missing_pe": 0, "missing_pb": 0},
        provenance={"source": "TEST"},
        derivative_turn={
            "daily_dif": {
                "stable": True,
                "kind": "bottom" if index % 2 == 0 else "top",
                "days_ahead": 8.0,
            },
            "weekly_dif": {"stable": True, "kind": "bottom", "days_ahead": 3.0},
        },
        benchmark_market="NDX" if market == "159941" else None,
    )


def test_build_points_has_exact_20_week_paths_and_20_pending() -> None:
    points = build_training_points([_simple_snapshot("399006", index) for index in range(80)])
    assert len(points) == 80
    assert all(len(point.future_path) == 20 for point in points[:-20] if point.future_path is not None)
    assert all(point.future_path is None for point in points[-20:])
    assert sum(point.outcome_available_date is None for point in points) == 20


def test_fixed_seed_samples_one_real_session_per_natural_week() -> None:
    sessions = [bar.trade_date for bar in _weekday_bars(35)]
    first = deterministic_weekly_sample_dates(sessions, "399006", seed=90210)
    second = deterministic_weekly_sample_dates(sessions, "399006", seed=90210)
    assert first == second
    assert all(day in sessions for day in first)
    assert len({day.isocalendar()[:2] for day in first}) == len(first)


def test_first_future_week_dif_turn_is_not_skipped() -> None:
    from backend.app.services.v33_training_service import _first_sign_turn

    # The first future natural week contains only four real sessions here.
    # A fixed week*5 conversion would incorrectly report five days.
    session_offsets = (0, 4, 9)
    assert _first_sign_turn((-0.2, 0.1, 0.2), session_offsets) == (4.0, "bottom")
    assert _first_sign_turn((0.2, -0.1, -0.2), session_offsets) == (4.0, "top")


def test_turn_errors_use_real_session_offsets_instead_of_week_times_five() -> None:
    actual_path = [0.0] * HORIZON_WEEKS
    actual_path[2] = 0.20
    actual_path[4] = -0.20
    session_offsets = (4, 8, 11, 16, 20, 25, 30, 35, 40, 45,
                       50, 55, 60, 65, 70, 75, 80, 85, 90, 95)
    point = V33TrainingPoint(
        sequence_number=1,
        market="399006",
        cutoff_date=_simple_snapshot("399006", 0).cutoff_date,
        snapshot=_simple_snapshot("399006", 0),
        future_path=tuple(actual_path),
        outcome_available_date=date(2020, 5, 22),
        future_session_offsets=session_offsets,
        actual_dif_turn_days=11.0,
        actual_dif_turn_kind="bottom",
    )
    zeros = (0.0,) * HORIZON_WEEKS
    forecast = V33Forecast(
        market="399006",
        cutoff_date=point.cutoff_date,
        p10=tuple(value - 0.05 for value in actual_path),
        p50=tuple(actual_path),
        p90=tuple(value + 0.05 for value in actual_path),
        expected=zeros,
        weekly_base=zeros,
        daily_correction=zeros,
        up_probability=0.0,
        sideways_probability=100.0,
        down_probability=0.0,
        direction="sideways",
        confidence=100.0,
        threshold=0.045,
        expected_max_drawdown=-20.0,
        predicted_high_week=1,
        predicted_low_week=2,
        dif_turn_kind="bottom",
        dif_turn_days=5.0,
        price_turn_days=None,
        turn_stable=True,
        analogue_calibration_count=0,
        model_version="test",
        model_state_hash="test",
    )

    metrics = V33TrainingService.evaluate(point, forecast)

    assert metrics["high_turn_error_days"] == 7.0  # session 11 - session 4
    assert metrics["low_turn_error_days"] == 12.0  # session 20 - session 8
    assert metrics["dif_turn_error_days"] == 6.0  # session 11 - session 5


def test_formal_progressive_window_never_skips_a_week_and_keeps_20_pending() -> None:
    points = build_training_points(
        [_simple_snapshot("399006", index) for index in range(90)],
        sampling_seed=330020,
    )
    repository = InMemoryV33TrainingRepository()
    result = V33TrainingService(repository).train_progressive(
        "399006",
        points,
        audit_start_date=points[60].cutoff_date,
        expected_iteration_count=30,
        weekly_sampling_seed=330020,
    )
    assert result.expected_iteration_count == result.audited_iteration_count == 30
    assert len(result.sampled_dates) == 30
    assert result.pending_count == 20
    assert result.full_count == 10
    assert result.state.optimizer_memory["residual_calibration_mode"] == "purged_oos"
    anchor = repository.warmup_anchors["399006"]
    root = state_from_payload(anchor["state_payload"])
    assert root.iteration_number == 0
    assert root.parent_state_hash is None
    assert root.optimizer_memory["warmup_only"] is True
    assert result.iterations[0].parent_state_hash == root.state_hash
    assert result.iterations[1].parent_state_hash == result.iterations[0].state_hash
    assert len(anchor["eligible_cutoff_dates"]) == root.training_sample_count

    # Re-saving the exact root is idempotent; any manifest drift is rejected.
    repository.save_warmup_anchor(
        root,
        build_cutoff_dates=tuple(
            date.fromisoformat(value) for value in anchor["build_cutoff_dates"]
        ),
        eligible_cutoff_dates=tuple(
            date.fromisoformat(value) for value in anchor["eligible_cutoff_dates"]
        ),
    )
    with pytest.raises(V33TrainingError, match="warm-up anchor changed"):
        repository.save_warmup_anchor(
            root,
            build_cutoff_dates=(root.trained_through,),
            eligible_cutoff_dates=(),
        )


def test_iteration_inherits_state_and_models_remain_market_specific() -> None:
    snapshots = [_simple_snapshot("399006", index) for index in range(65)]
    points = build_training_points(snapshots)
    eligible = [point for point in points[:30] if point.future_path is not None]
    service = V33TrainingService()
    first_current = replace(points[44], future_path=None, outcome_available_date=None)
    first, first_record = service.train_iteration(first_current, eligible[:24], None)
    second_current = replace(points[45], future_path=None, outcome_available_date=None)
    second, second_record = service.train_iteration(second_current, eligible[:25], first)

    assert first.market == second.market == "399006"
    assert second.parent_state_hash == first.state_hash
    assert second.iteration_number == first.iteration_number + 1
    assert len(second_record.forecast.p50) == 20
    assert all(
        low <= middle <= high
        for low, middle, high in zip(
            second_record.forecast.p10,
            second_record.forecast.p50,
            second_record.forecast.p90,
        )
    )
    ndx_etf_point = V33TrainingPoint(
        sequence_number=1,
        market="159941",
        cutoff_date=first_current.cutoff_date,
        snapshot=replace(first_current.snapshot, market="159941", benchmark_market="NDX"),
        future_path=None,
        outcome_available_date=None,
    )
    with pytest.raises(V33TrainingError):
        service.train_iteration(ndx_etf_point, eligible[:24], first)


def test_159941_cold_start_uses_12_real_labels_and_disables_challenger() -> None:
    etf_points = build_training_points(
        [_simple_snapshot("159941", index) for index in range(60)]
    )
    current = replace(etf_points[32], future_path=None, outcome_available_date=None)
    eligible = etf_points[:MIN_159941_COLD_START_SAMPLES]
    state, record = V33TrainingService().train_iteration(current, eligible, None)

    assert MIN_159941_COLD_START_SAMPLES == 12
    assert MIN_TRAIN_SAMPLES == 24
    assert state.training_sample_count == 12
    assert state.hyperparameters["ridge_alpha"] >= 40.0
    assert state.hyperparameters["boost_rounds"] <= 2.0
    assert state.optimizer_memory["training_mode"] == "small_sample_degraded"
    assert state.optimizer_memory["residual_calibration_mode"] == "small_sample_degraded"
    assert record.challenger_promoted is False
    assert record.promotion_reason == "small_sample_degraded_initial_champion"

    cyb_points = build_training_points(
        [_simple_snapshot("399006", index) for index in range(60)]
    )
    cyb_current = replace(cyb_points[32], future_path=None, outcome_available_date=None)
    with pytest.raises(V33TrainingError, match="insufficient real matured"):
        V33TrainingService().train_iteration(
            cyb_current, cyb_points[:MIN_159941_COLD_START_SAMPLES], None
        )


def test_159941_real_listing_span_keeps_all_512_formal_weeks() -> None:
    calendar = ExchangeCalendarProvider()
    sessions = calendar.sessions(
        "159941", date(2015, 7, 13), date(2026, 7, 31)
    )
    sampled = deterministic_weekly_sample_dates(sessions, "159941", seed=330020)
    formal_start = sampled[-1] - timedelta(days=3653)
    formal = [day for day in sampled if day >= formal_start]
    assert formal[0] == date(2016, 7, 29)
    assert len(formal) == 512

    grouped: dict[tuple[int, int], list[date]] = {}
    for day in sessions:
        grouped.setdefault(day.isocalendar()[:2], []).append(day)
    ordered_weeks = sorted(grouped)
    week_index = {key: index for index, key in enumerate(ordered_weeks)}
    session_index = {day: index + 1 for index, day in enumerate(sessions)}
    assert all(
        session_index[day] >= 100 and week_index[day.isocalendar()[:2]] >= 20
        for day in formal
    )

    first_formal = formal[0]
    first_week_index = week_index[first_formal.isocalendar()[:2]]
    weekly_cutoff = max(grouped[ordered_weeks[first_week_index - 1]])
    visible_sessions = [day for day in sessions if day <= first_formal]
    bars = [
        PriceBar(
            trade_date=day,
            open=1.0 + index * 0.0001,
            high=1.01 + index * 0.0001,
            low=0.99 + index * 0.0001,
            close=1.005 + index * 0.0001,
            volume=1_000_000.0 + index,
            turnover=10_000_000.0 + index,
            source="TEST_159941_LISTING_SPAN",
        )
        for index, day in enumerate(visible_sessions)
    ]
    snapshot = V33FeatureService().build_snapshot(
        "159941",
        bars,
        first_formal,
        weekly_cutoff_date=weekly_cutoff,
    )
    assert snapshot.daily_as_of == first_formal
    assert snapshot.provenance["weekly_bar_count"] >= WEEKLY_MACD_STABLE_BARS


def test_model_state_round_trip_and_tamper_detection() -> None:
    points = build_training_points([_simple_snapshot("399006", index) for index in range(60)])
    service = V33TrainingService()
    current = replace(points[44], future_path=None, outcome_available_date=None)
    state, _ = service.train_iteration(current, points[:24], None)
    payload = state_to_payload(state)
    restored = state_from_payload(payload)
    assert restored.state_hash == state.state_hash
    payload["intercept"][0] += 0.5
    with pytest.raises(V33TrainingError, match="state_hash"):
        state_from_payload(payload)


class _WeekdayCalendar:
    def sessions(self, market: str, start: date, end: date) -> tuple[date, ...]:
        assert market in ("399006", "159941")
        output = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                output.append(current)
            current += timedelta(days=1)
        return tuple(output)


def test_analysis_is_pure_uses_5_point_grid_and_five_allowed_ratios() -> None:
    points = build_training_points([_simple_snapshot("159941", index) for index in range(62)])
    repository = InMemoryV33TrainingRepository()
    training = V33TrainingService(repository)
    current = replace(points[44], future_path=None, outcome_available_date=None)
    state, _ = training.train_iteration(current, points[:24], None)
    repository.save_state(state)
    analysis = V33AnalysisService(training=training, champions=repository, calendar=_WeekdayCalendar())
    before = state_to_payload(state)
    result = analysis.run("159941", current.snapshot, current_position=0, calibration_points=points[:24])
    after = state_to_payload(repository.latest_state("159941"))

    assert result["training_mutated"] is False
    assert result["model"]["horizon_weeks"] == 20
    assert len(result["path"]["p50"]) == 20
    assert len(result["features"]["weekly"]) > 0
    assert result["features"]["daily_sequence_count"] == 100
    assert result["position"]["target"] % 5 == 0
    assert all(batch["percentage_points"] % 5 == 0 for batch in result["advice"]["batches"])
    assert result["advice"]["fund_etf_ratio"] in {"7:3", "6:4", "5:5", "4:6", "3:7"}
    assert before == after


def test_weekly_main_and_daily_corrector_have_enforced_responsibility_boundary() -> None:
    points = build_training_points([_simple_snapshot("399006", index) for index in range(65)])
    service = V33TrainingService()
    current = replace(points[44], future_path=None, outcome_available_date=None)
    state, record = service.train_iteration(current, points[:24], None)
    forecast = record.forecast

    assert np.allclose(
        np.asarray(forecast.expected),
        np.asarray(forecast.weekly_base) + np.asarray(forecast.daily_correction),
    )
    assert forecast.daily_correction[DAILY_CORRECTION_WEEKS:] == (0.0,) * (
        HORIZON_WEEKS - DAILY_CORRECTION_WEEKS
    )
    for correction, decay in zip(
        forecast.daily_correction[:DAILY_CORRECTION_WEEKS], DAILY_CORRECTION_DECAY
    ):
        assert abs(correction) <= DAILY_PATH_CORRECTION_MAX_ABS * decay + 1e-12
    assert all(
        not state.feature_names[stump.feature_index].startswith(
            ("daily_", "seq100_", "missing_daily_", "missing_seq100_")
        )
        for stump in state.stumps
    )

    changed_daily = {
        name: (None if value is None else float(value) * -7.0 + 0.123)
        for name, value in current.snapshot.daily.items()
    }
    changed_sequence = tuple(
        {
            **row,
            "dif": float(row["dif"]) * -5.0,
            "dea": float(row["dea"]) * 4.0,
            "macd_histogram": float(row["macd_histogram"]) * -6.0,
            "dif_slope_1": float(row["dif_slope_1"]) * -5.0,
            "dea_slope_1": float(row["dea_slope_1"]) * 4.0,
        }
        for row in current.snapshot.daily_sequence
    )
    daily_only_variant = replace(
        current.snapshot,
        daily=changed_daily,
        daily_sequence=changed_sequence,
    )
    variant = service.predict(state, daily_only_variant)

    assert np.allclose(variant.weekly_base, forecast.weekly_base, rtol=0.0, atol=1e-12)
    assert np.allclose(
        variant.expected[DAILY_CORRECTION_WEEKS:],
        forecast.expected[DAILY_CORRECTION_WEEKS:],
        rtol=0.0,
        atol=1e-12,
    )


def test_daily_corrector_cannot_reverse_high_confidence_weekly_action() -> None:
    points = build_training_points([_simple_snapshot("159941", index) for index in range(60)])
    training = V33TrainingService()
    current = replace(points[44], future_path=None, outcome_available_date=None)
    state, record = training.train_iteration(current, points[:24], None)
    weekly_base = tuple(0.007 * (week / HORIZON_WEEKS) for week in range(1, HORIZON_WEEKS + 1))
    daily_correction = tuple(
        -DAILY_PATH_CORRECTION_MAX_ABS * decay for decay in DAILY_CORRECTION_DECAY
    ) + (0.0,) * (HORIZON_WEEKS - DAILY_CORRECTION_WEEKS)
    forced_forecast = replace(
        record.forecast,
        weekly_base=weekly_base,
        daily_correction=daily_correction,
        expected=tuple(
            base + correction for base, correction in zip(weekly_base, daily_correction)
        ),
        p10=tuple(value - 0.02 for value in weekly_base),
        p50=weekly_base,
        p90=tuple(value + 0.02 for value in weekly_base),
        direction="up",
        confidence=80.0,
        up_probability=80.0,
        sideways_probability=15.0,
        down_probability=5.0,
        expected_max_drawdown=0.0,
    )

    class _ForcedTraining:
        @staticmethod
        def predict(*_args, **_kwargs):
            return forced_forecast

    analysis = V33AnalysisService(
        training=_ForcedTraining(),  # type: ignore[arg-type]
        champions=InMemoryV33TrainingRepository(),
        calendar=_WeekdayCalendar(),
    )
    result = analysis.analyze(current.snapshot, state, current_position=50)

    assert result["position"]["weekly_base_target"] > 50
    assert result["position"]["target"] >= 50
    assert abs(result["position"]["daily_adjustment"]) <= DAILY_POSITION_ADJUSTMENT_LIMIT
    assert abs(result["daily_corrector"]["confidence_adjustment"]) <= DAILY_CONFIDENCE_ADJUSTMENT_LIMIT


def test_loss_contains_all_approved_components() -> None:
    points = build_training_points([_simple_snapshot("399006", index) for index in range(65)])
    service = V33TrainingService()
    current = replace(points[45], future_path=None, outcome_available_date=None)
    state, record = service.train_iteration(current, points[:24], None)
    metrics = service.evaluate(points[24], record.forecast)
    assert metrics["composite_loss"] is not None
    assert metrics["wis_pinball_loss"] is not None
    assert metrics["brier_score"] is not None
    assert metrics["turn_loss"] is not None
    assert metrics["coverage_interval_loss"] is not None
    assert metrics["drawdown_loss"] is not None
    assert metrics["turnover_loss"] is not None
