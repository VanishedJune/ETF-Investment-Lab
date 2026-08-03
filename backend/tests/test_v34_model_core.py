from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import math

import numpy as np
import pytest

from backend.app.services.v33_feature_service import FeatureSnapshot
from backend.app.services.v34_feature_service import (
    V34FeatureError,
    assert_159941_self_only,
)
from backend.app.services.v34_scenario_service import (
    V34ScenarioService,
    _trend_state,
)
from backend.app.services.v34_training_service import (
    HORIZON_WEEKS,
    V34TrainingError,
    V34TrainingService,
    build_training_samples,
    effective_sample_count,
    eligible_fully_matured,
    maturity_status,
    overlap_weights,
    purged_walk_forward_folds,
)


def _snapshot(index: int, market: str = "399006") -> FeatureSnapshot:
    anchor = date(2024, 1, 5) + timedelta(days=7 * index)
    base = 100.0 * math.exp(0.002 * index + 0.025 * math.sin(index / 6.0))
    sequence = []
    for offset in range(99, -1, -1):
        day = anchor - timedelta(days=offset)
        close = base * math.exp(-0.0004 * offset + 0.002 * math.sin((index - offset) / 5))
        open_value = close * 0.998
        sequence.append(
            {
                "trade_date": day.isoformat(),
                "open": open_value,
                "high": close * 1.012,
                "low": open_value * 0.990,
                "close": close,
                "volume": 1_000_000.0 * (1.0 + 0.15 * math.sin((index - offset) / 4)),
                "turnover": close * 1_000_000.0,
                "dif": 0.1 * math.sin((index - offset) / 8),
                "dea": 0.08 * math.sin((index - offset - 1) / 8),
                "macd_histogram": 0.04 * math.sin((index - offset + 1) / 8),
            }
        )
    weekly = {
        "weekly_dif": 0.001 * math.sin(index / 5),
        "weekly_dea": 0.001 * math.sin((index - 1) / 5),
        "weekly_macd_histogram": 0.0004 * math.sin((index + 1) / 5),
        "weekly_dif_slope_1": 0.0002 * math.cos(index / 5),
        "weekly_dif_curvature_3": -0.00004 * math.sin(index / 5),
        "weekly_return_13": 0.03 * math.sin(index / 9),
        "weekly_volume_change_4": 0.1 * math.sin(index / 7),
    }
    daily = {
        "daily_dif": 0.001 * math.sin(index / 3),
        "daily_dea": 0.001 * math.sin((index - 1) / 3),
        "daily_macd_histogram": 0.0005 * math.cos(index / 3),
        "daily_dif_slope_3": 0.0002 * math.cos(index / 3),
        "daily_dea_slope_3": 0.00015 * math.cos((index - 1) / 3),
        "daily_histogram_speed_3": -0.0001 * math.sin(index / 3),
        "daily_return_20": 0.02 * math.sin(index / 8),
    }
    return FeatureSnapshot(
        market=market,
        cutoff_date=anchor,
        source_data_max_date=anchor,
        daily_as_of=anchor,
        weekly_as_of=anchor,
        weekly=weekly,
        daily=daily,
        daily_sequence=tuple(sequence),
        missing_masks={},
        provenance={"target": market, "data_boundary": "SELF_OHLCV_ONLY"},
        derivative_turn={},
        benchmark_market=None,
    )


def _fixture(market: str = "399006", count: int = 90):
    snapshots = tuple(_snapshot(index, market) for index in range(count))
    return snapshots, build_training_samples(snapshots)


def test_exact_13_week_labels_and_pending_tail() -> None:
    _snapshots, samples = _fixture(count=50)
    assert len(samples) == 50
    assert sum(sample.cumulative_return_path is None for sample in samples) == 13
    assert all(
        len(sample.cumulative_return_path) == HORIZON_WEEKS
        for sample in samples[:-13]
        if sample.cumulative_return_path is not None
    )
    assert all(sample.cumulative_return_path is None for sample in samples[-13:])


def test_maturity_states_do_not_fill_unrealized_labels() -> None:
    assert maturity_status(0) == "IMMATURE"
    assert maturity_status(1) == "MATURE_1W"
    assert maturity_status(4) == "MATURE_4W"
    assert maturity_status(8) == "MATURE_8W"
    assert maturity_status(13) == "FULLY_MATURE_13W"
    assert maturity_status(13, trained=True) == "TRAINED"
    _snapshots, samples = _fixture(count=55)
    cutoff = samples[40].anchor_date
    eligible = eligible_fully_matured(samples, cutoff)
    assert eligible
    assert all(sample.label_end_date is not None and sample.label_end_date <= cutoff for sample in eligible)


def test_purged_walk_forward_has_no_label_overlap_and_tracks_effective_count() -> None:
    _snapshots, samples = _fixture(count=100)
    matured = tuple(sample for sample in samples if sample.cumulative_return_path is not None)
    folds = purged_walk_forward_folds(matured)
    assert len(folds) >= 2
    for training, validation in folds:
        validation_start = matured[validation[0]].anchor_date
        assert all(matured[index].label_end_date < validation_start for index in training)
    weights = overlap_weights(matured)
    assert np.all(weights > 0)
    assert np.all(weights <= 1)
    assert 0 < effective_sample_count(weights) <= len(matured) + 1e-9


def test_fold_local_scaler_and_market_isolation() -> None:
    _snapshots, samples = _fixture(count=70)
    matured = tuple(sample for sample in samples if sample.cumulative_return_path is not None)
    service = V34TrainingService()
    state = service.fit(
        "399006",
        matured[:35],
        version="CYB_13W_V3.4.1",
        parent_version=None,
        trained_through=matured[34].label_end_date,
        effective_from=matured[35].anchor_date,
        alpha=4.0,
    )
    assert state.feature_fit_end_date == matured[34].label_end_date
    with pytest.raises(V34TrainingError, match="cross-market"):
        service.fit(
            "159941",
            matured[:35],
            version="bad",
            parent_version=None,
            trained_through=matured[34].label_end_date,
            effective_from=matured[35].anchor_date,
            alpha=4.0,
        )


def test_159941_forbidden_source_field_gate() -> None:
    assert_159941_self_only(("weekly_dif", "daily_volume_change_20", "open", "close"))
    for forbidden in ("NDX_close", "fx_usdcny", "qdii_nav", "premium", "macro_rate", "pe"):
        with pytest.raises(V34FeatureError, match="forbidden"):
            assert_159941_self_only(("weekly_dif", forbidden))


def test_daily_correction_applies_once_to_first_four_weekly_increments() -> None:
    snapshots, samples = _fixture(count=70)
    matured = tuple(sample for sample in samples if sample.cumulative_return_path is not None)
    service = V34TrainingService()
    state = service.fit(
        "399006",
        matured[:40],
        version="CYB_13W_V3.4.1",
        parent_version=None,
        trained_through=matured[39].label_end_date,
        effective_from=snapshots[-1].cutoff_date,
        alpha=5.0,
    )
    forecast = service.predict(state, snapshots[-1])
    increments = np.diff(np.concatenate(([0.0], np.asarray(forecast.daily_correction))))
    assert len(increments) == 13
    assert np.allclose(increments[4:], 0.0)


def test_scenario_kline_quantiles_probability_and_indicator_consistency() -> None:
    snapshots, samples = _fixture(count=100)
    matured = tuple(sample for sample in samples if sample.cumulative_return_path is not None)
    trainer = V34TrainingService()
    state = trainer.fit(
        "399006",
        matured[:70],
        version="CYB_13W_V3.4.1",
        parent_version=None,
        trained_through=matured[69].label_end_date,
        effective_from=snapshots[-1].cutoff_date,
        alpha=5.0,
    )
    path = trainer.predict(state, snapshots[-1])
    scenario = V34ScenarioService(scenario_count=1000).generate(
        state, path, snapshots[-1], matured, seed=123456
    )
    assert len(scenario.representative_ohlcv) == 13
    assert len(scenario.indicators) == 13
    assert len(scenario.price_quantiles) == 13
    for candle, quantile in zip(scenario.representative_ohlcv, scenario.price_quantiles):
        assert candle["low"] <= candle["open"] <= candle["high"]
        assert candle["low"] <= candle["close"] <= candle["high"]
        assert candle["low"] > 0
        assert quantile["close_p10"] <= quantile["close_p50"] <= quantile["close_p90"]
        assert quantile["low_quantile"] <= quantile["high_quantile"]
    assert sum(scenario.direction_probabilities.values()) == pytest.approx(100.0)
    assert sum(scenario.path_probabilities.values()) == pytest.approx(100.0)
    assert all(0.0 < value < 100.0 for value in scenario.direction_probabilities.values())
    assert all(0.0 < value < 100.0 for value in scenario.path_probabilities.values())
    assert scenario.scenario_audit["ohlc_legal_rate"] == 1.0
    assert scenario.scenario_audit["ema_initial_state_source"] == "pre_forecast_real_weekly_close_history"
    same = V34ScenarioService(scenario_count=1000).generate(
        state, path, snapshots[-1], matured, seed=123456
    )
    assert same.payload_hash == scenario.payload_hash
    assert same.representative_ohlcv == scenario.representative_ohlcv


def test_single_dif_sign_change_is_not_confirmed_uptrend() -> None:
    state = _trend_state(
        (100.0, 99.0, 99.5),
        (0.2, 0.15, 0.16),
        (0.21, 0.20, 0.19),
        (-0.02, -0.10, -0.06),
    )
    assert state in {"UPWARD_TURN", "UPWARD_TURN_UNCONFIRMED"}
    assert state != "UPTREND_CONFIRMED"
