from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from backend.app.services.v341_scenario_service import (
    ScenarioRandomPlan,
    apply_temperature,
    canonical_forecast_seed,
    fit_temperature,
    generate_return_scenarios,
    horizon_probabilities,
    model_reliability,
)
from backend.app.services.v341_training_service import (
    LOSS_SCHEMA_VERSION,
    candidate_alphas,
    conservative_independent_sample_count,
    multiclass_brier,
    normalized_loss_components,
    paired_block_bootstrap_interval,
    true_max_drawdown,
    weighted_ridge_fit,
    weighted_interval_score,
)


def test_candidate_alpha_grid_is_clipped_deduplicated_and_bounded() -> None:
    assert candidate_alphas(4.0) == (2.0, 3.0, 4.0, 5.0, 6.0)
    assert candidate_alphas(0.1) == (0.1, 0.125, 0.15)
    assert candidate_alphas(100.0) == (50.0, 75.0, 100.0)
    assert len(candidate_alphas(5.0)) <= 5


def test_conservative_effective_count_uses_non_overlapping_label_intervals() -> None:
    anchors = tuple(date(2024, 1, 5) + timedelta(days=7 * index) for index in range(52))
    intervals = tuple((anchor, anchor + timedelta(weeks=13)) for anchor in anchors)
    count = conservative_independent_sample_count(intervals)
    assert 3 <= count <= 4
    disjoint = tuple((anchor, anchor + timedelta(days=1)) for anchor in anchors)
    assert conservative_independent_sample_count(disjoint) == 52


def test_weighted_ridge_satisfies_weighted_intercept_normal_equation() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([[1.0], [3.0], [5.0], [7.0]])
    weights = np.asarray([1.0, 2.0, 3.0, 4.0])
    fit = weighted_ridge_fit(x, y, weights, alpha=0.1)
    predictions = fit.intercept + fit.transformed_x @ fit.coefficients
    residual = y - predictions
    normalized = weights / np.mean(weights)
    assert np.sum(residual[:, 0] * normalized) == pytest.approx(0.0, abs=1e-10)


def test_multiclass_brier_known_examples() -> None:
    assert multiclass_brier((1.0, 0.0, 0.0), 0) == pytest.approx(0.0)
    assert multiclass_brier((1 / 3, 1 / 3, 1 / 3), 0) == pytest.approx(1 / 3)
    assert multiclass_brier((0.0, 0.0, 1.0), 0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="sum"):
        multiclass_brier((0.6, 0.6, 0.0), 0)


def test_wis_penalizes_width_and_quantile_crossing() -> None:
    narrow = weighted_interval_score(0.9, 1.0, 1.1, 1.0)
    wide = weighted_interval_score(0.5, 1.0, 1.5, 1.0)
    assert narrow < wide
    assert weighted_interval_score(0.9, 1.0, 1.1, 0.5) > narrow
    with pytest.raises(ValueError, match="ordered"):
        weighted_interval_score(1.1, 1.0, 0.9, 1.0)


def test_true_drawdown_differs_from_minimum_cumulative_return() -> None:
    path = (0.20, 0.10, 0.30, 0.05)
    assert min(path) == pytest.approx(0.05)
    assert true_max_drawdown(path) == pytest.approx((1.30 - 1.05) / 1.30)


def test_loss_schema_has_required_components_and_weights() -> None:
    actual = np.tile(np.linspace(0.01, 0.13, 13), (4, 1))
    predicted = actual + 0.005
    scenarios = np.stack([predicted - 0.01, predicted, predicted + 0.01], axis=1)
    probabilities = {
        4: np.tile((0.7, 0.2, 0.1), (4, 1)),
        8: np.tile((0.7, 0.2, 0.1), (4, 1)),
        13: np.tile((0.7, 0.2, 0.1), (4, 1)),
    }
    thresholds = {4: 0.01, 8: 0.01, 13: 0.01}
    result = normalized_loss_components(
        actual,
        predicted,
        scenarios,
        probabilities,
        thresholds,
        scales=np.full(13, 0.02),
        recent_fold_losses=(0.5, 0.6, 0.55),
    )
    assert result["schema_version"] == LOSS_SCHEMA_VERSION
    assert set(result["components"]) == {
        "path_mae",
        "terminal_mae",
        "brier",
        "wis",
        "drawdown_volatility",
        "recent_stability",
    }
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert result["total"] >= 0.0


def test_paired_block_bootstrap_is_reproducible_and_keeps_identity_at_zero() -> None:
    zeros = np.zeros(52)
    first = paired_block_bootstrap_interval(
        zeros, block_length=13, fold_lengths=(26, 26), seed=7, draws=500
    )
    second = paired_block_bootstrap_interval(
        zeros, block_length=13, fold_lengths=(26, 26), seed=7, draws=500
    )
    assert first == second == (0.0, 0.0)
    improvement = paired_block_bootstrap_interval(
        np.full(52, -0.02),
        block_length=13,
        fold_lengths=(26, 26),
        seed=9,
        draws=500,
    )
    assert improvement[1] < 0.0


def test_paired_bootstrap_blocks_never_cross_fold_boundaries() -> None:
    values = np.concatenate((np.ones(13), np.full(13, 3.0)))
    interval = paired_block_bootstrap_interval(
        values,
        block_length=13,
        fold_lengths=(13, 13),
        seed=41,
        draws=500,
    )
    assert interval == pytest.approx((2.0, 2.0))


def test_temperature_calibration_is_versionable_and_changes_known_probabilities() -> None:
    raw = np.asarray([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.2, 0.7]] * 20)
    labels = np.asarray([0, 1, 2] * 20)
    temperature = fit_temperature(raw, labels)
    calibrated = apply_temperature(raw[0], temperature)
    assert 0.25 <= temperature <= 4.0
    assert np.sum(calibrated) == pytest.approx(1.0)
    assert not np.allclose(calibrated, raw[0])


def test_return_scenario_algebra_and_random_plan_are_exact() -> None:
    prediction = np.linspace(0.01, 0.13, 13)
    standardized = np.asarray([
        np.linspace(-1.0, 1.0, 13),
        np.linspace(1.0, -1.0, 13),
    ])
    source_sigma = np.asarray([0.02, 0.04])
    plan = ScenarioRandomPlan.build(
        seed=123,
        scenario_count=1200,
        residual_pool_size=2,
        empirical_pool_size=20,
    )
    scenarios = generate_return_scenarios(
        prediction,
        standardized,
        source_sigma,
        current_sigma=0.03,
        random_plan=plan,
    )
    selected = standardized[np.asarray(plan.residual_indices)]
    assert np.allclose(
        scenarios,
        np.expm1(np.log1p(prediction) + selected * 0.03),
    )
    assert np.all(scenarios > -1.0)
    assert np.array_equal(scenarios, generate_return_scenarios(
        prediction, standardized, source_sigma, 0.03, plan
    ))
    assert sum(group[3] for group in plan.residual_groups) == 1200
    assert plan.to_payload()["plan_hash"] == plan.plan_hash


def test_horizon_probabilities_are_cumulative_and_normalized() -> None:
    scenarios = np.zeros((1200, 13))
    scenarios[:600, :] = np.linspace(0.0, 0.20, 13)
    scenarios[600:900, :] = 0.0
    scenarios[900:, :] = np.linspace(0.0, -0.20, 13)
    result = horizon_probabilities(scenarios, {4: 0.02, 8: 0.03, 13: 0.04})
    assert set(result) == {4, 8, 13}
    for values in result.values():
        assert set(values) == {"up", "sideways", "down"}
        assert sum(values.values()) == pytest.approx(1.0)


def test_canonical_seed_changes_only_with_frozen_identity() -> None:
    first = canonical_forecast_seed("V3.4.1", "399006", date(2026, 7, 31), "MODEL-1", "SCENARIO-1")
    assert first == canonical_forecast_seed(
        "V3.4.1", "399006", date(2026, 7, 31), "MODEL-1", "SCENARIO-1"
    )
    assert first != canonical_forecast_seed(
        "V3.4.1", "399006", date(2026, 8, 7), "MODEL-1", "SCENARIO-1"
    )


def test_reliability_caps_are_not_prediction_probability() -> None:
    low_sample = model_reliability(
        effective_samples=10,
        calibration_status="SHRINKAGE_ONLY",
        brier=0.2,
        wis=0.3,
        coverage=0.8,
        recent_score=0.8,
        ood_score=0.0,
        consistent=True,
        health_status="MODEL_NORMAL",
    )
    assert low_sample["score"] <= 40
    assert low_sample["semantics"] == "model_reliability_not_prediction_probability"
