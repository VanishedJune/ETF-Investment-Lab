"""V3.5 protocol constants and deterministic strategy defaults.

The values in this module are the single source of truth for the
V3.5_CAPITAL_DRIVEN_8W model packages.  They are copied into every
``v35_model_packages.strategy_config_json`` row so that a frozen package can
never be affected by later edits here.
"""

from __future__ import annotations

from typing import Any, Mapping


PROTOCOL_VERSION = "V3.5_CAPITAL_DRIVEN_8W"
HORIZON_WEEKS = 8
FEATURE_VERSION = "V3.5_FEATURE_MANIFEST_1"
LOSS_SCHEMA_VERSION = "LOSS_SCHEMA_V35_1"
PROMOTION_RULE_VERSION = "PROMOTION_RULE_V35_1"
POLICY_VERSION_DEFAULT = "POLICY_V35_1"
CANDIDATE_RANDOM_PLAN_VERSION = "V3.5_CANDIDATE_PLAN_1"
RESIDUAL_SCHEMA_VERSION = "RESIDUAL_SCHEMA_V35_1"

# Initial Champion training sufficiency (from the approved plan; mirrors the
# V3.3/V3.4.1 sample contract applied to 8-week labels).
MIN_TRAIN_SAMPLES_BY_MARKET = {
    "399006": 24,
    "159941": 12,
}
SMALL_SAMPLE_CHAMPION_THRESHOLD = 12
PREWARMING_THRESHOLD = 12
MIN_VALIDATION_SAMPLES = 8
SMALL_SAMPLE_REGULARIZED = True

# Probability calibration thresholds (V3.4.1 protocol carried over).
PRELIMINARY_CALIBRATION_SAMPLES = 30
FORMAL_CALIBRATION_SAMPLES = 50

# Weekly processing cadence (V3.5 section 16).
PREDICTION_CHALLENGE_EVERY_MATURED = 4
STRATEGY_CHALLENGE_EVERY_MATURED = 8
PREDICTION_CHALLENGER_COUNT = 5
STRATEGY_CHALLENGER_COUNT = 3
ALPHA_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)
ALPHA_MIN = 0.1
ALPHA_MAX = 100.0

# StrategyScore weights (V3.5 section 6.1/6.2).
SCORE_WEIGHTS = {
    "399006": {
        "predicted_return": 0.30,
        "probability_edge": 0.25,
        "trend_momentum": 0.20,
        "risk": 0.15,
        "valuation": 0.10,
    },
    "159941": {
        "predicted_return": 0.35,
        "probability_edge": 0.30,
        "trend_momentum": 0.20,
        "risk": 0.15,
        "valuation": 0.00,
    },
}
ETF_SCORE_WEIGHTS = {
    "predicted_return": 0.35,
    "probability_edge": 0.30,
    "trend_momentum": 0.20,
    "risk": 0.15,
    "valuation": 0.00,
}


def score_weights_for(market: str) -> Mapping[str, float]:
    """Return StrategyScore weights for an index or ETF market."""

    if market == "399006":
        return SCORE_WEIGHTS["399006"]
    return ETF_SCORE_WEIGHTS


def min_train_samples_for(market: str) -> int:
    """Return the initial-Champion label threshold for a market."""

    return MIN_TRAIN_SAMPLES_BY_MARKET.get(market, 24)

# Base target position mapping (V3.5 section 6.3 with the approved fix: the
# highest band maps to 80% so it is consistent with the global 80% cap).
SCORE_POSITION_MAP: tuple[tuple[float, int], ...] = (
    (-0.60, 0),
    (-0.35, 10),
    (-0.15, 20),
    (0.10, 30),
    (0.30, 45),
    (0.55, 60),
    (float("inf"), 80),
)
MAX_POSITION_PP = 80
MIN_CASH_PP = 20
POSITION_GRID_PP = 5

# Confirmation-state position caps (V3.5 section 7).
STATE_POSITION_CAPS: Mapping[str, int] = {
    "BEARISH_CONFIRMED": 10,
    "TOP_CONFIRMED": 10,
    "SIGNAL_CONFLICT": 20,
    "UNCONFIRMED": 30,
    "NEGATIVE_DIF_RISING": 30,
    "PRICE_BOTTOM_CONFIRMED": 40,
    "DIF_BOTTOM_CONFIRMED": 40,
    "PARTIALLY_CONFIRMED_BOTTOM": 50,
    "TEMPORALLY_CONSISTENT_BOTTOM": 65,
    "POSITIVE_DIF_RISING": 60,
    "TREND_CONFIRMED": 80,
}

# Trading execution defaults (V3.5 sections 8-10).
EXECUTION_DEFAULTS: Mapping[str, Any] = {
    "max_batches": 4,
    "max_single_batch_pp": 20,
    "default_first_batch_pp": 10,
    "weak_first_batch_pp": 5,
    "strong_first_batch_pp": 20,
    "trend_add_pp": 10,
    "strong_trend_add_pp": 20,
    "recovery_first_batch_pp": 5,
    "recovery_second_batch_pp": 10,
    "cooldown_trading_days": 5,
    "slippage_bps": 5.0,
    "buy_commission_rate": 0.00025,
    "sell_commission_rate": 0.00025,
    "minimum_commission": 5.0,
    "stamp_duty_rate": 0.0,
    "transfer_fee_rate": 0.0,
    "other_fee_rate": 0.0,
    "maximum_single_sell_pp": 20,
    "light_reduce_pp": 5,
    "confirmed_reduce_pp": 10,
    "emergency_reduce_pp": 20,
}

# Promotion gates (V3.5 section 17).
PROMOTION_DEFAULTS: Mapping[str, Any] = {
    "minimum_evaluation_windows": 8,
    "minimum_excess_return": 0.0030,
    "minimum_win_rate": 0.55,
    "maximum_drawdown_degradation_pp": 2.0,
    "no_action_window_ratio_cap": 0.60,
    "bull_market_min_position_pp": 10,
    "bull_market_benchmark_return": 0.03,
    "stagnation_rounds_before_audit": 3,
}

# Initial from-scratch MODEL_PACKAGE defaults (approved plan).
INITIAL_PACKAGE_DEFAULTS: Mapping[str, Any] = {
    "alpha": 1.0,
    "training_window_mode": "ROLLING_520W",
    "feature_set_name": "CORE",
    "daily_adjustment_mode": "DAILY_ADJUSTMENT_OFF",
    "residual_multiplier": 1.0,
    "calibration_mode": "AUTO",
}


def default_strategy_config(market: str) -> dict[str, Any]:
    """Return the immutable V3.5 default strategy package for a market."""

    if market == "399006":
        weights = SCORE_WEIGHTS["399006"]
    elif len(str(market)) == 6 and str(market).isdigit():
        weights = ETF_SCORE_WEIGHTS
    else:
        raise ValueError(f"unknown V3.5 market {market}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "market": market,
        "policy_version": POLICY_VERSION_DEFAULT,
        "score_weights": dict(weights),
        "score_position_map": [
            {"score_gt": lower, "position_pp": position}
            for lower, position in SCORE_POSITION_MAP
        ],
        "max_position_pp": MAX_POSITION_PP,
        "min_cash_pp": MIN_CASH_PP,
        "position_grid_pp": POSITION_GRID_PP,
        "state_position_caps": dict(STATE_POSITION_CAPS),
        "execution": dict(EXECUTION_DEFAULTS),
        "promotion": dict(PROMOTION_DEFAULTS),
    }
