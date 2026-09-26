"""V3.5.1 challenge-mechanism constants and effective candidate rules."""

from __future__ import annotations

from typing import Any, Mapping


PROTOCOL_VERSION_351 = "V3.5.1_EFFECTIVE_CHALLENGER_AND_DYNAMIC_ETF_SLOTS"

# Prediction challenger pool (V3.5.1 section 5).
ALPHA_MULTIPLIERS_351 = (0.25, 0.50, 2.00, 4.00)
ALPHA_MIN_351 = 0.01
ALPHA_MAX_351 = 200.0
MAX_PREDICTION_CANDIDATES = 5
STRUCTURAL_CHALLENGE_EVERY = 4

# Low-frequency structural dimensions (one per challenge).
STRUCTURAL_DIMENSIONS = (
    "training_window_mode",
    "feature_set",
    "daily_adjustment_mode",
    "residual_scale",
)
TRAINING_WINDOW_VALUES = ("ROLLING_260W", "ROLLING_520W", "EXPANDING_AVAILABLE_HISTORY")
FEATURE_SET_VALUES = ("CORE_FEATURE_SET", "EXTENDED_FEATURE_SET")
DAILY_ADJUSTMENT_VALUES = (
    "DAILY_ADJUSTMENT_OFF",
    "DAILY_ADJUSTMENT_FIXED",
    "DAILY_ADJUSTMENT_REGIME_GATED",
)
RESIDUAL_SCALE_VALUES = (0.60, 0.80, 1.00, 1.20)

# Behavior divergence pre-filter thresholds.
DIVERGENCE_THRESHOLDS: Mapping[str, float] = {
    "forecast_min_abs_diff": 0.005,
    "forecast_divergence_ratio": 0.10,
    "strategy_score_ratio": 0.10,
    "base_position_ratio": 0.10,
    "target_position_ratio": 0.05,
    "trade_path_ratio": 0.03,
}
EQUITY_TOLERANCE_CNY = 0.01
EQUITY_PATH_REL_TOLERANCE = 1e-9

# MODEL_MAINTENANCE_REFRESH cadence and gates.
MAINTENANCE_EVERY_MATURED = 8
MAINTENANCE_GATES: Mapping[str, Any] = {
    "minimum_excess_return": -0.0005,
    "maximum_drawdown_degradation_pp": 0.5,
    "brier_degradation_max": 0.02,
    "mae_degradation_ratio_max": 1.15,
}

# STRONG_PROMOTION (same as V3.5 plus real trade divergence + quality line).
STRONG_PROMOTION_GATES: Mapping[str, Any] = {
    "minimum_evaluation_windows": 8,
    "minimum_excess_return": 0.0030,
    "minimum_win_rate": 0.55,
    "maximum_drawdown_degradation_pp": 2.0,
    "no_action_window_ratio_cap": 0.60,
    "bull_market_min_position_pp": 10.0,
    "minimum_trade_path_divergence_ratio": 0.0,
}

# STABLE_SMALL_EDGE_PROMOTION.
STABLE_SMALL_EDGE_GATES: Mapping[str, Any] = {
    "minimum_evaluation_windows": 16,
    "minimum_effective_independent_windows": 8,
    "minimum_excess_return": 0.0015,
    "maximum_excess_return": 0.0030,
    "minimum_win_rate": 0.58,
    "maximum_drawdown_degradation_pp": 0.5,
    "minimum_market_states": 3,
    "minimum_trade_path_divergence_ratio": 0.03,
    "no_action_window_ratio_cap": 0.60,
    "bull_market_min_position_pp": 10.0,
}
MINIMUM_PROMOTABLE_EXCESS_RETURN = 0.0015

# Rejection reason codes (V3.5.1 section 10).
REJECTION_REASONS = (
    "EFFECTIVELY_IDENTICAL",
    "NOT_SELECTED_INNER",
    "INSUFFICIENT_BEHAVIOR_DIVERGENCE",
    "INSUFFICIENT_EFFECTIVE_WINDOWS",
    "EXCESS_RETURN_BELOW_0_15",
    "EXCESS_RETURN_BELOW_0_30",
    "WIN_RATE_BELOW_THRESHOLD",
    "MEDIAN_PROFIT_GATE_FAILED",
    "DRAWDOWN_GATE_FAILED",
    "UP_MARKET_PARTICIPATION_FAILED",
    "NO_ACTION_DEPENDENCY",
    "PREDICTION_QUALITY_GATE_FAILED",
    "MARKET_REGIME_STABILITY_FAILED",
    "TRANSACTION_COST_EDGE_ERASED",
    "DATA_LEAKAGE_DETECTED",
    "INVALID_EXECUTION_PRICE",
)

# Strategy challenger parameter groups and allowed values.
STRATEGY_PARAMETER_GROUPS: Mapping[str, tuple[Any, ...]] = {
    "state_cap_UNCONFIRMED": (20, 30, 40, 50),
    "state_cap_NEGATIVE_DIF_RISING": (20, 30, 40),
    "state_cap_POSITIVE_DIF_RISING": (50, 60, 70),
    "state_cap_TREND_CONFIRMED": (70, 80),
    "first_probe_buy": (5, 10, 15),
    "trend_add": (5, 10, 15),
    "strong_trend_add": (10, 15, 20),
    "trend_confirmation_weeks": (1, 2, 3),
    "cooldown_days": (5, 10, 15),
    "light_reduce": (5, 10),
    "confirmed_reduce": (10, 15, 20),
}
BINDING_PARAMETER_KEYS = tuple(STRATEGY_PARAMETER_GROUPS)
MAX_STRATEGY_PARAMETER_GROUP_TRIES = 3

# Effective independent window threshold used for promotion channels.
MIN_INDEPENDENT_WINDOWS_STRONG = 8


def promotion_channel_for(
    *,
    raw_windows: int,
    effective_windows: int,
    mean_excess: float,
    median_profit_excess: float,
    win_rate: float,
    drawdown_degradation: float,
    participation: float,
    champion_participation: float,
    no_action_ratio: float,
    trade_path_divergence_ratio: float,
    quality_ok: bool,
) -> tuple[str | None, list[str], dict[str, float]]:
    """Return (channel, reason codes, gaps) for the two legal channels."""

    strong = STRONG_PROMOTION_GATES
    small = STABLE_SMALL_EDGE_GATES
    reason_codes: list[str] = []
    gaps: dict[str, float] = {}
    if not quality_ok:
        reason_codes.append("PREDICTION_QUALITY_GATE_FAILED")
    if raw_windows < int(strong["minimum_evaluation_windows"]):
        reason_codes.append("INSUFFICIENT_EFFECTIVE_WINDOWS")
        gaps["effective_window_gap"] = float(
            int(strong["minimum_evaluation_windows"]) - raw_windows
        )
    if mean_excess < MINIMUM_PROMOTABLE_EXCESS_RETURN:
        reason_codes.append("EXCESS_RETURN_BELOW_0_15")
        gaps["excess_return_gap"] = float(MINIMUM_PROMOTABLE_EXCESS_RETURN - mean_excess)
    elif mean_excess < float(strong["minimum_excess_return"]):
        reason_codes.append("EXCESS_RETURN_BELOW_0_30")
        gaps["excess_return_gap"] = float(strong["minimum_excess_return"] - mean_excess)
    if median_profit_excess <= 0.0:
        reason_codes.append("MEDIAN_PROFIT_GATE_FAILED")
        gaps["median_profit_gap"] = float(-median_profit_excess)
    if win_rate < float(strong["minimum_win_rate"]):
        reason_codes.append("WIN_RATE_BELOW_THRESHOLD")
        gaps["win_rate_gap"] = float(strong["minimum_win_rate"] - win_rate)
    if drawdown_degradation > float(strong["maximum_drawdown_degradation_pp"]) / 100.0:
        reason_codes.append("DRAWDOWN_GATE_FAILED")
        gaps["drawdown_gap"] = float(
            drawdown_degradation
            - float(strong["maximum_drawdown_degradation_pp"]) / 100.0
        )
    if participation < float(strong["bull_market_min_position_pp"]) / 100.0:
        reason_codes.append("NO_ACTION_DEPENDENCY")
        gaps["participation_gap"] = float(
            float(strong["bull_market_min_position_pp"]) / 100.0 - participation
        )
    if no_action_ratio > float(strong["no_action_window_ratio_cap"]):
        if "NO_ACTION_DEPENDENCY" not in reason_codes:
            reason_codes.append("NO_ACTION_DEPENDENCY")
    if trade_path_divergence_ratio < float(
        small["minimum_trade_path_divergence_ratio"]
    ):
        reason_codes.append("INSUFFICIENT_BEHAVIOR_DIVERGENCE")

    strong_ok = (
        quality_ok
        and raw_windows >= int(strong["minimum_evaluation_windows"])
        and mean_excess >= float(strong["minimum_excess_return"])
        and median_profit_excess > 0.0
        and win_rate >= float(strong["minimum_win_rate"])
        and drawdown_degradation
        <= float(strong["maximum_drawdown_degradation_pp"]) / 100.0
        and participation >= float(strong["bull_market_min_position_pp"]) / 100.0
        and no_action_ratio <= float(strong["no_action_window_ratio_cap"])
    )
    if strong_ok:
        return "STRONG_PROMOTION", [], {}
    small_ok = (
        quality_ok
        and raw_windows >= int(small["minimum_evaluation_windows"])
        and effective_windows >= int(small["minimum_effective_independent_windows"])
        and float(small["minimum_excess_return"])
        <= mean_excess
        < float(small["maximum_excess_return"])
        and median_profit_excess > 0.0
        and win_rate >= float(small["minimum_win_rate"])
        and drawdown_degradation
        <= float(small["maximum_drawdown_degradation_pp"]) / 100.0
        and participation >= champion_participation - 0.05
        and no_action_ratio <= float(small["no_action_window_ratio_cap"])
        and trade_path_divergence_ratio
        >= float(small["minimum_trade_path_divergence_ratio"])
    )
    if small_ok:
        return "STABLE_SMALL_EDGE_PROMOTION", [], {}
    if not reason_codes:
        reason_codes = ["EXCESS_RETURN_BELOW_0_30"]
    return None, reason_codes, gaps
