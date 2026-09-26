"""V3.6 protocol constants: activity health, gates and challenger dimensions."""

from __future__ import annotations

from typing import Any, Mapping


PROTOCOL_VERSION_36 = "V3.6_CAPITAL_DRIVEN_8W"

# Prediction alpha challengers (V3.6 P4). 1.00 belongs to maintenance only.
V36_ALPHA_MULTIPLIERS = (0.10, 0.25, 0.50, 1.50, 2.00, 4.00)
V36_ALPHA_MIN = 0.01
V36_ALPHA_MAX = 200.0

PREDICTION_CHALLENGE_EVERY_MATURED_36 = 4
STRATEGY_CHALLENGE_EVERY_MATURED_36 = 8
MAINTENANCE_EVERY_MATURED_36 = 8

RESIDUAL_SCALE_VALUES_36 = (0.60, 0.80, 1.00, 1.20)
RESIDUAL_WEIGHTING_VALUES_36 = ("UNIFORM", "EXP_DECAY")
CALIBRATION_MODE_VALUES_36 = ("TEMPERATURE", "PLATT", "ISOTONIC")
MODEL_FAMILY_VALUES_36 = ("RIDGE", "KERNEL_RIDGE")
REGIME_CORRECTION_VALUES_36 = ("OFF", "ON")

# Residual time weighting (P6).
RESIDUAL_HALF_LIFE_WEEKS = 50

# Calibration sample thresholds (P6).
CALIBRATION_UNAVAILABLE_SAMPLES = 50
CALIBRATION_PLATT_SAMPLES = 50
CALIBRATION_ISOTONIC_SAMPLES = 100

# Maintenance gates (P3).
MAINTENANCE_GATES_36: Mapping[str, Any] = {
    "mae_degradation_ratio_max": 1.02,
    "brier_degradation_max": 0.01,
    "maximum_drawdown_degradation_pp": 0.5,
    "minimum_excess_return": -0.0005,
    "minimum_mae_improvement": 0.01,
}

# Promotion gates (P5): activity health is a flag, not a forced-trade floor.
STRONG_PROMOTION_GATES_36: Mapping[str, Any] = {
    "minimum_evaluation_windows": 8,
    "minimum_excess_return": 0.0030,
    "minimum_win_rate": 0.55,
    "maximum_drawdown_degradation_pp": 2.0,
    "no_action_window_ratio_cap": 0.60,
    "bull_market_min_position_pp": 10.0,
    "minimum_trade_path_divergence_ratio": 0.0,
    "activity_health_not_worse": True,
}

STABLE_SMALL_EDGE_GATES_36: Mapping[str, Any] = {
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
    "activity_health_not_worse": True,
}
MINIMUM_PROMOTABLE_EXCESS_RETURN_36 = 0.0015

# Activity health (P2).
STRONG_UP_WINDOW_THRESHOLD = {"399006": 0.05, "ETF": 0.08}
ACTIVITY_LOOKBACK_WEEKS = 26
ACTIVITY_DEGRADED_MIN_STRONG_UP_WINDOWS = 3
ACTIVITY_DEGRADED_MAX_AVG_EXPOSURE_PP = 10.0

# Strategy challenger parameter groups (P7 additions).
STRATEGY_PARAMETER_GROUPS_36: Mapping[str, tuple[Any, ...]] = {
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
    "strong_signal_entry_floor_pp": (5, 10, 15),
    "vol_target_enabled": (False, True),
    "target_vol": (10.0, 15.0),
}

STRONG_SIGNAL_UP_PROB = 0.60
VOL_TARGET_CLAMP = (0.75, 1.25)

# Regime correction (P6): applied only when the challenger config enables it.
REGIME_CORRECTION_WEEKS = 4
REGIME_CORRECTION_SIGNAL_SCALE = 0.10
REGIME_CORRECTION_STRENGTH_DENOM = 0.08
REGIME_CORRECTION_MAX_PER_WEEK = 0.02


def strong_up_threshold_for(market: str) -> float:
    if market == "399006":
        return float(STRONG_UP_WINDOW_THRESHOLD["399006"])
    return float(STRONG_UP_WINDOW_THRESHOLD["ETF"])


def structural_dimensions_36() -> tuple[str, ...]:
    return (
        "training_window",
        "feature_set",
        "daily_adjustment_mode",
        "residual_weighting",
        "calibration_mode",
        "model_family",
        "regime_correction",
    )


def promotion_channel_for_36(
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
    activity_health_not_worse: bool,
) -> tuple[str | None, list[str], dict[str, float]]:
    """V3.6 promotion channels (P5): activity health is a flag, not forced trade."""

    strong = STRONG_PROMOTION_GATES_36
    small = STABLE_SMALL_EDGE_GATES_36
    reason_codes: list[str] = []
    gaps: dict[str, float] = {}
    if not quality_ok:
        reason_codes.append("PREDICTION_QUALITY_GATE_FAILED")
    if raw_windows < int(strong["minimum_evaluation_windows"]):
        reason_codes.append("INSUFFICIENT_EFFECTIVE_WINDOWS")
        gaps["effective_window_gap"] = float(
            int(strong["minimum_evaluation_windows"]) - raw_windows
        )
    if mean_excess < MINIMUM_PROMOTABLE_EXCESS_RETURN_36:
        reason_codes.append("EXCESS_RETURN_BELOW_0_15")
        gaps["excess_return_gap"] = float(
            MINIMUM_PROMOTABLE_EXCESS_RETURN_36 - mean_excess
        )
    elif mean_excess < float(strong["minimum_excess_return"]):
        reason_codes.append("EXCESS_RETURN_BELOW_0_30")
        gaps["excess_return_gap"] = float(
            strong["minimum_excess_return"] - mean_excess
        )
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
    if not activity_health_not_worse:
        reason_codes.append("ACTIVITY_HEALTH_GATE_FAILED")
        gaps["activity_health_gap"] = 1.0

    strong_ok = (
        quality_ok
        and activity_health_not_worse
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
        and activity_health_not_worse
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
