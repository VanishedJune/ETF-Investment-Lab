"""V3.7 protocol constants: multi-timeframe, DeepSeek AI, fusion and repair."""

from __future__ import annotations

from typing import Any, Mapping


PROTOCOL_VERSION_37 = "V3.7_DEEPSEEK_FUSION_8W"

HORIZON_WEEKS = 8
FEATURE_VERSION = "V3.7_FEATURE_MANIFEST_1"
PROMOTION_RULE_VERSION = "PROMOTION_RULE_V37_1"
POLICY_VERSION_DEFAULT = "POLICY_V37_1"

# Feature-set names for the three local architectures.
FEATURE_SET_WEEKLY_ONLY = "WEEKLY_ONLY"
FEATURE_SET_EARLY_FUSION = "EARLY_FUSION"
FEATURE_SET_LATE_FUSION = "LATE_FUSION"

# Initial challenger fusion weights (1-2W / 3-4W / 5-8W = daily/weekly).
LATE_FUSION_DAILY_WEIGHTS = (0.40, 0.30, 0.20)
LATE_FUSION_WEEKLY_WEIGHTS = (0.60, 0.70, 0.80)

# Weekly branch depth.
WEEKLY_BARS_MIN = 52
WEEKLY_BARS_TARGET = 104
DAILY_BARS_MIN = 60
DAILY_BARS_TARGET = 100

# MULTI_TIMEFRAME_STATE values.
MULTI_TIMEFRAME_STATES = (
    "BOTH_BULLISH",
    "BOTH_BEARISH",
    "DAILY_BULLISH_WEEKLY_BEARISH",
    "DAILY_BEARISH_WEEKLY_BULLISH",
    "DAILY_RECOVERY_WEEKLY_UNCONFIRMED",
    "WEEKLY_UPTREND_DAILY_PULLBACK",
    "NEUTRAL_MIXED",
)

# AI packet versions.
AI_PACKET_SCHEMA_VERSION = "MULTI_TIMEFRAME_AI_PACKET_V1"
AI_PROMPT_VERSION_INDEPENDENT = "V37_INDEPENDENT_ANALYST_1"
AI_PROMPT_VERSION_REVIEWER = "V37_MODEL_REVIEWER_1"
AI_PROVIDER = "deepseek"

# Strict AI JSON field enums.
AI_TREND_VALUES = {"BULLISH", "NEUTRAL_BULLISH", "NEUTRAL", "NEUTRAL_BEARISH", "BEARISH"}
AI_DIRECTION_TREND_VALUES = {"BULLISH", "NEUTRAL", "BEARISH"}
AI_RISK_VALUES = {"LOW", "MEDIUM", "HIGH"}

# AI calibration thresholds (matured Forward-OOS samples).
AI_CALIBRATION_UNAVAILABLE_SAMPLES = 30
AI_CALIBRATION_PRELIMINARY_SAMPLES = 50

# AI weight tiers by matured Forward-OOS windows (P7, corrected 2026-08-07):
# <30 -> AI_SHADOW (0%); 30-49 -> max 10%; 50-99 -> challengers up to 30%;
# >=100 -> max 40% after the OOS/returns/drawdown/activity gates pass.
AI_WEIGHT_TIERS: tuple[tuple[int, int], ...] = (
    (30, 10),
    (50, 30),
    (100, 40),
)
AI_WEIGHT_SHADOW_WINDOWS = 30
AI_WEIGHT_MAX = 40

# Fusion initial challenger candidates (local, ai).
FUSION_INITIAL_CANDIDATES: tuple[tuple[int, int], ...] = (
    (90, 10),
    (80, 20),
    (70, 30),
    (60, 40),
)

# Model-conflict / consensus states.
CONSENSUS_STATES = (
    "CONSENSUS_STRONG_BULL",
    "CONSENSUS_BULL",
    "CONSENSUS_NEUTRAL",
    "CONSENSUS_BEAR",
    "CONSENSUS_STRONG_BEAR",
)
CONFLICT_STATES = (
    "NONE",
    "LOCAL_BULL_AI_BEAR",
    "LOCAL_BEAR_AI_BULL",
    "SHORT_LONG_TIMEFRAME_CONFLICT",
    "HIGH_MODEL_CONFLICT",
)
CONFLICT_POLICIES = ("NONE", "REDUCE_5PP", "REDUCE_10PP", "WAIT_CONFIRMATION")

# Repair proposal whitelist (P9 / plan section 18; 14 items, not 13).
REPAIR_PARAMETER_WHITELIST = (
    "ridge_alpha",
    "training_window",
    "feature_subset",
    "daily_feature_set",
    "weekly_feature_set",
    "daily_weekly_fusion_weight",
    "residual_half_life",
    "calibration_mode",
    "model_family",
    "strong_signal_threshold",
    "entry_floor",
    "vol_target",
    "conflict_penalty",
    "risk_gate_parameter",
)

# A repair challenger may train/sanity-check on history up to the proposal
# anchor, but formal promotion evidence must come only from matured OOS
# windows that start at or after proposal_anchor + purge gap.
REPAIR_PROMOTION_PURGE_WEEKS = 8

# AI screening scopes: historical anonymized research is never allowed to
# feed formal calibration, weight unlocks or promotion.
AI_SCOPE_HISTORICAL_SCREENING = "HISTORICAL_SCREENING"
AI_SCOPE_FORWARD_OOS = "FORWARD_OOS"

# Transient failure codes that permit bounded retries (failed request rows
# are always retained and never overwritten or disguised as success).
AI_RETRYABLE_ERROR_CODES = (
    "NETWORK_ERROR",
    "TIMEOUT",
    "RATE_LIMITED",
    "MODEL_NOT_AVAILABLE",
)
AI_MAX_ATTEMPTS = 3

# Repair parameter validation ranges (only validated values are allowed).
REPAIR_PARAMETER_RANGES: Mapping[str, tuple[float, float]] = {
    "ridge_alpha": (0.01, 100.0),
    "training_window": (260.0, 1040.0),
    "residual_half_life": (10.0, 200.0),
    "daily_weekly_fusion_weight": (0.0, 0.60),
    "strong_signal_threshold": (0.50, 0.80),
    "entry_floor": (0.0, 0.20),
    "vol_target": (0.05, 0.25),
    "conflict_penalty": (0.0, 0.20),
    "risk_gate_parameter": (0.0, 1.0),
}

# Reviewer cadence (matured samples).
MODEL_REVIEW_EVERY_MATURED = 8

# AI request / fallback statuses.
AI_RESPONSE_OK = "OK"
AI_RESPONSE_INVALID = "AI_INVALID_RESPONSE"
AI_RESPONSE_API_UNAVAILABLE = "AI_API_UNAVAILABLE"
AI_RESPONSE_TIMEOUT = "AI_API_TIMEOUT"
AI_RESPONSE_CACHED = "CACHED"
AI_CALIBRATION_UNAVAILABLE = "AI_CALIBRATION_UNAVAILABLE"
AI_CALIBRATION_PRELIMINARY = "AI_PRELIMINARY_CALIBRATION"
AI_CALIBRATION_FULL = "AI_FULL_CALIBRATION"
AI_WEIGHT_DEGRADED = "AI_WEIGHT_DEGRADED"
AI_FALLBACK_LOCAL_ONLY = "AI_FALLBACK_LOCAL_ONLY"


def ai_weight_cap(forward_oos_windows: int) -> int:
    """Return the maximum AI fusion weight in pp for a matured window count.

    Corrected tiers (2026-08-07): <30 shadow/0%, 30-49 max 10%, 50-99 max
    30% (challenger-only), >=100 max 40% after verification gates.  The cap
    is an upper bound; actual weights still require OOS/returns/drawdown/
    activity gates to pass.
    """

    if forward_oos_windows < AI_WEIGHT_SHADOW_WINDOWS:
        return 0
    cap = 0
    for threshold, weight in AI_WEIGHT_TIERS:
        if forward_oos_windows >= threshold:
            cap = weight
    return min(cap, AI_WEIGHT_MAX)
