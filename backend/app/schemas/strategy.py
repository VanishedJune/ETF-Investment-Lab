"""Validated, versioned configuration for deterministic ETF rule strategies."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrategyWeights(BaseModel):
    """Relative weights. The engine normalizes the available positive weights."""

    model_config = ConfigDict(extra="forbid")

    valuation: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    trend: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    momentum: Decimal = Field(default=Decimal("0.15"), ge=0, le=1)
    volume: Decimal = Field(default=Decimal("0.10"), ge=0, le=1)
    volatility: Decimal = Field(default=Decimal("0.10"), ge=0, le=1)
    risk: Decimal = Field(default=Decimal("0.15"), ge=0, le=1)

    @model_validator(mode="after")
    def has_positive_total(self) -> "StrategyWeights":
        if sum(self.model_dump().values(), Decimal("0")) <= 0:
            raise ValueError("At least one strategy weight must be positive")
        return self


class StrategyThresholds(BaseModel):
    """Named thresholds used by the fixed, inspectable scoring formulas."""

    model_config = ConfigDict(extra="forbid")

    valuation_low_percentile: Decimal = Field(default=Decimal("0.30"), ge=0, le=1)
    valuation_high_percentile: Decimal = Field(default=Decimal("0.70"), ge=0, le=1)
    momentum_rsi_low: Decimal = Field(default=Decimal("35"), ge=0, le=100)
    momentum_rsi_high: Decimal = Field(default=Decimal("70"), ge=0, le=100)
    volume_ratio_low: Decimal = Field(default=Decimal("0.80"), ge=0)
    volume_ratio_high: Decimal = Field(default=Decimal("1.20"), ge=0)
    volatility_high: Decimal = Field(default=Decimal("0.35"), gt=0)
    drawdown_pause: Decimal = Field(default=Decimal("-0.20"), ge=-1, le=0)
    increase_score: Decimal = Field(default=Decimal("0.35"), ge=-1, le=1)
    reduce_score: Decimal = Field(default=Decimal("-0.20"), ge=-1, le=1)
    sell_partial_score: Decimal = Field(default=Decimal("-0.45"), ge=-1, le=1)
    pause_score: Decimal = Field(default=Decimal("-0.75"), ge=-1, le=1)

    @model_validator(mode="after")
    def has_ordered_thresholds(self) -> "StrategyThresholds":
        if self.valuation_low_percentile >= self.valuation_high_percentile:
            raise ValueError("valuation_low_percentile must be below valuation_high_percentile")
        if self.momentum_rsi_low >= self.momentum_rsi_high:
            raise ValueError("momentum_rsi_low must be below momentum_rsi_high")
        if self.volume_ratio_low > self.volume_ratio_high:
            raise ValueError("volume_ratio_low must not exceed volume_ratio_high")
        if not (
            self.pause_score <= self.sell_partial_score <= self.reduce_score < self.increase_score
        ):
            raise ValueError("recommendation score thresholds are unsafe or unordered")
        return self


class StrategyMultipliers(BaseModel):
    """Named non-negative research-plan multipliers by recommendation state."""

    model_config = ConfigDict(extra="forbid")

    increase: Decimal = Field(default=Decimal("1.30"), ge=0, le=3)
    normal: Decimal = Field(default=Decimal("1.00"), ge=0, le=3)
    reduce: Decimal = Field(default=Decimal("0.75"), ge=0, le=3)
    pause: Decimal = Field(default=Decimal("0.50"), ge=0, le=3)
    hold: Decimal = Field(default=Decimal("1.00"), ge=0, le=3)
    sell_partial: Decimal = Field(default=Decimal("0.70"), ge=0, le=3)


class StrategyConfig(BaseModel):
    """A complete, safe rule configuration stored with a strategy definition."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1.0", min_length=1, max_length=32)
    weights: StrategyWeights = Field(default_factory=StrategyWeights)
    thresholds: StrategyThresholds = Field(default_factory=StrategyThresholds)
    multipliers: StrategyMultipliers = Field(default_factory=StrategyMultipliers)
    maximum_sell_ratio: Decimal = Field(default=Decimal("0.30"), ge=0, le=Decimal("0.30"))
    minimum_holding_ratio: Decimal = Field(default=Decimal("0.20"), ge=Decimal("0.20"), le=1)


def strategy_config_from_mapping(
    values: Mapping[str, Any] | None,
    *,
    strategy_version: str | None = None,
) -> StrategyConfig:
    """Load modern config while safely mapping the shipped legacy field names.

    Existing local strategy rows used ``technical`` and ``max_single_sale_ratio``.
    They are read compatibly but never widened beyond the current safety limits.
    """
    raw: dict[str, Any] = deepcopy(dict(values or {}))
    weights = dict(raw.get("weights") or {})
    if "technical" in weights and "momentum" not in weights:
        weights["momentum"] = weights.pop("technical")
    else:
        weights.pop("technical", None)
    raw["weights"] = weights
    thresholds = dict(raw.get("thresholds") or {})
    thresholds.pop("buy_signal", None)
    thresholds.pop("reduce_signal", None)
    raw["thresholds"] = thresholds
    multipliers = dict(raw.get("multipliers") or {})
    if "combined" in multipliers and "increase" not in multipliers:
        multipliers["increase"] = multipliers["combined"]
    for legacy_key in ("valuation", "trend", "combined"):
        multipliers.pop(legacy_key, None)
    raw["multipliers"] = multipliers
    if "max_single_sale_ratio" in raw and "maximum_sell_ratio" not in raw:
        raw["maximum_sell_ratio"] = raw.pop("max_single_sale_ratio")
    if strategy_version is not None:
        configured_version = raw.get("version")
        if configured_version is not None and configured_version != strategy_version:
            raise ValueError("Strategy definition version must match its configuration version")
        raw["version"] = strategy_version
    return StrategyConfig.model_validate(raw)
