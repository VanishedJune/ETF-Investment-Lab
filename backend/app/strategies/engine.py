"""Explicit deterministic formulas for research-only ETF strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Mapping

from ..schemas.strategy import StrategyConfig


RECOMMENDATIONS = frozenset({"INCREASE", "NORMAL", "REDUCE", "PAUSE", "HOLD", "SELL_PARTIAL"})
_EIGHT_DP = Decimal("0.00000001")
MINIMUM_ACTION_CONFIDENCE = Decimal("0.50")


@dataclass(frozen=True, slots=True)
class StrategySnapshot:
    """Stored observations available no later than the requested research date."""

    instrument_code: str
    as_of_date: date
    data_cutoff: date | None
    close_price: Decimal | None
    volume: Decimal | None
    valuation: Mapping[str, Decimal | None]
    indicators: Mapping[str, Decimal | None]
    observations: Mapping[str, object]
    invalid_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyEvaluation:
    score: Decimal
    confidence: Decimal
    multiplier: Decimal
    recommendation: str
    suggested_sell_ratio: Decimal
    component_scores: Mapping[str, Decimal | None]
    reasons: tuple[str, ...]
    triggered_rules: tuple[str, ...]
    reverse_risks: tuple[str, ...]


def _decimal(value: Decimal | None) -> str:
    return "missing" if value is None else format(value, "f")


def _rounded(value: Decimal) -> Decimal:
    return value.quantize(_EIGHT_DP, rounding=ROUND_HALF_EVEN)


def _is_finite(value: Decimal | None) -> bool:
    return value is not None and value.is_finite()


class RuleStrategyEngine:
    """Score only the supplied stored snapshot; it never fetches or trades."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def evaluate(
        self,
        snapshot: StrategySnapshot,
        *,
        holding_ratio: Decimal | None = None,
    ) -> StrategyEvaluation:
        """Apply named thresholds and weights with reproducible numeric explanations."""
        reasons: list[str] = []
        rules: list[str] = []
        risks: list[str] = []
        for field in snapshot.invalid_fields:
            reasons.append(f"invalid:{field}")
            risks.append(f"invalid_input:{field}")
        components = {
            "valuation": self._valuation(snapshot, reasons, rules, risks),
            "trend": self._trend(snapshot, reasons, rules, risks),
            "momentum": self._momentum(snapshot, reasons, rules, risks),
            "volume": self._volume(snapshot, reasons, rules, risks),
            "volatility": self._volatility(snapshot, reasons, rules, risks),
            "risk": self._risk(snapshot, reasons, rules, risks),
        }
        score = self._weighted_score(components)
        confidence = self._confidence(snapshot, components)
        recommendation = self._recommendation(score, confidence, holding_ratio)
        if confidence < MINIMUM_ACTION_CONFIDENCE:
            rules.append("insufficient_data_hold")
            reasons.append(
                "confidence: "
                f"{_decimal(confidence)} < minimum_action_confidence="
                f"{_decimal(MINIMUM_ACTION_CONFIDENCE)}; recommendation=HOLD"
            )
        multiplier = getattr(self.config.multipliers, recommendation.lower())
        suggested_sell = self._suggested_sell_ratio(recommendation, holding_ratio)
        if recommendation == "SELL_PARTIAL":
            rules.append("sell_partial_safety_cap")
            reasons.append(
                "sell: suggested_sell_ratio="
                f"{_decimal(suggested_sell)} <= maximum_sell_ratio="
                f"{_decimal(self.config.maximum_sell_ratio)}"
            )
        return StrategyEvaluation(
            score=_rounded(score),
            confidence=_rounded(confidence),
            multiplier=_rounded(multiplier),
            recommendation=recommendation,
            suggested_sell_ratio=_rounded(suggested_sell),
            component_scores={name: None if value is None else _rounded(value) for name, value in components.items()},
            reasons=tuple(reasons),
            triggered_rules=tuple(rules),
            reverse_risks=tuple(risks),
        )

    def _valuation(
        self,
        snapshot: StrategySnapshot,
        reasons: list[str],
        rules: list[str],
        risks: list[str],
    ) -> Decimal | None:
        value = snapshot.valuation.get("valuation_percentile")
        if not _is_finite(value):
            if value is not None:
                reasons.append("invalid:valuation:valuation_percentile:non_finite")
            reasons.append("missing:valuation:valuation_percentile")
            risks.append("valuation_percentile_missing")
            return None
        thresholds = self.config.thresholds
        if value <= thresholds.valuation_low_percentile:
            rules.append("valuation_low_percentile")
            reasons.append(
                "valuation: valuation_percentile="
                f"{_decimal(value)} <= valuation_low_percentile="
                f"{_decimal(thresholds.valuation_low_percentile)}; score=1"
            )
            return Decimal("1")
        if value >= thresholds.valuation_high_percentile:
            rules.append("valuation_high_percentile")
            risks.append("valuation_high_percentile")
            reasons.append(
                "valuation: valuation_percentile="
                f"{_decimal(value)} >= valuation_high_percentile="
                f"{_decimal(thresholds.valuation_high_percentile)}; score=-1"
            )
            return Decimal("-1")
        reasons.append(f"valuation: valuation_percentile={_decimal(value)}; score=0")
        return Decimal("0")

    def _trend(
        self,
        snapshot: StrategySnapshot,
        reasons: list[str],
        rules: list[str],
        risks: list[str],
    ) -> Decimal | None:
        close = snapshot.close_price
        ma_20 = snapshot.indicators.get("ma_20")
        ma_60 = snapshot.indicators.get("ma_60")
        if not all(_is_finite(value) for value in (close, ma_20, ma_60)):
            for name, value in (("close_price", close), ("ma_20", ma_20), ("ma_60", ma_60)):
                if not _is_finite(value):
                    if value is not None:
                        reasons.append(f"invalid:trend:{name}:non_finite")
                    reasons.append(f"missing:trend:{name}")
            risks.append("trend_data_missing")
            return None
        close_score = Decimal("1") if close >= ma_20 else Decimal("-1")
        average_score = Decimal("1") if ma_20 >= ma_60 else Decimal("-1")
        if close_score > 0:
            rules.append("trend_close_above_ma_20")
        else:
            risks.append("trend_close_below_ma_20")
        if average_score > 0:
            rules.append("trend_ma_20_above_ma_60")
        else:
            risks.append("trend_ma_20_below_ma_60")
        score = (close_score + average_score) / Decimal("2")
        reasons.append(
            "trend: close_price="
            f"{_decimal(close)}, ma_20={_decimal(ma_20)}, ma_60={_decimal(ma_60)}; score={_decimal(score)}"
        )
        return score

    def _momentum(
        self,
        snapshot: StrategySnapshot,
        reasons: list[str],
        rules: list[str],
        risks: list[str],
    ) -> Decimal | None:
        rsi = snapshot.indicators.get("rsi_6")
        histogram = snapshot.indicators.get("macd_histogram")
        if not _is_finite(rsi) or not _is_finite(histogram):
            if not _is_finite(rsi):
                if rsi is not None:
                    reasons.append("invalid:momentum:rsi_6:non_finite")
                reasons.append("missing:momentum:rsi_6")
            if not _is_finite(histogram):
                if histogram is not None:
                    reasons.append("invalid:momentum:macd_histogram:non_finite")
                reasons.append("missing:momentum:macd_histogram")
            risks.append("momentum_data_missing")
            return None
        thresholds = self.config.thresholds
        rsi_score = Decimal("1") if rsi <= thresholds.momentum_rsi_low else Decimal("-1") if rsi >= thresholds.momentum_rsi_high else Decimal("0")
        macd_score = Decimal("1") if histogram >= 0 else Decimal("-1")
        if macd_score > 0:
            rules.append("momentum_macd_nonnegative")
        else:
            risks.append("momentum_macd_negative")
        if rsi_score < 0:
            risks.append("momentum_rsi_high")
        score = (rsi_score + macd_score) / Decimal("2")
        reasons.append(
            "momentum: rsi_6="
            f"{_decimal(rsi)}, macd_histogram={_decimal(histogram)}; score={_decimal(score)}"
        )
        return score

    def _volume(
        self,
        snapshot: StrategySnapshot,
        reasons: list[str],
        rules: list[str],
        risks: list[str],
    ) -> Decimal | None:
        volume = snapshot.volume
        average = snapshot.indicators.get("volume_ma_20")
        if not _is_finite(volume) or not _is_finite(average) or average <= 0:
            if not _is_finite(volume):
                if volume is not None:
                    reasons.append("invalid:volume:volume:non_finite")
                reasons.append("missing:volume:volume")
            if not _is_finite(average) or average <= 0:
                if average is not None and not average.is_finite():
                    reasons.append("invalid:volume:volume_ma_20:non_finite")
                reasons.append("missing:volume:volume_ma_20")
            risks.append("volume_data_missing")
            return None
        ratio = volume / average
        thresholds = self.config.thresholds
        if ratio >= thresholds.volume_ratio_high:
            rules.append("volume_above_average")
            score = Decimal("1")
        elif ratio <= thresholds.volume_ratio_low:
            risks.append("volume_below_average")
            score = Decimal("-1")
        else:
            score = Decimal("0")
        reasons.append(f"volume: volume_ratio={_decimal(ratio)}; score={_decimal(score)}")
        return score

    def _volatility(
        self,
        snapshot: StrategySnapshot,
        reasons: list[str],
        rules: list[str],
        risks: list[str],
    ) -> Decimal | None:
        value = snapshot.indicators.get("volatility_20")
        if not _is_finite(value):
            if value is not None:
                reasons.append("invalid:volatility:volatility_20:non_finite")
            reasons.append("missing:volatility:volatility_20")
            risks.append("volatility_data_missing")
            return None
        threshold = self.config.thresholds.volatility_high
        if value <= threshold:
            rules.append("volatility_within_limit")
            score = Decimal("1")
        else:
            risks.append("volatility_above_limit")
            score = Decimal("-1")
        reasons.append(
            f"volatility: volatility_20={_decimal(value)}, threshold={_decimal(threshold)}; score={_decimal(score)}"
        )
        return score

    def _risk(
        self,
        snapshot: StrategySnapshot,
        reasons: list[str],
        rules: list[str],
        risks: list[str],
    ) -> Decimal | None:
        drawdown = snapshot.indicators.get("current_drawdown")
        if not _is_finite(drawdown):
            if drawdown is not None:
                reasons.append("invalid:risk:current_drawdown:non_finite")
            reasons.append("missing:risk:current_drawdown")
            risks.append("drawdown_data_missing")
            return None
        threshold = self.config.thresholds.drawdown_pause
        if drawdown <= threshold:
            rules.append("risk_drawdown_pause")
            risks.append("drawdown_pause_threshold")
            score = Decimal("-1")
        else:
            rules.append("risk_drawdown_within_limit")
            score = Decimal("1")
        reasons.append(
            f"risk: current_drawdown={_decimal(drawdown)}, threshold={_decimal(threshold)}; score={_decimal(score)}"
        )
        return score

    def _weighted_score(self, components: Mapping[str, Decimal | None]) -> Decimal:
        weights = self.config.weights.model_dump()
        available = {name: value for name, value in components.items() if value is not None}
        total = sum((weights[name] for name in available), Decimal("0"))
        if total == 0:
            return Decimal("0")
        return sum((weights[name] * value for name, value in available.items()), Decimal("0")) / total

    @staticmethod
    def _confidence(
        snapshot: StrategySnapshot, components: Mapping[str, Decimal | None]) -> Decimal:
        expected = (
            snapshot.close_price,
            snapshot.volume,
            snapshot.valuation.get("valuation_percentile"),
            snapshot.indicators.get("ma_20"),
            snapshot.indicators.get("ma_60"),
            snapshot.indicators.get("rsi_6"),
            snapshot.indicators.get("macd_histogram"),
            snapshot.indicators.get("volume_ma_20"),
            snapshot.indicators.get("volatility_20"),
            snapshot.indicators.get("current_drawdown"),
        )
        completeness = Decimal(sum(_is_finite(value) for value in expected)) / Decimal(len(expected))
        available = [value for value in components.values() if value is not None]
        if available:
            positive = sum(value > 0 for value in available)
            negative = sum(value < 0 for value in available)
            agreement = Decimal(max(positive, negative)) / Decimal(len(available))
        else:
            agreement = Decimal("0")
        observation_factor = Decimal(len(available)) / Decimal("6")
        return (completeness * Decimal("0.60")) + (agreement * Decimal("0.25")) + (observation_factor * Decimal("0.15"))

    def _recommendation(
        self,
        score: Decimal,
        confidence: Decimal,
        holding_ratio: Decimal | None,
    ) -> str:
        thresholds = self.config.thresholds
        if confidence < MINIMUM_ACTION_CONFIDENCE:
            return "HOLD"
        may_sell = holding_ratio is None or holding_ratio > self.config.minimum_holding_ratio
        if score <= thresholds.pause_score:
            return "PAUSE"
        if score <= thresholds.sell_partial_score and may_sell:
            return "SELL_PARTIAL"
        if score <= thresholds.reduce_score:
            return "REDUCE"
        if score < 0:
            return "HOLD"
        if score < thresholds.increase_score:
            return "NORMAL"
        return "INCREASE"

    def _suggested_sell_ratio(self, recommendation: str, holding_ratio: Decimal | None) -> Decimal:
        if recommendation != "SELL_PARTIAL":
            return Decimal("0")
        ceiling = self.config.maximum_sell_ratio
        if holding_ratio is not None:
            ceiling = min(ceiling, max(Decimal("0"), holding_ratio - self.config.minimum_holding_ratio))
        return ceiling
