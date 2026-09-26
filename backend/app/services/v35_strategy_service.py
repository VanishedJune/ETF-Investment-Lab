"""V3.5 strategy layer: DIF quadrants, confirmation caps, StrategyScore and
graded execution batches.

The strategy layer only reads forecast output and the current V3.5 feature
snapshot.  It never mutates frozen model or forecast rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from backend.app.services.v35_config import (
    EXECUTION_DEFAULTS,
    SCORE_POSITION_MAP,
    STATE_POSITION_CAPS,
    default_strategy_config,
    score_weights_for,
)
from backend.app.services.v35_feature_service import V35FeatureSnapshot
from backend.app.services.v36_config import (
    STRONG_SIGNAL_UP_PROB,
    VOL_TARGET_CLAMP,
)


class V35StrategyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class V35StrategyDecision:
    market: str
    forecast_anchor_date: str
    dif_trend_state: str
    confirmation_status: str
    market_state: str
    strategy_score: float
    base_target_position_pp: int
    state_position_cap_pp: int
    final_target_position_pp: int
    batches: tuple[Mapping[str, Any], ...]
    components: Mapping[str, Any]
    reasons: tuple[str, ...]


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _position_from_score(score: float, config: Mapping[str, Any]) -> int:
    mapping = config["score_position_map"]
    for entry in mapping:
        if score <= entry["score_gt"]:
            return int(entry["position_pp"])
    return int(mapping[-1]["position_pp"])


def _snap_position(position: int, grid: int) -> int:
    return int(round(position / grid) * grid)


class V35StrategyService:
    """Deterministic V3.5 position strategy for one forecast anchor."""

    @staticmethod
    def dif_quadrant(snapshot: V35FeatureSnapshot) -> str:
        dif = snapshot.features.get("v35_dif")
        first = snapshot.features.get("v35_dif_first_change")
        second = snapshot.features.get("v35_dif_second_change")
        if dif is None or first is None:
            return "UNCONFIRMED"
        if dif < 0.0 and first < 0.0:
            return "NEGATIVE_DIF_FALLING"
        if dif < 0.0 and first > 0.0:
            return "NEGATIVE_DIF_RISING"
        if dif > 0.0 and first > 0.0:
            if second is not None and second < 0.0:
                return "POSITIVE_DIF_RISING_DECELERATING"
            if second is not None and second >= 0.0:
                return "POSITIVE_DIF_ACCELERATING"
            return "POSITIVE_DIF_RISING"
        if dif > 0.0 and first < 0.0:
            return "POSITIVE_DIF_FALLING"
        return "UNCONFIRMED"

    @staticmethod
    def confirmation_status(
        snapshot: V35FeatureSnapshot,
        dif_quadrant: str,
        probabilities_4: Sequence[float],
        probabilities_8: Sequence[float],
        *,
        trend_confirmation_weeks: int = 2,
    ) -> str:
        features = snapshot.features
        ma20_distance = features.get("v35_ma20_distance")
        ma20_slope = features.get("v35_ma20_slope")
        ma60_distance = features.get("v35_ma60_distance")
        rebound = features.get("v35_rebound_from_low")
        dif_duration = features.get("v35_dif_state_duration") or 0.0
        golden_duration = features.get("v35_golden_cross_duration") or 0.0
        price_below_ma20 = ma20_distance is not None and ma20_distance < 0.0
        ma20_flat_or_down = ma20_slope is None or ma20_slope <= 0.0
        down_edge_8 = (
            float(probabilities_8[2]) - float(probabilities_8[0])
            if len(probabilities_8) == 3
            else 0.0
        )
        up_edge_8 = -down_edge_8

        if dif_quadrant in ("NEGATIVE_DIF_FALLING",):
            if price_below_ma20 and ma20_flat_or_down and down_edge_8 > 0.15:
                return "BEARISH_CONFIRMED"
            return "UNCONFIRMED"
        if dif_quadrant in ("POSITIVE_DIF_FALLING",):
            if price_below_ma20 and ma20_flat_or_down:
                return "TOP_CONFIRMED"
            if down_edge_8 > 0.25 and price_below_ma20:
                return "SIGNAL_CONFLICT"
            return "UNCONFIRMED"
        if dif_quadrant == "NEGATIVE_DIF_RISING":
            price_bottom = rebound is not None and rebound > 0.0 and ma20_distance is not None
            dif_bottom = dif_duration >= 2.0
            if price_bottom and dif_bottom:
                return "PARTIALLY_CONFIRMED_BOTTOM"
            if price_bottom or dif_bottom:
                return "PRICE_BOTTOM_CONFIRMED" if price_bottom else "DIF_BOTTOM_CONFIRMED"
            return "NEGATIVE_DIF_RISING"
        if dif_quadrant in (
            "POSITIVE_DIF_RISING",
            "POSITIVE_DIF_RISING_DECELERATING",
            "POSITIVE_DIF_ACCELERATING",
        ):
            price_above_ma20 = ma20_distance is not None and ma20_distance > 0.0
            trend_up = (
                price_above_ma20
                and (ma20_slope is None or ma20_slope > 0.0)
                and (ma60_distance is None or ma60_distance > -0.05)
                and up_edge_8 > 0.0
                and golden_duration >= max(1, trend_confirmation_weeks)
            )
            if trend_up:
                return "TREND_CONFIRMED"
            return "POSITIVE_DIF_RISING"
        return "UNCONFIRMED"

    @staticmethod
    def market_state(snapshot: V35FeatureSnapshot) -> tuple[str, int]:
        code = snapshot.features.get("v35_market_state")
        duration = int(snapshot.features.get("v35_market_state_duration") or 1)
        names = {
            1.0: "UPTREND",
            -1.0: "DOWNTREND",
            0.0: "SIDEWAYS",
            2.0: "HIGH_VOLATILITY",
            -2.0: "LOW_VOLATILITY",
            -3.0: "DRAWDOWN",
            3.0: "RECOVERY",
        }
        return (names.get(code, "SIDEWAYS"), max(1, duration))

    def score_components(
        self,
        market: str,
        snapshot: V35FeatureSnapshot,
        expected_path: Sequence[float],
        p10_path: Sequence[float],
        probabilities_4: Sequence[float],
        probabilities_8: Sequence[float],
    ) -> dict[str, Any]:
        weights = score_weights_for(market)
        ret4 = float(expected_path[3]) if len(expected_path) >= 4 else 0.0
        ret8 = float(expected_path[-1]) if expected_path else 0.0
        predicted_return = _clip((ret4 + ret8) / 2.0 / 0.08)
        edge4 = (
            float(probabilities_4[0]) - float(probabilities_4[2])
            if len(probabilities_4) == 3
            else 0.0
        )
        edge8 = (
            float(probabilities_8[0]) - float(probabilities_8[2])
            if len(probabilities_8) == 3
            else 0.0
        )
        probability_edge = _clip((edge4 + edge8) / 2.0)

        dif = snapshot.features.get("v35_dif") or 0.0
        dif_first = snapshot.features.get("v35_dif_first_change")
        dif_first = 0.0 if dif_first is None else dif_first
        ma20_slope = snapshot.features.get("v35_ma20_slope")
        ma20_slope = 0.0 if ma20_slope is None else ma20_slope
        slope_norm = _clip(ma20_slope / 0.01)
        trend_momentum = _clip(
            0.40 * (1.0 if dif > 0 else -1.0)
            + 0.30 * (1.0 if dif_first > 0 else -1.0)
            + 0.30 * slope_norm
        )

        p10_8 = float(p10_path[-1]) if p10_path else 0.0
        drawdown = snapshot.features.get("v35_current_drawdown") or 0.0
        risk = _clip(-p10_8 / 0.15 - max(-drawdown / 0.25, 0.0) * 0.5)

        valuation = 0.0
        if market == "399006":
            percentile = snapshot.features.get("pe_expanding_percentile")
            if percentile is not None:
                valuation = _clip((0.60 - float(percentile) / 100.0) / 0.50)

        raw = (
            weights["predicted_return"] * predicted_return
            + weights["probability_edge"] * probability_edge
            + weights["trend_momentum"] * trend_momentum
            + weights["risk"] * risk
            + weights["valuation"] * valuation
        )
        return {
            "predicted_return": predicted_return,
            "probability_edge": probability_edge,
            "trend_momentum": trend_momentum,
            "risk": risk,
            "valuation": valuation,
            "score": _clip(raw),
        }

    def decide(
        self,
        market: str,
        snapshot: V35FeatureSnapshot,
        *,
        expected_path: Sequence[float],
        p10_path: Sequence[float],
        probabilities_4: Sequence[float],
        probabilities_8: Sequence[float],
        current_position_pp: int,
        reliability_score: float,
        health_status: str,
        ood_score: float,
        config: Mapping[str, Any] | None = None,
    ) -> V35StrategyDecision:
        package = dict(default_strategy_config(market) if config is None else config)
        weights = dict(package["score_weights"])
        components = self.score_components(
            market,
            snapshot,
            expected_path,
            p10_path,
            probabilities_4,
            probabilities_8,
        )
        score = float(components["score"])
        base_target = _snap_position(
            _position_from_score(score, package), int(package["position_grid_pp"])
        )
        quadrant = self.dif_quadrant(snapshot)
        confirmation = self.confirmation_status(
            snapshot,
            quadrant,
            probabilities_4,
            probabilities_8,
            trend_confirmation_weeks=int(
                package.get("trend_confirmation_weeks", 2)
            ),
        )
        caps = {str(key): int(value) for key, value in package["state_position_caps"].items()}
        state_cap = caps.get(confirmation, caps.get("UNCONFIRMED", 30))
        if quadrant == "POSITIVE_DIF_RISING_DECELERATING":
            state_cap = min(state_cap, 60)
        final_target = min(base_target, state_cap)
        final_target = _snap_position(final_target, int(package["position_grid_pp"]))

        execution = dict(package["execution"])
        if execution.get("vol_target_enabled"):
            sigma = float(snapshot.features.get("v35_volatility_26") or 0.20)
            target_vol = float(execution.get("target_vol", 10.0)) / 100.0
            multiplier = max(
                VOL_TARGET_CLAMP[0],
                min(
                    VOL_TARGET_CLAMP[1],
                    target_vol / max(sigma, 1e-9),
                ),
            )
            final_target = min(
                int(package.get("max_position_pp", 80)),
                int(round(final_target * multiplier / int(package["position_grid_pp"])))
                * int(package["position_grid_pp"]),
            )
        max_batches = int(execution["max_batches"])
        max_single = int(execution["max_single_batch_pp"])
        cooldown = int(execution["cooldown_trading_days"])
        reasons: list[str] = []
        batches: list[Mapping[str, Any]] = []

        emergency_sell = False
        if health_status == "MODEL_OUT_OF_DISTRIBUTION" and ood_score > 0.70:
            emergency_sell = True
            reasons.append("严重OOD且行情恶化")
        if confirmation == "TOP_CONFIRMED":
            reasons.append("已确认顶部")
        if confirmation == "BEARISH_CONFIRMED":
            reasons.append("明确看跌")

        if final_target > current_position_pp:
            if confirmation in ("TOP_CONFIRMED", "BEARISH_CONFIRMED") or health_status == "MODEL_OUT_OF_DISTRIBUTION":
                reasons.append("硬条件禁止新增仓位")
                batches = []
            else:
                remaining = final_target - current_position_pp
                strong = confirmation == "TREND_CONFIRMED" and float(components["probability_edge"]) > 0.15
                weak = confirmation in ("UNCONFIRMED", "NEGATIVE_DIF_RISING") or health_status == "MODEL_DEGRADED"
                strong_signal = bool(
                    len(probabilities_8) >= 3
                    and float(probabilities_8[0]) >= STRONG_SIGNAL_UP_PROB
                    and float(expected_path[-1]) > 0.0
                    and float(snapshot.features.get("v35_dif") or 0.0) > 0.0
                )
                first = (
                    int(execution["strong_first_batch_pp"])
                    if strong
                    else int(execution["weak_first_batch_pp"])
                    if weak
                    else int(execution["default_first_batch_pp"])
                )
                floor = int(execution.get("strong_signal_entry_floor_pp", 0) or 0)
                if strong_signal and floor > first:
                    first = floor
                first = min(first, remaining, max_single)
                position = current_position_pp
                for batch_number in range(1, max_batches + 1):
                    if position >= final_target:
                        break
                    if batch_number == 1:
                        change = first
                    elif quadrant in ("NEGATIVE_DIF_RISING",) and batch_number == 2:
                        change = min(int(execution["recovery_second_batch_pp"]), remaining - position, max_single)
                    elif quadrant.startswith("POSITIVE_DIF_RISING") and confirmation == "TREND_CONFIRMED":
                        change = min(int(execution["strong_trend_add_pp"]), remaining - position, max_single)
                    else:
                        change = min(int(execution["trend_add_pp"]), remaining - position, max_single)
                    if change < 5:
                        break
                    position += change
                    batches.append(
                        {
                            "batch_number": batch_number,
                            "action": "BUY",
                            "position_pp": position,
                            "batch_change_pp": change,
                            "target_position_pp": final_target,
                            "cooldown_trading_days": cooldown,
                            "condition": (
                                "趋势加仓：POSITIVE_DIF_RISING持续"
                                if quadrant.startswith("POSITIVE_DIF_RISING")
                                else "试探/恢复型买入"
                                if quadrant == "NEGATIVE_DIF_RISING"
                                else "常规分批买入"
                            ),
                        }
                    )
                if not batches and final_target > current_position_pp:
                    reasons.append("剩余目标仓位不足一个5%批次")
        elif final_target < current_position_pp:
            remaining = current_position_pp - final_target
            if emergency_sell:
                change = min(remaining, int(execution["emergency_reduce_pp"]), max_single)
                reasons.append("紧急风险控制")
            elif confirmation in ("TOP_CONFIRMED", "BEARISH_CONFIRMED"):
                change = min(remaining, int(execution["confirmed_reduce_pp"]))
                reasons.append("确认减仓")
            elif remaining <= 10:
                change = min(remaining, int(execution["light_reduce_pp"]))
                reasons.append("轻度减仓")
            else:
                change = min(remaining, int(execution["light_reduce_pp"]))
                reasons.append("停止加仓并小幅减仓")
            change = _snap_position(change, int(package["position_grid_pp"]))
            if change >= 5:
                batches.append(
                    {
                        "batch_number": 1,
                        "action": "SELL",
                        "position_pp": current_position_pp - change,
                        "batch_change_pp": change,
                        "target_position_pp": final_target,
                        "cooldown_trading_days": cooldown,
                        "condition": reasons[-1],
                    }
                )

        market_state, state_duration = self.market_state(snapshot)
        return V35StrategyDecision(
            market=market,
            forecast_anchor_date=snapshot.cutoff_date.isoformat(),
            dif_trend_state=quadrant,
            confirmation_status=confirmation,
            market_state=market_state,
            strategy_score=score,
            base_target_position_pp=base_target,
            state_position_cap_pp=state_cap,
            final_target_position_pp=final_target,
            batches=tuple(batches),
            components=components,
            reasons=tuple(reasons),
        )
