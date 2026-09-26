"""V3.5 weekly runtime: feature, forecast, strategy, account, challenge and
promotion in one idempotent per-anchor transaction.

The bootstrap walks every complete natural-week anchor in strict order.  A
week is processed only once; interrupted runs resume from the first missing
anchor.  All V3.5 rows live in ``v35_*`` tables; earlier-generation rows are
never selected, updated or deleted.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import zlib
from typing import Any, Mapping, Sequence

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V35BootstrapState,
    V35Challenge,
    V35ChallengerWindow,
    V35FeatureSnapshot as V35FeatureSnapshotRow,
    V35Forecast,
    V35ForecastCalibratorLink,
    V35ForecastEvaluation,
    V35HealthSnapshot,
    V351CandidateEvaluation,
    V351MaintenanceRefresh,
    V35ModelPackage,
    V35ModelVersion,
    V35PositionDecision,
    V35ProbabilityCalibrator,
    V35Promotion,
    V35RandomPlan,
    V35ResidualRecord,
    V35SimAccount,
    V35SimEvaluation,
    V35SimLedger,
    V35StrategySnapshot,
    V36AccountSnapshot,
    V36DecisionFunnel,
    V35TrainingIteration,
    V35TrainingProfile,
    V35TrainingRun,
    utc_now,
)
from backend.app.services.v35_config import (
    ALPHA_MAX,
    ALPHA_MIN,
    CANDIDATE_RANDOM_PLAN_VERSION,
    FEATURE_VERSION,
    FORMAL_CALIBRATION_SAMPLES,
    HORIZON_WEEKS,
    INITIAL_PACKAGE_DEFAULTS,
    PREWARMING_THRESHOLD,
    PREDICTION_CHALLENGE_EVERY_MATURED,
    PREDICTION_CHALLENGER_COUNT,
    PROTOCOL_VERSION,
    PROMOTION_DEFAULTS,
    PROMOTION_RULE_VERSION,
    RESIDUAL_SCHEMA_VERSION,
    SCORE_WEIGHTS,
    SMALL_SAMPLE_REGULARIZED,
    STRATEGY_CHALLENGE_EVERY_MATURED,
    STRATEGY_CHALLENGER_COUNT,
    default_strategy_config,
    min_train_samples_for,
)
from backend.app.services.v341_training_service import candidate_alphas
from backend.app.services.v35_feature_service import (
    FEATURE_SET_CORE,
    V35FeatureService,
    V35FeatureError,
    build_feature_manifest,
)
from backend.app.services.v34_feature_service import V34FeatureError
from backend.app.services.v35_simulation_service import (
    INITIAL_CAPITAL,
    load_weekly_bars,
    persist_continuous_result,
    persist_window_result,
    run_continuous_window,
    run_standard_window,
)
from backend.app.services.v351_behavior_service import (
    V351AnchorRecord,
    behavior_diff,
    effective_independent_window_count,
    passes_divergence_prefilter,
    persist_candidate_evaluation,
)
from backend.app.services.v351_config import (
    ALPHA_MAX_351,
    ALPHA_MIN_351,
    ALPHA_MULTIPLIERS_351,
    DAILY_ADJUSTMENT_VALUES,
    FEATURE_SET_VALUES,
    MAINTENANCE_EVERY_MATURED,
    MAINTENANCE_GATES,
    MAX_PREDICTION_CANDIDATES,
    MAX_STRATEGY_PARAMETER_GROUP_TRIES,
    MINIMUM_PROMOTABLE_EXCESS_RETURN,
    promotion_channel_for,
    RESIDUAL_SCALE_VALUES,
    STRATEGY_PARAMETER_GROUPS,
    STRUCTURAL_CHALLENGE_EVERY,
    STRUCTURAL_DIMENSIONS,
    STABLE_SMALL_EDGE_GATES,
    STRONG_PROMOTION_GATES,
    TRAINING_WINDOW_VALUES,
)
from backend.app.services.v36_config import (
    ACTIVITY_DEGRADED_MAX_AVG_EXPOSURE_PP,
    ACTIVITY_DEGRADED_MIN_STRONG_UP_WINDOWS,
    ACTIVITY_LOOKBACK_WEEKS,
    CALIBRATION_ISOTONIC_SAMPLES,
    CALIBRATION_PLATT_SAMPLES,
    CALIBRATION_UNAVAILABLE_SAMPLES,
    MAINTENANCE_EVERY_MATURED_36,
    MAINTENANCE_GATES_36,
    MINIMUM_PROMOTABLE_EXCESS_RETURN_36,
    MODEL_FAMILY_VALUES_36,
    PROTOCOL_VERSION_36,
    PREDICTION_CHALLENGE_EVERY_MATURED_36,
    REGIME_CORRECTION_MAX_PER_WEEK,
    REGIME_CORRECTION_SIGNAL_SCALE,
    REGIME_CORRECTION_STRENGTH_DENOM,
    REGIME_CORRECTION_VALUES_36,
    REGIME_CORRECTION_WEEKS,
    RESIDUAL_HALF_LIFE_WEEKS,
    RESIDUAL_SCALE_VALUES_36,
    RESIDUAL_WEIGHTING_VALUES_36,
    STABLE_SMALL_EDGE_GATES_36,
    STRATEGY_CHALLENGE_EVERY_MATURED_36,
    STRATEGY_PARAMETER_GROUPS_36,
    STRONG_PROMOTION_GATES_36,
    promotion_channel_for_36,
    V36_ALPHA_MAX,
    V36_ALPHA_MIN,
    V36_ALPHA_MULTIPLIERS,
    strong_up_threshold_for,
)
from backend.app.services.v35_strategy_service import (
    V35StrategyDecision,
    V35StrategyService,
)
from backend.app.services.v35_training_service import (
    V35ModelState,
    V35TrainingSample,
    V35TrainingService,
    state_from_payload,
    state_to_payload,
)


class V35RuntimeError(RuntimeError):
    pass


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _pct(value: float) -> Decimal:
    return Decimal(str(round(float(value), 8)))


def _money(value: float) -> Decimal:
    return Decimal(str(round(float(value), 2)))


def _snap_to_grid(position: float, grid: int = 5) -> int:
    return int(round(position / grid) * grid)


def _actual_class(value: float, threshold: float) -> str:
    if value > threshold:
        return "UP"
    if value < -threshold:
        return "DOWN"
    return "SIDEWAYS"


def _probabilities_from_expected(expected_return: float, sigma: float, threshold: float) -> list[float]:
    scale = max(sigma, 0.005)
    up = float(1.0 - _normal_cdf((threshold - expected_return) / scale))
    down = float(_normal_cdf((-threshold - expected_return) / scale))
    up = float(np.clip(up, 0.0, 1.0))
    down = float(np.clip(down, 0.0, 1.0 - up))
    sideways = float(np.clip(1.0 - up - down, 0.0, 1.0))
    total = up + down + sideways
    return [up / total, sideways / total, down / total]


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))


def _temperature_from_brier(brier: float, sample_count: int) -> float:
    if sample_count < 30:
        return 1.0
    raw = 1.0 + 0.5 * (1.0 - max(0.0, min(1.0, float(brier) / 0.25)))
    return float(np.clip(raw, 0.5, 2.0))


def _apply_temperature(
    probabilities: Sequence[float],
    temperature: float,
) -> list[float]:
    logits = np.log(np.maximum(np.asarray(probabilities, dtype=float), 1e-12))
    scaled = logits / max(float(temperature), 0.01)
    exp_values = np.exp(scaled - np.max(scaled))
    return (exp_values / np.sum(exp_values)).tolist()


def _reliability_components(
    *,
    raw_sample_count: int,
    calibration_status: str,
    ood_score: float,
    recent_mae: float | None,
) -> tuple[float, str]:
    score = 100.0
    reasons: list[str] = []
    if raw_sample_count < 30:
        score = min(score, 40.0)
        reasons.append("样本不足")
    if calibration_status != "FORMAL":
        score = min(score, 45.0)
        reasons.append("无正式校准")
    if ood_score > 0.55:
        score = min(score, 50.0)
        reasons.append("材料OOD")
    if recent_mae is not None and recent_mae > 0.20:
        score = min(score, 50.0)
        reasons.append("近期样本外误差偏高")
    if ood_score > 0.70:
        health = "MODEL_OUT_OF_DISTRIBUTION"
    elif score <= 50.0:
        health = "MODEL_DEGRADED"
    else:
        health = "MODEL_NORMAL"
    return float(score), health


def promotion_gate_decision(
    *,
    mean_excess: float,
    median_profit_excess: float,
    win_rate: float,
    max_drawdown_degradation: float,
    no_action_ratio: float,
    mean_position_pp: float,
    rules: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Return (decision, reason) for the sealed 8-week account gate."""

    thresholds = dict(PROMOTION_DEFAULTS if rules is None else rules)
    reasons: list[str] = []
    ok = True
    if mean_excess < float(thresholds["minimum_excess_return"]):
        ok = False
        reasons.append("平均超额收益不足0.30%")
    if median_profit_excess < 0.0:
        ok = False
        reasons.append("净利润中位数未超过Champion")
    if win_rate < float(thresholds["minimum_win_rate"]):
        ok = False
        reasons.append("战胜比例低于55%")
    if max_drawdown_degradation > float(thresholds["maximum_drawdown_degradation_pp"]) / 100.0:
        ok = False
        reasons.append("最大回撤恶化超过2个百分点")
    if no_action_ratio > float(thresholds["no_action_window_ratio_cap"]):
        ok = False
        reasons.append("长期空仓比例过高")
    if mean_position_pp < float(thresholds["bull_market_min_position_pp"]):
        ok = False
        reasons.append("平均仓位过低，疑似空仓避险")
    if ok:
        return "PROMOTED", "满足全部晋级门禁"
    return "REJECTED", "; ".join(reasons)


class _OnlineAccount:
    """Champion continuous position tracker used for frozen decisions."""

    def __init__(
        self,
        *,
        accounting_mode: str = "PERCENT",
        initial: Mapping[str, Any] | None = None,
    ) -> None:
        self.accounting_mode = accounting_mode
        self.equity = float((initial or {}).get("equity", INITIAL_CAPITAL))
        self.cash = float((initial or {}).get("cash", INITIAL_CAPITAL))
        self.position_pp = float((initial or {}).get("position_pp", 0.0))
        self.held_shares = float((initial or {}).get("held_shares", 0.0))
        self.average_cost = float((initial or {}).get("average_cost") or 0.0)
        self.sellable_shares = float((initial or {}).get("sellable_shares", 0.0))
        self.pending_sellable: list[tuple[date, float]] = list(
            (initial or {}).get("pending_sellable", [])
        )
        self.pending_batches: list[Mapping[str, Any]] = [
            dict(batch) for batch in (initial or {}).get("pending_batches", [])
        ]

    def _mark_sellable(self, anchor: date) -> None:
        if self.accounting_mode != "SHARES":
            return
        remaining: list[tuple[date, float]] = []
        for acquired, shares in self.pending_sellable:
            if acquired < anchor:
                self.sellable_shares += shares
            else:
                remaining.append((acquired, shares))
        self.pending_sellable = remaining

    def execute_and_mark(self, anchor: date, bar: Mapping[str, float | None]) -> None:
        if bar is None or bar["open"] in (None, 0.0) or bar["close"] in (None, 0.0):
            return
        self._mark_sellable(anchor)
        open_price = float(bar["open"])
        close_price = float(bar["close"])
        if self.pending_batches:
            batch = self.pending_batches[0]
            action = str(batch["action"])
            target = int(batch["target_position_pp"])
            change = int(batch["batch_change_pp"])
            if action == "BUY":
                change = min(change, max(0, target - self.position_pp))
            else:
                change = min(change, max(0, self.position_pp - target))
            if change >= 5:
                if self.accounting_mode == "SHARES":
                    notional = self.equity * change / 100.0
                    if action == "BUY":
                        shares = round(notional / open_price, 2)
                        if shares > 0:
                            new_cost = self.average_cost * self.held_shares + notional
                            self.held_shares += shares
                            self.average_cost = (
                                new_cost / self.held_shares if self.held_shares else 0.0
                            )
                            self.pending_sellable.append((anchor, shares))
                    else:
                        shares_to_sell = min(
                            self.sellable_shares, round(notional / open_price, 2)
                        )
                        if shares_to_sell >= 0.01:
                            self.held_shares = round(
                                self.held_shares - shares_to_sell, 2
                            )
                            self.sellable_shares = round(
                                self.sellable_shares - shares_to_sell, 2
                            )
                else:
                    self.position_pp += change if action == "BUY" else -change
                self.pending_batches = self.pending_batches[1:]
        market_return = close_price / open_price - 1.0
        if self.accounting_mode == "SHARES" and self.held_shares > 0:
            position_frac_at_open = (
                self.held_shares * open_price
            ) / max(self.equity, 1e-9)
            self.equity *= 1.0 + position_frac_at_open * market_return
            self.position_pp = (
                self.held_shares * close_price
            ) / max(self.equity, 1e-9) * 100.0
        else:
            self.equity *= 1.0 + (self.position_pp / 100.0) * market_return

    def set_decision(self, decision: V35StrategyDecision) -> None:
        self.pending_batches = list(decision.batches)

    def snapshot(self, anchor: date) -> dict[str, Any]:
        return {
            "anchor_date": anchor.isoformat(),
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "position_pp": round(self.position_pp, 2),
            "held_shares": round(self.held_shares, 2),
            "average_cost": round(self.average_cost, 6),
            "sellable_shares": round(self.sellable_shares, 2),
            "pending_sellable": list(self.pending_sellable),
            "pending_batches": [dict(batch) for batch in self.pending_batches],
        }


class V35RuntimeService:
    """Owns the V3.5 per-anchor pipeline and persistence."""

    feature_set_name: str = FEATURE_SET_CORE
    feature_version: str = FEATURE_VERSION
    promotion_rule_version: str = PROMOTION_RULE_VERSION

    def __init__(
        self,
        factory: sessionmaker,
        *,
        protocol_version: str = PROTOCOL_VERSION,
        v351: bool = False,
    ) -> None:
        self._factory = factory
        self._features = V35FeatureService()
        self._training = V35TrainingService()
        self._strategy = V35StrategyService()
        self.protocol_version = protocol_version
        self.v351 = v351
        self.v36 = protocol_version == PROTOCOL_VERSION_36
        self._calibrator_cache: dict[
            str, tuple[object, dict[int, V35ProbabilityCalibrator | None]]
        ] = {}

    def _cfg_alpha_multipliers(self) -> tuple[float, ...]:
        return V36_ALPHA_MULTIPLIERS if self.v36 else ALPHA_MULTIPLIERS_351

    def _cfg_alpha_bounds(self) -> tuple[float, float]:
        if self.v36:
            return V36_ALPHA_MIN, V36_ALPHA_MAX
        return ALPHA_MIN_351, ALPHA_MAX_351

    def _cfg_prediction_every(self) -> int:
        return PREDICTION_CHALLENGE_EVERY_MATURED_36 if self.v36 else PREDICTION_CHALLENGE_EVERY_MATURED

    def _cfg_strategy_every(self) -> int:
        return STRATEGY_CHALLENGE_EVERY_MATURED_36 if self.v36 else STRATEGY_CHALLENGE_EVERY_MATURED

    def _cfg_maintenance_every(self) -> int:
        return MAINTENANCE_EVERY_MATURED_36 if self.v36 else MAINTENANCE_EVERY_MATURED

    def _cfg_maintenance_gates(self) -> Mapping[str, Any]:
        return MAINTENANCE_GATES_36 if self.v36 else MAINTENANCE_GATES

    def _cfg_strategy_groups(self) -> Mapping[str, tuple[Any, ...]]:
        return STRATEGY_PARAMETER_GROUPS_36 if self.v36 else STRATEGY_PARAMETER_GROUPS

    def _cfg_residual_scale_values(self) -> tuple[float, ...]:
        return RESIDUAL_SCALE_VALUES_36 if self.v36 else RESIDUAL_SCALE_VALUES

    def _cfg_strong_gates(self) -> Mapping[str, Any]:
        return STRONG_PROMOTION_GATES_36 if self.v36 else STRONG_PROMOTION_GATES

    def _cfg_stable_gates(self) -> Mapping[str, Any]:
        return STABLE_SMALL_EDGE_GATES_36 if self.v36 else STABLE_SMALL_EDGE_GATES

    def _cfg_minimum_excess(self) -> float:
        return MINIMUM_PROMOTABLE_EXCESS_RETURN_36 if self.v36 else MINIMUM_PROMOTABLE_EXCESS_RETURN

    @staticmethod
    def validate_market(market: str) -> None:
        code = str(market)
        if code != "399006" and not (len(code) == 6 and code.isdigit()):
            raise V35RuntimeError("V3.5 supports 399006 and six-digit ETF codes")

    def _instrument_id(self, session: Session, market: str) -> int:
        row = session.scalar(select(Instrument).where(Instrument.code == market))
        if row is None:
            raise V35RuntimeError(f"missing instrument {market}")
        return row.id

    def _anchors(self, session: Session, market: str) -> tuple[date, ...]:
        return self._features.weekly_anchors(session, market)

    def _persist_profile(
        self,
        session: Session,
        market: str,
        feature_names: Sequence[str],
        manifest_hash: str,
    ) -> V35TrainingProfile:
        profile_id = f"{self.protocol_version}:{market}:CORE"
        payload = {
            "protocol_version": self.protocol_version,
            "market": market,
            "name": self.feature_set_name,
            "version": self.feature_version,
            "horizon_weeks": HORIZON_WEEKS,
            "ordered_feature_names": list(feature_names),
            "feature_manifest_hash": manifest_hash,
            "training_window_mode": INITIAL_PACKAGE_DEFAULTS["training_window_mode"],
            "feature_set_name": self.feature_set_name,
        }
        profile = session.get(V35TrainingProfile, profile_id)
        if profile is None:
            session.add(
                V35TrainingProfile(
                    id=profile_id,
                    protocol_version=self.protocol_version,
                    model_market=market,
                    training_window_mode=INITIAL_PACKAGE_DEFAULTS["training_window_mode"],
                    formal_training_weeks=520,
                    minimum_training_weeks=min_train_samples_for(market),
                    feature_warmup_weeks=52,
                    feature_set_name=self.feature_set_name,
                    feature_set_version=self.feature_version,
                    ordered_feature_names_json=list(feature_names),
                    feature_manifest_hash=manifest_hash,
                    horizon_weeks=HORIZON_WEEKS,
                    profile_json=payload,
                    profile_hash=_hash(payload),
                    created_at=utc_now(),
                )
            )
            return session.get(V35TrainingProfile, profile_id)
        return profile

    def _persist_feature(
        self,
        session: Session,
        market: str,
        anchor: date,
        snapshot: Any,
        manifest: Any,
    ) -> V35FeatureSnapshotRow:
        audit = self._features.audit_payload(snapshot, manifest)
        feature_id = f"{self.protocol_version}:{market}:{anchor.isoformat()}:{manifest.manifest_hash}"
        row = session.get(V35FeatureSnapshotRow, feature_id)
        if row is not None:
            return row
        session.add(
            V35FeatureSnapshotRow(
                id=feature_id,
                protocol_version=self.protocol_version,
                model_market=market,
                target_instrument_id=self._instrument_id(session, market),
                forecast_anchor_date=anchor,
                cutoff_at=datetime.combine(anchor, time.max, tzinfo=timezone.utc),
                source_max_date=snapshot.source_data_max_date,
                feature_manifest_hash=manifest.manifest_hash,
                feature_json=dict(snapshot.features),
                daily_sequence_json=list(snapshot.daily_sequence),
                provenance_json=dict(snapshot.provenance),
                leakage_audit_json=audit["leakage_audit"],
                snapshot_hash=audit["snapshot_hash"],
                created_at=utc_now(),
            )
        )
        session.flush()
        return session.get(V35FeatureSnapshotRow, feature_id)

    def _feature_snapshots(
        self,
        session: Session,
        market: str,
        through: date | None = None,
    ) -> tuple[Any, ...]:
        rows = session.scalars(
            select(V35FeatureSnapshotRow)
            .where(V35FeatureSnapshotRow.model_market == market)
            .order_by(V35FeatureSnapshotRow.forecast_anchor_date)
        ).all()
        if through is not None:
            rows = [row for row in rows if row.forecast_anchor_date <= through]
        return tuple(rows)

    def _matured_samples(
        self,
        session: Session,
        market: str,
        as_of: date,
    ) -> tuple[V35TrainingSample, ...]:
        rows = self._feature_snapshots(session, market, through=as_of)
        if len(rows) < HORIZON_WEEKS + 1:
            return ()
        result: list[V35TrainingSample] = []
        for index, row in enumerate(rows):
            if index + HORIZON_WEEKS >= len(rows):
                break
            futures = rows[index + 1 : index + HORIZON_WEEKS + 1]
            end_date = futures[-1].forecast_anchor_date
            if end_date > as_of:
                continue
            anchor = row.forecast_anchor_date
            closes = [float(row.daily_sequence_json[-1]["close"]) for row in (row, *futures)]
            path = tuple(closes[step] / closes[0] - 1.0 for step in range(1, HORIZON_WEEKS + 1))
            result.append(
                V35TrainingSample(
                    market=market,
                    anchor_date=anchor,
                    snapshot=replace(
                        self._empty_snapshot(market, anchor),
                        features=dict(row.feature_json),
                        daily_sequence=tuple(row.daily_sequence_json),
                        source_data_max_date=row.source_max_date,
                        provenance=dict(row.provenance_json),
                    ),
                    cumulative_return_path=path,
                    label_end_date=end_date,
                )
            )
        return tuple(result)

    @staticmethod
    def _empty_snapshot(market: str, anchor: date) -> Any:
        from backend.app.services.v35_feature_service import V35FeatureSnapshot

        return V35FeatureSnapshot(
            market=market,
            cutoff_date=anchor,
            source_data_max_date=anchor,
            features={},
            daily_sequence=(),
            provenance={},
        )

    def _persist_model(
        self,
        session: Session,
        market: str,
        state: V35ModelState,
        *,
        status: str,
        metrics: Mapping[str, Any],
        promotion_gate: Mapping[str, Any],
    ) -> V35ModelVersion:
        row = session.get(V35ModelVersion, state.version)
        if row is None:
            payload = state_to_payload(state)
            session.add(
                V35ModelVersion(
                    id=state.version,
                    protocol_version=self.protocol_version,
                    model_market=market,
                    version=state.version,
                    parent_model_id=state.parent_version,
                    profile_id=f"{self.protocol_version}:{market}:CORE",
                    status=status,
                    trained_through_date=state.trained_through_date,
                    effective_from_date=state.effective_from_date,
                    feature_anchor_max_date=state.feature_anchor_max_date,
                    label_observed_through_date=state.label_observed_through_date,
                    raw_matured_sample_count=state.raw_matured_sample_count,
                    effective_independent_sample_count=state.effective_independent_sample_count,
                    horizon_weeks=HORIZON_WEEKS,
                    ridge_alpha=_pct(state.ridge_alpha),
                    parameters_json=payload,
                    metrics_json=dict(metrics),
                    promotion_gate_json=dict(promotion_gate),
                    health_status=status,
                    parameter_hash=state.parameter_hash,
                    created_at=utc_now(),
                )
            )
            session.flush()
            return session.get(V35ModelVersion, state.version)
        return row

    def _persist_package(
        self,
        session: Session,
        market: str,
        *,
        version: str,
        kind: str,
        prediction_model_id: str,
        strategy_config: Mapping[str, Any],
        effective_from: date,
        parent_package_id: str | None = None,
    ) -> V35ModelPackage:
        row = session.get(V35ModelPackage, version)
        if row is not None:
            return row
        payload = {
            "protocol_version": self.protocol_version,
            "market": market,
            "version": version,
            "kind": kind,
            "prediction_model_id": prediction_model_id,
            "policy_version": strategy_config.get("policy_version"),
            "feature_version": self.feature_version,
            "promotion_rule_version": self.promotion_rule_version,
            "strategy_config": dict(strategy_config),
        }
        session.add(
            V35ModelPackage(
                id=version,
                protocol_version=self.protocol_version,
                model_market=market,
                version=version,
                package_kind=kind,
                parent_package_id=parent_package_id,
                prediction_model_id=prediction_model_id,
                policy_version=str(strategy_config.get("policy_version", "")),
                feature_version=self.feature_version,
                promotion_rule_version=self.promotion_rule_version,
                strategy_config_json=dict(strategy_config),
                effective_from_date=effective_from,
                package_hash=_hash(payload),
                created_at=utc_now(),
            )
        )
        session.flush()
        return session.get(V35ModelPackage, version)

    def _champion_package(self, session: Session, market: str, anchor: date) -> V35ModelPackage | None:
        state = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == self.protocol_version,
            )
        )
        if state is None or state.champion_package_id is None:
            return None
        package = session.get(V35ModelPackage, state.champion_package_id)
        if package is None or package.effective_from_date > anchor:
            return None
        return package

    def _staged_champion_package(self, session: Session, market: str) -> V35ModelPackage | None:
        state = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == self.protocol_version,
            )
        )
        if state is None or state.champion_package_id is None:
            return None
        return session.get(V35ModelPackage, state.champion_package_id)

    def _load_model_state(self, session: Session, package: V35ModelPackage) -> V35ModelState:
        model = session.get(V35ModelVersion, package.prediction_model_id)
        if model is None:
            raise V35RuntimeError(f"missing model {package.prediction_model_id}")
        return state_from_payload(model.parameters_json)

    def _random_plan(
        self,
        session: Session,
        market: str,
        anchor: date,
        package_id: str,
        residual_pool: Sequence[V35ResidualRecord],
        scenario_count: int,
        *,
        residual_multiplier: float = 1.0,
        residual_weighting: str = "UNIFORM",
        half_life_weeks: float = RESIDUAL_HALF_LIFE_WEEKS,
    ) -> V35RandomPlan:
        seed = int(
            _hash(
                {
                    "market": market,
                    "anchor": anchor.isoformat(),
                    "package": package_id,
                    "plan_version": CANDIDATE_RANDOM_PLAN_VERSION,
                }
            ),
            16,
        ) % (2**32)
        rng = np.random.default_rng(seed)
        pool_size = len(residual_pool)
        if residual_weighting == "EXP_DECAY" and pool_size > 0:
            age = np.arange(pool_size, dtype=float)
            weights = np.exp(-age * np.log(2.0) / float(half_life_weeks))
            weights = weights / float(weights.sum())
            indices = rng.choice(
                pool_size,
                size=(scenario_count, HORIZON_WEEKS),
                p=weights,
            ).tolist()
        else:
            indices = rng.integers(0, max(pool_size, 1), size=(scenario_count, HORIZON_WEEKS)).tolist()
        payload = {
            "indices": indices,
            "pool_size": pool_size,
            "residual_multiplier": float(residual_multiplier),
            "residual_unit": "STANDARDIZED_X_SIGMA",
            "sigma_version": "V351_FIX_A1_1",
            "residual_weighting": residual_weighting,
            "half_life_weeks": float(half_life_weeks),
        }
        compressed = zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        plan_hash = _hash(payload)
        plan = session.scalar(
            select(V35RandomPlan).where(
                V35RandomPlan.protocol_version == self.protocol_version,
                V35RandomPlan.model_market == market,
                V35RandomPlan.plan_hash == plan_hash,
            )
        )
        if plan is not None:
            return plan
        plan = V35RandomPlan(
            id=f"{self.protocol_version}:{market}:{plan_hash[:24]}",
            protocol_version=self.protocol_version,
            model_market=market,
            plan_scope="FORECAST",
            plan_version=CANDIDATE_RANDOM_PLAN_VERSION,
            seed=int(seed),
            scenario_count=scenario_count,
            residual_pool_identity=_hash(
                {"pool": [row.id for row in residual_pool], "market": market}
            ),
            residual_pool_size=pool_size,
            payload_codec="ZLIB_JSON",
            plan_payload_zlib=compressed,
            plan_hash=plan_hash,
            created_at=utc_now(),
        )
        session.add(plan)
        session.flush()
        return plan

    def _fit_calibrators(
        self,
        session: Session,
        market: str,
        as_of: date,
    ) -> dict[int, V35ProbabilityCalibrator | None]:
        calibrators: dict[int, V35ProbabilityCalibrator | None] = {}
        for horizon in (4, 8):
            evaluations = session.scalars(
                select(V35ForecastEvaluation)
                .join(
                    V35Forecast,
                    V35ForecastEvaluation.forecast_id == V35Forecast.id,
                )
                .where(
                    V35Forecast.model_market == market,
                    V35Forecast.protocol_version == self.protocol_version,
                    V35ForecastEvaluation.horizon_weeks == horizon,
                    V35ForecastEvaluation.evaluation_available_date <= as_of,
                )
                .order_by(V35ForecastEvaluation.evaluation_available_date)
            ).all()
            sample_count = len(evaluations)
            if sample_count < 30:
                calibrators[horizon] = None
                continue
            brier = float(np.mean([float(e.metrics_json.get("brier", 0.25)) for e in evaluations]))
            temperature = _temperature_from_brier(brier, sample_count)
            version = f"T{horizon:02d}_{as_of.isoformat()}"
            calibrator_id = f"{self.protocol_version}:{market}:{horizon}:{version}"
            row = session.get(V35ProbabilityCalibrator, calibrator_id)
            if row is None:
                payload = {
                    "horizon_weeks": horizon,
                    "temperature": temperature,
                    "brier": brier,
                    "sample_count": sample_count,
                }
                row = V35ProbabilityCalibrator(
                    id=calibrator_id,
                    protocol_version=self.protocol_version,
                    model_market=market,
                    horizon_weeks=horizon,
                    version=version,
                    model_family="weekly_ridge_residual_ensemble",
                    residual_schema_version=RESIDUAL_SCHEMA_VERSION,
                    fit_through_date=as_of,
                    effective_from_date=as_of,
                    raw_sample_count=sample_count,
                    effective_sample_count=sample_count,
                    status="FORMAL" if sample_count >= FORMAL_CALIBRATION_SAMPLES else "PRELIMINARY",
                    temperature=_pct(temperature),
                    metrics_json=payload,
                    calibration_hash=_hash(payload),
                    created_at=utc_now(),
                )
                session.add(row)
                session.flush()
            calibrators[horizon] = row
        return calibrators

    def _fit_calibrators_cached(
        self,
        session: Session,
        market: str,
        as_of: date,
    ) -> dict[int, V35ProbabilityCalibrator | None]:
        """Refit calibrators only when new evaluations arrived."""

        latest_id = session.scalar(
            select(func.max(V35ForecastEvaluation.id))
            .join(V35Forecast, V35ForecastEvaluation.forecast_id == V35Forecast.id)
            .where(
                V35Forecast.model_market == market,
                V35Forecast.protocol_version == self.protocol_version,
            )
        )
        cached = self._calibrator_cache.get(market)
        if cached is not None and cached[0] == latest_id:
            return cached[1]
        fitted = self._fit_calibrators(session, market, as_of)
        self._calibrator_cache[market] = (latest_id, fitted)
        return fitted

    def _generate_forecast(
        self,
        session: Session,
        market: str,
        anchor: date,
        package: V35ModelPackage,
        snapshot_row: V35FeatureSnapshotRow,
    ) -> V35Forecast:
        existing = session.scalar(
            select(V35Forecast).where(
                V35Forecast.model_market == market,
                V35Forecast.forecast_anchor_date == anchor,
                V35Forecast.protocol_version == self.protocol_version,
            )
        )
        if existing is not None:
            return existing
        state = self._load_model_state(session, package)
        snapshot = replace(
            self._empty_snapshot(market, anchor),
            features=dict(snapshot_row.feature_json),
            daily_sequence=tuple(snapshot_row.daily_sequence_json),
            source_data_max_date=snapshot_row.source_max_date,
            provenance=dict(snapshot_row.provenance_json),
        )
        forecast = self._training.predict(state, snapshot)
        expected = np.asarray(forecast.expected_path, dtype=float)
        model_config = self._v351_model_config(package)
        daily_mode = str(
            model_config.get("daily_adjustment_mode", "DAILY_ADJUSTMENT_OFF")
        )
        residual_scale = float(model_config.get("residual_scale", 1.0))
        residual_weighting = str(
            model_config.get("residual_weighting", "UNIFORM")
        )
        if str(model_config.get("regime_correction", "OFF")) == "ON":
            state_code = snapshot.features.get("v35_market_state")
            regime_signal = (
                1.0
                if state_code == 1.0
                else -1.0
                if state_code == -1.0
                else 0.0
            )
            ret_proxy = float(snapshot.features.get("v35_ret_26") or 0.0)
            regime_strength = float(
                np.clip(
                    ret_proxy / REGIME_CORRECTION_STRENGTH_DENOM,
                    -1.0,
                    1.0,
                )
            )
            regime_adjust = float(
                np.clip(
                    REGIME_CORRECTION_SIGNAL_SCALE
                    * regime_signal
                    * regime_strength,
                    -REGIME_CORRECTION_MAX_PER_WEEK,
                    REGIME_CORRECTION_MAX_PER_WEEK,
                )
            )
            expected[:REGIME_CORRECTION_WEEKS] += regime_adjust
        if daily_mode != "DAILY_ADJUSTMENT_OFF" and len(snapshot.daily_sequence) >= 6:
            closes = [
                float(row["close"])
                for row in snapshot.daily_sequence
                if row.get("close") is not None
            ]
            if len(closes) >= 6 and closes[-6] > 0:
                momentum = closes[-1] / closes[-6] - 1.0
                adjust = float(np.clip(momentum * 0.25, -0.02, 0.02))
                state_code = snapshot.features.get("v35_market_state")
                gated_ok = state_code in (1.0, 2.0)
                if daily_mode == "DAILY_ADJUSTMENT_FIXED" or (
                    daily_mode == "DAILY_ADJUSTMENT_REGIME_GATED" and gated_ok
                ):
                    expected[: min(2, len(expected))] += adjust
        residual_pool = tuple(
            session.scalars(
                select(V35ResidualRecord)
                .where(V35ResidualRecord.model_market == market)
                .order_by(V35ResidualRecord.matured_at.desc())
            ).all()
        )[:200]
        scenario_count = 1000
        random_plan = self._random_plan(
            session,
            market,
            anchor,
            package.id,
            residual_pool,
            scenario_count,
            residual_multiplier=residual_scale,
            residual_weighting=residual_weighting,
            half_life_weeks=RESIDUAL_HALF_LIFE_WEEKS,
        )
        pool = [
            np.asarray(json.loads(row.standardized_residual_json) if isinstance(row.standardized_residual_json, str) else row.standardized_residual_json, dtype=float)
            for row in residual_pool
        ]
        sigmas = [
            float(row.source_sigma or 1.0)
            for row in residual_pool
        ]
        payload_data = json.loads(zlib.decompress(random_plan.plan_payload_zlib).decode("utf-8"))
        indices = payload_data["indices"]
        multiplier = float(payload_data.get("residual_multiplier", 1.0))
        scenarios = np.empty((scenario_count, HORIZON_WEEKS), dtype=float)
        for row_index, index_row in enumerate(indices):
            residual_path = np.zeros(HORIZON_WEEKS, dtype=float)
            if pool:
                for week_index, pool_index in enumerate(index_row):
                    index = int(pool_index) % len(pool)
                    residual_path[week_index] = pool[index][week_index] * sigmas[index]
            scenarios[row_index] = expected + multiplier * residual_path
        p10 = np.quantile(scenarios, 0.10, axis=0)
        p50 = np.quantile(scenarios, 0.50, axis=0)
        p90 = np.quantile(scenarios, 0.90, axis=0)
        thresholds = {
            "4": float(np.std(expected[:4]) if len(expected) >= 4 else 0.02),
            "8": float(np.std(expected) if len(expected) else 0.02),
        }
        from backend.app.services.v351_payload_service import (
            load_history,
            predicted_series_and_indicators,
        )

        history = load_history(session, market, anchor)
        anchor_close = history["anchor_close"]
        predicted = (
            predicted_series_and_indicators(
                anchor_close=anchor_close,
                expected_path=expected.tolist(),
                p10_path=p10.tolist(),
                p90_path=p90.tolist(),
                history_closes=history["closes"],
                last_volume=history["last_volume"],
            )
            if anchor_close is not None
            else {"candles": [], "indicators": []}
        )
        sigma = forecast.forecast_sigma
        raw_probabilities = {
            "4": _probabilities_from_expected(float(expected[3]), sigma, thresholds["4"]),
            "8": _probabilities_from_expected(float(expected[-1]), sigma, thresholds["8"]),
        }
        calibrators = self._fit_calibrators_cached(session, market, anchor)
        horizon_probabilities: dict[str, list[float]] = {}
        calibrator_versions: dict[str, str | None] = {}
        for horizon in ("4", "8"):
            calibrator = calibrators[int(horizon)]
            if calibrator is None:
                horizon_probabilities[horizon] = raw_probabilities[horizon]
                calibrator_versions[horizon] = None
            else:
                horizon_probabilities[horizon] = _apply_temperature(
                    raw_probabilities[horizon], float(calibrator.temperature)
                )
                calibrator_versions[horizon] = calibrator.version
        ood = forecast.ood_diagnostics
        ood_score = float(ood["ood_score"])
        calibration_status = (
            "FORMAL"
            if all(calibrators.get(int(h)) is not None for h in ("4", "8"))
            and calibrators[4].status == "FORMAL"
            else "PRELIMINARY"
            if any(calibrators.get(int(h)) is not None for h in ("4", "8"))
            else "NONE"
        )
        reliability, health = _reliability_components(
            raw_sample_count=state.raw_matured_sample_count,
            calibration_status=calibration_status,
            ood_score=ood_score,
            recent_mae=state.validation_metrics.get("mean_oos_path_mae"),
        )
        label_end = self._anchor_after(session, market, anchor, steps=HORIZON_WEEKS)
        payload = {
            "protocol_version": self.protocol_version,
            "market": market,
            "anchor": anchor.isoformat(),
            "model_version": state.version,
            "package_id": package.id,
            "expected_path": list(expected),
            "p10": p10.tolist(),
            "p50": p50.tolist(),
            "p90": p90.tolist(),
            "probabilities": horizon_probabilities,
            "thresholds": thresholds,
            "ood": ood,
            "reliability": reliability,
            "health": health,
        }
        forecast_hash = _hash(payload)
        forecast_row = V35Forecast(
            protocol_version=self.protocol_version,
            model_market=market,
            target_instrument_id=self._instrument_id(session, market),
            feature_snapshot_id=snapshot_row.id,
            model_version_id=state.version,
            model_package_id=package.id,
            random_plan_id=random_plan.id,
            forecast_anchor_date=anchor,
            label_end_date=label_end,
            horizon_weeks=HORIZON_WEEKS,
            maturity_status="PENDING",
            scenario_seed=random_plan.seed,
            scenario_count=scenario_count,
            random_plan_hash=random_plan.plan_hash,
            expected_path_json=list(expected),
            representative_ohlcv_json=predicted["candles"],
            indicator_path_json=predicted["indicators"],
            price_quantiles_json=[p10.tolist(), p50.tolist(), p90.tolist()],
            horizon_probabilities_json=horizon_probabilities,
            thresholds_json=thresholds,
            calibrator_versions_json=calibrator_versions,
            residual_pool_json={
                "pool_size": len(residual_pool),
                "pool_identity": random_plan.residual_pool_identity,
            },
            model_reliability_score=_pct(reliability),
            reliability_components_json={"calibration_status": calibration_status},
            health_status=health,
            ood_json=ood,
            payload_json=payload,
            data_source_provenance=(
                self._data_provenance(session, market, anchor) if self.v36 else None
            ),
            forecast_hash=forecast_hash,
            created_at=utc_now(),
        )
        session.add(forecast_row)
        session.flush()
        for horizon in ("4", "8"):
            calibrator = calibrators[int(horizon)]
            if calibrator is None:
                continue
            session.add(
                V35ForecastCalibratorLink(
                    forecast_id=forecast_row.id,
                    calibrator_id=calibrator.id,
                    horizon_weeks=int(horizon),
                    created_at=utc_now(),
                )
            )
        session.add(
            V35HealthSnapshot(
                protocol_version=self.protocol_version,
                model_market=market,
                anchor_date=anchor,
                model_version_id=state.version,
                health_status=health,
                diagnostics_json=ood,
                reliability_score=_pct(reliability),
                health_hash=_hash({"reliability": reliability, "health": health, "ood": ood}),
                created_at=utc_now(),
            )
        )
        return forecast_row

    def _anchor_after(
        self,
        session: Session,
        market: str,
        anchor: date,
        *,
        steps: int,
    ) -> date | None:
        anchors = self._anchors(session, market)
        try:
            index = anchors.index(anchor)
        except ValueError:
            return None
        target = index + steps
        return anchors[target] if target < len(anchors) else None

    def _mature_forecasts(
        self,
        session: Session,
        market: str,
        as_of: date,
    ) -> tuple[V35Forecast, ...]:
        anchors = self._anchors(session, market)
        close_by_anchor = {
            anchor: float(bar["close"])
            for anchor, bar in load_weekly_bars(session, market, anchors).items()
        }
        forecasts = session.scalars(
            select(V35Forecast)
            .where(
                V35Forecast.model_market == market,
                V35Forecast.protocol_version == self.protocol_version,
                V35Forecast.maturity_status == "PENDING",
                V35Forecast.forecast_anchor_date < as_of,
            )
            .order_by(V35Forecast.forecast_anchor_date)
        ).all()
        matured: list[V35Forecast] = []
        for forecast in forecasts:
            anchor = forecast.forecast_anchor_date
            try:
                index = anchors.index(anchor)
            except ValueError:
                continue
            end_index = index + HORIZON_WEEKS
            if end_index >= len(anchors) or anchors[end_index] > as_of:
                continue
            end_anchor = anchors[end_index]
            start_close = close_by_anchor.get(anchor)
            if start_close is None or start_close <= 0.0:
                continue
            path = tuple(
                close_by_anchor.get(anchors[index + step], 0.0) / start_close - 1.0
                for step in range(1, HORIZON_WEEKS + 1)
            )
            if any(not math.isfinite(value) for value in path):
                continue
            forecast.maturity_status = "FULLY_MATURE_8W"
            for horizon in (4, 8):
                actual = path[horizon - 1]
                threshold = float(forecast.thresholds_json.get(str(horizon), 0.02))
                raw_prob = forecast.horizon_probabilities_json.get(str(horizon), [0.34, 0.33, 0.33])
                actual_class = _actual_class(actual, threshold)
                brier = float(np.mean((np.asarray(raw_prob) - np.asarray([1.0 if actual_class == "UP" else 0.0, 1.0 if actual_class == "SIDEWAYS" else 0.0, 1.0 if actual_class == "DOWN" else 0.0])) ** 2))
                expected = float(forecast.expected_path_json[horizon - 1])
                metrics = {
                    "actual_return": actual,
                    "expected_return": expected,
                    "absolute_error": abs(actual - expected),
                    "brier": brier,
                    "direction": actual_class,
                }
                existing = session.scalar(
                    select(V35ForecastEvaluation).where(
                        V35ForecastEvaluation.forecast_id == forecast.id,
                        V35ForecastEvaluation.horizon_weeks == horizon,
                    )
                )
                if existing is None:
                    session.add(
                        V35ForecastEvaluation(
                            forecast_id=forecast.id,
                            horizon_weeks=horizon,
                            evaluation_version="V3.5_EVAL_1",
                            evaluation_available_date=end_anchor,
                            actual_return=_pct(actual),
                            actual_class=actual_class,
                            metrics_json=metrics,
                            evaluation_hash=_hash(metrics),
                            created_at=utc_now(),
                        )
                    )
            if session.scalar(
                select(V35ResidualRecord).where(V35ResidualRecord.forecast_id == forecast.id)
            ) is None:
                expected_path = np.asarray(forecast.expected_path_json, dtype=float)
                actual_path = np.asarray(path, dtype=float)
                residual_path = actual_path - expected_path
                sigma = max(float(np.std(np.diff(np.concatenate(([0.0], residual_path))))), 0.005)
                session.add(
                    V35ResidualRecord(
                        forecast_id=forecast.id,
                        model_market=market,
                        matured_at=end_anchor,
                        model_family="weekly_ridge_residual_ensemble",
                        residual_schema_version=RESIDUAL_SCHEMA_VERSION,
                        return_unit="CUMULATIVE_8W",
                        prediction_path_json=list(expected_path),
                        actual_path_json=actual_path.tolist(),
                        residual_path_json=residual_path.tolist(),
                        standardized_residual_json=(residual_path / max(sigma, 1e-9)).tolist(),
                        source_sigma=_pct(sigma),
                        residual_hash=_hash({"forecast": forecast.id, "path": residual_path.tolist()}),
                        created_at=utc_now(),
                    )
                )
            matured.append(forecast)
        return tuple(matured)

    def _decisions_for_anchor(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> V35StrategyDecision | None:
        snapshot = session.scalar(
            select(V35StrategySnapshot).where(
                V35StrategySnapshot.model_market == market,
                V35StrategySnapshot.protocol_version == self.protocol_version,
                V35StrategySnapshot.forecast_anchor_date == anchor,
            )
        )
        if snapshot is None:
            return None
        batches = session.scalars(
            select(V35PositionDecision)
            .where(V35PositionDecision.strategy_snapshot_id == snapshot.id)
            .order_by(V35PositionDecision.batch_number)
        ).all()
        return V35StrategyDecision(
            market=market,
            forecast_anchor_date=snapshot.forecast_anchor_date.isoformat(),
            dif_trend_state=snapshot.dif_trend_state,
            confirmation_status=snapshot.confirmation_status,
            market_state="",
            strategy_score=float(snapshot.strategy_score),
            base_target_position_pp=snapshot.base_target_position_pp,
            state_position_cap_pp=snapshot.state_position_cap_pp,
            final_target_position_pp=snapshot.final_target_position_pp,
            batches=tuple(
                {
                    "batch_number": row.batch_number,
                    "action": row.action,
                    "position_pp": row.position_pp,
                    "batch_change_pp": row.batch_change_pp,
                    "target_position_pp": row.target_position_pp,
                    "cooldown_trading_days": int(
                        (row.condition_json or {}).get("cooldown_trading_days", 5)
                    ),
                    "condition": row.condition_json.get("condition", ""),
                }
                for row in batches
            ),
            components={},
            reasons=(),
        )

    def _activity_health(
        self,
        session: Session,
        market: str,
        anchor: date,
        package: V35ModelPackage,
    ) -> dict[str, Any]:
        """Post-hoc activity health from already-matured strong-up windows."""

        if not self.v36:
            return {}
        threshold = strong_up_threshold_for(market)
        lookback_start = anchor - timedelta(weeks=ACTIVITY_LOOKBACK_WEEKS)
        rows = session.execute(
            select(
                V35Forecast.forecast_anchor_date,
                V35Forecast.label_end_date,
                V35ForecastEvaluation.actual_return,
            )
            .join(
                V35ForecastEvaluation,
                V35ForecastEvaluation.forecast_id == V35Forecast.id,
            )
            .where(
                V35Forecast.model_market == market,
                V35Forecast.protocol_version == self.protocol_version,
                V35ForecastEvaluation.horizon_weeks == 8,
                V35ForecastEvaluation.evaluation_available_date <= anchor,
                V35ForecastEvaluation.evaluation_available_date >= lookback_start,
            )
        ).all()
        strong_windows = [
            (label_end, float(actual_return))
            for _anchor, label_end, actual_return in rows
            if label_end is not None
            and actual_return is not None
            and float(actual_return) >= threshold
        ]
        exposures: list[float] = []
        for label_end, _actual in strong_windows:
            evaluation = session.scalar(
                select(V35SimEvaluation)
                .where(
                    V35SimEvaluation.model_market == market,
                    V35SimEvaluation.model_package_id == package.id,
                    V35SimEvaluation.window_end_date == label_end,
                )
                .limit(1)
            )
            if evaluation is not None:
                exposures.append(float(evaluation.average_position_pp))
        avg_exposure = float(np.mean(exposures)) if exposures else 0.0
        degraded = bool(
            len(strong_windows) >= ACTIVITY_DEGRADED_MIN_STRONG_UP_WINDOWS
            and avg_exposure < ACTIVITY_DEGRADED_MAX_AVG_EXPOSURE_PP
        )
        return {
            "status": (
                "ACTIVITY_DEGRADED" if degraded else "ACTIVITY_HEALTHY"
            ),
            "strong_up_window_count": len(strong_windows),
            "strong_up_exposure_pp": round(avg_exposure, 2),
            "strong_up_miss_ratio": round(
                1.0 - avg_exposure / 100.0, 4
            )
            if exposures
            else 1.0,
            "up_opportunity_participation": round(avg_exposure / 100.0, 4),
        }

    def _data_provenance(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> dict[str, Any]:
        from backend.app.models.models import AppSetting

        latest_source = None
        instrument = session.scalar(select(Instrument).where(Instrument.code == market))
        if instrument is not None:
            latest_source = session.scalar(
                select(MarketPrice.source)
                .where(
                    MarketPrice.instrument_id == instrument.id,
                    MarketPrice.timeframe == "daily",
                    MarketPrice.trade_date <= anchor,
                )
                .order_by(MarketPrice.trade_date.desc())
                .limit(1)
            )
        calendar_fingerprint = None
        try:
            import akshare
            import hashlib

            calendar_path = (
                Path(akshare.__file__).resolve().parent / "file_fold" / "calendar.json"
            )
            if calendar_path.is_file():
                calendar_fingerprint = hashlib.sha256(
                    calendar_path.read_bytes()
                ).hexdigest()
        except Exception:  # noqa: BLE001 - provenance must never raise
            calendar_fingerprint = None
        refresh = session.scalar(
            select(AppSetting).where(AppSetting.key == "market_refresh_status")
        )
        refresh_status = dict(refresh.value or {}) if refresh is not None else {}
        return {
            "market_data_source": str(latest_source) if latest_source else None,
            "akshare_calendar_fingerprint": calendar_fingerprint,
            "refresh_run_id": refresh_status.get("last_success_at"),
            "anchor": anchor.isoformat(),
        }

    def _persist_strategy(
        self,
        session: Session,
        market: str,
        anchor: date,
        forecast: V35Forecast,
        package: V35ModelPackage,
        decision: V35StrategyDecision,
        *,
        turning_assessment: Mapping[str, Any] | None = None,
        funnel: Mapping[str, Any] | None = None,
        activity_health: Mapping[str, Any] | None = None,
    ) -> None:
        existing = session.scalar(
            select(V35StrategySnapshot).where(
                V35StrategySnapshot.model_market == market,
                V35StrategySnapshot.protocol_version == self.protocol_version,
                V35StrategySnapshot.forecast_anchor_date == anchor,
                V35StrategySnapshot.model_package_id == package.id,
            )
        )
        if existing is not None:
            return
        payload = {
            "market": market,
            "anchor": anchor.isoformat(),
            "package": package.id,
            "dif_trend_state": decision.dif_trend_state,
            "confirmation_status": decision.confirmation_status,
            "market_state": decision.market_state,
            "strategy_score": decision.strategy_score,
            "base_target_position_pp": decision.base_target_position_pp,
            "state_position_cap_pp": decision.state_position_cap_pp,
            "final_target_position_pp": decision.final_target_position_pp,
            "components": dict(decision.components),
            "reasons": list(decision.reasons),
            "batches": [dict(batch) for batch in decision.batches],
        }
        if turning_assessment is not None:
            payload["turning_assessment"] = dict(turning_assessment)
        if funnel is not None:
            payload["decision_funnel"] = dict(funnel)
        if activity_health is not None:
            payload["activity_health"] = dict(activity_health)
        strategy_hash = _hash(payload)
        row = V35StrategySnapshot(
            protocol_version=self.protocol_version,
            model_market=market,
            forecast_id=forecast.id,
            model_package_id=package.id,
            forecast_anchor_date=anchor,
            dif_trend_state=decision.dif_trend_state,
            confirmation_status=decision.confirmation_status,
            strategy_score=_pct(decision.strategy_score),
            base_target_position_pp=decision.base_target_position_pp,
            state_position_cap_pp=decision.state_position_cap_pp,
            final_target_position_pp=decision.final_target_position_pp,
            strategy_json=payload,
            data_source_provenance=(
                self._data_provenance(session, market, anchor) if self.v36 else None
            ),
            strategy_hash=strategy_hash,
            created_at=utc_now(),
        )
        session.add(row)
        session.flush()
        for batch_number, batch in enumerate(decision.batches, start=1):
            session.add(
                V35PositionDecision(
                    strategy_snapshot_id=row.id,
                    forecast_id=forecast.id,
                    batch_number=batch_number,
                    action=str(batch["action"]),
                    position_pp=int(batch["position_pp"]),
                    batch_change_pp=int(batch["batch_change_pp"]),
                    target_position_pp=int(batch["target_position_pp"]),
                    execution_anchor_date=None,
                    condition_json=dict(batch),
                    decision_hash=_hash({"strategy": strategy_hash, "batch": batch_number, "data": batch}),
                    created_at=utc_now(),
                )
            )

    def _ensure_champion(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> V35ModelPackage | None:
        existing = self._champion_package(session, market, anchor)
        if existing is not None:
            return existing
        matured = self._matured_samples(session, market, anchor)
        threshold = min_train_samples_for(market)
        small_sample = (
            market != "399006"
            and PREWARMING_THRESHOLD <= len(matured) < threshold
        )
        if len(matured) < PREWARMING_THRESHOLD or (
            len(matured) < threshold and not small_sample
        ):
            return None
        snapshot_rows = self._feature_snapshots(session, market, through=anchor)
        if not snapshot_rows:
            return None
        manifest = self._features.build_manifest(
            replace(
                self._empty_snapshot(market, anchor),
                features=dict(snapshot_rows[-1].feature_json),
            )
        )
        profile = self._persist_profile(session, market, manifest.ordered_feature_names, manifest.manifest_hash)
        validation = self._training.validation_metrics_for(
            market,
            matured,
            alpha=float(INITIAL_PACKAGE_DEFAULTS["alpha"]),
            feature_names=manifest.ordered_feature_names,
        )
        trained_through = anchor
        next_anchor = self._anchor_after(session, market, anchor, steps=1)
        if next_anchor is None:
            return None
        version = f"{self.protocol_version}:{market}:INITIAL:{anchor.isoformat()}"
        state = self._training.fit(
            market,
            matured,
            version=version,
            parent_version=None,
            trained_through=trained_through,
            effective_from=next_anchor,
            alpha=float(INITIAL_PACKAGE_DEFAULTS["alpha"]),
            feature_names=manifest.ordered_feature_names,
            validation_metrics=validation,
        )
        self._persist_model(
            session,
            market,
            state,
            status="ACTIVE",
            metrics=validation,
            promotion_gate={
                "kind": "SMALL_SAMPLE_INITIAL_FROM_SCRATCH"
                if small_sample
                else "INITIAL_FROM_SCRATCH"
            },
        )
        strategy_config = default_strategy_config(market)
        package = self._persist_package(
            session,
            market,
            version=version,
            kind="SMALL_SAMPLE_CHAMPION" if small_sample else "CHAMPION",
            prediction_model_id=state.version,
            strategy_config=strategy_config,
            effective_from=next_anchor,
        )
        bootstrap = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == self.protocol_version,
            )
        )
        if bootstrap is None:
            bootstrap = V35BootstrapState(
                protocol_version=self.protocol_version,
                model_market=market,
                state="RUNNING",
                first_formal_anchor=next_anchor,
                weekly_iteration_count=0,
                prediction_challenge_count=0,
                strategy_challenge_count=0,
                promotion_count=0,
                champion_package_id=package.id,
                state_json={"initial_model_anchor": anchor.isoformat()},
                state_hash=_hash({"initial": anchor.isoformat(), "package": package.id}),
                updated_at=utc_now(),
            )
            session.add(bootstrap)
        else:
            bootstrap.champion_package_id = package.id
            bootstrap.first_formal_anchor = bootstrap.first_formal_anchor or next_anchor
            bootstrap.state = "RUNNING"
            bootstrap.state_json = {
                **bootstrap.state_json,
                "initial_model_anchor": anchor.isoformat(),
                "initial_package": package.id,
            }
            bootstrap.updated_at = utc_now()
        return package

    def _run_window_accounts(
        self,
        session: Session,
        market: str,
        as_of: date,
        anchors: Sequence[date],
        package: V35ModelPackage,
        bars: Mapping[date, dict[str, float | None]],
    ) -> None:
        config = dict(package.strategy_config_json)
        try:
            end_index = anchors.index(as_of)
        except ValueError:
            return
        window_index = end_index - HORIZON_WEEKS + 1
        if window_index < 0:
            return
        window = anchors[window_index : window_index + HORIZON_WEEKS]
        if len(window) < HORIZON_WEEKS or window[-1] > as_of:
            return
        existing_account = session.scalar(
            select(V35SimAccount).where(
                V35SimAccount.model_market == market,
                V35SimAccount.scope == "STANDARD_8W",
                V35SimAccount.window_start_date == window[0],
                V35SimAccount.model_package_id == package.id,
            )
        )
        if existing_account is not None and existing_account.status == "COMPLETED":
            return
        decisions = {
            anchor: decision
            for anchor in window
            if (decision := self._decisions_for_anchor(session, market, anchor)) is not None
        }
        if len(decisions) != HORIZON_WEEKS:
            return
        result = run_standard_window(
            session,
            market=market,
            package_id=package.id,
            anchors=window,
            decisions=decisions,
            bars=bars,
            config=config,
            accounting_mode="SHARES" if self.v36 else "PERCENT",
            initial_snapshot=(
                self._snapshot_for(session, market, anchors[window_index - 1])
                if self.v36 and window_index > 0
                else None
            ),
        )
        persist_window_result(
            session,
            result,
            protocol_version=self.protocol_version,
        )

    def _process_week(
        self,
        session: Session,
        market: str,
        anchor: date,
        run_id: str,
        online: _OnlineAccount,
        bars: Mapping[date, dict[str, float | None]],
    ) -> dict[str, Any]:
        try:
            snapshot = self._features.load_snapshot(session, market, anchor)
        except (V35FeatureError, V34FeatureError) as exc:
            return {"status": "WARMUP", "anchor": anchor.isoformat(), "reason": str(exc)}
        feature_manifest = self._features.build_manifest(snapshot)
        snapshot_row = self._persist_feature(session, market, anchor, snapshot, feature_manifest)

        package = self._champion_package(session, market, anchor)
        if package is None:
            staged = self._staged_champion_package(session, market)
            if staged is not None:
                return {
                    "status": "WARMUP",
                    "anchor": anchor.isoformat(),
                    "reason": "champion staged and effective next week",
                }
            package = self._ensure_champion(session, market, anchor)
        if package is None:
            return {"status": "WARMUP", "anchor": anchor.isoformat()}
        if package.effective_from_date > anchor:
            return {
                "status": "WARMUP",
                "anchor": anchor.isoformat(),
                "reason": "champion effective from next complete week",
            }

        forecast_row = self._generate_forecast(session, market, anchor, package, snapshot_row)
        decision = self._strategy.decide(
            market,
            snapshot,
            expected_path=forecast_row.expected_path_json,
            p10_path=forecast_row.price_quantiles_json[0],
            probabilities_4=forecast_row.horizon_probabilities_json.get("4", [0.34, 0.33, 0.33]),
            probabilities_8=forecast_row.horizon_probabilities_json.get("8", [0.34, 0.33, 0.33]),
            current_position_pp=online.position_pp,
            reliability_score=float(forecast_row.model_reliability_score),
            health_status=forecast_row.health_status,
            ood_score=float(forecast_row.ood_json.get("ood_score", 0.0)),
            config=dict(package.strategy_config_json),
        )
        from backend.app.services.v351_payload_service import assess_turning_points

        indicator_path = forecast_row.indicator_path_json
        turning_assessment = assess_turning_points(
            expected_path=forecast_row.expected_path_json,
            predicted_dif=[float(row["dif"]) for row in indicator_path],
            predicted_dif_first=[
                float(row["dif_first_change"]) for row in indicator_path
            ],
            ma20_distance=snapshot.features.get("v35_ma20_distance"),
            ma20_slope=snapshot.features.get("v35_ma20_slope"),
            probabilities_8=forecast_row.horizon_probabilities_json.get(
                "8", [0.34, 0.33, 0.33]
            ),
        )
        probabilities_8 = forecast_row.horizon_probabilities_json.get(
            "8", [0.34, 0.33, 0.33]
        )
        bull_signal = int(
            float(probabilities_8[0]) > float(probabilities_8[2])
            or float(forecast_row.expected_path_json[-1]) > 0.0
        )
        base_positive = int(decision.base_target_position_pp > 0)
        funnel = {
            "bull_signal_count": bull_signal,
            "score_pass_count": int(decision.strategy_score > 0.0),
            "base_position_positive_count": base_positive,
            "risk_blocked_count": int(
                base_positive and decision.final_target_position_pp == 0
            ),
            "cooldown_blocked_count": 0,
            "final_position_positive_count": int(
                decision.final_target_position_pp > 0
            ),
            "trade_count": len(decision.batches),
            "actual_position_positive_count": int(online.position_pp > 0),
        }
        activity_health = self._activity_health(session, market, anchor, package)
        self._persist_decision_funnel(
            session, market, anchor, package.id, funnel
        )
        self._persist_strategy(
            session,
            market,
            anchor,
            forecast_row,
            package,
            decision,
            turning_assessment=turning_assessment,
            funnel=funnel,
            activity_health=activity_health,
        )
        online.set_decision(decision)
        if self.v36:
            self._persist_account_snapshot(session, market, anchor, online)

        matured = self._mature_forecasts(session, market, anchor)
        self._run_window_accounts(
            session,
            market,
            anchor,
            self._anchors(session, market),
            package,
            bars,
        )
        challenge_result = self._process_challenges(session, market, anchor, run_id)
        if self.v351:
            deferred = self._evaluate_pending_v351(session, market, anchor)
            if deferred:
                challenge_result["deferred_promotions"] = deferred
        iteration_number = self._iteration_count(session, market) + 1
        session.add(
            V35TrainingIteration(
                protocol_version=self.protocol_version,
                training_run_id=run_id,
                forecast_id=forecast_row.id,
                model_market=market,
                anchor_date=anchor,
                weekly_iteration_number=iteration_number,
                forecast_model_id=package.prediction_model_id,
                champion_after_model_id=package.prediction_model_id,
                training_triggered=bool(challenge_result.get("challenge_created")),
                promoted=bool(challenge_result.get("promoted")),
                newly_matured_count=len(matured),
                raw_matured_sample_count=0,
                effective_independent_sample_count=0,
                validation_json=challenge_result,
                completed_at=utc_now(),
            )
        )
        bootstrap = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == self.protocol_version,
            )
        )
        if bootstrap is not None:
            bootstrap.last_completed_anchor = anchor
            bootstrap.weekly_iteration_count = iteration_number
            bootstrap.updated_at = utc_now()
        return {
            "status": "PROCESSED",
            "anchor": anchor.isoformat(),
            "forecast_id": forecast_row.id,
            "newly_matured": len(matured),
            "challenge": challenge_result,
        }

    def _iteration_count(self, session: Session, market: str) -> int:
        bootstrap = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == self.protocol_version,
            )
        )
        return int(bootstrap.weekly_iteration_count) if bootstrap is not None else 0

    def _process_challenges(
        self,
        session: Session,
        market: str,
        anchor: date,
        run_id: str,
    ) -> dict[str, Any]:
        bootstrap = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == self.protocol_version,
            )
        )
        if bootstrap is None or bootstrap.champion_package_id is None:
            return {}
        champion = session.get(V35ModelPackage, bootstrap.champion_package_id)
        if champion is None:
            return {}
        state = bootstrap.state_json
        prediction_pending = int(state.get("prediction_pending_matured", 0))
        strategy_pending = int(state.get("strategy_pending_matured", 0))
        maintenance_pending = int(state.get("maintenance_pending_matured", 0))
        newly_matured = len(
            session.scalars(
                select(V35Forecast).where(
                    V35Forecast.model_market == market,
                    V35Forecast.protocol_version == self.protocol_version,
                    V35Forecast.maturity_status == "FULLY_MATURE_8W",
                    V35Forecast.forecast_anchor_date < anchor,
                )
            ).all()
        ) - int(state.get("matured_total_baseline", 0))
        prediction_pending += newly_matured
        strategy_pending += newly_matured
        maintenance_pending += newly_matured
        result: dict[str, Any] = {"newly_matured": newly_matured}
        if self.v351 and maintenance_pending >= self._cfg_maintenance_every():
            result["maintenance"] = self._maintenance_refresh(
                session, market, anchor, champion
            )
            maintenance_pending = 0
        prediction_due = prediction_pending >= self._cfg_prediction_every()
        strategy_due = strategy_pending >= self._cfg_strategy_every()
        family: str | None = None
        if prediction_due:
            family = "PREDICTION"
            prediction_pending = 0
        elif strategy_due:
            family = "STRATEGY"
            strategy_pending = 0
        if family is not None:
            challenge_result = self._run_challenge(
                session, market, anchor, champion, family, run_id
            )
            result.update(challenge_result)
            if family == "PREDICTION":
                bootstrap.prediction_challenge_count += 1
            else:
                bootstrap.strategy_challenge_count += 1
            if challenge_result.get("promoted"):
                bootstrap.promotion_count += 1
                bootstrap.champion_package_id = challenge_result["new_champion_id"]
        bootstrap.state_json = {
            **state,
            "prediction_pending_matured": prediction_pending,
            "strategy_pending_matured": strategy_pending,
            "maintenance_pending_matured": maintenance_pending,
            "matured_total_baseline": int(
                state.get("matured_total_baseline", 0)
            ) + newly_matured,
            "last_processed_anchor": anchor.isoformat(),
        }
        bootstrap.updated_at = utc_now()
        return result

    def _run_challenge(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
        family: str,
        run_id: str,
    ) -> dict[str, Any]:
        if self.v351:
            return self._run_challenge_v351(
                session, market, anchor, champion, family, run_id
            )
        challenge_id = f"{self.protocol_version}:{market}:{anchor.isoformat()}:{family}"
        existing = session.get(V35Challenge, challenge_id)
        if existing is not None:
            return {"challenge_created": False, "existing": challenge_id}
        challengers: list[V35ModelPackage] = []
        if family == "PREDICTION":
            champion_model = session.get(V35ModelVersion, champion.prediction_model_id)
            alphas = candidate_alphas(float(champion_model.ridge_alpha))
            matured = self._matured_samples(session, market, anchor)
            snapshot_rows = self._feature_snapshots(session, market, through=anchor)
            manifest = self._features.build_manifest(
                replace(
                    self._empty_snapshot(market, anchor),
                    features=dict(snapshot_rows[-1].feature_json) if snapshot_rows else {},
                )
            )
            for index, alpha in enumerate(alphas[:PREDICTION_CHALLENGER_COUNT]):
                version = f"{self.protocol_version}:{market}:PREDC:{anchor.isoformat()}:{index}"
                if session.get(V35ModelVersion, version) is not None:
                    state = state_from_payload(session.get(V35ModelVersion, version).parameters_json)
                else:
                    validation = self._training.validation_metrics_for(
                        market,
                        matured,
                        alpha=alpha,
                        feature_names=manifest.ordered_feature_names,
                    )
                    next_anchor = self._anchor_after(session, market, anchor, steps=1)
                    state = self._training.fit(
                        market,
                        matured,
                        version=version,
                        parent_version=champion.prediction_model_id,
                        trained_through=anchor,
                        effective_from=next_anchor or anchor,
                        alpha=alpha,
                        feature_names=manifest.ordered_feature_names,
                        validation_metrics=validation,
                    )
                    self._persist_model(
                        session, market, state, status="CANDIDATE", metrics=validation,
                        promotion_gate={"family": family, "alpha": alpha},
                    )
                package_version = f"{version}:PKG"
                challenger = self._persist_package(
                    session,
                    market,
                    version=package_version,
                    kind="PREDICTION_CHALLENGER",
                    prediction_model_id=state.version,
                    strategy_config=dict(champion.strategy_config_json),
                    effective_from=anchor,
                    parent_package_id=champion.id,
                )
                challengers.append(challenger)
        else:
            config = dict(champion.strategy_config_json)
            variants = self._strategy_variants(market, config)
            for index, variant in enumerate(variants[:STRATEGY_CHALLENGER_COUNT]):
                version = f"{self.protocol_version}:{market}:STRATC:{anchor.isoformat()}:{index}"
                challenger = self._persist_package(
                    session,
                    market,
                    version=version,
                    kind="STRATEGY_CHALLENGER",
                    prediction_model_id=champion.prediction_model_id,
                    strategy_config=variant,
                    effective_from=anchor,
                    parent_package_id=champion.id,
                )
                challengers.append(challenger)
        challenge = V35Challenge(
            id=challenge_id,
            protocol_version=self.protocol_version,
            model_market=market,
            anchor_date=anchor,
            challenger_family=family,
            champion_package_id=champion.id,
            candidate_count=len(challengers),
            status="EVALUATING",
            result_json={"candidates": [item.id for item in challengers]},
            created_at=utc_now(),
        )
        session.add(challenge)
        session.flush()
        promoted: V35ModelPackage | None = None
        promotion_records: list[dict[str, Any]] = []
        for challenger in challengers:
            evaluation = self._evaluate_challenger(
                session, market, anchor, challenge.id, champion, challenger, family
            )
            promotion_records.append(evaluation)
            if evaluation.get("promotion_decision") == "PROMOTED":
                promoted = challenger
                break
        challenge.status = "COMPLETED"
        challenge.result_json = {
            "candidates": [item.id for item in challengers],
            "promotions": promotion_records,
        }
        result: dict[str, Any] = {"challenge_created": True, "family": family, "candidates": len(challengers)}
        if promoted is not None:
            next_anchor = self._anchor_after(session, market, anchor, steps=1)
            session.add(
                V35Promotion(
                    protocol_version=self.protocol_version,
                    model_market=market,
                    challenge_id=challenge.id,
                    champion_package_id=champion.id,
                    candidate_package_id=promoted.id,
                    challenger_family=family,
                    evaluation_window_count=8,
                    promotion_decision="PROMOTED",
                    promotion_reason="通过8周账户收益与风险晋级门禁",
                    effective_from_date=next_anchor,
                    excess_profit=_money(promotion_records[-1].get("excess_profit", 0.0)),
                    excess_return=_pct(promotion_records[-1].get("excess_return", 0.0)),
                    metrics_json=promotion_records[-1],
                    promotion_hash=_hash(promotion_records[-1]),
                    created_at=utc_now(),
                )
            )
            result["promoted"] = True
            result["new_champion_id"] = promoted.id
        return result

    def _strategy_variants(
        self,
        market: str,
        champion_config: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        variants: list[dict[str, Any]] = []
        caps = list(dict(champion_config["state_position_caps"]).items())
        for delta in (-10, 10, 20):
            variant = json.loads(json.dumps(dict(champion_config)))
            variant_caps = dict(champion_config["state_position_caps"])
            key, current = caps[2]
            candidate = max(0, min(80, current + delta))
            if candidate != current:
                variant_caps[key] = candidate
                variant["state_position_caps"] = variant_caps
                variant["policy_version"] = f"{variant.get('policy_version', 'POLICY_V35_1')}:V{delta:+d}"
                variants.append(variant)
        if not variants:
            variants = [dict(champion_config)]
        return tuple(variants)

    # ------------------------------------------------------------------
    # V3.5.1 challenge mechanism
    # ------------------------------------------------------------------

    def _v351_model_config(self, package: V35ModelPackage) -> dict[str, Any]:
        config = dict(package.strategy_config_json)
        return dict(config.get("v351_model_config", {}))

    def _v351_binding_stats(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
    ) -> dict[str, int]:
        stats = {key: 0 for key in self._cfg_strategy_groups()}
        windows = self._matured_windows(session, market, anchor)[-12:]
        for window in windows:
            for window_anchor in window:
                snap = session.scalar(
                    select(V35StrategySnapshot).where(
                        V35StrategySnapshot.model_market == market,
                        V35StrategySnapshot.forecast_anchor_date == window_anchor,
                        V35StrategySnapshot.model_package_id == champion.id,
                    )
                )
                if snap is None:
                    continue
                cap_key = f"state_cap_{snap.confirmation_status}"
                if (
                    cap_key in stats
                    and snap.final_target_position_pp == snap.state_position_cap_pp
                    and snap.state_position_cap_pp < snap.base_target_position_pp
                ):
                    stats[cap_key] += 1
                batches = session.scalars(
                    select(V35PositionDecision)
                    .where(V35PositionDecision.strategy_snapshot_id == snap.id)
                    .order_by(V35PositionDecision.batch_number)
                ).all()
                buys = [row for row in batches if row.action == "BUY"]
                sells = [row for row in batches if row.action == "SELL"]
                if buys and buys[0].batch_change_pp in (5, 10, 15):
                    stats["first_probe_buy"] += 1
                if any(row.batch_change_pp == 10 for row in buys):
                    stats["trend_add"] += 1
                if any(row.batch_change_pp in (15, 20) for row in buys):
                    stats["strong_trend_add"] += 1
                if sells and sells[0].batch_change_pp == 5:
                    stats["light_reduce"] += 1
                if sells and sells[0].batch_change_pp in (10, 15, 20):
                    stats["confirmed_reduce"] += 1
                if snap.confirmation_status == "TREND_CONFIRMED":
                    stats["trend_confirmation_weeks"] += 1
                ledger_rows = session.scalars(
                    select(V35SimLedger)
                    .join(V35SimAccount, V35SimLedger.account_id == V35SimAccount.id)
                    .where(
                        V35SimAccount.model_market == market,
                        V35SimAccount.window_start_date == window[0],
                    )
                ).all()
                if any(
                    int(row.event_json.get("pending_batches", 0)) > 0
                    for row in ledger_rows
                ):
                    stats["cooldown_days"] += 1
        return stats

    def _v351_behavior_rows(
        self,
        session: Session,
        market: str,
        anchors: Sequence[date],
        package: V35ModelPackage,
        *,
        recompute_forecast: bool = False,
    ) -> tuple[V351AnchorRecord, ...]:
        records: list[V351AnchorRecord] = []
        for anchor in anchors:
            expected_path: list[float] | None = None
            strategy_score: float | None = None
            base_target: int | None = None
            final_target: int | None = None
            signals: tuple[tuple[str, int], ...] = ()
            trade_path: tuple[tuple[str, int], ...] = ()
            if recompute_forecast:
                state = self._load_model_state(session, package)
                if anchor < state.effective_from_date:
                    continue
                snapshot_row = session.scalar(
                    select(V35FeatureSnapshotRow).where(
                        V35FeatureSnapshotRow.model_market == market,
                        V35FeatureSnapshotRow.forecast_anchor_date == anchor,
                    )
                )
                if snapshot_row is not None:
                    state = self._load_model_state(session, package)
                    snapshot = replace(
                        self._empty_snapshot(market, anchor),
                        features=dict(snapshot_row.feature_json),
                        daily_sequence=tuple(snapshot_row.daily_sequence_json),
                        source_data_max_date=snapshot_row.source_max_date,
                        provenance=dict(snapshot_row.provenance_json),
                    )
                    forecast = self._training.predict(state, snapshot)
                    path = list(forecast.expected_path)
                    model_config = self._v351_model_config(package)
                    daily_mode = str(
                        model_config.get("daily_adjustment_mode", "DAILY_ADJUSTMENT_OFF")
                    )
                    if daily_mode != "DAILY_ADJUSTMENT_OFF" and len(snapshot.daily_sequence) >= 6:
                        closes = [
                            float(row["close"])
                            for row in snapshot.daily_sequence
                            if row.get("close") is not None
                        ]
                        if len(closes) >= 6 and closes[-6] > 0:
                            adjust = float(
                                np.clip((closes[-1] / closes[-6] - 1.0) * 0.25, -0.02, 0.02)
                            )
                            state_code = snapshot.features.get("v35_market_state")
                            gated_ok = state_code in (1.0, 2.0)
                            if daily_mode == "DAILY_ADJUSTMENT_FIXED" or (
                                daily_mode == "DAILY_ADJUSTMENT_REGIME_GATED" and gated_ok
                            ):
                                path = list(
                                    value + adjust if index < 2 else value
                                    for index, value in enumerate(path)
                                )
                    expected_path = path
                    sigma = forecast.forecast_sigma
                    decision = self._strategy.decide(
                        market,
                        snapshot,
                        expected_path=expected_path,
                        p10_path=[value - 1.5 * sigma for value in expected_path],
                        probabilities_4=_probabilities_from_expected(
                            float(expected_path[3]), sigma, 0.02
                        ),
                        probabilities_8=_probabilities_from_expected(
                            float(expected_path[-1]), sigma, 0.02
                        ),
                        current_position_pp=0,
                        reliability_score=70,
                        health_status="MODEL_NORMAL",
                        ood_score=0.1,
                        config=dict(package.strategy_config_json),
                    )
                    strategy_score = decision.strategy_score
                    base_target = decision.base_target_position_pp
                    final_target = decision.final_target_position_pp
                    signals = tuple(
                        (str(batch["action"]), int(batch["position_pp"]))
                        for batch in decision.batches
                    )
                    trade_path = tuple(
                        (str(batch["action"]), int(batch["batch_change_pp"]))
                        for batch in decision.batches
                    )
                    if self.v36:
                        probabilities_8 = _probabilities_from_expected(
                            float(expected_path[-1]), sigma, 0.02
                        )
                        self._persist_decision_funnel(
                            session,
                            market,
                            anchor,
                            package.id,
                            {
                                "bull_signal_count": int(
                                    probabilities_8[0] > probabilities_8[2]
                                    or float(expected_path[-1]) > 0.0
                                ),
                                "score_pass_count": int(
                                    decision.strategy_score > 0.0
                                ),
                                "base_position_positive_count": int(
                                    decision.base_target_position_pp > 0
                                ),
                                "risk_blocked_count": int(
                                    decision.base_target_position_pp > 0
                                    and decision.final_target_position_pp == 0
                                ),
                                "cooldown_blocked_count": 0,
                                "final_position_positive_count": int(
                                    decision.final_target_position_pp > 0
                                ),
                                "trade_count": len(decision.batches),
                                "actual_position_positive_count": 0,
                            },
                        )
            else:
                forecast = session.scalar(
                    select(V35Forecast).where(
                        V35Forecast.model_market == market,
                        V35Forecast.forecast_anchor_date == anchor,
                        V35Forecast.protocol_version == self.protocol_version,
                    )
                )
                if forecast is not None:
                    expected_path = [float(v) for v in forecast.expected_path_json]
                snap = session.scalar(
                    select(V35StrategySnapshot).where(
                        V35StrategySnapshot.model_market == market,
                        V35StrategySnapshot.forecast_anchor_date == anchor,
                        V35StrategySnapshot.protocol_version == self.protocol_version,
                    )
                )
                if snap is not None:
                    strategy_score = float(snap.strategy_score)
                    base_target = snap.base_target_position_pp
                    final_target = snap.final_target_position_pp
                    batches = session.scalars(
                        select(V35PositionDecision)
                        .where(V35PositionDecision.strategy_snapshot_id == snap.id)
                        .order_by(V35PositionDecision.batch_number)
                    ).all()
                    signals = tuple(
                        (row.action, row.position_pp) for row in batches
                    )
                    trade_path = tuple(
                        (row.action, row.batch_change_pp) for row in batches
                    )
            if (
                expected_path is None
                or strategy_score is None
                or base_target is None
                or final_target is None
            ):
                continue
            records.append(
                V351AnchorRecord(
                    anchor=anchor,
                    expected_path=expected_path,
                    strategy_score=strategy_score,
                    base_target_position_pp=base_target,
                    final_target_position_pp=final_target,
                    signals=signals,
                    trade_path=trade_path,
                    equity_path=(),
                )
            )
        return tuple(records)

    def _v351_prediction_candidates(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
        round_index: int,
    ) -> tuple[V35ModelPackage, ...]:
        champion_model = session.get(V35ModelVersion, champion.prediction_model_id)
        champion_config = self._v351_model_config(champion)
        alphas: list[float] = []
        alpha_min, alpha_max = self._cfg_alpha_bounds()
        for multiplier in self._cfg_alpha_multipliers():
            value = round(
                float(
                    np.clip(
                        float(champion_model.ridge_alpha) * multiplier,
                        alpha_min,
                        alpha_max,
                    )
                ),
                12,
            )
            if value not in alphas:
                alphas.append(value)
        matured = self._matured_samples(session, market, anchor)
        snapshot_rows = self._feature_snapshots(session, market, through=anchor)
        base_manifest = self._features.build_manifest(
            replace(
                self._empty_snapshot(market, anchor),
                features=dict(snapshot_rows[-1].feature_json) if snapshot_rows else {},
            )
        )
        candidates: list[V35ModelPackage] = []
        attempts: list[dict[str, Any]] = []
        for index, alpha in enumerate(alphas[:MAX_PREDICTION_CANDIDATES]):
            version = f"{self.protocol_version}:{market}:PREDC351:{anchor.isoformat()}:{index}"
            model_config = {
                **champion_config,
                "alpha": alpha,
                "training_window_mode": champion_config.get(
                    "training_window_mode", "ROLLING_520W"
                ),
                "feature_set_name": champion_config.get(
                    "feature_set_name", "CORE_FEATURE_SET"
                ),
                "daily_adjustment_mode": champion_config.get(
                    "daily_adjustment_mode", "DAILY_ADJUSTMENT_OFF"
                ),
                "residual_scale": champion_config.get("residual_scale", 1.0),
            }
            candidate = self._fit_v351_candidate(
                session,
                market,
                anchor,
                champion,
                version,
                model_config,
                matured,
                base_manifest,
                family="PREDICTION",
            )
            if candidate is None:
                continue
            attempts.append({"alpha": alpha, "generated": True})
            candidates.append(candidate)
        if round_index % STRUCTURAL_CHALLENGE_EVERY == 0 and len(candidates) < MAX_PREDICTION_CANDIDATES:
            if self.v36:
                dimensions = (
                    "training_window_mode",
                    "feature_set",
                    "daily_adjustment_mode",
                    "residual_scale",
                    "residual_weighting",
                    "calibration_mode",
                    "model_family",
                    "regime_correction",
                )
            else:
                dimensions = STRUCTURAL_DIMENSIONS
            dimension = dimensions[
                (round_index // STRUCTURAL_CHALLENGE_EVERY) % len(dimensions)
            ]
            model_config = dict(champion_config)
            if dimension == "training_window_mode":
                model_config["training_window_mode"] = next(
                    value
                    for value in TRAINING_WINDOW_VALUES
                    if value != champion_config.get("training_window_mode", "ROLLING_520W")
                )
            elif dimension == "feature_set":
                model_config["feature_set_name"] = next(
                    value
                    for value in FEATURE_SET_VALUES
                    if value != champion_config.get("feature_set_name", "CORE_FEATURE_SET")
                )
            elif dimension == "daily_adjustment_mode":
                model_config["daily_adjustment_mode"] = next(
                    value
                    for value in DAILY_ADJUSTMENT_VALUES
                    if value
                    != champion_config.get("daily_adjustment_mode", "DAILY_ADJUSTMENT_OFF")
                )
            elif dimension == "residual_scale":
                model_config["residual_scale"] = next(
                    value
                    for value in self._cfg_residual_scale_values()
                    if value != champion_config.get("residual_scale", 1.0)
                )
            elif dimension == "residual_weighting":
                model_config["residual_weighting"] = next(
                    value
                    for value in RESIDUAL_WEIGHTING_VALUES_36
                    if value
                    != champion_config.get("residual_weighting", "UNIFORM")
                )
            elif dimension == "calibration_mode":
                model_config["calibration_mode"] = next(
                    value
                    for value in (
                        "TEMPERATURE",
                        "PLATT",
                        "ISOTONIC",
                    )
                    if value
                    != champion_config.get("calibration_mode", "TEMPERATURE")
                )
            elif dimension == "model_family":
                model_config["model_family"] = next(
                    value
                    for value in MODEL_FAMILY_VALUES_36
                    if value != champion_config.get("model_family", "RIDGE")
                )
            elif dimension == "regime_correction":
                model_config["regime_correction"] = next(
                    value
                    for value in REGIME_CORRECTION_VALUES_36
                    if value
                    != champion_config.get("regime_correction", "OFF")
                )
            version = f"{self.protocol_version}:{market}:STRUCTC351:{anchor.isoformat()}"
            candidate = self._fit_v351_candidate(
                session,
                market,
                anchor,
                champion,
                version,
                model_config,
                matured,
                base_manifest,
                family="PREDICTION_STRUCTURAL",
            )
            if candidate is not None:
                attempts.append(
                    {"structural_dimension": dimension, "generated": True}
                )
                candidates.append(candidate)
        self._pending_v351_attempts = attempts
        return tuple(candidates)

    def _fit_v351_candidate(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
        version: str,
        model_config: Mapping[str, Any],
        matured: Sequence[V35TrainingSample],
        base_manifest: Any,
        *,
        family: str,
    ) -> V35ModelPackage | None:
        existing_model = session.get(V35ModelVersion, version)
        if existing_model is not None:
            state = state_from_payload(existing_model.parameters_json)
        else:
            training_window = str(
                model_config.get("training_window_mode", "ROLLING_520W")
            )
            limit = None
            if training_window == "ROLLING_520W":
                limit = 520
            elif training_window == "ROLLING_260W":
                limit = 260
            training_rows = tuple(matured[-limit:] if limit else matured)
            if len(training_rows) < 8:
                return None
            feature_set = str(
                model_config.get("feature_set_name", "CORE_FEATURE_SET")
            )
            feature_names = tuple(
                name
                for name in base_manifest.ordered_feature_names
                if feature_set == "EXTENDED_FEATURE_SET" or not name.startswith("seq100_")
            )
            validation = self._training.validation_metrics_for(
                market,
                training_rows,
                alpha=float(model_config.get("alpha", 1.0)),
                feature_names=feature_names,
            )
            next_anchor = self._anchor_after(session, market, anchor, steps=1)
            state = self._training.fit(
                market,
                training_rows,
                version=version,
                parent_version=champion.prediction_model_id,
                trained_through=anchor,
                effective_from=next_anchor or anchor,
                alpha=float(model_config.get("alpha", 1.0)),
                feature_names=feature_names,
                validation_metrics=validation,
            )
            self._persist_model(
                session,
                market,
                state,
                status="CANDIDATE",
                metrics=validation,
                promotion_gate={"family": family, "model_config": dict(model_config)},
            )
        package_version = f"{version}:PKG"
        strategy_config = {
            **dict(champion.strategy_config_json),
            "v351_model_config": dict(model_config),
        }
        return self._persist_package(
            session,
            market,
            version=package_version,
            kind="PREDICTION_CHALLENGER" if family == "PREDICTION" else "STRUCTURAL_CHALLENGER",
            prediction_model_id=state.version,
            strategy_config=strategy_config,
            effective_from=anchor,
            parent_package_id=champion.id,
        )

    def _v351_strategy_candidates(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
    ) -> tuple[V35ModelPackage, ...]:
        stats = self._v351_binding_stats(session, market, anchor, champion)
        ordered = sorted(
            (
                key
                for key, count in stats.items()
                if count > 0 and len(self._cfg_strategy_groups()[key]) > 1
            ),
            key=lambda key: -stats[key],
        )
        for group_index, key in enumerate(ordered[:MAX_STRATEGY_PARAMETER_GROUP_TRIES]):
            allowed = self._cfg_strategy_groups()[key]
            current = self._v351_current_parameter(champion, key)
            alternatives = [value for value in allowed if value != current][:3]
            if not alternatives:
                continue
            variants: list[tuple[dict[str, Any], Any]] = []
            for value in alternatives:
                variant = json.loads(json.dumps(dict(champion.strategy_config_json)))
                variant.pop("v351_model_config", None)
                self._v351_apply_parameter(variant, key, value)
                variants.append((variant, value))
            generated: list[V35ModelPackage] = []
            for index, (variant, value) in enumerate(variants):
                version = (
                    f"{self.protocol_version}:{market}:STRATC351:"
                    f"{anchor.isoformat()}:{group_index}:{index}"
                )
                challenger = self._persist_package(
                    session,
                    market,
                    version=version,
                    kind="STRATEGY_CHALLENGER",
                    prediction_model_id=champion.prediction_model_id,
                    strategy_config=variant,
                    effective_from=anchor,
                    parent_package_id=champion.id,
                )
                generated.append(challenger)
            if generated:
                return tuple(generated[:3])
        return ()

    def _v351_current_parameter(
        self,
        champion: V35ModelPackage,
        key: str,
    ) -> Any:
        config = dict(champion.strategy_config_json)
        if key.startswith("state_cap_"):
            return config["state_position_caps"].get(key.removeprefix("state_cap_"))
        execution = dict(config.get("execution", {}))
        mapping = {
            "first_probe_buy": "weak_first_batch_pp",
            "trend_add": "trend_add_pp",
            "strong_trend_add": "strong_trend_add_pp",
            "cooldown_days": "cooldown_trading_days",
            "light_reduce": "light_reduce_pp",
            "confirmed_reduce": "confirmed_reduce_pp",
            "strong_signal_entry_floor_pp": "strong_signal_entry_floor_pp",
            "vol_target_enabled": "vol_target_enabled",
            "target_vol": "target_vol",
        }
        if key in mapping:
            return execution.get(mapping[key])
        if key == "trend_confirmation_weeks":
            return config.get("trend_confirmation_weeks", 2)
        return None

    def _v351_apply_parameter(
        self,
        config: dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        if key.startswith("state_cap_"):
            config["state_position_caps"][key.removeprefix("state_cap_")] = value
            return
        if key == "trend_confirmation_weeks":
            config["trend_confirmation_weeks"] = value
            return
        mapping = {
            "first_probe_buy": "weak_first_batch_pp",
            "trend_add": "trend_add_pp",
            "strong_trend_add": "strong_trend_add_pp",
            "cooldown_days": "cooldown_trading_days",
            "light_reduce": "light_reduce_pp",
            "confirmed_reduce": "confirmed_reduce_pp",
            "strong_signal_entry_floor_pp": "strong_signal_entry_floor_pp",
            "vol_target_enabled": "vol_target_enabled",
            "target_vol": "target_vol",
        }
        if key in mapping:
            config["execution"][mapping[key]] = value

    def _run_challenge_v351(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
        family: str,
        run_id: str,
    ) -> dict[str, Any]:
        challenge_id = f"{self.protocol_version}:{market}:{anchor.isoformat()}:{family}"
        existing = session.get(V35Challenge, challenge_id)
        if existing is not None:
            return {"challenge_created": False, "existing": challenge_id}
        bootstrap = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == self.protocol_version,
            )
        )
        round_index = (
            int(bootstrap.prediction_challenge_count)
            if bootstrap is not None
            else 0
        )
        if family == "PREDICTION":
            candidates = self._v351_prediction_candidates(
                session, market, anchor, champion, round_index
            )
        else:
            candidates = self._v351_strategy_candidates(
                session, market, anchor, champion
            )
        challenge = V35Challenge(
            id=challenge_id,
            protocol_version=self.protocol_version,
            model_market=market,
            anchor_date=anchor,
            challenger_family=family,
            champion_package_id=champion.id,
            candidate_count=len(candidates),
            status="EVALUATING",
            result_json={
                "candidates": [item.id for item in candidates],
                "attempts": getattr(self, "_pending_v351_attempts", []),
                "mechanism": "V3.5.1_EFFECTIVE_CHALLENGER",
            },
            created_at=utc_now(),
        )
        session.add(challenge)
        session.flush()
        result: dict[str, Any] = {
            "challenge_created": True,
            "family": family,
            "candidates": len(candidates),
            "status": "EVALUATING",
        }
        return result

    def _evaluate_pending_v351(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> dict[str, Any]:
        """Evaluate deferred challengers once ≥8 post-effective windows matured."""

        challenges = session.scalars(
            select(V35Challenge).where(
                V35Challenge.model_market == market,
                V35Challenge.protocol_version == self.protocol_version,
                V35Challenge.status == "EVALUATING",
            )
        ).all()
        outcomes: dict[str, Any] = {}
        for challenge in challenges:
            champion = session.get(V35ModelPackage, challenge.champion_package_id)
            candidates = [
                session.get(V35ModelPackage, candidate_id)
                for candidate_id in challenge.result_json.get("candidates", [])
            ]
            candidates = [candidate for candidate in candidates if candidate is not None]
            if champion is None or not candidates:
                continue
            windows = [
                window
                for window in self._matured_windows(session, market, anchor)
                if window[0] >= candidates[0].effective_from_date
            ]
            if len(windows) < 8:
                continue
            anchors_seq = tuple(anchor_item for window in windows for anchor_item in window)
            diffs: dict[str, Any] = {}
            for candidate in candidates:
                champion_rows = self._v351_behavior_rows(
                    session, market, anchors_seq, champion
                )
                candidate_rows = self._v351_behavior_rows(
                    session,
                    market,
                    anchors_seq,
                    candidate,
                    recompute_forecast=True,
                )
                diffs[candidate.id] = behavior_diff(champion_rows, candidate_rows)
            selected = max(
                candidates,
                key=lambda item: (
                    float(
                        getattr(diffs[item.id], "target_position_divergence_ratio", 0.0)
                        or 0.0
                    ),
                    float(
                        getattr(diffs[item.id], "trade_path_divergence_ratio", 0.0)
                        or 0.0
                    ),
                    float(
                        getattr(diffs[item.id], "forecast_divergence_ratio", 0.0)
                        or 0.0
                    ),
                ),
                default=None,
            )
            if selected is None:
                continue
            for candidate in candidates:
                if candidate is not selected:
                    persist_candidate_evaluation(
                        session,
                        challenge_id=challenge.id,
                        candidate_package_id=candidate.id,
                        model_market=market,
                        challenger_family=challenge.challenger_family,
                        diff=diffs.get(candidate.id) or behavior_diff((), ()),
                        rejection_reason_codes=["NOT_SELECTED_INNER"],
                        gaps={},
                        promotion_channel=None,
                        raw_window_count=len(windows),
                        effective_window_count=len(windows),
                        prediction_quality={},
                    )
            evaluation = self._evaluate_challenger_v351(
                session,
                market,
                anchor,
                challenge.id,
                champion,
                selected,
                challenge.challenger_family,
                windows=windows,
            )
            challenge.status = "COMPLETED"
            challenge.result_json = {
                **challenge.result_json,
                "promotions": [evaluation],
                "evaluation_windows": [window[0].isoformat() for window in windows],
            }
            outcomes[challenge.id] = evaluation
            if evaluation.get("promotion_decision") == "PROMOTED":
                next_anchor = self._anchor_after(
                    session, market, challenge.anchor_date, steps=1
                )
                bootstrap = session.scalar(
                    select(V35BootstrapState).where(
                        V35BootstrapState.model_market == market,
                        V35BootstrapState.protocol_version == self.protocol_version,
                    )
                )
                if bootstrap is not None:
                    bootstrap.promotion_count += 1
                    bootstrap.champion_package_id = selected.id
                    bootstrap.updated_at = utc_now()
                session.add(
                    V35Promotion(
                        protocol_version=self.protocol_version,
                        model_market=market,
                        challenge_id=challenge.id,
                        champion_package_id=champion.id,
                        candidate_package_id=selected.id,
                        challenger_family=challenge.challenger_family,
                        evaluation_window_count=evaluation.get(
                            "evaluation_window_count", 0
                        ),
                        promotion_decision=str(
                            evaluation.get("promotion_channel", "PROMOTED")
                        ),
                        promotion_reason=str(evaluation.get("reason", "晋级门禁通过")),
                        effective_from_date=next_anchor,
                        excess_profit=_money(evaluation.get("excess_profit", 0.0)),
                        excess_return=_pct(evaluation.get("excess_return", 0.0)),
                        metrics_json=evaluation,
                        promotion_hash=_hash(evaluation),
                        created_at=utc_now(),
                    )
                )
        return outcomes

    def _finalize_pending_v351(self, session: Session, market: str) -> None:
        """Mark still-pending challenges as SHADOW_EVALUATION at end of history."""

        challenges = session.scalars(
            select(V35Challenge).where(
                V35Challenge.model_market == market,
                V35Challenge.protocol_version == self.protocol_version,
                V35Challenge.status == "EVALUATING",
            )
        ).all()
        for challenge in challenges:
            windows = self._matured_windows(session, market, challenge.anchor_date)
            challenge.status = "COMPLETED"
            challenge.result_json = {
                **challenge.result_json,
                "promotions": [
                    {
                        "promotion_decision": "SHADOW_EVALUATION",
                        "reason_codes": ["INSUFFICIENT_EFFECTIVE_WINDOWS"],
                        "evaluation_window_count": len(windows),
                    }
                ],
            }

    def _evaluate_challenger_v351(
        self,
        session: Session,
        market: str,
        anchor: date,
        challenge_id: str,
        champion: V35ModelPackage,
        challenger: V35ModelPackage,
        family: str,
        *,
        windows: Sequence[tuple[date, ...]] | None = None,
    ) -> dict[str, Any]:
        if windows is None:
            windows = self._matured_windows(session, market, anchor)
        windows = [
            window
            for window in windows
            if window[0] >= challenger.effective_from_date
        ]
        if len(windows) < 8:
            return {
                "candidate": challenger.id,
                "promotion_decision": "SHADOW_EVALUATION",
                "reason_codes": ["INSUFFICIENT_EFFECTIVE_WINDOWS"],
                "evaluation_window_count": len(windows),
            }
        anchors = tuple(a for window in windows for a in window)
        champion_rows = self._v351_behavior_rows(session, market, anchors, champion)
        candidate_rows = self._v351_behavior_rows(
            session, market, anchors, challenger, recompute_forecast=True
        )
        behavior = behavior_diff(champion_rows, candidate_rows)
        champion_metrics: list[dict[str, float]] = []
        candidate_metrics: list[dict[str, float]] = []
        comparable = 0
        for window in windows:
            champion_result = self._window_result(
                session, market, champion.id, window, load_weekly_bars(session, market, anchors)
            )
            if champion_result is None:
                continue
            candidate_result = self._challenger_window_result(
                session, market, challenge_id, challenger, window,
                load_weekly_bars(session, market, anchors), family,
            )
            if candidate_result is None:
                continue
            champion_metrics.append(champion_result)
            candidate_metrics.append(candidate_result)
            comparable += 1
        if behavior.effectively_identical:
            persist_candidate_evaluation(
                session,
                challenge_id=challenge_id,
                candidate_package_id=challenger.id,
                model_market=market,
                challenger_family=family,
                diff=behavior,
                rejection_reason_codes=["EFFECTIVELY_IDENTICAL"],
                gaps={},
                promotion_channel=None,
                raw_window_count=comparable,
                effective_window_count=comparable,
                prediction_quality={},
            )
            return {
                "candidate": challenger.id,
                "promotion_decision": "EFFECTIVELY_IDENTICAL",
                "reason": behavior.reason,
                "reason_codes": ["EFFECTIVELY_IDENTICAL"],
                "evaluation_window_count": comparable,
                "behavior": {
                    "forecast_divergence_ratio": behavior.forecast_divergence_ratio,
                    "target_position_divergence_ratio": behavior.target_position_divergence_ratio,
                    "trade_path_divergence_ratio": behavior.trade_path_divergence_ratio,
                },
            }
        if comparable < 8:
            persist_candidate_evaluation(
                session,
                challenge_id=challenge_id,
                candidate_package_id=challenger.id,
                model_market=market,
                challenger_family=family,
                diff=behavior,
                rejection_reason_codes=["INSUFFICIENT_EFFECTIVE_WINDOWS"],
                gaps={"effective_window_gap": 8 - comparable},
                promotion_channel=None,
                raw_window_count=comparable,
                effective_window_count=comparable,
                prediction_quality={},
            )
            return {
                "candidate": challenger.id,
                "promotion_decision": "REJECTED",
                "reason_codes": ["INSUFFICIENT_EFFECTIVE_WINDOWS"],
                "evaluation_window_count": comparable,
            }

        champion_returns = [float(row["net_return"]) for row in champion_metrics]
        candidate_returns = [float(row["net_return"]) for row in candidate_metrics]
        champion_profits = [float(row["net_profit"]) for row in champion_metrics]
        candidate_profits = [float(row["net_profit"]) for row in candidate_metrics]
        champion_drawdowns = [float(row["max_drawdown"]) for row in champion_metrics]
        candidate_drawdowns = [float(row["max_drawdown"]) for row in candidate_metrics]
        candidate_positions = [float(row["average_position_pp"]) for row in candidate_metrics]
        no_action_count = sum(
            1 for row in candidate_metrics if row["no_action_window"]
        )
        mean_excess = float(
            np.mean(np.asarray(candidate_returns) - np.asarray(champion_returns))
        )
        median_profit_excess = float(
            np.median(np.asarray(candidate_profits))
            - np.median(np.asarray(champion_profits))
        )
        win_rate = float(
            np.mean(np.asarray(candidate_returns) > np.asarray(champion_returns))
        )
        drawdown_degradation = float(
            np.max(np.asarray(candidate_drawdowns) - np.asarray(champion_drawdowns))
        )
        no_action_ratio = no_action_count / max(comparable, 1)
        participation = float(np.mean(candidate_positions)) / 100.0
        champion_participation = float(
            np.mean([float(row["average_position_pp"]) for row in champion_metrics])
        ) / 100.0
        intervals = [(window[0], window[-1]) for window in windows]
        effective_windows = effective_independent_window_count(intervals)

        model = session.get(V35ModelVersion, challenger.prediction_model_id)
        champion_model = session.get(V35ModelVersion, champion.prediction_model_id)
        prediction_quality = self._v351_prediction_quality(model, champion_model)
        quality_ok = bool(
            prediction_quality.get("pass")
        )
        if self.v36:
            activity_health_not_worse = bool(
                no_action_ratio
                <= float(
                    STRONG_PROMOTION_GATES_36["no_action_window_ratio_cap"]
                )
                and participation >= champion_participation - 0.05
            )
            channel, reason_codes, gaps = promotion_channel_for_36(
                raw_windows=comparable,
                effective_windows=effective_windows,
                mean_excess=mean_excess,
                median_profit_excess=median_profit_excess,
                win_rate=win_rate,
                drawdown_degradation=drawdown_degradation,
                participation=participation,
                champion_participation=champion_participation,
                no_action_ratio=no_action_ratio,
                trade_path_divergence_ratio=behavior.trade_path_divergence_ratio,
                quality_ok=quality_ok,
                activity_health_not_worse=activity_health_not_worse,
            )
        else:
            channel, reason_codes, gaps = promotion_channel_for(
                raw_windows=comparable,
                effective_windows=effective_windows,
                mean_excess=mean_excess,
                median_profit_excess=median_profit_excess,
                win_rate=win_rate,
                drawdown_degradation=drawdown_degradation,
                participation=participation,
                champion_participation=champion_participation,
                no_action_ratio=no_action_ratio,
                trade_path_divergence_ratio=behavior.trade_path_divergence_ratio,
                quality_ok=quality_ok,
            )
        if channel is None:
            decision = "REJECTED"
        else:
            decision = "PROMOTED"
        persist_candidate_evaluation(
            session,
            challenge_id=challenge_id,
            candidate_package_id=challenger.id,
            model_market=market,
            challenger_family=family,
            diff=behavior,
            rejection_reason_codes=reason_codes,
            gaps=gaps,
            promotion_channel=channel,
            raw_window_count=comparable,
            effective_window_count=effective_windows,
            prediction_quality=prediction_quality,
        )
        return {
            "candidate": challenger.id,
            "promotion_decision": decision,
            "promotion_channel": channel,
            "reason_codes": reason_codes,
            "reason": "; ".join(reason_codes) if reason_codes else "满足全部晋级门禁",
            "gaps": gaps,
            "evaluation_window_count": comparable,
            "effective_independent_window_count": effective_windows,
            "mean_excess_return": mean_excess,
            "median_profit_excess": median_profit_excess,
            "win_rate": win_rate,
            "max_drawdown_degradation": drawdown_degradation,
            "no_action_ratio": no_action_ratio,
            "mean_challenger_position_pp": float(np.mean(candidate_positions)),
            "excess_profit": mean_excess * 100000.0,
            "excess_return": mean_excess,
            "behavior": {
                "forecast_divergence_ratio": behavior.forecast_divergence_ratio,
                "target_position_divergence_ratio": behavior.target_position_divergence_ratio,
                "trade_path_divergence_ratio": behavior.trade_path_divergence_ratio,
            },
            "prediction_quality": prediction_quality,
        }

    def _v351_prediction_quality(
        self,
        candidate_model: V35ModelVersion | None,
        champion_model: V35ModelVersion | None,
    ) -> dict[str, Any]:
        if candidate_model is None or champion_model is None:
            return {"pass": True, "reason": "NO_MODEL_METRICS"}
        champion_metrics = dict(champion_model.metrics_json or {})
        candidate_metrics = dict(candidate_model.metrics_json or {})
        champion_mae = champion_metrics.get("mean_oos_path_mae")
        candidate_mae = candidate_metrics.get("mean_oos_path_mae")
        mae_ok = True
        if champion_mae is not None and candidate_mae is not None:
            mae_ok = float(candidate_mae) <= float(champion_mae) * float(
                self._cfg_maintenance_gates()["mae_degradation_ratio_max"]
            )
        return {
            "pass": mae_ok,
            "champion_mean_oos_path_mae": champion_mae,
            "candidate_mean_oos_path_mae": candidate_mae,
            "mae_ok": mae_ok,
        }

    def _maintenance_refresh(
        self,
        session: Session,
        market: str,
        anchor: date,
        champion: V35ModelPackage,
    ) -> dict[str, Any]:
        champion_model = session.get(V35ModelVersion, champion.prediction_model_id)
        if champion_model is None:
            return {"decision": "SKIPPED", "reason": "NO_CHAMPION_MODEL"}
        model_config = self._v351_model_config(champion)
        matured = self._matured_samples(session, market, anchor)
        training_window = str(
            model_config.get("training_window_mode", "ROLLING_520W")
        )
        limit = None
        if training_window == "ROLLING_520W":
            limit = 520
        elif training_window == "ROLLING_260W":
            limit = 260
        training_rows = tuple(matured[-limit:] if limit else matured)
        if len(training_rows) < 8:
            return {"decision": "SKIPPED", "reason": "INSUFFICIENT_SAMPLES"}
        snapshot_rows = self._feature_snapshots(session, market, through=anchor)
        manifest = self._features.build_manifest(
            replace(
                self._empty_snapshot(market, anchor),
                features=dict(snapshot_rows[-1].feature_json) if snapshot_rows else {},
            )
        )
        version = f"{self.protocol_version}:{market}:MAINT:{anchor.isoformat()}"
        validation = self._training.validation_metrics_for(
            market,
            training_rows,
            alpha=float(champion_model.ridge_alpha),
            feature_names=manifest.ordered_feature_names,
        )
        next_anchor = self._anchor_after(session, market, anchor, steps=1)
        state = self._training.fit(
            market,
            training_rows,
            version=version,
            parent_version=champion_model.id,
            trained_through=anchor,
            effective_from=next_anchor or anchor,
            alpha=float(champion_model.ridge_alpha),
            feature_names=manifest.ordered_feature_names,
            validation_metrics=validation,
        )
        self._persist_model(
            session,
            market,
            state,
            status="MAINTENANCE",
            metrics=validation,
            promotion_gate={"kind": "MODEL_MAINTENANCE_REFRESH"},
        )
        strategy_config = {
            **dict(champion.strategy_config_json),
            "v351_model_config": dict(model_config),
        }
        package = self._persist_package(
            session,
            market,
            version=f"{version}:PKG",
            kind="CHAMPION_MAINTENANCE",
            prediction_model_id=state.version,
            strategy_config=strategy_config,
            effective_from=next_anchor or anchor,
            parent_package_id=champion.id,
        )
        champion_metrics = dict(champion_model.metrics_json or {})
        champion_mae = champion_metrics.get("mean_oos_path_mae")
        candidate_mae = validation.get("mean_oos_path_mae")
        gates = self._cfg_maintenance_gates()
        mae_ok = (
            candidate_mae is not None
            and champion_mae is not None
            and float(candidate_mae)
            <= float(champion_mae) * float(gates["mae_degradation_ratio_max"])
        )
        if self.v36:
            improved = (
                candidate_mae is not None
                and champion_mae is not None
                and float(candidate_mae)
                <= float(champion_mae)
                * (1.0 - float(gates["minimum_mae_improvement"]))
            )
        else:
            improved = (
                candidate_mae is not None
                and champion_mae is not None
                and float(candidate_mae) < float(champion_mae)
            )
        if not mae_ok:
            payload = {
                "model_market": market,
                "anchor": anchor.isoformat(),
                "from_model_id": champion_model.id,
                "to_model_id": state.version,
                "from_package_id": champion.id,
                "to_package_id": package.id,
                "decision": "REJECTED",
                "reason_code": "MAE_DEGRADED",
            }
            session.add(
                V351MaintenanceRefresh(
                    model_market=market,
                    anchor_date=anchor,
                    from_model_id=champion_model.id,
                    to_model_id=state.version,
                    from_package_id=champion.id,
                    to_package_id=package.id,
                    decision="REJECTED",
                    reason_code="MAE_DEGRADED",
                    reason="MAE degraded beyond limit",
                    metrics_before_json=champion_metrics,
                    metrics_after_json=validation,
                    effective_from_date=None,
                    refresh_hash=_hash(payload),
                    created_at=utc_now(),
                )
            )
            return payload
        windows = self._matured_windows(session, market, anchor)[-8:]
        bars = load_weekly_bars(session, market, self._anchors(session, market))
        excess_returns: list[float] = []
        drawdown_diffs: list[float] = []
        for window in windows:
            champion_result = self._window_result(
                session, market, champion.id, window, bars
            )
            candidate_result = self._challenger_window_result(
                session,
                market,
                f"MAINT:{market}:{anchor.isoformat()}",
                package,
                window,
                bars,
                "MAINTENANCE",
                record_window=False,
            )
            if champion_result is None or candidate_result is None:
                continue
            excess_returns.append(
                float(candidate_result["net_return"]) - float(champion_result["net_return"])
            )
            drawdown_diffs.append(
                float(candidate_result["max_drawdown"])
                - float(champion_result["max_drawdown"])
            )
        mean_excess = float(np.mean(excess_returns)) if excess_returns else -1.0
        max_dd_diff = float(np.max(drawdown_diffs)) if drawdown_diffs else 1.0
        account_gate_ok = bool(
            max_dd_diff
            <= float(gates["maximum_drawdown_degradation_pp"]) / 100.0
            and mean_excess >= float(gates["minimum_excess_return"])
        )
        accepted = (
            account_gate_ok and (improved or mean_excess >= float(gates["minimum_excess_return"]))
        )
        decision = "MODEL_MAINTENANCE_REFRESH_ACCEPTED" if accepted else "REJECTED"
        reason_code = (
            "ACCEPTED"
            if accepted
            else "NO_IMPROVEMENT"
            if improved
            else "ACCOUNT_GATE_FAILED"
        )
        payload = {
            "model_market": market,
            "anchor": anchor.isoformat(),
            "from_model_id": champion_model.id,
            "to_model_id": state.version,
            "from_package_id": champion.id,
            "to_package_id": package.id,
            "decision": decision,
            "reason_code": reason_code,
            "mean_excess_return": mean_excess,
            "max_drawdown_degradation": max_dd_diff,
        }
        session.add(
            V351MaintenanceRefresh(
                model_market=market,
                anchor_date=anchor,
                from_model_id=champion_model.id,
                to_model_id=state.version,
                from_package_id=champion.id,
                to_package_id=package.id,
                decision=decision,
                reason_code=reason_code,
                reason="样本外误差改善且账户门禁通过" if accepted else "门禁未通过",
                metrics_before_json=champion_metrics,
                metrics_after_json=validation,
                effective_from_date=next_anchor if accepted else None,
                refresh_hash=_hash(payload),
                created_at=utc_now(),
            )
        )
        if accepted:
            bootstrap = session.scalar(
                select(V35BootstrapState).where(
                    V35BootstrapState.model_market == market,
                    V35BootstrapState.protocol_version == self.protocol_version,
                )
            )
            if bootstrap is not None:
                bootstrap.champion_package_id = package.id
                bootstrap.updated_at = utc_now()
        return payload

    def _evaluate_challenger(
        self,
        session: Session,
        market: str,
        anchor: date,
        challenge_id: str,
        champion: V35ModelPackage,
        challenger: V35ModelPackage,
        family: str,
    ) -> dict[str, Any]:
        anchors = self._anchors(session, market)
        bars = load_weekly_bars(session, market, anchors)
        windows = self._matured_windows(session, market, anchor)
        if len(windows) < 8:
            return {
                "candidate": challenger.id,
                "promotion_decision": "SHADOW_EVALUATION",
                "reason": "样本外8周窗口不足",
                "evaluation_window_count": len(windows),
            }
        windows = windows[-12:]
        champion_returns: list[float] = []
        challenger_returns: list[float] = []
        champion_profits: list[float] = []
        challenger_profits: list[float] = []
        champion_drawdowns: list[float] = []
        challenger_drawdowns: list[float] = []
        challenger_no_actions = 0
        challenger_positions: list[float] = []
        comparable = 0
        for window in windows:
            champion_result = self._window_result(
                session, market, champion.id, window, bars
            )
            if champion_result is None:
                continue
            challenger_result = self._challenger_window_result(
                session, market, challenge_id, challenger, window, bars, family
            )
            if challenger_result is None:
                continue
            champion_returns.append(champion_result["net_return"])
            challenger_returns.append(challenger_result["net_return"])
            champion_profits.append(champion_result["net_profit"])
            challenger_profits.append(challenger_result["net_profit"])
            champion_drawdowns.append(champion_result["max_drawdown"])
            challenger_drawdowns.append(challenger_result["max_drawdown"])
            challenger_positions.append(challenger_result["average_position_pp"])
            if challenger_result["no_action_window"]:
                challenger_no_actions += 1
            comparable += 1
        if comparable < 8:
            return {
                "candidate": challenger.id,
                "promotion_decision": "SHADOW_EVALUATION",
                "reason": f"可比较窗口不足（{comparable}/8）",
                "evaluation_window_count": comparable,
            }
        mean_excess = float(np.mean(np.asarray(challenger_returns) - np.asarray(champion_returns)))
        median_profit_excess = float(
            np.median(np.asarray(challenger_profits)) - np.median(np.asarray(champion_profits))
        )
        win_rate = float(np.mean(np.asarray(challenger_returns) > np.asarray(champion_returns)))
        drawdown_degradation = float(
            np.max(np.asarray(challenger_drawdowns) - np.asarray(champion_drawdowns))
        )
        no_action_ratio = challenger_no_actions / comparable
        decision, reason = promotion_gate_decision(
            mean_excess=mean_excess,
            median_profit_excess=median_profit_excess,
            win_rate=win_rate,
            max_drawdown_degradation=drawdown_degradation,
            no_action_ratio=no_action_ratio,
            mean_position_pp=float(np.mean(challenger_positions)),
        )
        return {
            "candidate": challenger.id,
            "promotion_decision": decision,
            "reason": reason,
            "evaluation_window_count": comparable,
            "mean_excess_return": mean_excess,
            "median_profit_excess": median_profit_excess,
            "win_rate": win_rate,
            "max_drawdown_degradation": drawdown_degradation,
            "no_action_ratio": no_action_ratio,
            "mean_challenger_position_pp": float(np.mean(challenger_positions)),
            "excess_profit": mean_excess * 100000.0,
            "excess_return": mean_excess,
        }

    def _matured_windows(self, session: Session, market: str, as_of: date) -> tuple[tuple[date, ...], ...]:
        anchors = self._anchors(session, market)
        result: list[tuple[date, ...]] = []
        for index in range(len(anchors) - HORIZON_WEEKS + 1):
            window = anchors[index : index + HORIZON_WEEKS]
            if len(window) == HORIZON_WEEKS and window[-1] <= as_of:
                result.append(window)
        return tuple(result)

    def _window_result(
        self,
        session: Session,
        market: str,
        package_id: str,
        window: Sequence[date],
        bars: Mapping[date, dict[str, float | None]],
    ) -> dict[str, Any] | None:
        decisions = {
            anchor: decision
            for anchor in window
            if (decision := self._decisions_for_anchor(session, market, anchor)) is not None
        }
        if len(decisions) != HORIZON_WEEKS:
            return None
        package = session.get(V35ModelPackage, package_id)
        if package is None:
            return None
        result = run_standard_window(
            session,
            market=market,
            package_id=package_id,
            anchors=window,
            decisions=decisions,
            bars=bars,
            config=dict(package.strategy_config_json),
        )
        persist_window_result(
            session,
            result,
            protocol_version=self.protocol_version,
        )
        return {
            "net_return": result.net_return,
            "net_profit": float(result.net_profit),
            "max_drawdown": result.max_drawdown,
            "average_position_pp": result.average_position_pp,
            "trade_count": result.trade_count,
            "no_action_window": result.no_action_window,
        }

    def _challenger_window_result(
        self,
        session: Session,
        market: str,
        challenge_id: str,
        challenger: V35ModelPackage,
        window: Sequence[date],
        bars: Mapping[date, dict[str, float | None]],
        family: str,
        *,
        record_window: bool = True,
    ) -> dict[str, Any] | None:
        existing = session.scalar(
            select(V35ChallengerWindow).where(
                V35ChallengerWindow.challenge_id == challenge_id,
                V35ChallengerWindow.candidate_package_id == challenger.id,
                V35ChallengerWindow.window_start_date == window[0],
            )
        )
        if existing is not None:
            return {
                "net_return": float(existing.net_return),
                "net_profit": float(existing.net_profit),
                "max_drawdown": float(existing.max_drawdown),
                "average_position_pp": float(existing.average_position_pp),
                "trade_count": existing.trade_count,
                "no_action_window": existing.no_action_window,
            }
        if family == "STRATEGY":
            decisions = self._recompute_decisions(session, market, window, challenger)
        else:
            decisions = {
                anchor: decision
                for anchor in window
                if (decision := self._decisions_for_anchor(session, market, anchor)) is not None
            }
        if len(decisions) != HORIZON_WEEKS:
            return None
        result = run_standard_window(
            session,
            market=market,
            package_id=challenger.id,
            anchors=window,
            decisions=decisions,
            bars=bars,
            config=dict(challenger.strategy_config_json),
        )
        persist_window_result(
            session,
            result,
            protocol_version=self.protocol_version,
        )
        if record_window:
            session.add(
                V35ChallengerWindow(
                    challenge_id=challenge_id,
                    candidate_package_id=challenger.id,
                    sim_account_id=result.account_id,
                    window_start_date=window[0],
                    window_end_date=window[-1],
                    net_return=_pct(result.net_return),
                    net_profit=result.net_profit,
                    max_drawdown=_pct(result.max_drawdown),
                    average_position_pp=_pct(result.average_position_pp),
                    trade_count=result.trade_count,
                    no_action_window=result.no_action_window,
                    created_at=utc_now(),
                )
            )
        return {
            "net_return": result.net_return,
            "net_profit": float(result.net_profit),
            "max_drawdown": result.max_drawdown,
            "average_position_pp": result.average_position_pp,
            "trade_count": result.trade_count,
            "no_action_window": result.no_action_window,
        }

    def _recompute_decisions(
        self,
        session: Session,
        market: str,
        window: Sequence[date],
        package: V35ModelPackage,
    ) -> dict[date, V35StrategyDecision]:
        decisions: dict[date, V35StrategyDecision] = {}
        position_pp = 0
        pending_batches: list[Mapping[str, Any]] = []
        if self.v36:
            previous_anchor = session.scalar(
                select(func.max(V36AccountSnapshot.anchor_date)).where(
                    V36AccountSnapshot.protocol_version
                    == self.protocol_version,
                    V36AccountSnapshot.model_market == market,
                    V36AccountSnapshot.anchor_date < window[0],
                )
            )
            snapshot = (
                self._snapshot_for(session, market, previous_anchor)
                if previous_anchor is not None
                else None
            )
            if snapshot is not None:
                position_pp = float(snapshot["position_pp"])
                pending_batches = [
                    dict(batch) for batch in snapshot.get("pending_batches", [])
                ]
        last_trade_sequence = -10
        for sequence, anchor in enumerate(window):
            # Execute the previous anchor's first pending batch at this week's
            # open (cooldown-aware), then generate this anchor's decision with
            # the resulting current position -- matching account simulation.
            if pending_batches:
                batch = pending_batches[0]
                action = str(batch["action"])
                target = int(batch["target_position_pp"])
                cooldown_days = int(batch.get("cooldown_trading_days", 5) or 5)
                min_gap = max(1, math.ceil(cooldown_days / 5))
                change = int(batch["batch_change_pp"])
                if (
                    sequence - last_trade_sequence >= min_gap
                    and (
                        (action == "BUY" and target > position_pp)
                        or (action == "SELL" and target < position_pp)
                    )
                ):
                    if action == "BUY":
                        change = min(change, max(0, target - position_pp))
                    else:
                        change = min(change, max(0, position_pp - target))
                    if change >= 5:
                        position_pp += change if action == "BUY" else -change
                        pending_batches = pending_batches[1:]
                        last_trade_sequence = sequence
            forecast = session.scalar(
                select(V35Forecast).where(
                    V35Forecast.model_market == market,
                    V35Forecast.protocol_version == self.protocol_version,
                    V35Forecast.forecast_anchor_date == anchor,
                )
            )
            snapshot_row = session.get(V35FeatureSnapshotRow, forecast.feature_snapshot_id) if forecast else None
            if forecast is None or snapshot_row is None:
                continue
            snapshot = replace(
                self._empty_snapshot(market, anchor),
                features=dict(snapshot_row.feature_json),
                daily_sequence=tuple(snapshot_row.daily_sequence_json),
                source_data_max_date=snapshot_row.source_max_date,
                provenance=dict(snapshot_row.provenance_json),
            )
            decisions[anchor] = self._strategy.decide(
                market,
                snapshot,
                expected_path=forecast.expected_path_json,
                p10_path=forecast.price_quantiles_json[0],
                probabilities_4=forecast.horizon_probabilities_json.get("4", [0.34, 0.33, 0.33]),
                probabilities_8=forecast.horizon_probabilities_json.get("8", [0.34, 0.33, 0.33]),
                current_position_pp=position_pp,
                reliability_score=float(forecast.model_reliability_score),
                health_status=forecast.health_status,
                ood_score=float(forecast.ood_json.get("ood_score", 0.0)),
                config=dict(package.strategy_config_json),
            )
            decision = decisions[anchor]
            if self.v36:
                hard_clear = decision.confirmation_status in (
                    "TOP_CONFIRMED",
                    "BEARISH_CONFIRMED",
                ) or any(
                    "紧急" in str(reason) or "OOD" in str(reason)
                    for reason in decision.reasons
                )
                lowered_below_position = (
                    decision.final_target_position_pp < position_pp
                )
                if hard_clear or lowered_below_position:
                    pending_batches = list(decision.batches)
                elif pending_batches:
                    pending_target = int(pending_batches[-1]["target_position_pp"])
                    if decision.final_target_position_pp > pending_target:
                        pending_batches = list(decision.batches)
                else:
                    pending_batches = list(decision.batches)
            else:
                pending_batches = list(decision.batches)
        return decisions

    def _rebuild_online_account(
        self,
        session: Session,
        market: str,
        anchors: Sequence[date],
        bars: Mapping[date, dict[str, float | None]],
        stop_anchor: date,
        *,
        accounting_mode: str = "PERCENT",
    ) -> _OnlineAccount:
        """Reconstruct the continuous position tracker from frozen decisions."""

        online = _OnlineAccount(accounting_mode=accounting_mode)
        for anchor in anchors:
            if anchor >= stop_anchor:
                break
            online.execute_and_mark(anchor, bars.get(anchor))
            decision = self._decisions_for_anchor(session, market, anchor)
            if decision is not None:
                online.set_decision(decision)
        return online

    def _snapshot_for(
        self,
        session: Session,
        market: str,
        anchor: date,
    ) -> dict[str, Any] | None:
        row = session.scalar(
            select(V36AccountSnapshot).where(
                V36AccountSnapshot.protocol_version == self.protocol_version,
                V36AccountSnapshot.model_market == market,
                V36AccountSnapshot.anchor_date == anchor,
            )
        )
        if row is None:
            return None
        return {
            "equity": float(row.equity),
            "cash": float(row.cash),
            "position_pp": float(row.position_pp),
            "held_shares": float(row.held_shares),
            "average_cost": float(row.average_cost or 0.0),
            "sellable_shares": float(row.sellable_shares),
            "pending_sellable": [
                (date.fromisoformat(acquired), float(shares))
                for acquired, shares in (row.cooldown_state_json or {}).get(
                    "pending_sellable", []
                )
            ],
            "pending_batches": list(row.pending_batches_json or []),
        }

    def _persist_account_snapshot(
        self,
        session: Session,
        market: str,
        anchor: date,
        online: _OnlineAccount,
    ) -> None:
        payload = online.snapshot(anchor)
        snapshot_hash = _hash(payload)
        row = session.scalar(
            select(V36AccountSnapshot).where(
                V36AccountSnapshot.protocol_version == self.protocol_version,
                V36AccountSnapshot.model_market == market,
                V36AccountSnapshot.anchor_date == anchor,
            )
        )
        cooldown_state = {
            "pending_sellable": [
                (acquired.isoformat(), round(shares, 2))
                for acquired, shares in online.pending_sellable
            ]
        }
        if row is None:
            row = V36AccountSnapshot(
                protocol_version=self.protocol_version,
                model_market=market,
                anchor_date=anchor,
                cash=_money(payload["cash"]),
                equity=_money(payload["equity"]),
                position_pp=round(payload["position_pp"]),
                held_shares=_money(payload["held_shares"]),
                average_cost=(
                    _money(payload["average_cost"])
                    if payload["average_cost"]
                    else None
                ),
                sellable_shares=_money(payload["sellable_shares"]),
                cooldown_state_json=cooldown_state,
                pending_batches_json=payload["pending_batches"],
                snapshot_hash=snapshot_hash,
                created_at=utc_now(),
            )
            session.add(row)
        else:
            row.cash = _money(payload["cash"])
            row.equity = _money(payload["equity"])
            row.position_pp = round(payload["position_pp"])
            row.held_shares = _money(payload["held_shares"])
            row.average_cost = (
                _money(payload["average_cost"]) if payload["average_cost"] else None
            )
            row.sellable_shares = _money(payload["sellable_shares"])
            row.cooldown_state_json = cooldown_state
            row.pending_batches_json = payload["pending_batches"]
            row.snapshot_hash = snapshot_hash

    def _persist_decision_funnel(
        self,
        session: Session,
        market: str,
        anchor: date,
        package_id: str,
        funnel: Mapping[str, Any],
    ) -> None:
        if not self.v36:
            return
        payload = {
            "market": market,
            "anchor": anchor.isoformat(),
            "package": package_id,
            **dict(funnel),
        }
        funnel_hash = _hash(payload)
        row = session.scalar(
            select(V36DecisionFunnel).where(
                V36DecisionFunnel.protocol_version == self.protocol_version,
                V36DecisionFunnel.model_market == market,
                V36DecisionFunnel.forecast_anchor_date == anchor,
                V36DecisionFunnel.model_package_id == package_id,
            )
        )
        values = {
            "bull_signal_count": int(funnel.get("bull_signal_count", 0)),
            "score_pass_count": int(funnel.get("score_pass_count", 0)),
            "base_position_positive_count": int(
                funnel.get("base_position_positive_count", 0)
            ),
            "risk_blocked_count": int(funnel.get("risk_blocked_count", 0)),
            "cooldown_blocked_count": int(funnel.get("cooldown_blocked_count", 0)),
            "final_position_positive_count": int(
                funnel.get("final_position_positive_count", 0)
            ),
            "trade_count": int(funnel.get("trade_count", 0)),
            "actual_position_positive_count": int(
                funnel.get("actual_position_positive_count", 0)
            ),
        }
        if row is None:
            session.add(
                V36DecisionFunnel(
                    protocol_version=self.protocol_version,
                    model_market=market,
                    forecast_anchor_date=anchor,
                    model_package_id=package_id,
                    **values,
                    funnel_hash=funnel_hash,
                    created_at=utc_now(),
                )
            )
        else:
            for key, value in values.items():
                setattr(row, key, value)
            row.funnel_hash = funnel_hash

    def bootstrap_sync(
        self,
        market: str,
        *,
        run_id: str | None = None,
        maximum_weeks: int | None = None,
    ) -> dict[str, Any]:
        self.validate_market(market)
        run_id = run_id or f"{self.protocol_version}:{market}:{utc_now().isoformat()}"
        with self._factory() as session, session.begin():
            existing_run = session.get(V35TrainingRun, run_id)
            if existing_run is None:
                session.add(
                    V35TrainingRun(
                        id=run_id,
                        protocol_version=self.protocol_version,
                        model_market=market,
                        profile_id=None,
                        run_type="BOOTSTRAP",
                        status="RUNNING",
                        current_stage="BOOTSTRAP",
                        horizon_weeks=HORIZON_WEEKS,
                        maximum_backlog_weeks=0,
                        started_at=utc_now(),
                        result_json={},
                    )
                )
        with self._factory() as session:
            anchors = self._anchors(session, market)
        processed = 0
        warmup = 0
        last_anchor: date | None = None
        with self._factory() as session, session.begin():
            bars = load_weekly_bars(session, market, anchors)
            bootstrap = session.scalar(
                select(V35BootstrapState).where(
                    V35BootstrapState.model_market == market,
                    V35BootstrapState.protocol_version == self.protocol_version,
                )
            )
            start_index = 0
            if bootstrap is not None and bootstrap.last_completed_anchor is not None:
                try:
                    start_index = anchors.index(bootstrap.last_completed_anchor) + 1
                except ValueError:
                    start_index = 0
            accounting_mode = "SHARES" if self.v36 else "PERCENT"
            online = _OnlineAccount(
                accounting_mode=accounting_mode,
                initial=(
                    self._snapshot_for(session, market, anchors[start_index - 1])
                    if self.v36 and start_index > 0
                    else None
                ),
            )
            for anchor in anchors[start_index:]:
                if maximum_weeks is not None and processed >= maximum_weeks:
                    break
                if session.scalar(
                    select(V35TrainingIteration).where(
                        V35TrainingIteration.protocol_version == self.protocol_version,
                        V35TrainingIteration.model_market == market,
                        V35TrainingIteration.anchor_date == anchor,
                    )
                ) is not None:
                    continue
                online.execute_and_mark(anchor, bars.get(anchor))
                outcome = self._process_week(session, market, anchor, run_id, online, bars)
                if outcome.get("status") == "WARMUP":
                    warmup += 1
                else:
                    processed += 1
                last_anchor = anchor
            if self.v351:
                self._finalize_pending_v351(session, market)
            run = session.get(V35TrainingRun, run_id)
            if run is not None:
                run.status = "COMPLETED"
                run.completed_at = utc_now()
                run.result_json = {
                    "processed": processed,
                    "warmup": warmup,
                    "last_anchor": last_anchor.isoformat() if last_anchor else None,
                }
            bootstrap = session.scalar(
                select(V35BootstrapState).where(
                    V35BootstrapState.model_market == market,
                    V35BootstrapState.protocol_version == self.protocol_version,
                )
            )
            if (
                bootstrap is not None
                and anchors
                and bootstrap.last_completed_anchor == anchors[-1]
            ):
                bootstrap.state = "COMPLETED"
                bootstrap.updated_at = utc_now()
        if processed or warmup:
            self._finalize_continuous(session_factory=self._factory, market=market, anchors=anchors)
        return {
            "market": market,
            "run_id": run_id,
            "processed_weeks": processed,
            "warmup_weeks": warmup,
            "last_anchor": last_anchor.isoformat() if last_anchor else None,
        }

    def _finalize_continuous(
        self,
        *,
        session_factory: sessionmaker,
        market: str,
        anchors: Sequence[date],
    ) -> None:
        with session_factory() as session, session.begin():
            package = self._champion_package(session, market, anchors[-1])
            if package is None:
                return
            decisions = {
                anchor: decision
                for anchor in anchors
                if (decision := self._decisions_for_anchor(session, market, anchor)) is not None
            }
            bars = load_weekly_bars(session, market, anchors)
            previous_anchor = (
                session.scalar(
                    select(func.max(V36AccountSnapshot.anchor_date)).where(
                        V36AccountSnapshot.protocol_version
                        == self.protocol_version,
                        V36AccountSnapshot.model_market == market,
                        V36AccountSnapshot.anchor_date < anchors[0],
                    )
                )
                if self.v36 and anchors
                else None
            )
            result = run_continuous_window(
                session,
                market=market,
                package_id=package.id,
                anchors=anchors,
                decisions=decisions,
                bars=bars,
                config=dict(package.strategy_config_json),
                accounting_mode="SHARES" if self.v36 else "PERCENT",
                initial_snapshot=(
                    self._snapshot_for(session, market, previous_anchor)
                    if self.v36 and previous_anchor is not None
                    else None
                ),
            )
            market_returns = [
                float(bar["close"]) / float(bar["open"]) - 1.0
                for bar in bars.values()
                if bar["open"] not in (None, 0.0) and bar["close"] not in (None, 0.0)
            ]
            elapsed_years = max(len(anchors) / 52.0, 0.01)
            persist_continuous_result(
                session,
                result,
                market_returns=market_returns,
                elapsed_years=elapsed_years,
                protocol_version=self.protocol_version,
            )

    def incremental_sync(self, market: str, *, run_id: str | None = None) -> dict[str, Any]:
        return self.bootstrap_sync(market, run_id=run_id)

    def status(self, market: str) -> dict[str, Any]:
        self.validate_market(market)
        with self._factory() as session:
            from sqlalchemy import func

            bootstrap = session.scalar(
                select(V35BootstrapState).where(
                    V35BootstrapState.model_market == market,
                    V35BootstrapState.protocol_version == self.protocol_version,
                )
            )
            forecast_count = int(
                session.scalar(
                    select(func.count())
                    .select_from(V35Forecast)
                    .where(
                        V35Forecast.model_market == market,
                        V35Forecast.protocol_version == self.protocol_version,
                    )
                )
                or 0
            )
            evaluation_count = int(
                session.scalar(
                    select(func.count())
                    .select_from(V35ForecastEvaluation)
                    .join(
                        V35Forecast,
                        V35ForecastEvaluation.forecast_id == V35Forecast.id,
                    )
                    .where(
                        V35Forecast.model_market == market,
                        V35Forecast.protocol_version == self.protocol_version,
                    )
                )
                or 0
            )
            return {
                "market": market,
                "protocol_version": self.protocol_version,
                "state": bootstrap.state if bootstrap is not None else "NOT_STARTED",
                "first_formal_anchor": (
                    bootstrap.first_formal_anchor.isoformat() if bootstrap and bootstrap.first_formal_anchor else None
                ),
                "last_completed_anchor": (
                    bootstrap.last_completed_anchor.isoformat() if bootstrap and bootstrap.last_completed_anchor else None
                ),
                "weekly_iteration_count": bootstrap.weekly_iteration_count if bootstrap else 0,
                "prediction_challenge_count": bootstrap.prediction_challenge_count if bootstrap else 0,
                "strategy_challenge_count": bootstrap.strategy_challenge_count if bootstrap else 0,
                "promotion_count": bootstrap.promotion_count if bootstrap else 0,
                "forecast_count": forecast_count,
                "evaluation_count": evaluation_count,
                "champion_package_id": bootstrap.champion_package_id if bootstrap else None,
            }

    def latest_forecast(self, market: str) -> dict[str, Any] | None:
        self.validate_market(market)
        with self._factory() as session:
            row = session.scalar(
                select(V35Forecast)
                .where(
                    V35Forecast.model_market == market,
                    V35Forecast.protocol_version == self.protocol_version,
                )
                .order_by(V35Forecast.forecast_anchor_date.desc())
            )
            if row is None:
                return None
            return {
                "forecast_anchor_date": row.forecast_anchor_date.isoformat(),
                "expected_path": row.expected_path_json,
                "price_quantiles": row.price_quantiles_json,
                "horizon_probabilities": row.horizon_probabilities_json,
                "health_status": row.health_status,
                "maturity_status": row.maturity_status,
                "forecast_hash": row.forecast_hash,
            }
