"""V3.7 DeepSeek Model Reviewer and LOCAL_REPAIR_CHALLENGER service.

Leakage rule (2026-08-07): the reviewer may use history up to the proposal
anchor T for training and sanity checks, but formal promotion evidence only
comes from matured OOS windows that start at or after T + purge gap (8
weeks).  Proposals may only change whitelisted parameters; architecture
changes are logged as proposals, never applied automatically.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import hashlib
import json
from typing import Any, Mapping

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    V35Forecast,
    V35BootstrapState,
    V35ModelPackage,
    V35PositionDecision,
    V35Promotion,
    V35StrategySnapshot,
    V36DecisionFunnel,
    V37AiRequest,
    V37ModelRepairProposal,
    V37ModelRepairRun,
    utc_now,
)
from backend.app.services.deepseek_client import (
    DeepSeekConfig,
    DeepSeekError,
    ERROR_INVALID_RESPONSE,
    chat_json_detailed,
)
from backend.app.services.v351_behavior_service import (
    effective_independent_window_count,
)
from backend.app.services.v35_simulation_service import (
    load_weekly_bars,
    run_standard_window,
)
from backend.app.services.v35_strategy_service import V35StrategyDecision
from backend.app.services.v37_config import (
    AI_MAX_ATTEMPTS,
    AI_PROMPT_VERSION_REVIEWER,
    AI_PROVIDER,
    AI_RETRYABLE_ERROR_CODES,
    AI_SCOPE_FORWARD_OOS,
    MODEL_REVIEW_EVERY_MATURED,
    PROTOCOL_VERSION_37,
    REPAIR_PARAMETER_RANGES,
    REPAIR_PARAMETER_WHITELIST,
    REPAIR_PROMOTION_PURGE_WEEKS,
)


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


def review_due(
    session: Session,
    market: str,
    anchor: date,
) -> tuple[bool, str]:
    """Return whether a DeepSeek model review is due."""

    last_proposal_anchor = session.scalar(
        select(func.max(V37ModelRepairProposal.anchor_date)).where(
            V37ModelRepairProposal.protocol_version == PROTOCOL_VERSION_37,
            V37ModelRepairProposal.model_market == market,
        )
    )
    matured_since = 0
    if last_proposal_anchor is None:
        matured_since = int(
            session.scalar(
                select(func.count())
                .select_from(V35Forecast)
                .where(
                    V35Forecast.protocol_version == PROTOCOL_VERSION_37,
                    V35Forecast.model_market == market,
                    V35Forecast.maturity_status == "FULLY_MATURE_8W",
                )
            )
            or 0
        )
    else:
        matured_since = int(
            session.scalar(
                select(func.count())
                .select_from(V35Forecast)
                .where(
                    V35Forecast.protocol_version == PROTOCOL_VERSION_37,
                    V35Forecast.model_market == market,
                    V35Forecast.maturity_status == "FULLY_MATURE_8W",
                    V35Forecast.forecast_anchor_date >= last_proposal_anchor,
                )
            )
            or 0
        )
    if matured_since >= MODEL_REVIEW_EVERY_MATURED:
        return True, f"MATURED_SAMPLES={matured_since}"
    # Drift triggers (lightweight checks).
    funnel_rows = session.scalars(
        select(V36DecisionFunnel)
        .where(
            V36DecisionFunnel.protocol_version == PROTOCOL_VERSION_37,
            V36DecisionFunnel.model_market == market,
        )
        .order_by(V36DecisionFunnel.forecast_anchor_date.desc())
        .limit(26)
    ).all()
    if len(funnel_rows) >= 12:
        risk_blocked = sum(row.risk_blocked_count for row in funnel_rows)
        bull_signals = sum(row.bull_signal_count for row in funnel_rows)
        if bull_signals > 0 and risk_blocked / bull_signals > 0.6:
            return True, "DECISION_FUNNEL_ABNORMAL"
    return False, ""


def build_review_context(
    session: Session,
    market: str,
    anchor: date,
) -> dict[str, Any]:
    """Assemble local diagnostics for the reviewer (no future data)."""

    forecasts = session.scalars(
        select(V35Forecast)
        .where(
            V35Forecast.protocol_version == PROTOCOL_VERSION_37,
            V35Forecast.model_market == market,
            V35Forecast.forecast_anchor_date <= anchor,
        )
        .order_by(V35Forecast.forecast_anchor_date.desc())
        .limit(16)
    ).all()
    funnel_rows = session.scalars(
        select(V36DecisionFunnel)
        .where(
            V36DecisionFunnel.protocol_version == PROTOCOL_VERSION_37,
            V36DecisionFunnel.model_market == market,
            V36DecisionFunnel.forecast_anchor_date <= anchor,
        )
        .order_by(V36DecisionFunnel.forecast_anchor_date.desc())
        .limit(26)
    ).all()
    return {
        "market": market,
        "anchor": anchor.isoformat(),
        "local_forecasts": [
            {
                "anchor": row.forecast_anchor_date.isoformat(),
                "expected_8w": float(row.expected_path_json[-1]),
                "prob_up_8w": float(
                    row.horizon_probabilities_json.get("8", [0.5, 0.0, 0.5])[0]
                ),
                "health": row.health_status,
                "reliability": float(row.model_reliability_score),
            }
            for row in forecasts
        ],
        "decision_funnel": [
            {
                "anchor": row.forecast_anchor_date.isoformat(),
                "bull_signal": row.bull_signal_count,
                "risk_blocked": row.risk_blocked_count,
                "final_positive": row.final_position_positive_count,
                "actual_positive": row.actual_position_positive_count,
            }
            for row in funnel_rows
        ],
    }


def build_reviewer_prompt(context: Mapping[str, Any]) -> tuple[str, str]:
    system = (
        "You are a model reviewer for a local 8-week quant system. Diagnose "
        "the local model from the provided point-in-time diagnostics and "
        "propose ONE whitelisted repair. Output exactly one JSON object with "
        "fields: diagnosis, target_layer, severity (LOW|MEDIUM|HIGH), "
        "proposed_change, parameter_changes (object of whitelisted parameters "
        "only), expected_effect, primary_risk, requires_full_replay (bool). "
        "Whitelisted parameters: ridge_alpha, training_window, feature_subset, "
        "daily_feature_set, weekly_feature_set, daily_weekly_fusion_weight, "
        "residual_half_life, calibration_mode, model_family, "
        "strong_signal_threshold, entry_floor, vol_target, conflict_penalty, "
        "risk_gate_parameter. Do not propose source-code or architecture "
        "changes here."
    )
    user = json.dumps(context, ensure_ascii=False, default=str)
    return system, user


def validate_repair_proposal(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a repair proposal against the whitelist."""

    diagnosis = str(parsed.get("diagnosis") or "").strip()
    target_layer = str(parsed.get("target_layer") or "").strip()
    severity = str(parsed.get("severity") or "")
    proposed_change = str(parsed.get("proposed_change") or "").strip()
    if not diagnosis or not target_layer or not proposed_change:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE,
            "diagnosis/target_layer/proposed_change must be non-empty",
        )
    if severity not in {"LOW", "MEDIUM", "HIGH"}:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "severity must be LOW|MEDIUM|HIGH"
        )
    parameter_changes = parsed.get("parameter_changes")
    if not isinstance(parameter_changes, dict):
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "parameter_changes must be an object"
        )
    validated_changes: dict[str, Any] = {}
    for key, value in parameter_changes.items():
        key = str(key)
        if key not in REPAIR_PARAMETER_WHITELIST:
            raise DeepSeekError(
                ERROR_INVALID_RESPONSE,
                f"parameter {key} is not whitelisted",
            )
        if key in REPAIR_PARAMETER_RANGES:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise DeepSeekError(
                    ERROR_INVALID_RESPONSE, f"{key} must be numeric"
                ) from exc
            lower, upper = REPAIR_PARAMETER_RANGES[key]
            if not lower <= number <= upper:
                raise DeepSeekError(
                    ERROR_INVALID_RESPONSE,
                    f"{key}={number} outside [{lower}, {upper}]",
                )
            validated_changes[key] = number
        else:
            if not isinstance(value, str) or not value.strip():
                raise DeepSeekError(
                    ERROR_INVALID_RESPONSE, f"{key} must be a non-empty string"
                )
            validated_changes[key] = value.strip()
    if not validated_changes:
        raise DeepSeekError(
            ERROR_INVALID_RESPONSE, "parameter_changes must not be empty"
        )
    requires_full_replay = bool(parsed.get("requires_full_replay", True))
    return {
        "diagnosis": diagnosis,
        "target_layer": target_layer,
        "severity": severity,
        "proposed_change": proposed_change,
        "parameter_changes": validated_changes,
        "expected_effect": str(parsed.get("expected_effect") or "").strip(),
        "primary_risk": str(parsed.get("primary_risk") or "").strip(),
        "requires_full_replay": requires_full_replay,
        "architecture_change_proposal": (
            dict(parsed["architecture_change_proposal"])
            if isinstance(parsed.get("architecture_change_proposal"), dict)
            else {}
        ),
    }


def ensure_repair_proposal(
    session: Session,
    market: str,
    anchor: date,
    config: DeepSeekConfig,
    *,
    trigger: str,
    generation: int = 1,
    max_attempts: int = AI_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Call the reviewer once (bounded retry) and persist a proposal."""

    existing = session.scalar(
        select(V37ModelRepairProposal).where(
            V37ModelRepairProposal.protocol_version == PROTOCOL_VERSION_37,
            V37ModelRepairProposal.model_market == market,
            V37ModelRepairProposal.anchor_date == anchor,
        )
    )
    if existing is not None:
        return {"cached": True, "proposal_id": existing.id}
    context = build_review_context(session, market, anchor)
    system, user = build_reviewer_prompt(context)
    input_hash = _hash(context)
    last_error: dict[str, Any] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            parsed, elapsed_ms, usage = chat_json_detailed(
                config, system=system, user=user
            )
            validated = validate_repair_proposal(parsed)
        except DeepSeekError as exc:
            session.add(
                V37AiRequest(
                    protocol_version=PROTOCOL_VERSION_37,
                    model_market=market,
                    forecast_anchor_date=anchor,
                    provider=AI_PROVIDER,
                    model_name=config.model,
                    model_version=None,
                    prompt_version=AI_PROMPT_VERSION_REVIEWER,
                    schema_version="V37_MODEL_REVIEWER_1",
                    ai_generation_version=generation,
                    attempt_number=attempt,
                    screening_scope=AI_SCOPE_FORWARD_OOS,
                    input_hash=input_hash,
                    output_hash=None,
                    raw_response_hash=None,
                    response_status=exc.code,
                    request_timestamp=utc_now(),
                    latency_ms=0,
                    token_usage_json={},
                    error_code=exc.code,
                    error_detail=exc.detail,
                    point_in_time_pass=True,
                    created_at=utc_now(),
                )
            )
            session.flush()
            last_error = {
                "status": exc.code,
                "error_code": exc.code,
                "error_detail": exc.detail,
                "attempt": attempt,
            }
            if exc.code not in AI_RETRYABLE_ERROR_CODES or attempt >= max_attempts:
                return {"cached": False, **last_error}
            continue
        output_hash = _hash(validated)
        session.add(
            V37AiRequest(
                protocol_version=PROTOCOL_VERSION_37,
                model_market=market,
                forecast_anchor_date=anchor,
                provider=AI_PROVIDER,
                model_name=config.model,
                model_version=None,
                prompt_version=AI_PROMPT_VERSION_REVIEWER,
                schema_version="V37_MODEL_REVIEWER_1",
                ai_generation_version=generation,
                attempt_number=attempt,
                screening_scope=AI_SCOPE_FORWARD_OOS,
                input_hash=input_hash,
                output_hash=output_hash,
                raw_response_hash=output_hash,
                response_status="OK",
                request_timestamp=utc_now(),
                latency_ms=elapsed_ms,
                token_usage_json=usage,
                error_code=None,
                error_detail=None,
                point_in_time_pass=True,
                created_at=utc_now(),
            )
        )
        proposal = V37ModelRepairProposal(
            protocol_version=PROTOCOL_VERSION_37,
            model_market=market,
            anchor_date=anchor,
            trigger=trigger,
            diagnosis=validated["diagnosis"],
            target_layer=validated["target_layer"],
            severity=validated["severity"],
            proposed_change=validated["proposed_change"],
            parameter_changes_json=validated["parameter_changes"],
            expected_effect=validated["expected_effect"],
            primary_risk=validated["primary_risk"],
            requires_full_replay=validated["requires_full_replay"],
            architecture_change_proposal_json=validated[
                "architecture_change_proposal"
            ],
            status="PENDING",
            proposal_hash=output_hash,
            created_at=utc_now(),
        )
        session.add(proposal)
        session.flush()
        return {
            "cached": False,
            "status": "OK",
            "proposal_id": proposal.id,
            "attempt": attempt,
        }
    return {"cached": False, **(last_error or {})}


def repair_eligible_windows(
    session: Session,
    market: str,
    proposal_anchor: date,
    *,
    purge_weeks: int = REPAIR_PROMOTION_PURGE_WEEKS,
) -> tuple[date, ...]:
    """Matured 8W windows eligible as formal repair-promotion evidence."""

    from backend.app.services.v37_feature_service import V37FeatureService

    anchors = V37FeatureService().weekly_anchors(session, market)
    try:
        proposal_index = anchors.index(proposal_anchor)
    except ValueError:
        return ()
    eligible_starts = anchors[proposal_index + purge_weeks :]
    return tuple(
        anchor
        for anchor in eligible_starts
        if anchor in anchors and anchors.index(anchor) + 8 <= len(anchors)
    )


def create_repair_challenger(
    session: Session,
    market: str,
    proposal: V37ModelRepairProposal,
    champion_package: V35ModelPackage,
    runtime: Any,
) -> V37ModelRepairRun:
    """Create a LOCAL_REPAIR_CHALLENGER package from a whitelisted proposal."""

    from backend.app.services.v37_config import (
        FEATURE_SET_EARLY_FUSION,
        FEATURE_SET_LATE_FUSION,
        FEATURE_SET_WEEKLY_ONLY,
    )

    changes = dict(proposal.parameter_changes_json)
    strategy_config = {
        **dict(champion_package.strategy_config_json),
        "v351_model_config": dict(
            champion_package.strategy_config_json.get("v351_model_config", {})
        ),
    }
    model_config = dict(strategy_config.get("v351_model_config", {}))
    architecture = str(
        model_config.get("architecture", FEATURE_SET_EARLY_FUSION)
    )
    if "daily_weekly_fusion_weight" in changes:
        model_config["daily_weekly_fusion_weight"] = changes.pop(
            "daily_weekly_fusion_weight"
        )
    if "feature_subset" in changes:
        feature_subset = str(changes.pop("feature_subset"))
        if feature_subset in {
            FEATURE_SET_WEEKLY_ONLY,
            FEATURE_SET_EARLY_FUSION,
            FEATURE_SET_LATE_FUSION,
        }:
            architecture = feature_subset
    if "daily_feature_set" in changes:
        model_config["daily_feature_set"] = changes.pop("daily_feature_set")
    if "weekly_feature_set" in changes:
        model_config["weekly_feature_set"] = changes.pop("weekly_feature_set")
    if "model_family" in changes:
        model_config["model_family"] = changes.pop("model_family")
    if "calibration_mode" in changes:
        model_config["calibration_mode"] = changes.pop("calibration_mode")
    for key in (
        "ridge_alpha",
        "training_window",
        "residual_half_life",
        "strong_signal_threshold",
        "entry_floor",
        "vol_target",
        "conflict_penalty",
        "risk_gate_parameter",
    ):
        if key in changes:
            model_config[key] = changes.pop(key)
    model_config["architecture"] = architecture
    strategy_config["v351_model_config"] = model_config
    version = (
        f"{PROTOCOL_VERSION_37}:{market}:REPAIR:"
        f"{proposal.anchor_date.isoformat()}:{proposal.id}"
    )
    package = runtime._persist_package(
        session,
        market,
        version=version,
        kind="LOCAL_REPAIR_CHALLENGER",
        prediction_model_id=champion_package.prediction_model_id,
        strategy_config=strategy_config,
        effective_from=proposal.anchor_date,
        parent_package_id=champion_package.id,
    )
    repair_run = V37ModelRepairRun(
        protocol_version=PROTOCOL_VERSION_37,
        model_market=market,
        proposal_id=proposal.id,
        challenger_package_id=package.id,
        training_run_id=None,
        status="CREATED",
        validation_json={
            "leakage_rule": (
                "formal_promotion_evidence_only_from_windows_after_"
                f"proposal_anchor_plus_{REPAIR_PROMOTION_PURGE_WEEKS}_weeks"
            ),
            "eligible_window_count": len(
                repair_eligible_windows(session, market, proposal.anchor_date)
            ),
        },
        created_at=utc_now(),
    )
    session.add(repair_run)
    proposal.status = "APPLIED"
    session.flush()
    return repair_run


def _champion_decision_for_anchor(
    session: Session,
    market: str,
    anchor: date,
    package: V35ModelPackage,
) -> V35StrategyDecision | None:
    """Rebuild a champion strategy decision from its persisted snapshot."""

    snapshot = session.scalar(
        select(V35StrategySnapshot).where(
            V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_37,
            V35StrategySnapshot.model_market == market,
            V35StrategySnapshot.forecast_anchor_date == anchor,
            V35StrategySnapshot.model_package_id == package.id,
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
        forecast_anchor_date=anchor.isoformat(),
        dif_trend_state=snapshot.dif_trend_state,
        confirmation_status=snapshot.confirmation_status,
        market_state=snapshot.confirmation_status,
        strategy_score=float(snapshot.strategy_score),
        base_target_position_pp=int(snapshot.base_target_position_pp),
        state_position_cap_pp=int(snapshot.state_position_cap_pp),
        final_target_position_pp=int(snapshot.final_target_position_pp),
        batches=tuple(
            {
                "action": batch.action,
                "target_position_pp": int(batch.target_position_pp),
                "batch_change_pp": int(batch.batch_change_pp),
                "cooldown_trading_days": int(
                    (batch.condition_json or {}).get(
                        "cooldown_trading_days", 5
                    )
                ),
                "condition": (batch.condition_json or {}).get("condition", ""),
            }
            for batch in batches
        ),
        components={},
        reasons=(),
    )


def _challenger_decision_for_anchor(
    session: Session,
    market: str,
    anchor: date,
    package: V35ModelPackage,
    runtime: Any,
) -> V35StrategyDecision | None:
    """Recompute a repair challenger decision point-in-time."""

    from backend.app.models.models import (
        V35FeatureSnapshot as V35FeatureSnapshotRow,
    )
    from backend.app.services.v35_runtime_service import (
        _probabilities_from_expected,
    )
    from backend.app.services.v35_feature_service import replace

    snapshot_row = session.scalar(
        select(V35FeatureSnapshotRow).where(
            V35FeatureSnapshotRow.protocol_version == PROTOCOL_VERSION_37,
            V35FeatureSnapshotRow.model_market == market,
            V35FeatureSnapshotRow.forecast_anchor_date == anchor,
        )
    )
    if snapshot_row is None:
        return None
    try:
        state = runtime._load_model_state(session, package)
        if anchor < state.effective_from_date:
            return None
        snapshot = replace(
            runtime._empty_snapshot(market, anchor),
            features=dict(snapshot_row.feature_json),
            daily_sequence=tuple(snapshot_row.daily_sequence_json),
            source_data_max_date=snapshot_row.source_max_date,
            provenance=dict(snapshot_row.provenance_json),
        )
        forecast = runtime._training.predict(state, snapshot)
    except Exception:
        return None
    expected_path = list(forecast.expected_path)
    sigma = forecast.forecast_sigma
    return runtime._strategy.decide(
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


def evaluate_repair_challenger(
    session: Session,
    market: str,
    repair_run: V37ModelRepairRun,
    runtime: Any,
    as_of: date,
) -> dict[str, Any]:
    """Sealed repair evaluation: only windows after proposal_anchor+8w count."""

    proposal = session.get(V37ModelRepairProposal, repair_run.proposal_id)
    if proposal is None or repair_run.challenger_package_id is None:
        repair_run.status = "FAILED"
        repair_run.validation_json = {"reason": "MISSING_PROPOSAL_OR_PACKAGE"}
        session.flush()
        return {"decision": "FAILED", "reason": "MISSING_PROPOSAL_OR_PACKAGE"}
    challenger = session.get(V35ModelPackage, repair_run.challenger_package_id)
    champion_state = session.scalar(
        select(V35BootstrapState).where(
            V35BootstrapState.protocol_version == PROTOCOL_VERSION_37,
            V35BootstrapState.model_market == market,
        )
    )
    if challenger is None or champion_state is None or champion_state.champion_package_id is None:
        repair_run.status = "FAILED"
        repair_run.validation_json = {"reason": "MISSING_PACKAGES"}
        session.flush()
        return {"decision": "FAILED", "reason": "MISSING_PACKAGES"}
    champion = session.get(V35ModelPackage, champion_state.champion_package_id)

    from backend.app.services.v37_feature_service import V37FeatureService

    anchors = V37FeatureService().weekly_anchors(session, market)
    try:
        proposal_index = anchors.index(proposal.anchor_date)
    except ValueError:
        repair_run.status = "FAILED"
        repair_run.validation_json = {"reason": "PROPOSAL_ANCHOR_NOT_FOUND"}
        session.flush()
        return {"decision": "FAILED", "reason": "PROPOSAL_ANCHOR_NOT_FOUND"}
    eligible_starts = anchors[
        proposal_index + REPAIR_PROMOTION_PURGE_WEEKS :
    ]
    windows: list[tuple[date, date]] = []
    for start in eligible_starts:
        try:
            index = anchors.index(start)
        except ValueError:
            continue
        end_index = index + 8 - 1
        if end_index < len(anchors) and anchors[end_index] <= as_of:
            windows.append((start, anchors[end_index]))
    independent = effective_independent_window_count(windows)
    if len(windows) < 8 or independent < 8:
        repair_run.status = "SHADOW_EVALUATION"
        repair_run.validation_json = {
            "reason": "INSUFFICIENT_ELIGIBLE_WINDOWS",
            "eligible_window_count": len(windows),
            "independent_window_count": independent,
            "leakage_rule": (
                "only_windows_starting_after_proposal_anchor_plus_8_weeks"
            ),
        }
        session.flush()
        return {
            "decision": "SHADOW_EVALUATION",
            "eligible_window_count": len(windows),
            "independent_window_count": independent,
        }

    bars = load_weekly_bars(session, market, anchors)
    results: dict[str, list[float]] = {
        "champion_net_return": [],
        "challenger_net_return": [],
        "champion_drawdown": [],
        "challenger_drawdown": [],
        "champion_position": [],
        "challenger_position": [],
    }
    for start, end in windows:
        window = anchors[
            anchors.index(start) : anchors.index(start) + 8
        ]
        champion_decisions: dict[date, V35StrategyDecision] = {}
        challenger_decisions: dict[date, V35StrategyDecision] = {}
        complete = True
        for anchor in window:
            champion_decision = _champion_decision_for_anchor(
                session, market, anchor, champion
            )
            challenger_decision = _challenger_decision_for_anchor(
                session, market, anchor, challenger, runtime
            )
            if champion_decision is None or challenger_decision is None:
                complete = False
                break
            champion_decisions[anchor] = champion_decision
            challenger_decisions[anchor] = challenger_decision
        if not complete:
            continue
        champion_result = run_standard_window(
            session,
            market=market,
            package_id=champion.id,
            anchors=window,
            decisions=champion_decisions,
            bars=bars,
            config=dict(champion.strategy_config_json),
            accounting_mode="PERCENT",
        )
        challenger_result = run_standard_window(
            session,
            market=market,
            package_id=challenger.id,
            anchors=window,
            decisions=challenger_decisions,
            bars=bars,
            config=dict(challenger.strategy_config_json),
            accounting_mode="PERCENT",
        )
        results["champion_net_return"].append(float(champion_result.net_return))
        results["challenger_net_return"].append(float(challenger_result.net_return))
        results["champion_drawdown"].append(float(champion_result.max_drawdown))
        results["challenger_drawdown"].append(float(challenger_result.max_drawdown))
        results["champion_position"].append(
            float(champion_result.average_position_pp)
        )
        results["challenger_position"].append(
            float(challenger_result.average_position_pp)
        )

    if not results["challenger_net_return"]:
        repair_run.status = "SHADOW_EVALUATION"
        repair_run.validation_json = {"reason": "NO_COMPARABLE_WINDOWS"}
        session.flush()
        return {"decision": "SHADOW_EVALUATION", "reason": "NO_COMPARABLE_WINDOWS"}
    excesses = [
        challenger - champion
        for challenger, champion in zip(
            results["challenger_net_return"], results["champion_net_return"]
        )
    ]
    mean_excess = float(np.mean(excesses))
    win_rate = float(np.mean([1.0 if value > 0.0 else 0.0 for value in excesses]))
    drawdown_degradation = float(
        np.mean(
            [
                challenger - champion
                for challenger, champion in zip(
                    results["challenger_drawdown"],
                    results["champion_drawdown"],
                )
            ]
        )
    )
    participation_diff = float(
        np.mean(
            [
                challenger - champion
                for challenger, champion in zip(
                    results["challenger_position"],
                    results["champion_position"],
                )
            ]
        )
    )
    reason_codes: list[str] = []
    if mean_excess < 0.0030:
        reason_codes.append("EXCESS_RETURN_BELOW_0_30")
    if win_rate < 0.55:
        reason_codes.append("WIN_RATE_BELOW_THRESHOLD")
    if drawdown_degradation > 0.02:
        reason_codes.append("DRAWDOWN_GATE_FAILED")
    if participation_diff < -0.05:
        reason_codes.append("PARTICIPATION_GATE_FAILED")
    validation = {
        "eligible_window_count": len(windows),
        "independent_window_count": independent,
        "mean_excess_return": round(mean_excess, 6),
        "win_rate": round(win_rate, 4),
        "drawdown_degradation_pp": round(drawdown_degradation * 100.0, 4),
        "participation_diff_pp": round(participation_diff * 100.0, 4),
        "reason_codes": reason_codes,
        "leakage_rule": (
            "only_windows_starting_after_proposal_anchor_plus_8_weeks"
        ),
    }
    if reason_codes:
        repair_run.status = "FAILED"
        repair_run.validation_json = validation
        repair_run.completed_at = utc_now()
        session.flush()
        return {"decision": "REJECTED", "reason_codes": reason_codes, **validation}

    challenge_id = (
        f"{PROTOCOL_VERSION_37}:{market}:REPAIR_CHALLENGE:"
        f"{proposal.anchor_date.isoformat()}:{repair_run.id}"
    )
    existing_promotion = session.scalar(
        select(V35Promotion).where(V35Promotion.challenge_id == challenge_id)
    )
    if existing_promotion is None:
        session.add(
            V35Promotion(
                protocol_version=PROTOCOL_VERSION_37,
                model_market=market,
                challenge_id=challenge_id,
                champion_package_id=champion.id,
                candidate_package_id=challenger.id,
                challenger_family="LOCAL_REPAIR_CHALLENGER",
                evaluation_window_count=len(windows),
                promotion_decision="PROMOTED",
                promotion_reason="LOCAL_AI_REPAIR_PROMOTION",
                effective_from_date=windows[-1][1],
                excess_profit=Decimal("0"),
                excess_return=Decimal(str(round(mean_excess, 8))),
                metrics_json=validation,
                promotion_hash=_hash(validation),
                created_at=utc_now(),
            )
        )
        champion_state.promotion_count = (champion_state.promotion_count or 0) + 1
    repair_run.status = "COMPLETED"
    repair_run.validation_json = validation
    repair_run.completed_at = utc_now()
    session.flush()
    return {"decision": "PROMOTED", **validation}
