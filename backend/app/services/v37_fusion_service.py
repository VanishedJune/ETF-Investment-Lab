"""V3.7 Quant-AI Fusion: deterministic challenger configs and accounts.

Fusion only uses ``ai_calibrated_probability``; when calibration is
unavailable the candidate degrades to LOCAL_ONLY (AI weight = 0).  Fusion
accounts run through the same execution engine and are scoped by their own
model_package_id so they can never overwrite Local or AI accounts.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import hashlib
import json
from typing import Any, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    V35ModelPackage,
    V35SimAccount,
    V35StrategySnapshot,
    V36AccountSnapshot,
    V37AiForecast,
    V37FusionConfig,
    V37FusionEvaluation,
    utc_now,
)
from backend.app.services.v35_runtime_service import _hash
from backend.app.services.v35_simulation_service import (
    load_weekly_bars,
    persist_window_result,
    run_standard_window,
)
from backend.app.services.v35_strategy_service import V35StrategyDecision
from backend.app.services.v36_calibration import apply_method
from backend.app.services.v37_ai_service import ai_calibration_status
from backend.app.services.v37_config import (
    AI_SCOPE_FORWARD_OOS,
    FUSION_INITIAL_CANDIDATES,
    PROTOCOL_VERSION_37,
)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def ensure_fusion_configs(
    session: Session,
    market: str,
) -> tuple[V37FusionConfig, ...]:
    """Create the four initial fusion challenger configs (idempotent)."""

    existing = session.scalars(
        select(V37FusionConfig).where(
            V37FusionConfig.protocol_version == PROTOCOL_VERSION_37,
            V37FusionConfig.model_market == market,
        )
    ).all()
    if existing:
        return tuple(existing)
    created: list[V37FusionConfig] = []
    for local_weight, ai_weight in FUSION_INITIAL_CANDIDATES:
        version = f"LOCAL_{local_weight:02d}_AI_{ai_weight:02d}"
        payload = {
            "protocol_version": PROTOCOL_VERSION_37,
            "market": market,
            "config_version": version,
            "local_weight": local_weight / 100.0,
            "ai_weight": ai_weight / 100.0,
            "conflict_policy": "NONE",
        }
        config = V37FusionConfig(
            protocol_version=PROTOCOL_VERSION_37,
            model_market=market,
            config_version=version,
            local_weight=Decimal(str(local_weight / 100.0)),
            ai_weight=Decimal(str(ai_weight / 100.0)),
            conflict_policy="NONE",
            strategy_config_json=payload,
            config_hash=_hash_payload(payload),
            created_at=utc_now(),
        )
        session.add(config)
        created.append(config)
    session.flush()
    return tuple(created)


def fusion_position(
    *,
    local_position_pp: int,
    local_score: float,
    ai_up_probability: float | None,
    config: V37FusionConfig,
    champion_strategy_config: Mapping[str, Any],
    conflict_state: str = "NONE",
) -> int:
    """Deterministic fusion target position (5% grid, capped at 80%)."""

    local_weight = float(config.local_weight)
    ai_weight = float(config.ai_weight)
    if ai_up_probability is None:
        ai_weight = 0.0
        local_weight = 1.0
    local_norm = float(max(0.0, min(1.0, local_score / 100.0)))
    ai_norm = (
        0.0
        if ai_up_probability is None
        else float(max(0.0, min(1.0, ai_up_probability)))
    )
    fused = local_weight * local_norm + ai_weight * ai_norm
    mapping = champion_strategy_config["score_position_map"]
    position = 0
    for entry in mapping:
        if fused <= float(entry["score_gt"]):
            position = int(entry["position_pp"])
            break
    else:
        position = int(mapping[-1]["position_pp"])
    conflict_policy = str(config.conflict_policy)
    if conflict_state != "NONE":
        if conflict_policy == "REDUCE_5PP":
            position -= 5
        elif conflict_policy == "REDUCE_10PP":
            position -= 10
        elif conflict_policy == "WAIT_CONFIRMATION":
            position = local_position_pp
    max_position = int(
        champion_strategy_config.get("max_position_pp", 80)
    )
    position = max(0, min(max_position, position))
    return int(round(position / 5.0) * 5)


def fusion_decision(
    market: str,
    anchor: date,
    *,
    local_snapshot: V35StrategySnapshot,
    ai_forecast: V37AiForecast | None,
    config: V37FusionConfig,
    champion_strategy_config: Mapping[str, Any],
    conflict_state: str,
) -> V35StrategyDecision:
    target = fusion_position(
        local_position_pp=int(local_snapshot.final_target_position_pp),
        local_score=float(local_snapshot.strategy_score),
        ai_up_probability=(
            float(ai_forecast.direction_scores_json.get("direction_score_8w", 0.5))
            if ai_forecast is not None
            else None
        ),
        config=config,
        champion_strategy_config=champion_strategy_config,
        conflict_state=conflict_state,
    )
    batches: list[Mapping[str, Any]] = []
    if target > 0:
        remaining = target
        first = min(10, target)
        for batch_number in range(1, 5):
            if remaining <= 0:
                break
            change = first if batch_number == 1 else min(10, remaining)
            change = max(5, int(round(change / 5.0) * 5))
            batches.append(
                {
                    "action": "BUY",
                    "target_position_pp": target,
                    "batch_change_pp": change,
                    "cooldown_trading_days": 5,
                }
            )
            remaining -= change
    return V35StrategyDecision(
        market=market,
        forecast_anchor_date=anchor.isoformat(),
        dif_trend_state=local_snapshot.dif_trend_state,
        confirmation_status=local_snapshot.confirmation_status,
        market_state=str(ai_forecast.multi_timeframe_state) if ai_forecast else "NEUTRAL_MIXED",
        strategy_score=float(local_snapshot.strategy_score),
        base_target_position_pp=target,
        state_position_cap_pp=80,
        final_target_position_pp=target,
        batches=tuple(batches),
        components={
            "path": "QUANT_AI_FUSION",
            "config_version": config.config_version,
            "local_weight": float(config.local_weight),
            "ai_weight": float(config.ai_weight),
            "conflict_state": conflict_state,
        },
        reasons=("QUANT_AI_FUSION",),
    )


def ensure_fusion_package(
    session: Session,
    market: str,
    config: V37FusionConfig,
    champion_package: V35ModelPackage,
) -> V35ModelPackage:
    version = f"{PROTOCOL_VERSION_37}:{market}:FUSION:{config.config_version}"
    package = session.get(V35ModelPackage, version)
    if package is not None:
        return package
    strategy_config = {
        **dict(champion_package.strategy_config_json),
        "v351_model_config": {
            "path": "QUANT_AI_FUSION",
            "config_version": config.config_version,
        },
        "policy_version": f"POLICY_V37_FUSION_{config.config_version}",
    }
    package = V35ModelPackage(
        id=version,
        protocol_version=PROTOCOL_VERSION_37,
        model_market=market,
        version=version,
        package_kind="QUANT_AI_FUSION_ACCOUNT",
        parent_package_id=champion_package.id,
        prediction_model_id=champion_package.prediction_model_id,
        policy_version=f"POLICY_V37_FUSION_{config.config_version}",
        feature_version="V3.7_FEATURE_MANIFEST_1",
        promotion_rule_version="PROMOTION_RULE_V37_1",
        strategy_config_json=strategy_config,
        effective_from_date=champion_package.effective_from_date,
        package_hash=_hash(strategy_config),
        created_at=utc_now(),
    )
    session.add(package)
    session.flush()
    return package


def run_fusion_window_accounts(
    session: Session,
    market: str,
    as_of: date,
    champion_package: V35ModelPackage,
) -> int:
    """Run 8-week fusion accounts for every fusion config (shared engine)."""

    from backend.app.services.v37_feature_service import V37FeatureService
    from backend.app.services.v37_ai_service import record_model_conflict

    anchors = V37FeatureService().weekly_anchors(session, market)
    try:
        end_index = anchors.index(as_of)
    except ValueError:
        return 0
    window_index = end_index - 8 + 1
    if window_index < 0:
        return 0
    window = anchors[window_index : window_index + 8]
    if len(window) < 8 or window[-1] > as_of:
        return 0
    configs = ensure_fusion_configs(session, market)
    bars = load_weekly_bars(session, market, window)
    ran = 0
    for config in configs:
        fusion_package = ensure_fusion_package(
            session, market, config, champion_package
        )
        existing = session.scalar(
            select(V35SimAccount).where(
                V35SimAccount.model_market == market,
                V35SimAccount.scope == "STANDARD_8W",
                V35SimAccount.window_start_date == window[0],
                V35SimAccount.model_package_id == fusion_package.id,
            )
        )
        if existing is not None and existing.status == "COMPLETED":
            continue
        decisions: dict[date, V35StrategyDecision] = {}
        conflict_states: dict[date, str] = {}
        for anchor in window:
            local_snapshot = session.scalar(
                select(V35StrategySnapshot).where(
                    V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_37,
                    V35StrategySnapshot.model_market == market,
                    V35StrategySnapshot.forecast_anchor_date == anchor,
                    V35StrategySnapshot.model_package_id
                    == champion_package.id,
                )
            )
            if local_snapshot is None:
                break
            ai_forecast = session.scalar(
                select(V37AiForecast).where(
                    V37AiForecast.protocol_version == PROTOCOL_VERSION_37,
                    V37AiForecast.model_market == market,
                    V37AiForecast.forecast_anchor_date == anchor,
                    V37AiForecast.screening_scope == AI_SCOPE_FORWARD_OOS,
                )
            )
            if ai_forecast is None:
                break
            _status, _count, calibrator = ai_calibration_status(
                session, market, anchor
            )
            calibrated = None
            if calibrator is not None:
                up_score = float(
                    ai_forecast.direction_scores_json.get(
                        "direction_score_8w", 0.5
                    )
                )
                calibrated = apply_method(
                    calibrator.calibrator_params_json,
                    [up_score, 0.5, 0.5],
                )[0]
            conflict = record_model_conflict(
                session, market, anchor, None, ai_forecast
            )
            conflict_state = conflict.conflict_state if conflict else "NONE"
            conflict_states[anchor] = conflict_state
            decisions[anchor] = fusion_decision(
                market,
                anchor,
                local_snapshot=local_snapshot,
                ai_forecast=ai_forecast,
                config=config,
                champion_strategy_config=dict(champion_package.strategy_config_json),
                conflict_state=conflict_state,
            )
        if len(decisions) != len(window):
            continue
        initial_snapshot = None
        if window_index > 0:
            initial_snapshot = session.scalar(
                select(V36AccountSnapshot)
                .where(
                    V36AccountSnapshot.protocol_version == PROTOCOL_VERSION_37,
                    V36AccountSnapshot.model_market == market,
                    V36AccountSnapshot.anchor_date == anchors[window_index - 1],
                )
            )
            if initial_snapshot is not None:
                initial_snapshot = {
                    "equity": float(initial_snapshot.equity),
                    "cash": float(initial_snapshot.cash),
                    "position_pp": int(initial_snapshot.position_pp),
                    "held_shares": float(initial_snapshot.held_shares),
                    "average_cost": (
                        float(initial_snapshot.average_cost)
                        if initial_snapshot.average_cost is not None
                        else 0.0
                    ),
                    "sellable_shares": float(initial_snapshot.sellable_shares),
                    "pending_sellable": [],
                    "pending_batches": list(initial_snapshot.pending_batches_json),
                }
        result = run_standard_window(
            session,
            market=market,
            package_id=fusion_package.id,
            anchors=window,
            decisions=decisions,
            bars=bars,
            config=dict(champion_package.strategy_config_json),
            accounting_mode="SHARES",
            initial_snapshot=initial_snapshot,
        )
        result_json = dict(result.result_json)
        result_json["path"] = "QUANT_AI_FUSION"
        result_json["config_version"] = config.config_version
        result = result.__class__(
            account_id=result.account_id,
            market=result.market,
            scope=result.scope,
            window_start=result.window_start,
            window_end=result.window_end,
            package_id=result.package_id,
            initial_capital=result.initial_capital,
            ending_equity=result.ending_equity,
            current_cash=result.current_cash,
            current_position_pp=result.current_position_pp,
            net_profit=result.net_profit,
            net_return=result.net_return,
            max_drawdown=result.max_drawdown,
            average_position_pp=result.average_position_pp,
            trade_count=result.trade_count,
            turnover=result.turnover,
            transaction_cost=result.transaction_cost,
            no_action_window=result.no_action_window,
            up_market_participation=result.up_market_participation,
            down_market_defense=result.down_market_defense,
            status=result.status,
            ledger=result.ledger,
            result_json=result_json,
            account_hash=result.account_hash,
        )
        persist_window_result(
            session,
            result,
            protocol_version=PROTOCOL_VERSION_37,
        )
        # Matured evaluation vs Local and AI accounts.
        local_account = session.scalar(
            select(V35SimAccount).where(
                V35SimAccount.model_market == market,
                V35SimAccount.scope == "STANDARD_8W",
                V35SimAccount.window_start_date == window[0],
                V35SimAccount.model_package_id == champion_package.id,
            )
        )
        ai_package_id = f"{PROTOCOL_VERSION_37}:{market}:DEEPSEEK_AI"
        ai_account = session.scalar(
            select(V35SimAccount).where(
                V35SimAccount.model_market == market,
                V35SimAccount.scope == "STANDARD_8W",
                V35SimAccount.window_start_date == window[0],
                V35SimAccount.model_package_id == ai_package_id,
            )
        )
        if local_account is not None and result.status == "COMPLETED":
            local_equity = float(local_account.ending_equity)
            ai_equity = (
                float(ai_account.ending_equity) if ai_account is not None else local_equity
            )
            fusion_equity = float(result.ending_equity)
            metrics = {
                "local_equity": local_equity,
                "ai_equity": ai_equity,
                "fusion_equity": fusion_equity,
                "window_start": window[0].isoformat(),
                "window_end": window[-1].isoformat(),
            }
            evaluation = V37FusionEvaluation(
                protocol_version=PROTOCOL_VERSION_37,
                model_market=market,
                fusion_config_id=config.id,
                window_start_date=window[0],
                window_end_date=window[-1],
                local_equity=Decimal(str(round(local_equity, 2))),
                ai_equity=Decimal(str(round(ai_equity, 2))),
                fusion_equity=Decimal(str(round(fusion_equity, 2))),
                local_net_return=Decimal(str(round(float(local_account.net_return), 8))),
                ai_net_return=Decimal(
                    str(round(float(ai_account.net_return), 8))
                    if ai_account is not None
                    else 0.0
                ),
                fusion_net_return=Decimal(str(round(float(result.net_return), 8))),
                win_vs_local=fusion_equity > local_equity,
                win_vs_ai=ai_account is not None and fusion_equity > ai_equity,
                max_drawdown=Decimal(str(round(float(result.max_drawdown), 8))),
                average_position_pp=Decimal(
                    str(round(float(result.average_position_pp), 4))
                ),
                metrics_json=metrics,
                evaluation_hash=_hash(metrics),
                created_at=utc_now(),
            )
            session.add(evaluation)
        ran += 1
    session.flush()
    return ran
