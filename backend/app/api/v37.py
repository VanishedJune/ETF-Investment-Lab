"""V3.7 public API routes (multi-timeframe local + DeepSeek AI + fusion)."""

from __future__ import annotations

from sqlalchemy import func, select
from fastapi import APIRouter, HTTPException, Request

from backend.app.services.v37_config import PROTOCOL_VERSION_37


router = APIRouter(prefix="/api/v37", tags=["v37"])


def _service(request: Request, name: str):
    return getattr(request.app.state, name, None)


def _require(request: Request, name: str):
    service = _service(request, name)
    if service is None:
        raise HTTPException(status_code=503, detail=f"{name} unavailable")
    return service


def _model_markets(request: Request) -> list[str]:
    from backend.app.models.models import V351InstrumentSlot

    factory = _require(request, "sessions")
    with factory() as session:
        slots = session.scalars(
            select(V351InstrumentSlot).where(V351InstrumentSlot.active.is_(True))
        ).all()
        codes = {slot.instrument_code for slot in slots}
    return sorted({"399006", "159941"} | codes)


def _require_model_market(request: Request, market: str) -> str:
    if market not in _model_markets(request):
        raise HTTPException(status_code=422, detail=f"unsupported v37 market {market}")
    return market


@router.get("/status")
def v37_status_all(request: Request) -> list[dict[str, object]]:
    runtime = _require(request, "v37_runtime")
    return [runtime.status(market) for market in _model_markets(request)]


@router.get("/status/{market}")
def v37_status_market(request: Request, market: str) -> dict[str, object]:
    return _require(request, "v37_runtime").status(
        _require_model_market(request, market)
    )


@router.get("/forecast/{market}")
def v37_forecast(request: Request, market: str) -> dict[str, object]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import (
        V35Forecast,
        V35PositionDecision,
        V35StrategySnapshot,
    )
    from backend.app.services.v351_payload_service import build_forecast_payload

    with factory() as session:
        forecast = session.scalar(
            select(V35Forecast)
            .where(
                V35Forecast.model_market == market,
                V35Forecast.protocol_version == PROTOCOL_VERSION_37,
            )
            .order_by(V35Forecast.forecast_anchor_date.desc())
        )
        if forecast is None:
            raise HTTPException(status_code=404, detail="no v37 forecast yet")
        snapshot = session.scalar(
            select(V35StrategySnapshot)
            .where(
                V35StrategySnapshot.model_market == market,
                V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_37,
                V35StrategySnapshot.forecast_anchor_date
                == forecast.forecast_anchor_date,
            )
        )
        batches = (
            list(
                session.scalars(
                    select(V35PositionDecision)
                    .where(
                        V35PositionDecision.strategy_snapshot_id == snapshot.id
                    )
                    .order_by(V35PositionDecision.batch_number)
                )
            )
            if snapshot is not None
            else []
        )
        return build_forecast_payload(
            session,
            market=market,
            forecast=forecast,
            strategy_snapshot=snapshot,
            batches=batches,
        )


@router.get("/strategy/{market}")
def v37_strategy(request: Request, market: str) -> dict[str, object]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import (
        V35PositionDecision,
        V35StrategySnapshot,
    )
    from backend.app.services.v351_payload_service import build_strategy_payload

    with factory() as session:
        snapshot = session.scalar(
            select(V35StrategySnapshot)
            .where(
                V35StrategySnapshot.model_market == market,
                V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_37,
            )
            .order_by(V35StrategySnapshot.forecast_anchor_date.desc())
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="no v37 strategy yet")
        batches = session.scalars(
            select(V35PositionDecision)
            .where(V35PositionDecision.strategy_snapshot_id == snapshot.id)
            .order_by(V35PositionDecision.batch_number)
        ).all()
        return build_strategy_payload(market, snapshot, batches)


@router.get("/ai/{market}")
def v37_ai(request: Request, market: str) -> dict[str, object]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import V37AiForecast

    with factory() as session:
        row = session.scalar(
            select(V37AiForecast)
            .where(
                V37AiForecast.protocol_version == PROTOCOL_VERSION_37,
                V37AiForecast.model_market == market,
                V37AiForecast.screening_scope == "FORWARD_OOS",
            )
            .order_by(V37AiForecast.forecast_anchor_date.desc())
        )
        if row is None:
            return {
                "market": market,
                "protocol_version": PROTOCOL_VERSION_37,
                "ai_forecast": None,
                "health": None,
            }
        from backend.app.services.v37_ai_service import ai_weight_status

        health = ai_weight_status(
            session, market, row.forecast_anchor_date
        )
        return {
            "market": market,
            "protocol_version": PROTOCOL_VERSION_37,
            "ai_forecast": {
                "anchor": row.forecast_anchor_date.isoformat(),
                "model": row.model_name,
                "prompt_version": row.prompt_version,
                "generation": row.ai_generation_version,
                "trend_1w": row.trend_1w,
                "trend_2w": row.trend_2w,
                "trend_4w": row.trend_4w,
                "trend_8w": row.trend_8w,
                "direction_scores": row.direction_scores_json,
                "daily_trend": row.daily_trend,
                "weekly_trend": row.weekly_trend,
                "multi_timeframe_state": row.multi_timeframe_state,
                "risk_level": row.risk_level,
                "confidence_raw": float(row.confidence_raw),
                "expected_return_4w": float(row.expected_return_4w),
                "expected_return_8w": float(row.expected_return_8w),
                "support_distance_pct": float(row.support_distance_pct),
                "resistance_distance_pct": float(row.resistance_distance_pct),
                "reason_codes": row.reason_codes_json,
                "status": row.status,
            },
            "health": {
                "status": health["status"],
                "calibration_status": health["calibration_status"],
                "forward_oos_window_count": health["forward_oos_window_count"],
                "weight_cap_pp": health["weight_cap_pp"],
                "metrics": health["metrics"],
            },
        }


@router.get("/funnel/{market}")
def v37_funnel(request: Request, market: str) -> dict[str, object]:
    """Decision funnel for LOCAL / DEEPSEEK_AI / QUANT_AI_FUSION (recent 52w)."""

    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import (
        V35BootstrapState,
        V35ModelPackage,
        V35StrategySnapshot,
        V36DecisionFunnel,
        V37AiForecast,
    )
    from backend.app.services.v37_ai_service import (
        ai_calibration_status,
        ai_target_position_pp,
    )
    from backend.app.services.v37_config import AI_SCOPE_FORWARD_OOS
    from backend.app.services.v37_fusion_service import (
        ensure_fusion_configs,
        fusion_position,
    )

    with factory() as session:
        local_rows = session.scalars(
            select(V36DecisionFunnel)
            .where(
                V36DecisionFunnel.protocol_version == PROTOCOL_VERSION_37,
                V36DecisionFunnel.model_market == market,
            )
            .order_by(V36DecisionFunnel.forecast_anchor_date.desc())
            .limit(52)
        ).all()
        local = [
            {
                "anchor": row.forecast_anchor_date.isoformat(),
                "bull_signal": row.bull_signal_count,
                "score_pass": row.score_pass_count,
                "base_position_positive": row.base_position_positive_count,
                "risk_blocked": row.risk_blocked_count,
                "cooldown_blocked": row.cooldown_blocked_count,
                "final_position_positive": row.final_position_positive_count,
                "trade_count": row.trade_count,
                "actual_position_positive": row.actual_position_positive_count,
            }
            for row in local_rows
        ]
        ai_rows = session.scalars(
            select(V37AiForecast)
            .where(
                V37AiForecast.protocol_version == PROTOCOL_VERSION_37,
                V37AiForecast.model_market == market,
                V37AiForecast.screening_scope == AI_SCOPE_FORWARD_OOS,
            )
            .order_by(V37AiForecast.forecast_anchor_date.desc())
            .limit(52)
        ).all()
        ai_funnel: list[dict[str, object]] = []
        for row in reversed(ai_rows):
            _status, _count, calibrator = ai_calibration_status(
                session, market, row.forecast_anchor_date
            )
            calibrated = None
            if calibrator is not None:
                from backend.app.services.v36_calibration import apply_method

                up_score = float(
                    row.direction_scores_json.get("direction_score_8w", 0.5)
                )
                calibrated = apply_method(
                    calibrator.calibrator_params_json, [up_score, 0.5, 0.5]
                )[0]
            target, batches = ai_target_position_pp(row, calibrated)
            bull_signal = int(
                row.trend_8w in ("BULLISH", "NEUTRAL_BULLISH")
                or float(row.direction_scores_json.get("direction_score_8w", 0.5)) >= 0.5
            )
            ai_funnel.append(
                {
                    "anchor": row.forecast_anchor_date.isoformat(),
                    "bull_signal": bull_signal,
                    "score_pass": int(float(row.confidence_raw) > 0.5),
                    "base_position_positive": int(target > 0),
                    "risk_blocked": int(bull_signal == 1 and target == 0),
                    "cooldown_blocked": 0,
                    "final_position_positive": int(target > 0),
                    "trade_count": len(batches),
                    "actual_position_positive": int(target > 0),
                }
            )
        champion = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.protocol_version == PROTOCOL_VERSION_37,
                V35BootstrapState.model_market == market,
            )
        )
        champion_config: dict = {}
        if champion is not None and champion.champion_package_id is not None:
            package = session.get(V35ModelPackage, champion.champion_package_id)
            if package is not None:
                champion_config = dict(package.strategy_config_json)
        configs = ensure_fusion_configs(session, market)
        fusion_config = configs[0] if configs else None
        fusion_funnel: list[dict[str, object]] = []
        if fusion_config is not None and champion_config:
            for row in reversed(ai_rows):
                local_snapshot = session.scalar(
                    select(V35StrategySnapshot).where(
                        V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_37,
                        V35StrategySnapshot.model_market == market,
                        V35StrategySnapshot.forecast_anchor_date
                        == row.forecast_anchor_date,
                    )
                )
                if local_snapshot is None:
                    continue
                _status, _count, calibrator = ai_calibration_status(
                    session, market, row.forecast_anchor_date
                )
                calibrated = None
                if calibrator is not None:
                    from backend.app.services.v36_calibration import apply_method

                    up_score = float(
                        row.direction_scores_json.get("direction_score_8w", 0.5)
                    )
                    calibrated = apply_method(
                        calibrator.calibrator_params_json, [up_score, 0.5, 0.5]
                    )[0]
                target = fusion_position(
                    local_position_pp=int(local_snapshot.final_target_position_pp),
                    local_score=float(local_snapshot.strategy_score),
                    ai_up_probability=calibrated,
                    config=fusion_config,
                    champion_strategy_config=champion_config,
                )
                bull_signal = int(
                    float(local_snapshot.strategy_score) > 0.0
                    or (
                        calibrated is not None
                        and calibrated >= 0.5
                    )
                )
                fusion_funnel.append(
                    {
                        "anchor": row.forecast_anchor_date.isoformat(),
                        "bull_signal": bull_signal,
                        "score_pass": int(float(local_snapshot.strategy_score) > 0.0),
                        "base_position_positive": int(target > 0),
                        "risk_blocked": int(bull_signal == 1 and target == 0),
                        "cooldown_blocked": 0,
                        "final_position_positive": int(target > 0),
                        "trade_count": 1 if target > 0 else 0,
                        "actual_position_positive": int(target > 0),
                    }
                )
        return {
            "market": market,
            "protocol_version": PROTOCOL_VERSION_37,
            "local": local,
            "ai": ai_funnel,
            "fusion": fusion_funnel,
            "fusion_config": (
                {
                    "config_version": fusion_config.config_version,
                    "local_weight": float(fusion_config.local_weight),
                    "ai_weight": float(fusion_config.ai_weight),
                }
                if fusion_config is not None
                else None
            ),
        }


@router.get("/fusion/{market}")
def v37_fusion(request: Request, market: str) -> dict[str, object]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import V37FusionConfig, V37FusionEvaluation

    with factory() as session:
        configs = session.scalars(
            select(V37FusionConfig)
            .where(
                V37FusionConfig.protocol_version == PROTOCOL_VERSION_37,
                V37FusionConfig.model_market == market,
            )
            .order_by(V37FusionConfig.config_version)
        ).all()
        evaluation_count = int(
            session.scalar(
                select(func.count())
                .select_from(V37FusionEvaluation)
                .where(
                    V37FusionEvaluation.protocol_version == PROTOCOL_VERSION_37,
                    V37FusionEvaluation.model_market == market,
                )
            )
            or 0
        )
        return {
            "market": market,
            "protocol_version": PROTOCOL_VERSION_37,
            "configs": [
                {
                    "config_version": config.config_version,
                    "local_weight": float(config.local_weight),
                    "ai_weight": float(config.ai_weight),
                    "conflict_policy": config.conflict_policy,
                }
                for config in configs
            ],
            "evaluation_count": evaluation_count,
        }


@router.get("/simulation/{market}")
def v37_simulation(request: Request, market: str) -> dict[str, object]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import (
        V35ContinuousAccount,
        V35SimAccount,
        V35SimEvaluation,
    )

    with factory() as session:
        accounts = session.scalars(
            select(V35SimAccount)
            .where(
                V35SimAccount.model_market == market,
                V35SimAccount.protocol_version == PROTOCOL_VERSION_37,
            )
            .order_by(V35SimAccount.window_start_date.desc())
            .limit(40)
        ).all()
        evaluation_count = int(
            session.scalar(
                select(func.count())
                .select_from(V35SimEvaluation)
                .where(V35SimEvaluation.model_market == market)
            )
            or 0
        )
        continuous = session.scalar(
            select(V35ContinuousAccount).where(
                V35ContinuousAccount.model_market == market,
                V35ContinuousAccount.protocol_version == PROTOCOL_VERSION_37,
            )
        )
        return {
            "market": market,
            "evaluation_count": evaluation_count,
            "accounts": [
                {
                    "scope": account.scope,
                    "window_start": (
                        account.window_start_date.isoformat()
                        if account.window_start_date
                        else None
                    ),
                    "ending_equity": float(account.ending_equity),
                    "net_return": float(account.net_return),
                    "max_drawdown": float(account.max_drawdown),
                    "average_position_pp": float(account.average_position_pp),
                    "trade_count": account.trade_count,
                    "no_action_window": account.no_action_window,
                    "status": account.status,
                    "path": (
                        account.result_json.get("path", "LOCAL_QUANT")
                        if isinstance(account.result_json, dict)
                        else "LOCAL_QUANT"
                    ),
                }
                for account in accounts
            ],
            "continuous": (
                {
                    "ending_equity": float(continuous.ending_equity),
                    "cumulative_return": float(continuous.cumulative_return),
                    "annualized_return": float(continuous.annualized_return),
                    "max_drawdown": float(continuous.max_drawdown),
                    "average_position_pp": float(continuous.average_position_pp),
                    "turnover": float(continuous.turnover),
                }
                if continuous is not None
                else None
            ),
        }


@router.get("/champions/{market}")
def v37_champions(request: Request, market: str) -> dict[str, object]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import V35BootstrapState, V35ModelPackage

    with factory() as session:
        state = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == PROTOCOL_VERSION_37,
            )
        )
        if state is None or state.champion_package_id is None:
            return {"market": market, "champion": None, "promotion_count": 0}
        package = session.get(V35ModelPackage, state.champion_package_id)
        return {
            "market": market,
            "champion": {
                "id": package.id,
                "kind": package.package_kind,
                "effective_from_date": package.effective_from_date.isoformat(),
                "prediction_model_id": package.prediction_model_id,
                "policy_version": package.policy_version,
            }
            if package is not None
            else None,
            "promotion_count": state.promotion_count,
            "prediction_challenge_count": state.prediction_challenge_count,
            "strategy_challenge_count": state.strategy_challenge_count,
        }


@router.get("/repair/proposals")
def v37_repair_proposals(request: Request) -> list[dict[str, object]]:
    factory = _require(request, "sessions")
    from backend.app.models.models import V37ModelRepairProposal

    with factory() as session:
        rows = session.scalars(
            select(V37ModelRepairProposal)
            .where(
                V37ModelRepairProposal.protocol_version == PROTOCOL_VERSION_37
            )
            .order_by(V37ModelRepairProposal.anchor_date.desc())
            .limit(100)
        ).all()
        return [
            {
                "id": row.id,
                "model_market": row.model_market,
                "anchor_date": row.anchor_date.isoformat(),
                "trigger": row.trigger,
                "diagnosis": row.diagnosis,
                "target_layer": row.target_layer,
                "severity": row.severity,
                "proposed_change": row.proposed_change,
                "parameter_changes": row.parameter_changes_json,
                "status": row.status,
            }
            for row in rows
        ]


@router.get("/model-conflicts/{market}")
def v37_model_conflicts(request: Request, market: str) -> list[dict[str, object]]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import V37ModelConflict

    with factory() as session:
        rows = session.scalars(
            select(V37ModelConflict)
            .where(
                V37ModelConflict.protocol_version == PROTOCOL_VERSION_37,
                V37ModelConflict.model_market == market,
            )
            .order_by(V37ModelConflict.forecast_anchor_date.desc())
            .limit(100)
        ).all()
        return [
            {
                "anchor": row.forecast_anchor_date.isoformat(),
                "consensus_state": row.consensus_state,
                "conflict_state": row.conflict_state,
                "local_up_probability_8w": float(row.local_up_probability_8w),
                "ai_up_probability_8w": float(row.ai_up_probability_8w),
                "conflict_score": float(row.conflict_score),
                "resolution": row.resolution,
            }
            for row in rows
        ]


@router.get("/challenges")
def v37_challenges(request: Request) -> list[dict[str, object]]:
    factory = _require(request, "sessions")
    from backend.app.models.models import V35Challenge

    with factory() as session:
        rows = session.scalars(
            select(V35Challenge)
            .where(V35Challenge.protocol_version == PROTOCOL_VERSION_37)
            .order_by(V35Challenge.anchor_date.desc())
            .limit(100)
        ).all()
        return [
            {
                "id": row.id,
                "model_market": row.model_market,
                "anchor_date": row.anchor_date.isoformat(),
                "challenger_family": row.challenger_family,
                "candidate_count": row.candidate_count,
                "status": row.status,
            }
            for row in rows
        ]
