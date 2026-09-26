"""V3.5.1 public API routes (ETF slots + challenge effectiveness)."""

from __future__ import annotations

from sqlalchemy import func, select
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from backend.app.services.v351_config import PROTOCOL_VERSION_351


router = APIRouter(prefix="/api/v351", tags=["v351"])


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
        raise HTTPException(status_code=422, detail=f"unsupported v351 market {market}")
    return market


@router.get("/instrument-slots")
def list_slots(request: Request) -> list[dict[str, object]]:
    return _require(request, "v351_slots").list_slots()


@router.get("/instrument-slots/{slot_id}")
def get_slot(request: Request, slot_id: str) -> dict[str, object]:
    slots = _require(request, "v351_slots").list_slots()
    for slot in slots:
        if slot["slot_id"] == slot_id:
            return slot
    raise HTTPException(status_code=404, detail=f"slot {slot_id} not found")


@router.post("/instrument-slots/{slot_id}/validate-replacement")
def validate_replacement(request: Request, slot_id: str, body: dict[str, str]) -> dict[str, object]:
    code = (body.get("instrument_code") or "").strip()
    try:
        return _require(request, "v351_slots").validate_replacement(
            slot_id, code, idempotency_key=body.get("idempotency_key")
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/instrument-slots/{slot_id}/replace", status_code=202)
def replace_slot(
    request: Request,
    slot_id: str,
    body: dict[str, str],
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    code = (body.get("instrument_code") or "").strip()
    protocol_version = (body.get("protocol_version") or "").strip()
    if not protocol_version:
        raise HTTPException(
            status_code=422,
            detail="protocol_version is required for write endpoints",
        )
    if protocol_version != "V3.5.1_EFFECTIVE_CHALLENGER_AND_DYNAMIC_ETF_SLOTS":
        raise HTTPException(
            status_code=422,
            detail="V3.6 slot replacement is not enabled yet; use V3.5.1 channel",
        )
    try:
        service = _require(request, "v351_slots")
        started = service.start_replace(
            slot_id,
            code,
            idempotency_key=body.get("idempotency_key"),
            maximum_weeks=None,
        )
        if started.get("reused"):
            return started
        background_tasks.add_task(
            service._run_replacement,
            slot_id,
            code,
            str(started["job_id"]),
            str(started["idempotency_key"]),
            None,
        )
        return {
            "job_id": started["job_id"],
            "state": "VALIDATING",
            "target_code": code,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - converted to a public error
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/instrument-slots/{slot_id}/replacement-status")
def replacement_status(request: Request, slot_id: str) -> dict[str, object] | None:
    result = _require(request, "v351_slots").replacement_status(slot_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no replacement job")
    return result


@router.post("/instrument-slots/{slot_id}/cancel-replacement")
def cancel_replacement(request: Request, slot_id: str) -> dict[str, object]:
    try:
        return _require(request, "v351_slots").cancel_replacement(slot_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/instrument-slots/{slot_id}/history")
def slot_history(request: Request, slot_id: str) -> list[dict[str, object]]:
    return _require(request, "v351_slots").slot_history(slot_id)


@router.get("/challenges/equivalence-summary")
def equivalence_summary(request: Request) -> dict[str, object]:
    from sqlalchemy import select

    factory = _require(request, "sessions")
    from backend.app.models.models import V351CandidateEvaluation

    with factory() as session:
        rows = session.scalars(select(V351CandidateEvaluation)).all()
        total = len(rows)
        identical = sum(1 for row in rows if row.effectively_identical)
        valid = total - identical
        by_market: dict[str, dict[str, int]] = {}
        for row in rows:
            market_stats = by_market.setdefault(
                row.model_market, {"total": 0, "identical": 0, "valid": 0}
            )
            market_stats["total"] += 1
            if row.effectively_identical:
                market_stats["identical"] += 1
            else:
                market_stats["valid"] += 1
        return {
            "total_candidates": total,
            "effectively_identical": identical,
            "valid_candidates": valid,
            "by_market": by_market,
        }


@router.get("/challenges/rejection-summary")
def rejection_summary(request: Request) -> dict[str, object]:
    from sqlalchemy import select

    factory = _require(request, "sessions")
    from backend.app.models.models import V351CandidateEvaluation

    with factory() as session:
        rows = session.scalars(select(V351CandidateEvaluation)).all()
        counts: dict[str, int] = {}
        by_market: dict[str, dict[str, int]] = {}
        for row in rows:
            for code in row.rejection_reason_codes_json:
                counts[str(code)] = counts.get(str(code), 0) + 1
            market_counts = by_market.setdefault(row.model_market, {})
            for code in row.rejection_reason_codes_json:
                market_counts[str(code)] = market_counts.get(str(code), 0) + 1
        return {"reasons": counts, "by_market": by_market}


@router.get("/challenges/{challenge_id}/behavior-diff")
def behavior_diff(request: Request, challenge_id: str) -> dict[str, object]:
    from sqlalchemy import select

    factory = _require(request, "sessions")
    from backend.app.models.models import V351CandidateEvaluation

    with factory() as session:
        rows = session.scalars(
            select(V351CandidateEvaluation).where(
                V351CandidateEvaluation.challenge_id == challenge_id
            )
        ).all()
        return {
            "challenge_id": challenge_id,
            "candidates": [
                {
                    "candidate_package_id": row.candidate_package_id,
                    "effectively_identical": row.effectively_identical,
                    "forecast_divergence_ratio": float(row.forecast_divergence_ratio),
                    "target_position_divergence_ratio": float(
                        row.target_position_divergence_ratio
                    ),
                    "trade_path_divergence_ratio": float(row.trade_path_divergence_ratio),
                    "rejection_reason_codes": row.rejection_reason_codes_json,
                    "gaps": row.gaps_json,
                    "promotion_channel": row.promotion_channel,
                }
                for row in rows
            ],
        }


@router.get("/promotions/maintenance-refresh")
def maintenance_refresh(request: Request) -> list[dict[str, object]]:
    from sqlalchemy import select

    factory = _require(request, "sessions")
    from backend.app.models.models import V351MaintenanceRefresh

    with factory() as session:
        rows = session.scalars(
            select(V351MaintenanceRefresh).order_by(
                V351MaintenanceRefresh.anchor_date.desc()
            )
        ).all()
        return [
            {
                "model_market": row.model_market,
                "anchor_date": row.anchor_date.isoformat(),
                "decision": row.decision,
                "reason": row.reason,
                "from_model_id": row.from_model_id,
                "to_model_id": row.to_model_id,
                "effective_from_date": (
                    row.effective_from_date.isoformat()
                    if row.effective_from_date
                    else None
                ),
            }
            for row in rows
        ]


@router.get("/status")
def v351_status_all(request: Request) -> list[dict[str, object]]:
    runtime = _require(request, "v351_runtime")
    return [runtime.status(market) for market in _model_markets(request)]


@router.get("/status/{market}")
def v351_status_market(request: Request, market: str) -> dict[str, object]:
    return _require(request, "v351_runtime").status(
        _require_model_market(request, market)
    )


@router.get("/forecast/{market}")
def v351_forecast(request: Request, market: str) -> dict[str, object]:
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
                V35Forecast.protocol_version == PROTOCOL_VERSION_351,
            )
            .order_by(V35Forecast.forecast_anchor_date.desc())
        )
        if forecast is None:
            raise HTTPException(status_code=404, detail="no v351 forecast yet")
        snapshot = session.scalar(
            select(V35StrategySnapshot)
            .where(
                V35StrategySnapshot.model_market == market,
                V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_351,
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
def v351_strategy(request: Request, market: str) -> dict[str, object]:
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
                V35StrategySnapshot.protocol_version == PROTOCOL_VERSION_351,
            )
            .order_by(V35StrategySnapshot.forecast_anchor_date.desc())
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="no v351 strategy yet")
        batches = session.scalars(
            select(V35PositionDecision)
            .where(V35PositionDecision.strategy_snapshot_id == snapshot.id)
            .order_by(V35PositionDecision.batch_number)
        ).all()
        return build_strategy_payload(market, snapshot, batches)


@router.get("/simulation/{market}")
def v351_simulation(request: Request, market: str) -> dict[str, object]:
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
                V35SimAccount.protocol_version == PROTOCOL_VERSION_351,
            )
            .order_by(V35SimAccount.window_start_date.desc())
            .limit(20)
        ).all()
        evaluation_count = session.scalar(
            select(func.count())
            .select_from(V35SimEvaluation)
            .where(V35SimEvaluation.model_market == market)
        )
        continuous = session.scalar(
            select(V35ContinuousAccount).where(
                V35ContinuousAccount.model_market == market,
                V35ContinuousAccount.protocol_version == PROTOCOL_VERSION_351,
            )
        )
        return {
            "market": market,
            "evaluation_count": int(evaluation_count or 0),
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
                    "buy_hold_return": float(continuous.buy_hold_return),
                    "fixed_30_return": float(continuous.fixed_30_return),
                    "cash_return": float(continuous.cash_return),
                }
                if continuous is not None
                else None
            ),
        }


@router.get("/champions/{market}")
def v351_champions(request: Request, market: str) -> dict[str, object]:
    market = _require_model_market(request, market)
    factory = _require(request, "sessions")
    from backend.app.models.models import V35BootstrapState, V35ModelPackage

    with factory() as session:
        state = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market,
                V35BootstrapState.protocol_version == PROTOCOL_VERSION_351,
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
