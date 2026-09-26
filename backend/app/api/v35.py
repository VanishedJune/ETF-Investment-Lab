"""V3.5 public API routes (read-only status plus controlled bootstrap start)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(prefix="/api/v35", tags=["v35"])


def _runtime(request: Request):
    return getattr(request.app.state, "v35_runtime", None)


def _market(value: str) -> str:
    if value not in ("399006", "159941"):
        raise HTTPException(status_code=422, detail="unsupported V3.5 market")
    return value


@router.get("/status")
def status_all(request: Request) -> list[dict[str, object]]:
    runtime = _runtime(request)
    if runtime is None:
        raise HTTPException(status_code=503, detail="V3.5 runtime unavailable")
    return [runtime.status(market) for market in ("399006", "159941")]


@router.get("/status/{market}")
def status_market(request: Request, market: str) -> dict[str, object]:
    return _runtime(request).status(_market(market))


@router.post("/bootstrap/{market}")
def start_bootstrap(
    request: Request,
    market: str,
    protocol_version: str | None = None,
) -> dict[str, object]:
    if protocol_version is None:
        raise HTTPException(
            status_code=422,
            detail="protocol_version is required for write endpoints",
        )
    if protocol_version != "V3.5_CAPITAL_DRIVEN_8W":
        raise HTTPException(
            status_code=422,
            detail="this endpoint only accepts protocol_version=V3.5_CAPITAL_DRIVEN_8W",
        )
    runtime = _runtime(request)
    try:
        return runtime.bootstrap_sync(_market(market))
    except Exception as exc:  # noqa: BLE001 - converted to a public error
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/forecast/{market}")
def latest_forecast(request: Request, market: str) -> dict[str, object] | None:
    runtime = _runtime(request)
    result = runtime.latest_forecast(_market(market))
    if result is None:
        raise HTTPException(status_code=404, detail="no V3.5 forecast yet")
    return result


@router.get("/strategy/{market}")
def latest_strategy(request: Request, market: str) -> dict[str, object] | None:
    from sqlalchemy import select

    runtime = _runtime(request)
    market_code = _market(market)
    factory = getattr(request.app.state, "sessions", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="session factory unavailable")
    from backend.app.models.models import (
        V35PositionDecision,
        V35StrategySnapshot,
    )

    with factory() as session:
        snapshot = session.scalar(
            select(V35StrategySnapshot)
            .where(V35StrategySnapshot.model_market == market_code)
            .order_by(V35StrategySnapshot.forecast_anchor_date.desc())
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="no V3.5 strategy yet")
        batches = session.scalars(
            select(V35PositionDecision)
            .where(V35PositionDecision.strategy_snapshot_id == snapshot.id)
            .order_by(V35PositionDecision.batch_number)
        ).all()
        return {
            "forecast_anchor_date": snapshot.forecast_anchor_date.isoformat(),
            "dif_trend_state": snapshot.dif_trend_state,
            "confirmation_status": snapshot.confirmation_status,
            "strategy_score": float(snapshot.strategy_score),
            "base_target_position_pp": snapshot.base_target_position_pp,
            "state_position_cap_pp": snapshot.state_position_cap_pp,
            "final_target_position_pp": snapshot.final_target_position_pp,
            "batches": [
                {
                    "batch_number": row.batch_number,
                    "action": row.action,
                    "position_pp": row.position_pp,
                    "batch_change_pp": row.batch_change_pp,
                    "target_position_pp": row.target_position_pp,
                    "condition": row.condition_json,
                }
                for row in batches
            ],
            "reasons": snapshot.strategy_json.get("reasons", []),
            "strategy_hash": snapshot.strategy_hash,
        }


@router.get("/simulation/{market}")
def simulation_summary(request: Request, market: str) -> dict[str, object]:
    from sqlalchemy import func, select

    market_code = _market(market)
    factory = getattr(request.app.state, "sessions", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="session factory unavailable")
    from backend.app.models.models import (
        V35ContinuousAccount,
        V35SimAccount,
        V35SimEvaluation,
    )

    with factory() as session:
        accounts = session.scalars(
            select(V35SimAccount)
            .where(V35SimAccount.model_market == market_code)
            .order_by(V35SimAccount.window_start_date.desc())
            .limit(20)
        ).all()
        evaluation_count = session.scalar(
            select(func.count())
            .select_from(V35SimEvaluation)
            .where(V35SimEvaluation.model_market == market_code)
        )
        continuous = session.scalar(
            select(V35ContinuousAccount).where(
                V35ContinuousAccount.model_market == market_code
            )
        )
        return {
            "market": market_code,
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
def champions(request: Request, market: str) -> dict[str, object]:
    from sqlalchemy import select

    market_code = _market(market)
    factory = getattr(request.app.state, "sessions", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="session factory unavailable")
    from backend.app.models.models import V35BootstrapState, V35ModelPackage

    with factory() as session:
        state = session.scalar(
            select(V35BootstrapState).where(
                V35BootstrapState.model_market == market_code
            )
        )
        if state is None or state.champion_package_id is None:
            return {"market": market_code, "champion": None, "promotion_count": 0}
        package = session.get(V35ModelPackage, state.champion_package_id)
        return {
            "market": market_code,
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
