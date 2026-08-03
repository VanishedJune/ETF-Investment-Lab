"""FastAPI entrypoint for the project-local ETF Investment Research Lab."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal
import logging
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterable, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.paths import project_root, resource_root
from backend.app.agent_iterations.storage import (
    AgentIterationStorage,
    ArtifactConsistencyError,
)
from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.errors import (
    PublicApiError,
    PublicConflictError,
    PublicDataUnavailableError,
    PublicNotFoundError,
    PublicValidationError,
)
from backend.app.forecasting import HistoricalSimilarityForecaster
from backend.app.models.models import (
    AppSetting,
    DataUpdateLog,
    IndicatorRecord,
    Instrument,
    MarketPrice,
    RealAccount,
    SimulationAccount,
    StrategyDefinition,
    StrategySignal,
)
from backend.app.schemas.real_account import (
    RealAccountCreate,
    RealAccountRead,
    RealTransactionCreate,
    RealTransactionUpdate,
)
from backend.app.schemas.investment_calendar import PositionEventCreate, PositionEventUpdate
from backend.app.schemas.simulation import (
    InvestmentPlanCreate,
    InvestmentPlanUpdate,
    SimulationAccountCreate,
    SimulationAccountRead,
)
from backend.app.schemas.weekly_analysis import (
    AnalysisTaskCreate,
    AnalysisTaskRead,
    LatestAdviceRead,
    ModelMetricsRead,
    TaskId,
)
from backend.app.services.backup_service import BackupService
from backend.app.services.agent_iteration_service import AgentIterationService
from backend.app.services.comparison_service import ComparisonService
from backend.app.services.csv_import import CsvImportService
from backend.app.services.indicator_service import IndicatorService
from backend.app.services.instrument_universe import (
    CHART_INSTRUMENTS,
    DISPLAY_ONLY_CODES,
    MODEL_MARKETS,
)
from backend.app.services.investment_calendar_service import InvestmentCalendarService
from backend.app.services.market_calendar import ExchangeCalendarProvider
from backend.app.services.market_data import MarketDataService
from backend.app.services.plan_service import InvestmentPlanService
from backend.app.services.providers import (
    AkShareEtfResearchProvider,
    AkShareIndexProvider,
    AkShareValuationProvider,
    YahooDirectIndexProvider,
)
from backend.app.services.real_account_service import RealAccountService
from backend.app.services.simulation_service import SimulationService
from backend.app.services.strategy_service import StrategyService
from backend.app.services.valuation_service import ValuationService
from backend.app.services.v31_analysis_service import DataGateBlocked, V31AnalysisService
from backend.app.services.v32_analysis_service import V32AnalysisService
from backend.app.services.v32_training_service import V32TrainingError, V32TrainingService
from backend.app.services.v33_data_service import V33DataService
from backend.app.services.v33_public_sources import V33PublicSourcesService
from backend.app.services.v33_runtime_service import V33RuntimeError, V33RuntimeService
from backend.app.services.v33_training_service import V33TrainingError
from backend.app.services.v34_runtime_service import V34RuntimeError, V34RuntimeService
from backend.app.services.v341_runtime_service import V341RuntimeError, V341RuntimeService
from backend.app.services.v342_turning_policy_service import (
    V342PolicyError,
    V342TurningPolicyService,
)
from backend.app.services.v343_market_ui_service import (
    V343MarketUIError,
    V343MarketUIService,
)
from backend.app.services.weekly_analysis_service import (
    WeeklyAnalysisUnavailable,
    build_weekly_analysis_service,
)
from backend.app.weekly_analysis.engine import DatasetQualityError
from backend.app.weekly_analysis.repository import (
    CheckpointNotFound,
    RepositoryCorruption,
    RepositoryError,
    TaskCheckpoint,
)


logger = logging.getLogger(__name__)


class WeeklyRunRequest(BaseModel):
    account_id: int = Field(gt=0)
    plan_id: int = Field(gt=0)
    week_date: date


class TradeRequest(BaseModel):
    instrument_code: str
    side: str
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0)
    execution_date: date
    notes: str | None = None


class ForecastRequest(BaseModel):
    window_days: int = Field(default=60, ge=20, le=260)
    horizon_weeks: int = Field(default=20, ge=1, le=26)
    candidate_count: int = Field(default=5, ge=1, le=12)


class SettingsWrite(BaseModel):
    value: Any
    category: str = "general"
    description: str | None = None


class V31RunRequest(BaseModel):
    refresh: bool = True


class V33ModelActionRequest(BaseModel):
    instrument_code: str


def _json(value: object) -> object:
    return jsonable_encoder(value, custom_encoder={Decimal: lambda item: format(item, "f")})


@dataclass
class _ApplicationRuntime:
    state_values: dict[str, object]
    task_manager: Any
    shutdown_values: tuple[Any, ...] = ()
    ref_count: int = 0


def _build_application_runtime() -> _ApplicationRuntime:
    root = project_root()
    initialize_database(
        legacy_weekly_analysis_audit_root=(
            root / "data" / "weekly_analysis_v2"
        )
    )
    factory = create_session_factory()
    investment_calendar = InvestmentCalendarService(factory)
    exchange_calendar = ExchangeCalendarProvider()
    market = MarketDataService(factory, calendar_provider=exchange_calendar)
    valuations = ValuationService(factory)
    indicators = IndicatorService(factory)
    # Upgrade only incomplete persisted chart payloads.  This is local and
    # deterministic: it never downloads data and never touches frozen model,
    # forecast, evaluation, calibrator, or Champion rows.
    for chart_instrument in sorted(CHART_INSTRUMENTS):
        indicators.recalculate_stale_timeframes(chart_instrument)
    v33_data = V33DataService(factory)
    v33_public_sources = V33PublicSourcesService(v33_data)

    def current_position(symbol: str) -> int | None:
        return investment_calendar.current_positions().get(symbol)

    weekly_analysis = build_weekly_analysis_service(
        factory,
        current_position_provider=current_position,
        audit_root=root / "data" / "weekly_analysis_v2",
    )
    task_manager = weekly_analysis.task_manager
    v32_training = V32TrainingService(
        factory,
        calendar=exchange_calendar,
    )
    v31_analysis = V31AnalysisService(
        factory,
        market=market,
        valuations=valuations,
        indicators=indicators,
        calendar=exchange_calendar,
        investment_calendar=investment_calendar,
    )
    v32_analysis = V32AnalysisService(
        factory,
        training=v32_training,
        v31_data_pipeline=v31_analysis,
        calendar=exchange_calendar,
        investment_calendar=investment_calendar,
    )
    v33_runtime = V33RuntimeService(
        factory,
        calendar=exchange_calendar,
        investment_calendar=investment_calendar,
        auxiliary_refresh=lambda: v33_public_sources.refresh_all().as_dict(),
    )
    v34_runtime = V34RuntimeService(
        factory,
        calendar=exchange_calendar,
        current_position_provider=current_position,
    )
    v341_runtime = V341RuntimeService(
        factory,
        calendar=exchange_calendar,
    )
    v342_policy = V342TurningPolicyService(
        factory,
        calendar=exchange_calendar,
    )
    v343_market_ui = V343MarketUIService(
        factory,
        calendar=exchange_calendar,
    )
    # Startup performs only the eligibility check.  A market with a new
    # complete week can queue exactly one incremental training task; an
    # unbootstrapped market remains untouched until its historical run exists.
    v32_training.submit_due_incrementals()
    v33_runtime.recover_interrupted_runs()
    v34_runtime.recover_interrupted_runs()
    v341_runtime.recover_interrupted_runs()
    for target_market in ("399006", "159941"):
        if v33_runtime.champion(target_market) is not None:
            v33_runtime.start_incremental(target_market)
        if v34_runtime.champion(target_market) is not None:
            v34_runtime.start_incremental(target_market)
        if (
            v341_runtime.champion(target_market) is not None
            and v341_runtime.incremental_due(target_market)
        ):
            v341_runtime.start_incremental(target_market)
    return _ApplicationRuntime(
        task_manager=task_manager,
        shutdown_values=(task_manager, v32_training, v33_runtime, v34_runtime, v341_runtime),
        state_values={
            "root": root,
            "sessions": factory,
            "market": market,
            "valuations": valuations,
            "indicators": indicators,
            "plans": InvestmentPlanService(factory),
            "simulations": SimulationService(factory),
            "real_accounts": RealAccountService(factory),
            "strategies": StrategyService(factory),
            "comparisons": ComparisonService(factory),
            "forecasts": HistoricalSimilarityForecaster(factory),
            "backups": BackupService(root),
            "agent_iterations": AgentIterationService(
                AgentIterationStorage(root / "data" / "model_iterations")
            ),
            "investment_calendar": investment_calendar,
            "weekly_analysis": weekly_analysis,
            "analysis_tasks": task_manager,
            "weekly_analysis_repository": task_manager.repository,
            "v31_analysis": v31_analysis,
            "v32_training": v32_training,
            "v32_analysis": v32_analysis,
            "v33_data": v33_data,
            "v33_public_sources": v33_public_sources,
            "v33_runtime": v33_runtime,
            "v34_runtime": v34_runtime,
            "v341_runtime": v341_runtime,
            "v342_policy": v342_policy,
            "v343_market_ui": v343_market_ui,
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime_lock = app.state._runtime_lock
    with runtime_lock:
        runtime = getattr(app.state, "_application_runtime", None)
        if runtime is None:
            runtime = _build_application_runtime()
            app.state._application_runtime = runtime
            for name, value in runtime.state_values.items():
                setattr(app.state, name, value)
        runtime.ref_count += 1
    try:
        yield
    finally:
        shutdown_values: tuple[Any, ...] = ()
        with runtime_lock:
            current = getattr(app.state, "_application_runtime", None)
            if current is runtime:
                runtime.ref_count -= 1
                if runtime.ref_count == 0:
                    for name, value in runtime.state_values.items():
                        if getattr(app.state, name, None) is value:
                            delattr(app.state, name)
                    delattr(app.state, "_application_runtime")
                    shutdown_values = runtime.shutdown_values
        for shutdown_service in shutdown_values:
            shutdown_service.shutdown(wait=False)


app = FastAPI(title="ETF Investment Research Lab", version="1.0.0", lifespan=lifespan)
app.state._runtime_lock = RLock()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, error: ValueError) -> JSONResponse:
    if isinstance(error, ValidationError):
        logger.error(
            "Pydantic validation error reached the global ValueError handler",
            exc_info=(type(error), error, error.__traceback__),
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    logger.warning(
        "Request validation raised an unsafe ValueError",
        exc_info=(type(error), error, error.__traceback__),
    )
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request"},
    )


@app.exception_handler(ArtifactConsistencyError)
async def artifact_consistency_handler(
    _request: Request,
    error: ArtifactConsistencyError,
) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})


def _service(request: Request, name: str) -> Any:
    return getattr(request.app.state, name)


def weekly_analysis_service(request: Request) -> Any:
    """Resolve the lifespan singleton; never compose services per request."""

    return _active_runtime_value(request, "weekly_analysis")


def analysis_task_manager(request: Request) -> Any:
    return _active_runtime_value(request, "analysis_tasks")


def weekly_analysis_repository(request: Request) -> Any:
    return _active_runtime_value(
        request, "weekly_analysis_repository"
    )


def _active_runtime_value(request: Request, name: str) -> Any:
    state = request.app.state
    runtime_lock = state._runtime_lock
    with runtime_lock:
        runtime = getattr(state, "_application_runtime", None)
        value = getattr(state, name)
        if runtime is not None and (
            runtime.ref_count < 1
            or runtime.state_values.get(name) is not value
        ):
            raise HTTPException(
                status_code=503,
                detail=PublicDataUnavailableError.public_detail,
            )
        return value


def _weekly_symbol(symbol: str) -> str:
    if symbol not in {"399006", "NDX"}:
        raise PublicValidationError(
            f"Unsupported weekly-analysis instrument {symbol!r}"
        )
    return symbol


def _v33_symbol(symbol: str) -> str:
    if symbol not in {"399006", "159941"}:
        raise PublicValidationError(
            f"V3.3 supports only 399006 and 159941; NDX is benchmark-only, got {symbol!r}"
        )
    return symbol


def _v34_symbol(symbol: str) -> str:
    if symbol not in {"399006", "159941"}:
        raise PublicValidationError(
            f"V3.4 supports only 399006 and 159941, got {symbol!r}"
        )
    return symbol


_PUBLIC_TASK_PROGRESS_KEYS = frozenset(
    {
        "stage",
        "completed_weeks",
        "total_weeks",
        "iteration_id",
        "work_version",
        "model_version",
        "last_checkpoint",
        "elapsed_seconds",
    }
)
_PUBLIC_TASK_STATUSES = frozenset(
    {
        "queued",
        "preparing_data",
        "building_features",
        "iterating",
        "validating",
        "generating_advice",
        "completed",
        "failed",
        "recoverable",
    }
)


def _public_task_progress(checkpoint: TaskCheckpoint) -> dict[str, object]:
    raw = checkpoint.progress
    if not isinstance(raw, Mapping):
        raise RepositoryCorruption("task progress has an invalid shape")
    progress = {
        key: raw[key]
        for key in _PUBLIC_TASK_PROGRESS_KEYS
        if key in raw
    }
    stage = progress.get("stage")
    if stage is not None and stage not in _PUBLIC_TASK_STATUSES:
        raise RepositoryCorruption("task progress stage is invalid")
    for key in ("completed_weeks", "total_weeks"):
        value = progress.get(key)
        if value is not None and (
            type(value) is not int or value < 0
        ):
            raise RepositoryCorruption(
                f"task progress {key} is invalid"
            )
    for key, prefix in (
        ("iteration_id", "I"),
        ("work_version", "W"),
        ("model_version", "M"),
    ):
        value = progress.get(key)
        if value is not None and (
            not isinstance(value, str)
            or re.fullmatch(rf"{prefix}[0-9]{{4,}}", value) is None
        ):
            raise RepositoryCorruption(
                f"task progress {key} is invalid"
            )
    checkpoint_value = progress.get("last_checkpoint")
    if checkpoint_value is not None and (
        not isinstance(checkpoint_value, str)
        or re.fullmatch(r"[0-9]{4}-W[0-9]{2}", checkpoint_value)
        is None
    ):
        raise RepositoryCorruption(
            "task progress last_checkpoint is invalid"
        )
    elapsed = progress.get("elapsed_seconds")
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float, Decimal))
        or elapsed < 0
    ):
        raise RepositoryCorruption(
            "task progress elapsed_seconds is invalid"
        )
    return progress


def _public_analysis_message(
    checkpoint: TaskCheckpoint,
) -> str | None:
    if checkpoint.status == "failed":
        return (
            "Weekly analysis failed; inspect local audit logs"
        )
    if checkpoint.status == "recoverable":
        return "Weekly analysis was interrupted and can be resumed"
    return None


def _analysis_task_payload(
    checkpoint: TaskCheckpoint,
    *,
    reused: bool = False,
) -> dict[str, object]:
    progress = _public_task_progress(checkpoint)
    return AnalysisTaskRead(
        id=checkpoint.task_id,
        instrument_code=checkpoint.symbol,
        status=checkpoint.status,
        completed_weeks=checkpoint.completed_weeks,
        total_weeks=checkpoint.total_weeks,
        last_iteration=checkpoint.last_iteration,
        last_work_version=checkpoint.last_work_version,
        last_model_version=checkpoint.last_model_version,
        last_completed_week=checkpoint.last_completed_week,
        progress=progress,
        message=_public_analysis_message(checkpoint),
        reused=reused,
    ).model_dump(mode="json")


def _raise_weekly_api_error(error: Exception) -> None:
    if isinstance(error, PublicApiError):
        raise HTTPException(
            status_code=error.status_code,
            detail=error.public_detail,
        ) from error
    if isinstance(error, CheckpointNotFound):
        public_error = PublicNotFoundError("analysis task does not exist")
        raise HTTPException(
            status_code=public_error.status_code,
            detail=public_error.public_detail,
        ) from error
    if isinstance(
        error,
        (
            DatasetQualityError,
            RepositoryCorruption,
            RepositoryError,
            WeeklyAnalysisUnavailable,
        ),
    ):
        logger.exception("Weekly analysis data is unavailable")
        public_error = PublicDataUnavailableError()
        raise HTTPException(
            status_code=public_error.status_code,
            detail=public_error.public_detail,
        ) from error
    logger.exception("Unexpected weekly analysis API failure")
    raise HTTPException(
        status_code=500,
        detail="Internal server error",
    ) from error


def _model_metrics_payload(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif hasattr(value, "to_dict"):
        payload = dict(value.to_dict())
    else:
        raise RepositoryCorruption("model metrics have an invalid shape")
    return ModelMetricsRead.model_validate(payload).model_dump(mode="json")


def _latest_advice_payload(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        advice_payload = getattr(value, "payload", None)
        if not isinstance(advice_payload, Mapping):
            raise RepositoryCorruption("latest advice has an invalid shape")
        payload = {
            **dict(advice_payload),
            "id": getattr(value, "id", None),
            "instrument_code": getattr(
                value,
                "symbol",
                advice_payload.get("instrument_code"),
            ),
            "iteration_id": getattr(value, "iteration_id", None),
            "work_version": getattr(value, "work_version", None),
            "model_version": getattr(
                value,
                "model_version",
                advice_payload.get("model_version"),
            ),
            "advice_generation": getattr(
                value, "advice_generation", None
            ),
            "advice_at": getattr(value, "advice_at", None),
        }
    validated = LatestAdviceRead.model_validate(payload)
    normalized = _json(validated.model_dump(mode="python"))
    if not isinstance(normalized, dict):
        raise RepositoryCorruption("latest advice has an invalid shape")
    return LatestAdviceRead.model_validate(normalized).model_dump(mode="json")


def _position_event_v2_payload(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    payload["instrument_code"] = payload.pop(
        "instrument_code",
        payload.pop("index", None),
    )
    payload["operation_date"] = payload.pop(
        "operation_date",
        payload.pop("date", None),
    )
    return payload


def _successful_aggregation_timeframes(result: object) -> tuple[str, ...]:
    if not isinstance(result, dict):
        return ()
    statuses = result.get("timeframe_status")
    if isinstance(statuses, dict):
        return tuple(
            timeframe
            for timeframe in ("weekly", "monthly")
            if isinstance(statuses.get(timeframe), dict)
            and statuses[timeframe].get("status") == "success"
        )
    return tuple(
        timeframe
        for timeframe in ("weekly", "monthly")
        if isinstance(result.get(timeframe), int)
    )


def _aggregation_status(result: object) -> str:
    if not isinstance(result, dict):
        return "failure"
    statuses = result.get("timeframe_status")
    if not isinstance(statuses, dict) or not statuses:
        return "failure"
    values = [
        (
            statuses[timeframe].get("status")
            if isinstance(statuses.get(timeframe), dict)
            else None
        )
        for timeframe in ("weekly", "monthly")
    ]
    succeeded = sum(value == "success" for value in values)
    if succeeded == len(values) and values:
        return "success"
    if succeeded:
        return "partial_failure"
    return "failure"


def _calculation_payload(result: object) -> dict[str, object]:
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    status = getattr(result, "status", None)
    return {
        "status": getattr(status, "value", status) or "unknown",
    }


def _recalculate_chart_timeframes(
    request: Request,
    instrument_code: str,
    timeframes: Iterable[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Synchronize only price timeframes whose publication succeeded."""
    indicator_service = _service(request, "indicators")
    selected_source = (
        indicator_service.chart_timeframes
        if timeframes is None
        else timeframes
    )
    selected = tuple(dict.fromkeys(selected_source))
    results: dict[str, dict[str, object]] = {}
    for timeframe in selected:
        try:
            calculation = indicator_service.recalculate(
                instrument_code, timeframe
            )
        except Exception as error:
            results[timeframe] = {
                "status": "error",
                "timeframe": timeframe,
                "error": str(error),
            }
            continue
        results[timeframe] = _calculation_payload(calculation)
    return results


def _aggregate_market_timeframes(
    request: Request,
    instrument_code: str,
    *,
    weekly_result=None,
    as_of: date | None = None,
) -> dict[str, object] | object:
    """Publish periods through the market service's injected exchange calendar."""
    return _service(request, "market").aggregate_periods(
        instrument_code,
        weekly_result=weekly_result,
        as_of=as_of,
    )


def _instrument(session: Session, code: str) -> Instrument:
    row = session.scalar(select(Instrument).where(Instrument.code == code))
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown instrument: {code}")
    return row


def _instrument_dict(row: Instrument) -> dict[str, object]:
    return {
        "id": row.id,
        "code": row.code,
        "name": row.name,
        "exchange": row.exchange,
        "currency": row.currency,
        "enabled": row.is_active,
        "description": row.description,
        "research_kind": (row.extra_data or {}).get("research_kind", row.category),
    }


def _price_dict(
    row: MarketPrice,
    *,
    volume_available: bool = True,
    is_complete: bool = True,
    period_status: str = "COMPLETE",
) -> dict[str, object]:
    return {
        "date": row.trade_date,
        "open": row.open_price or row.close_price,
        "high": row.high_price or row.close_price,
        "low": row.low_price or row.close_price,
        "close": row.close_price,
        "volume": (
            row.volume * row.volume_multiplier
            if volume_available and row.volume is not None
            else None
        ),
        "volume_multiplier": row.volume_multiplier,
        "amount": row.turnover,
        "source": row.source,
        "volume_source": row.volume_source,
        "timeframe": row.timeframe,
        "is_complete": is_complete,
        "period_status": period_status,
    }


def _provisional_period_row(
    daily_rows: list[MarketPrice],
    completed_rows: list[MarketPrice],
    timeframe: str,
    *,
    volume_available: bool,
) -> dict[str, object] | None:
    """Build a read-only current-period overlay; never persist it as complete."""

    if timeframe not in {"weekly", "monthly"} or not daily_rows:
        return None
    latest_daily = daily_rows[-1].trade_date
    period_key = (
        latest_daily.isocalendar()[:2]
        if timeframe == "weekly"
        else (latest_daily.year, latest_daily.month)
    )

    def key(value: date) -> tuple[int, int]:
        return (
            value.isocalendar()[:2]
            if timeframe == "weekly"
            else (value.year, value.month)
        )

    if completed_rows and key(completed_rows[-1].trade_date) == period_key:
        return None
    period_rows = [row for row in daily_rows if key(row.trade_date) == period_key]
    if not period_rows:
        return None
    volumes = [
        row.volume * row.volume_multiplier
        for row in period_rows
        if row.volume is not None
    ]
    turnovers = [row.turnover for row in period_rows if row.turnover is not None]
    return {
        "date": period_rows[-1].trade_date,
        "open": period_rows[0].open_price or period_rows[0].close_price,
        "high": max(row.high_price or row.close_price for row in period_rows),
        "low": min(row.low_price or row.close_price for row in period_rows),
        "close": period_rows[-1].close_price,
        "volume": sum(volumes, Decimal("0")) if volume_available and volumes else None,
        "volume_multiplier": 1,
        "amount": sum(turnovers, Decimal("0")) if turnovers else None,
        "source": "PROVISIONAL_DAILY_AGGREGATION",
        "volume_source": "PROVISIONAL_DAILY_AGGREGATION" if volumes else None,
        "timeframe": timeframe,
        "is_complete": False,
        "period_status": "INCOMPLETE_CURRENT_PERIOD",
    }


@app.get("/api/health")
def health(request: Request) -> object:
    root = request.app.state.root
    return _json({"status": "ok", "mode": "local_only", "database": "data/investment_lab.db", "frontend_built": (root / "frontend" / "dist" / "index.html").exists()})


@app.get("/api/agent-iterations/{instrument_code}")
def agent_iteration_summary(request: Request, instrument_code: str) -> object:
    return _json(_service(request, "agent_iterations").summary(instrument_code))


@app.get("/api/agent-iterations/{instrument_code}/{iteration_number}")
def agent_iteration_detail(
    request: Request,
    instrument_code: str,
    iteration_number: int,
) -> object:
    return _json(
        _service(request, "agent_iterations").detail(instrument_code, iteration_number)
    )


@app.get("/api/investment-calendar")
def investment_calendar(
    request: Request,
    instrument_code: str | None = None,
) -> object:
    return _json(
        _service(request, "investment_calendar").list_entries(instrument_code)
    )


@app.get("/api/investment-calendar/current-positions")
def investment_calendar_current_positions(request: Request) -> object:
    return _json(_service(request, "investment_calendar").current_positions())


@app.get("/api/investment-calendar/net-shares", deprecated=True)
def investment_calendar_net_shares(request: Request) -> object:
    return _json(_service(request, "investment_calendar").net_shares())


@app.post("/api/investment-calendar")
def create_investment_calendar_entry(
    request: Request,
    values: PositionEventCreate,
) -> object:
    return _json(_service(request, "investment_calendar").create(values))


@app.patch("/api/investment-calendar/{entry_id}")
def update_investment_calendar_entry(
    request: Request,
    entry_id: int,
    values: PositionEventUpdate,
) -> object:
    return _json(_service(request, "investment_calendar").update(entry_id, values))


@app.delete("/api/investment-calendar/{entry_id}")
def delete_investment_calendar_entry(
    request: Request,
    entry_id: int,
    confirmed: bool = False,
) -> object:
    if not confirmed:
        raise HTTPException(status_code=400, detail="confirmed=true is required")
    _service(request, "investment_calendar").delete(entry_id)
    return _json({"deleted": True, "id": entry_id})


@app.get("/api/instruments")
def instruments(request: Request) -> object:
    with request.app.state.sessions() as session:
        rows = session.scalars(select(Instrument).order_by(Instrument.code)).all()
        rows = [row for row in rows if row.code in {"399006", "159941"}]
        return _json([_instrument_dict(row) for row in rows])


@app.get("/api/dashboard")
def dashboard(request: Request) -> object:
    with request.app.state.sessions() as session:
        cards: list[dict[str, object]] = []
        for instrument in session.scalars(select(Instrument).order_by(Instrument.code)):
            if instrument.code not in {"399006", "159941"}:
                continue
            price = session.scalar(
                select(MarketPrice)
                .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
                .order_by(MarketPrice.trade_date.desc())
            )
            previous = session.scalar(
                select(MarketPrice)
                .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
                .order_by(MarketPrice.trade_date.desc()).offset(1)
            )
            change = None
            if price is not None and previous is not None and previous.close_price:
                change = (price.close_price - previous.close_price) / previous.close_price
            volume_available = instrument.code != "NDX"
            cards.append({
                **_instrument_dict(instrument),
                "latest": (
                    _price_dict(price, volume_available=volume_available)
                    if price
                    else None
                ),
                "change": change,
                "volume_availability": (
                    "available"
                    if volume_available
                    else "not_available_for_direct_index"
                ),
            })
        simulation_count = session.scalar(select(func.count()).select_from(SimulationAccount)) or 0
        real_count = session.scalar(select(func.count()).select_from(RealAccount)) or 0
    return _json({"instruments": cards, "simulation_accounts": simulation_count, "real_accounts": real_count, "disclaimer": "仅供个人研究与模拟，不连接券商、不自动交易；所有结果来自本地数据与确定性规则。"})


@app.get("/api/market/{instrument_code}/prices")
def prices(request: Request, instrument_code: str, timeframe: str = "daily") -> object:
    if timeframe not in {"daily", "weekly", "monthly"}:
        raise HTTPException(status_code=422, detail="timeframe must be daily, weekly, or monthly")
    with request.app.state.sessions() as session:
        instrument = _instrument(session, instrument_code)
        rows = list(session.scalars(
            select(MarketPrice)
            .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == timeframe)
            .order_by(MarketPrice.trade_date)
        ).all())
        daily_rows = (
            list(session.scalars(
                select(MarketPrice)
                .where(
                    MarketPrice.instrument_id == instrument.id,
                    MarketPrice.timeframe == "daily",
                )
                .order_by(MarketPrice.trade_date)
            ).all())
            if timeframe in {"weekly", "monthly"}
            else []
        )
    is_demo = bool(rows) and all((row.source or "").startswith("DEMO") for row in rows)
    volume_available = instrument_code != "NDX"
    payload_rows = [_price_dict(row, volume_available=volume_available) for row in rows]
    provisional = _provisional_period_row(
        daily_rows,
        rows,
        timeframe,
        volume_available=volume_available,
    )
    if provisional is not None:
        payload_rows.append(provisional)
    return _json({"instrument": _instrument_dict(instrument), "timeframe": timeframe, "rows": payload_rows, "source": payload_rows[-1]["source"] if payload_rows else None, "demo": is_demo, "updated_at": rows[-1].updated_at if rows else None, "volume_availability": "available" if volume_available else "not_available_for_direct_index", "current_period_status": provisional["period_status"] if provisional else "COMPLETE"})


@app.post("/api/market/{instrument_code}/refresh")
def refresh_market(request: Request, instrument_code: str) -> object:
    service = _service(request, "market")
    if instrument_code not in CHART_INSTRUMENTS | {"NDX"}:
        raise HTTPException(
            status_code=422,
            detail="Market refresh supports only the five chart instruments and the read-only NDX benchmark",
        )
    provider = (
        AkShareEtfResearchProvider()
        if instrument_code == "159941" or instrument_code in DISPLAY_ONLY_CODES
        else AkShareIndexProvider()
    )
    latest_daily = service.latest_daily_date(instrument_code)
    overlap_start = (
        date(2014, 2, 18)
        if instrument_code == "NDX"
        else (
            latest_daily - timedelta(days=14)
            if latest_daily is not None
            else None
        )
    )
    providers = (
        [YahooDirectIndexProvider(), provider]
        if instrument_code == "NDX"
        else [provider]
    )
    # The weekly-volume overlay validates all-or-nothing coverage against the
    # complete daily baseline, so its source request must cover full history
    # even when the daily price refresh uses a short overlap.
    weekly_result = provider.fetch_weekly(instrument_code, None, None)
    if instrument_code == "159941":
        v33_result = _service(request, "v33_data").refresh_159941(
            start_date=overlap_start,
            end_date=None,
        )
        result_succeeded = v33_result.status == "success"
        cutoff_date = v33_result.data_as_of
        payload: dict[str, object] = asdict(v33_result)
        payload.update(
            {
                "cutoff_date": cutoff_date,
                "cache_used": False,
                "demo": False,
                "error": None if result_succeeded else "; ".join(v33_result.issues),
            }
        )
    else:
        result = service.update_from_providers(
            instrument_code, providers, overlap_start, None
        )
        result_succeeded = bool(result.records)
        cutoff_date = result.cutoff_date
        payload = result.model_dump()
    aggregation_result: object | None = None
    indicator_results: dict[str, dict[str, object]] = {}
    if result_succeeded:
        aggregation_result = _aggregate_market_timeframes(
            request,
            instrument_code,
            weekly_result=weekly_result,
            as_of=cutoff_date,
        )
        indicator_results = _recalculate_chart_timeframes(
            request,
            instrument_code,
            ("daily", *_successful_aggregation_timeframes(aggregation_result)),
        )
    payload["aggregation"] = aggregation_result
    payload["aggregation_status"] = (
        _aggregation_status(aggregation_result)
        if aggregation_result is not None
        else "not_attempted"
    )
    payload["indicator_recalculation"] = indicator_results
    # A successful local quote refresh also completes previously saved real
    # trades whose market/indicator context was marked as pending.
    payload["real_accounts_recalculated"] = (
        _service(request, "real_accounts").recalculate_all_accounts()
        if result_succeeded and instrument_code in MODEL_MARKETS
        else 0
    )
    return _json(payload)


@app.post("/api/market/{instrument_code}/aggregate")
def aggregate_market(request: Request, instrument_code: str) -> object:
    result = _aggregate_market_timeframes(request, instrument_code)
    if isinstance(result, dict):
        _recalculate_chart_timeframes(
            request,
            instrument_code,
            _successful_aggregation_timeframes(result),
        )
    return _json(result)


@app.get("/api/valuations/{instrument_code}")
def valuations(request: Request, instrument_code: str) -> object:
    try:
        return _json(_service(request, "valuations").read(instrument_code))
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/valuations/{instrument_code}/series")
def valuation_series(request: Request, instrument_code: str) -> object:
    try:
        return _json(_service(request, "valuations").read(instrument_code)["series"])
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/valuations/{instrument_code}/refresh")
def refresh_valuations(request: Request, instrument_code: str) -> object:
    result = _service(request, "valuations").refresh(instrument_code, AkShareValuationProvider())
    payload = result.model_dump()
    payload["strategy_recomputed"] = False
    if result.status == "success":
        with request.app.state.sessions() as session:
            instrument = _instrument(session, instrument_code)
            strategy = session.scalar(
                select(StrategyDefinition)
                .where(StrategyDefinition.enabled.is_(True))
                .order_by(StrategyDefinition.id)
            )
            cutoff = session.scalar(
                select(MarketPrice.trade_date)
                .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
                .order_by(MarketPrice.trade_date.desc())
            )
        if strategy is not None and cutoff is not None:
            _service(request, "strategies").evaluate_and_store(strategy.name, instrument_code, cutoff)
            payload["strategy_recomputed"] = True
    if result.status == "success":
        payload["real_accounts_recalculated"] = _service(request, "real_accounts").recalculate_all_accounts()
    else:
        payload["real_accounts_recalculated"] = 0
    return _json(payload)


@app.post("/api/market/{instrument_code}/csv/preview")
async def preview_market_csv(request: Request, instrument_code: str) -> object:
    with request.app.state.sessions() as session:
        _instrument(session, instrument_code)
    return _json(CsvImportService(request.app.state.sessions).preview(await request.body()))


@app.post("/api/market/{instrument_code}/csv/import")
async def import_market_csv(request: Request, instrument_code: str) -> object:
    result = CsvImportService(request.app.state.sessions).confirm_import(instrument_code, await request.body())
    aggregation_result: object | None = None
    indicator_results: dict[str, dict[str, object]] = {}
    if result.added or result.updated:
        aggregation_result = _aggregate_market_timeframes(request, instrument_code)
        indicator_results = _recalculate_chart_timeframes(
            request,
            instrument_code,
            ("daily", *_successful_aggregation_timeframes(aggregation_result)),
        )
    payload = result.model_dump()
    payload["aggregation"] = aggregation_result
    payload["aggregation_status"] = (
        _aggregation_status(aggregation_result)
        if aggregation_result is not None
        else "not_attempted"
    )
    payload["indicator_recalculation"] = indicator_results
    payload["real_accounts_recalculated"] = (
        _service(request, "real_accounts").recalculate_all_accounts()
        if result.added or result.updated
        else 0
    )
    return _json(payload)


@app.post("/api/indicators/{instrument_code}/calculate")
def calculate_indicators(request: Request, instrument_code: str, timeframe: str = "daily") -> object:
    return _json(asdict(_service(request, "indicators").recalculate(instrument_code, timeframe)))


@app.get("/api/indicators/{instrument_code}")
def indicator_rows(request: Request, instrument_code: str, timeframe: str = "daily") -> object:
    with request.app.state.sessions() as session:
        instrument = _instrument(session, instrument_code)
        rows = session.scalars(
            select(IndicatorRecord)
            .where(IndicatorRecord.instrument_id == instrument.id, IndicatorRecord.timeframe == timeframe)
            .order_by(IndicatorRecord.indicator_date)
        ).all()
    return _json([{"date": row.indicator_date, **dict(row.indicator_values.get("values", {})), "metadata": row.indicator_values.get("metadata", {})} for row in rows])


@app.get("/api/plans")
def list_plans(request: Request, instrument_code: str | None = None) -> object:
    return _json(_service(request, "plans").list(instrument_code=instrument_code))


@app.post("/api/plans")
def create_plan(request: Request, values: InvestmentPlanCreate) -> object:
    return _json(_service(request, "plans").create(values))


@app.patch("/api/plans/{plan_id}")
def update_plan(request: Request, plan_id: int, changes: InvestmentPlanUpdate) -> object:
    return _json(_service(request, "plans").update(plan_id, changes))


@app.delete("/api/plans/{plan_id}")
def delete_plan(request: Request, plan_id: int, confirmed: bool = False) -> Response:
    if not confirmed:
        raise HTTPException(status_code=400, detail="Set confirmed=true after the UI confirmation dialog")
    _service(request, "plans").delete(plan_id)
    return Response(status_code=204)


@app.get("/api/simulations/accounts")
def simulation_accounts(request: Request) -> object:
    with request.app.state.sessions() as session:
        rows = session.scalars(select(SimulationAccount).order_by(SimulationAccount.id)).all()
        payload = [SimulationAccountRead(id=row.id, name=row.name, initial_cash=row.initial_cash, cash_balance=row.cash_balance, portfolio_value=row.portfolio_value, strategy_id=row.strategy_id, enabled=row.enabled) for row in rows]
    return _json(payload)


@app.post("/api/simulations/accounts")
def create_simulation_account(request: Request, values: SimulationAccountCreate) -> object:
    return _json(_service(request, "simulations").create_account(values))


@app.post("/api/simulations/run-weekly")
def run_weekly(request: Request, values: WeeklyRunRequest) -> object:
    result = _service(request, "simulations").run_weekly(values.account_id, values.plan_id, values.week_date)
    return _json({"transaction_id": result.transaction.id, "idempotent": result.idempotent, "theoretical_mode": result.theoretical_mode, "price_date": result.price_date})


@app.post("/api/simulations/{account_id}/trade")
def simulation_trade(request: Request, account_id: int, values: TradeRequest) -> object:
    row = _service(request, "simulations").execute_trade(account_id, values.instrument_code, values.side, values.quantity, values.price, values.execution_date, source="manual", notes=values.notes)
    return _json({"transaction_id": row.id, "cash_after": row.cash_after, "fee": row.fee})


@app.post("/api/simulations/{account_id}/recalculate")
def recalculate_simulation(request: Request, account_id: int) -> object:
    return _json(_service(request, "simulations").recalculate_account(account_id))


@app.get("/api/simulations/{account_id}/series")
def simulation_series(request: Request, account_id: int) -> object:
    return _json(_service(request, "simulations").chart_series(account_id))


@app.get("/api/fees")
def get_fees(request: Request) -> object:
    with request.app.state.sessions() as session:
        setting = session.scalar(select(AppSetting).where(AppSetting.key == "fee.configuration"))
    return _json(setting.value if setting else {})


@app.put("/api/fees")
def set_fees(request: Request, values: dict[str, object]) -> object:
    return _json(_service(request, "simulations").set_fee_configuration(values).as_dict())


@app.get("/api/real-accounts")
def real_accounts(request: Request) -> object:
    with request.app.state.sessions() as session:
        rows = session.scalars(select(RealAccount).order_by(RealAccount.id)).all()
        payload = [RealAccountRead(id=row.id, name=row.name, broker_name=row.broker_name, initial_cash=row.initial_cash, cash_balance=row.cash_balance, enabled=row.enabled) for row in rows]
    return _json(payload)


@app.post("/api/real-accounts")
def create_real_account(request: Request, values: RealAccountCreate) -> object:
    return _json(_service(request, "real_accounts").create_account(values))


@app.get("/api/real-accounts/{account_id}")
def get_real_account(request: Request, account_id: int) -> object:
    return _json(_service(request, "real_accounts").get_account(account_id))


@app.get("/api/real-accounts/{account_id}/transactions")
def real_transactions(request: Request, account_id: int) -> object:
    return _json(_service(request, "real_accounts").transactions(account_id))


@app.post("/api/real-accounts/{account_id}/transactions")
def create_real_transaction(request: Request, account_id: int, values: RealTransactionCreate) -> object:
    return _json(_service(request, "real_accounts").create_transaction(account_id, values))


@app.post("/api/real-accounts/{account_id}/transactions/preview")
def preview_real_transaction(request: Request, account_id: int, values: RealTransactionCreate) -> object:
    """Calculate the automatic shares, fee and local-market context before saving."""
    return _json(_service(request, "real_accounts").preview_transaction(account_id, values))


@app.patch("/api/real-accounts/{account_id}/transactions/{transaction_id}")
def update_real_transaction(request: Request, account_id: int, transaction_id: int, values: RealTransactionUpdate) -> object:
    return _json(_service(request, "real_accounts").update_transaction(account_id, transaction_id, values))


@app.delete("/api/real-accounts/{account_id}/transactions/{transaction_id}")
def delete_real_transaction(request: Request, account_id: int, transaction_id: int, confirmed: bool = False) -> Response:
    if not confirmed:
        raise HTTPException(status_code=400, detail="Set confirmed=true after the UI confirmation dialog")
    _service(request, "real_accounts").delete_transaction(account_id, transaction_id)
    return Response(status_code=204)


@app.get("/api/real-accounts/{account_id}/positions")
def real_positions(request: Request, account_id: int) -> object:
    return _json(_service(request, "real_accounts").positions(account_id))


@app.get("/api/real-accounts/{account_id}/series")
def real_series(
    request: Request,
    account_id: int,
    timeframe: str = "daily",
    start_date: date | None = None,
    end_date: date | None = None,
    instrument_code: str | None = None,
) -> object:
    return _json(
        _service(request, "real_accounts").chart_series(
            account_id,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            instrument_code=instrument_code,
        )
    )


@app.get("/api/real-accounts/{account_id}/comparison")
def real_account_comparison(request: Request, account_id: int) -> object:
    return _json(_service(request, "real_accounts").comparison_summary(account_id))


@app.get("/api/real-accounts/{account_id}/export")
def export_real_account(request: Request, account_id: int, format: str = "csv") -> Response:
    service = _service(request, "real_accounts")
    if format.lower() == "xlsx":
        return Response(
            content=service.export_xlsx(account_id),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="real-account-{account_id}.xlsx"'},
        )
    if format.lower() != "csv":
        raise HTTPException(status_code=400, detail="format must be csv or xlsx")
    return Response(
        content=service.export_csv(account_id),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="real-account-{account_id}.csv"'},
    )


@app.post("/api/real-accounts/{account_id}/import")
async def import_real_account(request: Request, account_id: int) -> object:
    return _json(_service(request, "real_accounts").import_csv(account_id, await request.body()))


@app.get("/api/comparison")
def compare_accounts(request: Request, real_account_id: int, simulation_account_id: int) -> object:
    return _json(_service(request, "comparisons").compare(real_account_id, simulation_account_id))


@app.get("/api/strategies")
def strategies(request: Request) -> object:
    with request.app.state.sessions() as session:
        rows = session.scalars(select(StrategyDefinition).order_by(StrategyDefinition.id)).all()
    return _json([{"id": row.id, "name": row.name, "version": row.version, "enabled": row.enabled, "parameters": row.parameters} for row in rows])


@app.post("/api/strategies/{instrument_code}/evaluate")
def evaluate_strategy(request: Request, instrument_code: str, as_of: date | None = None) -> object:
    with request.app.state.sessions() as session:
        instrument = _instrument(session, instrument_code)
        strategy = session.scalar(select(StrategyDefinition).where(StrategyDefinition.enabled.is_(True)).order_by(StrategyDefinition.id))
        cutoff = as_of or session.scalar(select(func.max(MarketPrice.trade_date)).where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily"))
    if strategy is None or cutoff is None:
        raise HTTPException(status_code=400, detail="A strategy and local market data are required")
    result = _service(request, "strategies").evaluate_and_store(strategy.name, instrument_code, cutoff)
    # The dated signal is the audit record.  Expose its rule evidence alongside
    # the concise result so the terminal can show why a buy/sell action was
    # proposed without relying on opaque or AI-generated explanations.
    with request.app.state.sessions() as session:
        signal = session.get(StrategySignal, result.signal_id)
        signal_data = dict(signal.signal_data or {}) if signal is not None else {}
    payload = asdict(result)
    payload.update(
        {
            "component_scores": signal_data.get("component_scores", {}),
            "triggered_rules": signal_data.get("triggered_rules", []),
            "reverse_risks": signal_data.get("reverse_risks", []),
            "reasons": signal_data.get("reasons", []),
            "signal_data": signal_data,
        }
    )
    return _json(payload)


@app.get("/api/signals")
def signals(request: Request, instrument_code: str | None = None) -> object:
    rows = _service(request, "strategies").list_signals(etf_code=instrument_code)
    return _json([{"id": row.id, "as_of_date": row.as_of_date, "recommendation": row.recommendation_type, "score": row.score, "confidence": row.confidence, "rationale": row.rationale, "data": row.signal_data} for row in rows])


@app.get("/api/reports")
def reports(request: Request, instrument_code: str | None = None) -> object:
    rows = _service(request, "strategies").list_reports(etf_code=instrument_code)
    return _json([{"id": row.id, "title": row.title, "date": row.report_date, "content": row.content, "status": row.status} for row in rows])


@app.post("/api/forecasts/{instrument_code}")
def create_forecast(request: Request, instrument_code: str, values: ForecastRequest) -> object:
    try:
        return _json(_service(request, "forecasts").run(instrument_code, **values.model_dump()))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/forecasts/{instrument_code}/availability")
def forecast_availability(
    request: Request,
    instrument_code: str,
    window_days: int = 60,
    horizon_weeks: int = 13,
    candidate_count: int = 5,
) -> object:
    try:
        return _json(
            _service(request, "forecasts").availability(
                instrument_code,
                window_days=window_days,
                horizon_weeks=horizon_weeks,
                candidate_count=candidate_count,
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/settings")
def settings(request: Request) -> object:
    with request.app.state.sessions() as session:
        rows = session.scalars(select(AppSetting).order_by(AppSetting.key)).all()
    return _json([{ "key": row.key, "value": row.value, "category": row.category, "description": row.description } for row in rows])


@app.put("/api/settings/{key}")
def write_setting(request: Request, key: str, values: SettingsWrite) -> object:
    with request.app.state.sessions() as session, session.begin():
        row = session.scalar(select(AppSetting).where(AppSetting.key == key))
        if row is None:
            row = AppSetting(key=key, value=values.value, category=values.category, description=values.description)
            session.add(row)
        else:
            row.value, row.category, row.description = values.value, values.category, values.description
        session.flush()
        payload = {"key": row.key, "value": row.value, "category": row.category, "description": row.description}
    return _json(payload)


@app.get("/api/data-update")
def update_log(request: Request) -> object:
    with request.app.state.sessions() as session:
        rows = session.scalars(select(DataUpdateLog).order_by(DataUpdateLog.id.desc()).limit(50)).all()
    return _json([{ "source": row.source, "status": row.status, "started_at": row.started_at, "completed_at": row.completed_at, "records_written": row.records_written, "message": row.message } for row in rows])


@app.get("/api/backups")
def list_backups(request: Request) -> object:
    return _json(_service(request, "backups").list())


@app.post("/api/backups")
def create_backup(request: Request) -> object:
    return _json(_service(request, "backups").create())


@app.post("/api/backups/{name}/restore")
def restore_backup(request: Request, name: str, confirmed: bool = False) -> object:
    if not confirmed:
        raise HTTPException(status_code=400, detail="Set confirmed=true after the UI confirmation dialog")
    return _json(_service(request, "backups").restore(name))


@app.get("/api/exports/all")
def export_all(request: Request) -> FileResponse:
    path = _service(request, "backups").export_zip()
    return FileResponse(path, media_type="application/zip", filename=path.name)


@app.post("/api/imports/all")
async def import_all(request: Request, confirmed: bool = False) -> object:
    if not confirmed:
        raise HTTPException(status_code=400, detail="Set confirmed=true after the UI confirmation dialog")
    return _json(_service(request, "backups").import_zip(await request.body()))


@app.post("/api/v2/analysis/tasks")
def create_weekly_analysis_task(
    values: AnalysisTaskCreate,
    service: Any = Depends(weekly_analysis_service),
) -> Response:
    try:
        symbol = _weekly_symbol(values.instrument_code)
        checkpoint = service.start(symbol)
        reused = checkpoint.status == "completed"
        return JSONResponse(
            status_code=200 if reused else 202,
            content=_analysis_task_payload(
                checkpoint,
                reused=reused,
            ),
        )
    except Exception as error:
        _raise_weekly_api_error(error)


@app.get("/api/v2/analysis/tasks/{task_id}")
def weekly_analysis_task_status(
    task_id: TaskId,
    manager: Any = Depends(analysis_task_manager),
) -> object:
    try:
        checkpoint = manager.get(task_id)
        return _analysis_task_payload(checkpoint)
    except Exception as error:
        _raise_weekly_api_error(error)


@app.post("/api/v2/analysis/tasks/{task_id}/resume")
def resume_weekly_analysis_task(
    task_id: TaskId,
    manager: Any = Depends(analysis_task_manager),
) -> Response:
    try:
        checkpoint = manager.get(task_id)
        if checkpoint.status != "recoverable":
            raise PublicConflictError(
                "analysis task is not recoverable"
            )
        latest_complete_week = checkpoint.progress.get(
            "latest_complete_week"
        )
        model_line = checkpoint.progress.get("model_line")
        if not isinstance(latest_complete_week, str) or not isinstance(
            model_line, str
        ):
            raise RepositoryCorruption(
                "recoverable task identity is incomplete"
            )
        resumed = manager.submit(
            checkpoint.symbol,
            latest_complete_week=latest_complete_week,
            model_line=model_line,
        )
        if resumed.task_id != checkpoint.task_id:
            raise RepositoryCorruption(
                "resume did not preserve the task identity"
            )
        return JSONResponse(
            status_code=202,
            content=_analysis_task_payload(resumed),
        )
    except Exception as error:
        _raise_weekly_api_error(error)


@app.get("/api/v2/models/{symbol}/metrics")
def weekly_model_metrics(
    symbol: str,
    repository: Any = Depends(weekly_analysis_repository),
) -> object:
    try:
        symbol = _weekly_symbol(symbol)
        return _model_metrics_payload(repository.metrics(symbol))
    except Exception as error:
        _raise_weekly_api_error(error)


@app.get("/api/v2/advice/{symbol}/latest")
def latest_weekly_advice(
    symbol: str,
    repository: Any = Depends(weekly_analysis_repository),
) -> object:
    try:
        symbol = _weekly_symbol(symbol)
        payload = _latest_advice_payload(
            repository.latest_advice(symbol)
        )
        if payload is None:
            raise PublicNotFoundError(
                f"no advice exists for {symbol}"
            )
        return payload
    except Exception as error:
        _raise_weekly_api_error(error)


@app.get("/api/v2/position-events")
def v2_position_events(
    request: Request,
    instrument_code: str | None = None,
) -> object:
    try:
        if instrument_code is not None:
            _v33_symbol(instrument_code)
        rows = _service(request, "investment_calendar").list_entries(
            instrument_code
        )
        return _json([_position_event_v2_payload(row) for row in rows])
    except Exception as error:
        _raise_weekly_api_error(error)


@app.post("/api/v2/position-events")
def create_v2_position_event(
    request: Request,
    values: PositionEventCreate,
) -> object:
    try:
        row = _service(request, "investment_calendar").create(values)
        return _json(_position_event_v2_payload(row))
    except Exception as error:
        _raise_weekly_api_error(error)


@app.patch("/api/v2/position-events/{entry_id}")
def update_v2_position_event(
    request: Request,
    entry_id: int,
    values: PositionEventUpdate,
) -> object:
    try:
        row = _service(request, "investment_calendar").update(
            entry_id, values
        )
        return _json(_position_event_v2_payload(row))
    except Exception as error:
        _raise_weekly_api_error(error)


@app.delete("/api/v2/position-events/{entry_id}")
def delete_v2_position_event(
    request: Request,
    entry_id: int,
    confirmed: bool = False,
) -> object:
    if not confirmed:
        error = PublicValidationError("confirmed=true is required")
        raise HTTPException(
            status_code=error.status_code,
            detail=error.public_detail,
        )
    try:
        _service(request, "investment_calendar").delete(entry_id)
        return _json({"deleted": True, "id": entry_id})
    except Exception as error:
        _raise_weekly_api_error(error)


@app.post("/api/v3.1/analysis/{symbol}/run")
def run_v31_analysis(
    request: Request,
    symbol: str,
    values: V31RunRequest,
) -> object:
    """Execute the complete V3.1 refresh-to-advice pipeline."""

    symbol = _weekly_symbol(symbol)
    return _json(
        _service(request, "v31_analysis").run(
            symbol,
            refresh=values.refresh,
        )
    )


@app.get("/api/v3.1/analysis/{symbol}/latest")
def latest_v31_analysis(request: Request, symbol: str) -> object:
    symbol = _weekly_symbol(symbol)
    payload = _service(request, "v31_analysis").latest(symbol)
    if payload is None:
        raise PublicNotFoundError(f"no V3.1 analysis exists for {symbol}")
    return _json(payload)


@app.get("/api/v3.1/analysis/runs/{run_id}")
def v31_analysis_run(request: Request, run_id: str) -> object:
    return _json(_service(request, "v31_analysis").get_run(run_id))


@app.get("/api/v3.2/training/status")
def v32_training_status(request: Request, market: str | None = None) -> object:
    try:
        return _json(_service(request, "v32_training").status(market))
    except V32TrainingError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v3.2/training/{market}/bootstrap", status_code=202)
def start_v32_bootstrap(request: Request, market: str) -> object:
    try:
        market = _weekly_symbol(market)
        return _json(_service(request, "v32_training").submit_bootstrap(market))
    except V32TrainingError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/v3.2/training/{market}/incremental", status_code=202)
def start_v32_incremental(request: Request, market: str) -> object:
    try:
        market = _weekly_symbol(market)
        return _json(_service(request, "v32_training").submit_incremental(market))
    except V32TrainingError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/v3.2/training/{market}/curve")
def v32_iteration_curve(request: Request, market: str) -> object:
    try:
        market = _weekly_symbol(market)
        return _json(_service(request, "v32_training").iteration_curve(market))
    except V32TrainingError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v3.2/models/{market}/champion")
def v32_champion(request: Request, market: str) -> object:
    try:
        market = _weekly_symbol(market)
        return _json(_service(request, "v32_training").champion(market))
    except V32TrainingError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/v3.2/analysis/{market}/run")
def run_v32_analysis(
    request: Request,
    market: str,
    values: V31RunRequest,
) -> object:
    """Run inference with the latest Champion without changing training state."""

    try:
        market = _weekly_symbol(market)
        return _json(
            _service(request, "v32_analysis").run(
                market,
                refresh=values.refresh,
            )
        )
    except (V32TrainingError, DataGateBlocked) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v3.2/analysis/{market}/latest")
def latest_v32_analysis(request: Request, market: str) -> object:
    market = _weekly_symbol(market)
    payload = _service(request, "v32_analysis").latest(market)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no V3.2 analysis exists for {market}")
    return _json(payload)


@app.get("/api/v3.2/analysis/runs/{run_id}")
def v32_analysis_run(request: Request, run_id: str) -> object:
    try:
        return _json(_service(request, "v32_analysis").get_run(run_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/v33/model/status")
def v33_model_status(request: Request) -> object:
    """Return polling state for both independent V3.3 target models."""

    return _json(_service(request, "v33_runtime").model_statuses())


@app.post("/api/v33/model/train", status_code=202)
def train_v33_model(
    request: Request,
    values: V33ModelActionRequest,
) -> object:
    """Start the first full bootstrap or one eligible weekly iteration."""

    try:
        market = _v33_symbol(values.instrument_code)
        runtime = _service(request, "v33_runtime")
        status = runtime.model_status(market)
        result = (
            runtime.start_incremental(market)
            if status["bootstrapped"]
            else runtime.start_bootstrap(market)
        )
        return _json({"implementation_revision": "V3.3-20W", **result})
    except (V33TrainingError, V33RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/v33/model/analysis")
def analyze_v33_model(
    request: Request,
    values: V33ModelActionRequest,
) -> object:
    """Run inference only; this endpoint is forbidden from training."""

    try:
        market = _v33_symbol(values.instrument_code)
        return _json(_service(request, "v33_runtime").run_analysis(market))
    except (V33TrainingError, V33RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v33/iterations/{market}")
def v33_iterations(request: Request, market: str) -> object:
    try:
        market = _v33_symbol(market)
        return _json(_service(request, "v33_runtime").curve(market))
    except (V33TrainingError, V33RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v33/forecast/{market}")
def latest_v33_forecast(request: Request, market: str) -> object:
    try:
        market = _v33_symbol(market)
        payload = _service(request, "v33_runtime").latest_analysis(market)
    except (V33TrainingError, V33RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no V3.3 analysis exists for {market}")
    return _json(payload)


@app.post("/api/v33/training/{market}/bootstrap", status_code=202)
def start_v33_bootstrap(request: Request, market: str) -> object:
    try:
        market = _v33_symbol(market)
        return _json(_service(request, "v33_runtime").start_bootstrap(market))
    except (V33TrainingError, V33RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/v33/training/{market}/incremental", status_code=202)
def start_v33_incremental(request: Request, market: str) -> object:
    try:
        market = _v33_symbol(market)
        return _json(_service(request, "v33_runtime").start_incremental(market))
    except (V33TrainingError, V33RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/v33/training/runs/{run_id}")
def v33_training_run(request: Request, run_id: str) -> object:
    try:
        return _json(_service(request, "v33_runtime").training_status(run_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/v33/models/{market}/champion")
def v33_champion(request: Request, market: str) -> object:
    try:
        market = _v33_symbol(market)
        payload = _service(request, "v33_runtime").champion(market)
    except (V33TrainingError, V33RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if payload is None:
        raise HTTPException(status_code=404, detail=f"no V3.3 Champion exists for {market}")
    return _json(payload)


@app.post("/api/v33/data/refresh-auxiliary")
def refresh_v33_auxiliary_sources(request: Request) -> object:
    """Refresh all independent public PIT sources and report every gap."""

    return _json(_service(request, "v33_public_sources").refresh_all().as_dict())


@app.get("/api/v34/model/status")
def v34_model_status(request: Request) -> object:
    return _json(_service(request, "v34_runtime").model_statuses())


@app.post("/api/v34/model/train", status_code=202)
def train_v34_model(
    request: Request,
    values: V33ModelActionRequest,
) -> object:
    try:
        market = _v34_symbol(values.instrument_code)
        runtime = _service(request, "v34_runtime")
        if runtime.champion(market) is None:
            return _json(runtime.start_bootstrap(market))
        return _json(runtime.start_incremental(market))
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error


@app.post("/api/v34/model/analysis")
def analyze_v34_model(
    request: Request,
    values: V33ModelActionRequest,
) -> object:
    try:
        return _json(
            _service(request, "v34_runtime").run_analysis(
                _v34_symbol(values.instrument_code)
            )
        )
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v34/iterations/{market}")
def v34_iterations(request: Request, market: str) -> object:
    try:
        return _json(
            _service(request, "v34_runtime").curve(_v34_symbol(market))
        )
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v34/forecast/{market}")
def latest_v34_forecast(request: Request, market: str) -> object:
    try:
        payload = _service(request, "v34_runtime").latest_forecast(
            _v34_symbol(market)
        )
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error
    if payload is None:
        raise HTTPException(status_code=404, detail="No V3.4 forecast is available")
    return _json(payload)


@app.post("/api/v34/training/{market}/bootstrap", status_code=202)
def start_v34_bootstrap(request: Request, market: str) -> object:
    try:
        return _json(
            _service(request, "v34_runtime").start_bootstrap(_v34_symbol(market))
        )
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error


@app.post("/api/v34/training/{market}/incremental", status_code=202)
def start_v34_incremental(request: Request, market: str) -> object:
    try:
        return _json(
            _service(request, "v34_runtime").start_incremental(_v34_symbol(market))
        )
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v34/training/runs/{run_id}")
def v34_training_run(request: Request, run_id: str) -> object:
    try:
        return _json(_service(request, "v34_runtime").training_status(run_id))
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v34/models/{market}/champion")
def v34_champion(request: Request, market: str) -> object:
    try:
        payload = _service(request, "v34_runtime").champion(_v34_symbol(market))
    except (V34RuntimeError, V34TrainingError) as error:
        raise PublicValidationError(str(error)) from error
    if payload is None:
        raise HTTPException(status_code=404, detail="No V3.4 champion is available")
    return _json(payload)


@app.get("/api/v341/model/status")
def v341_model_status(request: Request) -> object:
    return _json(_service(request, "v341_runtime").model_statuses())


@app.post("/api/v341/model/train", status_code=202)
def train_v341_model(request: Request, values: V33ModelActionRequest) -> object:
    try:
        market = _v34_symbol(values.instrument_code)
        runtime = _service(request, "v341_runtime")
        if runtime.champion(market) is None:
            return _json(runtime.start_bootstrap(market))
        return _json(runtime.start_incremental(market))
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error


@app.post("/api/v341/model/analysis")
def analyze_v341_model(request: Request, values: V33ModelActionRequest) -> object:
    try:
        return _json(
            _service(request, "v341_runtime").run_analysis(
                _v34_symbol(values.instrument_code)
            )
        )
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v341/iterations/{market}")
def v341_iterations(request: Request, market: str) -> object:
    try:
        return _json(_service(request, "v341_runtime").curve(_v34_symbol(market)))
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v341/forecast/{market}")
def latest_v341_forecast(request: Request, market: str) -> object:
    try:
        payload = _service(request, "v341_runtime").latest_forecast(
            _v34_symbol(market)
        )
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error
    if payload is None:
        raise HTTPException(status_code=404, detail="No V3.4.1 forecast is available")
    return _json(payload)


@app.post("/api/v341/training/{market}/bootstrap", status_code=202)
def start_v341_bootstrap(request: Request, market: str) -> object:
    try:
        return _json(
            _service(request, "v341_runtime").start_bootstrap(_v34_symbol(market))
        )
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error


@app.post("/api/v341/training/{market}/incremental", status_code=202)
def start_v341_incremental(request: Request, market: str) -> object:
    try:
        return _json(
            _service(request, "v341_runtime").start_incremental(_v34_symbol(market))
        )
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v341/training/runs/{run_id}")
def v341_training_run(request: Request, run_id: str) -> object:
    try:
        return _json(_service(request, "v341_runtime").training_status(run_id))
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v341/models/{market}/champion")
def v341_champion(request: Request, market: str) -> object:
    try:
        payload = _service(request, "v341_runtime").champion(_v34_symbol(market))
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error
    if payload is None:
        raise HTTPException(status_code=404, detail="No V3.4.1 champion is available")
    return _json(payload)


@app.post("/api/v342/policy/analysis")
def analyze_v342_policy(request: Request, values: V33ModelActionRequest) -> object:
    try:
        return _json(
            _service(request, "v342_policy").analyze(
                _v34_symbol(values.instrument_code)
            )
        )
    except V342PolicyError as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v342/policy/latest/{market}")
def latest_v342_policy(request: Request, market: str) -> object:
    try:
        payload = _service(request, "v342_policy").latest(_v34_symbol(market))
    except V342PolicyError as error:
        raise PublicValidationError(str(error)) from error
    if payload is None:
        raise HTTPException(status_code=404, detail="No V3.4.2 policy analysis is available")
    return _json(payload)


@app.post("/api/v342/policy/backtest")
def backtest_v342_policy(request: Request, values: V33ModelActionRequest) -> object:
    try:
        return _json(
            _service(request, "v342_policy").run_backtest(
                _v34_symbol(values.instrument_code)
            )
        )
    except V342PolicyError as error:
        raise PublicValidationError(str(error)) from error


@app.post("/api/v342/policy/backtest/live-oos")
def backtest_v342_policy_live_oos(
    request: Request, values: V33ModelActionRequest
) -> object:
    try:
        return _json(
            _service(request, "v342_policy").run_live_oos_backtest(
                _v34_symbol(values.instrument_code)
            )
        )
    except V342PolicyError as error:
        raise PublicValidationError(str(error)) from error


@app.post("/api/v342/policy/batches/evaluate")
def evaluate_v342_batches(request: Request, values: V33ModelActionRequest) -> object:
    try:
        return _json(
            _service(request, "v342_policy").evaluate_batches(
                _v34_symbol(values.instrument_code)
            )
        )
    except V342PolicyError as error:
        raise PublicValidationError(str(error)) from error


@app.post("/api/v343/model/analysis", status_code=202)
def analyze_v343_model(request: Request, values: V33ModelActionRequest) -> object:
    """Queue V3.4.1 inference without holding the HTTP request open."""

    try:
        market = _v34_symbol(values.instrument_code)
        return _json(_service(request, "v341_runtime").start_analysis(market))
    except V341RuntimeError as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v343/model/analysis/runs/{run_id}")
def v343_analysis_run(request: Request, run_id: str) -> object:
    """Poll a persisted analysis run and enrich only its completed result."""

    try:
        run = dict(_service(request, "v341_runtime").analysis_status(run_id))
        if run["status"] == "completed":
            market = str(run["market"])
            policy = _service(request, "v342_policy").analyze(market)
            run["result"] = _service(request, "v343_market_ui").enrich_forecast(
                market,
                run["result"],
                policy=policy,
            )
        return _json(run)
    except (V341RuntimeError, V342PolicyError, V343MarketUIError) as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v343/forecast/{market}")
def latest_v343_forecast(request: Request, market: str) -> object:
    """Return the latest frozen forecast with read-only chart history."""

    try:
        market = _v34_symbol(market)
        forecast = _service(request, "v341_runtime").latest_forecast(market)
        if forecast is None:
            raise HTTPException(
                status_code=404,
                detail="No V3.4.1 forecast is available",
            )
        policy = _service(request, "v342_policy").latest(market)
        return _json(
            _service(request, "v343_market_ui").enrich_forecast(
                market,
                forecast,
                policy=policy,
            )
        )
    except (V341RuntimeError, V342PolicyError, V343MarketUIError) as error:
        raise PublicValidationError(str(error)) from error


@app.get("/api/v343/iterations/{market}")
def v343_iterations(request: Request, market: str) -> object:
    """Expose V3.4.1 iteration maturity and actual-session deviations."""

    try:
        return _json(
            _service(request, "v343_market_ui").iteration_curve(
                _v34_symbol(market)
            )
        )
    except V343MarketUIError as error:
        raise PublicValidationError(str(error)) from error


@app.get("/")
def home(request: Request) -> Response:
    index = resource_root() / "frontend" / "dist" / "index.html"
    if not index.exists():
        return JSONResponse(status_code=503, content={"detail": "Frontend has not been built. Run build.bat or setup.bat."})
    return FileResponse(index)


@app.get("/{asset_path:path}")
def frontend_assets(request: Request, asset_path: str) -> Response:
    root = resource_root() / "frontend" / "dist"
    candidate = (root / asset_path).resolve()
    if root.exists() and candidate.is_file() and root.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = root / "index.html"
    if index.exists() and not asset_path.startswith("api/"):
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Not found")
