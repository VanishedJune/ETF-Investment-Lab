from __future__ import annotations

from datetime import date, timedelta
import math
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    V33Forecast,
    V33ModelVersion,
    V34Forecast,
    V34ModelVersion,
    V34TrainingIteration,
)
from backend.app.services.v33_feature_service import FeatureSnapshot
from backend.app.services.v34_runtime_service import V34RuntimeService
from backend.app.services.v34_training_service import (
    V34TrainingService,
    build_training_samples,
    eligible_fully_matured,
)
from backend.web import (
    V33ModelActionRequest,
    analyze_v34_model,
    latest_v34_forecast,
    train_v34_model,
    v34_iterations,
    v34_model_status,
)


class _Calendar:
    def sessions(self, _instrument_code: str, start_date: date, end_date: date):
        current = start_date
        output = []
        while current <= end_date:
            if current.weekday() < 5:
                output.append(current)
            current += timedelta(days=1)
        return output


def _snapshot(index: int) -> FeatureSnapshot:
    anchor = date(2024, 1, 5) + timedelta(days=7 * index)
    base = 100.0 * math.exp(0.001 * index + 0.02 * math.sin(index / 5.0))
    sequence = []
    for offset in range(99, -1, -1):
        day = anchor - timedelta(days=offset)
        close = base * math.exp(-0.0003 * offset)
        sequence.append(
            {
                "trade_date": day.isoformat(),
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0 * (1 + 0.1 * math.sin((index - offset) / 4)),
                "turnover": close * 1_000_000,
            }
        )
    return FeatureSnapshot(
        market="399006",
        cutoff_date=anchor,
        source_data_max_date=anchor,
        daily_as_of=anchor,
        weekly_as_of=anchor,
        weekly={
            "weekly_dif": math.sin(index / 5) / 1000,
            "weekly_dea": math.sin((index - 1) / 5) / 1000,
            "weekly_macd_histogram": math.cos(index / 5) / 2000,
            "weekly_dif_slope_1": math.cos(index / 5) / 5000,
            "weekly_return_13": math.sin(index / 8) / 30,
        },
        daily={
            "daily_dif": math.sin(index / 3) / 1000,
            "daily_dea": math.sin((index - 1) / 3) / 1000,
            "daily_macd_histogram": math.cos(index / 3) / 2000,
            "daily_dif_slope_3": math.cos(index / 3) / 5000,
            "daily_dea_slope_3": math.cos((index - 1) / 3) / 6000,
            "daily_histogram_speed_3": math.sin(index / 3) / 7000,
            "daily_return_20": math.sin(index / 8) / 50,
        },
        daily_sequence=tuple(sequence),
        missing_masks={},
        provenance={"target": "399006"},
        derivative_turn={},
    )


def _runtime(tmp_path: Path) -> V34RuntimeService:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    return V34RuntimeService(
        create_session_factory(database),
        calendar=_Calendar(),
        current_position_provider=lambda _market: 0,
        scenario_count=1000,
    )


def test_progressive_runtime_persists_13w_chain_and_preserves_v33(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(75))
    samples = build_training_samples(snapshots)
    formal = samples[45:]
    seed_rows = eligible_fully_matured(samples, formal[0].anchor_date)
    seed = V34TrainingService().fit(
        "399006",
        seed_rows,
        version="CYB_HYBRID_13W_V3.4.0",
        parent_version=None,
        trained_through=formal[0].anchor_date,
        effective_from=formal[0].anchor_date,
        alpha=5.0,
    )
    seed_id = runtime._persist_model(seed, status="champion")
    run_id = runtime._create_run("399006", "bootstrap", formal[-1].anchor_date)
    with runtime.sessions() as session:
        v33_before = (
            session.scalar(select(func.count()).select_from(V33ModelVersion)),
            session.scalar(select(func.count()).select_from(V33Forecast)),
        )
    result = runtime._progressive_run(run_id, "399006", samples, formal, seed, seed_id)
    assert result["status"] == "completed"
    with runtime.sessions() as session:
        iterations = list(
            session.scalars(
                select(V34TrainingIteration).order_by(
                    V34TrainingIteration.weekly_iteration_number
                )
            )
        )
        forecasts = list(session.scalars(select(V34Forecast)))
        models = {row.id: row for row in session.scalars(select(V34ModelVersion))}
        v33_after = (
            session.scalar(select(func.count()).select_from(V33ModelVersion)),
            session.scalar(select(func.count()).select_from(V33Forecast)),
        )
    assert len(iterations) == len(formal)
    assert len(forecasts) == len(formal)
    assert all(row.horizon_weeks == 13 and row.scenario_count >= 1000 for row in forecasts)
    assert all(len(row.representative_ohlcv_json) == 13 for row in forecasts)
    assert v33_before == v33_after
    for iteration, forecast in zip(iterations, forecasts):
        assert models[forecast.model_version_id].version == iteration.champion_before_version
        if iteration.promoted:
            assert iteration.champion_after_version != iteration.champion_before_version
    before = runtime._training_counts("399006")
    runtime._complete_anchors = lambda _market: tuple(item.anchor_date for item in samples)  # type: ignore[method-assign]
    payload = runtime.run_analysis("399006")
    after = runtime._training_counts("399006")
    assert payload["forecast_horizon_weeks"] == 13
    assert before == after
    runtime.shutdown()


class _ApiRuntime:
    def __init__(self, bootstrapped: bool) -> None:
        self.bootstrapped = bootstrapped
        self.calls = []

    def champion(self, market):
        self.calls.append(("champion", market))
        return {"version": "V3.4"} if self.bootstrapped else None

    def start_bootstrap(self, market):
        self.calls.append(("bootstrap", market))
        return {"run_id": "bootstrap", "market": market}

    def start_incremental(self, market):
        self.calls.append(("incremental", market))
        return {"run_id": "incremental", "market": market}

    def run_analysis(self, market):
        self.calls.append(("analysis", market))
        return {"market": market, "forecast_horizon_weeks": 13}

    def curve(self, market):
        return {"market": market, "points": []}

    def latest_forecast(self, market):
        return {"market": market, "forecast_horizon_weeks": 13}

    def model_statuses(self):
        return [{"market": "399006"}, {"market": "159941"}]


def _request(runtime):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(v34_runtime=runtime)))


def test_v34_web_contract_separates_training_and_analysis() -> None:
    fresh = _ApiRuntime(False)
    assert train_v34_model(
        _request(fresh), V33ModelActionRequest(instrument_code="159941")
    )["run_id"] == "bootstrap"
    trained = _ApiRuntime(True)
    assert train_v34_model(
        _request(trained), V33ModelActionRequest(instrument_code="399006")
    )["run_id"] == "incremental"
    before = list(trained.calls)
    assert analyze_v34_model(
        _request(trained), V33ModelActionRequest(instrument_code="399006")
    )["forecast_horizon_weeks"] == 13
    assert trained.calls[len(before):] == [("analysis", "399006")]
    assert len(v34_model_status(_request(trained))) == 2
    assert v34_iterations(_request(trained), "159941")["market"] == "159941"
    assert latest_v34_forecast(_request(trained), "399006")["forecast_horizon_weeks"] == 13
