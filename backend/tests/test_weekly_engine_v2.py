from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.weekly_analysis.domain import DailyBar, PeriodBar
from backend.app.weekly_analysis.features import ValuationPoint
from backend.app.weekly_analysis.optimizer import seed_model


def _engine_api() -> Any:
    return importlib.import_module("backend.app.weekly_analysis.engine")


def _service_api() -> Any:
    return importlib.import_module(
        "backend.app.services.weekly_analysis_service"
    )


def _week_key(day: date) -> str:
    iso_year, iso_week, _weekday = day.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _weekdays(start: date, end: date) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return tuple(values)


def _synthetic_dataset(
    symbol: str,
    as_of: date,
    *,
    start: date = date(2015, 1, 5),
) -> Any:
    module = _engine_api()
    calendar_end = as_of + timedelta(weeks=14)
    expected = _weekdays(start, calendar_end)
    visible_sessions = tuple(day for day in expected if day <= as_of)
    daily: list[DailyBar] = []
    for index, day in enumerate(visible_sessions):
        close = Decimal("100") + Decimal(index) / Decimal("100")
        volume = None if symbol == "NDX" else Decimal("1000") + index
        daily.append(
            DailyBar(
                trade_date=day,
                open_price=close,
                high_price=close + Decimal("1"),
                low_price=close - Decimal("1"),
                close_price=close,
                volume=volume,
                source="SYNTHETIC_DIRECT_INDEX",
            )
        )

    expected_by_week: dict[str, list[date]] = defaultdict(list)
    actual_by_week: dict[str, list[DailyBar]] = defaultdict(list)
    for day in expected:
        expected_by_week[_week_key(day)].append(day)
    for row in daily:
        actual_by_week[_week_key(row.trade_date)].append(row)
    complete_keys = tuple(
        sorted(
            key
            for key, sessions in expected_by_week.items()
            if sessions[-1] <= as_of
            and [row.trade_date for row in actual_by_week.get(key, ())]
            == sessions
        )
    )

    weekly: list[PeriodBar] = []
    valuations: list[ValuationPoint] = []
    for key in complete_keys:
        rows = actual_by_week[key]
        volume = (
            None
            if symbol == "NDX"
            else sum((row.volume for row in rows), Decimal())  # type: ignore[arg-type]
        )
        weekly.append(
            PeriodBar(
                timeframe="weekly",
                period_start=rows[0].trade_date,
                period_end=rows[-1].trade_date,
                open_price=rows[0].open_price,
                high_price=max(row.high_price for row in rows),  # type: ignore[type-var]
                low_price=min(row.low_price for row in rows),  # type: ignore[type-var]
                close_price=rows[-1].close_price,
                volume=volume,
                source="AGGREGATED_DAILY_VOLUME",
                volume_source=(
                    "VOLUME_UNAVAILABLE:DIRECT_INDEX"
                    if symbol == "NDX"
                    else "AGGREGATED_DAILY_VOLUME"
                ),
                price_source="SYNTHETIC_DIRECT_INDEX",
            )
        )
        valuations.append(
            ValuationPoint(
                valuation_date=rows[-1].trade_date,
                value=Decimal(20 + len(weekly) % 25),
            )
        )

    return module.VerifiedDataset(
        instrument_code=symbol,
        daily_bars=tuple(daily),
        completed_weekly_bars=tuple(weekly),
        valuations=tuple(valuations),
        trading_calendar=module.TradingCalendar(
            instrument_code=symbol,
            expected_trade_dates=expected,
            coverage_start=expected[0],
            coverage_end=expected[-1],
        ),
        complete_week_keys=complete_keys,
        current_position=50,
    )


class _Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date, int, int]] = []

    def load_verified_window(
        self,
        symbol: str,
        *,
        as_of: date,
        years: int,
        warmup_weeks: int,
    ) -> Any:
        self.calls.append((symbol, as_of, years, warmup_weeks))
        return _synthetic_dataset(symbol, as_of)


class _Repository:
    def __init__(self, *, growth_state=None, ndx_state=None) -> None:
        self.states = {
            "399006": growth_state or seed_model("399006"),
            "NDX": ndx_state or seed_model("NDX"),
        }
        self.committed: dict[str, list[str]] = defaultdict(list)
        self.advice_calls: dict[str, int] = defaultdict(int)

    def list_completed_weeks(self, symbol: str) -> set[str]:
        return set(self.states[symbol].week_records)

    def load_work_state(self, symbol: str):
        return self.states[symbol]

    def commit_iteration(self, symbol, iteration, work_state, labels) -> None:
        assert work_state.audits[-1] == iteration
        assert tuple(labels) == work_state.feedback
        self.states[symbol] = work_state
        self.committed[symbol].append(iteration.week_key)

    def save_advice(
        self, symbol: str, advice, *, generation_key=None
    ) -> int:
        assert advice.instrument_code == symbol
        self.advice_calls[symbol] += 1
        return self.advice_calls[symbol]


class _JudgmentPolicy:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.cutoffs: list[tuple[date, date | None]] = []
        self.positions: list[int | None] = []

    def judge(self, *, model, features, sample, current_position):
        module = _engine_api()
        self.calls += 1
        self.positions.append(current_position)
        self.cutoffs.append(
            (sample.sample_date, features.source_data_max_date)
        )
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic interruption")
        ma20 = features.features["ma_20"]
        ma60 = features.features["ma_60"]
        direction = "up" if ma20 is not None and ma60 is not None and ma20 >= ma60 else "down"
        return module.WeeklyJudgment(
            direction=direction,
            probability=Decimal("70"),
            confidence=Decimal("70"),
            target_position=70 if direction == "up" else 30,
        )


@pytest.fixture(scope="module")
def _baseline() -> tuple[Any, Any, Any, Any, Any]:
    module = _engine_api()
    provider = _Provider()
    repository = _Repository()
    policy = _JudgmentPolicy()
    as_of = date(2026, 7, 30)
    expected_dataset = _synthetic_dataset("399006", as_of)
    boundary = date(2016, 7, 30)
    expected = module.build_week_samples(
        (bar.trade_date for bar in expected_dataset.daily_bars),
        symbol="399006",
        complete_week_keys=expected_dataset.complete_week_keys,
    )
    expected = tuple(
        sample
        for sample in expected
        if sample.candidate_dates[0] >= boundary
    )

    engine = module.WeeklyAnalysisEngine(
        provider=provider,
        repository=repository,
        judgment_policy=policy,
    )
    result = engine.run("399006", as_of=as_of)
    return result, provider, repository, tuple(expected), policy


def test_engine_runs_once_per_actual_complete_week_in_the_ten_year_window(
    _baseline,
) -> None:
    result, provider, repository, expected, policy = _baseline
    as_of = date(2026, 7, 30)

    assert 515 <= len(expected) <= 525
    assert result.total_weeks == len(expected)
    assert result.processed_weeks == len(expected)
    assert repository.committed["399006"] == [
        sample.week_key for sample in expected
    ]
    assert provider.calls == [("399006", as_of, 10, 60)]
    assert all(
        source_max is None or source_max < sample_date
        for sample_date, source_max in policy.cutoffs
    )


def test_same_latest_complete_week_only_regenerates_advice(
    _baseline,
) -> None:
    module = _engine_api()
    provider = _Provider()
    _result, _provider, baseline_repository, _expected, _policy = (
        _baseline
    )
    repository = _Repository(
        growth_state=baseline_repository.states["399006"]
    )
    policy = _JudgmentPolicy()
    engine = module.WeeklyAnalysisEngine(
        provider=provider,
        repository=repository,
        judgment_policy=policy,
    )

    first = engine.run("399006", as_of=date(2026, 7, 30))
    before = repository.states["399006"]
    second = engine.run("399006", as_of=date(2026, 7, 30))

    assert second.processed_weeks == 0
    assert second.skipped_weeks == second.total_weeks
    assert repository.states["399006"] == before
    assert (
        second.iteration_id,
        second.work_version,
        second.model_version,
    ) == (
        first.iteration_id,
        first.work_version,
        first.model_version,
    )
    assert first.advice_generation == 1
    assert second.advice_generation == 2
    assert repository.advice_calls["399006"] == 2


def test_historical_step_uses_neutral_position_and_final_advice_uses_capture(
    _baseline,
) -> None:
    module = _engine_api()
    _result, _provider, baseline_repository, _expected, _policy = _baseline
    dataset = replace(
        _synthetic_dataset("399006", date(2026, 8, 2)),
        current_position=None,
    )

    class CapturedProvider:
        def load_verified_window(self, *_args, **_kwargs):
            return dataset

    repository = _Repository(
        growth_state=baseline_repository.states["399006"]
    )
    policy = _JudgmentPolicy()
    engine = module.WeeklyAnalysisEngine(
        provider=CapturedProvider(),
        repository=repository,
        judgment_policy=policy,
    )
    request = module.AnalysisRequest(
        latest_complete_week="2026-W31",
        model_line=module.MODEL_LINE,
        current_position=85,
        position_snapshot_key="position-85",
    )

    result = engine.run(
        "399006",
        as_of=date(2026, 8, 2),
        analysis_request=request,
    )

    assert result.processed_weeks == 1
    assert policy.positions == [50, 85]
    assert result.advice.current_position == 85


def test_incomplete_week_is_deferred_and_one_new_complete_week_appends_once(
    _baseline,
) -> None:
    module = _engine_api()
    provider = _Provider()
    _result, _provider, baseline_repository, _expected, _policy = _baseline
    repository = _Repository(
        growth_state=baseline_repository.states["399006"]
    )
    engine = module.WeeklyAnalysisEngine(
        provider=provider,
        repository=repository,
        judgment_policy=_JudgmentPolicy(),
    )

    assert "2026-W31" not in repository.states["399006"].week_records

    after = engine.run("399006", as_of=date(2026, 8, 2))
    repeated = engine.run("399006", as_of=date(2026, 8, 2))

    assert after.latest_complete_week == "2026-W31"
    assert after.processed_weeks == 1
    assert repeated.processed_weeks == 0
    assert repository.committed["399006"].count("2026-W31") == 1


def test_rolling_provider_window_ignores_persisted_weeks_that_aged_out(
    _baseline,
) -> None:
    module = _engine_api()
    _result, _provider, baseline_repository, _expected, _policy = _baseline
    persisted = baseline_repository.states["399006"]
    dataset = _synthetic_dataset("399006", date(2026, 8, 9))
    aged_out_key = next(iter(persisted.week_records))
    aged_out_bar = next(
        row
        for row in dataset.completed_weekly_bars
        if _week_key(row.period_end) == aged_out_key
    )
    rolled = replace(
        dataset,
        completed_weekly_bars=tuple(
            row
            for row in dataset.completed_weekly_bars
            if row is not aged_out_bar
        ),
        complete_week_keys=tuple(
            key for key in dataset.complete_week_keys if key != aged_out_key
        ),
    )

    class RollingProvider:
        def load_verified_window(self, *_args, **_kwargs):
            return rolled

    repository = _Repository(growth_state=persisted)
    engine = module.WeeklyAnalysisEngine(
        provider=RollingProvider(),
        repository=repository,
        judgment_policy=_JudgmentPolicy(),
    )
    before = len(persisted.audits)
    result = engine.run("399006", as_of=date(2026, 8, 9))

    assert result.processed_weeks == 2
    assert len(repository.states["399006"].audits) == before + 2
    assert repository.states["399006"].iteration_id == f"I{before + 2:04d}"


def test_valuation_coverage_estimates_only_isolated_two_session_gaps() -> None:
    service = _service_api()
    sessions = _weekdays(date(2026, 1, 5), date(2026, 1, 16))
    valuations = tuple(
        ValuationPoint(day, Decimal(index + 10))
        for index, day in enumerate(sessions)
        if day not in {sessions[2], sessions[6]}
    )

    checked = service._quality_checked_valuations(
        valuations,
        expected_sessions=sessions,
        analysis_start=sessions[0],
        analysis_end=sessions[-1],
    )

    assert tuple(point.valuation_date for point in checked) == sessions
    assert {
        point.valuation_date for point in checked if point.estimated
    } == {sessions[2], sessions[6]}


@pytest.mark.parametrize("gap_kind", ("internal", "terminal", "ancient_only"))
def test_valuation_coverage_blocks_long_or_stale_gaps(
    gap_kind: str,
) -> None:
    service = _service_api()
    sessions = _weekdays(date(2026, 1, 5), date(2026, 1, 23))
    if gap_kind == "internal":
        missing = set(sessions[4:7])
        valuations = tuple(
            ValuationPoint(day, Decimal("20"))
            for day in sessions
            if day not in missing
        )
    elif gap_kind == "terminal":
        valuations = tuple(
            ValuationPoint(day, Decimal("20")) for day in sessions[:-3]
        )
    else:
        valuations = tuple(
            ValuationPoint(
                date(2016, 1, 4) + timedelta(days=index),
                Decimal("20"),
            )
            for index in range(20)
        )

    with pytest.raises(
        service.DatasetQualityError,
        match="valuation.*(gap|stale|coverage)|valuation history",
    ):
        service._quality_checked_valuations(
            valuations,
            expected_sessions=sessions,
            analysis_start=sessions[0],
            analysis_end=sessions[-1],
        )


@pytest.mark.parametrize(
    "source",
    ("", "   ", "UNVERIFIED_VENDOR", "ETF_PROXY", "AKSHARE_FUND"),
)
def test_valuation_source_requires_explicit_direct_index_whitelist(
    source: str,
) -> None:
    service = _service_api()
    row = SimpleNamespace(
        valuation_date=date(2026, 1, 5),
        pe_ratio=Decimal("20"),
        pb_ratio=None,
        raw_values={"source": source, "proxy": False},
    )

    with pytest.raises(
        service.DatasetQualityError,
        match="valuation.*source|direct index|proxy|fund",
    ):
        service._valuation_point(row)


@pytest.mark.parametrize(
    "source",
    (
        "AKSHARE_CSINDEX",
        "OFFICIAL_DIRECT_INDEX",
        "VERIFIED_DIRECT_INDEX_CACHE",
    ),
)
def test_valuation_source_accepts_declared_direct_index_whitelist(
    source: str,
) -> None:
    service = _service_api()
    row = SimpleNamespace(
        valuation_date=date(2026, 1, 5),
        pe_ratio=Decimal("20"),
        pb_ratio=None,
        raw_values={"source": source, "proxy": False},
    )

    assert service._valuation_point(row) == ValuationPoint(
        date(2026, 1, 5),
        Decimal("20"),
    )


def test_engine_emit_never_swallows_lost_task_claim() -> None:
    engine = _engine_api()
    tasks = importlib.import_module("backend.app.weekly_analysis.tasks")
    update = engine.ProgressUpdate(
        stage="iterating",
        completed_weeks=0,
        total_weeks=1,
        iteration_id="I0000",
        work_version="W0000",
        model_version="M0001",
        last_checkpoint=None,
        elapsed_seconds=0.0,
    )

    def lost_claim(_update) -> None:
        raise tasks._TaskClaimLost("claim lost")

    with pytest.raises(tasks._TaskClaimLost, match="claim lost"):
        engine.WeeklyAnalysisEngine._emit(lost_claim, update)


def test_interruption_resumes_after_the_last_committed_week(
    _baseline,
) -> None:
    module = _engine_api()
    provider = _Provider()
    _result, _provider, baseline_repository, _expected, _policy = _baseline
    repository = _Repository(
        growth_state=baseline_repository.states["399006"]
    )
    failing = _JudgmentPolicy(fail_after=2)
    engine = module.WeeklyAnalysisEngine(
        provider=provider,
        repository=repository,
        judgment_policy=failing,
    )

    with pytest.raises(RuntimeError, match="interruption"):
        engine.run("399006", as_of=date(2026, 8, 16))
    committed_before_resume = tuple(repository.committed["399006"])
    assert committed_before_resume == ("2026-W31", "2026-W32")

    engine = module.WeeklyAnalysisEngine(
        provider=provider,
        repository=repository,
        judgment_policy=_JudgmentPolicy(),
    )
    result = engine.run("399006", as_of=date(2026, 8, 16))

    assert result.processed_weeks == 1
    assert tuple(repository.committed["399006"][:2]) == committed_before_resume
    assert len(repository.committed["399006"]) == len(
        set(repository.committed["399006"])
    )


def test_markets_keep_independent_state_and_invalid_identity_is_rejected() -> None:
    module = _engine_api()
    repository = _Repository()
    engine = module.WeeklyAnalysisEngine(
        provider=_Provider(),
        repository=repository,
        judgment_policy=_JudgmentPolicy(),
    )

    assert repository.states["399006"].instrument_code == "399006"
    assert repository.states["NDX"].instrument_code == "NDX"
    ndx = _synthetic_dataset("NDX", date(2026, 7, 30))
    sample = engine._analysis_samples(
        ndx, "NDX", date(2026, 7, 30)
    )[0]
    snapshot = engine._features_for_sample(ndx, sample)
    assert snapshot.quality_report.is_publishable
    assert (
        snapshot.quality_report.volume_availability
        == "not_available_for_direct_index"
    )
    with pytest.raises(ValueError, match="399006|NDX|supported"):
        engine.run("000688", as_of=date(2026, 7, 30))


def test_production_service_does_not_expose_as_of_and_missing_scorer_fails() -> None:
    service_module = _service_api()

    assert "as_of" not in inspect.signature(
        service_module.WeeklyAnalysisService.start
    ).parameters
    with pytest.raises(
        service_module.WeeklyAnalysisUnavailable,
        match="audited.*scorer|scorer.*available",
    ):
        service_module.UnavailableJudgmentPolicy().judge(
            model=seed_model("399006"),
            features=None,
            sample=None,
            current_position=None,
        )
