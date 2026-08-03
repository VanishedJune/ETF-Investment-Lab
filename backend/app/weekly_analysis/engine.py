"""Leakage-safe ten-year weekly baseline and incremental execution engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from threading import Lock
import time
from typing import Callable, Iterable, Mapping, Protocol

from .advice import Advice, build_advice
from .domain import DailyBar, PeriodBar
from .features import FeatureSnapshot, ValuationPoint, build_feature_snapshot
from .optimizer import (
    DailyClose,
    IterationLabel,
    IterationPrediction,
    OptimizerStepResult,
    TradingCalendar,
    WindowMetrics,
    WorkState,
    rolling_metrics,
    run_optimizer_step,
    update_labels,
)
from .repository import (
    DuplicateAdvice,
    TaskClaimLost,
    TaskExecutionGuard,
)
from .sampling import WeekSample, build_week_samples


SUPPORTED_MARKETS = frozenset({"399006", "NDX"})
MODEL_LINE = "weekly-v2"
ANALYSIS_YEARS = 10
WARMUP_WEEKS = 60
HISTORICAL_EVALUATION_POSITION = 50
_PROHIBITED_SOURCE_MARKERS = ("ETF", "FUND", "QQQ", "159941")


class DatasetQualityError(ValueError):
    """Verified source data cannot satisfy the V2 leakage/identity contract."""


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """One immutable service-captured task input identity."""

    latest_complete_week: str
    model_line: str
    current_position: int | None
    position_snapshot_key: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.latest_complete_week, str)
            or not self.latest_complete_week.strip()
        ):
            raise ValueError("latest_complete_week is required")
        if not isinstance(self.model_line, str) or not self.model_line.strip():
            raise ValueError("model_line is required")
        if self.current_position is not None and (
            type(self.current_position) is not int
            or not 0 <= self.current_position <= 100
        ):
            raise ValueError(
                "current_position must be an integer percent from 0 to 100"
            )
        if (
            not isinstance(self.position_snapshot_key, str)
            or not self.position_snapshot_key.strip()
            or len(self.position_snapshot_key) > 128
        ):
            raise ValueError(
                "position_snapshot_key must contain 1..128 characters"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "latest_complete_week": self.latest_complete_week,
            "model_line": self.model_line,
            "current_position": self.current_position,
            "position_snapshot_key": self.position_snapshot_key,
        }


@dataclass(frozen=True, slots=True)
class VerifiedDataset:
    """Read-only direct-index inputs plus an explicit exchange calendar."""

    instrument_code: str
    daily_bars: tuple[DailyBar, ...]
    completed_weekly_bars: tuple[PeriodBar, ...]
    valuations: tuple[ValuationPoint, ...]
    trading_calendar: TradingCalendar
    complete_week_keys: tuple[str, ...]
    current_position: int | None = None

    def __post_init__(self) -> None:
        if self.instrument_code not in SUPPORTED_MARKETS:
            raise DatasetQualityError("Only 399006 and NDX are supported")
        if self.trading_calendar.instrument_code != self.instrument_code:
            raise DatasetQualityError(
                "Verified dataset and trading calendar markets differ"
            )
        daily = tuple(self.daily_bars)
        weekly = tuple(self.completed_weekly_bars)
        valuations = tuple(self.valuations)
        complete_keys = tuple(self.complete_week_keys)
        if len({row.trade_date for row in daily}) != len(daily):
            raise DatasetQualityError("Daily source contains duplicate dates")
        if any(
            left.trade_date >= right.trade_date
            for left, right in zip(daily, daily[1:])
        ):
            raise DatasetQualityError(
                "Daily source dates must be strictly increasing"
            )
        if any(
            row.close_price is None
            or row.close_price <= 0
            or _source_is_prohibited(row.source)
            for row in daily
        ):
            raise DatasetQualityError(
                "Daily direct-index data failed price/source quality gates"
            )
        if any(
            row.timeframe != "weekly"
            or row.close_price is None
            or row.close_price <= 0
            or _source_is_prohibited(row.price_source)
            for row in weekly
        ):
            raise DatasetQualityError(
                "Completed weekly data failed price/source quality gates"
            )
        weekly_keys = tuple(_week_key(row.period_end) for row in weekly)
        if len(set(weekly_keys)) != len(weekly_keys):
            raise DatasetQualityError("Completed weekly keys are duplicated")
        if set(weekly_keys) != set(complete_keys):
            raise DatasetQualityError(
                "Completed weekly bars and completion keys disagree"
            )
        if complete_keys != tuple(sorted(set(complete_keys))):
            raise DatasetQualityError(
                "complete_week_keys must be sorted and unique"
            )
        if any(
            left.valuation_date >= right.valuation_date
            for left, right in zip(valuations, valuations[1:])
        ):
            raise DatasetQualityError(
                "Valuation dates must be strictly increasing"
            )
        if self.current_position is not None and (
            type(self.current_position) is not int
            or not 0 <= self.current_position <= 100
        ):
            raise DatasetQualityError(
                "current_position must be an integer percent from 0 to 100"
            )
        object.__setattr__(self, "daily_bars", daily)
        object.__setattr__(self, "completed_weekly_bars", weekly)
        object.__setattr__(self, "valuations", valuations)
        object.__setattr__(self, "complete_week_keys", complete_keys)


class VerifiedDatasetProvider(Protocol):
    """Narrow adapter boundary; implementations must not refresh the network."""

    def load_verified_window(
        self,
        symbol: str,
        *,
        as_of: date,
        years: int,
        warmup_weeks: int,
    ) -> VerifiedDataset: ...


@dataclass(frozen=True, slots=True)
class WeeklyJudgment:
    """Audited policy output used to construct Task 5 prediction inputs."""

    direction: str
    probability: Decimal
    confidence: Decimal
    target_position: int
    direction_probabilities: Mapping[str, Decimal] | None = None
    confirmation_conditions_met: bool = False

    def __post_init__(self) -> None:
        if self.direction not in {"up", "down", "neutral"}:
            raise ValueError("direction must be up, down, or neutral")
        probability = Decimal(self.probability)
        confidence = Decimal(self.confidence)
        if not Decimal() <= probability <= Decimal("100"):
            raise ValueError("probability must be in the 0..100 scale")
        if not Decimal() <= confidence <= Decimal("100"):
            raise ValueError("confidence must be in the 0..100 scale")
        if (
            type(self.target_position) is not int
            or not 0 <= self.target_position <= 100
        ):
            raise ValueError("target_position must be an integer percent")
        if type(self.confirmation_conditions_met) is not bool:
            raise ValueError(
                "confirmation_conditions_met must be a boolean"
            )
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "confidence", confidence)


class JudgmentPolicy(Protocol):
    """Explicit scorer adapter; no implicit/random signal substitution."""

    def judge(
        self,
        *,
        model: WorkState,
        features: FeatureSnapshot,
        sample: WeekSample,
        current_position: int | None,
    ) -> WeeklyJudgment: ...


class EngineRepository(Protocol):
    def list_completed_weeks(self, symbol: str) -> set[str]: ...

    def load_work_state(self, symbol: str) -> WorkState: ...

    def commit_iteration(
        self,
        symbol: str,
        iteration: object,
        work_state: WorkState,
        labels: Iterable[IterationLabel],
        *,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> None: ...

    def save_advice(
        self,
        symbol: str,
        advice: Advice,
        *,
        generation_key: str | None = None,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    stage: str
    completed_weeks: int
    total_weeks: int
    iteration_id: str
    work_version: str
    model_version: str
    last_checkpoint: str | None
    elapsed_seconds: float
    failure_reason: str | None = None
    recoverable_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "completed_weeks": self.completed_weeks,
            "total_weeks": self.total_weeks,
            "iteration_id": self.iteration_id,
            "work_version": self.work_version,
            "model_version": self.model_version,
            "last_checkpoint": self.last_checkpoint,
            "elapsed_seconds": self.elapsed_seconds,
            "failure_reason": self.failure_reason,
            "recoverable_reason": self.recoverable_reason,
        }


ProgressCallback = Callable[[ProgressUpdate], object]


@dataclass(frozen=True, slots=True)
class EngineRunResult:
    symbol: str
    latest_complete_week: str
    total_weeks: int
    processed_weeks: int
    skipped_weeks: int
    iteration_id: str
    work_version: str
    model_version: str
    advice: Advice
    advice_generation: int
    advice_persisted: bool


@dataclass(frozen=True, slots=True)
class StepInputs:
    """Concrete, inspectable inputs passed through Task 5 for one week."""

    sample: WeekSample
    features: FeatureSnapshot
    judgment: WeeklyJudgment
    advice: Advice
    prediction: IterationPrediction
    feedback: tuple[IterationLabel, ...]


class WeeklyAnalysisEngine:
    """Run the initial ten-year chain, then append each complete week once."""

    def __init__(
        self,
        *,
        provider: VerifiedDatasetProvider,
        repository: EngineRepository,
        judgment_policy: JudgmentPolicy,
    ) -> None:
        self.provider = provider
        self.repository = repository
        self.judgment_policy = judgment_policy
        self._advice_generation = {symbol: 0 for symbol in SUPPORTED_MARKETS}
        self._generation_lock = Lock()

    def run(
        self,
        symbol: str,
        as_of: date | None = None,
        progress_callback: ProgressCallback | None = None,
        advice_request_key: str | None = None,
        analysis_request: AnalysisRequest | None = None,
        execution_guard: TaskExecutionGuard | None = None,
    ) -> EngineRunResult:
        """Execute one market; ``as_of`` is reserved for tests/internal calls."""

        symbol = _validate_symbol(symbol)
        analysis_date = as_of or date.today()
        if not isinstance(analysis_date, date):
            raise TypeError("as_of must be a date")
        if analysis_request is not None and not isinstance(
            analysis_request, AnalysisRequest
        ):
            raise TypeError("analysis_request must be an AnalysisRequest")
        if execution_guard is not None and not isinstance(
            execution_guard, TaskExecutionGuard
        ):
            raise TypeError("execution_guard must be a TaskExecutionGuard")
        started = time.monotonic()
        initial_state = self.repository.load_work_state(symbol)
        self._emit(
            progress_callback,
            _progress(
                "preparing_data",
                initial_state,
                total=max(len(initial_state.audits), 0),
                started=started,
            ),
        )
        dataset = self.provider.load_verified_window(
            symbol,
            as_of=analysis_date,
            years=ANALYSIS_YEARS,
            warmup_weeks=WARMUP_WEEKS,
        )
        self._validate_dataset(dataset, symbol, analysis_date)
        samples = self._analysis_samples(dataset, symbol, analysis_date)
        if not samples:
            raise DatasetQualityError(
                "No complete actual trading week exists in the ten-year window"
            )
        samples_by_week = {sample.week_key: sample for sample in samples}
        feature_cache: dict[str, FeatureSnapshot] = {}
        if analysis_request is not None and (
            analysis_request.latest_complete_week != samples[-1].week_key
            or analysis_request.model_line != MODEL_LINE
        ):
            raise DatasetQualityError(
                "captured analysis request no longer matches verified data"
            )

        # ``initial_state`` is the repository's validated head object. Reading
        # completed weeks through another repository load would replace that
        # trusted object and make the first incremental commit fail identity
        # validation, especially on a fresh W0000 baseline.
        completed_at_start = set(initial_state.week_records)
        target_keys = {sample.week_key for sample in samples}
        unknown_completed = (completed_at_start & target_keys) - {
            _week_key(row.period_end)
            for row in dataset.completed_weekly_bars
        }
        if unknown_completed:
            raise DatasetQualityError(
                "Persisted completed weeks are absent from verified data: "
                + ",".join(sorted(unknown_completed))
            )
        pending = tuple(
            sample
            for sample in samples
            if sample.week_key not in completed_at_start
        )
        skipped = len(samples) - len(pending)
        absolute_total = len(initial_state.audits) + len(pending)
        daily_closes = _daily_closes(dataset)
        self._emit(
            progress_callback,
            _progress(
                "building_features",
                initial_state,
                total=absolute_total,
                started=started,
            ),
        )

        processed = 0
        latest_features: FeatureSnapshot | None = None
        latest_sample = samples[-1]
        state = initial_state
        if pending:
            self._emit(
                progress_callback,
                _progress(
                    "iterating",
                    state,
                    total=absolute_total,
                    started=started,
                ),
            )
        for sample in samples:
            if sample.week_key in completed_at_start:
                if sample == latest_sample:
                    latest_features = self._features_for_sample(
                        dataset, sample
                    )
                continue
            step = self._run_one_week(
                dataset,
                state,
                sample,
                daily_closes=daily_closes,
                samples_by_week=samples_by_week,
                feature_cache=feature_cache,
            )
            if processed == 0:
                self._emit(
                    progress_callback,
                    _progress(
                        "validating",
                        state,
                        total=absolute_total,
                        started=started,
                        last_checkpoint=sample.week_key,
                    ),
                )
            incremental_commit = getattr(
                self.repository, "commit_iteration_incremental", None
            )
            if callable(incremental_commit):
                commit_kwargs = (
                    {}
                    if execution_guard is None
                    else {"execution_guard": execution_guard}
                )
                incremental_commit(
                    symbol,
                    state,
                    step.optimizer_result,
                    step.inputs.feedback,
                    **commit_kwargs,
                )
            else:
                self.repository.commit_iteration(
                    symbol,
                    step.optimizer_result.audit,
                    step.optimizer_result.work_state,
                    step.inputs.feedback,
                    **(
                        {}
                        if execution_guard is None
                        else {"execution_guard": execution_guard}
                    ),
                )
            processed += 1
            latest_features = step.inputs.features
            committed_state = step.optimizer_result.work_state
            state = committed_state

        final_state = state
        if latest_features is None:
            latest_features = self._features_for_sample(
                dataset, latest_sample
            )
        self._emit(
            progress_callback,
            _progress(
                "generating_advice",
                final_state,
                total=max(absolute_total, len(final_state.audits)),
                started=started,
            ),
        )
        captured_position = (
            dataset.current_position
            if analysis_request is None
            else analysis_request.current_position
        )
        final_judgment = self.judgment_policy.judge(
            model=final_state,
            features=latest_features,
            sample=latest_sample,
            current_position=captured_position,
        )
        final_advice = _build_advice_from_judgment(
            model=final_state,
            features=latest_features,
            calendar=dataset.trading_calendar,
            current_position=captured_position,
            judgment=final_judgment,
        )
        advice_persisted = True
        try:
            self.repository.save_advice(
                symbol,
                final_advice,
                generation_key=(
                    (
                        analysis_request.position_snapshot_key
                        if analysis_request is not None
                        else advice_request_key
                    )
                    or f"weekly:{final_state.iteration_id}"
                ),
                **(
                    {}
                    if execution_guard is None
                    else {"execution_guard": execution_guard}
                ),
            )
        except DuplicateAdvice:
            advice_persisted = False
        with self._generation_lock:
            self._advice_generation[symbol] += 1
            advice_generation = self._advice_generation[symbol]
        self._emit(
            progress_callback,
            _progress(
                "completed",
                final_state,
                total=max(absolute_total, len(final_state.audits)),
                started=started,
            ),
        )
        return EngineRunResult(
            symbol=symbol,
            latest_complete_week=latest_sample.week_key,
            total_weeks=len(samples),
            processed_weeks=processed,
            skipped_weeks=skipped,
            iteration_id=final_state.iteration_id,
            work_version=final_state.work_version,
            model_version=final_state.current_model_version,
            advice=final_advice,
            advice_generation=advice_generation,
            advice_persisted=advice_persisted,
        )

    def _run_one_week(
        self,
        dataset: VerifiedDataset,
        state: WorkState,
        sample: WeekSample,
        *,
        daily_closes: tuple[DailyClose, ...] | None = None,
        samples_by_week: Mapping[str, WeekSample] | None = None,
        feature_cache: dict[str, FeatureSnapshot] | None = None,
    ) -> _StepExecution:
        features = self._features_for_sample(dataset, sample)
        if feature_cache is not None:
            feature_cache[sample.week_key] = features
        if not features.quality_report.is_publishable:
            blocking = tuple(
                issue.code
                for issue in features.quality_report.issues
                if issue.severity == "blocking"
            )
            raise DatasetQualityError(
                "Feature quality gate blocked "
                f"{sample.week_key}: {','.join(blocking)}"
            )
        judgment = self.judgment_policy.judge(
            model=state,
            features=features,
            sample=sample,
            current_position=HISTORICAL_EVALUATION_POSITION,
        )
        advice = _build_advice_from_judgment(
            model=state,
            features=features,
            calendar=dataset.trading_calendar,
            current_position=HISTORICAL_EVALUATION_POSITION,
            judgment=judgment,
        )
        prediction = _prediction_from_advice(
            state, sample, advice
        )
        historical_predictions = tuple(
            _prediction_from_label(label)
            for label in state.feedback
            if label.status != "mature_13w"
        )
        active_predictions = historical_predictions + (prediction,)
        earliest_cutoff = min(
            item.cutoff_date for item in active_predictions
        )
        visible_daily = tuple(
            row
            for row in (
                daily_closes
                if daily_closes is not None
                else _daily_closes(dataset)
            )
            if earliest_cutoff <= row.trade_date <= sample.sample_date
        )
        visible_calendar_dates = tuple(
            day
            for day in dataset.trading_calendar.expected_trade_dates
            if earliest_cutoff <= day <= sample.sample_date
        )
        label_calendar = TradingCalendar(
            instrument_code=dataset.instrument_code,
            expected_trade_dates=visible_calendar_dates,
            coverage_start=dataset.trading_calendar.coverage_start,
            coverage_end=dataset.trading_calendar.coverage_end,
        )
        labels = update_labels(
            active_predictions,
            visible_daily,
            sample.sample_date,
            expected_trade_dates=label_calendar,
            completed_week_ends=tuple(
                row.period_end
                for row in dataset.completed_weekly_bars
                if earliest_cutoff
                < row.period_end
                <= sample.sample_date
            ),
        )
        evaluator_identity = getattr(
            self.judgment_policy,
            "counterfactual_evaluator_identity",
            None,
        )
        evaluator = None
        if isinstance(evaluator_identity, str) and evaluator_identity.strip():
            evaluator = self._counterfactual_evaluator(
                dataset=dataset,
                state=state,
                as_of_sample=sample,
                daily_closes=(
                    daily_closes
                    if daily_closes is not None
                    else _daily_closes(dataset)
                ),
                samples_by_week=(
                    samples_by_week
                    if samples_by_week is not None
                    else {sample.week_key: sample}
                ),
                feature_cache=(
                    feature_cache if feature_cache is not None else {}
                ),
            )
        else:
            evaluator_identity = None
        result = run_optimizer_step(
            state,
            feedback=labels,
            week_key=sample.week_key,
            cutoff_date=sample.sample_date,
            source_data_max_date=features.source_data_max_date,
            prediction=prediction,
            evaluator=evaluator,
            evaluator_identity=evaluator_identity,
            data_integrity=features.quality_report.is_publishable,
        )
        return _StepExecution(
            inputs=StepInputs(
                sample=sample,
                features=features,
                judgment=judgment,
                advice=advice,
                prediction=prediction,
                feedback=result.work_state.feedback,
            ),
            optimizer_result=result,
        )

    def _counterfactual_evaluator(
        self,
        *,
        dataset: VerifiedDataset,
        state: WorkState,
        as_of_sample: WeekSample,
        daily_closes: tuple[DailyClose, ...],
        samples_by_week: Mapping[str, WeekSample],
        feature_cache: dict[str, FeatureSnapshot],
    ) -> Callable[..., Mapping[str, WindowMetrics]]:
        """Re-score mature historical weeks with one proposed weight vector.

        The closure is deliberately bounded by ``as_of_sample``.  It rebuilds
        predictions only for the active rolling window, then lets the existing
        label engine derive losses from closes that were visible at the current
        cutoff.  No later feature, close, label, or market is reachable.
        """

        def evaluate(
            *,
            weights: Mapping[str, Decimal],
            feedback: tuple[IterationLabel, ...],
            windows: tuple[str, ...],
        ) -> Mapping[str, WindowMetrics]:
            if not windows:
                return {}
            mature = tuple(
                sorted(
                    (
                        label
                        for label in feedback
                        if label.status == "mature_13w"
                    ),
                    key=lambda label: (
                        label.observed_through,
                        label.cutoff_date,
                        label.iteration_id,
                    ),
                )
            )
            if not mature:
                return {}
            selected_count = (
                52
                if "52" in windows
                else 20
                if "20" in windows
                else len(mature)
            )
            selected = mature[-selected_count:]
            candidate_model = replace(
                state,
                current_model_weights=weights,
            )
            predictions: list[IterationPrediction] = []
            for label in selected:
                historical_sample = samples_by_week.get(label.week_key)
                if historical_sample is None:
                    raise DatasetQualityError(
                        "Counterfactual sample is absent from the verified "
                        f"ten-year window: {label.week_key}"
                    )
                historical_features = feature_cache.get(label.week_key)
                if historical_features is None:
                    historical_features = self._features_for_sample(
                        dataset,
                        historical_sample,
                    )
                    feature_cache[label.week_key] = historical_features
                judgment = self.judgment_policy.judge(
                    model=candidate_model,
                    features=historical_features,
                    sample=historical_sample,
                    current_position=HISTORICAL_EVALUATION_POSITION,
                )
                advice = _build_advice_from_judgment(
                    # Advice constraints do not read model weights; keep the
                    # audited state here so its model-proof validation remains
                    # intact while the judgment above uses candidate weights.
                    model=state,
                    features=historical_features,
                    calendar=dataset.trading_calendar,
                    current_position=HISTORICAL_EVALUATION_POSITION,
                    judgment=judgment,
                )
                predictions.append(
                    _prediction_from_advice_for_iteration(
                        label.iteration_id,
                        historical_sample,
                        advice,
                    )
                )

            earliest_cutoff = min(
                prediction.cutoff_date for prediction in predictions
            )
            visible_daily = tuple(
                row
                for row in daily_closes
                if earliest_cutoff
                <= row.trade_date
                <= as_of_sample.sample_date
            )
            visible_calendar_dates = tuple(
                day
                for day in dataset.trading_calendar.expected_trade_dates
                if earliest_cutoff <= day <= as_of_sample.sample_date
            )
            evaluated_labels = update_labels(
                predictions,
                visible_daily,
                as_of_sample.sample_date,
                expected_trade_dates=TradingCalendar(
                    instrument_code=dataset.instrument_code,
                    expected_trade_dates=visible_calendar_dates,
                    coverage_start=dataset.trading_calendar.coverage_start,
                    coverage_end=dataset.trading_calendar.coverage_end,
                ),
                completed_week_ends=tuple(
                    row.period_end
                    for row in dataset.completed_weekly_bars
                    if earliest_cutoff
                    < row.period_end
                    <= as_of_sample.sample_date
                ),
            )
            if any(label.status != "mature_13w" for label in evaluated_labels):
                raise DatasetQualityError(
                    "Counterfactual evaluation selected an immature label"
                )
            metrics: dict[str, WindowMetrics] = {}
            for window_name in windows:
                requested_window = (
                    20 if window_name in {"expanding", "20"} else 52
                )
                metric = rolling_metrics(
                    evaluated_labels,
                    requested_window,
                    instrument_code=dataset.instrument_code,
                )
                if not isinstance(metric, WindowMetrics):
                    raise DatasetQualityError(
                        "Counterfactual rolling metric is unavailable"
                    )
                if metric.window != window_name:
                    raise DatasetQualityError(
                        "Counterfactual rolling window identity changed"
                    )
                metrics[window_name] = metric
            return metrics

        return evaluate

    def _features_for_sample(
        self,
        dataset: VerifiedDataset,
        sample: WeekSample,
    ) -> FeatureSnapshot:
        previous_end = sample.previous_completed_week_end
        if previous_end is None:
            raise DatasetQualityError(
                f"{sample.week_key} has no completed warmup week"
            )
        all_bars = tuple(
            row
            for row in dataset.completed_weekly_bars
            if row.period_end <= previous_end
        )
        bars = all_bars[-WARMUP_WEEKS:]
        valuations = tuple(
            point
            for point in dataset.valuations
            if point.valuation_date <= previous_end
        )
        expected_week_ends = tuple(row.period_end for row in bars)
        expected_trade_dates = tuple(
            day
            for day in dataset.trading_calendar.expected_trade_dates
            if bars[0].period_start <= day <= previous_end
        )
        snapshot = build_feature_snapshot(
            bars,
            valuations,
            sample.sample_date,
            dataset.instrument_code,
            expected_week_ends=expected_week_ends,
            expected_trade_dates=expected_trade_dates,
        )
        if (
            snapshot.source_data_max_date is not None
            and snapshot.source_data_max_date >= sample.sample_date
        ):
            raise DatasetQualityError(
                "Feature source must end before the sampled trading day"
            )
        return snapshot

    @staticmethod
    def _analysis_samples(
        dataset: VerifiedDataset,
        symbol: str,
        analysis_date: date,
    ) -> tuple[WeekSample, ...]:
        all_samples = build_week_samples(
            (row.trade_date for row in dataset.daily_bars),
            symbol=symbol,
            complete_week_keys=dataset.complete_week_keys,
        )
        boundary = _subtract_years(analysis_date, ANALYSIS_YEARS)
        return tuple(
            sample
            for sample in all_samples
            if sample.candidate_dates[0] >= boundary
            and sample.sample_date <= analysis_date
        )

    @staticmethod
    def _validate_dataset(
        dataset: VerifiedDataset,
        symbol: str,
        analysis_date: date,
    ) -> None:
        if not isinstance(dataset, VerifiedDataset):
            raise TypeError(
                "provider must return a VerifiedDataset quality boundary"
            )
        if dataset.instrument_code != symbol:
            raise DatasetQualityError(
                "Provider returned data for a different market"
            )
        after_as_of = tuple(
            row.trade_date
            for row in dataset.daily_bars
            if row.trade_date > analysis_date
        )
        if after_as_of:
            raise DatasetQualityError(
                "Verified daily data extends after analysis as_of"
            )

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        update: ProgressUpdate,
    ) -> None:
        if callback is None:
            return
        try:
            callback(update)
        except TaskClaimLost:
            raise
        except Exception:
            # A UI/progress failure can never roll back a committed week.
            return


@dataclass(frozen=True, slots=True)
class _StepExecution:
    inputs: StepInputs
    optimizer_result: OptimizerStepResult


def _build_advice_from_judgment(
    *,
    model: WorkState,
    features: FeatureSnapshot,
    calendar: TradingCalendar,
    current_position: int | None,
    judgment: WeeklyJudgment,
) -> Advice:
    return build_advice(
        model=model,
        latest_features=features,
        future_trading_calendar=calendar,
        current_position=current_position,
        target_position=judgment.target_position,
        direction=judgment.direction,
        probability=judgment.probability,
        confidence=judgment.confidence,
        direction_probabilities=judgment.direction_probabilities,
        confirmation_conditions_met=(
            judgment.confirmation_conditions_met
        ),
    )


def _prediction_from_advice(
    state: WorkState,
    sample: WeekSample,
    advice: Advice,
) -> IterationPrediction:
    next_number = int(state.iteration_id[1:]) + 1
    return _prediction_from_advice_for_iteration(
        f"I{next_number:04d}",
        sample,
        advice,
    )


def _prediction_from_advice_for_iteration(
    iteration_id: str,
    sample: WeekSample,
    advice: Advice,
) -> IterationPrediction:
    action_date = (
        advice.batches[0].expected_date
        if advice.batches
        else sample.sample_date
    )
    return IterationPrediction(
        iteration_id=iteration_id,
        instrument_code=sample.symbol,
        week_key=sample.week_key,
        cutoff_date=sample.sample_date,
        predicted_direction=advice.direction,
        operation_side=advice.operation_side,
        probability=advice.probability / Decimal("100"),
        predicted_action_date=action_date,
        target_position=Decimal(advice.target_position) / Decimal("100"),
        trade_count=len(advice.batches),
    )


def _prediction_from_label(label: IterationLabel) -> IterationPrediction:
    return IterationPrediction(
        iteration_id=label.iteration_id,
        instrument_code=label.instrument_code,
        week_key=label.week_key,
        cutoff_date=label.cutoff_date,
        predicted_direction=label.predicted_direction,
        operation_side=label.operation_side,
        probability=label.probability,
        predicted_action_date=label.predicted_action_date,
        target_position=label.target_position,
        trade_count=label.trade_count,
    )


def _daily_closes(dataset: VerifiedDataset) -> tuple[DailyClose, ...]:
    return tuple(
        DailyClose(
            trade_date=row.trade_date,
            close=row.close_price,  # type: ignore[arg-type]
            instrument_code=dataset.instrument_code,
        )
        for row in dataset.daily_bars
    )


def _progress(
    stage: str,
    state: WorkState,
    *,
    total: int,
    started: float,
    last_checkpoint: str | None = None,
) -> ProgressUpdate:
    return ProgressUpdate(
        stage=stage,
        completed_weeks=len(state.audits),
        total_weeks=max(total, len(state.audits)),
        iteration_id=state.iteration_id,
        work_version=state.work_version,
        model_version=state.current_model_version,
        last_checkpoint=(
            last_checkpoint
            if last_checkpoint is not None
            else None if not state.audits else state.audits[-1].week_key
        ),
        elapsed_seconds=max(time.monotonic() - started, 0.0),
    )


def _week_key(day: date) -> str:
    iso_year, iso_week, _weekday = day.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _source_is_prohibited(source: str | None) -> bool:
    normalized = (source or "").upper()
    return any(marker in normalized for marker in _PROHIBITED_SOURCE_MARKERS)


def _validate_symbol(symbol: str) -> str:
    if symbol not in SUPPORTED_MARKETS:
        raise ValueError("Only supported markets 399006 and NDX may run")
    return symbol


__all__ = [
    "ANALYSIS_YEARS",
    "DatasetQualityError",
    "EngineRunResult",
    "JudgmentPolicy",
    "MODEL_LINE",
    "ProgressUpdate",
    "StepInputs",
    "SUPPORTED_MARKETS",
    "TradingCalendar",
    "VerifiedDataset",
    "VerifiedDatasetProvider",
    "WARMUP_WEEKS",
    "WeeklyAnalysisEngine",
    "WeeklyJudgment",
    "build_week_samples",
]
