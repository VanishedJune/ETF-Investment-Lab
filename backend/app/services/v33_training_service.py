"""Deterministic V3.3 progressive training for a 20-week probability path.

This module is the model core, not a database task runner.  Persistence is
behind :class:`V33TrainingRepository`, which lets the V3.3 schema layer store
the exact immutable state/forecast without coupling this file to a migration.

Model structure
---------------

The primary forecast is a regularised multi-output weekly/fundamental model
plus a small gradient-boosted residual-stump ensemble.  A separate linear
corrector consumes the complete most-recent 100-session daily curve, may alter
only weeks 1-4 with deterministic decay, and is hard bounded.  It cannot alter
the 20-week terminal direction.  Historical neighbours are used *only* to
calibrate residual quantiles; they are never the central forecast.

For a standardised design matrix ``X`` and 20-week cumulative-return target
``Y``, the linear head is:

``B = argmin ||Y - X B||^2 + alpha ||B||^2 + l1 * ||B||_1``

Ridge is solved in primal or dual form and soft-thresholded for deterministic
elastic shrinkage.  Residual stumps then learn bounded nonlinear interactions.
P10/P90 are calibrated from purged out-of-sample residuals; the analogue layer
may blend at most 25 percent of its local residual correction.

Every validation split has a 20-week purge/embargo.  A training cutoff may use
an outcome only when ``outcome_available_date <= cutoff_date``.  Consequently
the most recent twenty forecasts are pending, never zero-filled/interpolated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date
import hashlib
import json
import math
import statistics
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .v33_feature_service import (
    FeatureSnapshot,
    HORIZON_WEEKS,
    SUPPORTED_MARKETS,
    V33FeatureError,
    flatten_snapshot,
)


PURGE_WEEKS = HORIZON_WEEKS
MIN_TRAIN_SAMPLES = 24
MIN_159941_COLD_START_SAMPLES = 12
VALIDATION_SAMPLES = 52
RECENT_VALIDATION_SAMPLES = 20
MAX_ANALOG_CALIBRATION_WEIGHT = 0.25
DAILY_CORRECTION_WEEKS = 4
DAILY_CORRECTION_DECAY = (1.0, 0.75, 0.5, 0.25)
# Maximum absolute cumulative-return correction at week 1.  Later allowed
# values are this cap multiplied by DAILY_CORRECTION_DECAY and weeks 5-20 are
# exactly zero.  Position/confidence limits live in the analysis layer.
DAILY_PATH_CORRECTION_MAX_ABS = 0.03
DAILY_POSITION_ADJUSTMENT_LIMIT = 10
DAILY_CONFIDENCE_ADJUSTMENT_LIMIT = 10
HIGH_CONFIDENCE_GATE = 70.0
FUND_ETF_RATIOS = ("7:3", "6:4", "5:5", "4:6", "3:7")


class V33TrainingError(RuntimeError):
    """The requested operation would violate the V3.3 training contract."""


@dataclass(frozen=True, slots=True)
class V33TrainingPoint:
    sequence_number: int
    market: str
    cutoff_date: date
    snapshot: FeatureSnapshot
    future_path: tuple[float, ...] | None
    outcome_available_date: date | None
    future_session_offsets: tuple[int, ...] = ()
    sampled_date: date | None = None
    sampling_seed: int = 330020
    actual_dif_turn_days: float | None = None
    actual_dif_turn_kind: str | None = None
    actual_price_turn_days: float | None = None
    actual_price_turn_kind: str | None = None

    def __post_init__(self) -> None:
        if self.market != self.snapshot.market:
            raise V33TrainingError("training point market/snapshot mismatch")
        if self.snapshot.source_data_max_date > self.cutoff_date:
            raise V33TrainingError("source_data_max_date exceeds training cutoff")
        if self.future_path is not None and len(self.future_path) != HORIZON_WEEKS:
            raise V33TrainingError("future path must contain exactly 20 weeks")
        if (self.future_path is None) != (self.outcome_available_date is None):
            raise V33TrainingError("future path and outcome availability must mature together")
        if self.future_path is not None:
            if len(self.future_session_offsets) != HORIZON_WEEKS:
                raise V33TrainingError(
                    "mature training point requires one real-session offset for every future week"
                )
            if any(value <= 0 for value in self.future_session_offsets):
                raise V33TrainingError("future-session offsets must be positive")
            if any(
                left >= right
                for left, right in zip(
                    self.future_session_offsets, self.future_session_offsets[1:]
                )
            ):
                raise V33TrainingError("future-session offsets must be strictly increasing")
        if self.sampled_date is not None and self.snapshot.source_data_max_date > self.sampled_date:
            raise V33TrainingError("sampled-day snapshot contains later daily data")


@dataclass(frozen=True, slots=True)
class ResidualStump:
    feature_index: int
    threshold: float
    left: tuple[float, ...]
    right: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class V33ModelState:
    market: str
    version: str
    iteration_number: int
    trained_through: date
    parent_state_hash: str | None
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    stumps: tuple[ResidualStump, ...]
    residual_p10: tuple[float, ...]
    residual_p50: tuple[float, ...]
    residual_p90: tuple[float, ...]
    terminal_residuals: tuple[float, ...]
    hyperparameters: Mapping[str, float]
    optimizer_memory: Mapping[str, Any]
    validation_metrics: Mapping[str, float | None]
    turn_lag_days: Mapping[str, float]
    training_sample_count: int
    state_hash: str


@dataclass(frozen=True, slots=True)
class V33Forecast:
    market: str
    cutoff_date: date
    p10: tuple[float, ...]
    p50: tuple[float, ...]
    p90: tuple[float, ...]
    expected: tuple[float, ...]
    weekly_base: tuple[float, ...]
    daily_correction: tuple[float, ...]
    up_probability: float
    sideways_probability: float
    down_probability: float
    direction: str
    confidence: float
    threshold: float
    expected_max_drawdown: float
    predicted_high_week: int
    predicted_low_week: int
    dif_turn_kind: str
    dif_turn_days: float | None
    price_turn_days: float | None
    turn_stable: bool
    analogue_calibration_count: int
    model_version: str
    model_state_hash: str


@dataclass(frozen=True, slots=True)
class V33IterationRecord:
    market: str
    iteration_number: int
    cutoff_date: date
    parent_state_hash: str | None
    state_hash: str
    champion_version: str
    challenger_promoted: bool
    promotion_reason: str
    forecast: V33Forecast
    status: str = "pending"
    evaluation: Mapping[str, float | None] | None = None


@dataclass(frozen=True, slots=True)
class V33ProgressiveResult:
    market: str
    state: V33ModelState
    iterations: tuple[V33IterationRecord, ...]
    pending_count: int
    full_count: int
    curve: tuple[Mapping[str, float | int | None], ...]
    expected_iteration_count: int
    audited_iteration_count: int
    weekly_sampling_seed: int
    sampled_dates: tuple[str, ...]


class V33TrainingRepository(Protocol):
    """Minimal persistence boundary used by a DB-backed integration layer."""

    def latest_state(self, market: str) -> V33ModelState | None: ...

    def save_state(self, state: V33ModelState) -> None: ...

    def save_iteration(self, iteration: V33IterationRecord) -> None: ...

    def save_warmup_anchor(
        self,
        state: V33ModelState,
        *,
        build_cutoff_dates: Sequence[date],
        eligible_cutoff_dates: Sequence[date],
    ) -> None: ...


class InMemoryV33TrainingRepository:
    """Test/standalone repository with identical training/inference semantics."""

    def __init__(self) -> None:
        self.states: dict[str, V33ModelState] = {}
        self.iterations: dict[str, list[V33IterationRecord]] = {market: [] for market in SUPPORTED_MARKETS}
        self.warmup_anchors: dict[str, dict[str, Any]] = {}

    def latest_state(self, market: str) -> V33ModelState | None:
        return self.states.get(market)

    def save_state(self, state: V33ModelState) -> None:
        self.states[state.market] = state

    def save_iteration(self, iteration: V33IterationRecord) -> None:
        rows = self.iterations.setdefault(iteration.market, [])
        for index, existing in enumerate(rows):
            if existing.iteration_number == iteration.iteration_number:
                rows[index] = iteration
                break
        else:
            rows.append(iteration)

    def save_warmup_anchor(
        self,
        state: V33ModelState,
        *,
        build_cutoff_dates: Sequence[date],
        eligible_cutoff_dates: Sequence[date],
    ) -> None:
        if state.iteration_number != 0 or state.parent_state_hash is not None:
            raise V33TrainingError("warm-up anchor must be the non-formal root state")
        payload = {
            "state_payload": state_to_payload(state),
            "state_hash": state.state_hash,
            "build_cutoff_dates": [value.isoformat() for value in build_cutoff_dates],
            "eligible_cutoff_dates": [value.isoformat() for value in eligible_cutoff_dates],
        }
        existing = self.warmup_anchors.get(state.market)
        if existing is not None and _hash_payload(existing) != _hash_payload(payload):
            raise V33TrainingError("warm-up anchor changed during an immutable bootstrap")
        self.warmup_anchors[state.market] = payload


def _snap5(value: float) -> int:
    return max(0, min(100, int(round(float(value) / 5.0) * 5)))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _quantile(values: np.ndarray, q: float, axis: int = 0) -> np.ndarray:
    if values.size == 0:
        return np.zeros(HORIZON_WEEKS, dtype=float)
    return np.quantile(values, q, axis=axis)


def _pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def _is_daily_corrector_feature(name: str) -> bool:
    """Keep the 100-session timing corrector disjoint from the weekly model."""

    return name.startswith(("daily_", "seq100_", "missing_daily_", "missing_seq100_"))


def _ridge_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
    l1: float,
) -> np.ndarray:
    if design.shape[1] == 0:
        return np.zeros((0, target.shape[1]), dtype=float)
    if design.shape[1] <= design.shape[0]:
        gram = design.T @ design + alpha * np.eye(design.shape[1])
        coefficients = np.linalg.solve(gram, design.T @ target)
    else:
        dual = np.linalg.solve(
            design @ design.T + alpha * np.eye(design.shape[0]), target
        )
        coefficients = design.T @ dual
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - l1, 0.0)


def _bounded_daily_correction(raw: np.ndarray) -> np.ndarray:
    """Bound and fade a daily residual correction without touching week 20."""

    values = np.asarray(raw, dtype=float)
    if values.shape[-1] != HORIZON_WEEKS:
        raise V33TrainingError("daily correction must contain exactly 20 weeks")
    decay = np.zeros(HORIZON_WEEKS, dtype=float)
    decay[:DAILY_CORRECTION_WEEKS] = np.asarray(
        DAILY_CORRECTION_DECAY, dtype=float
    )
    clipped = np.clip(
        values,
        -DAILY_PATH_CORRECTION_MAX_ABS,
        DAILY_PATH_CORRECTION_MAX_ABS,
    )
    return clipped * decay


def _max_drawdown_from_return_path(path: Sequence[float]) -> float:
    wealth = [1.0 + float(value) for value in path]
    peak = wealth[0]
    worst = 0.0
    for value in wealth:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1.0)
    return worst


def deterministic_weekly_sample_dates(
    sessions: Sequence[date],
    market: str,
    *,
    seed: int = 330020,
) -> tuple[date, ...]:
    """Pick one reproducible *real session* in every ISO natural week."""

    if market not in SUPPORTED_MARKETS:
        raise V33TrainingError("sampling supports only 399006 and 159941")
    grouped: dict[tuple[int, int], list[date]] = {}
    for session in sorted(set(sessions)):
        year, week, _ = session.isocalendar()
        grouped.setdefault((year, week), []).append(session)
    selected: list[date] = []
    for (year, week), candidates in sorted(grouped.items()):
        digest = hashlib.sha256(f"{seed}:{market}:{year}-W{week:02d}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(candidates)
        selected.append(candidates[index])
    return tuple(selected)


def build_training_points(
    snapshots: Sequence[FeatureSnapshot],
    *,
    sampling_seed: int = 330020,
) -> tuple[V33TrainingPoint, ...]:
    """Build exact 20-week labels; the final twenty points remain pending."""

    ordered = sorted(snapshots, key=lambda item: item.cutoff_date)
    if not ordered:
        return ()
    market = ordered[0].market
    if market not in SUPPORTED_MARKETS or any(snapshot.market != market for snapshot in ordered):
        raise V33TrainingError("training snapshots must belong to one supported market")
    if any(left.cutoff_date >= right.cutoff_date for left, right in zip(ordered, ordered[1:])):
        raise V33TrainingError("snapshot cutoffs must be unique and increasing")
    # Each feature snapshot contains the real exchange sessions visible at its
    # cutoff.  Their union reconstructs the audited session calendar across the
    # full progressive run, including exchange holidays and ad-hoc closures.
    # This calendar is the only clock used by turn-error metrics.
    trading_sessions = tuple(
        sorted(
            {
                date.fromisoformat(str(row["trade_date"]))
                for item in ordered
                for row in item.daily_sequence
                if row.get("trade_date") is not None
            }
        )
    )
    session_ordinals = {
        session: ordinal for ordinal, session in enumerate(trading_sessions)
    }
    sampled_sessions = {item.daily_as_of for item in ordered}
    if not sampled_sessions.issubset(session_ordinals):
        raise V33TrainingError("sampled cutoff is missing from the real trading-session sequence")
    closes = [float(snapshot.daily_sequence[-1]["close"]) for snapshot in ordered]
    dif_slopes = [snapshot.weekly.get("weekly_dif_slope_1") for snapshot in ordered]
    result: list[V33TrainingPoint] = []
    for index, snapshot in enumerate(ordered):
        if index + HORIZON_WEEKS >= len(ordered):
            path = None
            outcome_date = None
            future_session_offsets: tuple[int, ...] = ()
            dif_turn_days = None
            dif_turn_kind = None
            price_turn_days = None
            price_turn_kind = None
        else:
            path = tuple(
                closes[index + step] / closes[index] - 1.0
                for step in range(1, HORIZON_WEEKS + 1)
            )
            outcome_date = ordered[index + HORIZON_WEEKS].cutoff_date
            future_snapshots = ordered[index : index + HORIZON_WEEKS + 1]
            future_offsets = tuple(
                _session_distance(
                    session_ordinals,
                    ordered[index].daily_as_of,
                    future.daily_as_of,
                )
                for future in future_snapshots
            )
            future_session_offsets = future_offsets[1:]
            future_slopes = dif_slopes[index : index + HORIZON_WEEKS + 1]
            dif_turn_days, dif_turn_kind = _first_sign_turn(future_slopes, future_offsets)
            high_index = int(np.argmax(np.asarray(path, dtype=float))) + 1
            low_index = int(np.argmin(np.asarray(path, dtype=float))) + 1
            # The dominant terminal direction selects the economically relevant
            # price turn for lead/lag calibration; both high/low weeks remain in
            # every forecast/evaluation independently.
            if path[-1] >= 0.0:
                price_turn_days, price_turn_kind = float(future_offsets[high_index]), "top"
            else:
                price_turn_days, price_turn_kind = float(future_offsets[low_index]), "bottom"
        result.append(
            V33TrainingPoint(
                sequence_number=index + 1,
                market=market,
                cutoff_date=snapshot.cutoff_date,
                snapshot=snapshot,
                future_path=path,
                outcome_available_date=outcome_date,
                future_session_offsets=future_session_offsets,
                sampled_date=snapshot.daily_as_of,
                sampling_seed=sampling_seed,
                actual_dif_turn_days=dif_turn_days,
                actual_dif_turn_kind=dif_turn_kind,
                actual_price_turn_days=price_turn_days,
                actual_price_turn_kind=price_turn_kind,
            )
        )
    return tuple(result)


def _session_distance(
    session_ordinals: Mapping[date, int],
    start: date,
    end: date,
) -> int:
    if end < start:
        raise V33TrainingError("session-distance end precedes start")
    if start not in session_ordinals or end not in session_ordinals:
        raise V33TrainingError("session-distance endpoint is absent from exchange sessions")
    return session_ordinals[end] - session_ordinals[start]


def _first_sign_turn(
    slopes: Sequence[float | None],
    session_offsets: Sequence[int],
) -> tuple[float | None, str | None]:
    if len(slopes) != len(session_offsets):
        raise V33TrainingError("DIF slopes and real-session offsets must align")
    if any(left >= right for left, right in zip(session_offsets, session_offsets[1:])):
        raise V33TrainingError("DIF real-session offsets must be strictly increasing")
    previous: float | None = None if not slopes or slopes[0] is None else float(slopes[0])
    for index, value in enumerate(slopes[1:], start=1):
        if value is None:
            continue
        current = float(value)
        if previous is not None:
            if previous < 0.0 <= current:
                return float(session_offsets[index]), "bottom"
            if previous > 0.0 >= current:
                return float(session_offsets[index]), "top"
        previous = current
    return None, None


class V33TrainingService:
    """Progressive market-specific trainer and pure model predictor."""

    def __init__(self, repository: V33TrainingRepository | None = None) -> None:
        self.repository = repository or InMemoryV33TrainingRepository()
        self._feature_cache: dict[int, dict[str, float | None]] = {}
        self._active_state_hash: str | None = None
        self._active_transform_cache: dict[int, np.ndarray] = {}
        self._active_state_arrays: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] | None = None

    @staticmethod
    def validate_market(market: str) -> None:
        if market not in SUPPORTED_MARKETS:
            raise V33TrainingError("V3.3 supports only 399006 and 159941; NDX is benchmark-only")

    @staticmethod
    def _minimum_initial_samples(market: str) -> int:
        return MIN_159941_COLD_START_SAMPLES if market == "159941" else MIN_TRAIN_SAMPLES

    @staticmethod
    def _cold_start_hyperparameters(
        values: Mapping[str, float]
    ) -> dict[str, float]:
        return {
            **dict(values),
            "ridge_alpha": max(40.0, float(values["ridge_alpha"])),
            "l1_shrinkage": max(0.001, float(values["l1_shrinkage"])),
            "boost_rounds": min(2.0, float(values["boost_rounds"])),
            "learning_rate": min(0.04, float(values["learning_rate"])),
        }

    def fit_warmup_anchor(
        self,
        market: str,
        points: Sequence[V33TrainingPoint],
        *,
        first_formal_cutoff: date,
    ) -> tuple[V33ModelState, tuple[V33TrainingPoint, ...]]:
        """Rebuild the deterministic, non-formal state preceding iteration 1.

        The returned state is a real model fitted only from build-period labels
        that had fully matured by the first formal cutoff.  It is iteration 0
        for lineage purposes, but is never a formal training iteration or a
        published model version.
        """

        self.validate_market(market)
        ordered = sorted(points, key=lambda point: point.cutoff_date)
        if any(point.market != market for point in ordered):
            raise V33TrainingError("cross-market warm-up samples are forbidden")
        warmup = tuple(
            point
            for point in ordered
            if point.cutoff_date < first_formal_cutoff
            and point.future_path is not None
            and point.outcome_available_date is not None
            and point.outcome_available_date <= first_formal_cutoff
        )
        minimum_initial = self._minimum_initial_samples(market)
        if len(warmup) < minimum_initial:
            raise V33TrainingError(
                "formal ten-year window lacks the minimum real matured warm-up history"
            )
        prefix = "CYB" if market == "399006" else "GFNDXETF"
        hyperparameters = self._initial_hyperparameters(market)
        optimizer: dict[str, Any] = {
            "coordinate": 0,
            "direction": 1,
            "warmup_only": True,
        }
        if len(warmup) < MIN_TRAIN_SAMPLES:
            hyperparameters = self._cold_start_hyperparameters(hyperparameters)
            optimizer["training_mode"] = "small_sample_degraded"
        state = self._fit_state(
            market,
            warmup,
            hyperparameters,
            iteration_number=0,
            trained_through=first_formal_cutoff,
            parent_hash=None,
            version=f"{prefix}_HYBRID_20W_V3.3.0",
            optimizer_memory=optimizer,
        )
        return state, warmup

    def train_progressive(
        self,
        market: str,
        points: Sequence[V33TrainingPoint],
        *,
        initial_state: V33ModelState | None = None,
        audit_start_date: date | None = None,
        expected_iteration_count: int | None = None,
        weekly_sampling_seed: int = 330020,
    ) -> V33ProgressiveResult:
        """Run one inherited iteration per supplied complete trading week.

        Historical labels are present in ``points`` for later audit, but round
        N receives only labels whose full 20-week outcome was available at N's
        cutoff.  Saved predictions are never recomputed when they mature.
        """

        self.validate_market(market)
        ordered = sorted(points, key=lambda point: point.cutoff_date)
        if any(point.market != market for point in ordered):
            raise V33TrainingError("cross-market samples are forbidden")
        state = initial_state
        records: list[V33IterationRecord] = []
        formal_points = (
            ordered
            if audit_start_date is None
            else [point for point in ordered if point.cutoff_date >= audit_start_date]
        )
        if expected_iteration_count is not None and len(formal_points) != expected_iteration_count:
            raise V33TrainingError(
                f"expected {expected_iteration_count} audited natural-week iterations, got {len(formal_points)}"
            )
        if any(
            left.cutoff_date.isocalendar()[:2] == right.cutoff_date.isocalendar()[:2]
            for left, right in zip(formal_points, formal_points[1:])
        ):
            raise V33TrainingError("formal training contains more than one sample in an ISO natural week")
        if audit_start_date is not None and any(
            point.sampled_date is None
            or point.sampled_date != point.cutoff_date
            or point.sampling_seed != weekly_sampling_seed
            or point.snapshot.weekly_as_of.isocalendar()[:2]
            == point.sampled_date.isocalendar()[:2]
            for point in formal_points
        ):
            raise V33TrainingError(
                "formal samples require a fixed-seed real session and only earlier completed weekly bars"
            )

        if audit_start_date is not None and state is None:
            if not formal_points:
                raise V33TrainingError("formal ten-year window contains no training week")
            first_cutoff = formal_points[0].cutoff_date
            state, warmup = self.fit_warmup_anchor(
                market,
                ordered,
                first_formal_cutoff=first_cutoff,
            )
            self.repository.save_warmup_anchor(
                state,
                build_cutoff_dates=tuple(
                    point.cutoff_date
                    for point in ordered
                    if point.cutoff_date <= first_cutoff
                ),
                eligible_cutoff_dates=tuple(point.cutoff_date for point in warmup),
            )

        for current in formal_points:
            eligible = [
                point
                for point in ordered
                if point.cutoff_date < current.cutoff_date
                and point.future_path is not None
                and point.outcome_available_date is not None
                and point.outcome_available_date <= current.cutoff_date
            ]
            # 159941 is younger than the ten-year formal window plus a full
            # 20-week label horizon.  Its first formal round is therefore an
            # explicit, strongly regularised supervised cold start using only
            # the real matured ETF labels available then.  No NDX label or
            # fabricated prior is permitted.
            if state is None and len(eligible) < self._minimum_initial_samples(market):
                continue
            state, record = self.train_iteration(current, eligible, state)
            self.repository.save_state(state)
            # Persist the exact as-issued forecast immediately for crash-safe
            # resume.  A second idempotent save below only attaches maturity
            # evaluation; it never recomputes this forecast.
            self.repository.save_iteration(record)
            records.append(record)

        # Mature exact saved forecasts only; never regenerate an old forecast
        # with a later model.  The newest twenty remain explicitly pending.
        by_cutoff = {point.cutoff_date: point for point in ordered}
        matured_records: list[V33IterationRecord] = []
        for record in records:
            point = by_cutoff[record.cutoff_date]
            if point.future_path is None or point.outcome_available_date is None:
                matured_records.append(record)
                continue
            latest_cutoff = ordered[-1].cutoff_date
            if point.outcome_available_date > latest_cutoff:
                matured_records.append(record)
                continue
            metrics = self.evaluate(point, record.forecast)
            matured_records.append(replace(record, status="full", evaluation=metrics))
        records = matured_records
        for record in records:
            self.repository.save_iteration(record)
        pending = sum(record.status == "pending" for record in records)
        if len(records) >= HORIZON_WEEKS and pending != HORIZON_WEEKS:
            raise V33TrainingError(
                f"20-week audit invariant failed: expected 20 pending, got {pending}"
            )
        curve = tuple(
            {
                "iteration": record.iteration_number,
                "cutoff_date": record.cutoff_date.isoformat(),
                "status": record.status,
                "loss": None if record.evaluation is None else record.evaluation.get("composite_loss"),
                "price_turn_error_days": None if record.evaluation is None else record.evaluation.get("price_turn_error_days"),
                "dif_turn_error_days": None if record.evaluation is None else record.evaluation.get("dif_turn_error_days"),
            }
            for record in records
        )
        if state is None:
            raise V33TrainingError("no training state created")
        if audit_start_date is not None and len(records) != len(formal_points):
            raise V33TrainingError("an audited natural week was skipped")
        return V33ProgressiveResult(
            market=market,
            state=state,
            iterations=tuple(records),
            pending_count=pending,
            full_count=len(records) - pending,
            curve=curve,
            expected_iteration_count=(
                len(formal_points) if expected_iteration_count is None else expected_iteration_count
            ),
            audited_iteration_count=len(records),
            weekly_sampling_seed=weekly_sampling_seed,
            sampled_dates=tuple(
                (record.cutoff_date if by_cutoff[record.cutoff_date].sampled_date is None else by_cutoff[record.cutoff_date].sampled_date).isoformat()
                for record in records
            ),
        )

    def train_iteration(
        self,
        current: V33TrainingPoint,
        eligible: Sequence[V33TrainingPoint],
        previous: V33ModelState | None,
    ) -> tuple[V33ModelState, V33IterationRecord]:
        self.validate_market(current.market)
        if previous is not None and previous.market != current.market:
            raise V33TrainingError("iteration N cannot inherit another market's state")
        if any(
            point.outcome_available_date is None
            or point.outcome_available_date > current.cutoff_date
            or point.cutoff_date >= current.cutoff_date
            for point in eligible
        ):
            raise V33TrainingError("eligible samples contain unavailable future outcomes")
        if len(eligible) < self._minimum_initial_samples(current.market):
            raise V33TrainingError("insufficient real matured samples for market cold start")

        incumbent_hyper = (
            dict(previous.hyperparameters)
            if previous is not None
            else self._initial_hyperparameters(current.market)
        )
        cold_start = len(eligible) < MIN_TRAIN_SAMPLES
        if cold_start:
            incumbent_hyper = self._cold_start_hyperparameters(incumbent_hyper)
        optimizer = dict(previous.optimizer_memory) if previous is not None else {"coordinate": 0, "direction": 1}
        challenger_hyper = dict(incumbent_hyper)
        next_optimizer = dict(optimizer)
        promoted = previous is None and not cold_start
        reason = (
            "small_sample_degraded_initial_champion"
            if previous is None and cold_start
            else "initial_champion"
        )
        metrics: Mapping[str, float | None] = {}
        calibration_residuals: np.ndarray | None = None

        if len(eligible) >= MIN_TRAIN_SAMPLES + PURGE_WEEKS + 4:
            challenger_hyper, next_optimizer = self._challenger(
                incumbent_hyper, optimizer
            )
            train_rows, validation_rows = self._purged_split(eligible)
            incumbent_eval_state = self._fit_state(
                current.market,
                train_rows,
                incumbent_hyper,
                iteration_number=0,
                trained_through=current.cutoff_date,
                parent_hash=None,
                version="validation-incumbent",
                optimizer_memory=optimizer,
            )
            challenger_eval_state = self._fit_state(
                current.market,
                train_rows,
                challenger_hyper,
                iteration_number=0,
                trained_through=current.cutoff_date,
                parent_hash=None,
                version="validation-challenger",
                optimizer_memory=next_optimizer,
            )
            incumbent_metrics = self._validation_metrics(incumbent_eval_state, validation_rows, train_rows)
            challenger_metrics = self._validation_metrics(challenger_eval_state, validation_rows, train_rows)
            promoted, reason = self._promotion_gate(incumbent_metrics, challenger_metrics)
            metrics = challenger_metrics if promoted else incumbent_metrics
            selected_evaluation_state = challenger_eval_state if promoted else incumbent_eval_state
            calibration_residuals = self._residual_matrix(selected_evaluation_state, validation_rows)
        elif previous is not None:
            reason = (
                "small_sample_degraded_no_challenger"
                if len(eligible) < MIN_TRAIN_SAMPLES
                else "insufficient_purged_validation_keep_champion"
            )

        chosen_hyper = challenger_hyper if promoted else incumbent_hyper
        # Coefficients are re-estimated on the newly matured expanding sample
        # even when challenger hyperparameters are rejected.  Every iteration
        # therefore receives an immutable artifact version; ``promoted`` says
        # whether the challenger configuration won, not whether bytes changed.
        version_number = 1 if previous is None else previous.iteration_number + 1
        prefix = "CYB" if current.market == "399006" else "GFNDXETF"
        version = f"{prefix}_HYBRID_20W_V3.3.{version_number}"
        state = self._fit_state(
            current.market,
            eligible,
            chosen_hyper,
            iteration_number=1 if previous is None else previous.iteration_number + 1,
            trained_through=current.cutoff_date,
            parent_hash=None if previous is None else previous.state_hash,
            version=version,
            optimizer_memory=next_optimizer,
            validation_metrics=metrics,
            calibration_residuals=calibration_residuals,
            prior_calibration_state=previous,
        )
        forecast = self.predict(state, current.snapshot, calibration_points=eligible)
        record = V33IterationRecord(
            market=current.market,
            iteration_number=state.iteration_number,
            cutoff_date=current.cutoff_date,
            parent_state_hash=state.parent_state_hash,
            state_hash=state.state_hash,
            champion_version=state.version,
            challenger_promoted=promoted,
            promotion_reason=reason,
            forecast=forecast,
        )
        return state, record

    def predict(
        self,
        state: V33ModelState,
        snapshot: FeatureSnapshot,
        *,
        calibration_points: Sequence[V33TrainingPoint] = (),
    ) -> V33Forecast:
        """Pure inference; this method never writes state or repository rows."""

        if state.market != snapshot.market:
            raise V33TrainingError("model and inference market mismatch")
        before = state.state_hash
        vector = self._transform_one(snapshot, state)
        weekly_base, daily_correction = self._expected_components(state, vector)
        expected = weekly_base + daily_correction

        local_residuals = self._analogue_residuals(state, snapshot, calibration_points)
        analogue_count = len(local_residuals)
        base_p10 = np.asarray(state.residual_p10, dtype=float)
        base_p50 = np.asarray(state.residual_p50, dtype=float)
        base_p90 = np.asarray(state.residual_p90, dtype=float)
        if analogue_count >= 8:
            local = np.asarray(local_residuals, dtype=float)
            blend = min(MAX_ANALOG_CALIBRATION_WEIGHT, analogue_count / 320.0)
            base_p10 = (1.0 - blend) * base_p10 + blend * _quantile(local, 0.10)
            base_p50 = (1.0 - blend) * base_p50 + blend * _quantile(local, 0.50)
            base_p90 = (1.0 - blend) * base_p90 + blend * _quantile(local, 0.90)
        p10 = expected + base_p10
        p50 = expected + base_p50
        p90 = expected + base_p90
        stacked = np.sort(np.vstack([p10, p50, p90]), axis=0)
        p10, p50, p90 = stacked[0], stacked[1], stacked[2]

        threshold = 0.045 if snapshot.market == "399006" else 0.035
        residuals = np.asarray(state.terminal_residuals or (0.0,), dtype=float)
        terminal_scenarios = expected[-1] + residuals
        up = float(np.mean(terminal_scenarios > threshold))
        down = float(np.mean(terminal_scenarios < -threshold))
        sideways = max(0.0, 1.0 - up - down)
        probabilities = {"up": up, "sideways": sideways, "down": down}
        direction = max(probabilities, key=probabilities.get)
        high_week = int(np.argmax(p50)) + 1
        low_week = int(np.argmin(p50)) + 1
        turn = snapshot.derivative_turn.get("daily_dif", {})
        turn_kind = str(turn.get("kind") or "unstable")
        turn_stable = bool(turn.get("stable"))
        dif_turn_days = float(turn["days_ahead"]) if turn_stable and turn.get("days_ahead") is not None else None
        lag = float(state.turn_lag_days.get(turn_kind, 0.0))
        price_turn_days = None if dif_turn_days is None else max(0.0, dif_turn_days + lag)
        forecast = V33Forecast(
            market=snapshot.market,
            cutoff_date=snapshot.cutoff_date,
            p10=tuple(float(value) for value in p10),
            p50=tuple(float(value) for value in p50),
            p90=tuple(float(value) for value in p90),
            expected=tuple(float(value) for value in expected),
            weekly_base=tuple(float(value) for value in weekly_base),
            daily_correction=tuple(float(value) for value in daily_correction),
            up_probability=up * 100.0,
            sideways_probability=sideways * 100.0,
            down_probability=down * 100.0,
            direction=direction,
            confidence=max(probabilities.values()) * 100.0,
            threshold=threshold,
            expected_max_drawdown=_max_drawdown_from_return_path(expected) * 100.0,
            predicted_high_week=high_week,
            predicted_low_week=low_week,
            dif_turn_kind=turn_kind,
            dif_turn_days=dif_turn_days,
            price_turn_days=price_turn_days,
            turn_stable=turn_stable,
            analogue_calibration_count=analogue_count,
            model_version=state.version,
            model_state_hash=state.state_hash,
        )
        if state.state_hash != before:
            raise V33TrainingError("pure inference mutated model state")
        return forecast

    @staticmethod
    def evaluate(point: V33TrainingPoint, forecast: V33Forecast) -> dict[str, float | None]:
        if point.future_path is None:
            raise V33TrainingError("pending prediction cannot be evaluated")
        actual = np.asarray(point.future_path, dtype=float)
        p10 = np.asarray(forecast.p10, dtype=float)
        p50 = np.asarray(forecast.p50, dtype=float)
        p90 = np.asarray(forecast.p90, dtype=float)
        scale = max(float(np.std(actual)), 0.05)
        wis = (
            _pinball(actual, p10, 0.10)
            + _pinball(actual, p50, 0.50)
            + _pinball(actual, p90, 0.90)
        ) / (3.0 * scale)
        median_path = float(np.mean(np.abs(actual - p50)) / scale)
        terminal = float(abs(actual[-1] - p50[-1]) / scale)
        threshold = forecast.threshold
        actual_class = "up" if actual[-1] > threshold else "down" if actual[-1] < -threshold else "sideways"
        probabilities = {
            "up": forecast.up_probability / 100.0,
            "sideways": forecast.sideways_probability / 100.0,
            "down": forecast.down_probability / 100.0,
        }
        brier = sum((probabilities[key] - (1.0 if key == actual_class else 0.0)) ** 2 for key in probabilities) / 3.0
        inside = np.logical_and(actual >= p10, actual <= p90)
        coverage = float(np.mean(inside))
        interval = float(np.mean(np.maximum(0.0, p90 - p10)) / scale)
        coverage_loss = abs(coverage - 0.80) + 0.1 * interval
        actual_drawdown = _max_drawdown_from_return_path(actual)
        drawdown_loss = abs(actual_drawdown - forecast.expected_max_drawdown / 100.0) / scale
        if len(point.future_session_offsets) != HORIZON_WEEKS:
            raise V33TrainingError(
                "turn evaluation requires the audited real trading-session sequence"
            )
        if not 1 <= forecast.predicted_high_week <= HORIZON_WEEKS:
            raise V33TrainingError("predicted high week is outside the 20-week horizon")
        if not 1 <= forecast.predicted_low_week <= HORIZON_WEEKS:
            raise V33TrainingError("predicted low week is outside the 20-week horizon")
        session_offsets = point.future_session_offsets
        actual_high_index = int(np.argmax(actual))
        actual_low_index = int(np.argmin(actual))
        predicted_high_index = forecast.predicted_high_week - 1
        predicted_low_index = forecast.predicted_low_week - 1
        high_error_days = float(
            abs(
                session_offsets[actual_high_index]
                - session_offsets[predicted_high_index]
            )
        )
        low_error_days = float(
            abs(
                session_offsets[actual_low_index]
                - session_offsets[predicted_low_index]
            )
        )
        price_turn_error = None
        if point.actual_price_turn_days is not None and forecast.price_turn_days is not None:
            price_turn_error = abs(point.actual_price_turn_days - forecast.price_turn_days)
        dif_turn_error = None
        if point.actual_dif_turn_days is not None and forecast.dif_turn_days is not None:
            dif_turn_error = abs(point.actual_dif_turn_days - forecast.dif_turn_days)
        turn_loss = statistics.fmean(
            [value / 100.0 for value in (high_error_days, low_error_days, price_turn_error, dif_turn_error) if value is not None]
        )
        position = _snap5(50.0 + p50[-1] / max(threshold, 0.01) * 25.0)
        oracle = _snap5(50.0 + actual[-1] / max(threshold, 0.01) * 25.0)
        turnover_loss = abs(position - oracle) / 100.0
        composite = (
            0.30 * wis
            + 0.15 * (0.7 * median_path + 0.3 * terminal)
            + 0.15 * brier
            + 0.20 * turn_loss
            + 0.10 * coverage_loss
            + 0.05 * drawdown_loss
            + 0.05 * turnover_loss
        )
        return {
            "composite_loss": composite,
            "wis_pinball_loss": wis,
            "median_path_loss": median_path,
            "terminal_loss": terminal,
            "brier_score": brier,
            "coverage": coverage,
            "coverage_interval_loss": coverage_loss,
            "drawdown_loss": drawdown_loss,
            "high_turn_error_days": high_error_days,
            "low_turn_error_days": low_error_days,
            "price_turn_error_days": price_turn_error,
            "dif_turn_error_days": dif_turn_error,
            "turn_loss": turn_loss,
            "turnover_loss": turnover_loss,
            "direction_correct": 1.0 if forecast.direction == actual_class else 0.0,
        }

    def _fit_state(
        self,
        market: str,
        points: Sequence[V33TrainingPoint],
        hyperparameters: Mapping[str, float],
        *,
        iteration_number: int,
        trained_through: date,
        parent_hash: str | None,
        version: str,
        optimizer_memory: Mapping[str, Any],
        validation_metrics: Mapping[str, float | None] | None = None,
        calibration_residuals: np.ndarray | None = None,
        prior_calibration_state: V33ModelState | None = None,
    ) -> V33ModelState:
        if not points:
            raise V33TrainingError("V3.3 requires matured training samples")
        feature_maps = [self._feature_map(point.snapshot) for point in points]
        candidate_names = sorted({name for mapping in feature_maps for name in mapping})
        # Entirely missing columns carry no information and are removed.  A
        # missing mask remains, so median imputation is explicit and auditable.
        names = tuple(
            name
            for name in candidate_names
            if any(mapping.get(name) is not None for mapping in feature_maps)
        )
        raw = np.asarray(
            [
                [np.nan if mapping.get(name) is None else float(mapping[name]) for name in names]
                for mapping in feature_maps
            ],
            dtype=float,
        )
        medians = np.nanmedian(raw, axis=0)
        if np.any(~np.isfinite(medians)):
            raise V33TrainingError("all-missing feature escaped schema filter")
        imputed = np.where(np.isnan(raw), medians, raw)
        means = np.mean(imputed, axis=0)
        scales = np.std(imputed, axis=0)
        scales = np.where(scales < 1e-9, 1.0, scales)
        x = (imputed - means) / scales
        y = np.asarray([point.future_path for point in points], dtype=float)
        if y.shape != (len(points), HORIZON_WEEKS):
            raise V33TrainingError("training targets are not complete 20-week paths")
        intercept = np.mean(y, axis=0)
        centered_y = y - intercept
        alpha = float(hyperparameters["ridge_alpha"])
        l1 = float(hyperparameters["l1_shrinkage"])

        daily_indices = tuple(
            index for index, name in enumerate(names) if _is_daily_corrector_feature(name)
        )
        weekly_indices = tuple(
            index for index, name in enumerate(names) if not _is_daily_corrector_feature(name)
        )
        if not weekly_indices:
            raise V33TrainingError("weekly main model has no usable features")
        if not daily_indices:
            raise V33TrainingError("100-session daily corrector has no usable features")

        # Stage 1: the weekly/fundamental model owns the complete 20-week path,
        # including its bounded nonlinear interactions.
        weekly_design = x[:, weekly_indices]
        weekly_coefficients = _ridge_coefficients(
            weekly_design,
            centered_y,
            alpha=alpha,
            l1=l1,
        )
        weekly_base = intercept + weekly_design @ weekly_coefficients

        local_stumps, weekly_nonlinear = self._fit_stumps(
            weekly_design,
            y - weekly_base,
            rounds=int(hyperparameters["boost_rounds"]),
            learning_rate=float(hyperparameters["learning_rate"]),
        )
        stumps = [
            replace(stump, feature_index=weekly_indices[stump.feature_index])
            for stump in local_stumps
        ]
        weekly_base = weekly_base + weekly_nonlinear

        # Stage 2: the complete 100-session daily curve fits only the residual
        # left by the complete weekly model.  Its inference contribution is
        # bounded and is exactly zero after week 4.
        daily_design = x[:, daily_indices]
        daily_coefficients = _ridge_coefficients(
            daily_design,
            y - weekly_base,
            alpha=alpha,
            l1=l1,
        )
        daily_correction = _bounded_daily_correction(
            daily_design @ daily_coefficients
        )
        base = weekly_base + daily_correction
        coefficients = np.zeros((x.shape[1], HORIZON_WEEKS), dtype=float)
        coefficients[np.asarray(weekly_indices), :] = weekly_coefficients
        coefficients[np.asarray(daily_indices), :] = daily_coefficients
        fitted_residuals = y - base
        if calibration_residuals is not None:
            if calibration_residuals.ndim != 2 or calibration_residuals.shape[1] != HORIZON_WEEKS:
                raise V33TrainingError("purged residual calibration must have 20 columns")
            residual_p10 = _quantile(calibration_residuals, 0.10)
            residual_p50 = _quantile(calibration_residuals, 0.50)
            residual_p90 = _quantile(calibration_residuals, 0.90)
            terminal_residuals = calibration_residuals[:, -1]
            residual_calibration_mode = "purged_oos"
        elif (
            prior_calibration_state is not None
            and str(
                prior_calibration_state.optimizer_memory.get(
                    "residual_calibration_mode", ""
                )
            )
            in {"purged_oos", "inherited_previous_oos"}
        ):
            residual_p10 = np.asarray(prior_calibration_state.residual_p10, dtype=float)
            residual_p50 = np.asarray(prior_calibration_state.residual_p50, dtype=float)
            residual_p90 = np.asarray(prior_calibration_state.residual_p90, dtype=float)
            terminal_residuals = np.asarray(prior_calibration_state.terminal_residuals, dtype=float)
            residual_calibration_mode = "inherited_previous_oos"
        else:
            # Initial small-sample bootstrap only.  The state is explicitly
            # marked and its calibration is replaced as soon as the first
            # purged validation window becomes available.
            residual_p10 = _quantile(fitted_residuals, 0.10)
            residual_p50 = _quantile(fitted_residuals, 0.50)
            residual_p90 = _quantile(fitted_residuals, 0.90)
            terminal_residuals = fitted_residuals[:, -1]
            residual_calibration_mode = (
                "small_sample_degraded"
                if len(points) < MIN_TRAIN_SAMPLES
                else "bootstrap_in_sample_degraded"
            )
        turn_lag = self._turn_lag(points)
        optimizer_payload = {
            **dict(optimizer_memory),
            "residual_calibration_mode": residual_calibration_mode,
            "architecture": "weekly_main_plus_bounded_daily_corrector",
            "weekly_feature_count": len(weekly_indices),
            "daily_feature_count": len(daily_indices),
            "daily_correction_weeks": DAILY_CORRECTION_WEEKS,
            "daily_correction_decay": list(DAILY_CORRECTION_DECAY),
            "daily_path_correction_max_abs": DAILY_PATH_CORRECTION_MAX_ABS,
            "daily_position_adjustment_limit": DAILY_POSITION_ADJUSTMENT_LIMIT,
            "daily_confidence_adjustment_limit": DAILY_CONFIDENCE_ADJUSTMENT_LIMIT,
            "training_mode": (
                "small_sample_degraded"
                if len(points) < MIN_TRAIN_SAMPLES
                else "standard"
            ),
            "minimum_standard_training_samples": MIN_TRAIN_SAMPLES,
            "minimum_market_cold_start_samples": self._minimum_initial_samples(market),
        }
        payload = {
            "market": market,
            "version": version,
            "iteration_number": iteration_number,
            "trained_through": trained_through.isoformat(),
            "parent_state_hash": parent_hash,
            "feature_names": names,
            "medians": medians.tolist(),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "intercept": intercept.tolist(),
            "coefficients": coefficients.tolist(),
            "stumps": [asdict(stump) for stump in stumps],
            "residual_p10": residual_p10.tolist(),
            "residual_p50": residual_p50.tolist(),
            "residual_p90": residual_p90.tolist(),
            "hyperparameters": dict(hyperparameters),
            "optimizer_memory": optimizer_payload,
            "turn_lag_days": turn_lag,
            "training_sample_count": len(points),
        }
        state_hash = _hash_payload(payload)
        return V33ModelState(
            market=market,
            version=version,
            iteration_number=iteration_number,
            trained_through=trained_through,
            parent_state_hash=parent_hash,
            feature_names=names,
            medians=tuple(float(value) for value in medians),
            means=tuple(float(value) for value in means),
            scales=tuple(float(value) for value in scales),
            intercept=tuple(float(value) for value in intercept),
            coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
            stumps=tuple(stumps),
            residual_p10=tuple(float(value) for value in residual_p10),
            residual_p50=tuple(float(value) for value in residual_p50),
            residual_p90=tuple(float(value) for value in residual_p90),
            terminal_residuals=tuple(float(value) for value in terminal_residuals),
            hyperparameters=dict(hyperparameters),
            optimizer_memory=optimizer_payload,
            validation_metrics=dict(validation_metrics or {}),
            turn_lag_days=turn_lag,
            training_sample_count=len(points),
            state_hash=state_hash,
        )

    @staticmethod
    def _fit_stumps(
        x: np.ndarray,
        residual: np.ndarray,
        *,
        rounds: int,
        learning_rate: float,
    ) -> tuple[list[ResidualStump], np.ndarray]:
        fitted = np.zeros_like(residual)
        stumps: list[ResidualStump] = []
        if len(x) < 4 or x.shape[1] == 0:
            return stumps, fitted
        working = residual.copy()
        for round_number in range(max(0, rounds)):
            # Deterministic feature screening keeps the full 100-day vector
            # available while avoiding an O(features*samples*horizon) search.
            target = np.mean(working, axis=1)
            covariance = np.abs(x.T @ target)
            order = np.argsort(-covariance, kind="stable")[: min(48, x.shape[1])]
            best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
            for feature_index in order:
                threshold = float(np.median(x[:, feature_index]))
                left_mask = x[:, feature_index] <= threshold
                if int(np.sum(left_mask)) < 2 or int(np.sum(~left_mask)) < 2:
                    continue
                left = np.mean(working[left_mask], axis=0) * learning_rate
                right = np.mean(working[~left_mask], axis=0) * learning_rate
                candidate = np.where(left_mask[:, None], left, right)
                loss = float(np.mean((working - candidate) ** 2))
                item = (loss, int(feature_index), threshold, left, right)
                if best is None or item[0] < best[0]:
                    best = item
            if best is None:
                break
            _, feature_index, threshold, left, right = best
            contribution = np.where(x[:, feature_index, None] <= threshold, left, right)
            working -= contribution
            fitted += contribution
            stumps.append(
                ResidualStump(
                    feature_index=feature_index,
                    threshold=threshold,
                    left=tuple(float(value) for value in left),
                    right=tuple(float(value) for value in right),
                )
            )
        return stumps, fitted

    def _feature_map(self, snapshot: FeatureSnapshot) -> dict[str, float | None]:
        key = id(snapshot)
        cached = self._feature_cache.get(key)
        if cached is None:
            cached = flatten_snapshot(snapshot)
            self._feature_cache[key] = cached
        return cached

    def _transform_one(self, snapshot: FeatureSnapshot, state: V33ModelState) -> np.ndarray:
        self._activate_state_cache(state)
        cached = self._active_transform_cache.get(id(snapshot))
        if cached is not None:
            return cached
        mapping = self._feature_map(snapshot)
        values = np.asarray(
            [
                state.medians[index] if mapping.get(name) is None else float(mapping[name])
                for index, name in enumerate(state.feature_names)
            ],
            dtype=float,
        )
        transformed = (values - np.asarray(state.means)) / np.asarray(state.scales)
        self._active_transform_cache[id(snapshot)] = transformed
        return transformed

    def _activate_state_cache(self, state: V33ModelState) -> None:
        if self._active_state_hash == state.state_hash:
            return
        self._active_state_hash = state.state_hash
        self._active_transform_cache = {}
        self._active_state_arrays = None

    def _state_arrays(
        self, state: V33ModelState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._activate_state_cache(state)
        if self._active_state_arrays is None:
            weekly_indices = np.asarray(
                [
                    index
                    for index, name in enumerate(state.feature_names)
                    if not _is_daily_corrector_feature(name)
                ],
                dtype=int,
            )
            daily_indices = np.asarray(
                [
                    index
                    for index, name in enumerate(state.feature_names)
                    if _is_daily_corrector_feature(name)
                ],
                dtype=int,
            )
            self._active_state_arrays = (
                np.asarray(state.coefficients, dtype=float),
                np.asarray(state.intercept, dtype=float),
                weekly_indices,
                daily_indices,
            )
        return self._active_state_arrays

    def _analogue_residuals(
        self,
        state: V33ModelState,
        snapshot: FeatureSnapshot,
        points: Sequence[V33TrainingPoint],
    ) -> list[np.ndarray]:
        if not points:
            return []
        eligible = [
            point
            for point in points
            if point.future_path is not None
            and point.outcome_available_date is not None
            and point.outcome_available_date <= snapshot.cutoff_date
        ]
        if not eligible:
            return []
        current = self._transform_one(snapshot, state)
        matrix = np.vstack(
            [self._transform_one(point.snapshot, state) for point in eligible]
        )
        weekly, daily = self._expected_matrix(state, matrix)
        expected = weekly + daily
        outcomes = np.asarray([point.future_path for point in eligible], dtype=float)
        residuals = outcomes - expected
        distances = np.sqrt(np.mean((matrix - current) ** 2, axis=1))
        order = np.argsort(distances, kind="stable")[: min(40, len(eligible))]
        return [residuals[int(index)] for index in order]

    def _residual_matrix(
        self,
        state: V33ModelState,
        points: Sequence[V33TrainingPoint],
    ) -> np.ndarray:
        eligible = [point for point in points if point.future_path is not None]
        if not eligible:
            return np.empty((0, HORIZON_WEEKS), dtype=float)
        matrix = np.vstack(
            [self._transform_one(point.snapshot, state) for point in eligible]
        )
        weekly, daily = self._expected_matrix(state, matrix)
        outcomes = np.asarray([point.future_path for point in eligible], dtype=float)
        return outcomes - (weekly + daily)

    def _expected_components(
        self, state: V33ModelState, vector: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        weekly, daily = self._expected_matrix(state, np.asarray(vector, dtype=float)[None, :])
        return weekly[0], daily[0]

    def _expected_matrix(
        self, state: V33ModelState, matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(state.feature_names):
            raise V33TrainingError("standardised feature matrix has invalid shape")
        coefficients, intercept, weekly_indices, daily_indices = self._state_arrays(state)
        weekly = np.broadcast_to(intercept, (len(values), HORIZON_WEEKS)).copy()
        if weekly_indices.size:
            weekly += values[:, weekly_indices] @ coefficients[weekly_indices, :]
        daily_raw = (
            np.zeros((len(values), HORIZON_WEEKS), dtype=float)
            if not daily_indices.size
            else values[:, daily_indices] @ coefficients[daily_indices, :]
        )
        daily = _bounded_daily_correction(daily_raw)
        for stump in state.stumps:
            if _is_daily_corrector_feature(state.feature_names[stump.feature_index]):
                raise V33TrainingError("daily feature escaped into weekly residual stumps")
            left = np.asarray(stump.left, dtype=float)
            right = np.asarray(stump.right, dtype=float)
            weekly += np.where(
                values[:, stump.feature_index, None] <= stump.threshold,
                left,
                right,
            )
        return weekly, daily

    def _expected_from_vector(self, state: V33ModelState, vector: np.ndarray) -> np.ndarray:
        weekly, daily = self._expected_components(state, vector)
        return weekly + daily

    def _validation_metrics(
        self,
        state: V33ModelState,
        validation: Sequence[V33TrainingPoint],
        calibration: Sequence[V33TrainingPoint],
    ) -> dict[str, float | None]:
        rows = [self.evaluate(point, self.predict(state, point.snapshot, calibration_points=calibration)) for point in validation]
        if not rows:
            return {"composite_loss": None}
        numeric_keys = rows[0].keys()
        metrics: dict[str, float | None] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if row[key] is not None]
            metrics[key] = statistics.fmean(values) if values else None
        metrics["sample_count"] = float(len(rows))
        return metrics

    @staticmethod
    def _purged_split(
        points: Sequence[V33TrainingPoint],
    ) -> tuple[list[V33TrainingPoint], list[V33TrainingPoint]]:
        ordered = sorted(points, key=lambda point: point.cutoff_date)
        validation_size = min(VALIDATION_SAMPLES, max(4, len(ordered) // 4))
        validation_start = len(ordered) - validation_size
        training_end = validation_start - PURGE_WEEKS
        if training_end < MIN_TRAIN_SAMPLES:
            training_end = MIN_TRAIN_SAMPLES
            validation_start = min(len(ordered), training_end + PURGE_WEEKS)
        train = ordered[:training_end]
        validation = ordered[validation_start:]
        if not validation:
            raise V33TrainingError("purged split has no validation samples")
        if train[-1].outcome_available_date is not None and train[-1].outcome_available_date > validation[0].cutoff_date:
            raise V33TrainingError("20-week purge/embargo invariant failed")
        return train, validation

    @staticmethod
    def _promotion_gate(
        incumbent: Mapping[str, float | None], challenger: Mapping[str, float | None]
    ) -> tuple[bool, str]:
        incumbent_loss = incumbent.get("composite_loss")
        challenger_loss = challenger.get("composite_loss")
        if incumbent_loss is None or challenger_loss is None:
            return False, "missing_validation_metric"
        improvement = (float(incumbent_loss) - float(challenger_loss)) / max(abs(float(incumbent_loss)), 1e-9)
        if improvement < 0.002:
            return False, "composite_loss_not_improved"
        incumbent_direction = float(incumbent.get("direction_correct") or 0.0)
        challenger_direction = float(challenger.get("direction_correct") or 0.0)
        if challenger_direction < incumbent_direction - 0.02:
            return False, "direction_accuracy_regressed"
        incumbent_coverage = float(incumbent.get("coverage") or 0.0)
        challenger_coverage = float(challenger.get("coverage") or 0.0)
        if abs(challenger_coverage - 0.80) > abs(incumbent_coverage - 0.80) + 0.03:
            return False, "interval_coverage_regressed"
        return True, "purged_oos_gate_passed"

    @staticmethod
    def _initial_hyperparameters(market: str) -> dict[str, float]:
        return {
            "ridge_alpha": 12.0 if market == "399006" else 10.0,
            "l1_shrinkage": 0.0002,
            "boost_rounds": 8.0,
            "learning_rate": 0.08,
        }

    @staticmethod
    def _challenger(
        current: Mapping[str, float], optimizer: Mapping[str, Any]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        names = ("ridge_alpha", "l1_shrinkage", "boost_rounds", "learning_rate")
        coordinate = int(optimizer.get("coordinate", 0)) % len(names)
        direction = 1 if int(optimizer.get("direction", 1)) >= 0 else -1
        name = names[coordinate]
        candidate = dict(current)
        if name == "ridge_alpha":
            candidate[name] = max(1.0, min(40.0, candidate[name] * (1.0 + direction * 0.10)))
        elif name == "l1_shrinkage":
            candidate[name] = max(0.0, min(0.005, candidate[name] + direction * 0.00005))
        elif name == "boost_rounds":
            candidate[name] = float(max(2, min(20, int(candidate[name]) + direction)))
        else:
            candidate[name] = max(0.02, min(0.20, candidate[name] + direction * 0.01))
        next_coordinate = (coordinate + 1) % len(names)
        next_direction = -direction if next_coordinate == 0 else direction
        return candidate, {
            **dict(optimizer),
            "coordinate": next_coordinate,
            "direction": next_direction,
            "last_coordinate": name,
        }

    @staticmethod
    def _turn_lag(points: Sequence[V33TrainingPoint]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {"top": [], "bottom": []}
        for point in points:
            if (
                point.actual_dif_turn_days is not None
                and point.actual_price_turn_days is not None
                and point.actual_dif_turn_kind == point.actual_price_turn_kind
                and point.actual_dif_turn_kind in buckets
            ):
                buckets[point.actual_dif_turn_kind].append(
                    point.actual_price_turn_days - point.actual_dif_turn_days
                )
        return {
            kind: float(np.median(values)) if values else 0.0
            for kind, values in buckets.items()
        }


def state_to_payload(state: V33ModelState) -> dict[str, Any]:
    """JSON-safe exact state for the V3.3 repository/model artifact."""

    payload = asdict(state)
    payload["trained_through"] = state.trained_through.isoformat()
    # Normalise tuples and other dataclass containers to the exact mutable JSON
    # representation a SQLite JSON column/artifact will store.
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def state_from_payload(payload: Mapping[str, Any]) -> V33ModelState:
    """Rehydrate an exact state and verify its immutable audit hash."""

    stumps = tuple(ResidualStump(**row) for row in payload["stumps"])
    state = V33ModelState(
        market=str(payload["market"]),
        version=str(payload["version"]),
        iteration_number=int(payload["iteration_number"]),
        trained_through=date.fromisoformat(str(payload["trained_through"])),
        parent_state_hash=payload.get("parent_state_hash"),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        medians=tuple(float(value) for value in payload["medians"]),
        means=tuple(float(value) for value in payload["means"]),
        scales=tuple(float(value) for value in payload["scales"]),
        intercept=tuple(float(value) for value in payload["intercept"]),
        coefficients=tuple(tuple(float(item) for item in row) for row in payload["coefficients"]),
        stumps=stumps,
        residual_p10=tuple(float(value) for value in payload["residual_p10"]),
        residual_p50=tuple(float(value) for value in payload["residual_p50"]),
        residual_p90=tuple(float(value) for value in payload["residual_p90"]),
        terminal_residuals=tuple(float(value) for value in payload["terminal_residuals"]),
        hyperparameters=dict(payload["hyperparameters"]),
        optimizer_memory=dict(payload["optimizer_memory"]),
        validation_metrics=dict(payload.get("validation_metrics") or {}),
        turn_lag_days={str(key): float(value) for key, value in payload["turn_lag_days"].items()},
        training_sample_count=int(payload["training_sample_count"]),
        state_hash=str(payload["state_hash"]),
    )
    # State hashes intentionally cover learned parameters and optimizer state,
    # not validation display fields/terminal residual samples.
    expected = _hash_payload(
        {
            "market": state.market,
            "version": state.version,
            "iteration_number": state.iteration_number,
            "trained_through": state.trained_through.isoformat(),
            "parent_state_hash": state.parent_state_hash,
            "feature_names": state.feature_names,
            "medians": state.medians,
            "means": state.means,
            "scales": state.scales,
            "intercept": state.intercept,
            "coefficients": state.coefficients,
            "stumps": [asdict(stump) for stump in state.stumps],
            "residual_p10": state.residual_p10,
            "residual_p50": state.residual_p50,
            "residual_p90": state.residual_p90,
            "hyperparameters": dict(state.hyperparameters),
            "optimizer_memory": dict(state.optimizer_memory),
            "turn_lag_days": dict(state.turn_lag_days),
            "training_sample_count": state.training_sample_count,
        }
    )
    if expected != state.state_hash:
        raise V33TrainingError("V3.3 model artifact state_hash mismatch")
    return state
