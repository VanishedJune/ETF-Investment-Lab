"""Leakage-safe, immutable weekly feature snapshots for the V2 model boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, TypeVar

from ..indicators.calculator import (
    IndicatorCalculator,
    IndicatorParameters,
    PricePoint,
)
from .domain import (
    PeriodBar,
    QualityIssue,
    QualityReport,
    VolumeAvailability,
)


FEATURE_SET_VERSION = "weekly-features-v2"
WARMUP_WEEKS = 60
_NDX = "NDX"
_SIGNAL_RULE_VERSION = "deterministic_weekly_v2"
_GOLDEN_VALUATION_MAX = Decimal("30")
_BLACK_VALUATION_MIN = Decimal("70")
_SIGNAL_REQUIRED_CONFIRMATIONS = 5
_VALUATION_MIN_OBSERVATIONS = 20
_VALUATION_RULE_VERSION = "valuation-percentile-midrank-v1"
_VALUATION_TIE_POLICY = "midrank"
_VALUATION_PERCENTILE_FORMULA = (
    "100*(count_less+count_equal/2)/observation_count"
)

_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(Mapping[_K, _V]):
    """A read-only mapping backed only by immutable/private state."""

    __slots__ = ("__items", "__mapping")

    def __init__(
        self,
        values: Mapping[_K, _V] | Iterable[tuple[_K, _V]],
    ) -> None:
        copied = {
            key: _freeze_nested_value(value)
            for key, value in dict(values).items()
        }
        object.__setattr__(self, "_FrozenDict__items", tuple(copied.items()))
        object.__setattr__(
            self,
            "_FrozenDict__mapping",
            MappingProxyType(copied),
        )

    def __getitem__(self, key: _K) -> _V:
        return self.__mapping[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self.__mapping)

    def __len__(self) -> int:
        return len(self.__mapping)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self.__items)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.__items) == dict(other.items())

    def __hash__(self) -> int:
        return hash(frozenset(self.__items))


def _freeze_nested_value(value: _V) -> _V:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(
            _freeze_nested_value(item) for item in value
        )  # type: ignore[return-value]
    if isinstance(value, list):
        return tuple(
            _freeze_nested_value(item) for item in value
        )  # type: ignore[return-value]
    if isinstance(value, (set, frozenset)):
        return frozenset(
            _freeze_nested_value(item) for item in value
        )  # type: ignore[return-value]
    return value


@dataclass(frozen=True, slots=True)
class ValuationPoint:
    """One valuation observation known on its publication date."""

    valuation_date: date
    value: Decimal
    estimated: bool = False

    def __post_init__(self) -> None:
        if not self.value.is_finite():
            raise ValueError("Valuation value must be finite")
        if type(self.estimated) is not bool:
            raise ValueError("estimated must be a boolean")


class FutureDataError(ValueError):
    """A strict cutoff violation carrying its auditable blocking issue."""

    def __init__(self, quality_issue: QualityIssue) -> None:
        self.quality_issue = quality_issue
        super().__init__(quality_issue.message)


FeatureScalar = Decimal | str | int | bool | None | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """A deeply immutable feature result with canonical JSON serialization."""

    instrument_code: str
    cutoff_date: date
    source_data_max_date: date | None
    feature_set_version: str
    features: FrozenDict[str, FeatureScalar]
    availability: FrozenDict[str, str]
    quality_report: QualityReport
    audit_fields: FrozenDict[str, str | int]
    source_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_fields": dict(self.audit_fields),
            "availability": dict(self.availability),
            "cutoff_date": self.cutoff_date.isoformat(),
            "feature_set_version": self.feature_set_version,
            "features": {
                key: _json_value(value)
                for key, value in self.features.items()
            },
            "instrument_code": self.instrument_code,
            "quality_report": {
                "is_publishable": self.quality_report.is_publishable,
                "issues": [
                    {
                        "code": issue.code,
                        "end_date": (
                            issue.end_date.isoformat()
                            if issue.end_date is not None
                            else None
                        ),
                        "message": issue.message,
                        "severity": issue.severity,
                        "start_date": (
                            issue.start_date.isoformat()
                            if issue.start_date is not None
                            else None
                        ),
                    }
                    for issue in self.quality_report.issues
                ],
                "volume_availability": (
                    self.quality_report.volume_availability
                ),
            },
            "source_data_max_date": (
                self.source_data_max_date.isoformat()
                if self.source_data_max_date is not None
                else None
            ),
            "source_hash": self.source_hash,
        }

    def to_json(self) -> str:
        return _canonical_json_dumps(self.to_dict())


def _json_value(value: FeatureScalar) -> object:
    return _canonical_json_value(value)


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("Only finite Decimal values can be serialized")
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _canonical_json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_json_value(item) for item in value]
    return value


def _canonical_json_dumps(value: object) -> str:
    return json.dumps(
        _canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _issue(
    code: str,
    severity: str,
    start_date: date | None,
    end_date: date | None,
    message: str,
) -> QualityIssue:
    return QualityIssue(  # type: ignore[arg-type]
        code=code,
        severity=severity,
        start_date=start_date,
        end_date=end_date,
        message=message,
    )


def _iso_key(day: date) -> tuple[int, int]:
    iso_year, iso_week, _ = day.isocalendar()
    return iso_year, iso_week


def _validate_strict_cutoff(
    bars: tuple[PeriodBar, ...],
    valuations: tuple[ValuationPoint, ...],
    cutoff_date: date,
) -> None:
    future_bars = sorted(
        bar.period_end for bar in bars if bar.period_end > cutoff_date
    )
    if future_bars:
        raise FutureDataError(
            _issue(
                "WEEKLY_BAR_AFTER_CUTOFF",
                "blocking",
                future_bars[0],
                future_bars[-1],
                "Completed weekly bars cannot extend beyond cutoff_date.",
            )
        )
    future_valuations = sorted(
        point.valuation_date
        for point in valuations
        if point.valuation_date > cutoff_date
    )
    if future_valuations:
        raise FutureDataError(
            _issue(
                "VALUATION_AFTER_CUTOFF",
                "blocking",
                future_valuations[0],
                future_valuations[-1],
                "Valuation observations cannot extend beyond cutoff_date.",
            )
        )


def _ordered_bars(bars: tuple[PeriodBar, ...]) -> tuple[PeriodBar, ...]:
    ordered = tuple(sorted(bars, key=lambda bar: bar.period_end))
    seen_dates: set[date] = set()
    seen_weeks: set[tuple[int, int]] = set()
    for bar in ordered:
        if bar.timeframe != "weekly":
            raise ValueError("build_feature_snapshot accepts weekly PeriodBar rows only")
        if bar.period_end in seen_dates:
            raise ValueError(f"Duplicate weekly period_end: {bar.period_end}")
        if _iso_key(bar.period_end) in seen_weeks:
            raise ValueError(
                f"More than one weekly bar exists for ISO week {_iso_key(bar.period_end)}"
            )
        if bar.period_start > bar.period_end:
            raise ValueError("Weekly period_start cannot be later than period_end")
        if (
            bar.close_price is None
            or not bar.close_price.is_finite()
            or bar.close_price <= 0
        ):
            raise ValueError(
                f"Weekly close at {bar.period_end.isoformat()} must be finite and positive"
            )
        seen_dates.add(bar.period_end)
        seen_weeks.add(_iso_key(bar.period_end))
    return ordered


def _ordered_valuations(
    valuations: tuple[ValuationPoint, ...],
) -> tuple[ValuationPoint, ...]:
    ordered = tuple(sorted(valuations, key=lambda point: point.valuation_date))
    if len({point.valuation_date for point in ordered}) != len(ordered):
        raise ValueError("Valuation dates must be unique")
    return ordered


def _gap_issues(
    bars: tuple[PeriodBar, ...],
    cutoff_date: date,
    expected_week_ends: Iterable[date] | None,
    expected_trade_dates: Iterable[date] | None,
) -> tuple[QualityIssue, ...]:
    if expected_week_ends is None:
        return (
            _issue(
                "MISSING_GAP_CONTEXT",
                "blocking",
                bars[0].period_end if bars else None,
                cutoff_date,
                "Expected completed week ends are required to audit long gaps.",
            ),
        )
    expected = tuple(
        sorted({day for day in expected_week_ends if day <= cutoff_date})
    )
    actual_by_week = {_iso_key(bar.period_end): bar.period_end for bar in bars}
    expected_dates_by_week: dict[tuple[int, int], list[date]] = {}
    for expected_end in expected:
        expected_dates_by_week.setdefault(
            _iso_key(expected_end),
            [],
        ).append(expected_end)
    trade_dates_by_week: dict[tuple[int, int], list[date]] = {}
    if expected_trade_dates is not None:
        for trade_date in sorted(
            {day for day in expected_trade_dates if day <= cutoff_date}
        ):
            trade_dates_by_week.setdefault(_iso_key(trade_date), []).append(
                trade_date
            )

    issues: list[QualityIssue] = []
    expected_by_week: dict[tuple[int, int], date] = {}
    invalid_calendar_weeks: set[tuple[int, int]] = set()
    for week, week_ends in expected_dates_by_week.items():
        expected_end = max(week_ends)
        expected_by_week[week] = expected_end
        if len(week_ends) > 1:
            issues.append(
                _issue(
                    "MULTIPLE_EXPECTED_WEEK_ENDS",
                    "blocking",
                    min(week_ends),
                    max(week_ends),
                    "Only one expected endpoint is allowed per ISO week.",
                )
            )
        if expected_trade_dates is None:
            continue
        sessions = trade_dates_by_week.get(week, ())
        if not sessions:
            invalid_calendar_weeks.add(week)
            issues.append(
                _issue(
                    "CALENDAR_WEEK_WITHOUT_SESSIONS",
                    "blocking",
                    expected_end,
                    expected_end,
                    "Expected week end has no sessions in the supplied trading calendar.",
                )
            )
            continue
        calendar_end = max(sessions)
        if expected_end != calendar_end:
            invalid_calendar_weeks.add(week)
            issues.append(
                _issue(
                    "EXPECTED_WEEK_END_MISMATCH",
                    "blocking",
                    min(expected_end, calendar_end),
                    max(expected_end, calendar_end),
                    "Expected week end must equal the latest session in its ISO week.",
                )
            )

    for week, expected_end in expected_by_week.items():
        actual_end = actual_by_week.get(week)
        if actual_end is None:
            session_count = len(trade_dates_by_week.get(week, ()))
            calendar_confirmed = expected_trade_dates is not None
            if calendar_confirmed and week in invalid_calendar_weeks:
                continue
            is_short_week = calendar_confirmed and session_count <= 2
            issues.append(
                _issue(
                    (
                        "SHORT_EXPECTED_WEEK_GAP"
                        if is_short_week
                        else "MISSING_COMPLETED_WEEK"
                    ),
                    "warning" if is_short_week else "blocking",
                    expected_end,
                    expected_end,
                    (
                        "A short exchange-calendar week is absent from the input."
                        if is_short_week
                        else "An explicitly completed weekly period is absent from the input."
                    ),
                )
            )
            continue
        if actual_end != expected_end:
            if actual_end > expected_end:
                issues.append(
                    _issue(
                        "WEEK_END_AFTER_EXPECTED",
                        "blocking",
                        expected_end,
                        actual_end,
                        "Weekly endpoint is later than the explicit exchange-calendar endpoint.",
                    )
                )
                continue
            missing_sessions = [
                day
                for day in trade_dates_by_week.get(week, ())
                if actual_end < day <= expected_end
            ]
            calendar_confirmed = expected_trade_dates is not None
            if calendar_confirmed and week in invalid_calendar_weeks:
                continue
            if calendar_confirmed and not missing_sessions:
                issues.append(
                    _issue(
                        "CALENDAR_ENDPOINT_INCONSISTENT",
                        "blocking",
                        actual_end,
                        expected_end,
                        "Expected endpoint is not supported by later sessions in the supplied calendar.",
                    )
                )
                continue
            blocking = not calendar_confirmed or len(missing_sessions) > 2
            issues.append(
                _issue(
                    "WEEK_END_MISMATCH",
                    "blocking" if blocking else "warning",
                    min(actual_end, expected_end),
                    max(actual_end, expected_end),
                    "Weekly endpoint does not match the explicit exchange calendar.",
                )
            )
    for week, actual_end in actual_by_week.items():
        if week not in expected_by_week:
            issues.append(
                _issue(
                    "UNEXPECTED_COMPLETED_WEEK",
                    "blocking",
                    actual_end,
                    actual_end,
                    "A weekly bar is absent from the explicit completed-period sequence.",
                )
            )
    return tuple(issues)


def _valuation_features(
    valuations: tuple[ValuationPoint, ...],
) -> tuple[
    Decimal | None,
    Decimal | None,
    int,
    str,
    tuple[QualityIssue, ...],
]:
    observation_count = len(valuations)
    if not valuations:
        return (
            None,
            None,
            0,
            "missing",
            (
                _issue(
                    "VALUATION_UNAVAILABLE",
                    "warning",
                    None,
                    None,
                    "No valuation observation is visible at cutoff_date.",
                ),
            ),
        )
    latest = valuations[-1]
    if observation_count < _VALUATION_MIN_OBSERVATIONS:
        return (
            latest.value,
            None,
            observation_count,
            "insufficient_history",
            (
                _issue(
                    "INSUFFICIENT_VALUATION_HISTORY",
                    "warning",
                    valuations[0].valuation_date,
                    latest.valuation_date,
                    (
                        f"At least {_VALUATION_MIN_OBSERVATIONS} visible "
                        "valuation observations are required for a percentile."
                    ),
                ),
            ),
        )
    count_less = sum(
        point.value < latest.value for point in valuations
    )
    count_equal = sum(
        point.value == latest.value for point in valuations
    )
    percentile = (
        Decimal("100")
        * (
            Decimal(count_less)
            + Decimal(count_equal) / Decimal("2")
        )
        / Decimal(observation_count)
    )
    return (
        latest.value,
        percentile,
        observation_count,
        "available",
        (),
    )


def _volume_feature(
    bars: tuple[PeriodBar, ...],
    instrument_code: str,
    volume_window: int,
) -> tuple[
    Decimal | None,
    str,
    VolumeAvailability,
    tuple[QualityIssue, ...],
]:
    if instrument_code == _NDX:
        return (
            None,
            "not_available_for_direct_index",
            "not_available_for_direct_index",
            (
                _issue(
                    "DIRECT_INDEX_VOLUME_UNAVAILABLE",
                    "warning",
                    bars[0].period_end if bars else None,
                    bars[-1].period_end if bars else None,
                    "Reliable direct NDX volume is unavailable; volume features are excluded.",
                ),
            ),
        )
    window = bars[-volume_window:]
    missing = [
        bar.period_end
        for bar in window
        if (
            bar.volume is None
            or not bar.volume.is_finite()
            or bar.volume < 0
        )
    ]
    if len(window) < volume_window or missing:
        problem_dates = missing or [bar.period_end for bar in window]
        return (
            None,
            "missing_required_volume",
            "partially_unavailable",
            (
                _issue(
                    "INCOMPLETE_VOLUME_WINDOW",
                    "blocking",
                    problem_dates[0] if problem_dates else None,
                    problem_dates[-1] if problem_dates else None,
                    f"The {volume_window}-week direct-index volume window must be complete.",
                ),
            ),
        )
    volumes = [bar.volume for bar in window]
    assert all(volume is not None for volume in volumes)
    average = sum(volumes, Decimal("0")) / Decimal(volume_window)
    if average == 0:
        return (
            None,
            "zero_average",
            "partially_unavailable",
            (
                _issue(
                    "ZERO_VOLUME_AVERAGE",
                    "blocking",
                    window[0].period_end,
                    window[-1].period_end,
                    f"The {volume_window}-week volume average is zero.",
                ),
            ),
        )
    current = volumes[-1]
    assert current is not None
    return current / average, "available", "available", ()


def _signal_features(
    snapshots: tuple[object, ...],
    close: Decimal | None,
    valuation_percentile: Decimal | None,
    *,
    is_available: bool,
) -> dict[str, FeatureScalar]:
    unavailable: dict[str, FeatureScalar] = {
        "golden_point": None,
        "golden_strength": None,
        "golden_point_reasons": None,
        "black_point": None,
        "black_strength": None,
        "black_point_reasons": None,
    }
    if (
        not is_available
        or valuation_percentile is None
        or len(snapshots) < 2
        or close is None
    ):
        return unavailable

    previous = snapshots[-2]
    current = snapshots[-1]
    previous_values = previous.values  # type: ignore[attr-defined]
    current_values = current.values  # type: ignore[attr-defined]
    previous_dif = previous_values["dif"]
    previous_dea = previous_values["dea"]
    current_dif = current_values["dif"]
    current_dea = current_values["dea"]
    previous_histogram = previous_values["macd_histogram"]
    current_histogram = current_values["macd_histogram"]
    ma20 = current_values["ma_20"]
    if any(
        value is None
        for value in (
            previous_dif,
            previous_dea,
            current_dif,
            current_dea,
            previous_histogram,
            current_histogram,
            ma20,
        )
    ):
        return unavailable

    crossed_up = previous_dif <= previous_dea and current_dif > current_dea
    crossed_down = previous_dif >= previous_dea and current_dif < current_dea
    golden_checks = (
        ("dif_crossed_above_dea", crossed_up),
        ("dif_at_or_below_zero", current_dif <= 0),
        ("macd_histogram_rising", current_histogram > previous_histogram),
        (
            "valuation_percentile_low",
            valuation_percentile is not None
            and valuation_percentile <= _GOLDEN_VALUATION_MAX,
        ),
        ("price_at_or_above_ma20", close >= ma20),
    )
    black_checks = (
        ("dif_crossed_below_dea", crossed_down),
        ("dif_at_or_above_zero", current_dif >= 0),
        ("macd_histogram_falling", current_histogram < previous_histogram),
        (
            "valuation_percentile_high",
            valuation_percentile is not None
            and valuation_percentile >= _BLACK_VALUATION_MIN,
        ),
        ("price_below_ma20", close < ma20),
    )
    golden_reasons = (
        tuple(reason for reason, passed in golden_checks if passed)
        if crossed_up
        else ()
    )
    black_reasons = (
        tuple(reason for reason, passed in black_checks if passed)
        if crossed_down
        else ()
    )
    golden_strength = len(golden_reasons)
    black_strength = len(black_reasons)
    return {
        "golden_point": (
            golden_strength >= _SIGNAL_REQUIRED_CONFIRMATIONS
        ),
        "golden_strength": golden_strength,
        "golden_point_reasons": golden_reasons,
        "black_point": black_strength >= _SIGNAL_REQUIRED_CONFIRMATIONS,
        "black_strength": black_strength,
        "black_point_reasons": black_reasons,
    }


def _signal_rule_configuration() -> dict[str, object]:
    return {
        "black_valuation_min": _BLACK_VALUATION_MIN,
        "golden_valuation_max": _GOLDEN_VALUATION_MAX,
        "required_confirmations": _SIGNAL_REQUIRED_CONFIRMATIONS,
        "rule_version": _SIGNAL_RULE_VERSION,
        "uses_dif_dea_cross": True,
        "uses_histogram_direction": True,
        "uses_ma20_position": True,
        "uses_zero_axis_position": True,
    }


def _valuation_rule_configuration() -> dict[str, object]:
    return {
        "minimum_observations": _VALUATION_MIN_OBSERVATIONS,
        "percentile_formula": _VALUATION_PERCENTILE_FORMULA,
        "rule_version": _VALUATION_RULE_VERSION,
        "tie_policy": _VALUATION_TIE_POLICY,
    }


def _quality_hash_payload(report: QualityReport) -> dict[str, object]:
    return {
        "is_publishable": report.is_publishable,
        "issues": [
            {
                "code": issue.code,
                "end_date": (
                    issue.end_date.isoformat()
                    if issue.end_date is not None
                    else None
                ),
                "message": issue.message,
                "severity": issue.severity,
                "start_date": (
                    issue.start_date.isoformat()
                    if issue.start_date is not None
                    else None
                ),
            }
            for issue in report.issues
        ],
        "volume_availability": report.volume_availability,
    }


def _source_hash(
    *,
    bars: tuple[PeriodBar, ...],
    valuations: tuple[ValuationPoint, ...],
    cutoff_date: date,
    instrument_code: str,
    expected_week_ends: Iterable[date] | None,
    expected_trade_dates: Iterable[date] | None,
    indicator_formula_label: str,
    indicator_parameters: Mapping[str, object],
    quality_report: QualityReport,
) -> str:
    payload = {
        "bars": [
            {
                "adjusted_close": bar.adjusted_close_price,
                "close": bar.close_price,
                "end": bar.period_end.isoformat(),
                "high": bar.high_price,
                "low": bar.low_price,
                "open": bar.open_price,
                "price_source": bar.price_source,
                "source": bar.source,
                "start": bar.period_start.isoformat(),
                "turnover": bar.turnover,
                "volume": bar.volume,
                "volume_source": bar.volume_source,
            }
            for bar in bars
        ],
        "cutoff_date": cutoff_date.isoformat(),
        "expected_week_ends": (
            sorted(day.isoformat() for day in expected_week_ends)
            if expected_week_ends is not None
            else None
        ),
        "expected_trade_dates": (
            sorted(day.isoformat() for day in expected_trade_dates)
            if expected_trade_dates is not None
            else None
        ),
        "feature_set_version": FEATURE_SET_VERSION,
        "indicator_formula_label": indicator_formula_label,
        "indicator_parameters": dict(indicator_parameters),
        "instrument_code": instrument_code,
        "quality_report": _quality_hash_payload(quality_report),
        "signal_rules": _signal_rule_configuration(),
        "valuation_rules": _valuation_rule_configuration(),
        "valuations": [
            {
                "date": point.valuation_date.isoformat(),
                "value": point.value,
                "estimated": point.estimated,
            }
            for point in valuations
        ],
    }
    encoded = _canonical_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_cutoff_dates(
    values: Iterable[date] | None,
    cutoff_date: date,
) -> tuple[date, ...] | None:
    if values is None:
        return None
    return tuple(sorted({day for day in values if day <= cutoff_date}))


def build_feature_snapshot(
    completed_weekly_bars: Iterable[PeriodBar],
    valuations: Iterable[ValuationPoint],
    cutoff_date: date,
    instrument_code: str,
    *,
    expected_week_ends: Iterable[date] | None = None,
    expected_trade_dates: Iterable[date] | None = None,
) -> FeatureSnapshot:
    """Build one strict-cutoff weekly snapshot without database or network I/O."""

    if not instrument_code:
        raise ValueError("instrument_code is required")
    raw_bars = tuple(completed_weekly_bars)
    raw_valuations = tuple(valuations)
    _validate_strict_cutoff(raw_bars, raw_valuations, cutoff_date)
    bars = _ordered_bars(raw_bars)
    valuation_rows = _ordered_valuations(raw_valuations)
    expected_end_rows = _canonical_cutoff_dates(
        expected_week_ends,
        cutoff_date,
    )
    expected_trade_date_rows = _canonical_cutoff_dates(
        expected_trade_dates,
        cutoff_date,
    )
    issues = list(
        _gap_issues(
            bars,
            cutoff_date,
            expected_end_rows,
            expected_trade_date_rows,
        )
    )
    if not bars:
        issues.append(
            _issue(
                "NO_COMPLETED_WEEKLY_DATA",
                "blocking",
                None,
                cutoff_date,
                "At least one completed weekly bar is required.",
            )
        )
    if len(bars) < WARMUP_WEEKS:
        issues.append(
            _issue(
                "INSUFFICIENT_WARMUP",
                "blocking",
                bars[0].period_end if bars else None,
                bars[-1].period_end if bars else None,
                f"At least {WARMUP_WEEKS} completed weekly bars are required.",
            )
        )

    (
        valuation_value,
        valuation_percentile,
        valuation_observation_count,
        valuation_status,
        valuation_issues,
    ) = _valuation_features(valuation_rows)
    issues.extend(valuation_issues)
    parameters = IndicatorParameters()
    volume_ratio, volume_status, volume_availability, volume_issues = (
        _volume_feature(
            bars,
            instrument_code,
            parameters.volume_ma_period,
        )
    )
    issues.extend(volume_issues)

    calculation = IndicatorCalculator(parameters).calculate(
        [
            PricePoint(
                trade_date=bar.period_end,
                close=bar.close_price,  # type: ignore[arg-type]
                volume=bar.volume,
            )
            for bar in bars
        ],
        "weekly",
    )
    latest_values: Mapping[str, Decimal | None] = (
        calculation.snapshots[-1].values
        if calculation.snapshots
        else {}
    )
    dif = latest_values.get("dif")
    dea = latest_values.get("dea")
    features: dict[str, FeatureScalar] = {
        "ema_12": latest_values.get("ema_12"),
        "ema_26": latest_values.get("ema_26"),
        "dif": dif,
        "dip": dif,
        "dea": dea,
        "eda": dea,
        "macd_histogram": latest_values.get("macd_histogram"),
        "rsi_6": latest_values.get("rsi_6"),
        "ma_20": latest_values.get("ma_20"),
        "ma_60": latest_values.get("ma_60"),
        "volume_ratio": volume_ratio,
        "volatility_20": latest_values.get("volatility_20"),
        "volatility": latest_values.get("volatility_20"),
        "current_drawdown": latest_values.get("current_drawdown"),
        "drawdown": latest_values.get("current_drawdown"),
        "running_drawdown": latest_values.get("running_drawdown"),
        "valuation_value": valuation_value,
        "valuation_percentile": valuation_percentile,
        "valuation_observation_count": valuation_observation_count,
    }
    technical_status = (
        "available"
        if len(bars) >= WARMUP_WEEKS
        else "insufficient_warmup"
    )
    if technical_status != "available":
        signal_status = technical_status
    elif valuation_status == "missing":
        signal_status = "missing_valuation"
    elif valuation_status != "available":
        signal_status = "insufficient_valuation_history"
    else:
        signal_status = "available"
    close = bars[-1].close_price if bars else None
    features.update(
        _signal_features(
            calculation.snapshots,
            close,
            valuation_percentile,
            is_available=signal_status == "available",
        )
    )

    availability = {
        name: technical_status
        for name in (
            "ema_12",
            "ema_26",
            "dif",
            "dip",
            "dea",
            "eda",
            "macd_histogram",
            "rsi_6",
            "ma_20",
            "ma_60",
            "volatility_20",
            "volatility",
            "current_drawdown",
            "drawdown",
            "running_drawdown",
        )
    }
    for name in (
        "golden_point",
        "golden_strength",
        "golden_point_reasons",
        "black_point",
        "black_strength",
        "black_point_reasons",
    ):
        availability[name] = signal_status
    availability["volume_ratio"] = volume_status
    availability["valuation_value"] = (
        "available" if valuation_value is not None else "missing"
    )
    availability["valuation_percentile"] = valuation_status
    availability["valuation_observation_count"] = "available"

    issue_tuple = tuple(issues)
    quality_report = QualityReport(
        is_publishable=not any(
            issue.severity == "blocking" for issue in issue_tuple
        ),
        issues=issue_tuple,
        volume_availability=volume_availability,
    )
    source_dates: Iterator[date] = iter(
        [bar.period_end for bar in bars]
        + [point.valuation_date for point in valuation_rows]
    )
    source_data_max_date = max(source_dates, default=None)
    assert (
        source_data_max_date is None or source_data_max_date <= cutoff_date
    )
    audit_fields: FrozenDict[str, str | int] = FrozenDict(
        {
            "dip_alias_of": "dif",
            "eda_alias_of": "dea",
            "indicator_formula_label": calculation.formula_label,
            "indicator_parameters_json": _canonical_json_dumps(
                parameters.as_dict()
            ),
            "macd_histogram_multiplier": (
                parameters.macd_histogram_multiplier
            ),
            "signal_engine": _SIGNAL_RULE_VERSION,
            "signal_rules_json": _canonical_json_dumps(
                _signal_rule_configuration()
            ),
            "valuation_min_observations": _VALUATION_MIN_OBSERVATIONS,
            "valuation_percentile_formula": (
                _VALUATION_PERCENTILE_FORMULA
            ),
            "valuation_rule_version": _VALUATION_RULE_VERSION,
            "valuation_tie_policy": _VALUATION_TIE_POLICY,
            "warmup_weeks": WARMUP_WEEKS,
            "weekly_definition": "completed_exchange_calendar_period_end",
        }
    )
    return FeatureSnapshot(
        instrument_code=instrument_code,
        cutoff_date=cutoff_date,
        source_data_max_date=source_data_max_date,
        feature_set_version=FEATURE_SET_VERSION,
        features=FrozenDict(features),
        availability=FrozenDict(availability),
        quality_report=quality_report,
        audit_fields=audit_fields,
        source_hash=_source_hash(
            bars=bars,
            valuations=valuation_rows,
            cutoff_date=cutoff_date,
            instrument_code=instrument_code,
            expected_week_ends=expected_end_rows,
            expected_trade_dates=expected_trade_date_rows,
            indicator_formula_label=calculation.formula_label,
            indicator_parameters=parameters.as_dict(),
            quality_report=quality_report,
        ),
    )
