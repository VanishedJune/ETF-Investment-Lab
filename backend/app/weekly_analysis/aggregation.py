"""Pure period aggregation and quality gates for direct market data."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from decimal import Decimal
from typing import Iterable, Sequence

from .domain import (
    AggregationResult,
    DailyBar,
    PeriodBar,
    QualityIssue,
    QualityReport,
    Timeframe,
    UpstreamWeeklyVolume,
    VolumeAvailability,
)


_DIRECT_INDEX_CODES = frozenset({"000688", "399006", "NDX"})
_PROHIBITED_SOURCE_MARKERS = ("ETF", "QQQ", "FUND", "159941")
_AGGREGATED_VOLUME_SOURCE = "AGGREGATED_DAILY_VOLUME"
_UNAVAILABLE_DIRECT_INDEX_VOLUME_SOURCE = "VOLUME_UNAVAILABLE:DIRECT_INDEX"
_UPSTREAM_399006_VOLUME_UNITS = {"AKSHARE_INDEX_WEEKLY": "shares"}
_REPAIRABLE_VOLUME_ISSUE_CODES = frozenset(
    {
        "INCOMPLETE_DAILY_VOLUME",
        "MISSING_PERIOD_VOLUME",
        "INTERNAL_PERIOD_VOLUME_GAP",
    }
)


def _period_key(day: date, timeframe: Timeframe) -> tuple[int, int]:
    if timeframe == "weekly":
        iso_year, iso_week, _ = day.isocalendar()
        return iso_year, iso_week
    return day.year, day.month


def _is_finite_positive(value: Decimal | None) -> bool:
    return value is not None and value.is_finite() and value > 0


def _is_finite_nonnegative(value: Decimal | None) -> bool:
    return value is not None and value.is_finite() and value >= 0


def _source_is_prohibited(source: str | None) -> bool:
    normalized = (source or "").upper()
    return any(marker in normalized for marker in _PROHIBITED_SOURCE_MARKERS)


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


def _report(
    issues: Iterable[QualityIssue],
    volume_availability: VolumeAvailability,
) -> QualityReport:
    issue_tuple = tuple(issues)
    return QualityReport(
        is_publishable=not any(issue.severity == "blocking" for issue in issue_tuple),
        issues=issue_tuple,
        volume_availability=volume_availability,
    )


def _merge_issues(*groups: Iterable[QualityIssue]) -> tuple[QualityIssue, ...]:
    merged: list[QualityIssue] = []
    seen: set[tuple[object, ...]] = set()
    for group in groups:
        for issue in group:
            identity = (
                issue.code,
                issue.severity,
                issue.start_date,
                issue.end_date,
                issue.message,
            )
            if identity not in seen:
                seen.add(identity)
                merged.append(issue)
    return tuple(merged)


def _daily_ohlc_issue(bar: DailyBar) -> QualityIssue | None:
    values = (bar.open_price, bar.high_price, bar.low_price, bar.close_price)
    if not all(_is_finite_positive(value) for value in values):
        return _issue(
            "INVALID_DAILY_OHLC",
            "blocking",
            bar.trade_date,
            bar.trade_date,
            "Daily OHLC values must all be finite and strictly positive.",
        )
    assert all(value is not None for value in values)
    open_price, high_price, low_price, close_price = values
    if high_price < max(open_price, low_price, close_price) or low_price > min(
        open_price, high_price, close_price
    ):
        return _issue(
            "INVALID_DAILY_OHLC_RELATION",
            "blocking",
            bar.trade_date,
            bar.trade_date,
            "Daily high/low values do not contain open and close.",
        )
    return None


def aggregate_bars(
    daily: Iterable[DailyBar],
    timeframe: Timeframe,
    instrument_code: str,
    *,
    expected_trade_dates: Iterable[date] | None = None,
    as_of: date | None = None,
) -> AggregationResult:
    """Aggregate direct daily bars without concealing invalid or missing values."""
    if timeframe not in ("weekly", "monthly"):
        raise ValueError(f"Unsupported aggregation timeframe: {timeframe}")

    input_bars = tuple(daily)
    issues: list[QualityIssue] = []
    if not input_bars:
        issues.append(
            _issue(
                "NO_DAILY_DATA",
                "blocking",
                None,
                None,
                "At least one daily bar is required to publish a period series.",
            )
        )
        return AggregationResult((), _report(issues, "available"))
    if expected_trade_dates is None or as_of is None:
        issues.append(
            _issue(
                "MISSING_COMPLETENESS_CONTEXT",
                "blocking",
                None,
                None,
                "Expected trading dates and as_of are required for period publication.",
            )
        )
        return AggregationResult((), _report(issues, "available"))
    expected_dates = tuple(expected_trade_dates)
    if len(expected_dates) != len(set(expected_dates)):
        issues.append(
            _issue(
                "DUPLICATE_EXPECTED_TRADE_DATE",
                "blocking",
                min(expected_dates, default=None),
                max(expected_dates, default=None),
                "Expected trading calendar contains duplicate dates.",
            )
        )
        return AggregationResult((), _report(issues, "available"))
    dates = [bar.trade_date for bar in input_bars]
    duplicate_dates = sorted(
        day
        for day, count in Counter(dates).items()
        if count > 1
    )
    if duplicate_dates:
        issues.append(
            _issue(
                "DUPLICATE_DAILY_DATE",
                "blocking",
                duplicate_dates[0],
                duplicate_dates[-1],
                "Daily input contains duplicate trade dates.",
            )
        )
    if any(current <= previous for previous, current in zip(dates, dates[1:])):
        if not duplicate_dates:
            issues.append(
                _issue(
                    "NON_MONOTONIC_DAILY_DATE",
                    "blocking",
                    min(dates, default=None),
                    max(dates, default=None),
                    "Daily input dates must be strictly increasing.",
                )
            )
    if issues:
        return AggregationResult((), _report(issues, "available"))

    ordered = tuple(sorted(input_bars, key=lambda bar: bar.trade_date))
    for bar in ordered:
        ohlc_issue = _daily_ohlc_issue(bar)
        if ohlc_issue is not None:
            issues.append(ohlc_issue)
        if instrument_code in _DIRECT_INDEX_CODES and _source_is_prohibited(bar.source):
            issues.append(
                _issue(
                    "PROHIBITED_ETF_SOURCE",
                    "blocking",
                    bar.trade_date,
                    bar.trade_date,
                    "Direct-index aggregation cannot consume ETF or fund data.",
                )
            )
    early_availability: VolumeAvailability = (
        "not_available_for_direct_index" if instrument_code == "NDX" else "available"
    )
    if any(issue.severity == "blocking" for issue in issues):
        return AggregationResult((), _report(issues, early_availability))

    ndx_volume_available = instrument_code != "NDX"
    volume_availability: VolumeAvailability = (
        "available" if ndx_volume_available else "not_available_for_direct_index"
    )
    if instrument_code == "NDX" and not ndx_volume_available:
        issues.append(
            _issue(
                "DIRECT_INDEX_VOLUME_UNAVAILABLE",
                "warning",
                ordered[0].trade_date if ordered else None,
                ordered[-1].trade_date if ordered else None,
                "Reliable direct NDX volume is unavailable; volume-dependent models must exclude it.",
            )
        )

    grouped: dict[tuple[int, int], list[DailyBar]] = defaultdict(list)
    for bar in ordered:
        grouped[_period_key(bar.trade_date, timeframe)].append(bar)
    expected_by_period: dict[tuple[int, int], set[date]] = defaultdict(set)
    for expected_date in expected_dates:
        expected_by_period[_period_key(expected_date, timeframe)].add(expected_date)
    actual_by_period = {
        key: {bar.trade_date for bar in bars}
        for key, bars in grouped.items()
    }
    first_key = _period_key(ordered[0].trade_date, timeframe)
    last_key = _period_key(ordered[-1].trade_date, timeframe)
    relevant_keys = {
        key
        for key in set(actual_by_period) | set(expected_by_period)
        if first_key <= key <= last_key
    }
    deferred_keys: set[tuple[int, int]] = set()
    for key in sorted(relevant_keys):
        actual_dates = actual_by_period.get(key, set())
        expected_period_dates = expected_by_period.get(key, set())
        unexpected = sorted(actual_dates - expected_period_dates)
        if unexpected:
            issues.append(
                _issue(
                    "UNEXPECTED_TRADE_DATE",
                    "blocking",
                    unexpected[0],
                    unexpected[-1],
                    "Daily input contains dates absent from the explicit trading calendar.",
                )
            )
        missing = sorted(expected_period_dates - actual_dates)
        if missing:
            missing_by_as_of = [
                missing_date for missing_date in missing if missing_date <= as_of
            ]
            missing_after_as_of = [
                missing_date for missing_date in missing if missing_date > as_of
            ]
            if missing_by_as_of:
                issues.append(
                    _issue(
                        "MISSING_EXPECTED_TRADE_DATE",
                        "blocking",
                        missing_by_as_of[0],
                        missing_by_as_of[-1],
                        "Published periods must contain every date in the explicit trading calendar.",
                    )
                )
            incomplete_current = (
                key == _period_key(as_of, timeframe)
                and bool(missing_after_as_of)
            )
            if incomplete_current:
                deferred_keys.add(key)
                issues.append(
                    _issue(
                        "INCOMPLETE_CURRENT_PERIOD",
                        "warning",
                        missing_after_as_of[0],
                        missing_after_as_of[-1],
                        "The current period is deferred until every expected session is available.",
                    )
                )
            elif missing_after_as_of:
                issues.append(
                    _issue(
                        "MISSING_EXPECTED_TRADE_DATE",
                        "blocking",
                        missing_after_as_of[0],
                        missing_after_as_of[-1],
                        "Published periods must contain every date in the explicit trading calendar.",
                    )
                )
    after_cutoff = [bar.trade_date for bar in ordered if bar.trade_date > as_of]
    if after_cutoff:
        issues.append(
            _issue(
                "DAILY_AFTER_AS_OF",
                "blocking",
                after_cutoff[0],
                after_cutoff[-1],
                "Daily input cannot extend beyond the aggregation as_of date.",
            )
        )

    periods: list[PeriodBar] = []
    for key, bars in grouped.items():
        if key in deferred_keys:
            continue
        start = bars[0].trade_date
        end = bars[-1].trade_date
        sources = sorted({bar.source for bar in bars if bar.source})
        price_source = sources[0] if len(sources) == 1 else ",".join(sources)
        missing_or_invalid_volume = any(
            not _is_finite_nonnegative(bar.volume) for bar in bars
        )
        if instrument_code == "NDX" and not ndx_volume_available:
            volume = None
            volume_source = _UNAVAILABLE_DIRECT_INDEX_VOLUME_SOURCE
        elif missing_or_invalid_volume:
            volume = None
            volume_source = _AGGREGATED_VOLUME_SOURCE
            issues.append(
                _issue(
                    "INCOMPLETE_DAILY_VOLUME",
                    "blocking",
                    start,
                    end,
                    "Every daily volume in a published period must be finite and nonnegative.",
                )
            )
        else:
            volume_values = [bar.volume for bar in bars]
            assert all(value is not None for value in volume_values)
            volume = sum(volume_values, Decimal("0"))
            volume_source = _AGGREGATED_VOLUME_SOURCE
            zero_dates = [bar.trade_date for bar in bars if bar.volume == 0]
            if zero_dates:
                issues.append(
                    _issue(
                        "ZERO_DAILY_VOLUME",
                        "warning",
                        zero_dates[0],
                        zero_dates[-1],
                        "A real zero daily volume was preserved in the period sum.",
                    )
                )

        turnover_values = [bar.turnover for bar in bars]
        turnover = (
            sum((value for value in turnover_values if value is not None), Decimal("0"))
            if turnover_values and all(value is not None for value in turnover_values)
            else None
        )
        periods.append(
            PeriodBar(
                timeframe=timeframe,
                period_start=start,
                period_end=end,
                open_price=bars[0].open_price,
                high_price=max(bar.high_price for bar in bars),  # type: ignore[type-var]
                low_price=min(bar.low_price for bar in bars),  # type: ignore[type-var]
                close_price=bars[-1].close_price,
                adjusted_close_price=bars[-1].adjusted_close_price,
                volume=volume,
                turnover=turnover,
                source=_AGGREGATED_VOLUME_SOURCE,
                volume_source=volume_source,
                price_source=price_source,
            )
        )

    if not periods and deferred_keys:
        return AggregationResult((), _report(issues, volume_availability))

    validated = validate_series(periods, instrument_code)
    combined = _merge_issues(issues, validated.report.issues)
    availability = (
        "not_available_for_direct_index"
        if volume_availability == "not_available_for_direct_index"
        else validated.report.volume_availability
    )
    return AggregationResult(validated.bars, _report(combined, availability))


def validate_series(
    periods: Iterable[PeriodBar],
    instrument_code: str,
) -> AggregationResult:
    """Validate period identity, OHLC relationships, source audit, and volume continuity."""
    bars = tuple(sorted(periods, key=lambda bar: (bar.period_start, bar.period_end)))
    issues: list[QualityIssue] = []
    if not bars:
        issues.append(
            _issue(
                "NO_PERIOD_DATA",
                "blocking",
                None,
                None,
                "At least one period bar is required for series publication.",
            )
        )
        return AggregationResult((), _report(issues, "available"))
    seen: set[tuple[str, tuple[int, int]]] = set()
    for bar in bars:
        identity = (bar.timeframe, _period_key(bar.period_end, bar.timeframe))
        if identity in seen:
            issues.append(
                _issue(
                    "DUPLICATE_PERIOD",
                    "blocking",
                    bar.period_start,
                    bar.period_end,
                    "Series contains more than one row for the same period.",
                )
            )
        seen.add(identity)
        if (
            bar.period_start > bar.period_end
            or _period_key(bar.period_start, bar.timeframe)
            != _period_key(bar.period_end, bar.timeframe)
        ):
            issues.append(
                _issue(
                    "INVALID_PERIOD_BOUNDARY",
                    "blocking",
                    bar.period_start,
                    bar.period_end,
                    "Period start/end must be ordered and belong to one period bucket.",
                )
            )
        values = (bar.open_price, bar.high_price, bar.low_price, bar.close_price)
        if not all(_is_finite_positive(value) for value in values):
            issues.append(
                _issue(
                    "INVALID_PERIOD_OHLC",
                    "blocking",
                    bar.period_start,
                    bar.period_end,
                    "Period OHLC values must all be finite and strictly positive.",
                )
            )
        else:
            open_price, high_price, low_price, close_price = values
            assert all(value is not None for value in values)
            if high_price < max(open_price, low_price, close_price) or low_price > min(
                open_price, high_price, close_price
            ):
                issues.append(
                    _issue(
                        "INVALID_PERIOD_OHLC_RELATION",
                        "blocking",
                        bar.period_start,
                        bar.period_end,
                        "Period high/low values do not contain open and close.",
                    )
                )
        if not bar.source or not bar.volume_source or not bar.price_source:
            issues.append(
                _issue(
                    "MISSING_PERIOD_SOURCE",
                    "blocking",
                    bar.period_start,
                    bar.period_end,
                    "Every period must retain price and volume provenance.",
                )
            )
        if bar.volume is not None and not _is_finite_nonnegative(bar.volume):
            issues.append(
                _issue(
                    "INVALID_PERIOD_VOLUME",
                    "blocking",
                    bar.period_start,
                    bar.period_end,
                    "Period volume must be finite and nonnegative when available.",
                )
            )

    missing_indexes = [index for index, bar in enumerate(bars) if bar.volume is None]
    volume_availability: VolumeAvailability = "available"
    if missing_indexes:
        if instrument_code == "NDX" and all(
            bar.volume_source == _UNAVAILABLE_DIRECT_INDEX_VOLUME_SOURCE
            for bar in bars
        ):
            volume_availability = "not_available_for_direct_index"
        else:
            volume_availability = "partially_unavailable"
            present_indexes = [
                index
                for index, bar in enumerate(bars)
                if bar.volume is not None
            ]
            first_present = min(present_indexes, default=None)
            last_present = max(present_indexes, default=None)
            for index in missing_indexes:
                bar = bars[index]
                internal = (
                    first_present is not None
                    and last_present is not None
                    and first_present < index < last_present
                )
                issues.append(
                    _issue(
                        "INTERNAL_PERIOD_VOLUME_GAP" if internal else "MISSING_PERIOD_VOLUME",
                        "blocking",
                        bar.period_start,
                        bar.period_end,
                        "A volume-declared period series cannot contain null volume.",
                    )
                )
    return AggregationResult(bars, _report(issues, volume_availability))


def apply_upstream_weekly_volume(
    base_result: AggregationResult,
    upstream: Sequence[UpstreamWeeklyVolume],
    *,
    instrument_code: str,
    upstream_source: str,
) -> AggregationResult:
    """Audit native direct-index weekly volume before applying an all-or-fallback overlay."""
    baseline = base_result.bars

    def fallback(code: str, message: str) -> AggregationResult:
        issue = _issue(
            code,
            "warning",
            baseline[0].period_start if baseline else None,
            baseline[-1].period_end if baseline else None,
            message,
        )
        validated = validate_series(baseline, instrument_code)
        return AggregationResult(
            validated.bars,
            _report(
                _merge_issues(
                    base_result.report.issues,
                    (issue,),
                    validated.report.issues,
                ),
                base_result.report.volume_availability,
            ),
        )

    if instrument_code != "399006":
        return fallback(
            "UPSTREAM_VOLUME_WRONG_INSTRUMENT",
            "Native weekly volume overlay is only approved for direct index 399006.",
        )
    expected_unit = _UPSTREAM_399006_VOLUME_UNITS.get(upstream_source)
    if expected_unit is None:
        return fallback(
            "UPSTREAM_VOLUME_PROHIBITED_SOURCE",
            "Upstream weekly volume source is not an approved direct-index capability.",
        )
    if not upstream:
        return fallback("UPSTREAM_VOLUME_EMPTY", "No upstream weekly volume was supplied.")
    if any(
        not item.source
        or item.source != upstream_source
        or item.unit != expected_unit
        for item in upstream
    ):
        return fallback(
            "UPSTREAM_VOLUME_SOURCE_MISMATCH",
            "Every upstream row must retain the approved direct-index source and unit.",
        )
    if any(not _is_finite_positive(item.volume) for item in upstream):
        return fallback(
            "UPSTREAM_VOLUME_INVALID_VALUE",
            "Every upstream weekly volume must be finite and strictly positive.",
        )

    upstream_dates = [item.trade_date for item in upstream]
    upstream_keys = [_period_key(item.trade_date, "weekly") for item in upstream]
    if len(upstream_dates) != len(set(upstream_dates)) or len(upstream_keys) != len(
        set(upstream_keys)
    ):
        return fallback(
            "UPSTREAM_VOLUME_DUPLICATE_WEEK",
            "Upstream weekly volume contains duplicate week mappings.",
        )
    baseline_weekly = tuple(bar for bar in baseline if bar.timeframe == "weekly")
    baseline_by_end = {bar.period_end: bar for bar in baseline_weekly}
    baseline_ends = set(baseline_by_end)
    if set(upstream_dates) != baseline_ends:
        if len(upstream_dates) != len(baseline_ends):
            return fallback(
                "UPSTREAM_VOLUME_INCOMPLETE_COVERAGE",
                "Upstream weekly volume must cover every derived week exactly once.",
            )
        return fallback(
            "UPSTREAM_VOLUME_ENDPOINT_MISMATCH",
            "Every upstream date must exactly equal its derived period_end.",
        )

    ratios = [
        item.volume / baseline_by_end[item.trade_date].volume
        for item in upstream
        if item.volume is not None
        and baseline_by_end[item.trade_date].volume is not None
        and baseline_by_end[item.trade_date].volume > 0
    ]
    if len(ratios) != len(baseline_weekly):
        return fallback(
            "UPSTREAM_VOLUME_NO_SCALE_REFERENCE",
            "Upstream volume has no positive daily-derived overlap for unit validation.",
        )
    if any(ratio < Decimal("0.8") or ratio > Decimal("1.2") for ratio in ratios):
        return fallback(
            "UPSTREAM_VOLUME_INCONSISTENT_SCALE",
            "Same-unit upstream volume must remain within 20% of daily-derived volume.",
        )

    upstream_by_end = {item.trade_date: item for item in upstream}
    overlaid = tuple(
        replace(
            bar,
            volume=upstream_by_end[bar.period_end].volume,
            source=upstream_by_end[bar.period_end].source,
            volume_source=upstream_by_end[bar.period_end].source,
        )
        if bar.period_end in upstream_by_end
        else bar
        for bar in baseline
    )
    validated = validate_series(overlaid, instrument_code)
    preserved_base_issues = tuple(
        issue
        for issue in base_result.report.issues
        if issue.code not in _REPAIRABLE_VOLUME_ISSUE_CODES
    )
    return AggregationResult(
        validated.bars,
        _report(
            _merge_issues(preserved_base_issues, validated.report.issues),
            validated.report.volume_availability,
        ),
    )
