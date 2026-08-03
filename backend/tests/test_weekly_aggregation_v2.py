from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal
import importlib

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    DataUpdateLog,
    IndicatorRecord,
    Instrument,
    MarketPrice,
)
from backend.app.schemas.market import MarketDataRecord, ProviderResult
from backend.app.services.indicator_service import (
    IndicatorCalculationStatus,
    IndicatorService,
)
from backend.app.services.market_data import MarketDataService


class StaticCalendarProvider:
    def __init__(self, sessions: tuple[date, ...]) -> None:
        self.configured_sessions = sessions
        self.calls: list[tuple[str, date, date]] = []

    def sessions(
        self,
        instrument_code: str,
        start_date: date,
        end_date: date,
    ) -> tuple[date, ...]:
        self.calls.append((instrument_code, start_date, end_date))
        return tuple(
            day
            for day in self.configured_sessions
            if start_date <= day <= end_date
        )


def _domain():
    return importlib.import_module("backend.app.weekly_analysis.domain")


def _aggregation():
    return importlib.import_module("backend.app.weekly_analysis.aggregation")


def _daily(
    trade_date: date,
    close: str,
    *,
    volume: str | None = "100",
    source: str = "DIRECT_INDEX_DAILY",
):
    DailyBar = _domain().DailyBar
    close_value = Decimal(close)
    values = dict(
        trade_date=trade_date,
        open_price=close_value - Decimal("1"),
        high_price=close_value + Decimal("2"),
        low_price=close_value - Decimal("3"),
        close_price=close_value,
        volume=None if volume is None else Decimal(volume),
        source=source,
    )
    return DailyBar(**values)


def _aggregate_complete(
    daily,
    timeframe: str,
    instrument_code: str,
    *,
    expected_trade_dates=None,
    as_of: date | None = None,
):
    bars = list(daily)
    expected = (
        tuple(expected_trade_dates)
        if expected_trade_dates is not None
        else tuple(bar.trade_date for bar in bars)
    )
    cutoff = as_of or max(expected, default=date(2026, 1, 1))
    return _aggregation().aggregate_bars(
        bars,
        timeframe,
        instrument_code,
        expected_trade_dates=expected,
        as_of=cutoff,
    )


def _stored_record(
    trade_date: date,
    close: str,
    *,
    volume: str | None = "100",
) -> MarketDataRecord:
    close_value = Decimal(close)
    return MarketDataRecord(
        trade_date=trade_date,
        open_price=close_value - Decimal("1"),
        high_price=close_value + Decimal("2"),
        low_price=close_value - Decimal("3"),
        close_price=close_value,
        volume=None if volume is None else Decimal(volume),
        source="DIRECT_INDEX_DAILY",
    )


def test_domain_bars_and_quality_objects_are_immutable() -> None:
    domain = _domain()
    bar = _daily(date(2026, 7, 6), "100")
    issue = domain.QualityIssue(
        code="TEST",
        severity="warning",
        start_date=date(2026, 7, 6),
        end_date=date(2026, 7, 6),
        message="test",
    )
    report = domain.QualityReport(
        is_publishable=True,
        issues=(issue,),
        volume_availability="available",
    )

    with pytest.raises(FrozenInstanceError):
        bar.close_price = Decimal("1")
    with pytest.raises(FrozenInstanceError):
        report.is_publishable = False


def test_weekly_and_monthly_ohlcv_use_first_high_low_last_and_full_volume_sum() -> None:
    daily = [
        _daily(date(2026, 6, 30), "100", volume="10"),
        _daily(date(2026, 7, 1), "110", volume="20"),
        _daily(date(2026, 7, 3), "105", volume="30"),
    ]

    weekly = _aggregate_complete(daily, "weekly", "399006")
    monthly = _aggregate_complete(daily, "monthly", "399006")

    assert weekly.report.is_publishable is True
    assert len(weekly.bars) == 1  # One ISO week even though the week crosses a month boundary.
    assert (
        weekly.bars[0].period_start,
        weekly.bars[0].period_end,
        weekly.bars[0].open_price,
        weekly.bars[0].high_price,
        weekly.bars[0].low_price,
        weekly.bars[0].close_price,
        weekly.bars[0].volume,
    ) == (
        date(2026, 6, 30),
        date(2026, 7, 3),
        Decimal("99"),
        Decimal("112"),
        Decimal("97"),
        Decimal("105"),
        Decimal("60"),
    )
    assert [(bar.period_end, bar.volume) for bar in monthly.bars] == [
        (date(2026, 6, 30), Decimal("10")),
        (date(2026, 7, 3), Decimal("50")),
    ]
    assert {bar.source for bar in weekly.bars} == {"AGGREGATED_DAILY_VOLUME"}
    assert {bar.volume_source for bar in weekly.bars} == {"AGGREGATED_DAILY_VOLUME"}
    assert {bar.price_source for bar in weekly.bars} == {"DIRECT_INDEX_DAILY"}


def test_weekly_grouping_handles_four_and_five_week_months_without_month_buckets() -> None:
    january = [
        _daily(day, str(100 + index))
        for index, day in enumerate(
            (
                date(2026, 1, 2),
                date(2026, 1, 5),
                date(2026, 1, 12),
                date(2026, 1, 19),
                date(2026, 1, 26),
            )
        )
    ]
    february = [
        _daily(day, str(110 + index))
        for index, day in enumerate(
            (date(2026, 2, 2), date(2026, 2, 9), date(2026, 2, 16), date(2026, 2, 23))
        )
    ]

    result = _aggregate_complete(january + february, "weekly", "399006")

    assert len(result.bars) == 9
    assert [bar.period_end for bar in result.bars[-4:]] == [
        date(2026, 2, 2),
        date(2026, 2, 9),
        date(2026, 2, 16),
        date(2026, 2, 23),
    ]


@pytest.mark.parametrize(
    "daily_values,expected_code",
    [
        (
            [
                (date(2026, 7, 6), "100"),
                (date(2026, 7, 6), "101"),
            ],
            "DUPLICATE_DAILY_DATE",
        ),
        (
            [
                (date(2026, 7, 7), "101"),
                (date(2026, 7, 6), "100"),
            ],
            "NON_MONOTONIC_DAILY_DATE",
        ),
    ],
)
def test_daily_date_duplicates_and_non_monotonic_input_are_blocking(
    daily_values,
    expected_code,
) -> None:
    daily = [_daily(day, close) for day, close in daily_values]
    result = _aggregate_complete(
        daily,
        "weekly",
        "399006",
        expected_trade_dates=sorted({day for day, _close in daily_values}),
    )

    assert result.bars == ()
    assert result.report.is_publishable is False
    assert expected_code in {issue.code for issue in result.report.issues}


def test_399006_missing_one_daily_volume_never_partially_sums_period() -> None:
    result = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100", volume="100"),
            _daily(date(2026, 7, 7), "101", volume=None),
            _daily(date(2026, 7, 8), "102", volume="300"),
        ],
        "weekly",
        "399006",
    )

    assert result.bars[0].volume is None
    assert result.report.is_publishable is False
    assert "INCOMPLETE_DAILY_VOLUME" in {issue.code for issue in result.report.issues}
    assert result.bars[0].volume != Decimal("400")


def test_empty_daily_series_is_blocking_instead_of_publishing_an_empty_replacement() -> None:
    result = _aggregation().aggregate_bars(
        [],
        "weekly",
        "399006",
        expected_trade_dates=(),
        as_of=date(2026, 7, 6),
    )

    assert result.bars == ()
    assert result.report.is_publishable is False
    assert "NO_DAILY_DATA" in {issue.code for issue in result.report.issues}


def test_real_zero_volume_is_preserved_but_warned() -> None:
    result = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100", volume="0"),
            _daily(date(2026, 7, 7), "101", volume="10"),
        ],
        "weekly",
        "399006",
    )

    assert result.bars[0].volume == Decimal("10")
    assert result.report.is_publishable is True
    assert "ZERO_DAILY_VOLUME" in {issue.code for issue in result.report.issues}


def test_invalid_ohlc_is_blocking_and_not_publishable() -> None:
    DailyBar = _domain().DailyBar
    result = _aggregate_complete(
        [
            DailyBar(
                trade_date=date(2026, 7, 6),
                open_price=Decimal("100"),
                high_price=Decimal("NaN"),
                low_price=Decimal("99"),
                close_price=Decimal("100"),
                volume=Decimal("10"),
                source="DIRECT_INDEX_DAILY",
            )
        ],
        "weekly",
        "399006",
    )

    assert result.report.is_publishable is False
    assert "INVALID_DAILY_OHLC" in {issue.code for issue in result.report.issues}


def test_explicit_trading_calendar_blocks_internal_missing_trade_date() -> None:
    result = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100"),
            _daily(date(2026, 7, 8), "102"),
        ],
        "weekly",
        "399006",
        expected_trade_dates=(
            date(2026, 7, 6),
            date(2026, 7, 7),
            date(2026, 7, 8),
        ),
        as_of=date(2026, 7, 8),
    )

    assert result.report.is_publishable is False
    issue = next(issue for issue in result.report.issues if issue.code == "MISSING_EXPECTED_TRADE_DATE")
    assert issue.start_date == date(2026, 7, 7)
    assert issue.end_date == date(2026, 7, 7)


def test_current_period_future_defer_does_not_hide_an_internal_gap() -> None:
    result = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100"),
            _daily(date(2026, 7, 8), "102"),
        ],
        "weekly",
        "399006",
        expected_trade_dates=tuple(date(2026, 7, day) for day in range(6, 11)),
        as_of=date(2026, 7, 8),
    )

    assert result.report.is_publishable is False
    issue = next(
        issue
        for issue in result.report.issues
        if issue.code == "MISSING_EXPECTED_TRADE_DATE"
    )
    assert (issue.start_date, issue.end_date) == (
        date(2026, 7, 7),
        date(2026, 7, 7),
    )


def test_explicit_trading_calendar_defers_current_incomplete_period() -> None:
    result = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100"),
            _daily(date(2026, 7, 7), "101"),
            _daily(date(2026, 7, 8), "102"),
        ],
        "weekly",
        "399006",
        expected_trade_dates=tuple(date(2026, 7, day) for day in range(6, 11)),
        as_of=date(2026, 7, 8),
    )

    assert result.bars == ()
    assert result.report.is_publishable is True
    issue = next(
        issue for issue in result.report.issues if issue.code == "INCOMPLETE_CURRENT_PERIOD"
    )
    assert issue.severity == "warning"


def test_completed_week_publishes_while_current_month_is_deferred() -> None:
    daily = [
        _daily(date(2026, 7, day), str(100 + day))
        for day in range(6, 11)
    ]
    july_sessions = tuple(
        date(2026, 7, day)
        for day in range(6, 32)
        if date(2026, 7, day).weekday() < 5
    )

    weekly = _aggregate_complete(
        daily,
        "weekly",
        "399006",
        expected_trade_dates=july_sessions,
        as_of=date(2026, 7, 10),
    )
    monthly = _aggregate_complete(
        daily,
        "monthly",
        "399006",
        expected_trade_dates=july_sessions,
        as_of=date(2026, 7, 10),
    )

    assert [bar.period_end for bar in weekly.bars] == [date(2026, 7, 10)]
    assert weekly.report.is_publishable is True
    assert monthly.bars == ()
    assert monthly.report.is_publishable is True
    assert "INCOMPLETE_CURRENT_PERIOD" in {
        issue.code for issue in monthly.report.issues
    }


def test_historical_month_publishes_while_current_week_and_month_are_deferred() -> None:
    january_sessions = tuple(date(2026, 1, day) for day in range(26, 31))
    february_sessions = tuple(
        date(2026, 2, day)
        for day in range(1, 29)
        if date(2026, 2, day).weekday() < 5
    )
    daily = [
        *[
            _daily(day, str(100 + index))
            for index, day in enumerate(january_sessions)
        ],
        _daily(date(2026, 2, 2), "110"),
        _daily(date(2026, 2, 3), "111"),
    ]
    expected = january_sessions + february_sessions

    weekly = _aggregate_complete(
        daily,
        "weekly",
        "399006",
        expected_trade_dates=expected,
        as_of=date(2026, 2, 3),
    )
    monthly = _aggregate_complete(
        daily,
        "monthly",
        "399006",
        expected_trade_dates=expected,
        as_of=date(2026, 2, 3),
    )

    assert [bar.period_end for bar in weekly.bars] == [date(2026, 1, 30)]
    assert [bar.period_end for bar in monthly.bars] == [date(2026, 1, 30)]
    assert weekly.report.is_publishable is True
    assert monthly.report.is_publishable is True


def test_explicit_calendar_does_not_treat_a_weekday_holiday_as_missing() -> None:
    expected = (date(2026, 7, 6), date(2026, 7, 8))
    result = _aggregate_complete(
        [_daily(expected[0], "100"), _daily(expected[1], "102")],
        "weekly",
        "399006",
        expected_trade_dates=expected,
        as_of=expected[-1],
    )

    assert result.report.is_publishable is True


def test_iso_week_grouping_crosses_calendar_year_without_splitting() -> None:
    daily = [
        _daily(date(2025, 12, 29), "100"),
        _daily(date(2025, 12, 31), "101"),
        _daily(date(2026, 1, 2), "102"),
        _daily(date(2026, 1, 5), "103"),
    ]

    result = _aggregate_complete(daily, "weekly", "399006")

    assert [(bar.period_start, bar.period_end) for bar in result.bars] == [
        (date(2025, 12, 29), date(2026, 1, 2)),
        (date(2026, 1, 5), date(2026, 1, 5)),
    ]


def test_validate_series_rejects_internal_volume_hole_and_missing_source() -> None:
    domain = _domain()
    validate_series = _aggregation().validate_series
    periods = (
        domain.PeriodBar(
            timeframe="weekly",
            period_start=date(2026, 7, 6),
            period_end=date(2026, 7, 10),
            open_price=Decimal("100"),
            high_price=Decimal("105"),
            low_price=Decimal("99"),
            close_price=Decimal("104"),
            volume=Decimal("10"),
            source="DIRECT_INDEX_DAILY",
            volume_source="AGGREGATED_DAILY_VOLUME",
            price_source="DIRECT_INDEX_DAILY",
        ),
        domain.PeriodBar(
            timeframe="weekly",
            period_start=date(2026, 7, 13),
            period_end=date(2026, 7, 17),
            open_price=Decimal("104"),
            high_price=Decimal("106"),
            low_price=Decimal("100"),
            close_price=Decimal("101"),
            volume=None,
            source="",
            volume_source="",
            price_source="",
        ),
        domain.PeriodBar(
            timeframe="weekly",
            period_start=date(2026, 7, 20),
            period_end=date(2026, 7, 24),
            open_price=Decimal("101"),
            high_price=Decimal("109"),
            low_price=Decimal("100"),
            close_price=Decimal("108"),
            volume=Decimal("12"),
            source="DIRECT_INDEX_DAILY",
            volume_source="AGGREGATED_DAILY_VOLUME",
            price_source="DIRECT_INDEX_DAILY",
        ),
    )

    result = validate_series(periods, "399006")

    assert result.report.is_publishable is False
    assert {"INTERNAL_PERIOD_VOLUME_GAP", "MISSING_PERIOD_SOURCE"} <= {
        issue.code for issue in result.report.issues
    }


def test_validate_series_empty_input_is_blocking() -> None:
    result = _aggregation().validate_series((), "399006")

    assert result.bars == ()
    assert result.report.is_publishable is False
    assert "NO_PERIOD_DATA" in {issue.code for issue in result.report.issues}


def test_valid_direct_index_upstream_weekly_volume_covers_and_overrides_every_week() -> None:
    aggregation = _aggregation()
    domain = _domain()
    baseline = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100", volume="100"),
            _daily(date(2026, 7, 7), "101", volume="100"),
            _daily(date(2026, 7, 13), "102", volume="100"),
            _daily(date(2026, 7, 14), "103", volume="100"),
            _daily(date(2026, 7, 20), "104", volume="100"),
        ],
        "weekly",
        "399006",
    )
    upstream = (
        domain.UpstreamWeeklyVolume(
            date(2026, 7, 7), Decimal("200"), "AKSHARE_INDEX_WEEKLY", "shares"
        ),
        domain.UpstreamWeeklyVolume(
            date(2026, 7, 14), Decimal("200"), "AKSHARE_INDEX_WEEKLY", "shares"
        ),
        domain.UpstreamWeeklyVolume(
            date(2026, 7, 20), Decimal("100"), "AKSHARE_INDEX_WEEKLY", "shares"
        ),
    )

    overlaid = aggregation.apply_upstream_weekly_volume(
        baseline,
        upstream,
        instrument_code="399006",
        upstream_source="AKSHARE_INDEX_WEEKLY",
    )

    assert [bar.volume for bar in overlaid.bars] == [Decimal("200"), Decimal("200"), Decimal("100")]
    assert {bar.source for bar in overlaid.bars} == {"AKSHARE_INDEX_WEEKLY"}
    assert {bar.volume_source for bar in overlaid.bars} == {"AKSHARE_INDEX_WEEKLY"}
    assert {bar.price_source for bar in overlaid.bars} == {"DIRECT_INDEX_DAILY"}
    assert overlaid.report.is_publishable is True


@pytest.mark.parametrize("upstream_volume", ["200", "2000"])
def test_upstream_volume_never_discards_base_blocking_quality(
    upstream_volume: str,
) -> None:
    aggregation = _aggregation()
    domain = _domain()
    baseline = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100", volume="100"),
            _daily(date(2026, 7, 7), "101", volume="100"),
        ],
        "weekly",
        "399006",
    )
    missing_issue = domain.QualityIssue(
        code="MISSING_EXPECTED_TRADE_DATE",
        severity="blocking",
        start_date=date(2026, 7, 8),
        end_date=date(2026, 7, 8),
        message="Base completeness failure must survive volume overlay.",
    )
    base_result = domain.AggregationResult(
        bars=baseline.bars,
        report=domain.QualityReport(
            is_publishable=False,
            issues=(*baseline.report.issues, missing_issue),
            volume_availability=baseline.report.volume_availability,
        ),
    )

    result = aggregation.apply_upstream_weekly_volume(
        base_result,
        (
            domain.UpstreamWeeklyVolume(
                date(2026, 7, 7),
                Decimal(upstream_volume),
                "AKSHARE_INDEX_WEEKLY",
                "shares",
            ),
        ),
        instrument_code="399006",
        upstream_source="AKSHARE_INDEX_WEEKLY",
    )

    assert result.report.is_publishable is False
    assert "MISSING_EXPECTED_TRADE_DATE" in {
        issue.code for issue in result.report.issues
    }


@pytest.mark.parametrize(
    "source,upstream,unit",
    [
        (
            "QQQ_ETF_WEEKLY",
            ((date(2026, 7, 7), "200"), (date(2026, 7, 14), "200")),
            "shares",
        ),
        (
            "AKSHARE_INDEX_WEEKLY",
            ((date(2026, 7, 7), "200"), (date(2026, 7, 7), "200")),
            "shares",
        ),
        (
            "AKSHARE_INDEX_WEEKLY",
            ((date(2026, 7, 10), "200"), (date(2026, 7, 17), "200")),
            "shares",
        ),
        (
            "AKSHARE_INDEX_WEEKLY",
            ((date(2026, 7, 24), "200"),),
            "shares",
        ),
        (
            "AKSHARE_INDEX_WEEKLY",
            ((date(2026, 7, 7), "200"), (date(2026, 7, 14), "200")),
            "lots",
        ),
        (
            "AKSHARE_INDEX_WEEKLY",
            ((date(2026, 7, 7), "0"), (date(2026, 7, 14), "200")),
            "shares",
        ),
    ],
)
def test_bad_upstream_weekly_volume_is_auditable_and_falls_back_without_holes(
    source: str,
    upstream: tuple[tuple[date, str], ...],
    unit: str,
) -> None:
    aggregation = _aggregation()
    domain = _domain()
    baseline = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100"),
            _daily(date(2026, 7, 7), "101"),
            _daily(date(2026, 7, 13), "102"),
            _daily(date(2026, 7, 14), "103"),
        ],
        "weekly",
        "399006",
    )
    candidates = tuple(
        domain.UpstreamWeeklyVolume(day, Decimal(volume), source, unit) for day, volume in upstream
    )

    result = aggregation.apply_upstream_weekly_volume(
        baseline,
        candidates,
        instrument_code="399006",
        upstream_source=source,
    )

    assert result.bars == baseline.bars
    assert result.report.is_publishable is True
    assert any(issue.severity == "warning" for issue in result.report.issues)
    assert all(bar.volume is not None for bar in result.bars)
    assert {bar.volume_source for bar in result.bars} == {"AGGREGATED_DAILY_VOLUME"}


def test_upstream_missing_one_derived_week_falls_back_entire_series() -> None:
    domain = _domain()
    aggregation = _aggregation()
    baseline = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100"),
            _daily(date(2026, 7, 7), "101"),
            _daily(date(2026, 7, 13), "102"),
            _daily(date(2026, 7, 14), "103"),
        ],
        "weekly",
        "399006",
    )

    result = aggregation.apply_upstream_weekly_volume(
        baseline,
        (
            domain.UpstreamWeeklyVolume(
                date(2026, 7, 7), Decimal("200"), "AKSHARE_INDEX_WEEKLY", "shares"
            ),
        ),
        instrument_code="399006",
        upstream_source="AKSHARE_INDEX_WEEKLY",
    )

    assert result.bars == baseline.bars
    assert "UPSTREAM_VOLUME_INCOMPLETE_COVERAGE" in {
        issue.code for issue in result.report.issues
    }


def test_upstream_same_multiplier_unit_anomaly_falls_back_entire_series() -> None:
    domain = _domain()
    aggregation = _aggregation()
    baseline = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "100"),
            _daily(date(2026, 7, 7), "101"),
            _daily(date(2026, 7, 13), "102"),
            _daily(date(2026, 7, 14), "103"),
        ],
        "weekly",
        "399006",
    )

    result = aggregation.apply_upstream_weekly_volume(
        baseline,
        (
            domain.UpstreamWeeklyVolume(
                date(2026, 7, 7), Decimal("2000"), "AKSHARE_INDEX_WEEKLY", "shares"
            ),
            domain.UpstreamWeeklyVolume(
                date(2026, 7, 14), Decimal("2000"), "AKSHARE_INDEX_WEEKLY", "shares"
            ),
        ),
        instrument_code="399006",
        upstream_source="AKSHARE_INDEX_WEEKLY",
    )

    assert result.bars == baseline.bars
    assert "UPSTREAM_VOLUME_INCONSISTENT_SCALE" in {
        issue.code for issue in result.report.issues
    }


def test_single_week_upstream_scale_anomaly_falls_back() -> None:
    domain = _domain()
    aggregation = _aggregation()
    baseline = _aggregate_complete(
        [_daily(date(2026, 7, 6), "100"), _daily(date(2026, 7, 7), "101")],
        "weekly",
        "399006",
    )

    result = aggregation.apply_upstream_weekly_volume(
        baseline,
        (
            domain.UpstreamWeeklyVolume(
                date(2026, 7, 7), Decimal("2000"), "AKSHARE_INDEX_WEEKLY", "shares"
            ),
        ),
        instrument_code="399006",
        upstream_source="AKSHARE_INDEX_WEEKLY",
    )

    assert result.bars == baseline.bars
    assert "UPSTREAM_VOLUME_INCONSISTENT_SCALE" in {
        issue.code for issue in result.report.issues
    }


def test_ndx_missing_direct_volume_is_explicitly_unavailable_and_never_fabricated_as_zero() -> None:
    result = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "20000", volume="100", source="STOOQ_INDEX"),
            _daily(date(2026, 7, 7), "20100", volume=None, source="STOOQ_INDEX"),
            _daily(date(2026, 7, 13), "20200", volume="300", source="STOOQ_INDEX"),
        ],
        "weekly",
        "NDX",
    )

    assert result.report.is_publishable is True
    assert result.report.volume_availability == "not_available_for_direct_index"
    assert [bar.volume for bar in result.bars] == [None, None]
    assert all(bar.volume_source == "VOLUME_UNAVAILABLE:DIRECT_INDEX" for bar in result.bars)
    assert "DIRECT_INDEX_VOLUME_UNAVAILABLE" in {issue.code for issue in result.report.issues}
    assert all(bar.volume != Decimal("0") for bar in result.bars)


def test_ndx_positive_direct_volume_is_still_unavailable() -> None:
    result = _aggregate_complete(
        [
            _daily(
                date(2026, 7, 6),
                "20000",
                volume="100",
                source="NDX_DIRECT_INDEX_VOLUME",
            ),
            _daily(
                date(2026, 7, 7),
                "20100",
                volume="200",
                source="NDX_DIRECT_INDEX_VOLUME",
            ),
        ],
        "weekly",
        "NDX",
    )

    assert result.report.is_publishable is True
    assert result.report.volume_availability == "not_available_for_direct_index"
    assert result.bars[0].volume is None


@pytest.mark.parametrize(
    "source",
    [
        "CSV",
        "UNKNOWN",
        "DEMO_INDEX",
        "QQQ",
        "SOME_ETF",
        "STOOQ_INDEX",
        "NDX_DIRECT_INDEX_VOLUME",
    ],
)
def test_ndx_unverified_or_unapproved_positive_volume_is_unavailable(
    source: str,
) -> None:
    result = _aggregate_complete(
        [
            _daily(
                date(2026, 7, 6),
                "20000",
                volume="100",
                source=source,
            )
        ],
        "weekly",
        "NDX",
    )

    assert result.report.volume_availability == "not_available_for_direct_index"
    if source in {"QQQ", "SOME_ETF"}:
        assert result.bars == ()
        assert result.report.is_publishable is False
    else:
        assert result.bars[0].volume is None


def test_ndx_demo_volume_is_not_treated_as_reliable_direct_index_volume() -> None:
    result = _aggregate_complete(
        [
            _daily(date(2026, 7, 6), "20000", volume="100", source="DEMO_INDEX"),
            _daily(date(2026, 7, 7), "20100", volume="200", source="DEMO_INDEX"),
        ],
        "weekly",
        "NDX",
    )

    assert result.report.is_publishable is True
    assert result.report.volume_availability == "not_available_for_direct_index"
    assert result.bars[0].volume is None
    assert result.bars[0].volume_source == "VOLUME_UNAVAILABLE:DIRECT_INDEX"


def test_ndx_demo_fallback_does_not_fabricate_daily_volume(tmp_path) -> None:
    class FailingNdxProvider:
        source = "DIRECT_INDEX_PROVIDER"

        def fetch(self, instrument_code, start_date=None, end_date=None):
            return ProviderResult.failed(self.source, "offline")

    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    service = MarketDataService(create_session_factory(database))

    response = service.update_from_provider("NDX", FailingNdxProvider())

    assert response.demo is True
    assert response.records
    assert all(record.source == "DEMO_INDEX" for record in response.records)
    assert all(record.volume is None for record in response.records)


def test_blocking_quality_failure_keeps_last_good_weekly_and_monthly_rows(tmp_path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    engine = create_database_engine(database)
    service = MarketDataService(session_factory)
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(date(2026, 7, 6), "100", volume="100"),
                _stored_record(date(2026, 7, 7), "101", volume="200"),
            ],
        ),
    )
    service.aggregate_timeframes(
        "399006",
        expected_trade_dates=(date(2026, 7, 6), date(2026, 7, 7)),
        as_of=date(2026, 7, 7),
    )
    with Session(engine) as session:
        before = session.execute(
            select(
                MarketPrice.timeframe,
                MarketPrice.trade_date,
                MarketPrice.close_price,
                MarketPrice.volume,
                MarketPrice.source,
            )
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe.in_(("weekly", "monthly")),
            )
            .order_by(MarketPrice.timeframe)
        ).all()
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [_stored_record(date(2026, 7, 8), "102", volume=None)],
        ),
    )

    with pytest.raises(ValueError, match="INCOMPLETE_DAILY_VOLUME"):
        service.aggregate_timeframes(
            "399006",
            expected_trade_dates=(
                date(2026, 7, 6),
                date(2026, 7, 7),
                date(2026, 7, 8),
            ),
            as_of=date(2026, 7, 8),
        )

    with Session(engine) as session:
        after = session.execute(
            select(
                MarketPrice.timeframe,
                MarketPrice.trade_date,
                MarketPrice.close_price,
                MarketPrice.volume,
                MarketPrice.source,
            )
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe.in_(("weekly", "monthly")),
            )
            .order_by(MarketPrice.timeframe)
        ).all()
    assert after == before


def test_blocking_quality_failure_raises_before_any_period_or_cleanup_dml(tmp_path) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    service = MarketDataService(session_factory)
    invalid = _stored_record(date(2026, 7, 6), "100", volume=None).model_copy(
        update={"amount": Decimal("1000")}
    )
    service.store_result(
        "399006",
        ProviderResult.success("DIRECT_INDEX_DAILY", [invalid]),
    )
    statements: list[str] = []
    engine = session_factory.kw["bind"]

    def capture_dml(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture_dml)
    try:
        with pytest.raises(ValueError, match="INCOMPLETE_DAILY_VOLUME"):
            service.aggregate_timeframes(
                "399006",
                expected_trade_dates=(date(2026, 7, 6),),
                as_of=date(2026, 7, 6),
            )
    finally:
        event.remove(engine, "before_cursor_execute", capture_dml)

    assert statements == []


def test_service_persists_endpoint_and_auditable_volume_source_without_internal_nulls(
    tmp_path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    engine = create_database_engine(database)
    service = MarketDataService(session_factory)
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(date(2026, 7, 6), "100", volume="100"),
                _stored_record(date(2026, 7, 8), "102", volume="200"),
            ],
        ),
    )

    service.aggregate_timeframes(
        "399006",
        expected_trade_dates=(date(2026, 7, 6), date(2026, 7, 8)),
        as_of=date(2026, 7, 8),
    )

    with Session(engine) as session:
        periods = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe.in_(("weekly", "monthly")),
            )
        ).all()
    assert {period.trade_date for period in periods} == {date(2026, 7, 8)}
    assert all(period.open_price is not None for period in periods)
    assert all(period.high_price is not None for period in periods)
    assert all(period.low_price is not None for period in periods)
    assert all(period.close_price is not None for period in periods)
    assert all(period.volume is not None for period in periods)
    assert {period.source for period in periods} == {"DIRECT_INDEX_DAILY"}
    assert {period.volume_source for period in periods} == {
        "AGGREGATED_DAILY_VOLUME"
    }


def test_aggregate_periods_derives_context_from_injected_exchange_calendar(
    tmp_path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    july_sessions = tuple(
        date(2026, 7, day)
        for day in range(6, 32)
        if date(2026, 7, day).weekday() < 5
    )
    calendar = StaticCalendarProvider(july_sessions)
    service = MarketDataService(session_factory, calendar_provider=calendar)
    completed_week = tuple(date(2026, 7, day) for day in range(6, 11))
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(day, str(100 + index), volume="100")
                for index, day in enumerate(completed_week)
            ],
        ),
    )

    result = service.aggregate_periods("399006")

    assert result["weekly"] == 1
    assert result["monthly"] == 0
    assert result["timeframe_status"]["weekly"]["status"] == "success"
    assert result["timeframe_status"]["monthly"]["status"] == "success"
    assert calendar.calls == [
        ("399006", date(2026, 7, 6), date(2026, 7, 31))
    ]


def test_display_only_etf_aggregates_real_sessions_without_fabricating_suspension(
    tmp_path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    expected = (
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 8),
        date(2026, 7, 9),
        date(2026, 7, 10),
    )
    calendar = StaticCalendarProvider(expected)
    service = MarketDataService(session_factory, calendar_provider=calendar)
    actual = tuple(day for day in expected if day != date(2026, 7, 8))
    service.store_result(
        "512690",
        ProviderResult.success(
            "DIRECT_ETF_DAILY",
            [
                _stored_record(day, str(100 + index), volume="100")
                for index, day in enumerate(actual)
            ],
        ),
    )

    result = service.aggregate_periods("512690")

    assert result["weekly"] == 1
    with session_factory() as session:
        daily_dates = tuple(
            session.scalars(
                select(MarketPrice.trade_date)
                .join(Instrument)
                .where(Instrument.code == "512690", MarketPrice.timeframe == "daily")
                .order_by(MarketPrice.trade_date)
            )
        )
    assert daily_dates == actual


@pytest.mark.parametrize("instrument_code", ["399006", "NDX"])
def test_preopen_or_not_yet_closed_market_uses_latest_daily_as_cutoff(
    tmp_path,
    instrument_code: str,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    latest_daily = date(2026, 7, 29)
    remaining_sessions = (
        date(2026, 7, 29),
        date(2026, 7, 30),
        date(2026, 7, 31),
    )
    calendar = StaticCalendarProvider(remaining_sessions)
    service = MarketDataService(session_factory, calendar_provider=calendar)
    service.store_result(
        instrument_code,
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [_stored_record(latest_daily, "20000", volume="100")],
        ),
    )

    result = service.aggregate_periods(instrument_code)

    assert result["weekly"] == 0
    assert result["monthly"] == 0
    assert calendar.calls == [
        (instrument_code, latest_daily, date(2026, 8, 2))
    ]


def test_completed_current_week_publishes_while_month_keeps_last_period(
    tmp_path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    engine = create_database_engine(database)
    service = MarketDataService(session_factory)
    completed = tuple(date(2026, 1, day) for day in range(26, 31))
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(day, str(100 + index), volume="100")
                for index, day in enumerate(completed)
            ],
        ),
    )
    service.aggregate_timeframes(
        "399006",
        expected_trade_dates=completed,
        as_of=completed[-1],
    )
    completed_week = tuple(date(2026, 2, day) for day in range(2, 7))
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(day, str(110 + index), volume="100")
                for index, day in enumerate(completed_week)
            ],
        ),
    )
    february_sessions = tuple(
        date(2026, 2, day)
        for day in range(1, 29)
        if date(2026, 2, day).weekday() < 5
    )
    expected = completed + february_sessions

    service.aggregate_timeframes(
        "399006",
        expected_trade_dates=expected,
        as_of=completed_week[-1],
    )

    with Session(engine) as session:
        after = session.execute(
            select(
                MarketPrice.timeframe,
                MarketPrice.trade_date,
                MarketPrice.close_price,
            )
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe.in_(("weekly", "monthly")),
            )
            .order_by(MarketPrice.timeframe)
        ).all()
    assert after == [
        ("monthly", date(2026, 1, 30), Decimal("104")),
        ("weekly", date(2026, 1, 30), Decimal("104")),
        ("weekly", date(2026, 2, 6), Decimal("114")),
    ]


@pytest.mark.parametrize(
    ("failing_timeframe", "successful_timeframe"),
    [("weekly", "monthly"), ("monthly", "weekly")],
)
def test_timeframe_write_failure_rolls_back_only_that_timeframe(
    tmp_path,
    failing_timeframe: str,
    successful_timeframe: str,
) -> None:
    class FaultInjectingMarketDataService(MarketDataService):
        def _aggregate(
            self,
            session,
            instrument_id,
            daily,
            timeframe,
            instrument_code,
            *,
            aggregation=None,
        ):
            counts = super()._aggregate(
                session,
                instrument_id,
                daily,
                timeframe,
                instrument_code,
                aggregation=aggregation,
            )
            if timeframe == failing_timeframe:
                session.flush()
                raise RuntimeError(f"injected {timeframe} write failure")
            return counts

    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    engine = create_database_engine(database)
    service = FaultInjectingMarketDataService(session_factory)
    trading_days = tuple(date(2026, 1, day) for day in range(26, 31))
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(day, str(100 + index), volume="100")
                for index, day in enumerate(trading_days)
            ],
        ),
    )

    result = service.aggregate_timeframes(
        "399006",
        expected_trade_dates=trading_days,
        as_of=trading_days[-1],
    )

    assert result[failing_timeframe] == 0
    assert result[successful_timeframe] == 1
    assert result["timeframe_status"][failing_timeframe]["status"] == "error"
    assert result["timeframe_status"][successful_timeframe]["status"] == "success"
    with Session(engine) as session:
        periods = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe.in_(("weekly", "monthly")),
            )
        ).all()
        logs = session.scalars(
            select(DataUpdateLog)
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                DataUpdateLog.source == "AGGREGATION",
            )
        ).all()
    assert {period.timeframe for period in periods} == {successful_timeframe}
    assert {
        log.timeframe: log.status
        for log in logs
    } == {
        failing_timeframe: "error",
        successful_timeframe: "success",
    }


@pytest.mark.parametrize(
    ("failing_timeframe", "successful_timeframe"),
    [("weekly", "monthly"), ("monthly", "weekly")],
)
def test_success_log_failure_rolls_back_only_its_timeframe(
    tmp_path,
    failing_timeframe: str,
    successful_timeframe: str,
) -> None:
    class FaultInjectingMarketDataService(MarketDataService):
        def _add_log(self, session, **values):
            if (
                values.get("timeframe") == failing_timeframe
                and values.get("status") == "success"
            ):
                raise RuntimeError(f"injected {failing_timeframe} log failure")
            return super()._add_log(session, **values)

    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    engine = create_database_engine(database)
    service = FaultInjectingMarketDataService(session_factory)
    trading_days = tuple(date(2026, 1, day) for day in range(26, 31))
    service.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(day, str(100 + index), volume="100")
                for index, day in enumerate(trading_days)
            ],
        ),
    )

    result = service.aggregate_timeframes(
        "399006",
        expected_trade_dates=trading_days,
        as_of=trading_days[-1],
    )

    assert result["timeframe_status"][failing_timeframe]["status"] == "error"
    assert result["timeframe_status"][successful_timeframe]["status"] == "success"
    with Session(engine) as session:
        periods = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(
                Instrument.code == "399006",
                MarketPrice.timeframe.in_(("weekly", "monthly")),
            )
        ).all()
    assert {period.timeframe for period in periods} == {successful_timeframe}


def test_aggregate_timeframes_and_indicators_share_exact_persisted_period_endpoints(
    tmp_path,
) -> None:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    session_factory = create_session_factory(database)
    engine = create_database_engine(database)
    market = MarketDataService(session_factory)
    trading_days = (
        date(2026, 1, 29),
        date(2026, 1, 30),
        date(2026, 2, 2),
        date(2026, 2, 3),
    )
    market.store_result(
        "399006",
        ProviderResult.success(
            "DIRECT_INDEX_DAILY",
            [
                _stored_record(day, str(100 + index), volume=str(100 + index * 10))
                for index, day in enumerate(trading_days)
            ],
        ),
    )

    market.aggregate_timeframes(
        "399006",
        expected_trade_dates=trading_days,
        as_of=trading_days[-1],
    )
    results = IndicatorService(session_factory).recalculate_all_timeframes("399006")

    assert all(result.status is IndicatorCalculationStatus.SUCCESS for result in results)
    expected_dates = {
        "daily": set(trading_days),
        "weekly": {date(2026, 1, 30), date(2026, 2, 3)},
        "monthly": {date(2026, 1, 30), date(2026, 2, 3)},
    }
    with Session(engine) as session:
        prices = session.scalars(
            select(MarketPrice)
            .join(Instrument)
            .where(Instrument.code == "399006")
            .order_by(MarketPrice.timeframe, MarketPrice.trade_date)
        ).all()
        indicators = session.scalars(
            select(IndicatorRecord)
            .join(Instrument)
            .where(Instrument.code == "399006")
            .order_by(IndicatorRecord.timeframe, IndicatorRecord.indicator_date)
        ).all()

    for timeframe, dates in expected_dates.items():
        assert {row.trade_date for row in prices if row.timeframe == timeframe} == dates
        assert {
            row.indicator_date for row in indicators if row.timeframe == timeframe
        } == dates
    weekly = [row for row in prices if row.timeframe == "weekly"]
    assert [(row.close_price, row.volume * row.volume_multiplier) for row in weekly] == [
        (Decimal("101"), Decimal("210")),
        (Decimal("103"), Decimal("250")),
    ]
    for record in indicators:
        values = record.indicator_values["values"]
        assert values["dif"] is not None
        assert values["dea"] is not None
        assert values["macd_histogram"] is not None
        assert values["volume_ma_20"] is not None
