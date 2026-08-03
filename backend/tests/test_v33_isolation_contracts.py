from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from backend.app.services.market_calendar import ExchangeCalendarProvider
from backend.app.services.v33_analysis_service import V33AnalysisError, V33AnalysisService
from backend.app.services.v33_feature_service import HORIZON_WEEKS
from backend.app.services.v33_runtime_service import V33RuntimeService
from backend.app.services.v33_training_service import FUND_ETF_RATIOS


class _RecordingShenzhenCalendar:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date]] = []

    def sessions(self, market: str, start: date, end: date) -> tuple[date, ...]:
        self.calls.append((market, start, end))
        rows: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                rows.append(current)
            current += timedelta(days=1)
        return tuple(rows)


def test_159941_analysis_and_week_sampling_keep_shenzhen_instrument_identity() -> None:
    calendar = _RecordingShenzhenCalendar()
    analysis = V33AnalysisService(
        training=SimpleNamespace(),  # type: ignore[arg-type]
        champions=SimpleNamespace(),  # type: ignore[arg-type]
        calendar=calendar,
    )

    future = analysis._future_sessions("159941", date(2026, 7, 31))
    assert len(future) == HORIZON_WEEKS * 5
    assert calendar.calls[0][0] == "159941"

    actual = tuple(
        date(2026, 7, 27) + timedelta(days=offset) for offset in range(5)
    )
    runtime_stub = SimpleNamespace(calendar=calendar)
    complete = V33RuntimeService._complete_week_sessions(
        runtime_stub, "159941", actual  # type: ignore[arg-type]
    )
    assert complete
    assert all(market == "159941" for market, _start, _end in calendar.calls)


@pytest.mark.parametrize("market", ["399006", "159941"])
def test_production_calendar_preserves_shenzhen_identity_and_discloses_proxy(
    market: str,
) -> None:
    metadata = ExchangeCalendarProvider().metadata(market)

    assert metadata == {
        "target_exchange": "XSHE",
        "schedule_provider": "exchange_calendars",
        "schedule_name": "XSHG",
        "equivalent_mainland_schedule_proxy": True,
    }


@pytest.mark.parametrize("market", ["399006", "159941"])
def test_production_calendar_clamps_packaged_schedule_without_inventing_sessions(
    market: str,
) -> None:
    calendar = ExchangeCalendarProvider()
    requested = calendar.sessions(
        market,
        date(2026, 8, 1),
        date(2027, 1, 27),
    )
    assert len(requested) >= HORIZON_WEEKS * 5
    assert requested[-1] <= date(2026, 12, 31)

    analysis = V33AnalysisService(
        training=SimpleNamespace(),  # type: ignore[arg-type]
        champions=SimpleNamespace(),  # type: ignore[arg-type]
        calendar=calendar,
    )
    assert len(analysis._future_sessions(market, date(2026, 7, 31))) == 100


def test_analysis_fails_clearly_when_published_calendar_has_too_few_sessions() -> None:
    analysis = V33AnalysisService(
        training=SimpleNamespace(),  # type: ignore[arg-type]
        champions=SimpleNamespace(),  # type: ignore[arg-type]
        calendar=ExchangeCalendarProvider(),
    )
    with pytest.raises(V33AnalysisError, match="fewer than 100 future sessions"):
        analysis._future_sessions("399006", date(2026, 12, 1))


@pytest.mark.parametrize(
    ("terminal_return", "expected"),
    [
        (0.10, "7:3"),
        (0.03, "6:4"),
        (0.00, "5:5"),
        (-0.03, "4:6"),
        (-0.10, "3:7"),
    ],
)
def test_fund_etf_output_is_exactly_one_of_five_integer_ratios(
    terminal_return: float,
    expected: str,
) -> None:
    forecast = SimpleNamespace(p50=(0.0,) * 19 + (terminal_return,))
    ratio = V33AnalysisService._fund_etf_ratio(forecast)  # type: ignore[arg-type]
    assert ratio == expected
    assert ratio in FUND_ETF_RATIOS
    assert set(FUND_ETF_RATIOS) == {"7:3", "6:4", "5:5", "4:6", "3:7"}


def test_all_position_execution_batches_remain_on_five_point_grid() -> None:
    for total in range(0, 101, 5):
        batches = V33AnalysisService._split_batches(total)
        assert sum(batches) == total
        assert len(batches) <= 4
        assert all(batch > 0 and batch % 5 == 0 for batch in batches)
