from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta

import pytest

from backend.app import weekly_analysis
from backend.app.weekly_analysis import sampling as sampling_module
from backend.app.weekly_analysis.sampling import build_week_samples


def _week_key(day: date) -> str:
    iso_year, iso_week, _ = day.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _weekdays(start: date, end: date) -> tuple[date, ...]:
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return tuple(days)


def _seed_selecting(
    *, symbol: str, week_key: str, candidate_count: int, index: int
) -> int:
    for seed in range(100_000):
        key = f"{seed}:{symbol}:{week_key}"
        limit = 256 - (256 % candidate_count)
        counter = 0
        while True:
            material = key if counter == 0 else f"{key}:{counter}"
            for value in hashlib.sha256(material.encode("utf-8")).digest():
                if value < limit:
                    if value % candidate_count == index:
                        return seed
                    break
            else:
                counter += 1
                continue
            break
    raise AssertionError("No deterministic seed found")


def test_sampling_uses_iso_weeks_and_assigns_month_from_last_real_session() -> None:
    trading_dates = (
        date(2026, 1, 29),
        date(2026, 1, 30),
        date(2026, 2, 2),
        date(2026, 2, 3),
    )

    samples = build_week_samples(
        trading_dates,
        symbol="399006",
        complete_week_keys=("2026-W05", "2026-W06"),
    )

    assert [(sample.week_key, sample.assigned_month) for sample in samples] == [
        ("2026-W05", "2026-01"),
        ("2026-W06", "2026-02"),
    ]
    assert samples[0].candidate_dates == (
        date(2026, 1, 29),
        date(2026, 1, 30),
    )
    assert samples[1].candidate_dates == (
        date(2026, 2, 2),
        date(2026, 2, 3),
    )


def test_sampling_is_repeatable_uses_only_candidates_and_is_symbol_independent() -> None:
    trading_dates = _weekdays(date(2026, 1, 5), date(2026, 3, 27))
    complete = tuple(sorted({_week_key(day) for day in trading_dates}))

    first = build_week_samples(
        trading_dates,
        symbol="399006",
        complete_week_keys=complete,
    )
    second = build_week_samples(
        trading_dates,
        symbol="399006",
        complete_week_keys=complete,
    )
    ndx = build_week_samples(
        trading_dates,
        symbol="NDX",
        complete_week_keys=complete,
    )

    assert first == second
    assert all(sample.sample_date in sample.candidate_dates for sample in first)
    assert [sample.sample_date for sample in first] != [
        sample.sample_date for sample in ndx
    ]


def test_sampling_normalizes_duplicate_non_monotonic_input_without_fake_sessions() -> None:
    actual = (
        date(2026, 1, 5),
        date(2026, 1, 6),
        date(2026, 1, 8),
        date(2026, 1, 9),
    )
    noisy = (actual[2], actual[0], actual[2], actual[3], actual[1])

    clean = build_week_samples(
        actual,
        symbol="399006",
        complete_week_keys=("2026-W02",),
    )
    normalized = build_week_samples(
        noisy,
        symbol="399006",
        complete_week_keys=("2026-W02",),
    )

    assert normalized == clean
    assert normalized[0].candidate_dates == actual
    assert build_week_samples(
        (),
        symbol="399006",
        complete_week_keys=(),
    ) == ()


def test_sampling_handles_iso_year_boundary_and_a_five_week_natural_month() -> None:
    dates = (
        date(2025, 12, 29),
        date(2025, 12, 31),
        date(2026, 1, 2),
        date(2026, 1, 5),
        date(2026, 1, 9),
        date(2026, 1, 12),
        date(2026, 1, 16),
        date(2026, 1, 19),
        date(2026, 1, 23),
        date(2026, 1, 26),
        date(2026, 1, 30),
    )
    complete = tuple(sorted({_week_key(day) for day in dates}))

    samples = build_week_samples(
        dates,
        symbol="399006",
        complete_week_keys=complete,
    )

    assert samples[0].week_key == "2026-W01"
    assert samples[0].candidate_dates == (
        date(2025, 12, 29),
        date(2025, 12, 31),
        date(2026, 1, 2),
    )
    assert samples[0].assigned_month == "2026-01"
    assert sum(sample.assigned_month == "2026-01" for sample in samples) == 5


def test_latest_incomplete_week_requires_explicit_completion_state() -> None:
    dates = (
        date(2026, 1, 29),
        date(2026, 1, 30),
        date(2026, 2, 2),
        date(2026, 2, 3),
    )

    samples = build_week_samples(
        dates,
        symbol="399006",
        completed_through=date(2026, 1, 30),
    )

    assert [sample.week_key for sample in samples] == ["2026-W05"]
    with pytest.raises(ValueError, match="completion"):
        build_week_samples(dates, symbol="399006")
    with pytest.raises(ValueError, match="only one"):
        build_week_samples(
            dates,
            symbol="399006",
            completed_through=date(2026, 1, 30),
            complete_week_keys=("2026-W05",),
        )


@pytest.mark.parametrize("sample_index", [0, 4])
def test_previous_completed_week_end_prevents_monday_or_friday_sample_leakage(
    sample_index: int,
) -> None:
    dates = _weekdays(date(2026, 1, 5), date(2026, 1, 16))
    second_week = "2026-W03"
    seed = _seed_selecting(
        symbol="399006",
        week_key=second_week,
        candidate_count=5,
        index=sample_index,
    )

    samples = build_week_samples(
        dates,
        seed=seed,
        symbol="399006",
        complete_week_keys=("2026-W02", "2026-W03"),
    )

    assert samples[1].sample_date.weekday() == sample_index
    assert samples[1].previous_completed_week_end == date(2026, 1, 9)
    assert samples[1].previous_completed_week_end < min(samples[1].candidate_dates)


def test_sample_source_hash_is_stable_and_changes_with_real_input_or_parameters() -> None:
    dates = _weekdays(date(2026, 1, 5), date(2026, 1, 9))
    base = build_week_samples(
        dates,
        symbol="399006",
        complete_week_keys=("2026-W02",),
    )[0]
    reordered = build_week_samples(
        tuple(reversed(dates)) + (dates[0],),
        symbol="399006",
        complete_week_keys=("2026-W02",),
    )[0]
    changed_candidates = build_week_samples(
        dates[:-1],
        symbol="399006",
        complete_week_keys=("2026-W02",),
    )[0]
    changed_seed = build_week_samples(
        dates,
        seed=1,
        symbol="399006",
        complete_week_keys=("2026-W02",),
    )[0]

    assert reordered.source_hash == base.source_hash
    assert changed_candidates.source_hash != base.source_hash
    assert changed_seed.source_hash != base.source_hash
    assert base.seed == 20260730
    assert base.algorithm_version == "weekly-v2"


def test_source_hash_binds_previous_safe_cutoff_and_full_completion_selection() -> None:
    dates = _weekdays(date(2026, 1, 5), date(2026, 1, 23))

    earlier_previous = build_week_samples(
        dates,
        symbol="399006",
        complete_week_keys=("2026-W02", "2026-W04"),
    )[-1]
    later_previous = build_week_samples(
        dates,
        symbol="399006",
        complete_week_keys=("2026-W03", "2026-W04"),
    )[-1]
    same_previous_different_selection = build_week_samples(
        dates,
        symbol="399006",
        complete_week_keys=("2026-W02", "2026-W03", "2026-W04"),
    )[-1]

    assert earlier_previous.candidate_dates == later_previous.candidate_dates
    assert (
        earlier_previous.previous_completed_week_end
        != later_previous.previous_completed_week_end
    )
    assert earlier_previous.source_hash != later_previous.source_hash
    assert (
        later_previous.previous_completed_week_end
        == same_previous_different_selection.previous_completed_week_end
    )
    assert (
        later_previous.source_hash
        != same_previous_different_selection.source_hash
    )


@pytest.mark.parametrize(
    ("candidate_count", "digest_values", "expected"),
    [
        (3, (255, 4), 1),
        (5, (255, 10), 0),
    ],
)
def test_uniform_index_rejects_biased_digest_tail_before_selecting(
    candidate_count: int,
    digest_values: tuple[int, ...],
    expected: int,
) -> None:
    assert sampling_module._uniform_index_from_bytes(  # type: ignore[attr-defined]
        candidate_count,
        iter(digest_values),
    ) == expected


def test_sampling_is_reproducible_across_python_hash_seeds_and_processes() -> None:
    script = """
import json
from datetime import date, timedelta
from backend.app.weekly_analysis.sampling import build_week_samples
start = date(2026, 1, 5)
days = tuple(start + timedelta(days=i) for i in range(5))
sample = build_week_samples(
    days,
    symbol="399006",
    complete_week_keys=("2026-W02",),
)[0]
print(json.dumps({
    "sample_date": sample.sample_date.isoformat(),
    "source_hash": sample.source_hash,
}, sort_keys=True))
"""
    outputs: list[dict[str, str]] = []
    for hash_seed in ("1", "987654"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=environment,
            text=True,
        )
        outputs.append(json.loads(output))

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("symbol", ["399006", "NDX"])
def test_ten_year_window_has_one_sample_per_actual_complete_iso_week(
    symbol: str,
) -> None:
    dates = _weekdays(date(2016, 8, 1), date(2026, 7, 24))
    if symbol == "NDX":
        # Static XNAS-like holiday removals change candidates, never week identity.
        dates = tuple(
            day
            for day in dates
            if not (day.month == 7 and day.day == 4)
        )
    complete = tuple(sorted({_week_key(day) for day in dates}))

    samples = build_week_samples(
        dates,
        symbol=symbol,
        complete_week_keys=complete,
    )

    assert 515 <= len(samples) <= 525
    assert len(samples) == len(set(sample.week_key for sample in samples))
    assert {sample.week_key for sample in samples} == set(complete)


def test_sampling_is_available_from_the_weekly_analysis_public_boundary() -> None:
    assert weekly_analysis.build_week_samples is build_week_samples
