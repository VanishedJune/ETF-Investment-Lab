"""Deterministic sampling of real sessions from explicitly completed ISO weeks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Iterator


SAMPLING_RULE_VERSION = "sha256-byte-rejection-v1"
COMPLETION_RULE_VERSION = "explicit-completed-iso-weeks-v1"


@dataclass(frozen=True, slots=True)
class WeekSample:
    """One auditable checkpoint sampled from a completed natural week."""

    week_key: str
    assigned_month: str
    sample_date: date
    candidate_dates: tuple[date, ...]
    previous_completed_week_end: date | None
    seed: int
    algorithm_version: str
    sampling_rule_version: str
    completion_rule_version: str
    completion_mode: str
    completed_week_keys: tuple[str, ...]
    completed_through: date | None
    symbol: str
    source_hash: str


def _week_key(day: date) -> str:
    iso_year, iso_week, _ = day.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _source_hash(
    *,
    candidates: tuple[date, ...],
    seed: int,
    algorithm_version: str,
    symbol: str,
    week_key: str,
    previous_completed_week_end: date | None,
    completion_mode: str,
    completed_week_keys: tuple[str, ...],
    completed_through: date | None,
) -> str:
    payload = {
        "algorithm_version": algorithm_version,
        "candidate_dates": [day.isoformat() for day in candidates],
        "completion_rule_version": COMPLETION_RULE_VERSION,
        "completion_state": {
            "completed_through": (
                completed_through.isoformat()
                if completed_through is not None
                else None
            ),
            "mode": completion_mode,
            "selected_week_keys": list(completed_week_keys),
        },
        "previous_completed_week_end": (
            previous_completed_week_end.isoformat()
            if previous_completed_week_end is not None
            else None
        ),
        "sampling_rule_version": SAMPLING_RULE_VERSION,
        "seed": seed,
        "symbol": symbol,
        "week_key": week_key,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _uniform_index_from_bytes(
    candidate_count: int,
    digest_bytes: Iterable[int],
) -> int:
    """Map a deterministic byte stream uniformly with rejection sampling."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    width = max(1, (candidate_count.bit_length() + 7) // 8)
    range_size = 1 << (width * 8)
    acceptance_limit = range_size - (range_size % candidate_count)
    stream = iter(digest_bytes)
    while True:
        chunk: list[int] = []
        for _ in range(width):
            try:
                value = next(stream)
            except StopIteration as error:
                raise ValueError(
                    "Digest stream ended before an unbiased index was selected"
                ) from error
            if not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError("Digest stream values must be bytes")
            chunk.append(value)
        candidate = int.from_bytes(bytes(chunk), "big")
        if candidate < acceptance_limit:
            return candidate % candidate_count


def _sha256_byte_stream(seed: int, symbol: str, week_key: str) -> Iterator[int]:
    random_key = f"{seed}:{symbol}:{week_key}"
    counter = 0
    while True:
        material = random_key if counter == 0 else f"{random_key}:{counter}"
        yield from hashlib.sha256(material.encode("utf-8")).digest()
        counter += 1


def build_week_samples(
    trading_dates: Iterable[date],
    seed: int = 20260730,
    algorithm_version: str = "weekly-v2",
    symbol: str = "",
    *,
    completed_through: date | None = None,
    complete_week_keys: Iterable[str] | None = None,
) -> tuple[WeekSample, ...]:
    """Return one deterministic real-session sample per explicitly completed week.

    Duplicate and non-monotonic source dates are normalized to sorted unique
    sessions. Completion is never inferred from the weekday of the last row.
    Callers must provide either a trusted ``completed_through`` boundary or the
    exact ISO week keys that their exchange calendar has marked complete.
    """

    if not symbol:
        raise ValueError("symbol is required for independent deterministic sampling")
    if not algorithm_version:
        raise ValueError("algorithm_version must not be empty")
    if completed_through is not None and complete_week_keys is not None:
        raise ValueError(
            "Provide only one completion state: completed_through or complete_week_keys"
        )
    if completed_through is None and complete_week_keys is None:
        raise ValueError(
            "Explicit completion state is required; completion is never guessed"
        )

    normalized = tuple(sorted(set(trading_dates)))
    if any(not isinstance(day, date) for day in normalized):
        raise TypeError("trading_dates must contain date values")
    if not normalized:
        return ()

    grouped: dict[str, list[date]] = defaultdict(list)
    for day in normalized:
        grouped[_week_key(day)].append(day)

    if complete_week_keys is not None:
        completed_keys = frozenset(complete_week_keys)
        selected_keys = sorted(key for key in grouped if key in completed_keys)
        completion_mode = "complete_week_keys"
        completion_boundary = None
    else:
        assert completed_through is not None
        selected_keys = sorted(
            key
            for key, candidates in grouped.items()
            if candidates[-1] <= completed_through
        )
        completion_mode = "completed_through"
        completion_boundary = completed_through
    selected_key_tuple = tuple(selected_keys)

    samples: list[WeekSample] = []
    previous_week_end: date | None = None
    for week_key in selected_keys:
        candidates = tuple(grouped[week_key])
        sample_index = _uniform_index_from_bytes(
            len(candidates),
            _sha256_byte_stream(seed, symbol, week_key),
        )
        samples.append(
            WeekSample(
                week_key=week_key,
                assigned_month=candidates[-1].strftime("%Y-%m"),
                sample_date=candidates[sample_index],
                candidate_dates=candidates,
                previous_completed_week_end=previous_week_end,
                seed=seed,
                algorithm_version=algorithm_version,
                sampling_rule_version=SAMPLING_RULE_VERSION,
                completion_rule_version=COMPLETION_RULE_VERSION,
                completion_mode=completion_mode,
                completed_week_keys=selected_key_tuple,
                completed_through=completion_boundary,
                symbol=symbol,
                source_hash=_source_hash(
                    candidates=candidates,
                    seed=seed,
                    algorithm_version=algorithm_version,
                    symbol=symbol,
                    week_key=week_key,
                    previous_completed_week_end=previous_week_end,
                    completion_mode=completion_mode,
                    completed_week_keys=selected_key_tuple,
                    completed_through=completion_boundary,
                ),
            )
        )
        previous_week_end = candidates[-1]
    return tuple(samples)
