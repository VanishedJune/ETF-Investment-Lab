"""V3.5.1 behavior-equivalence detection for challengers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.models import (
    V351CandidateEvaluation,
    utc_now,
)
from backend.app.services.v351_config import (
    DIVERGENCE_THRESHOLDS,
    EQUITY_PATH_REL_TOLERANCE,
    EQUITY_TOLERANCE_CNY,
    PROTOCOL_VERSION_351,
)


@dataclass(frozen=True, slots=True)
class V351BehaviorDiff:
    forecast_divergence_ratio: float
    strategy_score_divergence_ratio: float
    base_position_divergence_ratio: float
    target_position_divergence_ratio: float
    signal_divergence_ratio: float
    trade_path_divergence_ratio: float
    equity_path_divergence_ratio: float
    effectively_identical: bool
    reason: str


@dataclass(frozen=True, slots=True)
class V351AnchorRecord:
    anchor: date
    expected_path: Sequence[float]
    strategy_score: float
    base_target_position_pp: int
    final_target_position_pp: int
    signals: Sequence[tuple[str, int]]
    trade_path: Sequence[tuple[str, int]]
    equity_path: Sequence[float]


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _ratio(champion: Sequence[float], candidate: Sequence[float]) -> float:
    if not champion or len(champion) != len(candidate):
        return 0.0
    return _fraction(
        sum(
            1
            for left, right in zip(champion, candidate)
            if abs(left - right) > 1e-6
        ),
        len(champion),
    )


def _path_ratio(
    champion: Sequence[float],
    candidate: Sequence[float],
    *,
    relative_tolerance: float,
) -> float:
    if not champion or len(champion) != len(candidate):
        return 0.0
    differing = 0
    for left, right in zip(champion, candidate):
        scale = max(abs(left), abs(right), 1e-9)
        if abs(left - right) / scale > relative_tolerance:
            differing += 1
    return _fraction(differing, len(champion))


def behavior_diff(
    champion: Sequence[V351AnchorRecord],
    candidate: Sequence[V351AnchorRecord],
) -> V351BehaviorDiff:
    """Compute the seven divergence ratios and equivalence flag."""

    champion_by_anchor = {row.anchor: row for row in champion}
    candidate_by_anchor = {row.anchor: row for row in candidate}
    anchors = sorted(set(champion_by_anchor) & set(candidate_by_anchor))
    if not anchors:
        return V351BehaviorDiff(
            forecast_divergence_ratio=0.0,
            strategy_score_divergence_ratio=0.0,
            base_position_divergence_ratio=0.0,
            target_position_divergence_ratio=0.0,
            signal_divergence_ratio=0.0,
            trade_path_divergence_ratio=0.0,
            equity_path_divergence_ratio=0.0,
            effectively_identical=True,
            reason="NO_COMMON_ANCHORS",
        )

    forecast_diff_weeks = 0
    forecast_weeks = 0
    strategy_diff = 0
    base_diff = 0
    target_diff = 0
    signal_diff = 0
    trade_nodes = 0
    trade_diff_nodes = 0
    equity_diff_weeks = 0
    equity_weeks = 0
    ending_equity_diffs: list[float] = []
    for anchor in anchors:
        left = champion_by_anchor[anchor]
        right = candidate_by_anchor[anchor]
        forecast_weeks += max(len(left.expected_path), len(right.expected_path))
        forecast_diff_weeks += sum(
            1
            for a, b in zip(left.expected_path, right.expected_path)
            if abs(a - b) > 1e-4
        )
        strategy_diff += 1 if abs(left.strategy_score - right.strategy_score) > 1e-4 else 0
        base_diff += 1 if left.base_target_position_pp != right.base_target_position_pp else 0
        target_diff += 1 if left.final_target_position_pp != right.final_target_position_pp else 0
        signal_diff += 1 if left.signals != right.signals else 0
        left_trades = tuple(left.trade_path)
        right_trades = tuple(right.trade_path)
        nodes = max(len(left_trades), len(right_trades))
        trade_nodes += nodes
        trade_diff_nodes += sum(
            1
            for index in range(nodes)
            if (
                index >= len(left_trades)
                or index >= len(right_trades)
                or left_trades[index] != right_trades[index]
            )
        )
        equity_weeks += max(len(left.equity_path), len(right.equity_path))
        equity_diff_weeks += sum(
            1
            for a, b in zip(left.equity_path, right.equity_path)
            if abs(a - b) / max(abs(a), abs(b), 1e-9) > EQUITY_PATH_REL_TOLERANCE
        )
        if left.equity_path and right.equity_path:
            ending_equity_diffs.append(
                abs(left.equity_path[-1] - right.equity_path[-1])
            )

    forecast_divergence = _fraction(forecast_diff_weeks, forecast_weeks)
    strategy_divergence = _fraction(strategy_diff, len(anchors))
    base_divergence = _fraction(base_diff, len(anchors))
    target_divergence = _fraction(target_diff, len(anchors))
    signal_divergence = _fraction(signal_diff, len(anchors))
    trade_divergence = _fraction(trade_diff_nodes, trade_nodes)
    equity_divergence = _fraction(equity_diff_weeks, equity_weeks)

    identical = (
        target_diff == 0
        and signal_diff == 0
        and trade_diff_nodes == 0
        and (not ending_equity_diffs or max(ending_equity_diffs) < EQUITY_TOLERANCE_CNY)
        and equity_diff_weeks == 0
    )
    reason = (
        "最终目标仓位、买卖信号、交易流水、期末资产与净值路径完全一致"
        if identical
        else "存在行为差异"
    )
    return V351BehaviorDiff(
        forecast_divergence_ratio=forecast_divergence,
        strategy_score_divergence_ratio=strategy_divergence,
        base_position_divergence_ratio=base_divergence,
        target_position_divergence_ratio=target_divergence,
        signal_divergence_ratio=signal_divergence,
        trade_path_divergence_ratio=trade_divergence,
        equity_path_divergence_ratio=equity_divergence,
        effectively_identical=identical,
        reason=reason,
    )


def passes_divergence_prefilter(diff: V351BehaviorDiff) -> bool:
    """Return whether a candidate produces real decision differences."""

    thresholds = DIVERGENCE_THRESHOLDS
    return (
        diff.forecast_divergence_ratio >= thresholds["forecast_divergence_ratio"]
        or diff.strategy_score_divergence_ratio >= thresholds["strategy_score_ratio"]
        or diff.base_position_divergence_ratio >= thresholds["base_position_ratio"]
        or diff.target_position_divergence_ratio >= thresholds["target_position_ratio"]
        or diff.trade_path_divergence_ratio >= thresholds["trade_path_ratio"]
    )


def persist_candidate_evaluation(
    session: Session,
    *,
    challenge_id: str,
    candidate_package_id: str,
    model_market: str,
    challenger_family: str,
    diff: V351BehaviorDiff,
    rejection_reason_codes: Sequence[str],
    gaps: Mapping[str, float],
    promotion_channel: str | None,
    raw_window_count: int,
    effective_window_count: int,
    prediction_quality: Mapping[str, Any],
) -> V351CandidateEvaluation:
    existing = session.scalar(
        select(V351CandidateEvaluation).where(
            V351CandidateEvaluation.challenge_id == challenge_id,
            V351CandidateEvaluation.candidate_package_id == candidate_package_id,
        )
    )
    if existing is not None:
        return existing
    payload = {
        "challenge_id": challenge_id,
        "candidate_package_id": candidate_package_id,
        "model_market": model_market,
        "challenger_family": challenger_family,
        "effectively_identical": diff.effectively_identical,
        "reason": diff.reason,
        "divergence": {
            "forecast": diff.forecast_divergence_ratio,
            "strategy_score": diff.strategy_score_divergence_ratio,
            "base_position": diff.base_position_divergence_ratio,
            "target_position": diff.target_position_divergence_ratio,
            "signal": diff.signal_divergence_ratio,
            "trade_path": diff.trade_path_divergence_ratio,
            "equity_path": diff.equity_path_divergence_ratio,
        },
        "rejection_reason_codes": list(rejection_reason_codes),
        "gaps": dict(gaps),
        "promotion_channel": promotion_channel,
        "raw_window_count": raw_window_count,
        "effective_window_count": effective_window_count,
        "prediction_quality": dict(prediction_quality),
    }
    evaluation_hash = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    row = V351CandidateEvaluation(
        protocol_version=PROTOCOL_VERSION_351,
        challenge_id=challenge_id,
        candidate_package_id=candidate_package_id,
        model_market=model_market,
        challenger_family=challenger_family,
        effectively_identical=diff.effectively_identical,
        equivalence_reason=diff.reason,
        forecast_divergence_ratio=Decimal(str(round(diff.forecast_divergence_ratio, 8))),
        strategy_score_divergence_ratio=Decimal(
            str(round(diff.strategy_score_divergence_ratio, 8))
        ),
        base_position_divergence_ratio=Decimal(
            str(round(diff.base_position_divergence_ratio, 8))
        ),
        target_position_divergence_ratio=Decimal(
            str(round(diff.target_position_divergence_ratio, 8))
        ),
        signal_divergence_ratio=Decimal(str(round(diff.signal_divergence_ratio, 8))),
        trade_path_divergence_ratio=Decimal(str(round(diff.trade_path_divergence_ratio, 8))),
        equity_path_divergence_ratio=Decimal(str(round(diff.equity_path_divergence_ratio, 8))),
        rejection_reason_codes_json=list(rejection_reason_codes),
        gaps_json=dict(gaps),
        promotion_channel=promotion_channel,
        raw_evaluation_window_count=raw_window_count,
        effective_independent_window_count=effective_window_count,
        prediction_quality_json=dict(prediction_quality),
        evaluation_hash=evaluation_hash,
        created_at=utc_now(),
    )
    session.add(row)
    session.flush()
    return row


def anchor_records_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[V351AnchorRecord, ...]:
    """Convert persisted row dicts into anchor records."""

    result: list[V351AnchorRecord] = []
    for row in rows:
        result.append(
            V351AnchorRecord(
                anchor=date.fromisoformat(str(row["anchor"])),
                expected_path=[float(v) for v in row["expected_path"]],
                strategy_score=float(row["strategy_score"]),
                base_target_position_pp=int(row["base_target_position_pp"]),
                final_target_position_pp=int(row["final_target_position_pp"]),
                signals=tuple(
                    (str(action), int(pp))
                    for action, pp in row.get("signals", [])
                ),
                trade_path=tuple(
                    (str(action), int(pp))
                    for action, pp in row.get("trade_path", [])
                ),
                equity_path=[float(v) for v in row.get("equity_path", [])],
            )
        )
    return tuple(result)


def effective_independent_window_count(intervals: Sequence[tuple[date, date]]) -> int:
    """Conservative count of non-overlapping closed intervals."""

    normalized = sorted(intervals, key=lambda item: (item[1], item[0]))
    count = 0
    last_end: date | None = None
    for start, end in normalized:
        if last_end is None or start >= last_end:
            count += 1
            last_end = end
    return count
