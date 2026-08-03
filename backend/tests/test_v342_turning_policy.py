from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func, select

import pytest

from backend.app.services.v342_turning_policy_service import (
    BacktestResult,
    TurningAssessmentResult,
    TurningCandidate,
    assess_turning_points,
    backtest_policy,
    build_policy_decision,
    observed_market_confirmation,
    transition_batch_state,
    V342TurningPolicyService,
)
from backend.app.models.models import (
    V341Forecast,
    V342PolicyVersion,
    V342PolicyBatch,
    V342PolicyBatchEvent,
    V342PolicyRun,
    V342TurningAssessment,
)
from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.services.v34_training_service import build_training_samples


ANCHOR = date(2026, 8, 7)


def _history() -> list[dict[str, object]]:
    rows = []
    for index in range(60):
        close = 100.0 + 0.02 * ((index % 5) - 2)
        rows.append(
            {
                "date": (ANCHOR - timedelta(days=7 * (59 - index))).isoformat(),
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


def _bars(closes: list[float]) -> list[dict[str, object]]:
    return [
        {
            "week": index + 1,
            "week_start": (ANCHOR + timedelta(days=7 * index + 1)).isoformat(),
            "week_end": (ANCHOR + timedelta(days=7 * (index + 1))).isoformat(),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume_p50": 1_000_000,
        }
        for index, close in enumerate(closes)
    ]


def _indicators(d1: list[float]) -> list[dict[str, object]]:
    dif = []
    current = 0.0
    for value in d1:
        current += value
        dif.append(current)
    dea = [value - 0.005 for value in dif]
    macd = [2.0 * (left - right) for left, right in zip(dif, dea)]
    return [
        {
            "week": index + 1,
            "week_end": (ANCHOR + timedelta(days=7 * (index + 1))).isoformat(),
            "dif": dif[index],
            "dea": dea[index],
            "macd": macd[index] + index * 0.0001,
            "dif_first_change": d1[index],
            "dif_second_change": d1[index] - (d1[index - 1] if index else 0.0),
        }
        for index in range(13)
    ]


def _assessment(kind: str = "BOTTOM", consistency: str = "TEMPORALLY_CONSISTENT"):
    candidate = TurningCandidate(
        signal_kind="PRICE",
        turn_kind=kind,
        candidate_week=5,
        window_start_date=ANCHOR,
        window_end_date=ANCHOR,
        classification="VALID_TURN",
        local_extremum_met=True,
        direction_reversal_met=True,
        prominence_value=0.1,
        minimum_prominence=0.01,
        persistence_periods=1,
        minimum_persistence_periods=1,
        move_magnitude=0.2,
        minimum_move_magnitude=0.02,
        neutral_threshold=0.01,
        confirmation_periods=1,
        confirmation_status="CONFIRMED",
        evidence={},
    )
    dif_candidate = TurningCandidate(
        **{**candidate.__dict__, "signal_kind": "DIF"}
    )
    return TurningAssessmentResult(
        window_extrema={},
        candidates=(candidate, dif_candidate),
        price_turn_status="CONFIRMED",
        dif_turn_status="CONFIRMED",
        consistency_status=consistency,
        top_bottom_conflict=consistency == "TYPE_CONFLICT",
        criteria={},
        input_hash="input",
        assessment_hash="assessment",
    )


def test_monotonic_path_has_extrema_but_no_valid_price_turn() -> None:
    result = assess_turning_points(
        _bars([100 + index for index in range(13)]),
        _indicators([0.01] * 13),
        _history(),
    )
    assert result.window_extrema["lowest"]["week"] == 1
    assert result.window_extrema["highest"]["week"] == 13
    assert result.price_turn_status == "NO_VALID_TURN"
    assert not any(
        item.signal_kind == "PRICE" and item.classification == "VALID_TURN"
        for item in result.candidates
    )


def test_window_extrema_use_ohlc_high_and_low_instead_of_close() -> None:
    bars = _bars([100 + index for index in range(13)])
    bars[4]["high"] = 999.0
    bars[6]["low"] = 1.0
    result = assess_turning_points(bars, _indicators([0.01] * 13), _history())
    assert result.window_extrema["highest"]["week"] == 5
    assert result.window_extrema["highest"]["price"] == 999.0
    assert result.window_extrema["lowest"]["week"] == 7
    assert result.window_extrema["lowest"]["price"] == 1.0


def test_market_ohlc_uses_one_direct_etf_qfq_scale() -> None:
    from types import SimpleNamespace
    import backend.app.services.v342_turning_policy_service as policy_module

    row = SimpleNamespace(
        open_price=9.0,
        high_price=11.0,
        low_price=8.0,
        close_price=10.0,
        adjusted_close_price=5.0,
    )
    scaled = policy_module._scaled_market_ohlc(row)
    assert scaled == {"open": 4.5, "high": 5.5, "low": 4.0, "close": 5.0}


def test_shallow_price_noise_and_tiny_dif_cross_do_not_confirm() -> None:
    closes = [100.0, 99.99, 100.0] + [100.0 + index * 0.001 for index in range(10)]
    d1 = [-1e-12, 1e-12] + [1e-12] * 11
    result = assess_turning_points(_bars(closes), _indicators(d1), _history())
    assert result.price_turn_status == "NO_VALID_TURN"
    assert result.dif_turn_status == "NO_VALID_TURN"


def test_real_v_bottom_needs_and_receives_following_confirmation() -> None:
    closes = [100, 99, 98, 96, 90, 96, 98, 99, 100, 101, 102, 103, 104]
    d1 = [-0.02, -0.02, -0.02, -0.02, 0.02, 0.015, 0.012, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    result = assess_turning_points(_bars(closes), _indicators(d1), _history())
    assert result.price_turn_status == "CONFIRMED"
    assert result.dif_turn_status == "CONFIRMED"
    assert result.consistency_status == "TEMPORALLY_CONSISTENT"
    assert any(
        item.turn_kind == "BOTTOM" and item.confirmation_status == "CONFIRMED"
        for item in result.candidates
    )


def test_last_week_dif_cross_remains_pending() -> None:
    d1 = [-0.02] * 12 + [0.02]
    result = assess_turning_points(
        _bars([100 + index for index in range(13)]),
        _indicators(d1),
        _history(),
    )
    assert result.dif_turn_status == "PENDING"
    assert not any(
        item.signal_kind == "DIF" and item.confirmation_status == "CONFIRMED"
        for item in result.candidates
    )


def test_price_bottom_and_dif_top_are_never_consistent() -> None:
    closes = [100, 99, 98, 96, 90, 96, 98, 99, 100, 101, 102, 103, 104]
    d1 = [0.02, 0.02, 0.02, 0.02, -0.02, -0.015, -0.012, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
    indicators = _indicators(d1)
    # Mirror DEA/MACD confirmation for a top.
    for index, row in enumerate(indicators):
        row["dea"] = float(row["dif"]) + 0.005
        row["macd"] = -0.01 - index * 0.001
    result = assess_turning_points(_bars(closes), indicators, _history())
    assert result.price_turn_status == "CONFIRMED"
    assert result.dif_turn_status == "CONFIRMED"
    assert result.consistency_status == "TYPE_CONFLICT"
    assert result.top_bottom_conflict is True


def test_probability_edge_cannot_jump_zero_position_to_high_exposure() -> None:
    decision = build_policy_decision(
        current_position=0,
        horizon_probabilities={
            13: {"calibrated": {"up": 0.4655, "sideways": 0.3757, "down": 0.1588}}
        },
        reliability_score=68,
        health_status="MODEL_NORMAL",
        assessment=_assessment(),
        anchor_date=ANCHOR,
    )
    assert decision.uncapped_target == 70
    assert decision.next_executable_position == 10
    assert decision.next_executable_position < decision.target_position
    assert len(decision.batches) <= 4
    assert all(int(item["change_pp"]) <= 20 for item in decision.batches)
    assert all(int(item["change_pp"]) % 5 == 0 for item in decision.batches)


def test_week12_turn_does_not_schedule_or_execute_batches_in_first_month() -> None:
    turn_start = ANCHOR + timedelta(days=7 * 10)
    turn_end = ANCHOR + timedelta(days=7 * 12)
    base = _assessment()
    candidates = tuple(
        TurningCandidate(
            **{
                **candidate.__dict__,
                "candidate_week": 12,
                "window_start_date": turn_start,
                "window_end_date": turn_end,
            }
        )
        for candidate in base.candidates
    )
    assessment = TurningAssessmentResult(
        **{**base.__dict__, "candidates": candidates}
    )
    decision = build_policy_decision(
        current_position=0,
        horizon_probabilities={13: {"calibrated": {"up": 0.8, "sideways": 0.1, "down": 0.1}}},
        reliability_score=90,
        health_status="MODEL_NORMAL",
        assessment=assessment,
        observed_confirmation_met=True,
        anchor_date=ANCHOR,
    )
    assert decision.batches
    assert date.fromisoformat(decision.batches[0]["window_start_date"]) >= turn_start
    assert all(
        date.fromisoformat(batch["window_start_date"]) >= turn_start
        for batch in decision.batches
    )
    assert decision.next_executable_position == 0
    assert decision.batches[0]["initial_state"] == "WAITING_CONFIRMATION"


def test_observed_dif_cross_requires_a_complete_following_period(monkeypatch) -> None:
    import backend.app.services.v342_turning_policy_service as policy_module

    history = _history()[-30:]
    d1 = np.full(29, -0.1, dtype=float)
    d1[-2] = -0.2
    d1[-1] = 0.3
    dif = np.concatenate(([0.0], np.cumsum(d1)))
    dea = np.linspace(-1.0, 1.0, 30)
    histogram = np.linspace(-1.0, 1.0, 30)
    monkeypatch.setattr(policy_module, "_macd", lambda _closes: (dif, dea, histogram))
    assert observed_market_confirmation(history, action="BUY", dif_epsilon=0.05) is False


def test_conflict_degraded_and_cooldown_states_block_unsafe_increase() -> None:
    probabilities = {13: {"calibrated": {"up": 0.8, "sideways": 0.1, "down": 0.1}}}
    conflict = build_policy_decision(
        current_position=0,
        horizon_probabilities=probabilities,
        reliability_score=90,
        health_status="MODEL_NORMAL",
        assessment=_assessment(consistency="TYPE_CONFLICT"),
        anchor_date=ANCHOR,
    )
    degraded = build_policy_decision(
        current_position=20,
        horizon_probabilities=probabilities,
        reliability_score=90,
        health_status="MODEL_DEGRADED",
        assessment=_assessment(),
        anchor_date=ANCHOR,
    )
    cooldown = build_policy_decision(
        current_position=0,
        horizon_probabilities=probabilities,
        reliability_score=90,
        health_status="MODEL_NORMAL",
        assessment=_assessment(),
        cooldown_eligible=False,
        anchor_date=ANCHOR,
    )
    assert conflict.action == "HOLD"
    assert degraded.next_executable_position == 20
    assert cooldown.next_executable_position == 0
    assert cooldown.batches[0]["initial_state"] == "WAITING_CONFIRMATION"


def test_batch_state_requires_previous_execution_cooldown_and_confirmation() -> None:
    assert transition_batch_state(
        current_state="WAITING_CONFIRMATION",
        previous_executed=True,
        cooldown_sessions=99,
        confirmation_met=True,
        invalidated=False,
        expired=False,
        execution_observed=False,
        not_started=True,
    ) == "WAITING_CONFIRMATION"
    assert transition_batch_state(
        current_state="WAITING_CONFIRMATION",
        previous_executed=False,
        cooldown_sessions=99,
        confirmation_met=True,
        invalidated=False,
        expired=False,
        execution_observed=False,
    ) == "DEFERRED"
    assert transition_batch_state(
        current_state="DEFERRED",
        previous_executed=True,
        cooldown_sessions=4,
        confirmation_met=True,
        invalidated=False,
        expired=False,
        execution_observed=False,
    ) == "DEFERRED"
    assert transition_batch_state(
        current_state="DEFERRED",
        previous_executed=True,
        cooldown_sessions=5,
        confirmation_met=True,
        invalidated=False,
        expired=False,
        execution_observed=False,
    ) == "ELIGIBLE"
    assert transition_batch_state(
        current_state="ELIGIBLE",
        previous_executed=True,
        cooldown_sessions=5,
        confirmation_met=True,
        invalidated=False,
        expired=False,
        execution_observed=True,
    ) == "EXECUTED_OBSERVED"
    assert transition_batch_state(
        current_state="ELIGIBLE",
        previous_executed=True,
        cooldown_sessions=5,
        confirmation_met=True,
        invalidated=True,
        expired=False,
        execution_observed=True,
    ) == "CANCELLED"
    assert transition_batch_state(
        current_state="ELIGIBLE",
        previous_executed=True,
        cooldown_sessions=5,
        confirmation_met=True,
        invalidated=False,
        expired=True,
        execution_observed=True,
    ) == "EXPIRED"


def test_backtest_cost_turnover_drawdown_and_benchmarks_are_auditable() -> None:
    prices = []
    for index, value in enumerate((100, 110, 99, 105)):
        close_day = ANCHOR + timedelta(days=7 * index)
        open_day = close_day - timedelta(days=4)
        prices.append(
            {
                "date": close_day.isoformat(),
                "open_date": open_day.isoformat(),
                "trading_dates": [
                    (open_day + timedelta(days=offset)).isoformat()
                    for offset in range(5)
                ],
                "open": value,
                "close": value,
            }
        )
    decisions = [
        {
            "anchor_date": prices[0]["date"],
            "forecast_id": 1,
            "next_executable_position": 10,
        },
        {
            "anchor_date": prices[1]["date"],
            "forecast_id": 2,
            "next_executable_position": 20,
        },
    ]
    free: BacktestResult = backtest_policy(decisions, prices, transaction_cost_bps=0)
    paid: BacktestResult = backtest_policy(decisions, prices, transaction_cost_bps=10)
    assert paid.metrics["turnover"] == pytest.approx(0.2)
    assert paid.metrics["transaction_cost"] == pytest.approx(0.0002)
    assert paid.metrics["realized_return"] < free.metrics["realized_return"]
    assert paid.metrics["maximum_drawdown"] <= 0.0
    assert paid.metrics["risk_adjusted_metric"] == "ANNUALIZED_WEEKLY_SHARPE_RF_0"
    assert paid.metrics["fixed_dca_semantics"] == "FIXED_5PP_WEEKLY_EXPOSURE_RAMP"
    assert "buy_hold_return" in paid.metrics
    assert "excess_vs_fixed_dca" in paid.metrics


def test_backtest_does_not_assign_pre_open_gap_to_new_position() -> None:
    prices = [
        {"date": ANCHOR.isoformat(), "open": 100.0, "close": 100.0},
        {
            "date": (ANCHOR + timedelta(days=7)).isoformat(),
            "open": 200.0,
            "close": 200.0,
        },
    ]
    decisions = [
        {
            "anchor_date": ANCHOR.isoformat(),
            "forecast_id": 1,
            "next_executable_position": 100,
        }
    ]
    result = backtest_policy(decisions, prices, transaction_cost_bps=0)
    assert result.metrics["realized_return"] == pytest.approx(0.0)
    assert result.points[0]["overnight_return"] == pytest.approx(1.0)
    assert result.points[0]["open_to_close_return"] == pytest.approx(0.0)
    assert result.points[0]["old_position_pp"] == 0
    assert result.points[0]["position_pp"] == 100


def test_backtest_replays_dependent_batches_and_cooldown(monkeypatch) -> None:
    import backend.app.services.v342_turning_policy_service as policy_module

    monkeypatch.setattr(policy_module, "observed_market_confirmation", lambda *args, **kwargs: True)
    prices = []
    for index in range(4):
        close_day = ANCHOR + timedelta(days=7 * index)
        open_day = close_day - timedelta(days=4)
        prices.append(
            {
                "date": close_day.isoformat(),
                "open_date": open_day.isoformat(),
                "trading_dates": [
                    (open_day + timedelta(days=offset)).isoformat()
                    for offset in range(5)
                ],
                "open": 100.0 + index,
                "close": 100.5 + index,
            }
        )
    decisions = [
        {
            "anchor_date": prices[0]["date"],
            "forecast_id": 1,
            "action": "BUY",
            "cooldown_eligible": True,
            "dif_epsilon": 0.01,
            "batches": [
                {
                    "batch_number": 1,
                    "action": "BUY",
                    "change_pp": 10,
                    "target_after_pp": 10,
                    "window_start_date": prices[0]["date"],
                    "window_end_date": prices[3]["date"],
                    "initial_state": "WAITING_CONFIRMATION",
                    "trigger": {},
                },
                {
                    "batch_number": 2,
                    "action": "BUY",
                    "change_pp": 10,
                    "target_after_pp": 20,
                    "window_start_date": prices[0]["date"],
                    "window_end_date": prices[3]["date"],
                    "initial_state": "WAITING_CONFIRMATION",
                    "trigger": {"requires_previous_executed": True},
                },
            ],
        }
    ]
    result = backtest_policy(decisions, prices, transaction_cost_bps=0)
    assert [point["position_pp"] for point in result.points[:2]] == [10, 20]
    assert result.metrics["turnover"] == pytest.approx(0.2)
    events = [event["event"] for point in result.points for event in point["policy_events"]]
    assert events.count("SIMULATED_BATCH_FILLED_AT_NEXT_WEEK_OPEN") == 2
    assert result.metrics["conditional_batch_replay"] is True


def test_backtest_does_not_fill_after_batch_window_end(monkeypatch) -> None:
    import backend.app.services.v342_turning_policy_service as policy_module

    monkeypatch.setattr(policy_module, "observed_market_confirmation", lambda *args, **kwargs: True)
    prices = [
        {"date": ANCHOR.isoformat(), "open": 100.0, "close": 100.0},
        {
            "date": (ANCHOR + timedelta(days=7)).isoformat(),
            "open": 101.0,
            "close": 102.0,
        },
    ]
    decisions = [
        {
            "anchor_date": ANCHOR.isoformat(),
            "forecast_id": 1,
            "action": "BUY",
            "cooldown_eligible": True,
            "dif_epsilon": 0.01,
            "batches": [
                {
                    "batch_number": 1,
                    "action": "BUY",
                    "change_pp": 10,
                    "target_after_pp": 10,
                    "window_start_date": ANCHOR.isoformat(),
                    "window_end_date": ANCHOR.isoformat(),
                    "initial_state": "WAITING_CONFIRMATION",
                    "trigger": {},
                }
            ],
        }
    ]
    result = backtest_policy(decisions, prices, transaction_cost_bps=0)
    assert result.points[0]["position_pp"] == 0
    assert not any(
        event["event"] == "SIMULATED_BATCH_FILLED_AT_NEXT_WEEK_OPEN"
        for event in result.points[0]["policy_events"]
    )
    assert any(
        event.get("to_state") == "EXPIRED"
        for event in result.points[0]["policy_events"]
    )


def test_backtest_uses_actual_week_open_date_for_window(monkeypatch) -> None:
    import backend.app.services.v342_turning_policy_service as policy_module

    monkeypatch.setattr(policy_module, "observed_market_confirmation", lambda *args, **kwargs: True)
    next_close = ANCHOR + timedelta(days=7)
    actual_open = ANCHOR + timedelta(days=3)
    window_start = ANCHOR + timedelta(days=5)
    prices = [
        {"date": ANCHOR.isoformat(), "open": 100.0, "close": 100.0},
        {
            "date": next_close.isoformat(),
            "open_date": actual_open.isoformat(),
            "trading_dates": [
                (actual_open + timedelta(days=offset)).isoformat()
                for offset in range(5)
            ],
            "open": 90.0,
            "close": 100.0,
        },
    ]
    decisions = [
        {
            "anchor_date": ANCHOR.isoformat(),
            "forecast_id": 1,
            "action": "BUY",
            "cooldown_eligible": True,
            "dif_epsilon": 0.01,
            "batches": [
                {
                    "batch_number": 1,
                    "action": "BUY",
                    "change_pp": 10,
                    "target_after_pp": 10,
                    "window_start_date": window_start.isoformat(),
                    "window_end_date": next_close.isoformat(),
                    "initial_state": "WAITING_CONFIRMATION",
                    "trigger": {},
                }
            ],
        }
    ]
    result = backtest_policy(decisions, prices, transaction_cost_bps=0)
    assert result.points[0]["position_pp"] == 0
    assert any(
        event.get("to_state") == "WAITING_CONFIRMATION"
        for event in result.points[0]["policy_events"]
    ) is False
    assert not any(
        event["event"] == "SIMULATED_BATCH_FILLED_AT_NEXT_WEEK_OPEN"
        for event in result.points[0]["policy_events"]
    )


def test_backtest_cooldown_counts_actual_sessions_not_weeks(monkeypatch) -> None:
    import backend.app.services.v342_turning_policy_service as policy_module

    monkeypatch.setattr(policy_module, "observed_market_confirmation", lambda *args, **kwargs: True)
    prices = [
        {
            "date": "2026-08-07",
            "open_date": "2026-08-03",
            "trading_dates": ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
            "open": 100.0,
            "close": 100.0,
        },
        {
            "date": "2026-08-13",
            "open_date": "2026-08-10",
            "trading_dates": ["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"],
            "open": 101.0,
            "close": 101.0,
        },
        {
            "date": "2026-08-21",
            "open_date": "2026-08-17",
            "trading_dates": ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"],
            "open": 102.0,
            "close": 102.0,
        },
        {
            "date": "2026-08-28",
            "open_date": "2026-08-24",
            "trading_dates": ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"],
            "open": 103.0,
            "close": 103.0,
        },
    ]
    decisions = [
        {
            "anchor_date": prices[0]["date"],
            "forecast_id": 1,
            "action": "BUY",
            "cooldown_eligible": True,
            "dif_epsilon": 0.01,
            "batches": [
                {
                    "batch_number": 1,
                    "action": "BUY",
                    "change_pp": 10,
                    "target_after_pp": 10,
                    "window_start_date": "2026-08-10",
                    "window_end_date": "2026-08-28",
                    "initial_state": "WAITING_CONFIRMATION",
                    "trigger": {},
                },
                {
                    "batch_number": 2,
                    "action": "BUY",
                    "change_pp": 10,
                    "target_after_pp": 20,
                    "window_start_date": "2026-08-10",
                    "window_end_date": "2026-08-28",
                    "initial_state": "WAITING_CONFIRMATION",
                    "trigger": {"requires_previous_executed": True},
                },
            ],
        }
    ]
    result = backtest_policy(decisions, prices, transaction_cost_bps=0)
    assert [point["position_pp"] for point in result.points] == [10, 10, 20]


def test_new_forecast_cannot_reset_cross_plan_cooldown(monkeypatch) -> None:
    import backend.app.services.v342_turning_policy_service as policy_module

    monkeypatch.setattr(policy_module, "observed_market_confirmation", lambda *args, **kwargs: True)
    prices = [
        {
            "date": "2026-08-07",
            "open_date": "2026-08-03",
            "trading_dates": ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
            "open": 100.0,
            "close": 100.0,
        },
        {
            "date": "2026-08-13",
            "open_date": "2026-08-10",
            "trading_dates": ["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"],
            "open": 101.0,
            "close": 101.0,
        },
        {
            "date": "2026-08-21",
            "open_date": "2026-08-17",
            "trading_dates": ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"],
            "open": 102.0,
            "close": 102.0,
        },
    ]

    def plan(anchor_date: str, forecast_id: int, target: int) -> dict[str, object]:
        return {
            "anchor_date": anchor_date,
            "forecast_id": forecast_id,
            "action": "BUY",
            "cooldown_eligible": True,
            "dif_epsilon": 0.01,
            "batches": [
                {
                    "batch_number": 1,
                    "action": "BUY",
                    "change_pp": 10,
                    "target_after_pp": target,
                    "window_start_date": "2026-08-10",
                    "window_end_date": "2026-08-21",
                    "initial_state": "WAITING_CONFIRMATION",
                    "trigger": {},
                }
            ],
        }

    result = backtest_policy(
        [plan(prices[0]["date"], 1, 10), plan(prices[1]["date"], 2, 20)],
        prices,
        transaction_cost_bps=0,
    )
    assert [point["position_pp"] for point in result.points] == [10, 10]
    second_week_events = result.points[1]["policy_events"]
    assert not any(
        event["event"] == "SIMULATED_BATCH_FILLED_AT_NEXT_WEEK_OPEN"
        for event in second_week_events
    )


def test_v342_payload_types_do_not_expose_best_date_deviation_fields() -> None:
    forbidden = {
        "actual_turn_date",
        "predicted_action_date",
        "deviation_days",
        "confirmation_offset",
    }
    assessment = _assessment()
    serialized = str(assessment)
    assert not any(name in serialized for name in forbidden)


def test_runtime_policy_is_idempotent_and_does_not_mutate_frozen_v341(tmp_path) -> None:
    from backend.tests.test_v341_runtime import (
        _Calendar,
        _runtime,
        _seed_runtime,
        _snapshot,
    )

    runtime = _runtime(tmp_path)
    snapshots = tuple(_snapshot(index) for index in range(60))
    samples = build_training_samples(snapshots)
    current, manifest, run_id = _seed_runtime(runtime, snapshots, samples, 59)
    runtime._process_week(
        run_id,
        "399006",
        current.anchor_date,
        snapshots,
        samples,
        manifest,
    )
    with runtime.sessions() as session:
        frozen = session.scalar(select(V341Forecast))
        assert frozen is not None
        before = (
            frozen.forecast_hash,
            frozen.payload_json,
            frozen.turning_windows_json,
        )
    service = V342TurningPolicyService(
        runtime.sessions,
        calendar=_Calendar(),
        now_provider=lambda: datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    first = service.analyze("399006")
    second = service.analyze("399006")
    assert second == first
    assert first["semantics"]["window_extrema_are_not_trading_turns"] is True
    assert first["decision"]["next_executable_position"] <= 20
    with runtime.sessions() as session:
        after = session.scalar(select(V341Forecast))
        assert after is not None
        assert (after.forecast_hash, after.payload_json, after.turning_windows_json) == before
        assert session.scalar(select(func.count()).select_from(V342PolicyRun)) == 1
        assert session.scalar(select(func.count()).select_from(V342TurningAssessment)) == 1
        run = session.scalar(select(V342PolicyRun))
        assert run is not None
        batch = session.scalar(select(V342PolicyBatch))
        if batch is None:
            batch = V342PolicyBatch(
                policy_run_id=run.id,
                batch_number=1,
                action="BUY",
                change_pp=5,
                target_after_pp=5,
                window_start_date=run.forecast_anchor_date,
                window_end_date=run.forecast_anchor_date + timedelta(days=30),
                initial_state="ELIGIBLE",
                depends_on_batch_id=None,
                trigger_definition_json={},
                invalidation_definition_json={},
                batch_hash="manual-batch",
                created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
            )
            session.add(batch)
            session.flush()
            session.add(
                V342PolicyBatchEvent(
                    batch_id=batch.id,
                    from_state=None,
                    to_state="ELIGIBLE",
                    evaluated_anchor_date=run.forecast_anchor_date,
                    event_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
                    trigger_snapshot_json={},
                    reason_json={},
                    event_hash="manual-event",
                    created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
                )
            )
        session.commit()
        evaluation_date = run.forecast_anchor_date + timedelta(days=7)
    evaluated = service.evaluate_batches("399006", as_of=evaluation_date)
    assert evaluated["batches"][0]["execution_observed"] is False
    assert evaluated["batches"][0]["state_after"] != "EXECUTED_OBSERVED"
    with runtime.sessions() as session, session.begin():
        service._supersede_previous_runs(
            session,
            market="399006",
            new_run_id="new-same-direction-run",
            evaluation_date=evaluation_date,
        )
    with runtime.sessions() as session:
        latest_batch_event = session.scalar(
            select(V342PolicyBatchEvent)
            .order_by(V342PolicyBatchEvent.id.desc())
            .limit(1)
        )
        assert latest_batch_event is not None
        assert latest_batch_event.to_state == "SUPERSEDED"
    runtime.shutdown()


def test_policy_version_identity_is_call_order_independent(tmp_path) -> None:
    class Calendar:
        def sessions(self, _market, start_date, end_date):
            return [start_date, end_date] if start_date != end_date else [start_date]

    identities = []
    for index, order in enumerate(((False, True), (True, False))):
        database = tmp_path / f"order-{index}" / "investment_lab.db"
        initialize_database(database, tmp_path / f"config-{index}")
        service = V342TurningPolicyService(
            create_session_factory(database),
            calendar=Calendar(),
            now_provider=lambda: datetime(2026, 8, 2, tzinfo=timezone.utc),
        )
        with service.session_factory() as session, session.begin():
            for retrospective in order:
                service._ensure_policy(
                    session, "399006", retrospective=retrospective
                )
        with service.session_factory() as session:
            rows = session.scalars(
                select(V342PolicyVersion)
                .where(V342PolicyVersion.model_market == "399006")
                .order_by(V342PolicyVersion.version)
            ).all()
            identities.append(
                [
                    (
                        row.id,
                        row.version,
                        row.status,
                        row.effective_from_date,
                        row.config_hash,
                    )
                    for row in rows
                ]
            )
        not_due = service.run_live_oos_backtest("399006")
        assert not_due["status"] == "NOT_DUE"
    assert identities[0] == identities[1]
