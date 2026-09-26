"""Fast unit tests for the V3.5.1 A1-A3 model fixes."""

from __future__ import annotations

from datetime import date, timedelta

from backend.app.services.v35_config import default_strategy_config
from backend.app.services.v35_feature_service import V35FeatureSnapshot
from backend.app.services.v35_simulation_service import _run_window
from backend.app.services.v35_runtime_service import _OnlineAccount
from backend.app.services.v35_strategy_service import (
    V35StrategyDecision,
    V35StrategyService,
)


def _snapshot() -> V35FeatureSnapshot:
    return V35FeatureSnapshot(
        market="159941",
        cutoff_date=date(2026, 8, 5),
        source_data_max_date=date(2026, 8, 5),
        features={
            "v35_dif": 1.0,
            "v35_dif_first_change": -1.0,
            "v35_dif_second_change": -1.0,
            "v35_ma20_distance": -0.01,
            "v35_ma20_slope": 0.0,
            "v35_ma60_distance": None,
            "v35_rebound_from_low": None,
            "v35_dif_state_duration": 3.0,
            "v35_golden_cross_duration": 0.0,
            "v35_market_state": 1.0,
            "v35_market_state_duration": 2.0,
            "v35_current_drawdown": 0.05,
            "pe_expanding_percentile": None,
        },
        daily_sequence=(),
        provenance={},
    )


def test_strategy_emits_sell_when_current_position_above_target() -> None:
    decision = V35StrategyService().decide(
        "159941",
        _snapshot(),
        expected_path=[-0.02, -0.03, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10],
        p10_path=[-0.05, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18, -0.20],
        probabilities_4=[0.20, 0.30, 0.50],
        probabilities_8=[0.15, 0.20, 0.65],
        current_position_pp=50,
        reliability_score=70.0,
        health_status="MODEL_NORMAL",
        ood_score=0.1,
        config=default_strategy_config("159941"),
    )
    assert decision.confirmation_status == "TOP_CONFIRMED"
    assert decision.final_target_position_pp < 50
    assert any(batch["action"] == "SELL" for batch in decision.batches)


def _decision(
    *,
    batches: tuple[dict, ...],
    cooldown: int = 5,
    final_target: int = 20,
    confirmation_status: str = "TREND_CONFIRMED",
    reasons: tuple[str, ...] = (),
) -> V35StrategyDecision:
    return V35StrategyDecision(
        market="TEST999",
        forecast_anchor_date="2026-01-02",
        dif_trend_state="POSITIVE_DIF_RISING",
        confirmation_status=confirmation_status,
        market_state="UPTREND",
        strategy_score=0.5,
        base_target_position_pp=20,
        state_position_cap_pp=80,
        final_target_position_pp=final_target,
        batches=batches,
        components={},
        reasons=reasons,
    )


def _bars(anchors: list[date]) -> dict[date, dict[str, float]]:
    return {
        anchor: {"open": 100.0, "close": 101.0, "volume": 1000.0}
        for anchor in anchors
    }


def _traded_sequences(ledger) -> list[int]:
    return [
        row.sequence
        for row in ledger
        if row.trade_action in ("BUY", "SELL")
    ]


def test_window_cooldown_controls_batch_spacing() -> None:
    anchors = [date(2026, 1, 2) + timedelta(weeks=index) for index in range(4)]
    buy_batch = {
        "batch_number": 1,
        "action": "BUY",
        "position_pp": 5,
        "batch_change_pp": 5,
        "target_position_pp": 20,
        "cooldown_trading_days": 10,
        "condition": "test",
    }
    second_batch = {
        "batch_number": 2,
        "action": "BUY",
        "position_pp": 10,
        "batch_change_pp": 5,
        "target_position_pp": 20,
        "cooldown_trading_days": 10,
        "condition": "test",
    }
    decisions = {
        anchors[0]: _decision(
            batches=(buy_batch, second_batch), cooldown=10
        )
    }
    result = _run_window(
        market="TEST999",
        package_id="pkg",
        scope="STANDARD_8W",
        anchors=anchors,
        decisions=decisions,
        bars=_bars(anchors),
        config=default_strategy_config("159941"),
    )
    assert _traded_sequences(result.ledger) == [1, 3]


def test_window_default_cooldown_executes_consecutive_weeks() -> None:
    anchors = [date(2026, 1, 2) + timedelta(weeks=index) for index in range(3)]
    batches = tuple(
        {
            "batch_number": index + 1,
            "action": "BUY",
            "position_pp": 5 * (index + 1),
            "batch_change_pp": 5,
            "target_position_pp": 20,
            "cooldown_trading_days": 5,
            "condition": "test",
        }
        for index in range(2)
    )
    result = _run_window(
        market="TEST999",
        package_id="pkg",
        scope="STANDARD_8W",
        anchors=anchors,
        decisions={anchors[0]: _decision(batches=batches, cooldown=5)},
        bars=_bars(anchors),
        config=default_strategy_config("159941"),
    )
    assert _traded_sequences(result.ledger) == [1, 2]


def test_pending_batches_survive_empty_decision() -> None:
    anchors = [date(2026, 1, 2) + timedelta(weeks=index) for index in range(3)]
    batches = tuple(
        {
            "batch_number": index + 1,
            "action": "BUY",
            "position_pp": 5 * (index + 1),
            "batch_change_pp": 5,
            "target_position_pp": 20,
            "cooldown_trading_days": 5,
            "condition": "test",
        }
        for index in range(2)
    )
    result = _run_window(
        market="TEST999",
        package_id="pkg",
        scope="STANDARD_8W",
        anchors=anchors,
        decisions={anchors[0]: _decision(batches=batches)},
        bars=_bars(anchors),
        config=default_strategy_config("159941"),
    )
    assert _traded_sequences(result.ledger) == [1, 2]
    assert result.current_position_pp == 10


def test_pending_batches_cleared_on_top_confirmed() -> None:
    anchors = [date(2026, 1, 2) + timedelta(weeks=index) for index in range(3)]
    batches = tuple(
        {
            "batch_number": index + 1,
            "action": "BUY",
            "position_pp": 5 * (index + 1),
            "batch_change_pp": 5,
            "target_position_pp": 20,
            "cooldown_trading_days": 5,
            "condition": "test",
        }
        for index in range(2)
    )
    decisions = {
        anchors[0]: _decision(batches=batches),
        anchors[1]: _decision(
            batches=(),
            confirmation_status="TOP_CONFIRMED",
            final_target=0,
        ),
    }
    result = _run_window(
        market="TEST999",
        package_id="pkg",
        scope="STANDARD_8W",
        anchors=anchors,
        decisions=decisions,
        bars=_bars(anchors),
        config=default_strategy_config("159941"),
    )
    assert _traded_sequences(result.ledger) == [1]
    assert result.current_position_pp == 5


def test_pending_batches_cleared_when_target_lowered_below_position() -> None:
    anchors = [date(2026, 1, 2) + timedelta(weeks=index) for index in range(3)]
    batches = tuple(
        {
            "batch_number": index + 1,
            "action": "BUY",
            "position_pp": 5 * (index + 1),
            "batch_change_pp": 5,
            "target_position_pp": 20,
            "cooldown_trading_days": 5,
            "condition": "test",
        }
        for index in range(2)
    )
    decisions = {
        anchors[0]: _decision(batches=batches),
        anchors[1]: _decision(
            batches=(),
            confirmation_status="UNCONFIRMED",
            final_target=0,
        ),
    }
    result = _run_window(
        market="TEST999",
        package_id="pkg",
        scope="STANDARD_8W",
        anchors=anchors,
        decisions=decisions,
        bars=_bars(anchors),
        config=default_strategy_config("159941"),
    )
    assert _traded_sequences(result.ledger) == [1]
    assert result.current_position_pp == 5


def test_online_snapshot_restores_pending_batches() -> None:
    batch = {
        "batch_number": 2,
        "action": "BUY",
        "position_pp": 10,
        "batch_change_pp": 5,
        "target_position_pp": 20,
        "cooldown_trading_days": 5,
        "condition": "test",
    }
    online = _OnlineAccount(
        accounting_mode="SHARES",
        initial={
            "equity": 100000.0,
            "cash": 90000.0,
            "position_pp": 5.0,
            "held_shares": 1000.0,
            "average_cost": 1.0,
            "sellable_shares": 1000.0,
            "pending_batches": [batch],
        },
    )
    assert len(online.pending_batches) == 1
    assert online.held_shares == 1000.0
    assert online.sellable_shares == 1000.0
