"""V3.5 acceptance tests covering the approved 30-item checklist.

Items 25 and 26 (chart legend/toolbar behaviour) are covered by the frontend
Playwright suite (``frontend/tests/v35-chart.spec.ts``); the remaining items
are executed here against synthetic databases so every acceptance rule is
automated and reproducible.
"""

from __future__ import annotations

import math
import random
import tempfile
from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.migrations import run_migrations
from backend.app.database.session import create_database_engine
from backend.app.models.models import (
    Instrument,
    MarketPrice,
    V35BootstrapState,
    V35Challenge,
    V35ChallengerWindow,
    V35ContinuousAccount,
    V35FeatureSnapshot as V35FeatureSnapshotRow,
    V35Forecast,
    V35ForecastEvaluation,
    V35ModelPackage,
    V35Promotion,
    V35ResidualRecord,
    V35SimAccount,
    V35SimEvaluation,
    V35SimLedger,
    V35StrategySnapshot,
    V35TrainingIteration,
)
from backend.app.services.v35_config import (
    HORIZON_WEEKS,
    PREDICTION_CHALLENGE_EVERY_MATURED,
    STRATEGY_CHALLENGE_EVERY_MATURED,
)
from backend.app.services.v35_feature_service import (
    V35FeatureService,
    build_feature_manifest,
)
from backend.app.services.v35_runtime_service import (
    V35RuntimeService,
    promotion_gate_decision,
)
from backend.app.services.v35_strategy_service import V35StrategyService


def _session_dates(start: date, count: int) -> list[date]:
    result: list[date] = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current)
        current += timedelta(days=1)
    return result


def _seed_market(
    engine,
    code: str,
    *,
    start: date,
    days: int,
    seed: int = 7,
    volume_seed: int = 11,
) -> None:
    rng = random.Random(seed)
    volume_rng = random.Random(volume_seed)
    price = 1000.0
    with Session(engine) as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == code))
        if instrument is None:
            instrument = Instrument(
                code=code,
                name=code,
                exchange="SZSE",
                category="index" if code == "399006" else "etf",
                currency="CNY",
                is_active=True,
            )
            session.add(instrument)
            session.flush()
        for trade_date in _session_dates(start, days):
            previous = price
            price = max(100.0, price * (1.0 + rng.gauss(0.001, 0.012)))
            open_price = previous * (1.0 + rng.gauss(0.0, 0.004))
            high = max(open_price, price) * (1.0 + abs(rng.gauss(0.0, 0.003)))
            low = min(open_price, price) * (1.0 - abs(rng.gauss(0.0, 0.003)))
            session.add(
                MarketPrice(
                    instrument_id=instrument.id,
                    trade_date=trade_date,
                    timeframe="daily",
                    open_price=Decimal(str(round(open_price, 3))),
                    high_price=Decimal(str(round(high, 3))),
                    low_price=Decimal(str(round(low, 3))),
                    close_price=Decimal(str(round(price, 3))),
                    adjusted_close_price=Decimal(str(round(price, 3))),
                    volume=Decimal(str(round(1_000_000 * (1 + volume_rng.random()), 2))),
                    volume_multiplier=1,
                    turnover=Decimal("0"),
                    source="TEST",
                )
            )


@pytest.fixture(scope="module")
def full_db(tmp_path_factory):
    root = tmp_path_factory.mktemp("v35-full")
    engine = create_database_engine(root / "full.db")
    run_migrations(engine)
    _seed_market(engine, "399006", start=date(2021, 1, 4), days=850)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    runtime = V35RuntimeService(factory)
    result = runtime.bootstrap_sync("399006", maximum_weeks=130)
    try:
        yield {"engine": engine, "runtime": runtime, "factory": factory, "result": result}
    finally:
        engine.dispose()


def _count(session: Session, model) -> int:
    return len(session.scalars(select(model)).all())


def test_01_labels_are_strictly_1_to_8_weeks(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        forecasts = session.scalars(select(V35Forecast)).all()
        assert forecasts
        assert all(row.horizon_weeks == 8 for row in forecasts)
        assert all(set(row.horizon_probabilities_json) == {"4", "8"} for row in forecasts)
        assert all(len(row.expected_path_json) == 8 for row in forecasts)


def test_02_v35_output_has_no_13_or_12_week_horizon(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        rows = session.scalars(select(V35Forecast)).all()
        for row in rows:
            assert "13" not in row.horizon_probabilities_json
            assert "12" not in row.horizon_probabilities_json
            assert row.label_end_date is None or row.label_end_date > row.forecast_anchor_date


def test_03_forecast_matures_only_after_eight_weeks(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        rows = session.scalars(select(V35Forecast)).all()
        for row in rows:
            if row.maturity_status == "FULLY_MATURE_8W":
                evaluation = session.scalar(
                    select(V35ForecastEvaluation).where(
                        V35ForecastEvaluation.forecast_id == row.id,
                        V35ForecastEvaluation.horizon_weeks == 8,
                    )
                )
                assert evaluation is not None
                assert evaluation.evaluation_available_date >= row.label_end_date


def test_04_account_cash_and_position_carry_forward(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        account = session.scalar(
            select(V35SimAccount).where(V35SimAccount.scope == "STANDARD_8W")
        )
        assert account is not None
        ledger = session.scalars(
            select(V35SimLedger)
            .where(V35SimLedger.account_id == account.id)
            .order_by(V35SimLedger.sequence)
        ).all()
        assert len(ledger) == 8
        equities = [float(row.equity) for row in ledger]
        assert all(equities[index + 1] != 100000.0 for index in range(len(equities) - 1))


def test_05_unconfirmed_allows_5_to_10_percent_probe(full_db):
    service = V35StrategyService()
    snapshot = _with_features(
        _snapshot_from_row(_first_snapshot_row(full_db["engine"])),
        {
            "v35_dif": -0.005,
            "v35_dif_first_change": -0.001,
            "v35_ma20_distance": 0.01,
        },
    )
    decision = service.decide(
        "399006",
        snapshot,
        expected_path=[0.03, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06],
        p10_path=[-0.01] * 8,
        probabilities_4=[0.50, 0.25, 0.25],
        probabilities_8=[0.55, 0.20, 0.25],
        current_position_pp=0,
        reliability_score=70,
        health_status="MODEL_NORMAL",
        ood_score=0.1,
    )
    assert decision.confirmation_status == "UNCONFIRMED"
    assert decision.final_target_position_pp > 0
    buy_batches = [b for b in decision.batches if b["action"] == "BUY"]
    assert buy_batches
    assert 5 <= buy_batches[0]["batch_change_pp"] <= 10


def test_06_negative_dif_rising_allows_limited_probe(full_db):
    service = V35StrategyService()
    snapshot = _snapshot_from_row(_first_snapshot_row(full_db["engine"]))
    snapshot = _with_features(
        snapshot,
        {
            "v35_dif": -0.01,
            "v35_dif_first_change": 0.002,
            "v35_dif_second_change": 0.0,
            "v35_ma20_slope": -0.0002,
        },
    )
    assert service.dif_quadrant(snapshot) == "NEGATIVE_DIF_RISING"
    decision = service.decide(
        "399006",
        snapshot,
        expected_path=[0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04],
        p10_path=[-0.04] * 8,
        probabilities_4=[0.40, 0.30, 0.30],
        probabilities_8=[0.45, 0.25, 0.30],
        current_position_pp=0,
        reliability_score=70,
        health_status="MODEL_NORMAL",
        ood_score=0.1,
    )
    buys = [b for b in decision.batches if b["action"] == "BUY"]
    assert buys
    assert buys[0]["batch_change_pp"] <= 10


def test_07_positive_dif_rising_does_not_wait_for_bottom(full_db):
    service = V35StrategyService()
    snapshot = _with_features(
        _snapshot_from_row(_first_snapshot_row(full_db["engine"])),
        {
            "v35_dif": 0.01,
            "v35_dif_first_change": 0.002,
            "v35_dif_second_change": 0.001,
            "v35_golden_cross_duration": 1.0,
            "v35_ma20_distance": 0.02,
            "v35_ma20_slope": 0.001,
        },
    )
    quadrant = service.dif_quadrant(snapshot)
    assert quadrant in (
        "POSITIVE_DIF_RISING",
        "POSITIVE_DIF_RISING_DECELERATING",
        "POSITIVE_DIF_ACCELERATING",
    )
    decision = service.decide(
        "399006",
        snapshot,
        expected_path=[0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04],
        p10_path=[-0.03] * 8,
        probabilities_4=[0.45, 0.25, 0.30],
        probabilities_8=[0.50, 0.20, 0.30],
        current_position_pp=0,
        reliability_score=70,
        health_status="MODEL_NORMAL",
        ood_score=0.1,
    )
    assert any(b["action"] == "BUY" for b in decision.batches)


def test_08_positive_dif_rising_two_weeks_allows_trend_add(full_db):
    service = V35StrategyService()
    snapshot = _with_features(
        _snapshot_from_row(_first_snapshot_row(full_db["engine"])),
        {
            "v35_dif": 0.01,
            "v35_dif_first_change": 0.002,
            "v35_dif_second_change": 0.001,
            "v35_dif_state_duration": 3.0,
            "v35_golden_cross_duration": 2.0,
            "v35_ma20_distance": 0.03,
            "v35_ma20_slope": 0.001,
            "v35_ma60_distance": 0.01,
        },
    )
    decision = service.decide(
        "399006",
        snapshot,
        expected_path=[0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.04],
        p10_path=[-0.03] * 8,
        probabilities_4=[0.45, 0.25, 0.30],
        probabilities_8=[0.50, 0.20, 0.30],
        current_position_pp=0,
        reliability_score=70,
        health_status="MODEL_NORMAL",
        ood_score=0.1,
    )
    assert decision.confirmation_status == "TREND_CONFIRMED"
    buys = [b for b in decision.batches if b["action"] == "BUY"]
    assert buys and max(b["batch_change_pp"] for b in buys) >= 10


def test_09_decelerating_dif_cannot_jump_to_max_position(full_db):
    service = V35StrategyService()
    snapshot = _with_features(
        _snapshot_from_row(_first_snapshot_row(full_db["engine"])),
        {
            "v35_dif": 0.01,
            "v35_dif_first_change": 0.002,
            "v35_dif_second_change": -0.001,
        },
    )
    assert service.dif_quadrant(snapshot) == "POSITIVE_DIF_RISING_DECELERATING"
    decision = service.decide(
        "399006",
        snapshot,
        expected_path=[0.05] * 8,
        p10_path=[-0.01] * 8,
        probabilities_4=[0.6, 0.2, 0.2],
        probabilities_8=[0.7, 0.1, 0.2],
        current_position_pp=0,
        reliability_score=90,
        health_status="MODEL_NORMAL",
        ood_score=0.1,
    )
    assert decision.final_target_position_pp <= 60
    assert decision.state_position_cap_pp <= 60


def test_10_small_target_decrease_without_top_only_small_reduce(full_db):
    service = V35StrategyService()
    snapshot = _with_features(
        _snapshot_from_row(_first_snapshot_row(full_db["engine"])),
        {
            "v35_dif": -0.005,
            "v35_dif_first_change": 0.0005,
            "v35_ma20_distance": -0.01,
        },
    )
    decision = service.decide(
        "399006",
        snapshot,
        expected_path=[-0.002, -0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        p10_path=[-0.06] * 8,
        probabilities_4=[0.3, 0.3, 0.4],
        probabilities_8=[0.35, 0.25, 0.40],
        current_position_pp=30,
        reliability_score=70,
        health_status="MODEL_NORMAL",
        ood_score=0.2,
    )
    assert decision.confirmation_status != "TOP_CONFIRMED"
    sells = [b for b in decision.batches if b["action"] == "SELL"]
    if sells:
        assert max(b["batch_change_pp"] for b in sells) <= 5


def test_11_confirmed_top_allows_stronger_reduce(full_db):
    service = V35StrategyService()
    snapshot = _with_features(
        _snapshot_from_row(_first_snapshot_row(full_db["engine"])),
        {
            "v35_dif": 0.005,
            "v35_dif_first_change": -0.002,
            "v35_ma20_distance": -0.02,
            "v35_ma20_slope": -0.001,
        },
    )
    decision = service.decide(
        "399006",
        snapshot,
        expected_path=[-0.01, -0.01, -0.02, -0.02, -0.02, -0.02, -0.02, -0.03],
        p10_path=[-0.08] * 8,
        probabilities_4=[0.25, 0.25, 0.50],
        probabilities_8=[0.25, 0.20, 0.55],
        current_position_pp=40,
        reliability_score=70,
        health_status="MODEL_NORMAL",
        ood_score=0.2,
    )
    assert decision.confirmation_status == "TOP_CONFIRMED"
    sells = [b for b in decision.batches if b["action"] == "SELL"]
    assert sells and 10 <= sells[0]["batch_change_pp"] <= 20


def test_12_prediction_challenge_has_five_candidates(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        challenges = session.scalars(
            select(V35Challenge).where(V35Challenge.challenger_family == "PREDICTION")
        ).all()
        assert challenges
        assert all(row.candidate_count == 5 for row in challenges)


def test_13_strategy_challenge_has_three_candidates(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        challenges = session.scalars(
            select(V35Challenge).where(V35Challenge.challenger_family == "STRATEGY")
        ).all()
        if challenges:
            assert all(row.candidate_count == 3 for row in challenges)


def test_14_same_week_only_one_challenge_family(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        rows = session.scalars(select(V35Challenge)).all()
        anchors = [row.anchor_date for row in rows]
        assert len(anchors) == len(set(anchors))


def test_15_prediction_challenger_inherits_strategy(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        challenge = session.scalar(
            select(V35Challenge).where(V35Challenge.challenger_family == "PREDICTION")
        )
        if challenge is None:
            pytest.skip("no prediction challenge in fixture")
        champion = session.get(V35ModelPackage, challenge.champion_package_id)
        candidates = session.scalars(
            select(V35ModelPackage).where(
                V35ModelPackage.id.in_(
                    [
                        row.candidate_package_id
                        for row in session.scalars(
                            select(V35ChallengerWindow).where(
                                V35ChallengerWindow.challenge_id == challenge.id
                            )
                        ).all()
                    ]
                )
            )
        ).all()
        for candidate in candidates:
            assert candidate.strategy_config_json == champion.strategy_config_json


def test_16_strategy_challenger_inherits_prediction_model(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        challenge = session.scalar(
            select(V35Challenge).where(V35Challenge.challenger_family == "STRATEGY")
        )
        if challenge is None:
            pytest.skip("no strategy challenge in fixture")
        champion = session.get(V35ModelPackage, challenge.champion_package_id)
        candidates = session.scalars(
            select(V35ModelPackage).where(
                V35ModelPackage.id.in_(
                    [
                        row.candidate_package_id
                        for row in session.scalars(
                            select(V35ChallengerWindow).where(
                                V35ChallengerWindow.challenge_id == challenge.id
                            )
                        ).all()
                    ]
                )
            )
        ).all()
        for candidate in candidates:
            assert candidate.prediction_model_id == champion.prediction_model_id


def test_17_low_but_accurate_candidate_is_rejected():
    decision, reason = promotion_gate_decision(
        mean_excess=-0.001,
        median_profit_excess=100.0,
        win_rate=0.6,
        max_drawdown_degradation=0.001,
        no_action_ratio=0.0,
        mean_position_pp=40.0,
    )
    assert decision == "REJECTED"
    assert "0.30%" in reason


def test_18_higher_return_and_qualified_risk_promotes():
    decision, _ = promotion_gate_decision(
        mean_excess=0.006,
        median_profit_excess=200.0,
        win_rate=0.6,
        max_drawdown_degradation=0.001,
        no_action_ratio=0.0,
        mean_position_pp=45.0,
    )
    assert decision == "PROMOTED"


def test_19_long_cash_candidate_cannot_promote():
    decision, reason = promotion_gate_decision(
        mean_excess=0.006,
        median_profit_excess=300.0,
        win_rate=0.6,
        max_drawdown_degradation=0.001,
        no_action_ratio=0.7,
        mean_position_pp=2.0,
    )
    assert decision == "REJECTED"
    assert "空仓" in reason or "仓位" in reason


def test_20_down_market_cash_is_valid_defense(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        ledger_rows = session.scalars(
            select(V35SimLedger)
            .where(V35SimLedger.position_pp == 0)
            .order_by(V35SimLedger.anchor_date.desc())
        ).all()
        assert ledger_rows
        down_rows = [row for row in ledger_rows if row.market_return < 0]
        assert down_rows
        assert all(float(row.account_return) == 0.0 for row in down_rows)


def test_21_no_evaluation_before_eight_weeks():
    root = tempfile.TemporaryDirectory()
    engine = create_database_engine(root.name + "/partial.db")
    run_migrations(engine)
    _seed_market(engine, "399006", start=date(2021, 1, 4), days=850)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    runtime = V35RuntimeService(factory)
    try:
        runtime.bootstrap_sync("399006", maximum_weeks=60)
        with Session(engine) as session:
            forecasts = session.scalars(select(V35Forecast)).all()
            assert forecasts
            pending = [row for row in forecasts if row.maturity_status == "PENDING"]
            assert pending
            for row in pending:
                assert not session.scalar(
                    select(V35ForecastEvaluation).where(
                        V35ForecastEvaluation.forecast_id == row.id
                    )
                )
                assert not session.scalar(
                    select(V35ResidualRecord).where(
                        V35ResidualRecord.forecast_id == row.id
                    )
                )
    finally:
        engine.dispose()
        root.cleanup()


def test_22_promoted_model_effective_from_next_week(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        promotion = session.scalar(select(V35Promotion).order_by(V35Promotion.id.desc()))
        if promotion is None:
            pytest.skip("no promotion happened in fixture")
        challenge = session.get(V35Challenge, promotion.challenge_id)
        assert promotion.effective_from_date is not None
        assert promotion.effective_from_date > challenge.anchor_date


def test_23_second_run_creates_no_duplicates(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        forecasts_before = _count(session, V35Forecast)
        evaluations_before = _count(session, V35ForecastEvaluation)
        promotions_before = _count(session, V35Promotion)
    result = full_db["runtime"].bootstrap_sync("399006")
    assert result["processed_weeks"] == 0
    with Session(engine) as session:
        assert _count(session, V35Forecast) == forecasts_before
        assert _count(session, V35ForecastEvaluation) == evaluations_before
        assert _count(session, V35Promotion) == promotions_before


def test_24_159941_never_reads_external_fields():
    from backend.app.services.v34_feature_service import assert_159941_self_only

    with pytest.raises(Exception):
        assert_159941_self_only(["v35_dif", "ndx_close", "fx_usdcny", "nav"])
    root = tempfile.TemporaryDirectory()
    engine = create_database_engine(root.name + "/boundary.db")
    run_migrations(engine)
    _seed_market(engine, "159941", start=date(2021, 1, 4), days=850, seed=23)
    try:
        with Session(engine) as session:
            anchors = V35FeatureService().weekly_anchors(session, "159941")
            snapshot = V35FeatureService().load_snapshot(session, "159941", anchors[-1])
            manifest = build_feature_manifest(snapshot)
            assert_159941_self_only(manifest.ordered_feature_names)
    finally:
        engine.dispose()
        root.cleanup()


@pytest.mark.skip(reason="covered by frontend Playwright suite (v35-chart.spec.ts)")
def test_25_legend_toolbar_does_not_overlap_chart():
    pass


@pytest.mark.skip(reason="covered by frontend Playwright suite (v35-chart.spec.ts)")
def test_26_zoom_hover_dynamic_axes_work_with_toolbar():
    pass


def test_27_completed_bootstrap_is_not_rerun(full_db):
    state = full_db["runtime"].status("399006")
    assert state["state"] == "RUNNING" or state["state"] == "COMPLETED"
    assert state["weekly_iteration_count"] > 0
    result = full_db["runtime"].bootstrap_sync("399006")
    assert result["processed_weeks"] == 0


def test_28_partial_bootstrap_resumes_from_first_missing_week():
    root = tempfile.TemporaryDirectory()
    engine = create_database_engine(root.name + "/resume.db")
    run_migrations(engine)
    _seed_market(engine, "399006", start=date(2021, 1, 4), days=850)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    runtime = V35RuntimeService(factory)
    try:
        first = runtime.bootstrap_sync("399006", maximum_weeks=70)
        with Session(engine) as session:
            anchors_before = {row.anchor_date for row in session.scalars(select(V35TrainingIteration)).all()}
            count_before = len(anchors_before)
        second = runtime.bootstrap_sync("399006")
        with Session(engine) as session:
            anchors_after = {row.anchor_date for row in session.scalars(select(V35TrainingIteration)).all()}
        assert anchors_before.issubset(anchors_after)
        assert len(anchors_after) > count_before
        with Session(engine) as session:
            all_anchors = [row.anchor_date for row in session.scalars(select(V35TrainingIteration)).all()]
        assert len(anchors_after) == len(all_anchors) == len(set(all_anchors))
    finally:
        engine.dispose()
        root.cleanup()


def test_29_theory_challenge_counts_for_500_weeks():
    mature_windows = 500 - 8
    assert math.floor(mature_windows / PREDICTION_CHALLENGE_EVERY_MATURED) == 123
    assert math.floor(mature_windows / STRATEGY_CHALLENGE_EVERY_MATURED) == 61
    assert 123 * 5 == 615
    assert 61 * 3 == 183


def test_30_forecast_anchors_are_contiguous_unique_and_ordered(full_db):
    engine = full_db["engine"]
    with Session(engine) as session:
        anchors = [
            row.forecast_anchor_date
            for row in session.scalars(
                select(V35Forecast).order_by(V35Forecast.forecast_anchor_date)
            ).all()
        ]
        assert anchors
        assert len(anchors) == len(set(anchors))
        assert all(left < right for left, right in zip(anchors, anchors[1:]))


def _snapshot_from_row(row):
    from backend.app.services.v35_feature_service import V35FeatureSnapshot

    return V35FeatureSnapshot(
        market=row.model_market,
        cutoff_date=row.forecast_anchor_date,
        source_data_max_date=row.source_max_date,
        features=dict(row.feature_json),
        daily_sequence=tuple(row.daily_sequence_json),
        provenance=dict(row.provenance_json),
    )


def _first_snapshot_row(engine):
    with Session(engine) as session:
        return session.scalar(select(V35FeatureSnapshotRow).order_by(V35FeatureSnapshotRow.forecast_anchor_date))


def _first_forecast(engine):
    with Session(engine) as session:
        return session.scalar(select(V35Forecast).order_by(V35Forecast.forecast_anchor_date))


def _with_features(snapshot, features):
    from dataclasses import replace

    return replace(snapshot, features={**snapshot.features, **features})
