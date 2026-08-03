from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from decimal import Decimal

import pytest

from backend.app import weekly_analysis
from backend.app.weekly_analysis.advice import (
    ADVICE_VERSION,
    Advice,
    AdviceBatch,
    build_advice,
    fund_etf_ratio,
)
from backend.app.weekly_analysis.domain import QualityReport
from backend.app.weekly_analysis.features import (
    FEATURE_SET_VERSION,
    FeatureSnapshot,
    FrozenDict,
)
from backend.app.weekly_analysis.optimizer import (
    CrossMarketError,
    IterationLabel,
    TradingCalendar,
    WindowMetrics,
    WorkState,
    run_optimizer_step,
    seed_model,
)


CUTOFF = date(2026, 7, 24)


def _weekdays(start: date, count: int) -> tuple[date, ...]:
    values: list[date] = []
    current = start
    while len(values) < count:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    return tuple(values)


def _snapshot(
    *,
    instrument_code: str = "399006",
    valuation_percentile: int = 50,
    golden_point: bool = False,
    golden_strength: int = 0,
    black_point: bool = False,
    black_strength: int = 0,
    publishable: bool = True,
) -> FeatureSnapshot:
    features = FrozenDict(
        {
            "valuation_percentile": Decimal(valuation_percentile),
            "golden_point": golden_point,
            "golden_strength": golden_strength,
            "golden_point_reasons": (
                ("valuation_percentile_low", "dif_crossed_above_dea")
                if golden_point
                else ()
            ),
            "black_point": black_point,
            "black_strength": black_strength,
            "black_point_reasons": (
                ("valuation_percentile_high", "dif_crossed_below_dea")
                if black_point
                else ()
            ),
        }
    )
    availability = FrozenDict(
        {name: "available" for name in features}
    )
    return FeatureSnapshot(
        instrument_code=instrument_code,
        cutoff_date=CUTOFF,
        source_data_max_date=CUTOFF,
        feature_set_version=FEATURE_SET_VERSION,
        features=features,
        availability=availability,
        quality_report=QualityReport(
            is_publishable=publishable,
            issues=(),
            volume_availability="available",
        ),
        audit_fields=FrozenDict({"warmup_weeks": 60}),
        source_hash="a" * 64,
    )


def _calendar(
    instrument_code: str = "399006",
    *,
    include_past: bool = True,
) -> TradingCalendar:
    future = _weekdays(CUTOFF + timedelta(days=1), 65)
    values = (
        _weekdays(CUTOFF - timedelta(days=7), 5) + future
        if include_past
        else future
    )
    return TradingCalendar(
        instrument_code=instrument_code,
        expected_trade_dates=values,
    )


def _advice(
    *,
    current_position: int | None,
    target_position: int,
    direction: str,
    probability: int,
    confidence: int,
    snapshot: FeatureSnapshot | None = None,
    calendar: TradingCalendar | None = None,
    confirmation_conditions_met: bool = False,
    market_state: str | None = None,
    model: WorkState | None = None,
) -> Advice:
    return build_advice(
        model=model or seed_model("399006"),
        latest_features=snapshot or _snapshot(),
        future_trading_calendar=calendar or _calendar(),
        current_position=current_position,
        target_position=target_position,
        direction=direction,
        probability=probability,
        confidence=confidence,
        confirmation_conditions_met=confirmation_conditions_met,
        market_state=market_state,
    )


def _audited_model() -> WorkState:
    return run_optimizer_step(
        seed_model("399006"),
        feedback=(),
        week_key="2026-W30",
        cutoff_date=CUTOFF,
        source_data_max_date=CUTOFF,
    ).work_state


def _accepted_audited_model() -> WorkState:
    label = IterationLabel(
        iteration_id="I0001",
        instrument_code="399006",
        week_key="2026-W17",
        cutoff_date=date(2026, 4, 24),
        predicted_direction="up",
        operation_side="buy",
        probability=Decimal("0.8"),
        predicted_action_date=date(2026, 4, 27),
        target_position=Decimal("0.75"),
        trade_count=1,
        status="mature_13w",
        observed_through=CUTOFF,
        visible_row_count=65,
        partial_weight=Decimal("1"),
        actual_direction="up",
        actual_turn_date=date(2026, 4, 27),
        turning_deviation_sessions=0,
        loss_components=FrozenDict(
            {
                "turning_deviation": Decimal("0"),
                "direction_calibration": Decimal("0"),
                "adverse_excursion": Decimal("0"),
                "overtrading": Decimal("0"),
            }
        ),
        total_loss=Decimal("0"),
        direction_hit=True,
        calibration_formula="brier=(p-y)^2",
    )
    current = WindowMetrics(
        instrument_code="399006",
        window="expanding",
        sample_count=1,
        total_loss=Decimal("0.50"),
        direction_hit_rate=Decimal("0.60"),
        calibration_loss=Decimal("0.20"),
        overtrade_penalty=Decimal("0.20"),
    )
    candidate = replace(current, total_loss=Decimal("0.49"))
    return run_optimizer_step(
        seed_model("399006"),
        feedback=(label,),
        week_key="2026-W30",
        cutoff_date=CUTOFF,
        source_data_max_date=CUTOFF,
        current_metrics={"expanding": current},
        candidate_metrics={"expanding": candidate},
    ).work_state


@pytest.mark.parametrize(
    ("direction", "probability", "confidence", "expected"),
    [
        ("up", 78, 80, "7:3"),
        ("up", 61, 65, "6:4"),
        ("neutral", 50, 62, "5:5"),
        ("down", 62, 66, "4:6"),
        ("down", 81, 84, "3:7"),
    ],
)
def test_ratio_is_one_of_five_integer_tiers(
    direction: str,
    probability: int,
    confidence: int,
    expected: str,
) -> None:
    assert fund_etf_ratio(direction, probability, confidence) == expected


@pytest.mark.parametrize(
    ("direction", "probability", "confidence"),
    [("up", 90, 54), ("up", 54, 90), ("down", 90, 54), ("down", 54, 90)],
)
def test_ratio_uses_balanced_tier_for_low_or_conflicting_signal(
    direction: str,
    probability: int,
    confidence: int,
) -> None:
    assert fund_etf_ratio(direction, probability, confidence) == "5:5"


def test_overvalued_strong_trend_reduces_only_requested_twenty_points() -> None:
    advice = _advice(
        current_position=90,
        target_position=70,
        direction="up",
        probability=74,
        confidence=76,
        snapshot=_snapshot(valuation_percentile=88),
    )

    assert advice.market_state == "overvalued_strong_trend"
    assert advice.operation_side == "sell"
    assert advice.total_adjustment == 20
    assert 1 < len(advice.batches) <= 4
    assert sum(batch.percent for batch in advice.batches) == 20
    assert max(batch.percent for batch in advice.batches) < 90


def test_overvalued_multi_peak_uses_four_batches_totalling_eighty() -> None:
    advice = _advice(
        current_position=95,
        target_position=15,
        direction="down",
        probability=82,
        confidence=86,
        snapshot=_snapshot(
            valuation_percentile=95,
            black_point=True,
            black_strength=5,
        ),
    )

    assert advice.market_state == "overvalued_multi_peak"
    assert len(advice.batches) == 4
    assert sum(batch.percent for batch in advice.batches) == 80
    assert [batch.percent for batch in advice.batches] == sorted(
        (batch.percent for batch in advice.batches),
        reverse=True,
    )


def test_undervalued_weak_trend_only_adds_ten_point_probe() -> None:
    advice = _advice(
        current_position=10,
        target_position=20,
        direction="up",
        probability=59,
        confidence=61,
        snapshot=_snapshot(valuation_percentile=12),
    )

    assert advice.market_state == "undervalued_weak_trend"
    assert advice.operation_side == "buy"
    assert tuple(batch.percent for batch in advice.batches) == (10,)


def test_market_state_cannot_override_feature_derived_risk_limit() -> None:
    snapshot = _snapshot(valuation_percentile=10)
    derived = _advice(
        current_position=10,
        target_position=80,
        direction="up",
        probability=80,
        confidence=80,
        snapshot=snapshot,
    )

    assert derived.market_state == "undervalued_weak_trend"
    assert derived.target_position == 20
    with pytest.raises(ValueError, match="market_state"):
        _advice(
            current_position=10,
            target_position=80,
            direction="up",
            probability=80,
            confidence=80,
            snapshot=snapshot,
            market_state="bullish",
        )


def test_matching_market_state_is_only_a_consistency_assertion() -> None:
    advice = _advice(
        current_position=10,
        target_position=80,
        direction="up",
        probability=80,
        confidence=80,
        snapshot=_snapshot(valuation_percentile=10),
        market_state="undervalued_weak_trend",
    )

    assert advice.target_position == 20
    assert tuple(batch.percent for batch in advice.batches) == (10,)


def test_undervalued_reversal_resonance_uses_four_batches_for_sixty() -> None:
    advice = _advice(
        current_position=20,
        target_position=80,
        direction="up",
        probability=83,
        confidence=87,
        snapshot=_snapshot(
            valuation_percentile=10,
            golden_point=True,
            golden_strength=5,
        ),
    )

    assert advice.market_state == "undervalued_reversal_resonance"
    assert len(advice.batches) == 4
    assert sum(batch.percent for batch in advice.batches) == 60
    assert [batch.percent for batch in advice.batches] == sorted(
        (batch.percent for batch in advice.batches),
        reverse=True,
    )


def test_batches_are_five_multiples_conserve_position_and_use_calendar() -> None:
    calendar = _calendar()
    advice = _advice(
        current_position=95,
        target_position=15,
        direction="down",
        probability=81,
        confidence=84,
        snapshot=_snapshot(
            valuation_percentile=95,
            black_point=True,
            black_strength=5,
        ),
        calendar=calendar,
    )

    assert len(advice.batches) <= 4
    assert all(batch.percent % 5 == 0 for batch in advice.batches)
    assert sum(batch.percent for batch in advice.batches) == 80
    assert all(
        batch.tolerance_trading_days == 3
        for batch in advice.batches
    )
    assert all(
        batch.expected_date in calendar.expected_trade_dates
        and CUTOFF < batch.expected_date <= CUTOFF + timedelta(weeks=13)
        for batch in advice.batches
    )
    assert (
        advice.current_position + advice.signed_position_change
        == advice.target_position
    )


def test_unset_position_returns_analysis_and_range_without_batches_or_dates() -> None:
    advice = _advice(
        current_position=None,
        target_position=82,
        direction="up",
        probability=80,
        confidence=82,
        snapshot=_snapshot(
            valuation_percentile=10,
            golden_point=True,
            golden_strength=5,
        ),
    )

    assert advice.direction == "up"
    assert advice.target_position == 80
    assert advice.target_position_range == (75, 85)
    assert advice.current_position is None
    assert advice.total_adjustment is None
    assert advice.batches == ()


def test_low_confidence_holds_position_without_inventing_dates() -> None:
    advice = _advice(
        current_position=40,
        target_position=80,
        direction="up",
        probability=80,
        confidence=54,
        snapshot=_snapshot(valuation_percentile=20),
    )

    assert advice.recommendation == "wait_confirmation"
    assert advice.operation_side == "hold"
    assert advice.target_position == advice.current_position == 40
    assert advice.total_adjustment == 0
    assert advice.batches == ()


def test_unpublishable_features_hold_without_inventing_dates() -> None:
    advice = _advice(
        current_position=40,
        target_position=80,
        direction="up",
        probability=80,
        confidence=80,
        snapshot=_snapshot(valuation_percentile=20, publishable=False),
    )

    assert advice.recommendation == "wait_confirmation"
    assert advice.operation_side == "hold"
    assert advice.batches == ()
    assert "feature_snapshot_not_publishable" in advice.audit_notes


def test_later_batches_wait_until_confirmation_condition_is_met() -> None:
    advice = _advice(
        current_position=20,
        target_position=80,
        direction="up",
        probability=83,
        confidence=87,
        snapshot=_snapshot(
            valuation_percentile=10,
            golden_point=True,
            golden_strength=5,
        ),
        confirmation_conditions_met=False,
    )

    assert advice.batches[0].status == "ready"
    assert all(
        batch.status == "waiting_confirmation"
        for batch in advice.batches[1:]
    )
    assert all(batch.confirmation_condition for batch in advice.batches)


def test_confirmation_conditions_met_requires_a_real_boolean() -> None:
    with pytest.raises(ValueError, match="confirmation_conditions_met"):
        _advice(
            current_position=20,
            target_position=80,
            direction="up",
            probability=83,
            confidence=87,
            snapshot=_snapshot(
                valuation_percentile=10,
                golden_point=True,
                golden_strength=5,
            ),
            confirmation_conditions_met="false",  # type: ignore[arg-type]
        )


def test_positions_are_quantized_to_nearest_five_before_splitting() -> None:
    advice = _advice(
        current_position=93,
        target_position=17,
        direction="down",
        probability=82,
        confidence=86,
        snapshot=_snapshot(
            valuation_percentile=95,
            black_point=True,
            black_strength=5,
        ),
    )

    assert advice.current_position == 95
    assert advice.target_position == 15
    assert sum(batch.percent for batch in advice.batches) == 80


def test_model_snapshot_and_calendar_cannot_mix_markets() -> None:
    with pytest.raises(CrossMarketError):
        build_advice(
            model=seed_model("399006"),
            latest_features=_snapshot(instrument_code="NDX"),
            future_trading_calendar=_calendar("399006"),
            current_position=40,
            target_position=60,
            direction="up",
            probability=80,
            confidence=80,
        )


def test_relabelled_model_state_cannot_cross_market_identity() -> None:
    relabelled = replace(
        seed_model("399006"),
        instrument_code="NDX",
    )

    with pytest.raises(CrossMarketError):
        build_advice(
            model=relabelled,
            latest_features=_snapshot(instrument_code="NDX"),
            future_trading_calendar=_calendar("NDX"),
            current_position=40,
            target_position=60,
            direction="up",
            probability=80,
            confidence=80,
        )


def test_valid_audited_model_state_can_generate_advice() -> None:
    advice = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
        model=_audited_model(),
    )

    assert advice.model_version == "M0001"


def test_task5_accepted_model_state_can_generate_advice() -> None:
    model = _accepted_audited_model()

    advice = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
        model=model,
    )

    assert model.audits[-1].accepted is True
    assert advice.model_version == "M0002"


@pytest.mark.parametrize("tamper", ["version", "weights"])
def test_audited_model_state_must_match_its_proven_history(
    tamper: str,
) -> None:
    model = _audited_model()
    if tamper == "version":
        corrupted = replace(
            model,
            current_model_version="M9999",
        )
    else:
        corrupted = replace(
            model,
            current_model_weights=seed_model("NDX").current_model_weights,
        )

    with pytest.raises(ValueError, match="model|weight|audit|history"):
        _advice(
            current_position=40,
            target_position=60,
            direction="up",
            probability=80,
            confidence=80,
            model=corrupted,
        )


def test_audited_model_rejects_impossible_accepted_weight_transition() -> None:
    model = _audited_model()
    audit = model.audits[-1]
    impossible_weights = seed_model("NDX").current_model_weights
    forged_audit = replace(
        audit,
        accepted=True,
        model_version="M0002",
        candidate_weights=impossible_weights,
    )
    forged = replace(
        model,
        current_model_version="M0002",
        current_model_weights=impossible_weights,
        audits=(forged_audit,),
        week_records=FrozenDict(
            {forged_audit.week_key: forged_audit}
        ),
    )

    with pytest.raises(ValueError, match="candidate|weight|transition"):
        _advice(
            current_position=40,
            target_position=60,
            direction="up",
            probability=80,
            confidence=80,
            model=forged,
        )


def test_rejected_task5_gate_cannot_be_forged_into_accepted_model() -> None:
    model = _audited_model()
    audit = model.audits[-1]
    assert audit.accepted is False
    assert audit.reasons == ("insufficient_mature_feedback",)

    forged_audit = replace(
        audit,
        accepted=True,
        reasons=(),
        model_version="M0002",
    )
    forged = replace(
        model,
        current_model_version="M0002",
        current_model_weights=forged_audit.candidate_weights,
        audits=(forged_audit,),
        week_records=FrozenDict(
            {forged_audit.week_key: forged_audit}
        ),
    )

    with pytest.raises(ValueError, match="accept|gate|reason"):
        _advice(
            current_position=40,
            target_position=60,
            direction="up",
            probability=80,
            confidence=80,
            model=forged,
        )


def test_mutable_feature_snapshot_contract_is_rejected_before_use() -> None:
    trusted = _snapshot(valuation_percentile=10)
    mutable_features = dict(trusted.features)
    untrusted = replace(trusted, features=mutable_features)
    mutable_features["valuation_percentile"] = Decimal("95")

    with pytest.raises(ValueError, match="FrozenDict|immutable"):
        _advice(
            current_position=10,
            target_position=80,
            direction="up",
            probability=80,
            confidence=80,
            snapshot=untrusted,
        )


def test_advice_is_immutable_auditable_and_exported() -> None:
    advice = _advice(
        current_position=20,
        target_position=80,
        direction="up",
        probability=83,
        confidence=87,
        snapshot=_snapshot(
            valuation_percentile=10,
            golden_point=True,
            golden_strength=5,
        ),
    )

    with pytest.raises(FrozenInstanceError):
        advice.confidence = Decimal("1")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        advice.batches[0].status = "ready"  # type: ignore[misc]
    assert advice.advice_version == ADVICE_VERSION
    assert advice.model_version == "M0001"
    assert advice.data_cutoff_date == CUTOFF
    assert advice.source_data_max_date == CUTOFF
    assert advice.forecast_horizon_weeks == 13
    assert set(advice.direction_probabilities) == {
        "up",
        "down",
        "neutral",
    }
    assert '"advice_version":"weekly-advice-v2"' in advice.to_json()
    assert weekly_analysis.Advice is Advice
    assert weekly_analysis.AdviceBatch is AdviceBatch
    assert weekly_analysis.build_advice is build_advice
    assert weekly_analysis.fund_etf_ratio is fund_etf_ratio


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recommendation", "hold"),
        ("fund_etf_ratio", "3:7"),
        ("market_state", "arbitrary"),
        ("feature_set_version", "bogus"),
    ],
)
def test_advice_rejects_cross_field_association_tampering(
    field: str,
    value: object,
) -> None:
    advice = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
    )

    with pytest.raises(ValueError):
        replace(advice, **{field: value})


def test_market_state_is_recomputed_from_audited_feature_inputs() -> None:
    advice = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
    )
    fields = dict(advice.audit_fields)
    fields["market_state_inputs"] = FrozenDict(
        {
            "black_point": False,
            "black_strength": 0,
            "golden_point": False,
            "golden_strength": 0,
            "valuation_percentile": Decimal("95"),
        }
    )

    with pytest.raises(ValueError, match="market_state"):
        replace(advice, audit_fields=fields)


def test_confirmation_status_vector_is_bound_to_audit_gate() -> None:
    waiting = _advice(
        current_position=20,
        target_position=80,
        direction="up",
        probability=83,
        confidence=87,
        snapshot=_snapshot(
            valuation_percentile=10,
            golden_point=True,
            golden_strength=5,
        ),
        confirmation_conditions_met=False,
    )
    all_ready = tuple(
        replace(batch, status="ready")
        for batch in waiting.batches
    )
    all_waiting = tuple(
        replace(batch, status="waiting_confirmation")
        for batch in waiting.batches
    )
    ready = _advice(
        current_position=20,
        target_position=80,
        direction="up",
        probability=83,
        confidence=87,
        snapshot=_snapshot(
            valuation_percentile=10,
            golden_point=True,
            golden_strength=5,
        ),
        confirmation_conditions_met=True,
    )
    one_waiting = ready.batches[:1] + (
        replace(ready.batches[1], status="waiting_confirmation"),
    ) + ready.batches[2:]

    for advice, batches in (
        (waiting, all_ready),
        (waiting, all_waiting),
        (ready, one_waiting),
    ):
        with pytest.raises(ValueError, match="confirmation|status"):
            replace(advice, batches=batches)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tolerance_trading_days", Decimal("3")),
        ("tolerance_trading_days", True),
        ("confirmation_condition", True),
    ],
)
def test_batch_rejects_non_strict_field_types(
    field: str,
    value: object,
) -> None:
    batch = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
    ).batches[0]

    with pytest.raises(ValueError):
        replace(batch, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_adjustment", False),
        ("total_adjustment", Decimal("0")),
        ("signed_position_change", False),
        ("signed_position_change", Decimal("0")),
        ("forecast_horizon_weeks", Decimal("13")),
    ],
)
def test_advice_rejects_non_strict_integer_field_types(
    field: str,
    value: object,
) -> None:
    advice = _advice(
        current_position=40,
        target_position=40,
        direction="up",
        probability=80,
        confidence=80,
    )

    with pytest.raises(ValueError):
        replace(advice, **{field: value})


def test_audit_batch_count_rejects_bool_disguised_as_one() -> None:
    advice = _advice(
        current_position=10,
        target_position=20,
        direction="up",
        probability=80,
        confidence=80,
        snapshot=_snapshot(valuation_percentile=10),
    )
    fields = dict(advice.audit_fields)
    fields["batch_count"] = True

    with pytest.raises(ValueError, match="batch_count"):
        replace(advice, audit_fields=fields)


def test_advice_rejects_model_version_zero() -> None:
    advice = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
    )

    with pytest.raises(ValueError, match="M0001"):
        replace(advice, model_version="M0000")


def test_build_advice_rejects_unvalidated_model_version_zero() -> None:
    model = replace(
        seed_model("399006"),
        current_model_version="M0000",
    )

    with pytest.raises(ValueError, match="M0001"):
        build_advice(
            model=model,
            latest_features=_snapshot(),
            future_trading_calendar=_calendar(),
            current_position=40,
            target_position=60,
            direction="up",
            probability=80,
            confidence=80,
        )


def test_target_position_range_is_deeply_frozen_as_a_tuple() -> None:
    advice = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
    )
    source_range = [55, 65]

    replaced = replace(
        advice,
        target_position_range=source_range,  # type: ignore[arg-type]
    )
    source_range[0] = 0

    assert isinstance(replaced.target_position_range, tuple)
    assert replaced.target_position_range == (55, 65)


@pytest.mark.parametrize(
    "invalid_range",
    [
        (False, 65),
        (55.0, 65),
        (56, 65),
        (-5, 65),
        (65, 55),
    ],
)
def test_target_position_range_rejects_invalid_boundaries(
    invalid_range: tuple[object, object],
) -> None:
    advice = _advice(
        current_position=40,
        target_position=60,
        direction="up",
        probability=80,
        confidence=80,
    )

    with pytest.raises(ValueError):
        replace(
            advice,
            target_position_range=invalid_range,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("direction", "sideways"),
        ("probability", 101),
        ("confidence", -1),
        ("current_position", 101),
        ("target_position", -1),
    ],
)
def test_invalid_advice_inputs_are_rejected(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "model": seed_model("399006"),
        "latest_features": _snapshot(),
        "future_trading_calendar": _calendar(),
        "current_position": 40,
        "target_position": 60,
        "direction": "up",
        "probability": 80,
        "confidence": 80,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        build_advice(**kwargs)  # type: ignore[arg-type]


def test_source_data_after_cutoff_is_rejected() -> None:
    snapshot = replace(
        _snapshot(),
        source_data_max_date=CUTOFF + timedelta(days=1),
    )

    with pytest.raises(ValueError, match="source_data_max_date"):
        _advice(
            current_position=40,
            target_position=60,
            direction="up",
            probability=80,
            confidence=80,
            snapshot=snapshot,
        )
