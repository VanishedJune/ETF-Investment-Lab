from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from backend.app.services import weekly_analysis_service as service_module
from backend.app.weekly_analysis.domain import QualityReport
from backend.app.weekly_analysis.features import (
    FEATURE_SET_VERSION,
    FeatureSnapshot,
    FrozenDict,
)
from backend.app.weekly_analysis.optimizer import seed_model


CUTOFF = date(2026, 7, 24)


def _snapshot(**overrides: object) -> FeatureSnapshot:
    values: dict[str, object] = {
        "valuation_percentile": Decimal("50"),
        "dif": Decimal("0"),
        "dip": Decimal("0"),
        "dea": Decimal("0"),
        "eda": Decimal("0"),
        "macd_histogram": Decimal("0"),
        "ema_12": Decimal("100"),
        "ma_20": Decimal("100"),
        "ma_60": Decimal("100"),
        "rsi_6": Decimal("50"),
        "volume_ratio": Decimal("1"),
        "volatility_20": Decimal("0.03"),
        "current_drawdown": Decimal("0"),
        "golden_point": False,
        "golden_strength": 0,
        "black_point": False,
        "black_strength": 0,
    }
    values.update(overrides)
    frozen = FrozenDict(values)
    return FeatureSnapshot(
        instrument_code="399006",
        cutoff_date=CUTOFF,
        source_data_max_date=CUTOFF,
        feature_set_version=FEATURE_SET_VERSION,
        features=frozen,
        availability=FrozenDict(
            {name: "available" for name in frozen}
        ),
        quality_report=QualityReport(
            is_publishable=True,
            issues=(),
            volume_availability="available",
        ),
        audit_fields=FrozenDict({"warmup_weeks": 60}),
        source_hash="a" * 64,
    )


def _judge(snapshot: FeatureSnapshot, current_position: int = 50):
    return service_module.RuleBasedJudgmentPolicy().judge(
        model=seed_model("399006"),
        features=snapshot,
        sample=SimpleNamespace(symbol="399006"),
        current_position=current_position,
    )


@pytest.mark.parametrize(
    ("snapshot", "direction", "target"),
    [
        (
            _snapshot(
                valuation_percentile=Decimal("94"),
                dif=Decimal("1.0"),
                dip=Decimal("1.0"),
                dea=Decimal("1.3"),
                eda=Decimal("1.3"),
                macd_histogram=Decimal("-0.6"),
                ema_12=Decimal("112"),
                ma_20=Decimal("108"),
                ma_60=Decimal("101"),
            ),
            "up",
            70,
        ),
        (
            _snapshot(
                valuation_percentile=Decimal("97"),
                dif=Decimal("-0.8"),
                dip=Decimal("-0.8"),
                dea=Decimal("0.2"),
                eda=Decimal("0.2"),
                macd_histogram=Decimal("-2.0"),
                ema_12=Decimal("92"),
                ma_20=Decimal("100"),
                ma_60=Decimal("104"),
                black_point=True,
                black_strength=5,
            ),
            "down",
            15,
        ),
        (
            _snapshot(
                valuation_percentile=Decimal("8"),
                dif=Decimal("-2.0"),
                dip=Decimal("-2.0"),
                dea=Decimal("-1.2"),
                eda=Decimal("-1.2"),
                macd_histogram=Decimal("-1.6"),
                ema_12=Decimal("91"),
                ma_20=Decimal("100"),
                ma_60=Decimal("106"),
            ),
            "up",
            20,
        ),
        (
            _snapshot(
                valuation_percentile=Decimal("12"),
                dif=Decimal("1.4"),
                dip=Decimal("1.4"),
                dea=Decimal("0.7"),
                eda=Decimal("0.7"),
                macd_histogram=Decimal("1.4"),
                ema_12=Decimal("111"),
                ma_20=Decimal("104"),
                ma_60=Decimal("99"),
                golden_point=True,
                golden_strength=5,
            ),
            "up",
            80,
        ),
    ],
)
def test_rule_policy_covers_the_four_required_market_scenarios(
    snapshot: FeatureSnapshot,
    direction: str,
    target: int,
) -> None:
    judgment = _judge(snapshot)

    assert judgment.direction == direction
    assert judgment.target_position == target
    assert judgment.target_position % 5 == 0
    assert judgment.direction_probabilities is not None
    assert (
        sum(judgment.direction_probabilities.values(), Decimal())
        == Decimal("100")
    )
    assert (
        judgment.direction_probabilities[judgment.direction]
        == judgment.probability
    )


def test_rule_policy_market_score_does_not_depend_on_user_position() -> None:
    snapshot = _snapshot(
        valuation_percentile=Decimal("97"),
        dif=Decimal("-0.8"),
        dea=Decimal("0.2"),
        macd_histogram=Decimal("-2"),
        ema_12=Decimal("92"),
        ma_20=Decimal("100"),
        ma_60=Decimal("104"),
        black_point=True,
        black_strength=5,
    )

    assert _judge(snapshot, 10) == _judge(snapshot, 95)


def test_production_composition_defaults_to_rule_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Provider:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class Repository:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class Engine:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class Manager:
        def __init__(self, **kwargs) -> None:
            self.engine = kwargs["engine"]

    monkeypatch.setattr(
        service_module,
        "LocalVerifiedDatasetProvider",
        Provider,
    )
    monkeypatch.setattr(
        service_module,
        "WeeklyAnalysisRepository",
        Repository,
    )
    monkeypatch.setattr(
        service_module,
        "WeeklyAnalysisEngine",
        Engine,
    )
    monkeypatch.setattr(
        service_module,
        "WeeklyAnalysisTaskManager",
        Manager,
    )

    service_module.build_weekly_analysis_service(lambda: None)

    assert isinstance(
        captured["judgment_policy"],
        service_module.RuleBasedJudgmentPolicy,
    )
