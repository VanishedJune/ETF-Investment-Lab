from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    ResearchReport,
    StrategyDefinition,
    StrategySignal,
    ValuationRecord,
)
from backend.app.schemas.strategy import (
    StrategyConfig,
    StrategyMultipliers,
    StrategyThresholds,
    StrategyWeights,
    strategy_config_from_mapping,
)
from backend.app.strategies.engine import RuleStrategyEngine, StrategySnapshot
from backend.app.services.strategy_service import StrategyService


AS_OF = date(2026, 2, 2)


def _safe_config() -> StrategyConfig:
    return StrategyConfig(
        version="2.0",
        weights=StrategyWeights(
            valuation=Decimal("0.25"),
            trend=Decimal("0.25"),
            momentum=Decimal("0.15"),
            volume=Decimal("0.10"),
            volatility=Decimal("0.10"),
            risk=Decimal("0.15"),
        ),
        thresholds=StrategyThresholds(
            valuation_low_percentile=Decimal("0.30"),
            valuation_high_percentile=Decimal("0.70"),
            momentum_rsi_low=Decimal("35"),
            momentum_rsi_high=Decimal("70"),
            volume_ratio_low=Decimal("0.80"),
            volume_ratio_high=Decimal("1.20"),
            volatility_high=Decimal("0.35"),
            drawdown_pause=Decimal("-0.20"),
            increase_score=Decimal("0.35"),
            reduce_score=Decimal("-0.20"),
            sell_partial_score=Decimal("-0.45"),
            pause_score=Decimal("-0.75"),
        ),
        multipliers=StrategyMultipliers(
            increase=Decimal("1.30"),
            normal=Decimal("1.00"),
            reduce=Decimal("0.75"),
            pause=Decimal("0.50"),
            hold=Decimal("1.00"),
            sell_partial=Decimal("0.70"),
        ),
        maximum_sell_ratio=Decimal("0.30"),
        minimum_holding_ratio=Decimal("0.20"),
    )


def _snapshot(*, bullish: bool = True, missing: bool = False) -> StrategySnapshot:
    if missing:
        return StrategySnapshot(
            instrument_code="589850",
            as_of_date=AS_OF,
            data_cutoff=AS_OF,
            close_price=None,
            volume=None,
            valuation={},
            indicators={},
            observations={},
        )
    return StrategySnapshot(
        instrument_code="589850",
        as_of_date=AS_OF,
        data_cutoff=AS_OF,
        close_price=Decimal("120" if bullish else "80"),
        volume=Decimal("1300" if bullish else "700"),
        valuation={"valuation_percentile": Decimal("0.15" if bullish else "0.85")},
        indicators={
            "ma_20": Decimal("100"),
            "ma_60": Decimal("95" if bullish else "105"),
            "rsi_6": Decimal("55" if bullish else "78"),
            "macd_histogram": Decimal("2" if bullish else "-2"),
            "volume_ma_20": Decimal("1000"),
            "volatility_20": Decimal("0.15" if bullish else "0.50"),
            "current_drawdown": Decimal("-0.05" if bullish else "-0.35"),
            "running_drawdown": Decimal("-0.10" if bullish else "-0.40"),
        },
        observations={"price_date": AS_OF.isoformat(), "valuation_date": AS_OF.isoformat()},
    )


def _make_database(tmp_path: Path) -> Path:
    database = tmp_path / "data" / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    return database


def _seed_strategy_data(database: Path, *, add_future: bool = False) -> None:
    engine = create_database_engine(database)
    with engine.begin() as connection:
        instrument_id = connection.execute(
            select(Instrument.id).where(Instrument.code == "589850")
        ).scalar_one()
        connection.execute(
            MarketPrice.__table__.insert().values(
                instrument_id=instrument_id,
                trade_date=AS_OF,
                timeframe="daily",
                close_price=Decimal("120"),
                volume=Decimal("1300"),
                source="TEST",
            )
        )
        connection.execute(
            ValuationRecord.__table__.insert().values(
                instrument_id=instrument_id,
                valuation_date=AS_OF,
                valuation_percentile=Decimal("0.15"),
                pe_ratio=Decimal("20"),
                pb_ratio=Decimal("2"),
                dividend_yield=Decimal("0.02"),
                raw_values={},
            )
        )
        connection.execute(
            IndicatorRecord.__table__.insert().values(
                instrument_id=instrument_id,
                indicator_date=AS_OF,
                timeframe="daily",
                indicator_name="technical_indicators",
                indicator_values={
                    "values": {
                        "ma_20": "100",
                        "ma_60": "95",
                        "rsi_6": "55",
                        "macd_histogram": "2",
                        "volume_ma_20": "1000",
                        "volatility_20": "0.15",
                        "current_drawdown": "-0.05",
                        "running_drawdown": "-0.10",
                    },
                    "metadata": {},
                },
            )
        )
        if add_future:
            future = AS_OF + timedelta(days=1)
            connection.execute(
                MarketPrice.__table__.insert().values(
                    instrument_id=instrument_id,
                    trade_date=future,
                    timeframe="daily",
                    close_price=Decimal("10000"),
                    volume=Decimal("1"),
                    source="FUTURE",
                )
            )
            connection.execute(
                ValuationRecord.__table__.insert().values(
                    instrument_id=instrument_id,
                    valuation_date=future,
                    valuation_percentile=Decimal("0.99"),
                    raw_values={},
                )
            )


def _create_strategy(database: Path, config: StrategyConfig | None = None) -> None:
    with Session(create_database_engine(database)) as session, session.begin():
        session.add(
            StrategyDefinition(
                name="test-rule-strategy",
                strategy_type="rule",
                parameters=(config or _safe_config()).model_dump(mode="json"),
                version=(config or _safe_config()).version,
            )
        )


def test_engine_scores_known_trend_and_valuation_conditions_deterministically() -> None:
    engine = RuleStrategyEngine(_safe_config())

    bullish_first = engine.evaluate(_snapshot(bullish=True))
    bullish_second = engine.evaluate(_snapshot(bullish=True))
    bearish = engine.evaluate(_snapshot(bullish=False))

    assert bullish_first == bullish_second
    assert bullish_first.score > bearish.score
    assert bullish_first.multiplier > bearish.multiplier
    assert bullish_first.recommendation == "INCREASE"
    assert "valuation_low_percentile" in bullish_first.triggered_rules
    assert "trend_close_above_ma_20" in bullish_first.triggered_rules


def test_missing_strategy_data_degrades_confidence_and_records_missing_reasons() -> None:
    signal = RuleStrategyEngine(_safe_config()).evaluate(_snapshot(missing=True))

    assert signal.confidence < Decimal("0.50")
    assert signal.recommendation in {"INCREASE", "NORMAL", "REDUCE", "PAUSE", "HOLD", "SELL_PARTIAL"}
    assert signal.recommendation != "SELL_ALL"
    assert any(reason.startswith("missing:") for reason in signal.reasons)


def test_pause_precedes_sell_partial_at_maximum_negative_score() -> None:
    signal = RuleStrategyEngine(_safe_config()).evaluate(
        _snapshot(bullish=False), holding_ratio=Decimal("0.80")
    )

    assert signal.score == Decimal("-1")
    assert signal.recommendation == "PAUSE"
    assert signal.suggested_sell_ratio == Decimal("0")


def test_moderately_negative_score_still_uses_bounded_partial_sell() -> None:
    baseline = _snapshot(bullish=False)
    indicators = dict(baseline.indicators)
    indicators["volatility_20"] = Decimal("0.15")
    snapshot = replace(baseline, volume=Decimal("1300"), indicators=indicators)

    signal = RuleStrategyEngine(_safe_config()).evaluate(snapshot, holding_ratio=Decimal("0.25"))

    assert signal.score == Decimal("-0.60")
    assert signal.recommendation == "SELL_PARTIAL"
    assert signal.suggested_sell_ratio == Decimal("0.05")
    assert Decimal("0.25") - signal.suggested_sell_ratio >= Decimal("0.20")


@pytest.mark.parametrize(
    "update",
    (
        {"maximum_sell_ratio": Decimal("0.31")},
        {"minimum_holding_ratio": Decimal("0.19")},
        {"weights": {"valuation": Decimal("0"), "trend": Decimal("0"), "momentum": Decimal("0"), "volume": Decimal("0"), "volatility": Decimal("0"), "risk": Decimal("0")}},
    ),
)
def test_strategy_config_rejects_unsafe_sell_holding_and_weight_values(update: dict[str, object]) -> None:
    payload = _safe_config().model_dump()
    payload.update(update)
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate(payload)


def test_strategy_config_reads_legacy_default_parameters_without_widening_safety_limits() -> None:
    config = strategy_config_from_mapping(
        {
            "weights": {"valuation": 0.4, "trend": 0.35, "technical": 0.25},
            "thresholds": {"buy_signal": 0.65, "reduce_signal": 0.75},
            "multipliers": {"valuation": 1.3, "trend": 1.15, "combined": 1.4},
            "max_single_sale_ratio": 0.3,
            "minimum_holding_ratio": 0.2,
        },
        strategy_version="1.0",
    )

    assert config.version == "1.0"
    assert config.weights.momentum == Decimal("0.25")
    assert config.multipliers.increase == Decimal("1.4")
    assert config.maximum_sell_ratio == Decimal("0.30")


def test_service_uses_only_as_of_or_earlier_records_and_persists_signal_and_fixed_report(
    tmp_path: Path,
) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database, add_future=True)
    _create_strategy(database)
    service = StrategyService(create_session_factory(database))

    first = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)
    second = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)

    assert first.signal_id == second.signal_id
    assert first.snapshot.close_price == Decimal("120")
    assert first.snapshot.data_cutoff == AS_OF
    assert first.recommendation in {"INCREASE", "NORMAL", "REDUCE", "PAUSE", "HOLD", "SELL_PARTIAL"}
    assert first.report_id == second.report_id
    assert "## 数据来源与截止日期" in first.markdown
    assert "## 价格、估值与指标" in first.markdown
    assert "## 规则触发与反向风险" in first.markdown
    assert "## 研究用途免责声明" in first.markdown
    assert "不构成投资建议" in first.markdown
    with Session(create_database_engine(database)) as session:
        assert session.scalar(select(func.count()).select_from(StrategySignal)) == 1
        assert session.scalar(select(func.count()).select_from(ResearchReport)) == 1


def test_service_applies_plan_amount_bounds_and_exports_csv_rows(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    with Session(create_database_engine(database)) as session, session.begin():
        instrument_id = session.scalar(select(Instrument.id).where(Instrument.code == "589850"))
        session.add(
            InvestmentPlan(
                instrument_id=instrument_id,
                name="bounded plan",
                amount=Decimal("100"),
                rule_parameters={"minimum_amount": "80", "maximum_amount": "110"},
            )
        )

    outcome = StrategyService(create_session_factory(database)).evaluate_and_store(
        "test-rule-strategy", "589850", AS_OF, plan_name="bounded plan"
    )

    assert outcome.suggested_buy_amount == Decimal("110")
    rows = StrategyService(create_session_factory(database)).export_csv_rows(etf_code="589850")
    assert rows == [
        {
            "as_of_date": AS_OF.isoformat(),
            "instrument_code": "589850",
            "recommendation": outcome.recommendation,
            "score": str(outcome.score),
            "confidence": str(outcome.confidence),
            "strategy_version": "2.0",
        }
    ]


def test_same_version_config_mutation_creates_immutable_hashed_signal_and_report(
    tmp_path: Path,
) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    service = StrategyService(create_session_factory(database))

    original = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)
    with Session(create_database_engine(database)) as session, session.begin():
        strategy = session.scalar(
            select(StrategyDefinition).where(StrategyDefinition.name == "test-rule-strategy")
        )
        assert strategy is not None
        parameters = dict(strategy.parameters)
        multipliers = dict(parameters["multipliers"])
        multipliers["increase"] = "1.10"
        parameters["multipliers"] = multipliers
        strategy.parameters = parameters
    changed = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)

    assert original.signal_id != changed.signal_id
    assert original.report_id != changed.report_id
    with Session(create_database_engine(database)) as session:
        signals = session.scalars(select(StrategySignal).order_by(StrategySignal.id)).all()
        reports = session.scalars(select(ResearchReport).order_by(ResearchReport.id)).all()
    assert len(signals) == 2
    assert len(reports) == 2
    assert signals[0].strategy_version == signals[1].strategy_version == "2.0"
    assert signals[0].strategy_config_hash != signals[1].strategy_config_hash
    assert signals[0].signal_data["strategy_config_json"]
    assert signals[0].signal_data["strategy_config_hash"] == signals[0].strategy_config_hash
    assert signals[0].strategy_config_hash in (signals[0].rationale or "")
    assert reports[0].strategy_id == reports[1].strategy_id == signals[0].strategy_id
    assert reports[0].strategy_config_hash != reports[1].strategy_config_hash


def test_report_csv_export_reads_persisted_report_identity_rows(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    service = StrategyService(create_session_factory(database))

    result = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)

    rows = service.export_report_csv_rows(etf_code="589850", report_date=AS_OF)
    assert rows == [
        {
            "report_date": AS_OF.isoformat(),
            "instrument_code": "589850",
            "report_type": "strategy_research",
            "strategy_version": "2.0",
            "strategy_config_hash": result.strategy_config_hash,
            "source_data_hash": result.source_data_hash,
            "status": "generated",
        }
    ]


def test_all_missing_snapshot_holds_and_never_suggests_a_plan_buy(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    _create_strategy(database)

    result = StrategyService(create_session_factory(database)).evaluate_and_store(
        "test-rule-strategy", "589850", AS_OF
    )

    assert result.recommendation == "HOLD"
    assert result.confidence == Decimal("0")
    assert result.suggested_buy_amount == Decimal("0")


def test_corrected_historical_source_data_creates_immutable_signal_and_report_revision(
    tmp_path: Path,
) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    service = StrategyService(create_session_factory(database))

    original = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)
    with Session(create_database_engine(database)) as session, session.begin():
        instrument_id = session.scalar(select(Instrument.id).where(Instrument.code == "589850"))
        price = session.scalar(
            select(MarketPrice).where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.trade_date == AS_OF,
                MarketPrice.timeframe == "daily",
            )
        )
        assert price is not None
        price.close_price = Decimal("121")
    corrected = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)

    assert corrected.signal_id != original.signal_id
    assert corrected.report_id != original.report_id
    assert corrected.source_data_hash != original.source_data_hash
    with Session(create_database_engine(database)) as session:
        signals = session.scalars(select(StrategySignal).order_by(StrategySignal.id)).all()
        reports = session.scalars(select(ResearchReport).order_by(ResearchReport.id)).all()
    assert len(signals) == len(reports) == 2
    assert signals[0].rationale == "\n".join(signals[0].signal_data["reasons"])
    assert reports[0].content == original.markdown


def test_holding_ratio_is_hashed_as_an_immutable_evaluation_input(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    with Session(create_database_engine(database)) as session, session.begin():
        price = session.scalar(select(MarketPrice).where(MarketPrice.trade_date == AS_OF))
        valuation = session.scalar(select(ValuationRecord).where(ValuationRecord.valuation_date == AS_OF))
        indicator = session.scalar(select(IndicatorRecord).where(IndicatorRecord.indicator_date == AS_OF))
        assert price is not None and valuation is not None and indicator is not None
        price.close_price = Decimal("80")
        price.volume = Decimal("1300")
        valuation.valuation_percentile = Decimal("0.85")
        payload = dict(indicator.indicator_values)
        values = dict(payload["values"])
        values.update(
            {
                "ma_20": "100",
                "ma_60": "105",
                "rsi_6": "78",
                "macd_histogram": "-2",
                "volume_ma_20": "1000",
                "volatility_20": "0.15",
                "current_drawdown": "-0.35",
            }
        )
        payload["values"] = values
        indicator.indicator_values = payload
    service = StrategyService(create_session_factory(database))

    high_holding = service.evaluate_and_store(
        "test-rule-strategy", "589850", AS_OF, holding_ratio=Decimal("1.0")
    )
    minimum_holding = service.evaluate_and_store(
        "test-rule-strategy", "589850", AS_OF, holding_ratio=Decimal("0.2")
    )

    assert high_holding.recommendation == "SELL_PARTIAL"
    assert minimum_holding.recommendation == "REDUCE"
    assert high_holding.source_data_hash != minimum_holding.source_data_hash
    assert high_holding.signal_id != minimum_holding.signal_id
    assert high_holding.report_id != minimum_holding.report_id
    with Session(create_database_engine(database)) as session:
        signals = session.scalars(select(StrategySignal).order_by(StrategySignal.id)).all()
    assert [signal.signal_data["source_snapshot"]["holding_ratio"] for signal in signals] == [
        "1.0",
        "0.2",
    ]


def test_instrument_rename_creates_an_immutable_report_revision(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    service = StrategyService(create_session_factory(database))

    original = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)
    with Session(create_database_engine(database)) as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "589850"))
        assert instrument is not None
        instrument.name = "Renamed ETF"
    renamed = service.evaluate_and_store("test-rule-strategy", "589850", AS_OF)

    assert renamed.source_data_hash != original.source_data_hash
    assert renamed.report_id != original.report_id
    assert "Renamed ETF" in renamed.markdown
    with Session(create_database_engine(database)) as session:
        reports = session.scalars(select(ResearchReport).order_by(ResearchReport.id)).all()
    assert len(reports) == 2
    assert reports[0].content == original.markdown
    assert reports[1].content == renamed.markdown
    assert reports[1].strategy_config_hash == reports[0].strategy_config_hash


def test_concurrent_identical_evaluation_is_idempotent(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    barrier = Barrier(2)

    class BarrierStrategyService(StrategyService):
        def _store_signal(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            barrier.wait(timeout=10)
            return super()._store_signal(*args, **kwargs)

    service = BarrierStrategyService(create_session_factory(database))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result(timeout=20)
            for future in (
                executor.submit(service.evaluate_and_store, "test-rule-strategy", "589850", AS_OF),
                executor.submit(service.evaluate_and_store, "test-rule-strategy", "589850", AS_OF),
            )
        ]

    assert {result.signal_id for result in results} == {results[0].signal_id}
    assert {result.report_id for result in results} == {results[0].report_id}
    with Session(create_database_engine(database)) as session:
        assert session.scalar(select(func.count()).select_from(StrategySignal)) == 1
        assert session.scalar(select(func.count()).select_from(ResearchReport)) == 1


def test_nonfinite_indicator_json_is_a_typed_invalid_missing_input_not_a_score(tmp_path: Path) -> None:
    database = _make_database(tmp_path)
    _seed_strategy_data(database)
    _create_strategy(database)
    with Session(create_database_engine(database)) as session, session.begin():
        indicator = session.scalar(select(IndicatorRecord).where(IndicatorRecord.indicator_date == AS_OF))
        assert indicator is not None
        values = dict(indicator.indicator_values)
        metrics = dict(values["values"])
        metrics["rsi_6"] = "NaN"
        values["values"] = metrics
        indicator.indicator_values = values

    result = StrategyService(create_session_factory(database)).evaluate_and_store(
        "test-rule-strategy", "589850", AS_OF
    )

    assert "indicator:rsi_6:non_finite" in result.snapshot.invalid_fields
    with Session(create_database_engine(database)) as session:
        signal = session.scalar(select(StrategySignal))
    assert signal is not None
    assert "invalid:indicator:rsi_6:non_finite" in (signal.rationale or "")
