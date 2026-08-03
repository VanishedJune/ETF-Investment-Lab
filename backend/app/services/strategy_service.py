"""Local persistence and exports for deterministic ETF strategy research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from typing import Callable, Mapping

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..models.models import (
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    ResearchReport,
    StrategyDefinition,
    StrategySignal,
    ValuationRecord,
)
from ..schemas.strategy import StrategyConfig, strategy_config_from_mapping
from ..strategies.engine import RuleStrategyEngine, StrategyEvaluation, StrategySnapshot
from ..strategies.report_template import render_strategy_report


@dataclass(frozen=True, slots=True)
class StrategyServiceResult:
    signal_id: int
    report_id: int
    snapshot: StrategySnapshot
    score: Decimal
    confidence: Decimal
    multiplier: Decimal
    recommendation: str
    suggested_buy_amount: Decimal | None
    suggested_sell_ratio: Decimal
    strategy_config_hash: str
    source_data_hash: str
    markdown: str


def _decimal(
    value: object,
    *,
    invalid_fields: list[str] | None = None,
    field: str | None = None,
) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        if invalid_fields is not None and field is not None:
            invalid_fields.append(f"{field}:invalid_decimal")
        return None
    if not parsed.is_finite():
        if invalid_fields is not None and field is not None:
            invalid_fields.append(f"{field}:non_finite")
        return None
    return parsed


def _serialized_decimal(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _csv_decimal(value: Decimal | None) -> str:
    return "" if value is None else format(value, ".8f")


def _canonical_strategy_config(config: StrategyConfig) -> tuple[dict[str, object], str, str]:
    """Serialize the validated config identically on every supported platform."""
    payload = config.model_dump(mode="json")
    canonical_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return payload, canonical_json, hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _canonical_source_snapshot(
    snapshot: StrategySnapshot,
    plan: InvestmentPlan | None,
    holding_ratio: Decimal | None,
    instrument_name: str,
) -> tuple[dict[str, object], str, str]:
    """Hash every local input that can change a dated strategy outcome."""
    plan_snapshot: dict[str, object] | None = None
    if plan is not None:
        plan_snapshot = {
            "id": plan.id,
            "amount": _serialized_decimal(plan.amount),
            "frequency": plan.frequency,
            "enabled": plan.enabled,
            "rule_parameters": dict(plan.rule_parameters or {}),
        }
    payload: dict[str, object] = {
        "instrument_code": snapshot.instrument_code,
        "instrument_name": instrument_name,
        "as_of_date": snapshot.as_of_date.isoformat(),
        "data_cutoff": snapshot.data_cutoff.isoformat() if snapshot.data_cutoff else None,
        "close_price": _serialized_decimal(snapshot.close_price),
        "volume": _serialized_decimal(snapshot.volume),
        "valuation": {key: _serialized_decimal(value) for key, value in snapshot.valuation.items()},
        "indicators": {key: _serialized_decimal(value) for key, value in snapshot.indicators.items()},
        "observations": dict(snapshot.observations),
        "invalid_fields": list(snapshot.invalid_fields),
        "plan": plan_snapshot,
        "holding_ratio": _serialized_decimal(holding_ratio),
    }
    canonical_json = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return payload, canonical_json, hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class StrategyService:
    """Builds date-bounded snapshots and stores only local research artifacts."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    def evaluate_and_store(
        self,
        strategy_name: str,
        instrument_code: str,
        as_of_date: date,
        *,
        holding_ratio: Decimal | None = None,
        plan_name: str | None = None,
    ) -> StrategyServiceResult:
        """Evaluate one strategy date with stored same-or-earlier records only."""
        if holding_ratio is not None and (
            not holding_ratio.is_finite() or not Decimal("0") <= holding_ratio <= Decimal("1")
        ):
            raise ValueError("holding_ratio must be between 0 and 1")
        with self.session_factory() as session, session.begin():
            strategy = session.scalar(
                select(StrategyDefinition).where(StrategyDefinition.name == strategy_name)
            )
            if strategy is None:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            if not strategy.enabled:
                raise ValueError(f"Strategy is disabled: {strategy_name}")
            instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
            if instrument is None:
                raise ValueError(f"Unknown instrument code: {instrument_code}")
            config = strategy_config_from_mapping(strategy.parameters, strategy_version=strategy.version)
            config_payload, config_json, config_hash = _canonical_strategy_config(config)
            snapshot = self._build_snapshot(session, instrument, as_of_date)
            evaluation = RuleStrategyEngine(config).evaluate(snapshot, holding_ratio=holding_ratio)
            plan = self._plan(session, instrument.id, plan_name)
            suggested_buy = self._suggested_buy_amount(plan, evaluation)
            source_payload, source_json, source_hash = _canonical_source_snapshot(
                snapshot, plan, holding_ratio, instrument.name
            )
            signal = self._store_signal(
                session,
                strategy,
                instrument,
                snapshot,
                config,
                config_payload,
                config_json,
                config_hash,
                source_payload,
                source_json,
                source_hash,
                evaluation,
                suggested_buy,
            )
            markdown = render_strategy_report(
                instrument_name=instrument.name,
                snapshot=snapshot,
                evaluation=evaluation,
                strategy_version=config.version,
                strategy_config_hash=config_hash,
                source_data_hash=source_hash,
                suggested_buy_amount=suggested_buy,
            )
            report = self._store_report(
                session,
                strategy,
                instrument,
                as_of_date,
                config.version,
                config_hash,
                source_hash,
                markdown,
            )
            session.flush()
            return StrategyServiceResult(
                signal_id=signal.id,
                report_id=report.id,
                snapshot=snapshot,
                score=evaluation.score,
                confidence=evaluation.confidence,
                multiplier=evaluation.multiplier,
                recommendation=evaluation.recommendation,
                suggested_buy_amount=suggested_buy,
                suggested_sell_ratio=evaluation.suggested_sell_ratio,
                strategy_config_hash=config_hash,
                source_data_hash=source_hash,
                markdown=markdown,
            )

    @staticmethod
    def _build_snapshot(session: Session, instrument: Instrument, as_of_date: date) -> StrategySnapshot:
        invalid_fields: list[str] = []
        price = session.scalar(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= as_of_date,
            )
            .order_by(MarketPrice.trade_date.desc(), MarketPrice.id.desc())
        )
        valuation = session.scalar(
            select(ValuationRecord)
            .where(
                ValuationRecord.instrument_id == instrument.id,
                ValuationRecord.valuation_date <= as_of_date,
            )
            .order_by(ValuationRecord.valuation_date.desc(), ValuationRecord.id.desc())
        )
        indicator = session.scalar(
            select(IndicatorRecord)
            .where(
                IndicatorRecord.instrument_id == instrument.id,
                IndicatorRecord.timeframe == "daily",
                IndicatorRecord.indicator_date <= as_of_date,
            )
            .order_by(IndicatorRecord.indicator_date.desc(), IndicatorRecord.id.desc())
        )
        indicator_values = dict((indicator.indicator_values or {}).get("values") or {}) if indicator else {}
        expected_indicators = (
            "ma_20",
            "ma_60",
            "rsi_6",
            "macd_histogram",
            "volume_ma_20",
            "volatility_20",
            "current_drawdown",
            "running_drawdown",
        )
        dates = [
            row_date
            for row_date in (
                price.trade_date if price else None,
                valuation.valuation_date if valuation else None,
                indicator.indicator_date if indicator else None,
            )
            if row_date is not None
        ]
        price_count = session.scalar(
            select(func.count())
            .select_from(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= as_of_date,
            )
        )
        return StrategySnapshot(
            instrument_code=instrument.code,
            as_of_date=as_of_date,
            data_cutoff=max(dates) if dates else None,
            close_price=_decimal(
                price.close_price if price else None,
                invalid_fields=invalid_fields,
                field="price:close_price",
            ),
            volume=_decimal(
                (
                    price.volume * Decimal(price.volume_multiplier)
                    if price is not None and price.volume is not None
                    else None
                ),
                invalid_fields=invalid_fields,
                field="price:volume",
            ),
            valuation={
                "valuation_percentile": _decimal(
                    valuation.valuation_percentile if valuation else None,
                    invalid_fields=invalid_fields,
                    field="valuation:valuation_percentile",
                ),
                "pe_ratio": _decimal(
                    valuation.pe_ratio if valuation else None,
                    invalid_fields=invalid_fields,
                    field="valuation:pe_ratio",
                ),
                "pb_ratio": _decimal(
                    valuation.pb_ratio if valuation else None,
                    invalid_fields=invalid_fields,
                    field="valuation:pb_ratio",
                ),
                "dividend_yield": _decimal(
                    valuation.dividend_yield if valuation else None,
                    invalid_fields=invalid_fields,
                    field="valuation:dividend_yield",
                ),
            },
            indicators={
                name: _decimal(
                    indicator_values.get(name),
                    invalid_fields=invalid_fields,
                    field=f"indicator:{name}",
                )
                for name in expected_indicators
            },
            observations={
                "price_date": price.trade_date.isoformat() if price else None,
                "valuation_date": valuation.valuation_date.isoformat() if valuation else None,
                "indicator_date": indicator.indicator_date.isoformat() if indicator else None,
                "price_source": price.source if price and price.source else "本地存储",
                "market_observations": int(price_count or 0),
            },
            invalid_fields=tuple(invalid_fields),
        )

    @staticmethod
    def _plan(session: Session, instrument_id: int, plan_name: str | None) -> InvestmentPlan | None:
        statement: Select[tuple[InvestmentPlan]] = select(InvestmentPlan).where(
            InvestmentPlan.instrument_id == instrument_id,
            InvestmentPlan.enabled.is_(True),
        )
        if plan_name is not None:
            statement = statement.where(InvestmentPlan.name == plan_name)
        return session.scalar(statement.order_by(InvestmentPlan.id))

    @staticmethod
    def _suggested_buy_amount(
        plan: InvestmentPlan | None,
        evaluation: StrategyEvaluation,
    ) -> Decimal | None:
        if plan is None:
            return None
        if evaluation.recommendation not in {"INCREASE", "NORMAL"}:
            return Decimal("0")
        parameters = dict(plan.rule_parameters or {})
        minimum = _decimal(parameters.get("minimum_amount", parameters.get("min_amount")))
        maximum = _decimal(parameters.get("maximum_amount", parameters.get("max_amount")))
        if minimum is not None and minimum < 0:
            raise ValueError("Plan minimum amount must be nonnegative")
        if maximum is not None and maximum < 0:
            raise ValueError("Plan maximum amount must be nonnegative")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("Plan minimum amount cannot exceed maximum amount")
        amount = plan.amount * evaluation.multiplier
        if minimum is not None:
            amount = max(amount, minimum)
        if maximum is not None:
            amount = min(amount, maximum)
        return amount

    def _store_signal(
        self,
        session: Session,
        strategy: StrategyDefinition,
        instrument: Instrument,
        snapshot: StrategySnapshot,
        config: StrategyConfig,
        config_payload: Mapping[str, object],
        config_json: str,
        config_hash: str,
        source_payload: Mapping[str, object],
        source_json: str,
        source_hash: str,
        evaluation: StrategyEvaluation,
        suggested_buy: Decimal | None,
    ) -> StrategySignal:
        identity = {
            "strategy_id": strategy.id,
            "instrument_id": instrument.id,
            "as_of_date": snapshot.as_of_date,
            "strategy_version": config.version,
            "strategy_config_hash": config_hash,
            "source_data_hash": source_hash,
        }
        rationale_reasons = (
            *evaluation.reasons,
            f"strategy_config_sha256={config_hash}",
            f"source_data_sha256={source_hash}",
        )
        data = self._signal_data(
            snapshot,
            config_payload,
            config_json,
            config_hash,
            source_payload,
            source_json,
            source_hash,
            evaluation,
            suggested_buy,
            rationale_reasons,
        )
        values = {
            "signal_at": datetime.combine(snapshot.as_of_date, time.min, tzinfo=timezone.utc),
            "recommendation_type": evaluation.recommendation,
            "score": evaluation.score,
            "confidence": evaluation.confidence,
            "target_allocation": None,
            "rationale": "\n".join(rationale_reasons),
            "signal_data": data,
        }
        insert = sqlite_insert(StrategySignal).values(**identity, **values)
        session.execute(
            insert.on_conflict_do_nothing(
                index_elements=(
                    "strategy_id",
                    "instrument_id",
                    "as_of_date",
                    "strategy_version",
                    "strategy_config_hash",
                    "source_data_hash",
                )
            )
        )
        signal = session.scalar(select(StrategySignal).filter_by(**identity))
        if signal is None:  # Defensive: SQLite conflict handling is atomic.
            raise RuntimeError("Could not persist deterministic strategy signal")
        return signal

    @staticmethod
    def _signal_data(
        snapshot: StrategySnapshot,
        config_payload: Mapping[str, object],
        config_json: str,
        config_hash: str,
        source_payload: Mapping[str, object],
        source_json: str,
        source_hash: str,
        evaluation: StrategyEvaluation,
        suggested_buy: Decimal | None,
        rationale_reasons: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "strategy_version": str(config_payload["version"]),
            "strategy_config": dict(config_payload),
            "strategy_config_json": config_json,
            "strategy_config_hash": config_hash,
            "source_snapshot": dict(source_payload),
            "source_snapshot_json": source_json,
            "source_data_hash": source_hash,
            "as_of_date": snapshot.as_of_date.isoformat(),
            "data_cutoff": snapshot.data_cutoff.isoformat() if snapshot.data_cutoff else None,
            "close_price": _serialized_decimal(snapshot.close_price),
            "volume": _serialized_decimal(snapshot.volume),
            "valuation": {key: _serialized_decimal(value) for key, value in snapshot.valuation.items()},
            "indicators": {key: _serialized_decimal(value) for key, value in snapshot.indicators.items()},
            "observations": dict(snapshot.observations),
            "invalid_fields": list(snapshot.invalid_fields),
            "component_scores": {
                key: _serialized_decimal(value) for key, value in evaluation.component_scores.items()
            },
            "triggered_rules": list(evaluation.triggered_rules),
            "reverse_risks": list(evaluation.reverse_risks),
            "reasons": list(rationale_reasons),
            "multiplier": _serialized_decimal(evaluation.multiplier),
            "suggested_buy_amount": _serialized_decimal(suggested_buy),
            "suggested_sell_ratio": _serialized_decimal(evaluation.suggested_sell_ratio),
        }

    @staticmethod
    def _store_report(
        session: Session,
        strategy: StrategyDefinition,
        instrument: Instrument,
        report_date: date,
        strategy_version: str,
        strategy_config_hash: str,
        source_data_hash: str,
        markdown: str,
    ) -> ResearchReport:
        identity = {
            "instrument_id": instrument.id,
            "report_date": report_date,
            "report_type": "strategy_research",
            "strategy_id": strategy.id,
            "strategy_version": strategy_version,
            "strategy_config_hash": strategy_config_hash,
            "source_data_hash": source_data_hash,
        }
        title = (
            f"{instrument.code} {report_date.isoformat()} 规则策略研究报告 "
            f"{strategy_config_hash[:8]}-{source_data_hash[:8]}"
        )
        insert = sqlite_insert(ResearchReport).values(
            **identity,
            title=title,
            content=markdown,
            status="generated",
        )
        session.execute(
            insert.on_conflict_do_nothing(
                index_elements=(
                    "instrument_id",
                    "report_date",
                    "report_type",
                    "strategy_id",
                    "strategy_version",
                    "strategy_config_hash",
                    "source_data_hash",
                )
            )
        )
        report = session.scalar(select(ResearchReport).filter_by(**identity))
        if report is None:  # Defensive: SQLite conflict handling is atomic.
            raise RuntimeError("Could not persist deterministic research report")
        return report

    def list_signals(
        self,
        *,
        etf_code: str | None = None,
        as_of_date: date | None = None,
    ) -> list[StrategySignal]:
        with self.session_factory() as session:
            statement = select(StrategySignal).join(Instrument).order_by(
                StrategySignal.as_of_date, StrategySignal.id
            )
            if etf_code is not None:
                statement = statement.where(Instrument.code == etf_code)
            if as_of_date is not None:
                statement = statement.where(StrategySignal.as_of_date == as_of_date)
            return session.scalars(statement).all()

    def list_reports(
        self,
        *,
        etf_code: str | None = None,
        report_date: date | None = None,
    ) -> list[ResearchReport]:
        with self.session_factory() as session:
            statement = select(ResearchReport).join(Instrument).where(
                ResearchReport.report_type == "strategy_research"
            ).order_by(ResearchReport.report_date, ResearchReport.id)
            if etf_code is not None:
                statement = statement.where(Instrument.code == etf_code)
            if report_date is not None:
                statement = statement.where(ResearchReport.report_date == report_date)
            return session.scalars(statement).all()

    def export_csv_rows(
        self,
        *,
        etf_code: str | None = None,
        as_of_date: date | None = None,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for signal in self.list_signals(etf_code=etf_code, as_of_date=as_of_date):
            with self.session_factory() as session:
                instrument_code = session.scalar(
                    select(Instrument.code).where(Instrument.id == signal.instrument_id)
                )
            rows.append(
                {
                    "as_of_date": signal.as_of_date.isoformat(),
                    "instrument_code": instrument_code or "",
                    "recommendation": signal.recommendation_type,
                    "score": _csv_decimal(signal.score),
                    "confidence": _csv_decimal(signal.confidence),
                    "strategy_version": signal.strategy_version,
                }
            )
        return rows

    def export_report_csv_rows(
        self,
        *,
        etf_code: str | None = None,
        report_date: date | None = None,
    ) -> list[dict[str, str]]:
        """Return persisted fixed-template report metadata as stable CSV-ready rows."""
        rows: list[dict[str, str]] = []
        for report in self.list_reports(etf_code=etf_code, report_date=report_date):
            with self.session_factory() as session:
                instrument_code = session.scalar(
                    select(Instrument.code).where(Instrument.id == report.instrument_id)
                )
            rows.append(
                {
                    "report_date": report.report_date.isoformat(),
                    "instrument_code": instrument_code or "",
                    "report_type": report.report_type,
                    "strategy_version": report.strategy_version or "",
                    "strategy_config_hash": report.strategy_config_hash or "",
                    "source_data_hash": report.source_data_hash or "",
                    "status": report.status,
                }
            )
        return rows
