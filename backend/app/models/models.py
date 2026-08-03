"""SQLAlchemy models for the local ETF Investment Research Lab database."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class FixedPointDecimal(TypeDecorator[Decimal]):
    """Persist Decimal values as scaled SQLite INTEGERs without binary rounding.

    Query code must not use raw SQL arithmetic or ``func.avg`` on these
    columns: their values are scaled integers. Use the explicitly named
    helpers in ``backend.app.database.fixed_point`` for multiplication,
    division, and averages.
    """

    impl = Integer
    cache_ok = True

    def __init__(self, scale: int = 8) -> None:
        super().__init__()
        self.scale = scale
        self._factor = Decimal(10) ** scale

    def to_storage(self, value: Decimal | int | float | str) -> int:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
        scaled_value = decimal_value * self._factor
        integral_value = scaled_value.to_integral_value()
        if scaled_value != integral_value:
            raise ValueError(
                f"{decimal_value} cannot be represented exactly with {self.scale} decimal places"
            )
        storage_value = int(integral_value)
        if not -(2**63) <= storage_value < 2**63:
            raise ValueError(f"{decimal_value} exceeds SQLite INTEGER storage range")
        return storage_value

    def process_bind_param(
        self, value: Decimal | int | float | str | None, dialect: Any
    ) -> int | None:
        return None if value is None else self.to_storage(value)

    def process_result_value(self, value: int | None, dialect: Any) -> Decimal | None:
        return None if value is None else Decimal(value) / self._factor


MONEY = FixedPointDecimal(scale=8)
PERCENTAGE = FixedPointDecimal(scale=8)


def utc_now() -> datetime:
    """Provide timezone-aware UTC timestamps for all persistent records."""
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator[datetime]):
    """Keep timestamps timezone-aware even when SQLite returns naive values."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all local-only persistence models."""


class TimestampMixin:
    """Shared timezone-aware creation and update timestamps."""

    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, onupdate=utc_now, nullable=False
    )


class Instrument(TimestampMixin, Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(64), default="etf", nullable=False)
    currency: Mapped[str] = mapped_column(String(8), default="CNY", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    extra_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    market_prices: Mapped[list[MarketPrice]] = relationship(back_populates="instrument")
    valuation_records: Mapped[list[ValuationRecord]] = relationship(back_populates="instrument")
    indicator_records: Mapped[list[IndicatorRecord]] = relationship(back_populates="instrument")
    investment_plans: Mapped[list[InvestmentPlan]] = relationship(back_populates="instrument")
    strategy_signals: Mapped[list[StrategySignal]] = relationship(back_populates="instrument")
    simulation_transactions: Mapped[list[SimulationTransaction]] = relationship(back_populates="instrument")
    simulation_positions: Mapped[list[SimulationPosition]] = relationship(back_populates="instrument")
    real_transactions: Mapped[list[RealTransaction]] = relationship(back_populates="instrument")
    real_positions: Mapped[list[RealPosition]] = relationship(back_populates="instrument")
    research_reports: Mapped[list[ResearchReport]] = relationship(back_populates="instrument")
    decision_logs: Mapped[list[DecisionLog]] = relationship(back_populates="instrument")
    forecast_runs: Mapped[list[ForecastRun]] = relationship(back_populates="instrument")
    data_update_logs: Mapped[list[DataUpdateLog]] = relationship(back_populates="instrument")
    investment_calendar_entries: Mapped[list[InvestmentCalendarEntry]] = relationship(
        back_populates="instrument"
    )


class MarketPrice(TimestampMixin, Base):
    __tablename__ = "market_prices"
    __table_args__ = (
        UniqueConstraint("instrument_id", "trade_date", "timeframe", name="uq_market_price_period"),
        CheckConstraint("timeframe IN ('daily', 'weekly', 'monthly')", name="ck_market_price_timeframe"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    open_price: Mapped[Decimal | None] = mapped_column(MONEY)
    high_price: Mapped[Decimal | None] = mapped_column(MONEY)
    low_price: Mapped[Decimal | None] = mapped_column(MONEY)
    close_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    adjusted_close_price: Mapped[Decimal | None] = mapped_column(MONEY)
    volume: Mapped[Decimal | None] = mapped_column(MONEY)
    volume_multiplier: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    turnover: Mapped[Decimal | None] = mapped_column(MONEY)
    source: Mapped[str | None] = mapped_column(String(64))
    volume_source: Mapped[str | None] = mapped_column(String(64))

    instrument: Mapped[Instrument] = relationship(back_populates="market_prices")


class ValuationRecord(TimestampMixin, Base):
    __tablename__ = "valuation_records"
    __table_args__ = (UniqueConstraint("instrument_id", "valuation_date", name="uq_valuation_record_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    valuation_date: Mapped[date] = mapped_column(Date, nullable=False)
    pe_ratio: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    pb_ratio: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    dividend_yield: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    valuation_percentile: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    fair_value: Mapped[Decimal | None] = mapped_column(MONEY)
    raw_values: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    instrument: Mapped[Instrument] = relationship(back_populates="valuation_records")


class IndicatorRecord(TimestampMixin, Base):
    __tablename__ = "indicator_records"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "indicator_date", "timeframe", name="uq_indicator_record"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    indicator_date: Mapped[date] = mapped_column(Date, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), default="daily", nullable=False)
    indicator_name: Mapped[str] = mapped_column(String(64), nullable=False)
    indicator_value: Mapped[Decimal | None] = mapped_column(MONEY)
    indicator_values: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    instrument: Mapped[Instrument] = relationship(back_populates="indicator_records")


class InvestmentPlan(TimestampMixin, Base):
    __tablename__ = "investment_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    frequency: Mapped[str] = mapped_column(String(16), default="weekly", nullable=False)
    amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    weekday: Mapped[int | None] = mapped_column(Integer)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    rule_parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    instrument: Mapped[Instrument] = relationship(back_populates="investment_plans")
    simulation_transactions: Mapped[list[SimulationTransaction]] = relationship(back_populates="plan")


class StrategyDefinition(TimestampMixin, Base):
    __tablename__ = "strategy_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    strategy_type: Mapped[str] = mapped_column(String(64), default="rule", nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    version: Mapped[str] = mapped_column(String(32), default="1.0", nullable=False)

    signals: Mapped[list[StrategySignal]] = relationship(back_populates="strategy")
    research_reports: Mapped[list[ResearchReport]] = relationship(back_populates="strategy")
    simulation_accounts: Mapped[list[SimulationAccount]] = relationship(back_populates="strategy")
    decision_logs: Mapped[list[DecisionLog]] = relationship(back_populates="strategy")


class StrategySignal(TimestampMixin, Base):
    __tablename__ = "strategy_signals"
    __table_args__ = (
        CheckConstraint(
            "recommendation_type IN ('INCREASE', 'NORMAL', 'REDUCE', 'PAUSE', 'HOLD', 'SELL_PARTIAL')",
            name="ck_strategy_signal_recommendation",
        ),
        UniqueConstraint(
            "strategy_id",
            "instrument_id",
            "as_of_date",
            "strategy_version",
            "strategy_config_hash",
            "source_data_hash",
            name="uq_strategy_signal_snapshot",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategy_definitions.id"), nullable=False, index=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    signal_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    strategy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    strategy_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    recommendation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    target_allocation: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    confidence: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    rationale: Mapped[str | None] = mapped_column(Text)
    signal_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    strategy: Mapped[StrategyDefinition] = relationship(back_populates="signals")
    instrument: Mapped[Instrument] = relationship(back_populates="strategy_signals")
    simulation_transactions: Mapped[list[SimulationTransaction]] = relationship(back_populates="signal")


class SimulationAccount(TimestampMixin, Base):
    __tablename__ = "simulation_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_definitions.id"), index=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    initial_cash: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    portfolio_value: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    strategy: Mapped[StrategyDefinition | None] = relationship(back_populates="simulation_accounts")
    transactions: Mapped[list[SimulationTransaction]] = relationship(back_populates="account")
    positions: Mapped[list[SimulationPosition]] = relationship(back_populates="account")
    daily_snapshots: Mapped[list[SimulationDailySnapshot]] = relationship(back_populates="account")


class SimulationTransaction(TimestampMixin, Base):
    __tablename__ = "simulation_transactions"
    __table_args__ = (
        UniqueConstraint(
            "account_id", "plan_id", "execution_date", "source_identity",
            name="uq_simulation_transaction_week_source",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("simulation_accounts.id"), nullable=False, index=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    plan_id: Mapped[int | None] = mapped_column(ForeignKey("investment_plans.id"), index=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_signals.id"), index=True)
    transaction_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    execution_date: Mapped[date | None] = mapped_column(Date, index=True)
    source_identity: Mapped[str] = mapped_column(String(128), default="manual", nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    fee: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    requested_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    executed_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    commission: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    stamp_duty: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    transfer_fee: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    other_fee: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    commission_rate: Mapped[Decimal] = mapped_column(PERCENTAGE, default=Decimal("0"), nullable=False)
    stamp_duty_rate: Mapped[Decimal] = mapped_column(PERCENTAGE, default=Decimal("0"), nullable=False)
    transfer_fee_rate: Mapped[Decimal] = mapped_column(PERCENTAGE, default=Decimal("0"), nullable=False)
    other_fee_rate: Mapped[Decimal] = mapped_column(PERCENTAGE, default=Decimal("0"), nullable=False)
    minimum_commission: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    minimum_commission_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    cash_after: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    is_theoretical: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    price_date: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)

    account: Mapped[SimulationAccount] = relationship(back_populates="transactions")
    instrument: Mapped[Instrument] = relationship(back_populates="simulation_transactions")
    plan: Mapped[InvestmentPlan | None] = relationship(back_populates="simulation_transactions")
    signal: Mapped[StrategySignal | None] = relationship(back_populates="simulation_transactions")


class SimulationPosition(TimestampMixin, Base):
    __tablename__ = "simulation_positions"
    __table_args__ = (UniqueConstraint("account_id", "instrument_id", name="uq_simulation_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("simulation_accounts.id"), nullable=False, index=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    quantity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    average_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    total_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    market_value: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)

    account: Mapped[SimulationAccount] = relationship(back_populates="positions")
    instrument: Mapped[Instrument] = relationship(back_populates="simulation_positions")


class SimulationDailySnapshot(TimestampMixin, Base):
    __tablename__ = "simulation_daily_snapshots"
    __table_args__ = (UniqueConstraint("account_id", "snapshot_date", name="uq_simulation_daily_snapshot"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("simulation_accounts.id"), nullable=False, index=True)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    market_value: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    total_assets: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    external_cash_flow: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    daily_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    cumulative_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    time_weighted_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    money_weighted_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    annualized_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    max_drawdown: Mapped[Decimal | None] = mapped_column(PERCENTAGE)

    account: Mapped[SimulationAccount] = relationship(back_populates="daily_snapshots")


class RealAccount(TimestampMixin, Base):
    __tablename__ = "real_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    broker_name: Mapped[str | None] = mapped_column(String(128))
    initial_cash: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    transactions: Mapped[list[RealTransaction]] = relationship(back_populates="account")
    positions: Mapped[list[RealPosition]] = relationship(back_populates="account")
    snapshots: Mapped[list[RealAccountSnapshot]] = relationship(back_populates="account")


class RealTransaction(TimestampMixin, Base):
    __tablename__ = "real_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("real_accounts.id"), nullable=False, index=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    transaction_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    fee: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    planned_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    cash_after: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    holding_quantity_after: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    average_cost_after: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    cost_basis: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    net_proceeds: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    external_reference: Mapped[str | None] = mapped_column(String(128))
    market_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    account: Mapped[RealAccount] = relationship(back_populates="transactions")
    instrument: Mapped[Instrument] = relationship(back_populates="real_transactions")


class RealPosition(TimestampMixin, Base):
    """Current exact average-cost holdings rebuilt from the manual trade ledger."""

    __tablename__ = "real_positions"
    __table_args__ = (UniqueConstraint("account_id", "instrument_id", name="uq_real_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("real_accounts.id"), nullable=False, index=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    quantity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    average_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    total_cost: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    market_value: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    fees_paid: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)

    account: Mapped[RealAccount] = relationship(back_populates="positions")
    instrument: Mapped[Instrument] = relationship(back_populates="real_positions")


class RealAccountSnapshot(TimestampMixin, Base):
    __tablename__ = "real_account_snapshots"
    __table_args__ = (UniqueConstraint("account_id", "snapshot_date", name="uq_real_account_snapshot"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("real_accounts.id"), nullable=False, index=True)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    market_value: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    total_assets: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    total_contribution: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    external_cash_flow: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    total_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    holding_quantity: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    benchmark_total_assets: Mapped[Decimal | None] = mapped_column(MONEY)
    benchmark_total_pnl: Mapped[Decimal | None] = mapped_column(MONEY)
    benchmark_quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    fees_paid: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"), nullable=False)
    daily_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    total_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    time_weighted_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    money_weighted_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    annualized_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    max_drawdown: Mapped[Decimal | None] = mapped_column(PERCENTAGE)

    account: Mapped[RealAccount] = relationship(back_populates="snapshots")


class ResearchReport(TimestampMixin, Base):
    __tablename__ = "research_reports"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "report_date",
            "report_type",
            "strategy_id",
            "strategy_version",
            "strategy_config_hash",
            "source_data_hash",
            name="uq_research_report_strategy_snapshot",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int | None] = mapped_column(ForeignKey("instruments.id"), index=True)
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_definitions.id"), index=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32))
    strategy_config_hash: Mapped[str | None] = mapped_column(String(64))
    source_data_hash: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    report_date: Mapped[date] = mapped_column(Date, default=date.today, nullable=False)
    report_type: Mapped[str] = mapped_column(String(64), default="research", nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)

    instrument: Mapped[Instrument | None] = relationship(back_populates="research_reports")
    strategy: Mapped[StrategyDefinition | None] = relationship(back_populates="research_reports")


class DecisionLog(TimestampMixin, Base):
    __tablename__ = "decision_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int | None] = mapped_column(ForeignKey("instruments.id"), index=True)
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_definitions.id"), index=True)
    decision_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    planned_amount: Mapped[Decimal | None] = mapped_column(MONEY)
    confidence: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    rationale: Mapped[str | None] = mapped_column(Text)
    decision_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    instrument: Mapped[Instrument | None] = relationship(back_populates="decision_logs")
    strategy: Mapped[StrategyDefinition | None] = relationship(back_populates="decision_logs")


class ForecastRun(TimestampMixin, Base):
    __tablename__ = "forecast_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int | None] = mapped_column(ForeignKey("instruments.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    run_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    instrument: Mapped[Instrument | None] = relationship(back_populates="forecast_runs")
    points: Mapped[list[ForecastPoint]] = relationship(back_populates="forecast_run")


class ForecastPoint(TimestampMixin, Base):
    __tablename__ = "forecast_points"
    __table_args__ = (UniqueConstraint("forecast_run_id", "forecast_date", name="uq_forecast_point_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_run_id: Mapped[int] = mapped_column(ForeignKey("forecast_runs.id"), nullable=False, index=True)
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)
    predicted_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    lower_bound: Mapped[Decimal | None] = mapped_column(MONEY)
    upper_bound: Mapped[Decimal | None] = mapped_column(MONEY)
    confidence: Mapped[Decimal | None] = mapped_column(PERCENTAGE)

    forecast_run: Mapped[ForecastRun] = relationship(back_populates="points")


class InvestmentCalendarEntry(TimestampMixin, Base):
    __tablename__ = "investment_calendar_entries"
    __table_args__ = (
        CheckConstraint("product_type IN ('fund', 'etf')", name="ck_calendar_product_type"),
        CheckConstraint("side IN ('buy', 'sell')", name="ck_calendar_side"),
        CheckConstraint("quantity > 0", name="ck_calendar_quantity_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False, index=True)
    product_name: Mapped[str] = mapped_column(String(128), nullable=False)
    product_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    product_type: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    operation_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    actual_price: Mapped[Decimal | None] = mapped_column(MONEY)
    note: Mapped[str | None] = mapped_column(Text)

    instrument: Mapped[Instrument] = relationship(back_populates="investment_calendar_entries")


class AppSetting(TimestampMixin, Base):
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    category: Mapped[str] = mapped_column(String(64), default="general", nullable=False)
    description: Mapped[str | None] = mapped_column(Text)


class DataUpdateLog(TimestampMixin, Base):
    __tablename__ = "data_update_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int | None] = mapped_column(ForeignKey("instruments.id"), index=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str | None] = mapped_column(String(64))
    timeframe: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    records_received: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_written: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    records_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message: Mapped[str | None] = mapped_column(Text)

    instrument: Mapped[Instrument | None] = relationship(back_populates="data_update_logs")


class V2MarketDataCache(TimestampMixin, Base):
    __tablename__ = "v2_market_data_cache"
    __table_args__ = (
        UniqueConstraint("instrument_id", "cache_key", name="uq_v2_market_data_cache_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    cache_key: Mapped[str] = mapped_column(String(256), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_start_date: Mapped[date | None] = mapped_column(Date)
    requested_end_date: Mapped[date | None] = mapped_column(Date)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String(32), default="ready", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2PeriodBar(TimestampMixin, Base):
    __tablename__ = "v2_period_bars"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "timeframe",
            "period_key",
            name="uq_v2_period_bar_period",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    period_key: Mapped[str] = mapped_column(String(32), nullable=False)
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    open_price: Mapped[Decimal | None] = mapped_column(MONEY)
    high_price: Mapped[Decimal | None] = mapped_column(MONEY)
    low_price: Mapped[Decimal | None] = mapped_column(MONEY)
    close_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    adjusted_close_price: Mapped[Decimal | None] = mapped_column(MONEY)
    volume: Mapped[Decimal | None] = mapped_column(MONEY)
    source: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="complete", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2FeatureSnapshot(TimestampMixin, Base):
    __tablename__ = "v2_feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "week_key",
            "feature_set_version",
            name="uq_v2_feature_snapshot_week_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    week_key: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    feature_set_version: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="complete", nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2WeekSample(TimestampMixin, Base):
    __tablename__ = "v2_week_samples"
    __table_args__ = (
        UniqueConstraint("instrument_id", "week_key", name="uq_v2_week_sample_week"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    week_key: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    week_start: Mapped[date] = mapped_column(Date, nullable=False)
    week_end: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    target_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2AnalysisIteration(TimestampMixin, Base):
    __tablename__ = "v2_analysis_iterations"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "iteration_number",
            name="uq_v2_analysis_iteration_number",
        ),
        UniqueConstraint(
            "instrument_id",
            "iteration_number",
            "work_number",
            name="uq_v2_analysis_iteration_work",
        ),
        UniqueConstraint(
            "instrument_id",
            "week_key",
            name="uq_v2_analysis_iteration_week",
        ),
        ForeignKeyConstraint(
            ("instrument_id", "week_key"),
            ("v2_week_samples.instrument_id", "v2_week_samples.week_key"),
            name="fk_v2_analysis_iteration_week_sample",
        ),
        CheckConstraint("iteration_number > 0", name="ck_v2_analysis_iteration_number_positive"),
        CheckConstraint("work_number > 0", name="ck_v2_analysis_iteration_work_number_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    week_key: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    work_number: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2IterationLabel(TimestampMixin, Base):
    __tablename__ = "v2_iteration_labels"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "week_key",
            name="uq_v2_iteration_label_week",
        ),
        ForeignKeyConstraint(
            ("instrument_id", "week_key"),
            (
                "v2_analysis_iterations.instrument_id",
                "v2_analysis_iterations.week_key",
            ),
            name="fk_v2_iteration_label_iteration",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    week_key: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    label_date: Mapped[date | None] = mapped_column(Date)
    maturity_date: Mapped[date | None] = mapped_column(Date)
    label_value: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    label: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2ModelVersion(TimestampMixin, Base):
    __tablename__ = "v2_model_versions"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "model_number",
            name="uq_v2_model_version_number",
        ),
        CheckConstraint("model_number > 0", name="ck_v2_model_version_number_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    model_number: Mapped[int] = mapped_column(Integer, nullable=False)
    trained_through_week_key: Mapped[str | None] = mapped_column(String(16))
    training_started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    training_completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2AnalysisTask(TimestampMixin, Base):
    __tablename__ = "v2_analysis_tasks"
    __table_args__ = (
        UniqueConstraint("task_key", name="uq_v2_analysis_task_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    task_key: Mapped[str] = mapped_column(String(128), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    week_key: Mapped[str | None] = mapped_column(String(16), index=True)
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    worker_token: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    result_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2PositionEvent(TimestampMixin, Base):
    __tablename__ = "v2_position_events"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "operation_date",
            "sequence",
            name="uq_v2_position_event_sequence",
        ),
        CheckConstraint(
            "direction IN ('increase', 'decrease')",
            name="ck_v2_position_event_direction",
        ),
        CheckConstraint(
            "change_percent >= 500000000 "
            "AND change_percent <= 10000000000 "
            "AND change_percent % 500000000 = 0",
            name="ck_v2_position_event_change_percent",
        ),
        CheckConstraint("sequence > 0", name="ck_v2_position_event_sequence_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    operation_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    change_percent: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="recorded", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2PositionSnapshot(TimestampMixin, Base):
    __tablename__ = "v2_position_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "position_event_id",
            name="uq_v2_position_snapshot_event",
        ),
        CheckConstraint(
            "position_percent >= 0 AND position_percent <= 10000000000",
            name="ck_v2_position_snapshot_percent",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_event_id: Mapped[int] = mapped_column(
        ForeignKey("v2_position_events.id"), nullable=False, index=True
    )
    position_percent: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="current", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V2AdviceHistory(TimestampMixin, Base):
    __tablename__ = "v2_advice_history"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "model_number",
            "work_number",
            "iteration_number",
            "advice_generation",
            name="uq_v2_advice_history_provenance_generation",
        ),
        UniqueConstraint(
            "instrument_id",
            "model_number",
            "work_number",
            "iteration_number",
            "generation_key",
            name="uq_v2_advice_history_generation_key",
        ),
        ForeignKeyConstraint(
            ("instrument_id", "model_number"),
            ("v2_model_versions.instrument_id", "v2_model_versions.model_number"),
            name="fk_v2_advice_history_model",
        ),
        ForeignKeyConstraint(
            ("instrument_id", "iteration_number", "work_number"),
            (
                "v2_analysis_iterations.instrument_id",
                "v2_analysis_iterations.iteration_number",
                "v2_analysis_iterations.work_number",
            ),
            name="fk_v2_advice_history_iteration",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    model_number: Mapped[int] = mapped_column(Integer, nullable=False)
    work_number: Mapped[int] = mapped_column(Integer, nullable=False)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    advice_generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    generation_key: Mapped[str] = mapped_column(
        String(128), nullable=False, default="default"
    )
    advice_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="published", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    audit_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V31AnalysisRun(Base):
    """One auditable V3.1 button execution and its stage log."""

    __tablename__ = "v31_analysis_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    price_data_as_of: Mapped[date | None] = mapped_column(Date)
    valuation_data_as_of: Mapped[date | None] = mapped_column(Date)
    weekly_data_as_of: Mapped[date | None] = mapped_column(Date)
    data_gate_status: Mapped[str] = mapped_column(String(32), nullable=False)
    degraded_reasons_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(String(64))
    data_snapshot_id: Mapped[str | None] = mapped_column(String(96))
    stages_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V31ModelVersion(Base):
    """Independent weekly/daily Champion or Challenger model artifact."""

    __tablename__ = "v31_model_versions"
    __table_args__ = (
        UniqueConstraint("market", "model_type", "version", name="uq_v31_model_version"),
    )

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    methodology_version: Mapped[str] = mapped_column(String(64), nullable=False)
    training_end_date: Mapped[date | None] = mapped_column(Date)
    validation_start_date: Mapped[date | None] = mapped_column(Date)
    validation_end_date: Mapped[date | None] = mapped_column(Date)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V31WeeklyForecast(Base):
    __tablename__ = "v31_weekly_forecasts"
    __table_args__ = (
        UniqueConstraint("market", "forecast_date", "model_version", name="uq_v31_forecast"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    data_snapshot_id: Mapped[str] = mapped_column(String(96), nullable=False)
    weekly_direction: Mapped[str] = mapped_column(String(16), nullable=False)
    weekly_confidence: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    base_target_position: Mapped[int] = mapped_column(Integer, nullable=False)
    p10_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    p50_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    p90_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    expected_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    up_probability: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    sideways_probability: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    down_probability: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    expected_max_drawdown: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    high_week_range: Mapped[str] = mapped_column(String(32), nullable=False)
    low_week_range: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V31DailyCorrection(Base):
    __tablename__ = "v31_daily_corrections"
    __table_args__ = (UniqueConstraint("weekly_forecast_id", name="uq_v31_daily_forecast"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    weekly_forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v31_weekly_forecasts.id"), nullable=False, index=True
    )
    correction_date: Mapped[date] = mapped_column(Date, nullable=False)
    daily_state: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_adjustment: Mapped[int] = mapped_column(Integer, nullable=False)
    position_adjustment: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_window_start: Mapped[date | None] = mapped_column(Date)
    execution_window_end: Mapped[date | None] = mapped_column(Date)
    trigger_conditions_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    invalidation_conditions_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V31ModelEvaluation(Base):
    __tablename__ = "v31_model_evaluations"
    __table_args__ = (UniqueConstraint("forecast_id", name="uq_v31_evaluation_forecast"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v31_weekly_forecasts.id"), nullable=False, index=True
    )
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    path_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    terminal_return_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    direction_score: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    quantile_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    interval_coverage: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    high_date_error: Mapped[int | None] = mapped_column(Integer)
    low_date_error: Mapped[int | None] = mapped_column(Integer)
    turnover: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    net_strategy_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V31InstrumentMarketMapping(TimestampMixin, Base):
    __tablename__ = "v31_instrument_market_mappings"
    __table_args__ = (
        UniqueConstraint("instrument_id", "market_model", name="uq_v31_instrument_market"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False)
    market_model: Mapped[str] = mapped_column(String(64), nullable=False)
    exposure_weight: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    benchmark: Mapped[str] = mapped_column(String(32), nullable=False)
    correlation: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    tracking_error: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    mapping_confidence: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V32TrainingRun(Base):
    """One auditable V3.2 bootstrap or weekly incremental training task."""

    __tablename__ = "v32_training_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    run_type: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_through_week: Mapped[str | None] = mapped_column(String(16))
    source_parent_iteration: Mapped[int | None] = mapped_column(Integer)
    created_iteration_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    stages_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V32TrainingIteration(Base):
    """A single progressive V3.2 I-state with an explicit parent."""

    __tablename__ = "v32_training_iterations"
    __table_args__ = (
        UniqueConstraint("market", "iteration_number", name="uq_v32_iteration_number"),
        UniqueConstraint("market", "week_key", name="uq_v32_iteration_week"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v32_training_runs.id"), nullable=False, index=True
    )
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_iteration_number: Mapped[int | None] = mapped_column(Integer)
    week_key: Mapped[str] = mapped_column(String(16), nullable=False)
    cutoff_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    champion_weekly_version: Mapped[str] = mapped_column(String(64), nullable=False)
    champion_daily_version: Mapped[str] = mapped_column(String(64), nullable=False)
    challenger_version: Mapped[str | None] = mapped_column(String(64))
    promoted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    data_snapshot_id: Mapped[str] = mapped_column(String(96), nullable=False)
    input_features_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    forecast_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    evaluation_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    optimizer_state_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    composite_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    path_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    terminal_return_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    direction_score: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    interval_coverage: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    high_week_error: Mapped[int | None] = mapped_column(Integer)
    low_week_error: Mapped[int | None] = mapped_column(Integer)
    completed_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V32TrainingCheckpoint(Base):
    """Latest recoverable checkpoint for one training run and market."""

    __tablename__ = "v32_training_checkpoints"
    __table_args__ = (
        UniqueConstraint("training_run_id", "market", name="uq_v32_checkpoint_run_market"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v32_training_runs.id"), nullable=False, index=True
    )
    market: Mapped[str] = mapped_column(String(16), nullable=False)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    week_key: Mapped[str] = mapped_column(String(16), nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V32ModelVersion(Base):
    """Independent weekly or daily V3.2 Champion/Challenger artifact."""

    __tablename__ = "v32_model_versions"
    __table_args__ = (
        UniqueConstraint("market", "model_type", "version", name="uq_v32_model_version"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_version: Mapped[str | None] = mapped_column(String(64))
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    methodology_version: Mapped[str] = mapped_column(String(64), nullable=False)
    trained_through_date: Mapped[date] = mapped_column(Date, nullable=False)
    validation_start_date: Mapped[date | None] = mapped_column(Date)
    validation_end_date: Mapped[date | None] = mapped_column(Date)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V32OptimizerState(Base):
    """Durable state inherited by the next weekly V3.2 training iteration."""

    __tablename__ = "v32_optimizer_states"
    __table_args__ = (UniqueConstraint("market", name="uq_v32_optimizer_market"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    last_training_week_key: Mapped[str] = mapped_column(String(16), nullable=False)
    champion_weekly_version: Mapped[str] = mapped_column(String(64), nullable=False)
    champion_daily_version: Mapped[str] = mapped_column(String(64), nullable=False)
    weekly_parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    daily_parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    optimizer_memory_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    last_successful_training_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    next_training_eligible_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V32DataSnapshot(Base):
    """Immutable identity for the leakage-safe inputs of a training or analysis run."""

    __tablename__ = "v32_data_snapshots"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    cutoff_date: Mapped[date] = mapped_column(Date, nullable=False)
    daily_data_as_of: Mapped[date] = mapped_column(Date, nullable=False)
    weekly_data_as_of: Mapped[date] = mapped_column(Date, nullable=False)
    valuation_data_as_of: Mapped[date | None] = mapped_column(Date)
    source_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V32WeeklyForecast(Base):
    """A persisted V3.2 13-week probability path."""

    __tablename__ = "v32_weekly_forecasts"
    __table_args__ = (
        UniqueConstraint("market", "forecast_date", "model_version", name="uq_v32_forecast"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    training_iteration_id: Mapped[int | None] = mapped_column(
        ForeignKey("v32_training_iterations.id"), index=True
    )
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    data_snapshot_id: Mapped[str] = mapped_column(String(96), nullable=False)
    weekly_direction: Mapped[str] = mapped_column(String(16), nullable=False)
    weekly_confidence: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    base_target_position: Mapped[int] = mapped_column(Integer, nullable=False)
    p10_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    p50_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    p90_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    expected_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    up_probability: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    sideways_probability: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    down_probability: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    expected_max_drawdown: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    high_week_range: Mapped[str] = mapped_column(String(32), nullable=False)
    low_week_range: Mapped[str] = mapped_column(String(32), nullable=False)
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V32DailyCorrection(Base):
    __tablename__ = "v32_daily_corrections"
    __table_args__ = (UniqueConstraint("weekly_forecast_id", name="uq_v32_daily_forecast"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    weekly_forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v32_weekly_forecasts.id"), nullable=False, index=True
    )
    correction_date: Mapped[date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    daily_state: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_adjustment: Mapped[int] = mapped_column(Integer, nullable=False)
    position_adjustment: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_window_start: Mapped[date | None] = mapped_column(Date)
    execution_window_end: Mapped[date | None] = mapped_column(Date)
    trigger_conditions_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    invalidation_conditions_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class V32ModelEvaluation(Base):
    __tablename__ = "v32_model_evaluations"
    __table_args__ = (UniqueConstraint("forecast_id", name="uq_v32_evaluation_forecast"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v32_weekly_forecasts.id"), nullable=False, index=True
    )
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    composite_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    path_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    terminal_return_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    direction_score: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    quantile_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    interval_coverage: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    high_week_error: Mapped[int | None] = mapped_column(Integer)
    low_week_error: Mapped[int | None] = mapped_column(Integer)
    turnover: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    net_strategy_return: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V32AnalysisRun(Base):
    """A pure-inference V3.2 current-market analysis run."""

    __tablename__ = "v32_analysis_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    market: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))
    data_snapshot_id: Mapped[str | None] = mapped_column(String(96))
    price_data_as_of: Mapped[date | None] = mapped_column(Date)
    weekly_data_as_of: Mapped[date | None] = mapped_column(Date)
    valuation_data_as_of: Mapped[date | None] = mapped_column(Date)
    data_gate_status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    stages_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class V32AppState(Base):
    __tablename__ = "v32_app_state"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33InstrumentRole(Base):
    """Point-in-time role of an instrument in one independent V3.3 market model.

    V3.3 deliberately separates the instrument that can be traded from any
    benchmark used as an explanatory input.  In particular, ``159941`` is the
    tradable CNY target and ``NDX`` is a non-tradable benchmark/audit series.
    """

    __tablename__ = "v33_instrument_roles"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "model_market", "role", name="uq_v33_instrument_role"
        ),
        CheckConstraint(
            "role IN ('tradable', 'benchmark', 'legacy_audit')",
            name="ck_v33_instrument_role",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(24), nullable=False)
    benchmark_instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id"), index=True
    )
    currency: Mapped[str] = mapped_column(String(8), nullable=False)
    exchange_timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    vintage: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33RawIngestion(Base):
    """Append-only record of one external/local source request and its payload."""

    __tablename__ = "v33_raw_ingestions"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id"), index=True
    )
    dataset: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    requested_start_date: Mapped[date | None] = mapped_column(Date)
    requested_end_date: Mapped[date | None] = mapped_column(Date)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    vintage: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_payload_json: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(JSON)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    records_received: Mapped[int] = mapped_column(Integer, nullable=False)
    records_written: Mapped[int] = mapped_column(Integer, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33SourceRelease(Base):
    """Publication/vintage identity used by point-in-time feature joins."""

    __tablename__ = "v33_source_releases"
    __table_args__ = (
        UniqueConstraint(
            "dataset",
            "series_code",
            "effective_date",
            "source",
            "raw_payload_hash",
            name="uq_v33_source_release",
        ),
        CheckConstraint(
            "published_at <= available_at",
            name="ck_v33_source_release_availability",
        ),
    )

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    ingestion_id: Mapped[str | None] = mapped_column(
        ForeignKey("v33_raw_ingestions.id"), index=True
    )
    instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id"), index=True
    )
    dataset: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    series_code: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    observation_period_start: Mapped[date | None] = mapped_column(Date)
    observation_period_end: Mapped[date | None] = mapped_column(Date)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    published_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)
    vintage: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33MarketBar(Base):
    """Append-only PIT OHLCV revision for a tradable or benchmark instrument."""

    __tablename__ = "v33_market_bars"
    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "trade_date",
            "timeframe",
            "source",
            "raw_payload_hash",
            name="uq_v33_market_bar_revision",
        ),
        CheckConstraint(
            "timeframe IN ('daily', 'weekly', 'monthly')",
            name="ck_v33_market_bar_timeframe",
        ),
        CheckConstraint(
            "open_price > 0 AND high_price > 0 AND low_price > 0 AND close_price > 0",
            name="ck_v33_market_bar_positive_ohlc",
        ),
        CheckConstraint(
            "high_price >= open_price AND high_price >= close_price "
            "AND high_price >= low_price AND low_price <= open_price "
            "AND low_price <= close_price",
            name="ck_v33_market_bar_ohlc_range",
        ),
        CheckConstraint(
            "published_at <= available_at AND available_at <= cutoff_at",
            name="ck_v33_market_bar_availability",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("v33_raw_ingestions.id"), nullable=False, index=True
    )
    source_release_id: Mapped[str] = mapped_column(
        ForeignKey("v33_source_releases.id"), nullable=False, index=True
    )
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    open_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    high_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    low_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    close_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    adjusted_close_price: Mapped[Decimal | None] = mapped_column(MONEY)
    volume: Mapped[Decimal | None] = mapped_column(MONEY)
    volume_multiplier: Mapped[int] = mapped_column(Integer, nullable=False)
    turnover: Mapped[Decimal | None] = mapped_column(MONEY)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    published_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    cutoff_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    vintage: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33PointInTimeObservation(Base):
    """One valuation, earnings, flow, FX, rate, or macro observation revision."""

    __tablename__ = "v33_point_in_time_observations"
    __table_args__ = (
        UniqueConstraint(
            "series_code",
            "effective_date",
            "available_at",
            "vintage",
            "source",
            name="uq_v33_pit_observation_revision",
        ),
        CheckConstraint(
            "numeric_value IS NOT NULL OR text_value IS NOT NULL OR payload_json IS NOT NULL",
            name="ck_v33_pit_observation_has_value",
        ),
        CheckConstraint(
            "published_at <= available_at AND available_at <= cutoff_at",
            name="ck_v33_pit_observation_availability",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id"), index=True
    )
    ingestion_id: Mapped[str | None] = mapped_column(
        ForeignKey("v33_raw_ingestions.id"), index=True
    )
    source_release_id: Mapped[str] = mapped_column(
        ForeignKey("v33_source_releases.id"), nullable=False, index=True
    )
    dataset: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    series_code: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    observation_period: Mapped[str | None] = mapped_column(String(32))
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    published_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    available_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    cutoff_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    vintage: Mapped[str] = mapped_column(String(64), nullable=False)
    numeric_value: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    text_value: Mapped[str | None] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33QualityAssessment(Base):
    """Persist every hard/soft data gate result without fabricating values."""

    __tablename__ = "v33_quality_assessments"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    ingestion_id: Mapped[str | None] = mapped_column(
        ForeignKey("v33_raw_ingestions.id"), index=True
    )
    instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id"), index=True
    )
    dataset: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    gate_version: Mapped[str] = mapped_column(String(64), nullable=False)
    assessed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    coverage_start_date: Mapped[date | None] = mapped_column(Date)
    coverage_end_date: Mapped[date | None] = mapped_column(Date)
    records_received: Mapped[int] = mapped_column(Integer, nullable=False)
    records_valid: Mapped[int] = mapped_column(Integer, nullable=False)
    records_rejected: Mapped[int] = mapped_column(Integer, nullable=False)
    required_fields_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    missing_fields_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    issues_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    warnings_json: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    assessed_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33FeatureSnapshot(Base):
    """Immutable leakage-safe feature vector/sequence identity at one cutoff."""

    __tablename__ = "v33_feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "model_market", "cutoff_date", "feature_version", name="uq_v33_feature_snapshot"
        ),
        CheckConstraint(
            "source_max_available_at <= cutoff_available_at",
            name="ck_v33_feature_snapshot_no_future",
        ),
    )

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    cutoff_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    cutoff_available_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    daily_data_as_of: Mapped[date] = mapped_column(Date, nullable=False)
    weekly_data_as_of: Mapped[date] = mapped_column(Date, nullable=False)
    source_max_available_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    daily_session_count: Mapped[int] = mapped_column(Integer, nullable=False)
    weekly_bar_count: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    missing_features_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    source_release_ids_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    leakage_audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, nullable=False)


class V33TrainingRun(Base):
    """One V3.3 bootstrap or at-most-once weekly incremental task."""

    __tablename__ = "v33_training_runs"
    __table_args__ = (
        CheckConstraint("horizon_weeks = 20", name="ck_v33_training_horizon"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    run_type: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    requested_through_week: Mapped[str | None] = mapped_column(String(16))
    source_parent_iteration: Mapped[int | None] = mapped_column(Integer)
    created_iteration_count: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    stages_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class V33TrainingIteration(Base):
    """Progressive V3.3 iteration inheriting the prior optimizer state."""

    __tablename__ = "v33_training_iterations"
    __table_args__ = (
        UniqueConstraint(
            "model_market", "iteration_number", name="uq_v33_iteration_number"
        ),
        UniqueConstraint("model_market", "week_key", name="uq_v33_iteration_week"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v33_training_runs.id"), nullable=False, index=True
    )
    feature_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("v33_feature_snapshots.id"), nullable=False, index=True
    )
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_iteration_number: Mapped[int | None] = mapped_column(Integer)
    week_key: Mapped[str] = mapped_column(String(16), nullable=False)
    cutoff_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    champion_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    challenger_model_version: Mapped[str | None] = mapped_column(String(64))
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    forecast_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evaluation_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    optimizer_state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    composite_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    wis_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    price_turn_error_days: Mapped[int | None] = mapped_column(Integer)
    dif_zero_error_days: Mapped[int | None] = mapped_column(Integer)
    completed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class V33TrainingCheckpoint(Base):
    __tablename__ = "v33_training_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id", "model_market", name="uq_v33_checkpoint_run_market"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v33_training_runs.id"), nullable=False, index=True
    )
    model_market: Mapped[str] = mapped_column(String(32), nullable=False)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    week_key: Mapped[str] = mapped_column(String(16), nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V33ModelVersion(Base):
    """Independent weekly, daily-turning-point, regime, or ensemble artifact."""

    __tablename__ = "v33_model_versions"
    __table_args__ = (
        UniqueConstraint(
            "model_market", "model_type", "version", name="uq_v33_model_version"
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(48), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_version: Mapped[str | None] = mapped_column(String(64))
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    methodology_version: Mapped[str] = mapped_column(String(64), nullable=False)
    trained_through_date: Mapped[date] = mapped_column(Date, nullable=False)
    validation_start_date: Mapped[date | None] = mapped_column(Date)
    validation_end_date: Mapped[date | None] = mapped_column(Date)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V33OptimizerState(Base):
    __tablename__ = "v33_optimizer_states"
    __table_args__ = (
        UniqueConstraint("model_market", name="uq_v33_optimizer_market"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    last_training_week_key: Mapped[str] = mapped_column(String(16), nullable=False)
    champion_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    optimizer_memory_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    last_successful_training_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    next_training_eligible_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V33Forecast(Base):
    """Saved 20-week distribution, turning-point windows, and execution target."""

    __tablename__ = "v33_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "model_market", "forecast_date", "model_version", name="uq_v33_forecast"
        ),
        CheckConstraint("horizon_weeks = 20", name="ck_v33_forecast_horizon"),
        CheckConstraint(
            "target_position IS NULL OR "
            "(target_position >= 0 AND target_position <= 100 AND target_position % 5 = 0)",
            name="ck_v33_forecast_position_grid",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    training_iteration_id: Mapped[int | None] = mapped_column(
        ForeignKey("v33_training_iterations.id"), index=True
    )
    feature_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("v33_feature_snapshots.id"), nullable=False, index=True
    )
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    weekly_direction: Mapped[str | None] = mapped_column(String(16))
    confidence: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    p10_path_json: Mapped[list[Any] | None] = mapped_column(JSON)
    p50_path_json: Mapped[list[Any] | None] = mapped_column(JSON)
    p90_path_json: Mapped[list[Any] | None] = mapped_column(JSON)
    expected_path_json: Mapped[list[Any] | None] = mapped_column(JSON)
    up_probability: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    sideways_probability: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    down_probability: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    expected_max_drawdown: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    expected_high_start_date: Mapped[date | None] = mapped_column(Date)
    expected_high_end_date: Mapped[date | None] = mapped_column(Date)
    expected_low_start_date: Mapped[date | None] = mapped_column(Date)
    expected_low_end_date: Mapped[date | None] = mapped_column(Date)
    dif_turn_start_date: Mapped[date | None] = mapped_column(Date)
    dif_turn_end_date: Mapped[date | None] = mapped_column(Date)
    price_turn_start_date: Mapped[date | None] = mapped_column(Date)
    price_turn_end_date: Mapped[date | None] = mapped_column(Date)
    target_position: Mapped[int | None] = mapped_column(Integer)
    execution_batches_json: Mapped[list[Any] | None] = mapped_column(JSON)
    qdii_decomposition_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    forecast_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V33ModelEvaluation(Base):
    """Measured out-of-sample score; pending rows keep every metric NULL."""

    __tablename__ = "v33_model_evaluations"
    __table_args__ = (
        UniqueConstraint("forecast_id", name="uq_v33_evaluation_forecast"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v33_forecasts.id"), nullable=False, index=True
    )
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    composite_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    wis_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    pinball_loss: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    p50_path_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    terminal_return_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    brier_score: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    calibration_error: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    interval_coverage: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    interval_width: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    dif_turn_error_days: Mapped[int | None] = mapped_column(Integer)
    price_turn_error_days: Mapped[int | None] = mapped_column(Integer)
    coverage_3d: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    coverage_5d: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    coverage_10d: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    turning_direction_f1: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    false_turn_alert_rate: Mapped[Decimal | None] = mapped_column(PERCENTAGE)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evaluated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class V33AnalysisRun(Base):
    """Pure-inference V3.3 run; it never mutates training state."""

    __tablename__ = "v33_analysis_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))
    feature_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("v33_feature_snapshots.id"), index=True
    )
    forecast_id: Mapped[int | None] = mapped_column(
        ForeignKey("v33_forecasts.id"), index=True
    )
    price_data_as_of: Mapped[date | None] = mapped_column(Date)
    weekly_data_as_of: Mapped[date | None] = mapped_column(Date)
    valuation_data_as_of: Mapped[date | None] = mapped_column(Date)
    macro_data_as_of: Mapped[date | None] = mapped_column(Date)
    data_gate_status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    stages_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class V34FeatureSnapshot(Base):
    """Immutable V3.4 feature/scaler input frozen at a weekly forecast anchor."""

    __tablename__ = "v34_feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "model_market", "forecast_anchor_date", "feature_version",
            name="uq_v34_feature_snapshot",
        ),
        CheckConstraint(
            "source_max_date <= forecast_anchor_date",
            name="ck_v34_feature_snapshot_no_future",
        ),
    )

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    forecast_anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    source_max_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_fit_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scaler_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    daily_sequence_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    weekly_state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_fields_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    leakage_audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V34TrainingRun(Base):
    """One V3.4 bootstrap or incremental weekly learning task."""

    __tablename__ = "v34_training_runs"
    __table_args__ = (
        CheckConstraint("horizon_weeks = 13", name="ck_v34_training_horizon"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    run_type: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    requested_through_anchor: Mapped[date | None] = mapped_column(Date)
    weekly_iteration_count: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_training_count: Mapped[int] = mapped_column(Integer, nullable=False)
    champion_promotion_count: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class V34ModelVersion(Base):
    """Versioned candidate/champion with next-anchor activation semantics."""

    __tablename__ = "v34_model_versions"
    __table_args__ = (
        UniqueConstraint("model_market", "version", name="uq_v34_model_version"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_version: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    trained_through_date: Mapped[date] = mapped_column(Date, nullable=False)
    effective_from_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scaler_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scenario_adapter_version: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_matured_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_independent_sample_count: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    promotion_gate_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V34OptimizerState(Base):
    __tablename__ = "v34_optimizer_states"
    __table_args__ = (
        UniqueConstraint("model_market", name="uq_v34_optimizer_market"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    last_anchor_date: Mapped[date] = mapped_column(Date, nullable=False)
    weekly_iteration_count: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_training_count: Mapped[int] = mapped_column(Integer, nullable=False)
    champion_promotion_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fully_matured_since_retrain: Mapped[int] = mapped_column(Integer, nullable=False)
    champion_version: Mapped[str] = mapped_column(String(64), nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V34ScenarioAdapter(Base):
    __tablename__ = "v34_scenario_adapters"
    __table_args__ = (
        UniqueConstraint("model_market", "version", name="uq_v34_scenario_adapter"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_fit_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    scenario_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V34ProbabilityCalibrator(Base):
    __tablename__ = "v34_probability_calibrators"
    __table_args__ = (
        UniqueConstraint(
            "model_market", "horizon_weeks", "version",
            name="uq_v34_probability_calibrator",
        ),
        CheckConstraint(
            "horizon_weeks IN (1, 4, 8, 13)",
            name="ck_v34_calibrator_horizon",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    fit_through_date: Mapped[date] = mapped_column(Date, nullable=False)
    raw_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_sample_count: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    calibration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V34Forecast(Base):
    """Frozen 13-week scenario forecast issued before any labels are revealed."""

    __tablename__ = "v34_forecasts"
    __table_args__ = (
        UniqueConstraint("model_market", "forecast_anchor_date", name="uq_v34_forecast_anchor"),
        CheckConstraint("horizon_weeks = 13", name="ck_v34_forecast_horizon"),
        CheckConstraint("scenario_count >= 1000", name="ck_v34_forecast_scenarios"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    feature_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("v34_feature_snapshots.id"), nullable=False, index=True
    )
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("v34_model_versions.id"), nullable=False, index=True
    )
    forecast_anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    label_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    scenario_adapter_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scenario_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario_count: Mapped[int] = mapped_column(Integer, nullable=False)
    representative_ohlcv_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    indicator_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    price_quantiles_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    direction_probabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    path_probabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    calibration_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    turning_windows_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    consistency_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    advice_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    forecast_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V34ModelEvaluation(Base):
    __tablename__ = "v34_model_evaluations"
    __table_args__ = (
        UniqueConstraint("forecast_id", name="uq_v34_evaluation_forecast"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v34_forecasts.id"), nullable=False, index=True
    )
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    realized_week_count: Mapped[int] = mapped_column(Integer, nullable=False)
    realized_labels_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evaluated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class V34TrainingIteration(Base):
    __tablename__ = "v34_training_iterations"
    __table_args__ = (
        UniqueConstraint("model_market", "anchor_date", name="uq_v34_iteration_anchor"),
        UniqueConstraint(
            "model_market", "weekly_iteration_number", name="uq_v34_iteration_number"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v34_training_runs.id"), nullable=False, index=True
    )
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v34_forecasts.id"), nullable=False, index=True
    )
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    weekly_iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    champion_before_version: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_version: Mapped[str | None] = mapped_column(String(64))
    champion_after_version: Mapped[str] = mapped_column(String(64), nullable=False)
    training_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    newly_matured_count: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_matured_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_independent_sample_count: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    validation_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    promotion_gate_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V34AnalysisRun(Base):
    """Pure-inference V3.4 analysis; training counters are never mutated."""

    __tablename__ = "v34_analysis_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    forecast_id: Mapped[int | None] = mapped_column(
        ForeignKey("v34_forecasts.id"), index=True
    )
    model_version: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_anchor_date: Mapped[date | None] = mapped_column(Date)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class V341TrainingProfile(Base):
    """Frozen executable protocol used by one V3.4.1 market chain."""

    __tablename__ = "v341_training_profiles"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "profile_hash",
            name="uq_v341_training_profile",
        ),
        CheckConstraint(
            "training_window_mode IN ('ROLLING_520W', 'EXPANDING_AVAILABLE_HISTORY')",
            name="ck_v341_training_window_mode",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    training_window_mode: Mapped[str] = mapped_column(String(40), nullable=False)
    formal_training_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_training_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_warmup_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_set_name: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(64), nullable=False)
    ordered_feature_names_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    feature_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    loss_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    random_plan_version: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold_formula_version: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    profile_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341FeatureSnapshot(Base):
    """Point-in-time feature record isolated from the frozen V3.4 store."""

    __tablename__ = "v341_feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "forecast_anchor_date", "feature_manifest_hash",
            name="uq_v341_feature_snapshot",
        ),
        CheckConstraint(
            "source_max_date <= forecast_anchor_date",
            name="ck_v341_feature_snapshot_no_future",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    forecast_anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    cutoff_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    source_max_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    daily_sequence_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    leakage_audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341TrainingRun(Base):
    __tablename__ = "v341_training_runs"
    __table_args__ = (
        CheckConstraint("horizon_weeks = 13", name="ck_v341_training_horizon"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("v341_training_profiles.id"), nullable=False, index=True
    )
    run_type: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_stage: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    requested_through_anchor: Mapped[date | None] = mapped_column(Date)
    maximum_backlog_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class V341ModelVersion(Base):
    __tablename__ = "v341_model_versions"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "version",
            name="uq_v341_model_version",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    parent_model_id: Mapped[str | None] = mapped_column(
        ForeignKey("v341_model_versions.id"), index=True
    )
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("v341_training_profiles.id"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    trained_through_date: Mapped[date] = mapped_column(Date, nullable=False)
    effective_from_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    feature_anchor_max_date: Mapped[date] = mapped_column(Date, nullable=False)
    label_observed_through_date: Mapped[date] = mapped_column(Date, nullable=False)
    raw_matured_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_independent_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    promotion_gate_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    health_status: Mapped[str] = mapped_column(String(40), nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341OptimizerState(Base):
    __tablename__ = "v341_optimizer_states"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", name="uq_v341_optimizer_market"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("v341_training_profiles.id"), nullable=False, index=True
    )
    last_anchor_date: Mapped[date] = mapped_column(Date, nullable=False)
    last_consumed_mature_anchor: Mapped[date | None] = mapped_column(Date)
    last_candidate_anchor: Mapped[date | None] = mapped_column(Date)
    last_structure_audit_anchor: Mapped[date | None] = mapped_column(Date)
    weekly_iteration_count: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_training_count: Mapped[int] = mapped_column(Integer, nullable=False)
    champion_promotion_count: Mapped[int] = mapped_column(Integer, nullable=False)
    champion_model_id: Mapped[str] = mapped_column(
        ForeignKey("v341_model_versions.id"), nullable=False, index=True
    )
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341ScenarioAdapter(Base):
    __tablename__ = "v341_scenario_adapters"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "version",
            name="uq_v341_scenario_adapter",
        ),
        CheckConstraint("scenario_count >= 1000", name="ck_v341_adapter_scenarios"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    residual_schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    random_plan_version: Mapped[str] = mapped_column(String(80), nullable=False)
    scenario_count: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341ProbabilityCalibrator(Base):
    __tablename__ = "v341_probability_calibrators"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "horizon_weeks", "version",
            name="uq_v341_probability_calibrator",
        ),
        CheckConstraint(
            "horizon_weeks IN (4, 8, 13)", name="ck_v341_calibrator_horizon"
        ),
    )

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    model_family: Mapped[str] = mapped_column(String(80), nullable=False)
    scenario_adapter_version: Mapped[str] = mapped_column(String(80), nullable=False)
    residual_schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    threshold_formula_version: Mapped[str] = mapped_column(String(80), nullable=False)
    fit_through_date: Mapped[date] = mapped_column(Date, nullable=False)
    effective_from_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    raw_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    temperature: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    calibration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341RandomPlan(Base):
    """Append-only, compressed realization of a frozen stochastic plan."""

    __tablename__ = "v341_random_plans"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "plan_hash",
            name="uq_v341_random_plan_hash",
        ),
        CheckConstraint("scenario_count >= 1000", name="ck_v341_random_plan_scenarios"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    plan_scope: Mapped[str] = mapped_column(String(32), nullable=False)
    plan_version: Mapped[str] = mapped_column(String(80), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario_count: Mapped[int] = mapped_column(Integer, nullable=False)
    residual_pool_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    residual_pool_size: Mapped[int] = mapped_column(Integer, nullable=False)
    empirical_pool_size: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_codec: Mapped[str] = mapped_column(String(32), nullable=False)
    plan_payload_zlib: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341OuterEvaluationBlock(Base):
    """Sealed outer validation block; one row means one irreversible use."""

    __tablename__ = "v341_outer_evaluation_blocks"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "block_hash",
            name="uq_v341_outer_block_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v341_training_runs.id"), nullable=False, index=True
    )
    anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    validation_start_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    validation_end_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    training_anchors_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    validation_anchors_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    block_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    consumed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341Forecast(Base):
    __tablename__ = "v341_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "forecast_anchor_date",
            name="uq_v341_forecast_anchor",
        ),
        CheckConstraint("horizon_weeks = 13", name="ck_v341_forecast_horizon"),
        CheckConstraint("scenario_count >= 1000", name="ck_v341_forecast_scenarios"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    target_instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id"), nullable=False, index=True
    )
    feature_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("v341_feature_snapshots.id"), nullable=False, index=True
    )
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("v341_model_versions.id"), nullable=False, index=True
    )
    scenario_adapter_id: Mapped[str] = mapped_column(
        ForeignKey("v341_scenario_adapters.id"), nullable=False, index=True
    )
    random_plan_id: Mapped[str] = mapped_column(
        ForeignKey("v341_random_plans.id"), nullable=False, index=True
    )
    forecast_anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    label_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    maturity_status: Mapped[str] = mapped_column(String(32), nullable=False)
    scenario_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario_count: Mapped[int] = mapped_column(Integer, nullable=False)
    random_plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    representative_ohlcv_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    indicator_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    price_quantiles_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    horizon_probabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    thresholds_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    calibrator_versions_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    residual_pool_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    path_probabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    model_reliability_score: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    reliability_components_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    health_status: Mapped[str] = mapped_column(String(40), nullable=False)
    turning_windows_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    consistency_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    forecast_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341ForecastEvaluation(Base):
    __tablename__ = "v341_forecast_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "forecast_id", "horizon_weeks", "evaluation_version",
            name="uq_v341_forecast_evaluation",
        ),
        CheckConstraint(
            "horizon_weeks IN (4, 8, 13)", name="ck_v341_evaluation_horizon"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v341_forecasts.id"), nullable=False, index=True
    )
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluation_available_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    actual_return: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    actual_class: Mapped[str] = mapped_column(String(16), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    market_data_version_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evaluation_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341ForecastCalibratorLink(Base):
    __tablename__ = "v341_forecast_calibrators"
    __table_args__ = (
        UniqueConstraint(
            "forecast_id", "horizon_weeks", name="uq_v341_forecast_calibrator_horizon"
        ),
        CheckConstraint(
            "horizon_weeks IN (4, 8, 13)", name="ck_v341_forecast_calibrator_horizon"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v341_forecasts.id"), nullable=False, index=True
    )
    calibrator_id: Mapped[str] = mapped_column(
        ForeignKey("v341_probability_calibrators.id"), nullable=False, index=True
    )
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341ResidualRecord(Base):
    __tablename__ = "v341_residual_records"
    __table_args__ = (
        UniqueConstraint("forecast_id", name="uq_v341_residual_forecast"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v341_forecasts.id"), nullable=False, index=True
    )
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    matured_at: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    model_family: Mapped[str] = mapped_column(String(80), nullable=False)
    scenario_adapter_version: Mapped[str] = mapped_column(String(80), nullable=False)
    residual_schema_version: Mapped[str] = mapped_column(String(80), nullable=False)
    return_unit: Mapped[str] = mapped_column(String(40), nullable=False)
    prediction_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    actual_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    residual_path_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    standardized_residual_json: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    source_sigma: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    compatibility_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    residual_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341CandidateTrial(Base):
    __tablename__ = "v341_candidate_trials"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id", "anchor_date", "candidate_model_id",
            name="uq_v341_candidate_trial",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v341_training_runs.id"), nullable=False, index=True
    )
    anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    champion_model_id: Mapped[str] = mapped_column(
        ForeignKey("v341_model_versions.id"), nullable=False, index=True
    )
    candidate_model_id: Mapped[str] = mapped_column(
        ForeignKey("v341_model_versions.id"), nullable=False, index=True
    )
    random_plan_id: Mapped[str] = mapped_column(
        ForeignKey("v341_random_plans.id"), nullable=False, index=True
    )
    alpha: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    candidate_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    outer_evaluation_block_id: Mapped[str] = mapped_column(
        ForeignKey("v341_outer_evaluation_blocks.id"), nullable=False, index=True
    )
    random_plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    loss_components_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    promotion_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341ModelHealthSnapshot(Base):
    __tablename__ = "v341_model_health_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "anchor_date",
            name="uq_v341_model_health_anchor",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("v341_model_versions.id"), nullable=False, index=True
    )
    health_status: Mapped[str] = mapped_column(String(40), nullable=False)
    consecutive_entry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    consecutive_exit_count: Mapped[int] = mapped_column(Integer, nullable=False)
    diagnostics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    reliability_score: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    health_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341TrainingIteration(Base):
    __tablename__ = "v341_training_iterations"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "anchor_date",
            name="uq_v341_iteration_anchor",
        ),
        UniqueConstraint(
            "protocol_version", "model_market", "weekly_iteration_number",
            name="uq_v341_iteration_number",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("v341_training_runs.id"), nullable=False, index=True
    )
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v341_forecasts.id"), nullable=False, index=True
    )
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    weekly_iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    forecast_model_id: Mapped[str] = mapped_column(
        ForeignKey("v341_model_versions.id"), nullable=False, index=True
    )
    champion_after_model_id: Mapped[str] = mapped_column(
        ForeignKey("v341_model_versions.id"), nullable=False, index=True
    )
    training_triggered: Mapped[bool] = mapped_column(Boolean, nullable=False)
    promoted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    newly_matured_count: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_matured_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_independent_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    evaluation_available_date: Mapped[date | None] = mapped_column(Date)
    validation_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V341AnalysisRun(Base):
    __tablename__ = "v341_analysis_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    forecast_id: Mapped[int | None] = mapped_column(
        ForeignKey("v341_forecasts.id"), index=True
    )
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("v341_model_versions.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_anchor_date: Mapped[date | None] = mapped_column(Date)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class V342PolicyVersion(Base):
    """Versioned policy configuration, isolated from the V3.4.1 predictor."""

    __tablename__ = "v342_policy_versions"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "version",
            name="uq_v342_policy_version",
        ),
        CheckConstraint(
            "max_single_change_pp BETWEEN 0 AND 100 "
            "AND max_single_change_pp % 5 = 0",
            name="ck_v342_policy_max_change_grid",
        ),
        CheckConstraint(
            "degraded_max_position_pp BETWEEN 0 AND 100 "
            "AND degraded_max_position_pp % 5 = 0",
            name="ck_v342_policy_degraded_cap_grid",
        ),
        CheckConstraint(
            "minimum_cooldown_sessions >= 0",
            name="ck_v342_policy_cooldown_nonnegative",
        ),
        CheckConstraint(
            "transaction_cost_bps >= 0",
            name="ck_v342_policy_cost_nonnegative",
        ),
        CheckConstraint(
            "uncertain_action IN ('HOLD', 'REDUCE_ONLY')",
            name="ck_v342_policy_uncertain_action",
        ),
        Index(
            "ix_v342_policy_market_status_effective",
            "model_market", "status", "effective_from_date",
        ),
        Index(
            "uq_v342_policy_one_champion",
            "protocol_version", "model_market",
            unique=True,
            sqlite_where=text("status = 'CHAMPION'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    parent_policy_id: Mapped[str | None] = mapped_column(
        ForeignKey("v342_policy_versions.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    effective_from_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    max_single_change_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_cooldown_sessions: Mapped[int] = mapped_column(Integer, nullable=False)
    transaction_cost_bps: Mapped[int] = mapped_column(Integer, nullable=False)
    degraded_max_position_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    uncertain_action: Mapped[str] = mapped_column(String(16), nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V342TurningAssessment(Base):
    """Immutable turning assessment derived from one frozen V3.4.1 forecast."""

    __tablename__ = "v342_turning_assessments"
    __table_args__ = (
        UniqueConstraint(
            "forecast_id", "policy_version_id", "assessment_version",
            name="uq_v342_turning_assessment",
        ),
        CheckConstraint(
            "consistency_status IN "
            "('UNCONFIRMED', 'TYPE_CONFLICT', 'TEMPORALLY_CONSISTENT', "
            "'TIME_CONFLICT', 'AMBIGUOUS_TWO_SIDED')",
            name="ck_v342_turning_consistency",
        ),
        CheckConstraint(
            "top_bottom_conflict = 0 OR consistency_status <> 'TEMPORALLY_CONSISTENT'",
            name="ck_v342_turning_conflict_not_consistent",
        ),
        Index(
            "ix_v342_turning_market_anchor",
            "model_market", "forecast_anchor_date",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v341_forecasts.id"), nullable=False, index=True
    )
    policy_version_id: Mapped[str] = mapped_column(
        ForeignKey("v342_policy_versions.id"), nullable=False, index=True
    )
    forecast_anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    assessment_version: Mapped[str] = mapped_column(String(64), nullable=False)
    window_extrema_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    price_turn_status: Mapped[str] = mapped_column(String(24), nullable=False)
    dif_turn_status: Mapped[str] = mapped_column(String(24), nullable=False)
    consistency_status: Mapped[str] = mapped_column(String(32), nullable=False)
    top_bottom_conflict: Mapped[bool] = mapped_column(Boolean, nullable=False)
    criteria_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    assessment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V342TurningCandidate(Base):
    __tablename__ = "v342_turning_candidates"
    __table_args__ = (
        UniqueConstraint(
            "assessment_id", "signal_kind", "turn_kind", "candidate_week",
            name="uq_v342_turning_candidate",
        ),
        CheckConstraint(
            "signal_kind IN ('PRICE', 'DIF')",
            name="ck_v342_candidate_signal_kind",
        ),
        CheckConstraint(
            "turn_kind IN ('TOP', 'BOTTOM')",
            name="ck_v342_candidate_turn_kind",
        ),
        CheckConstraint(
            "candidate_week BETWEEN 1 AND 13",
            name="ck_v342_candidate_week",
        ),
        CheckConstraint(
            "classification IN ('WINDOW_EXTREME', 'CANDIDATE', 'VALID_TURN')",
            name="ck_v342_candidate_classification",
        ),
        CheckConstraint(
            "confirmation_status IN ('NOT_REQUIRED', 'PENDING', 'CONFIRMED', 'REJECTED')",
            name="ck_v342_candidate_confirmation",
        ),
        Index(
            "ix_v342_candidate_signal_status",
            "assessment_id", "signal_kind", "confirmation_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assessment_id: Mapped[str] = mapped_column(
        ForeignKey("v342_turning_assessments.id"), nullable=False, index=True
    )
    signal_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    turn_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    candidate_week: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    window_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    classification: Mapped[str] = mapped_column(String(24), nullable=False)
    local_extremum_met: Mapped[bool] = mapped_column(Boolean, nullable=False)
    direction_reversal_met: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prominence_value: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    minimum_prominence: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    persistence_periods: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_persistence_periods: Mapped[int] = mapped_column(Integer, nullable=False)
    move_magnitude: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    minimum_move_magnitude: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    neutral_threshold: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    confirmation_periods: Mapped[int] = mapped_column(Integer, nullable=False)
    confirmation_status: Mapped[str] = mapped_column(String(24), nullable=False)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V342PolicyRun(Base):
    __tablename__ = "v342_policy_runs"
    __table_args__ = (
        UniqueConstraint(
            "protocol_version", "model_market", "input_identity_hash",
            name="uq_v342_policy_run_identity",
        ),
        CheckConstraint(
            "run_kind IN ('LIVE_ANALYSIS', 'OOS_REPLAY', 'RETROSPECTIVE')",
            name="ck_v342_policy_run_kind",
        ),
        CheckConstraint(
            "current_position_pp BETWEEN 0 AND 100 AND current_position_pp % 5 = 0",
            name="ck_v342_run_current_grid",
        ),
        CheckConstraint(
            "uncapped_target_pp BETWEEN 0 AND 100 AND uncapped_target_pp % 5 = 0",
            name="ck_v342_run_uncapped_grid",
        ),
        CheckConstraint(
            "target_position_pp BETWEEN 0 AND 100 AND target_position_pp % 5 = 0",
            name="ck_v342_run_target_grid",
        ),
        CheckConstraint(
            "next_executable_position_pp BETWEEN 0 AND 100 "
            "AND next_executable_position_pp % 5 = 0",
            name="ck_v342_run_next_executable_grid",
        ),
        CheckConstraint(
            "total_change_pp BETWEEN 0 AND 100 AND total_change_pp % 5 = 0",
            name="ck_v342_run_change_grid",
        ),
        CheckConstraint(
            "action IN ('BUY', 'SELL', 'HOLD')",
            name="ck_v342_run_action",
        ),
        CheckConstraint(
            "fund_etf_ratio IN ('7:3', '6:4', '5:5', '4:6', '3:7')",
            name="ck_v342_run_fund_etf_ratio",
        ),
        Index(
            "ix_v342_run_market_anchor_created",
            "model_market", "forecast_anchor_date", "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    run_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    forecast_id: Mapped[int] = mapped_column(
        ForeignKey("v341_forecasts.id"), nullable=False, index=True
    )
    policy_version_id: Mapped[str] = mapped_column(
        ForeignKey("v342_policy_versions.id"), nullable=False, index=True
    )
    turning_assessment_id: Mapped[str] = mapped_column(
        ForeignKey("v342_turning_assessments.id"), nullable=False, index=True
    )
    forecast_anchor_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    position_source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    position_source_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("v2_position_events.id"), index=True
    )
    position_effective_date: Mapped[date | None] = mapped_column(Date)
    position_updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    current_position_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    uncapped_target_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    target_position_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    next_executable_position_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    total_change_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    fund_etf_ratio: Mapped[str] = mapped_column(String(8), nullable=False)
    health_status: Mapped[str] = mapped_column(String(40), nullable=False)
    reliability_score: Mapped[int] = mapped_column(Integer, nullable=False)
    cooldown_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    input_identity_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    run_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V342PolicyBatch(Base):
    __tablename__ = "v342_policy_batches"
    __table_args__ = (
        UniqueConstraint(
            "policy_run_id", "batch_number", name="uq_v342_policy_batch"
        ),
        CheckConstraint(
            "batch_number BETWEEN 1 AND 4",
            name="ck_v342_batch_number",
        ),
        CheckConstraint(
            "action IN ('BUY', 'SELL')",
            name="ck_v342_batch_action",
        ),
        CheckConstraint(
            "change_pp > 0 AND change_pp <= 100 AND change_pp % 5 = 0",
            name="ck_v342_batch_change_grid",
        ),
        CheckConstraint(
            "target_after_pp BETWEEN 0 AND 100 AND target_after_pp % 5 = 0",
            name="ck_v342_batch_target_grid",
        ),
        CheckConstraint(
            "initial_state IN ('WAITING_CONFIRMATION', 'ELIGIBLE')",
            name="ck_v342_batch_initial_state",
        ),
        Index("ix_v342_batch_run_number", "policy_run_id", "batch_number"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_run_id: Mapped[str] = mapped_column(
        ForeignKey("v342_policy_runs.id"), nullable=False, index=True
    )
    batch_number: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    change_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    target_after_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    window_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    initial_state: Mapped[str] = mapped_column(String(32), nullable=False)
    depends_on_batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("v342_policy_batches.id"), index=True
    )
    trigger_definition_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    invalidation_definition_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    batch_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V342PolicyBatchEvent(Base):
    __tablename__ = "v342_policy_batch_events"
    __table_args__ = (
        UniqueConstraint("batch_id", "event_hash", name="uq_v342_batch_event"),
        CheckConstraint(
            "to_state IN ('WAITING_CONFIRMATION', 'ELIGIBLE', 'DEFERRED', "
            "'CANCELLED', 'EXPIRED', 'SUPERSEDED', 'EXECUTED_OBSERVED')",
            name="ck_v342_batch_event_state",
        ),
        Index("ix_v342_batch_event_time", "batch_id", "event_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("v342_policy_batches.id"), nullable=False, index=True
    )
    from_state: Mapped[str | None] = mapped_column(String(32))
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    evaluated_anchor_date: Mapped[date] = mapped_column(Date, nullable=False)
    event_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    trigger_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    reason_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V342StrategyBacktestRun(Base):
    __tablename__ = "v342_strategy_backtest_runs"
    __table_args__ = (
        UniqueConstraint(
            "policy_version_id", "model_market", "forecast_set_hash",
            "cost_model_hash", "evaluation_available_through", "price_set_hash",
            name="uq_v342_backtest_identity",
        ),
        Index(
            "ix_v342_backtest_market_policy_status",
            "model_market", "policy_version_id", "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    protocol_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_market: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    policy_version_id: Mapped[str] = mapped_column(
        ForeignKey("v342_policy_versions.id"), nullable=False, index=True
    )
    start_anchor_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_anchor_date: Mapped[date] = mapped_column(Date, nullable=False)
    evaluation_available_through: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_set_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    cost_model_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    price_set_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    decision_count: Mapped[int] = mapped_column(Integer, nullable=False)
    realized_return: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    maximum_drawdown: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    risk_adjusted_return: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    risk_adjusted_metric: Mapped[str] = mapped_column(String(32), nullable=False)
    turnover: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    transaction_cost: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    buy_hold_return: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    fixed_dca_return: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    excess_vs_buy_hold: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    excess_vs_fixed_dca: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    leakage_audit_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    run_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class V342StrategyBacktestPoint(Base):
    __tablename__ = "v342_strategy_backtest_points"
    __table_args__ = (
        UniqueConstraint(
            "backtest_run_id", "sequence", name="uq_v342_backtest_point_sequence"
        ),
        UniqueConstraint(
            "backtest_run_id", "point_date", name="uq_v342_backtest_point_date"
        ),
        CheckConstraint(
            "position_pp BETWEEN 0 AND 100 AND position_pp % 5 = 0",
            name="ck_v342_backtest_position_grid",
        ),
        CheckConstraint(
            "position_change_pp BETWEEN -100 AND 100 "
            "AND position_change_pp % 5 = 0",
            name="ck_v342_backtest_change_grid",
        ),
        Index("ix_v342_backtest_point_date", "backtest_run_id", "point_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    backtest_run_id: Mapped[str] = mapped_column(
        ForeignKey("v342_strategy_backtest_runs.id"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    point_date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_id: Mapped[int | None] = mapped_column(
        ForeignKey("v341_forecasts.id"), index=True
    )
    position_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    position_change_pp: Mapped[int] = mapped_column(Integer, nullable=False)
    gross_return: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    net_return: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    turnover: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    transaction_cost: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    strategy_equity: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    strategy_drawdown: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    buy_hold_equity: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    fixed_dca_equity: Mapped[Decimal] = mapped_column(PERCENTAGE, nullable=False)
    event_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    point_hash: Mapped[str] = mapped_column(String(64), nullable=False)
