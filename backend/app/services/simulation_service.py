"""Atomic, local-only weekly ETF paper-simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from contextlib import contextmanager
from typing import Callable, Iterator, Mapping

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import DEFAULT_FEE_CONFIG
from ..models.models import (
    AppSetting,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    SimulationAccount,
    SimulationDailySnapshot,
    SimulationPosition,
    SimulationTransaction,
    StrategyDefinition,
    StrategySignal,
)
from ..schemas.simulation import (
    SimulationAccountCreate,
    SimulationAccountRead,
    SimulationExecutionSettings,
)
from ..simulation.accounting import (
    PositionState,
    apply_buy,
    apply_sell,
    calculate_return_metrics,
)
from ..simulation.engine import MONEY_QUANTUM, money, size_buy_order
from ..simulation.fees import FeeBreakdown, FeeConfiguration, calculate_trade_fees


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class WeeklySimulationResult:
    transaction: SimulationTransaction
    price_date: date | None
    idempotent: bool
    theoretical_mode: bool


def _finite_nonnegative(value: object, field: str) -> Decimal:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal_value.is_finite() or decimal_value < ZERO:
        raise ValueError(f"{field} must be a finite nonnegative decimal")
    return decimal_value


def _frozen_fees(side: str, gross: Decimal, config: FeeConfiguration) -> FeeBreakdown:
    raw = calculate_trade_fees(gross, side, config)
    return FeeBreakdown(
        commission=money(raw.commission),
        stamp_duty=money(raw.stamp_duty),
        transfer_fee=money(raw.transfer_fee),
        other_fee=money(raw.other_fee),
        commission_rate=raw.commission_rate,
        stamp_duty_rate=raw.stamp_duty_rate,
        transfer_fee_rate=raw.transfer_fee_rate,
        other_fee_rate=raw.other_fee_rate,
        minimum_commission=raw.minimum_commission,
        minimum_commission_enabled=raw.minimum_commission_enabled,
    )


class SimulationService:
    """Owns no broker integration and persists all paper-account changes together."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    @contextmanager
    def _account_write_session(self) -> Iterator[Session]:
        """Serialize account mutations before any balance/position read occurs."""
        session = self.session_factory()
        try:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def create_account(self, values: SimulationAccountCreate) -> SimulationAccountRead:
        with self.session_factory() as session, session.begin():
            if session.scalar(select(SimulationAccount).where(SimulationAccount.name == values.name)):
                raise ValueError(f"Simulation account already exists: {values.name}")
            if values.strategy_id is not None and session.get(StrategyDefinition, values.strategy_id) is None:
                raise ValueError(f"Unknown strategy: {values.strategy_id}")
            account = SimulationAccount(
                name=values.name,
                initial_cash=values.initial_cash,
                cash_balance=values.initial_cash,
                portfolio_value=ZERO,
                strategy_id=values.strategy_id,
                enabled=values.enabled,
                notes=values.notes,
            )
            session.add(account)
            session.flush()
            return self._account_read(account)

    @staticmethod
    def _account_read(account: SimulationAccount) -> SimulationAccountRead:
        return SimulationAccountRead(
            id=account.id,
            name=account.name,
            initial_cash=account.initial_cash,
            cash_balance=account.cash_balance,
            portfolio_value=account.portfolio_value,
            strategy_id=account.strategy_id,
            enabled=account.enabled,
        )

    @staticmethod
    def _fee_config(session: Session) -> FeeConfiguration:
        setting = session.scalar(select(AppSetting).where(AppSetting.key == "fee.configuration"))
        return FeeConfiguration.from_mapping(setting.value if setting is not None else DEFAULT_FEE_CONFIG)

    def set_fee_configuration(self, changes: Mapping[str, object]) -> FeeConfiguration:
        with self.session_factory() as session, session.begin():
            setting = session.scalar(select(AppSetting).where(AppSetting.key == "fee.configuration"))
            existing: dict[str, object] = (
                dict(setting.value) if setting is not None and isinstance(setting.value, dict) else dict(DEFAULT_FEE_CONFIG)
            )
            existing.update(dict(changes))
            configuration = FeeConfiguration.from_mapping(existing)
            if setting is None:
                setting = AppSetting(
                    key="fee.configuration",
                    value=configuration.as_dict(),
                    category="default",
                    description="ETF fee configuration",
                )
                session.add(setting)
            else:
                setting.value = configuration.as_dict()
            return configuration

    @staticmethod
    def _position(session: Session, account_id: int, instrument_id: int) -> SimulationPosition:
        position = session.scalar(
            select(SimulationPosition).where(
                SimulationPosition.account_id == account_id,
                SimulationPosition.instrument_id == instrument_id,
            )
        )
        if position is None:
            position = SimulationPosition(
                account_id=account_id,
                instrument_id=instrument_id,
                quantity=ZERO,
                average_cost=ZERO,
                total_cost=ZERO,
                market_value=ZERO,
                unrealized_pnl=ZERO,
                realized_pnl=ZERO,
            )
            session.add(position)
            session.flush()
        return position

    @staticmethod
    def _position_state(position: SimulationPosition) -> PositionState:
        return PositionState(
            quantity=position.quantity,
            average_cost=position.average_cost,
            total_cost=position.total_cost,
            realized_pnl=position.realized_pnl,
        )

    @staticmethod
    def _write_position(position: SimulationPosition, state: PositionState) -> None:
        position.quantity = state.quantity
        position.average_cost = state.average_cost
        position.total_cost = state.total_cost
        position.realized_pnl = state.realized_pnl

    @staticmethod
    def _transaction(
        *,
        account: SimulationAccount,
        instrument: Instrument,
        plan_id: int | None,
        signal_id: int | None,
        execution_date: date,
        source: str,
        side: str,
        requested_amount: Decimal,
        quantity: Decimal,
        price: Decimal,
        fees: FeeBreakdown,
        cash_after: Decimal,
        realized_pnl: Decimal = ZERO,
        theoretical: bool = False,
        price_date: date | None = None,
        notes: str | None = None,
    ) -> SimulationTransaction:
        gross = money(quantity * price)
        return SimulationTransaction(
            account_id=account.id,
            instrument_id=instrument.id,
            plan_id=plan_id,
            signal_id=signal_id,
            transaction_at=datetime.combine(execution_date, time.min, tzinfo=timezone.utc),
            execution_date=execution_date,
            source_identity=source,
            transaction_type=side,
            quantity=quantity,
            price=price,
            amount=gross,
            fee=money(fees.total),
            requested_amount=money(requested_amount),
            executed_amount=gross,
            commission=fees.commission,
            stamp_duty=fees.stamp_duty,
            transfer_fee=fees.transfer_fee,
            other_fee=fees.other_fee,
            commission_rate=fees.commission_rate,
            stamp_duty_rate=fees.stamp_duty_rate,
            transfer_fee_rate=fees.transfer_fee_rate,
            other_fee_rate=fees.other_fee_rate,
            minimum_commission=fees.minimum_commission,
            minimum_commission_enabled=fees.minimum_commission_enabled,
            cash_after=cash_after,
            realized_pnl=realized_pnl,
            is_theoretical=theoretical,
            price_date=price_date,
            notes=notes,
        )

    def _snapshot(
        self,
        session: Session,
        account: SimulationAccount,
        snapshot_date: date,
        *,
        external_cash_flow: Decimal,
    ) -> SimulationDailySnapshot:
        positions = session.scalars(
            select(SimulationPosition).where(SimulationPosition.account_id == account.id)
        ).all()
        market_total = ZERO
        for position in positions:
            price = session.scalar(
                select(MarketPrice)
                .where(
                    MarketPrice.instrument_id == position.instrument_id,
                    MarketPrice.timeframe == "daily",
                    MarketPrice.trade_date <= snapshot_date,
                )
                .order_by(MarketPrice.trade_date.desc(), MarketPrice.id.desc())
            )
            mark = price.close_price if price is not None else ZERO
            position.market_value = money(position.quantity * mark)
            position.unrealized_pnl = money(position.market_value - position.total_cost)
            market_total = money(market_total + position.market_value)
        account.portfolio_value = market_total
        total_assets = money(account.cash_balance + market_total)
        current = session.scalar(
            select(SimulationDailySnapshot).where(
                SimulationDailySnapshot.account_id == account.id,
                SimulationDailySnapshot.snapshot_date == snapshot_date,
            )
        )
        prior = session.scalar(
            select(SimulationDailySnapshot)
            .where(
                SimulationDailySnapshot.account_id == account.id,
                SimulationDailySnapshot.snapshot_date < snapshot_date,
            )
            .order_by(SimulationDailySnapshot.snapshot_date.desc())
        )
        other_snapshots = session.scalars(
            select(SimulationDailySnapshot)
            .where(
                SimulationDailySnapshot.account_id == account.id,
                SimulationDailySnapshot.snapshot_date != snapshot_date,
            )
            .order_by(SimulationDailySnapshot.snapshot_date)
        ).all()
        historical_snapshots = [
            row for row in other_snapshots if row.snapshot_date < snapshot_date
        ]
        aggregate_cash_flow = money(
            (current.external_cash_flow if current is not None else ZERO) + external_cash_flow
        )
        contributed = money(
            account.initial_cash
            + sum((row.external_cash_flow for row in historical_snapshots), ZERO)
            + aggregate_cash_flow
        )
        metrics = calculate_return_metrics(
            total_assets=total_assets,
            prior_assets=prior.total_assets if prior is not None else None,
            external_cash_flow=aggregate_cash_flow,
            contributed_capital=contributed,
            historical_assets=[row.total_assets for row in historical_snapshots],
            prior_time_weighted_return=prior.time_weighted_return if prior is not None else None,
            cashflow_history=[
                (row.snapshot_date, row.external_cash_flow)
                for row in historical_snapshots
            ] + [(snapshot_date, aggregate_cash_flow)],
            valuation_date=snapshot_date,
            initial_cash=account.initial_cash,
        )
        if current is None:
            current = SimulationDailySnapshot(account_id=account.id, snapshot_date=snapshot_date)
            session.add(current)
        current.cash_balance = account.cash_balance
        current.market_value = market_total
        current.total_assets = total_assets
        current.external_cash_flow = aggregate_cash_flow
        current.daily_return = metrics.daily_return
        current.cumulative_return = metrics.cumulative_return
        current.time_weighted_return = metrics.time_weighted_return
        current.money_weighted_return = metrics.money_weighted_return
        current.annualized_return = metrics.annualized_return
        current.max_drawdown = metrics.max_drawdown
        return current

    @staticmethod
    def _latest_price(
        session: Session, instrument_id: int, execution_date: date, rule: str
    ) -> MarketPrice:
        cutoff_operator = MarketPrice.trade_date < execution_date if rule == "PREVIOUS_CLOSE" else MarketPrice.trade_date <= execution_date
        price = session.scalar(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.timeframe == "daily",
                cutoff_operator,
            )
            .order_by(MarketPrice.trade_date.desc(), MarketPrice.id.desc())
        )
        if price is None:
            raise ValueError("No eligible stored price exists for the execution date")
        if rule == "OPEN" and price.open_price is None:
            raise ValueError("Eligible stored price has no open price")
        return price

    @staticmethod
    def _signal(
        session: Session, account: SimulationAccount, instrument_id: int, execution_date: date
    ) -> StrategySignal | None:
        if account.strategy_id is None:
            return None
        return session.scalar(
            select(StrategySignal)
            .where(
                StrategySignal.strategy_id == account.strategy_id,
                StrategySignal.instrument_id == instrument_id,
                StrategySignal.as_of_date <= execution_date,
            )
            .order_by(StrategySignal.as_of_date.desc(), StrategySignal.id.desc())
        )

    @staticmethod
    def _weekly_amount(plan: InvestmentPlan, signal: StrategySignal | None) -> Decimal:
        rules = dict(plan.rule_parameters or {})
        amount = plan.amount
        recommendation = signal.recommendation_type if signal is not None else "NORMAL"
        if recommendation in {"PAUSE", "SELL_PARTIAL"}:
            if recommendation == "PAUSE" and not bool(rules.get("allow_pause", True)):
                return amount
            return ZERO
        data = dict(signal.signal_data or {}) if signal is not None else {}
        suggested = data.get("suggested_buy_amount")
        if suggested is not None:
            try:
                amount = _finite_nonnegative(suggested, "suggested_buy_amount")
            except (ArithmeticError, ValueError):
                amount = plan.amount
        else:
            multiplier = data.get("multiplier", "1")
            try:
                amount = money(plan.amount * _finite_nonnegative(multiplier, "signal multiplier"))
            except (ArithmeticError, ValueError):
                amount = plan.amount
        minimum = rules.get("minimum_weekly_amount")
        maximum = rules.get("maximum_weekly_amount")
        if minimum is not None:
            amount = max(amount, _finite_nonnegative(minimum, "minimum_weekly_amount"))
        if maximum is not None:
            amount = min(amount, _finite_nonnegative(maximum, "maximum_weekly_amount"))
        return money(amount)

    def execute_trade(
        self,
        account_id: int,
        instrument_code: str,
        side: str,
        trade_quantity: Decimal,
        price: Decimal,
        execution_date: date,
        *,
        source: str,
        notes: str | None = None,
    ) -> SimulationTransaction:
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        qty = _finite_nonnegative(trade_quantity, "trade_quantity")
        if qty <= ZERO:
            raise ValueError("trade_quantity must be positive")
        execution_price = _finite_nonnegative(price, "price")
        if execution_price <= ZERO:
            raise ValueError("price must be positive")
        with self._account_write_session() as session:
            account = session.get(SimulationAccount, account_id)
            if account is None:
                raise ValueError(f"Unknown simulation account: {account_id}")
            if not account.enabled:
                raise ValueError("Simulation account is disabled")
            instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
            if instrument is None:
                raise ValueError(f"Unknown instrument code: {instrument_code}")
            gross = money(qty * execution_price)
            fees = _frozen_fees(normalized_side, gross, self._fee_config(session))
            position = self._position(session, account.id, instrument.id)
            if normalized_side == "BUY":
                accounting = apply_buy(account.cash_balance, self._position_state(position), qty, execution_price, fees.total)
            else:
                accounting = apply_sell(account.cash_balance, self._position_state(position), qty, execution_price, fees.total)
            account.cash_balance = accounting.cash_balance
            self._write_position(position, accounting.position)
            transaction = self._transaction(
                account=account,
                instrument=instrument,
                plan_id=None,
                signal_id=None,
                execution_date=execution_date,
                source=source,
                side=normalized_side,
                requested_amount=gross,
                quantity=qty,
                price=execution_price,
                fees=fees,
                cash_after=account.cash_balance,
                realized_pnl=accounting.realized_pnl,
                notes=notes,
            )
            session.add(transaction)
            self._snapshot(session, account, execution_date, external_cash_flow=ZERO)
            session.flush()
            return transaction

    def run_weekly(
        self, account_id: int, plan_id: int, week_date: date, *, source: str = "scheduled"
    ) -> WeeklySimulationResult:
        week_start = week_date - timedelta(days=week_date.weekday())
        with self._account_write_session() as session:
            account = session.get(SimulationAccount, account_id)
            plan = session.get(InvestmentPlan, plan_id)
            if account is None:
                raise ValueError(f"Unknown simulation account: {account_id}")
            if plan is None:
                raise ValueError(f"Unknown investment plan: {plan_id}")
            if not account.enabled or not plan.enabled:
                raise ValueError("Simulation account or investment plan is disabled")
            if plan.frequency != "weekly":
                raise ValueError("Only weekly investment plans can be simulated")
            execution_date = week_start + timedelta(days=plan.weekday if plan.weekday is not None else 1)
            if plan.start_date is not None and execution_date < plan.start_date:
                raise ValueError("Plan has not started")
            if plan.end_date is not None and execution_date > plan.end_date:
                raise ValueError("Plan has ended")
            existing = session.scalar(
                select(SimulationTransaction).where(
                    SimulationTransaction.account_id == account.id,
                    SimulationTransaction.plan_id == plan.id,
                    SimulationTransaction.execution_date == execution_date,
                    SimulationTransaction.source_identity == source,
                )
            )
            if existing is not None:
                return WeeklySimulationResult(existing, existing.price_date, True, existing.is_theoretical)
            instrument = session.get(Instrument, plan.instrument_id)
            if instrument is None:
                raise RuntimeError("Plan instrument is missing")
            rules = dict(plan.rule_parameters or {})
            execution_settings = SimulationExecutionSettings(
                mode=rules.get("mode", "REAL_LOT"),
                lot_size=rules.get("lot_size", 100),
                execution_price_rule=rules.get("execution_price_rule", "CLOSE"),
            )
            price_rule = execution_settings.execution_price_rule
            stored_price = self._latest_price(session, instrument.id, execution_date, price_rule)
            execution_price = stored_price.open_price if price_rule == "OPEN" else stored_price.close_price
            assert execution_price is not None
            signal = self._signal(session, account, instrument.id, execution_date)
            amount = self._weekly_amount(plan, signal)
            config = self._fee_config(session)
            mode = execution_settings.mode
            lot_size = execution_settings.lot_size
            recommendation = signal.recommendation_type if signal is not None else "NORMAL"
            position = self._position(session, account.id, instrument.id)
            if recommendation == "SELL_PARTIAL" and position.quantity > ZERO:
                transaction = self._weekly_sell(
                    session, account, plan, instrument, position, signal, execution_date,
                    source, execution_price, stored_price.trade_date, config, mode, lot_size,
                )
                self._snapshot(session, account, execution_date, external_cash_flow=ZERO)
                session.flush()
                return WeeklySimulationResult(transaction, stored_price.trade_date, False, transaction.is_theoretical)
            if amount == ZERO:
                no_fees = _frozen_fees("BUY", ZERO, config)
                transaction = self._transaction(
                    account=account, instrument=instrument, plan_id=plan.id,
                    signal_id=signal.id if signal else None, execution_date=execution_date,
                    source=source, side="SKIP", requested_amount=ZERO, quantity=ZERO,
                    price=execution_price, fees=no_fees, cash_after=account.cash_balance,
                    price_date=stored_price.trade_date, notes="PAUSED_OR_NO_ALLOCATION",
                )
                session.add(transaction)
                self._snapshot(session, account, execution_date, external_cash_flow=ZERO)
                session.flush()
                return WeeklySimulationResult(transaction, stored_price.trade_date, False, False)
            order = size_buy_order(amount, account.cash_balance, execution_price, lot_size, mode, config)
            account.cash_balance = order.available_cash
            if order.quantity > ZERO:
                accounting = apply_buy(
                    account.cash_balance,
                    self._position_state(position),
                    order.quantity,
                    execution_price,
                    order.fees.total,
                )
                account.cash_balance = accounting.cash_balance
                self._write_position(position, accounting.position)
            note = "THEORETICAL_FRACTIONAL" if order.theoretical else order.reason
            transaction = self._transaction(
                account=account, instrument=instrument, plan_id=plan.id,
                signal_id=signal.id if signal else None, execution_date=execution_date,
                source=source, side="BUY" if order.quantity > ZERO else "SKIP",
                requested_amount=amount, quantity=order.quantity, price=execution_price,
                fees=order.fees, cash_after=account.cash_balance,
                theoretical=order.theoretical, price_date=stored_price.trade_date, notes=note,
            )
            session.add(transaction)
            self._snapshot(session, account, execution_date, external_cash_flow=amount)
            session.flush()
            return WeeklySimulationResult(transaction, stored_price.trade_date, False, order.theoretical)

    def _weekly_sell(
        self,
        session: Session,
        account: SimulationAccount,
        plan: InvestmentPlan,
        instrument: Instrument,
        position: SimulationPosition,
        signal: StrategySignal,
        execution_date: date,
        source: str,
        price: Decimal,
        price_date: date,
        config: FeeConfiguration,
        mode: str,
        lot_size: int,
    ) -> SimulationTransaction:
        strategy = session.get(StrategyDefinition, account.strategy_id) if account.strategy_id else None
        parameters = dict(strategy.parameters or {}) if strategy is not None else {}
        signal_data = dict(signal.signal_data or {})
        max_ratio = _finite_nonnegative(parameters.get("maximum_sell_ratio", "1"), "maximum_sell_ratio")
        min_holding_ratio = _finite_nonnegative(parameters.get("minimum_holding_ratio", "0"), "minimum_holding_ratio")
        requested_ratio = _finite_nonnegative(signal_data.get("suggested_sell_ratio", max_ratio), "suggested_sell_ratio")
        allowable = max(ZERO, position.quantity * (Decimal("1") - min_holding_ratio))
        sell_quantity = min(position.quantity * min(requested_ratio, max_ratio), allowable)
        if mode.upper() == "REAL_LOT":
            sell_quantity = (sell_quantity / Decimal(lot_size)).to_integral_value(rounding=ROUND_DOWN) * Decimal(lot_size)
        else:
            sell_quantity = sell_quantity.quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)
        if sell_quantity <= ZERO:
            fees = _frozen_fees("SELL", ZERO, config)
            transaction = self._transaction(
                account=account, instrument=instrument, plan_id=plan.id, signal_id=signal.id,
                execution_date=execution_date, source=source, side="SKIP", requested_amount=ZERO,
                quantity=ZERO, price=price, fees=fees, cash_after=account.cash_balance,
                price_date=price_date, notes="SELL_CONSTRAINED_BY_MIN_HOLDING",
            )
            session.add(transaction)
            return transaction
        gross = money(sell_quantity * price)
        fees = _frozen_fees("SELL", gross, config)
        accounting = apply_sell(account.cash_balance, self._position_state(position), sell_quantity, price, fees.total)
        account.cash_balance = accounting.cash_balance
        self._write_position(position, accounting.position)
        transaction = self._transaction(
            account=account, instrument=instrument, plan_id=plan.id, signal_id=signal.id,
            execution_date=execution_date, source=source, side="SELL", requested_amount=gross,
            quantity=sell_quantity, price=price, fees=fees, cash_after=account.cash_balance,
            realized_pnl=accounting.realized_pnl, theoretical=mode.upper() == "FRACTIONAL",
            price_date=price_date,
            notes="THEORETICAL_FRACTIONAL" if mode.upper() == "FRACTIONAL" else None,
        )
        session.add(transaction)
        return transaction

    def recalculate_account(
        self, account_id: int, *, through_date: date | None = None
    ) -> SimulationAccountRead:
        """Rebuild positions, cash, and daily snapshots from frozen trade history.

        Fees stored on transactions are deliberately replayed as recorded; a
        later fee-setting change therefore cannot rewrite simulated history.
        """
        if through_date is not None:
            raise ValueError("cutoff recalculation is unsupported because it must not mutate later state")
        with self._account_write_session() as session:
            account = session.get(SimulationAccount, account_id)
            if account is None:
                raise ValueError(f"Unknown simulation account: {account_id}")
            statement = select(SimulationTransaction).where(
                SimulationTransaction.account_id == account_id,
            )
            if through_date is not None:
                statement = statement.where(SimulationTransaction.execution_date <= through_date)
            transactions = session.scalars(
                statement.order_by(SimulationTransaction.execution_date, SimulationTransaction.id)
            ).all()
            session.execute(
                delete(SimulationDailySnapshot).where(SimulationDailySnapshot.account_id == account_id)
            )
            session.execute(delete(SimulationPosition).where(SimulationPosition.account_id == account_id))
            account.cash_balance = account.initial_cash
            account.portfolio_value = ZERO
            by_date: dict[date, list[SimulationTransaction]] = {}
            for transaction in transactions:
                if transaction.execution_date is None:
                    continue
                by_date.setdefault(transaction.execution_date, []).append(transaction)
            for transaction_date, dated_transactions in by_date.items():
                cash_flow = ZERO
                for transaction in dated_transactions:
                    if transaction.plan_id is not None and transaction.transaction_type in {"BUY", "SKIP"}:
                        cash_flow = money(cash_flow + transaction.requested_amount)
                        account.cash_balance = money(account.cash_balance + transaction.requested_amount)
                    if transaction.quantity <= ZERO:
                        transaction.cash_after = account.cash_balance
                        continue
                    position = self._position(session, account.id, transaction.instrument_id)
                    state = self._position_state(position)
                    if transaction.transaction_type == "BUY":
                        accounting = apply_buy(
                            account.cash_balance, state, transaction.quantity, transaction.price, transaction.fee
                        )
                    elif transaction.transaction_type == "SELL":
                        accounting = apply_sell(
                            account.cash_balance, state, transaction.quantity, transaction.price, transaction.fee
                        )
                    else:
                        continue
                    account.cash_balance = accounting.cash_balance
                    self._write_position(position, accounting.position)
                    transaction.cash_after = account.cash_balance
                    transaction.realized_pnl = accounting.realized_pnl
                self._snapshot(session, account, transaction_date, external_cash_flow=cash_flow)
            session.flush()
            return self._account_read(account)

    def chart_series(self, account_id: int) -> list[dict[str, date | Decimal | None]]:
        """Return ordered local daily snapshots ready for a client-side chart."""
        with self.session_factory() as session:
            if session.get(SimulationAccount, account_id) is None:
                raise ValueError(f"Unknown simulation account: {account_id}")
            snapshots = session.scalars(
                select(SimulationDailySnapshot)
                .where(SimulationDailySnapshot.account_id == account_id)
                .order_by(SimulationDailySnapshot.snapshot_date)
            ).all()
            return [
                {
                    "date": snapshot.snapshot_date,
                    "cash_balance": snapshot.cash_balance,
                    "market_value": snapshot.market_value,
                    "total_assets": snapshot.total_assets,
                    "cumulative_return": snapshot.cumulative_return,
                    "time_weighted_return": snapshot.time_weighted_return,
                    "money_weighted_return": snapshot.money_weighted_return,
                }
                for snapshot in snapshots
            ]
