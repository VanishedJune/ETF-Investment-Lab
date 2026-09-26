"""Auditable, local-only real-account ledger.

The only user-entered trade facts are instrument, direction, date, actual
execution price and an optional note.  Everything else is rebuilt from the
local plan, fee settings, stored market history and the chronological ledger.
No broker connection or order execution path exists in this module.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import date, datetime, time, timezone
from decimal import Decimal, ROUND_DOWN
from io import BytesIO, StringIO
from typing import Any, Iterator, TextIO

from openpyxl import Workbook
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from ..models.models import (
    AppSetting,
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    RealAccount,
    RealAccountSnapshot,
    RealPosition,
    RealTransaction,
    ValuationRecord,
)
from ..schemas.real_account import (
    CsvImportSummary,
    RealAccountCreate,
    RealAccountRead,
    RealAccountSnapshotRead,
    RealPositionRead,
    RealTransactionCreate,
    RealTransactionRead,
    RealTransactionUpdate,
)
from ..simulation.accounting import PositionState, apply_buy, apply_sell, calculate_return_metrics
from ..simulation.engine import money
from ..simulation.fees import FeeConfiguration, calculate_trade_fees


ZERO = Decimal("0")
EPSILON = Decimal("0.00000001")
CSV_FIELDS = ("transaction_date", "instrument_code", "side", "price", "notes")
INDEX_BY_ETF = {"589850": "000688", "159915": "399006", "159941": "NDX"}


class RealAccountNotFoundError(ValueError):
    """Raised when a requested local real account is absent."""


class RealAccountNoDataError(ValueError):
    """Raised when an operation needs a generated asset history."""


class RealAccountService:
    """Persist automatic trades and replay the complete ledger deterministically."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    @contextmanager
    def _account_write_session(self) -> Iterator[Session]:
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

    @staticmethod
    def _account_read(account: RealAccount) -> RealAccountRead:
        return RealAccountRead(
            id=account.id,
            name=account.name,
            broker_name=account.broker_name,
            initial_cash=account.initial_cash,
            cash_balance=account.cash_balance,
            enabled=account.enabled,
        )

    @staticmethod
    def _trade_date(transaction: RealTransaction) -> date:
        return transaction.transaction_at.date()

    @staticmethod
    def _as_transaction_datetime(transaction_date: date) -> datetime:
        return datetime.combine(transaction_date, time.min, tzinfo=timezone.utc)

    @staticmethod
    def _account(session: Session, account_id: int) -> RealAccount:
        account = session.get(RealAccount, account_id)
        if account is None:
            raise RealAccountNotFoundError(f"Unknown real account: {account_id}")
        return account

    @staticmethod
    def _instrument(session: Session, instrument_code: str) -> Instrument:
        instrument = session.scalar(select(Instrument).where(Instrument.code == instrument_code))
        if instrument is None:
            raise ValueError(f"Unknown instrument code: {instrument_code}")
        return instrument

    @staticmethod
    def _position_state(position: RealPosition) -> PositionState:
        return PositionState(
            quantity=position.quantity,
            average_cost=position.average_cost,
            total_cost=position.total_cost,
            realized_pnl=position.realized_pnl,
        )

    @staticmethod
    def _write_position(position: RealPosition, state: PositionState) -> None:
        position.quantity = state.quantity
        position.average_cost = state.average_cost
        position.total_cost = state.total_cost
        position.realized_pnl = state.realized_pnl

    @staticmethod
    def _position(session: Session, account_id: int, instrument_id: int) -> RealPosition:
        position = session.scalar(
            select(RealPosition).where(
                RealPosition.account_id == account_id,
                RealPosition.instrument_id == instrument_id,
            )
        )
        if position is None:
            position = RealPosition(
                account_id=account_id,
                instrument_id=instrument_id,
                quantity=ZERO,
                average_cost=ZERO,
                total_cost=ZERO,
                market_value=ZERO,
                unrealized_pnl=ZERO,
                realized_pnl=ZERO,
                fees_paid=ZERO,
            )
            session.add(position)
            session.flush()
        return position

    @staticmethod
    def _decimal_json(value: Decimal | None) -> str | None:
        if value is None:
            return None
        rendered = format(value, "f").rstrip("0").rstrip(".")
        return rendered or "0"

    @staticmethod
    def _json_decimal(value: object) -> Decimal | None:
        if value is None or value == "":
            return None
        try:
            result = Decimal(str(value))
        except Exception:
            return None
        return result if result.is_finite() else None

    def _fee_configuration(self, session: Session) -> FeeConfiguration:
        setting = session.scalar(select(AppSetting).where(AppSetting.key == "fee.configuration"))
        if setting is not None:
            return FeeConfiguration.from_mapping(setting.value)
        # A missing local setting must never create an unannounced charge.
        return FeeConfiguration.from_mapping(
            {
                "buy_commission_rate": 0,
                "sell_commission_rate": 0,
                "minimum_commission": 0,
                "minimum_commission_enabled": False,
                "etf_stamp_duty_rate": 0,
                "transfer_fee_rate": 0,
                "other_fee_rate": 0,
            }
        )

    @staticmethod
    def _plan_for_instrument(session: Session, instrument_id: int) -> InvestmentPlan:
        plan = session.scalar(
            select(InvestmentPlan)
            .where(InvestmentPlan.instrument_id == instrument_id)
            .order_by(InvestmentPlan.enabled.desc(), InvestmentPlan.id)
        )
        if plan is None:
            raise ValueError("该ETF尚未设置关联定投金额，请先在定投计划中设置单次投入金额。")
        if plan.amount <= ZERO:
            raise ValueError("该ETF关联的定投计划没有有效的单次投入金额。")
        return plan

    @staticmethod
    def _lot_step(plan: InvestmentPlan) -> Decimal:
        settings = dict(plan.rule_parameters or {})
        if str(settings.get("mode", "REAL_LOT")).upper() == "FRACTIONAL":
            return EPSILON
        lot = settings.get("lot_size", 100)
        try:
            return Decimal(str(lot))
        except Exception:
            return Decimal("100")

    @staticmethod
    def _floor_to_step(quantity: Decimal, step: Decimal) -> Decimal:
        if quantity <= ZERO or step <= ZERO:
            return ZERO
        return (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step

    def _buy_quantities(
        self,
        *,
        plan_amount: Decimal,
        price: Decimal,
        plan: InvestmentPlan,
        fee_configuration: FeeConfiguration,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """Return quantity, gross, fee and unspent plan cash without overspending."""
        provisional = calculate_trade_fees(plan_amount, "BUY", fee_configuration).total
        available = max(ZERO, plan_amount - provisional)
        quantity = self._floor_to_step(available / price, self._lot_step(plan))
        if quantity <= ZERO:
            raise ValueError("关联定投金额不足以按当前价格买入最小交易单位。")
        gross = money(quantity * price)
        fees = money(calculate_trade_fees(gross, "BUY", fee_configuration).total)
        while quantity > ZERO and gross + fees > plan_amount:
            quantity = self._floor_to_step(quantity - self._lot_step(plan), self._lot_step(plan))
            gross = money(quantity * price)
            fees = money(calculate_trade_fees(gross, "BUY", fee_configuration).total) if quantity else ZERO
        if quantity <= ZERO:
            raise ValueError("关联定投金额不足以覆盖最小交易单位及手续费。")
        return quantity, gross, fees, money(plan_amount - gross - fees)

    def _market_snapshot(
        self, session: Session, instrument: Instrument, requested_date: date
    ) -> dict[str, object]:
        """Capture the nearest historical quote and local valuation/indicator state."""
        price = session.scalar(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument.id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= requested_date,
            )
            .order_by(MarketPrice.trade_date.desc(), MarketPrice.id.desc())
        )
        index_code = INDEX_BY_ETF.get(instrument.code)
        index = self._instrument(session, index_code) if index_code else None
        index_price = (
            session.scalar(
                select(MarketPrice)
                .where(
                    MarketPrice.instrument_id == index.id,
                    MarketPrice.timeframe == "daily",
                    MarketPrice.trade_date <= requested_date,
                )
                .order_by(MarketPrice.trade_date.desc(), MarketPrice.id.desc())
            )
            if index is not None
            else None
        )
        valuation_instrument_id = instrument.id if price is not None else (index.id if index else instrument.id)
        valuation = session.scalar(
            select(ValuationRecord)
            .where(
                ValuationRecord.instrument_id == valuation_instrument_id,
                ValuationRecord.valuation_date <= requested_date,
            )
            .order_by(ValuationRecord.valuation_date.desc(), ValuationRecord.id.desc())
        )
        indicator_instrument_id = instrument.id if price is not None else (index.id if index else instrument.id)
        indicator = session.scalar(
            select(IndicatorRecord)
            .where(
                IndicatorRecord.instrument_id == indicator_instrument_id,
                IndicatorRecord.timeframe == "daily",
                IndicatorRecord.indicator_date <= requested_date,
            )
            .order_by(IndicatorRecord.indicator_date.desc(), IndicatorRecord.id.desc())
        )
        values = dict((indicator.indicator_values or {}).get("values", {})) if indicator else {}
        dif = self._json_decimal(values.get("dif"))
        dea = self._json_decimal(values.get("dea"))
        histogram = self._json_decimal(values.get("macd_histogram"))
        effective_date = price.trade_date if price is not None else (index_price.trade_date if index_price else None)
        sources = [source for source in (price.source if price else None, index_price.source if index_price else None) if source]
        latest_times = [item.updated_at for item in (price, index_price, valuation, indicator) if item is not None]
        return {
            "status": "complete" if price is not None else "待补全",
            "requested_date": requested_date.isoformat(),
            "effective_market_date": effective_date.isoformat() if effective_date else None,
            "non_trading_day_note": (
                f"该日期为非交易日，指标采用最近交易日数据：{effective_date.isoformat()}。"
                if effective_date is not None and effective_date != requested_date
                else None
            ),
            "etf_close": self._decimal_json(price.close_price if price else None),
            "reference_nav": self._decimal_json(price.close_price if price else None),
            "index_code": index_code,
            "index_close": self._decimal_json(index_price.close_price if index_price else None),
            "pe_ratio": self._decimal_json(valuation.pe_ratio if valuation else None),
            "pb_ratio": self._decimal_json(valuation.pb_ratio if valuation else None),
            "valuation_percentile": self._decimal_json(valuation.valuation_percentile if valuation else None),
            "valuation_status": self._valuation_status(valuation.valuation_percentile if valuation else None),
            "dif": self._decimal_json(dif),
            "dea": self._decimal_json(dea),
            "macd_histogram": self._decimal_json(histogram),
            "macd_status": self._macd_status(dif, dea, histogram),
            "dif_position": self._zero_position(dif),
            "dea_position": self._zero_position(dea),
            "rsi": self._decimal_json(self._json_decimal(values.get("rsi_6"))),
            "volume": self._decimal_json(price.volume if price else None),
            "data_source": " / ".join(sources) if sources else "本地行情缓存待补全",
            "data_updated_at": max(latest_times).isoformat() if latest_times else None,
        }

    @staticmethod
    def _zero_position(value: Decimal | None) -> str | None:
        if value is None:
            return None
        return "零轴上方" if value > ZERO else "零轴下方" if value < ZERO else "零轴附近"

    @staticmethod
    def _macd_status(dif: Decimal | None, dea: Decimal | None, histogram: Decimal | None) -> str | None:
        if dif is None or dea is None:
            return None
        cross = "金叉区" if dif >= dea else "死叉区"
        if histogram is None:
            return cross
        return f"{cross} / {'红柱' if histogram >= ZERO else '绿柱'}"

    @staticmethod
    def _valuation_status(percentile: Decimal | None) -> str | None:
        if percentile is None:
            return None
        if percentile <= Decimal("0.3"):
            return "低估区"
        if percentile >= Decimal("0.7"):
            return "高估区"
        return "中性区"

    def _market_close(self, session: Session, instrument_id: int, as_of: date) -> Decimal | None:
        row = session.scalar(
            select(MarketPrice)
            .where(
                MarketPrice.instrument_id == instrument_id,
                MarketPrice.timeframe == "daily",
                MarketPrice.trade_date <= as_of,
            )
            .order_by(MarketPrice.trade_date.desc(), MarketPrice.id.desc())
        )
        return row.close_price if row else None

    def _preview_values(
        self,
        session: Session,
        account: RealAccount,
        instrument: Instrument,
        values: RealTransactionCreate,
    ) -> dict[str, object]:
        fees = self._fee_configuration(session)
        market = self._market_snapshot(session, instrument, values.transaction_date)
        close = self._json_decimal(market.get("etf_close"))
        deviation = money((values.price - close) / close) if close and close > ZERO else None
        warning = bool(deviation is not None and abs(deviation) > Decimal("0.1"))
        if values.side == "BUY":
            plan = self._plan_for_instrument(session, instrument.id)
            quantity, gross, fee, remainder = self._buy_quantities(
                plan_amount=plan.amount,
                price=values.price,
                plan=plan,
                fee_configuration=fees,
            )
            return {
                "side": "BUY",
                "plan_amount": plan.amount,
                "quantity": quantity,
                "gross_amount": gross,
                "fee": fee,
                "remaining_cash": remainder,
                "market": market,
                "price_deviation": deviation,
                "price_deviation_warning": warning,
                "fee_configuration": fees.as_dict(),
            }
        holding = self._available_quantity_before(session, account.id, instrument.id, values.transaction_date)
        if holding <= ZERO:
            raise ValueError("当前没有可卖份额，无法新增卖出交易。")
        gross = money(holding * values.price)
        fee = money(calculate_trade_fees(gross, "SELL", fees).total)
        cost = self._cost_before(session, account.id, instrument.id, values.transaction_date)
        return {
            "side": "SELL",
            "available_quantity": holding,
            "quantity": holding,
            "gross_amount": gross,
            "fee": fee,
            "net_proceeds": money(gross - fee),
            "estimated_realized_pnl": money(gross - fee - cost),
            "remaining_quantity": ZERO,
            "market": market,
            "price_deviation": deviation,
            "price_deviation_warning": warning,
            "fee_configuration": fees.as_dict(),
        }

    def _available_quantity_before(
        self, session: Session, account_id: int, instrument_id: int, transaction_date: date
    ) -> Decimal:
        rows = session.scalars(
            select(RealTransaction)
            .where(
                RealTransaction.account_id == account_id,
                RealTransaction.instrument_id == instrument_id,
                RealTransaction.transaction_at <= self._as_transaction_datetime(transaction_date),
            )
            .order_by(RealTransaction.transaction_at, RealTransaction.id)
        ).all()
        holding = ZERO
        for row in rows:
            holding += row.quantity if row.transaction_type == "BUY" else -row.quantity
        return money(max(ZERO, holding))

    def _cost_before(
        self, session: Session, account_id: int, instrument_id: int, transaction_date: date
    ) -> Decimal:
        rows = session.scalars(
            select(RealTransaction)
            .where(
                RealTransaction.account_id == account_id,
                RealTransaction.instrument_id == instrument_id,
                RealTransaction.transaction_at <= self._as_transaction_datetime(transaction_date),
            )
            .order_by(RealTransaction.transaction_at, RealTransaction.id)
        ).all()
        quantity = total_cost = ZERO
        for row in rows:
            if row.transaction_type == "BUY":
                quantity += row.quantity
                total_cost += row.amount + row.fee
            elif quantity > ZERO:
                released = total_cost if row.quantity == quantity else money(total_cost / quantity * row.quantity)
                total_cost -= released
                quantity -= row.quantity
        return money(total_cost)

    def create_account(self, values: RealAccountCreate) -> RealAccountRead:
        if not isinstance(values, RealAccountCreate):
            raise TypeError("values must be a RealAccountCreate")
        with self._account_write_session() as session:
            if session.scalar(select(RealAccount).where(RealAccount.name == values.name)) is not None:
                raise ValueError(f"Real account already exists: {values.name}")
            account = RealAccount(
                name=values.name,
                broker_name=values.broker_name,
                initial_cash=values.initial_cash,
                cash_balance=values.initial_cash,
                enabled=values.enabled,
                notes=values.notes,
            )
            session.add(account)
            session.flush()
            return self._account_read(account)

    def get_account(self, account_id: int) -> RealAccountRead:
        with self.session_factory() as session:
            return self._account_read(self._account(session, account_id))

    def preview_transaction(self, account_id: int, values: RealTransactionCreate) -> dict[str, object]:
        with self.session_factory() as session:
            account = self._account(session, account_id)
            instrument = self._instrument(session, values.instrument_code)
            return self._preview_values(session, account, instrument, values)

    def create_transaction(self, account_id: int, values: RealTransactionCreate) -> RealTransactionRead:
        if not isinstance(values, RealTransactionCreate):
            raise TypeError("values must be a RealTransactionCreate")
        with self._account_write_session() as session:
            account = self._account(session, account_id)
            if not account.enabled:
                raise ValueError("Real account is disabled")
            instrument = self._instrument(session, values.instrument_code)
            duplicate = session.scalar(
                select(RealTransaction).where(
                    RealTransaction.account_id == account.id,
                    RealTransaction.instrument_id == instrument.id,
                    RealTransaction.transaction_at == self._as_transaction_datetime(values.transaction_date),
                    RealTransaction.transaction_type == values.side,
                    RealTransaction.price == values.price,
                )
            )
            if duplicate is not None:
                raise ValueError("检测到重复交易记录，请确认后再保存。")
            preview = self._preview_values(session, account, instrument, values)
            transaction = RealTransaction(
                account_id=account.id,
                instrument_id=instrument.id,
                transaction_at=self._as_transaction_datetime(values.transaction_date),
                transaction_type=values.side,
                quantity=preview["quantity"],
                price=values.price,
                amount=preview["gross_amount"],
                fee=preview["fee"],
                planned_amount=preview.get("plan_amount", ZERO),
                cash_after=ZERO,
                holding_quantity_after=ZERO,
                average_cost_after=ZERO,
                cost_basis=ZERO,
                net_proceeds=ZERO,
                realized_pnl=ZERO,
                market_snapshot=preview["market"],
                notes=values.notes,
            )
            session.add(transaction)
            session.flush()
            self._recalculate(session, account)
            session.flush()
            return self._transaction_read(session, transaction, instrument.code)

    add_transaction = create_transaction

    def update_transaction(
        self, account_id: int, transaction_id: int, changes: RealTransactionUpdate
    ) -> RealTransactionRead:
        if not isinstance(changes, RealTransactionUpdate):
            raise TypeError("changes must be a RealTransactionUpdate")
        with self._account_write_session() as session:
            account = self._account(session, account_id)
            transaction = session.scalar(
                select(RealTransaction).where(
                    RealTransaction.id == transaction_id, RealTransaction.account_id == account.id
                )
            )
            if transaction is None:
                raise ValueError(f"Unknown real transaction: {transaction_id}")
            previous = session.get(Instrument, transaction.instrument_id)
            if previous is None:
                raise RuntimeError("Real transaction instrument is missing")
            merged = {
                "instrument_code": previous.code,
                "side": transaction.transaction_type,
                "transaction_date": transaction.transaction_at.date(),
                "price": transaction.price,
                "notes": transaction.notes,
            }
            merged.update(changes.model_dump(exclude_unset=True))
            validated = RealTransactionCreate.model_validate(merged)
            instrument = self._instrument(session, validated.instrument_code)
            transaction.instrument_id = instrument.id
            transaction.transaction_at = self._as_transaction_datetime(validated.transaction_date)
            transaction.transaction_type = validated.side
            transaction.price = validated.price
            transaction.notes = validated.notes
            if transaction.transaction_type == "BUY":
                # An edit remains auditable: preserve the originally allocated
                # plan amount unless this is a legacy manual row.
                if transaction.planned_amount <= ZERO:
                    transaction.planned_amount = money(transaction.amount + transaction.fee)
            self._recalculate(session, account)
            session.flush()
            return self._transaction_read(session, transaction, instrument.code)

    def delete_transaction(self, account_id: int, transaction_id: int) -> None:
        with self._account_write_session() as session:
            account = self._account(session, account_id)
            transaction = session.scalar(
                select(RealTransaction).where(
                    RealTransaction.id == transaction_id, RealTransaction.account_id == account.id
                )
            )
            if transaction is None:
                raise ValueError(f"Unknown real transaction: {transaction_id}")
            session.delete(transaction)
            session.flush()
            self._recalculate(session, account)

    remove_transaction = delete_transaction

    def _mark_positions(self, session: Session, account: RealAccount, snapshot_date: date) -> tuple[Decimal, Decimal, Decimal]:
        market_total = realized_total = unrealized_total = ZERO
        positions = session.scalars(select(RealPosition).where(RealPosition.account_id == account.id)).all()
        for position in positions:
            close = self._market_close(session, position.instrument_id, snapshot_date) or ZERO
            position.market_value = money(position.quantity * close)
            position.unrealized_pnl = money(position.market_value - position.total_cost)
            market_total += position.market_value
            realized_total += position.realized_pnl
            unrealized_total += position.unrealized_pnl
        return money(market_total), money(realized_total), money(unrealized_total)

    def _snapshot(
        self,
        session: Session,
        account: RealAccount,
        snapshot_date: date,
        *,
        total_contribution: Decimal,
        external_cash_flow: Decimal,
        fees_paid: Decimal,
        cashflow_history: list[tuple[date, Decimal]],
        benchmark_assets: Decimal | None,
        benchmark_quantity: Decimal | None,
    ) -> RealAccountSnapshot:
        market_total, realized, unrealized = self._mark_positions(session, account, snapshot_date)
        total_assets = money(account.cash_balance + market_total)
        historical = session.scalars(
            select(RealAccountSnapshot)
            .where(
                RealAccountSnapshot.account_id == account.id,
                RealAccountSnapshot.snapshot_date < snapshot_date,
            )
            .order_by(RealAccountSnapshot.snapshot_date)
        ).all()
        prior = historical[-1] if historical else None
        metrics = calculate_return_metrics(
            total_assets=total_assets,
            prior_assets=prior.total_assets if prior else None,
            external_cash_flow=external_cash_flow,
            contributed_capital=total_contribution,
            historical_assets=[row.total_assets for row in historical],
            prior_time_weighted_return=prior.time_weighted_return if prior else None,
            cashflow_history=cashflow_history,
            valuation_date=snapshot_date,
            initial_cash=account.initial_cash,
        )
        snapshot = RealAccountSnapshot(
            account_id=account.id,
            snapshot_date=snapshot_date,
            cash_balance=account.cash_balance,
            market_value=market_total,
            total_assets=total_assets,
            total_contribution=total_contribution,
            external_cash_flow=external_cash_flow,
            total_pnl=money(total_assets - total_contribution),
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            holding_quantity=money(sum((p.quantity for p in session.scalars(select(RealPosition).where(RealPosition.account_id == account.id)).all()), ZERO)),
            fees_paid=fees_paid,
            daily_return=metrics.daily_return,
            total_return=metrics.cumulative_return,
            time_weighted_return=metrics.time_weighted_return,
            money_weighted_return=metrics.money_weighted_return,
            annualized_return=metrics.annualized_return,
            max_drawdown=metrics.max_drawdown,
            benchmark_total_assets=benchmark_assets,
            benchmark_total_pnl=money(benchmark_assets - total_contribution) if benchmark_assets is not None else None,
            benchmark_quantity=benchmark_quantity,
        )
        session.add(snapshot)
        return snapshot

    def _recalculate(self, session: Session, account: RealAccount) -> None:
        """Replay every record by date then creation order, including historical backfills."""
        transactions = session.scalars(
            select(RealTransaction)
            .where(RealTransaction.account_id == account.id)
            .order_by(RealTransaction.transaction_at, RealTransaction.id)
        ).all()
        session.execute(delete(RealAccountSnapshot).where(RealAccountSnapshot.account_id == account.id))
        session.execute(delete(RealPosition).where(RealPosition.account_id == account.id))
        account.cash_balance = account.initial_cash
        if not transactions:
            return
        instruments = {row.id: row for row in session.scalars(select(Instrument).where(Instrument.id.in_({t.instrument_id for t in transactions}))).all()}
        fee_configuration = self._fee_configuration(session)
        grouped: dict[date, list[RealTransaction]] = {}
        for transaction in transactions:
            grouped.setdefault(self._trade_date(transaction), []).append(transaction)
        all_market_dates = set(
            session.scalars(
                select(MarketPrice.trade_date)
                .where(
                    MarketPrice.instrument_id.in_(set(instruments)),
                    MarketPrice.timeframe == "daily",
                    MarketPrice.trade_date >= min(grouped),
                )
                .order_by(MarketPrice.trade_date)
            ).all()
        )
        all_dates = sorted(all_market_dates | set(grouped))
        total_contribution = account.initial_cash
        fees_paid = ZERO
        cashflow_history: list[tuple[date, Decimal]] = []
        if account.initial_cash > ZERO:
            cashflow_history.append((all_dates[0], account.initial_cash))
        benchmark_cash = account.initial_cash
        benchmark_quantities: dict[int, Decimal] = {instrument_id: ZERO for instrument_id in instruments}

        for snapshot_date in all_dates:
            daily_flow = ZERO
            for transaction in grouped.get(snapshot_date, []):
                instrument = instruments[transaction.instrument_id]
                position = self._position(session, account.id, transaction.instrument_id)
                state = self._position_state(position)
                market = self._market_snapshot(session, instrument, snapshot_date)
                if transaction.transaction_type == "BUY":
                    plan_amount = transaction.planned_amount
                    if plan_amount <= ZERO:  # a pre-v12 manual ledger row remains readable and exact
                        accounting = apply_buy(account.cash_balance, state, transaction.quantity, transaction.price, transaction.fee)
                    else:
                        plan = self._plan_for_instrument(session, instrument.id)
                        quantity, gross, fee, _remainder = self._buy_quantities(
                            plan_amount=plan_amount,
                            price=transaction.price,
                            plan=plan,
                            fee_configuration=fee_configuration,
                        )
                        transaction.quantity, transaction.amount, transaction.fee = quantity, gross, fee
                        accounting = apply_buy(account.cash_balance + plan_amount, state, quantity, transaction.price, fee)
                        daily_flow += plan_amount
                        total_contribution += plan_amount
                        cashflow_history.append((snapshot_date, plan_amount))
                        benchmark_price = self._market_close(session, instrument.id, snapshot_date) or transaction.price
                        benchmark_quantity, benchmark_gross, benchmark_fee, _ = self._buy_quantities(
                            plan_amount=plan_amount,
                            price=benchmark_price,
                            plan=plan,
                            fee_configuration=fee_configuration,
                        )
                        benchmark_quantities[instrument.id] += benchmark_quantity
                        benchmark_cash = money(benchmark_cash + plan_amount - benchmark_gross - benchmark_fee)
                    transaction.cost_basis = money(transaction.amount + transaction.fee)
                    transaction.net_proceeds = ZERO
                    transaction.realized_pnl = ZERO
                elif transaction.transaction_type == "SELL":
                    if state.quantity <= ZERO:
                        raise ValueError("当前没有可卖份额，无法新增卖出交易。")
                    transaction.quantity = state.quantity
                    transaction.amount = money(transaction.quantity * transaction.price)
                    transaction.fee = money(calculate_trade_fees(transaction.amount, "SELL", fee_configuration).total)
                    transaction.cost_basis = state.total_cost
                    accounting = apply_sell(account.cash_balance, state, transaction.quantity, transaction.price, transaction.fee)
                    transaction.net_proceeds = money(transaction.amount - transaction.fee)
                    transaction.realized_pnl = accounting.realized_pnl
                else:
                    raise ValueError("real transaction side must be BUY or SELL")
                account.cash_balance = accounting.cash_balance
                self._write_position(position, accounting.position)
                position.fees_paid = money(position.fees_paid + transaction.fee)
                fees_paid += transaction.fee
                transaction.cash_after = account.cash_balance
                transaction.holding_quantity_after = accounting.position.quantity
                transaction.average_cost_after = accounting.position.average_cost
                transaction.market_snapshot = market
                close = self._json_decimal(market.get("etf_close"))
                if close and close > ZERO:
                    transaction.market_snapshot["price_deviation"] = self._decimal_json(money((transaction.price - close) / close))
                    transaction.market_snapshot["price_deviation_warning"] = abs(transaction.price - close) / close > Decimal("0.1")
                else:
                    transaction.market_snapshot["price_deviation"] = None
                    transaction.market_snapshot["price_deviation_warning"] = False

            benchmark_market = ZERO
            benchmark_quantity_total = ZERO
            for instrument_id, quantity in benchmark_quantities.items():
                benchmark_quantity_total += quantity
                benchmark_market += quantity * (self._market_close(session, instrument_id, snapshot_date) or ZERO)
            benchmark_assets = money(benchmark_cash + benchmark_market)
            self._snapshot(
                session,
                account,
                snapshot_date,
                total_contribution=total_contribution,
                external_cash_flow=daily_flow,
                fees_paid=money(fees_paid),
                cashflow_history=cashflow_history,
                benchmark_assets=benchmark_assets,
                benchmark_quantity=money(benchmark_quantity_total),
            )

    def recalculate_account(self, account_id: int) -> RealAccountRead:
        with self._account_write_session() as session:
            account = self._account(session, account_id)
            self._recalculate(session, account)
            session.flush()
            return self._account_read(account)

    def recalculate_all_accounts(self) -> int:
        """Backfill pending quote/indicator snapshots after a local data refresh."""
        with self.session_factory() as session:
            account_ids = list(session.scalars(select(RealAccount.id).order_by(RealAccount.id)))
        for account_id in account_ids:
            self.recalculate_account(account_id)
        return len(account_ids)

    def _transaction_read(self, session: Session, transaction: RealTransaction, instrument_code: str) -> RealTransactionRead:
        position = session.scalar(
            select(RealPosition).where(
                RealPosition.account_id == transaction.account_id,
                RealPosition.instrument_id == transaction.instrument_id,
            )
        )
        cumulative_realized = sum(
            session.scalars(
                select(RealTransaction.realized_pnl).where(
                    RealTransaction.account_id == transaction.account_id,
                    RealTransaction.transaction_at <= transaction.transaction_at,
                )
            ).all(),
            ZERO,
        )
        latest = session.scalar(
            select(RealAccountSnapshot)
            .where(RealAccountSnapshot.account_id == transaction.account_id)
            .order_by(RealAccountSnapshot.snapshot_date.desc(), RealAccountSnapshot.id.desc())
        )
        unrealized = position.unrealized_pnl if position else ZERO
        return RealTransactionRead(
            id=transaction.id,
            account_id=transaction.account_id,
            instrument_id=transaction.instrument_id,
            instrument_code=instrument_code,
            side=transaction.transaction_type,
            transaction_date=transaction.transaction_at.date(),
            quantity=transaction.quantity,
            price=transaction.price,
            amount=transaction.amount,
            fee=transaction.fee,
            planned_amount=transaction.planned_amount,
            cash_after=transaction.cash_after,
            holding_quantity_after=transaction.holding_quantity_after,
            average_cost_after=transaction.average_cost_after,
            cost_basis=transaction.cost_basis,
            net_proceeds=transaction.net_proceeds,
            realized_pnl=transaction.realized_pnl,
            cumulative_realized_pnl=money(cumulative_realized),
            current_unrealized_pnl=unrealized,
            total_pnl=latest.total_pnl if latest else ZERO,
            total_return=latest.total_return if latest else None,
            market_snapshot=dict(transaction.market_snapshot or {}),
            notes=transaction.notes,
        )

    def transactions(self, account_id: int) -> list[RealTransactionRead]:
        with self.session_factory() as session:
            self._account(session, account_id)
            rows = session.execute(
                select(RealTransaction, Instrument.code)
                .join(Instrument, Instrument.id == RealTransaction.instrument_id)
                .where(RealTransaction.account_id == account_id)
                .order_by(RealTransaction.transaction_at, RealTransaction.id)
            ).all()
            return [self._transaction_read(session, transaction, code) for transaction, code in rows]

    def positions(self, account_id: int) -> list[RealPositionRead]:
        with self.session_factory() as session:
            self._account(session, account_id)
            rows = session.execute(
                select(RealPosition, Instrument.code)
                .join(Instrument, Instrument.id == RealPosition.instrument_id)
                .where(RealPosition.account_id == account_id)
                .order_by(Instrument.code)
            ).all()
            return [
                RealPositionRead(
                    id=position.id,
                    account_id=position.account_id,
                    instrument_id=position.instrument_id,
                    instrument_code=code,
                    quantity=position.quantity,
                    average_cost=position.average_cost,
                    total_cost=position.total_cost,
                    market_value=position.market_value,
                    unrealized_pnl=position.unrealized_pnl,
                    realized_pnl=position.realized_pnl,
                    fees_paid=position.fees_paid,
                )
                for position, code in rows
            ]

    @staticmethod
    def _snapshot_read(snapshot: RealAccountSnapshot) -> RealAccountSnapshotRead:
        return RealAccountSnapshotRead(
            snapshot_date=snapshot.snapshot_date,
            total_contribution=snapshot.total_contribution,
            cash_balance=snapshot.cash_balance,
            market_value=snapshot.market_value,
            total_assets=snapshot.total_assets,
            total_pnl=snapshot.total_pnl,
            realized_pnl=snapshot.realized_pnl,
            unrealized_pnl=snapshot.unrealized_pnl,
            holding_quantity=snapshot.holding_quantity,
            benchmark_total_assets=snapshot.benchmark_total_assets,
            benchmark_total_pnl=snapshot.benchmark_total_pnl,
            benchmark_quantity=snapshot.benchmark_quantity,
            fees_paid=snapshot.fees_paid,
            total_return=snapshot.total_return,
            time_weighted_return=snapshot.time_weighted_return,
            money_weighted_return=snapshot.money_weighted_return,
            annualized_return=snapshot.annualized_return,
            max_drawdown=snapshot.max_drawdown,
        )

    def snapshots(self, account_id: int) -> list[RealAccountSnapshotRead]:
        with self.session_factory() as session:
            self._account(session, account_id)
            rows = session.scalars(
                select(RealAccountSnapshot)
                .where(RealAccountSnapshot.account_id == account_id)
                .order_by(RealAccountSnapshot.snapshot_date)
            ).all()
            return [self._snapshot_read(row) for row in rows]

    @staticmethod
    def _period_end(day: date, timeframe: str) -> tuple[int, int]:
        if timeframe == "weekly":
            iso = day.isocalendar()
            return iso.year, iso.week
        if timeframe == "monthly":
            return day.year, day.month
        return day.year, day.toordinal()

    def chart_series(
        self,
        account_id: int,
        *,
        timeframe: str = "daily",
        start_date: date | None = None,
        end_date: date | None = None,
        instrument_code: str | None = None,
    ) -> list[dict[str, date | Decimal | None]]:
        if timeframe not in {"daily", "weekly", "monthly"}:
            raise ValueError("timeframe must be daily, weekly, or monthly")
        snapshots = self.snapshots(account_id)
        if start_date:
            snapshots = [row for row in snapshots if row.snapshot_date >= start_date]
        if end_date:
            snapshots = [row for row in snapshots if row.snapshot_date <= end_date]
        if not snapshots:
            raise RealAccountNoDataError(f"No real-account snapshots exist for account: {account_id}")
        grouped: dict[tuple[int, int], RealAccountSnapshotRead] = {}
        for snapshot in snapshots:
            grouped[self._period_end(snapshot.snapshot_date, timeframe)] = snapshot
        price_by_date: dict[date, Decimal | None] = {}
        if instrument_code:
            with self.session_factory() as session:
                instrument = self._instrument(session, instrument_code)
                prices = session.scalars(
                    select(MarketPrice)
                    .where(MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == "daily")
                    .order_by(MarketPrice.trade_date, MarketPrice.id)
                ).all()
                latest: Decimal | None = None
                iterator = iter(prices)
                current = next(iterator, None)
                for row in grouped.values():
                    while current is not None and current.trade_date <= row.snapshot_date:
                        latest = current.close_price
                        current = next(iterator, None)
                    price_by_date[row.snapshot_date] = latest
        return [
            {
                "date": row.snapshot_date,
                "contribution": row.total_contribution,
                "cash_balance": row.cash_balance,
                "market_value": row.market_value,
                "total_assets": row.total_assets,
                "total_pnl": row.total_pnl,
                "realized_pnl": row.realized_pnl,
                "unrealized_pnl": row.unrealized_pnl,
                "holding_quantity": row.holding_quantity,
                "benchmark_total_assets": row.benchmark_total_assets,
                "benchmark_total_pnl": row.benchmark_total_pnl,
                "benchmark_quantity": row.benchmark_quantity,
                "fees_paid": row.fees_paid,
                "total_return": row.total_return,
                "market_price": price_by_date.get(row.snapshot_date),
            }
            for row in grouped.values()
        ]

    def comparison_summary(self, account_id: int) -> dict[str, Decimal | None]:
        snapshots = self.snapshots(account_id)
        if not snapshots:
            raise RealAccountNoDataError(f"No real-account snapshots exist for account: {account_id}")
        latest = snapshots[-1]
        benchmark_assets = latest.benchmark_total_assets
        benchmark_pnl = latest.benchmark_total_pnl
        contribution = latest.total_contribution
        return {
            "actual_total_assets": latest.total_assets,
            "benchmark_total_assets": benchmark_assets,
            "actual_total_pnl": latest.total_pnl,
            "benchmark_total_pnl": benchmark_pnl,
            "actual_total_return": latest.total_return,
            "benchmark_total_return": money(benchmark_pnl / contribution) if benchmark_pnl is not None and contribution > ZERO else None,
            "outperformance": money(latest.total_assets - benchmark_assets) if benchmark_assets is not None else None,
            "actual_quantity": latest.holding_quantity,
            "benchmark_quantity": latest.benchmark_quantity,
            "quantity_difference": money(latest.holding_quantity - latest.benchmark_quantity) if latest.benchmark_quantity is not None else None,
        }

    def export_csv(self, account_id: int) -> str:
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for transaction in self.transactions(account_id):
            writer.writerow(
                {
                    "transaction_date": transaction.transaction_date.isoformat(),
                    "instrument_code": transaction.instrument_code,
                    "side": transaction.side,
                    "price": format(transaction.price, "f"),
                    "notes": transaction.notes or "",
                }
            )
        return output.getvalue()

    def export_xlsx(self, account_id: int) -> bytes:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "交易收益明细"
        headers = [
            "序号", "ETF代码", "操作类型", "操作日期", "实际成交价格", "当日收盘价", "交易金额", "交易份额", "手续费",
            "交易后持有份额", "交易后平均成本", "本次已实现收益", "累计已实现收益", "当前未实现收益", "累计收益", "累计收益率",
            "市盈率", "市净率", "估值百分位", "DIF", "DEA", "MACD状态", "备注",
        ]
        sheet.append(headers)
        for number, row in enumerate(self.transactions(account_id), start=1):
            market = row.market_snapshot
            sheet.append([
                number, row.instrument_code, row.side, row.transaction_date.isoformat(), float(row.price), market.get("etf_close"), float(row.amount),
                float(row.quantity), float(row.fee), float(row.holding_quantity_after), float(row.average_cost_after), float(row.realized_pnl),
                float(row.cumulative_realized_pnl), float(row.current_unrealized_pnl), float(row.total_pnl), float(row.total_return or ZERO),
                market.get("pe_ratio"), market.get("pb_ratio"), market.get("valuation_percentile"), market.get("dif"), market.get("dea"),
                market.get("macd_status"), row.notes or "",
            ])
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = 16
        output = BytesIO()
        workbook.save(output)
        return output.getvalue()

    @staticmethod
    def _csv_text(content: str | bytes | TextIO) -> str:
        if isinstance(content, bytes):
            return content.decode("utf-8-sig")
        if isinstance(content, str):
            return content
        if hasattr(content, "read"):
            value = content.read()
            return value.decode("utf-8-sig") if isinstance(value, bytes) else value
        raise TypeError("CSV content must be str, bytes, or a text stream")

    def import_csv(self, account_id: int, content: str | bytes | TextIO) -> CsvImportSummary:
        source = self._csv_text(content)
        reader = csv.DictReader(StringIO(source))
        if reader.fieldnames is None:
            return CsvImportSummary(imported=0, rejected=1, errors=["CSV header is required"])
        missing = [field for field in CSV_FIELDS[:4] if field not in reader.fieldnames]
        if missing:
            return CsvImportSummary(imported=0, rejected=1, errors=[f"CSV is missing required columns: {', '.join(missing)}"])
        records: list[RealTransactionCreate] = []
        errors: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                records.append(RealTransactionCreate.model_validate({key: row.get(key) for key in CSV_FIELDS}))
            except Exception as error:
                errors.append(f"row {row_number}: {error}")
        if errors:
            return CsvImportSummary(imported=0, rejected=len(errors), errors=errors)
        ids: list[int] = []
        for record in records:
            try:
                ids.append(self.create_transaction(account_id, record).id)
            except Exception as error:
                return CsvImportSummary(imported=0, rejected=1, errors=[str(error)])
        return CsvImportSummary(imported=len(ids), rejected=0, transaction_ids=ids)
