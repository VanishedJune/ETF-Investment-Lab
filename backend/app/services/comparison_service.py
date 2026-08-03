"""Read-only local comparison of actual manual and simulated account records."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ..models.models import (
    RealAccount,
    RealAccountSnapshot,
    SimulationAccount,
    SimulationDailySnapshot,
    SimulationTransaction,
)
from ..schemas.comparison import AccountComparisonRead, ComparisonSeriesPoint
from ..simulation.engine import money
from .real_account_service import RealAccountNotFoundError


ZERO = Decimal("0")


class SimulationAccountNotFoundError(ValueError):
    """Raised when the requested simulation account is absent."""


class ComparisonNoDataError(ValueError):
    """Raised when either selected account has no local daily projection."""


class ComparisonService:
    """Align locally stored snapshots without broker/API or future-price access."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    @staticmethod
    def _valid_id(value: int, field_name: str) -> int:
        if type(value) is not int or value <= 0:
            raise TypeError(f"{field_name} must be a positive integer")
        return value

    @staticmethod
    def _difference(left: Decimal | None, right: Decimal | None) -> Decimal | None:
        return money(left - right) if left is not None and right is not None else None

    @staticmethod
    def _simulation_fees(
        transactions: list[SimulationTransaction], through: date
    ) -> Decimal:
        return money(
            sum(
                (
                    transaction.fee
                    for transaction in transactions
                    if (transaction.execution_date or transaction.transaction_at.date()) <= through
                ),
                ZERO,
            )
        )

    def compare(self, real_account_id: int, simulation_account_id: int) -> AccountComparisonRead:
        real_account_id = self._valid_id(real_account_id, "real_account_id")
        simulation_account_id = self._valid_id(simulation_account_id, "simulation_account_id")
        with self.session_factory() as session:
            real_account = session.get(RealAccount, real_account_id)
            if real_account is None:
                raise RealAccountNotFoundError(f"Unknown real account: {real_account_id}")
            simulation_account = session.get(SimulationAccount, simulation_account_id)
            if simulation_account is None:
                raise SimulationAccountNotFoundError(
                    f"Unknown simulation account: {simulation_account_id}"
                )
            real_snapshots = session.scalars(
                select(RealAccountSnapshot)
                .where(RealAccountSnapshot.account_id == real_account_id)
                .order_by(RealAccountSnapshot.snapshot_date)
            ).all()
            simulation_snapshots = session.scalars(
                select(SimulationDailySnapshot)
                .where(SimulationDailySnapshot.account_id == simulation_account_id)
                .order_by(SimulationDailySnapshot.snapshot_date)
            ).all()
            if not real_snapshots or not simulation_snapshots:
                raise ComparisonNoDataError(
                    "Both selected accounts need at least one stored daily snapshot"
                )
            simulation_transactions = session.scalars(
                select(SimulationTransaction)
                .where(SimulationTransaction.account_id == simulation_account_id)
                .order_by(SimulationTransaction.execution_date, SimulationTransaction.id)
            ).all()
            return self._aligned(
                real_account,
                simulation_account,
                real_snapshots,
                simulation_snapshots,
                simulation_transactions,
            )

    def _aligned(
        self,
        real_account: RealAccount,
        simulation_account: SimulationAccount,
        real_snapshots: list[RealAccountSnapshot],
        simulation_snapshots: list[SimulationDailySnapshot],
        simulation_transactions: list[SimulationTransaction],
    ) -> AccountComparisonRead:
        dates = sorted(
            {snapshot.snapshot_date for snapshot in real_snapshots}
            | {snapshot.snapshot_date for snapshot in simulation_snapshots}
        )
        real_by_date = {snapshot.snapshot_date: snapshot for snapshot in real_snapshots}
        simulation_by_date = {snapshot.snapshot_date: snapshot for snapshot in simulation_snapshots}
        current_real: RealAccountSnapshot | None = None
        current_simulation: SimulationDailySnapshot | None = None
        simulation_contribution = simulation_account.initial_cash
        points: list[ComparisonSeriesPoint] = []
        for snapshot_date in dates:
            next_real = real_by_date.get(snapshot_date)
            next_simulation = simulation_by_date.get(snapshot_date)
            if next_real is not None:
                current_real = next_real
            if next_simulation is not None:
                current_simulation = next_simulation
                simulation_contribution = money(
                    simulation_contribution + current_simulation.external_cash_flow
                )
            real_contribution = (
                current_real.total_contribution if current_real is not None else None
            )
            sim_contribution = simulation_contribution if current_simulation is not None else None
            real_assets = current_real.total_assets if current_real is not None else None
            sim_assets = current_simulation.total_assets if current_simulation is not None else None
            real_cash = current_real.cash_balance if current_real is not None else None
            sim_cash = current_simulation.cash_balance if current_simulation is not None else None
            real_market = current_real.market_value if current_real is not None else None
            sim_market = current_simulation.market_value if current_simulation is not None else None
            real_return = current_real.total_return if current_real is not None else None
            sim_return = current_simulation.cumulative_return if current_simulation is not None else None
            real_twr = current_real.time_weighted_return if current_real is not None else None
            sim_twr = current_simulation.time_weighted_return if current_simulation is not None else None
            real_mwr = current_real.money_weighted_return if current_real is not None else None
            sim_mwr = current_simulation.money_weighted_return if current_simulation is not None else None
            real_annualized = current_real.annualized_return if current_real is not None else None
            sim_annualized = current_simulation.annualized_return if current_simulation is not None else None
            real_drawdown = current_real.max_drawdown if current_real is not None else None
            sim_drawdown = current_simulation.max_drawdown if current_simulation is not None else None
            real_fees = current_real.fees_paid if current_real is not None else None
            sim_fees = (
                self._simulation_fees(
                    simulation_transactions, current_simulation.snapshot_date
                )
                if current_simulation is not None
                else None
            )
            points.append(
                ComparisonSeriesPoint(
                    date=snapshot_date,
                    real_contribution=real_contribution,
                    simulation_contribution=sim_contribution,
                    contribution_difference=self._difference(real_contribution, sim_contribution),
                    real_total_assets=real_assets,
                    simulation_total_assets=sim_assets,
                    total_assets_difference=self._difference(real_assets, sim_assets),
                    real_cash=real_cash,
                    simulation_cash=sim_cash,
                    cash_difference=self._difference(real_cash, sim_cash),
                    real_market_value=real_market,
                    simulation_market_value=sim_market,
                    market_value_difference=self._difference(real_market, sim_market),
                    real_total_return=real_return,
                    simulation_total_return=sim_return,
                    total_return_difference=self._difference(real_return, sim_return),
                    real_time_weighted_return=real_twr,
                    simulation_time_weighted_return=sim_twr,
                    time_weighted_return_difference=self._difference(real_twr, sim_twr),
                    real_money_weighted_return=real_mwr,
                    simulation_money_weighted_return=sim_mwr,
                    money_weighted_return_difference=self._difference(real_mwr, sim_mwr),
                    real_annualized_return=real_annualized,
                    simulation_annualized_return=sim_annualized,
                    annualized_return_difference=self._difference(real_annualized, sim_annualized),
                    real_max_drawdown=real_drawdown,
                    simulation_max_drawdown=sim_drawdown,
                    max_drawdown_difference=self._difference(real_drawdown, sim_drawdown),
                    real_fees=real_fees,
                    simulation_fees=sim_fees,
                    fees_difference=self._difference(real_fees, sim_fees),
                )
            )
        return AccountComparisonRead(
            real_account_id=real_account.id,
            simulation_account_id=simulation_account.id,
            series=points,
        )

    def chart_series(self, real_account_id: int, simulation_account_id: int) -> list[ComparisonSeriesPoint]:
        """Return the aligned series directly for chart adapters."""
        return self.compare(real_account_id, simulation_account_id).series
