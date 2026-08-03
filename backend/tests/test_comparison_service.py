from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_database_engine, create_session_factory
from backend.app.models.models import (
    AppSetting,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    RealAccount,
    SimulationAccount,
    SimulationDailySnapshot,
)
from backend.app.schemas.real_account import RealAccountCreate, RealTransactionCreate
from backend.app.services.comparison_service import ComparisonService
from backend.app.services.comparison_service import ComparisonNoDataError
from backend.app.services.real_account_service import RealAccountService


def test_comparison_aligns_daily_series_and_exposes_all_metric_differences(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    factory = create_session_factory(database)
    real_service = RealAccountService(factory)
    real = real_service.create_account(RealAccountCreate(name="real", initial_cash=Decimal("100")))
    with Session(create_database_engine(database)) as session, session.begin():
        instrument = session.query(Instrument).filter_by(code="589850").one()
        plan = session.scalar(select(InvestmentPlan).where(InvestmentPlan.instrument_id == instrument.id))
        assert plan is not None
        plan.amount = Decimal("100")
        plan.rule_parameters = {"mode": "FRACTIONAL", "lot_size": 1}
        fees = session.scalar(select(AppSetting).where(AppSetting.key == "fee.configuration"))
        assert fees is not None
        fees.value = {
            "buy_commission_rate": 0, "sell_commission_rate": 0, "minimum_commission": 0,
            "minimum_commission_enabled": False, "etf_stamp_duty_rate": 0,
            "transfer_fee_rate": 0, "other_fee_rate": 0,
        }
        session.add_all(
            [
                MarketPrice(instrument_id=instrument.id, trade_date=date(2026, 7, 2), timeframe="daily", close_price=Decimal("12")),
                MarketPrice(instrument_id=instrument.id, trade_date=date(2026, 7, 3), timeframe="daily", close_price=Decimal("12")),
            ]
        )
        simulation = SimulationAccount(
            name="simulation",
            initial_cash=Decimal("100"),
            cash_balance=Decimal("20"),
            portfolio_value=Decimal("100"),
        )
        session.add(simulation)
        session.flush()
        session.add_all(
            [
                SimulationDailySnapshot(
                    account_id=simulation.id,
                    snapshot_date=date(2026, 7, 1),
                    cash_balance=Decimal("100"),
                    market_value=Decimal("0"),
                    total_assets=Decimal("100"),
                    external_cash_flow=Decimal("0"),
                    cumulative_return=Decimal("0"),
                    time_weighted_return=Decimal("0"),
                    money_weighted_return=None,
                    annualized_return=None,
                    max_drawdown=Decimal("0"),
                ),
                SimulationDailySnapshot(
                    account_id=simulation.id,
                    snapshot_date=date(2026, 7, 3),
                    cash_balance=Decimal("20"),
                    market_value=Decimal("100"),
                    total_assets=Decimal("120"),
                    external_cash_flow=Decimal("0"),
                    cumulative_return=Decimal("0.2"),
                    time_weighted_return=Decimal("0.2"),
                    money_weighted_return=None,
                    annualized_return=None,
                    max_drawdown=Decimal("-0.1"),
                ),
            ]
        )
        simulation_id = simulation.id

    real_service.create_transaction(
        real.id,
        RealTransactionCreate(
            instrument_code="589850",
            side="BUY",
            transaction_date=date(2026, 7, 2),
            price=Decimal("10"),
        ),
    )
    comparison = ComparisonService(factory).compare(real.id, simulation_id)

    assert [row.date for row in comparison.series] == [
        date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3),
    ]
    final = comparison.series[-1]
    assert final.total_assets_difference is not None
    assert final.cash_difference is not None
    assert final.contribution_difference == Decimal("100")
    assert final.time_weighted_return_difference is not None
    assert final.max_drawdown_difference is not None
    assert final.fees_difference == Decimal("0")


def test_comparison_rejects_typed_invalid_ids_and_reports_missing_snapshot_data(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    factory = create_session_factory(database)
    with Session(create_database_engine(database)) as session:
        real_id = session.query(RealAccount.id).first()[0]
        simulation_id = session.query(SimulationAccount.id).first()[0]
    service = ComparisonService(factory)
    with pytest.raises(TypeError, match="real_account_id"):
        service.compare("1", 1)  # type: ignore[arg-type]
    with pytest.raises(ComparisonNoDataError):
        service.compare(real_id, simulation_id)
