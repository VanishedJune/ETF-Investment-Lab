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
    IndicatorRecord,
    Instrument,
    InvestmentPlan,
    MarketPrice,
    RealAccountSnapshot,
    ValuationRecord,
)
from backend.app.schemas.real_account import RealAccountCreate, RealTransactionCreate, RealTransactionUpdate
from backend.app.services.real_account_service import RealAccountNoDataError, RealAccountService


def _automatic_service(tmp_path: Path) -> tuple[RealAccountService, int, Path]:
    database = tmp_path / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    with Session(create_database_engine(database)) as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "589850"))
        assert instrument is not None
        plan = session.scalar(select(InvestmentPlan).where(InvestmentPlan.instrument_id == instrument.id))
        assert plan is not None
        plan.amount = Decimal("100")
        plan.rule_parameters = {"mode": "FRACTIONAL", "lot_size": 1}
        fees = session.scalar(select(AppSetting).where(AppSetting.key == "fee.configuration"))
        assert fees is not None
        fees.value = {
            "buy_commission_rate": 0,
            "sell_commission_rate": 0,
            "minimum_commission": 0,
            "minimum_commission_enabled": False,
            "etf_stamp_duty_rate": 0,
            "transfer_fee_rate": 0,
            "other_fee_rate": 0,
        }
        session.add_all(
            [
                MarketPrice(instrument_id=instrument.id, trade_date=date(2026, 7, 1), timeframe="daily", close_price=Decimal("10"), source="test"),
                MarketPrice(instrument_id=instrument.id, trade_date=date(2026, 7, 2), timeframe="daily", close_price=Decimal("12"), source="test"),
                MarketPrice(instrument_id=instrument.id, trade_date=date(2026, 7, 3), timeframe="daily", close_price=Decimal("11"), source="test"),
            ]
        )
    service = RealAccountService(create_session_factory(database))
    account = service.create_account(RealAccountCreate(name=f"automatic-{tmp_path.name}"))
    return service, account.id, database


def _trade(service: RealAccountService, account_id: int, when: date, side: str, price: str):
    return service.create_transaction(
        account_id,
        RealTransactionCreate(
            instrument_code="589850", side=side, transaction_date=when, price=Decimal(price)
        ),
    )


def test_real_trade_uses_linked_plan_then_auto_sells_only_prior_holdings(tmp_path: Path) -> None:
    """The trade form carries no manually-entered money, shares, or fee fields."""
    service, account_id, _database = _automatic_service(tmp_path)
    buy = _trade(service, account_id, date(2026, 7, 1), "BUY", "10")
    sale = _trade(service, account_id, date(2026, 7, 2), "SELL", "12")

    assert buy.quantity == Decimal("10")
    assert buy.amount == Decimal("100")
    assert buy.planned_amount == Decimal("100")
    assert sale.quantity == Decimal("10")
    assert sale.realized_pnl == Decimal("20")
    assert service.positions(account_id)[0].quantity == Decimal("0")
    assert service.get_account(account_id).cash_balance == Decimal("120")


def test_historical_buy_backfill_replays_later_auto_sale_and_daily_asset_series(tmp_path: Path) -> None:
    service, account_id, database = _automatic_service(tmp_path)
    _trade(service, account_id, date(2026, 7, 1), "BUY", "10")
    sale = _trade(service, account_id, date(2026, 7, 3), "SELL", "11")
    _trade(service, account_id, date(2026, 7, 2), "BUY", "12")

    rows = service.transactions(account_id)
    assert [row.transaction_date for row in rows] == [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)]
    assert rows[-1].id == sale.id
    assert rows[-1].quantity == Decimal("18.33333333")
    assert service.positions(account_id)[0].quantity == Decimal("0")
    series = service.chart_series(account_id)
    assert [row["date"] for row in series] == [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)]
    assert series[-1]["benchmark_total_assets"] is not None
    with Session(create_database_engine(database)) as session:
        stored = session.scalars(
            select(RealAccountSnapshot).where(RealAccountSnapshot.account_id == account_id)
        ).all()
    assert len(stored) == 3


def test_trade_market_snapshot_uses_previous_trading_day_and_never_overwrites_actual_price(tmp_path: Path) -> None:
    service, account_id, database = _automatic_service(tmp_path)
    with Session(create_database_engine(database)) as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "589850"))
        assert instrument is not None
        session.add(
            ValuationRecord(
                instrument_id=instrument.id,
                valuation_date=date(2026, 7, 3),
                pe_ratio=Decimal("20"),
                pb_ratio=Decimal("2"),
                valuation_percentile=Decimal("0.25"),
            )
        )
        session.add(
            IndicatorRecord(
                instrument_id=instrument.id,
                indicator_date=date(2026, 7, 3),
                timeframe="daily",
                indicator_name="technical_indicators",
                indicator_values={"values": {"dif": "1", "dea": "0.5", "macd_histogram": "1", "rsi_6": "55"}},
            )
        )
    buy = _trade(service, account_id, date(2026, 7, 5), "BUY", "9")

    assert buy.price == Decimal("9")
    assert buy.market_snapshot["effective_market_date"] == "2026-07-03"
    assert buy.market_snapshot["non_trading_day_note"] == "该日期为非交易日，指标采用最近交易日数据：2026-07-03。"
    assert buy.market_snapshot["valuation_status"] == "低估区"
    assert buy.market_snapshot["macd_status"] == "金叉区 / 红柱"
    assert buy.market_snapshot["price_deviation_warning"] is True
    assert service.comparison_summary(account_id)["benchmark_total_assets"] is not None


def test_real_trade_blocks_missing_plan_and_sale_without_prior_holding(tmp_path: Path) -> None:
    service, account_id, database = _automatic_service(tmp_path)
    with Session(create_database_engine(database)) as session, session.begin():
        instrument = session.scalar(select(Instrument).where(Instrument.code == "589850"))
        assert instrument is not None
        plan = session.scalar(select(InvestmentPlan).where(InvestmentPlan.instrument_id == instrument.id))
        assert plan is not None
        session.delete(plan)
    with pytest.raises(ValueError, match="尚未设置关联定投金额"):
        _trade(service, account_id, date(2026, 7, 1), "BUY", "10")
    with pytest.raises(ValueError, match="当前没有可卖份额"):
        _trade(service, account_id, date(2026, 7, 1), "SELL", "10")


def test_update_and_delete_rebuild_automatic_projection_and_exports(tmp_path: Path) -> None:
    service, account_id, _database = _automatic_service(tmp_path)
    first = _trade(service, account_id, date(2026, 7, 1), "BUY", "10")
    service.update_transaction(account_id, first.id, RealTransactionUpdate(price=Decimal("8")))
    updated = service.transactions(account_id)[0]
    assert updated.quantity == Decimal("12.5")
    assert updated.holding_quantity_after == Decimal("12.5")
    assert "price" in service.export_csv(account_id).splitlines()[0]
    assert service.export_xlsx(account_id).startswith(b"PK")
    service.delete_transaction(account_id, first.id)
    with pytest.raises(RealAccountNoDataError):
        service.chart_series(account_id)
