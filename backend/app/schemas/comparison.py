"""Chart-ready typed contracts for real-versus-simulation comparisons."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field


class ComparisonSeriesPoint(BaseModel):
    date: date
    real_contribution: Decimal | None
    simulation_contribution: Decimal | None
    contribution_difference: Decimal | None
    real_total_assets: Decimal | None
    simulation_total_assets: Decimal | None
    total_assets_difference: Decimal | None
    real_cash: Decimal | None
    simulation_cash: Decimal | None
    cash_difference: Decimal | None
    real_market_value: Decimal | None
    simulation_market_value: Decimal | None
    market_value_difference: Decimal | None
    real_total_return: Decimal | None
    simulation_total_return: Decimal | None
    total_return_difference: Decimal | None
    real_time_weighted_return: Decimal | None
    simulation_time_weighted_return: Decimal | None
    time_weighted_return_difference: Decimal | None
    real_money_weighted_return: Decimal | None
    simulation_money_weighted_return: Decimal | None
    money_weighted_return_difference: Decimal | None
    real_annualized_return: Decimal | None
    simulation_annualized_return: Decimal | None
    annualized_return_difference: Decimal | None
    real_max_drawdown: Decimal | None
    simulation_max_drawdown: Decimal | None
    max_drawdown_difference: Decimal | None
    real_fees: Decimal | None
    simulation_fees: Decimal | None
    fees_difference: Decimal | None


class AccountComparisonRead(BaseModel):
    real_account_id: int
    simulation_account_id: int
    series: list[ComparisonSeriesPoint] = Field(default_factory=list)

