from decimal import Decimal

import pytest

from backend.app.simulation.fees import FeeConfiguration, calculate_trade_fees


def test_etf_buy_fee_uses_enabled_minimum_commission_exactly() -> None:
    fees = calculate_trade_fees(
        Decimal("1000"),
        "BUY",
        FeeConfiguration(),
    )

    assert fees.commission == Decimal("5")
    assert fees.stamp_duty == Decimal("0")
    assert fees.total == Decimal("5")


def test_minimum_commission_can_be_disabled() -> None:
    fees = calculate_trade_fees(
        Decimal("1000"),
        "SELL",
        FeeConfiguration(minimum_commission_enabled=False),
    )

    assert fees.commission == Decimal("0.25000")
    assert fees.total == Decimal("0.25000")


@pytest.mark.parametrize("invalid", ["NaN", "Infinity", "-0.01"])
def test_fee_configuration_rejects_nonfinite_or_negative_rates(invalid: str) -> None:
    with pytest.raises(ValueError):
        FeeConfiguration(buy_commission_rate=Decimal(invalid))
