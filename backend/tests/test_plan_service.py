from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.schemas.simulation import InvestmentPlanCreate
from backend.app.services.plan_service import InvestmentPlanService


def test_plan_crud_persists_weekly_execution_settings(tmp_path: Path) -> None:
    database = tmp_path / "investment_lab.db"
    initialize_database(database, tmp_path / "config")
    service = InvestmentPlanService(create_session_factory(database))

    created = service.create(
        InvestmentPlanCreate(
            instrument_code="589850",
            name="exact weekly plan",
            weekly_amount=Decimal("150.00"),
            execution_weekday=1,
            execution_price_rule="PREVIOUS_CLOSE",
            start_date=date(2026, 7, 1),
            minimum_weekly_amount=Decimal("100"),
            maximum_weekly_amount=Decimal("200"),
            allow_pause=True,
            mode="REAL_LOT",
            lot_size=100,
        )
    )

    loaded = service.get(created.id)
    assert loaded.weekly_amount == Decimal("150.00")
    assert loaded.execution_price_rule == "PREVIOUS_CLOSE"
    assert loaded.mode == "REAL_LOT"
    assert loaded.lot_size == 100
    assert loaded.allow_pause is True


@pytest.mark.parametrize(
    "values",
    [
        {"weekly_amount": Decimal("0")},
        {"execution_weekday": 7},
        {"minimum_weekly_amount": Decimal("200"), "maximum_weekly_amount": Decimal("100")},
        {"start_date": date(2026, 7, 9), "end_date": date(2026, 7, 1)},
        {"mode": "INVALID"},
    ],
)
def test_plan_schema_rejects_invalid_weekly_constraints(values: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "instrument_code": "589850",
        "name": "invalid plan",
        "weekly_amount": Decimal("150"),
        "execution_weekday": 1,
    }
    fields.update(values)

    with pytest.raises(ValueError):
        InvestmentPlanCreate(**fields)
