"""SQLite-backed investment plan CRUD with strict weekly-plan validation."""

from __future__ import annotations

from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.models import Instrument, InvestmentPlan
from ..schemas.simulation import InvestmentPlanCreate, InvestmentPlanRead, InvestmentPlanUpdate


class InvestmentPlanService:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    @staticmethod
    def _settings(plan: InvestmentPlan) -> dict[str, object]:
        settings = dict(plan.rule_parameters or {})
        return {
            "weekly_amount": plan.amount,
            "execution_weekday": plan.weekday if plan.weekday is not None else 1,
            "execution_price_rule": settings.get("execution_price_rule", "CLOSE"),
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "minimum_weekly_amount": settings.get("minimum_weekly_amount"),
            "maximum_weekly_amount": settings.get("maximum_weekly_amount"),
            "enabled": plan.enabled,
            "allow_pause": settings.get("allow_pause", True),
            "mode": settings.get("mode", "REAL_LOT"),
            "lot_size": settings.get("lot_size", 100),
        }

    @classmethod
    def _read(cls, plan: InvestmentPlan, instrument_code: str) -> InvestmentPlanRead:
        return InvestmentPlanRead(
            id=plan.id,
            instrument_id=plan.instrument_id,
            instrument_code=instrument_code,
            name=plan.name,
            **cls._settings(plan),
        )

    @staticmethod
    def _parameters(values: InvestmentPlanCreate | InvestmentPlanRead) -> dict[str, object]:
        return {
            "minimum_weekly_amount": str(values.minimum_weekly_amount)
            if values.minimum_weekly_amount is not None
            else None,
            "maximum_weekly_amount": str(values.maximum_weekly_amount)
            if values.maximum_weekly_amount is not None
            else None,
            "execution_price_rule": values.execution_price_rule,
            "allow_pause": values.allow_pause,
            "mode": values.mode,
            "lot_size": values.lot_size,
        }

    def create(self, values: InvestmentPlanCreate) -> InvestmentPlanRead:
        with self.session_factory() as session, session.begin():
            instrument = session.scalar(select(Instrument).where(Instrument.code == values.instrument_code))
            if instrument is None:
                raise ValueError(f"Unknown instrument code: {values.instrument_code}")
            if session.scalar(select(InvestmentPlan).where(InvestmentPlan.name == values.name)) is not None:
                raise ValueError(f"Investment plan already exists: {values.name}")
            plan = InvestmentPlan(
                instrument_id=instrument.id,
                name=values.name,
                frequency="weekly",
                amount=values.weekly_amount,
                weekday=values.execution_weekday,
                start_date=values.start_date,
                end_date=values.end_date,
                enabled=values.enabled,
                rule_parameters=self._parameters(values),
            )
            session.add(plan)
            session.flush()
            return self._read(plan, instrument.code)

    def get(self, plan_id: int) -> InvestmentPlanRead:
        with self.session_factory() as session:
            row = session.execute(
                select(InvestmentPlan, Instrument.code)
                .join(Instrument)
                .where(InvestmentPlan.id == plan_id)
            ).one_or_none()
            if row is None:
                raise ValueError(f"Unknown investment plan: {plan_id}")
            return self._read(row[0], row[1])

    def list(self, *, instrument_code: str | None = None) -> list[InvestmentPlanRead]:
        with self.session_factory() as session:
            statement = select(InvestmentPlan, Instrument.code).join(Instrument).order_by(InvestmentPlan.id)
            if instrument_code is not None:
                statement = statement.where(Instrument.code == instrument_code)
            return [self._read(plan, code) for plan, code in session.execute(statement).all()]

    def update(self, plan_id: int, changes: InvestmentPlanUpdate) -> InvestmentPlanRead:
        with self.session_factory() as session, session.begin():
            row = session.execute(
                select(InvestmentPlan, Instrument.code)
                .join(Instrument)
                .where(InvestmentPlan.id == plan_id)
            ).one_or_none()
            if row is None:
                raise ValueError(f"Unknown investment plan: {plan_id}")
            plan, instrument_code = row
            merged: dict[str, object] = {
                "instrument_code": instrument_code,
                "name": plan.name,
                **self._settings(plan),
            }
            merged.update(changes.model_dump(exclude_unset=True))
            validated = InvestmentPlanCreate(**merged)
            duplicate = session.scalar(
                select(InvestmentPlan).where(
                    InvestmentPlan.name == validated.name,
                    InvestmentPlan.id != plan_id,
                )
            )
            if duplicate is not None:
                raise ValueError(f"Investment plan already exists: {validated.name}")
            plan.name = validated.name
            plan.frequency = "weekly"
            plan.amount = validated.weekly_amount
            plan.weekday = validated.execution_weekday
            plan.start_date = validated.start_date
            plan.end_date = validated.end_date
            plan.enabled = validated.enabled
            plan.rule_parameters = self._parameters(validated)
            session.flush()
            return self._read(plan, instrument_code)

    def delete(self, plan_id: int) -> None:
        with self.session_factory() as session, session.begin():
            plan = session.get(InvestmentPlan, plan_id)
            if plan is None:
                raise ValueError(f"Unknown investment plan: {plan_id}")
            session.delete(plan)
