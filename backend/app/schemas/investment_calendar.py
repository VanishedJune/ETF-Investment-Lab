from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


# Calendar entries are deliberately independent of the model pipeline.  Keep
# the complete chart/ETF catalog here so every shipped AI-assistant instrument
# can be recorded even though none of them triggers inference.
# Syntax here, authoritative slot/ledger membership in the service.
InstrumentCode = Annotated[str, Field(pattern=r"^(?:399006|[15]\d{5})$")]
PositionDirection = Literal["increase", "decrease"]


class PositionEventCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument_code: InstrumentCode
    direction: PositionDirection
    operation_date: date
    change_percent: int = Field(ge=5, le=100, multiple_of=5)
    note: str | None = Field(default=None, max_length=1000)


class PositionEventUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument_code: InstrumentCode | None = None
    direction: PositionDirection | None = None
    operation_date: date | None = None
    change_percent: int | None = Field(default=None, ge=5, le=100, multiple_of=5)
    note: str | None = Field(default=None, max_length=1000)


# Compatibility for project-local imports that have not yet adopted the V2 name.
CalendarEntryCreate = PositionEventCreate
CalendarEntryUpdate = PositionEventUpdate
