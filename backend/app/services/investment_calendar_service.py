from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import time
from typing import Callable, TypeVar

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from backend.app.errors import (
    PublicConflictError,
    PublicDataUnavailableError,
    PublicNotFoundError,
    PublicValidationError,
)
from backend.app.models.models import (
    Instrument,
    V2PositionEvent,
    V2PositionSnapshot,
)
from backend.app.schemas.investment_calendar import PositionEventCreate, PositionEventUpdate


SUPPORTED_INDEXES = ("399006", "159941")
MAX_WRITE_RETRIES = 3
WriteResult = TypeVar("WriteResult")


class InvestmentCalendarService:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self.session_factory = session_factory

    @staticmethod
    def _is_sqlite_lock_error(error: OperationalError) -> bool:
        message = str(error.orig).lower()
        return any(
            marker in message
            for marker in (
                "database is locked",
                "database table is locked",
                "database schema is locked",
                "database is busy",
            )
        )

    @staticmethod
    def _is_sequence_race(error: IntegrityError) -> bool:
        message = str(error.orig).lower()
        return (
            "uq_v2_position_event_sequence" in message
            or (
                "unique constraint failed" in message
                and "v2_position_events.instrument_id" in message
                and "v2_position_events.operation_date" in message
                and "v2_position_events.sequence" in message
            )
        )

    def _run_write(
        self,
        operation: Callable[[Session], WriteResult],
    ) -> WriteResult:
        for retry_number in range(MAX_WRITE_RETRIES + 1):
            with self.session_factory() as session:
                is_sqlite = session.get_bind().dialect.name == "sqlite"
                try:
                    if is_sqlite:
                        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                    else:
                        session.begin()
                    result = operation(session)
                    session.commit()
                    return result
                except OperationalError as error:
                    session.rollback()
                    can_retry = (
                        is_sqlite
                        and self._is_sqlite_lock_error(error)
                        and retry_number < MAX_WRITE_RETRIES
                    )
                    if not can_retry:
                        raise
                except IntegrityError as error:
                    session.rollback()
                    if not (
                        self._is_sequence_race(error)
                        and retry_number < MAX_WRITE_RETRIES
                    ):
                        raise
                except Exception:
                    session.rollback()
                    raise
            time.sleep(0.02 * (retry_number + 1))
        raise RuntimeError("unreachable write retry state")

    @staticmethod
    def _instrument(session: Session, code: str) -> Instrument:
        if code not in SUPPORTED_INDEXES:
            raise PublicValidationError(
                f"不支持的标的：{code}，V3.3仓位日历仅支持399006和159941"
            )
        instrument = session.scalar(select(Instrument).where(Instrument.code == code))
        if instrument is None:
            raise PublicDataUnavailableError(f"指数不存在：{code}")
        return instrument

    @staticmethod
    def _validated_percent(value: object) -> int:
        try:
            percent = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise PublicValidationError(
                "仓位变动百分比必须是5到100之间的5倍数"
            ) from None
        if (
            percent != percent.to_integral_value()
            or percent < 5
            or percent > 100
            or percent % 5 != 0
        ):
            raise PublicValidationError(
                "仓位变动百分比必须是5到100之间的5倍数"
            )
        return int(percent)

    @classmethod
    def _row(
        cls,
        event: V2PositionEvent,
        instrument_code: str,
        position_after: Decimal | int | None,
    ) -> dict[str, object]:
        return {
            "id": event.id,
            "index": instrument_code,
            "direction": event.direction,
            "date": event.operation_date,
            "change_percent": cls._validated_percent(event.change_percent),
            "sequence": event.sequence,
            "position_after": (
                None if position_after is None else int(Decimal(position_after))
            ),
            "note": event.note,
            "created_at": event.created_at,
            "updated_at": event.updated_at,
        }

    @staticmethod
    def _delete_snapshots(session: Session, instrument_id: int) -> None:
        event_ids = select(V2PositionEvent.id).where(
            V2PositionEvent.instrument_id == instrument_id
        )
        session.execute(
            delete(V2PositionSnapshot).where(
                V2PositionSnapshot.position_event_id.in_(event_ids)
            )
        )

    @classmethod
    def _replay(cls, session: Session, instrument_id: int) -> int | None:
        cls._delete_snapshots(session, instrument_id)
        events = session.scalars(
            select(V2PositionEvent)
            .where(V2PositionEvent.instrument_id == instrument_id)
            .order_by(
                V2PositionEvent.operation_date.asc(),
                V2PositionEvent.sequence.asc(),
                V2PositionEvent.created_at.asc(),
                V2PositionEvent.id.asc(),
            )
        ).all()
        if not events:
            return None

        position = 0
        for index, event in enumerate(events):
            change = cls._validated_percent(event.change_percent)
            next_position = (
                position + change
                if event.direction == "increase"
                else position - change
            )
            if index == 0 and event.direction != "increase":
                raise PublicConflictError(
                    "仓位冲突："
                    f"{event.operation_date.isoformat()} 第一条事件必须为increase，"
                    f"结果仓位为{next_position}%"
                )
            if next_position < 0 or next_position > 100:
                raise PublicConflictError(
                    "仓位冲突："
                    f"{event.operation_date.isoformat()} 结果仓位为{next_position}%"
                    "，允许范围为0%到100%"
                )
            position = next_position
            session.add(
                V2PositionSnapshot(
                    position_event_id=event.id,
                    position_percent=position,
                    status="current",
                    payload={},
                    audit_data={},
                )
            )
        session.flush()
        return position

    @staticmethod
    def _next_sequence(
        session: Session,
        instrument_id: int,
        operation_date: date,
    ) -> int:
        current = session.scalar(
            select(func.max(V2PositionEvent.sequence)).where(
                V2PositionEvent.instrument_id == instrument_id,
                V2PositionEvent.operation_date == operation_date,
            )
        )
        return int(current or 0) + 1

    @staticmethod
    def _compact_sequences(
        session: Session,
        instrument_id: int,
        operation_date: date,
    ) -> None:
        events = session.scalars(
            select(V2PositionEvent)
            .where(
                V2PositionEvent.instrument_id == instrument_id,
                V2PositionEvent.operation_date == operation_date,
            )
            .order_by(
                V2PositionEvent.sequence.asc(),
                V2PositionEvent.created_at.asc(),
                V2PositionEvent.id.asc(),
            )
        ).all()
        current_max = max((event.sequence for event in events), default=0)
        for offset, event in enumerate(events, start=1):
            event.sequence = current_max + offset
        session.flush()
        for sequence, event in enumerate(events, start=1):
            event.sequence = sequence
        session.flush()

    @classmethod
    def _event_row(cls, session: Session, event_id: int) -> dict[str, object]:
        result = session.execute(
            select(
                V2PositionEvent,
                Instrument.code,
                V2PositionSnapshot.position_percent,
            )
            .join(Instrument, Instrument.id == V2PositionEvent.instrument_id)
            .outerjoin(
                V2PositionSnapshot,
                V2PositionSnapshot.position_event_id == V2PositionEvent.id,
            )
            .where(V2PositionEvent.id == event_id)
        ).one()
        return cls._row(result[0], result[1], result[2])

    def create(self, values: PositionEventCreate) -> dict[str, object]:
        def write(session: Session) -> dict[str, object]:
            instrument = self._instrument(session, values.instrument_code)
            change_percent = self._validated_percent(values.change_percent)
            event = V2PositionEvent(
                instrument_id=instrument.id,
                direction=values.direction,
                operation_date=values.operation_date,
                change_percent=change_percent,
                note=values.note,
                sequence=self._next_sequence(
                    session,
                    instrument.id,
                    values.operation_date,
                ),
                status="recorded",
                payload={},
                audit_data={},
            )
            session.add(event)
            session.flush()
            self._replay(session, instrument.id)
            return self._event_row(session, event.id)

        return self._run_write(write)

    def list_entries(self, instrument_code: str | None = None) -> list[dict[str, object]]:
        with self.session_factory() as session:
            query = (
                select(
                    V2PositionEvent,
                    Instrument.code,
                    V2PositionSnapshot.position_percent,
                )
                .join(Instrument, Instrument.id == V2PositionEvent.instrument_id)
                .outerjoin(
                    V2PositionSnapshot,
                    V2PositionSnapshot.position_event_id == V2PositionEvent.id,
                )
                .order_by(
                    V2PositionEvent.operation_date.desc(),
                    V2PositionEvent.sequence.desc(),
                    V2PositionEvent.created_at.desc(),
                    V2PositionEvent.id.desc(),
                )
            )
            if instrument_code is not None:
                instrument = self._instrument(session, instrument_code)
                query = query.where(V2PositionEvent.instrument_id == instrument.id)
            return [
                self._row(event, code, position)
                for event, code, position in session.execute(query).all()
            ]

    def update(
        self,
        entry_id: int,
        changes: PositionEventUpdate,
    ) -> dict[str, object]:
        def write(session: Session) -> dict[str, object]:
            event = session.get(V2PositionEvent, entry_id)
            if event is None:
                raise PublicNotFoundError(f"仓位事件不存在：{entry_id}")

            values = changes.model_dump(exclude_unset=True)
            for required_field in (
                "instrument_code",
                "direction",
                "operation_date",
                "change_percent",
            ):
                if required_field in values and values[required_field] is None:
                    raise PublicValidationError(
                        f"{required_field}不能为null"
                    )

            old_instrument_id = event.instrument_id
            old_operation_date = event.operation_date
            target_instrument = (
                self._instrument(session, values["instrument_code"])
                if "instrument_code" in values
                else session.get(Instrument, event.instrument_id)
            )
            if target_instrument is None:
                raise PublicDataUnavailableError(
                    f"指数不存在：{event.instrument_id}"
                )
            target_date = values.get("operation_date", event.operation_date)
            moved = (
                target_instrument.id != old_instrument_id
                or target_date != old_operation_date
            )
            target_sequence = (
                self._next_sequence(session, target_instrument.id, target_date)
                if moved
                else event.sequence
            )

            event.instrument_id = target_instrument.id
            event.operation_date = target_date
            event.sequence = target_sequence
            if "direction" in values:
                event.direction = values["direction"]
            if "change_percent" in values:
                event.change_percent = self._validated_percent(
                    values["change_percent"]
                )
            if "note" in values:
                event.note = values["note"]
            session.flush()

            if moved:
                self._compact_sequences(
                    session,
                    old_instrument_id,
                    old_operation_date,
                )

            affected_instruments = [old_instrument_id]
            if target_instrument.id != old_instrument_id:
                affected_instruments.append(target_instrument.id)
            for instrument_id in affected_instruments:
                self._replay(session, instrument_id)
            return self._event_row(session, event.id)

        return self._run_write(write)

    def delete(self, entry_id: int) -> None:
        def write(session: Session) -> None:
            event = session.get(V2PositionEvent, entry_id)
            if event is None:
                raise PublicNotFoundError(f"仓位事件不存在：{entry_id}")
            instrument_id = event.instrument_id
            operation_date = event.operation_date

            self._delete_snapshots(session, instrument_id)
            session.flush()
            session.delete(event)
            session.flush()
            self._compact_sequences(session, instrument_id, operation_date)
            self._replay(session, instrument_id)

        self._run_write(write)

    def current_positions(self) -> dict[str, int | None]:
        positions: dict[str, int | None] = {}
        with self.session_factory() as session:
            for code in SUPPORTED_INDEXES:
                instrument = self._instrument(session, code)
                value = session.scalar(
                    select(V2PositionSnapshot.position_percent)
                    .join(
                        V2PositionEvent,
                        V2PositionEvent.id
                        == V2PositionSnapshot.position_event_id,
                    )
                    .where(V2PositionEvent.instrument_id == instrument.id)
                    .order_by(
                        V2PositionEvent.operation_date.desc(),
                        V2PositionEvent.sequence.desc(),
                        V2PositionEvent.created_at.desc(),
                        V2PositionEvent.id.desc(),
                    )
                    .limit(1)
                )
                # A missing ledger is an explicit cleared position in V3.1.
                # This lets the first analysis produce actionable buy batches
                # while the UI still labels the source as the default 0% state.
                positions[code] = 0 if value is None else int(Decimal(value))
        return positions

    def net_shares(self) -> dict[str, int | None]:
        """Deprecated compatibility wrapper returning V2 position percentages."""
        return self.current_positions()
