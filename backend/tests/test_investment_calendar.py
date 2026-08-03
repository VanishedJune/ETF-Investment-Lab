from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Barrier
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database.initialize import initialize_database
from backend.app.database.session import create_session_factory
from backend.app.models.models import (
    Base,
    Instrument,
    V2PositionEvent,
    V2PositionSnapshot,
)
from backend.app.schemas.investment_calendar import PositionEventCreate, PositionEventUpdate
from backend.app.services.investment_calendar_service import InvestmentCalendarService


def _service(tmp_path, *, run_full_initialization: bool = False):
    database = tmp_path / "data" / "investment_lab.db"
    if run_full_initialization:
        initialize_database(database, tmp_path / "config")
        factory = create_session_factory(database)
    else:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        with Session(engine) as session, session.begin():
            session.add_all(
                [
                    Instrument(
                        code="399006",
                        name="创业板指数",
                        exchange="SZSE",
                        category="index",
                        currency="CNY",
                    ),
                    Instrument(
                        code="159941",
                        name="纳斯达克100指数",
                        exchange="NASDAQ",
                        category="index",
                        currency="USD",
                    ),
                ]
            )
        factory = sessionmaker(bind=engine, expire_on_commit=False)
    return InvestmentCalendarService(factory), database, factory


def _event(
    *,
    instrument_code: str = "399006",
    direction: str = "increase",
    operation_date: date = date(2026, 7, 1),
    change_percent: int = 20,
    note: str | None = None,
) -> PositionEventCreate:
    return PositionEventCreate(
        instrument_code=instrument_code,
        direction=direction,
        operation_date=operation_date,
        change_percent=change_percent,
        note=note,
    )


def _concurrent_service():
    database_name = f"position-events-{uuid4().hex}"
    engine = create_engine(
        "sqlite+pysqlite:///"
        f"file:{database_name}?mode=memory&cache=shared&uri=true",
        connect_args={"check_same_thread": False, "timeout": 1},
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Instrument(
                    code="399006",
                    name="创业板指数",
                    exchange="SZSE",
                    category="index",
                    currency="CNY",
                ),
                Instrument(
                    code="159941",
                    name="纳斯达克100指数",
                    exchange="NASDAQ",
                    category="index",
                    currency="USD",
                ),
            ]
        )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return InvestmentCalendarService(factory), factory, engine


def test_position_event_crud_persists_and_reopens_from_sqlite(tmp_path) -> None:
    service, database, factory = _service(tmp_path, run_full_initialization=True)
    first = service.create(
        _event(operation_date=date(2026, 7, 1), change_percent=20, note="首次建仓")
    )
    second = service.create(
        _event(operation_date=date(2026, 7, 8), change_percent=15, note="继续加仓")
    )

    updated = service.update(
        second["id"],
        PositionEventUpdate(change_percent=25, note="调整后的加仓"),
    )
    reopened = InvestmentCalendarService(create_session_factory(database))

    assert first == {
        "id": first["id"],
        "index": "399006",
        "direction": "increase",
        "date": date(2026, 7, 1),
        "change_percent": 20,
        "sequence": 1,
        "position_after": 20,
        "note": "首次建仓",
        "created_at": first["created_at"],
        "updated_at": first["updated_at"],
    }
    assert updated["position_after"] == 45
    assert reopened.current_positions() == {"399006": 45, "159941": 0}
    assert [row["position_after"] for row in reopened.list_entries("399006")] == [45, 20]

    with factory() as session:
        assert session.scalar(select(func.count()).select_from(V2PositionEvent)) == 2
        assert session.scalar(select(func.count()).select_from(V2PositionSnapshot)) == 2

    reopened.delete(first["id"])
    assert reopened.list_entries("399006") == [
        {
            **updated,
            "position_after": 25,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instrument_code", "000688"),
        ("direction", "hold"),
        ("change_percent", 7),
        ("change_percent", 0),
        ("change_percent", 105),
    ],
)
def test_position_event_schema_rejects_invalid_contract_values(field, value) -> None:
    values = {
        "instrument_code": "159941",
        "direction": "increase",
        "operation_date": date(2026, 7, 1),
        "change_percent": 10,
    }

    with pytest.raises(ValidationError):
        PositionEventCreate(**{**values, field: value})


def test_no_events_return_cleared_zero_current_positions(tmp_path) -> None:
    service, _database, _factory = _service(tmp_path)

    assert service.list_entries() == []
    assert service.current_positions() == {"399006": 0, "159941": 0}
    assert service.net_shares() == {"399006": 0, "159941": 0}


def test_service_defensively_rejects_non_multiple_percent_and_rolls_back(tmp_path) -> None:
    service, _database, factory = _service(tmp_path)
    bypassed_schema = PositionEventCreate.model_construct(
        instrument_code="399006",
        direction="increase",
        operation_date=date(2026, 7, 1),
        change_percent=7,
        note=None,
    )

    with pytest.raises(ValueError, match="5到100之间的5倍数"):
        service.create(bypassed_schema)

    with factory() as session:
        assert session.scalar(select(func.count()).select_from(V2PositionEvent)) == 0
        assert session.scalar(select(func.count()).select_from(V2PositionSnapshot)) == 0


def test_first_event_must_increase_and_failed_create_rolls_back(tmp_path) -> None:
    service, _database, factory = _service(tmp_path)

    with pytest.raises(ValueError, match=r"2026-07-01.*-20%"):
        service.create(_event(direction="decrease", change_percent=20))

    assert service.current_positions()["399006"] == 0
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(V2PositionEvent)) == 0
        assert session.scalar(select(func.count()).select_from(V2PositionSnapshot)) == 0


def test_backfilled_earlier_event_replays_in_chronological_order(tmp_path) -> None:
    service, _database, _factory = _service(tmp_path)
    later = service.create(_event(operation_date=date(2026, 7, 20), change_percent=50))

    earlier = service.create(_event(operation_date=date(2026, 7, 10), change_percent=30))
    rows = service.list_entries("399006")

    assert [(row["id"], row["position_after"]) for row in rows] == [
        (later["id"], 80),
        (earlier["id"], 30),
    ]
    assert service.current_positions()["399006"] == 80


def test_update_and_delete_replay_every_remaining_event(tmp_path) -> None:
    service, _database, factory = _service(tmp_path)
    first = service.create(_event(operation_date=date(2026, 7, 1), change_percent=30))
    middle = service.create(_event(operation_date=date(2026, 7, 2), change_percent=20))
    last = service.create(
        _event(
            direction="decrease",
            operation_date=date(2026, 7, 3),
            change_percent=10,
        )
    )

    service.update(middle["id"], PositionEventUpdate(change_percent=40))
    assert [row["position_after"] for row in service.list_entries("399006")] == [60, 70, 30]

    service.delete(middle["id"])
    assert [row["position_after"] for row in service.list_entries("399006")] == [20, 30]
    assert service.current_positions()["399006"] == 20

    with factory() as session:
        snapshots = session.scalars(select(V2PositionSnapshot)).all()
        event_ids = set(session.scalars(select(V2PositionEvent.id)))
        assert len(snapshots) == 2
        assert {snapshot.position_event_id for snapshot in snapshots} == event_ids
        assert last["id"] in event_ids
        assert first["id"] in event_ids


def test_overflow_create_update_and_invalid_delete_roll_back_entire_transaction(tmp_path) -> None:
    service, _database, factory = _service(tmp_path)
    first = service.create(_event(operation_date=date(2026, 7, 1), change_percent=60))
    second = service.create(_event(operation_date=date(2026, 7, 2), change_percent=30))

    with pytest.raises(ValueError, match=r"2026-07-03.*120%"):
        service.create(
            _event(operation_date=date(2026, 7, 3), change_percent=30)
        )
    with pytest.raises(ValueError, match=r"2026-07-02.*110%"):
        service.update(second["id"], PositionEventUpdate(change_percent=50))
    service.update(second["id"], PositionEventUpdate(direction="decrease"))
    with pytest.raises(ValueError, match=r"2026-07-02.*-30%"):
        service.delete(first["id"])

    assert [
        (row["id"], row["change_percent"], row["position_after"])
        for row in service.list_entries("399006")
    ] == [
        (second["id"], 30, 30),
        (first["id"], 60, 60),
    ]
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(V2PositionEvent)) == 2
        assert session.scalar(select(func.count()).select_from(V2PositionSnapshot)) == 2


def test_same_day_sequence_is_automatic_reassigned_and_compacted(tmp_path) -> None:
    service, _database, _factory = _service(tmp_path)
    day_one = date(2026, 7, 1)
    day_two = date(2026, 7, 2)
    first = service.create(_event(operation_date=day_one, change_percent=10))
    second = service.create(_event(operation_date=day_one, change_percent=10))
    third = service.create(_event(operation_date=day_one, change_percent=10))
    target = service.create(_event(operation_date=day_two, change_percent=10))

    moved = service.update(first["id"], PositionEventUpdate(operation_date=day_two))

    assert moved["sequence"] == 2
    assert [
        (row["id"], row["sequence"])
        for row in service.list_entries("399006")
        if row["date"] == day_one
    ] == [(third["id"], 2), (second["id"], 1)]
    assert [
        (row["id"], row["sequence"])
        for row in service.list_entries("399006")
        if row["date"] == day_two
    ] == [(first["id"], 2), (target["id"], 1)]

    service.delete(second["id"])
    assert [
        row["sequence"]
        for row in service.list_entries("399006")
        if row["date"] == day_one
    ] == [1]


def test_two_indexes_replay_in_complete_isolation(tmp_path) -> None:
    service, _database, _factory = _service(tmp_path)
    cn = service.create(_event(change_percent=40))
    us = service.create(
        _event(instrument_code="159941", change_percent=70, note="纳指仓位")
    )

    service.update(cn["id"], PositionEventUpdate(change_percent=50))
    assert service.current_positions() == {"399006": 50, "159941": 70}

    service.delete(cn["id"])
    assert service.current_positions() == {"399006": 0, "159941": 70}
    assert service.list_entries("159941")[0]["id"] == us["id"]
    assert service.list_entries("159941")[0]["position_after"] == 70


def test_moving_event_to_other_index_appends_sequence_and_replays_both(tmp_path) -> None:
    service, _database, factory = _service(tmp_path)
    same_day = date(2026, 7, 1)
    us = service.create(
        _event(
            instrument_code="159941",
            operation_date=same_day,
            change_percent=20,
        )
    )
    moved = service.create(
        _event(operation_date=same_day, change_percent=30)
    )
    remaining_cn = service.create(
        _event(operation_date=date(2026, 7, 2), change_percent=10)
    )

    result = service.update(
        moved["id"],
        PositionEventUpdate(instrument_code="159941"),
    )

    assert result["index"] == "159941"
    assert result["sequence"] == 2
    assert result["position_after"] == 50
    assert service.current_positions() == {"399006": 10, "159941": 50}
    assert [
        (row["id"], row["sequence"], row["position_after"])
        for row in service.list_entries("159941")
    ] == [
        (moved["id"], 2, 50),
        (us["id"], 1, 20),
    ]
    assert service.list_entries("399006")[0]["id"] == remaining_cn["id"]

    with factory() as session:
        snapshots = session.scalars(select(V2PositionSnapshot)).all()
        assert len(snapshots) == 3
        assert len({snapshot.position_event_id for snapshot in snapshots}) == 3


def test_sequence_compaction_avoids_unique_collision_when_ids_and_sequences_reverse(
    tmp_path,
) -> None:
    service, _database, _factory = _service(tmp_path)
    day_one = date(2026, 7, 1)
    day_two = date(2026, 7, 2)
    smaller_id = service.create(_event(operation_date=day_one, change_percent=10))
    sequence_one = service.create(_event(operation_date=day_two, change_percent=10))
    later_id = service.create(_event(operation_date=day_two, change_percent=10))
    moved = service.update(
        smaller_id["id"],
        PositionEventUpdate(operation_date=day_two),
    )
    assert moved["sequence"] == 3

    service.delete(sequence_one["id"])

    rows = service.list_entries("399006")
    assert [(row["id"], row["sequence"]) for row in rows] == [
        (smaller_id["id"], 2),
        (later_id["id"], 1),
    ]
    assert service.current_positions()["399006"] == 20


def test_concurrent_same_day_creates_serialize_sequences_and_snapshots() -> None:
    service, factory, engine = _concurrent_service()
    start = Barrier(3)

    def create_event(note: str):
        start.wait(timeout=5)
        return service.create(_event(change_percent=10, note=note))

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(create_event, "thread-one"),
                executor.submit(create_event, "thread-two"),
            ]
            start.wait(timeout=5)
            created = [future.result(timeout=10) for future in futures]

        assert sorted(row["sequence"] for row in created) == [1, 2]
        assert service.current_positions()["399006"] == 20
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(V2PositionEvent)) == 2
            assert session.scalar(select(func.count()).select_from(V2PositionSnapshot)) == 2
            event_ids = set(session.scalars(select(V2PositionEvent.id)))
            snapshot_event_ids = set(
                session.scalars(select(V2PositionSnapshot.position_event_id))
            )
            assert snapshot_event_ids == event_ids
    finally:
        engine.dispose()


def test_concurrent_failed_replay_rolls_back_without_partial_snapshots() -> None:
    service, factory, engine = _concurrent_service()
    base = service.create(_event(change_percent=80))
    start = Barrier(3)

    def create_event(values):
        start.wait(timeout=5)
        return service.create(values)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            invalid = executor.submit(
                create_event,
                _event(
                    operation_date=date(2026, 7, 2),
                    change_percent=30,
                ),
            )
            valid = executor.submit(
                create_event,
                _event(
                    instrument_code="159941",
                    operation_date=date(2026, 7, 2),
                    change_percent=20,
                ),
            )
            start.wait(timeout=5)
            with pytest.raises(ValueError, match="110%"):
                invalid.result(timeout=10)
            valid_result = valid.result(timeout=10)

        assert valid_result["position_after"] == 20
        assert service.current_positions() == {"399006": 80, "159941": 20}
        with factory() as session:
            events = session.scalars(select(V2PositionEvent)).all()
            snapshots = session.scalars(select(V2PositionSnapshot)).all()
            assert {event.id for event in events} == {base["id"], valid_result["id"]}
            assert {snapshot.position_event_id for snapshot in snapshots} == {
                base["id"],
                valid_result["id"],
            }
    finally:
        engine.dispose()
