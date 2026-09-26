"""Data-only regression: no application lifespan, training, or live database."""
from datetime import date
from types import SimpleNamespace
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.models.models import Base, Instrument, V351InstrumentSlot, V2PositionEvent
from backend.app.services.v351_slot_service import seed_default_slots
from backend.app.services.instrument_universe import AI_ASSISTANT_ETFS, calendar_codes
from backend.app.services.investment_calendar_service import InvestmentCalendarService
from backend.app.schemas.investment_calendar import PositionEventCreate
from backend.web import instruments


@pytest.fixture
def factory():
    engine = create_engine('sqlite://', poolclass=StaticPool, connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session, session.begin():
        seed_default_slots(session)
        session.add(Instrument(code='512010', name='医药ETF', exchange='SSE', category='etf', currency='CNY'))
    yield factory
    engine.dispose()


def test_live_eight_follow_trading_priority(factory):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(sessions=factory)))
    rows = instruments(request)
    assert [row['code'] for row in rows if row['is_current_slot']] == list(AI_ASSISTANT_ETFS)
    assert [row['slot_order'] for row in rows if row['is_current_slot']] == list(range(1, 9))


def test_replacement_calendar_accepts_new_code_and_keeps_old_ledger(factory):
    service = InvestmentCalendarService(factory)
    old = service.create(PositionEventCreate(instrument_code='512010', direction='increase', operation_date=date(2026, 9, 1), change_percent=20))
    with factory() as session, session.begin():
        slot = session.scalar(select(V351InstrumentSlot).where(V351InstrumentSlot.instrument_code == '515220'))
        session.add(Instrument(code='515790', name='替换测试ETF', exchange='SSE', category='etf', currency='CNY'))
        slot.instrument_code = '515790'
    new = service.create(PositionEventCreate(instrument_code='515790', direction='increase', operation_date=date(2026, 9, 2), change_percent=30))
    assert service.current_positions()['512010'] == 20
    assert service.current_positions()['515790'] == 30
    assert service.list_entries('512010')[0]['id'] == old['id']
    service.update(new['id'], __import__('backend.app.schemas.investment_calendar', fromlist=['PositionEventUpdate']).PositionEventUpdate(change_percent=25))
    assert service.current_positions()['515790'] == 25
    # Fresh service (like reopening the app) sees the same persisted identities.
    assert InvestmentCalendarService(factory).current_positions()['515790'] == 25
    service.delete(new['id'])
    assert service.current_positions()['515790'] == 0
    assert service.current_positions()['512010'] == 20
    with factory() as session:
        assert '515790' in calendar_codes(session)
        assert session.get(V2PositionEvent, old['id']).instrument_id != 0


def test_unregistered_code_is_not_accepted(factory):
    with pytest.raises(Exception, match='不支持'):
        InvestmentCalendarService(factory).create(PositionEventCreate(
            instrument_code='123456', direction='increase', operation_date=date(2026, 9, 2), change_percent=20))


def test_seed_does_not_reset_user_replacement(factory):
    with factory() as session, session.begin():
        slot = session.scalar(select(V351InstrumentSlot).where(V351InstrumentSlot.slot_id == 'ETF_SLOT_05'))
        slot.instrument_code = '515790'
        seed_default_slots(session)
    with factory() as session:
        assert session.scalar(select(V351InstrumentSlot.instrument_code).where(V351InstrumentSlot.slot_id == 'ETF_SLOT_05')) == '515790'
