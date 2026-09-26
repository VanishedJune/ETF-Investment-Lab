"""Explicit, one-time data-only directory alignment. No model workflow calls."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select
from backend.app.database.session import create_session_factory
from backend.app.models.models import Instrument, MarketPrice, V351InstrumentSlot, V351InstrumentSlotHistory, utc_now
from backend.app.services.v351_slot_service import DEFAULT_SLOTS, seed_default_slots
from backend.app.services.instrument_universe import AI_ASSISTANT_ETFS
from backend.app.services.indicator_service import IndicatorService
from backend.app.services.market_data import MarketDataService


def ledger_rows(connection):
    tables = ['v2_position_events', 'v2_position_snapshots']
    return {name: connection.execute(f'SELECT * FROM {name} ORDER BY id').fetchall() for name in tables}


def align(database: Path, source: Path, apply: bool):
    priority = json.loads((source / 'model_iteration/configs/investment_priority.json').read_text(encoding='utf-8'))['priority']
    if priority != list(AI_ASSISTANT_ETFS):
        raise ValueError('Trading priority differs from the reviewed catalog; no changes made.')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        before_ledger = ledger_rows(conn)
        before_slots = conn.execute('SELECT slot_id,instrument_code,slot_order FROM v351_instrument_slots ORDER BY slot_order').fetchall()
        existing_codes = {row[0] for row in conn.execute('SELECT code FROM instruments')}
    specs = {s['slot_id']: s for s in DEFAULT_SLOTS}
    allowed_old = {'ETF_SLOT_01': '159941', 'ETF_SLOT_02': '518600', 'ETF_SLOT_05': '512010'}
    for slot_id, code, _ in before_slots:
        if slot_id not in specs or code not in {specs[slot_id]['instrument_code'], allowed_old.get(slot_id)}:
            raise ValueError(f'Unexpected user binding {slot_id}: {code}; preserve it for review.')
    new_codes = [code for code in priority if code not in existing_codes]
    imported = {}
    for code in new_codes:
        paths = list((source / '数据').glob(f'{code}_*_日线.csv'))
        if len(paths) != 1:
            raise ValueError(f'Expected one daily OHLCV file for {code}')
        with paths[0].open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        prior = ''
        for row in rows:
            if row['code'] != code or row['date'] <= prior or date.fromisoformat(row['date']) > date.today():
                raise ValueError(f'Invalid code/date sequence: {code}')
            values = [Decimal(row[k]) for k in ('raw_open', 'raw_high', 'raw_low', 'raw_close', 'adj_close')]
            if any(not x.is_finite() or x <= 0 for x in values):
                raise ValueError(f'Invalid price for {code} {row["date"]}')
            o, h, l, c, _ = values
            if not l <= min(o, c) <= max(o, c) <= h:
                raise ValueError(f'Invalid OHLC for {code}')
            prior = row['date']
        if not rows:
            raise ValueError(f'Empty daily data: {code}')
        imported[code] = rows
    report = {'priority': priority, 'before_slots': before_slots,
              'imports': {code: {'rows': len(rows), 'cutoff': rows[-1]['date']} for code, rows in imported.items()},
              'mode': 'apply' if apply else 'check'}
    if not apply:
        return report
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    backup = database.parent / 'backups' / f'catalog-before-{stamp}.db'
    backup.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(backup) as dest:
        src.backup(dest)
    factory = create_session_factory(database)
    with factory() as session, session.begin():
        seed_default_slots(session)
        now = utc_now()
        for slot in session.scalars(select(V351InstrumentSlot)).all():
            spec = specs[slot.slot_id]
            code = spec['instrument_code']
            instrument = session.scalar(select(Instrument).where(Instrument.code == code))
            if instrument is None:
                instrument = Instrument(code=code, name=spec['instrument_name'], exchange=spec['exchange'],
                    category='etf', currency='CNY', is_active=True,
                    extra_data={'role': 'display_only', 'model_eligible': False, 'advice_eligible': False,
                                'price_basis': 'raw_OHLC_and_separate_adjusted_close', 'source': str(source)})
                session.add(instrument)
                session.flush()
            if slot.instrument_code != code:
                old = slot.instrument_code
                binding = session.scalar(select(V351InstrumentSlotHistory).where(
                    V351InstrumentSlotHistory.slot_id == slot.slot_id,
                    V351InstrumentSlotHistory.binding_version == slot.binding_version))
                if binding:
                    binding.effective_to = now
                slot.binding_version = f'CATALOG-{stamp}'
                slot.replaced_from_code = old
                slot.bound_at = now
                session.add(V351InstrumentSlotHistory(slot_id=slot.slot_id, binding_version=slot.binding_version,
                    instrument_code=code, action='REPLACE', effective_from=now,
                    reason=f'User-approved trading catalog alignment: {old} -> {code}; no ledger transfer', created_at=now))
            slot.instrument_code = code
            slot.instrument_name = spec['instrument_name']
            slot.exchange = spec['exchange']
            slot.slot_order = priority.index(code) + 1
            slot.model_enabled = False
            slot.model_status = 'DATA_ONLY'
            slot.active = True
            slot.replacement_status = 'READY'
            slot.updated_at = now
            for row in imported.get(code, []):
                session.add(MarketPrice(instrument_id=instrument.id, trade_date=date.fromisoformat(row['date']),
                    timeframe='daily', open_price=Decimal(row['raw_open']), high_price=Decimal(row['raw_high']),
                    low_price=Decimal(row['raw_low']), close_price=Decimal(row['raw_close']),
                    adjusted_close_price=Decimal(row['adj_close']),
                    volume=Decimal(row['volume']) if row['volume'] else None,
                    turnover=Decimal(row['amount']) if row['amount'] else None, volume_multiplier=1,
                    source='AI_ASSISTANT_CSV', volume_source='AI_ASSISTANT_CSV'))
    try:
        market = MarketDataService(factory)
        indicators = IndicatorService(factory)
        for code in imported:
            market.aggregate_periods(code, as_of=date.fromisoformat(imported[code][-1]['date']))
            results = indicators.recalculate_all_timeframes(code)
            if any(result.status.value != 'success' for result in results):
                raise RuntimeError(f'Indicator calculation failed: {code}')
        with factory() as session, session.begin():
            for slot in session.scalars(select(V351InstrumentSlot)):
                instrument = session.scalar(select(Instrument).where(Instrument.code == slot.instrument_code))
                slot.data_start_date, slot.data_end_date = session.execute(select(func.min(MarketPrice.trade_date), func.max(MarketPrice.trade_date)).where(
                    MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == 'daily')).one()
                slot.history_week_count = session.scalar(select(func.count()).select_from(MarketPrice).where(
                    MarketPrice.instrument_id == instrument.id, MarketPrice.timeframe == 'weekly'))
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
            if ledger_rows(conn) != before_ledger:
                raise RuntimeError('Ledger changed unexpectedly. Stop and inspect retained backup.')
        report['ledger_unchanged'] = True
        report['backup'] = str(backup)
        return report
    except Exception:
        # Keep the original backup; do not overwrite potentially concurrent user writes.
        print(f'Alignment requires attention; recoverable database: {backup}', file=sys.stderr)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--database', type=Path, default=ROOT / 'data/investment_lab.db')
    parser.add_argument('--source', type=Path, default=ROOT.parent / 'AI-Investment-Assistant')
    args = parser.parse_args()
    print(json.dumps(align(args.database, args.source, args.apply), ensure_ascii=False, indent=2))
