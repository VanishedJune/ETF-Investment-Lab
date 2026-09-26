"""Instrument roles for the data-only chart and investment-calendar UI."""

from __future__ import annotations

from typing import Final


MODEL_MARKETS: Final[frozenset[str]] = frozenset({"399006", "159941"})

# The eight instruments shipped by the sibling AI Investment Assistant.  In
# this build they are data/catalog entries only; no model or AI permission is
# implied by their presence here.
AI_ASSISTANT_ETFS: Final[dict[str, dict[str, str]]] = {
    "159915": {"name": "创业板ETF易方达", "exchange": "SZSE", "exchange_prefix": "sz"},
    "517520": {"name": "黄金股ETF永赢", "exchange": "SSE", "exchange_prefix": "sh"},
    "516150": {"name": "稀土ETF嘉实", "exchange": "SSE", "exchange_prefix": "sh"},
    "159622": {"name": "创新药ETF东财", "exchange": "SZSE", "exchange_prefix": "sz"},
    "159611": {"name": "电力ETF广发", "exchange": "SZSE", "exchange_prefix": "sz"},
    "515220": {"name": "煤炭ETF国泰", "exchange": "SSE", "exchange_prefix": "sh"},
    "512800": {"name": "华宝银行ETF", "exchange": "SSE", "exchange_prefix": "sh"},
    "512690": {"name": "鹏华酒ETF", "exchange": "SSE", "exchange_prefix": "sh"},
}

LEGACY_CALENDAR_CODES = frozenset({"399006", "159941", "518600", "512010"})


def current_slot_rows(session):
    """The persisted bindings, not an old UI whitelist, own the live order."""
    from sqlalchemy import inspect, select
    from backend.app.models.models import V351InstrumentSlot
    if not inspect(session.get_bind()).has_table(V351InstrumentSlot.__tablename__):
        return []
    return list(session.scalars(select(V351InstrumentSlot).where(
        V351InstrumentSlot.active.is_(True)
    ).order_by(V351InstrumentSlot.slot_order, V351InstrumentSlot.id)))


def calendar_codes(session):
    """Current bindings plus original ledger identities; never relabel trades."""
    from sqlalchemy import select
    from backend.app.models.models import Instrument, V2PositionEvent
    slots = current_slot_rows(session)
    active = [row.instrument_code for row in slots] or list(AI_ASSISTANT_ETFS)
    historical = list(session.scalars(select(Instrument.code).join(
        V2PositionEvent, V2PositionEvent.instrument_id == Instrument.id
    ).distinct().order_by(Instrument.code)))
    return tuple(dict.fromkeys([*active, *sorted(LEGACY_CALENDAR_CODES), *historical]))

# The six default tabs are 399006 plus the five active ETF slots.  The
# remaining catalog items remain selectable from the extra-universe/calendar.
CORE_PANEL_CODES: Final[tuple[str, ...]] = ("399006", *AI_ASSISTANT_ETFS)
CALENDAR_INSTRUMENT_CODES: Final[frozenset[str]] = frozenset(
    {"399006", *AI_ASSISTANT_ETFS}
)

# Every ETF in the simplified product is display/calendar data only.  The
# legacy model universe is retained for historical database compatibility but
# is not eligible for V3.7 UI actions, training, inference, or advice.
DISPLAY_ONLY_ETFS: Final[dict[str, dict[str, str]]] = dict(AI_ASSISTANT_ETFS)
DISPLAY_ONLY_CODES: Final[frozenset[str]] = frozenset(DISPLAY_ONLY_ETFS)
CHART_INSTRUMENTS: Final[frozenset[str]] = frozenset(CORE_PANEL_CODES)
READABLE_MARKETS: Final[frozenset[str]] = CALENDAR_INSTRUMENT_CODES | {"NDX"}


def etf_exchange_prefix(instrument_code: str) -> str:
    """Return the explicit public-quote prefix for a supported ETF."""

    if instrument_code in AI_ASSISTANT_ETFS:
        return AI_ASSISTANT_ETFS[instrument_code]["exchange_prefix"]
    if is_etf_code(instrument_code):
        return "sh" if instrument_code.startswith(("5", "6")) else "sz"
    raise ValueError(f"unsupported ETF instrument: {instrument_code}")


def is_etf_code(instrument_code: str) -> bool:
    """Return whether a six-digit code is an ETF-style instrument code."""

    if not isinstance(instrument_code, str) or len(instrument_code) != 6:
        return False
    if not instrument_code.isdigit():
        return False
    return instrument_code.startswith(("1", "5"))
