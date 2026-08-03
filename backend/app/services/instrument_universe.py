"""Explicit chart/model instrument roles for the desktop research workbench."""

from __future__ import annotations

from typing import Final


MODEL_MARKETS: Final[frozenset[str]] = frozenset({"399006", "159941"})
DISPLAY_ONLY_ETFS: Final[dict[str, dict[str, str]]] = {
    "518600": {
        "name": "广发黄金ETF",
        "exchange": "SSE",
        "exchange_prefix": "sh",
    },
    "512800": {
        "name": "华宝银行ETF",
        "exchange": "SSE",
        "exchange_prefix": "sh",
    },
    "512690": {
        "name": "鹏华酒ETF",
        "exchange": "SSE",
        "exchange_prefix": "sh",
    },
}
DISPLAY_ONLY_CODES: Final[frozenset[str]] = frozenset(DISPLAY_ONLY_ETFS)
CHART_INSTRUMENTS: Final[frozenset[str]] = MODEL_MARKETS | DISPLAY_ONLY_CODES
READABLE_MARKETS: Final[frozenset[str]] = CHART_INSTRUMENTS | {"NDX"}


def etf_exchange_prefix(instrument_code: str) -> str:
    """Return the explicit public-quote prefix for a supported ETF."""

    if instrument_code in DISPLAY_ONLY_ETFS:
        return DISPLAY_ONLY_ETFS[instrument_code]["exchange_prefix"]
    if instrument_code == "159941":
        return "sz"
    raise ValueError(f"unsupported ETF instrument: {instrument_code}")
