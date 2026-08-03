"""Exchange-session calendars used to prove period completeness."""

from __future__ import annotations

from datetime import date
from typing import Protocol


class CalendarProvider(Protocol):
    def sessions(
        self,
        instrument_code: str,
        start_date: date,
        end_date: date,
    ) -> tuple[date, ...]:
        """Return the exchange's expected sessions in the inclusive range."""


class ExchangeCalendarProvider:
    """Production adapter backed by exchange-calendars."""

    # ``exchange_calendars`` 4.13 exposes XSHG but not a separate XSHE
    # calendar.  Shanghai and Shenzhen use the same jointly announced
    # mainland exchange holiday schedule, so XSHG is a valid schedule source
    # for the two Shenzhen targets.  Keep the *target exchange identity*
    # separate from that implementation source so persisted advice never
    # falsely claims that a Shenzhen instrument is listed in Shanghai.
    _CALENDAR_BINDINGS = {
        "000688": ("XSHG", "XSHG", False),
        "399006": ("XSHG", "XSHE", True),
        "159941": ("XSHG", "XSHE", True),
        # Display-only ETFs still need a real exchange calendar so their
        # downloaded daily bars can be quality-gated and aggregated into
        # complete weekly/monthly OHLCV.  They remain excluded from every
        # model/training allow-list in ``instrument_universe``.
        "518600": ("XSHG", "XSHG", False),
        "512800": ("XSHG", "XSHG", False),
        "512690": ("XSHG", "XSHG", False),
        "NDX": ("XNAS", "XNAS", False),
    }

    def metadata(self, instrument_code: str) -> dict[str, str | bool]:
        """Return the target exchange and auditable schedule provenance."""

        try:
            schedule_name, target_exchange, equivalent_proxy = self._CALENDAR_BINDINGS[
                instrument_code
            ]
        except KeyError as error:
            raise ValueError(
                f"No exchange calendar configured for instrument: {instrument_code}"
            ) from error
        return {
            "target_exchange": target_exchange,
            "schedule_provider": "exchange_calendars",
            "schedule_name": schedule_name,
            "equivalent_mainland_schedule_proxy": equivalent_proxy,
        }

    def sessions(
        self,
        instrument_code: str,
        start_date: date,
        end_date: date,
    ) -> tuple[date, ...]:
        try:
            calendar_name = self._CALENDAR_BINDINGS[instrument_code][0]
        except KeyError as error:
            raise ValueError(
                f"No exchange calendar configured for instrument: {instrument_code}"
            ) from error

        import exchange_calendars as exchange_calendars

        calendar = exchange_calendars.get_calendar(calendar_name)
        # ``exchange_calendars`` ships a finite schedule.  Callers that ask
        # for a forecasting buffer can legitimately extend a few weeks past
        # the packaged last session even when the requested number of future
        # sessions is still available before that boundary.  Clamp to the
        # real published schedule and let the caller decide whether the
        # returned session count is sufficient; never invent future dates.
        first_session = calendar.first_session.date()
        last_session = calendar.last_session.date()
        if end_date < first_session or start_date > last_session:
            return ()
        bounded_start = max(start_date, first_session)
        bounded_end = min(end_date, last_session)
        return tuple(
            session.date()
            for session in calendar.sessions_in_range(bounded_start, bounded_end)
        )
