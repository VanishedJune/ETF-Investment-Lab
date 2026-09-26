"""Exchange-session calendars used to prove period completeness."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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

    @staticmethod
    def _binding_for(instrument_code: str) -> tuple[str, str, bool]:
        if instrument_code in ExchangeCalendarProvider._CALENDAR_BINDINGS:
            return ExchangeCalendarProvider._CALENDAR_BINDINGS[instrument_code]
        if instrument_code.startswith(("5", "6")):
            return ("XSHG", "XSHG", False)
        if instrument_code.startswith(("0", "1", "3")):
            return ("XSHG", "XSHE", True)
        raise ValueError(
            f"No exchange calendar configured for instrument: {instrument_code}"
        )

    def metadata(self, instrument_code: str) -> dict[str, str | bool]:
        """Return the target exchange and auditable schedule provenance."""

        schedule_name, target_exchange, equivalent_proxy = self._binding_for(
            instrument_code
        )
        return {
            "target_exchange": target_exchange,
            "schedule_provider": "exchange_calendars",
            "schedule_name": schedule_name,
            "equivalent_mainland_schedule_proxy": equivalent_proxy,
        }

    def latest_completed_session(
        self,
        instrument_code: str,
        *,
        as_of: datetime | None = None,
        availability_delay: timedelta = timedelta(minutes=30),
    ) -> date | None:
        """Return the latest session whose close data should be public.

        A calendar date alone is not enough for freshness checks: before the
        exchange closes, today's row must not be required, and public quote
        vendors need a small publication buffer after the official close.
        Exchange-calendars supplies the real close timestamp (including US
        daylight-saving changes), so the result is auditable for both the
        mainland ETF panel and the legacy NDX benchmark.
        """

        current = as_of or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if availability_delay < timedelta(0):
            raise ValueError("availability_delay cannot be negative")

        calendar_name = self._binding_for(instrument_code)[0]
        import exchange_calendars as exchange_calendars

        calendar = exchange_calendars.get_calendar(calendar_name)
        current_utc = current.astimezone(timezone.utc)
        last_session = calendar.last_session.date()
        # An exhausted packaged schedule must never make old local data look
        # fresh indefinitely.  Let the caller download and report the calendar
        # verification failure instead of returning a stale terminal session.
        if current_utc.date() > last_session + timedelta(days=7):
            raise ValueError(
                f"{calendar_name} exchange calendar ends at {last_session.isoformat()}"
            )

        bounded_end = min(current_utc.date(), last_session)
        bounded_start = max(
            calendar.first_session.date(),
            bounded_end - timedelta(days=31),
        )
        sessions = calendar.sessions_in_range(bounded_start, bounded_end)
        for session in reversed(sessions):
            close_at = calendar.session_close(session).to_pydatetime()
            if close_at.tzinfo is None or close_at.utcoffset() is None:
                close_at = close_at.replace(tzinfo=timezone.utc)
            if current_utc >= close_at.astimezone(timezone.utc) + availability_delay:
                return session.date()
        return None

    def sessions(
        self,
        instrument_code: str,
        start_date: date,
        end_date: date,
    ) -> tuple[date, ...]:
        calendar_name = self._binding_for(instrument_code)[0]

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
