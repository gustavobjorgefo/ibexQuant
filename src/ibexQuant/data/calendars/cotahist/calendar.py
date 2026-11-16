# ibexQuant\src\ibexQuant\data\calendars\cotahist\calendar.py

"""
TradingCalendar of the days B3 actually traded, from the COTAHIST base (ADR 0016, H6.8).

Data source and provenance
--------------------------
The sessions are the distinct ``session_date`` values of ``curated/b3/bars_daily``, which
holds every COTAHIST file ingested (ADR 0011). A day is a session because B3 published
trades for it, not because a holiday rule says so: this is the calendar the index periods
of ADR 0015 use, and the one a B3 panel is aligned to.

Contract
--------
- Sessions are read on first use, only the ``session_date`` column of each partition, and
  kept in memory for the life of the object; a new object sees the latest ingestion.
- Coverage ends at the last ingested session: future sessions are not projected.

What does not belong here
-------------------------
- Other markets, and projecting future sessions for a live engine — see
  ``PandasMarketCalendarsCalendar``.
"""

from __future__ import annotations

import pandas as pd

from ibexQuant.data.calendars.calendar import TradingCalendar
from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS, PARTITION_FILE
from ibexQuant.data.ingestion.provider import DateLike
from ibexQuant.data.store import DataStore


class CotahistCalendarError(FileNotFoundError):
    """The B3 base has no daily bars to take sessions from."""


class CotahistCalendar(TradingCalendar):
    """
    B3 sessions as recorded in COTAHIST.

    Parameters
    ----------
    store : DataStore
        The data store holding ``curated/b3/bars_daily``.

    Examples
    --------
    >>> calendar = CotahistCalendar(DataStore.open())  # doctest: +SKIP
    >>> calendar.valid_sessions("2024-12-20", "2025-01-03")  # doctest: +SKIP
    """

    def __init__(self, store: DataStore) -> None:
        self._store: DataStore = store
        self._sessions: pd.DatetimeIndex | None = None

    # --- public API ---

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """
        Every session of the base, ascending.

        Raises
        ------
        CotahistCalendarError
            If there is no ``bars_daily`` partition.
        """
        if self._sessions is None:
            self._sessions = self._read()
        return self._sessions

    def valid_sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        """See :meth:`TradingCalendar.valid_sessions`."""
        sessions = self.sessions
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        return sessions[(sessions >= first) & (sessions <= last)]

    def offset(self, session: DateLike, bars: int) -> pd.Timestamp:
        """
        Return the session ``bars`` sessions before ``session``, or the first one there is.

        ``session`` need not be a session itself: counting starts from the first
        session on or after it.

        Raises
        ------
        ValueError
            If ``bars`` is negative.
        """
        if bars < 0:
            raise ValueError(f"bars must be zero or positive, got {bars}.")
        sessions = self.sessions
        position = int(sessions.searchsorted(pd.Timestamp(session).normalize()))
        return sessions[min(max(position - bars, 0), len(sessions) - 1)]

    # --- helpers ---

    def _read(self) -> pd.DatetimeIndex:
        folder = self._store.curated(*CURATED_PARTS)
        paths = sorted(folder.glob(f"year=*/{PARTITION_FILE}")) if folder.is_dir() else []
        if not paths:
            raise CotahistCalendarError(
                "No bars_daily partition. Run `ibexquant data ingest cotahist --all` first."
            )
        days = [pd.read_parquet(path, columns=["session_date"])["session_date"] for path in paths]
        index = pd.DatetimeIndex(pd.concat(days).unique()).normalize()
        return pd.DatetimeIndex(index.unique().sort_values().as_unit("ns"), name=None)
