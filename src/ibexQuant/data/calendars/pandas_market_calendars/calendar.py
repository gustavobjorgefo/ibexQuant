# ibexQuant\src\ibexQuant\data\calendars\pandas_market_calendars\calendar.py

"""
TradingCalendar backed by the pandas_market_calendars package.

Data source and provenance
---------------------------
pandas_market_calendars ships every holiday and session rule as static
code inside the package itself — it never fetches live data from an
exchange or a paid data vendor at runtime. It originated in Quantopian's
trading_calendars project (used by Zipline) and is now maintained
independently, open source (MIT), by community contributors.

That is good enough for most research and backtesting use, but it is
NOT an official feed from any exchange with a contractual accuracy
guarantee. Movable and municipal holidays in particular can lag a real
calendar change until a contributor notices and a new release ships.
Spot-check an unfamiliar year against the exchange's own published
calendar before trusting this blindly in anything that risks real
capital — do not treat it as a certified source.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pandas_market_calendars as mcal

from ibexQuant.data.calendars.calendar import TradingCalendar
from ibexQuant.data.ingestion.provider import DateLike


class PandasMarketCalendarsCalendar(TradingCalendar):
    """
    TradingCalendar backed by pandas_market_calendars.

    Parameters
    ----------
    name : str
        A calendar name pandas_market_calendars recognizes (e.g.
        ``"B3"``, ``"XNYS"``). See
        ``pandas_market_calendars.get_calendar_names()`` for the full,
        current list — it is authoritative, not this docstring.

    Examples
    --------
    >>> b3 = PandasMarketCalendarsCalendar("B3")
    >>> b3.valid_sessions("2024-01-01", "2024-01-05")
    """

    def __init__(self, name: str) -> None:
        self._calendar = mcal.get_calendar(name)

    def valid_sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        """See :meth:`TradingCalendar.valid_sessions`."""
        sessions = self._calendar.valid_days(start_date=start, end_date=end)
        return cast(pd.DatetimeIndex, sessions.tz_localize(None))
