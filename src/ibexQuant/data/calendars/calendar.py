# ibexQuant\src\ibexQuant\data\calendars\calendar.py

"""
Abstract contract for trading-calendar sources, and calendar alignment.

Responsibility
--------------
Two things live here, deliberately together: the minimal
:class:`TradingCalendar` contract any calendar source must satisfy, and
``align_to_calendar``, the one function that uses it to reindex OHLCV
bars (as returned by any ``MarketDataProvider``) against real trading
sessions. A calendar source on its own is not useful to the rest of the
codebase — what matters is the alignment it enables.

Per-symbol range, not a global one
-----------------------------------
When ``start``/``end`` are not given, the range checked for a symbol is
that symbol's own observed ``[min, max]`` timestamp — never the whole
input frame's. A frame holding a long-listed stock next to one that IPO'd
last year is normal (e.g. a full request across a whole index), and
checking the newer listing against the older stock's range would flag
"missing" sessions for a period where the newer symbol simply did not
exist yet. That is not a data quality problem, so it must not be
reported as one.

Unexpected timestamps (dropped, logged, not raised)
-----------------------------------------------------
A timestamp present in ``df`` that is not one of the calendar's valid
sessions for that symbol (e.g. a vendor mistakenly including a Saturday)
is logged as a warning and does not appear in the returned frame — the
returned index is exactly the calendar's valid sessions, nothing else.
This is intentionally a log, not an exception or a returned report: it
is expected to be rare, and does not (yet) warrant the same per-row
audit trail as ``ibexQuant.data.ingestion.validation.check_bars``. If
that assumption stops holding, this is the place to revisit it.

What deliberately does not belong here
---------------------------------------
- Deciding what to do about the ``NaN`` that alignment reveals (a real
  gap within a valid session). That is still the caller's call, exactly
  as it is everywhere else in this codebase.
- Intraday session times, breaks, or auctions. This operates at the
  session (day) level only; intraday scheduling is a tick-data concern
  for later.
- Value-level sanity of the bars themselves (``high >= low``, etc.).
  That is ``ibexQuant.data.ingestion.validation.check_bars``, a separate
  step in the pipeline.
- Any specific calendar's data source or how much to trust it. That
  belongs in the module docstring of each concrete ``TradingCalendar`` —
  every source has its own provenance and its own caveats.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import cast

import pandas as pd

from ibexQuant.data.ingestion.provider import DateLike, validate_bars_shape

logger: logging.Logger = logging.getLogger(__name__)


class TradingCalendar(ABC):
    """
    Abstract source of valid trading sessions for one market or exchange.
    """

    @abstractmethod
    def valid_sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        """
        Return the valid trading sessions in ``[start, end]``, inclusive.

        Parameters
        ----------
        start : str or pandas.Timestamp
        end : str or pandas.Timestamp

        Returns
        -------
        pandas.DatetimeIndex
            Naive (no timezone) timestamps, one per valid session, sorted
            ascending — consistent with the daily-bar contract used
            throughout the ingestion layer.
        """
        ...


def align_to_calendar(
    df: pd.DataFrame,
    calendar: TradingCalendar | Mapping[str, TradingCalendar],
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> pd.DataFrame:
    """
    Reindex OHLCV bars, per symbol, against their trading calendar.

    After this, a missing row within a symbol's checked range is a real
    gap — a valid session with no data — never a non-trading day.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by ``MultiIndex(timestamp, symbol)``, as returned by any
        ``MarketDataProvider``.
    calendar : TradingCalendar or Mapping[str, TradingCalendar]
        A single calendar applied to every symbol, or a mapping from
        symbol to its own calendar for a multi-market frame.
    start, end : str or pandas.Timestamp, optional
        Range to align each symbol against. When omitted, defaults to
        that symbol's own observed ``[min, max]`` timestamp in ``df``
        (see the module docstring for why this is per-symbol, not
        global).

    Returns
    -------
    pd.DataFrame
        Indexed by ``MultiIndex(timestamp, symbol)`` restricted to each
        symbol's valid trading sessions in range. Columns and dtypes are
        unchanged from ``df``; a session with no original row becomes a
        row of ``NaN``.

    Raises
    ------
    ValueError
        If ``df`` is not indexed by ``MultiIndex(timestamp, symbol)``, or
        if ``calendar`` is a mapping missing an entry for a symbol
        present in ``df``.
    """
    validate_bars_shape(df)

    aligned_frames: list[pd.DataFrame] = []
    for symbol in df.index.get_level_values("symbol").unique():
        symbol_calendar = _resolve_calendar(calendar, symbol)
        symbol_df = cast(pd.DataFrame, df.xs(symbol, level="symbol", drop_level=False))
        symbol_timestamps = symbol_df.index.get_level_values("timestamp")

        range_start = start if start is not None else symbol_timestamps.min()
        range_end = end if end is not None else symbol_timestamps.max()
        sessions = symbol_calendar.valid_sessions(range_start, range_end)

        _warn_on_unexpected_timestamps(symbol, symbol_timestamps, sessions)

        new_index = pd.MultiIndex.from_product([sessions, [symbol]], names=df.index.names)
        aligned_frames.append(symbol_df.reindex(new_index))

    return pd.concat(aligned_frames).sort_index()


# --- internal helpers ---


def _resolve_calendar(
    calendar: TradingCalendar | Mapping[str, TradingCalendar], symbol: str
) -> TradingCalendar:
    """Return the calendar to use for ``symbol``."""
    if not isinstance(calendar, Mapping):
        return calendar
    if symbol not in calendar:
        raise ValueError(f"No calendar provided for symbol {symbol!r}.")
    return calendar[symbol]


def _warn_on_unexpected_timestamps(
    symbol: str, observed: pd.Index, sessions: pd.DatetimeIndex
) -> None:
    """Log a warning for any observed timestamp outside the valid sessions."""
    unexpected = observed.difference(sessions)
    if len(unexpected) > 0:
        logger.warning(
            "%s has %d timestamp(s) outside its calendar's valid sessions "
            "(dropped from the aligned result): %s",
            symbol,
            len(unexpected),
            list(unexpected[:5]),
        )
