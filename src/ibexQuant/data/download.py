# ibexQuant\src\ibexQuant\data\download.py

"""
One-call daily bars for research, in the shape of ``yfinance.download`` (ADR 0015, H5.10).

Responsibility
--------------
Open the configured B3 base, ask :class:`B3Provider` for the bars and return
them wide: one ticker gives a frame of fields, several give columns
``(field, ticker)``.

Contract
--------
- The numbers are the provider's: same raw bars, same adjustment rules
  (ADR 0014), same handling of renames and of unknown tickers (left out with
  a warning; an error when none has data). This module only reshapes.
- ``adjustment`` is ``"total_return"`` (default), ``"splits"`` or ``"raw"``;
  ``anchor`` applies to the adjusted modes only.
- A ticker given as a string gives flat columns; a list, even of one, gives
  columns ``(field, ticker)`` in the order asked.
- The index holds the sessions (``timestamp``), ascending; a ticker missing on
  a session has NaN there.

What does not belong here
-------------------------
- Point-in-time universes. A list of tickers chosen today carries
  survivorship bias: backtests take the universe from
  ``ibexQuant.data.universe.universe_mask`` and the bars from the provider.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pandas as pd

from ibexQuant.data.adjustment import AdjustmentMode
from ibexQuant.data.ingestion.b3.provider import B3Provider
from ibexQuant.data.ingestion.provider import DateLike
from ibexQuant.data.store import DataStore

FIELD_LEVEL: Final[str] = "field"
TICKER_LEVEL: Final[str] = "ticker"
SESSION_AXIS: Final[str] = "timestamp"


def download(
    tickers: str | Sequence[str],
    start: DateLike,
    end: DateLike,
    *,
    adjustment: str = AdjustmentMode.TOTAL_RETURN.value,
    anchor: DateLike | None = None,
    fields: Sequence[str] | None = None,
    store: DataStore | None = None,
) -> pd.DataFrame:
    """
    Return daily bars of B3 tickers, wide.

    Parameters
    ----------
    tickers : str or sequence of str
        One ticker (flat columns) or a list (columns ``(field, ticker)``).
    start, end : str or pandas.Timestamp
        Inclusive range of sessions.
    adjustment : {"total_return", "splits", "raw"}, default "total_return"
        ``total_return`` removes splits and cash (like ``auto_adjust`` in
        yfinance), ``splits`` only share-count events, ``raw`` serves the
        COTAHIST bars as traded.
    anchor : str or pandas.Timestamp, optional
        Date the adjusted prices are relative to; defaults to ``end``.
    fields : sequence of str, optional
        Subset of ``open``, ``high``, ``low``, ``close``, ``volume`` and
        ``traded_value``; OHLCV by default.
    store : DataStore, optional
        Defaults to the configured data root (``DataStore.open()``).

    Returns
    -------
    pandas.DataFrame
        Indexed by session.

    Raises
    ------
    ValueError
        If ``adjustment`` is unknown, ``anchor`` is given with ``raw``, a
        field is unknown, or no ticker has data in the range.

    Examples
    --------
    >>> import ibexQuant as iq
    >>> iq.download("PETR4", "2024-01-01", "2024-12-31")  # doctest: +SKIP
    >>> iq.download(
    ...     ["PETR4", "VALE3"], "2024-01-01", "2024-12-31", fields=["close"]
    ... )  # doctest: +SKIP
    """
    try:
        mode = AdjustmentMode(adjustment)
    except ValueError as error:
        choices = [member.value for member in AdjustmentMode]
        raise ValueError(f"adjustment must be one of {choices}, got {adjustment!r}.") from error
    if mode is AdjustmentMode.RAW and anchor is not None:
        raise ValueError("anchor applies to adjusted bars only; drop it or pick an adjustment.")

    provider = B3Provider(store or DataStore.open())
    single = isinstance(tickers, str)
    asked = [tickers] if isinstance(tickers, str) else list(dict.fromkeys(tickers))
    if mode is AdjustmentMode.RAW:
        bars = provider.get_bars(asked, start, end, fields, adjusted=False)
    else:
        bars = provider.get_bars(
            asked, start, end, fields, adjusted=True, anchor=anchor, adjustment=mode.value
        )

    symbols = set(bars.index.get_level_values("symbol"))
    frames = {ticker: _one(bars, ticker) for ticker in asked if ticker in symbols}
    columns = list(bars.columns)
    if single:
        return frames[asked[0]]
    wide = pd.concat(frames, axis=1, names=[TICKER_LEVEL, FIELD_LEVEL])
    wide = wide.swaplevel(axis=1)
    order = [(field, ticker) for field in columns for ticker in frames]
    wide = wide.reindex(
        columns=pd.MultiIndex.from_tuples(order, names=[FIELD_LEVEL, TICKER_LEVEL])
    )
    wide.index.name = SESSION_AXIS
    return wide


def _one(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = bars[bars.index.get_level_values("symbol") == ticker]
    frame = pd.DataFrame(
        rows.to_numpy(), index=rows.index.get_level_values("timestamp"), columns=bars.columns
    )
    frame = frame.astype("float64")
    frame.index.name = SESSION_AXIS
    frame.columns.name = None
    return frame
