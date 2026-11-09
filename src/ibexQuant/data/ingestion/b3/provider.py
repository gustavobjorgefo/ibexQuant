# ibexQuant\src\ibexQuant\data\ingestion\b3\provider.py

"""
:class:`MarketDataProvider` over the curated B3 daily bars.

Responsibility
--------------
Serve ``curated/b3/bars_daily`` in the shape every provider promises
(``MultiIndex(timestamp, symbol)``, float64 OHLCV), so that research code can
switch from yfinance to the B3 base without other changes (ADR 0011, H1.13).

Contract
--------
- Symbols are B3 tickers as published (``PETR4``, no ``.SA`` suffix).
- Only the round-lot cash market (``010``) is served. The odd-lot market
  trades the same security under another ticker (``PETR4F``) and stays in
  ``curated/`` for liquidity studies.
- ``volume`` is the number of shares traded.
- Prices are raw: ``adjusted=True`` raises ``ValueError`` until H4.
- A ticker with two rows on one session raises ``ValueError``: the contract
  forbids duplicate ``(timestamp, symbol)`` pairs, and choosing one row
  silently would pick between two securities.

What does not belong here
-------------------------
- Building the table — see ``bars_daily``.
- Value checks and calendar alignment — see ``check_bars`` and
  ``align_to_calendar``, applied to the result by the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import pandas as pd

from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS, PARTITION_FILE
from ibexQuant.data.ingestion.provider import DateLike, MarketDataProvider
from ibexQuant.data.store import DataStore

logger: logging.Logger = logging.getLogger(__name__)

ROUND_LOT_MARKET: Final[int] = 10
_SOURCE_COLUMNS: Final[dict[str, str]] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume_shares",
}


class B3Provider(MarketDataProvider):
    """
    Daily OHLCV bars from the B3 historical data base.

    Parameters
    ----------
    store : DataStore
        The data store holding ``curated/b3/bars_daily``.

    Examples
    --------
    >>> provider = B3Provider(DataStore.open())  # doctest: +SKIP
    >>> provider.get_bars(
    ...     ["PETR4", "VALE3"], "2024-01-01", "2024-12-31", adjusted=False
    ... )  # doctest: +SKIP
    """

    def __init__(self, store: DataStore) -> None:
        super().__init__(frequency="1d")
        self._store: DataStore = store

    # --- public API ---

    def get_bars(
        self,
        symbols: str | Sequence[str],
        start: DateLike,
        end: DateLike,
        fields: Sequence[str] | None = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Return raw daily bars for one or more B3 tickers.

        See :meth:`MarketDataProvider.get_bars` for the shared contract.

        Raises
        ------
        ValueError
            If ``adjusted`` is true (not available before H4), if ``fields``
            is invalid, if ``start`` is after ``end``, if no requested ticker
            has data in the range, or if a ticker has two rows on one session.
        """
        if adjusted:
            raise ValueError(
                "B3Provider serves raw prices only until anchored adjustment (H4); "
                "pass adjusted=False."
            )
        columns = self._resolve_fields(fields)
        tickers = [symbols] if isinstance(symbols, str) else list(dict.fromkeys(symbols))
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if first > last:
            raise ValueError(f"start ({first.date()}) is after end ({last.date()}).")

        rows = self._read(tickers, first, last)
        missing = sorted(set(tickers) - set(rows["ticker"]))
        for ticker in missing:
            logger.warning(
                "No B3 data for %s between %s and %s", ticker, first.date(), last.date()
            )
        if rows.empty:
            raise ValueError(f"No B3 data for {tickers} between {first.date()} and {last.date()}.")

        repeated = rows.duplicated(subset=["session_date", "ticker"], keep=False)
        if repeated.any():
            pairs = rows.loc[repeated, ["session_date", "ticker"]].drop_duplicates()
            raise ValueError(f"More than one round-lot row per session: {pairs.values.tolist()}.")

        bars = pd.DataFrame(
            {field: rows[_SOURCE_COLUMNS[field]].to_numpy(dtype="float64") for field in columns},
            index=pd.MultiIndex.from_arrays(
                [
                    pd.DatetimeIndex(rows["session_date"]).as_unit("ns"),
                    rows["ticker"].astype(str).to_numpy(),
                ],
                names=["timestamp", "symbol"],
            ),
        )
        return bars.sort_index()

    # --- helpers ---

    def _read(self, tickers: list[str], first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
        """Read the partitions that overlap ``[first, last]``, round lot only."""
        frames = []
        for year in range(first.year, last.year + 1):
            path = self._store.curated(*CURATED_PARTS, f"year={year}", PARTITION_FILE)
            if not path.is_file():
                continue
            frames.append(
                pd.read_parquet(
                    path,
                    engine="pyarrow",
                    columns=["session_date", "ticker", *_SOURCE_COLUMNS.values()],
                    filters=[("market_type", "==", ROUND_LOT_MARKET), ("ticker", "in", tickers)],
                )
            )
        if not frames:
            return pd.DataFrame(columns=["session_date", "ticker", *_SOURCE_COLUMNS.values()])
        rows = pd.concat(frames, ignore_index=True)
        within = (rows["session_date"] >= first) & (rows["session_date"] <= last)
        return rows.loc[within].reset_index(drop=True)
