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
- Renames are followed by default (ADR 0012, H2.14): a ticker is resolved
  to its security, and the security's whole round-lot history in the range
  is served under the requested ticker. ``follow_renames=False`` serves the
  ticker's own rows only, as in H1; there, a ticker with two rows on one
  session raises ``ValueError``.

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
    follow_renames : bool, default True
        Serve each ticker's whole security across ticker changes; requires
        the security master (``ibexquant data build security-master``).

    Examples
    --------
    >>> provider = B3Provider(DataStore.open())  # doctest: +SKIP
    >>> provider.get_bars(
    ...     ["PETR4", "VALE3"], "2024-01-01", "2024-12-31", adjusted=False
    ... )  # doctest: +SKIP
    """

    def __init__(self, store: DataStore, follow_renames: bool = True) -> None:
        super().__init__(frequency="1d")
        self._store: DataStore = store
        self._follow_renames: bool = follow_renames

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

        See :meth:`MarketDataProvider.get_bars` for the shared contract. With
        ``follow_renames`` (the default), each ticker is resolved to its
        security on the last session up to ``end`` on which it traded, and the
        whole security's history in the range is returned under that ticker
        (ADR 0012, H2.14).

        Raises
        ------
        ValueError
            If ``adjusted`` is true (not available before H4), if ``fields``
            is invalid, if ``start`` is after ``end``, if no requested ticker
            has data in the range, if two requested tickers are the same
            security, or (without ``follow_renames``) if a ticker has two rows
            on one session.
        SecurityMasterNotBuiltError
            With ``follow_renames``, if the security master was never built.
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

        if self._follow_renames:
            rows = self._read_securities(tickers, first, last)
        else:
            rows = self._read(tickers, first, last).assign(label=lambda frame: frame["ticker"])
            repeated = rows.duplicated(subset=["session_date", "label"], keep=False)
            if repeated.any():
                pairs = rows.loc[repeated, ["session_date", "label"]].drop_duplicates()
                raise ValueError(
                    f"More than one round-lot row per session: {pairs.values.tolist()}."
                )

        for ticker in sorted(set(tickers) - set(rows["label"])):
            logger.warning("No B3 data for %s from %s to %s", ticker, first.date(), last.date())
        if rows.empty:
            raise ValueError(f"No B3 data for {tickers} between {first.date()} and {last.date()}.")

        bars = pd.DataFrame(
            {field: rows[_SOURCE_COLUMNS[field]].to_numpy(dtype="float64") for field in columns},
            index=pd.MultiIndex.from_arrays(
                [
                    pd.DatetimeIndex(rows["session_date"]).as_unit("ns"),
                    rows["label"].astype(str).to_numpy(),
                ],
                names=["timestamp", "symbol"],
            ),
        )
        return bars.sort_index()

    # --- helpers ---

    def _read_securities(
        self, tickers: list[str], first: pd.Timestamp, last: pd.Timestamp
    ) -> pd.DataFrame:
        """Read every round-lot listing of each requested ticker's security."""
        # Imported here: the reference package imports this package's build module,
        # so a module-level import would be circular.
        from ibexQuant.data.reference import SecurityMaster

        table = SecurityMaster.load(self._store).table
        round_lot = table[table["market_type"] == ROUND_LOT_MARKET]
        label_of: dict[tuple[str, str], str] = {}
        security_of: dict[str, str] = {}
        for ticker in tickers:
            own = round_lot[(round_lot["ticker"] == ticker) & (round_lot["first_session"] <= last)]
            if own.empty:
                continue
            security = str(own.sort_values("first_session").iloc[-1]["security_id"])
            if security in security_of:
                raise ValueError(
                    f"{security_of[security]} and {ticker} are the same security ({security})."
                )
            security_of[security] = ticker
            members = round_lot[round_lot["security_id"] == security]
            for member_ticker, isin in zip(members["ticker"], members["isin"], strict=True):
                label_of[(str(member_ticker), str(isin))] = ticker

        wanted = sorted({ticker for ticker, _ in label_of})
        rows = self._read(wanted, first, last)
        keys = list(zip(rows["ticker"].astype(str), rows["isin"].astype(str), strict=True))
        labels = pd.Series([label_of.get(key) for key in keys], index=rows.index, dtype="object")
        rows = rows.assign(label=labels).dropna(subset=["label"])
        # A security trading under two tickers on one session (a change of ticker
        # with the same ISIN): the more traded row represents the session.
        rows = rows.sort_values("volume_shares", ascending=False, kind="stable")
        rows = rows.drop_duplicates(subset=["session_date", "label"], keep="first")
        return rows.reset_index(drop=True)

    def _read(self, tickers: list[str], first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
        """Read the partitions that overlap ``[first, last]``, round lot only."""
        columns = ["session_date", "ticker", "isin", *_SOURCE_COLUMNS.values()]
        frames = []
        for year in range(first.year, last.year + 1):
            path = self._store.curated(*CURATED_PARTS, f"year={year}", PARTITION_FILE)
            if not path.is_file() or not tickers:
                continue
            frames.append(
                pd.read_parquet(
                    path,
                    engine="pyarrow",
                    columns=columns,
                    filters=[("market_type", "==", ROUND_LOT_MARKET), ("ticker", "in", tickers)],
                )
            )
        if not frames:
            return pd.DataFrame(columns=columns)
        rows = pd.concat(frames, ignore_index=True)
        within = (rows["session_date"] >= first) & (rows["session_date"] <= last)
        return rows.loc[within].reset_index(drop=True)
