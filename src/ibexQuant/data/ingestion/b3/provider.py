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
- ``adjusted=True`` (the default) serves prices adjusted relative to an
  anchor date (ADR 0014): ``total_return`` by default, or ``splits``; the
  anchor defaults to ``end`` and may not precede it. Factors come from the
  derived factor table, joined by security, so a followed ticker is adjusted
  along its whole chain. ``adjusted=False`` serves raw prices.
- A served range that contains an unexplained jump of the ledger is logged
  as a warning: the raw move stays in the adjusted series (H4.9).
- Renames are followed by default (ADR 0012, H2.14): a ticker is resolved
  to its security, and the security's whole round-lot history in the range
  is served under the requested ticker. ``follow_renames=False`` serves the
  ticker's own rows only, as in H1; there, a ticker with two rows on one
  session raises ``ValueError``.

What does not belong here
-------------------------
- Building the table — see ``bars_daily``.
- The factor arithmetic — see ``ibexQuant.data.adjustment``.
- Value checks and calendar alignment — see ``check_bars`` and
  ``align_to_calendar``, applied to the result by the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import pandas as pd

from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS, PARTITION_FILE
from ibexQuant.data.ingestion.provider import DateLike, MarketDataProvider
from ibexQuant.data.store import DataStore

if TYPE_CHECKING:
    from ibexQuant.data.adjustment import AdjustmentMode

logger: logging.Logger = logging.getLogger(__name__)

ROUND_LOT_MARKET: Final[int] = 10
ADJUSTED_MODES: Final[tuple[str, ...]] = ("total_return", "splits")
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
    >>> provider.get_bars(["PETR4", "VALE3"], "2024-01-01", "2024-12-31")  # doctest: +SKIP
    >>> provider.get_bars(
    ...     "PETR4", "2019-01-01", "2019-12-31", anchor="2019-12-31", adjustment="splits"
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
        *,
        anchor: DateLike | None = None,
        adjustment: str | None = None,
    ) -> pd.DataFrame:
        """
        Return daily bars for one or more B3 tickers.

        See :meth:`MarketDataProvider.get_bars` for the shared contract. With
        ``follow_renames`` (the default), each ticker is resolved to its
        security on the last session up to ``end`` on which it traded, and the
        whole security's history in the range is returned under that ticker
        (ADR 0012, H2.14).

        Parameters
        ----------
        anchor : str or pandas.Timestamp, optional
            Date the adjusted prices are relative to: bars on and after it are
            raw (ADR 0014, H4.4). Defaults to ``end``; may not precede it.
        adjustment : {"total_return", "splits"}, optional
            Events removed by the adjustment. Defaults to ``"total_return"``.

        Raises
        ------
        ValueError
            If ``anchor`` or ``adjustment`` is given with ``adjusted=False``,
            if ``adjustment`` is unknown, if ``anchor`` precedes ``end``, if
            ``fields`` is invalid, if ``start`` is after ``end``, if no
            requested ticker has data in the range, if two requested tickers
            are the same security, or (without ``follow_renames``) if a ticker
            has two rows on one session.
        SecurityMasterNotBuiltError
            With ``follow_renames`` or ``adjusted``, if the security master was
            never built.
        AdjustmentFactorsNotBuiltError
            With ``adjusted``, if the factor table was never built.
        StaleAdjustmentFactorsError
            With ``adjusted``, if the ledger changed after the factors were built.
        """
        mode = self._resolve_adjustment(adjusted, anchor, adjustment)
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
        if mode is not None:
            pivot = last if anchor is None else pd.Timestamp(anchor).normalize()
            if pivot < last:
                raise ValueError(
                    f"anchor ({pivot.date()}) is before end ({last.date()}): bars after it "
                    "would keep unadjusted jumps."
                )
            rows = self._adjusted(rows, pivot, mode, first, last)

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

    @staticmethod
    def _resolve_adjustment(
        adjusted: bool, anchor: DateLike | None, adjustment: str | None
    ) -> AdjustmentMode | None:
        """The adjustment mode, or ``None`` for raw bars."""
        # Imported here for the same reason as SecurityMaster (see _read_securities).
        from ibexQuant.data.adjustment import AdjustmentMode

        if not adjusted:
            if anchor is not None or adjustment is not None:
                raise ValueError("anchor and adjustment apply to adjusted bars only.")
            return None
        name = adjustment or AdjustmentMode.TOTAL_RETURN
        if name not in ADJUSTED_MODES:
            raise ValueError(
                f"adjustment must be one of {list(ADJUSTED_MODES)}, got {name!r}; "
                "for raw bars pass adjusted=False."
            )
        return AdjustmentMode(name)

    def _adjusted(
        self,
        rows: pd.DataFrame,
        anchor: pd.Timestamp,
        mode: AdjustmentMode,
        first: pd.Timestamp,
        last: pd.Timestamp,
    ) -> pd.DataFrame:
        """Apply the anchored factors to raw rows and warn about unexplained jumps."""
        from ibexQuant.data.adjustment import adjust, read_adjustment_factors

        if "security_id" not in rows:
            rows = rows.assign(security_id=rows["isin"].map(self._security_of_isin()))
        factors = read_adjustment_factors(self._store)
        self._warn_unexplained_jumps(rows, first, last)
        frame = rows.rename(columns={"volume_shares": "volume"})
        frame = adjust(frame, factors, anchor, mode)
        return frame.rename(columns={"volume": "volume_shares"})

    def _security_of_isin(self) -> dict[str, str]:
        from ibexQuant.data.reference import SecurityMaster

        table = SecurityMaster.load(self._store).table
        return dict(zip(table["isin"].astype(str), table["security_id"].astype(str), strict=True))

    def _warn_unexplained_jumps(
        self, rows: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp
    ) -> None:
        from ibexQuant.data.reference.corporate_actions import ExSessionStatus, read_ledger

        _, ex = read_ledger(self._store)
        jumps = ex[
            (ex["status"] == ExSessionStatus.UNEXPLAINED_JUMP)
            & ex["security_id"].isin(set(rows["security_id"]))
            & (ex["session_date"] > first)
            & (ex["session_date"] <= last)
        ]
        for ticker, day, ratio in zip(
            jumps["ticker"], jumps["session_date"].dt.date, jumps["open_ratio"], strict=True
        ):
            logger.warning(
                "Unexplained jump kept in adjusted bars: %s on %s (open / previous close %.3f)",
                ticker,
                day,
                ratio,
            )

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
        rows = rows.assign(
            label=labels, security_id=labels.map(self._security_by_label(security_of))
        ).dropna(subset=["label"])
        # A security trading under two tickers on one session (a change of ticker
        # with the same ISIN): the more traded row represents the session.
        rows = rows.sort_values("volume_shares", ascending=False, kind="stable")
        rows = rows.drop_duplicates(subset=["session_date", "label"], keep="first")
        return rows.reset_index(drop=True)

    @staticmethod
    def _security_by_label(security_of: dict[str, str]) -> dict[str, str]:
        return {ticker: security for security, ticker in security_of.items()}

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
