# ibexQuant\src\ibexQuant\data\ingestion\yfinance\provider.py

"""
yfinance-backed implementation of the MarketDataProvider contract.

Responsibility
--------------
Adapt the yfinance package to :class:`MarketDataProvider`: download OHLCV
data per symbol and normalize it to the canonical schema (long/tidy
``MultiIndex(timestamp, symbol)``, lowercase ``float64`` columns).

What does not belong here
--------------------------
- Retrying failed downloads. yfinance's network calls occasionally fail
  transiently, but retry policy belongs to a decorator wrapping this
  provider (e.g. a future ``RetryingProvider``), not to the source
  provider itself — kept out deliberately for now, tracked as backlog.
- Caching to disk — a future ``CachingProvider`` wraps this class instead.
- Trading calendar alignment or value-level sanity checks — both are
  applied to this provider's output by separate, source-agnostic
  functions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import pandas as pd
import yfinance

from ibexQuant.data.ingestion.provider import OHLCV_FIELDS, DateLike, MarketDataProvider

logger: logging.Logger = logging.getLogger(__name__)

_COLUMN_RENAME: Final[dict[str, str]] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YFinanceProvider(MarketDataProvider):
    """
    OHLCV provider backed by Yahoo Finance via the yfinance package.

    Symbols are downloaded one at a time rather than in a single batched
    call. This trades a small amount of speed for the ability to isolate
    exactly which symbols failed, which is what :meth:`get_bars`'s partial
    failure policy requires. This is intentionally not optimized for
    throughput — a future ``CachingProvider`` is expected to remove most
    repeated downloads anyway.

    Parameters
    ----------
    frequency : str
        yfinance interval string (e.g. ``"1d"``, ``"1h"``), passed through
        unchanged to ``yfinance.download``'s ``interval`` argument.

    Examples
    --------
    >>> provider = YFinanceProvider(frequency="1d")
    >>> data = provider.get_bars(["PETR4.SA", "VALE3.SA"], "2020-01-01", "2024-01-01")
    """

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
        Return OHLCV bars for one or more symbols from Yahoo Finance.

        See :meth:`MarketDataProvider.get_bars` for the full contract.
        ``adjusted`` maps directly to yfinance's ``auto_adjust``: when
        ``True``, the entire OHLC series is split/dividend-adjusted by
        yfinance itself, not just the close.
        """
        resolved_fields = self._resolve_fields(fields)
        requested_symbols = self._normalize_symbols(symbols)

        frames: dict[str, pd.DataFrame] = {}
        for symbol in requested_symbols:
            frame = self._download_one(symbol, start, end, adjusted)
            if frame is not None:
                frames[symbol] = frame

        if not frames:
            raise ValueError(
                f"No data returned for any of the requested symbols: {requested_symbols}."
            )

        combined = pd.concat(frames, names=["symbol"])
        combined = combined.reorder_levels(["timestamp", "symbol"]).sort_index()
        return combined[list(resolved_fields)]

    # --- internal helpers ---

    def _download_one(
        self,
        symbol: str,
        start: DateLike,
        end: DateLike,
        adjusted: bool,
    ) -> pd.DataFrame | None:
        """
        Download and normalize a single symbol.

        Returns
        -------
        pd.DataFrame or None
            Normalized OHLCV frame indexed by timestamp, or ``None`` if the
            download failed or returned no rows. Failures are logged, not
            raised — total failure across all symbols is handled by the
            caller.
        """
        try:
            raw = yfinance.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=self.frequency,
                auto_adjust=adjusted,
                progress=False,
            )
        except Exception as exc:
            # yfinance surfaces network, parsing, and upstream API failures
            # under a mix of exception types with no stable common base
            # beyond Exception. Caught broadly and deliberately here, at
            # this external boundary only, and always logged rather than
            # swallowed silently.
            logger.warning("Failed to download %s: %s", symbol, exc)
            return None

        if raw.empty:
            logger.warning("No data returned for %s", symbol)
            return None

        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a raw yfinance frame to the canonical OHLCV schema.

        Parameters
        ----------
        raw : pd.DataFrame
            Untouched output of ``yfinance.download`` for a single symbol.

        Returns
        -------
        pd.DataFrame
            Indexed by a ``"timestamp"``-named ``DatetimeIndex``, columns
            renamed to :data:`OHLCV_FIELDS`, dtype ``float64``, sorted and
            deduplicated by timestamp.
        """
        df = raw.copy()

        if isinstance(df.columns, pd.MultiIndex):
            # yfinance names its column levels ("Price", "Ticker") — read
            # by name, not position, so a future reordering of levels
            # upstream does not silently break this.
            df.columns = df.columns.get_level_values("Price")

        df = df.rename(columns=_COLUMN_RENAME)[list(OHLCV_FIELDS)].astype("float64")
        df.columns.name = None  # drop yfinance's leftover "Price" column-axis name
        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"

        return df[~df.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _normalize_symbols(symbols: str | Sequence[str]) -> list[str]:
        """Normalize the ``symbols`` argument to a list of strings."""
        if isinstance(symbols, str):
            return [symbols]
        return list(symbols)
