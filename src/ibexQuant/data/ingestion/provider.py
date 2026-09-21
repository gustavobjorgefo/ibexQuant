# ibexQuant\src\ibexQuant\data\ingestion\provider.py

"""
Abstract contract for market data ingestion providers.

This module defines :class:`MarketDataProvider`, the single abstraction
every market data source (yfinance, B3 parquet exports, future venues) must
implement in order to be usable anywhere else in ibexQuant.

Responsibility
--------------
Deliver bar-level OHLCV market data for one or more symbols in a single,
predictable shape, regardless of the underlying source, so that every
downstream consumer (data quality checks, calendar alignment, feature
engineering, backtesting) can rely on one contract instead of one per
source.

Contract summary
-----------------
- ``get_bars`` always returns a :class:`pandas.DataFrame` indexed by a
  ``pandas.MultiIndex`` with levels ``("timestamp", "symbol")``, sorted
  ascending, regardless of how many symbols were requested.
- Column names and dtypes are normalized (``open``/``high``/``low``/
  ``close``/``volume``, all ``float64``) independently of the source's
  native naming.
- Missing data is represented as ``NaN`` and never silently filled.
- Duplicate ``(timestamp, symbol)`` pairs are never returned.
- ``frequency`` describes the source as a whole and is fixed at
  construction time. ``adjusted`` is a per-call choice, since a single
  source can serve both adjusted and raw prices on request.

What does not belong here
--------------------------
- Data quality checks (spike detection, OHLC sanity) — see
  ``ibexQuant.data.ingestion.validation``.
- Trading calendar alignment (distinguishing a real gap from a non-trading
  day) — see ``ibexQuant.data.ingestion.calendar``.
- Caching, retries, or rate limiting — implemented by providers that wrap
  another :class:`MarketDataProvider` (e.g. a future ``CachingProvider``),
  never by a source provider itself.
- Real-time/streaming data — a distinct concern, addressed by a separate
  abstraction later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Final, TypeAlias

import pandas as pd

DateLike: TypeAlias = str | pd.Timestamp
"""A date or timestamp accepted at the provider boundary."""

OHLCV_FIELDS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


def validate_bars_shape(df: pd.DataFrame) -> None:
    """
    Confirm ``df`` is indexed the way every MarketDataProvider promises.

    Shared by any function that consumes MarketDataProvider output and
    needs to fail fast on a frame that was never shaped by one in the
    first place — e.g. ``ibexQuant.data.ingestion.validation.check_bars``
    and ``ibexQuant.data.calendars.calendar.align_to_calendar``.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to check.

    Raises
    ------
    ValueError
        If the index is not a ``MultiIndex`` named ``(timestamp, symbol)``.
    """
    if not isinstance(df.index, pd.MultiIndex) or list(df.index.names) != [
        "timestamp",
        "symbol",
    ]:
        actual = list(df.index.names) if isinstance(df.index, pd.MultiIndex) else type(df.index)
        raise ValueError(
            "Expected a MarketDataProvider-shaped DataFrame indexed by "
            f"MultiIndex(timestamp, symbol); got {actual!r}."
        )


class MarketDataProvider(ABC):
    """
    Abstract source of OHLCV market data.

    A concrete provider wraps exactly one data source (e.g. yfinance, a
    local parquet export) and exposes it through :meth:`get_bars`. The bar
    ``frequency`` the source natively serves is fixed at construction time,
    because it describes *what kind of source* this is, not a per-request
    option. Whether prices are corporate-action adjusted is instead a
    per-call choice — see the ``adjusted`` parameter of :meth:`get_bars`.

    Parameters
    ----------
    frequency : str
        Native bar frequency of this source (e.g. ``"1d"``). A single
        provider instance always serves the same frequency.

    Examples
    --------
    >>> provider = YFinanceProvider(frequency="1d")
    >>> data = provider.get_bars(
    ...     ["PETR4.SA", "VALE3.SA"], "2020-01-01", "2024-01-01", adjusted=True
    ... )
    >>> data.index.names
    FrozenList(['timestamp', 'symbol'])
    """

    def __init__(self, frequency: str) -> None:
        self._frequency: str = frequency

    # --- properties ---

    @property
    def frequency(self) -> str:
        """str: Native bar frequency served by this provider."""
        return self._frequency

    # --- public API ---

    @abstractmethod
    def get_bars(
        self,
        symbols: str | Sequence[str],
        start: DateLike,
        end: DateLike,
        fields: Sequence[str] | None = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Return OHLCV bars for one or more symbols.

        Parameters
        ----------
        symbols : str or Sequence[str]
            A single symbol or a sequence of symbols. The return shape is
            identical in both cases — there is no special-cased
            single-symbol output.
        start : str or pandas.Timestamp
            Inclusive start of the requested range.
        end : str or pandas.Timestamp
            Inclusive end of the requested range.
        fields : Sequence[str], optional
            Subset of :data:`OHLCV_FIELDS` to return. Defaults to all five.
        adjusted : bool, default True
            Whether returned prices are adjusted for splits and dividends.
            ``close`` (and any other price field) carries one meaning or
            the other, never both at once — there is no separate
            ``adj_close`` column. A provider that cannot honor the
            requested value must raise ``ValueError``.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``MultiIndex(timestamp, symbol)``, sorted ascending.
            Columns are the requested ``fields`` (or all of
            :data:`OHLCV_FIELDS`), dtype ``float64``. Missing observations
            are ``NaN``; no ``(timestamp, symbol)`` pair repeats.

        Raises
        ------
        ValueError
            If ``fields`` contains a value outside :data:`OHLCV_FIELDS`, or
            if none of the requested ``symbols`` returned any data.

        Notes
        -----
        A partial failure — some symbols returning data and others not —
        is not an error: the missing symbols are simply absent from the
        result and the failure is logged by the implementation. Only a
        *total* failure (no symbol returned data) raises.

        This method does not validate value-level sanity (e.g. ``high >=
        low``) and does not align results to a trading calendar. Both are
        the responsibility of separate, source-agnostic functions applied
        to the result.
        """
        ...

    # --- shared helpers for subclasses ---

    def _resolve_fields(self, fields: Sequence[str] | None) -> tuple[str, ...]:
        """
        Validate and normalize a requested field subset.

        Centralized here so every subclass validates ``fields`` the same
        way instead of each reimplementing the check.

        Parameters
        ----------
        fields : Sequence[str], optional
            Requested field subset, or ``None`` for all of
            :data:`OHLCV_FIELDS`.

        Returns
        -------
        tuple[str, ...]
            The validated fields, in :data:`OHLCV_FIELDS` order.

        Raises
        ------
        ValueError
            If ``fields`` contains a value outside :data:`OHLCV_FIELDS`.
        """
        if fields is None:
            return OHLCV_FIELDS

        unknown = set(fields) - set(OHLCV_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown field(s) {sorted(unknown)}. Valid fields are {OHLCV_FIELDS}."
            )
        return tuple(field for field in OHLCV_FIELDS if field in fields)
