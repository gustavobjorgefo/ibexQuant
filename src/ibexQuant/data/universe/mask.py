# ibexQuant\src\ibexQuant\data\universe\mask.py

"""
Universe specifications and the membership mask (ADR 0015, H5.6, H5.7).

Responsibility
--------------
Say which securities belong to a universe at each session, as a boolean
frame sessions × ``security_id``.

Contract
--------
- ``Index("IBOV")``: the members of the period in force at each session, from
  the constituents table. With ``fill=False`` the periods taken from the
  liquidity rule are left empty.
- ``Tickers([...])``: every security that ever carried one of the tickers,
  over its whole round-lot life, so a renamed ticker stays in. A ticker the
  master does not know is logged and skipped.
- ``LiquidityRule(top=...)``: at each period's base session, the ``top``
  most traded eligible shares, from data up to that session only.
- ``a | b``, ``a & b`` and ``a - b`` combine universes session by session.
- The mask is membership only. Whether a security traded on a given session
  is the panel's business (missing bars); sessions outside every period have
  no member.

What does not belong here
-------------------------
- Weights (rejected, D4) and intra-period changes of B3's portfolios.
- Labelling securities with tickers: that happens when the mask joins a
  panel, with the provider's rule (ADR 0014).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from ibexQuant.data.ingestion.b3.index_portfolios import PortfolioSource
from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.store import DataStore
from ibexQuant.data.universe.calendar import assign_periods
from ibexQuant.data.universe.constituents import read_index_constituents, read_index_periods
from ibexQuant.data.universe.liquidity import (
    IBOV_LIQUIDITY,
    ROUND_LOT_MARKET,
    LiquiditySettings,
    TradedValues,
    base_session,
    read_traded_values,
    select_by_liquidity,
)

logger: logging.Logger = logging.getLogger(__name__)

SESSION_AXIS: Final[str] = "session_date"
SECURITY_AXIS: Final[str] = "security_id"


class UniverseData:
    """
    The tables a mask is computed from, read once and shared by every part of
    a combined universe.

    Parameters
    ----------
    store : DataStore
        The data store, with the universe built (``ibexquant data build universe``).
    """

    def __init__(self, store: DataStore) -> None:
        self._store: DataStore = store
        self.periods: pd.DataFrame = read_index_periods(store)
        self.constituents: pd.DataFrame = read_index_constituents(store)
        self.master: SecurityMaster = SecurityMaster.load(store)
        self._traded: TradedValues | None = None

    @property
    def traded(self) -> TradedValues:
        """Daily traded values, read on first use (only the liquidity rule needs them)."""
        if self._traded is None:
            self._traded = read_traded_values(self._store, self.master)
        return self._traded


# --- Specifications ---


class Universe(ABC):
    """A set of securities that may change at every session."""

    @abstractmethod
    def mask(self, data: UniverseData, sessions: pd.DatetimeIndex) -> pd.DataFrame:
        """Return membership as a boolean frame ``sessions`` × ``security_id``."""
        ...

    def __or__(self, other: Universe) -> Universe:
        return Combined(self, other, "|")

    def __and__(self, other: Universe) -> Universe:
        return Combined(self, other, "&")

    def __sub__(self, other: Universe) -> Universe:
        return Combined(self, other, "-")


@dataclass(frozen=True)
class Index(Universe):
    """
    Members of a B3 index, period by period.

    Parameters
    ----------
    code : str
        Index code as B3 names its sheet today (``IBOV``, ``IBXX``, ``SMLL``...).
    fill : bool, default True
        Keep the periods filled by the liquidity rule; ``False`` leaves them empty.
    """

    code: str
    fill: bool = True

    def mask(self, data: UniverseData, sessions: pd.DatetimeIndex) -> pd.DataFrame:
        table = data.constituents
        rows = table[(table["index"] == self.code) & table["security_id"].notna()]
        if not self.fill:
            rows = rows[rows["source"] != PortfolioSource.LIQUIDITY_RULE.value]
        members = {
            str(period): set(group["security_id"].astype(str))
            for period, group in rows.groupby("period")
        }
        return _by_period(sessions, data.periods, members)


@dataclass(frozen=True, init=False)
class Tickers(Universe):
    """
    A fixed list of tickers, each followed across renames.

    Parameters
    ----------
    tickers : sequence of str
        B3 tickers (``PETR4``, ``VALE3``...).
    """

    tickers: tuple[str, ...]

    def __init__(self, tickers: Sequence[str] | str) -> None:
        values = (tickers,) if isinstance(tickers, str) else tuple(dict.fromkeys(tickers))
        object.__setattr__(self, "tickers", values)

    def mask(self, data: UniverseData, sessions: pd.DatetimeIndex) -> pd.DataFrame:
        table = data.master.table
        round_lot = table[table["market_type"] == ROUND_LOT_MARKET]
        known = set(round_lot["ticker"].astype(str))
        for ticker in self.tickers:
            if ticker not in known:
                logger.warning("Ticker %s is not in the security master; left out.", ticker)
        securities = set(round_lot.loc[round_lot["ticker"].isin(self.tickers), "security_id"])
        own = round_lot[round_lot["security_id"].isin(securities)]
        lives = own.groupby(own["security_id"].astype(str)).agg(
            first=("first_session", "min"), last=("last_session", "max")
        )
        days = sessions.normalize()
        columns: dict[str, object] = {
            str(security): (days >= first) & (days <= last)
            for security, first, last in zip(
                lives.index, lives["first"], lives["last"], strict=True
            )
        }
        return _frame(sessions, columns)


@dataclass(frozen=True)
class LiquidityRule(Universe):
    """
    The most traded shares at each period's base session.

    Parameters
    ----------
    top : int
        Number of securities per period.
    lookback_months : int, default 12
        Window before the base session.
    min_presence : float, default 0.95
        Share of the window's sessions a security must have traded.
    """

    top: int
    lookback_months: int = IBOV_LIQUIDITY.lookback_months
    min_presence: float = IBOV_LIQUIDITY.min_presence

    def __post_init__(self) -> None:
        if self.top < 1:
            raise ValueError(f"top must be at least 1, got {self.top}.")
        LiquiditySettings(self.lookback_months, self.min_presence)

    def mask(self, data: UniverseData, sessions: pd.DatetimeIndex) -> pd.DataFrame:
        settings = LiquiditySettings(self.lookback_months, self.min_presence)
        traded = data.traded
        wanted = set(assign_periods(sessions, data.periods).dropna().astype(str))
        members: dict[str, set[str]] = {}
        for label, start in zip(
            data.periods["period"], data.periods["effective_from"], strict=True
        ):
            if str(label) not in wanted:
                continue
            base = base_session(traded.calendar, start)
            if base is not None:
                members[str(label)] = set(select_by_liquidity(traded, base, self.top, settings))
        return _by_period(sessions, data.periods, members)


_OPERATIONS: Final[dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]]] = {
    "|": lambda left, right: left | right,
    "&": lambda left, right: left & right,
    "-": lambda left, right: left & ~right,
}


@dataclass(frozen=True)
class Combined(Universe):
    """Two universes combined by union (``|``), intersection (``&``) or difference (``-``)."""

    left: Universe
    right: Universe
    operation: str

    def __post_init__(self) -> None:
        if self.operation not in _OPERATIONS:
            raise ValueError(
                f"operation must be one of {sorted(_OPERATIONS)}, got {self.operation!r}."
            )

    def mask(self, data: UniverseData, sessions: pd.DatetimeIndex) -> pd.DataFrame:
        left = self.left.mask(data, sessions)
        right = self.right.mask(data, sessions)
        columns = left.columns.union(right.columns)
        left = left.reindex(columns=columns, fill_value=False)
        right = right.reindex(columns=columns, fill_value=False)
        return _OPERATIONS[self.operation](left, right)


# --- Entry point ---


def universe_mask(
    universe: Universe,
    sessions: pd.DatetimeIndex,
    store: DataStore | None = None,
    *,
    data: UniverseData | None = None,
) -> pd.DataFrame:
    """
    Return the membership of ``universe`` at each session.

    Parameters
    ----------
    universe : Universe
        ``Index``, ``Tickers``, ``LiquidityRule`` or a combination.
    sessions : pandas.DatetimeIndex
        Sessions to evaluate, in the order wanted.
    store : DataStore, optional
        Defaults to the configured data root (``DataStore.open()``).
    data : UniverseData, optional
        Tables already read, to share between calls (``B3Dataset`` does);
        ``store`` is not used when given.

    Returns
    -------
    pandas.DataFrame
        Boolean, indexed by ``sessions`` (``session_date``), one column per
        ``security_id`` that is a member on at least one session, sorted.

    Raises
    ------
    UniverseNotBuiltError
        If the universe was never built.

    Examples
    --------
    >>> mask = universe_mask(Index("IBXX") - Index("IBOV"), sessions)  # doctest: +SKIP
    """
    tables = data if data is not None else UniverseData(store or DataStore.open())
    result = universe.mask(tables, sessions)
    result = result.loc[:, result.any(axis=0)]
    result = result.reindex(columns=sorted(result.columns)).astype(bool)
    result.index.name = SESSION_AXIS
    result.columns.name = SECURITY_AXIS
    return result


# --- Helpers ---


def _by_period(
    sessions: pd.DatetimeIndex, periods: pd.DataFrame, members: dict[str, set[str]]
) -> pd.DataFrame:
    """Each session takes the members of the period in force."""
    securities = sorted(set().union(*members.values())) if members else []
    grid = pd.DataFrame(False, index=pd.Index(sorted(members), dtype=object), columns=securities)
    for label, chosen in members.items():
        grid.loc[label, sorted(chosen)] = True
    labels = assign_periods(sessions, periods)
    keys = [str(label) if isinstance(label, str) else "" for label in labels.astype(object)]
    rows = grid.reindex(keys, fill_value=False)
    rows.index = sessions
    return rows.astype(bool)


def _frame(sessions: pd.DatetimeIndex, columns: dict[str, object]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=sessions, columns=pd.Index([], dtype=object), dtype=bool)
    return pd.DataFrame(columns, index=sessions).astype(bool)
