# ibexQuant\src\ibexQuant\backtest\walkforward\universe.py

"""
Layer 2 of the walk-forward: which securities a window may trade (ADR 0017, B1.5).

Responsibility
--------------
Turn a :class:`Window` into a :class:`Selection` of candidates, and say where a window must
be cut because the universe changes inside it.

Contract
--------
- A :class:`UniverseRule` reads the dataset and the window, nothing else; it never looks at
  bars or at a strategy.
- :class:`StaticUniverse` takes any universe argument of ``ds.panel``; its candidates are
  its members on any session of the test window, and the test panel keeps its mask. It
  never cuts.
- :class:`IndexComposition` takes the composition of the quadrimester that contains the
  test window, from its first session (the 3rd preview is known before the month turns).
  With ``at_rebalance="split"`` it cuts a window at each quadrimester start inside it.
  A period without a composition raises :class:`WalkForwardError`.

What does not belong here
-------------------------
- Whether a candidate has enough data: layer 3.
- Selection that changes bar by bar from the strategy's results: the strategy's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd

from ibexQuant.backtest.walkforward.errors import WalkForwardError
from ibexQuant.backtest.walkforward.schedule import MONTH, Window
from ibexQuant.data.dataset import B3Dataset, as_universe
from ibexQuant.data.universe import Universe

AT_REBALANCE: Final[tuple[str, ...]] = ("split", "hold")
QUADRIMESTER_MONTHS: Final[int] = 4


@dataclass(frozen=True)
class Selection:
    """
    The candidates of a window.

    Attributes
    ----------
    security_ids : tuple of str
        Candidate ``security_id`` values.
    unresolved : tuple of str
        Tickers of members with no ``security_id``; dropped as ``unresolved``.
    universe : Universe or None
        A universe whose own mask the test panel keeps; ``None`` for a fixed set.
    index, period, source : str or None
        The index period used, when there is one.
    """

    security_ids: tuple[str, ...]
    unresolved: tuple[str, ...] = ()
    universe: Universe | None = None
    index: str | None = None
    period: str | None = None
    source: str | None = None


class UniverseRule(ABC):
    """Which securities a window may trade, and where the universe changes inside it."""

    @abstractmethod
    def select(self, ds: B3Dataset, window: Window) -> Selection:
        """
        Return the candidates of ``window``.

        Raises
        ------
        WalkForwardError
            If the window has no candidate the rule can name.
        """
        ...

    def cuts(self, window: Window) -> tuple[pd.Period, ...]:
        """Return the months, after the first test month, that start a new universe."""
        return ()


class StaticUniverse(UniverseRule):
    """
    Any universe of ``ds.panel``: a ticker list, an index code, ``Index``, combinations.

    Parameters
    ----------
    universe : Universe, str or sequence of str
        The universe argument, as ``ds.panel`` takes it.

    Raises
    ------
    TypeError
        If ``universe`` is not a ``Universe``, a string or a sequence of strings.
    """

    def __init__(self, universe: Universe | str | Sequence[str]) -> None:
        self.universe: Universe = as_universe(universe)

    def select(self, ds: B3Dataset, window: Window) -> Selection:
        members = ds.members(self.universe, window.test_start, window.test_end)
        if members.columns.empty:
            raise WalkForwardError(
                f"{self.universe} has no member from {window.test_start.date()} "
                f"to {window.test_end.date()}."
            )
        return Selection(security_ids=tuple(members.columns), universe=self.universe)

    def __repr__(self) -> str:
        return f"StaticUniverse({self.universe})"


class IndexComposition(UniverseRule):
    """
    The members of an index's quadrimester portfolio, from the quadrimester's first session.

    Parameters
    ----------
    index : str
        Index code (``IBOV``, ``IBXX``...).
    at_rebalance : {"split", "hold"}, default "split"
        ``"split"`` cuts a test window at each quadrimester start inside it; ``"hold"``
        keeps the composition of the window's start to its end.

    Raises
    ------
    ValueError
        If ``at_rebalance`` is unknown.
    """

    def __init__(self, index: str, at_rebalance: Literal["split", "hold"] = "split") -> None:
        if at_rebalance not in AT_REBALANCE:
            raise ValueError(
                f"at_rebalance must be one of {list(AT_REBALANCE)}, got {at_rebalance!r}."
            )
        self.index: str = index
        self.at_rebalance: str = at_rebalance

    def select(self, ds: B3Dataset, window: Window) -> Selection:
        period = quadrimester(window.test_start.to_period(MONTH))
        where = f"(test window from {window.test_start.date()})"
        try:
            members = ds.composition(self.index, period)
        except ValueError as error:
            raise WalkForwardError(f"{error} {where}") from error
        if members.empty:
            raise WalkForwardError(f"{self.index} has no composition in period {period} {where}.")
        resolved = members["security_id"].fillna("") != ""
        return Selection(
            security_ids=tuple(members.loc[resolved, "security_id"].astype(str)),
            unresolved=tuple(members.loc[~resolved, "ticker"].astype(str)),
            index=self.index,
            period=period,
            source=", ".join(sorted(set(members["source"].astype(str)))),
        )

    def cuts(self, window: Window) -> tuple[pd.Period, ...]:
        if self.at_rebalance == "hold":
            return ()
        first = window.test_start.to_period(MONTH)
        last = window.test_end.to_period(MONTH)
        months = pd.period_range(first + 1, last, freq=MONTH)
        return tuple(month for month in months if (month.month - 1) % QUADRIMESTER_MONTHS == 0)

    def __repr__(self) -> str:
        return f"IndexComposition({self.index!r}, at_rebalance={self.at_rebalance!r})"


def quadrimester(month: pd.Period) -> str:
    """Return the label of the index period whose quadrimester contains ``month``."""
    start = (month.month - 1) // QUADRIMESTER_MONTHS * QUADRIMESTER_MONTHS + 1
    return f"{month.year}-{start:02d}"
