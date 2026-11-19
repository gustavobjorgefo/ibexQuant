# ibexQuant\src\ibexQuant\backtest\walkforward\eligibility.py

"""
Layer 3 of the walk-forward: which candidates have the data to be fitted and traded.

Responsibility
--------------
Keep or drop each candidate of a window, with a reason (ADR 0017, B1.6).

Contract
--------
- An :class:`EligibilityRule` returns, for each ``security_id``, the reason it is dropped,
  or ``<NA>`` when it is kept. It reads data up to the decision session only.
- :class:`HasHistory` looks at the date of the first session, :class:`StillTrading` at
  the date of the last one, both over every ticker a security carried.
  :class:`MinimumPresence` counts the sessions with a close in the raw bars.
- Coverage: ``"train"`` is the ``warmup`` bars before the training window plus the window
  itself; ``"test"`` is the ``warmup`` bars before the decision session.

What does not belong here
-------------------------
- Which securities are candidates: layer 2.
- A selection that changes bar by bar from the strategy's own results (top N by rolling
  Sharpe ratio): it is part of the strategy (E1), or a stateful component of the event loop
  (E2, E3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Final, Literal

import pandas as pd

from ibexQuant.backtest.walkforward.schedule import Window
from ibexQuant.data.dataset import B3Dataset
from ibexQuant.data.universe import Securities

COVERS: Final[tuple[str, ...]] = ("train", "test")
SHORT_HISTORY: Final[str] = "short_history"
STOPPED_TRADING: Final[str] = "stopped_trading"
LOW_PRESENCE: Final[str] = "low_presence"


class EligibilityRule(ABC):
    """Keep or drop each candidate of a window, with a reason."""

    @abstractmethod
    def apply(
        self, ds: B3Dataset, window: Window, security_ids: Sequence[str], warmup: int
    ) -> pd.Series:
        """
        Return why each security is dropped.

        Parameters
        ----------
        ds : B3Dataset
            The dataset.
        window : Window
            The window being decided.
        security_ids : sequence of str
            The candidates still kept.
        warmup : int
            Bars of history the strategy needs before each range.

        Returns
        -------
        pandas.Series
            ``string`` dtype, indexed by ``security_id`` in the given order: the reason it
            is dropped, ``<NA>`` when it is kept.
        """
        ...


class HasHistory(EligibilityRule):
    """
    A security whose history starts early enough for the strategy.

    Parameters
    ----------
    cover : {"train", "test"}, default "train"
        ``"train"``: ``warmup`` bars before the training window plus the window itself.
        ``"test"``: ``warmup`` bars before the decision session (for studies that fit
        nothing).

    Raises
    ------
    ValueError
        If ``cover`` is unknown.
    """

    def __init__(self, cover: Literal["train", "test"] = "train") -> None:
        self.cover: str = _check_cover(cover)

    def apply(
        self, ds: B3Dataset, window: Window, security_ids: Sequence[str], warmup: int
    ) -> pd.Series:
        start = window.train_start if self.cover == "train" else window.decision_start
        sessions = ds.calendar.sessions
        position = int(sessions.searchsorted(start)) - warmup
        first = _spans(ds, security_ids)["first_session"]
        if position < 0:
            # A warmup that reaches before the base cannot be met by anyone.
            return _reasons(pd.Series(False, index=first.index), SHORT_HISTORY)
        return _reasons(first <= sessions[position], SHORT_HISTORY)

    def __repr__(self) -> str:
        return f"HasHistory(cover={self.cover!r})"


class StillTrading(EligibilityRule):
    """A security that still trades on the decision session or later."""

    def apply(
        self, ds: B3Dataset, window: Window, security_ids: Sequence[str], warmup: int
    ) -> pd.Series:
        last = _spans(ds, security_ids)["last_session"]
        return _reasons(last >= window.decision_start, STOPPED_TRADING)

    def __repr__(self) -> str:
        return "StillTrading()"


class MinimumPresence(EligibilityRule):
    """
    A security that traded on at least ``fraction`` of the sessions covered.

    Parameters
    ----------
    fraction : float
        Share of sessions with a trade, in ``(0, 1]``.
    cover : {"train", "test"}, default "train"
        ``"train"``: the training window. ``"test"``: the ``warmup`` bars before the
        decision session.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``(0, 1]`` or ``cover`` is unknown.
    """

    def __init__(self, fraction: float, cover: Literal["train", "test"] = "train") -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
        self.fraction: float = fraction
        self.cover: str = _check_cover(cover)

    def apply(
        self, ds: B3Dataset, window: Window, security_ids: Sequence[str], warmup: int
    ) -> pd.Series:
        if self.cover == "train":
            start, end = window.train_start, window.train_end
        else:
            end = window.decision_start
            start = ds.calendar.offset(end, warmup)
        panel = ds.panel(Securities(security_ids), start, end, adjustment="raw", fields=["close"])
        close = panel.field("close").reindex(columns=list(security_ids))
        presence = close.notna().mean()
        return _reasons(presence >= self.fraction, LOW_PRESENCE)

    def __repr__(self) -> str:
        return f"MinimumPresence({self.fraction}, cover={self.cover!r})"


def _check_cover(cover: str) -> str:
    if cover not in COVERS:
        raise ValueError(f"cover must be one of {list(COVERS)}, got {cover!r}.")
    return cover


def _spans(ds: B3Dataset, security_ids: Sequence[str]) -> pd.DataFrame:
    """First and last session of each security, over every ticker it carried."""
    history = ds.ticker_history(list(security_ids))
    spans = history.groupby("security_id").agg(
        first_session=("first_session", "min"), last_session=("last_session", "max")
    )
    return spans.reindex(list(security_ids))


def _reasons(kept: pd.Series, reason: str) -> pd.Series:
    """``reason`` where ``kept`` is false, ``<NA>`` elsewhere."""
    result = pd.Series(pd.NA, index=kept.index, dtype="string")
    result[~kept.astype(bool)] = reason
    return result
