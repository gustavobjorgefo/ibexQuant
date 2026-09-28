# ibexQuant\src\ibexQuant\features\kernels\rolling.py

"""
Generic fixed-window kernels.

Provides, each with its incremental state:

- :func:`rolling_sum` / :class:`RollingSumState` — sum of the last *window*
  observations. Over log returns, it is the log return of the whole window
  (momentum).
- :func:`rolling_std` / :class:`RollingStdState` — standard deviation of the
  last *window* observations, ``ddof=1`` by default (sample estimator, as in
  pandas).

Both run in observation time (ADR 0003) and are strict: no value is emitted
until *window* observations are available (``min_periods = window``). The
lookback is ``window - 1``.
"""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from ibexQuant.features.kernels._observation_time import in_observation_time
from ibexQuant.features.kernels._validation import require_frame, require_integer
from ibexQuant.features.kernels._window import FixedWindowState

# --- Batch kernels ---


def rolling_sum(values: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Sum the last *window* observations of each column.

    Parameters
    ----------
    values : pd.DataFrame
        Wide values, timestamp x symbol.
    window : int
        Number of observations. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` until *window* observations are available and
        wherever *values* is ``NaN``.

    Raises
    ------
    TypeError
        If *values* is not a DataFrame or *window* is not an integer.
    ValueError
        If *window* is < 1.
    """
    values = require_frame(values)
    window = require_integer("window", window, minimum=1)
    return in_observation_time(
        values, lambda block: block.rolling(window, min_periods=window).sum()
    )


def rolling_std(values: pd.DataFrame, window: int, ddof: int = 1) -> pd.DataFrame:
    """
    Standard deviation of the last *window* observations of each column.

    Parameters
    ----------
    values : pd.DataFrame
        Wide values, timestamp x symbol.
    window : int
        Number of observations. Must be > *ddof*.
    ddof : int, default 1
        Delta degrees of freedom: the divisor is ``window - ddof``. ``1``
        gives the sample estimator (pandas default); ``0`` the population
        one (NumPy default).

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` until *window* observations are available and
        wherever *values* is ``NaN``.

    Raises
    ------
    TypeError
        If *values* is not a DataFrame, or *window* or *ddof* is not an
        integer.
    ValueError
        If *ddof* is < 0 or *window* is not > *ddof*.
    """
    values = require_frame(values)
    ddof = require_integer("ddof", ddof, minimum=0)
    window = require_integer("window", window, minimum=ddof + 1)
    return in_observation_time(
        values, lambda block: block.rolling(window, min_periods=window).std(ddof=ddof)
    )


# --- Incremental states ---


class RollingSumState(FixedWindowState):
    """
    Incremental counterpart of :func:`rolling_sum`.

    Parameters
    ----------
    window : int
        Number of observations. Must be >= 1.
    """

    def _compute(self, window: deque[float]) -> float:
        return math.fsum(window)


class RollingStdState(FixedWindowState):
    """
    Incremental counterpart of :func:`rolling_std`.

    Parameters
    ----------
    window : int
        Number of observations. Must be > *ddof*.
    ddof : int, default 1
        Delta degrees of freedom.

    Raises
    ------
    TypeError
        If *window* or *ddof* is not an integer.
    ValueError
        If *ddof* is < 0 or *window* is not > *ddof*.
    """

    def __init__(self, window: int, ddof: int = 1) -> None:
        self._ddof: int = require_integer("ddof", ddof, minimum=0)
        super().__init__(require_integer("window", window, minimum=self._ddof + 1))

    def _compute(self, window: deque[float]) -> float:
        # Two-pass algorithm: exact mean first, then squared deviations, which
        # avoids the cancellation of the naive sum-of-squares formula.
        mean: float = math.fsum(window) / len(window)
        squared_deviations: float = math.fsum((value - mean) ** 2 for value in window)
        return math.sqrt(squared_deviations / (len(window) - self._ddof))
