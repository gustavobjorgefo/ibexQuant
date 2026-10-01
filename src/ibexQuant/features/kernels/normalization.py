# ibexQuant\src\ibexQuant\features\kernels\normalization.py

"""
Normalization kernels.

Provides, with its incremental state:

- :func:`rolling_zscore` / :class:`RollingZScoreState` — distance of the
  current value from the mean of the last *window* observations, in units of
  their standard deviation. Lookback ``window - 1``.

The window **includes** the current value, like every other window kernel:
the z-score equals ``(x - sma(x, w)) / rolling_std(x, w)``, the logic of
Bollinger bands. A consequence is that it is bounded: with ``ddof=1`` its
absolute value never exceeds ``(window - 1) / sqrt(window)`` — about 4.25
for a 20-bar window — however extreme the observation, because the
observation itself pulls the mean and the deviation towards it.

A z-score against the *previous* window (unbounded, measuring surprise
relative to history) is built by composing features with a one-observation
lag on the statistics.

Z-scores are scale invariant, so they are valid on adjusted prices
(ADR 0005). A window with zero dispersion has no z-score: ``NaN``.
"""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from ibexQuant._validation import require_integer
from ibexQuant.features.kernels._observation_time import in_observation_time
from ibexQuant.features.kernels._validation import require_frame
from ibexQuant.features.kernels._window import FixedWindowState

# --- Batch kernels ---


def rolling_zscore(values: pd.DataFrame, window: int, ddof: int = 1) -> pd.DataFrame:
    """
    Z-score of the current value within the last *window* observations.

    Parameters
    ----------
    values : pd.DataFrame
        Wide values, timestamp x symbol.
    window : int
        Number of observations, current one included. Must be >= 2 and
        > *ddof*.
    ddof : int, default 1
        Delta degrees of freedom of the standard deviation.

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` until *window* observations are available,
        wherever *values* is ``NaN``, and where the window has zero
        dispersion.

    Raises
    ------
    TypeError
        If *values* is not a DataFrame, or *window* or *ddof* is not an
        integer.
    ValueError
        If *ddof* is < 0, or *window* is < 2 or not > *ddof*.
    """
    values = require_frame(values)
    ddof = require_integer("ddof", ddof, minimum=0)
    window = require_integer("window", window, minimum=max(2, ddof + 1))

    def compute(block: pd.DataFrame) -> pd.DataFrame:
        rolling = block.rolling(window, min_periods=window)
        # Zero dispersion is detected exactly (all values equal) rather than
        # through the computed deviation, which may carry ~1e-17 of noise.
        dispersed: pd.DataFrame = rolling.max() > rolling.min()
        return (block - rolling.mean()) / rolling.std(ddof=ddof).where(dispersed)

    return in_observation_time(values, compute)


# --- Incremental states ---


class RollingZScoreState(FixedWindowState):
    """
    Incremental counterpart of :func:`rolling_zscore`.

    Parameters
    ----------
    window : int
        Number of observations, current one included. Must be >= 2 and
        > *ddof*.
    ddof : int, default 1
        Delta degrees of freedom of the standard deviation.

    Raises
    ------
    TypeError
        If *window* or *ddof* is not an integer.
    ValueError
        If *ddof* is < 0, or *window* is < 2 or not > *ddof*.
    """

    def __init__(self, window: int, ddof: int = 1) -> None:
        self._ddof: int = require_integer("ddof", ddof, minimum=0)
        super().__init__(require_integer("window", window, minimum=max(2, self._ddof + 1)))

    def _compute(self, window: deque[float]) -> float:
        # Zero dispersion is detected exactly: fsum(window) / n does not always
        # reproduce a repeated value (e.g. 0.1), leaving ~1e-17 of spurious
        # deviation that would turn a constant window into a z-score of +-0.8.
        if max(window) == min(window):
            return math.nan
        mean: float = math.fsum(window) / len(window)
        squared_deviations: float = math.fsum((value - mean) ** 2 for value in window)
        deviation: float = math.sqrt(squared_deviations / (len(window) - self._ddof))
        return (window[-1] - mean) / deviation
