# ibexQuant\src\ibexQuant\features\kernels\moving_averages.py

"""
Moving average kernels.

Provides, each with its incremental state:

- :func:`sma` / :class:`SmaState` — simple (equally weighted) moving average
  of the last *window* observations. Strict: no value until *window*
  observations are available; lookback ``window - 1``.

Runs in observation time (ADR 0003). Note that a moving average of prices is
not scale invariant; on adjusted prices, use it through a ratio such as
``close / sma`` (ADR 0005).
"""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from ibexQuant.features.kernels._observation_time import in_observation_time
from ibexQuant.features.kernels._validation import require_frame, require_integer
from ibexQuant.features.kernels._window import FixedWindowState

# --- Batch kernels ---


def sma(values: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Simple moving average of the last *window* observations of each column.

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
        values, lambda block: block.rolling(window, min_periods=window).mean()
    )


# --- Incremental states ---


class SmaState(FixedWindowState):
    """
    Incremental counterpart of :func:`sma`.

    Parameters
    ----------
    window : int
        Number of observations. Must be >= 1.
    """

    def _compute(self, window: deque[float]) -> float:
        return math.fsum(window) / len(window)
