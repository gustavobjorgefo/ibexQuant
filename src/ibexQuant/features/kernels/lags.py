# ibexQuant\src\ibexQuant\features\kernels\lags.py

"""
Lag kernel.

Provides, with its incremental state:

- :func:`lag` / :class:`LagState` — the value observed *periods*
  observations earlier. Lookback ``periods``.

Lags are counted in **observations**, not sessions (observation time,
ADR 0003). For an asset without gaps the two coincide. After a suspension,
``lag(1)`` on the day trading resumes is the last price before the
suspension, several sessions back. This keeps ``log_returns(p)`` equal to
``log(p) - lag(log(p), 1)`` on every row, gaps included, and avoids erasing
``periods`` rows after every gap. How many sessions an observation spans is
a separate piece of information, exposed by its own feature.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

from ibexQuant.features.kernels._observation_time import in_observation_time
from ibexQuant.features.kernels._validation import require_frame, require_integer
from ibexQuant.features.kernels._window import FixedWindowState

# --- Batch kernels ---


def lag(values: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Value observed *periods* observations earlier, for each column.

    Parameters
    ----------
    values : pd.DataFrame
        Wide values, timestamp x symbol.
    periods : int
        Number of observations to look back. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` for each asset's first *periods* observations and
        wherever *values* is ``NaN``.

    Raises
    ------
    TypeError
        If *values* is not a DataFrame or *periods* is not an integer.
    ValueError
        If *periods* is < 1.
    """
    values = require_frame(values)
    periods = require_integer("periods", periods, minimum=1)
    return in_observation_time(values, lambda block: block.shift(periods))


# --- Incremental states ---


class LagState(FixedWindowState):
    """
    Incremental counterpart of :func:`lag`.

    Keeps the last ``periods + 1`` observations and returns the oldest.

    Parameters
    ----------
    periods : int
        Number of observations to look back. Must be >= 1.

    Raises
    ------
    TypeError
        If *periods* is not an integer.
    ValueError
        If *periods* is < 1.
    """

    def __init__(self, periods: int) -> None:
        super().__init__(require_integer("periods", periods, minimum=1) + 1)

    def _compute(self, window: deque[float]) -> float:
        return window[0]
