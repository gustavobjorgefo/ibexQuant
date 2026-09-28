# ibexQuant\src\ibexQuant\features\kernels\volatility.py

"""
Volatility kernels.

Provides, each with its incremental state:

- :func:`historical_volatility` / :class:`HistoricalVolatilityState` —
  close-to-close volatility: the sample standard deviation of the last
  *window* returns. A domain-named entry point to
  :func:`ibexQuant.features.kernels.rolling.rolling_std`, which it calls
  unchanged; lookback ``window - 1``.
- :func:`ewma_volatility` / :class:`EwmaVolatilityState` — RiskMetrics
  exponentially weighted volatility of returns (ADR 0008, D1.8):

      sigma2_t = (1 - alpha) * sigma2_{t-1} + alpha * r_t ** 2

  The mean return is assumed to be zero. The recursion starts from the first
  squared return, ``sigma2_0 = r_0 ** 2``, and values are emitted once that
  starting point weighs at most *tolerance* (ADR 0008, D1.6).

The two estimators differ in one convention besides the weighting:
historical volatility subtracts the window mean, as the classic
close-to-close estimator does, while RiskMetrics assumes a zero mean. Both
are market standards; for daily returns the difference is small.

Annualization is not applied: multiply by ``sqrt(periods_per_year)`` where a
readable scale is needed.

Timing
------
Both volatilities at *t* include the return of *t*: it is known only after
bar *t* closes. It is causal and is the natural forecast for *t + 1*. To size
a position traded within bar *t* itself, lag it by one bar.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

from ibexQuant.features.kernels._observation_time import in_observation_time
from ibexQuant.features.kernels._validation import require_frame
from ibexQuant.features.kernels.exponential import (
    DEFAULT_EXPONENTIAL_TOLERANCE,
    exponential_lookback,
)
from ibexQuant.features.kernels.moving_averages import EmaState
from ibexQuant.features.kernels.rolling import RollingStdState, rolling_std
from ibexQuant.features.kernels.state import IncrementalState

# --- Batch kernels ---


def historical_volatility(returns: pd.DataFrame, window: int, ddof: int = 1) -> pd.DataFrame:
    """
    Close-to-close volatility: standard deviation of the last *window* returns.

    Identical to :func:`ibexQuant.features.kernels.rolling.rolling_std`; this
    name only states the intent when the input is a return series.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide returns, timestamp x symbol (typically log returns).
    window : int
        Number of returns. Must be > *ddof*.
    ddof : int, default 1
        Delta degrees of freedom (``1``: sample estimator).

    Returns
    -------
    pd.DataFrame
        Same shape, per-period (not annualized) volatility. ``NaN`` until
        *window* observations are available and wherever *returns* is
        ``NaN``. The value at *t* includes the return of *t*; lag it to size
        positions within bar *t*.

    Raises
    ------
    TypeError
        If *returns* is not a DataFrame, or *window* or *ddof* is not an
        integer.
    ValueError
        If *ddof* is < 0 or *window* is not > *ddof*.
    """
    return rolling_std(returns, window, ddof)


def ewma_volatility(
    returns: pd.DataFrame,
    alpha: float,
    tolerance: float = DEFAULT_EXPONENTIAL_TOLERANCE,
) -> pd.DataFrame:
    """
    RiskMetrics exponentially weighted volatility of each column.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide returns, timestamp x symbol (typically log returns).
    alpha : float
        Smoothing factor in ``(0, 1)``: the weight of the newest squared
        return. For the RiskMetrics daily standard use
        ``alpha_from_decay(0.94)``, i.e. ``alpha = 0.06`` — not ``0.94``.
    tolerance : float, default ``DEFAULT_EXPONENTIAL_TOLERANCE``
        Maximum weight of the starting point in any emitted value.

    Returns
    -------
    pd.DataFrame
        Same shape, per-period (not annualized) volatility. ``NaN`` until
        ``exponential_lookback(alpha, tolerance) + 1`` observations are
        available and wherever *returns* is ``NaN``. The value at *t*
        includes the return of *t*; lag it to size positions within bar *t*.

    Raises
    ------
    TypeError
        If *returns* is not a DataFrame, or *alpha* or *tolerance* is not a
        real number.
    ValueError
        If *alpha* or *tolerance* is outside ``(0, 1)``.
    """
    returns = require_frame(returns)
    observations: int = exponential_lookback(alpha, tolerance) + 1

    def compute(block: pd.DataFrame) -> pd.DataFrame:
        variance = (block**2).ewm(alpha=alpha, adjust=False, min_periods=observations).mean()
        # NumPy ufuncs applied to a DataFrame return a DataFrame (__array_ufunc__),
        # but the stubs type them as ndarray.
        return cast(pd.DataFrame, np.sqrt(variance))

    return in_observation_time(returns, compute)


# --- Incremental states ---


class HistoricalVolatilityState(RollingStdState):
    """
    Incremental counterpart of :func:`historical_volatility`.

    Identical to :class:`ibexQuant.features.kernels.rolling.RollingStdState`.

    Parameters
    ----------
    window : int
        Number of returns. Must be > *ddof*.
    ddof : int, default 1
        Delta degrees of freedom.
    """


class EwmaVolatilityState(IncrementalState):
    """
    Incremental counterpart of :func:`ewma_volatility`.

    An exponential average of squared returns, returned as its square root.

    Parameters
    ----------
    alpha : float
        Smoothing factor in ``(0, 1)``.
    tolerance : float, default ``DEFAULT_EXPONENTIAL_TOLERANCE``
        Maximum weight of the starting point in any emitted value.

    Raises
    ------
    TypeError
        If *alpha* or *tolerance* is not a real number.
    ValueError
        If *alpha* or *tolerance* is outside ``(0, 1)``.
    """

    def __init__(self, alpha: float, tolerance: float = DEFAULT_EXPONENTIAL_TOLERANCE) -> None:
        self._variance: EmaState = EmaState(alpha, tolerance)

    @property
    def lookback(self) -> int:
        return self._variance.lookback

    def update(self, value: float) -> float:
        # NaN propagates: the inner state ignores it and sqrt(NaN) is NaN.
        return math.sqrt(self._variance.update(value * value))
