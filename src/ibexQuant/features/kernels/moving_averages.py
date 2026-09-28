# ibexQuant\src\ibexQuant\features\kernels\moving_averages.py

"""
Moving average kernels.

Provides, each with its incremental state:

- :func:`sma` / :class:`SmaState` — simple (equally weighted) moving average
  of the last *window* observations. Strict: no value until *window*
  observations are available; lookback ``window - 1``.
- :func:`ema` / :class:`EmaState` — exponential moving average with
  smoothing factor *alpha* (weight of the newest observation, ADR 0008
  D1.7). Starts from the first observation and emits values once the
  starting point weighs at most *tolerance* (ADR 0008 D1.6).

Both run in observation time (ADR 0003). Note that a moving average of prices is
not scale invariant; on adjusted prices, use it through a ratio such as
``close / sma`` or ``close / ema`` (ADR 0005).
"""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from ibexQuant.features.kernels._observation_time import in_observation_time
from ibexQuant.features.kernels._validation import require_frame, require_integer
from ibexQuant.features.kernels._window import FixedWindowState
from ibexQuant.features.kernels.exponential import (
    DEFAULT_EXPONENTIAL_TOLERANCE,
    exponential_lookback,
)
from ibexQuant.features.kernels.state import IncrementalState

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


def ema(
    values: pd.DataFrame,
    alpha: float,
    tolerance: float = DEFAULT_EXPONENTIAL_TOLERANCE,
) -> pd.DataFrame:
    """
    Exponential moving average of each column.

    Computes ``ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}``, starting from
    ``ema_0 = x_0`` (pandas ``adjust=False``).

    Parameters
    ----------
    values : pd.DataFrame
        Wide values, timestamp x symbol.
    alpha : float
        Smoothing factor in ``(0, 1)``: the weight of the newest observation.
        Convert other parameterizations with ``alpha_from_span``,
        ``alpha_from_halflife`` or ``alpha_from_decay``.
    tolerance : float, default ``DEFAULT_EXPONENTIAL_TOLERANCE``
        Maximum weight of the starting point in any emitted value.

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` until ``exponential_lookback(alpha, tolerance) + 1``
        observations are available and wherever *values* is ``NaN``.

    Raises
    ------
    TypeError
        If *values* is not a DataFrame, or *alpha* or *tolerance* is not a
        real number.
    ValueError
        If *alpha* or *tolerance* is outside ``(0, 1)``.
    """
    values = require_frame(values)
    observations: int = exponential_lookback(alpha, tolerance) + 1
    return in_observation_time(
        values,
        lambda block: block.ewm(alpha=alpha, adjust=False, min_periods=observations).mean(),
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


class EmaState(IncrementalState):
    """
    Incremental counterpart of :func:`ema`.

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
        self._lookback: int = exponential_lookback(alpha, tolerance)
        self._alpha: float = float(alpha)
        self._average: float | None = None
        self._observations: int = 0

    @property
    def lookback(self) -> int:
        return self._lookback

    def update(self, value: float) -> float:
        if math.isnan(value):
            return math.nan

        if self._average is None:
            self._average = value
        else:
            self._average = self._alpha * value + (1.0 - self._alpha) * self._average
        self._observations += 1

        if self._observations <= self._lookback:
            return math.nan
        return self._average
