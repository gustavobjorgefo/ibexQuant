# ibexQuant\src\ibexQuant\features\kernels\returns.py

"""
Return kernels: simple and logarithmic returns from prices.

Provides, each with its incremental state:

- :func:`simple_returns` / :class:`SimpleReturnState` —
  ``p_t / p_{t-1} - 1``. Additive across assets: a portfolio's return is the
  weighted sum of its assets' simple returns. Use for portfolio aggregation.
- :func:`log_returns` / :class:`LogReturnState` —
  ``ln(p_t / p_{t-1})``. Additive over time: the sum of *n* consecutive log
  returns is the log return over the *n* periods. Use for features (e.g.
  momentum as a rolling sum).

Both follow the observation-time policy (ADR 0003): after a gap, the return
is computed against the asset's previous observation, and the output is
``NaN`` while the asset is not observable. Both have a lookback of one bar.

Prices must be strictly positive. Non-positive prices raise instead of
silently producing ``NaN`` or ``inf``.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Final, cast

import numpy as np
import pandas as pd

from ibexQuant.features.kernels._observation_time import in_observation_time
from ibexQuant.features.kernels._validation import require_frame
from ibexQuant.features.kernels.state import IncrementalState

RETURNS_LOOKBACK: Final[int] = 1


# --- Batch kernels ---


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns, ``p_t / p_{t-1} - 1``, in observation time.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide prices, timestamp x symbol. Every non-``NaN`` value must be
        strictly positive.

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` at each asset's first observation and wherever
        *prices* is ``NaN``.

    Raises
    ------
    TypeError
        If *prices* is not a DataFrame.
    ValueError
        If any price is zero or negative.
    """
    prices = _require_positive_prices(require_frame(prices))
    return in_observation_time(prices, lambda block: block / block.shift(1) - 1.0)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns, ``ln(p_t / p_{t-1})``, in observation time.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide prices, timestamp x symbol. Every non-``NaN`` value must be
        strictly positive.

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` at each asset's first observation and wherever
        *prices* is ``NaN``.

    Raises
    ------
    TypeError
        If *prices* is not a DataFrame.
    ValueError
        If any price is zero or negative.
    """
    prices = _require_positive_prices(require_frame(prices))
    return in_observation_time(prices, _log_ratio)


# --- Incremental states ---


class _ReturnState(IncrementalState):
    """Shared bookkeeping of return states: the previous observed price."""

    def __init__(self) -> None:
        self._previous: float | None = None

    @property
    def lookback(self) -> int:
        return RETURNS_LOOKBACK

    def update(self, value: float) -> float:
        if math.isnan(value):
            return math.nan
        if value <= 0.0:
            raise ValueError(f"Prices must be strictly positive, got {value}.")

        previous: float | None = self._previous
        self._previous = value
        if previous is None:
            return math.nan
        return self._compute(value, previous)

    @abstractmethod
    def _compute(self, price: float, previous: float) -> float: ...


class SimpleReturnState(_ReturnState):
    """Incremental counterpart of :func:`simple_returns`."""

    def _compute(self, price: float, previous: float) -> float:
        return price / previous - 1.0


class LogReturnState(_ReturnState):
    """Incremental counterpart of :func:`log_returns`."""

    def _compute(self, price: float, previous: float) -> float:
        return math.log(price / previous)


# --- Helpers ---


def _require_positive_prices(prices: pd.DataFrame) -> pd.DataFrame:
    non_positive: pd.DataFrame = prices.le(0.0)
    if non_positive.to_numpy().any():
        symbols: list[str] = list(prices.columns[non_positive.any()])
        raise ValueError(f"Prices must be strictly positive; non-positive values in {symbols}.")
    return prices


def _log_ratio(block: pd.DataFrame) -> pd.DataFrame:
    # NumPy ufuncs applied to a DataFrame return a DataFrame (__array_ufunc__),
    # but the stubs type them as ndarray.
    return cast(pd.DataFrame, np.log(block / block.shift(1)))
