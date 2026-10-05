# ibexQuant\src\ibexQuant\analytics\_reduce.py

"""
Reduction helpers shared by scalar analytics.

Scalar analytics (metrics, distribution statistics, benchmark comparisons)
reduce each return series to one number. The helpers below hold the rules
they share, so each is written once:

- :func:`reduce_per_series` — the shape contract (Series in, float out;
  DataFrame in, one value per column), validation, and the rejection of
  empty input.
- :func:`sample_standard_deviation` — ``ddof=1``, exactly zero for constant
  values.
- :func:`safe_ratio` — ``NaN`` instead of division by zero.

What does not belong here
-------------------------
The statistics themselves, which live in the public modules.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from ibexQuant.analytics._types import ReturnsT
from ibexQuant.analytics._validation import require_returns


def reduce_per_series(
    returns: pd.Series | pd.DataFrame, statistic: Callable[[pd.Series], float]
) -> float | pd.Series:
    """
    Apply *statistic* to each series over its observed period.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    statistic : callable
        Reduces one series, without missing values, to a float.

    Returns
    -------
    float or pd.Series
        The statistic for a Series; one value per column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    # Validation guarantees NaN only at the edges, so dropna just trims them.
    if isinstance(returns, pd.Series):
        series = _require_observations(require_returns(returns))
        return statistic(series.dropna())

    # Anything that is not a Series is validated here, so non-pandas input
    # still gets a TypeError.
    frame = _require_observations(require_returns(returns))
    values = [statistic(column.dropna()) for _, column in frame.items()]
    return pd.Series(values, index=frame.columns, dtype="float64")


def sample_standard_deviation(values: pd.Series) -> float:
    """
    Sample standard deviation (``ddof=1``), exactly zero for constant values.

    Parameters
    ----------
    values : pd.Series
        Values without missing entries.

    Returns
    -------
    float
        ``NaN`` for fewer than two values.
    """
    if len(values) < 2:
        return math.nan
    # The mean of a repeated value is not always that value in floating point
    # (e.g. 0.1), which would leave ~1e-17 of spurious dispersion and turn a
    # ratio over zero risk into a huge number instead of NaN.
    if is_constant(values):
        return 0.0
    return float(values.std(ddof=1))


def is_constant(values: pd.Series) -> bool:
    """Whether every value is identical, compared exactly."""
    return bool(values.max() == values.min())


def safe_ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``; ``NaN`` when the denominator is zero or NaN."""
    if math.isnan(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator


def _require_observations(returns: ReturnsT) -> ReturnsT:
    """Reject empty input: a statistic of nothing signals a bug upstream."""
    if returns.empty:
        raise ValueError("returns must have at least one observation.")
    return returns
