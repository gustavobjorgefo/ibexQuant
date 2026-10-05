# ibexQuant\src\ibexQuant\analytics\distribution.py

"""
Distribution statistics: the shape of the returns of one period.

Provides:

- :func:`mean_return` and :func:`standard_deviation` — center and dispersion.
- :func:`skewness` and :func:`kurtosis` — asymmetry and tail weight.
- :func:`quantile`, :func:`best_period` and :func:`worst_period` — points of
  the distribution.
- :func:`positive_ratio` and :func:`negative_ratio` — how often returns are
  gains or losses.
- :func:`value_at_risk` and :func:`conditional_value_at_risk` — the loss
  tail.

Contract
--------
Inputs and output shapes follow :mod:`ibexQuant.analytics.metrics`: a Series
gives a ``float``, a DataFrame gives one value per column, each measured over
its own observed period. Empty input raises ``ValueError``; a statistic that
is undefined for a valid input is ``NaN``.

Conventions
-----------
- Every number is **per period**, never annualized: a daily series gives
  daily statistics. Annualized figures live in
  :mod:`ibexQuant.analytics.metrics`.
- Losses are negative numbers, as everywhere in ibexQuant: a 95% VaR of
  ``-0.021`` means a 2.1% loss, matching the sign of ``max_drawdown``.
- Skewness and kurtosis are the bias-adjusted sample estimates computed by
  pandas (the same as Excel's ``SKEW`` and ``KURT``); kurtosis is in excess
  of the normal distribution's, so a normal gives 0.
- Quantiles interpolate linearly between neighbouring observations, the
  pandas and NumPy default.
- A return of exactly zero is neither positive nor negative.

What does not belong here
-------------------------
Hypothesis tests (normality, significance of the mean or of the Sharpe
ratio), which return a statistic and a p-value rather than a description,
and belong in a separate inference module.
"""

from __future__ import annotations

import math
from typing import Final, cast, overload

import pandas as pd

from ibexQuant._validation import require_real
from ibexQuant.analytics._reduce import (
    is_constant,
    reduce_per_series,
    sample_standard_deviation,
)

# The bias-adjusted estimators need these many observations to be defined.
MINIMUM_SKEWNESS_OBSERVATIONS: Final[int] = 3
MINIMUM_KURTOSIS_OBSERVATIONS: Final[int] = 4


# --- Center and dispersion ---


@overload
def mean_return(returns: pd.Series) -> float: ...
@overload
def mean_return(returns: pd.DataFrame) -> pd.Series: ...
def mean_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the arithmetic mean return per period.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Mean return; one value per column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(returns, lambda series: float(series.mean()))


@overload
def standard_deviation(returns: pd.Series) -> float: ...
@overload
def standard_deviation(returns: pd.DataFrame) -> pd.Series: ...
def standard_deviation(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the sample standard deviation per period (``ddof=1``).

    The per-period counterpart of
    :func:`~ibexQuant.analytics.metrics.annualized_volatility`.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Standard deviation; one value per column for a DataFrame. Exactly
        ``0.0`` for constant returns, ``NaN`` for a single observation.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(returns, sample_standard_deviation)


# --- Shape ---


@overload
def skewness(returns: pd.Series) -> float: ...
@overload
def skewness(returns: pd.DataFrame) -> pd.Series: ...
def skewness(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the bias-adjusted sample skewness.

    Positive skewness means a longer right tail (rare large gains), negative
    a longer left tail (rare large losses).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Skewness; one value per column for a DataFrame. ``NaN`` for fewer
        than three observations or constant returns.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(
        returns,
        lambda series: (
            cast(float, series.skew())
            if _has_shape(series, MINIMUM_SKEWNESS_OBSERVATIONS)
            else math.nan
        ),
    )


@overload
def kurtosis(returns: pd.Series) -> float: ...
@overload
def kurtosis(returns: pd.DataFrame) -> pd.Series: ...
def kurtosis(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the bias-adjusted sample excess kurtosis.

    Measured against the normal distribution: 0 is normal, positive means
    fatter tails (extreme returns more frequent than a normal predicts).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Excess kurtosis; one value per column for a DataFrame. ``NaN`` for
        fewer than four observations or constant returns.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(
        returns,
        lambda series: (
            cast(float, series.kurt())
            if _has_shape(series, MINIMUM_KURTOSIS_OBSERVATIONS)
            else math.nan
        ),
    )


# --- Points of the distribution ---


@overload
def quantile(returns: pd.Series, q: float) -> float: ...
@overload
def quantile(returns: pd.DataFrame, q: float) -> pd.Series: ...
def quantile(returns: pd.Series | pd.DataFrame, q: float) -> float | pd.Series:
    """
    Compute the *q*-quantile of the returns, interpolating linearly.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    q : float
        Quantile, in ``[0, 1]``.

    Returns
    -------
    float or pd.Series
        Quantile; one value per column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, or *q* is not
        a real number in ``[0, 1]``.
    """
    q = _require_probability("q", q)
    return reduce_per_series(returns, lambda series: float(series.quantile(q)))


@overload
def best_period(returns: pd.Series) -> float: ...
@overload
def best_period(returns: pd.DataFrame) -> pd.Series: ...
def best_period(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the largest single-period return.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Maximum return; one value per column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(returns, lambda series: float(series.max()))


@overload
def worst_period(returns: pd.Series) -> float: ...
@overload
def worst_period(returns: pd.DataFrame) -> pd.Series: ...
def worst_period(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the smallest single-period return.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Minimum return (negative for a loss); one value per column for a
        DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(returns, lambda series: float(series.min()))


# --- Frequency of gains and losses ---


@overload
def positive_ratio(returns: pd.Series) -> float: ...
@overload
def positive_ratio(returns: pd.DataFrame) -> pd.Series: ...
def positive_ratio(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the fraction of periods with a positive return.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Fraction in ``[0, 1]``; one value per column for a DataFrame. Zero
        returns count as neither positive nor negative, so
        ``positive_ratio + negative_ratio`` can be below 1.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(returns, lambda series: float((series > 0.0).mean()))


@overload
def negative_ratio(returns: pd.Series) -> float: ...
@overload
def negative_ratio(returns: pd.DataFrame) -> pd.Series: ...
def negative_ratio(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the fraction of periods with a negative return.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Fraction in ``[0, 1]``; one value per column for a DataFrame. Zero
        returns count as neither positive nor negative.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return reduce_per_series(returns, lambda series: float((series < 0.0).mean()))


# --- Loss tail ---


@overload
def value_at_risk(returns: pd.Series, confidence: float = ...) -> float: ...
@overload
def value_at_risk(returns: pd.DataFrame, confidence: float = ...) -> pd.Series: ...
def value_at_risk(
    returns: pd.Series | pd.DataFrame, confidence: float = 0.95
) -> float | pd.Series:
    """
    Compute the historical value at risk, the ``1 - confidence`` quantile.

    A 95% VaR of ``-0.021`` reads: in 95% of the periods the return was above
    -2.1%; only the worst 5% fell further.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    confidence : float, default 0.95
        Confidence level, strictly between 0 and 1.

    Returns
    -------
    float or pd.Series
        VaR as a return (negative for a loss); one value per column for a
        DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, or
        *confidence* is not a real number strictly between 0 and 1.
    """
    tail = 1.0 - require_real("confidence", confidence, above=0.0, below=1.0)
    return reduce_per_series(returns, lambda series: float(series.quantile(tail)))


@overload
def conditional_value_at_risk(returns: pd.Series, confidence: float = ...) -> float: ...
@overload
def conditional_value_at_risk(returns: pd.DataFrame, confidence: float = ...) -> pd.Series: ...
def conditional_value_at_risk(
    returns: pd.Series | pd.DataFrame, confidence: float = 0.95
) -> float | pd.Series:
    """
    Compute the conditional value at risk (expected shortfall).

    The mean of the returns at or below the value at risk: where VaR says how
    bad the threshold is, CVaR says how bad it gets beyond it.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    confidence : float, default 0.95
        Confidence level, strictly between 0 and 1.

    Returns
    -------
    float or pd.Series
        CVaR as a return, never above the VaR; one value per column for a
        DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, or
        *confidence* is not a real number strictly between 0 and 1.
    """
    tail = 1.0 - require_real("confidence", confidence, above=0.0, below=1.0)

    def statistic(series: pd.Series) -> float:
        threshold = series.quantile(tail)
        # Never empty: the minimum is always at or below any quantile.
        return float(series[series <= threshold].mean())

    return reduce_per_series(returns, statistic)


# --- Helpers ---


def _has_shape(series: pd.Series, minimum_observations: int) -> bool:
    """Whether skewness or kurtosis is defined: enough values, not constant (0/0)."""
    return len(series) >= minimum_observations and not is_constant(series)


def _require_probability(name: str, value: object) -> float:
    """Require a real number in the closed interval [0, 1]."""
    probability = require_real(name, value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {probability}.")
    return probability
