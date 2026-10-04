# ibexQuant\src\ibexQuant\analytics\metrics.py

"""
Performance metrics: one number per return series.

Provides:

- :func:`total_return` and :func:`cagr` — how much the series earned.
- :func:`annualized_volatility`, :func:`max_drawdown` and
  :func:`max_drawdown_duration` — how much it fluctuated and fell.
- :func:`sharpe_ratio`, :func:`sortino_ratio` and :func:`calmar_ratio` —
  return per unit of risk.

Contract
--------
Inputs follow :mod:`ibexQuant.analytics.returns`: periodic simple returns,
as a ``pd.Series`` or a ``pd.DataFrame`` (one column per series), with
missing values only at the edges of each column. A Series gives a ``float``;
a DataFrame gives a ``pd.Series`` indexed by its columns. Each column is
measured over its own observed period, so a benchmark that starts later does
not shorten the strategy.

Conventions
-----------
- Annualization uses an explicit ``periods_per_year`` (252 for B3 sessions,
  :data:`B3_SESSIONS_PER_YEAR`); it is never assumed.
- Standard deviations use ``ddof=1`` (sample estimate).
- ``risk_free`` follows :func:`~ibexQuant.analytics.returns.excess_returns`:
  a ``float`` is an annual rate, a ``pd.Series`` holds periodic returns.
- Empty input raises ``ValueError``: asking for a metric of nothing is a bug
  upstream. A metric that is mathematically undefined for a valid input
  (volatility of one observation, a ratio over zero risk) is ``NaN``, never
  ``inf``.
- No minimum length is imposed: annualizing a few sessions is arithmetically
  correct and statistically meaningless, and flagging it is the job of the
  summary or report, not of these functions.

What does not belong here
-------------------------
Per-period distribution statistics (skewness, quantiles, tail losses),
metrics over individual trades, and comparisons against a benchmark, each in
its own analytics module (``distribution``, ``trades`` and ``relative``,
implemented after this one).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, overload

import pandas as pd

from ibexQuant._validation import require_integer
from ibexQuant.analytics._types import ReturnsT
from ibexQuant.analytics._validation import require_returns
from ibexQuant.analytics.returns import (
    cumulative_returns,
    drawdown,
    drawdown_periods,
    excess_returns,
)

B3_SESSIONS_PER_YEAR: Final[int] = 252


# --- Return ---


@overload
def total_return(returns: pd.Series) -> float: ...
@overload
def total_return(returns: pd.DataFrame) -> pd.Series: ...
def total_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the total compounded return, ``∏(1 + r) - 1``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Total return; one value per column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return _per_series(returns, lambda series: float(cumulative_returns(series).iloc[-1]))


@overload
def cagr(returns: pd.Series, *, periods_per_year: int) -> float: ...
@overload
def cagr(returns: pd.DataFrame, *, periods_per_year: int) -> pd.Series: ...
def cagr(returns: pd.Series | pd.DataFrame, *, periods_per_year: int) -> float | pd.Series:
    """
    Compute the compound annual growth rate, ``(1 + total)^(N / n) - 1``.

    ``N`` is *periods_per_year* and ``n`` the number of observed periods, so
    time is measured in sessions, not calendar days.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Annualized compounded return; one value per column for a DataFrame.
        Over less than a year it extrapolates: a few strong sessions give a
        very large CAGR.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, or
        *periods_per_year* is not a positive integer.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)

    def statistic(series: pd.Series) -> float:
        growth = 1.0 + float(cumulative_returns(series).iloc[-1])
        return float(growth ** (periods_per_year / len(series))) - 1.0

    return _per_series(returns, statistic)


# --- Risk ---


@overload
def annualized_volatility(returns: pd.Series, *, periods_per_year: int) -> float: ...
@overload
def annualized_volatility(returns: pd.DataFrame, *, periods_per_year: int) -> pd.Series: ...
def annualized_volatility(
    returns: pd.Series | pd.DataFrame, *, periods_per_year: int
) -> float | pd.Series:
    """
    Compute the annualized volatility, ``std(r) · √N``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Annualized sample standard deviation; one value per column for a
        DataFrame. Exactly ``0.0`` for constant returns, ``NaN`` for a single
        observation.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, or
        *periods_per_year* is not a positive integer.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    scale = math.sqrt(periods_per_year)
    return _per_series(returns, lambda series: _standard_deviation(series) * scale)


@overload
def max_drawdown(returns: pd.Series) -> float: ...
@overload
def max_drawdown(returns: pd.DataFrame) -> pd.Series: ...
def max_drawdown(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the deepest drawdown, ``min(drawdown)``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Largest fall from a running peak, as a non-positive number (``-0.25``
        is a 25% drawdown); ``0.0`` if the series never fell. One value per
        column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """
    return _per_series(returns, lambda series: float(drawdown(series).min()))


@overload
def max_drawdown_duration(returns: pd.Series) -> float: ...
@overload
def max_drawdown_duration(returns: pd.DataFrame) -> pd.Series: ...
def max_drawdown_duration(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Compute the longest drawdown episode, in periods.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    float or pd.Series
        Longest ``duration`` among the episodes of
        :func:`~ibexQuant.analytics.returns.drawdown_periods`, counting an
        episode still open at the end of the series; ``0.0`` if the series
        never fell. One value per column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty.
    """

    def statistic(series: pd.Series) -> float:
        durations = drawdown_periods(series)["duration"]
        return float(durations.max()) if len(durations) else 0.0

    return _per_series(returns, statistic)


# --- Risk-adjusted ---


@overload
def sharpe_ratio(
    returns: pd.Series, risk_free: float | pd.Series = ..., *, periods_per_year: int
) -> float: ...
@overload
def sharpe_ratio(
    returns: pd.DataFrame, risk_free: float | pd.Series = ..., *, periods_per_year: int
) -> pd.Series: ...
def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> float | pd.Series:
    """
    Compute the annualized Sharpe ratio, ``mean(r - rf) / std(r - rf) · √N``.

    The arithmetic mean and the standard deviation are both taken over excess
    returns (Lo, 2002), which matters when the risk-free rate varies.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series); see
        :func:`~ibexQuant.analytics.returns.excess_returns`.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Sharpe ratio; one value per column for a DataFrame. ``NaN`` when the
        excess returns have zero dispersion or a single observation.

    Raises
    ------
    TypeError, ValueError
        If *returns* or *risk_free* break the analytics contract, *returns* is
        empty, or *periods_per_year* is not a positive integer.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    scale = math.sqrt(periods_per_year)

    def statistic(series: pd.Series) -> float:
        excess = excess_returns(series, risk_free, periods_per_year=periods_per_year)
        return _ratio(float(excess.mean()), _standard_deviation(excess)) * scale

    return _per_series(returns, statistic)


@overload
def sortino_ratio(
    returns: pd.Series, risk_free: float | pd.Series = ..., *, periods_per_year: int
) -> float: ...
@overload
def sortino_ratio(
    returns: pd.DataFrame, risk_free: float | pd.Series = ..., *, periods_per_year: int
) -> pd.Series: ...
def sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> float | pd.Series:
    """
    Compute the annualized Sortino ratio, ``mean(r - rf) / DD · √N``.

    ``DD`` is the downside deviation, ``√mean(min(r - rf, 0)²)``, averaged
    over **all** periods: periods above the risk-free rate contribute zero
    (Sortino and Price, 1994). Unlike Sharpe, gains above the target are not
    penalized as risk.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series), also the target
        below which a return counts as downside.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Sortino ratio; one value per column for a DataFrame. ``NaN`` when no
        period falls below the risk-free rate.

    Raises
    ------
    TypeError, ValueError
        If *returns* or *risk_free* break the analytics contract, *returns* is
        empty, or *periods_per_year* is not a positive integer.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    scale = math.sqrt(periods_per_year)

    def statistic(series: pd.Series) -> float:
        excess = excess_returns(series, risk_free, periods_per_year=periods_per_year)
        downside = math.sqrt(float((excess.clip(upper=0.0) ** 2).mean()))
        return _ratio(float(excess.mean()), downside) * scale

    return _per_series(returns, statistic)


@overload
def calmar_ratio(returns: pd.Series, *, periods_per_year: int) -> float: ...
@overload
def calmar_ratio(returns: pd.DataFrame, *, periods_per_year: int) -> pd.Series: ...
def calmar_ratio(returns: pd.Series | pd.DataFrame, *, periods_per_year: int) -> float | pd.Series:
    """
    Compute the Calmar ratio, ``CAGR / |max drawdown|``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Calmar ratio; one value per column for a DataFrame. ``NaN`` when the
        series never fell.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, or
        *periods_per_year* is not a positive integer.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)

    def statistic(series: pd.Series) -> float:
        growth_rate = cagr(series, periods_per_year=periods_per_year)
        return _ratio(growth_rate, abs(max_drawdown(series)))

    return _per_series(returns, statistic)


# --- Helpers ---


def _per_series(
    returns: pd.Series | pd.DataFrame, statistic: Callable[[pd.Series], float]
) -> float | pd.Series:
    """Apply *statistic* to each series over its observed period."""
    # Validation guarantees NaN only at the edges, so dropna just trims them.
    if isinstance(returns, pd.Series):
        series = _require_observations(require_returns(returns))
        return statistic(series.dropna())

    # Anything that is not a Series is validated here, so non-pandas input
    # still gets a TypeError.
    frame = _require_observations(require_returns(returns))
    values = [statistic(column.dropna()) for _, column in frame.items()]
    return pd.Series(values, index=frame.columns, dtype="float64")


def _require_observations(returns: ReturnsT) -> ReturnsT:
    """Reject empty input: a metric of nothing signals a bug upstream."""
    if returns.empty:
        raise ValueError("returns must have at least one observation.")
    return returns


def _standard_deviation(values: pd.Series) -> float:
    """Sample standard deviation, exactly zero for constant values."""
    if len(values) < 2:
        return math.nan
    # The mean of a repeated value is not always that value in floating point
    # (e.g. 0.1), which would leave ~1e-17 of spurious dispersion and turn a
    # ratio over zero risk into a huge number instead of NaN.
    if values.max() == values.min():
        return 0.0
    return float(values.std(ddof=1))


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, NaN when the denominator is zero or NaN."""
    if math.isnan(denominator) or denominator == 0.0:
        return math.nan
    return numerator / denominator
