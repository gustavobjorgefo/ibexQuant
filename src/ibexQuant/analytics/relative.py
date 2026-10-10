# ibexQuant\src\ibexQuant\analytics\relative.py

"""
Relative analytics: how a strategy behaved against a benchmark.

Provides:

- :func:`active_returns` — the period-by-period difference ``r - b``.
- :func:`correlation`, :func:`beta` and :func:`alpha` — co-movement, market
  sensitivity, and the return left after the market's share.
- :func:`tracking_error` and :func:`information_ratio` — how far the
  strategy strays from the benchmark, and whether straying paid.
- :func:`up_capture` and :func:`down_capture` — how much of the benchmark's
  rises and falls the strategy takes.
- :func:`excess_cagr` — the annual return above the benchmark's.

Contract
--------
``returns`` follows :mod:`ibexQuant.analytics.metrics`: a Series gives a
``float``, a DataFrame gives one value per column, so several strategies are
compared with the same benchmark in one call. ``benchmark`` is a Series of
periodic simple returns on **exactly the same index** as ``returns``: any
difference in dates or order raises. Edge ``NaN`` remains allowed on both
sides (ADR 0009, AN3), so a strategy and a benchmark that start on different
dates share one index; every metric uses the rows where both are observed.

Conventions
-----------
- Beta and alpha are the CAPM regression of ``r - rf`` on ``b - rf``
  (Jensen's alpha). ``risk_free`` follows
  :func:`~ibexQuant.analytics.returns.excess_returns`.
- Tracking error and information ratio do not take a risk-free rate: it
  cancels out, ``(r - rf) - (b - rf) = r - b``.
- Alpha is annualized arithmetically, ``α · N``, as Sharpe and the
  information ratio annualize the mean; it reads as the intercept of the
  same regression that gives beta. empyrical compounds it instead.
- Capture ratios compound: the annualized compounded return of the strategy
  over the benchmark's up (or down) periods, divided by the benchmark's over
  the same periods (the Morningstar convention). A benchmark return of
  exactly zero is neither up nor down.
- ``excess_cagr`` is the difference of CAGRs over the common period, the
  global market convention, rather than the geometric ratio.

What does not belong here
-------------------------
Rolling versions of these metrics, in :mod:`ibexQuant.analytics.rolling`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import overload

import pandas as pd

from ibexQuant._validation import require_integer
from ibexQuant.analytics._reduce import (
    is_constant,
    reduce_per_series,
    safe_ratio,
    sample_standard_deviation,
)
from ibexQuant.analytics._types import ReturnsT
from ibexQuant.analytics._validation import require_returns, require_series
from ibexQuant.analytics.metrics import cagr
from ibexQuant.analytics.returns import excess_returns

type PairStatistic = Callable[[pd.Series, pd.Series], float]


# --- Active returns ---


def active_returns(returns: ReturnsT, benchmark: pd.Series) -> ReturnsT:
    """
    Compute the active returns, ``r - b`` for each period.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*; ``NaN`` where either side is unobserved.

    Raises
    ------
    TypeError, ValueError
        If either input breaks the analytics contract, the indexes differ, or
        a strategy never overlaps the benchmark.
    """
    benchmark = _require_benchmark(returns, benchmark)
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    for _, column in frame.items():
        _paired(column, benchmark)
    active: ReturnsT = returns.sub(benchmark, axis=0)
    return active


# --- Co-movement ---


@overload
def correlation(returns: pd.Series, benchmark: pd.Series) -> float: ...
@overload
def correlation(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series: ...
def correlation(returns: pd.Series | pd.DataFrame, benchmark: pd.Series) -> float | pd.Series:
    """
    Compute the Pearson correlation with the benchmark.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.

    Returns
    -------
    float or pd.Series
        Correlation in ``[-1, 1]``; one value per column for a DataFrame.
        ``NaN`` with fewer than two common periods or when either side is
        constant.

    Raises
    ------
    TypeError, ValueError
        If either input breaks the analytics contract, *returns* is empty,
        the indexes differ, or a strategy never overlaps the benchmark.
    """

    def statistic(strategy: pd.Series, market: pd.Series) -> float:
        if len(strategy) < 2 or is_constant(strategy) or is_constant(market):
            return math.nan
        return float(strategy.corr(market))

    return _reduce_against(returns, benchmark, statistic)


@overload
def beta(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float | pd.Series = ...,
    *,
    periods_per_year: int,
) -> float: ...
@overload
def beta(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    risk_free: float | pd.Series = ...,
    *,
    periods_per_year: int,
) -> pd.Series: ...
def beta(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> float | pd.Series:
    """
    Compute the CAPM beta, ``cov(r - rf, b - rf) / var(b - rf)``.

    A beta of 1.2 means the strategy tends to move 1.2% when the benchmark
    moves 1%.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series).
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions), used to convert an
        annual *risk_free*.

    Returns
    -------
    float or pd.Series
        Beta; one value per column for a DataFrame. ``NaN`` with fewer than
        two common periods or a constant benchmark.

    Raises
    ------
    TypeError, ValueError
        If an input breaks the analytics contract, *returns* is empty, the
        indexes differ, a strategy never overlaps the benchmark, or
        *periods_per_year* is not a positive integer.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    return _reduce_against(
        returns,
        benchmark,
        lambda strategy, market: _regression(strategy, market, risk_free, periods_per_year)[0],
    )


@overload
def alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float | pd.Series = ...,
    *,
    periods_per_year: int,
) -> float: ...
@overload
def alpha(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    risk_free: float | pd.Series = ...,
    *,
    periods_per_year: int,
) -> pd.Series: ...
def alpha(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> float | pd.Series:
    """
    Compute Jensen's alpha, annualized arithmetically.

    The intercept of the regression of ``r - rf`` on ``b - rf``, multiplied
    by ``N``: the annual return left after removing the part explained by
    the market, ``(mean(r - rf) - β · mean(b - rf)) · N``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series).
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Annualized alpha; one value per column for a DataFrame. ``NaN``
        whenever beta is.

    Raises
    ------
    TypeError, ValueError
        Under the same conditions as :func:`beta`.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    return _reduce_against(
        returns,
        benchmark,
        lambda strategy, market: (
            _regression(strategy, market, risk_free, periods_per_year)[1] * periods_per_year
        ),
    )


# --- Active risk ---


@overload
def tracking_error(
    returns: pd.Series, benchmark: pd.Series, *, periods_per_year: int
) -> float: ...
@overload
def tracking_error(
    returns: pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> pd.Series: ...
def tracking_error(
    returns: pd.Series | pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> float | pd.Series:
    """
    Compute the annualized tracking error, ``std(r - b) · √N``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Tracking error; one value per column for a DataFrame. Exactly
        ``0.0`` when the strategy replicates the benchmark, ``NaN`` with a
        single common period.

    Raises
    ------
    TypeError, ValueError
        If either input breaks the analytics contract, *returns* is empty,
        the indexes differ, a strategy never overlaps the benchmark, or
        *periods_per_year* is not a positive integer.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    scale = math.sqrt(periods_per_year)
    return _reduce_against(
        returns,
        benchmark,
        lambda strategy, market: sample_standard_deviation(strategy - market) * scale,
    )


@overload
def information_ratio(
    returns: pd.Series, benchmark: pd.Series, *, periods_per_year: int
) -> float: ...
@overload
def information_ratio(
    returns: pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> pd.Series: ...
def information_ratio(
    returns: pd.Series | pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> float | pd.Series:
    """
    Compute the information ratio, ``mean(r - b) / std(r - b) · √N``.

    The Sharpe ratio of the active returns: active return per unit of
    tracking error.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Information ratio; one value per column for a DataFrame. ``NaN``
        when the tracking error is zero or undefined.

    Raises
    ------
    TypeError, ValueError
        Under the same conditions as :func:`tracking_error`.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    scale = math.sqrt(periods_per_year)

    def statistic(strategy: pd.Series, market: pd.Series) -> float:
        active = strategy - market
        return safe_ratio(float(active.mean()), sample_standard_deviation(active)) * scale

    return _reduce_against(returns, benchmark, statistic)


# --- Capture ---


@overload
def up_capture(returns: pd.Series, benchmark: pd.Series, *, periods_per_year: int) -> float: ...
@overload
def up_capture(
    returns: pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> pd.Series: ...
def up_capture(
    returns: pd.Series | pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> float | pd.Series:
    """
    Compute the up-market capture ratio.

    The strategy's annualized compounded return over the periods where the
    benchmark rose, divided by the benchmark's over the same periods. Above 1
    the strategy gains more than the market when the market gains.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Capture ratio (``1.0`` is 100%); one value per column for a
        DataFrame. ``NaN`` when the benchmark never rose.

    Raises
    ------
    TypeError, ValueError
        Under the same conditions as :func:`tracking_error`.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    return _reduce_against(
        returns,
        benchmark,
        lambda strategy, market: _capture(strategy, market, market > 0.0, periods_per_year),
    )


@overload
def down_capture(returns: pd.Series, benchmark: pd.Series, *, periods_per_year: int) -> float: ...
@overload
def down_capture(
    returns: pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> pd.Series: ...
def down_capture(
    returns: pd.Series | pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> float | pd.Series:
    """
    Compute the down-market capture ratio.

    The strategy's annualized compounded return over the periods where the
    benchmark fell, divided by the benchmark's over the same periods. Below 1
    the strategy loses less than the market when the market falls; above 1
    it loses more.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Capture ratio (``1.0`` is 100%); one value per column for a
        DataFrame. ``NaN`` when the benchmark never fell.

    Raises
    ------
    TypeError, ValueError
        Under the same conditions as :func:`tracking_error`.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    return _reduce_against(
        returns,
        benchmark,
        lambda strategy, market: _capture(strategy, market, market < 0.0, periods_per_year),
    )


# --- Excess return ---


@overload
def excess_cagr(returns: pd.Series, benchmark: pd.Series, *, periods_per_year: int) -> float: ...
@overload
def excess_cagr(
    returns: pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> pd.Series: ...
def excess_cagr(
    returns: pd.Series | pd.DataFrame, benchmark: pd.Series, *, periods_per_year: int
) -> float | pd.Series:
    """
    Compute the annual return above the benchmark, ``CAGR(r) - CAGR(b)``.

    Both CAGRs are measured over the common period, so the comparison is
    like for like even when the series start on different dates.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    float or pd.Series
        Difference of CAGRs; one value per column for a DataFrame.

    Raises
    ------
    TypeError, ValueError
        Under the same conditions as :func:`tracking_error`.
    """
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    return _reduce_against(
        returns,
        benchmark,
        lambda strategy, market: (
            cagr(strategy, periods_per_year=periods_per_year)
            - cagr(market, periods_per_year=periods_per_year)
        ),
    )


# --- Helpers ---


def _require_benchmark(returns: pd.Series | pd.DataFrame, benchmark: object) -> pd.Series:
    """Validate the benchmark and require the exact index of *returns*."""
    # Branching keeps each call on one concrete type; anything that is not a
    # Series is validated as a DataFrame and rejected there.
    index = (
        require_returns(returns).index
        if isinstance(returns, pd.Series)
        else require_returns(returns).index
    )
    market = require_returns(require_series(benchmark, name="benchmark"), name="benchmark")
    if not market.index.equals(index):
        raise ValueError(
            "benchmark must have exactly the same index as returns (same dates, same order)."
        )
    return market


def _reduce_against(
    returns: pd.Series | pd.DataFrame, benchmark: pd.Series, statistic: PairStatistic
) -> float | pd.Series:
    """Apply *statistic* to each strategy and the benchmark over their common periods."""
    market = _require_benchmark(returns, benchmark)

    def against_benchmark(strategy: pd.Series) -> float:
        return statistic(*_paired(strategy, market))

    return reduce_per_series(returns, against_benchmark)


def _paired(strategy: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The rows where both the strategy and the benchmark are observed."""
    aligned = benchmark.reindex(strategy.index)
    both = strategy.notna() & aligned.notna()
    if not both.any():
        label = "returns" if strategy.name is None else strategy.name
        raise ValueError(f"{label!r} has no period in common with the benchmark.")
    return strategy[both], aligned[both]


def _regression(
    strategy: pd.Series, market: pd.Series, risk_free: float | pd.Series, periods_per_year: int
) -> tuple[float, float]:
    """Beta and the per-period intercept of ``r - rf`` regressed on ``b - rf``."""
    excess_strategy = excess_returns(strategy, risk_free, periods_per_year=periods_per_year)
    excess_market = excess_returns(market, risk_free, periods_per_year=periods_per_year)
    if len(excess_market) < 2 or is_constant(excess_market):
        return math.nan, math.nan

    covariance = excess_strategy.cov(excess_market)
    slope = covariance / sample_standard_deviation(excess_market) ** 2
    intercept = float(excess_strategy.mean()) - slope * float(excess_market.mean())
    return slope, intercept


def _capture(
    strategy: pd.Series, market: pd.Series, regime: pd.Series, periods_per_year: int
) -> float:
    """Ratio of compounded annualized returns over the periods selected by *regime*."""
    if not regime.any():
        return math.nan
    return safe_ratio(
        cagr(strategy[regime], periods_per_year=periods_per_year),
        cagr(market[regime], periods_per_year=periods_per_year),
    )
