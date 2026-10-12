# ibexQuant\src\ibexQuant\analytics\rolling.py

"""
Rolling analytics: how performance metrics evolved over time.

Provides trailing-window versions of metrics from
:mod:`ibexQuant.analytics.metrics` and :mod:`ibexQuant.analytics.relative`:

- :func:`rolling_return` — compounded return of each window.
- :func:`rolling_volatility`, :func:`rolling_sharpe` and
  :func:`rolling_sortino` — risk and risk-adjusted return.
- :func:`rolling_beta` and :func:`rolling_correlation` — market exposure
  against a benchmark.

Contract
--------
Inputs follow the analytics contract (ADR 0009). The output has the shape
and index of ``returns``; the value at *t* uses the window of ``window`` bars
ending at *t*, inclusive, so it depends only on the past. The window is a
number of bars, independent of the timeframe.

Conventions
-----------
- A value exists only once its window is full: the first ``window - 1`` bars
  of each column are ``NaN``, counted from that column's first observation.
- Each window follows the convention of its full-sample metric: annualized
  with an explicit ``periods_per_year``, ``risk_free`` as in
  :func:`~ibexQuant.analytics.returns.excess_returns`, and a benchmark on
  the exact index of the returns.
- pandas updates rolling moments with running sums, which leave residual
  dispersion (~1e-15) in windows of identical values. Such windows are
  detected exactly (rolling max equals rolling min): volatility is ``0.0``
  and every ratio over zero dispersion is ``NaN`` (ADR 0009, AN4).
- For the same reason a window's value matches the full-sample metric on
  that window within numerical tolerance, not bit for bit.

What does not belong here
-------------------------
Rolling quantities used as trading inputs (e.g. trading only while a
strategy's rolling Sharpe exceeds a threshold): they must update bar by bar
in live trading and are feature nodes, built on :mod:`ibexQuant.features`.
These functions describe a strategy ex post and never feed a decision.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from ibexQuant._validation import require_integer
from ibexQuant.analytics._types import ReturnsT
from ibexQuant.analytics._validation import (
    require_benchmark,
    require_observations,
    require_returns,
)
from ibexQuant.analytics.returns import excess_returns

type PairRolling = Callable[[pd.Series, pd.Series, int], pd.Series]


# --- Return ---


def rolling_return(returns: ReturnsT, window: int) -> ReturnsT:
    """
    Compute the compounded return of each trailing window, ``∏(1 + r) - 1``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    window : int
        Bars per window, at least 1.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*; not annualized.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, or *window*
        is not an integer of at least 1.
    """
    window = require_integer("window", window, minimum=1)
    returns = require_observations(require_returns(returns))
    # Each window is compounded exactly: a running product would not survive a
    # -100% return, and a sum of logarithms breaks on log(0).
    compounded: ReturnsT = returns.rolling(window, min_periods=window).apply(_compound, raw=True)
    return compounded


# --- Risk ---


def rolling_volatility(returns: ReturnsT, window: int, *, periods_per_year: int) -> ReturnsT:
    """
    Compute the annualized volatility of each trailing window, ``std(r) · √N``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    window : int
        Bars per window, at least 2.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*. Exactly ``0.0`` in windows of identical
        returns.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract or is empty, *window* is
        not an integer of at least 2, or *periods_per_year* is not a positive
        integer.
    """
    window = _require_dispersion_window(window)
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    returns = require_observations(require_returns(returns))
    volatility: ReturnsT = _rolling_std(returns, window) * math.sqrt(periods_per_year)
    return volatility


# --- Risk-adjusted ---


def rolling_sharpe(
    returns: ReturnsT,
    window: int,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> ReturnsT:
    """
    Compute the annualized Sharpe ratio of each trailing window.

    The convention of :func:`~ibexQuant.analytics.metrics.sharpe_ratio`:
    ``mean(r - rf) / std(r - rf) · √N``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    window : int
        Bars per window, at least 2.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series).
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*. ``NaN`` in windows of zero dispersion.

    Raises
    ------
    TypeError, ValueError
        If *returns* or *risk_free* break the analytics contract, *returns*
        is empty, *window* is not an integer of at least 2, or
        *periods_per_year* is not a positive integer.
    """
    window = _require_dispersion_window(window)
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    excess = excess_returns(
        require_observations(require_returns(returns)),
        risk_free,
        periods_per_year=periods_per_year,
    )
    mean = excess.rolling(window, min_periods=window).mean()
    deviation = _rolling_std(excess, window)
    sharpe: ReturnsT = mean / deviation.where(deviation != 0.0) * math.sqrt(periods_per_year)
    return sharpe


def rolling_sortino(
    returns: ReturnsT,
    window: int,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> ReturnsT:
    """
    Compute the annualized Sortino ratio of each trailing window.

    The convention of :func:`~ibexQuant.analytics.metrics.sortino_ratio`: the
    mean excess return over the downside deviation, averaged over all bars of
    the window.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    window : int
        Bars per window, at least 2.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series), also the target
        below which a return counts as downside.
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions).

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*. ``NaN`` in windows with no bar below the
        risk-free rate.

    Raises
    ------
    TypeError, ValueError
        Under the same conditions as :func:`rolling_sharpe`.
    """
    window = _require_dispersion_window(window)
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    excess = excess_returns(
        require_observations(require_returns(returns)),
        risk_free,
        periods_per_year=periods_per_year,
    )
    mean = excess.rolling(window, min_periods=window).mean()
    downside = ((excess.clip(upper=0.0) ** 2).rolling(window, min_periods=window).mean()) ** 0.5
    # Counting bars below target is exact (sums of 0 and 1), unlike the running
    # sum of squares, which may keep a residual after the last loss leaves.
    below_target = (excess < 0.0).astype(float).where(excess.notna())
    has_downside = below_target.rolling(window, min_periods=window).sum() > 0.0
    sortino: ReturnsT = mean / downside.where(has_downside) * math.sqrt(periods_per_year)
    return sortino


# --- Against a benchmark ---


def rolling_beta(
    returns: ReturnsT,
    benchmark: pd.Series,
    window: int,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> ReturnsT:
    """
    Compute the CAPM beta of each trailing window.

    The convention of :func:`~ibexQuant.analytics.relative.beta`:
    ``cov(r - rf, b - rf) / var(b - rf)``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    window : int
        Bars per window, at least 2.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series).
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions), used to convert an
        annual *risk_free*.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*; a window needs ``window`` bars where both
        the strategy and the benchmark are observed. ``NaN`` where the
        benchmark is constant over the window.

    Raises
    ------
    TypeError, ValueError
        If an input breaks the analytics contract, *returns* is empty, the
        indexes differ, a strategy never overlaps the benchmark, *window* is
        not an integer of at least 2, or *periods_per_year* is not a positive
        integer.
    """
    window = _require_dispersion_window(window)
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)
    market = require_benchmark(returns, benchmark)
    excess_strategy = excess_returns(
        require_observations(returns), risk_free, periods_per_year=periods_per_year
    )
    excess_market = excess_returns(market, risk_free, periods_per_year=periods_per_year)

    def beta_of(strategy: pd.Series, market: pd.Series, window: int) -> pd.Series:
        covariance = strategy.rolling(window, min_periods=window).cov(market)
        variance = _rolling_std(market, window) ** 2
        return covariance / variance.where(variance != 0.0)

    return _against_benchmark(excess_strategy, excess_market, window, beta_of)


def rolling_correlation(returns: ReturnsT, benchmark: pd.Series, window: int) -> ReturnsT:
    """
    Compute the Pearson correlation with the benchmark over each window.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns of one or more strategies.
    benchmark : pd.Series
        Periodic simple returns of the benchmark, on the same index.
    window : int
        Bars per window, at least 2.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*. ``NaN`` where either side is constant over
        the window.

    Raises
    ------
    TypeError, ValueError
        If either input breaks the analytics contract, *returns* is empty,
        the indexes differ, a strategy never overlaps the benchmark, or
        *window* is not an integer of at least 2.
    """
    window = _require_dispersion_window(window)
    market = require_benchmark(returns, benchmark)

    def correlation_of(strategy: pd.Series, market: pd.Series, window: int) -> pd.Series:
        correlation = strategy.rolling(window, min_periods=window).corr(market)
        dispersed = (_rolling_std(strategy, window) != 0.0) & (_rolling_std(market, window) != 0.0)
        return correlation.where(dispersed)

    return _against_benchmark(require_observations(returns), market, window, correlation_of)


# --- Helpers ---


def _require_dispersion_window(window: object) -> int:
    """A window for dispersion metrics: a sample deviation needs two bars."""
    return require_integer("window", window, minimum=2)


def _compound(window_returns: np.ndarray) -> float:
    """Compounded return of one window."""
    return float(np.prod(1.0 + window_returns)) - 1.0


def _rolling_std(values: ReturnsT, window: int) -> ReturnsT:
    """Rolling sample deviation (``ddof=1``), exactly zero for constant windows."""
    rolling = values.rolling(window, min_periods=window)
    deviation = rolling.std(ddof=1)
    # Max and min are exact (no running sums), so equality detects a constant
    # window reliably; NaN comparisons are False, leaving warm-up untouched.
    constant = rolling.max() == rolling.min()
    exact: ReturnsT = deviation.mask(constant, 0.0)
    return exact


def _against_benchmark(
    returns: ReturnsT, benchmark: pd.Series, window: int, compute: PairRolling
) -> ReturnsT:
    """Apply *compute* to each strategy paired with the benchmark."""

    def paired(strategy: pd.Series) -> pd.Series:
        both = strategy.notna() & benchmark.notna()
        if not both.any():
            label = "returns" if strategy.name is None else strategy.name
            raise ValueError(f"{label!r} has no period in common with the benchmark.")
        # Masking both sides to the common rows keeps every window counting
        # the same bars for strategy and benchmark.
        result = compute(strategy.where(both), benchmark.where(both), window)
        result.name = strategy.name
        return result

    if isinstance(returns, pd.Series):
        return paired(returns)
    result: ReturnsT = pd.DataFrame(
        {column: paired(values) for column, values in returns.items()},
        index=returns.index,
        columns=returns.columns,
    )
    return result
