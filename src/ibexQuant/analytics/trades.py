# ibexQuant\src\ibexQuant\analytics\trades.py

"""
Trade analytics: metrics over individual round-trip trades.

A trade is one round trip: opened, closed, one result. The unit is the same
whatever the holding period, so a day trade and a five-day swing trade are
measured alike, and the difference between styles shows up as a metric
(:func:`average_holding_period`) rather than as a branch in the code.

Provides two groups, split by the data each one needs:

- **Outcome metrics** take ``trade_results``, a ``pd.Series`` with one result
  per trade in chronological order of exit. The results may be returns or
  PnL; every metric reports in the unit it receives (expectancy in % for
  returns, in currency for PnL).

  - :func:`trade_count`, :func:`win_rate` and :func:`loss_rate`;
  - :func:`expectancy`, :func:`trade_standard_deviation` and
    :func:`expectancy_t_stat`;
  - :func:`average_win`, :func:`average_loss` and :func:`payoff_ratio`;
  - :func:`gross_profit`, :func:`gross_loss` and :func:`profit_factor`;
  - :func:`best_trade` and :func:`worst_trade`;
  - :func:`max_consecutive_wins` and :func:`max_consecutive_losses`.

- **Timing metrics** take a ``trades`` table with ``entry_time`` and
  ``exit_time`` columns, and the ``sessions`` of the period (e.g. the index
  of the daily return series):

  - :func:`trade_frequency`, :func:`exposure` and
    :func:`average_holding_period`.

Conventions
-----------
- A result of exactly zero is neither a win nor a loss: it is left out of
  the win and loss rates and averages, and it breaks a winning or losing
  streak.
- Losses are negative numbers, as everywhere in ibexQuant
  (:func:`average_loss`, :func:`gross_loss`, :func:`worst_trade`); ratios
  divide by their absolute value and are positive.
- Holding periods count sessions inclusively: a trade opened and closed in
  the same session lasts one session.
- A period without trades is a legitimate outcome, not a bug: counts and
  sums are zero, and metrics that need at least one trade are ``NaN``. This
  differs from return analytics, where empty input raises, because a
  strategy that never traded in a window still produced a result.
- Undefined ratios (no losses for the profit factor, a single trade for the
  t-statistic) are ``NaN``, never ``inf``.

What does not belong here
-------------------------
Metrics over the daily returns of the strategy (Sharpe, volatility,
drawdown), which measure risk on every session, inside trades too, and live
in :mod:`ibexQuant.analytics.metrics`. Producing the trades table, which is
the backtest's job.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ibexQuant._validation import require_integer
from ibexQuant.analytics._reduce import safe_ratio, sample_standard_deviation
from ibexQuant.analytics._validation import (
    require_sessions,
    require_trade_results,
    require_trades,
)

# --- Counts and rates ---


def trade_count(trade_results: pd.Series) -> int:
    """
    Count the trades.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    int
        Number of trades; ``0`` for a period without trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    return len(require_trade_results(trade_results))


def win_rate(trade_results: pd.Series) -> float:
    """
    Compute the fraction of trades with a positive result.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Fraction in ``[0, 1]``. Zero results count as neither win nor loss,
        so ``win_rate + loss_rate`` can be below 1. ``NaN`` without trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    return float((results > 0.0).mean()) if len(results) else math.nan


def loss_rate(trade_results: pd.Series) -> float:
    """
    Compute the fraction of trades with a negative result.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Fraction in ``[0, 1]``; zero results are excluded. ``NaN`` without
        trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    return float((results < 0.0).mean()) if len(results) else math.nan


# --- Expectancy ---


def expectancy(trade_results: pd.Series) -> float:
    """
    Compute the mean result per trade.

    Equivalently ``win_rate · average_win + loss_rate · average_loss``: a
    low win rate is fine when the average win is large enough.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Mean result, in the unit of *trade_results*. ``NaN`` without trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    return float(results.mean()) if len(results) else math.nan


def trade_standard_deviation(trade_results: pd.Series) -> float:
    """
    Compute the sample standard deviation of the trade results (``ddof=1``).

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Standard deviation, in the unit of *trade_results*. Exactly ``0.0``
        for identical results, ``NaN`` for fewer than two trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    return sample_standard_deviation(require_trade_results(trade_results))


def expectancy_t_stat(trade_results: pd.Series) -> float:
    """
    Compute the t-statistic of the expectancy, ``√n · mean / std``.

    How many standard errors the mean result lies from zero: the same
    expectancy is more convincing over 500 trades than over 20. Known in
    trading as the System Quality Number (SQN), here without a cap on ``n``.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Dimensionless t-statistic. ``NaN`` for fewer than two trades or
        identical results.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    if len(results) < 2:
        return math.nan
    deviation = sample_standard_deviation(results)
    return safe_ratio(float(results.mean()), deviation) * math.sqrt(len(results))


# --- Wins and losses ---


def average_win(trade_results: pd.Series) -> float:
    """
    Compute the mean result of the winning trades.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Mean of the positive results. ``NaN`` without winning trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    wins = _wins(require_trade_results(trade_results))
    return float(wins.mean()) if len(wins) else math.nan


def average_loss(trade_results: pd.Series) -> float:
    """
    Compute the mean result of the losing trades.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Mean of the negative results, as a negative number. ``NaN`` without
        losing trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    losses = _losses(require_trade_results(trade_results))
    return float(losses.mean()) if len(losses) else math.nan


def payoff_ratio(trade_results: pd.Series) -> float:
    """
    Compute the payoff ratio, ``average_win / |average_loss|``.

    How much the average win pays for the average loss: 2.0 means a typical
    win is worth two typical losses.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Positive ratio. ``NaN`` without winning or without losing trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    return safe_ratio(average_win(results), abs(average_loss(results)))


def gross_profit(trade_results: pd.Series) -> float:
    """
    Compute the sum of the winning results.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade. Summing is exact for PnL; for
        returns it is a sum, not a compounded total, so it is meaningful
        mainly when position size is constant.

    Returns
    -------
    float
        Sum of the positive results; ``0.0`` without winning trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    return float(_wins(require_trade_results(trade_results)).sum())


def gross_loss(trade_results: pd.Series) -> float:
    """
    Compute the sum of the losing results.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade; see :func:`gross_profit` on
        summing returns.

    Returns
    -------
    float
        Sum of the negative results, as a non-positive number; ``0.0``
        without losing trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    return float(_losses(require_trade_results(trade_results)).sum())


def profit_factor(trade_results: pd.Series) -> float:
    """
    Compute the profit factor, ``gross_profit / |gross_loss|``.

    Above 1 the winners outweigh the losers. Two strategies with the same net
    result can differ widely here: +30% / -5% gives 6.0, +80% / -55% gives
    1.45.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade; see :func:`gross_profit` on
        summing returns.

    Returns
    -------
    float
        Positive ratio. ``NaN`` without losing trades, where it is undefined.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    return safe_ratio(gross_profit(results), abs(gross_loss(results)))


def best_trade(trade_results: pd.Series) -> float:
    """
    Compute the largest single-trade result.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Maximum result. ``NaN`` without trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    return float(results.max()) if len(results) else math.nan


def worst_trade(trade_results: pd.Series) -> float:
    """
    Compute the smallest single-trade result.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade.

    Returns
    -------
    float
        Minimum result, negative for a loss. ``NaN`` without trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    results = require_trade_results(trade_results)
    return float(results.min()) if len(results) else math.nan


# --- Streaks ---


def max_consecutive_wins(trade_results: pd.Series) -> int:
    """
    Compute the longest run of consecutive winning trades.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade, in chronological order.

    Returns
    -------
    int
        Length of the longest run; a zero result ends a run. ``0`` without
        winning trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    return _longest_run(require_trade_results(trade_results) > 0.0)


def max_consecutive_losses(trade_results: pd.Series) -> int:
    """
    Compute the longest run of consecutive losing trades.

    The worst sequence of stops the strategy has needed to sit through, in
    capital and in patience.

    Parameters
    ----------
    trade_results : pd.Series
        One result (return or PnL) per trade, in chronological order.

    Returns
    -------
    int
        Length of the longest run; a zero result ends a run. ``0`` without
        losing trades.

    Raises
    ------
    TypeError, ValueError
        If *trade_results* is not a numeric Series of finite values.
    """
    return _longest_run(require_trade_results(trade_results) < 0.0)


# --- Timing ---


def trade_frequency(
    trades: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    annualized: bool = False,
    periods_per_year: int | None = None,
) -> float:
    """
    Compute how often the strategy trades, ``n_trades / n_sessions``.

    For a day trade strategy, 5 trades in a 21-session month give 5/21.

    Parameters
    ----------
    trades : pd.DataFrame
        One row per trade, with ``entry_time`` and ``exit_time``.
    sessions : pd.DatetimeIndex
        Every session of the period, traded or not, one timestamp per day.
    annualized : bool, default False
        If True, report trades per year instead of per session.
    periods_per_year : int, optional
        Sessions in one year (252 for B3). Required when *annualized* is
        True, rejected otherwise.

    Returns
    -------
    float
        Trades per session, or per year when *annualized*. ``0.0`` without
        trades.

    Raises
    ------
    TypeError, ValueError
        If *trades* or *sessions* break their contract, a trade falls on a
        date outside *sessions*, or *periods_per_year* does not match
        *annualized*.
    """
    _, session_dates = _session_positions(trades, sessions)
    per_session = len(trades) / len(session_dates)

    if not annualized:
        if periods_per_year is not None:
            raise ValueError("periods_per_year only applies when annualized=True.")
        return per_session
    if periods_per_year is None:
        raise ValueError("periods_per_year is required when annualized=True.")
    return per_session * require_integer("periods_per_year", periods_per_year, minimum=1)


def exposure(trades: pd.DataFrame, sessions: pd.DatetimeIndex) -> float:
    """
    Compute the fraction of sessions with at least one open position.

    A session counts once however many trades are open in it, and a trade
    opened and closed in the same session counts that session.

    Parameters
    ----------
    trades : pd.DataFrame
        One row per trade, with ``entry_time`` and ``exit_time``.
    sessions : pd.DatetimeIndex
        Every session of the period, traded or not.

    Returns
    -------
    float
        Fraction in ``[0, 1]``; ``0.0`` without trades.

    Raises
    ------
    TypeError, ValueError
        If *trades* or *sessions* break their contract, or a trade falls on a
        date outside *sessions*.
    """
    positions, session_dates = _session_positions(trades, sessions)
    in_position = np.zeros(len(session_dates), dtype=bool)
    for entry, exit_ in positions:
        in_position[entry : exit_ + 1] = True
    return float(in_position.mean())


def average_holding_period(trades: pd.DataFrame, sessions: pd.DatetimeIndex) -> float:
    """
    Compute the mean number of sessions a trade stays open, inclusively.

    A day trade lasts 1 session; trades held for 4, 3 and 5 sessions average
    4. This is the metric that tells day trading from swing trading.

    Parameters
    ----------
    trades : pd.DataFrame
        One row per trade, with ``entry_time`` and ``exit_time``.
    sessions : pd.DatetimeIndex
        Every session of the period, traded or not.

    Returns
    -------
    float
        Mean holding period, in sessions. ``NaN`` without trades.

    Raises
    ------
    TypeError, ValueError
        If *trades* or *sessions* break their contract, or a trade falls on a
        date outside *sessions*.
    """
    positions, _ = _session_positions(trades, sessions)
    if not positions:
        return math.nan
    return float(np.mean([exit_ - entry + 1 for entry, exit_ in positions]))


# --- Helpers ---


def _wins(results: pd.Series) -> pd.Series:
    """The strictly positive results."""
    return results[results > 0.0]


def _losses(results: pd.Series) -> pd.Series:
    """The strictly negative results."""
    return results[results < 0.0]


def _longest_run(flags: pd.Series) -> int:
    """Length of the longest run of consecutive True values."""
    if not flags.any():
        return 0
    # Each change of value starts a new run; summing a run of True counts it.
    run_id = (flags != flags.shift()).cumsum()
    return int(flags.groupby(run_id).sum().max())


def _session_positions(
    trades: pd.DataFrame, sessions: pd.DatetimeIndex
) -> tuple[list[tuple[int, int]], pd.DatetimeIndex]:
    """
    Locate each trade's entry and exit sessions.

    Trades are matched to sessions by calendar date, so intraday entry and
    exit times fall on the session of their day.

    Returns
    -------
    list of (int, int)
        Entry and exit positions in *sessions*, one pair per trade.
    pd.DatetimeIndex
        The session dates.
    """
    trades = require_trades(trades)
    session_dates = require_sessions(sessions).normalize()
    if not session_dates.is_unique:
        raise ValueError("sessions must hold one timestamp per day (daily sessions).")

    entries = session_dates.get_indexer(pd.DatetimeIndex(trades["entry_time"]).normalize())
    exits = session_dates.get_indexer(pd.DatetimeIndex(trades["exit_time"]).normalize())
    outside = (entries < 0) | (exits < 0)
    if outside.any():
        first = trades.index[int(np.argmax(outside))]
        raise ValueError(
            f"trades must fall on dates in sessions; trade {first!r} enters or exits outside them."
        )
    return list(zip(entries.tolist(), exits.tolist(), strict=True)), session_dates
