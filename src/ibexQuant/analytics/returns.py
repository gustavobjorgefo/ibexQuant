# ibexQuant\src\ibexQuant\analytics\returns.py

"""
Return transformations: series derived from periodic simple returns.

Provides:

- :func:`equity_curve`, :func:`cumulative_returns` and
  :func:`returns_from_equity` — compounding, ``∏(1 + r)``, and its inverse.
- :func:`drawdown` and :func:`drawdown_periods` — distance from the running
  peak, and the table of drawdown episodes.
- :func:`period_returns` and :func:`monthly_returns_table` — returns
  compounded by month, quarter or year.
- :func:`excess_returns` — returns net of a risk-free rate.

Contract
--------
Inputs are periodic simple returns (``0.01`` is 1%), as a ``pd.Series`` or a
``pd.DataFrame`` with one column per series, indexed by a strictly increasing
``DatetimeIndex``. Missing values are allowed only at the edges of each
column (see :mod:`ibexQuant.analytics._validation`). Every function returns
the shape it received, unless documented as Series-only.

Edge ``NaN`` is neutral: before a series starts and after it ends, it neither
earns nor loses, and its derived values are ``NaN``.

What does not belong here
-------------------------
Scalar summaries (CAGR, volatility, Sharpe, maximum drawdown), in
:mod:`ibexQuant.analytics.metrics`; comparisons against a benchmark, in
:mod:`ibexQuant.analytics.relative`; producing returns from positions, which
is the backtest's job.
"""

from __future__ import annotations

from typing import Final, Literal, cast, get_args

import pandas as pd

from ibexQuant._validation import require_integer, require_real
from ibexQuant.analytics._types import ReturnsT
from ibexQuant.analytics._validation import (
    MINIMUM_RETURN,
    require_equity,
    require_returns,
    require_series,
)

Period = Literal["month", "quarter", "year"]

PERIODS: Final[tuple[str, ...]] = get_args(Period)
# pandas offset aliases anchored at the period end ("ME" is month end).
PERIOD_RULES: Final[dict[str, str]] = {"month": "ME", "quarter": "QE", "year": "YE"}
MONTHS: Final[tuple[int, ...]] = tuple(range(1, 13))
YEAR_TOTAL_COLUMN: Final[str] = "total"


# --- Compounding ---


def equity_curve(returns: ReturnsT, *, initial_value: float = 1.0) -> ReturnsT:
    """
    Compound returns into an equity curve, ``V_0 · ∏(1 + r)``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    initial_value : float, default 1.0
        Value before the first return. Must be positive.

    Returns
    -------
    pd.Series or pd.DataFrame
        Equity after each period, same shape as *returns*. The first value is
        ``initial_value · (1 + r_1)``: the initial value itself has no
        timestamp, since it precedes the first return.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract, or *initial_value* is not
        a positive finite number.
    """
    returns = require_returns(returns)
    initial_value = require_real("initial_value", initial_value, above=0.0)
    return initial_value * (1.0 + returns).cumprod()


def cumulative_returns(returns: ReturnsT) -> ReturnsT:
    """
    Compound returns into the cumulative return, ``∏(1 + r) - 1``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    pd.Series or pd.DataFrame
        Return accumulated up to each period, same shape as *returns*.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract.
    """
    returns = require_returns(returns)
    return (1.0 + returns).cumprod() - 1.0


def returns_from_equity(equity: ReturnsT) -> ReturnsT:
    """
    Recover periodic simple returns from an equity curve, ``V_t / V_{t-1} - 1``.

    The inverse of :func:`equity_curve`, for sources that report value (a
    backtest's or a live account's equity) rather than returns.

    Parameters
    ----------
    equity : pd.Series or pd.DataFrame
        Equity values. Every observed value must be strictly positive.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *equity*. ``NaN`` at each column's first observation,
        which has no previous value to compare with.

    Raises
    ------
    TypeError, ValueError
        If *equity* breaks the analytics contract or has a non-positive value.
    """
    return require_equity(equity).pct_change()


# --- Drawdowns ---


def drawdown(returns: ReturnsT) -> ReturnsT:
    """
    Compute the drawdown, ``V_t / max(V_0, ..., V_t) - 1``.

    The running peak starts at the initial capital, not at the first equity
    value: a loss in the first period is already a drawdown.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same shape as *returns*. Zero at a new peak, negative below it, and
        ``-1.0`` after a total loss.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract.
    """
    equity = (1.0 + require_returns(returns)).cumprod()
    # clip(lower=1.0) places the initial capital at the start of every column.
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1.0


def drawdown_periods(returns: pd.Series) -> pd.DataFrame:
    """
    List every drawdown episode of a single return series.

    An episode starts when the equity falls below its running peak and ends
    on the first period it closes back at or above that peak.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns of one series.

    Returns
    -------
    pd.DataFrame
        One row per episode, in chronological order, with columns:

        - ``peak`` — last timestamp at the peak before the episode. For an
          episode opening the series, the peak is the initial capital, which
          has no timestamp; the first observation is used.
        - ``trough`` — timestamp of the deepest point.
        - ``recovery`` — first timestamp back at the peak; ``NaT`` if the
          episode is still open at the end of the series.
        - ``depth`` — drawdown at the trough (negative).
        - ``peak_to_trough`` — periods from peak to trough.
        - ``trough_to_recovery`` — periods from trough to recovery; ``<NA>``
          if still open.
        - ``duration`` — periods from peak to recovery or, if still open, to
          the end of the series.

        Empty, with these columns, when the series never draws down.

    Raises
    ------
    TypeError
        If *returns* is not a Series.
    TypeError, ValueError
        If *returns* breaks the analytics contract.
    """
    returns = require_returns(require_series(returns, name="returns"))
    # Validation guarantees NaN only at the edges, so this just trims them.
    depth = drawdown(returns.dropna())
    index = cast(pd.DatetimeIndex, depth.index)
    last = len(depth) - 1

    underwater = depth < 0.0
    # Consecutive underwater periods share an id; each new episode increments it.
    episode = (underwater != underwater.shift(fill_value=False)).cumsum()

    peaks: list[pd.Timestamp] = []
    troughs: list[pd.Timestamp] = []
    recoveries: list[pd.Timestamp | None] = []
    depths: list[float] = []
    peak_to_trough: list[int] = []
    trough_to_recovery: list[int | None] = []
    durations: list[int] = []

    for _, segment in depth[underwater].groupby(episode[underwater]):
        # get_loc returns an int for a unique index, which validation guarantees.
        start = cast(int, index.get_loc(segment.index[0]))
        trough = cast(int, index.get_loc(segment.idxmin()))
        after = cast(int, index.get_loc(segment.index[-1])) + 1
        recovered = after <= last
        peak = max(start - 1, 0)
        end = after if recovered else last

        peaks.append(index[peak])
        troughs.append(index[trough])
        recoveries.append(index[after] if recovered else None)
        depths.append(float(segment.min()))
        peak_to_trough.append(trough - peak)
        trough_to_recovery.append(after - trough if recovered else None)
        durations.append(end - peak)

    return pd.DataFrame(
        {
            "peak": pd.DatetimeIndex(peaks, dtype=index.dtype),
            "trough": pd.DatetimeIndex(troughs, dtype=index.dtype),
            "recovery": pd.DatetimeIndex(recoveries, dtype=index.dtype),
            "depth": pd.Series(depths, dtype="float64"),
            "peak_to_trough": pd.array(peak_to_trough, dtype="Int64"),
            "trough_to_recovery": pd.array(trough_to_recovery, dtype="Int64"),
            "duration": pd.array(durations, dtype="Int64"),
        }
    )


# --- Aggregation ---


def period_returns(returns: ReturnsT, period: Period) -> ReturnsT:
    """
    Compound returns within each calendar period, ``∏(1 + r) - 1``.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    period : {"month", "quarter", "year"}
        Calendar period to compound over.

    Returns
    -------
    pd.Series or pd.DataFrame
        One row per calendar period from the first to the last timestamp,
        labeled by the period's calendar end (e.g. 2024-01-31 for January,
        even if the last session was the 30th). A column with no observation
        in a period is ``NaN`` there, never 0%. The first and last periods
        are compounded over the sessions available, so they may be partial.

    Raises
    ------
    TypeError, ValueError
        If *returns* breaks the analytics contract.
    ValueError
        If *period* is not one of the accepted values.
    """
    returns = require_returns(returns)
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}.")

    # min_count=1: a period with no observation is NaN, not an empty product
    # of 1.0 that would read as a 0% return.
    return (1.0 + returns).resample(PERIOD_RULES[period]).prod(min_count=1) - 1.0


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Arrange the monthly returns of a single series as a year x month table.

    Parameters
    ----------
    returns : pd.Series
        Periodic simple returns of one series.

    Returns
    -------
    pd.DataFrame
        One row per year (index named ``"year"``), columns ``1`` to ``12`` with
        that month's compounded return, and ``"total"`` with the year's
        compounded return. Months without data are ``NaN``.

    Raises
    ------
    TypeError
        If *returns* is not a Series.
    TypeError, ValueError
        If *returns* breaks the analytics contract.
    """
    returns = require_returns(require_series(returns, name="returns"))
    monthly = period_returns(returns, "month")
    yearly = period_returns(returns, "year")

    month_end = cast(pd.DatetimeIndex, monthly.index)
    table = (
        pd.DataFrame(
            {"year": month_end.year, "month": month_end.month, "value": monthly.to_numpy()}
        )
        .pivot(index="year", columns="month", values="value")
        .reindex(index=cast(pd.DatetimeIndex, yearly.index).year, columns=list(MONTHS))
    )
    table[YEAR_TOTAL_COLUMN] = yearly.to_numpy()
    return table.rename_axis(index="year", columns=None)


# --- Risk-free ---


def excess_returns(
    returns: ReturnsT,
    risk_free: float | pd.Series = 0.0,
    *,
    periods_per_year: int,
) -> ReturnsT:
    """
    Subtract the risk-free return from each period, ``r_t - rf_t``.

    The type of *risk_free* decides how it is read, so an annual rate is
    never mistaken for a periodic one:

    - ``float`` — an **annual** rate, converted to the period geometrically,
      ``(1 + rf)^(1 / periods_per_year) - 1``, so that compounding it over a
      year gives back *rf*.
    - ``pd.Series`` — **periodic** returns already (e.g. the daily CDI),
      subtracted as given. It must cover every timestamp where *returns* is
      observed.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Periodic simple returns.
    risk_free : float or pd.Series, default 0.0
        Annual rate (float) or periodic returns (Series).
    periods_per_year : int
        Periods in one year (252 for B3 daily sessions). Required, so the
        frequency is never assumed; unused when *risk_free* is a Series.

    Returns
    -------
    pd.Series or pd.DataFrame
        Excess returns, same shape as *returns*.

    Raises
    ------
    TypeError, ValueError
        If *returns* or a Series *risk_free* breaks the analytics contract, a
        float *risk_free* is not a finite rate above -100%, or
        *periods_per_year* is not a positive integer.
    ValueError
        If a Series *risk_free* is missing a timestamp where *returns* is
        observed.
    """
    returns = require_returns(returns)
    periods_per_year = require_integer("periods_per_year", periods_per_year, minimum=1)

    if isinstance(risk_free, pd.Series):
        periodic = _aligned_risk_free(require_returns(risk_free, name="risk_free"), returns)
        # axis=0 aligns on the index, subtracting the same rate from every column.
        excess: ReturnsT = returns.sub(periodic, axis=0)
        return excess

    annual = require_real("risk_free", risk_free, above=MINIMUM_RETURN)
    periodic_rate = (1.0 + annual) ** (1.0 / periods_per_year) - 1.0
    shifted: ReturnsT = returns - periodic_rate
    return shifted


# --- Helpers ---


def _aligned_risk_free(risk_free: pd.Series, returns: pd.Series | pd.DataFrame) -> pd.Series:
    """Periodic risk-free returns on the returns index; reject uncovered timestamps."""
    aligned = risk_free.reindex(returns.index)
    observed = returns.notna() if isinstance(returns, pd.Series) else returns.notna().any(axis=1)
    uncovered = observed & aligned.isna()
    if uncovered.any():
        raise ValueError(
            f"risk_free must cover every timestamp where returns is observed; "
            f"missing at {uncovered.idxmax()} and {int(uncovered.sum()) - 1} other(s)."
        )
    return aligned
