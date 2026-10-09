# ibexQuant\src\ibexQuant\analytics\_validation.py

"""
Input validation for analytics functions.

Analytics consume the periodic simple returns of a strategy (or the equity
curve those returns produce). Every public function validates its input with
the helpers below, so the contract is enforced in one place:

- a ``pd.Series`` or ``pd.DataFrame`` (one column per series);
- a ``DatetimeIndex``, strictly increasing (sorted, no duplicates);
- numeric, finite values;
- missing values only at the edges of each column.

Edge ``NaN`` lets series of different lengths share one DataFrame (a strategy
and a benchmark starting on different dates). Interior ``NaN`` is rejected: a
strategy is never "unobservable" mid-history, since out of the market it
earns cash, so a hole means a bug upstream, and filling it with zero would
hide the bug.

Trade analytics have their own inputs, validated here too: a series of
per-trade results (:func:`require_trade_results`), a table of trades with
entry and exit times (:func:`require_trades`), and the sessions of the
period they belong to (:func:`require_sessions`).

What does not belong here
-------------------------
Scalar parameter rules (integers, reals), shared across packages in
:mod:`ibexQuant._validation`.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Final

import numpy as np
import pandas as pd

from ibexQuant.analytics._types import ReturnsT

# Below -100% the value of a position would be negative and compounding
# breaks; exactly -100% (total loss) is a legitimate return.
MINIMUM_RETURN: Final[float] = -1.0

# Columns a trades table must have; any other column is ignored.
TRADE_TIME_COLUMNS: Final[tuple[str, str]] = ("entry_time", "exit_time")


def require_returns(returns: ReturnsT, *, name: str = "returns") -> ReturnsT:
    """
    Require periodic simple returns satisfying the analytics contract.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Candidate returns, timestamp x series.
    name : str, default "returns"
        Name used in error messages.

    Returns
    -------
    pd.Series or pd.DataFrame
        *returns*, unchanged.

    Raises
    ------
    TypeError
        If *returns* is not a Series or DataFrame, its index is not a
        ``DatetimeIndex``, or its values are not numeric.
    ValueError
        If the index is not strictly increasing, a value is infinite or below
        -100%, a column has interior ``NaN``, or a column has no observations.
    """
    frame = _require_time_series(returns, name)
    if (frame < MINIMUM_RETURN).any().any():
        raise ValueError(f"{name} must be >= -100% (-1.0); found a value below it.")
    return returns


def require_equity(equity: ReturnsT, *, name: str = "equity") -> ReturnsT:
    """
    Require an equity curve satisfying the analytics contract.

    Parameters
    ----------
    equity : pd.Series or pd.DataFrame
        Candidate equity values, timestamp x series.
    name : str, default "equity"
        Name used in error messages.

    Returns
    -------
    pd.Series or pd.DataFrame
        *equity*, unchanged.

    Raises
    ------
    TypeError
        Under the same conditions as :func:`require_returns`.
    ValueError
        Under the same structural conditions as :func:`require_returns`, or if
        any value is zero or negative (no return can be computed from it).
    """
    frame = _require_time_series(equity, name)
    if (frame <= 0.0).any().any():
        raise ValueError(f"{name} must be strictly positive; found a zero or negative value.")
    return equity


def require_series(values: object, *, name: str) -> pd.Series:
    """
    Require a ``pd.Series``, for functions that only make sense per series.

    Parameters
    ----------
    values : object
        Candidate input.
    name : str
        Name used in error messages.

    Returns
    -------
    pd.Series
        *values*, unchanged.

    Raises
    ------
    TypeError
        If *values* is not a Series (a DataFrame included).
    """
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pd.Series, got {type(values)!r}.")
    return values


# --- Trades ---


def require_trade_results(results: object, *, name: str = "trade_results") -> pd.Series:
    """
    Require per-trade results: one finite number per trade.

    The values may be returns or PnL; the order of the series is the
    chronological order of the trades, and the index is free (exit times,
    a counter, or repeated timestamps for trades closed together).

    Parameters
    ----------
    results : object
        Candidate results.
    name : str, default "trade_results"
        Name used in error messages.

    Returns
    -------
    pd.Series
        *results*, unchanged. May be empty: a period without trades is a
        legitimate outcome.

    Raises
    ------
    TypeError
        If *results* is not a numeric Series.
    ValueError
        If any value is missing or infinite.
    """
    series = require_series(results, name=name)
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{name} must be numeric, got dtype {series.dtype}.")

    values = series.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"{name} must not contain missing values: every trade has a result.")
    if np.isinf(values).any():
        raise ValueError(f"{name} must be finite; found an infinite value.")
    return series


def require_trades(trades: object, *, name: str = "trades") -> pd.DataFrame:
    """
    Require a table of trades with entry and exit times.

    Parameters
    ----------
    trades : object
        Candidate table, one row per trade, with datetime columns
        ``entry_time`` and ``exit_time``. Other columns are ignored.
    name : str, default "trades"
        Name used in error messages.

    Returns
    -------
    pd.DataFrame
        *trades*, unchanged. May be empty.

    Raises
    ------
    TypeError
        If *trades* is not a DataFrame or a time column is not datetime.
    ValueError
        If a time column is missing, has missing values, or a trade exits
        before it enters.
    """
    if not isinstance(trades, pd.DataFrame):
        raise TypeError(f"{name} must be a pd.DataFrame, got {type(trades)!r}.")

    missing = [column for column in TRADE_TIME_COLUMNS if column not in trades.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s) {missing}.")

    for column in TRADE_TIME_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(trades[column]):
            raise TypeError(
                f"{name} column {column!r} must be datetime, got {trades[column].dtype}."
            )
        if trades[column].isna().any():
            raise ValueError(f"{name} column {column!r} must not contain missing values.")

    if (trades["exit_time"] < trades["entry_time"]).any():
        raise ValueError(f"{name} has a trade whose exit_time precedes its entry_time.")
    return trades


def require_sessions(sessions: object, *, name: str = "sessions") -> pd.DatetimeIndex:
    """
    Require the sessions of a period: a non-empty, strictly increasing index.

    Parameters
    ----------
    sessions : object
        Candidate sessions, e.g. the index of a daily return series.
    name : str, default "sessions"
        Name used in error messages.

    Returns
    -------
    pd.DatetimeIndex
        *sessions*, unchanged.

    Raises
    ------
    TypeError
        If *sessions* is not a DatetimeIndex.
    ValueError
        If it is empty or not strictly increasing.
    """
    if not isinstance(sessions, pd.DatetimeIndex):
        raise TypeError(f"{name} must be a pd.DatetimeIndex, got {type(sessions)!r}.")
    if sessions.empty:
        raise ValueError(f"{name} must contain at least one session.")
    if not (sessions.is_monotonic_increasing and sessions.is_unique):
        raise ValueError(f"{name} must be strictly increasing (sorted, no duplicates).")
    return sessions


# --- Structural checks ---


def _require_time_series(values: object, name: str) -> pd.DataFrame:
    """Run the checks shared by returns and equity; return the input as a frame."""
    if not isinstance(values, (pd.Series, pd.DataFrame)):
        raise TypeError(f"{name} must be a pd.Series or pd.DataFrame, got {type(values)!r}.")

    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(values.index).__name__}.")

    if not (values.index.is_monotonic_increasing and values.index.is_unique):
        raise ValueError(f"{name} index must be strictly increasing (sorted, no duplicates).")

    # One code path for both shapes: a Series is checked as a one-column frame.
    frame = (
        values.to_frame(name if values.name is None else values.name)
        if isinstance(values, pd.Series)
        else values
    )

    if not all(
        pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
        for dtype in frame.dtypes
    ):
        raise TypeError(
            f"{name} must be numeric; got dtypes {sorted(set(map(str, frame.dtypes)))}."
        )

    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} must be finite; found an infinite value.")

    for label, column in frame.items():
        _require_edge_only_gaps(column, label, name)
    return frame


def _require_edge_only_gaps(column: pd.Series, label: Hashable, name: str) -> None:
    """Reject a column with interior NaN or with no observation at all."""
    if column.empty:
        return

    first, last = column.first_valid_index(), column.last_valid_index()
    if first is None or last is None:
        raise ValueError(f"{name} column {label!r} has no observations.")

    # Between the first and last observations, every value must be present.
    gaps = column.loc[first:last].isna()
    if gaps.any():
        raise ValueError(
            f"{name} column {label!r} has missing values inside its history, first at "
            f"{gaps.idxmax()}. Only leading and trailing NaN are allowed."
        )
