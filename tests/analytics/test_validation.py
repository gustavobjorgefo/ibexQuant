# ibexQuant\tests\analytics\test_validation.py

"""Tests for :mod:`ibexQuant.analytics._validation`."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.analytics._validation import (
    require_equity,
    require_returns,
    require_series,
    require_sessions,
    require_trades,
)

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=4)


def make_returns(values: list[float], name: str = "strategy") -> pd.Series:
    return pd.Series(values, index=SESSIONS, name=name, dtype="float64")


# --- Accepted inputs ---


def test_require_returns_returns_the_same_object() -> None:
    returns = make_returns([0.01, -0.02, 0.0, 0.03])

    assert require_returns(returns) is returns


def test_require_returns_accepts_dataframes_with_edge_gaps() -> None:
    returns = pd.DataFrame(
        {"strategy": [0.01, 0.02, 0.03, NAN], "benchmark": [NAN, NAN, 0.01, 0.02]},
        index=SESSIONS,
    )

    assert require_returns(returns) is returns


def test_require_returns_accepts_a_total_loss() -> None:
    returns = make_returns([0.01, -1.0, 0.0, 0.0])

    assert require_returns(returns) is returns


def test_require_returns_accepts_integer_dtypes() -> None:
    returns = pd.Series([0, 0, 0, 0], index=SESSIONS, dtype="int64")

    assert require_returns(returns) is returns


def test_require_returns_accepts_empty_input() -> None:
    returns = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")

    assert require_returns(returns) is returns


# --- Structural errors ---


@pytest.mark.parametrize("value", [[0.01, 0.02], np.array([0.01]), 0.01, None])
def test_require_returns_rejects_non_pandas_inputs(value: object) -> None:
    with pytest.raises(TypeError, match=r"pd.Series or pd.DataFrame"):
        require_returns(value)  # type: ignore[type-var]


def test_require_returns_rejects_non_datetime_index() -> None:
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(TypeError, match="DatetimeIndex"):
        require_returns(returns)


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex(["2024-01-02", "2024-01-01"]),
        pd.DatetimeIndex(["2024-01-01", "2024-01-01"]),
    ],
    ids=["unsorted", "duplicated"],
)
def test_require_returns_rejects_non_increasing_index(index: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        require_returns(pd.Series([0.01, 0.02], index=index))


@pytest.mark.parametrize("dtype", ["bool", "object"])
def test_require_returns_rejects_non_numeric_dtypes(dtype: str) -> None:
    returns = pd.Series([True, False, True, False], index=SESSIONS, dtype=dtype)

    with pytest.raises(TypeError, match="must be numeric"):
        require_returns(returns)


# --- Value errors ---


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_require_returns_rejects_infinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        require_returns(make_returns([0.01, value, 0.0, 0.0]))


def test_require_returns_rejects_returns_below_total_loss() -> None:
    with pytest.raises(ValueError, match=r">= -100%"):
        require_returns(make_returns([0.01, -1.5, 0.0, 0.0]))


def test_require_returns_rejects_interior_gaps_naming_column_and_date() -> None:
    returns = pd.DataFrame(
        {"strategy": [0.01, 0.02, 0.03, 0.04], "benchmark": [0.01, NAN, NAN, 0.02]},
        index=SESSIONS,
    )

    with pytest.raises(ValueError, match=r"'benchmark'.*2024-01-02"):
        require_returns(returns)


def test_require_returns_rejects_columns_without_observations() -> None:
    returns = pd.DataFrame(
        {"strategy": [0.01, 0.02, 0.03, 0.04], "benchmark": [NAN] * 4}, index=SESSIONS
    )

    with pytest.raises(ValueError, match="'benchmark' has no observations"):
        require_returns(returns)


def test_unnamed_series_are_reported_by_parameter_name() -> None:
    returns = pd.Series([0.01, NAN, 0.02, 0.03], index=SESSIONS)

    with pytest.raises(ValueError, match="column 'risk_free'"):
        require_returns(returns, name="risk_free")


# --- Equity ---


def test_require_equity_accepts_positive_values() -> None:
    equity = make_returns([100.0, 101.0, 99.0, NAN])

    assert require_equity(equity) is equity


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_require_equity_rejects_non_positive_values(value: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        require_equity(make_returns([100.0, value, 99.0, 98.0]))


# --- Series-only inputs ---


def test_require_series_returns_the_same_series() -> None:
    returns = make_returns([0.01, 0.02, 0.03, 0.04])

    assert require_series(returns, name="returns") is returns


def test_require_series_rejects_dataframes() -> None:
    with pytest.raises(TypeError, match="returns must be a pd.Series"):
        require_series(pd.DataFrame({"a": [0.01]}), name="returns")


# --- Trades table ---


def make_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "exit_time": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )


def test_require_trades_returns_the_same_table() -> None:
    trades = make_trades()

    assert require_trades(trades) is trades


def test_require_trades_rejects_non_dataframes() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        require_trades(make_trades()["entry_time"])


def test_require_trades_names_missing_columns() -> None:
    with pytest.raises(ValueError, match=r"missing required column\(s\) \['exit_time'\]"):
        require_trades(make_trades().drop(columns="exit_time"))


def test_require_trades_rejects_non_datetime_columns() -> None:
    trades = make_trades().assign(entry_time=["2024-01-01", "2024-01-03"])

    with pytest.raises(TypeError, match="'entry_time' must be datetime"):
        require_trades(trades)


def test_require_trades_rejects_missing_times() -> None:
    trades = make_trades()
    trades.loc[1, "exit_time"] = pd.NaT

    with pytest.raises(ValueError, match="'exit_time' must not contain missing"):
        require_trades(trades)


def test_require_trades_rejects_exit_before_entry() -> None:
    trades = make_trades()
    trades.loc[0, "exit_time"] = pd.Timestamp("2023-12-29")

    with pytest.raises(ValueError, match="exit_time precedes its entry_time"):
        require_trades(trades)


# --- Sessions ---


def test_require_sessions_returns_the_same_index() -> None:
    assert require_sessions(SESSIONS) is SESSIONS


def test_require_sessions_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="pd.DatetimeIndex"):
        require_sessions(list(SESSIONS))


def test_require_sessions_rejects_empty_index() -> None:
    with pytest.raises(ValueError, match="at least one session"):
        require_sessions(pd.DatetimeIndex([]))


def test_require_sessions_rejects_unsorted_index() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        require_sessions(SESSIONS[::-1])
