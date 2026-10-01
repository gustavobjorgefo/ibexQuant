# ibexQuant\tests\analytics\test_validation.py

"""Tests for :mod:`ibexQuant.analytics._validation`."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.analytics._validation import require_equity, require_returns, require_series

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
