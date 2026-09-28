# ibexQuant\tests\features\kernels\test_lags.py

"""Tests for :mod:`ibexQuant.features.kernels.lags`."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import LagState, lag, log_returns

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=5, name="timestamp")

# ADR 0003 example: two-session suspension.
SUSPENDED: Final[list[float]] = [30.00, 30.30, NAN, NAN, 31.50]


def make_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=SESSIONS, dtype="float64").rename_axis(columns="symbol")


# --- Observation time ---


def test_lag_counts_observations_not_sessions() -> None:
    result = lag(make_frame({"PETR4.SA": SUSPENDED}), periods=1)["PETR4.SA"].to_numpy()

    # On resumption, lag(1) is the last price before the suspension.
    np.testing.assert_array_equal(result, [NAN, 30.00, NAN, NAN, 30.30])


def test_log_returns_equal_log_price_minus_its_lag_even_across_gaps() -> None:
    prices = make_frame({"PETR4.SA": SUSPENDED})
    log_prices = np.log(prices)

    pd.testing.assert_frame_equal(log_returns(prices), log_prices - lag(log_prices, 1))


def test_lag_matches_pandas_shift_without_gaps() -> None:
    frame = make_frame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 4.0, 3.0, 2.0, 1.0]})

    pd.testing.assert_frame_equal(lag(frame, 2), frame.shift(2))


def test_lag_after_late_listing() -> None:
    frame = make_frame({"A": [NAN, NAN, 1.0, 2.0, 3.0]})

    np.testing.assert_array_equal(lag(frame, 1)["A"].to_numpy(), [NAN, NAN, NAN, 1.0, 2.0])


# --- Validation ---


@pytest.mark.parametrize("periods", [0, -1])
def test_periods_must_be_positive(periods: int) -> None:
    with pytest.raises(ValueError, match="periods must be >= 1"):
        lag(make_frame({"A": SUSPENDED}), periods)


def test_periods_must_be_an_integer() -> None:
    with pytest.raises(TypeError, match="periods must be an integer"):
        lag(make_frame({"A": SUSPENDED}), 1.0)  # type: ignore[arg-type]


def test_series_input_raises() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        lag(pd.Series([1.0, 2.0]), 1)  # type: ignore[arg-type]


# --- Incremental state ---


def test_state_lookback_equals_periods() -> None:
    assert LagState(5).lookback == 5


def test_state_follows_observations() -> None:
    state = LagState(1)
    outputs = [state.update(value) for value in SUSPENDED]

    np.testing.assert_array_equal(outputs, [NAN, 30.00, NAN, NAN, 30.30])


def test_state_validates_periods() -> None:
    with pytest.raises(ValueError, match="periods must be >= 1"):
        LagState(0)


def test_state_first_values_are_nan() -> None:
    state = LagState(2)

    assert math.isnan(state.update(1.0))
    assert math.isnan(state.update(2.0))
    assert state.update(3.0) == 1.0


# --- Parity ---


@pytest.mark.parametrize("periods", [1, 5])
@pytest.mark.parametrize("seed", [0, 1])
def test_batch_and_incremental_agree(
    periods: int,
    seed: int,
    random_prices: Callable[[int], pd.DataFrame],
    assert_parity: Callable[..., None],
) -> None:
    assert_parity(
        lambda frame: lag(frame, periods), lambda: LagState(periods), random_prices(seed)
    )
