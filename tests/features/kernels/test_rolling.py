# ibexQuant\tests\features\kernels\test_rolling.py

"""Tests for :mod:`ibexQuant.features.kernels.rolling` and ``sma``."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import (
    RollingStdState,
    RollingSumState,
    SmaState,
    log_returns,
    rolling_std,
    rolling_sum,
    sma,
)
from ibexQuant.features.kernels.state import IncrementalState

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=6, name="timestamp")

# One suspension: observation time sees [1, 2, 4, 5, 6].
GAPPED: Final[list[float]] = [1.0, 2.0, NAN, 4.0, 5.0, 6.0]


def make_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=SESSIONS, dtype="float64").rename_axis(columns="symbol")


def complete_frame() -> pd.DataFrame:
    return make_frame(
        {"A": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0], "B": [10.0, 10.5, 9.8, 10.1, 10.9, 11.2]}
    )


# --- Hand-computed values in observation time ---


def test_rolling_sum_skips_the_gap() -> None:
    result = rolling_sum(make_frame({"A": GAPPED}), window=3)["A"].to_numpy()

    np.testing.assert_allclose(result, [NAN, NAN, NAN, 7.0, 11.0, 15.0], equal_nan=True)


def test_sma_skips_the_gap() -> None:
    result = sma(make_frame({"A": GAPPED}), window=3)["A"].to_numpy()

    np.testing.assert_allclose(result, [NAN, NAN, NAN, 7 / 3, 11 / 3, 5.0], equal_nan=True)


def test_rolling_std_skips_the_gap() -> None:
    result = rolling_std(make_frame({"A": GAPPED}), window=3)["A"].to_numpy()

    expected = [
        NAN,
        NAN,
        NAN,
        statistics.stdev([1.0, 2.0, 4.0]),
        statistics.stdev([2.0, 4.0, 5.0]),
        statistics.stdev([4.0, 5.0, 6.0]),
    ]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_rolling_std_population_divisor() -> None:
    result = rolling_std(make_frame({"A": GAPPED}), window=3, ddof=0)["A"].iloc[-1]

    assert result == pytest.approx(statistics.pstdev([4.0, 5.0, 6.0]))


# --- Agreement with pandas on complete data ---


@pytest.mark.parametrize("window", [1, 2, 4])
def test_rolling_sum_matches_pandas(window: int) -> None:
    frame = complete_frame()

    pd.testing.assert_frame_equal(rolling_sum(frame, window), frame.rolling(window).sum())


@pytest.mark.parametrize("window", [1, 2, 4])
def test_sma_matches_pandas(window: int) -> None:
    frame = complete_frame()

    pd.testing.assert_frame_equal(sma(frame, window), frame.rolling(window).mean())


@pytest.mark.parametrize(("window", "ddof"), [(2, 1), (4, 1), (1, 0), (4, 0)])
def test_rolling_std_matches_pandas(window: int, ddof: int) -> None:
    frame = complete_frame()

    pd.testing.assert_frame_equal(
        rolling_std(frame, window, ddof), frame.rolling(window).std(ddof=ddof)
    )


# --- Properties ---


def test_rolling_sum_of_log_returns_is_the_window_log_return() -> None:
    prices = make_frame({"A": [10.0, 11.0, NAN, 12.0, 12.6, 13.0]})

    momentum = rolling_sum(log_returns(prices), window=3)["A"].iloc[-1]

    # Observed prices are [10, 11, 12, 12.6, 13]; the last three returns span 11 -> 13.
    assert momentum == pytest.approx(math.log(13.0 / 11.0))


def test_constant_window_has_zero_std() -> None:
    result = rolling_std(make_frame({"A": [1.1] * 6}), window=3)["A"]

    assert (result.iloc[2:] == 0.0).all()


# --- Validation ---


@pytest.mark.parametrize("kernel", [rolling_sum, sma])
def test_window_must_be_positive(kernel: Callable[[pd.DataFrame, int], pd.DataFrame]) -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        kernel(complete_frame(), 0)


@pytest.mark.parametrize("kernel", [rolling_sum, sma])
def test_window_must_be_an_integer(kernel: Callable[[pd.DataFrame, int], pd.DataFrame]) -> None:
    with pytest.raises(TypeError, match="window must be an integer"):
        kernel(complete_frame(), 20.0)  # type: ignore[arg-type]


def test_rolling_std_window_must_exceed_ddof() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        rolling_std(complete_frame(), window=1)


def test_rolling_std_ddof_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        rolling_std(complete_frame(), window=3, ddof=-1)


@pytest.mark.parametrize("kernel", [rolling_sum, sma])
def test_series_input_raises(kernel: Callable[[pd.DataFrame, int], pd.DataFrame]) -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        kernel(pd.Series([1.0, 2.0]), 2)  # type: ignore[arg-type]


# --- Incremental states ---


@pytest.mark.parametrize("state_type", [RollingSumState, SmaState, RollingStdState])
def test_state_lookback_is_window_minus_one(state_type: type[RollingSumState]) -> None:
    assert state_type(20).lookback == 19


def test_state_warms_up_after_window_observations() -> None:
    state = SmaState(3)

    assert math.isnan(state.update(1.0))
    assert math.isnan(state.update(2.0))
    assert state.update(3.0) == pytest.approx(2.0)


def test_state_nan_input_is_not_an_observation() -> None:
    state = RollingSumState(3)
    outputs = [state.update(value) for value in GAPPED]

    np.testing.assert_allclose(outputs, [NAN, NAN, NAN, 7.0, 11.0, 15.0], equal_nan=True)


def test_std_state_validates_parameters() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        RollingStdState(1)
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        RollingStdState(3, ddof=-1)


def test_window_state_validates_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        SmaState(0)


# --- Parity ---

type StateFactory = Callable[[], IncrementalState]

CASES: Final[list[object]] = [
    pytest.param(lambda f: rolling_sum(f, 20), lambda: RollingSumState(20), id="sum_20"),
    pytest.param(lambda f: sma(f, 20), lambda: SmaState(20), id="sma_20"),
    pytest.param(lambda f: rolling_std(f, 20), lambda: RollingStdState(20), id="std_20"),
    pytest.param(
        lambda f: rolling_std(f, 5, ddof=0), lambda: RollingStdState(5, ddof=0), id="std_5_ddof0"
    ),
]


@pytest.mark.parametrize(("kernel", "make_state"), CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("on_returns", [False, True], ids=["prices", "log_returns"])
def test_batch_and_incremental_agree(
    kernel: Callable[[pd.DataFrame], pd.DataFrame],
    make_state: StateFactory,
    seed: int,
    on_returns: bool,
    random_prices: Callable[[int], pd.DataFrame],
    assert_parity: Callable[..., None],
) -> None:
    prices = random_prices(seed)
    values = log_returns(prices) if on_returns else prices

    assert_parity(kernel, make_state, values)
