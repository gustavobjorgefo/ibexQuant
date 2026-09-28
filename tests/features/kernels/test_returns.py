# ibexQuant\tests\features\kernels\test_returns.py

"""Tests for :mod:`ibexQuant.features.kernels.returns`."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import (
    LogReturnState,
    SimpleReturnState,
    log_returns,
    simple_returns,
)
from ibexQuant.features.kernels.state import IncrementalState

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=5, name="timestamp")

type ReturnKernel = Callable[[pd.DataFrame], pd.DataFrame]

KERNELS: Final[list[ReturnKernel]] = [simple_returns, log_returns]
STATES: Final[list[type[IncrementalState]]] = [SimpleReturnState, LogReturnState]


def make_prices(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=SESSIONS, dtype="float64").rename_axis(columns="symbol")


# --- Hand-computed values (ADR 0003 example) ---


def test_log_returns_skip_the_suspension() -> None:
    prices = make_prices({"PETR4.SA": [30.00, 30.30, NAN, NAN, 31.50]})

    result = log_returns(prices)["PETR4.SA"].to_numpy()

    np.testing.assert_allclose(
        result,
        [NAN, math.log(30.30 / 30.00), NAN, NAN, math.log(31.50 / 30.30)],
        equal_nan=True,
    )
    assert result[1] == pytest.approx(0.00995, abs=1e-5)
    assert result[4] == pytest.approx(0.03884, abs=1e-5)


def test_simple_returns_skip_the_suspension() -> None:
    prices = make_prices({"PETR4.SA": [30.00, 30.30, NAN, NAN, 31.50]})

    result = simple_returns(prices)["PETR4.SA"].to_numpy()

    np.testing.assert_allclose(result, [NAN, 0.01, NAN, NAN, 31.50 / 30.30 - 1.0], equal_nan=True)


# --- Agreement with pandas on complete data ---


def test_log_returns_match_pandas_without_gaps() -> None:
    prices = make_prices({"A": [10.0, 11.0, 10.5, 12.0, 12.6], "B": [5.0, 4.9, 5.1, 5.1, 5.3]})

    pd.testing.assert_frame_equal(log_returns(prices), np.log(prices).diff(), rtol=1e-12)


def test_simple_returns_match_pandas_without_gaps() -> None:
    prices = make_prices({"A": [10.0, 11.0, 10.5, 12.0, 12.6], "B": [5.0, 4.9, 5.1, 5.1, 5.3]})

    pd.testing.assert_frame_equal(simple_returns(prices), prices.pct_change(), rtol=1e-12)


# --- Properties ---


def test_log_returns_are_additive_over_time() -> None:
    prices = make_prices({"A": [10.0, 11.0, NAN, 12.0, 12.6]})

    total = log_returns(prices)["A"].sum()

    assert total == pytest.approx(math.log(12.6 / 10.0))


@pytest.mark.parametrize("kernel", KERNELS)
def test_first_observation_after_listing_is_nan(kernel: ReturnKernel) -> None:
    prices = make_prices({"A": [NAN, NAN, 10.0, 11.0, 12.0]})

    result = kernel(prices)["A"]

    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].notna().all()


@pytest.mark.parametrize("kernel", KERNELS)
def test_shape_is_preserved_and_input_untouched(kernel: ReturnKernel) -> None:
    prices = make_prices({"B": [5.0, NAN, 5.1, 5.1, 5.3], "A": [10.0, 11.0, 10.5, 12.0, 12.6]})
    original = prices.copy()

    result = kernel(prices)

    assert result.index.equals(prices.index)
    assert list(result.columns) == ["B", "A"]
    assert result.dtypes.eq("float64").all()
    pd.testing.assert_frame_equal(prices, original)


# --- Validation ---


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_non_positive_prices_raise(kernel: ReturnKernel, bad_price: float) -> None:
    prices = make_prices(
        {"A": [10.0, 11.0, 12.0, 13.0, 14.0], "B": [5.0, bad_price, 5.0, 5.0, 5.0]}
    )

    with pytest.raises(ValueError, match=r"strictly positive.*\['B'\]"):
        kernel(prices)


@pytest.mark.parametrize("kernel", KERNELS)
def test_series_input_raises(kernel: ReturnKernel) -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        kernel(pd.Series([1.0, 2.0]))  # type: ignore[arg-type]


# --- Incremental states ---


@pytest.mark.parametrize("state_type", STATES)
def test_state_lookback_is_one(state_type: type[IncrementalState]) -> None:
    assert state_type().lookback == 1


@pytest.mark.parametrize("state_type", STATES)
def test_state_warms_up_on_first_observation(state_type: type[IncrementalState]) -> None:
    state = state_type()

    assert math.isnan(state.update(10.0))
    assert not math.isnan(state.update(11.0))


def test_state_nan_input_is_not_an_observation() -> None:
    state = LogReturnState()
    state.update(30.00)

    assert math.isnan(state.update(NAN))
    assert state.update(31.50) == pytest.approx(math.log(31.50 / 30.00))


@pytest.mark.parametrize("state_type", STATES)
@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_state_non_positive_price_raises(
    state_type: type[IncrementalState], bad_price: float
) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        state_type().update(bad_price)


def test_state_repr() -> None:
    assert repr(LogReturnState()) == "LogReturnState(lookback=1)"


# --- Parity ---


@pytest.mark.parametrize(("kernel", "state_type"), list(zip(KERNELS, STATES, strict=True)))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_and_incremental_agree(
    kernel: ReturnKernel,
    state_type: type[IncrementalState],
    seed: int,
    random_prices: Callable[[int], pd.DataFrame],
    assert_parity: Callable[..., None],
) -> None:
    assert_parity(kernel, state_type, random_prices(seed))
