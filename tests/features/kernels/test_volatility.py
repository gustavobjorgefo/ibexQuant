# ibexQuant\tests\features\kernels\test_volatility.py

"""Tests for :mod:`ibexQuant.features.kernels.volatility`."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import (
    DEFAULT_EXPONENTIAL_TOLERANCE,
    EwmaVolatilityState,
    HistoricalVolatilityState,
    RollingStdState,
    alpha_from_decay,
    ewma_volatility,
    exponential_lookback,
    historical_volatility,
    log_returns,
    rolling_std,
)

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=5, name="timestamp")

ALPHA: Final[float] = 0.5
TOLERANCE: Final[float] = 0.25  # lookback 2
RISKMETRICS: Final[float] = alpha_from_decay(0.94)


def make_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=SESSIONS, dtype="float64").rename_axis(columns="symbol")


def random_returns(length: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-01", periods=length, name="timestamp")
    return pd.DataFrame(
        rng.normal(0.0005, 0.02, size=(length, 2)),
        index=index,
        columns=pd.Index(["A", "B"], name="symbol"),
    )


# --- Historical volatility ---


@pytest.mark.parametrize(("window", "ddof"), [(20, 1), (5, 0)])
def test_historical_volatility_is_rolling_std(window: int, ddof: int) -> None:
    returns = random_returns()

    pd.testing.assert_frame_equal(
        historical_volatility(returns, window, ddof), rolling_std(returns, window, ddof)
    )


def test_historical_volatility_state_is_a_rolling_std_state() -> None:
    state = HistoricalVolatilityState(20)

    assert isinstance(state, RollingStdState)
    assert repr(state) == "HistoricalVolatilityState(lookback=19)"


def test_historical_volatility_validates_parameters() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        historical_volatility(random_returns(), window=1)


def test_historical_volatility_demeans_unlike_riskmetrics() -> None:
    # Constant returns: zero dispersion around the mean, but a non-zero
    # RiskMetrics volatility (zero-mean assumption).
    frame = make_frame({"A": [0.01] * 5})

    assert (historical_volatility(frame, window=3)["A"].iloc[2:] == 0.0).all()


@pytest.mark.parametrize("seed", [0, 1])
def test_historical_batch_and_incremental_agree(
    seed: int,
    random_prices: Callable[[int], pd.DataFrame],
    assert_parity: Callable[..., None],
) -> None:
    assert_parity(
        lambda frame: historical_volatility(frame, 20),
        lambda: HistoricalVolatilityState(20),
        log_returns(random_prices(seed)),
    )


# --- EWMA: hand-computed values ---


def test_recursion_starts_from_first_squared_return_and_skips_the_gap() -> None:
    # Observed returns [0.1, -0.2, 0.3, 0.0]:
    # variance = 0.01, 0.025, 0.0575, 0.02875 (alpha = 0.5).
    frame = make_frame({"A": [0.1, -0.2, NAN, 0.3, 0.0]})

    result = ewma_volatility(frame, ALPHA, TOLERANCE)["A"].to_numpy()

    np.testing.assert_allclose(
        result, [NAN, NAN, NAN, math.sqrt(0.0575), math.sqrt(0.02875)], equal_nan=True
    )


def test_riskmetrics_recursion() -> None:
    returns = random_returns()
    lookback = exponential_lookback(RISKMETRICS)

    variance = returns["A"].iloc[0] ** 2
    expected = []
    for value in returns["A"]:
        variance = 0.94 * variance + 0.06 * value**2
        expected.append(math.sqrt(variance))
    # The first recursion step is sigma2_0 = r_0 ** 2, which the loop reproduces
    # because 0.94 * r_0 ** 2 + 0.06 * r_0 ** 2 == r_0 ** 2.

    result = ewma_volatility(returns, RISKMETRICS)["A"].to_numpy()

    np.testing.assert_allclose(result[lookback:], expected[lookback:], rtol=1e-12)


# --- Properties ---


def test_mean_is_assumed_zero() -> None:
    # A constant return has zero dispersion around its mean, but RiskMetrics
    # does not demean: the volatility is the return's absolute value.
    frame = make_frame({"A": [0.01] * 5})

    result = ewma_volatility(frame, ALPHA, TOLERANCE)["A"]

    np.testing.assert_allclose(result.iloc[2:].to_numpy(), 0.01)


def test_matches_pandas_on_squared_returns() -> None:
    returns = random_returns()
    lookback = exponential_lookback(RISKMETRICS)

    result = ewma_volatility(returns, RISKMETRICS)
    expected = np.sqrt((returns**2).ewm(alpha=RISKMETRICS, adjust=False).mean())

    pd.testing.assert_frame_equal(result.iloc[lookback:], expected.iloc[lookback:])
    assert result.iloc[:lookback].isna().all().all()


def test_is_nan_where_input_is_nan() -> None:
    frame = make_frame({"A": [0.1, -0.2, 0.3, NAN, NAN]})

    result = ewma_volatility(frame, ALPHA, TOLERANCE)["A"]

    assert result.iloc[3:].isna().all()


# --- Validation ---


def test_decay_passed_as_alpha_is_still_valid_but_wrong() -> None:
    # 0.94 is a valid alpha, so it cannot be rejected; this documents why the
    # conversion must be explicit (ADR 0008, D1.7): it weighs the last return
    # at 94% instead of 6%.
    assert exponential_lookback(0.94) < exponential_lookback(RISKMETRICS)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_alpha_outside_open_interval_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        ewma_volatility(random_returns(), alpha)


def test_series_input_raises() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        ewma_volatility(pd.Series([0.1, 0.2]), RISKMETRICS)  # type: ignore[arg-type]


# --- Incremental state ---


def test_state_lookback_and_warm_up() -> None:
    state = EwmaVolatilityState(ALPHA, TOLERANCE)
    outputs = [state.update(value) for value in [0.1, -0.2, NAN, 0.3, 0.0]]

    assert state.lookback == 2
    np.testing.assert_allclose(
        outputs, [NAN, NAN, NAN, math.sqrt(0.0575), math.sqrt(0.02875)], equal_nan=True
    )


def test_state_default_tolerance() -> None:
    assert EwmaVolatilityState(RISKMETRICS).lookback == 112


# --- Parity ---


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_and_incremental_agree_over_full_history(
    seed: int,
    random_prices: Callable[[int], pd.DataFrame],
    assert_parity: Callable[..., None],
) -> None:
    returns = log_returns(random_prices(seed))

    assert_parity(
        lambda frame: ewma_volatility(frame, RISKMETRICS),
        lambda: EwmaVolatilityState(RISKMETRICS),
        returns,
    )


def test_state_warmed_up_with_lookback_bars_is_within_tolerance(
    random_prices: Callable[[int], pd.DataFrame],
) -> None:
    # The bound holds on the variance: its gap is at most (1 - alpha) ** lookback
    # times the range of the squared returns (ADR 0008, D1.6).
    returns = log_returns(random_prices(1))
    lookback = exponential_lookback(RISKMETRICS)
    batch = ewma_volatility(returns, RISKMETRICS)

    for column in returns.columns:
        observed = returns[column].dropna()
        expected = batch[column].reindex(observed.index)
        bound = DEFAULT_EXPONENTIAL_TOLERANCE * float(np.ptp(observed.to_numpy() ** 2))
        for end in range(lookback, len(observed)):
            state = EwmaVolatilityState(RISKMETRICS)
            for value in observed.iloc[end - lookback : end + 1]:
                warmed = state.update(float(value))
            assert abs(warmed**2 - expected.iloc[end] ** 2) <= bound
