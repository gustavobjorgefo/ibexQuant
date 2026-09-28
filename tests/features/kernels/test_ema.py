# ibexQuant\tests\features\kernels\test_ema.py

"""Tests for :func:`ibexQuant.features.kernels.ema` and its state."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import (
    DEFAULT_EXPONENTIAL_TOLERANCE,
    EmaState,
    alpha_from_span,
    ema,
    exponential_lookback,
)

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=6, name="timestamp")

# alpha = 0.5 with tolerance 0.25 gives lookback 2: values from the 3rd observation.
ALPHA: Final[float] = 0.5
TOLERANCE: Final[float] = 0.25
SPAN_20: Final[float] = alpha_from_span(20)


def make_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=SESSIONS, dtype="float64").rename_axis(columns="symbol")


def random_walk(length: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = 30.0 + np.cumsum(rng.normal(0.0, 0.5, size=(length, 2)), axis=0)
    index = pd.bdate_range("2024-01-01", periods=length, name="timestamp")
    return pd.DataFrame(values, index=index, columns=pd.Index(["A", "B"], name="symbol"))


# --- Hand-computed values ---


def test_ema_recursion_skips_the_gap() -> None:
    # Observed [1, 2, 4, 5, 6]: ema = 1, 1.5, 2.75, 3.875, 4.9375.
    frame = make_frame({"A": [1.0, 2.0, NAN, 4.0, 5.0, 6.0]})

    result = ema(frame, ALPHA, TOLERANCE)["A"].to_numpy()

    np.testing.assert_allclose(result, [NAN, NAN, NAN, 2.75, 3.875, 4.9375], equal_nan=True)


# --- Agreement with pandas ---


def test_ema_matches_pandas_after_warm_up() -> None:
    frame = random_walk()
    lookback = exponential_lookback(SPAN_20)

    result = ema(frame, SPAN_20)
    expected = frame.ewm(span=20, adjust=False).mean()

    assert result.iloc[:lookback].isna().all().all()
    pd.testing.assert_frame_equal(result.iloc[lookback:], expected.iloc[lookback:])


def test_ema_is_nan_where_input_is_nan() -> None:
    # pandas' ewm carries the last value forward through NaN rows.
    frame = make_frame({"A": [1.0, 2.0, 3.0, 4.0, NAN, NAN]})

    result = ema(frame, ALPHA, TOLERANCE)["A"]

    assert result.iloc[4:].isna().all()


@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 1e-4])
def test_warm_up_length_follows_the_tolerance(tolerance: float) -> None:
    frame = random_walk()

    leading_nan = int(ema(frame, SPAN_20, tolerance)["A"].isna().sum())

    assert leading_nan == exponential_lookback(SPAN_20, tolerance)


# --- Validation ---


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_alpha_outside_open_interval_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        ema(random_walk(), alpha)


def test_invalid_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        ema(random_walk(), SPAN_20, tolerance=0.0)


def test_series_input_raises() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        ema(pd.Series([1.0, 2.0]), SPAN_20)  # type: ignore[arg-type]


# --- Incremental state ---


def test_state_lookback_and_warm_up() -> None:
    state = EmaState(ALPHA, TOLERANCE)

    assert state.lookback == 2
    assert math.isnan(state.update(1.0))
    assert math.isnan(state.update(2.0))
    assert state.update(4.0) == pytest.approx(2.75)


def test_state_nan_input_is_not_an_observation() -> None:
    state = EmaState(ALPHA, TOLERANCE)
    outputs = [state.update(value) for value in [1.0, 2.0, NAN, 4.0, 5.0, 6.0]]

    np.testing.assert_allclose(outputs, [NAN, NAN, NAN, 2.75, 3.875, 4.9375], equal_nan=True)


def test_state_default_tolerance() -> None:
    assert EmaState(SPAN_20).lookback == exponential_lookback(SPAN_20)


def test_state_validates_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        EmaState(1.0)


# --- Parity ---


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_and_incremental_agree_over_full_history(
    seed: int,
    random_prices: Callable[[int], pd.DataFrame],
    assert_parity: Callable[..., None],
) -> None:
    assert_parity(
        lambda frame: ema(frame, SPAN_20), lambda: EmaState(SPAN_20), random_prices(seed)
    )


def test_state_warmed_up_with_lookback_bars_is_within_tolerance(
    random_prices: Callable[[int], pd.DataFrame],
) -> None:
    # A state fed only the last lookback + 1 observations differs from the
    # full-history batch by (1 - alpha) ** lookback times a gap bounded by the
    # range of the data (ADR 0008, D1.6).
    prices = random_prices(0)
    lookback = exponential_lookback(SPAN_20)
    batch = ema(prices, SPAN_20)

    for column in prices.columns:
        observed = prices[column].dropna()
        expected = batch[column].reindex(observed.index)
        bound = DEFAULT_EXPONENTIAL_TOLERANCE * float(np.ptp(observed.to_numpy()))
        for end in range(lookback, len(observed)):
            state = EmaState(SPAN_20)
            for value in observed.iloc[end - lookback : end + 1]:
                warmed = state.update(float(value))
            assert abs(warmed - expected.iloc[end]) <= bound
