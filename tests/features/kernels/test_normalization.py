# ibexQuant\tests\features\kernels\test_normalization.py

"""Tests for :mod:`ibexQuant.features.kernels.normalization`."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import (
    RollingZScoreState,
    log_returns,
    rolling_std,
    rolling_zscore,
    sma,
)

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=6, name="timestamp")


def make_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=SESSIONS, dtype="float64").rename_axis(columns="symbol")


def zscore(window: list[float]) -> float:
    return (window[-1] - statistics.mean(window)) / statistics.stdev(window)


# --- Hand-computed values ---


def test_zscore_skips_the_gap_and_includes_the_current_value() -> None:
    # Observed [1, 2, 4, 5, 7]; windows of 3 end at 4, 5 and 7.
    frame = make_frame({"A": [1.0, 2.0, NAN, 4.0, 5.0, 7.0]})

    result = rolling_zscore(frame, window=3)["A"].to_numpy()

    expected = [NAN, NAN, NAN, zscore([1, 2, 4]), zscore([2, 4, 5]), zscore([4, 5, 7])]
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_zscore_is_the_composition_of_sma_and_rolling_std() -> None:
    frame = make_frame({"A": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0], "B": [2.0, 1.0, NAN, 3.0, 5.0, 4.0]})

    expected = (frame - sma(frame, 3)) / rolling_std(frame, 3)

    pd.testing.assert_frame_equal(rolling_zscore(frame, 3), expected)


# --- Properties ---


@pytest.mark.parametrize("window", [3, 20, 60])
def test_zscore_is_bounded_when_the_window_includes_the_current_value(window: int) -> None:
    # The most extreme case: all previous values equal, then an outlier.
    values = [0.0] * (window - 1) + [1_000.0]
    index = pd.bdate_range("2024-01-01", periods=window, name="timestamp")
    frame = pd.DataFrame({"A": values}, index=index)

    result = rolling_zscore(frame, window)["A"].iloc[-1]

    assert result == pytest.approx((window - 1) / math.sqrt(window))


def test_constant_window_gives_nan() -> None:
    frame = make_frame({"A": [1.0, 2.0, 5.0, 5.0, 5.0, 6.0]})

    result = rolling_zscore(frame, window=3)["A"]

    assert math.isnan(result.iloc[4])
    assert not math.isnan(result.iloc[5])


def test_zscore_is_scale_invariant() -> None:
    frame = make_frame({"A": [10.0, 11.0, 10.5, 12.0, 11.5, 12.5]})

    pd.testing.assert_frame_equal(rolling_zscore(frame, 3), rolling_zscore(frame * 7.3, 3))


# --- Validation ---


def test_window_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        rolling_zscore(make_frame({"A": [1.0] * 6}), window=1, ddof=0)


def test_ddof_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        rolling_zscore(make_frame({"A": [1.0] * 6}), window=3, ddof=-1)


def test_series_input_raises() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        rolling_zscore(pd.Series([1.0, 2.0, 3.0]), 2)  # type: ignore[arg-type]


# --- Incremental state ---


def test_state_lookback_and_values() -> None:
    state = RollingZScoreState(3)
    outputs = [state.update(value) for value in [1.0, 2.0, NAN, 4.0, 5.0, 7.0]]

    assert state.lookback == 2
    np.testing.assert_allclose(
        outputs,
        [NAN, NAN, NAN, zscore([1, 2, 4]), zscore([2, 4, 5]), zscore([4, 5, 7])],
        equal_nan=True,
    )


@pytest.mark.parametrize("value", [5.0, 0.1, 0.7])
def test_state_constant_window_gives_nan(value: float) -> None:
    # Regression: fsum([0.1] * 3) / 3 is not exactly 0.1, which used to leave
    # ~1e-17 of deviation and a z-score of about +-0.82 instead of NaN.
    state = RollingZScoreState(3)
    outputs = [state.update(value) for _ in range(4)]
    batch = rolling_zscore(make_frame({"A": [value] * 6}), window=3)["A"]

    assert all(math.isnan(output) for output in outputs)
    assert batch.isna().all()


def test_state_validates_parameters() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        RollingZScoreState(1, ddof=0)
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        RollingZScoreState(3, ddof=-1)


# --- Parity ---


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("on_returns", [False, True], ids=["prices", "log_returns"])
def test_batch_and_incremental_agree(
    seed: int,
    on_returns: bool,
    random_prices: Callable[[int], pd.DataFrame],
    assert_parity: Callable[..., None],
) -> None:
    prices = random_prices(seed)
    values = log_returns(prices) if on_returns else prices

    assert_parity(lambda frame: rolling_zscore(frame, 20), lambda: RollingZScoreState(20), values)
