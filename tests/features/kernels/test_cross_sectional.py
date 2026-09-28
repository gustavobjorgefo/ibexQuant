# ibexQuant\tests\features\kernels\test_cross_sectional.py

"""Tests for :mod:`ibexQuant.features.kernels.cross_sectional`."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import cs_rank, cs_zscore, log_returns

NAN: Final[float] = np.nan
SYMBOLS: Final[list[str]] = ["A", "B", "C", "D"]

type CrossSectionalKernel = Callable[..., pd.DataFrame]

KERNELS: Final[list[CrossSectionalKernel]] = [cs_rank, cs_zscore]


def make_frame(rows: list[list[float]]) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(rows), name="timestamp")
    return pd.DataFrame(rows, index=index, columns=pd.Index(SYMBOLS, name="symbol"))


def make_mask(frame: pd.DataFrame, rows: list[list[bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=frame.index, columns=frame.columns)


# --- cs_rank ---


def test_rank_spans_zero_to_one_whatever_the_universe_size() -> None:
    frame = make_frame([[3.0, 1.0, 2.0, NAN], [4.0, 1.0, 3.0, 2.0]])

    result = cs_rank(frame)

    np.testing.assert_allclose(result.iloc[0].to_numpy(), [1.0, 0.0, 0.5, NAN], equal_nan=True)
    np.testing.assert_allclose(result.iloc[1].to_numpy(), [1.0, 0.0, 2 / 3, 1 / 3])


@pytest.mark.parametrize("assets", [3, 7, 100])
def test_rank_median_is_one_half(assets: int) -> None:
    values = np.arange(assets, dtype="float64").reshape(1, -1)
    frame = pd.DataFrame(values, columns=[f"S{i}" for i in range(assets)])

    result = cs_rank(frame).iloc[0]

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert result.mean() == pytest.approx(0.5)


def test_rank_ties_take_the_average_rank() -> None:
    frame = make_frame([[3.0, 1.0, 2.0, 2.0]])

    np.testing.assert_allclose(cs_rank(frame).iloc[0].to_numpy(), [1.0, 0.0, 0.5, 0.5])


# --- cs_zscore ---


def test_zscore_uses_the_row_statistics() -> None:
    row = [3.0, 1.0, 2.0, 6.0]
    frame = make_frame([row])

    expected = [(value - statistics.mean(row)) / statistics.stdev(row) for value in row]

    np.testing.assert_allclose(cs_zscore(frame).iloc[0].to_numpy(), expected)


def test_zscore_population_divisor() -> None:
    row = [3.0, 1.0, 2.0, 6.0]
    frame = make_frame([row])

    expected = [(value - statistics.mean(row)) / statistics.pstdev(row) for value in row]

    np.testing.assert_allclose(cs_zscore(frame, ddof=0).iloc[0].to_numpy(), expected)


@pytest.mark.parametrize("value", [0.1, 0.7, 5.0])
def test_zscore_of_a_constant_row_is_nan(value: float) -> None:
    # pandas' row std of [0.1] * 4 is ~1.7e-17, not 0; zero dispersion must
    # still be detected.
    frame = make_frame([[value] * 4])

    assert cs_zscore(frame).isna().all().all()


def test_zscore_requires_more_assets_than_ddof() -> None:
    frame = make_frame([[1.0, 2.0, NAN, NAN]])

    assert cs_zscore(frame, ddof=1).notna().sum().sum() == 2
    assert cs_zscore(frame, ddof=2).isna().all().all()


def test_zscore_rejects_negative_ddof() -> None:
    with pytest.raises(ValueError, match="ddof must be >= 0"):
        cs_zscore(make_frame([[1.0, 2.0, 3.0, 4.0]]), ddof=-1)


# --- Shared behaviour ---


@pytest.mark.parametrize("kernel", KERNELS)
def test_rows_with_fewer_than_two_assets_are_nan(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[5.0, NAN, NAN, NAN], [NAN, NAN, NAN, NAN]])

    assert kernel(frame).isna().all().all()


@pytest.mark.parametrize("kernel", KERNELS)
def test_mask_excludes_assets_from_statistics_and_output(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[100.0, 1.0, 2.0, 3.0]])
    mask = make_mask(frame, [[False, True, True, True]])

    result = kernel(frame, universe_mask=mask)
    without_a = kernel(frame[["B", "C", "D"]])

    assert math.isnan(result.iloc[0, 0])
    np.testing.assert_allclose(result.iloc[0, 1:].to_numpy(), without_a.iloc[0].to_numpy())


@pytest.mark.parametrize("kernel", KERNELS)
def test_mask_changes_over_time(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    mask = make_mask(frame, [[True, True, True, False], [True, True, True, True]])

    result = kernel(frame, universe_mask=mask)

    assert math.isnan(result.iloc[0, 3])
    assert not math.isnan(result.iloc[1, 3])


@pytest.mark.parametrize("kernel", KERNELS)
def test_output_shape_and_input_untouched(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[3.0, 1.0, NAN, 2.0], [1.0, 2.0, 3.0, 4.0]])
    original = frame.copy()

    result = kernel(frame)

    assert result.index.equals(frame.index)
    assert result.columns.equals(frame.columns)
    assert result.dtypes.eq("float64").all()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("kernel", KERNELS)
def test_scale_invariant(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[3.0, 1.0, 2.0, 6.0], [1.5, 2.5, 0.5, 4.0]])

    pd.testing.assert_frame_equal(kernel(frame), kernel(frame * 4.2))


# --- Mask validation ---


@pytest.mark.parametrize("kernel", KERNELS)
def test_mask_must_be_a_dataframe(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[1.0, 2.0, 3.0, 4.0]])

    with pytest.raises(TypeError, match="universe_mask"):
        kernel(frame, universe_mask=np.ones((1, 4), dtype=bool))


@pytest.mark.parametrize("kernel", KERNELS)
def test_mask_must_be_aligned(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[1.0, 2.0, 3.0, 4.0]])
    mask = pd.DataFrame([[True, True, True]], index=frame.index, columns=["A", "B", "C"])

    with pytest.raises(ValueError, match="same index and columns"):
        kernel(frame, universe_mask=mask)


@pytest.mark.parametrize("kernel", KERNELS)
def test_mask_must_be_bool(kernel: CrossSectionalKernel) -> None:
    frame = make_frame([[1.0, 2.0, 3.0, 4.0]])
    mask = make_mask(frame, [[True, True, True, True]]).astype("float64")

    with pytest.raises(ValueError, match="must be bool"):
        kernel(frame, universe_mask=mask)


@pytest.mark.parametrize("kernel", KERNELS)
def test_series_input_raises(kernel: CrossSectionalKernel) -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        kernel(pd.Series([1.0, 2.0]))


# --- Parity: the row-by-row application is the incremental mode ---


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("seed", [0, 1])
def test_row_by_row_equals_whole_frame(
    kernel: CrossSectionalKernel,
    seed: int,
    random_prices: Callable[[int], pd.DataFrame],
) -> None:
    returns = log_returns(random_prices(seed))
    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(
        rng.random(returns.shape) > 0.2, index=returns.index, columns=returns.columns
    )

    whole = kernel(returns, universe_mask=mask)
    row_by_row = pd.concat(
        [kernel(returns.iloc[[i]], universe_mask=mask.iloc[[i]]) for i in range(len(returns))]
    )

    pd.testing.assert_frame_equal(row_by_row, whole)
