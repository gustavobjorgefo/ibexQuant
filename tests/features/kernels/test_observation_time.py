# ibexQuant\tests\features\kernels\test_observation_time.py

"""Tests for :mod:`ibexQuant.features.kernels._observation_time`."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels._observation_time import has_interior_gaps, in_observation_time

NAN: Final[float] = np.nan
SESSIONS: Final[pd.DatetimeIndex] = pd.date_range("2024-01-01", periods=5, name="timestamp")


def make_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=SESSIONS, dtype="float64").rename_axis(columns="symbol")


def lag_one(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.shift(1)


# --- has_interior_gaps ---


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], False),  # complete
        ([NAN, NAN, 3.0, 4.0, 5.0], False),  # listed later
        ([1.0, 2.0, 3.0, NAN, NAN], False),  # delisted
        ([NAN, 2.0, 3.0, 4.0, NAN], False),  # both edges
        ([NAN, NAN, NAN, NAN, NAN], False),  # never observed
        ([1.0, NAN, 3.0, 4.0, 5.0], True),  # suspension
        ([NAN, 2.0, NAN, NAN, 5.0], True),  # late listing and suspension
    ],
)
def test_has_interior_gaps(values: list[float], expected: bool) -> None:
    frame = make_frame({"A": values})

    assert bool(has_interior_gaps(frame)["A"]) is expected


# --- in_observation_time ---


def test_gapped_column_skips_the_gap() -> None:
    frame = make_frame({"A": [1.0, 2.0, NAN, NAN, 5.0]})

    result = in_observation_time(frame, lag_one)

    # Observation-time lag: 5.0 is preceded by 2.0, its previous observation.
    np.testing.assert_array_equal(result["A"].to_numpy(), [NAN, 1.0, NAN, NAN, 2.0])


def test_complete_and_gapped_columns_are_mixed_correctly() -> None:
    frame = make_frame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [10.0, NAN, 30.0, 40.0, 50.0]})

    result = in_observation_time(frame, lag_one)

    expected = make_frame({"A": [NAN, 1.0, 2.0, 3.0, 4.0], "B": [NAN, NAN, 10.0, 30.0, 40.0]})
    pd.testing.assert_frame_equal(result, expected)


def test_fast_path_computes_ungapped_columns_in_one_call() -> None:
    frame = make_frame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [NAN, 2.0, 3.0, 4.0, 5.0],
            "C": [1.0, NAN, 3.0, 4.0, 5.0],
        }
    )
    calls: list[list[str]] = []

    def spy(block: pd.DataFrame) -> pd.DataFrame:
        calls.append(list(block.columns))
        return block.shift(1)

    in_observation_time(frame, spy)

    assert calls == [["A", "B"], ["C"]]


def test_all_gapped_columns_skip_the_fast_path() -> None:
    frame = make_frame({"A": [1.0, NAN, 3.0, 4.0, 5.0], "B": [1.0, 2.0, NAN, 4.0, 5.0]})
    calls: list[list[str]] = []

    def spy(block: pd.DataFrame) -> pd.DataFrame:
        calls.append(list(block.columns))
        return block.shift(1)

    in_observation_time(frame, spy)

    assert calls == [["A"], ["B"]]


def test_result_preserves_index_and_column_order() -> None:
    frame = make_frame({"C": [1.0, NAN, 3.0, 4.0, 5.0], "A": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = in_observation_time(frame, lag_one)

    assert list(result.columns) == ["C", "A"]
    assert result.index.equals(frame.index)
    assert result.dtypes.eq("float64").all()


def test_output_is_nan_wherever_input_is_nan() -> None:
    # ffill carries values into NaN rows; the helper must mask them anyway,
    # including trailing NaN handled by the fast path.
    frame = make_frame({"A": [1.0, 2.0, 3.0, NAN, NAN], "B": [1.0, NAN, 3.0, 4.0, 5.0]})

    result = in_observation_time(frame, lambda block: block.ffill())

    assert result.isna().equals(frame.isna())


def test_never_observed_column_stays_nan() -> None:
    frame = make_frame({"A": [NAN] * 5, "B": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = in_observation_time(frame, lag_one)

    assert result["A"].isna().all()


def test_input_is_not_modified() -> None:
    frame = make_frame({"A": [1.0, NAN, 3.0, 4.0, 5.0]})
    original = frame.copy()

    in_observation_time(frame, lag_one)

    pd.testing.assert_frame_equal(frame, original)
