# ibexQuant\tests\features\kernels\test_validation.py

"""Tests for :mod:`ibexQuant.features.kernels._validation`."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels._validation import require_frame, require_integer, require_real

# --- require_frame ---


def test_require_frame_returns_the_same_frame() -> None:
    frame = pd.DataFrame({"PETR4.SA": [1.0]})

    assert require_frame(frame) is frame


def test_require_frame_rejects_series() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        require_frame(pd.Series([1.0]))


# --- require_integer ---


@pytest.mark.parametrize("value", [1, 20, np.int64(20)])
def test_require_integer_accepts_python_and_numpy_integers(value: object) -> None:
    result = require_integer("window", value, minimum=1)

    assert result == int(value)  # type: ignore[call-overload]
    assert type(result) is int


@pytest.mark.parametrize("value", [True, 3.0, "3", None])
def test_require_integer_rejects_non_integers(value: object) -> None:
    with pytest.raises(TypeError, match="window must be an integer"):
        require_integer("window", value, minimum=1)


def test_require_integer_rejects_values_below_minimum() -> None:
    with pytest.raises(ValueError, match="window must be >= 1, got 0"):
        require_integer("window", 0, minimum=1)


# --- require_real ---


@pytest.mark.parametrize("value", [0.5, 1, np.float64(0.5), np.int32(2)])
def test_require_real_accepts_python_and_numpy_numbers(value: object) -> None:
    result = require_real("x", value)

    assert result == float(value)  # type: ignore[arg-type]
    assert type(result) is float


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_require_real_rejects_non_numbers(value: object) -> None:
    with pytest.raises(TypeError, match="x must be a real number"):
        require_real("x", value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_require_real_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        require_real("x", value)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_require_real_bounds_are_exclusive(value: float) -> None:
    with pytest.raises(ValueError):
        require_real("alpha", value, above=0.0, below=1.0)


def test_require_real_accepts_values_inside_bounds() -> None:
    assert require_real("alpha", 0.06, above=0.0, below=1.0) == 0.06
