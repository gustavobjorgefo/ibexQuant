# ibexQuant\tests\features\kernels\test_validation.py

"""Tests for :mod:`ibexQuant.features.kernels._validation`."""

from __future__ import annotations

import pandas as pd
import pytest

from ibexQuant.features.kernels._validation import require_frame

# --- require_frame ---


def test_require_frame_returns_the_same_frame() -> None:
    frame = pd.DataFrame({"PETR4.SA": [1.0]})

    assert require_frame(frame) is frame


def test_require_frame_rejects_series() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        require_frame(pd.Series([1.0]))
