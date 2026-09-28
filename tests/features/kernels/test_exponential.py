# ibexQuant\tests\features\kernels\test_exponential.py

"""Tests for :mod:`ibexQuant.features.kernels.exponential`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels import (
    DEFAULT_EXPONENTIAL_TOLERANCE,
    alpha_from_decay,
    alpha_from_halflife,
    alpha_from_span,
    exponential_lookback,
)

# --- Conversions ---


def test_alpha_from_span() -> None:
    assert alpha_from_span(20) == pytest.approx(2 / 21)


def test_alpha_from_halflife_halves_the_weight() -> None:
    alpha = alpha_from_halflife(10)

    assert (1 - alpha) ** 10 == pytest.approx(0.5)


def test_alpha_from_decay_is_the_riskmetrics_complement() -> None:
    assert alpha_from_decay(0.94) == pytest.approx(0.06)


@pytest.mark.parametrize(
    ("pandas_kwargs", "alpha"),
    [
        ({"span": 20}, alpha_from_span(20)),
        ({"halflife": 10}, alpha_from_halflife(10)),
    ],
)
def test_conversions_match_pandas(pandas_kwargs: dict[str, float], alpha: float) -> None:
    series = pd.Series(np.random.default_rng(0).normal(size=50))

    expected = series.ewm(adjust=False, **pandas_kwargs).mean()
    result = series.ewm(alpha=alpha, adjust=False).mean()

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("span", [1, 0.5, -3])
def test_alpha_from_span_rejects_span_not_above_one(span: float) -> None:
    with pytest.raises(ValueError, match="span must be > 1"):
        alpha_from_span(span)


@pytest.mark.parametrize("halflife", [0, -1])
def test_alpha_from_halflife_rejects_non_positive(halflife: float) -> None:
    with pytest.raises(ValueError, match="halflife must be > 0"):
        alpha_from_halflife(halflife)


@pytest.mark.parametrize("decay", [0, 1, 1.5])
def test_alpha_from_decay_rejects_outside_unit_interval(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        alpha_from_decay(decay)


# --- Lookback ---


def test_default_tolerance() -> None:
    assert DEFAULT_EXPONENTIAL_TOLERANCE == 1e-3


@pytest.mark.parametrize(
    ("tolerance", "span_20", "decay_094"),
    [(1e-2, 47, 75), (1e-3, 70, 112), (1e-4, 93, 149)],
)
def test_lookback_reference_values(tolerance: float, span_20: int, decay_094: int) -> None:
    assert exponential_lookback(alpha_from_span(20), tolerance) == span_20
    assert exponential_lookback(alpha_from_decay(0.94), tolerance) == decay_094


@pytest.mark.parametrize("alpha", [0.01, 0.06, 2 / 21, 0.3, 0.9])
@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 1e-6])
def test_lookback_is_the_smallest_sufficient_number_of_bars(
    alpha: float, tolerance: float
) -> None:
    lookback = exponential_lookback(alpha, tolerance)

    assert (1 - alpha) ** lookback <= tolerance
    assert lookback == 0 or (1 - alpha) ** (lookback - 1) > tolerance


@pytest.mark.parametrize(("alpha", "bars"), [(0.5, 2), (0.1, 4), (0.05, 5)])
def test_lookback_handles_exact_integer_ratio(alpha: float, bars: int) -> None:
    # The tolerance equals the starting weight after exactly `bars` bars. In
    # floating point, log(tolerance) / log(1 - alpha) can land just above the
    # integer, and a naive ceil would return bars + 1.
    tolerance = (1 - alpha) ** bars

    assert exponential_lookback(alpha, tolerance) == bars


def test_lookback_matches_the_starting_point_influence() -> None:
    alpha = alpha_from_span(20)
    lookback = exponential_lookback(alpha)
    history = np.random.default_rng(1).normal(size=500)

    # Two EWMAs over the same recent data, started from different points,
    # differ by at most the starting weight times the spread of the starts.
    full = pd.Series(history).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    short = pd.Series(history[-(lookback + 1) :]).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    spread = np.ptp(history)

    assert abs(full - short) <= DEFAULT_EXPONENTIAL_TOLERANCE * spread


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_lookback_rejects_alpha_outside_open_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        exponential_lookback(alpha)


@pytest.mark.parametrize("tolerance", [0.0, 1.0])
def test_lookback_rejects_tolerance_outside_open_interval(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        exponential_lookback(0.1, tolerance)
