# ibexQuant\tests\analytics\test_distribution.py

"""Tests for :mod:`ibexQuant.analytics.distribution`."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import empyrical
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ibexQuant.analytics import (
    best_period,
    conditional_value_at_risk,
    kurtosis,
    mean_return,
    negative_ratio,
    positive_ratio,
    quantile,
    skewness,
    standard_deviation,
    value_at_risk,
    worst_period,
)

NAN: Final[float] = np.nan

# Five periods, one of them flat. Sorted: -0.02, -0.01, 0.00, 0.02, 0.03.
RETURNS: Final[list[float]] = [0.02, -0.01, 0.03, -0.02, 0.0]
# Mean 0.004; squared deviations sum to 0.00172.
STD: Final[float] = math.sqrt(0.00172 / 4)

type Statistic = Callable[[pd.Series], float]

STATISTICS: Final[dict[str, Statistic]] = {
    "mean_return": mean_return,
    "standard_deviation": standard_deviation,
    "skewness": skewness,
    "kurtosis": kurtosis,
    "quantile": lambda returns: quantile(returns, 0.25),
    "best_period": best_period,
    "worst_period": worst_period,
    "positive_ratio": positive_ratio,
    "negative_ratio": negative_ratio,
    "value_at_risk": value_at_risk,
    "conditional_value_at_risk": conditional_value_at_risk,
}


def make_returns(values: list[float] = RETURNS, name: str = "strategy") -> pd.Series:
    index = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=index, name=name, dtype="float64")


@pytest.fixture
def random_returns() -> pd.DataFrame:
    """Three years of daily returns for two series; the second starts a year late."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2021-01-01", "2023-12-31")
    values = rng.normal(0.0004, 0.012, size=(len(index), 2))
    returns = pd.DataFrame(values, index=index, columns=["strategy", "benchmark"])
    returns.loc[:"2021-12-31", "benchmark"] = NAN
    return returns


# --- Hand-computed values ---


def test_mean_and_standard_deviation() -> None:
    returns = make_returns()

    assert mean_return(returns) == pytest.approx(0.004)
    assert standard_deviation(returns) == pytest.approx(STD)


def test_best_and_worst_period() -> None:
    returns = make_returns()

    assert best_period(returns) == 0.03
    assert worst_period(returns) == -0.02


def test_zero_returns_are_neither_positive_nor_negative() -> None:
    returns = make_returns()

    assert positive_ratio(returns) == pytest.approx(0.4)
    assert negative_ratio(returns) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0.0, -0.02), (0.1, -0.016), (0.25, -0.01), (0.5, 0.0), (1.0, 0.03)],
)
def test_quantile_interpolates_linearly(q: float, expected: float) -> None:
    # Position (n - 1) * q in the sorted values: 0.1 falls 40% of the way
    # from -0.02 to -0.01.
    assert quantile(make_returns(), q) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("confidence", "var", "cvar"),
    [(0.90, -0.016, -0.02), (0.75, -0.01, -0.015)],
)
def test_value_at_risk_and_expected_shortfall(confidence: float, var: float, cvar: float) -> None:
    returns = make_returns()

    assert value_at_risk(returns, confidence) == pytest.approx(var)
    assert conditional_value_at_risk(returns, confidence) == pytest.approx(cvar)


def test_symmetric_returns_have_zero_skewness() -> None:
    symmetric = make_returns([-0.02, -0.01, 0.0, 0.01, 0.02])

    assert skewness(symmetric) == pytest.approx(0.0, abs=1e-12)


# --- Reference implementations ---


def test_shape_moments_match_scipy_bias_adjusted(random_returns: pd.DataFrame) -> None:
    strategy = random_returns["strategy"]

    assert skewness(strategy) == pytest.approx(stats.skew(strategy, bias=False))
    assert kurtosis(strategy) == pytest.approx(stats.kurtosis(strategy, fisher=True, bias=False))


def test_tail_measures_match_empyrical(random_returns: pd.DataFrame) -> None:
    strategy = random_returns["strategy"]

    assert value_at_risk(strategy, 0.95) == pytest.approx(
        empyrical.value_at_risk(strategy, cutoff=0.05)
    )
    assert conditional_value_at_risk(strategy, 0.95) == pytest.approx(
        empyrical.conditional_value_at_risk(strategy, cutoff=0.05)
    )


def test_fat_tails_give_positive_excess_kurtosis() -> None:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2000-01-03", periods=5000)
    normal = pd.Series(rng.normal(0.0, 0.01, len(index)), index=index)
    fat_tailed = pd.Series(rng.standard_t(3, len(index)) * 0.01, index=index)

    assert kurtosis(normal) == pytest.approx(0.0, abs=0.2)
    assert kurtosis(fat_tailed) > 2.0


def test_expected_shortfall_is_never_above_value_at_risk(random_returns: pd.DataFrame) -> None:
    for confidence in (0.90, 0.95, 0.99):
        var = value_at_risk(random_returns, confidence)
        cvar = conditional_value_at_risk(random_returns, confidence)
        assert (cvar <= var).all()


# --- Undefined values and errors ---


@pytest.mark.parametrize("statistic", STATISTICS.values(), ids=STATISTICS.keys())
def test_empty_input_raises(statistic: Statistic) -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")

    with pytest.raises(ValueError, match="at least one observation"):
        statistic(empty)


def test_shape_moments_need_enough_observations() -> None:
    assert math.isnan(skewness(make_returns([0.01, 0.02])))
    assert not math.isnan(skewness(make_returns([0.01, 0.02, 0.04])))
    assert math.isnan(kurtosis(make_returns([0.01, 0.02, 0.04])))
    assert not math.isnan(kurtosis(make_returns([0.01, 0.02, 0.04, 0.03])))


@pytest.mark.parametrize("value", [0.1, 0.7, 0.0])
def test_constant_returns(value: float) -> None:
    constant = make_returns([value] * 6)

    assert standard_deviation(constant) == 0.0
    assert math.isnan(skewness(constant))
    assert math.isnan(kurtosis(constant))


@pytest.mark.parametrize("q", [-0.1, 1.1, np.nan])
def test_quantile_rejects_values_outside_unit_interval(q: float) -> None:
    with pytest.raises(ValueError, match="q must"):
        quantile(make_returns(), q)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
@pytest.mark.parametrize("measure", [value_at_risk, conditional_value_at_risk])
def test_tail_measures_reject_invalid_confidence(
    measure: Callable[[pd.Series, float], float], confidence: float
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        measure(make_returns(), confidence)


def test_boolean_parameters_are_rejected() -> None:
    with pytest.raises(TypeError, match="q must be a real number"):
        quantile(make_returns(), True)


# --- Shape contract ---


@pytest.mark.parametrize("statistic", STATISTICS.values(), ids=STATISTICS.keys())
def test_series_in_gives_float_out(statistic: Statistic, random_returns: pd.DataFrame) -> None:
    assert isinstance(statistic(random_returns["strategy"]), float)


@pytest.mark.parametrize("statistic", STATISTICS.values(), ids=STATISTICS.keys())
def test_each_dataframe_column_is_measured_over_its_own_period(
    statistic: Statistic, random_returns: pd.DataFrame
) -> None:
    result = statistic(random_returns)  # type: ignore[arg-type]

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["strategy", "benchmark"]
    for column in random_returns.columns:
        assert result[column] == pytest.approx(statistic(random_returns[column].dropna()))
