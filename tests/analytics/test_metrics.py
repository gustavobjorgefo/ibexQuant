# ibexQuant\tests\analytics\test_metrics.py

"""Tests for :mod:`ibexQuant.analytics.metrics`."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import empyrical
import numpy as np
import pandas as pd
import pytest

from ibexQuant.analytics import (
    B3_SESSIONS_PER_YEAR,
    annualized_volatility,
    cagr,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)

NAN: Final[float] = np.nan
N: Final[int] = B3_SESSIONS_PER_YEAR

# Four sessions, small enough to verify by hand. With periods_per_year=4 the
# sample spans exactly one "year", so CAGR equals the total return and the
# annualization factor is sqrt(4) = 2.
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=4)
RETURNS: Final[list[float]] = [0.02, -0.01, 0.03, -0.02]
TOTAL: Final[float] = 1.02 * 0.99 * 1.03 * 0.98 - 1.0
# Sample standard deviation: mean 0.005, squared deviations sum to 0.0017.
STD: Final[float] = math.sqrt(0.0017 / 3)
# Downside deviation over all four periods: sqrt((0 + 0.01² + 0 + 0.02²) / 4).
DOWNSIDE: Final[float] = math.sqrt(0.0005 / 4)

type Metric = Callable[[pd.Series], float]

METRICS: Final[dict[str, Metric]] = {
    "total_return": total_return,
    "cagr": lambda returns: cagr(returns, periods_per_year=N),
    "annualized_volatility": lambda returns: annualized_volatility(returns, periods_per_year=N),
    "max_drawdown": max_drawdown,
    "max_drawdown_duration": max_drawdown_duration,
    "sharpe_ratio": lambda returns: sharpe_ratio(returns, 0.10, periods_per_year=N),
    "sortino_ratio": lambda returns: sortino_ratio(returns, 0.10, periods_per_year=N),
    "calmar_ratio": lambda returns: calmar_ratio(returns, periods_per_year=N),
}

ANNUALIZED: Final[dict[str, Callable[..., object]]] = {
    "cagr": cagr,
    "annualized_volatility": annualized_volatility,
    "sharpe_ratio": sharpe_ratio,
    "sortino_ratio": sortino_ratio,
    "calmar_ratio": calmar_ratio,
}


def make_returns(values: list[float] = RETURNS, name: str = "strategy") -> pd.Series:
    index = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=index, name=name, dtype="float64")


@pytest.fixture
def random_returns() -> pd.DataFrame:
    """Three years of daily returns for two series; the second starts a year late."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2021-01-01", "2023-12-31")
    values = rng.normal(0.0004, 0.012, size=(len(index), 2))
    returns = pd.DataFrame(values, index=index, columns=["strategy", "benchmark"])
    returns.loc[:"2021-12-31", "benchmark"] = NAN
    return returns


# --- Hand-computed values ---


def test_total_return_compounds_every_period() -> None:
    assert total_return(make_returns()) == pytest.approx(TOTAL)


def test_cagr_equals_total_return_over_exactly_one_year() -> None:
    assert cagr(make_returns(), periods_per_year=4) == pytest.approx(TOTAL)


def test_cagr_annualizes_by_sessions() -> None:
    # Four sessions out of eight per year: half a year, so growth is squared.
    assert cagr(make_returns(), periods_per_year=8) == pytest.approx((1.0 + TOTAL) ** 2 - 1.0)


def test_annualized_volatility_scales_the_sample_deviation() -> None:
    assert annualized_volatility(make_returns(), periods_per_year=4) == pytest.approx(STD * 2)


def test_sharpe_ratio_divides_mean_by_deviation() -> None:
    assert sharpe_ratio(make_returns(), periods_per_year=4) == pytest.approx(0.005 / STD * 2)


def test_sortino_ratio_uses_downside_deviation_over_all_periods() -> None:
    assert sortino_ratio(make_returns(), periods_per_year=4) == pytest.approx(0.005 / DOWNSIDE * 2)


def test_max_drawdown_is_the_deepest_fall() -> None:
    # Drawdowns are 0%, -1%, 0%, -2%.
    assert max_drawdown(make_returns()) == pytest.approx(-0.02)


def test_max_drawdown_duration_is_the_longest_episode() -> None:
    # Episodes: peak at session 0 recovered at 2 (2 periods); peak at 2, still
    # open at the end (1 period).
    assert max_drawdown_duration(make_returns()) == 2.0


def test_calmar_ratio_divides_cagr_by_max_drawdown() -> None:
    assert calmar_ratio(make_returns(), periods_per_year=4) == pytest.approx(TOTAL / 0.02)


# --- Reference implementation ---


def test_metrics_match_empyrical(random_returns: pd.DataFrame) -> None:
    strategy = random_returns["strategy"]
    annual_rate = 0.10
    # empyrical takes the periodic rate; ours converts the annual rate itself.
    periodic_rate = (1.0 + annual_rate) ** (1.0 / N) - 1.0

    assert cagr(strategy, periods_per_year=N) == pytest.approx(empyrical.cagr(strategy))
    assert annualized_volatility(strategy, periods_per_year=N) == pytest.approx(
        empyrical.annual_volatility(strategy)
    )
    assert sharpe_ratio(strategy, annual_rate, periods_per_year=N) == pytest.approx(
        empyrical.sharpe_ratio(strategy, risk_free=periodic_rate)
    )
    assert sortino_ratio(strategy, annual_rate, periods_per_year=N) == pytest.approx(
        empyrical.sortino_ratio(strategy, required_return=periodic_rate)
    )
    assert max_drawdown(strategy) == pytest.approx(empyrical.max_drawdown(strategy))
    assert calmar_ratio(strategy, periods_per_year=N) == pytest.approx(
        empyrical.calmar_ratio(strategy)
    )


# --- Risk-free rate ---


def test_constant_risk_free_series_matches_the_equivalent_annual_rate() -> None:
    returns = make_returns()
    periodic = (1.0 + 0.10) ** (1.0 / 4) - 1.0
    risk_free = pd.Series(periodic, index=SESSIONS)

    for ratio in (sharpe_ratio, sortino_ratio):
        assert ratio(returns, risk_free, periods_per_year=4) == pytest.approx(
            ratio(returns, 0.10, periods_per_year=4)
        )


def test_sharpe_ratio_measures_dispersion_of_excess_returns() -> None:
    returns = make_returns()
    risk_free = pd.Series([0.001, 0.003, 0.002, 0.004], index=SESSIONS)
    excess = returns - risk_free

    result = sharpe_ratio(returns, risk_free, periods_per_year=4)

    assert result == pytest.approx(excess.mean() / excess.std(ddof=1) * 2)


# --- Undefined values and errors ---


@pytest.mark.parametrize("metric", METRICS.values(), ids=METRICS.keys())
def test_empty_input_raises(metric: Metric) -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")

    with pytest.raises(ValueError, match="at least one observation"):
        metric(empty)


def test_single_observation_has_no_dispersion() -> None:
    single = make_returns([0.01])

    assert total_return(single) == pytest.approx(0.01)
    assert math.isnan(annualized_volatility(single, periods_per_year=N))
    assert math.isnan(sharpe_ratio(single, periods_per_year=N))


@pytest.mark.parametrize("value", [0.1, 0.7, 0.0])
def test_constant_returns_have_exactly_zero_volatility(value: float) -> None:
    constant = make_returns([value] * 5)

    assert annualized_volatility(constant, periods_per_year=N) == 0.0
    assert math.isnan(sharpe_ratio(constant, periods_per_year=N))


def test_sortino_ratio_is_nan_without_downside() -> None:
    assert math.isnan(sortino_ratio(make_returns([0.01, 0.02, 0.0]), periods_per_year=N))


def test_series_that_never_falls() -> None:
    rising = make_returns([0.01, 0.02, 0.01])

    assert max_drawdown(rising) == 0.0
    assert max_drawdown_duration(rising) == 0.0
    assert math.isnan(calmar_ratio(rising, periods_per_year=N))


def test_total_loss() -> None:
    wiped = make_returns([0.05, -1.0, 0.0])

    assert total_return(wiped) == -1.0
    assert cagr(wiped, periods_per_year=N) == -1.0
    assert max_drawdown(wiped) == -1.0


def test_max_drawdown_duration_counts_an_open_episode() -> None:
    # Peak at the first session, never recovered: the episode runs to the end.
    falling = make_returns([0.05, -0.01, -0.01, 0.0, -0.01, 0.0])

    assert max_drawdown_duration(falling) == 5.0


@pytest.mark.parametrize("metric", ANNUALIZED.values(), ids=ANNUALIZED.keys())
@pytest.mark.parametrize("periods_per_year", [0, 252.0])
def test_annualized_metrics_reject_invalid_periods_per_year(
    metric: Callable[..., object], periods_per_year: object
) -> None:
    with pytest.raises((TypeError, ValueError), match="periods_per_year"):
        metric(make_returns(), periods_per_year=periods_per_year)


def test_non_pandas_input_raises() -> None:
    with pytest.raises(TypeError, match=r"pd.Series or pd.DataFrame"):
        total_return([0.01, 0.02])  # type: ignore[call-overload]


# --- Shape contract ---


@pytest.mark.parametrize("metric", METRICS.values(), ids=METRICS.keys())
def test_series_in_gives_float_out(metric: Metric, random_returns: pd.DataFrame) -> None:
    assert isinstance(metric(random_returns["strategy"]), float)


@pytest.mark.parametrize("metric", METRICS.values(), ids=METRICS.keys())
def test_each_dataframe_column_is_measured_over_its_own_period(
    metric: Metric, random_returns: pd.DataFrame
) -> None:
    result = metric(random_returns)  # type: ignore[arg-type]

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["strategy", "benchmark"]
    for column in random_returns.columns:
        assert result[column] == pytest.approx(metric(random_returns[column].dropna()))
