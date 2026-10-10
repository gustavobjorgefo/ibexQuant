# ibexQuant\tests\analytics\test_relative.py

"""Tests for :mod:`ibexQuant.analytics.relative`."""

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
    active_returns,
    alpha,
    beta,
    cagr,
    correlation,
    down_capture,
    excess_cagr,
    information_ratio,
    tracking_error,
    up_capture,
)

NAN: Final[float] = np.nan
N: Final[int] = B3_SESSIONS_PER_YEAR

# Five periods: two up, two down and one flat for the benchmark.
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=5)
BENCHMARK: Final[list[float]] = [0.01, -0.02, 0.03, 0.0, -0.01]
STRATEGY: Final[list[float]] = [0.02, -0.01, 0.02, 0.01, -0.02]

type RelativeMetric = Callable[[pd.Series, pd.Series], float]

METRICS: Final[dict[str, RelativeMetric]] = {
    "correlation": correlation,
    "beta": lambda returns, benchmark: beta(returns, benchmark, 0.10, periods_per_year=N),
    "alpha": lambda returns, benchmark: alpha(returns, benchmark, 0.10, periods_per_year=N),
    "tracking_error": lambda returns, benchmark: tracking_error(
        returns, benchmark, periods_per_year=N
    ),
    "information_ratio": lambda returns, benchmark: information_ratio(
        returns, benchmark, periods_per_year=N
    ),
    "up_capture": lambda returns, benchmark: up_capture(returns, benchmark, periods_per_year=N),
    "down_capture": lambda returns, benchmark: down_capture(
        returns, benchmark, periods_per_year=N
    ),
    "excess_cagr": lambda returns, benchmark: excess_cagr(returns, benchmark, periods_per_year=N),
}


def make_series(values: list[float], name: str = "strategy") -> pd.Series:
    index = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=index, name=name, dtype="float64")


@pytest.fixture
def market() -> pd.Series:
    """Three years of daily benchmark returns."""
    rng = np.random.default_rng(21)
    index = pd.bdate_range("2021-01-01", "2023-12-31")
    return pd.Series(rng.normal(0.0004, 0.012, len(index)), index=index, name="benchmark")


@pytest.fixture
def strategies(market: pd.Series) -> pd.DataFrame:
    """Two strategies with beta near 0.8 and 1.2; the second starts a year late."""
    rng = np.random.default_rng(22)
    noise = rng.normal(0.0, 0.006, size=(len(market), 2))
    frame = pd.DataFrame(
        {
            "defensive": 0.0002 + 0.8 * market + noise[:, 0],
            "aggressive": 1.2 * market + noise[:, 1],
        },
        index=market.index,
    )
    frame.loc[:"2021-12-31", "aggressive"] = NAN
    return frame


# --- Exact relationships ---


def test_a_leveraged_copy_has_beta_two_and_no_alpha() -> None:
    benchmark = make_series(BENCHMARK, "benchmark")
    leveraged = 2.0 * benchmark

    assert beta(leveraged, benchmark, periods_per_year=N) == pytest.approx(2.0)
    assert alpha(leveraged, benchmark, periods_per_year=N) == pytest.approx(0.0, abs=1e-15)
    assert correlation(leveraged, benchmark) == pytest.approx(1.0)


def test_a_constant_premium_is_pure_alpha() -> None:
    # Binary fractions, so (b + premium) - b is exactly the premium. With
    # decimal values such as 0.011 - 0.01 the difference carries ~1e-18 of
    # rounding, a genuine (if negligible) tracking error.
    benchmark = make_series([2**-6, -(2**-5), 2**-4, 0.0, -(2**-7)], "benchmark")
    premium = benchmark + 2**-10

    assert beta(premium, benchmark, periods_per_year=N) == pytest.approx(1.0)
    assert alpha(premium, benchmark, periods_per_year=N) == pytest.approx(2**-10 * N)
    assert tracking_error(premium, benchmark, periods_per_year=N) == 0.0
    assert math.isnan(information_ratio(premium, benchmark, periods_per_year=N))


def test_the_benchmark_against_itself(market: pd.Series) -> None:
    assert tracking_error(market, market, periods_per_year=N) == 0.0
    assert up_capture(market, market, periods_per_year=N) == pytest.approx(1.0)
    assert down_capture(market, market, periods_per_year=N) == pytest.approx(1.0)
    assert excess_cagr(market, market, periods_per_year=N) == 0.0


def test_capture_ratios_compound_over_each_regime() -> None:
    returns = make_series(STRATEGY)
    benchmark = make_series(BENCHMARK, "benchmark")

    # Up periods are the 1st and 3rd; down periods the 2nd and 5th; the flat
    # 4th period is in neither. Two periods per year: each regime is one year.
    up = up_capture(returns, benchmark, periods_per_year=2)
    down = down_capture(returns, benchmark, periods_per_year=2)

    assert up == pytest.approx((1.02 * 1.02 - 1.0) / (1.01 * 1.03 - 1.0))
    assert down == pytest.approx((0.99 * 0.98 - 1.0) / (0.98 * 0.99 - 1.0))


def test_tracking_error_and_information_ratio_measure_active_returns() -> None:
    returns = make_series(STRATEGY)
    benchmark = make_series(BENCHMARK, "benchmark")
    active = returns - benchmark

    assert tracking_error(returns, benchmark, periods_per_year=N) == pytest.approx(
        active.std(ddof=1) * math.sqrt(N)
    )
    assert information_ratio(returns, benchmark, periods_per_year=N) == pytest.approx(
        active.mean() / active.std(ddof=1) * math.sqrt(N)
    )


# --- Reference implementations ---


def test_beta_and_captures_match_empyrical(strategies: pd.DataFrame, market: pd.Series) -> None:
    returns = strategies["defensive"]
    periodic_rate = (1.0 + 0.10) ** (1.0 / N) - 1.0

    assert beta(returns, market, 0.10, periods_per_year=N) == pytest.approx(
        empyrical.beta(returns, market, risk_free=periodic_rate)
    )
    assert up_capture(returns, market, periods_per_year=N) == pytest.approx(
        empyrical.up_capture(returns, market)
    )
    assert down_capture(returns, market, periods_per_year=N) == pytest.approx(
        empyrical.down_capture(returns, market)
    )


@pytest.mark.parametrize("annual_rate", [0.0, 0.10])
def test_alpha_and_beta_are_the_excess_return_regression(
    strategies: pd.DataFrame, market: pd.Series, annual_rate: float
) -> None:
    returns = strategies["defensive"]
    periodic_rate = (1.0 + annual_rate) ** (1.0 / N) - 1.0
    slope, intercept = np.polyfit(market - periodic_rate, returns - periodic_rate, deg=1)

    assert beta(returns, market, annual_rate, periods_per_year=N) == pytest.approx(slope)
    # Annualized arithmetically: the per-period intercept times N.
    assert alpha(returns, market, annual_rate, periods_per_year=N) == pytest.approx(intercept * N)


def test_correlation_matches_numpy(strategies: pd.DataFrame, market: pd.Series) -> None:
    returns = strategies["defensive"]

    assert correlation(returns, market) == pytest.approx(np.corrcoef(returns, market)[0, 1])


# --- Common period ---


def test_excess_cagr_compares_over_the_common_period(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    late = strategies["aggressive"]
    common = late.notna()

    expected = cagr(late[common], periods_per_year=N) - cagr(market[common], periods_per_year=N)

    assert excess_cagr(late, market, periods_per_year=N) == pytest.approx(expected)


def test_a_later_benchmark_restricts_the_comparison() -> None:
    returns = make_series(STRATEGY)
    benchmark = make_series([NAN, *BENCHMARK[1:]], "benchmark")

    result = tracking_error(returns, benchmark, periods_per_year=N)

    active = returns.iloc[1:] - benchmark.iloc[1:]
    assert result == pytest.approx(active.std(ddof=1) * math.sqrt(N))


def test_active_returns_keep_the_shape_and_mark_unpaired_rows(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    result = active_returns(strategies, market)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["defensive", "aggressive"]
    assert result["aggressive"].loc[:"2021-12-31"].isna().all()
    pd.testing.assert_series_equal(
        result["defensive"], strategies["defensive"] - market, check_names=False
    )


# --- Alignment errors ---


def test_benchmark_on_a_different_index_is_rejected() -> None:
    returns = make_series(STRATEGY)
    shifted = pd.Series(BENCHMARK, index=SESSIONS + pd.offsets.BDay(1))

    with pytest.raises(ValueError, match="exactly the same index"):
        beta(returns, shifted, periods_per_year=N)


def test_benchmark_with_fewer_rows_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly the same index"):
        correlation(make_series(STRATEGY), make_series(BENCHMARK[:4], "benchmark"))


def test_benchmark_with_interior_gaps_is_rejected() -> None:
    holed = make_series([0.01, NAN, 0.03, 0.0, -0.01], "benchmark")

    with pytest.raises(ValueError, match="missing values inside its history"):
        correlation(make_series(STRATEGY), holed)


def test_series_without_common_periods_are_rejected() -> None:
    early = make_series([0.01, 0.02, NAN, NAN, NAN])
    late = make_series([NAN, NAN, NAN, 0.01, -0.01], "benchmark")

    with pytest.raises(ValueError, match="no period in common"):
        tracking_error(early, late, periods_per_year=N)
    with pytest.raises(ValueError, match="no period in common"):
        active_returns(early, late)


def test_benchmark_must_be_a_series() -> None:
    returns = make_series(STRATEGY)

    with pytest.raises(TypeError, match="benchmark must be a pd.Series"):
        correlation(returns, returns.to_frame())  # type: ignore[arg-type]


@pytest.mark.parametrize("periods_per_year", [0, 252.0])
def test_annualized_metrics_reject_invalid_periods_per_year(periods_per_year: object) -> None:
    returns = make_series(STRATEGY)
    benchmark = make_series(BENCHMARK, "benchmark")

    for metric in (beta, alpha, tracking_error, information_ratio, excess_cagr):
        with pytest.raises((TypeError, ValueError), match="periods_per_year"):
            metric(returns, benchmark, periods_per_year=periods_per_year)  # type: ignore[operator]
    for capture in (up_capture, down_capture):
        with pytest.raises((TypeError, ValueError), match="periods_per_year"):
            capture(returns, benchmark, periods_per_year=periods_per_year)  # type: ignore[arg-type]


# --- Undefined values ---


def test_constant_benchmark_has_no_beta_or_correlation() -> None:
    returns = make_series(STRATEGY)
    flat = make_series([0.001] * 5, "benchmark")

    assert math.isnan(beta(returns, flat, periods_per_year=N))
    assert math.isnan(alpha(returns, flat, periods_per_year=N))
    assert math.isnan(correlation(returns, flat))


def test_single_common_period_is_undefined() -> None:
    returns = make_series([0.01, NAN, NAN])
    benchmark = make_series([0.02, 0.01, -0.01], "benchmark")

    assert math.isnan(beta(returns, benchmark, periods_per_year=N))
    assert math.isnan(tracking_error(returns, benchmark, periods_per_year=N))
    assert math.isnan(correlation(returns, benchmark))


def test_capture_is_nan_without_that_regime() -> None:
    returns = make_series([0.01, 0.02, 0.0])
    rising = make_series([0.01, 0.03, 0.0], "benchmark")

    assert math.isnan(down_capture(returns, rising, periods_per_year=N))
    assert not math.isnan(up_capture(returns, rising, periods_per_year=N))


def test_empty_returns_raise() -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")

    with pytest.raises(ValueError, match="at least one observation"):
        correlation(empty, empty)


# --- Shape contract ---


@pytest.mark.parametrize("metric", METRICS.values(), ids=METRICS.keys())
def test_series_in_gives_float_out(
    metric: RelativeMetric, strategies: pd.DataFrame, market: pd.Series
) -> None:
    assert isinstance(metric(strategies["defensive"], market), float)


@pytest.mark.parametrize("metric", METRICS.values(), ids=METRICS.keys())
def test_each_strategy_column_is_compared_independently(
    metric: RelativeMetric, strategies: pd.DataFrame, market: pd.Series
) -> None:
    result = metric(strategies, market)  # type: ignore[arg-type]

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["defensive", "aggressive"]
    for column in strategies.columns:
        assert result[column] == pytest.approx(metric(strategies[column], market))
