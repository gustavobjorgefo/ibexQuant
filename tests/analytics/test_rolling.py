# ibexQuant\tests\analytics\test_rolling.py

"""Tests for :mod:`ibexQuant.analytics.rolling`."""

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
    beta,
    correlation,
    rolling_beta,
    rolling_correlation,
    rolling_return,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from ibexQuant.features.kernels import historical_volatility

NAN: Final[float] = np.nan
N: Final[int] = B3_SESSIONS_PER_YEAR
WINDOW: Final[int] = 63

# Rolling moments use running sums; recomputing a window from scratch agrees
# within the tolerance fixed for kernels (ADR 0008, amendment of 2026-09-28).
RTOL: Final[float] = 1e-9
ATOL: Final[float] = 1e-12

type Rolling = Callable[[pd.Series], pd.Series]
type WindowMetric = Callable[[pd.Series], float]

# Each rolling function paired with the full-sample metric it must reproduce
# on every window.
SINGLE_SERIES: Final[dict[str, tuple[Rolling, WindowMetric]]] = {
    "return": (lambda returns: rolling_return(returns, WINDOW), total_return),
    "volatility": (
        lambda returns: rolling_volatility(returns, WINDOW, periods_per_year=N),
        lambda window: annualized_volatility(window, periods_per_year=N),
    ),
    "sharpe": (
        lambda returns: rolling_sharpe(returns, WINDOW, 0.10, periods_per_year=N),
        lambda window: sharpe_ratio(window, 0.10, periods_per_year=N),
    ),
    "sortino": (
        lambda returns: rolling_sortino(returns, WINDOW, 0.10, periods_per_year=N),
        lambda window: sortino_ratio(window, 0.10, periods_per_year=N),
    ),
}

ALL_ROLLING: Final[dict[str, Callable[[pd.DataFrame, pd.Series], object]]] = {
    "return": lambda returns, _: rolling_return(returns, WINDOW),
    "volatility": lambda returns, _: rolling_volatility(returns, WINDOW, periods_per_year=N),
    "sharpe": lambda returns, _: rolling_sharpe(returns, WINDOW, periods_per_year=N),
    "sortino": lambda returns, _: rolling_sortino(returns, WINDOW, periods_per_year=N),
    "beta": lambda returns, market: rolling_beta(returns, market, WINDOW, periods_per_year=N),
    "correlation": lambda returns, market: rolling_correlation(returns, market, WINDOW),
}


def make_series(values: list[float] | np.ndarray, name: str = "strategy") -> pd.Series:
    index = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=index, name=name, dtype="float64")


@pytest.fixture
def market() -> pd.Series:
    """Two years of daily benchmark returns."""
    rng = np.random.default_rng(31)
    index = pd.bdate_range("2022-01-03", periods=500)
    return pd.Series(rng.normal(0.0004, 0.012, len(index)), index=index, name="benchmark")


@pytest.fixture
def strategies(market: pd.Series) -> pd.DataFrame:
    """Two strategies on the benchmark's index; the second starts 100 bars late."""
    rng = np.random.default_rng(32)
    noise = rng.normal(0.0, 0.006, size=(len(market), 2))
    frame = pd.DataFrame(
        {"defensive": 0.8 * market + noise[:, 0], "aggressive": 1.2 * market + noise[:, 1]},
        index=market.index,
    )
    frame.iloc[:100, 1] = NAN
    return frame


# --- Each window reproduces its full-sample metric ---


@pytest.mark.parametrize("pair", SINGLE_SERIES.values(), ids=SINGLE_SERIES.keys())
def test_each_window_matches_the_full_sample_metric(
    pair: tuple[Rolling, WindowMetric], strategies: pd.DataFrame
) -> None:
    rolling, metric = pair
    returns = strategies["defensive"]
    result = rolling(returns)

    for end in (WINDOW, 200, len(returns)):
        expected = metric(returns.iloc[end - WINDOW : end])
        assert result.iloc[end - 1] == pytest.approx(expected, rel=RTOL, abs=ATOL)


def test_beta_and_correlation_windows_match_relative_metrics(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    returns = strategies["defensive"]
    betas = rolling_beta(returns, market, WINDOW, 0.10, periods_per_year=N)
    correlations = rolling_correlation(returns, market, WINDOW)

    for end in (WINDOW, 300, len(returns)):
        window = slice(end - WINDOW, end)
        assert betas.iloc[end - 1] == pytest.approx(
            beta(returns.iloc[window], market.iloc[window], 0.10, periods_per_year=N),
            rel=RTOL,
            abs=ATOL,
        )
        assert correlations.iloc[end - 1] == pytest.approx(
            correlation(returns.iloc[window], market.iloc[window]), rel=RTOL, abs=ATOL
        )


# --- Reference implementations ---


def test_sharpe_and_beta_match_empyrical(strategies: pd.DataFrame, market: pd.Series) -> None:
    returns = strategies["defensive"]

    np.testing.assert_allclose(
        rolling_sharpe(returns, WINDOW, periods_per_year=N).iloc[WINDOW - 1 :].to_numpy(),
        empyrical.roll_sharpe_ratio(returns, window=WINDOW).to_numpy(),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        rolling_beta(returns, market, WINDOW, periods_per_year=N).iloc[WINDOW - 1 :].to_numpy(),
        empyrical.roll_beta(returns, market, window=WINDOW).to_numpy(),
        rtol=RTOL,
    )


def test_volatility_agrees_with_the_feature_kernel(strategies: pd.DataFrame) -> None:
    # The kernel serves features in observation time; on gap-free returns both
    # implementations must give the same number, annualization apart.
    returns = strategies["defensive"]

    kernel = historical_volatility(returns.to_frame(), WINDOW)["defensive"]

    pd.testing.assert_series_equal(
        rolling_volatility(returns, WINDOW, periods_per_year=N),
        kernel * math.sqrt(N),
        rtol=RTOL,
        atol=ATOL,
    )


# --- Warm-up and edges ---


def test_values_start_once_the_window_is_full(strategies: pd.DataFrame) -> None:
    result = rolling_volatility(strategies, WINDOW, periods_per_year=N)

    assert result["defensive"].iloc[: WINDOW - 1].isna().all()
    assert result["defensive"].iloc[WINDOW - 1 :].notna().all()
    # The late column warms up from its own first observation.
    assert result["aggressive"].iloc[: 100 + WINDOW - 1].isna().all()
    assert result["aggressive"].iloc[100 + WINDOW - 1 :].notna().all()


def test_a_later_strategy_pairs_only_common_bars_with_the_benchmark(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    late = strategies["aggressive"]

    result = rolling_beta(late, market, WINDOW, periods_per_year=N)

    first = 100 + WINDOW - 1
    assert result.iloc[:first].isna().all()
    window = slice(100, 100 + WINDOW)
    assert result.iloc[first] == pytest.approx(
        beta(late.iloc[window], market.iloc[window], periods_per_year=N), rel=RTOL
    )


def test_trailing_gaps_end_the_rolling_values() -> None:
    returns = make_series([0.01, -0.02, 0.03, 0.01, NAN, NAN])

    result = rolling_return(returns, 2)

    assert result.iloc[-2:].isna().all()
    assert result.iloc[3] == pytest.approx(1.03 * 1.01 - 1.0)


def test_window_longer_than_the_series_gives_no_values() -> None:
    result = rolling_sharpe(make_series([0.01, -0.02, 0.03]), 10, periods_per_year=N)

    assert result.isna().all()


# --- Exact zero dispersion ---


def test_constant_windows_have_exactly_zero_volatility_despite_running_sums() -> None:
    # Large moves followed by identical returns leave pandas' running sums with
    # ~1e-8 of residual deviation; without exact detection the Sharpe ratio of
    # the last window would be around 1.6e8 instead of NaN.
    rng = np.random.default_rng(6)
    values = np.r_[np.clip(rng.normal(0.0, 0.5, 40), -0.9, None), [0.1] * 25]
    returns = make_series(values)
    assert returns.rolling(20).std().iloc[-1] != 0.0

    assert rolling_volatility(returns, 20, periods_per_year=N).iloc[-1] == 0.0
    assert math.isnan(rolling_sharpe(returns, 20, periods_per_year=N).iloc[-1])


def test_strategy_in_cash_has_no_ratios() -> None:
    returns = make_series([0.01, -0.02, 0.015, *([0.0] * 10)])

    assert rolling_volatility(returns, 5, periods_per_year=N).iloc[-1] == 0.0
    assert math.isnan(rolling_sharpe(returns, 5, periods_per_year=N).iloc[-1])
    assert math.isnan(rolling_sortino(returns, 5, periods_per_year=N).iloc[-1])


def test_sortino_is_nan_once_the_last_loss_leaves_the_window() -> None:
    returns = make_series([-0.03, 0.01, 0.02, 0.01, 0.03, 0.02])

    result = rolling_sortino(returns, 3, periods_per_year=N)

    assert not math.isnan(result.iloc[2])
    assert result.iloc[3:].isna().all()


def test_constant_benchmark_windows_have_no_beta_or_correlation() -> None:
    benchmark = make_series([0.01, -0.02, 0.03, *([0.001] * 5)], "benchmark")
    returns = make_series([0.02, -0.01, 0.02, 0.01, -0.02, 0.0, 0.01, 0.02])

    assert math.isnan(rolling_beta(returns, benchmark, 4, periods_per_year=N).iloc[-1])
    assert math.isnan(rolling_correlation(returns, benchmark, 4).iloc[-1])


def test_rolling_return_survives_a_total_loss() -> None:
    returns = make_series([0.05, -1.0, 0.10, 0.20])

    result = rolling_return(returns, 2)

    np.testing.assert_allclose(result.to_numpy(), [NAN, -1.0, -1.0, 1.10 * 1.20 - 1.0])


# --- Shape contract ---


@pytest.mark.parametrize("compute", ALL_ROLLING.values(), ids=ALL_ROLLING.keys())
def test_series_in_gives_named_series_out(
    compute: Callable[[pd.Series, pd.Series], object],
    strategies: pd.DataFrame,
    market: pd.Series,
) -> None:
    result = compute(strategies["defensive"], market)

    assert isinstance(result, pd.Series)
    assert result.name == "defensive"
    assert result.index.equals(market.index)


@pytest.mark.parametrize("compute", ALL_ROLLING.values(), ids=ALL_ROLLING.keys())
def test_each_dataframe_column_matches_its_series(
    compute: Callable[[pd.DataFrame | pd.Series, pd.Series], object],
    strategies: pd.DataFrame,
    market: pd.Series,
) -> None:
    result = compute(strategies, market)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["defensive", "aggressive"]
    for column in strategies.columns:
        pd.testing.assert_series_equal(result[column], compute(strategies[column], market))  # type: ignore[arg-type]


# --- Errors ---


@pytest.mark.parametrize(
    ("compute", "minimum"),
    [
        (lambda returns, window: rolling_return(returns, window), 1),
        (lambda returns, window: rolling_volatility(returns, window, periods_per_year=N), 2),
        (lambda returns, window: rolling_sharpe(returns, window, periods_per_year=N), 2),
        (lambda returns, window: rolling_sortino(returns, window, periods_per_year=N), 2),
        (lambda returns, window: rolling_correlation(returns, returns, window), 2),
    ],
    ids=["return", "volatility", "sharpe", "sortino", "correlation"],
)
def test_windows_below_the_minimum_are_rejected(
    compute: Callable[[pd.Series, object], object], minimum: int
) -> None:
    returns = make_series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError, match="window"):
        compute(returns, minimum - 1)
    with pytest.raises(TypeError, match="window"):
        compute(returns, 2.0)


def test_empty_returns_raise() -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64")

    with pytest.raises(ValueError, match="at least one observation"):
        rolling_return(empty, 2)
    with pytest.raises(ValueError, match="at least one observation"):
        rolling_beta(empty, empty, 2, periods_per_year=N)


def test_benchmark_on_a_different_index_is_rejected(market: pd.Series) -> None:
    with pytest.raises(ValueError, match="exactly the same index"):
        rolling_beta(market, market.iloc[1:], WINDOW, periods_per_year=N)


def test_strategy_without_common_bars_is_rejected() -> None:
    early = make_series([0.01, 0.02, NAN, NAN])
    late = make_series([NAN, NAN, 0.01, -0.01], "benchmark")

    with pytest.raises(ValueError, match="no period in common"):
        rolling_correlation(early, late, 2)


def test_annualized_functions_reject_invalid_periods_per_year() -> None:
    returns = make_series([0.01, -0.02, 0.03])

    with pytest.raises(ValueError, match="periods_per_year"):
        rolling_volatility(returns, 2, periods_per_year=0)
    with pytest.raises(TypeError, match="periods_per_year"):
        rolling_sharpe(returns, 2, periods_per_year=252.0)  # type: ignore[arg-type]
