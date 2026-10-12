# ibexQuant\tests\analytics\test_summary.py

"""Tests for :mod:`ibexQuant.analytics.summary`."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.analytics import (
    B3_SESSIONS_PER_YEAR,
    METRIC_UNITS,
    beta,
    cagr,
    expectancy,
    exposure,
    performance_summary,
    profit_factor,
    sharpe_ratio,
    total_return,
    trade_summary,
    value_at_risk,
)

NAN: Final[float] = np.nan
N: Final[int] = B3_SESSIONS_PER_YEAR
LATE_START: Final[int] = 120


@pytest.fixture
def market() -> pd.Series:
    """Three years of daily benchmark returns, starting LATE_START bars late."""
    rng = np.random.default_rng(41)
    index = pd.bdate_range("2021-01-01", periods=3 * N)
    values = pd.Series(rng.normal(0.0004, 0.012, len(index)), index=index, name="BOVA11")
    values.iloc[:LATE_START] = NAN
    return values


@pytest.fixture
def strategies(market: pd.Series) -> pd.DataFrame:
    """Two strategies observed over the whole index."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0001, 0.006, size=(len(market), 2))
    base = market.fillna(0.0).to_numpy()[:, np.newaxis]
    return pd.DataFrame(
        base * [0.8, 1.2] + noise, index=market.index, columns=["defensive", "aggressive"]
    )


# --- Performance table ---


def test_table_has_metrics_as_rows_and_series_as_columns(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    summary = performance_summary(strategies, market, periods_per_year=N)

    assert list(summary.metrics.columns) == ["defensive", "aggressive", "BOVA11"]
    assert summary.metrics.index.names == ["section", "metric"]
    assert list(summary.metrics.index.get_level_values("section").unique()) == [
        "return",
        "risk",
        "risk_adjusted",
        "distribution",
        "benchmark",
    ]
    assert (summary.metrics.dtypes == "float64").all()


def test_every_cell_is_the_analytics_function_over_the_common_period(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    summary = performance_summary(strategies, market, 0.10, periods_per_year=N)
    common = strategies.iloc[LATE_START:]
    common_market = market.iloc[LATE_START:]
    metrics = summary.metrics

    assert metrics.loc[("return", "total_return"), "defensive"] == pytest.approx(
        total_return(common["defensive"])
    )
    assert metrics.loc[("risk_adjusted", "sharpe_ratio"), "BOVA11"] == pytest.approx(
        sharpe_ratio(common_market, 0.10, periods_per_year=N)
    )
    assert metrics.loc[("distribution", "value_at_risk"), "aggressive"] == pytest.approx(
        value_at_risk(common["aggressive"], 0.95)
    )
    assert metrics.loc[("benchmark", "beta"), "aggressive"] == pytest.approx(
        beta(common["aggressive"], common_market, 0.10, periods_per_year=N)
    )


def test_period_and_conventions_are_recorded(strategies: pd.DataFrame, market: pd.Series) -> None:
    summary = performance_summary(strategies, market, 0.10, periods_per_year=N, confidence=0.99)

    assert summary.start == market.index[LATE_START]
    assert summary.end == market.index[-1]
    assert summary.periods == 3 * N - LATE_START
    assert summary.years == pytest.approx((3 * N - LATE_START) / N)
    assert summary.periods_per_year == N
    assert summary.risk_free == 0.10
    assert summary.confidence == 0.99
    assert summary.common_period is True
    assert summary.short_columns == ()


def test_each_column_over_its_own_period_when_requested(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    summary = performance_summary(strategies, market, periods_per_year=N, common_period=False)

    assert summary.start == strategies.index[0]
    assert summary.periods == 3 * N
    assert summary.metrics.loc[("return", "total_return"), "defensive"] == pytest.approx(
        total_return(strategies["defensive"])
    )
    assert summary.metrics.loc[("return", "total_return"), "BOVA11"] == pytest.approx(
        total_return(market.dropna())
    )


def test_benchmark_rows_are_blank_in_the_benchmark_column(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    benchmark_rows = performance_summary(strategies, market, periods_per_year=N).metrics.loc[
        "benchmark"
    ]

    assert benchmark_rows["BOVA11"].isna().all()
    assert benchmark_rows[["defensive", "aggressive"]].notna().all().all()


def test_without_benchmark_there_is_no_benchmark_section(strategies: pd.DataFrame) -> None:
    summary = performance_summary(strategies, periods_per_year=N)

    assert "benchmark" not in summary.metrics.index.get_level_values("section")
    assert list(summary.metrics.columns) == ["defensive", "aggressive"]


def test_single_unnamed_series_becomes_a_strategy_column(strategies: pd.DataFrame) -> None:
    unnamed = strategies["defensive"].rename(None)

    summary = performance_summary(unnamed, periods_per_year=N)

    assert list(summary.metrics.columns) == ["strategy"]


def test_unnamed_benchmark_becomes_a_benchmark_column(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    summary = performance_summary(strategies, market.rename(None), periods_per_year=N)

    assert list(summary.metrics.columns) == ["defensive", "aggressive", "benchmark"]
    assert market.name == "BOVA11"


def test_periodic_risk_free_series_is_recorded_by_name(strategies: pd.DataFrame) -> None:
    cdi = pd.Series(0.0004, index=strategies.index, name="CDI")

    summary = performance_summary(strategies, risk_free=cdi, periods_per_year=N)

    assert summary.risk_free == "CDI"
    assert summary.metrics.loc[("risk_adjusted", "sharpe_ratio"), "defensive"] == pytest.approx(
        sharpe_ratio(strategies["defensive"], cdi, periods_per_year=N)
    )


# --- GIPS: no annualized returns below one year ---


def test_annualized_returns_are_suppressed_below_one_year(strategies: pd.DataFrame) -> None:
    short = strategies.iloc[: N - 1]

    summary = performance_summary(short, periods_per_year=N)
    metrics = summary.metrics

    assert metrics.loc[("return", "cagr")].isna().all()
    assert metrics.loc[("risk_adjusted", "calmar_ratio")].isna().all()
    assert metrics.loc[("return", "total_return")].notna().all()
    assert metrics.loc[("risk_adjusted", "sharpe_ratio")].notna().all()
    assert summary.short_columns == ("defensive", "aggressive")


def test_one_full_year_is_annualized(strategies: pd.DataFrame) -> None:
    year = strategies.iloc[:N]

    summary = performance_summary(year, periods_per_year=N)

    assert summary.metrics.loc[("return", "cagr"), "defensive"] == pytest.approx(
        cagr(year["defensive"], periods_per_year=N)
    )


def test_only_the_short_column_is_suppressed_over_own_periods(
    strategies: pd.DataFrame,
) -> None:
    late = strategies.copy()
    late.iloc[: 3 * N - 100, 1] = NAN

    summary = performance_summary(late, periods_per_year=N, common_period=False)

    assert not math.isnan(summary.metrics.loc[("return", "cagr"), "defensive"])
    assert math.isnan(summary.metrics.loc[("return", "cagr"), "aggressive"])
    assert summary.short_columns == ("aggressive",)


def test_excess_cagr_is_suppressed_for_short_strategies(market: pd.Series) -> None:
    window = market.iloc[LATE_START : LATE_START + 100]
    strategy = (window * 0.5).rename("half")

    summary = performance_summary(strategy, window, periods_per_year=N)

    assert math.isnan(summary.metrics.loc[("benchmark", "excess_cagr"), "half"])
    assert not math.isnan(summary.metrics.loc[("benchmark", "beta"), "half"])


# --- Text rendering ---


def test_text_shows_period_conventions_and_raw_values(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    summary = performance_summary(strategies, market, 0.10, periods_per_year=N)

    text = str(summary)

    assert text.startswith("Performance summary")
    assert f"Period: {summary.start.date()} -> {summary.end.date()}" in text
    assert "risk-free 0.1000 annual" in text
    assert "VaR and CVaR at 95%" in text
    assert "sharpe_ratio" in text
    assert "n/a" in text
    assert "GIPS" not in text


def test_text_explains_suppressed_annualized_returns(strategies: pd.DataFrame) -> None:
    text = performance_summary(strategies.iloc[:100], periods_per_year=N).to_text()

    assert "Less than one year of data for defensive, aggressive" in text
    assert "(GIPS)" in text


# --- Performance errors ---


def test_benchmark_name_clashing_with_a_strategy_is_rejected(
    strategies: pd.DataFrame, market: pd.Series
) -> None:
    with pytest.raises(ValueError, match="clashes"):
        performance_summary(strategies, market.rename("defensive"), periods_per_year=N)


def test_series_without_common_period_are_rejected() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {"early": [0.01, 0.02, NAN, NAN], "late": [NAN, NAN, 0.01, -0.01]}, index=index
    )

    with pytest.raises(ValueError, match="no period in common"):
        performance_summary(returns, periods_per_year=N)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_invalid_confidence_is_rejected(strategies: pd.DataFrame, confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        performance_summary(strategies, periods_per_year=N, confidence=confidence)


def test_empty_returns_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        performance_summary(
            pd.Series([], index=pd.DatetimeIndex([]), dtype="float64"), periods_per_year=N
        )


# --- Trade table ---


@pytest.fixture
def trade_results() -> pd.Series:
    return pd.Series([0.02, -0.01, 0.0, 0.03, -0.02], name="return")


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2024-01-02", "2024-01-04", "2024-01-08", "2024-01-09", "2024-01-11"]
            ),
            "exit_time": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-08", "2024-01-10", "2024-01-12"]
            ),
        }
    )


SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=10)


def test_trade_table_without_timing(trade_results: pd.Series) -> None:
    summary = trade_summary(trade_results)

    assert list(summary.metrics.index.get_level_values("section").unique()) == ["outcome"]
    assert summary.metrics[("outcome", "trade_count")] == 5.0
    assert summary.metrics[("outcome", "expectancy")] == pytest.approx(expectancy(trade_results))
    assert summary.metrics[("outcome", "profit_factor")] == pytest.approx(
        profit_factor(trade_results)
    )


def test_trade_table_with_timing(trade_results: pd.Series, trades: pd.DataFrame) -> None:
    summary = trade_summary(trade_results, trades, SESSIONS)

    assert summary.metrics[("timing", "exposure")] == pytest.approx(exposure(trades, SESSIONS))
    assert summary.metrics[("timing", "trade_frequency")] == pytest.approx(5 / 10)
    assert summary.annualized_frequency is False


def test_annualized_trade_frequency(trade_results: pd.Series, trades: pd.DataFrame) -> None:
    summary = trade_summary(trade_results, trades, SESSIONS, annualized=True, periods_per_year=N)

    assert summary.metrics[("timing", "trade_frequency")] == pytest.approx(5 / 10 * N)
    assert "per year" in str(summary)


def test_trade_table_without_trades() -> None:
    summary = trade_summary(pd.Series([], dtype="float64"))

    assert summary.metrics[("outcome", "trade_count")] == 0.0
    assert math.isnan(summary.metrics[("outcome", "expectancy")])
    assert "n/a" in summary.to_text()


def test_trades_and_sessions_must_come_together(
    trade_results: pd.Series, trades: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="together"):
        trade_summary(trade_results, trades)
    with pytest.raises(ValueError, match="together"):
        trade_summary(trade_results, sessions=SESSIONS)


def test_trade_text_rendering(trade_results: pd.Series) -> None:
    text = str(trade_summary(trade_results))

    assert text.startswith("Trade summary")
    assert "per bar" in text
    assert "profit_factor" in text


# --- Units ---


def test_every_row_has_a_unit(
    strategies: pd.DataFrame,
    market: pd.Series,
    trade_results: pd.Series,
    trades: pd.DataFrame,
) -> None:
    performance = performance_summary(strategies, market, periods_per_year=N).metrics
    trade = trade_summary(trade_results, trades, SESSIONS).metrics
    rows = {*performance.index.get_level_values("metric"), *trade.index.get_level_values("metric")}

    assert rows == set(METRIC_UNITS)
