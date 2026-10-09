# ibexQuant\tests\analytics\test_trades.py

"""Tests for :mod:`ibexQuant.analytics.trades`."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.analytics import (
    average_holding_period,
    average_loss,
    average_win,
    best_trade,
    expectancy,
    expectancy_t_stat,
    exposure,
    gross_loss,
    gross_profit,
    loss_rate,
    max_consecutive_losses,
    max_consecutive_wins,
    payoff_ratio,
    profit_factor,
    trade_count,
    trade_frequency,
    trade_standard_deviation,
    win_rate,
    worst_trade,
)

# Eight trades: three wins, four losses and one flat. In order:
# win, loss, flat, loss, win, loss, loss, win.
RESULTS: Final[list[float]] = [0.02, -0.01, 0.0, -0.005, 0.03, -0.01, -0.02, 0.01]

# Twenty sessions, 2024-01-01 to 2024-01-26.
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-01", periods=20)

type OutcomeMetric = Callable[[pd.Series], float]

UNDEFINED_WITHOUT_TRADES: Final[dict[str, OutcomeMetric]] = {
    "win_rate": win_rate,
    "loss_rate": loss_rate,
    "expectancy": expectancy,
    "trade_standard_deviation": trade_standard_deviation,
    "expectancy_t_stat": expectancy_t_stat,
    "average_win": average_win,
    "average_loss": average_loss,
    "payoff_ratio": payoff_ratio,
    "profit_factor": profit_factor,
    "best_trade": best_trade,
    "worst_trade": worst_trade,
}


def make_results(values: list[float] = RESULTS) -> pd.Series:
    return pd.Series(values, dtype="float64", name="return")


def make_trades(*spans: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime([entry for entry, _ in spans]),
            "exit_time": pd.to_datetime([exit_ for _, exit_ in spans]),
        }
    )


@pytest.fixture
def swing_trades() -> pd.DataFrame:
    """Three trades held for 4, 3 and 5 sessions: 12 sessions in position."""
    return make_trades(
        ("2024-01-02", "2024-01-05"),
        ("2024-01-09", "2024-01-11"),
        ("2024-01-15", "2024-01-19"),
    )


# --- Counts and rates ---


def test_counts_and_rates_leave_flat_trades_out() -> None:
    results = make_results()

    assert trade_count(results) == 8
    assert win_rate(results) == pytest.approx(3 / 8)
    assert loss_rate(results) == pytest.approx(4 / 8)


# --- Expectancy ---


def test_expectancy_is_the_mean_result() -> None:
    assert expectancy(make_results()) == pytest.approx(0.015 / 8)


def test_expectancy_decomposes_into_wins_and_losses() -> None:
    results = make_results()

    decomposed = win_rate(results) * average_win(results) + loss_rate(results) * average_loss(
        results
    )

    assert expectancy(results) == pytest.approx(decomposed)


def test_trade_standard_deviation_is_the_sample_estimate() -> None:
    results = make_results()

    assert trade_standard_deviation(results) == pytest.approx(results.std(ddof=1))


def test_expectancy_t_stat_scales_with_the_number_of_trades() -> None:
    results = make_results()
    expected = results.mean() / results.std(ddof=1) * math.sqrt(8)

    assert expectancy_t_stat(results) == pytest.approx(expected)
    # The same results repeated: same mean and almost the same deviation,
    # but twice the evidence.
    doubled = make_results(RESULTS * 2)
    assert expectancy_t_stat(doubled) > expectancy_t_stat(results)


@pytest.mark.parametrize("values", [[0.01], [0.01, 0.01, 0.01]], ids=["single", "identical"])
def test_expectancy_t_stat_is_nan_without_dispersion(values: list[float]) -> None:
    assert math.isnan(expectancy_t_stat(make_results(values)))


# --- Wins and losses ---


def test_averages_of_wins_and_losses() -> None:
    results = make_results()

    assert average_win(results) == pytest.approx(0.06 / 3)
    assert average_loss(results) == pytest.approx(-0.045 / 4)
    assert payoff_ratio(results) == pytest.approx((0.06 / 3) / (0.045 / 4))


def test_gross_profit_loss_and_profit_factor() -> None:
    results = make_results()

    assert gross_profit(results) == pytest.approx(0.06)
    assert gross_loss(results) == pytest.approx(-0.045)
    assert profit_factor(results) == pytest.approx(0.06 / 0.045)


def test_profit_factor_tells_apart_equal_net_results() -> None:
    # Both net +25%; the second earns it with far smaller losses.
    volatile = make_results([0.40, 0.40, -0.30, -0.25])
    steady = make_results([0.15, 0.15, -0.05])

    assert profit_factor(volatile) == pytest.approx(0.80 / 0.55)
    assert profit_factor(steady) == pytest.approx(6.0)


def test_best_and_worst_trade() -> None:
    results = make_results()

    assert best_trade(results) == 0.03
    assert worst_trade(results) == -0.02


def test_metrics_report_in_the_unit_received() -> None:
    pnl = make_results([150.0, -80.0, 230.0])

    assert expectancy(pnl) == pytest.approx(100.0)
    assert gross_loss(pnl) == -80.0
    assert profit_factor(pnl) == pytest.approx(380.0 / 80.0)


def test_ratios_are_nan_without_losses() -> None:
    winners = make_results([0.01, 0.02])

    assert gross_loss(winners) == 0.0
    assert math.isnan(average_loss(winners))
    assert math.isnan(payoff_ratio(winners))
    assert math.isnan(profit_factor(winners))


# --- Streaks ---


def test_longest_streaks() -> None:
    results = make_results()

    assert max_consecutive_wins(results) == 1
    assert max_consecutive_losses(results) == 2


def test_flat_trade_breaks_a_streak() -> None:
    # Three losses, a flat trade, one loss: runs of 3 and 1, not 4.
    results = make_results([-0.01, -0.01, -0.01, 0.0, -0.01, 0.02, 0.02])

    assert max_consecutive_losses(results) == 3
    assert max_consecutive_wins(results) == 2


def test_streaks_are_zero_without_wins_or_losses() -> None:
    assert max_consecutive_wins(make_results([-0.01, 0.0])) == 0
    assert max_consecutive_losses(make_results([0.01, 0.0])) == 0


# --- Without trades ---


def test_counts_and_sums_are_zero_without_trades() -> None:
    empty = make_results([])

    assert trade_count(empty) == 0
    assert gross_profit(empty) == 0.0
    assert gross_loss(empty) == 0.0
    assert max_consecutive_wins(empty) == 0
    assert max_consecutive_losses(empty) == 0


@pytest.mark.parametrize(
    "metric", UNDEFINED_WITHOUT_TRADES.values(), ids=UNDEFINED_WITHOUT_TRADES.keys()
)
def test_per_trade_metrics_are_nan_without_trades(metric: OutcomeMetric) -> None:
    assert math.isnan(metric(make_results([])))


# --- Timing ---


def test_swing_trades_timing(swing_trades: pd.DataFrame) -> None:
    assert trade_frequency(swing_trades, SESSIONS) == pytest.approx(3 / 20)
    assert exposure(swing_trades, SESSIONS) == pytest.approx(12 / 20)
    assert average_holding_period(swing_trades, SESSIONS) == pytest.approx(4.0)


def test_annualized_trade_frequency(swing_trades: pd.DataFrame) -> None:
    result = trade_frequency(swing_trades, SESSIONS, annualized=True, periods_per_year=252)

    assert result == pytest.approx(3 / 20 * 252)


def test_intraday_trades_last_one_session() -> None:
    day_trades = make_trades(
        ("2024-01-02 10:15", "2024-01-02 16:30"),
        ("2024-01-03 11:00", "2024-01-03 17:45"),
    )

    assert average_holding_period(day_trades, SESSIONS) == 1.0
    assert exposure(day_trades, SESSIONS) == pytest.approx(2 / 20)


def test_overlapping_trades_count_each_session_once() -> None:
    overlapping = make_trades(("2024-01-02", "2024-01-05"), ("2024-01-04", "2024-01-08"))

    # Sessions Jan 2, 3, 4, 5 and 8: five distinct sessions in position.
    assert exposure(overlapping, SESSIONS) == pytest.approx(5 / 20)


def test_timing_without_trades() -> None:
    no_trades = make_trades()

    assert trade_frequency(no_trades, SESSIONS) == 0.0
    assert exposure(no_trades, SESSIONS) == 0.0
    assert math.isnan(average_holding_period(no_trades, SESSIONS))


def test_extra_columns_in_trades_are_ignored(swing_trades: pd.DataFrame) -> None:
    enriched = swing_trades.assign(symbol="PETR4", side="long", pnl=[1.0, -2.0, 3.0])

    assert exposure(enriched, SESSIONS) == exposure(swing_trades, SESSIONS)


def test_trade_frequency_requires_matching_annualization(swing_trades: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="required when annualized"):
        trade_frequency(swing_trades, SESSIONS, annualized=True)
    with pytest.raises(ValueError, match="only applies when annualized"):
        trade_frequency(swing_trades, SESSIONS, periods_per_year=252)
    with pytest.raises(ValueError, match="periods_per_year"):
        trade_frequency(swing_trades, SESSIONS, annualized=True, periods_per_year=0)


def test_trades_outside_sessions_are_rejected() -> None:
    weekend = make_trades(("2024-01-06", "2024-01-08"))

    with pytest.raises(ValueError, match="outside"):
        exposure(weekend, SESSIONS)


def test_sessions_must_be_daily(swing_trades: pd.DataFrame) -> None:
    hourly = pd.date_range("2024-01-02 10:00", periods=8, freq="h")

    with pytest.raises(ValueError, match="one timestamp per day"):
        exposure(swing_trades, hourly)


# --- Validation of trade results ---


@pytest.mark.parametrize(
    ("values", "error", "match"),
    [
        ([0.01, np.nan], ValueError, "missing values"),
        ([0.01, np.inf], ValueError, "finite"),
    ],
    ids=["nan", "inf"],
)
def test_trade_results_must_be_complete_and_finite(
    values: list[float], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        expectancy(make_results(values))


def test_trade_results_must_be_a_numeric_series() -> None:
    with pytest.raises(TypeError, match="pd.Series"):
        expectancy(make_results().to_frame())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be numeric"):
        expectancy(pd.Series([True, False]))


def test_trade_results_accept_repeated_exit_times() -> None:
    # Two trades closed at the same instant (two symbols at the close).
    close = pd.Timestamp("2024-01-02 17:00")
    results = pd.Series([0.01, -0.02], index=[close, close])

    assert trade_count(results) == 2
