# ibexQuant\tests\analytics\test_returns.py

"""Tests for :mod:`ibexQuant.analytics.returns`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import empyrical
import numpy as np
import pandas as pd
import pytest

from ibexQuant.analytics import (
    cumulative_returns,
    drawdown,
    drawdown_periods,
    equity_curve,
    excess_returns,
    monthly_returns_table,
    period_returns,
    returns_from_equity,
)

NAN: Final[float] = np.nan
SESSIONS_PER_YEAR: Final[int] = 252

# Eight sessions across a month boundary, with three drawdown episodes: one
# opening the series, one recovered, and one still open at the end.
SESSIONS: Final[pd.DatetimeIndex] = pd.bdate_range("2024-01-29", periods=8)
RETURNS: Final[list[float]] = [-0.05, 0.02, 0.04, -0.10, 0.05, 0.10, -0.01, 0.0]
EQUITY: Final[list[float]] = [
    0.95,
    0.95 * 1.02,
    0.95 * 1.02 * 1.04,
    0.95 * 1.02 * 1.04 * 0.90,
    0.95 * 1.02 * 1.04 * 0.90 * 1.05,
    0.95 * 1.02 * 1.04 * 0.90 * 1.05 * 1.10,
    0.95 * 1.02 * 1.04 * 0.90 * 1.05 * 1.10 * 0.99,
    0.95 * 1.02 * 1.04 * 0.90 * 1.05 * 1.10 * 0.99,
]

type SeriesTransform = Callable[[pd.Series], pd.Series]

SHAPE_PRESERVING: Final[dict[str, SeriesTransform]] = {
    "equity_curve": equity_curve,
    "cumulative_returns": cumulative_returns,
    "drawdown": drawdown,
    "period_returns": lambda returns: period_returns(returns, "month"),
    "excess_returns": lambda returns: excess_returns(
        returns, 0.10, periods_per_year=SESSIONS_PER_YEAR
    ),
}


def make_returns(values: list[float] = RETURNS, name: str = "strategy") -> pd.Series:
    return pd.Series(values, index=SESSIONS, name=name, dtype="float64")


@pytest.fixture
def random_returns() -> pd.DataFrame:
    """Three years of daily returns for two series; the second starts a year late."""
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2021-01-01", "2023-12-31")
    values = rng.normal(0.0004, 0.012, size=(len(index), 2))
    returns = pd.DataFrame(values, index=index, columns=["strategy", "benchmark"])
    returns.loc[:"2021-12-31", "benchmark"] = NAN
    return returns


# --- Compounding ---


def test_equity_curve_compounds_returns() -> None:
    result = equity_curve(make_returns())

    np.testing.assert_allclose(result.to_numpy(), EQUITY)


def test_equity_curve_scales_with_initial_value() -> None:
    result = equity_curve(make_returns(), initial_value=100_000.0)

    np.testing.assert_allclose(result.to_numpy(), np.array(EQUITY) * 100_000.0)


@pytest.mark.parametrize("initial_value", [0.0, -1.0, np.nan])
def test_equity_curve_rejects_non_positive_initial_value(initial_value: float) -> None:
    with pytest.raises(ValueError, match="initial_value"):
        equity_curve(make_returns(), initial_value=initial_value)


def test_cumulative_returns_are_equity_minus_one() -> None:
    result = cumulative_returns(make_returns())

    np.testing.assert_allclose(result.to_numpy(), np.array(EQUITY) - 1.0)


def test_cumulative_returns_match_empyrical(random_returns: pd.DataFrame) -> None:
    strategy = random_returns["strategy"]

    np.testing.assert_allclose(
        cumulative_returns(strategy).to_numpy(), empyrical.cum_returns(strategy).to_numpy()
    )


def test_returns_from_equity_inverts_equity_curve(random_returns: pd.DataFrame) -> None:
    recovered = returns_from_equity(equity_curve(random_returns, initial_value=50.0))

    # The first observation of each column has no previous value; every other
    # return is recovered exactly.
    expected = random_returns.copy()
    expected.iloc[0, 0] = NAN
    expected.loc[pd.Timestamp("2022-01-03"), "benchmark"] = NAN
    pd.testing.assert_frame_equal(recovered, expected, rtol=1e-12)


def test_returns_from_equity_computes_ratios() -> None:
    equity = pd.Series([100.0, 110.0, 99.0], index=SESSIONS[:3])

    np.testing.assert_allclose(returns_from_equity(equity).to_numpy(), [NAN, 0.10, -0.10])


def test_returns_from_equity_rejects_non_positive_equity() -> None:
    equity = pd.Series([100.0, 0.0, 99.0], index=SESSIONS[:3])

    with pytest.raises(ValueError, match="strictly positive"):
        returns_from_equity(equity)


# --- Drawdown ---


def test_drawdown_measures_distance_from_running_peak() -> None:
    result = drawdown(make_returns())

    expected = [-0.05, -0.031, 0.0, -0.10, EQUITY[4] / EQUITY[2] - 1.0, 0.0, -0.01, -0.01]
    np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-15)


def test_drawdown_counts_a_first_period_loss() -> None:
    # The initial capital is the first peak; a naive equity / equity.cummax()
    # would report zero here.
    result = drawdown(make_returns([-0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))

    assert result.iloc[0] == pytest.approx(-0.05)


def test_drawdown_stays_at_minus_one_after_total_loss() -> None:
    result = drawdown(make_returns([0.10, -1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(result.to_numpy()[1:], -1.0)


def test_drawdown_starts_each_column_at_its_own_initial_capital() -> None:
    returns = pd.DataFrame(
        {"strategy": make_returns().to_numpy(), "late": [NAN, NAN, -0.05, *RETURNS[3:]]},
        index=SESSIONS,
    )

    result = drawdown(returns)["late"]

    assert np.isnan(result.iloc[:2]).all()
    assert result.iloc[2] == pytest.approx(-0.05)


def test_maximum_drawdown_matches_empyrical(random_returns: pd.DataFrame) -> None:
    strategy = random_returns["strategy"]

    assert drawdown(strategy).min() == pytest.approx(empyrical.max_drawdown(strategy))


# --- Drawdown periods ---


def test_drawdown_periods_lists_every_episode() -> None:
    result = drawdown_periods(make_returns())

    expected = pd.DataFrame(
        {
            "peak": SESSIONS[[0, 2, 5]],
            "trough": SESSIONS[[0, 3, 6]],
            "recovery": pd.DatetimeIndex([SESSIONS[2], SESSIONS[5], pd.NaT]),
            "depth": [-0.05, -0.10, -0.01],
            "peak_to_trough": pd.array([0, 1, 1], dtype="Int64"),
            "trough_to_recovery": pd.array([2, 2, pd.NA], dtype="Int64"),
            "duration": pd.array([2, 3, 2], dtype="Int64"),
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_index_type=False)


def test_drawdown_periods_is_empty_without_drawdowns() -> None:
    result = drawdown_periods(make_returns([0.01] * 8))

    assert result.empty
    assert list(result.columns) == [
        "peak",
        "trough",
        "recovery",
        "depth",
        "peak_to_trough",
        "trough_to_recovery",
        "duration",
    ]


def test_drawdown_periods_ignores_edge_gaps() -> None:
    padded = make_returns([NAN, *RETURNS[1:7], NAN])

    result = drawdown_periods(padded)

    assert result["peak"].iloc[0] == SESSIONS[2]
    assert result["trough"].iloc[0] == SESSIONS[3]


def test_drawdown_periods_deepest_matches_drawdown_minimum(random_returns: pd.DataFrame) -> None:
    strategy = random_returns["strategy"]

    result = drawdown_periods(strategy)

    assert result["depth"].min() == pytest.approx(drawdown(strategy).min())
    assert (result["duration"] >= result["peak_to_trough"]).all()


def test_drawdown_periods_rejects_dataframes() -> None:
    with pytest.raises(TypeError, match="pd.Series"):
        drawdown_periods(make_returns().to_frame())  # type: ignore[arg-type]


# --- Period returns ---


def test_period_returns_compound_within_each_month() -> None:
    result = period_returns(make_returns(), "month")

    # Labeled by calendar period end, not by the last session (2024-02-07).
    assert list(result.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    np.testing.assert_allclose(
        result.to_numpy(),
        [0.95 * 1.02 * 1.04 - 1.0, 0.90 * 1.05 * 1.10 * 0.99 * 1.0 - 1.0],
    )


@pytest.mark.parametrize("period", ["month", "quarter", "year"])
def test_period_returns_compound_back_to_the_total(
    random_returns: pd.DataFrame, period: str
) -> None:
    result = period_returns(random_returns, period)  # type: ignore[arg-type]

    np.testing.assert_allclose(
        cumulative_returns(result).iloc[-1].to_numpy(),
        cumulative_returns(random_returns).iloc[-1].to_numpy(),
    )


def test_period_returns_are_nan_where_a_column_has_no_data(random_returns: pd.DataFrame) -> None:
    result = period_returns(random_returns, "year")

    # The benchmark has no 2021 data: NaN, never an invented 0%.
    assert np.isnan(result["benchmark"].iloc[0])
    assert not np.isnan(result["benchmark"].iloc[1:]).any()


def test_period_returns_mark_periods_without_rows_as_nan() -> None:
    # No row at all in February: the month exists in the result, as NaN.
    index = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-03-01", "2024-03-04"])
    returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=index)

    result = period_returns(returns, "month")

    assert list(result.index.month) == [1, 2, 3]
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(1.03 * 1.04 - 1.0)


def test_period_returns_rejects_unknown_periods() -> None:
    with pytest.raises(ValueError, match="period must be one of"):
        period_returns(make_returns(), "week")  # type: ignore[arg-type]


def test_period_returns_of_empty_input_are_empty() -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype="float64", name="strategy")

    result = period_returns(empty, "month")

    assert result.empty
    assert result.name == "strategy"


# --- Monthly returns table ---


def test_monthly_returns_table_places_months_and_totals() -> None:
    result = monthly_returns_table(make_returns())

    assert list(result.index) == [2024]
    assert result.index.name == "year"
    assert list(result.columns) == [*range(1, 13), "total"]
    assert result.loc[2024, 1] == pytest.approx(0.95 * 1.02 * 1.04 - 1.0)
    assert result.loc[2024, "total"] == pytest.approx(EQUITY[-1] - 1.0)
    assert result.loc[2024, 3:12].isna().all()


def test_monthly_returns_table_rows_compound_to_the_total(random_returns: pd.DataFrame) -> None:
    result = monthly_returns_table(random_returns["strategy"])

    assert list(result.index) == [2021, 2022, 2023]
    compounded = (1.0 + result.loc[:, 1:12]).prod(axis=1) - 1.0
    np.testing.assert_allclose(compounded.to_numpy(), result["total"].to_numpy())


def test_monthly_returns_table_rejects_dataframes() -> None:
    with pytest.raises(TypeError, match="pd.Series"):
        monthly_returns_table(make_returns().to_frame())  # type: ignore[arg-type]


# --- Excess returns ---


def test_excess_returns_with_zero_rate_are_the_returns() -> None:
    returns = make_returns()

    pd.testing.assert_series_equal(
        excess_returns(returns, periods_per_year=SESSIONS_PER_YEAR), returns
    )


def test_annual_rate_is_converted_geometrically() -> None:
    returns = make_returns([0.0] * 8)

    periodic = -excess_returns(returns, 0.10, periods_per_year=SESSIONS_PER_YEAR).iloc[0]

    # Compounding the periodic rate over a year gives back the annual rate.
    assert (1.0 + periodic) ** SESSIONS_PER_YEAR - 1.0 == pytest.approx(0.10)


def test_risk_free_series_is_subtracted_per_period_from_every_column() -> None:
    returns = pd.DataFrame({"strategy": RETURNS, "benchmark": [0.0] * 8}, index=SESSIONS)
    risk_free = pd.Series([0.001] * 8, index=SESSIONS)

    result = excess_returns(returns, risk_free, periods_per_year=SESSIONS_PER_YEAR)

    np.testing.assert_allclose(result.to_numpy(), returns.to_numpy() - 0.001)


def test_risk_free_series_may_be_missing_where_returns_are_not_observed() -> None:
    returns = make_returns([NAN, *RETURNS[1:]])
    risk_free = pd.Series([0.001] * 7, index=SESSIONS[1:])

    result = excess_returns(returns, risk_free, periods_per_year=SESSIONS_PER_YEAR)

    assert np.isnan(result.iloc[0])
    np.testing.assert_allclose(result.iloc[1:].to_numpy(), np.array(RETURNS[1:]) - 0.001)


def test_risk_free_series_must_cover_observed_returns() -> None:
    risk_free = pd.Series([0.001] * 6, index=SESSIONS[:6])

    with pytest.raises(ValueError, match=r"missing at 2024-02-06.* and 1 other"):
        excess_returns(make_returns(), risk_free, periods_per_year=SESSIONS_PER_YEAR)


@pytest.mark.parametrize("periods_per_year", [0, 252.0])
def test_excess_returns_reject_invalid_periods_per_year(periods_per_year: object) -> None:
    with pytest.raises((TypeError, ValueError), match="periods_per_year"):
        excess_returns(make_returns(), 0.10, periods_per_year=periods_per_year)  # type: ignore[arg-type]


def test_annual_rate_must_be_above_total_loss() -> None:
    with pytest.raises(ValueError, match="risk_free"):
        excess_returns(make_returns(), -1.0, periods_per_year=SESSIONS_PER_YEAR)


# --- Shape contract ---


@pytest.mark.parametrize("transform", SHAPE_PRESERVING.values(), ids=SHAPE_PRESERVING.keys())
def test_series_in_gives_named_series_out(transform: SeriesTransform) -> None:
    result = transform(make_returns())

    assert isinstance(result, pd.Series)
    assert result.name == "strategy"


@pytest.mark.parametrize("transform", SHAPE_PRESERVING.values(), ids=SHAPE_PRESERVING.keys())
def test_each_dataframe_column_matches_its_series(
    transform: SeriesTransform, random_returns: pd.DataFrame
) -> None:
    frame_result = transform(random_returns)  # type: ignore[arg-type]

    assert isinstance(frame_result, pd.DataFrame)
    for column in random_returns.columns:
        pd.testing.assert_series_equal(frame_result[column], transform(random_returns[column]))


@pytest.mark.parametrize("transform", SHAPE_PRESERVING.values(), ids=SHAPE_PRESERVING.keys())
def test_inputs_are_not_modified(transform: SeriesTransform, random_returns: pd.DataFrame) -> None:
    original = random_returns.copy()

    transform(random_returns)  # type: ignore[arg-type]

    pd.testing.assert_frame_equal(random_returns, original)
