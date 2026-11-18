# ibexQuant\tests\backtest\test_vectorized.py

"""E1 position engine (ADR 0017, B1.9-B1.16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibexQuant.backtest import BacktestResult, equal_weight, simulate_weights
from ibexQuant.data.panel import Panel

SESSIONS = pd.bdate_range("2025-01-06", periods=6)
NAN = np.nan


def panel(close: dict[str, list[float]], open_: dict[str, list[float]] | None = None) -> Panel:
    closes = pd.DataFrame(close, index=SESSIONS, dtype=float)
    opens = closes if open_ is None else pd.DataFrame(open_, index=SESSIONS, dtype=float)
    return Panel({"open": opens, "close": closes})


def rows(*entries: tuple[int, dict[str, float]]) -> pd.DataFrame:
    """Weights by session position."""
    return pd.DataFrame([w for _, w in entries], index=SESSIONS[[i for i, _ in entries]])


def adds_up(result: BacktestResult) -> None:
    """Every fraction is over the previous equity: the pieces make the return."""
    total = result.contributions.sum(axis=1) - result.costs
    np.testing.assert_allclose(result.returns, total, atol=1e-15)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)


BARS = panel(
    {"A": [10, 11, 12, 12, 13, 13], "B": [20, 20, 18, 18, 18, 19]},
    {"A": [10, 10, 11, 12, 12, 14], "B": [20, 21, 19, 18, 18, 18]},
)


# --- execution ---


def test_next_close_buys_and_drifts() -> None:
    result = simulate_weights(BARS, rows((0, {"A": 0.5, "B": 0.5})), execution="next_close")
    a, b = 0.5 / 11, 0.5 / 20  # units bought at the close of session 1
    value = a * np.array([11, 12, 12, 13, 13]) + b * np.array([20, 18, 18, 18, 19])
    assert result.returns.index.equals(SESSIONS[1:])
    np.testing.assert_allclose(
        result.returns,
        np.diff(np.concatenate([[1.0], value])) / np.concatenate([[1.0], value[:-1]]),
    )
    # Drift: no rebalance after the first, so A's weight follows its price.
    np.testing.assert_allclose(result.weights["A"].iloc[-1], a * 13 / value[-1])
    assert result.weights["cash"].abs().max() < 1e-15
    assert result.turnover.tolist() == [1.0, 0, 0, 0, 0]
    adds_up(result)


def test_next_open_splits_the_session() -> None:
    weights = rows((0, {"A": 1.0}), (2, {"B": 1.0}))
    result = simulate_weights(BARS, weights)
    # Session 1: A bought at the open (10), held to the close (11).
    assert result.returns.iloc[0] == pytest.approx(11 / 10 - 1)
    # Session 3: A's overnight (12 -> 12) belongs to the old holding; B is bought at 18.
    equity_2 = 12 / 10
    assert result.returns.iloc[2] == pytest.approx((equity_2 * 12 / 12) / equity_2 - 1)
    assert result.weights.loc[SESSIONS[3], "B"] == pytest.approx(1.0)
    assert result.weights.loc[SESSIONS[3], "A"] == 0
    assert result.turnover.iloc[2] == pytest.approx(2.0)
    adds_up(result)


def test_a_missing_column_is_zero_and_shorts_fund_cash() -> None:
    result = simulate_weights(BARS, rows((0, {"A": -0.5})), execution="next_close")
    assert result.weights.loc[SESSIONS[1], "cash"] == pytest.approx(1.5)
    assert result.returns.iloc[1] == pytest.approx(-0.5 * (12 / 11 - 1))
    assert result.net_exposure.iloc[0] == pytest.approx(-0.5)
    assert result.gross_exposure.iloc[0] == pytest.approx(0.5)
    # The short grows with the price: -0.5 * 12/11 over an equity of 1.5 - 0.5 * 12/11.
    assert result.net_exposure.iloc[1] == pytest.approx(-6 / 10.5)
    adds_up(result)


# --- securities that do not trade ---


def test_a_suspended_security_keeps_its_units() -> None:
    bars = panel({"A": [10, 10, 10, 10, 10, 10], "B": [20, 20, 20, NAN, 22, 22]})
    weights = rows((0, {"B": 1.0}), (2, {"A": 1.0}))
    result = simulate_weights(bars, weights, execution="next_close")
    # B has no bar on session 3: it stays, and A is bought with the whole equity.
    assert result.weights.loc[SESSIONS[3], "B"] == pytest.approx(1.0)
    assert result.weights.loc[SESSIONS[3], "A"] == pytest.approx(1.0)
    assert result.weights.loc[SESSIONS[3], "cash"] == pytest.approx(-1.0)
    assert result.returns.iloc[3] == pytest.approx(0.1)  # B from 20 to 22, on equity 1
    adds_up(result)


def test_an_exit_is_cash_at_the_last_close() -> None:
    bars = panel({"A": [10, 10, 12, NAN, NAN, NAN], "B": [20] * 6})
    weights = rows((0, {"A": 1.0}), (3, {"A": 0.5, "B": 0.5}))
    result = simulate_weights(bars, weights, execution="next_close")
    assert result.weights.loc[SESSIONS[3], "cash"] == pytest.approx(1.0)
    assert result.equity.iloc[-1] == pytest.approx(1.2)
    # The exit is not a trade; the later target on A is zero.
    assert result.turnover.iloc[2] == 0
    assert result.turnover.iloc[3] == pytest.approx(0.5)
    assert result.weights.loc[SESSIONS[4], "A"] == 0
    adds_up(result)


def test_a_target_before_the_first_bar() -> None:
    bars = panel({"A": [10] * 6, "B": [NAN, NAN, NAN, 20, 20, 20]})
    with pytest.raises(ValueError, match=r"Target on \['B'\] before their first bar"):
        simulate_weights(bars, rows((1, {"B": 1.0})), execution="next_close")
    result = simulate_weights(bars, rows((2, {"B": 1.0})), execution="next_close")
    assert result.weights.loc[SESSIONS[3], "B"] == pytest.approx(1.0)


def test_no_open_is_no_trade() -> None:
    bars = panel({"A": [10] * 6}, {"A": [10, NAN, 10, 10, 10, 10]})
    result = simulate_weights(bars, rows((0, {"A": 1.0})))
    assert result.weights["cash"].iloc[0] == 1.0


# --- costs ---


def test_costs_are_paid_per_side() -> None:
    weights = rows((0, {"A": 1.0}), (2, {"B": 1.0}))
    result = simulate_weights(BARS, weights, execution="next_close", cost_bps=10)
    assert result.costs.iloc[0] == pytest.approx(0.001)
    # Selling A and buying B: two sides of about the equity each.
    assert result.costs.iloc[2] == pytest.approx(0.001 * result.turnover.iloc[2])
    assert result.returns.iloc[0] == pytest.approx(-0.001)
    adds_up(result)


# --- result ---


def test_result_derivations() -> None:
    result = simulate_weights(BARS, rows((0, {"A": 1.0})), execution="next_close")
    np.testing.assert_allclose(result.equity, [1.0, 12 / 11, 12 / 11, 13 / 11, 13 / 11])
    assert result.equity.name == "equity"
    assert result.returns.name == "portfolio"
    summary = result.summary(periods_per_year=252)
    total = summary.metrics.loc[("return", "total_return"), "portfolio"]
    assert total == pytest.approx(13 / 11 - 1)


# --- validation ---


@pytest.mark.parametrize(
    ("weights", "arguments", "message"),
    [
        (rows((0, {"A": 1.0})), {"execution": "close"}, "execution must be one of"),
        (rows((0, {"A": 1.0})), {"cost_bps": -1}, "cost_bps must be zero or positive"),
        (pd.DataFrame(), {}, "weights has no rebalance"),
        (rows((2, {"A": 1.0}), (1, {"A": 1.0})), {}, "strictly increasing"),
        (rows((0, {"C": 1.0})), {}, r"not in bars: \['C'\]"),
        (
            pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2025-01-04")]),
            {},
            "2025-01-04 00:00:00 is not a session of bars",
        ),
        (rows((5, {"A": 1.0})), {}, "2025-01-13 is the last session of bars"),
        (rows((0, {"A": np.inf})), {}, "weights must be finite"),
    ],
)
def test_arguments_are_checked(
    weights: pd.DataFrame, arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_weights(BARS, weights, **arguments)  # type: ignore[arg-type]


# --- equal_weight ---


def test_equal_weight() -> None:
    mask = pd.DataFrame({"A": [True, True, False], "B": [True, False, False]})
    weights = equal_weight(mask)
    assert weights.to_numpy().tolist() == [[0.5, 0.5], [1.0, 0.0], [0.0, 0.0]]
    with pytest.raises(TypeError, match="mask must be boolean"):
        equal_weight(mask.astype(int))
