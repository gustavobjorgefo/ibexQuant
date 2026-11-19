# ibexQuant\tests\backtest\test_daytrade.py

"""E1 day-trade engine (ADR 0017, B1.17-B1.21)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibexQuant.backtest import DayTradeOrders, DayTradeResult, simulate_day_trades
from ibexQuant.data.panel import Panel

SESSIONS = pd.bdate_range("2025-01-06", periods=3)
NAN = np.nan


def bars(**bar: tuple[float, float, float, float]) -> Panel:
    """Security A: a flat bar on session 0 and session 2, ``bar`` (O, H, L, C) on session 1."""
    rows = [(10.0, 10.0, 10.0, 10.0), bar.get("A", (10.0, 10.0, 10.0, 10.0)), (10.0,) * 4]
    frames = {
        name: pd.DataFrame({"A": [row[i] for row in rows]}, index=SESSIONS, dtype=float)
        for i, name in enumerate(("open", "high", "low", "close"))
    }
    return Panel(frames)


def order(value: float) -> pd.DataFrame:
    """One order on A, decided at session 0 for session 1."""
    return pd.DataFrame({"A": [value]}, index=SESSIONS[:1])


def one_trade(result: DayTradeResult) -> pd.Series:
    assert len(result.trades) == 1
    trade: pd.Series = result.trades.iloc[0]
    return trade


# --- fills ---


@pytest.mark.parametrize(
    ("bar", "filled", "entry"),
    [
        ((10.5, 11.0, 10.0, 10.8), False, NAN),  # the low touches the limit: no fill
        ((10.5, 11.0, 9.9, 10.8), True, 10.0),  # traded through it: at the limit
        ((9.5, 10.0, 9.3, 9.8), True, 9.5),  # the auction crossed it: at the open
    ],
)
def test_a_buy_limit_fills_strictly_below(
    bar: tuple[float, float, float, float], filled: bool, entry: float
) -> None:
    result = simulate_day_trades(bars(A=bar), DayTradeOrders(size=0.5, buy_limit=order(10.0)))
    assert len(result.trades) == int(filled)
    if filled:
        trade = one_trade(result)
        assert (trade["side"], trade["entry_price"], trade["exit_reason"]) == (
            "long",
            entry,
            "close",
        )
        assert trade["return"] == pytest.approx(bar[3] / entry - 1)
        assert result.returns.iloc[0] == pytest.approx(0.5 * (bar[3] / entry - 1))
    else:
        assert result.returns.iloc[0] == 0


def test_a_sell_limit_is_the_mirror() -> None:
    touched = simulate_day_trades(
        bars(A=(10.5, 11.0, 10.2, 10.8)), DayTradeOrders(size=1.0, sell_limit=order(11.0))
    )
    assert touched.trades.empty
    gapped = simulate_day_trades(
        bars(A=(11.5, 11.6, 10.6, 10.8)), DayTradeOrders(size=1.0, sell_limit=order(11.0))
    )
    trade = one_trade(gapped)
    assert (trade["side"], trade["entry_price"]) == ("short", 11.5)
    assert trade["return"] == pytest.approx(1 - 10.8 / 11.5)


def test_both_sides_fill_on_one_session() -> None:
    orders = DayTradeOrders(size=0.5, buy_limit=order(10.0), sell_limit=order(11.0))
    result = simulate_day_trades(bars(A=(10.5, 11.5, 9.5, 10.5)), orders)
    assert result.trades["side"].tolist() == ["long", "short"]
    # Long 10 -> 10.5 and short 11 -> 10.5: the spread between the limits, less the drift.
    assert result.returns.iloc[0] == pytest.approx(0.5 * (0.05 + 0.5 / 11))


def test_side_opens_at_the_open() -> None:
    long = simulate_day_trades(bars(A=(10.0, 11.0, 9.0, 10.5)), DayTradeOrders(1.0, side=order(1)))
    assert one_trade(long)["return"] == pytest.approx(0.05)
    short = simulate_day_trades(
        bars(A=(10.0, 11.0, 9.0, 10.5)), DayTradeOrders(1.0, side=order(-1))
    )
    assert one_trade(short)["return"] == pytest.approx(-0.05)
    nothing = simulate_day_trades(
        bars(A=(10.0, 11.0, 9.0, 10.5)), DayTradeOrders(1.0, side=order(0))
    )
    assert nothing.trades.empty


# --- stops ---


def test_a_long_stop_is_hit_when_the_low_reaches_it() -> None:
    orders = DayTradeOrders(size=1.0, buy_limit=order(10.0), buy_stop=order(9.5))
    touched = simulate_day_trades(bars(A=(10.5, 11.0, 9.5, 10.8)), orders)
    trade = one_trade(touched)
    assert (trade["exit_price"], trade["exit_reason"]) == (9.5, "stop")
    assert trade["return"] == pytest.approx(-0.05)
    above = simulate_day_trades(bars(A=(10.5, 11.0, 9.6, 10.8)), orders)
    assert one_trade(above)["exit_reason"] == "close"


def test_an_auction_through_the_stop_exits_at_the_open() -> None:
    orders = DayTradeOrders(size=1.0, side=order(1), buy_stop=order(9.5))
    trade = one_trade(simulate_day_trades(bars(A=(9.0, 9.8, 8.8, 9.6)), orders))
    assert (trade["entry_price"], trade["exit_price"], trade["exit_reason"]) == (9.0, 9.0, "stop")
    assert trade["return"] == 0


def test_a_short_stop() -> None:
    orders = DayTradeOrders(size=1.0, sell_limit=order(11.0), sell_stop=order(11.5))
    trade = one_trade(simulate_day_trades(bars(A=(10.5, 11.6, 10.4, 10.8)), orders))
    assert (trade["exit_price"], trade["exit_reason"]) == (11.5, "stop")
    assert trade["return"] == pytest.approx(1 - 11.5 / 11)
    gapped = DayTradeOrders(size=1.0, side=order(-1), sell_stop=order(11.0))
    trade = one_trade(simulate_day_trades(bars(A=(11.2, 11.6, 10.4, 10.8)), gapped))
    assert (trade["exit_price"], trade["return"]) == (11.2, 0)


# --- what does not trade ---


def test_no_complete_bar_no_trade() -> None:
    result = simulate_day_trades(
        bars(A=(10.5, NAN, 9.0, 10.8)), DayTradeOrders(size=1.0, buy_limit=order(10.0))
    )
    assert result.trades.empty
    assert result.returns.tolist() == [0.0, 0.0]


def test_size_by_frame_and_zero() -> None:
    bar = bars(A=(10.0, 11.0, 9.0, 10.5))
    framed = simulate_day_trades(bar, DayTradeOrders(size=order(0.25), side=order(1)))
    assert framed.returns.iloc[0] == pytest.approx(0.25 * 0.05)
    assert one_trade(framed)["size"] == 0.25
    assert simulate_day_trades(bar, DayTradeOrders(size=order(0.0), side=order(1))).trades.empty


# --- costs and result ---


def test_costs_are_paid_on_both_sides() -> None:
    orders = DayTradeOrders(size=0.5, buy_limit=order(10.0))
    result = simulate_day_trades(bars(A=(10.5, 11.0, 9.9, 10.8)), orders, cost_bps=10)
    fees = 0.001 * (1 + 10.8 / 10)  # the entry notional and the exit notional
    assert one_trade(result)["return"] == pytest.approx(0.08 - fees)
    assert result.costs.iloc[0] == pytest.approx(0.5 * fees)
    assert result.returns.iloc[0] == pytest.approx(0.5 * (0.08 - fees))


def test_result() -> None:
    result = simulate_day_trades(
        bars(A=(10.0, 11.0, 9.0, 10.5)), DayTradeOrders(size=1.0, side=order(1))
    )
    assert result.returns.index.equals(SESSIONS[1:])
    assert result.returns.name == "portfolio"
    pd.testing.assert_series_equal(
        result.contributions.sum(axis=1), result.returns, check_names=False
    )
    assert result.equity.tolist() == pytest.approx([1.05, 1.05])
    assert result.trades.columns.tolist() == [
        "security_id",
        "side",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "exit_reason",
        "size",
        "return",
    ]
    trades = result.trade_summary().metrics
    assert trades.loc[("outcome", "trade_count")] == 1
    benchmark = pd.Series(0.01, index=SESSIONS)
    summary = result.summary(periods_per_year=252, benchmark=benchmark)
    assert summary.metrics.loc[("return", "total_return"), "portfolio"] == pytest.approx(0.05)


# --- validation ---


@pytest.mark.parametrize(
    ("orders", "arguments", "message"),
    [
        (DayTradeOrders(1.0, side=order(1)), {"cost_bps": -1}, "cost_bps must be zero"),
        (DayTradeOrders(1.0), {}, "limits or side, not both and not neither"),
        (
            DayTradeOrders(1.0, buy_limit=order(10.0), side=order(1)),
            {},
            "limits or side, not both",
        ),
        (DayTradeOrders(-1.0, side=order(1)), {}, "size must be finite and zero or positive"),
        (DayTradeOrders(1.0, side=pd.DataFrame(columns=["A"])), {}, "side has no decision"),
        (DayTradeOrders(order(np.inf), side=order(1)), {}, "size must be finite"),
        (
            DayTradeOrders(1.0, buy_limit=order(10.0), buy_stop=order(10.0)),
            {},
            "A stop must be below its buy limit",
        ),
        (
            DayTradeOrders(1.0, sell_limit=order(11.0), sell_stop=order(10.0)),
            {},
            "above its sell limit",
        ),
        (
            DayTradeOrders(1.0, side=pd.DataFrame({"A": [1]}, index=SESSIONS[2:])),
            {},
            "is the last session of bars",
        ),
        (
            DayTradeOrders(1.0, side=pd.DataFrame({"B": [1]}, index=SESSIONS[:1])),
            {},
            r"side names securities that are not in bars: \['B'\]",
        ),
    ],
)
def test_arguments_are_checked(
    orders: DayTradeOrders, arguments: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_day_trades(bars(), orders, **arguments)


def test_bars_need_every_field() -> None:
    closes = Panel({"close": pd.DataFrame({"A": [10.0] * 3}, index=SESSIONS)})
    with pytest.raises(ValueError, match=r"missing \['open', 'high', 'low'\]"):
        simulate_day_trades(closes, DayTradeOrders(1.0, side=order(1)))
