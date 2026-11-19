# ibexQuant\src\ibexQuant\backtest\daytrade.py

"""
E1 day-trade engine: orders that open and close inside one daily bar (ADR 0017, B1.17-B1.21).

Responsibility
--------------
Turn orders decided at the close of ``d - 1`` into trades on the bar of ``d``, flat at its
close, and into the daily returns of an account that never carries a position.

Contract
--------
- Orders enter the book at the opening auction. A buy limit fills when the low is strictly
  below it, at the lower of the open and the limit; a sell limit is the mirror. With
  ``side``, positions open at the open.
- A long stop is hit when the low reaches it, a short stop when the high does; the exit is
  at the stop, or at the open when the auction went through it. The fill comes first.
  Every other position closes at the close.
- Sizes are fractions of the equity of the previous close; costs are paid per side on the
  traded notional. ``returns == contributions.sum(axis=1)``, costs included.
- An order on a security without a complete bar on ``d`` does nothing.

What does not belong here
-------------------------
- Positions carried across closes: the position engine (``simulate_weights``).
- Raw prices, ticks and lots: E2. Levels here are on the panel's adjusted scale (B1.20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from ibexQuant.analytics import (
    PerformanceSummary,
    TradeSummary,
    performance_summary,
    trade_summary,
)
from ibexQuant.backtest._alignment import execution_positions
from ibexQuant.data.panel import Panel

BASIS_POINT: Final[float] = 1e-4
PORTFOLIO: Final[str] = "portfolio"
BAR_FIELDS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "security_id",
    "side",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "exit_reason",
    "size",
    "return",
)


@dataclass(frozen=True)
class DayTradeOrders:
    """
    Orders for the session after each decision, one column per security.

    Attributes
    ----------
    size : float or pandas.DataFrame
        Notional of each order as a fraction of the equity of the previous close.
    buy_limit, sell_limit : pandas.DataFrame, optional
        Limit price to buy, and to sell short; ``NaN`` is no order.
    side : pandas.DataFrame, optional
        Instead of limits: +1 buys and -1 sells short at the open; 0 or ``NaN`` is no order.
    buy_stop, sell_stop : pandas.DataFrame, optional
        Stop of each side; ``NaN`` is no stop.
    """

    size: float | pd.DataFrame
    buy_limit: pd.DataFrame | None = None
    sell_limit: pd.DataFrame | None = None
    side: pd.DataFrame | None = None
    buy_stop: pd.DataFrame | None = None
    sell_stop: pd.DataFrame | None = None


@dataclass(frozen=True)
class DayTradeResult:
    """
    What a day-trade simulation produced, by session from the first execution.

    Attributes
    ----------
    returns : pandas.Series
        Profit and loss over the equity of the previous close, costs included.
    contributions : pandas.DataFrame
        The same, by security; its rows add up to ``returns``.
    costs : pandas.Series
        Costs over the equity of the previous close.
    trades : pandas.DataFrame
        One row per fill (:data:`TRADE_COLUMNS`), ``return`` net of costs.
    """

    returns: pd.Series
    contributions: pd.DataFrame
    costs: pd.Series
    trades: pd.DataFrame = field(repr=False)

    @property
    def equity(self) -> pd.Series:
        """Equity of the account, starting from 1 before the first execution."""
        return (1.0 + self.returns).cumprod().rename("equity")

    def summary(
        self,
        *,
        periods_per_year: int,
        benchmark: pd.Series | None = None,
        risk_free: float | pd.Series = 0.0,
    ) -> PerformanceSummary:
        """Return the performance table of ``returns`` (ADR 0009), benchmark aligned."""
        aligned = None if benchmark is None else benchmark.reindex(self.returns.index)
        return performance_summary(
            self.returns, aligned, risk_free, periods_per_year=periods_per_year
        )

    def trade_summary(self) -> TradeSummary:
        """Return the trade table of ADR 0009 from the trades' net returns."""
        sessions = pd.DatetimeIndex(self.returns.index)
        return trade_summary(self.trades["return"], self.trades, sessions)


def simulate_day_trades(
    bars: Panel, orders: DayTradeOrders, *, cost_bps: float = 0.0
) -> DayTradeResult:
    """
    Simulate orders that open and close inside the bar after their decision.

    Parameters
    ----------
    bars : Panel
        Bars with open, high, low and close on one scale (adjusted, B1.20).
    orders : DayTradeOrders
        Limits or sides, sizes and stops, indexed by decision session.
    cost_bps : float, default 0.0
        Cost per side, in basis points of the traded notional.

    Returns
    -------
    DayTradeResult

    Raises
    ------
    ValueError
        If ``bars`` lacks a field, the orders have both or neither of limits and sides, a
        size is negative or not finite, a stop is on the wrong side of its limit, a
        decision is not a session of ``bars`` (or is its last), or a column is not one of
        its securities.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be zero or positive, got {cost_bps}.")
    missing = [name for name in BAR_FIELDS if name not in bars.fields]
    if missing:
        raise ValueError(f"bars must have {list(BAR_FIELDS)}; missing {missing}.")
    has_limits = orders.buy_limit is not None or orders.sell_limit is not None
    if has_limits == (orders.side is not None):
        raise ValueError("Orders need limits or side, not both and not neither.")

    sessions, symbols = bars.index, list(bars.symbols)
    open_, high, low, close = (bars.field(name).to_numpy(dtype=float) for name in BAR_FIELDS)
    complete = ~(np.isnan(open_) | np.isnan(high) | np.isnan(low) | np.isnan(close))
    grid = _Grid(sessions, symbols)

    size = grid.size(orders.size)
    buy_stop, sell_stop = (
        grid.place(orders.buy_stop, "buy_stop"),
        grid.place(orders.sell_stop, "sell_stop"),
    )
    if orders.side is not None:
        side = np.nan_to_num(grid.place(orders.side, "side"))
        long_entry = np.where(complete & (side > 0), open_, np.nan)
        short_entry = np.where(complete & (side < 0), open_, np.nan)
    else:
        buy = grid.place(orders.buy_limit, "buy_limit")
        sell = grid.place(orders.sell_limit, "sell_limit")
        if (buy_stop >= buy).any() or (sell_stop <= sell).any():
            raise ValueError("A stop must be below its buy limit and above its sell limit.")
        # In the book from the auction: an open beyond the limit fills at the open.
        long_entry = np.where(complete & (low < buy), np.fmin(open_, buy), np.nan)
        short_entry = np.where(complete & (high > sell), np.fmax(open_, sell), np.nan)

    long_stopped = ~np.isnan(long_entry) & (low <= buy_stop)
    short_stopped = ~np.isnan(short_entry) & (high >= sell_stop)
    long_exit = np.where(long_stopped, np.fmin(open_, buy_stop), close)
    short_exit = np.where(short_stopped, np.fmax(open_, sell_stop), close)
    legs = (
        _Leg("long", long_entry, long_exit, long_stopped, long_exit / long_entry - 1.0),
        _Leg("short", short_entry, short_exit, short_stopped, 1.0 - short_exit / short_entry),
    )
    return _result(legs, size, cost_bps * BASIS_POINT, min(grid.first), sessions, symbols)


# --- helpers ---


class _Grid:
    """Order frames placed on the bars' grid, at the session of their execution."""

    def __init__(self, sessions: pd.DatetimeIndex, symbols: list[str]) -> None:
        self.sessions: pd.DatetimeIndex = sessions
        self.symbols: list[str] = symbols
        self.first: list[int] = []  # first execution of each frame placed

    def place(self, values: pd.DataFrame | None, name: str) -> np.ndarray:
        grid = np.full((len(self.sessions), len(self.symbols)), np.nan)
        if values is None:
            return grid
        positions = execution_positions(values, self.sessions, self.symbols, name)
        grid[positions] = values.reindex(columns=self.symbols).to_numpy(dtype=float)
        self.first.append(int(positions[0]))
        return grid

    def size(self, size: float | pd.DataFrame) -> np.ndarray:
        if isinstance(size, pd.DataFrame):
            values = self.place(size, "size")
        else:
            values = np.full((len(self.sessions), len(self.symbols)), float(size))
        if not (np.isnan(values) | (np.isfinite(values) & (values >= 0))).all():
            raise ValueError("size must be finite and zero or positive.")
        result: np.ndarray = np.nan_to_num(values)
        return result


@dataclass(frozen=True)
class _Leg:
    """One side of the day: entry and exit prices, stops and gross returns, per cell."""

    side: str
    entry: np.ndarray
    exit: np.ndarray
    stopped: np.ndarray
    gross: np.ndarray


def _result(
    legs: tuple[_Leg, ...],
    size: np.ndarray,
    cost: float,
    start: int,
    sessions: pd.DatetimeIndex,
    symbols: list[str],
) -> DayTradeResult:
    """Net returns of each fill, their sum by security and session, and the trades."""
    pnl = np.zeros((len(sessions), len(symbols)))
    paid = np.zeros(len(sessions))
    trades: list[pd.DataFrame] = []
    for leg in legs:
        filled = ~np.isnan(leg.entry) & (size > 0)
        # Both sides pay: the entry notional, and the exit notional the units fetch.
        fees = cost * (1.0 + leg.exit / leg.entry)
        net = np.where(filled, leg.gross - fees, 0.0)
        pnl += np.where(filled, size * net, 0.0)
        paid += np.where(filled, size * fees, 0.0).sum(axis=1)
        day, column = np.nonzero(filled)
        trades.append(
            pd.DataFrame(
                {
                    "security_id": [symbols[i] for i in column],
                    "side": leg.side,
                    "entry_time": sessions[day],
                    "exit_time": sessions[day],
                    "entry_price": leg.entry[day, column],
                    "exit_price": leg.exit[day, column],
                    "exit_reason": np.where(leg.stopped[day, column], "stop", "close"),
                    "size": size[day, column],
                    "return": net[day, column],
                }
            )
        )
    index = sessions[start:]
    table = pd.concat(trades, ignore_index=True)
    table = table.sort_values(["entry_time", "security_id", "side"], kind="stable")
    contributions = pd.DataFrame(pnl[start:], index=index, columns=symbols)
    return DayTradeResult(
        returns=contributions.sum(axis=1).rename(PORTFOLIO),
        contributions=contributions,
        costs=pd.Series(paid[start:], index=index, name="costs"),
        trades=table.reset_index(drop=True)[list(TRADE_COLUMNS)],
    )
