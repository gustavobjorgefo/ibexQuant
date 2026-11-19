# ibexQuant\src\ibexQuant\backtest\vectorized.py

"""
E1 position engine: target weights simulated on adjusted bars (ADR 0017, B1.9-B1.15).

Responsibility
--------------
Turn target weights, decided at closes, into the returns of one portfolio that holds
fractional units of adjusted prices, with drift between rebalances, exits as cash and a
linear cost.

Contract
--------
- A row of ``weights`` is a rebalance and the whole target portfolio, decided at the close
  of its session and executed in the next one (``next_open`` or ``next_close``).
- Positions are signed units; short proceeds go to cash; cash earns nothing and may be
  negative.
- A security without a bar on the execution session trades at its last close; after its
  last bar its holding is cash at the last close, at no cost; a target on a security with
  no price yet raises ``ValueError``.
- Every fraction of the result is over the equity at the previous close, so
  ``returns == contributions.sum(axis=1) - costs``.
- The only loop runs over rebalances and exits; between them, values are computed for the
  whole segment at once.

What does not belong here
-------------------------
- Signals, weights and fitting: the strategy's.
- Raw prices, event cash, ticks, lots and order types: the E2 engine (P9, P11, P12).
- Metrics: ``ibexQuant.analytics`` reads ``BacktestResult.returns``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from ibexQuant.analytics import PerformanceSummary, performance_summary
from ibexQuant.backtest._alignment import execution_positions
from ibexQuant.data.panel import Panel

EXECUTIONS: Final[tuple[str, ...]] = ("next_open", "next_close")
BASIS_POINT: Final[float] = 1e-4
PORTFOLIO: Final[str] = "portfolio"
CASH: Final[str] = "cash"


@dataclass(frozen=True)
class BacktestResult:
    """
    What a simulation produced, by session from the first execution.

    Every fraction is over the equity at the previous close, so
    ``returns == contributions.sum(axis=1) - costs``.

    Attributes
    ----------
    returns : pandas.Series
        Simple return of the portfolio, named ``portfolio``.
    contributions : pandas.DataFrame
        Each security's profit and loss over the previous equity.
    weights : pandas.DataFrame
        Weights held at each close, after drift, with a ``cash`` column.
    turnover : pandas.Series
        Traded notional over the previous equity.
    costs : pandas.Series
        Costs over the previous equity.
    """

    returns: pd.Series
    contributions: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    @property
    def equity(self) -> pd.Series:
        """Equity of the portfolio, starting from 1 before the first execution."""
        return (1.0 + self.returns).cumprod().rename("equity")

    @property
    def gross_exposure(self) -> pd.Series:
        """Sum of the absolute weights of the securities at each close."""
        return self.weights.drop(columns=CASH).abs().sum(axis=1).rename("gross_exposure")

    @property
    def net_exposure(self) -> pd.Series:
        """Sum of the signed weights of the securities at each close."""
        return self.weights.drop(columns=CASH).sum(axis=1).rename("net_exposure")

    def summary(
        self,
        *,
        periods_per_year: int,
        benchmark: pd.Series | None = None,
        risk_free: float | pd.Series = 0.0,
    ) -> PerformanceSummary:
        """
        Return the performance table of ``returns`` (ADR 0009).

        A benchmark is aligned to the sessions of ``returns``: dates outside them are
        dropped and missing ones are ``NaN``, which the table accepts only at the edges.
        """
        aligned = None if benchmark is None else benchmark.reindex(self.returns.index)
        return performance_summary(
            self.returns, aligned, risk_free, periods_per_year=periods_per_year
        )


def simulate_weights(
    bars: Panel,
    weights: pd.DataFrame,
    *,
    execution: Literal["next_open", "next_close"] = "next_open",
    cost_bps: float = 0.0,
) -> BacktestResult:
    """
    Simulate target weights decided at closes.

    Parameters
    ----------
    bars : Panel
        Adjusted bars with ``close``, and ``open`` for ``next_open``, covering every
        security in ``weights`` from the first decision to the last session to value.
    weights : pandas.DataFrame
        Target fractions of equity, one row per rebalance, indexed by decision session,
        one column per security; a missing column or ``NaN`` is zero.
    execution : {"next_open", "next_close"}, default "next_open"
        Price of the session after each decision at which the rebalance trades.
    cost_bps : float, default 0.0
        Cost per side, in basis points of the traded notional.

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        If an argument is invalid, a decision session is not a session of ``bars`` (or is
        its last), a column is not a security of ``bars``, or a target names a security
        before its first price.
    """
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {list(EXECUTIONS)}, got {execution!r}.")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be zero or positive, got {cost_bps}.")
    market = _Market.of(bars, execution)
    targets = _targets(weights, market)
    start = min(targets)
    count, width = len(market.sessions), len(market.symbols)
    exits = {int(last) + 1 for last in market.last_bar if start <= last + 1 < count}

    units, cash = np.zeros(width), 1.0
    held, balance = np.zeros((count, width)), np.zeros(count)
    flows, traded, cost = np.zeros((count, width)), np.zeros(count), np.zeros(count)
    events = sorted(set(targets) | exits)
    for event, end in zip(events, [*events[1:], count], strict=True):
        gone = (market.last_bar == event - 1) & (units != 0)
        if gone.any():
            # An exit is cash at the last close, at no cost and outside the turnover (B1.12).
            flows[event, gone] = -units[gone] * market.marked[event - 1, gone]
            cash -= float(flows[event, gone].sum())
            units[gone] = 0.0
        if event in targets:
            new, trade = market.rebalance(units, cash, targets[event], event)
            flows[event] += trade
            traded[event] = float(np.abs(trade).sum())
            cost[event] = cost_bps * BASIS_POINT * traded[event]
            cash -= float(trade.sum()) + cost[event]
            units = new
        held[event:end] = units
        balance[event:end] = cash
    return _result(market, start, held, balance, flows, traded, cost)


# --- helpers ---


@dataclass(frozen=True)
class _Market:
    """Prices of the run as arrays, and where each security's bars start and end."""

    sessions: pd.DatetimeIndex
    symbols: list[str]
    price: np.ndarray  # execution price: open or close
    marked: np.ndarray  # last close, carried over sessions without a bar
    has_bar: np.ndarray
    first_bar: np.ndarray  # position of the first bar; len(sessions) if none
    last_bar: np.ndarray  # position of the last bar; -1 if none

    @classmethod
    def of(cls, bars: Panel, execution: str) -> _Market:
        close = bars.field("close").to_numpy(dtype=float)
        price = bars.field("open").to_numpy(dtype=float) if execution == "next_open" else close
        has_bar = ~np.isnan(close)
        any_bar = has_bar.any(axis=0)
        count = len(bars.index)
        return cls(
            sessions=bars.index,
            symbols=list(bars.symbols),
            price=price,
            marked=pd.DataFrame(close).ffill().to_numpy(),
            has_bar=has_bar,
            first_bar=np.where(any_bar, has_bar.argmax(axis=0), count),
            last_bar=np.where(any_bar, count - 1 - has_bar[::-1].argmax(axis=0), -1),
        )

    def rebalance(
        self, units: np.ndarray, cash: float, target: np.ndarray, event: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Units after trading to ``target`` on session ``event``, and the traded values."""
        target = np.where(self.last_bar < event, 0.0, target)
        execution = self.price[event]
        tradable = self.has_bar[event] & ~np.isnan(execution)
        # A security without a bar trades at its last close, as a holder marking it would:
        # E1 never keeps a holding it cannot price (B1.12).
        price = np.where(tradable, execution, self.marked[event - 1])
        priced = ~np.isnan(price)
        early = (target != 0) & ~priced
        if early.any():
            names = [self.symbols[i] for i in np.flatnonzero(early)]
            raise ValueError(
                f"Target on {names} before their first price, executed on "
                f"{self.sessions[event].date()}."
            )
        price = np.where(priced, price, 1.0)
        equity = cash + float(np.sum(units * price))
        new = target * equity / price
        return new, (new - units) * price


def _targets(weights: pd.DataFrame, market: _Market) -> dict[int, np.ndarray]:
    """Target rows by execution position: the session after each decision."""
    if weights.empty:
        raise ValueError("weights has no rebalance.")
    positions = execution_positions(weights, market.sessions, market.symbols, "weights")
    values = weights.reindex(columns=market.symbols).fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite.")
    return {int(position): row for position, row in zip(positions, values, strict=True)}


def _result(
    market: _Market,
    start: int,
    held: np.ndarray,
    balance: np.ndarray,
    flows: np.ndarray,
    traded: np.ndarray,
    cost: np.ndarray,
) -> BacktestResult:
    """Values, returns and fractions from the units and the cash of each session."""
    symbols = market.symbols
    values = np.where(held != 0, held * np.nan_to_num(market.marked), 0.0)
    equity = balance + values.sum(axis=1)
    previous_values = np.vstack([np.zeros((1, len(symbols))), values[:-1]])
    previous_equity = np.concatenate([[1.0], equity[start:-1]])
    pnl = (values - previous_values - flows)[start:]

    index = market.sessions[start:]
    weights = pd.DataFrame(values[start:] / equity[start:, None], index=index, columns=symbols)
    weights[CASH] = balance[start:] / equity[start:]
    return BacktestResult(
        returns=pd.Series(equity[start:] / previous_equity - 1.0, index=index, name=PORTFOLIO),
        contributions=pd.DataFrame(pnl / previous_equity[:, None], index=index, columns=symbols),
        weights=weights,
        turnover=pd.Series(traded[start:] / previous_equity, index=index, name="turnover"),
        costs=pd.Series(cost[start:] / previous_equity, index=index, name="costs"),
    )
