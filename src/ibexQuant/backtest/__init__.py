# ibexQuant\src\ibexQuant\backtest\__init__.py

"""
Backtests on top of the data base (ADR 0017).

The walk-forward (``ibexQuant.backtest.walkforward``) cuts history into training
and test windows, in layers usable alone. The engines simulate what a strategy
does with them: E1 vectorized now, for positions (:func:`simulate_weights`) and for
day trades on daily bars (:func:`simulate_day_trades`); E2 and E3 in their own
records.

Strategies, their parameters and their fitting never live here.
"""

from __future__ import annotations

from ibexQuant.backtest.daytrade import DayTradeOrders, DayTradeResult, simulate_day_trades
from ibexQuant.backtest.vectorized import BacktestResult, simulate_weights
from ibexQuant.backtest.weights import equal_weight

__all__ = [
    "BacktestResult",
    "DayTradeOrders",
    "DayTradeResult",
    "equal_weight",
    "simulate_day_trades",
    "simulate_weights",
]
