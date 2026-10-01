# ibexQuant\src\ibexQuant\analytics\__init__.py

"""
Performance analytics: numbers derived from the returns of a strategy.

Every function consumes periodic simple returns, as a ``pd.Series`` or a
``pd.DataFrame`` with one column per series (e.g. strategy and benchmark), and
returns the same shape it received. Analytics never produce returns (the
backtest's job) and never draw (:mod:`ibexQuant.plotting`); they depend only
on pandas and NumPy, so any consumer can use them without plotting libraries.
"""

from __future__ import annotations

from ibexQuant.analytics.returns import (
    Period,
    cumulative_returns,
    drawdown,
    drawdown_periods,
    equity_curve,
    excess_returns,
    monthly_returns_table,
    period_returns,
    returns_from_equity,
)

__all__ = [
    "Period",
    "cumulative_returns",
    "drawdown",
    "drawdown_periods",
    "equity_curve",
    "excess_returns",
    "monthly_returns_table",
    "period_returns",
    "returns_from_equity",
]
