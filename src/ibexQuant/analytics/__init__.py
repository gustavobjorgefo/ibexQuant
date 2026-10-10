# ibexQuant\src\ibexQuant\analytics\__init__.py

"""
Performance analytics: numbers derived from the results of a strategy.

Return analytics (``returns``, ``metrics``, ``distribution``) consume
periodic simple returns, as a ``pd.Series`` or a ``pd.DataFrame`` with one
column per series (e.g. strategy and benchmark). Trade analytics (``trades``)
consume one result per round-trip trade, plus the entry and exit times for
timing metrics.

Analytics never produce returns or trades (the backtest's job) and never draw
(:mod:`ibexQuant.plotting`); they depend only on pandas and NumPy, so any
consumer can use them without plotting libraries.
"""

from __future__ import annotations

from ibexQuant.analytics.distribution import (
    best_period,
    conditional_value_at_risk,
    kurtosis,
    mean_return,
    negative_ratio,
    positive_ratio,
    quantile,
    skewness,
    standard_deviation,
    value_at_risk,
    worst_period,
)
from ibexQuant.analytics.metrics import (
    B3_SESSIONS_PER_YEAR,
    annualized_volatility,
    cagr,
    calmar_ratio,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)
from ibexQuant.analytics.relative import (
    active_returns,
    alpha,
    beta,
    correlation,
    down_capture,
    excess_cagr,
    information_ratio,
    tracking_error,
    up_capture,
)
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
from ibexQuant.analytics.trades import (
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

__all__ = [
    "B3_SESSIONS_PER_YEAR",
    "Period",
    "active_returns",
    "alpha",
    "annualized_volatility",
    "average_holding_period",
    "average_loss",
    "average_win",
    "best_period",
    "best_trade",
    "beta",
    "cagr",
    "calmar_ratio",
    "conditional_value_at_risk",
    "correlation",
    "cumulative_returns",
    "down_capture",
    "drawdown",
    "drawdown_periods",
    "equity_curve",
    "excess_cagr",
    "excess_returns",
    "expectancy",
    "expectancy_t_stat",
    "exposure",
    "gross_loss",
    "gross_profit",
    "information_ratio",
    "kurtosis",
    "loss_rate",
    "max_consecutive_losses",
    "max_consecutive_wins",
    "max_drawdown",
    "max_drawdown_duration",
    "mean_return",
    "monthly_returns_table",
    "negative_ratio",
    "payoff_ratio",
    "period_returns",
    "positive_ratio",
    "profit_factor",
    "quantile",
    "returns_from_equity",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "standard_deviation",
    "total_return",
    "tracking_error",
    "trade_count",
    "trade_frequency",
    "trade_standard_deviation",
    "up_capture",
    "value_at_risk",
    "win_rate",
    "worst_period",
    "worst_trade",
]
