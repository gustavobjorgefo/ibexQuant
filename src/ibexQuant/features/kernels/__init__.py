# ibexQuant\src\ibexQuant\features\kernels\__init__.py

"""
Numerical kernels: the pure math behind every feature.

Each kernel is a function over a wide DataFrame (timestamp x symbol) and,
where applicable, an incremental state computing the same value one bar at a
time. Kernels know nothing about feature graphs, caching or engines.

Conventions shared by all kernels are recorded in
``docs/decisions/0008-kernel-conventions.md``; per-kernel conventions are
documented in each docstring.
"""

from __future__ import annotations

from ibexQuant.features.kernels.exponential import (
    DEFAULT_EXPONENTIAL_TOLERANCE,
    alpha_from_decay,
    alpha_from_halflife,
    alpha_from_span,
    exponential_lookback,
)
from ibexQuant.features.kernels.moving_averages import EmaState, SmaState, ema, sma
from ibexQuant.features.kernels.returns import (
    LogReturnState,
    SimpleReturnState,
    log_returns,
    simple_returns,
)
from ibexQuant.features.kernels.rolling import (
    RollingStdState,
    RollingSumState,
    rolling_std,
    rolling_sum,
)
from ibexQuant.features.kernels.state import IncrementalState
from ibexQuant.features.kernels.volatility import (
    EwmaVolatilityState,
    HistoricalVolatilityState,
    ewma_volatility,
    historical_volatility,
)

__all__ = [
    "DEFAULT_EXPONENTIAL_TOLERANCE",
    "EmaState",
    "EwmaVolatilityState",
    "HistoricalVolatilityState",
    "IncrementalState",
    "LogReturnState",
    "RollingStdState",
    "RollingSumState",
    "SimpleReturnState",
    "SmaState",
    "alpha_from_decay",
    "alpha_from_halflife",
    "alpha_from_span",
    "ema",
    "ewma_volatility",
    "exponential_lookback",
    "historical_volatility",
    "log_returns",
    "rolling_std",
    "rolling_sum",
    "simple_returns",
    "sma",
]
