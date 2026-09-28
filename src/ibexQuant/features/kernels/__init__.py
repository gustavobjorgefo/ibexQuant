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

__all__ = [
    "DEFAULT_EXPONENTIAL_TOLERANCE",
    "alpha_from_decay",
    "alpha_from_halflife",
    "alpha_from_span",
    "exponential_lookback",
]
