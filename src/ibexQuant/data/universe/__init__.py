# ibexQuant\src\ibexQuant\data\universe\__init__.py

"""
Point-in-time universe of the B3 base (ADR 0015).

Index periods from the B3 calendar, index compositions by period resolved to
securities, and (in later steps) the universe mask.

Requires the ``b3`` extra (``pyarrow``, ``pyyaml``, ``openpyxl``).
"""

from __future__ import annotations

from ibexQuant.data.universe.calendar import (
    RULE_VERSIONS,
    PeriodOverride,
    PeriodOverrides,
    RuleVersion,
    StartRule,
    assign_periods,
    index_periods,
    load_period_overrides,
)
from ibexQuant.data.universe.constituents import (
    UniverseBuildError,
    UniverseError,
    UniverseNotBuiltError,
    UniverseSummary,
    build_universe,
    read_index_constituents,
    read_index_periods,
)

__all__ = [
    "RULE_VERSIONS",
    "PeriodOverride",
    "PeriodOverrides",
    "RuleVersion",
    "StartRule",
    "UniverseBuildError",
    "UniverseError",
    "UniverseNotBuiltError",
    "UniverseSummary",
    "assign_periods",
    "build_universe",
    "index_periods",
    "load_period_overrides",
    "read_index_constituents",
    "read_index_periods",
]
