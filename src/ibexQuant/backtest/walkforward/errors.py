# ibexQuant\src\ibexQuant\backtest\walkforward\errors.py

"""Errors of ``ibexQuant.backtest.walkforward`` (ADR 0017)."""

from __future__ import annotations


class WalkForwardError(ValueError):
    """A walk-forward cannot be built from the data at hand."""
