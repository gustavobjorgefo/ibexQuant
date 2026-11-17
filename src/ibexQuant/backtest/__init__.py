# ibexQuant\src\ibexQuant\backtest\__init__.py

"""
Backtests on top of the data base (ADR 0017).

The walk-forward (``ibexQuant.backtest.walkforward``) cuts history into training
and test windows, in layers usable alone. The engines simulate what a strategy
does with them: E1 vectorized now; E2 and E3 in their own records.

Strategies, their parameters and their fitting never live here.
"""

from __future__ import annotations
