# ibexQuant\src\ibexQuant\backtest\walkforward\__init__.py

"""
Walk-forward in layers (ADR 0017, B1.3).

1. Time: :class:`WalkForwardSchedule` cuts the calendar into training and test
   :class:`Window` objects.
2. Universe, 3. eligibility and 4. folds come in the next steps.

Each layer knows nothing of the layers above it, so a study can stop at any of
them.
"""

from __future__ import annotations

from ibexQuant.backtest.walkforward.errors import WalkForwardError
from ibexQuant.backtest.walkforward.schedule import WalkForwardSchedule, Window

__all__ = ["WalkForwardError", "WalkForwardSchedule", "Window"]
