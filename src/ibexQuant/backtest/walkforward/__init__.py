# ibexQuant\src\ibexQuant\backtest\walkforward\__init__.py

"""
Walk-forward in layers (ADR 0017, B1.3).

1. Time: :class:`WalkForwardSchedule` cuts the calendar into training and test
   :class:`Window` objects.
2. Universe: a :class:`UniverseRule` (:class:`StaticUniverse`,
   :class:`IndexComposition`) gives the candidates of a window.
3. Eligibility: :class:`EligibilityRule` objects (:class:`HasHistory`,
   :class:`StillTrading`, :class:`MinimumPresence`) keep the candidates with enough
   data.
4. Folds: :class:`WalkForward` assembles the layers into :class:`Fold` objects, whose
   ``load`` returns their :class:`FoldData`.

Each layer knows nothing of the layers above it, so a study can stop at any of
them.
"""

from __future__ import annotations

from ibexQuant.backtest.walkforward.eligibility import (
    EligibilityRule,
    HasHistory,
    MinimumPresence,
    StillTrading,
)
from ibexQuant.backtest.walkforward.errors import WalkForwardError
from ibexQuant.backtest.walkforward.folds import Fold, FoldData, WalkForward
from ibexQuant.backtest.walkforward.schedule import WalkForwardSchedule, Window
from ibexQuant.backtest.walkforward.universe import (
    IndexComposition,
    Selection,
    StaticUniverse,
    UniverseRule,
)

__all__ = [
    "EligibilityRule",
    "Fold",
    "FoldData",
    "HasHistory",
    "IndexComposition",
    "MinimumPresence",
    "Selection",
    "StaticUniverse",
    "StillTrading",
    "UniverseRule",
    "WalkForward",
    "WalkForwardError",
    "WalkForwardSchedule",
    "Window",
]
