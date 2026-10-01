# ibexQuant\src\ibexQuant\analytics\_types.py

"""
Type definitions shared by analytics modules.

Analytics functions accept a single series (``pd.Series``) or several aligned
series (``pd.DataFrame``, one column each) and return the same shape they
received. :data:`ReturnsT` expresses that contract to the type checker.
"""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

# A Series in gives a Series out; a DataFrame in gives a DataFrame out. A
# module-level TypeVar (rather than PEP 695 type parameters) keeps this
# constraint defined once for every analytics function that shares it.
ReturnsT = TypeVar("ReturnsT", pd.Series, pd.DataFrame)
