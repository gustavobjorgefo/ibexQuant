# ibexQuant\src\ibexQuant\features\kernels\_window.py

"""
Shared buffering for fixed-window incremental states.

A fixed-window state keeps the last ``window`` valid observations in a ring
buffer and recomputes its statistic from the buffer on every update.

Recomputing costs O(window) per update, but it is exact at every step.
Running sums updated by adding the new value and subtracting the dropped one
are O(1), but accumulate floating-point error over long streams, which
matters in live trading where a state may run for months.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections import deque

from ibexQuant.features.kernels._validation import require_integer
from ibexQuant.features.kernels.state import IncrementalState


class FixedWindowState(IncrementalState):
    """
    Base class for states computed over the last ``window`` observations.

    Parameters
    ----------
    window : int
        Number of observations in the window. Must be >= 1.

    Raises
    ------
    TypeError
        If *window* is not an integer.
    ValueError
        If *window* is < 1.
    """

    def __init__(self, window: int) -> None:
        self._window: int = require_integer("window", window, minimum=1)
        self._buffer: deque[float] = deque(maxlen=self._window)

    # --- IncrementalState interface ---

    @property
    def lookback(self) -> int:
        return self._window - 1

    def update(self, value: float) -> float:
        if math.isnan(value):
            return math.nan

        self._buffer.append(value)
        if len(self._buffer) < self._window:
            return math.nan
        return self._compute(self._buffer)

    # --- Subclass hook ---

    @abstractmethod
    def _compute(self, window: deque[float]) -> float:
        """Compute the statistic over a full window, oldest value first."""
        ...
