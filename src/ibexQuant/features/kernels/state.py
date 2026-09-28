# ibexQuant\src\ibexQuant\features\kernels\state.py

"""
Contract for incremental kernel states.

An incremental state computes, one observation at a time, the same value its
batch kernel computes over a whole column. It is the building block of
bar-by-bar execution (event-driven backtests and live trading).

Contract (ADR 0008, D1.11)
--------------------------
- One state per asset, updated with scalars.
- :meth:`IncrementalState.update` returns ``NaN`` until ``lookback + 1``
  observations have been seen.
- A ``NaN`` input is not an observation: the state does not change and
  ``NaN`` is returned. This reproduces the batch treatment of gaps.
- Fed a column's values in order, a state reproduces the batch kernel's
  output for that column (verified by parity tests).

What does not belong here
-------------------------
Orchestration of states across a feature graph and across assets, which is
the responsibility of the streaming engine (Phase 7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class IncrementalState(ABC):
    """
    Abstract base class for the incremental counterpart of a batch kernel.

    Subclasses hold only what is needed to produce the next value (e.g. the
    previous price, a ring buffer, a running average) and never the full
    history.
    """

    # --- Abstract interface ---

    @property
    @abstractmethod
    def lookback(self) -> int:
        """
        int: Past observations needed beyond the current one.

        :meth:`update` returns its first non-``NaN`` value on the
        ``lookback + 1``-th valid observation.
        """
        ...

    @abstractmethod
    def update(self, value: float) -> float:
        """
        Consume one observation and return the current kernel value.

        Parameters
        ----------
        value : float
            The asset's newest observation. ``NaN`` means "no observation":
            the state is left unchanged.

        Returns
        -------
        float
            The kernel value after this observation, or ``NaN`` while
            warming up or when *value* is ``NaN``.
        """
        ...

    # --- Default behaviour ---

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lookback={self.lookback})"
