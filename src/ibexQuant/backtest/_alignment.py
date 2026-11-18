# ibexQuant\src\ibexQuant\backtest\_alignment.py

"""
Decisions to executions, shared by the E1 engines (ADR 0017, B1.4).

Responsibility
--------------
Check a frame indexed by decision session against the sessions of a panel, and give the
position of the session that executes each decision: the next one.

Contract
--------
- Decision sessions are sessions of the panel, strictly increasing, and never the last
  session (it has no next one).
- Columns are securities of the panel.

What does not belong here
-------------------------
- What an order or a weight means: each engine's.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def execution_positions(
    frame: pd.DataFrame, sessions: pd.DatetimeIndex, symbols: Sequence[str], name: str
) -> np.ndarray:
    """
    Return the position in ``sessions`` of the session after each row's decision.

    Raises
    ------
    ValueError
        If ``frame`` is empty, its index is not strictly increasing, a column is not in
        ``symbols``, a decision is not a session or is the last one.
    """
    if frame.empty:
        raise ValueError(f"{name} has no decision.")
    if not (frame.index.is_monotonic_increasing and frame.index.is_unique):
        raise ValueError(f"{name} index must be strictly increasing (sorted, no duplicates).")
    unknown = sorted(set(map(str, frame.columns)) - set(symbols))
    if unknown:
        raise ValueError(f"{name} names securities that are not in bars: {unknown}.")
    positions = sessions.get_indexer(pd.DatetimeIndex(frame.index))
    if (positions < 0).any():
        raise ValueError(f"Decision {frame.index[positions < 0][0]} is not a session of bars.")
    if positions[-1] == len(sessions) - 1:
        raise ValueError(f"Decision {sessions[-1].date()} is the last session of bars.")
    return positions + 1
