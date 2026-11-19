# ibexQuant\src\ibexQuant\backtest\weights.py

"""
Target-weight helpers for the E1 engine (ADR 0017, B1.16).

Responsibility
--------------
Turn simple inputs (a universe mask) into target weights that ``simulate_weights`` takes.

Contract
--------
- Weights are fractions of equity, one row per decision session, one column per security.
- A row with no eligible security is all cash.

What does not belong here
-------------------------
- Signals, risk models and position sizing with state: the strategy's, and E2's.
"""

from __future__ import annotations

import pandas as pd


def equal_weight(mask: pd.DataFrame) -> pd.DataFrame:
    """
    Return ``1 / n`` over the true cells of each row, zero elsewhere.

    Parameters
    ----------
    mask : pandas.DataFrame
        Boolean, decision sessions by security.

    Returns
    -------
    pandas.DataFrame
        Float weights of the same shape; a row without a true cell is all zero (cash).

    Raises
    ------
    TypeError
        If ``mask`` is not boolean.
    """
    if not all(pd.api.types.is_bool_dtype(dtype) for dtype in mask.dtypes):
        raise TypeError("mask must be boolean.")
    counts = mask.sum(axis=1)
    weights = mask.astype(float).div(counts.where(counts > 0), axis=0)
    return weights.fillna(0.0)
