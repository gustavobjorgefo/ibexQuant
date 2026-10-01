# ibexQuant\src\ibexQuant\features\kernels\_validation.py

"""
Kernel input validation shared by kernels and, later, by feature nodes.

Kernels are public functions, also called directly from notebooks, so they
validate their own inputs. Feature nodes call these same helpers at
construction, keeping a single source for each rule. Scalar parameter rules
(``require_integer``, ``require_real``) are shared with other packages and
live in :mod:`ibexQuant._validation`.

What does not belong here
-------------------------
Validation of data invariants (alignment, ordering, dtypes), which is the
responsibility of :class:`ibexQuant.data.panel.Panel`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def require_frame(values: object) -> pd.DataFrame:
    """
    Require a pandas DataFrame.

    Parameters
    ----------
    values : object
        Candidate kernel input.

    Returns
    -------
    pd.DataFrame
        *values*, unchanged.

    Raises
    ------
    TypeError
        If *values* is not a DataFrame (a Series included: single assets are
        passed as one-column DataFrames).
    """
    if not isinstance(values, pd.DataFrame):
        raise TypeError(f"Expected a pd.DataFrame, got {type(values)!r}.")
    return values


def require_mask(mask: object, values: pd.DataFrame) -> pd.DataFrame:
    """
    Require a boolean mask aligned with *values*.

    Parameters
    ----------
    mask : object
        Candidate universe mask.
    values : pd.DataFrame
        The DataFrame the mask applies to.

    Returns
    -------
    pd.DataFrame
        *mask*, unchanged.

    Raises
    ------
    TypeError
        If *mask* is not a DataFrame.
    ValueError
        If *mask* does not share the index and columns of *values*, or is not
        ``bool`` (which also excludes missing values).
    """
    if not isinstance(mask, pd.DataFrame):
        raise TypeError(f"Expected universe_mask to be a pd.DataFrame, got {type(mask)!r}.")

    if not mask.index.equals(values.index) or not mask.columns.equals(values.columns):
        raise ValueError("universe_mask must have the same index and columns as values.")

    if not all(dtype == np.dtype("bool") for dtype in mask.dtypes):
        raise ValueError(
            f"universe_mask must be bool; got dtypes {sorted(set(map(str, mask.dtypes)))}."
        )
    return mask
