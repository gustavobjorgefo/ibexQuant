# ibexQuant\src\ibexQuant\features\kernels\_validation.py

"""
Parameter validation shared by kernels and, later, by feature nodes.

Kernels are public functions, also called directly from notebooks, so they
validate their own parameters. Feature nodes call these same helpers at
construction, keeping a single source for each rule.

Every helper returns the validated value converted to a plain Python type,
so it can be used inline: ``window = require_integer("window", window,
minimum=1)``.

What does not belong here
-------------------------
Validation of data invariants (alignment, ordering, dtypes), which is the
responsibility of :class:`ibexQuant.data.panel.Panel`.
"""

from __future__ import annotations

import math

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


def require_integer(name: str, value: object, *, minimum: int) -> int:
    """
    Require an integer greater than or equal to *minimum*.

    Parameters
    ----------
    name : str
        Parameter name, used in error messages.
    value : object
        Candidate value. Python and NumPy integers are accepted; ``bool`` and
        floats (even integral ones such as ``3.0``) are rejected.
    minimum : int
        Inclusive lower bound.

    Returns
    -------
    int
        *value* as a Python ``int``.

    Raises
    ------
    TypeError
        If *value* is not an integer.
    ValueError
        If *value* is below *minimum*.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value)!r}.")

    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {integer}.")
    return integer


def require_real(
    name: str,
    value: object,
    *,
    above: float | None = None,
    below: float | None = None,
) -> float:
    """
    Require a finite real number strictly inside the given bounds.

    Parameters
    ----------
    name : str
        Parameter name, used in error messages.
    value : object
        Candidate value. Python and NumPy numbers are accepted; ``bool`` is
        rejected.
    above : float, optional
        Exclusive lower bound.
    below : float, optional
        Exclusive upper bound.

    Returns
    -------
    float
        *value* as a Python ``float``.

    Raises
    ------
    TypeError
        If *value* is not a real number.
    ValueError
        If *value* is not finite or lies outside the open interval.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value)!r}.")

    real = float(value)
    if not math.isfinite(real):
        raise ValueError(f"{name} must be finite, got {real}.")

    if above is not None and real <= above:
        raise ValueError(f"{name} must be > {above}, got {real}.")

    if below is not None and real >= below:
        raise ValueError(f"{name} must be < {below}, got {real}.")
    return real
