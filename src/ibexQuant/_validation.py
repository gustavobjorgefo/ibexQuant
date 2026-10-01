# ibexQuant\src\ibexQuant\_validation.py

"""
Scalar parameter validation shared across ibexQuant packages.

Generic rules that do not depend on any domain (an integer with a minimum, a
finite real inside bounds) live here, so each rule has a single source no
matter how many packages enforce it.

Every helper returns the validated value converted to a plain Python type,
so it can be used inline: ``window = require_integer("window", window,
minimum=1)``.

What does not belong here
-------------------------
Validation of domain objects (kernel inputs, universe masks, return series),
which lives in the package that defines the contract.
"""

from __future__ import annotations

import math

import numpy as np


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
