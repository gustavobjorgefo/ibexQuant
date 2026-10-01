# ibexQuant\src\ibexQuant\features\kernels\exponential.py

"""
Parameterization and warm-up of exponentially weighted kernels.

Every exponential kernel in ibexQuant takes a single smoothing factor,
``alpha``, always in the pandas sense: the weight of the **newest**
observation,

    ewma_t = alpha * x_t + (1 - alpha) * ewma_{t-1}

Other parameterizations are converted explicitly with the functions below.
This avoids a classic trap: in RiskMetrics, ``0.94`` is the decay (the weight
of the previous value), which corresponds to ``alpha = 0.06``, not ``0.94``.

Exponential kernels have infinite memory, so their warm-up is defined by a
tolerance: :func:`exponential_lookback` returns the number of past bars after
which the weight still carried by the starting point is at most that
tolerance. Kernels emit ``NaN`` until then. See ADR 0008.
"""

from __future__ import annotations

import math
from typing import Final

from ibexQuant._validation import require_real

DEFAULT_EXPONENTIAL_TOLERANCE: Final[float] = 1e-3


def alpha_from_span(span: float) -> float:
    """
    Convert a span into a smoothing factor: ``alpha = 2 / (span + 1)``.

    Parameters
    ----------
    span : float
        Span in bars. Must be > 1 (a span of 1 means no smoothing at all).

    Returns
    -------
    float
        Smoothing factor in ``(0, 1)``.

    Raises
    ------
    TypeError
        If *span* is not a real number.
    ValueError
        If *span* is not finite or not > 1.
    """
    return 2.0 / (require_real("span", span, above=1.0) + 1.0)


def alpha_from_halflife(halflife: float) -> float:
    """
    Convert a half-life into a smoothing factor.

    After *halflife* bars, an observation's weight has halved:
    ``(1 - alpha) ** halflife == 0.5``.

    Parameters
    ----------
    halflife : float
        Half-life in bars. Must be > 0.

    Returns
    -------
    float
        Smoothing factor in ``(0, 1)``.

    Raises
    ------
    TypeError
        If *halflife* is not a real number.
    ValueError
        If *halflife* is not finite or not > 0.
    """
    return 1.0 - math.pow(0.5, 1.0 / require_real("halflife", halflife, above=0.0))


def alpha_from_decay(decay: float) -> float:
    """
    Convert a decay factor (RiskMetrics lambda) into a smoothing factor.

    The decay is the weight of the **previous** value, so
    ``alpha = 1 - decay``. RiskMetrics' daily ``decay = 0.94`` gives
    ``alpha = 0.06``.

    Parameters
    ----------
    decay : float
        Decay factor in ``(0, 1)``.

    Returns
    -------
    float
        Smoothing factor in ``(0, 1)``.

    Raises
    ------
    TypeError
        If *decay* is not a real number.
    ValueError
        If *decay* is not finite or outside ``(0, 1)``.
    """
    return 1.0 - require_real("decay", decay, above=0.0, below=1.0)


def exponential_lookback(
    alpha: float,
    tolerance: float = DEFAULT_EXPONENTIAL_TOLERANCE,
) -> int:
    """
    Past bars needed before an exponential kernel's value is reliable.

    With *L* past bars beyond the current one, the starting observation still
    carries a weight of ``(1 - alpha) ** L``. The lookback is the smallest *L*
    for which that weight is at most *tolerance*.

    Parameters
    ----------
    alpha : float
        Smoothing factor in ``(0, 1)``.
    tolerance : float, default ``DEFAULT_EXPONENTIAL_TOLERANCE``
        Maximum weight of the starting point, in ``(0, 1)``.

    Returns
    -------
    int
        Lookback in bars. A kernel needs ``lookback + 1`` observations to
        emit its first value.

    Raises
    ------
    TypeError
        If *alpha* or *tolerance* is not a real number.
    ValueError
        If *alpha* or *tolerance* is outside ``(0, 1)``.

    Examples
    --------
    >>> exponential_lookback(alpha_from_span(20))
    70
    >>> exponential_lookback(alpha_from_decay(0.94))
    112
    """
    alpha = require_real("alpha", alpha, above=0.0, below=1.0)
    tolerance = require_real("tolerance", tolerance, above=0.0, below=1.0)

    retained: float = 1.0 - alpha
    lookback: int = math.ceil(math.log(tolerance) / math.log(retained))
    # Floating-point error can push an exact integer ratio up by one.
    if lookback > 0 and retained ** (lookback - 1) <= tolerance:
        lookback -= 1
    return lookback
