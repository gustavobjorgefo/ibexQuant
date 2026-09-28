# ibexQuant\src\ibexQuant\features\kernels\cross_sectional.py

"""
Cross-sectional kernels: statistics across assets at each timestamp.

Provides:

- :func:`cs_rank` — rank of each asset within its row, normalized to
  ``[0, 1]`` as ``(rank - 1) / (n - 1)``: the lowest value is 0, the highest
  is 1 and the median is 0.5, whatever the number of assets *n*. Ties take
  their average rank.
- :func:`cs_zscore` — distance of each asset from the row mean, in units of
  the row standard deviation (``ddof=1`` by default).

Both accept an optional universe mask (ADR 0004): assets outside the
universe at a timestamp are excluded from that row's statistics **and** are
``NaN`` in the output. A ``NaN`` value is treated the same way. Rows with
fewer than two eligible assets are ``NaN``.

No incremental states
---------------------
A cross-sectional value at *t* depends only on row *t*; there is no past to
keep and the lookback is zero. The incremental counterpart of each kernel is
the kernel itself applied to a one-row DataFrame, so this module defines no
:class:`~ibexQuant.features.kernels.state.IncrementalState` subclasses
(ADR 0008, amendment). Streaming execution must deliver every asset of a
timestamp together.

Cross-sectional kernels do not use the observation-time helper: they operate
row-wise, and pandas already skips ``NaN`` within a row.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from ibexQuant.features.kernels._validation import require_frame, require_integer, require_mask

MINIMUM_ASSETS: Final[int] = 2


# --- Batch kernels ---


def cs_rank(values: pd.DataFrame, universe_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Rank each asset within its row, normalized to ``[0, 1]``.

    Parameters
    ----------
    values : pd.DataFrame
        Wide values, timestamp x symbol.
    universe_mask : pd.DataFrame, optional
        Boolean, same index and columns as *values*. ``False`` excludes the
        asset at that timestamp. Defaults to every asset.

    Returns
    -------
    pd.DataFrame
        Same shape, values in ``[0, 1]``. ``NaN`` for excluded or missing
        assets, and for whole rows with fewer than two eligible assets.

    Raises
    ------
    TypeError
        If *values* or *universe_mask* is not a DataFrame.
    ValueError
        If *universe_mask* is not aligned with *values* or is not ``bool``.
    """
    eligible: pd.DataFrame = _eligible(values, universe_mask)
    counts: pd.Series = eligible.count(axis=1)
    ranks: pd.DataFrame = eligible.rank(axis=1, method="average")
    return (ranks - 1.0).div((counts - 1).where(counts >= MINIMUM_ASSETS), axis=0)


def cs_zscore(
    values: pd.DataFrame,
    universe_mask: pd.DataFrame | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """
    Z-score of each asset relative to its row.

    Parameters
    ----------
    values : pd.DataFrame
        Wide values, timestamp x symbol.
    universe_mask : pd.DataFrame, optional
        Boolean, same index and columns as *values*. ``False`` excludes the
        asset at that timestamp. Defaults to every asset.
    ddof : int, default 1
        Delta degrees of freedom of the row standard deviation.

    Returns
    -------
    pd.DataFrame
        Same shape. ``NaN`` for excluded or missing assets, and for whole rows
        with fewer than ``max(2, ddof + 1)`` eligible assets or with zero
        dispersion.

    Raises
    ------
    TypeError
        If *values* or *universe_mask* is not a DataFrame, or *ddof* is not an
        integer.
    ValueError
        If *ddof* is < 0, or *universe_mask* is not aligned with *values* or
        is not ``bool``.
    """
    ddof = require_integer("ddof", ddof, minimum=0)
    eligible: pd.DataFrame = _eligible(values, universe_mask)

    counts: pd.Series = eligible.count(axis=1)
    # Zero dispersion is detected exactly (all values equal) rather than
    # through the computed deviation, which may carry ~1e-17 of noise.
    valid: pd.Series = (counts >= max(MINIMUM_ASSETS, ddof + 1)) & (
        eligible.max(axis=1) > eligible.min(axis=1)
    )
    deviation: pd.Series = eligible.std(axis=1, ddof=ddof).where(valid)
    return eligible.sub(eligible.mean(axis=1), axis=0).div(deviation, axis=0)


# --- Helpers ---


def _eligible(values: pd.DataFrame, universe_mask: pd.DataFrame | None) -> pd.DataFrame:
    """Return *values* with assets outside the universe set to ``NaN``."""
    values = require_frame(values)
    if universe_mask is None:
        return values
    return values.where(require_mask(universe_mask, values))
