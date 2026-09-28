# ibexQuant\src\ibexQuant\features\kernels\_observation_time.py

"""
Observation-time execution of time-series kernels.

Implements the missing data policy of ADR 0003 in one place, so that no
kernel reimplements it: a time-series computation runs over each asset's own
valid observations, gaps are skipped and never filled, and ``NaN`` in the
output means "not observable at this timestamp".

Two execution paths produce identical results:

- **Fast path** — columns without interior gaps are computed together in one
  vectorized call. Leading ``NaN`` (before listing) and trailing ``NaN``
  (after delisting) do not require the slow path, because no valid value
  lies between them.
- **Per-column path** — each column with interior gaps is computed on its
  valid observations only and reindexed back onto the full index.

What does not belong here
-------------------------
Cross-sectional kernels, which operate row-wise and do not use this module.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def has_interior_gaps(values: pd.DataFrame) -> pd.Series:
    """
    Flag, per column, whether any ``NaN`` lies between two valid values.

    Parameters
    ----------
    values : pd.DataFrame
        Wide DataFrame, timestamp x symbol.

    Returns
    -------
    pd.Series
        Boolean, indexed by column. ``True`` if the column has at least one
        interior gap.
    """
    observed: pd.DataFrame = values.notna()
    seen_before: pd.DataFrame = observed.cummax()
    seen_after: pd.DataFrame = observed.iloc[::-1].cummax().iloc[::-1]
    interior: pd.DataFrame = ~observed & seen_before & seen_after
    return interior.any()


def in_observation_time(
    values: pd.DataFrame,
    compute: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """
    Apply a time-series computation in each column's observation time.

    Parameters
    ----------
    values : pd.DataFrame
        Wide DataFrame, timestamp x symbol.
    compute : Callable[[pd.DataFrame], pd.DataFrame]
        Vectorized time-series computation. It must return a DataFrame with
        the same index and columns as its input and treat each column
        independently.

    Returns
    -------
    pd.DataFrame
        Same index and column order as *values*. Every cell where *values*
        is ``NaN`` is ``NaN`` in the result, whatever *compute* produced
        there: an asset that is not observable has no feature value.
    """
    gapped: pd.Series = has_interior_gaps(values)

    pieces: list[pd.DataFrame] = []
    if not gapped.all():
        pieces.append(compute(values.loc[:, ~gapped]))

    for column in values.columns[gapped]:
        observed: pd.DataFrame = values[[column]].dropna()
        pieces.append(compute(observed).reindex(values.index))

    result: pd.DataFrame = pd.concat(pieces, axis=1).reindex(columns=values.columns)
    # Some pandas operations carry values forward through NaN rows (e.g.
    # ewm().mean()); masking enforces the policy regardless of `compute`.
    return result.where(values.notna())
