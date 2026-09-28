# ibexQuant\tests\features\kernels\conftest.py

"""
Shared fixtures for kernel tests.

Provides reproducible random price panels with every kind of missing data
(late listing, delisting, interior gaps) and a parity check that feeds each
column, value by value, into a fresh incremental state and compares the
result with the batch kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.features.kernels.state import IncrementalState

# pandas rolling kernels update sums online, while states recompute each window
# exactly; the two differ by floating-point noise (up to ~1e-10 relative on
# realistic data, and absolute ~1e-16 on sums close to zero). Any convention
# error (ddof, adjust, window alignment) is orders of magnitude larger.
PARITY_RTOL: Final[float] = 1e-9
PARITY_ATOL: Final[float] = 1e-12

type PriceFactory = Callable[[int], pd.DataFrame]
type ParityCheck = Callable[
    [Callable[[pd.DataFrame], pd.DataFrame], Callable[[], IncrementalState], pd.DataFrame],
    None,
]


@pytest.fixture
def random_prices() -> PriceFactory:
    """
    Build a reproducible wide price panel for a given seed.

    Columns cover the missing-data cases: ``COMPLETE`` (no NaN), ``LISTED``
    (leading NaN), ``DELISTED`` (trailing NaN) and ``SUSPENDED`` (interior
    gaps, including a multi-session one).
    """

    def build(seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        sessions = pd.bdate_range("2024-01-01", periods=250, name="timestamp")
        symbols = ["COMPLETE", "LISTED", "DELISTED", "SUSPENDED"]
        log_steps = rng.normal(0.0, 0.02, size=(len(sessions), len(symbols)))
        prices = pd.DataFrame(
            30.0 * np.exp(np.cumsum(log_steps, axis=0)),
            index=sessions,
            columns=pd.Index(symbols, name="symbol"),
        )
        prices.iloc[:40, 1] = np.nan
        prices.iloc[-30:, 2] = np.nan
        gaps = rng.choice(np.arange(10, 240), size=12, replace=False)
        prices.iloc[gaps, 3] = np.nan
        prices.iloc[100:105, 3] = np.nan
        return prices

    return build


@pytest.fixture
def assert_parity() -> ParityCheck:
    """Assert that an incremental state reproduces its batch kernel, column by column."""

    def check(
        batch: Callable[[pd.DataFrame], pd.DataFrame],
        make_state: Callable[[], IncrementalState],
        values: pd.DataFrame,
    ) -> None:
        expected = batch(values)
        for column in values.columns:
            state = make_state()
            streamed = [state.update(float(value)) for value in values[column]]
            np.testing.assert_allclose(
                streamed,
                expected[column].to_numpy(),
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
                equal_nan=True,
                err_msg=f"batch and incremental differ for column {column!r}",
            )

    return check
