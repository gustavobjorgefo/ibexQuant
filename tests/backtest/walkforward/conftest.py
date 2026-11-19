# ibexQuant\tests\backtest\walkforward\conftest.py

"""The synthetic base of ``tests/support/universe.py``, built once for the walk-forward tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from support.universe import write_base, write_portfolios

from ibexQuant.backtest.walkforward import WalkForwardSchedule, Window
from ibexQuant.data.adjustment import build_adjustment_factors
from ibexQuant.data.dataset import B3Dataset
from ibexQuant.data.reference.corporate_action_overrides import ActionOverrides
from ibexQuant.data.reference.corporate_actions import build_corporate_actions
from ibexQuant.data.store import DataStore
from ibexQuant.data.universe import LiquiditySettings, PeriodOverrides, build_universe


@pytest.fixture(scope="session")
def ds(tmp_path_factory: pytest.TempPathFactory) -> B3Dataset:
    """
    Sessions 2024-03-01 to 2025-02-28; index periods 2024-05, 2024-09 and 2025-01.

    IBOV 2024-09 has GONE3 without a security, LATE3 listed on 2024-09-02 and QUIT3,
    which stops on 2024-10-15; PAUS3 (IBXX) has no bar on 2024-09-02.
    """
    store = DataStore.initialize(tmp_path_factory.mktemp("walkforward") / "data")
    write_portfolios(write_base(store))
    build_corporate_actions(store, ActionOverrides.none())
    build_adjustment_factors(store)
    build_universe(store, PeriodOverrides.none(), LiquiditySettings(lookback_months=1))
    return B3Dataset.open(store)


@pytest.fixture
def window_for(ds: B3Dataset) -> Callable[[str], Window]:
    """Build the window testing one month after two months of training."""

    def build(month: str) -> Window:
        schedule = WalkForwardSchedule(month, month, train_months=2, test_months=1)
        return schedule.window(month, month, ds.calendar)

    return build
