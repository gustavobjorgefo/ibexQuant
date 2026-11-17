# ibexQuant\tests\backtest\walkforward\test_universe.py

"""Layer 2 of the walk-forward: universe rules (ADR 0017, B1.5)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from ibexQuant.backtest.walkforward import (
    IndexComposition,
    StaticUniverse,
    WalkForwardError,
    WalkForwardSchedule,
    Window,
)
from ibexQuant.backtest.walkforward.universe import quadrimester
from ibexQuant.data.dataset import B3Dataset
from ibexQuant.data.universe import Index, Tickers

SEPTEMBER = ("BRLATE", "BROLDX", "BRPETR", "BRQUIT", "BRVALE")


@pytest.mark.parametrize(
    ("month", "label"),
    [
        ("2024-01", "2024-01"),
        ("2024-04", "2024-01"),
        ("2024-05", "2024-05"),
        ("2024-12", "2024-09"),
    ],
)
def test_quadrimester(month: str, label: str) -> None:
    assert quadrimester(pd.Period(month, "M")) == label


# --- IndexComposition ---


def test_index_composition_from_the_first(
    ds: B3Dataset, window_for: Callable[[str], Window]
) -> None:
    # The September portfolio applies from 2024-09-02, B3's first Monday, and from the 1st here.
    selection = IndexComposition("IBOV").select(ds, window_for("2024-10"))
    assert selection.security_ids == SEPTEMBER
    assert selection.unresolved == ("GONE3",)
    assert (selection.index, selection.period, selection.source) == (
        "IBOV",
        "2024-09",
        "third_preview",
    )
    assert selection.universe is None


def test_index_composition_without_a_composition(
    ds: B3Dataset, window_for: Callable[[str], Window]
) -> None:
    with pytest.raises(WalkForwardError, match=r"IBXX has no composition in period 2024-05 \("):
        IndexComposition("IBXX").select(ds, window_for("2024-06"))


def test_index_composition_outside_the_periods_of_the_base(ds: B3Dataset) -> None:
    window = WalkForwardSchedule("2025-02", "2025-02", train_months=1, test_months=1).window(
        "2024-04", "2024-04", ds.calendar
    )
    with pytest.raises(WalkForwardError, match=r"2024-01 is not a period of the base .* \(test"):
        IndexComposition("IBOV").select(ds, window)


def test_index_composition_cuts(ds: B3Dataset) -> None:
    schedule = WalkForwardSchedule("2024-08", "2024-12", train_months=2, test_months=5)
    window = schedule.window("2024-08", "2024-12", ds.calendar)
    assert IndexComposition("IBOV").cuts(window) == (pd.Period("2024-09", "M"),)
    assert IndexComposition("IBOV", at_rebalance="hold").cuts(window) == ()
    inside = schedule.window("2024-09", "2024-12", ds.calendar)
    assert IndexComposition("IBOV").cuts(inside) == ()


def test_index_composition_checks_at_rebalance() -> None:
    with pytest.raises(ValueError, match="at_rebalance must be one of"):
        IndexComposition("IBOV", at_rebalance="never")  # type: ignore[arg-type]
    assert repr(IndexComposition("IBXX")) == "IndexComposition('IBXX', at_rebalance='split')"


# --- StaticUniverse ---


def test_static_tickers(ds: B3Dataset, window_for: Callable[[str], Window]) -> None:
    rule = StaticUniverse(["VALE3", "PETR4"])
    selection = rule.select(ds, window_for("2024-10"))
    assert selection.security_ids == ("BRPETR", "BRVALE")
    assert selection.universe == Tickers(["VALE3", "PETR4"])
    assert selection.index is None
    assert rule.cuts(window_for("2024-10")) == ()


def test_static_index_follows_b3_dates(ds: B3Dataset, window_for: Callable[[str], Window]) -> None:
    # August holds May's portfolio; September holds September's from Monday 2024-09-02.
    assert StaticUniverse("IBOV").select(ds, window_for("2024-08")).security_ids == ("BRPETR",)
    assert (
        StaticUniverse(Index("IBOV")).select(ds, window_for("2024-09")).security_ids == SEPTEMBER
    )


def test_static_without_members(ds: B3Dataset, window_for: Callable[[str], Window]) -> None:
    with pytest.raises(WalkForwardError, match="has no member from 2024-06-03 to 2024-06-28"):
        StaticUniverse(Index("IBXX")).select(ds, window_for("2024-06"))
    assert repr(StaticUniverse("IBXX")) == f"StaticUniverse({Index('IBXX')})"
