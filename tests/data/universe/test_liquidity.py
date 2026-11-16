# ibexQuant\tests\data\universe\test_liquidity.py

"""Liquidity ranking of B3 shares (ADR 0015, H5.6, H5.9)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from support.universe import write_base

from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.store import DataStore
from ibexQuant.data.universe import UniverseNotBuiltError
from ibexQuant.data.universe.liquidity import (
    TRADED_COLUMNS,
    LiquiditySettings,
    TradedValues,
    base_session,
    read_traded_values,
    select_by_liquidity,
)

DAY = pd.Timestamp


def traded(rows: list[tuple[str, str, float, bool]], calendar: pd.DatetimeIndex) -> TradedValues:
    table = pd.DataFrame(rows, columns=list(TRADED_COLUMNS))
    table = table.assign(session_date=pd.to_datetime(table["session_date"])).sort_values(
        "session_date", kind="stable"
    )
    return TradedValues(table.reset_index(drop=True), calendar)


CALENDAR = pd.bdate_range("2024-01-01", "2024-03-29")
ONE_MONTH = LiquiditySettings(lookback_months=1)


def daily(
    security: str, value: float, eligible: bool = True, every: int = 1
) -> list[tuple[str, str, float, bool]]:
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    return [
        (str(day.date()), security, value, eligible)
        for number, day in enumerate(days)
        if number % every == 0
    ]


def test_settings_are_checked() -> None:
    with pytest.raises(ValueError, match="lookback_months"):
        LiquiditySettings(lookback_months=0)
    with pytest.raises(ValueError, match="min_presence"):
        LiquiditySettings(min_presence=0)
    with pytest.raises(ValueError, match="min_presence"):
        LiquiditySettings(min_presence=1.5)


def test_the_base_session_is_three_sessions_before_the_start() -> None:
    assert base_session(CALENDAR, DAY("2024-03-04")) == DAY("2024-02-28")
    assert base_session(CALENDAR, DAY("2024-01-02")) is None


def test_securities_rank_by_median_traded_value() -> None:
    values = traded(daily("B", 10.0) + daily("A", 30.0) + daily("C", 20.0), CALENDAR)
    assert select_by_liquidity(values, DAY("2024-03-29"), 2, ONE_MONTH) == ["A", "C"]


def test_ties_are_broken_by_security_id() -> None:
    values = traded(daily("B", 10.0) + daily("A", 10.0), CALENDAR)
    assert select_by_liquidity(values, DAY("2024-03-29"), 5, ONE_MONTH) == ["A", "B"]


def test_presence_and_eligibility_filter() -> None:
    rows = daily("A", 10.0) + daily("FUND", 99.0, eligible=False) + daily("RARE", 50.0, every=2)
    values = traded(rows, CALENDAR)
    assert select_by_liquidity(values, DAY("2024-03-29"), 5, ONE_MONTH) == ["A"]
    loose = LiquiditySettings(lookback_months=1, min_presence=0.05)
    # Sessions without trading count as zero in the median.
    assert select_by_liquidity(values, DAY("2024-03-29"), 5, loose) == ["RARE", "A"]


def test_nothing_after_the_base_session_is_seen() -> None:
    late = [
        (str(day.date()), "LATE", 1e9, True) for day in pd.bdate_range("2024-03-01", "2024-03-29")
    ]
    values = traded(daily("A", 10.0) + late, CALENDAR)
    assert select_by_liquidity(values, DAY("2024-02-29"), 5, ONE_MONTH) == ["A"]


def test_a_window_before_the_data_has_no_ranking() -> None:
    values = traded(daily("A", 10.0), CALENDAR)
    assert select_by_liquidity(values, DAY("2024-01-15"), 5, ONE_MONTH) == []
    assert select_by_liquidity(traded([], pd.DatetimeIndex([])), DAY("2024-03-29"), 5) == []


def test_a_window_without_sessions_or_trading_has_no_ranking() -> None:
    sparse = pd.DatetimeIndex(["2024-01-02", "2024-03-29"])
    assert select_by_liquidity(traded([], sparse), DAY("2024-03-15"), 5, ONE_MONTH) == []
    assert select_by_liquidity(traded([], CALENDAR), DAY("2024-03-29"), 5, ONE_MONTH) == []


def test_top_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top"):
        select_by_liquidity(traded([], CALENDAR), DAY("2024-03-29"), 0)


def test_traded_values_are_read_from_the_bars(tmp_path: Path) -> None:
    store = write_base(DataStore.initialize(tmp_path / "data"))
    values = read_traded_values(store, SecurityMaster.load(store))
    table = values.table
    assert tuple(table.columns) == TRADED_COLUMNS
    assert table["session_date"].is_monotonic_increasing
    assert "BRALIEN" not in set(table["security_id"])  # unknown ISIN left out
    day = table[table["session_date"] == DAY("2024-05-02")].set_index("security_id")
    assert day.loc["BRPETR", "traded_value"] == 100e6
    assert not day.loc["BRFUND", "eligible"]
    assert values.calendar[0] == DAY("2024-03-01")


def test_reading_without_bars_says_how_to_build(tmp_path: Path) -> None:
    store = write_base(DataStore.initialize(tmp_path / "data"))
    master = SecurityMaster.load(store)
    shutil.rmtree(store.curated("b3", "bars_daily"))
    with pytest.raises(UniverseNotBuiltError, match="ingest cotahist"):
        read_traded_values(store, master)
