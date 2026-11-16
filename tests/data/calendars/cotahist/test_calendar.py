# ibexQuant\tests\data\calendars\cotahist\test_calendar.py

"""CotahistCalendar: B3 sessions as recorded in the base (ADR 0016, H6.8)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ibexQuant.data.calendars.calendar import TradingCalendar, align_to_calendar
from ibexQuant.data.calendars.cotahist.calendar import CotahistCalendar, CotahistCalendarError
from ibexQuant.data.store import DataStore

DAY = pd.Timestamp
# 2024-12-24/25 and 2024-12-31/2025-01-01: no trading.
SESSIONS = ["2024-12-20", "2024-12-23", "2024-12-26", "2024-12-27", "2024-12-30", "2025-01-02"]


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    store = DataStore.initialize(tmp_path / "data")
    frame = pd.DataFrame({"session_date": pd.to_datetime(SESSIONS * 2), "ticker": "X"})
    for year, rows in frame.groupby(frame["session_date"].dt.year):
        path = store.curated("b3", "bars_daily", f"year={year}", "part-0.parquet")
        path.parent.mkdir(parents=True)
        rows.to_parquet(path, index=False)
    return store


def test_sessions_are_the_days_in_the_base(store: DataStore) -> None:
    calendar = CotahistCalendar(store)
    assert isinstance(calendar, TradingCalendar)
    assert calendar.sessions.tolist() == [DAY(day) for day in SESSIONS]
    assert calendar.sessions.tz is None


def test_valid_sessions_in_a_range(store: DataStore) -> None:
    sessions = CotahistCalendar(store).valid_sessions("2024-12-24", "2024-12-31 15:00")
    assert sessions.tolist() == [DAY("2024-12-26"), DAY("2024-12-27"), DAY("2024-12-30")]


def test_sessions_are_read_once(store: DataStore) -> None:
    calendar = CotahistCalendar(store)
    first = calendar.sessions
    (store.curated("b3", "bars_daily", "year=2025", "part-0.parquet")).unlink()
    assert calendar.sessions is first
    assert len(CotahistCalendar(store).sessions) == 5  # a new object sees the change


def test_offset_counts_sessions_back(store: DataStore) -> None:
    calendar = CotahistCalendar(store)
    assert calendar.offset("2025-01-02", 0) == DAY("2025-01-02")
    assert calendar.offset("2025-01-02", 2) == DAY("2024-12-27")
    assert calendar.offset("2024-12-25", 1) == DAY("2024-12-23")  # from the next session
    assert calendar.offset("2024-12-23", 10) == DAY("2024-12-20")  # the first there is
    assert calendar.offset("2025-06-01", 0) == DAY("2025-01-02")  # the last there is
    with pytest.raises(ValueError, match="bars"):
        calendar.offset("2025-01-02", -1)


def test_alignment_with_the_base_calendar(store: DataStore) -> None:
    bars = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [(DAY("2024-12-20"), "A"), (DAY("2024-12-27"), "A")], names=["timestamp", "symbol"]
        ),
    )
    aligned = align_to_calendar(bars, CotahistCalendar(store))
    assert aligned.index.get_level_values("timestamp").tolist() == [
        DAY("2024-12-20"),
        DAY("2024-12-23"),
        DAY("2024-12-26"),
        DAY("2024-12-27"),
    ]


def test_without_bars_the_calendar_says_how_to_build(tmp_path: Path) -> None:
    calendar = CotahistCalendar(DataStore.initialize(tmp_path / "data"))
    with pytest.raises(CotahistCalendarError, match="ingest cotahist"):
        calendar.valid_sessions("2024-01-01", "2024-12-31")
