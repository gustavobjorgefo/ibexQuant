# ibexQuant\tests\data\calendars\pandas_market_calendars\test_calendar.py

"""
Unit tests for PandasMarketCalendarsCalendar.

These use the real pandas_market_calendars package rather than a mock:
it ships all calendar rules as static code and makes no network calls,
so exercising it directly is fast, deterministic, and actually confirms
the wrapper behaves as documented.
"""

from __future__ import annotations

import pandas as pd

from ibexQuant.data.calendars.calendar import align_to_calendar
from ibexQuant.data.calendars.pandas_market_calendars.calendar import (
    PandasMarketCalendarsCalendar,
)


def test_valid_sessions_returns_naive_timestamps() -> None:
    calendar = PandasMarketCalendarsCalendar("B3")
    sessions = calendar.valid_sessions("2024-01-01", "2024-01-05")

    assert sessions.tz is None


def test_valid_sessions_excludes_new_years_day() -> None:
    calendar = PandasMarketCalendarsCalendar("B3")
    sessions = calendar.valid_sessions("2024-01-01", "2024-01-05")

    assert pd.Timestamp("2024-01-01") not in sessions
    assert pd.Timestamp("2024-01-02") in sessions


def test_valid_sessions_excludes_weekends() -> None:
    calendar = PandasMarketCalendarsCalendar("B3")
    sessions = calendar.valid_sessions("2024-01-01", "2024-01-08")

    assert pd.Timestamp("2024-01-06") not in sessions  # Saturday
    assert pd.Timestamp("2024-01-07") not in sessions  # Sunday


def test_align_to_calendar_reveals_a_real_gap_against_the_real_b3_calendar() -> None:
    df = pd.DataFrame(
        {"close": [10.0, 12.0]},
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "PETR4.SA"),
                (pd.Timestamp("2024-01-04"), "PETR4.SA"),
            ],
            names=["timestamp", "symbol"],
        ),
    )

    aligned = align_to_calendar(df, PandasMarketCalendarsCalendar("B3"))

    assert pd.isna(aligned.loc[(pd.Timestamp("2024-01-03"), "PETR4.SA"), "close"])
