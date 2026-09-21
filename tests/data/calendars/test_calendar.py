# ibexQuant\tests\data\calendars\test_calendar.py

"""
Unit tests for align_to_calendar, using a stub TradingCalendar so these
never depend on pandas_market_calendars or any real exchange rules.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ibexQuant.data.calendars.calendar import TradingCalendar, align_to_calendar

# Deliberately skips 2024-01-06/07 (a weekend) to simulate a real calendar.
_ALL_SESSIONS = pd.DatetimeIndex(
    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
)


class _StubCalendar(TradingCalendar):
    """A calendar with a fixed, known set of valid sessions, for testing."""

    def __init__(self, sessions: pd.DatetimeIndex = _ALL_SESSIONS) -> None:
        self._sessions = sessions

    def valid_sessions(self, start: object, end: object) -> pd.DatetimeIndex:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        return self._sessions[(self._sessions >= start_ts) & (self._sessions <= end_ts)]


def _make_bars(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.set_index(["timestamp", "symbol"]).sort_index()


# --- shape validation ---


def test_align_to_calendar_rejects_non_multiindex_frame() -> None:
    df = pd.DataFrame({"close": [10.0]})
    with pytest.raises(ValueError, match="MultiIndex"):
        align_to_calendar(df, _StubCalendar())


# --- default range is per symbol, not global ---


def test_align_to_calendar_uses_each_symbols_own_range_by_default() -> None:
    df = _make_bars(
        [
            {"timestamp": "2024-01-02", "symbol": "OLD", "close": 10.0},
            {"timestamp": "2024-01-08", "symbol": "OLD", "close": 11.0},
            {"timestamp": "2024-01-04", "symbol": "NEW", "close": 20.0},
            {"timestamp": "2024-01-05", "symbol": "NEW", "close": 21.0},
        ]
    )
    aligned = align_to_calendar(df, _StubCalendar())

    new_timestamps = list(aligned.xs("NEW", level="symbol").index)
    assert new_timestamps == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]


def test_align_to_calendar_reveals_a_real_gap_as_nan() -> None:
    df = _make_bars(
        [
            {"timestamp": "2024-01-02", "symbol": "PETR4.SA", "close": 10.0},
            # 2024-01-03 is missing here: a real gap, not a holiday.
            {"timestamp": "2024-01-04", "symbol": "PETR4.SA", "close": 12.0},
        ]
    )
    aligned = align_to_calendar(df, _StubCalendar())

    gap_value = aligned.loc[(pd.Timestamp("2024-01-03"), "PETR4.SA"), "close"]
    assert pd.isna(gap_value)


def test_align_to_calendar_leaves_a_fully_covered_symbol_unchanged() -> None:
    df = _make_bars(
        [
            {"timestamp": str(ts.date()), "symbol": "PETR4.SA", "close": float(i)}
            for i, ts in enumerate(_ALL_SESSIONS)
        ]
    )
    aligned = align_to_calendar(df, _StubCalendar())

    pd.testing.assert_frame_equal(aligned, df)


# --- explicit range overrides the per-symbol default ---


def test_align_to_calendar_explicit_range_applies_to_every_symbol() -> None:
    df = _make_bars([{"timestamp": "2024-01-04", "symbol": "PETR4.SA", "close": 10.0}])

    aligned = align_to_calendar(df, _StubCalendar(), start="2024-01-02", end="2024-01-05")

    assert len(aligned) == 4  # 01-02, 01-03, 01-04, 01-05


# --- a calendar per symbol ---


def test_align_to_calendar_accepts_a_calendar_per_symbol() -> None:
    df = _make_bars(
        [
            {"timestamp": "2024-01-02", "symbol": "A", "close": 1.0},
            {"timestamp": "2024-01-02", "symbol": "B", "close": 2.0},
        ]
    )
    calendars = {"A": _StubCalendar(), "B": _StubCalendar(pd.DatetimeIndex(["2024-01-02"]))}

    aligned = align_to_calendar(df, calendars, start="2024-01-02", end="2024-01-02")

    assert len(aligned) == 2


def test_align_to_calendar_raises_for_symbol_missing_from_mapping() -> None:
    df = _make_bars([{"timestamp": "2024-01-02", "symbol": "UNMAPPED", "close": 1.0}])

    with pytest.raises(ValueError, match="No calendar provided"):
        align_to_calendar(df, {"OTHER": _StubCalendar()})


# --- timestamps outside valid sessions ---


def test_align_to_calendar_drops_and_logs_timestamps_outside_valid_sessions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = _make_bars(
        [
            {"timestamp": "2024-01-02", "symbol": "PETR4.SA", "close": 10.0},
            {"timestamp": "2024-01-06", "symbol": "PETR4.SA", "close": 99.0},  # a Saturday
        ]
    )
    with caplog.at_level("WARNING"):
        aligned = align_to_calendar(df, _StubCalendar(), start="2024-01-02", end="2024-01-08")

    assert pd.Timestamp("2024-01-06") not in aligned.index.get_level_values("timestamp")
    assert "outside its calendar's valid sessions" in caplog.text
