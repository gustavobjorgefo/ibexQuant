# ibexQuant\tests\backtest\walkforward\test_schedule.py

"""Layer 1 of the walk-forward: training and test windows (ADR 0017, B1.4)."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import ibexQuant.backtest
from ibexQuant.backtest.walkforward import WalkForwardError, WalkForwardSchedule, Window
from ibexQuant.data.calendars.calendar import TradingCalendar
from ibexQuant.data.ingestion.provider import DateLike


class BusinessDays(TradingCalendar):
    """Every weekday from ``first`` to ``last``."""

    def __init__(self, first: str = "2015-01-01", last: str = "2026-11-06") -> None:
        self.sessions = pd.bdate_range(first, last)

    def valid_sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        return self.sessions[(self.sessions >= first) & (self.sessions <= last)]


CALENDAR = BusinessDays()


def day(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


# --- windows ---


def test_twelve_and_one() -> None:
    windows = WalkForwardSchedule("2025-01", "2025-03", train_months=12, test_months=1).windows(
        CALENDAR
    )
    assert len(windows) == 3
    assert windows[0] == Window(
        train_start=day("2024-01-01"),
        train_end=day("2024-12-31"),
        decision_start=day("2024-12-31"),
        decision_end=day("2025-01-30"),
        test_start=day("2025-01-01"),
        test_end=day("2025-01-31"),
    )
    # The last test session of a window is decided by the next one.
    assert windows[1].decision_start == windows[0].test_end
    assert windows[1].train_start == day("2024-02-01")


def test_eighteen_and_three() -> None:
    windows = WalkForwardSchedule("2025-01", "2025-12", train_months=18, test_months=3).windows(
        CALENDAR
    )
    assert [w.test_start.month for w in windows] == [1, 4, 7, 10]
    second = windows[1]
    assert second.train_start == day("2023-10-02")  # first weekday of October 2023
    assert second.train_end == day("2025-03-31")
    assert second.test_end == day("2025-06-30")
    assert second.decision_end == day("2025-06-27")


def test_expanding_training_starts_with_the_first_window() -> None:
    schedule = WalkForwardSchedule(
        "2025-01", "2025-06", train_months=12, test_months=2, expanding=True
    )
    windows = schedule.windows(CALENDAR)
    assert schedule.first_train_month == pd.Period("2024-01", "M")
    assert {w.train_start for w in windows} == {day("2024-01-01")}
    assert windows[-1].train_end == day("2025-04-30")


def test_the_last_window_is_cut_at_the_end() -> None:
    windows = WalkForwardSchedule("2025-01", "2025-10", train_months=12, test_months=3).windows(
        CALENDAR
    )
    assert windows[-1].test_start == day("2025-10-01")
    assert windows[-1].test_end == day("2025-10-31")


def test_a_date_stands_for_its_month() -> None:
    schedule = WalkForwardSchedule(
        pd.Timestamp("2025-01-17"), "2025-02-03", train_months=12, test_months=1
    )
    assert schedule.start == pd.Period("2025-01", "M")
    assert (schedule.start, schedule.end) == (pd.Period("2025-01", "M"), pd.Period("2025-02", "M"))


# --- step ---


def test_tiles_overlap_and_gaps() -> None:
    tiled = WalkForwardSchedule("2025-01", "2025-12", train_months=12, test_months=3)
    overlapping = WalkForwardSchedule(
        "2025-01", "2025-03", train_months=12, test_months=3, step_months=1
    )
    gapped = WalkForwardSchedule(
        "2025-01", "2025-12", train_months=12, test_months=1, step_months=3
    )
    assert tiled.tiles
    assert not overlapping.tiles
    assert not gapped.tiles
    starts = [w.test_start.month for w in overlapping.windows(CALENDAR)]
    assert starts == [1, 2, 3]
    assert overlapping.windows(CALENDAR)[0].test_end == day("2025-03-31")
    assert [w.test_start.month for w in gapped.windows(CALENDAR)] == [1, 4, 7, 10]


# --- the calendar's limits ---


def test_a_window_ends_with_the_data() -> None:
    windows = WalkForwardSchedule("2026-10", "2026-11", train_months=12, test_months=1).windows(
        CALENDAR
    )
    assert windows[-1].test_start == day("2026-11-02")
    assert windows[-1].test_end == day("2026-11-06")
    assert windows[-1].decision_end == day("2026-11-05")


def test_windows_without_sessions_are_left_out(caplog: pytest.LogCaptureFixture) -> None:
    schedule = WalkForwardSchedule("2026-10", "2027-03", train_months=12, test_months=1)
    with caplog.at_level(logging.WARNING):
        windows = schedule.windows(CALENDAR)
    assert len(windows) == 2
    assert "4 test window(s) without a session left out, from 2026-12" in caplog.text


def test_no_window_with_sessions_is_an_error() -> None:
    schedule = WalkForwardSchedule("2027-01", "2027-03", train_months=12, test_months=1)
    with pytest.raises(WalkForwardError, match="No test window from 2027-01 to 2027-03"):
        schedule.windows(CALENDAR)


def test_training_before_the_calendar_is_an_error() -> None:
    late = BusinessDays(first="2015-03-16")
    schedule = WalkForwardSchedule("2016-01", "2016-06", train_months=12, test_months=1)
    with pytest.raises(WalkForwardError, match=r"starts in 2015-01, before the calendar"):
        schedule.windows(late)
    message = r"starts in 2015-03, before the calendar \(first session 2015-04-01\)"
    with pytest.raises(WalkForwardError, match=message):
        WalkForwardSchedule("2016-02", "2016-02", train_months=11, test_months=1).windows(
            BusinessDays(first="2015-04-01")
        )


def test_a_training_window_with_no_session_at_all() -> None:
    with pytest.raises(WalkForwardError, match=r"starts in 2014-01, before the calendar\.$"):
        WalkForwardSchedule("2015-01", "2015-01", train_months=12, test_months=1).windows(CALENDAR)


# --- one window ---


def test_window_builds_one_piece() -> None:
    schedule = WalkForwardSchedule("2025-01", "2025-12", train_months=18, test_months=3)
    piece = schedule.window("2025-05", "2025-06", CALENDAR)
    assert piece.train_start == day("2023-11-01")
    assert piece.decision_start == day("2025-04-30")
    assert piece.test_start == day("2025-05-01")
    assert piece.test_end == day("2025-06-30")


def test_a_one_session_window_is_decided_at_the_decision_session() -> None:
    short = BusinessDays(last="2026-11-02")
    piece = WalkForwardSchedule("2026-11", "2026-11", train_months=12, test_months=1).window(
        "2026-11", "2026-11", short
    )
    assert piece.test_start == piece.test_end == day("2026-11-02")
    assert piece.decision_start == piece.decision_end == day("2026-10-30")


def test_window_checks_its_months() -> None:
    schedule = WalkForwardSchedule("2025-01", "2025-12", train_months=12, test_months=1)
    with pytest.raises(ValueError, match="first 2025-06 is after last 2025-05"):
        schedule.window("2025-06", "2025-05", CALENDAR)
    with pytest.raises(WalkForwardError, match="2027-01 to 2027-01 have no session"):
        schedule.window("2027-01", "2027-01", CALENDAR)


# --- frame and repr ---


def test_frame() -> None:
    schedule = WalkForwardSchedule("2025-01", "2025-02", train_months=12, test_months=1)
    frame = schedule.frame(CALENDAR)
    assert frame.columns.tolist() == [
        "train_start",
        "train_end",
        "decision_start",
        "decision_end",
        "test_start",
        "test_end",
        "train_sessions",
        "test_sessions",
    ]
    assert frame["test_sessions"].tolist() == [23, 20]
    assert frame["train_sessions"].iloc[0] == len(pd.bdate_range("2024-01-01", "2024-12-31"))


def test_repr() -> None:
    schedule = WalkForwardSchedule("2025-01", "2025-12", train_months=18, test_months=3)
    assert repr(schedule) == (
        "WalkForwardSchedule(2025-01 to 2025-12, train=18m, test=3m, step=3m, expanding=False)"
    )


# --- validation ---


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"train_months": 0}, "train_months must be at least 1, got 0"),
        ({"test_months": 0}, "test_months must be at least 1, got 0"),
        ({"step_months": 0}, "step_months must be at least 1, got 0"),
        ({"start": "2026-01"}, "start 2026-01 is after end 2025-12"),
        ({"end": "december"}, r"end is not a month \(YYYY-MM\) or a date \(YYYY-MM-DD\)"),
        ({"end": "2025-1"}, "end is not a month"),
        ({"end": "2025-13"}, "end is not a month"),
    ],
)
def test_arguments_are_checked(arguments: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "start": "2025-01",
        "end": "2025-12",
        "train_months": 12,
        "test_months": 1,
    }
    values.update(arguments)
    with pytest.raises(ValueError, match=message):
        WalkForwardSchedule(**values)  # type: ignore[arg-type]


def test_the_package_documents_itself() -> None:
    assert ibexQuant.backtest.__doc__ is not None
