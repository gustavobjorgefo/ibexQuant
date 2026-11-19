# ibexQuant\src\ibexQuant\backtest\walkforward\schedule.py

"""
Layer 1 of the walk-forward: when a strategy is fitted and when it trades (ADR 0017, B1.4).

Responsibility
--------------
Cut a trading calendar into windows: a test window of whole calendar months, the training
window of the months before it, and the sessions whose close makes the test window's
orders.

Contract
--------
- Durations are calendar months, free: 12 and 1, 18 and 3, 36 and 6. Windows start
  every ``step_months`` (default ``test_months``); a training window is rolling, or
  starts with the first one when ``expanding``.
- Only ``TradingCalendar.valid_sessions`` is used: any calendar works, and no bar is
  ever counted.
- Decisions are taken at the close and executed in the next session. The decision
  session is the last session before the test window; it ends the training window.
- The last window is cut at ``end`` (D1). A window whose test months have no session yet
  is left out with a warning; if none has a session, :class:`WalkForwardError` (D2). A
  training window that starts before the calendar raises :class:`WalkForwardError` (D3).
- Overlapping or gapped test windows are allowed; ``tiles`` says whether they cover the
  range exactly once (D4).

What does not belong here
-------------------------
- Securities, indices and eligibility: layers 2 and 3.
- Bars: a window holds only dates; layer 4 loads the data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

import pandas as pd

from ibexQuant.backtest.walkforward.errors import WalkForwardError
from ibexQuant.data.calendars.calendar import TradingCalendar
from ibexQuant.data.ingestion.provider import DateLike

logger: logging.Logger = logging.getLogger(__name__)

MONTH: Final[str] = "M"
# pandas reads loose text ("december" is December of year 1): only ISO months and dates pass.
MONTH_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}(-\d{2})?")
WINDOW_COLUMNS: Final[tuple[str, ...]] = (
    "train_start",
    "train_end",
    "decision_start",
    "decision_end",
    "test_start",
    "test_end",
    "train_sessions",
    "test_sessions",
)


@dataclass(frozen=True)
class Window:
    """
    One training window and the test window that follows it.

    Attributes
    ----------
    train_start, train_end : pandas.Timestamp
        First and last session of the training window; ``train_end`` is the
        decision session.
    decision_start, decision_end : pandas.Timestamp
        Sessions whose close makes the window's orders: from the decision session
        to the second to last test session.
    test_start, test_end : pandas.Timestamp
        Sessions where those orders are executed and held.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    decision_start: pd.Timestamp
    decision_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSchedule:
    """
    Training and test windows over calendar months.

    Parameters
    ----------
    start, end : str or pandas.Timestamp
        First and last test month: ``"2019-01"``, or a date (``"2019-01-15"`` or a
        timestamp) standing for its month.
    train_months : int
        Calendar months of the training window.
    test_months : int
        Calendar months of each test window.
    step_months : int, optional
        Months between the starts of two test windows; defaults to ``test_months``.
    expanding : bool, default False
        Every training window starts with the first one, instead of rolling.

    Raises
    ------
    ValueError
        If a duration is below one month, ``start`` is after ``end`` or a month
        cannot be read.

    Examples
    --------
    >>> schedule = WalkForwardSchedule("2019-01", "2026-10", train_months=18, test_months=3)
    >>> windows = schedule.windows(ds.calendar)  # doctest: +SKIP
    """

    def __init__(
        self,
        start: DateLike,
        end: DateLike,
        *,
        train_months: int,
        test_months: int,
        step_months: int | None = None,
        expanding: bool = False,
    ) -> None:
        step = test_months if step_months is None else step_months
        for name, value in (
            ("train_months", train_months),
            ("test_months", test_months),
            ("step_months", step),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}.")
        self.start: pd.Period = _month(start, "start")
        self.end: pd.Period = _month(end, "end")
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}.")
        self.train_months: int = train_months
        self.test_months: int = test_months
        self.step_months: int = step
        self.expanding: bool = expanding

    # --- properties ---

    @property
    def tiles(self) -> bool:
        """Whether the test windows cover the range once, without overlap or gap."""
        return self.step_months == self.test_months

    @property
    def first_train_month(self) -> pd.Period:
        """First month of the first training window, where every expanding one starts."""
        return self.start - self.train_months

    # --- windows ---

    def windows(self, calendar: TradingCalendar) -> list[Window]:
        """
        Return the windows of the schedule, in order.

        Raises
        ------
        WalkForwardError
            If no test window has a session, or a training window starts before the
            calendar.
        """
        result: list[Window] = []
        missing: list[pd.Period] = []
        first = self.start
        while first <= self.end:
            last = min(first + (self.test_months - 1), self.end)
            window = self._build(first, last, calendar)
            if window is None:
                missing.append(first)
            else:
                result.append(window)
            first += self.step_months
        if not result:
            raise WalkForwardError(
                f"No test window from {self.start} to {self.end} has a session in the calendar."
            )
        if missing:
            logger.warning(
                "%d test window(s) without a session left out, from %s.", len(missing), missing[0]
            )
        return result

    def window(
        self, first: DateLike | pd.Period, last: DateLike | pd.Period, calendar: TradingCalendar
    ) -> Window:
        """
        Return the window testing the months ``first`` to ``last``.

        The same rule as :meth:`windows`, for one range of test months: layer 2 uses it
        to cut a window where an index portfolio changes.

        Raises
        ------
        ValueError
            If ``first`` is after ``last``.
        WalkForwardError
            If the test months have no session, or the training window starts before
            the calendar.
        """
        first_month, last_month = _month(first, "first"), _month(last, "last")
        if first_month > last_month:
            raise ValueError(f"first {first_month} is after last {last_month}.")
        window = self._build(first_month, last_month, calendar)
        if window is None:
            raise WalkForwardError(
                f"The test months {first_month} to {last_month} have no session in the calendar."
            )
        return window

    def frame(self, calendar: TradingCalendar) -> pd.DataFrame:
        """Return the windows as a frame, one row each, with their session counts."""
        rows = []
        for window in self.windows(calendar):
            train = calendar.valid_sessions(window.train_start, window.train_end)
            test = calendar.valid_sessions(window.test_start, window.test_end)
            rows.append({**vars(window), "train_sessions": len(train), "test_sessions": len(test)})
        return pd.DataFrame(rows, columns=list(WINDOW_COLUMNS))

    def __repr__(self) -> str:
        return (
            f"WalkForwardSchedule({self.start} to {self.end}, train={self.train_months}m, "
            f"test={self.test_months}m, step={self.step_months}m, expanding={self.expanding})"
        )

    # --- helpers ---

    def _build(
        self, first: pd.Period, last: pd.Period, calendar: TradingCalendar
    ) -> Window | None:
        test = calendar.valid_sessions(first.start_time, last.end_time.normalize())
        if test.empty:
            return None
        train_first = self.first_train_month if self.expanding else first - self.train_months
        train = calendar.valid_sessions(train_first.start_time, (first - 1).end_time.normalize())
        if train.empty or train[0].to_period(MONTH) != train_first:
            raise WalkForwardError(
                f"The training window of {first} starts in {train_first}, before the "
                f"calendar{'' if train.empty else f' (first session {train[0].date()})'}."
            )
        decisions_end = test[-2] if len(test) > 1 else train[-1]
        return Window(
            train_start=train[0],
            train_end=train[-1],
            decision_start=train[-1],
            decision_end=decisions_end,
            test_start=test[0],
            test_end=test[-1],
        )


def _month(value: DateLike | pd.Period, name: str) -> pd.Period:
    """Read a month from ``"YYYY-MM"``, ``"YYYY-MM-DD"``, a timestamp or a period."""
    message = f"{name} is not a month (YYYY-MM) or a date (YYYY-MM-DD): {value!r}."
    if isinstance(value, str) and MONTH_PATTERN.fullmatch(value) is None:
        raise ValueError(message)
    try:
        return pd.Period(value, freq=MONTH)
    except ValueError as error:
        raise ValueError(message) from error
