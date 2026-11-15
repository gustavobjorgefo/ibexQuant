# ibexQuant\tests\data\universe\test_index_periods.py

"""Index periods from the B3 calendar and ``index_periods.yaml`` (ADR 0015, H5.3)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from ibexQuant.data.reference.overrides import OverridesFormatError
from ibexQuant.data.universe.calendar import (
    OVERRIDE_RULE,
    PERIODS_FILE,
    PeriodOverride,
    PeriodOverrides,
    StartRule,
    assign_periods,
    index_periods,
    load_period_overrides,
    parse_period,
    period_label,
    rule_date,
    rule_for,
)

# Days without trading that matter for the starts and ends checked below.
HOLIDAYS = [
    "2018-01-01", "2018-05-01", "2018-12-24", "2018-12-25", "2018-12-31", "2019-01-01",
    "2020-09-07", "2022-12-30", "2023-05-01", "2023-12-25", "2023-12-29", "2024-01-01",
    "2026-09-07", "2027-01-01",
]  # fmt: skip

# The 21 starts of the hand-kept table, 2017-09 to 2024-05.
KNOWN_STARTS = [
    "2017-09-04", "2018-01-02", "2018-05-07", "2018-09-03", "2019-01-07", "2019-05-06",
    "2019-09-02", "2020-01-06", "2020-05-04", "2020-09-08", "2021-01-04", "2021-05-03",
    "2021-09-06", "2022-01-03", "2022-05-02", "2022-09-05", "2023-01-02", "2023-05-02",
    "2023-09-04", "2024-01-02", "2024-05-06",
]  # fmt: skip


def sessions(start: str, end: str) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    return days[~days.isin(pd.DatetimeIndex(HOLIDAYS))]


def periods_of(start: str, end: str, overrides: PeriodOverrides | None = None) -> pd.DataFrame:
    return index_periods(sessions(start, end), overrides).set_index("period")


# --- labels and rules ---


def test_labels_round_trip() -> None:
    assert period_label(2025, 9) == "2025-09"
    assert parse_period("2025-09") == (2025, 9)


@pytest.mark.parametrize("bad", ["2025-03", "25-09", "2025-9"])
def test_a_label_must_be_january_may_or_september(bad: str) -> None:
    with pytest.raises(ValueError, match="not a period label"):
        parse_period(bad)


def test_a_period_cannot_start_in_another_month() -> None:
    with pytest.raises(ValueError, match="not month 3"):
        period_label(2025, 3)


def test_rule_versions() -> None:
    assert rule_for("2011-05") is StartRule.FIRST_SESSION
    assert rule_for("2011-09") is StartRule.FIRST_MONDAY
    assert rule_for("2026-09") is StartRule.FIRST_MONDAY
    assert rule_for("2027-01") is StartRule.SECOND_MONDAY


def test_rule_dates() -> None:
    assert rule_date(2010, 9, StartRule.FIRST_SESSION) == dt.date(2010, 9, 1)
    assert rule_date(2010, 9, StartRule.FIRST_MONDAY) == dt.date(2010, 9, 6)
    assert rule_date(2018, 1, StartRule.FIRST_MONDAY) == dt.date(2018, 1, 1)
    assert rule_date(2027, 1, StartRule.SECOND_MONDAY) == dt.date(2027, 1, 11)


# --- periods ---


def test_the_rule_reproduces_the_hand_kept_starts() -> None:
    table = periods_of("2017-06-01", "2024-08-30")
    assert table["effective_from"].dt.strftime("%Y-%m-%d").tolist() == KNOWN_STARTS


def test_a_period_ends_on_the_session_before_the_next_start() -> None:
    table = periods_of("2017-06-01", "2024-08-30")
    # The three ends the hand-kept table had wrong.
    assert table.loc["2018-09", "effective_to"] == pd.Timestamp("2019-01-04")
    assert table.loc["2022-09", "effective_to"] == pd.Timestamp("2022-12-29")
    assert table.loc["2023-09", "effective_to"] == pd.Timestamp("2023-12-28")


def test_the_last_period_ends_on_the_last_session() -> None:
    table = periods_of("2024-01-01", "2024-08-30")
    assert table.index[-1] == "2024-05"
    assert table.loc["2024-05", "effective_to"] == pd.Timestamp("2024-08-30")


def test_the_rule_changes_between_versions() -> None:
    table = periods_of("2010-06-01", "2011-12-30")
    assert table.loc["2010-09", "effective_from"] == pd.Timestamp("2010-09-01")
    assert table.loc["2011-01", "effective_from"] == pd.Timestamp("2011-01-03")
    assert table.loc["2011-05", "effective_from"] == pd.Timestamp("2011-05-02")
    assert table.loc["2011-09", "effective_from"] == pd.Timestamp("2011-09-05")
    assert table.loc["2011-09", "rule"] == "first_monday"
    assert table.loc["2010-09", "rule"] == "first_session"


def test_from_2027_the_start_is_the_second_monday() -> None:
    table = periods_of("2026-08-03", "2027-05-31")
    assert table.loc["2026-09", "effective_from"] == pd.Timestamp("2026-09-08")
    assert table.loc["2027-01", "effective_from"] == pd.Timestamp("2027-01-11")
    assert table.loc["2026-09", "effective_to"] == pd.Timestamp("2027-01-08")
    assert table.loc["2027-05", "effective_from"] == pd.Timestamp("2027-05-10")


def test_no_period_starts_on_or_before_the_first_session() -> None:
    table = periods_of("2018-01-02", "2018-06-29")
    assert table.index.tolist() == ["2018-05"]


def test_a_period_whose_start_is_not_in_the_calendar_yet_is_left_out() -> None:
    table = periods_of("2026-08-03", "2026-12-30")
    assert table.index.tolist() == ["2026-09"]


def test_an_empty_calendar_has_no_periods() -> None:
    assert index_periods(pd.DatetimeIndex([])).empty
    assert index_periods(sessions("2018-02-01", "2018-03-30")).empty


def test_duplicate_and_unordered_sessions_are_accepted() -> None:
    days = sessions("2018-03-01", "2018-06-29")
    table = index_periods(days.append(days[::-1]))
    assert table["period"].tolist() == ["2018-05"]


# --- overrides ---


def moved(period: str, start: str) -> PeriodOverrides:
    return PeriodOverrides((PeriodOverride(period, pd.Timestamp(start), "test"),), "hash")


def test_an_override_moves_one_start() -> None:
    table = periods_of("2018-03-01", "2018-10-31", moved("2018-05", "2018-05-14"))
    assert table.loc["2018-05", "effective_from"] == pd.Timestamp("2018-05-14")
    assert table.loc["2018-05", "rule"] == OVERRIDE_RULE


def test_an_override_beyond_the_calendar_is_ignored() -> None:
    table = periods_of("2018-03-01", "2018-06-29", moved("2018-09", "2018-09-10"))
    assert table.index.tolist() == ["2018-05"]


def test_an_override_must_be_a_session() -> None:
    with pytest.raises(ValueError, match="is no session"):
        periods_of("2018-03-01", "2018-06-29", moved("2018-05", "2018-05-01"))


def test_an_override_cannot_reorder_the_periods() -> None:
    with pytest.raises(ValueError, match="out of order"):
        periods_of("2018-03-01", "2018-10-31", moved("2018-05", "2018-09-04"))


# --- assigning sessions ---


def test_each_session_gets_the_period_in_force() -> None:
    table = index_periods(sessions("2018-03-01", "2018-10-31"))
    days = pd.DatetimeIndex(["2018-04-30", "2018-05-07", "2018-08-31", "2018-09-03"])
    labels = assign_periods(days, table)
    assert labels.isna().tolist() == [True, False, False, False]
    assert labels.iloc[1:].tolist() == ["2018-05", "2018-05", "2018-09"]


# --- the overrides file ---


def write(tmp_path: Path, text: str) -> Path:
    (tmp_path / PERIODS_FILE).write_text(text, encoding="utf-8")
    return tmp_path


def test_without_the_file_there_are_no_overrides(tmp_path: Path) -> None:
    assert load_period_overrides(tmp_path) == PeriodOverrides.none()


def test_the_file_is_read(tmp_path: Path) -> None:
    root = write(
        tmp_path,
        "version: 1\nperiods:\n  - {period: 2027-01, start: 2027-01-11, note: ' OC 068 '}\n",
    )
    overrides = load_period_overrides(root)
    assert overrides.entries == (PeriodOverride("2027-01", pd.Timestamp("2027-01-11"), "OC 068"),)
    assert overrides.sha256 is not None


def test_an_empty_list_is_no_overrides(tmp_path: Path) -> None:
    assert load_period_overrides(write(tmp_path, "version: 1\n")).entries == ()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("version: [1\n", "not valid YAML"),
        ("- 1\n", "must be a mapping"),
        ("version: 1\nother: 1\n", "unknown key"),
        ("version: 2\n", "version must be 1"),
        ("version: 1\nperiods: {}\n", "must be a list"),
        ("version: 1\nperiods: [1]\n", "must be a mapping"),
        ("version: 1\nperiods: [{period: 2027-01, start: 2027-01-11}]\n", "needs exactly"),
        ("version: 1\nperiods: [{period: 2027-02, start: 2027-02-01, note: x}]\n", "period label"),
        ("version: 1\nperiods: [{period: 2027-01, start: soon, note: x}]\n", "ISO date"),
        (
            "version: 1\nperiods: [{period: 2027-01, start: 2027-01-11 10:00:00, note: x}]\n",
            "ISO date",
        ),
        ("version: 1\nperiods: [{period: 2027-01, start: 2027-01-11, note: ' '}]\n", "note"),
        (
            "version: 1\nperiods:\n"
            "  - {period: 2027-01, start: 2027-01-11, note: x}\n"
            "  - {period: 2027-01, start: 2027-01-12, note: y}\n",
            "set twice",
        ),
    ],
)
def test_a_malformed_file_is_refused(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(OverridesFormatError, match=message):
        load_period_overrides(write(tmp_path, text))
