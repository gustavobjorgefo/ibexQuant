# ibexQuant\src\ibexQuant\data\universe\calendar.py

"""
Validity periods of the B3 index portfolios (ADR 0015, H5.3).

Responsibility
--------------
Compute, from the B3 session calendar, when each four-month portfolio
(January, May, September) starts and ends, and read the one-off changes of
``index_periods.yaml``.

Contract
--------
- The start of a period follows the rule version in force for it
  (:data:`RULE_VERSIONS`): the first session of the month up to 2011-05, the
  first Monday from 2011-09, the second Monday from 2027-01; a Monday without
  trading moves the start to the next session.
- A period ends on the session before the next period starts; the last
  period known ends on the last session of the calendar.
- A period is labelled ``YYYY-MM`` by the month it starts in.
- Only periods whose rule date falls after the first session are built:
  without a session before it, the first session of the data is not
  necessarily the start of a period.
- An override replaces the start of one period; it must be a session, and
  the periods must stay in order.

What does not belong here
-------------------------
- Exact publication dates of previews, for a rebalance trade: they are
  checked against B3's notices, outside the base.
- Which securities belong to a period — see ``constituents``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

import pandas as pd
import yaml

from ibexQuant.data.reference.overrides import OverridesFormatError

PERIODS_FILE: Final[str] = "index_periods.yaml"
SCHEMA_VERSION: Final[int] = 1
PERIOD_MONTHS: Final[tuple[int, ...]] = (1, 5, 9)
PERIOD_COLUMNS: Final[tuple[str, ...]] = ("period", "effective_from", "effective_to", "rule")
OVERRIDE_RULE: Final[str] = "override"

_PERIOD_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\d{4})-(01|05|09)$")
_TOP_KEYS: Final[frozenset[str]] = frozenset({"version", "periods"})
_ENTRY_KEYS: Final[frozenset[str]] = frozenset({"period", "start", "note"})
_MONDAY: Final[int] = 0


class StartRule(StrEnum):
    """How the first session of a period is chosen."""

    FIRST_SESSION = "first_session"
    FIRST_MONDAY = "first_monday"
    SECOND_MONDAY = "second_monday"


@dataclass(frozen=True, slots=True)
class RuleVersion:
    """A start rule and the first period it applies to (``None``: from the beginning)."""

    first_period: str | None
    rule: StartRule
    source: str


RULE_VERSIONS: Final[tuple[RuleVersion, ...]] = (
    RuleVersion(None, StartRule.FIRST_SESSION, "Ibovespa methodology before 2011"),
    RuleVersion("2011-09", StartRule.FIRST_MONDAY, "Ofício Circular 007/2011-DT"),
    RuleVersion("2027-01", StartRule.SECOND_MONDAY, "Ofício Circular 068/2026-VPC"),
)


@dataclass(frozen=True, slots=True)
class PeriodOverride:
    """A one-off start set by B3, outside the rule."""

    period: str
    start: pd.Timestamp
    note: str


@dataclass(frozen=True, slots=True)
class PeriodOverrides:
    """The entries of ``index_periods.yaml`` and the hash of the file read."""

    entries: tuple[PeriodOverride, ...]
    sha256: str | None

    @classmethod
    def none(cls) -> Self:
        """No overrides."""
        return cls((), None)


# --- Labels and rules ---


def period_label(year: int, month: int) -> str:
    """
    Return the label ``YYYY-MM`` of the period starting in ``month`` of ``year``.

    Raises
    ------
    ValueError
        If ``month`` is not January, May or September.
    """
    if month not in PERIOD_MONTHS:
        raise ValueError(f"Index periods start in January, May or September, not month {month}.")
    return f"{year:04d}-{month:02d}"


def parse_period(label: str) -> tuple[int, int]:
    """
    Return ``(year, month)`` of a label ``YYYY-MM``.

    Raises
    ------
    ValueError
        If the label is not a year and January, May or September.
    """
    match = _PERIOD_PATTERN.match(label)
    if match is None:
        raise ValueError(f"{label!r} is not a period label (YYYY-01, YYYY-05 or YYYY-09).")
    return int(match.group(1)), int(match.group(2))


def rule_for(period: str, versions: Sequence[RuleVersion] = RULE_VERSIONS) -> StartRule:
    """Return the start rule in force for ``period``."""
    rule = versions[0].rule
    for version in versions:
        if version.first_period is None or version.first_period <= period:
            rule = version.rule
    return rule


def rule_date(year: int, month: int, rule: StartRule) -> dt.date:
    """
    Return the calendar date a rule points to, before skipping days without trading.

    The first session of the month is looked for from the 1st; the Mondays are
    calendar Mondays.
    """
    first = dt.date(year, month, 1)
    if rule is StartRule.FIRST_SESSION:
        return first
    first_monday = first + dt.timedelta(days=(_MONDAY - first.weekday()) % 7)
    if rule is StartRule.FIRST_MONDAY:
        return first_monday
    return first_monday + dt.timedelta(days=7)


# --- Periods ---


def index_periods(
    sessions: pd.DatetimeIndex,
    overrides: PeriodOverrides | None = None,
    versions: Sequence[RuleVersion] = RULE_VERSIONS,
) -> pd.DataFrame:
    """
    Compute the periods covered by a session calendar.

    Parameters
    ----------
    sessions : pandas.DatetimeIndex
        Trading sessions of B3, any order; duplicates are ignored.
    overrides : PeriodOverrides, optional
        One-off starts; an override for a period not covered is ignored.
    versions : sequence of RuleVersion
        Rule versions in time order; :data:`RULE_VERSIONS` by default.

    Returns
    -------
    pandas.DataFrame
        One row per period, in time order: ``period``, ``effective_from``,
        ``effective_to``, ``rule``.

    Raises
    ------
    ValueError
        If an override's start is not a session, or the starts are not in order.
    """
    calendar = sessions.normalize().unique().sort_values()
    if calendar.empty:
        return _empty_periods()
    first, last = calendar[0], calendar[-1]
    moved = {entry.period: entry for entry in (overrides or PeriodOverrides.none()).entries}

    labels: list[str] = []
    starts: list[pd.Timestamp] = []
    rules: list[str] = []
    for year in range(first.year, last.year + 1):
        for month in PERIOD_MONTHS:
            label = period_label(year, month)
            rule = rule_for(label, versions)
            anchor = pd.Timestamp(rule_date(year, month, rule))
            if anchor <= first:
                continue
            if label in moved:
                start = moved[label].start
                if start > last:
                    continue
                if start not in calendar:
                    raise ValueError(
                        f"{PERIODS_FILE}: {label} start {start.date()} is no session."
                    )
                labels.append(label)
                starts.append(start)
                rules.append(OVERRIDE_RULE)
                continue
            position = calendar.searchsorted(anchor)
            if position == len(calendar):
                continue
            labels.append(label)
            starts.append(calendar[int(position)])
            rules.append(rule.value)

    if not labels:
        return _empty_periods()
    start_index = pd.DatetimeIndex(starts)
    if not start_index.is_monotonic_increasing or start_index.has_duplicates:
        raise ValueError(
            f"Period starts are out of order: {dict(zip(labels, starts, strict=True))}."
        )
    ends = [calendar[int(calendar.searchsorted(start)) - 1] for start in starts[1:]] + [last]
    return pd.DataFrame(
        {
            "period": labels,
            "effective_from": start_index,
            "effective_to": pd.DatetimeIndex(ends),
            "rule": rules,
        }
    )


def assign_periods(sessions: pd.DatetimeIndex, periods: pd.DataFrame) -> pd.Series:
    """
    Return the period in force at each session (missing before the first period).

    Parameters
    ----------
    sessions : pandas.DatetimeIndex
        Sessions to label, any order.
    periods : pandas.DataFrame
        The output of :func:`index_periods`.
    """
    starts = pd.DatetimeIndex(periods["effective_from"])
    positions = starts.searchsorted(sessions.normalize(), side="right") - 1
    labels = periods["period"].to_numpy(dtype=object)
    values = [labels[position] if position >= 0 else None for position in positions]
    return pd.Series(values, index=sessions, dtype="string", name="period")


def _empty_periods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": pd.Series(dtype="string"),
            "effective_from": pd.Series(dtype="datetime64[ns]"),
            "effective_to": pd.Series(dtype="datetime64[ns]"),
            "rule": pd.Series(dtype="string"),
        }
    )


# --- Overrides file ---


def load_period_overrides(root: Path) -> PeriodOverrides:
    """
    Read ``index_periods.yaml`` from the overrides root, if present.

    The file holds ``version: 1`` and ``periods``, a list of entries with
    ``period`` (``YYYY-MM``), ``start`` (ISO date) and ``note``.

    Raises
    ------
    OverridesFormatError
        If the file is not valid YAML or breaks the schema.
    """
    path = root / PERIODS_FILE
    if not path.is_file():
        return PeriodOverrides.none()
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise OverridesFormatError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(document, dict):
        raise OverridesFormatError(f"{path} must be a mapping with 'version: 1'.")
    unknown = sorted(set(document) - _TOP_KEYS)
    if unknown:
        raise OverridesFormatError(f"{path}: unknown key(s) {unknown}.")
    if document.get("version") != SCHEMA_VERSION:
        raise OverridesFormatError(
            f"{path}: version must be {SCHEMA_VERSION}, got {document.get('version')!r}."
        )
    entries = document.get("periods")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise OverridesFormatError(f"{path}: 'periods' must be a list.")
    parsed = tuple(
        _override(entry, f"{path}: periods[{index}]") for index, entry in enumerate(entries)
    )
    seen: set[str] = set()
    for entry in parsed:
        if entry.period in seen:
            raise OverridesFormatError(f"{path}: period {entry.period} is set twice.")
        seen.add(entry.period)
    return PeriodOverrides(parsed, hashlib.sha256(raw).hexdigest())


def _override(entry: object, where: str) -> PeriodOverride:
    if not isinstance(entry, dict):
        raise OverridesFormatError(f"{where} must be a mapping.")
    keys = set(entry)
    if keys != _ENTRY_KEYS:
        raise OverridesFormatError(
            f"{where} needs exactly {sorted(_ENTRY_KEYS)}, got {sorted(keys)}."
        )
    label = str(entry["period"])
    try:
        parse_period(label)
    except ValueError as error:
        raise OverridesFormatError(f"{where}: {error}") from error
    start = entry["start"]
    if not isinstance(start, dt.date) or isinstance(start, dt.datetime):
        raise OverridesFormatError(f"{where}: start must be an ISO date, got {start!r}.")
    note = entry["note"]
    if not isinstance(note, str) or not note.strip():
        raise OverridesFormatError(f"{where}: note must be a non-empty text.")
    return PeriodOverride(label, pd.Timestamp(start), note.strip())
