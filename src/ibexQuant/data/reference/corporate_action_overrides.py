# ibexQuant\src\ibexQuant\data\reference\corporate_action_overrides.py

"""
Hand-made corporate action decisions: ``corporate_actions.yaml`` (ADR 0013, H3.11).

Responsibility
--------------
Read and validate the shape of the file in the overrides root: events to
add (of any type, exchanges and conversions included), source events to
reject, and ex-sessions accepted without an event. Every entry has a note.

Contract
--------
- The file is optional: without it there are no decisions.
- Dates are ISO (``2024-04-30``); types are :class:`EventType` names; cash
  types need ``cash_amount``, the others ``share_ratio``; spin-offs,
  exchanges and conversions also need ``target_isin``.
- Whether an entry matches something in the data is checked by the build.

What does not belong here
-------------------------
- Applying the decisions — see ``corporate_actions``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import pandas as pd
import yaml

from ibexQuant.data.reference.corporate_action_sources import (
    SOURCE_EVENT_COLUMNS,
    Source,
)
from ibexQuant.data.reference.event_types import CASH_TYPES, EventType
from ibexQuant.data.reference.overrides import OverridesFormatError

ACTIONS_FILE: Final[str] = "corporate_actions.yaml"
SCHEMA_VERSION: Final[int] = 1
_TOP_KEYS: Final[frozenset[str]] = frozenset({"version", "events", "rejections", "accepted"})
_TARGET_TYPES: Final[frozenset[EventType]] = frozenset(
    {EventType.SPIN_OFF, EventType.EXCHANGE, EventType.CONVERSION}
)


@dataclass(frozen=True, slots=True)
class Rejection:
    """A source event marked wrong: matched by ISIN, ex-date, type and source."""

    isin: str
    ex_date: pd.Timestamp
    event_type: str
    source: str
    note: str


@dataclass(frozen=True, slots=True)
class Accepted:
    """An ex-session reviewed and found to need no event."""

    isin: str
    session: pd.Timestamp
    note: str


@dataclass(frozen=True, slots=True)
class ActionOverrides:
    """
    The decisions of one ``corporate_actions.yaml``.

    Attributes
    ----------
    events : pandas.DataFrame
        Added events, columns of the source events, ``source`` = ``manual``.
    rejections : tuple[Rejection, ...]
    accepted : tuple[Accepted, ...]
    sha256 : str or None
        Hash of the file; ``None`` when there is none.
    """

    events: pd.DataFrame
    rejections: tuple[Rejection, ...]
    accepted: tuple[Accepted, ...]
    sha256: str | None

    @classmethod
    def none(cls) -> Self:
        """No decisions."""
        return cls(_events([]), (), (), None)


def load_action_overrides(root: Path) -> ActionOverrides:
    """
    Read ``corporate_actions.yaml`` from the overrides root, if present.

    Raises
    ------
    OverridesFormatError
        If the file is not valid YAML or breaks the schema.
    """
    path = root / ACTIONS_FILE
    if not path.is_file():
        return ActionOverrides.none()
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
    events = [
        _event(entry, f"{path}: events[{index}]")
        for index, entry in enumerate(_list(document, "events", path))
    ]
    rejections = tuple(
        _rejection(entry, f"{path}: rejections[{index}]")
        for index, entry in enumerate(_list(document, "rejections", path))
    )
    accepted = tuple(
        _accepted(entry, f"{path}: accepted[{index}]")
        for index, entry in enumerate(_list(document, "accepted", path))
    )
    return ActionOverrides(_events(events), rejections, accepted, hashlib.sha256(raw).hexdigest())


# --- Entries ---


def _list(document: dict[str, Any], key: str, path: Path) -> list[Any]:
    value = document.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise OverridesFormatError(f"{path}: '{key}' must be a list.")
    return value


def _event(entry: object, where: str) -> dict[str, object]:
    fields = _mapping(entry, where)
    isin = _isin(fields, "isin", where)
    kind = _type(fields, where)
    row: dict[str, object] = {
        "isin": isin,
        "last_cum_date": _date(fields, "last_cum_date", where),
        "event_type": kind.value,
        "target_isin": "",
        "label": _note(fields, where),
    }
    if kind in CASH_TYPES:
        row["cash_amount"] = _number(fields, "cash_amount", where)
    else:
        row["share_ratio"] = _number(fields, "share_ratio", where)
    if kind in _TARGET_TYPES:
        row["target_isin"] = _isin(fields, "target_isin", where)
    return row


def _rejection(entry: object, where: str) -> Rejection:
    fields = _mapping(entry, where)
    source = str(fields.get("source", ""))
    if source not in {item.value for item in Source} - {Source.MANUAL.value}:
        raise OverridesFormatError(f"{where}: unknown source {source!r}.")
    return Rejection(
        isin=_isin(fields, "isin", where),
        ex_date=_date(fields, "ex_date", where),
        event_type=_type(fields, where).value,
        source=source,
        note=_note(fields, where),
    )


def _accepted(entry: object, where: str) -> Accepted:
    fields = _mapping(entry, where)
    return Accepted(
        isin=_isin(fields, "isin", where),
        session=_date(fields, "session", where),
        note=_note(fields, where),
    )


def _mapping(entry: object, where: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise OverridesFormatError(f"{where} must be a mapping.")
    return entry


def _isin(fields: dict[str, Any], key: str, where: str) -> str:
    value = str(fields.get(key, "")).strip()
    if len(value) != 12 or not value.isalnum():
        raise OverridesFormatError(f"{where}: '{key}' must be a 12-character ISIN.")
    return value


def _type(fields: dict[str, Any], where: str) -> EventType:
    try:
        return EventType(str(fields.get("type", "")))
    except ValueError as error:
        raise OverridesFormatError(f"{where}: unknown type {fields.get('type')!r}.") from error


def _date(fields: dict[str, Any], key: str, where: str) -> pd.Timestamp:
    value = fields.get(key)
    if not isinstance(value, dt.date):
        raise OverridesFormatError(f"{where}: '{key}' must be an ISO date (2024-04-30).")
    return pd.Timestamp(value)


def _number(fields: dict[str, Any], key: str, where: str) -> float:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OverridesFormatError(f"{where}: '{key}' must be a number.")
    return float(value)


def _note(fields: dict[str, Any], where: str) -> str:
    value = fields.get("note")
    if not isinstance(value, str) or not value.strip():
        raise OverridesFormatError(f"{where}: 'note' must be non-empty text.")
    return value.strip()


def _events(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).reindex(columns=list(SOURCE_EVENT_COLUMNS))
    frame["source"] = Source.MANUAL.value
    frame["origin"] = "overrides"
    for column in ("last_cum_date", "payment_date", "approved_on"):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in ("cash_amount", "share_ratio", "subscription_price"):
        frame[column] = frame[column].astype("float64")
    frame["distribution_number"] = frame["distribution_number"].astype("Int64")
    for column in ("isin", "event_type", "target_isin", "label"):
        frame[column] = frame[column].astype(object).fillna("").astype(str)
    return frame
