# ibexQuant\src\ibexQuant\data\reference\corporate_actions.py

"""
Build of the corporate actions ledger (ADR 0013).

Responsibility
--------------
Read the skeleton and every source, place each source event on an ex-session,
merge the sources by priority, give each ex-session a status (inferring
splits where the price says so) and write the ledger tables to
``curated/b3/``.

Contract
--------
- Inputs: ``bars_daily`` and the security master (curated), the B3
  listed-company snapshots, the *Earnings credited* files and the frozen
  bootstrap (raw), and the overrides. The build never downloads.
- Date events get ``ex_date`` = the first B3 session after their data-com;
  events placed by distribution number get the ex-session as ``ex_date`` and
  the session before it as data-com (H3.1, H3.3).
- Events are merged per ISIN, ex-session (or ex-date when unplaced) and type;
  the highest-priority source supplies the events, lower sources confirm them
  one to one by amount or are written as conflicts (H3.8).
- Bootstrap events are kept only for securities without any B3 cash event
  and only when they match an ex-session (H3.4).
- Outputs are replaced atomically; a build is a pure function of its inputs.

What does not belong here
-------------------------
- Reading files — see the source readers.
- Price adjustment (H4).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS as BARS_PARTS
from ibexQuant.data.ingestion.b3.bars_daily import PARTITION_FILE
from ibexQuant.data.ingestion.b3.earnings_credited import credited_files, read_earnings_credited
from ibexQuant.data.ingestion.b3.listed_companies import (
    read_listed_companies,
    snapshot_folders,
)
from ibexQuant.data.reference.corporate_action_overrides import (
    Accepted,
    ActionOverrides,
    Rejection,
)
from ibexQuant.data.reference.corporate_action_sources import (
    SOURCE_EVENT_COLUMNS,
    SOURCE_PRIORITY,
    UNRESOLVED_COLUMNS,
    IsinLocator,
    Source,
    bootstrap_events,
    credited_events,
    listed_events,
)
from ibexQuant.data.reference.errors import SecurityMasterNotBuiltError
from ibexQuant.data.reference.event_types import CASH_TYPES, EventType
from ibexQuant.data.reference.ex_sessions import (
    BAR_COLUMNS,
    EX_SESSION_COLUMNS,
    MatchOutcome,
    distribution_runs,
    ex_sessions_from_runs,
    match_by_date,
    match_by_distribution,
)
from ibexQuant.data.reference.security_master import SecurityMaster, implied_factor
from ibexQuant.data.store import DataStore
from ibexQuant.data.store.atomic import replace_atomically

CURATED_PARTS: Final[tuple[str, ...]] = ("b3",)
EVENTS_FILE: Final[str] = "corporate_actions.parquet"
EX_SESSIONS_FILE: Final[str] = "ex_sessions.parquet"
CONFLICTS_FILE: Final[str] = "corporate_action_conflicts.parquet"
UNRESOLVED_FILE: Final[str] = "corporate_action_unresolved.parquet"
SUMMARY_FILE: Final[str] = "corporate_actions.json"
BOOTSTRAP_PARTS: Final[tuple[str, ...]] = ("bootstrap", "statusinvest_cash.parquet")
SUMMARY_SCHEMA_VERSION: Final[int] = 1

JUMP_BAND: Final[tuple[float, float]] = (0.7, 1.4)
"""Open over previous close outside this band is a jump (H3.6)."""
CASH_TOLERANCE: Final[float] = 0.005
RATIO_TOLERANCE: Final[float] = 0.01
"""Relative tolerances for matching amounts across sources (H3.8)."""

EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "security_id",
    "isin",
    "ex_date",
    "last_cum_date",
    "event_type",
    "cash_amount",
    "share_ratio",
    "subscription_price",
    "target_isin",
    "payment_date",
    "approved_on",
    "source",
    "origin",
    "label",
    "status",
    "placement",
    "ex_session",
    "confirmed_by",
)
EX_SESSION_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "security_id",
    *EX_SESSION_COLUMNS,
    "status",
    "event_ids",
)
CONFLICT_COLUMNS: Final[tuple[str, ...]] = (
    "isin",
    "anchor_date",
    "event_type",
    "source",
    "origin",
    "amount",
    "kept_source",
    "kept_amounts",
)


class LedgerBuildError(ValueError):
    """An override does not match the data it decides on."""


class LedgerNotBuiltError(FileNotFoundError):
    """The ledger tables do not exist yet."""


class EventStatus:
    """Values of the ledger's ``status`` column (H3.9)."""

    CONFIRMED: Final[str] = "confirmed"
    INFERRED: Final[str] = "inferred"
    REVIEW: Final[str] = "review"
    REJECTED: Final[str] = "rejected"


class ExSessionStatus:
    """Values of the skeleton's ``status`` column (H3.6)."""

    EXPLAINED: Final[str] = "explained"
    INFERRED: Final[str] = "inferred"
    UNEXPLAINED_JUMP: Final[str] = "unexplained_jump"
    UNEXPLAINED: Final[str] = "unexplained"
    ACCEPTED: Final[str] = "accepted"


EX_SESSION_STATUSES: Final[tuple[str, ...]] = (
    ExSessionStatus.EXPLAINED,
    ExSessionStatus.INFERRED,
    ExSessionStatus.UNEXPLAINED_JUMP,
    ExSessionStatus.UNEXPLAINED,
    ExSessionStatus.ACCEPTED,
)


def read_ledger(store: DataStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the ledger events and the ex-session skeleton.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``corporate_actions.parquet`` and ``ex_sessions.parquet``.

    Raises
    ------
    LedgerNotBuiltError
        If the ledger was never built.
    """
    folder = store.curated(*CURATED_PARTS)
    paths = (folder / EVENTS_FILE, folder / EX_SESSIONS_FILE)
    if not all(path.is_file() for path in paths):
        raise LedgerNotBuiltError(
            "No corporate actions ledger yet. Run `ibexquant data build corporate-actions`."
        )
    events, ex = (pd.read_parquet(path) for path in paths)
    return events, ex


@dataclass(frozen=True, slots=True)
class LedgerSummary:
    """
    Outcome of one build, also written to ``corporate_actions.json``.

    Attributes
    ----------
    events : dict[str, dict[str, int]]
        Ledger events by source, by type and by status.
    ex_sessions : dict[str, int]
        Ex-sessions by status.
    unexplained_by_year : dict[str, dict[str, int]]
        ``unexplained`` and ``unexplained_jump`` ex-sessions per year.
    conflicts : int
        Lower-priority events whose amounts disagree with the kept ones.
    unresolved : dict[str, int]
        Source rows that could not be placed on an ISIN, by reason.
    bootstrap_dropped : dict[str, int]
        Bootstrap events left out: securities with B3 cash events, and
        events that match no ex-session.
    inputs : dict[str, object]
        Snapshots, credited files and file hashes used.
    built_at : str
        UTC build time, ISO format.
    """

    events: dict[str, dict[str, int]]
    ex_sessions: dict[str, int]
    unexplained_by_year: dict[str, dict[str, int]]
    conflicts: int
    unresolved: dict[str, int]
    bootstrap_dropped: dict[str, int]
    inputs: dict[str, object] = field(default_factory=dict)
    built_at: str = ""


# --- Build ---


def build_corporate_actions(store: DataStore, overrides: ActionOverrides) -> LedgerSummary:
    """
    Build the ledger tables in ``curated/b3/``.

    Parameters
    ----------
    store : DataStore
        The data store.
    overrides : ActionOverrides
        Decisions of ``corporate_actions.yaml`` (``ActionOverrides.none()``
        to build without them).

    Returns
    -------
    LedgerSummary

    Raises
    ------
    SecurityMasterNotBuiltError
        If ``bars_daily`` or the security master is missing.
    LedgerBuildError
        If a rejection or an accepted ex-session matches nothing.
    """
    bars = _load_bars(store)
    runs = distribution_runs(bars)
    calendar = pd.DatetimeIndex(np.sort(bars["session_date"].unique()))
    master = SecurityMaster.load(store).table
    locator = IsinLocator(master)

    snapshots = snapshot_folders(store)
    listed, unresolved_listed = listed_events(read_listed_companies(snapshots), locator)
    credited_paths = credited_files(store)
    credited = credited_events(read_earnings_credited(credited_paths))
    bootstrap, unresolved_bootstrap, bootstrap_path = _bootstrap(store, locator)

    sources = [
        frame for frame in (overrides.events, listed, credited, bootstrap) if not frame.empty
    ]
    events = (
        pd.concat(sources, ignore_index=True)
        if sources
        else pd.DataFrame(columns=list(SOURCE_EVENT_COLUMNS))
    )
    events = _place(events, runs, calendar)
    events, dropped = _filter_bootstrap(events)
    events, conflicts = _merge_sources(events)
    pending = list(overrides.rejections)
    events["rejected"] = _rejected(events, pending)

    security_of = dict(zip(master["isin"], master["security_id"], strict=False))
    ex = ex_sessions_from_runs(runs)
    ex, inferred = _explain(ex, events[~events["rejected"]])
    if not inferred.empty:
        inferred["rejected"] = _rejected(inferred, pending)
        ex = _unexplain(ex, inferred[inferred["rejected"]])
        events = pd.concat([events, inferred], ignore_index=True)
    if pending:
        first = pending[0]
        raise LedgerBuildError(
            f"Rejection of {first.event_type} {first.isin} {first.ex_date.date()} "
            f"({first.source}) matches no event."
        )
    ex = _accept(ex, overrides.accepted)
    events = _finish_events(events, security_of)
    ex = _finish_ex_sessions(ex, events, security_of)
    unresolved = pd.concat([unresolved_listed, unresolved_bootstrap], ignore_index=True)

    _write(store, events, ex, conflicts, unresolved)
    summary = LedgerSummary(
        events={
            "source": _counts(events["source"]),
            "type": _counts(events["event_type"]),
            "status": _counts(events["status"]),
        },
        ex_sessions=_counts(ex["status"]),
        unexplained_by_year=_unexplained_by_year(ex),
        conflicts=len(conflicts),
        unresolved=_counts(unresolved["reason"]),
        bootstrap_dropped=dropped,
        inputs={
            "snapshots": [folder.name for folder in snapshots],
            "credited_files": len(credited_paths),
            "bootstrap": _sha256(bootstrap_path),
            "overrides": overrides.sha256,
        },
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    _write_summary(store, summary)
    return summary


# --- Inputs ---


def _load_bars(store: DataStore) -> pd.DataFrame:
    folder = store.curated(*BARS_PARTS)
    paths = sorted(folder.glob(f"year=*/{PARTITION_FILE}")) if folder.is_dir() else []
    if not paths:
        raise SecurityMasterNotBuiltError(
            "No bars_daily partition. Run `ibexquant data ingest cotahist --all` first."
        )
    bars = pd.concat([pd.read_parquet(path, columns=list(BAR_COLUMNS)) for path in paths])
    bars["isin"] = bars["isin"].astype(str)
    bars["ticker"] = bars["ticker"].astype(str)
    return bars.reset_index(drop=True)


def _bootstrap(store: DataStore, locator: IsinLocator) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    path = store.raw(*BOOTSTRAP_PARTS)
    if not path.is_file():
        empty = pd.DataFrame(columns=list(SOURCE_EVENT_COLUMNS))
        return empty, pd.DataFrame(columns=list(UNRESOLVED_COLUMNS)), path
    events, unresolved = bootstrap_events(pd.read_parquet(path), locator)
    return events, unresolved, path


# --- Placing ---


def _place(events: pd.DataFrame, runs: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    events = events.reset_index(drop=True).copy()
    events["ex_date"] = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    by_date = events["last_cum_date"].notna()
    by_number = ~by_date & events["distribution_number"].notna()
    placement = pd.Series(MatchOutcome.NOT_TRADED.value, index=events.index, dtype=object)
    ex_session = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")

    if by_date.any():
        matched = match_by_date(
            events.loc[by_date, "isin"], events.loc[by_date, "last_cum_date"], runs
        )
        placement[by_date] = matched["outcome"].astype(str)
        ex_session[by_date] = matched["ex_session"]
        positions = calendar.searchsorted(events.loc[by_date, "last_cum_date"], side="right")
        events.loc[by_date, "ex_date"] = _sessions_at(calendar, positions)
    if by_number.any():
        matched = match_by_distribution(
            events.loc[by_number, "isin"], events.loc[by_number, "distribution_number"], runs
        )
        placement[by_number] = matched["outcome"].astype(str)
        ex_session[by_number] = matched["ex_session"]
        events.loc[by_number, "ex_date"] = matched["ex_session"]
        positions = calendar.searchsorted(matched["ex_session"].dropna(), side="left") - 1
        placed = matched["ex_session"].notna()
        events.loc[placed[placed].index, "last_cum_date"] = _sessions_at(calendar, positions)

    events["placement"] = placement
    events["ex_session"] = ex_session
    events["anchor_date"] = events["ex_session"].fillna(events["ex_date"])
    return events


def _sessions_at(calendar: pd.DatetimeIndex, positions: np.ndarray) -> np.ndarray:
    valid = (positions >= 0) & (positions < len(calendar))
    out = np.full(len(positions), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    out[valid] = calendar.to_numpy()[positions[valid]]
    return out


def _filter_bootstrap(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    is_bootstrap = events["source"] == Source.BOOTSTRAP.value
    b3_cash = events["source"].isin([Source.B3_LISTED.value, Source.B3_CREDITED.value]) & events[
        "event_type"
    ].isin([kind.value for kind in CASH_TYPES])
    covered = set(events.loc[b3_cash, "isin"])
    with_b3 = is_bootstrap & events["isin"].isin(covered)
    unmatched = is_bootstrap & ~with_b3 & (events["placement"] != MatchOutcome.MATCHED.value)
    dropped = {"security_has_b3_cash": int(with_b3.sum()), "no_ex_session": int(unmatched.sum())}
    return events[~(with_b3 | unmatched)].reset_index(drop=True), dropped


# --- Merging sources ---


def _amount(events: pd.DataFrame) -> pd.Series:
    cash = events["event_type"].isin([kind.value for kind in CASH_TYPES])
    return events["cash_amount"].where(cash, events["share_ratio"])


def _merge_sources(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    events["priority"] = events["source"].map(SOURCE_PRIORITY)
    events["amount"] = _amount(events)
    events["confirmed_by"] = ""
    key = ["isin", "anchor_date", "event_type"]
    best = events.groupby(key, dropna=False)["priority"].transform("min")
    kept = events[events["priority"] == best]
    lower = events[events["priority"] != best]
    if lower.empty:
        return kept.drop(columns=["priority", "amount"]), pd.DataFrame(
            columns=list(CONFLICT_COLUMNS)
        )

    conflicts = []
    confirmations: dict[int, list[str]] = {}
    kept_groups = {name: group for name, group in kept.groupby(key, dropna=False)}
    for name, group in lower.groupby(key, dropna=False):
        primary = kept_groups[name]
        amounts: dict[int, float] = dict(
            zip(primary.index, primary["amount"].astype(float).tolist(), strict=True)
        )
        free = list(amounts)
        cash = name[2] in {kind.value for kind in CASH_TYPES}
        tolerance = CASH_TOLERANCE if cash else RATIO_TOLERANCE
        for _, row in group.iterrows():
            match = next(
                (
                    candidate
                    for candidate in free
                    if _close(float(row["amount"]), amounts[candidate], tolerance)
                ),
                None,
            )
            if match is not None:
                free.remove(match)
                confirmations.setdefault(match, []).append(str(row["source"]))
                continue
            conflicts.append(
                {
                    "isin": name[0],
                    "anchor_date": name[1],
                    "event_type": name[2],
                    "source": row["source"],
                    "origin": row["origin"],
                    "amount": row["amount"],
                    "kept_source": primary["source"].iloc[0],
                    "kept_amounts": ", ".join(f"{value:.10g}" for value in primary["amount"]),
                }
            )
    for index, sources in confirmations.items():
        kept.at[index, "confirmed_by"] = ",".join(sorted(set(sources)))
    return (
        kept.drop(columns=["priority", "amount"]),
        pd.DataFrame(conflicts, columns=list(CONFLICT_COLUMNS)),
    )


def _close(value: float, reference: float, tolerance: float) -> bool:
    if math.isnan(value) or math.isnan(reference):
        return math.isnan(value) and math.isnan(reference)
    return abs(value - reference) <= tolerance * max(abs(reference), 1e-12)


# --- Overrides ---


def _rejected(events: pd.DataFrame, pending: list[Rejection]) -> pd.Series:
    """Mark the events each pending rejection names, removing the rejections used."""
    marked = pd.Series(False, index=events.index)
    for rejection in list(pending):
        hit = (
            (events["isin"] == rejection.isin)
            & (events["ex_date"] == rejection.ex_date)
            & (events["event_type"] == rejection.event_type)
            & (events["source"] == rejection.source)
        )
        if hit.any():
            marked |= hit
            pending.remove(rejection)
    return marked


def _unexplain(ex: pd.DataFrame, rejected: pd.DataFrame) -> pd.DataFrame:
    keys = set(zip(rejected["isin"], rejected["ex_session"], strict=True))
    hit = [key in keys for key in zip(ex["isin"], ex["session_date"], strict=True)]
    status = ex["status"].where(~pd.Series(hit, index=ex.index), ExSessionStatus.UNEXPLAINED_JUMP)
    return ex.assign(status=status)


def _accept(ex: pd.DataFrame, accepted: tuple[Accepted, ...]) -> pd.DataFrame:
    status = ex["status"].copy()
    for entry in accepted:
        hit = (ex["isin"] == entry.isin) & (ex["session_date"] == entry.session)
        if not hit.any():
            raise LedgerBuildError(
                f"Accepted ex-session {entry.isin} {entry.session.date()} is not in the skeleton."
            )
        if (status[hit] == ExSessionStatus.EXPLAINED).all():
            raise LedgerBuildError(
                f"Accepted ex-session {entry.isin} {entry.session.date()} is already explained."
            )
        status[hit] = ExSessionStatus.ACCEPTED
    return ex.assign(status=status)


# --- Explaining the skeleton ---


def _explain(ex: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    placed = events[
        (events["placement"] == MatchOutcome.MATCHED.value) & events["ex_session"].notna()
    ]
    explained = set(zip(placed["isin"], placed["ex_session"], strict=True))
    status, inferred_rows = [], []
    low, high = JUMP_BAND
    rows = zip(
        ex["isin"].tolist(),
        ex["session_date"].tolist(),
        ex["previous_session"].tolist(),
        ex["open_ratio"].astype(float).tolist(),
        strict=True,
    )
    for isin, session, previous, ratio in rows:
        if (isin, session) in explained:
            status.append(ExSessionStatus.EXPLAINED)
            continue
        if low <= ratio <= high or not math.isfinite(ratio) or ratio <= 0:
            status.append(ExSessionStatus.UNEXPLAINED)
            continue
        factor = implied_factor(1.0 / ratio)
        if factor is None:
            status.append(ExSessionStatus.UNEXPLAINED_JUMP)
            continue
        status.append(ExSessionStatus.INFERRED)
        inferred_rows.append(
            {
                "isin": isin,
                "last_cum_date": previous,
                "ex_date": session,
                "event_type": (
                    EventType.SPLIT.value if factor > 1 else EventType.REVERSE_SPLIT.value
                ),
                "share_ratio": factor,
                "source": Source.INFERRED.value,
                "origin": f"open/close {ratio:.6g}",
                "label": "",
                "placement": MatchOutcome.MATCHED.value,
                "ex_session": session,
                "anchor_date": session,
                "confirmed_by": "",
            }
        )
    ex = ex.assign(status=status)
    inferred = pd.DataFrame(inferred_rows)
    return ex, inferred


# --- Finishing ---


def _finish_events(events: pd.DataFrame, security_of: dict[str, str]) -> pd.DataFrame:
    rejected = events["rejected"].fillna(False).astype(bool)
    events = events.reindex(columns=[*EVENT_COLUMNS, "anchor_date"]).copy()
    events["security_id"] = events["isin"].map(security_of).fillna("")
    events["status"] = np.select(
        [
            rejected,
            events["source"] == Source.INFERRED.value,
            events["placement"] == MatchOutcome.NO_EX_SESSION.value,
        ],
        [EventStatus.REJECTED, EventStatus.INFERRED, EventStatus.REVIEW],
        EventStatus.CONFIRMED,
    )
    for column in ("ex_date", "last_cum_date", "payment_date", "approved_on", "ex_session"):
        events[column] = pd.to_datetime(events[column]).astype("datetime64[ns]")
    for column in ("cash_amount", "share_ratio", "subscription_price"):
        events[column] = events[column].astype("float64")
    for column in ("target_isin", "origin", "label", "confirmed_by"):
        events[column] = events[column].astype(object).fillna("").astype(str)
    events = events.sort_values(["isin", "anchor_date", "event_type", "source"], kind="stable")
    occurrence = events.groupby(
        ["isin", "ex_date", "event_type", "source", "origin"], dropna=False
    ).cumcount()
    # map(str), not astype(str): the string dtype keeps NaT and NaN as missing values.
    parts = zip(
        events["isin"].map(str),
        events["ex_date"].map(str),
        events["event_type"].map(str),
        events["cash_amount"].map("{:.10g}".format),
        events["share_ratio"].map("{:.10g}".format),
        events["source"].map(str),
        events["origin"].map(str),
        occurrence.map(str),
        strict=True,
    )
    events["event_id"] = [hashlib.sha1("|".join(part).encode()).hexdigest()[:16] for part in parts]
    return events[list(EVENT_COLUMNS)].reset_index(drop=True)


def _finish_ex_sessions(
    ex: pd.DataFrame, events: pd.DataFrame, security_of: dict[str, str]
) -> pd.DataFrame:
    placed = events[events["ex_session"].notna()]
    ids = placed.groupby(["isin", "ex_session"])["event_id"].agg(",".join)
    keys = pd.MultiIndex.from_arrays([ex["isin"], ex["session_date"]])
    ex = ex.assign(
        security_id=ex["isin"].map(security_of).fillna(""),
        event_ids=ids.reindex(keys).fillna("").to_numpy(),
    )
    return ex[list(EX_SESSION_TABLE_COLUMNS)].reset_index(drop=True)


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(count) for key, count in values.value_counts().sort_index().items()}


def _unexplained_by_year(ex: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for status in (ExSessionStatus.UNEXPLAINED, ExSessionStatus.UNEXPLAINED_JUMP):
        years = ex.loc[ex["status"] == status, "session_date"].dt.year
        out[status] = {
            str(year): int(count) for year, count in years.value_counts().sort_index().items()
        }
    return out


# --- Writing ---


def _write(
    store: DataStore,
    events: pd.DataFrame,
    ex: pd.DataFrame,
    conflicts: pd.DataFrame,
    unresolved: pd.DataFrame,
) -> None:
    folder = store.curated(*CURATED_PARTS)
    folder.mkdir(parents=True, exist_ok=True)
    tables = (
        (EVENTS_FILE, events),
        (EX_SESSIONS_FILE, ex),
        (CONFLICTS_FILE, conflicts),
        (UNRESOLVED_FILE, unresolved),
    )
    for name, frame in tables:
        replace_atomically(folder / name, _parquet_writer(frame))


def _parquet_writer(frame: pd.DataFrame) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    return write


def _write_summary(store: DataStore, summary: LedgerSummary) -> None:
    document = {"schema_version": SUMMARY_SCHEMA_VERSION, **asdict(summary)}
    text = json.dumps(document, indent=2) + "\n"
    replace_atomically(
        store.curated(*CURATED_PARTS, SUMMARY_FILE),
        lambda temporary: temporary.write_text(text, encoding="utf-8"),
    )


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
