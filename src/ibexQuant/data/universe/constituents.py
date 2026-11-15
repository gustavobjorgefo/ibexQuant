# ibexQuant\src\ibexQuant\data\universe\constituents.py

"""
Index compositions by period, resolved to securities (ADR 0015, H5.1–H5.5).

Responsibility
--------------
Build ``curated/b3/index_periods.parquet`` from the session calendar and
``curated/b3/index_constituents.parquet`` from the raw portfolio files, one
composition per index and period from the best source at hand, every ticker
resolved to a ``security_id`` with the security master.

Contract
--------
- Sources rank ``definitive`` > ``third_preview`` > ``second_preview`` >
  ``retroactive`` (H5.1); the source and file of every row are recorded, and
  the compositions set aside for a better one are counted.
- Two files of the same kind, index and period must list the same tickers;
  the first by file name is kept.
- A ticker resolves on its base session, or on the start of its period when
  it did not carry that ticker then (H5.5). An unresolved ticker keeps an
  empty ``security_id`` and is listed in the summary; fixes go to
  ``ticker_links.yaml``.
- The build reads only raw files, the bars and the master: it can be rerun at
  any time and gives the same tables.

What does not belong here
-------------------------
- Reading the files — see ``ibexQuant.data.ingestion.b3.index_portfolios``.
- Period dates — see ``calendar``.
- Filling the periods without a composition with the liquidity rule, and the
  mask — later steps of H5.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd

from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS as BARS_PARTS
from ibexQuant.data.ingestion.b3.bars_daily import PARTITION_FILE
from ibexQuant.data.ingestion.b3.index_portfolios import (
    RAW_PARTS,
    RETROACTIVE_DIR,
    RETROACTIVE_FILE,
    SOURCE_PRIORITY,
    TITLE_SHEET,
    WORKBOOKS_DIR,
    IndexPortfolioFormatError,
    PortfolioSource,
    read_retroactive,
    read_workbook,
    workbook_paths,
)
from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.store import DataStore
from ibexQuant.data.store.atomic import replace_atomically
from ibexQuant.data.universe.calendar import PeriodOverrides, index_periods

CURATED_PARTS: Final[tuple[str, ...]] = ("b3",)
CONSTITUENTS_FILE: Final[str] = "index_constituents.parquet"
PERIODS_TABLE_FILE: Final[str] = "index_periods.parquet"
SUMMARY_FILE: Final[str] = "index_constituents.json"
SUMMARY_SCHEMA_VERSION: Final[int] = 1

CONSTITUENT_COLUMNS: Final[tuple[str, ...]] = (
    "index",
    "period",
    "ticker",
    "security_id",
    "name",
    "share_class",
    "theoretical_quantity",
    "weight_pct",
    "base_session",
    "source",
    "source_file",
)
STRICT_INDICES: Final[tuple[str, ...]] = ("IBOV", "IBXX")
"""Indices whose tickers must all resolve to close H5 (H5.5)."""

FIRST_IBOV_PERIOD: Final[str] = "2003-01"
"""The first period with a composition from B3 (the retroactive series)."""


class UniverseError(Exception):
    """Base class of every error raised by ``ibexQuant.data.universe``."""


class UniverseNotBuiltError(UniverseError, FileNotFoundError):
    """The universe tables, or the tables they are built from, do not exist yet."""


class UniverseBuildError(UniverseError, ValueError):
    """The raw portfolio files contradict each other or cannot be read."""


@dataclass(frozen=True, slots=True)
class UniverseSummary:
    """
    Outcome of one build, also written to ``index_constituents.json``.

    Attributes
    ----------
    periods : int
        Periods of the calendar.
    first_period, last_period : str or None
        Labels of the first and last period of the calendar.
    compositions : dict[str, int]
        Compositions kept (index and period), by source.
    superseded : dict[str, int]
        Compositions set aside for a better source, by source.
    rows : int
        Rows of the constituents table.
    indices : int
        Distinct indices.
    unresolved : dict[str, int]
        Rows without ``security_id``, by index.
    unresolved_strict : list[dict[str, str]]
        Those rows for :data:`STRICT_INDICES`: index, period, ticker.
    ibov_gaps : list[str]
        Periods from :data:`FIRST_IBOV_PERIOD` without an ``IBOV`` composition.
    outside_calendar : list[str]
        Periods with a composition that the calendar does not cover.
    inputs : dict[str, str | None]
        SHA-256 of each raw file and of ``index_periods.yaml``, by name.
    built_at : str
        UTC build time, ISO format.
    """

    periods: int
    first_period: str | None
    last_period: str | None
    compositions: dict[str, int]
    superseded: dict[str, int]
    rows: int
    indices: int
    unresolved: dict[str, int]
    unresolved_strict: list[dict[str, str]]
    ibov_gaps: list[str]
    outside_calendar: list[str]
    inputs: dict[str, str | None]
    built_at: str


# --- Build ---


def build_universe(store: DataStore, overrides: PeriodOverrides) -> UniverseSummary:
    """
    Build the periods and the constituents tables and their summary.

    Parameters
    ----------
    store : DataStore
        The data store, with ``bars_daily`` and the security master built.
    overrides : PeriodOverrides
        One-off period starts; ``PeriodOverrides.none()`` for the rule only.

    Returns
    -------
    UniverseSummary
        Counts and inputs of the build.

    Raises
    ------
    UniverseNotBuiltError
        If there are no bars or no master.
    UniverseBuildError
        If raw files cannot be read (all of them are listed), two files of one
        kind disagree, or an override breaks the calendar.
    SecurityMasterNotBuiltError
        If the master was never built.
    """
    sessions = _sessions(store)
    try:
        periods = index_periods(sessions, overrides)
    except ValueError as error:
        raise UniverseBuildError(str(error)) from error
    master = SecurityMaster.load(store)

    root = store.raw(*RAW_PARTS)
    inputs: dict[str, str | None] = {}
    candidates: list[pd.DataFrame] = []
    # Every unreadable file is reported at once, so one run lists all fixes to make.
    problems: list[str] = []
    for path in workbook_paths(root / WORKBOOKS_DIR):
        inputs[f"{WORKBOOKS_DIR}/{path.name}"] = _sha256(path)
        try:
            book = read_workbook(path)
        except IndexPortfolioFormatError as error:
            problems.append(str(error))
            continue
        candidates.append(
            book.members.assign(
                period=book.period, source=book.source.value, source_file=path.name
            )
        )
    retroactive = root / RETROACTIVE_DIR / RETROACTIVE_FILE
    if retroactive.is_file():
        inputs[f"{RETROACTIVE_DIR}/{RETROACTIVE_FILE}"] = _sha256(retroactive)
        try:
            candidates.append(
                read_retroactive(retroactive).assign(
                    source=PortfolioSource.RETROACTIVE.value, source_file=retroactive.name
                )
            )
        except IndexPortfolioFormatError as error:
            problems.append(str(error))
    if problems:
        raise UniverseBuildError("\n".join(problems))
    inputs["index_periods.yaml"] = overrides.sha256

    members = _deduplicate(pd.concat(candidates, ignore_index=True)) if candidates else None
    kept, superseded = _best_source(members) if members is not None else (_empty(), {})
    constituents = _resolve(kept, periods, master)

    summary = _summary(constituents, periods, superseded, inputs)
    _write(store, constituents, periods, summary)
    return summary


def read_index_constituents(store: DataStore) -> pd.DataFrame:
    """
    Read ``curated/b3/index_constituents.parquet``.

    Raises
    ------
    UniverseNotBuiltError
        If the universe was never built.
    """
    return _read(store, CONSTITUENTS_FILE)


def read_index_periods(store: DataStore) -> pd.DataFrame:
    """
    Read ``curated/b3/index_periods.parquet``.

    Raises
    ------
    UniverseNotBuiltError
        If the universe was never built.
    """
    return _read(store, PERIODS_TABLE_FILE)


# --- Steps ---


def _sessions(store: DataStore) -> pd.DatetimeIndex:
    folder = store.curated(*BARS_PARTS)
    paths = sorted(folder.glob(f"year=*/{PARTITION_FILE}")) if folder.is_dir() else []
    if not paths:
        raise UniverseNotBuiltError(
            "No bars_daily partition. Run `ibexquant data ingest cotahist --all` first."
        )
    days = [pd.read_parquet(path, columns=["session_date"])["session_date"] for path in paths]
    return pd.DatetimeIndex(pd.concat(days).unique()).sort_values()


def _deduplicate(members: pd.DataFrame) -> pd.DataFrame:
    """Keep one file per index, period and source; files of one kind must agree."""
    keys = ["index", "period", "source"]
    tickers = members.groupby([*keys, "source_file"], sort=True)["ticker"].agg(frozenset)
    for key, group in tickers.groupby(level=keys):
        if group.nunique() > 1:
            names = group.index.get_level_values("source_file").tolist()
            raise UniverseBuildError(f"{key}: files {names} list different members.")
    kept_files = tickers.reset_index()[[*keys, "source_file"]].drop_duplicates(keys)
    return members.merge(kept_files, on=[*keys, "source_file"], how="inner")


def _best_source(members: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rank = {source.value: position for position, source in enumerate(SOURCE_PRIORITY)}
    compositions = members[["index", "period", "source"]].drop_duplicates()
    compositions = compositions.assign(rank=compositions["source"].map(rank))
    best = compositions.sort_values("rank").drop_duplicates(["index", "period"])
    set_aside = compositions.merge(best, how="left", indicator=True)
    superseded = _counts(set_aside[set_aside["_merge"] == "left_only"]["source"])
    kept = members.merge(best[["index", "period", "source"]], on=["index", "period", "source"])
    return kept, superseded


def _resolve(members: pd.DataFrame, periods: pd.DataFrame, master: SecurityMaster) -> pd.DataFrame:
    """Add ``security_id``: on the base session, else on the period's start."""
    frame = members.merge(
        periods[["period", "effective_from"]], on="period", how="left"
    ).reset_index(drop=True)
    spans = master.table[["ticker", "security_id", "first_session", "last_session"]].astype(
        {"ticker": "string", "security_id": "string"}
    )
    spans = spans.astype({"first_session": "datetime64[ns]", "last_session": "datetime64[ns]"})
    found = _lookup(frame, frame["base_session"], spans)
    missing = found.isna()
    if missing.any():
        fallback = _lookup(frame[missing], frame.loc[missing, "effective_from"], spans)
        found = found.where(~missing, fallback)
    frame["security_id"] = found
    frame = frame.sort_values(["index", "period", "ticker"]).reset_index(drop=True)
    return frame[list(CONSTITUENT_COLUMNS)].astype(
        {
            "index": "string",
            "period": "string",
            "ticker": "string",
            "security_id": "string",
            "source": "string",
            "source_file": "string",
        }
    )


def _lookup(rows: pd.DataFrame, as_of: pd.Series, spans: pd.DataFrame) -> pd.Series:
    """The one security carrying each row's ticker on ``as_of``; missing if none or several."""
    probe = pd.DataFrame(
        {
            "row": rows.index,
            "ticker": rows["ticker"].astype("string"),
            "as_of": pd.to_datetime(as_of).astype("datetime64[ns]").to_numpy(),
        }
    )
    hits = probe.merge(spans, on="ticker", how="inner")
    hits = hits[(hits["first_session"] <= hits["as_of"]) & (hits["as_of"] <= hits["last_session"])]
    unique = hits.drop_duplicates(["row", "security_id"])
    single = unique[~unique["row"].duplicated(keep=False)].set_index("row")["security_id"]
    return single.reindex(rows.index).astype("string")


def _summary(
    constituents: pd.DataFrame,
    periods: pd.DataFrame,
    superseded: dict[str, int],
    inputs: dict[str, str | None],
) -> UniverseSummary:
    labels = periods["period"].astype(str).tolist()
    compositions = constituents[["index", "period", "source"]].drop_duplicates()
    unresolved = constituents[constituents["security_id"].isna()]
    strict = unresolved[unresolved["index"].isin(STRICT_INDICES)]
    ibov_periods = set(
        compositions.loc[compositions["index"] == TITLE_SHEET, "period"].astype(str)
    )
    return UniverseSummary(
        periods=len(labels),
        first_period=labels[0] if labels else None,
        last_period=labels[-1] if labels else None,
        compositions=_counts(compositions["source"]),
        superseded=superseded,
        rows=len(constituents),
        indices=int(constituents["index"].nunique()),
        unresolved=_counts(unresolved["index"]),
        unresolved_strict=[
            {"index": str(row.index), "period": str(row.period), "ticker": str(row.ticker)}
            for row in strict.itertuples(index=False)
        ],
        ibov_gaps=[
            label for label in labels if label >= FIRST_IBOV_PERIOD and label not in ibov_periods
        ],
        outside_calendar=sorted(set(compositions["period"].astype(str)) - set(labels)),
        inputs=inputs,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# --- Storage ---


def _write(
    store: DataStore, constituents: pd.DataFrame, periods: pd.DataFrame, summary: UniverseSummary
) -> None:
    folder = store.curated(*CURATED_PARTS)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in ((CONSTITUENTS_FILE, constituents), (PERIODS_TABLE_FILE, periods)):
        replace_atomically(folder / name, _parquet_writer(frame))
    text = json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, **asdict(summary)}, indent=2)
    replace_atomically(
        folder / SUMMARY_FILE, lambda path: path.write_text(text + "\n", encoding="utf-8")
    )


def _read(store: DataStore, name: str) -> pd.DataFrame:
    path = store.curated(*CURATED_PARTS, name)
    if not path.is_file():
        raise UniverseNotBuiltError("No universe yet. Run `ibexquant data build universe`.")
    return pd.read_parquet(path)


def _parquet_writer(frame: pd.DataFrame) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        frame.to_parquet(path, engine="pyarrow", index=False)

    return write


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(count) for key, count in values.value_counts().sort_index().items()}


def _empty() -> pd.DataFrame:
    text = {
        column: pd.Series(dtype="string")
        for column in ("index", "period", "ticker", "name", "share_class", "source", "source_file")
    }
    return pd.DataFrame(
        {
            **text,
            "theoretical_quantity": pd.Series(dtype="float64"),
            "weight_pct": pd.Series(dtype="float64"),
            "base_session": pd.Series(dtype="datetime64[ns]"),
        }
    )
