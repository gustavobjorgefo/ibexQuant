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
- ``IBOV`` periods without a composition from B3 (before 2003, May 2014 to
  May 2017) take the liquidity rule (source ``liquidity_rule``), with as many
  members as the nearest real portfolio; the rule's overlap with the real
  index is reported for every real portfolio from :data:`OVERLAP_FROM`.
- Members whose security stops trading inside the period are written to
  ``curated/b3/index_exits.parquet`` (H5.8); members of a period that did not
  trade on its first session and are not exits are counted in the summary.
- The build reads only raw files, the bars, the master and the ledger: it can
  be rerun at any time and gives the same tables.

What does not belong here
-------------------------
- Reading the files — see ``ibexQuant.data.ingestion.b3.index_portfolios``.
- Period dates — see ``calendar``.
- The ranking itself — see ``liquidity``; finding exits — see ``exits``; the
  mask — see ``mask``.
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
from ibexQuant.data.reference.corporate_actions import LedgerNotBuiltError, read_ledger
from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.store import DataStore
from ibexQuant.data.store.atomic import replace_atomically
from ibexQuant.data.universe.calendar import PeriodOverrides, index_periods
from ibexQuant.data.universe.errors import UniverseBuildError, UniverseNotBuiltError
from ibexQuant.data.universe.exits import find_exits, last_sessions
from ibexQuant.data.universe.liquidity import (
    IBOV_LIQUIDITY,
    ROUND_LOT_MARKET,
    LiquiditySettings,
    TradedValues,
    base_session,
    read_traded_values,
    select_by_liquidity,
)

CURATED_PARTS: Final[tuple[str, ...]] = ("b3",)
CONSTITUENTS_FILE: Final[str] = "index_constituents.parquet"
PERIODS_TABLE_FILE: Final[str] = "index_periods.parquet"
EXITS_FILE: Final[str] = "index_exits.parquet"
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

OVERLAP_FROM: Final[str] = "2017-09"
"""First period of B3's own files, where the liquidity rule is compared with the index."""

REAL_SOURCES: Final[frozenset[str]] = frozenset(
    source.value for source in SOURCE_PRIORITY if source is not PortfolioSource.LIQUIDITY_RULE
)


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
        Periods without an ``IBOV`` composition, after the liquidity rule.
    liquidity_periods : list[str]
        ``IBOV`` periods filled by the liquidity rule.
    liquidity_overlap : dict[str, float]
        Share of each real ``IBOV`` portfolio from :data:`OVERLAP_FROM` that
        the liquidity rule, with as many members, selects.
    exits : dict[str, int]
        Rows of ``index_exits.parquet``, by reason.
    absent_at_start : dict[str, int]
        Members without a bar on their period's first session and not in the
        exits, by index.
    absent_at_start_strict : list[dict[str, str]]
        Those members for :data:`STRICT_INDICES`: index, period, ticker.
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
    liquidity_periods: list[str]
    liquidity_overlap: dict[str, float]
    exits: dict[str, int]
    absent_at_start: dict[str, int]
    absent_at_start_strict: list[dict[str, str]]
    outside_calendar: list[str]
    inputs: dict[str, str | None]
    built_at: str


# --- Build ---


def build_universe(
    store: DataStore,
    overrides: PeriodOverrides,
    liquidity: LiquiditySettings = IBOV_LIQUIDITY,
) -> UniverseSummary:
    """
    Build the periods, constituents and exits tables and their summary.

    Parameters
    ----------
    store : DataStore
        The data store, with ``bars_daily``, the security master and the
        corporate actions ledger built.
    overrides : PeriodOverrides
        One-off period starts; ``PeriodOverrides.none()`` for the rule only.
    liquidity : LiquiditySettings
        The rule for ``IBOV`` periods without a composition; Ibovespa-like
        defaults (H5.9).

    Returns
    -------
    UniverseSummary
        Counts and inputs of the build.

    Raises
    ------
    UniverseNotBuiltError
        If there are no bars or no ledger.
    UniverseBuildError
        If raw files cannot be read (all of them are listed), two files of one
        kind disagree, or an override breaks the calendar.
    SecurityMasterNotBuiltError
        If the master was never built.
    """
    master = SecurityMaster.load(store)
    traded = read_traded_values(store, master)
    try:
        periods = index_periods(traded.calendar, overrides)
    except ValueError as error:
        raise UniverseBuildError(str(error)) from error
    try:
        events, _ = read_ledger(store)
    except LedgerNotBuiltError as error:
        raise UniverseNotBuiltError(str(error)) from error

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
    real = _resolve(kept, periods, master)
    filled = _liquidity_fill(real, periods, master, traded, liquidity)
    constituents = (
        pd.concat([real, filled], ignore_index=True)
        .sort_values(["index", "period", "ticker"])
        .reset_index(drop=True)
    )
    exits = find_exits(constituents, periods, last_sessions(master), events)
    absent = _absent_at_start(constituents, periods, traded, exits)
    overlap = _overlap(real, periods, traded, liquidity)

    summary = _summary(constituents, periods, superseded, inputs, exits, absent, overlap)
    _write(
        store,
        {CONSTITUENTS_FILE: constituents, PERIODS_TABLE_FILE: periods, EXITS_FILE: exits},
        summary,
    )
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


def read_index_exits(store: DataStore) -> pd.DataFrame:
    """
    Read ``curated/b3/index_exits.parquet``.

    Raises
    ------
    UniverseNotBuiltError
        If the universe was never built.
    """
    return _read(store, EXITS_FILE)


# --- Steps ---


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


def _liquidity_fill(
    real: pd.DataFrame,
    periods: pd.DataFrame,
    master: SecurityMaster,
    traded: TradedValues,
    settings: LiquiditySettings,
) -> pd.DataFrame:
    """``IBOV`` rows from the liquidity rule for the periods B3 has no portfolio for."""
    ibov = real[real["index"] == TITLE_SHEET]
    sizes = ibov.groupby("period")["ticker"].size()
    labels = periods["period"].astype(str).tolist()
    position = {label: number for number, label in enumerate(labels)}
    known = [(position[label], int(size)) for label, size in sizes.items() if label in position]
    rows: list[pd.DataFrame] = []
    for number, (label, start) in enumerate(zip(labels, periods["effective_from"], strict=True)):
        if label in sizes.index or not known:
            continue
        # Nearest real portfolio by number of periods; the earlier one on a tie.
        top = min(known, key=lambda item: (abs(item[0] - number), item[0]))[1]
        base = base_session(traded.calendar, start)
        if base is not None and (selected := select_by_liquidity(traded, base, top, settings)):
            rows.append(_rule_rows(label, base, selected, master))
    if not rows:
        return real.iloc[0:0]
    return pd.concat(rows, ignore_index=True)[list(CONSTITUENT_COLUMNS)].astype(
        real.dtypes.to_dict()
    )


def _rule_rows(
    label: str, base: pd.Timestamp, selected: list[str], master: SecurityMaster
) -> pd.DataFrame:
    """Rows for securities chosen by the rule, each under its ticker on the base session."""
    table = master.table
    spans = table[
        (table["market_type"] == ROUND_LOT_MARKET)
        & table["security_id"].astype(str).isin(selected)
        & (table["first_session"] <= base)
    ].sort_values(["security_id", "last_session", "ticker"])
    ticker_of = spans.drop_duplicates("security_id", keep="last").set_index("security_id")[
        "ticker"
    ]
    return pd.DataFrame(
        {
            "index": TITLE_SHEET,
            "period": label,
            "ticker": [str(ticker_of.get(security, "")) for security in selected],
            "security_id": selected,
            "name": "",
            "share_class": "",
            "theoretical_quantity": float("nan"),
            "weight_pct": float("nan"),
            "base_session": base,
            "source": PortfolioSource.LIQUIDITY_RULE.value,
            "source_file": "",
        }
    )


def _overlap(
    real: pd.DataFrame,
    periods: pd.DataFrame,
    traded: TradedValues,
    settings: LiquiditySettings,
) -> dict[str, float]:
    """Share of each real ``IBOV`` portfolio the rule selects with as many members."""
    ibov = real[(real["index"] == TITLE_SHEET) & (real["period"] >= OVERLAP_FROM)].merge(
        periods[["period", "effective_from"]], on="period", how="inner"
    )
    shares: dict[str, float] = {}
    for label, group in ibov.groupby("period", sort=True):
        base = base_session(traded.calendar, group["effective_from"].iloc[0])
        size = len(group)
        if base is not None and (selected := select_by_liquidity(traded, base, size, settings)):
            members = set(group["security_id"].dropna().astype(str))
            shares[str(label)] = round(len(members & set(selected)) / size, 4)
    return shares


def _absent_at_start(
    constituents: pd.DataFrame,
    periods: pd.DataFrame,
    traded: TradedValues,
    exits: pd.DataFrame,
) -> pd.DataFrame:
    """Members without a bar on the period's first session that are not exits."""
    members = constituents.dropna(subset=["security_id"]).merge(
        periods[["period", "effective_from"]], on="period", how="inner"
    )
    table = traded.table
    starts = table["session_date"].isin(periods["effective_from"])
    bars = table.loc[starts, ["session_date", "security_id"]].rename(
        columns={"session_date": "effective_from"}
    )
    probe = members.astype({"security_id": "string"}).merge(
        bars.assign(traded=True), on=["effective_from", "security_id"], how="left"
    )
    probe = probe.merge(
        exits[["index", "period", "security_id"]].assign(exit=True),
        on=["index", "period", "security_id"],
        how="left",
    )
    missing = probe[probe["traded"].isna() & probe["exit"].isna()]
    return missing[["index", "period", "ticker"]].reset_index(drop=True)


def _summary(
    constituents: pd.DataFrame,
    periods: pd.DataFrame,
    superseded: dict[str, int],
    inputs: dict[str, str | None],
    exits: pd.DataFrame,
    absent: pd.DataFrame,
    overlap: dict[str, float],
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
        ibov_gaps=[label for label in labels if label not in ibov_periods],
        liquidity_periods=sorted(
            compositions.loc[
                compositions["source"] == PortfolioSource.LIQUIDITY_RULE.value, "period"
            ].astype(str)
        ),
        liquidity_overlap=overlap,
        exits=_counts(exits["reason"]),
        absent_at_start=_counts(absent["index"]),
        absent_at_start_strict=[
            {"index": str(row.index), "period": str(row.period), "ticker": str(row.ticker)}
            for row in absent[absent["index"].isin(STRICT_INDICES)].itertuples(index=False)
        ],
        outside_calendar=sorted(set(compositions["period"].astype(str)) - set(labels)),
        inputs=inputs,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# --- Storage ---


def _write(store: DataStore, tables: dict[str, pd.DataFrame], summary: UniverseSummary) -> None:
    folder = store.curated(*CURATED_PARTS)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
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
