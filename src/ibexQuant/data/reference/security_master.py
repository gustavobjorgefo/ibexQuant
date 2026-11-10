# ibexQuant\src\ibexQuant\data\reference\security_master.py

"""
Security master: permanent identifiers across ticker changes (ADR 0012).

Responsibility
--------------
Cut ``bars_daily`` into spans — the sessions of one ``(ticker, isin,
market_type)`` — and join them into securities with the rules of H2.4:
same ISIN, internal code to ISIN, adjacent renames confirmed by CNPJ, and
hand-made overrides. Write the master, the candidates and a summary
(H2.12), and answer ticker-to-security questions (H2.13).

Contract
--------
- A security is a chain of ISINs in time; a link never merges two
  securities nor joins listings that overlapped (H2.5).
- Unconfirmed candidates stay unlinked (H2.7).
- The ``security_id`` is the ISIN of the security's earliest span (H2.11);
  identifier changes against the previous build are reported.

What does not belong here
-------------------------
- Exchange ratios, conversions and mergers between securities — H3.
- Serving bars — see ``ibexQuant.data.ingestion.b3.provider``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Self

import pandas as pd

from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS as BARS_PARTS
from ibexQuant.data.ingestion.b3.bars_daily import MANIFEST_FILE, PARTITION_FILE
from ibexQuant.data.ingestion.cvm import fca_files, read_fca_securities
from ibexQuant.data.ingestion.provider import DateLike
from ibexQuant.data.reference.errors import (
    SecurityMasterBuildError,
    SecurityMasterNotBuiltError,
    SecurityNotFoundError,
)
from ibexQuant.data.reference.overrides import IsinPair, LinkOverrides
from ibexQuant.data.store import DataStore
from ibexQuant.data.store.atomic import replace_atomically

CURATED_PARTS: Final[tuple[str, ...]] = ("b3",)
MASTER_FILE: Final[str] = "security_master.parquet"
CANDIDATES_FILE: Final[str] = "link_candidates.parquet"
SUMMARY_FILE: Final[str] = "security_master.json"
SUMMARY_SCHEMA_VERSION: Final[int] = 1

MAX_GAP_SESSIONS: Final[int] = 5
"""Sessions allowed between the end of a listing and the start of its successor."""

LINKABLE_TYPES: Final[frozenset[str]] = frozenset({"AC", "UN", "CT", "BD"})
"""ISIN type codes (characters 7-8) of shares, units, fund quotas and BDRs."""

ROUND_LOT: Final[int] = 10

MASTER_COLUMNS: Final[tuple[str, ...]] = (
    "security_id",
    "ticker",
    "isin",
    "market_type",
    "first_session",
    "last_session",
    "sessions",
    "link_rule",
    "cnpj",
)

# Link rules recorded per span (H2.12).
FOUNDING: Final[str] = "founding"
SAME_ISIN: Final[str] = "same_isin"
INTERNAL_CODE: Final[str] = "internal_code"
CNPJ: Final[str] = "cnpj"
OVERRIDE: Final[str] = "override"

# Candidate statuses (H2.12).
LINKED_BY_CNPJ: Final[str] = "linked_by_cnpj"
LINKED_BY_OVERRIDE: Final[str] = "linked_by_override"
REJECTED: Final[str] = "rejected"
PENDING: Final[str] = "pending"
CONFLICT: Final[str] = "conflict"
SUPERSEDED: Final[str] = "superseded"
STATUSES: Final[tuple[str, ...]] = (
    LINKED_BY_CNPJ,
    LINKED_BY_OVERRIDE,
    REJECTED,
    PENDING,
    CONFLICT,
    SUPERSEDED,
)


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """
    Outcome of one build, also written to ``security_master.json``.

    Attributes
    ----------
    spans, isins, securities : int
        Rows of the master, distinct ISINs, distinct securities.
    links : dict[str, int]
        Links between different ISINs, by rule.
    candidates : dict[str, int]
        Rule-3 candidates, by status.
    ambiguous_internal_codes : int
        Pre-1995 codes with more than one possible ISIN successor (not linked).
    ambiguous_issuers : list[str]
        Issuer codes that map to more than one CNPJ (not used).
    identifier_changes : int
        Spans whose ``security_id`` differs from the previous build.
    inputs : dict[str, str | None]
        SHA-256 of each input file, by name.
    built_at : str
        UTC build time, ISO format.
    """

    spans: int
    isins: int
    securities: int
    links: dict[str, int]
    candidates: dict[str, int]
    ambiguous_internal_codes: int
    ambiguous_issuers: list[str]
    identifier_changes: int
    inputs: dict[str, str | None]
    built_at: str


def is_isin(code: str) -> bool:
    """
    Tell whether ``code`` is a well-formed ISIN with a valid check digit (ISO 6166).

    Pre-1995 exchange-internal codes (``ACESACON``) are not.
    """
    if len(code) != 12 or not code[:2].isalpha() or not code[2:11].isalnum():
        return False
    if not code[11].isdigit():
        return False
    digits = "".join(str(int(char, 36)) for char in code[:11].upper())
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 0:
            value = value * 2 - 9 if value * 2 > 9 else value * 2
        total += value
    return (10 - total % 10) % 10 == int(code[11])


def build_security_master(store: DataStore, overrides: LinkOverrides) -> BuildSummary:
    """
    Build the master, the candidates and the summary from ``bars_daily``.

    Parameters
    ----------
    store : DataStore
        The data store, with ``bars_daily`` built (H1).
    overrides : LinkOverrides
        Hand-made decisions; ``LinkOverrides.none()`` to build without them.

    Returns
    -------
    BuildSummary
        Counts and inputs of the build.

    Raises
    ------
    SecurityMasterNotBuiltError
        If ``bars_daily`` has no partition.
    SecurityMasterBuildError
        If an override names an unknown ISIN, is not in time order, rejects a
        pair that is not a candidate, or breaks a chain.
    """
    bars = _load_bars(store)
    sessions = pd.DatetimeIndex(sorted(bars["session_date"].unique()))
    spans = _spans(bars, sessions)
    nodes = _nodes(spans)

    fca_paths = fca_files(store)
    issuer_cnpj, ambiguous_issuers = _issuer_cnpj(read_fca_securities(fca_paths), spans)
    nodes["cnpj"] = nodes["isin"].map(lambda isin: issuer_cnpj.get(isin[2:6]))

    chain = _Chain(nodes)
    rejected = {(pair.from_isin, pair.to_isin) for pair in overrides.rejections}
    for pair in overrides.links:
        chain.link_override(pair)
    ambiguous_codes = _link_internal_codes(chain, nodes, spans)
    candidates = _candidates(nodes, sessions, bars)
    _check_rejections(overrides.rejections, candidates)
    candidates["status"] = _decide(chain, candidates, rejected, overrides.links)

    master = _master(spans, nodes, chain)
    changes = _identifier_changes(store, master)
    _write(store, master, candidates)

    inputs: dict[str, str | None] = {
        "bars_daily/manifest.json": _sha256(store.curated(*BARS_PARTS, MANIFEST_FILE)),
        "ticker_links.yaml": overrides.sha256,
        **{f"cvm/fca/{path.name}": _sha256(path) for path in fca_paths},
    }
    summary = BuildSummary(
        spans=len(master),
        isins=int(master["isin"].nunique()),
        securities=int(master["security_id"].nunique()),
        links=chain.counts(),
        candidates={status: int((candidates["status"] == status).sum()) for status in STATUSES},
        ambiguous_internal_codes=ambiguous_codes,
        ambiguous_issuers=sorted(ambiguous_issuers),
        identifier_changes=changes,
        inputs=inputs,
        built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    _write_summary(store, summary)
    return summary


def read_candidates(store: DataStore) -> pd.DataFrame:
    """
    Return ``link_candidates.parquet``.

    Raises
    ------
    SecurityMasterNotBuiltError
        If the master was never built.
    """
    path = store.curated(*CURATED_PARTS, CANDIDATES_FILE)
    if not path.is_file():
        raise SecurityMasterNotBuiltError(
            "No security master yet. Run `ibexquant data build security-master`."
        )
    return pd.read_parquet(path)


class SecurityMaster:
    """
    Read access to the built master (ADR 0012, H2.13).

    Parameters
    ----------
    table : pandas.DataFrame
        The master, with :data:`MASTER_COLUMNS`.

    Examples
    --------
    >>> master = SecurityMaster.load(DataStore.open())  # doctest: +SKIP
    >>> master.resolve("BHIA3", "2024-06-03")  # doctest: +SKIP
    'BRVVARACNOR1'
    """

    def __init__(self, table: pd.DataFrame) -> None:
        missing = sorted(set(MASTER_COLUMNS) - set(table.columns))
        if missing:
            raise ValueError(f"The master table lacks column(s) {missing}.")
        self._table: pd.DataFrame = table.reset_index(drop=True)

    @classmethod
    def load(cls, store: DataStore) -> Self:
        """
        Load ``curated/b3/security_master.parquet``.

        Raises
        ------
        SecurityMasterNotBuiltError
            If the master was never built.
        """
        path = store.curated(*CURATED_PARTS, MASTER_FILE)
        if not path.is_file():
            raise SecurityMasterNotBuiltError(
                "No security master yet. Run `ibexquant data build security-master`."
            )
        return cls(pd.read_parquet(path))

    # --- Queries ---

    @property
    def table(self) -> pd.DataFrame:
        """The master, one row per span."""
        return self._table.copy()

    def resolve(self, ticker: str, as_of: DateLike) -> str:
        """
        Return the ``security_id`` of the span carrying ``ticker`` on ``as_of``.

        Raises
        ------
        SecurityNotFoundError
            If no span of ``ticker`` covers ``as_of``.
        ValueError
            If two securities carried ``ticker`` on ``as_of``.
        """
        day = pd.Timestamp(as_of).normalize()
        table = self._table
        rows = table[
            (table["ticker"] == ticker)
            & (table["first_session"] <= day)
            & (table["last_session"] >= day)
        ]
        found = rows["security_id"].unique()
        if len(found) == 0:
            raise SecurityNotFoundError(f"No listing of {ticker} on {day.date()}.")
        if len(found) > 1:
            raise ValueError(f"{ticker} carried {sorted(found)} on {day.date()}.")
        return str(found[0])

    def spans(self, security_id: str) -> pd.DataFrame:
        """
        Return the spans of ``security_id``, in time order.

        Raises
        ------
        SecurityNotFoundError
            If the identifier is unknown.
        """
        rows = self._table[self._table["security_id"] == security_id]
        if rows.empty:
            raise SecurityNotFoundError(f"Unknown security {security_id}.")
        return rows.sort_values(["first_session", "ticker"]).reset_index(drop=True)

    def ticker_at(self, security_id: str, as_of: DateLike) -> str:
        """
        Return the round-lot ticker of ``security_id`` on ``as_of``.

        When two tickers of the same ISIN overlap during a change, the one
        that started later is returned.

        Raises
        ------
        SecurityNotFoundError
            If the security had no round-lot listing on ``as_of``.
        """
        day = pd.Timestamp(as_of).normalize()
        rows = self.spans(security_id)
        rows = rows[
            (rows["market_type"] == ROUND_LOT)
            & (rows["first_session"] <= day)
            & (rows["last_session"] >= day)
        ]
        if rows.empty:
            raise SecurityNotFoundError(f"{security_id} had no round-lot listing on {day.date()}.")
        return str(rows.sort_values("first_session").iloc[-1]["ticker"])


# --- Inputs ---


def _load_bars(store: DataStore) -> pd.DataFrame:
    folder = store.curated(*BARS_PARTS)
    paths = sorted(folder.glob(f"year=*/{PARTITION_FILE}")) if folder.is_dir() else []
    if not paths:
        raise SecurityMasterNotBuiltError(
            "No bars_daily partition. Run `ibexquant data ingest cotahist --all` first."
        )
    columns = ["session_date", "ticker", "isin", "market_type", "open", "close"]
    bars = pd.concat([pd.read_parquet(path, columns=columns) for path in paths])
    bars["ticker"] = bars["ticker"].astype(str)
    bars["isin"] = bars["isin"].astype(str)
    return bars.reset_index(drop=True)


def _spans(bars: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    spans = (
        bars.groupby(["ticker", "isin", "market_type"], observed=True)["session_date"]
        .agg(first_session="min", last_session="max", sessions="count")
        .reset_index()
    )
    spans["first_pos"] = sessions.get_indexer(pd.DatetimeIndex(spans["first_session"]))
    spans["last_pos"] = sessions.get_indexer(pd.DatetimeIndex(spans["last_session"]))
    return spans


def _nodes(spans: pd.DataFrame) -> pd.DataFrame:
    """One row per ISIN; timing from round-lot spans when there are any."""
    round_lot = spans[spans["market_type"] == ROUND_LOT]
    timed = pd.concat([round_lot, spans[~spans["isin"].isin(round_lot["isin"])]])
    first = timed.sort_values("first_pos").groupby("isin").first()
    last = timed.sort_values("last_pos").groupby("isin").last()
    nodes = pd.DataFrame(
        {
            "isin": first.index,
            "first_pos": first["first_pos"].to_numpy(),
            "last_pos": last.loc[first.index, "last_pos"].to_numpy(),
            "first_session": first["first_session"].to_numpy(),
            "last_session": last.loc[first.index, "last_session"].to_numpy(),
            "first_ticker": first["ticker"].to_numpy(),
            "last_ticker": last.loc[first.index, "ticker"].to_numpy(),
        }
    )
    nodes["is_isin"] = nodes["isin"].map(is_isin)
    nodes["kind"] = nodes["isin"].str[6:11].where(nodes["is_isin"])
    return nodes.set_index("isin", drop=False)


def _issuer_cnpj(fca: pd.DataFrame, spans: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    """Map issuer codes to CNPJs through the tickers each FCA year names (H2.6)."""
    if fca.empty:
        return {}, set()
    round_lot = spans[spans["market_type"] == ROUND_LOT]
    joined = fca.astype({"ticker": str, "cnpj": str}).merge(round_lot, on="ticker")
    start = pd.to_datetime(joined["reference_year"].astype(str) + "-01-01")
    end = pd.to_datetime(joined["reference_year"].astype(str) + "-12-31")
    joined = joined[(joined["first_session"] <= end) & (joined["last_session"] >= start)]
    joined = joined[joined["isin"].map(is_isin).astype(bool)]
    if joined.empty:
        return {}, set()
    by_issuer = joined.assign(issuer=joined["isin"].str[2:6]).groupby("issuer")["cnpj"].unique()
    mapping = {
        str(issuer): str(cnpjs[0]) for issuer, cnpjs in by_issuer.items() if len(cnpjs) == 1
    }
    ambiguous = {str(issuer) for issuer, cnpjs in by_issuer.items() if len(cnpjs) > 1}
    return mapping, ambiguous


# --- Linking ---


class _Chain:
    """Predecessor / successor links between ISINs, enforcing H2.5."""

    def __init__(self, nodes: pd.DataFrame) -> None:
        self._first: dict[str, int] = {
            str(isin): int(pos) for isin, pos in nodes["first_pos"].items()
        }
        self._last: dict[str, int] = {
            str(isin): int(pos) for isin, pos in nodes["last_pos"].items()
        }
        self.successor: dict[str, str] = {}
        self.predecessor: dict[str, str] = {}
        self.rule: dict[str, str] = {}

    def can_link(self, old: str, new: str) -> bool:
        return old not in self.successor and new not in self.predecessor

    def link(self, old: str, new: str, rule: str) -> None:
        self.successor[old] = new
        self.predecessor[new] = old
        self.rule[new] = rule

    def link_override(self, pair: IsinPair) -> None:
        old, new = pair.from_isin, pair.to_isin
        for isin in (old, new):
            if isin not in self._first:
                raise SecurityMasterBuildError(f"Override {old} -> {new}: unknown ISIN {isin}.")
        if self._first[new] <= self._last[old]:
            raise SecurityMasterBuildError(
                f"Override {old} -> {new}: {new} traded before {old} ended; a security "
                "already trading is a conversion (H3), not a link."
            )
        if not self.can_link(old, new):
            raise SecurityMasterBuildError(
                f"Override {old} -> {new} gives a listing two successors or two "
                "predecessors; a security is a chain (H2.5)."
            )
        self.link(old, new, OVERRIDE)

    def counts(self) -> dict[str, int]:
        rules = list(self.rule.values())
        return {rule: rules.count(rule) for rule in (INTERNAL_CODE, CNPJ, OVERRIDE)}


def _link_internal_codes(chain: _Chain, nodes: pd.DataFrame, spans: pd.DataFrame) -> int:
    """Rule 2: a pre-1995 code continues as the only ISIN its ticker carries next."""
    round_lot = spans[spans["market_type"] == ROUND_LOT]
    pairs = round_lot.merge(round_lot, on="ticker", suffixes=("_old", "_new"))
    pairs = pairs[
        ~pairs["isin_old"].map(is_isin)
        & pairs["isin_new"].map(is_isin)
        & (pairs["first_pos_new"] > pairs["last_pos_old"])
    ]
    ambiguous = 0
    for old, group in pairs.groupby("isin_old"):
        successors = group["isin_new"].unique()
        if len(successors) != 1:
            ambiguous += 1
            continue
        new = str(successors[0])
        if chain.can_link(str(old), new):
            chain.link(str(old), new, INTERNAL_CODE)
    return ambiguous


def _candidates(
    nodes: pd.DataFrame, sessions: pd.DatetimeIndex, bars: pd.DataFrame
) -> pd.DataFrame:
    """Rule 3: one listing ends, another of the same kind starts within the gap."""
    flat = nodes.reset_index(drop=True)
    linkable = flat[flat["is_isin"] & flat["kind"].str[:2].isin(LINKABLE_TYPES)]
    pairs = linkable.merge(linkable, on="kind", suffixes=("_old", "_new"))
    gap = pairs["first_pos_new"] - pairs["last_pos_old"]
    pairs = pairs[(gap >= 1) & (gap <= MAX_GAP_SESSIONS)].copy()
    pairs["gap_sessions"] = gap[pairs.index]

    round_lot = bars[bars["market_type"] == ROUND_LOT]
    last_close = round_lot.set_index(["isin", "session_date"])["close"]
    first_open = round_lot.set_index(["isin", "session_date"])["open"]
    last_close = last_close[~last_close.index.duplicated()]
    first_open = first_open[~first_open.index.duplicated()]
    old_keys = pd.MultiIndex.from_arrays([pairs["isin_old"], pairs["last_session_old"]])
    new_keys = pd.MultiIndex.from_arrays([pairs["isin_new"], pairs["first_session_new"]])
    old_close = last_close.reindex(old_keys).to_numpy()
    new_open = first_open.reindex(new_keys).to_numpy()

    return pd.DataFrame(
        {
            "from_isin": pairs["isin_old"].to_numpy(),
            "from_ticker": pairs["last_ticker_old"].to_numpy(),
            "from_last_session": pairs["last_session_old"].to_numpy(),
            "to_isin": pairs["isin_new"].to_numpy(),
            "to_ticker": pairs["first_ticker_new"].to_numpy(),
            "to_first_session": pairs["first_session_new"].to_numpy(),
            "gap_sessions": pairs["gap_sessions"].to_numpy(dtype="int64"),
            "from_cnpj": pairs["cnpj_old"].to_numpy(),
            "to_cnpj": pairs["cnpj_new"].to_numpy(),
            "old_close": old_close,
            "new_open": new_open,
            "ratio": new_open / old_close,
        }
    ).sort_values(["to_first_session", "from_isin", "to_isin"], ignore_index=True)


def _check_rejections(rejections: Iterable[IsinPair], candidates: pd.DataFrame) -> None:
    known = set(zip(candidates["from_isin"], candidates["to_isin"], strict=True))
    for pair in rejections:
        if (pair.from_isin, pair.to_isin) not in known:
            raise SecurityMasterBuildError(
                f"Rejection {pair.from_isin} -> {pair.to_isin} is not a candidate."
            )


def _decide(
    chain: _Chain,
    candidates: pd.DataFrame,
    rejected: set[tuple[str, str]],
    override_links: Iterable[IsinPair],
) -> list[str]:
    """
    Status of each candidate, linking the CNPJ-confirmed ones (rule 3).

    Confirmed pairs are taken closest first (smallest gap), so that a listing
    confirmed with two partners keeps the nearer one; two partners at the
    same distance are a conflict, and neither is linked.
    """
    overridden = {(pair.from_isin, pair.to_isin) for pair in override_links}
    pairs = _pairs(candidates)
    decided = [pair in overridden or pair in rejected for pair in pairs]
    confirmed = (
        candidates["from_cnpj"].notna()
        & (candidates["from_cnpj"] == candidates["to_cnpj"])
        & ~pd.Series(decided, index=candidates.index)
    )

    status: dict[int, str] = {}
    for _, level in candidates[confirmed].groupby("gap_sessions", sort=True):
        open_ = level[[chain.can_link(old, new) for old, new in _pairs(level)]]
        crowded = set(open_["from_isin"][open_["from_isin"].duplicated(keep=False)]) | set(
            open_["to_isin"][open_["to_isin"].duplicated(keep=False)]
        )
        for index, (old, new) in zip(level.index, _pairs(level), strict=True):
            if old in crowded or new in crowded:
                status[index] = CONFLICT
            elif chain.can_link(old, new):
                chain.link(old, new, CNPJ)
                status[index] = LINKED_BY_CNPJ

    result = []
    for index, (old, new) in zip(candidates.index, pairs, strict=True):
        if (old, new) in overridden:
            result.append(LINKED_BY_OVERRIDE)
        elif (old, new) in rejected:
            result.append(REJECTED)
        elif index in status:
            result.append(status[index])
        elif old in chain.successor or new in chain.predecessor:
            result.append(SUPERSEDED)
        else:
            result.append(PENDING)
    return result


def _pairs(candidates: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(candidates["from_isin"], candidates["to_isin"], strict=True))


# --- Output ---


def _master(spans: pd.DataFrame, nodes: pd.DataFrame, chain: _Chain) -> pd.DataFrame:
    security: dict[str, str] = {}
    for isin in nodes.sort_values("first_pos")["isin"]:
        security[isin] = security[chain.predecessor[isin]] if isin in chain.predecessor else isin

    master = spans.sort_values(["first_pos", "ticker", "market_type"]).copy()
    master["security_id"] = master["isin"].map(security)
    first_of_isin = ~master.duplicated("isin")
    node_rule = master["isin"].map(lambda isin: chain.rule.get(isin, FOUNDING))
    master["link_rule"] = node_rule.where(first_of_isin, SAME_ISIN)
    master["cnpj"] = master["isin"].map(nodes["cnpj"]).astype("string")
    master["ticker"] = master["ticker"].astype("string")
    master["isin"] = master["isin"].astype("string")
    master["security_id"] = master["security_id"].astype("string")
    master["link_rule"] = master["link_rule"].astype("string")
    master["market_type"] = master["market_type"].astype("int16")
    master["sessions"] = master["sessions"].astype("int64")
    return master[list(MASTER_COLUMNS)].reset_index(drop=True)


def _identifier_changes(store: DataStore, master: pd.DataFrame) -> int:
    path = store.curated(*CURATED_PARTS, MASTER_FILE)
    if not path.is_file():
        return 0
    key = ["ticker", "isin", "market_type"]
    previous = pd.read_parquet(path, columns=[*key, "security_id"])
    joined = master[[*key, "security_id"]].merge(previous, on=key, suffixes=("", "_before"))
    return int((joined["security_id"] != joined["security_id_before"]).sum())


def _write(store: DataStore, master: pd.DataFrame, candidates: pd.DataFrame) -> None:
    folder = store.curated(*CURATED_PARTS)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in ((MASTER_FILE, master), (CANDIDATES_FILE, candidates)):
        replace_atomically(folder / name, _parquet_writer(frame))


def _parquet_writer(frame: pd.DataFrame) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    return write


def _write_summary(store: DataStore, summary: BuildSummary) -> None:
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
