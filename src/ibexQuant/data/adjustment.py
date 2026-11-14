# ibexQuant\src\ibexQuant\data\adjustment.py

"""
Anchored price adjustment of the B3 base (ADR 0014).

Responsibility
--------------
Turn the corporate actions ledger (ADR 0013) into one factor per ex-session
(:func:`build_adjustment_factors`), and apply those factors to raw bars
relative to an anchor date (:func:`adjust`).

Contract
--------
- An ex-session's factor combines every ``confirmed`` or ``inferred`` event
  placed on it (H4.1, H4.2): with ``P`` the previous raw close, ``V`` the
  value handed to the holder per share (cash gross, spin-off targets at
  their price, the theoretical value of subscription rights) and ``R`` the
  product of the share ratios, ``f_total = (1 - V / P) / R`` and
  ``f_splits = 1 / R``.
- A factor that cannot exist (``V >= P``, no previous close, no price for a
  spin-off target) is not applied and is listed with the reason (H4.3).
- Conversions and exchanges into a security of the same chain are share
  events; into another security, and redemptions and ``OTHER`` events, they
  get no factor (H4.2).
- ``P_adj(s; A) = P_raw(s) * prod(f_e for s < ex_e <= A)`` for every price
  column; volume takes the share ratios only (H4.4–H4.6).
- The factor table is derived and records the SHA-256 of the ledger files
  it came from; :func:`read_adjustment_factors` refuses a table built from
  another ledger (H4.8, H4.9).

What does not belong here
-------------------------
- Placing events on ex-sessions — see ``corporate_actions``.
- Reading bars, resolving tickers and warning about unexplained jumps — see
  ``B3Provider``.
- Crediting event cash to a portfolio — the backtest engine's option (P9).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS as BARS_PARTS
from ibexQuant.data.ingestion.b3.bars_daily import PARTITION_FILE
from ibexQuant.data.reference.corporate_actions import (
    CURATED_PARTS as LEDGER_PARTS,
)
from ibexQuant.data.reference.corporate_actions import (
    EVENTS_FILE,
    EX_SESSIONS_FILE,
    EventStatus,
    ExSessionStatus,
    read_ledger,
)
from ibexQuant.data.reference.event_types import CASH_TYPES, EventType
from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.store import DataStore
from ibexQuant.data.store.atomic import replace_atomically

DERIVED_PARTS: Final[tuple[str, ...]] = ("b3",)
FACTORS_FILE: Final[str] = "adjustment_factors.parquet"
REJECTED_FILE: Final[str] = "adjustment_rejected.parquet"
SUMMARY_FILE: Final[str] = "adjustment_factors.json"
SUMMARY_SCHEMA_VERSION: Final[int] = 1

TARGET_LOOKAHEAD_SESSIONS: Final[int] = 5
"""A spin-off target without a close on the ex-session may take its first close this late."""

FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    "security_id",
    "isin",
    "ex_date",
    "previous_close",
    "cash_value",
    "share_ratio",
    "f_total",
    "f_splits",
    "event_ids",
)
REJECTED_COLUMNS: Final[tuple[str, ...]] = (
    "security_id",
    "isin",
    "ex_date",
    "reason",
    "event_ids",
)
ROUND_LOT_MARKET: Final[int] = 10
PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
VOLUME_COLUMN: Final[str] = "volume"

PAID_TYPES: Final[frozenset[str]] = frozenset(CASH_TYPES - {EventType.REDEMPTION})
"""Cash events that leave the holder with the share; a redemption takes it away (H4.2)."""
SHARE_TYPES: Final[frozenset[str]] = frozenset(
    {EventType.SPLIT, EventType.REVERSE_SPLIT, EventType.STOCK_BONUS}
)
CHAIN_TYPES: Final[frozenset[str]] = frozenset({EventType.CONVERSION, EventType.EXCHANGE})
APPLIED_STATUSES: Final[frozenset[str]] = frozenset({EventStatus.CONFIRMED, EventStatus.INFERRED})


class AdjustmentMode(StrEnum):
    """Which events an adjusted series removes (H4.5)."""

    RAW = "raw"
    SPLITS = "splits"
    TOTAL_RETURN = "total_return"


class RejectionReason(StrEnum):
    """Why an ex-session gets no factor (H4.3)."""

    NO_PREVIOUS_CLOSE = "no_previous_close"
    NO_TARGET_PRICE = "no_target_price"
    VALUE_EXCEEDS_PRICE = "value_exceeds_price"


class AdjustmentFactorsNotBuiltError(FileNotFoundError):
    """The factor table does not exist yet."""


class StaleAdjustmentFactorsError(RuntimeError):
    """The factor table was built from another ledger than the one on disk."""


@dataclass(frozen=True, slots=True)
class FactorSummary:
    """
    Outcome of one build, also written to ``adjustment_factors.json``.

    Attributes
    ----------
    factors : int
        Ex-sessions with a factor.
    rejected : dict[str, int]
        Ex-sessions without a factor, by reason (H4.3).
    ignored : dict[str, int]
        Events that never take a factor, by type: terminal conversions and
        exchanges, redemptions, ``OTHER`` (H4.2).
    unexplained : dict[str, int]
        Ex-sessions of the ledger with no event, by status: their raw move
        stays in adjusted series.
    ledger : dict[str, str]
        SHA-256 of the ledger files the factors were built from.
    built_at : str
        UTC build time, ISO format.
    """

    factors: int
    rejected: dict[str, int]
    ignored: dict[str, int]
    unexplained: dict[str, int]
    ledger: dict[str, str]
    built_at: str


# --- Build ---


def build_adjustment_factors(store: DataStore) -> FactorSummary:
    """
    Compute one factor per ex-session of the ledger and write the tables.

    Parameters
    ----------
    store : DataStore
        Data root with the security master, the ledger and ``bars_daily``.

    Returns
    -------
    FactorSummary
        Counts of the build, also written to ``adjustment_factors.json``.

    Raises
    ------
    LedgerNotBuiltError
        If the ledger was never built.
    SecurityMasterNotBuiltError
        If the security master was never built.
    """
    events, ex = read_ledger(store)
    master = SecurityMaster.load(store).table
    security_of = dict(zip(master["isin"], master["security_id"], strict=True))
    spin_offs = events[events["event_type"] == EventType.SPIN_OFF]
    targets = _target_closes(store, spin_offs)

    factors, rejected, ignored = compute_factors(events, ex, security_of, targets)
    unexplained = ex[
        ex["status"].isin([ExSessionStatus.UNEXPLAINED, ExSessionStatus.UNEXPLAINED_JUMP])
    ]
    summary = FactorSummary(
        factors=len(factors),
        rejected=_counts(rejected["reason"]),
        ignored=ignored,
        unexplained=_counts(unexplained["status"]),
        ledger=_ledger_hashes(store),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    _write(store, factors, rejected, summary)
    return summary


def compute_factors(
    events: pd.DataFrame,
    ex: pd.DataFrame,
    security_of: dict[str, str],
    target_closes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Combine the events placed on each ex-session into its factor (H4.1–H4.3).

    Parameters
    ----------
    events : pandas.DataFrame
        Ledger events (``corporate_actions.parquet``).
    ex : pandas.DataFrame
        Ledger ex-sessions (``ex_sessions.parquet``), for ``previous_close``.
    security_of : dict[str, str]
        ISIN to ``security_id``, from the security master.
    target_closes : pandas.DataFrame
        ``isin``, ``session_date``, ``close`` of spin-off targets, and
        ``earliest_ex``: the earliest ex-session that may take this close,
        :data:`TARGET_LOOKAHEAD_SESSIONS` market sessions before it.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, dict[str, int]]
        Factors (:data:`FACTOR_COLUMNS`), rejected ex-sessions
        (:data:`REJECTED_COLUMNS`) and ignored events by type.
    """
    placed = events[events["status"].isin(APPLIED_STATUSES) & events["ex_session"].notna()].copy()
    kind = placed["event_type"].astype(str)
    chained = kind.isin(CHAIN_TYPES) & (
        placed["target_isin"].map(security_of) == placed["security_id"]
    )
    applied = kind.isin(PAID_TYPES | SHARE_TYPES | {EventType.SPIN_OFF, EventType.SUBSCRIPTION})
    ignored = _counts(kind[~(applied | chained)])
    placed = placed[applied | chained]
    if placed.empty:
        return _empty(FACTOR_COLUMNS), _empty(REJECTED_COLUMNS), ignored

    previous = ex.set_index(["isin", "session_date"])["previous_close"]
    keys = pd.MultiIndex.from_arrays([placed["isin"], placed["ex_session"]])
    price = previous.reindex(keys).to_numpy(dtype="float64")
    placed = placed.assign(previous_close=price)

    kind = placed["event_type"].astype(str)
    ratio = placed["share_ratio"].astype("float64")
    cash = placed["cash_amount"].astype("float64").where(kind.isin(PAID_TYPES), 0.0)
    subscription = (
        ratio * np.maximum(price - placed["subscription_price"].astype("float64"), 0.0)
    ) / (1.0 + ratio)
    target = _target_value(placed, target_closes)
    placed = placed.assign(
        value=cash.fillna(0.0)
        + subscription.where(kind == EventType.SUBSCRIPTION, 0.0).fillna(0.0)
        + target,
        ratio=ratio.where(kind.isin(SHARE_TYPES | CHAIN_TYPES), 1.0),
        missing_target=(kind == EventType.SPIN_OFF) & target.isna(),
    )

    sessions = (
        placed.groupby(["security_id", "ex_session"], sort=True)
        .agg(
            isin=("isin", "first"),
            previous_close=("previous_close", "first"),
            cash_value=("value", "sum"),
            share_ratio=("ratio", "prod"),
            missing_target=("missing_target", "any"),
            event_ids=("event_id", ",".join),
        )
        .reset_index()
        .rename(columns={"ex_session": "ex_date"})
    )
    reason = np.select(
        [
            sessions["previous_close"].isna().to_numpy(),
            sessions["missing_target"].to_numpy(dtype=bool),
            (sessions["cash_value"] >= sessions["previous_close"]).to_numpy(),
        ],
        [
            RejectionReason.NO_PREVIOUS_CLOSE.value,
            RejectionReason.NO_TARGET_PRICE.value,
            RejectionReason.VALUE_EXCEEDS_PRICE.value,
        ],
        default="",
    )
    sessions = sessions.assign(reason=reason)

    rejected = sessions[sessions["reason"] != ""]
    kept = sessions[sessions["reason"] == ""]
    kept = kept.assign(
        f_total=(1.0 - kept["cash_value"] / kept["previous_close"]) / kept["share_ratio"],
        f_splits=1.0 / kept["share_ratio"],
    )
    return (
        kept[list(FACTOR_COLUMNS)].reset_index(drop=True),
        rejected[list(REJECTED_COLUMNS)].reset_index(drop=True),
        ignored,
    )


# --- Read ---


def read_adjustment_factors(store: DataStore) -> pd.DataFrame:
    """
    Load the factor table, checking that it matches the ledger on disk.

    Raises
    ------
    AdjustmentFactorsNotBuiltError
        If the factors were never built.
    StaleAdjustmentFactorsError
        If the ledger changed since the factors were built.
    """
    folder = store.derived(*DERIVED_PARTS)
    table, summary = folder / FACTORS_FILE, folder / SUMMARY_FILE
    if not (table.is_file() and summary.is_file()):
        raise AdjustmentFactorsNotBuiltError(
            "No adjustment factors yet. Run `ibexquant data build adjustment-factors`."
        )
    recorded = json.loads(summary.read_text(encoding="utf-8"))["ledger"]
    if recorded != _ledger_hashes(store):
        raise StaleAdjustmentFactorsError(
            "The corporate actions ledger changed after the adjustment factors were built. "
            "Run `ibexquant data build adjustment-factors`."
        )
    return pd.read_parquet(table)


# --- Apply ---


def adjust(
    bars: pd.DataFrame,
    factors: pd.DataFrame,
    anchor: pd.Timestamp,
    mode: AdjustmentMode | str = AdjustmentMode.TOTAL_RETURN,
) -> pd.DataFrame:
    """
    Adjust raw bars relative to an anchor date (H4.4–H4.6).

    Parameters
    ----------
    bars : pandas.DataFrame
        Long frame with ``session_date``, ``security_id`` and any of
        :data:`PRICE_COLUMNS` and :data:`VOLUME_COLUMN`.
    factors : pandas.DataFrame
        Factor table (:data:`FACTOR_COLUMNS`).
    anchor : pandas.Timestamp
        Events with ``ex_date <= anchor`` are applied to the bars before them.
    mode : AdjustmentMode or str, default ``"total_return"``
        ``"raw"``, ``"splits"`` or ``"total_return"``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``bars`` with adjusted prices and volume, same row order.

    Raises
    ------
    ValueError
        If ``mode`` is not a known mode.
    """
    mode = AdjustmentMode(mode)
    out = bars.copy()
    if mode is AdjustmentMode.RAW or out.empty:
        return out
    factor_column = "f_total" if mode is AdjustmentMode.TOTAL_RETURN else "f_splits"
    used = factors[factors["ex_date"] <= anchor]
    by_security: dict[str, pd.DataFrame] = {
        str(key): group for key, group in used.groupby("security_id", sort=False)
    }

    price_multiplier = np.ones(len(out))
    volume_multiplier = np.ones(len(out))
    sessions = out["session_date"].to_numpy(dtype="datetime64[ns]")
    for security, positions in out.groupby("security_id", sort=False).indices.items():
        group = by_security.get(str(security))
        if group is None:
            continue
        group = group.sort_values("ex_date")
        dates = group["ex_date"].to_numpy(dtype="datetime64[ns]")
        # Suffix products: entry k multiplies every bar before the k-th ex-date.
        prices = _suffix_products(group[factor_column].to_numpy(dtype="float64"))
        shares = _suffix_products(group["share_ratio"].to_numpy(dtype="float64"))
        first_after = np.searchsorted(dates, sessions[positions], side="right")
        price_multiplier[positions] = prices[first_after]
        volume_multiplier[positions] = shares[first_after]

    for column in PRICE_COLUMNS:
        if column in out:
            out[column] = out[column].to_numpy(dtype="float64") * price_multiplier
    if VOLUME_COLUMN in out:
        out[VOLUME_COLUMN] = out[VOLUME_COLUMN].to_numpy(dtype="float64") * volume_multiplier
    return out


# --- Helpers ---


def _suffix_products(values: np.ndarray) -> np.ndarray:
    """``out[k] = prod(values[k:])``, with ``out[len(values)] = 1``."""
    return np.append(np.cumprod(values[::-1])[::-1], 1.0)


def _target_value(placed: pd.DataFrame, target_closes: pd.DataFrame) -> pd.Series:
    """``r * P_target`` for spin-offs, ``NaN`` when the target has no price, 0 otherwise."""
    value = pd.Series(0.0, index=placed.index)
    spin = placed[placed["event_type"].astype(str) == EventType.SPIN_OFF]
    if spin.empty:
        return value
    # Parquet and pandas disagree on datetime units and string dtypes: align the keys.
    left = spin[["ex_session", "target_isin", "share_ratio"]].reset_index()
    left = left.astype({"ex_session": "datetime64[ns]", "target_isin": "object"})
    left = left.sort_values("ex_session", kind="stable")
    right = target_closes.astype(
        {"isin": "object", "session_date": "datetime64[ns]", "earliest_ex": "datetime64[ns]"}
    ).sort_values("session_date", kind="stable")
    found = pd.merge_asof(
        left,
        right,
        left_on="ex_session",
        right_on="session_date",
        left_by="target_isin",
        right_by="isin",
        direction="forward",
    )
    in_time = found["session_date"].notna() & (found["earliest_ex"] <= found["ex_session"])
    priced = found["share_ratio"].astype("float64") * found["close"].astype("float64")
    value[found["index"].to_numpy()] = priced.where(in_time, np.nan).to_numpy()
    return value


def _target_closes(store: DataStore, spin_offs: pd.DataFrame) -> pd.DataFrame:
    """Round-lot closes of the spin-off targets, each with its earliest ex-session."""
    if spin_offs.empty:
        return _no_closes()
    first = pd.Timestamp(spin_offs["ex_session"].min())
    last = pd.Timestamp(spin_offs["ex_session"].max())
    frames = []
    for year in range(first.year, last.year + 2):
        path = store.curated(*BARS_PARTS, f"year={year}", PARTITION_FILE)
        if path.is_file():
            frames.append(
                pd.read_parquet(path, columns=["session_date", "isin", "market_type", "close"])
            )
    if not frames:
        return _no_closes()
    bars = pd.concat(frames, ignore_index=True)
    return target_closes_from(bars, set(spin_offs["target_isin"].dropna()))


def target_closes_from(bars: pd.DataFrame, targets: set[str]) -> pd.DataFrame:
    """
    Select the targets' round-lot closes and the earliest ex-session each may price.

    Parameters
    ----------
    bars : pandas.DataFrame
        ``session_date``, ``isin``, ``market_type`` and ``close`` of every
        security, so that the market calendar is complete.
    targets : set[str]
        ISINs of the spin-off targets.

    Returns
    -------
    pandas.DataFrame
        ``isin``, ``session_date``, ``close``, ``earliest_ex``.
    """
    calendar = pd.DatetimeIndex(sorted(bars["session_date"].unique()))
    earlier = np.maximum(np.arange(len(calendar)) - TARGET_LOOKAHEAD_SESSIONS, 0)
    earliest = pd.Series(calendar[earlier], index=calendar)
    rows = bars[(bars["market_type"] == ROUND_LOT_MARKET) & bars["isin"].isin(targets)]
    closes = rows[["isin", "session_date", "close"]].reset_index(drop=True)
    return closes.assign(earliest_ex=earliest.reindex(closes["session_date"]).to_numpy())


def _no_closes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isin": pd.Series(dtype="object"),
            "session_date": pd.Series(dtype="datetime64[ns]"),
            "close": pd.Series(dtype="float64"),
            "earliest_ex": pd.Series(dtype="datetime64[ns]"),
        }
    )


def _ledger_hashes(store: DataStore) -> dict[str, str]:
    folder = store.curated(*LEDGER_PARTS)
    return {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in (EVENTS_FILE, EX_SESSIONS_FILE)
    }


def _counts(values: Iterable[object] | pd.Series) -> dict[str, int]:
    series = pd.Series(list(values), dtype="object")
    return {str(key): int(count) for key, count in series.value_counts().sort_index().items()}


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _write(
    store: DataStore, factors: pd.DataFrame, rejected: pd.DataFrame, summary: FactorSummary
) -> None:
    folder = store.derived(*DERIVED_PARTS)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in ((FACTORS_FILE, factors), (REJECTED_FILE, rejected)):
        replace_atomically(folder / name, _parquet_writer(frame))
    text = json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, **asdict(summary)}, indent=2)
    replace_atomically(
        folder / SUMMARY_FILE,
        lambda path: path.write_text(text + "\n", encoding="utf-8"),
    )


def _parquet_writer(frame: pd.DataFrame) -> Callable[[Path], None]:
    def write(path: Path) -> None:
        frame.to_parquet(path, engine="pyarrow", index=False)

    return write
