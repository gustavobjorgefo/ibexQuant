# ibexQuant\src\ibexQuant\data\reference\corporate_action_sources.py

"""
Source events of the corporate actions ledger, in one shape.

Responsibility
--------------
Turn what each source says into rows of :data:`SOURCE_EVENT_COLUMNS`: an ISIN,
a data-com or a distribution number, an event type and its amounts (ADR 0013,
H3.4, H3.5 and H3.13).

Contract
--------
- Only events of round-lot shares and units (ISIN types ``AC``, ``UN``,
  ``CD``) are kept; funds, BDRs and receipts are outside the ledger.
- B3's cash history names an issuer code and a share class, not an ISIN. The
  class gives ISIN characters 7–11; the ISINs of ``bars_daily`` that start
  with ``BR``, the code and those characters point to securities; the ISIN of
  those securities in force on the data-com is the event's ISIN (H3.13). The
  bootstrap's tickers are placed the same way, from the ticker's securities.
  A row that resolves to no ISIN or to several is returned apart, with the
  reason.
- B3's recent cash events repeat part of the cash history; both are
  ``b3_listed``, and a row present in both is kept as many times as the
  source that lists it most.

What does not belong here
-------------------------
- Placing events on ex-sessions, merging sources, statuses — see the ledger.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd

from ibexQuant.data.ingestion.b3.listed_companies import ListedCompanies
from ibexQuant.data.reference.event_types import (
    CREDITED_LABELS,
    LISTED_CASH_LABELS,
    SUBSCRIPTION_LABELS,
    EventType,
    event_type,
    stock_event,
)
from ibexQuant.data.reference.ex_sessions import ROUND_LOT, SKELETON_TYPES

SOURCE_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "isin",
    "last_cum_date",
    "distribution_number",
    "event_type",
    "cash_amount",
    "share_ratio",
    "target_isin",
    "subscription_price",
    "payment_date",
    "approved_on",
    "source",
    "origin",
    "label",
)
UNRESOLVED_COLUMNS: Final[tuple[str, ...]] = ("source", "origin", "key", "last_cum_date", "reason")

CLASS_KINDS: Final[dict[str, str]] = {
    "ON": "ACNOR",
    "PN": "ACNPR",
    **{f"PN{letter}": f"ACNP{letter}" for letter in "ABCDEFGHV"},
    "UNT": "CDAM",
}
"""Share class of B3's cash history -> ISIN characters 7–11 (H3.13)."""

AMOUNT_DECIMALS: Final[int] = 8


class Source(StrEnum):
    """Ledger sources, by priority (H3.4)."""

    MANUAL = "manual"
    B3_LISTED = "b3_listed"
    B3_CREDITED = "b3_credited"
    BOOTSTRAP = "bootstrap"
    INFERRED = "inferred"


SOURCE_PRIORITY: Final[dict[str, int]] = {source.value: rank for rank, source in enumerate(Source)}


# --- ISIN in force on a date ---


class IsinLocator:
    """
    Find the ISIN a security traded under on a date (H3.13).

    Parameters
    ----------
    master : pandas.DataFrame
        The security master table (ADR 0012).
    """

    def __init__(self, master: pd.DataFrame) -> None:
        round_lot = master[master["market_type"] == ROUND_LOT]
        intervals = (
            round_lot.groupby(["security_id", "isin"], observed=True)
            .agg(first=("first_session", "min"), last=("last_session", "max"))
            .reset_index()
            .sort_values(["security_id", "first"])
        )
        self._intervals: dict[str, pd.DataFrame] = {
            str(security): group.reset_index(drop=True)
            for security, group in intervals.groupby("security_id", sort=False)
        }
        self._by_isin: dict[str, str] = dict(
            zip(round_lot["isin"].astype(str), round_lot["security_id"].astype(str), strict=True)
        )
        self._by_ticker: dict[str, set[str]] = {}
        for ticker, security in zip(
            round_lot["ticker"].astype(str), round_lot["security_id"].astype(str), strict=True
        ):
            self._by_ticker.setdefault(ticker, set()).add(security)
        self._isins: np.ndarray = np.array(sorted(self._by_isin))

    def securities_with_prefix(self, prefix: str) -> set[str]:
        """Return the securities of the ISINs that start with ``prefix``."""
        start = int(np.searchsorted(self._isins, prefix, side="left"))
        found: set[str] = set()
        for isin in self._isins[start:]:
            if not isin.startswith(prefix):
                break
            found.add(self._by_isin[isin])
        return found

    def securities_with_ticker(self, ticker: str) -> set[str]:
        """Return the securities that ever traded round lots under ``ticker``."""
        return set(self._by_ticker.get(ticker, set()))

    def isin_at(self, securities: Iterable[str], day: pd.Timestamp) -> tuple[str | None, str]:
        """
        Return the one ISIN of ``securities`` that traded on a span containing ``day``.

        Returns
        -------
        tuple[str | None, str]
            The ISIN and an empty reason, or ``None`` and the reason.
        """
        found: set[str] = set()
        for security in securities:
            intervals = self._intervals[security]
            inside = intervals[(intervals["first"] <= day) & (intervals["last"] >= day)]
            found.update(inside["isin"])
        if len(found) == 1:
            return found.pop(), ""
        return None, "no ISIN traded on the date" if not found else "several ISINs"


# --- B3 listed companies ---


def listed_events(
    listed: ListedCompanies, locator: IsinLocator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the events of B3's listed-company snapshots and the rows not placed.

    Parameters
    ----------
    listed : ListedCompanies
        Output of :func:`~ibexQuant.data.ingestion.b3.listed_companies.read_listed_companies`.
    locator : IsinLocator
        Built on the security master.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Events (:data:`SOURCE_EVENT_COLUMNS`) and unresolved cash-history
        rows (:data:`UNRESOLVED_COLUMNS`).
    """
    stock = _stock(listed.stock)
    recent = _cash(listed.cash_recent, listed.cash_recent["isin"], "cash_recent")
    history_isins, unresolved = _class_isins(listed.cash_history, locator)
    history = _cash(listed.cash_history, history_isins, "cash_history")
    subscriptions = _subscriptions(listed.subscriptions)
    cash = _union_by_count([recent, history])
    events = pd.concat([stock, cash, subscriptions], ignore_index=True)
    return _keep_skeleton_isins(events), unresolved


def _stock(stock: pd.DataFrame) -> pd.DataFrame:
    kinds: list[str | None] = []
    ratios: list[float] = []
    targets: list[str] = []
    for isin, target, label, factor in zip(
        stock["isin"].astype(str),
        stock["target_isin"].astype(str),
        stock["label"].astype(str),
        stock["factor"].astype(float),
        strict=True,
    ):
        other = bool(target) and target != isin
        kind, ratio = stock_event(label, factor, other)
        kinds.append(None if kind is None else kind.value)
        ratios.append(ratio)
        targets.append(target if other else "")
    events = pd.DataFrame(
        {
            "isin": stock["isin"],
            "last_cum_date": stock["last_cum_date"],
            "event_type": kinds,
            "share_ratio": ratios,
            "target_isin": targets,
            "approved_on": stock["approved_on"],
            "origin": "stock:" + stock["code"],
            "label": stock["label"],
        },
        index=stock.index,
    )
    return _frame(events[events["event_type"].notna()], Source.B3_LISTED)


def _cash(table: pd.DataFrame, isins: pd.Series, part: str) -> pd.DataFrame:
    if table.empty:
        return _frame([], Source.B3_LISTED)
    kinds = _types(table["label"], LISTED_CASH_LABELS, "listed cash")
    keep = isins.notna() & kinds.notna()
    events = pd.DataFrame(
        {
            "isin": isins[keep],
            "last_cum_date": table.loc[keep, "last_cum_date"],
            "event_type": kinds[keep],
            "cash_amount": table.loc[keep, "cash_amount"],
            "payment_date": table.loc[keep, "payment_date"] if "payment_date" in table else pd.NaT,
            "approved_on": table.loc[keep, "approved_on"],
            "origin": part + ":" + table.loc[keep, "code"],
            "label": table.loc[keep, "label"],
        }
    )
    return _frame(events, Source.B3_LISTED)


def _subscriptions(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return _frame([], Source.B3_LISTED)
    kinds = _types(table["label"], SUBSCRIPTION_LABELS, "subscription")
    events = pd.DataFrame(
        {
            "isin": table["isin"],
            "last_cum_date": table["last_cum_date"],
            "event_type": kinds,
            "share_ratio": table["percentage"] / 100.0,
            "target_isin": table["target_isin"],
            "subscription_price": table["price"],
            "approved_on": table["approved_on"],
            "origin": "subscriptions:" + table["code"],
            "label": table["label"],
        }
    )
    return _frame(events, Source.B3_LISTED)


def _class_isins(history: pd.DataFrame, locator: IsinLocator) -> tuple[pd.Series, pd.DataFrame]:
    found: dict[object, str | None] = {}
    unresolved = []
    for (code, share_class), group in history.groupby(["code", "share_class"], sort=False):
        kind = CLASS_KINDS.get(str(share_class))
        securities = locator.securities_with_prefix(f"BR{code}{kind}") if kind else set()
        for index, day in zip(group.index, pd.DatetimeIndex(group["last_cum_date"]), strict=True):
            isin, reason = (
                locator.isin_at(securities, day)
                if securities
                else (None, f"no ISIN for class {share_class}")
            )
            found[index] = isin
            if isin is None:
                unresolved.append(
                    {
                        "source": Source.B3_LISTED.value,
                        "origin": f"cash_history:{code}",
                        "key": f"{code} {share_class}",
                        "last_cum_date": day,
                        "reason": reason,
                    }
                )
    isins = pd.Series([found.get(index) for index in history.index], index=history.index)
    return isins, pd.DataFrame(unresolved, columns=list(UNRESOLVED_COLUMNS))


# --- Earnings credited ---


def credited_events(credited: pd.DataFrame) -> pd.DataFrame:
    """
    Return the events of the *Earnings credited* files.

    They carry a distribution number instead of a data-com (H3.3).
    """
    if credited.empty:
        return _frame([], Source.B3_CREDITED)
    kinds = _types(credited["label"], CREDITED_LABELS, "credited")
    keep = kinds.notna()
    events = pd.DataFrame(
        {
            "isin": credited.loc[keep, "isin"],
            "distribution_number": credited.loc[keep, "distribution_number"],
            "event_type": kinds[keep],
            "cash_amount": credited.loc[keep, "value_brl"],
            "payment_date": credited.loc[keep, "credit_date"],
            "approved_on": credited.loc[keep, "approved_on"],
            "origin": credited.loc[keep, "source_file"],
            "label": credited.loc[keep, "label"],
        }
    )
    return _keep_skeleton_isins(_frame(events, Source.B3_CREDITED))


# --- Bootstrap ---

BOOTSTRAP_COLUMNS: Final[tuple[str, ...]] = (
    "ticker",
    "last_cum_date",
    "event_type",
    "cash_amount",
    "payment_date",
    "retrieved_on",
)
"""Canonical schema of ``raw/bootstrap/statusinvest_cash.parquet`` (H3.10)."""


class BootstrapFormatError(ValueError):
    """The bootstrap file does not follow :data:`BOOTSTRAP_COLUMNS`."""


def bootstrap_events(
    bootstrap: pd.DataFrame, locator: IsinLocator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the bootstrap's cash events, placed on ISINs through their tickers.

    Raises
    ------
    BootstrapFormatError
        If a column is missing or an event type is not a cash type.
    """
    missing = set(BOOTSTRAP_COLUMNS) - set(bootstrap.columns)
    if missing:
        raise BootstrapFormatError(f"Bootstrap lacks columns {sorted(missing)}.")
    cash = {
        EventType.CASH_DIVIDEND,
        EventType.INTEREST_ON_CAPITAL,
        EventType.INCOME,
        EventType.RETURN_OF_CAPITAL,
    }
    unknown = set(bootstrap["event_type"]) - {kind.value for kind in cash}
    if unknown:
        raise BootstrapFormatError(f"Bootstrap event types {sorted(unknown)} are not cash.")
    isins: list[str | None] = []
    unresolved = []
    days = pd.DatetimeIndex(pd.to_datetime(bootstrap["last_cum_date"]))
    for ticker, day in zip(bootstrap["ticker"].astype(str), days, strict=True):
        isin, reason = locator.isin_at(locator.securities_with_ticker(ticker), day)
        isins.append(isin)
        if isin is None:
            unresolved.append(
                {
                    "source": Source.BOOTSTRAP.value,
                    "origin": "bootstrap",
                    "key": ticker,
                    "last_cum_date": day,
                    "reason": reason,
                }
            )
    events = pd.DataFrame(
        {
            "isin": isins,
            "last_cum_date": days,
            "event_type": bootstrap["event_type"].to_numpy(),
            "cash_amount": bootstrap["cash_amount"].to_numpy(),
            "payment_date": bootstrap["payment_date"].to_numpy(),
            "origin": ("bootstrap:" + bootstrap["ticker"].astype(str)).to_numpy(),
            "label": bootstrap["event_type"].to_numpy(),
        }
    )
    events = _keep_skeleton_isins(_frame(events[events["isin"].notna()], Source.BOOTSTRAP))
    return events, pd.DataFrame(unresolved, columns=list(UNRESOLVED_COLUMNS))


# --- Helpers ---


def _types(labels: pd.Series, mapping: dict[str, EventType | None], name: str) -> pd.Series:
    def lookup(label: str) -> str | None:
        kind = event_type(label, mapping, name)
        return None if kind is None else kind.value

    return labels.map(lookup)


def _frame(data: pd.DataFrame | list[dict[str, object]], source: Source) -> pd.DataFrame:
    frame = pd.DataFrame(data).reindex(columns=list(SOURCE_EVENT_COLUMNS))
    frame["source"] = source.value
    return _typed(frame)


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("last_cum_date", "payment_date", "approved_on"):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in ("cash_amount", "share_ratio", "subscription_price"):
        frame[column] = frame[column].astype("float64")
    frame["distribution_number"] = frame["distribution_number"].astype("Int64")
    for column in ("isin", "event_type", "target_isin", "source", "origin", "label"):
        frame[column] = frame[column].astype(object).fillna("").astype("str")
    return frame


def _keep_skeleton_isins(events: pd.DataFrame) -> pd.DataFrame:
    keep = events["isin"].str[6:8].isin(SKELETON_TYPES)
    return events[keep].reset_index(drop=True)


def _union_by_count(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge frames that repeat each other: each event as many times as its largest count."""
    columns = ["isin", "last_cum_date", "event_type", "amount_key"]
    tagged = []
    for rank, frame in enumerate(frames):
        keyed = frame.assign(amount_key=frame["cash_amount"].round(AMOUNT_DECIMALS))
        keyed["copy"] = keyed.groupby(columns, dropna=False).cumcount()
        keyed["rank"] = rank
        tagged.append(keyed)
    joined = pd.concat(tagged, ignore_index=True).sort_values("rank", kind="stable")
    kept = joined.drop_duplicates([*columns, "copy"], keep="first")
    return kept.drop(columns=["amount_key", "copy", "rank"]).reset_index(drop=True)
