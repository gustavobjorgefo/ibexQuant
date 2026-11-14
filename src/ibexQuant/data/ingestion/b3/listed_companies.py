# ibexQuant\src\ibexQuant\data\ingestion\b3\listed_companies.py

"""
Reader of B3's listed-company snapshots (ADR 0013, H3.4 and H3.10).

Responsibility
--------------
Turn the JSON answers saved under ``raw/b3/listed_companies/YYYY-MM-DD/``
into four tidy tables: stock events, recent cash events, the cash history
and subscriptions, with numbers and dates parsed.

Contract
--------
- ``supplement/<CODE>.json`` is the answer of ``GetListedSupplementCompany``;
  ``cash_dividends/<CODE>/page-NNN.json`` the pages of
  ``GetListedCashDividends`` under the issuer's current trading name, and
  ``page-<NAME>-NNN.json`` the pages under an old trading name of the same
  root (renamed issuers keep their history under the old name). A
  zero-byte or empty answer means B3 had nothing for the code and is
  skipped; any other unreadable file is an error.
- Numbers use a decimal comma and a thousands point (``1.234,56``). Dates
  are day-first; B3's placeholders ``31/12/9999`` and ``01/01/1900`` and
  empty strings are missing dates.
- Cash-history values quoted per 1,000 shares (``quotedPerShares``) are
  divided to a value per share, and so is the closing price (H3.5).
- Labels and the B3 factor are returned as published; turning them into
  events belongs to the ledger.
- Snapshots are merged by keeping every row ever seen: within one answer,
  identical rows are distinct events (installments), so each distinct row
  is kept as many times as the answer that listed it most. An answer is one
  trading name in one snapshot: the same history returned under the current
  and an old name is one history, not two.

What does not belong here
-------------------------
- Downloading: the B3 data collector places the files.
- Resolving the share class of the cash history to an ISIN (H3.13).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from ibexQuant.data.store import DataStore

RAW_PARTS: Final[tuple[str, ...]] = ("b3", "listed_companies")
SNAPSHOT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MISSING_DATES: Final[frozenset[str]] = frozenset({"", "31/12/9999", "01/01/1900"})

STOCK_COLUMNS: Final[tuple[str, ...]] = (
    "snapshot",
    "code",
    "isin",
    "target_isin",
    "label",
    "factor",
    "last_cum_date",
    "approved_on",
)
CASH_RECENT_COLUMNS: Final[tuple[str, ...]] = (
    "snapshot",
    "code",
    "isin",
    "label",
    "cash_amount",
    "last_cum_date",
    "approved_on",
    "payment_date",
    "related_to",
)
CASH_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "snapshot",
    "code",
    "share_class",
    "label",
    "cash_amount",
    "last_cum_date",
    "approved_on",
    "closing_price",
)
SUBSCRIPTION_COLUMNS: Final[tuple[str, ...]] = (
    "snapshot",
    "code",
    "isin",
    "target_isin",
    "label",
    "percentage",
    "price",
    "last_cum_date",
    "approved_on",
    "subscription_date",
)
_ANSWER: Final[str] = "answer"
_DATE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"snapshot", "last_cum_date", "approved_on", "payment_date", "subscription_date"}
)
_NUMBER_COLUMNS: Final[frozenset[str]] = frozenset(
    {"factor", "cash_amount", "closing_price", "percentage", "price"}
)


class ListedCompaniesFormatError(ValueError):
    """A saved B3 answer is not the JSON shape the reader relies on."""


@dataclass(frozen=True, slots=True)
class ListedCompanies:
    """
    The events of every listed-company snapshot.

    Attributes
    ----------
    stock : pandas.DataFrame
        Columns :data:`STOCK_COLUMNS`. ``target_isin`` is the asset the
        holders received (B3's ``assetIssued``), empty when none.
    cash_recent : pandas.DataFrame
        Columns :data:`CASH_RECENT_COLUMNS`, from the supplement.
    cash_history : pandas.DataFrame
        Columns :data:`CASH_HISTORY_COLUMNS`, values per share.
    subscriptions : pandas.DataFrame
        Columns :data:`SUBSCRIPTION_COLUMNS`.
    """

    stock: pd.DataFrame
    cash_recent: pd.DataFrame
    cash_history: pd.DataFrame
    subscriptions: pd.DataFrame


def snapshot_folders(store: DataStore) -> list[Path]:
    """Return the snapshot folders of the raw zone, oldest first."""
    root = store.raw(*RAW_PARTS)
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.iterdir() if path.is_dir() and SNAPSHOT_PATTERN.match(path.name)
    )


def read_listed_companies(folders: list[Path]) -> ListedCompanies:
    """
    Read and merge the given snapshot folders.

    Parameters
    ----------
    folders : list[Path]
        Folders named ``YYYY-MM-DD`` (see :func:`snapshot_folders`).

    Returns
    -------
    ListedCompanies

    Raises
    ------
    ListedCompaniesFormatError
        If a non-empty answer is not valid JSON of the expected shape.
    """
    rows: dict[str, list[dict[str, Any]]] = {
        "stock": [],
        "cash_recent": [],
        "cash_history": [],
        "subscriptions": [],
    }
    for folder in folders:
        for kind, found in _read_snapshot(folder).items():
            rows[kind].extend(found)
    return ListedCompanies(
        stock=_merge(rows["stock"], STOCK_COLUMNS),
        cash_recent=_merge(rows["cash_recent"], CASH_RECENT_COLUMNS),
        cash_history=_merge(rows["cash_history"], CASH_HISTORY_COLUMNS),
        subscriptions=_merge(rows["subscriptions"], SUBSCRIPTION_COLUMNS),
    )


# --- One snapshot ---


def _read_snapshot(folder: Path) -> dict[str, list[dict[str, Any]]]:
    snapshot = folder.name
    found: dict[str, list[dict[str, Any]]] = {
        "stock": [],
        "cash_recent": [],
        "cash_history": [],
        "subscriptions": [],
    }
    for path in sorted((folder / "supplement").glob("*.json")):
        company = _load(path)
        if not company:
            continue
        if not isinstance(company, list) or not isinstance(company[0], dict):
            raise ListedCompaniesFormatError(f"{path}: expected a list with one company.")
        data = company[0]
        code = path.stem
        base = {"snapshot": snapshot, "code": code}
        for item in data.get("stockDividends") or []:
            found["stock"].append(
                base
                | {
                    "isin": item["isinCode"],
                    "target_isin": item.get("assetIssued") or "",
                    "label": item["label"],
                    "factor": item["factor"],
                    "last_cum_date": item["lastDatePrior"],
                    "approved_on": item.get("approvedOn", ""),
                }
            )
        for item in data.get("cashDividends") or []:
            found["cash_recent"].append(
                base
                | {
                    "isin": item["isinCode"],
                    "label": item["label"],
                    "cash_amount": item["rate"],
                    "last_cum_date": item["lastDatePrior"],
                    "approved_on": item.get("approvedOn", ""),
                    "payment_date": item.get("paymentDate", ""),
                    "related_to": item.get("relatedTo", ""),
                }
            )
        for item in data.get("subscriptions") or []:
            found["subscriptions"].append(
                base
                | {
                    "isin": item["isinCode"],
                    "target_isin": item.get("assetIssued") or "",
                    "label": item["label"],
                    "percentage": item["percentage"],
                    "price": item["priceUnit"],
                    "last_cum_date": item["lastDatePrior"],
                    "approved_on": item.get("approvedOn", ""),
                    "subscription_date": item.get("subscriptionDate", ""),
                }
            )
    for path in sorted((folder / "cash_dividends").glob("*/page-*.json")):
        page = _load(path)
        if not page:
            continue
        if not isinstance(page, dict) or not isinstance(page.get("results"), list):
            raise ListedCompaniesFormatError(f"{path}: expected an object with 'results'.")
        for item in page["results"]:
            quoted = int(item.get("quotedPerShares") or 1)
            found["cash_history"].append(
                {
                    "snapshot": snapshot,
                    "code": path.parent.name,
                    "share_class": item["typeStock"],
                    "label": item["corporateAction"],
                    "cash_amount": _number(item["valueCash"]) / quoted,
                    "last_cum_date": item["lastDatePriorEx"],
                    "approved_on": item.get("dateApproval") or "",
                    "closing_price": _number(item.get("closingPricePriorExDate") or "0") / quoted,
                    _ANSWER: _trading_name_of(path),
                }
            )
    return found


def _trading_name_of(page: Path) -> str:
    # page-001.json -> "" (current name); page-JBS-001.json -> "JBS".
    return "-".join(page.stem.split("-")[1:-1])


def _load(path: Path) -> Any:
    content = path.read_bytes()
    if not content.strip():
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ListedCompaniesFormatError(f"{path}: not valid JSON ({error}).") from error


# --- Tables ---


def _merge(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=[*columns, _ANSWER])
    table[_ANSWER] = table[_ANSWER].astype(object).fillna("").astype(str)
    for column in columns:
        if column in _DATE_COLUMNS:
            table[column] = _dates(table[column])
        elif column in _NUMBER_COLUMNS:
            table[column] = table[column].map(_number).astype("float64")
        else:
            table[column] = table[column].astype("str").str.strip()
    if table.empty:
        return table[list(columns)]
    content = [column for column in columns if column != "snapshot"]
    # Identical rows within an answer are installments; the same row seen in
    # several answers is one event. Keep each row its largest count.
    counted = table.assign(
        copy=table.groupby(["snapshot", _ANSWER, *content], dropna=False).cumcount()
    )
    kept = counted.sort_values("snapshot").drop_duplicates([*content, "copy"], keep="last")
    return (
        kept.drop(columns="copy")
        .sort_values(["code", "last_cum_date"], kind="stable")[list(columns)]
        .reset_index(drop=True)
    )


def _number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(".", "").replace(",", ".")
    return float(text)


def _dates(values: pd.Series) -> pd.Series:
    text = values.astype("str").str.strip()
    text = text.where(~text.isin(MISSING_DATES))
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    parsed = pd.to_datetime(text.where(~iso), format="%d/%m/%Y")
    parsed[iso] = pd.to_datetime(text[iso], format="%Y-%m-%d")
    return parsed.astype("datetime64[ns]")
