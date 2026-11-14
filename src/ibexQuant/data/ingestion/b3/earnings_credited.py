# ibexQuant\src\ibexQuant\data\ingestion\b3\earnings_credited.py

"""
Reader of B3's daily *Earnings credited* report (*Crédito de proventos*).

Responsibility
--------------
Turn the files in ``raw/b3/earnings_credited/YYYY/YYYY-MM-DD.csv`` into one
row per credit: ISIN, distribution number, B3 label, value per share and
dates (ADR 0013, H3.4 and H3.10).

Contract
--------
- The file name is the credit date. The report has been published in two
  layouts: an English header with decimal points and month-first credit
  dates, and a Portuguese header with decimal commas and day-first dates.
  The header decides the layout; the credit column must agree with the file
  name, which settles the date order without guessing.
- Approval dates are day-first in both layouts; ``-`` means none.
- The ISIN column carries the ISIN and the distribution number the credit
  belongs to (``BRPETRACNOR9 213``); both are required.
- Labels are returned as published (upper case, accents kept). Mapping them
  to event types belongs to the ledger.

What does not belong here
-------------------------
- Downloading: the B3 data collector places the files.
- Matching credits to ex-sessions — see ``ibexQuant.data.reference``.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pandas as pd

from ibexQuant.data.store import DataStore

RAW_PARTS: Final[tuple[str, ...]] = ("b3", "earnings_credited")
FILE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")
CREDITED_COLUMNS: Final[tuple[str, ...]] = (
    "credit_date",
    "isin",
    "distribution_number",
    "label",
    "value_brl",
    "approved_on",
    "issuer",
    "source_file",
)
_HEADERS: Final[dict[str, tuple[str, str]]] = {
    # first header cell -> (decimal separator, credit date order)
    "Issuer": (".", "month_first"),
    "Emissor": (",", "day_first"),
}
_FIELD_COUNT: Final[int] = 6


class EarningsCreditedFormatError(ValueError):
    """An *Earnings credited* file breaks the layout the reader relies on."""


def credited_files(store: DataStore) -> list[Path]:
    """
    Return the *Earnings credited* files of the raw zone, by date.

    Parameters
    ----------
    store : DataStore
        The data store.
    """
    folder = store.raw(*RAW_PARTS)
    if not folder.is_dir():
        return []
    found = [path for path in folder.glob("*/*.csv") if FILE_NAME_PATTERN.match(path.name)]
    return sorted(found, key=lambda path: path.name)


def read_earnings_credited(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Read the credits of the given files.

    Parameters
    ----------
    paths : Iterable[Path]
        ``YYYY-MM-DD.csv`` files.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`CREDITED_COLUMNS`, in file order.

    Raises
    ------
    EarningsCreditedFormatError
        If a file name is not a date, the header is missing, an ISIN lacks
        its distribution number, or the credit dates disagree with the name.
    """
    frames = [_read_one(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True)


def _read_one(path: Path) -> pd.DataFrame:
    match = FILE_NAME_PATTERN.match(path.name)
    if match is None:
        raise EarningsCreditedFormatError(f"{path.name} is not named YYYY-MM-DD.csv.")
    day = pd.Timestamp(match.group(1))

    lines = path.read_bytes().decode("utf-8-sig").splitlines()
    header = next(
        (
            index
            for index, line in enumerate(lines)
            if line.split(";", 1)[0].strip() in _HEADERS and line.count(";") == _FIELD_COUNT - 1
        ),
        None,
    )
    if header is None:
        raise EarningsCreditedFormatError(f"{path.name}: no header line ({sorted(_HEADERS)}).")
    decimal, date_order = _HEADERS[lines[header].split(";", 1)[0].strip()]

    rows = [line for line in lines[header + 1 :] if line.count(";") == _FIELD_COUNT - 1]
    if not rows:
        return _empty()
    table = pd.read_csv(io.StringIO("\n".join(rows)), sep=";", header=None, dtype=str)
    issuer, code, label, approved, value, credited = (table[i].str.strip() for i in range(6))

    parts = code.str.split(r"\s+", n=1, expand=True)
    if parts.shape[1] < 2 or parts[1].isna().any():
        raise EarningsCreditedFormatError(
            f"{path.name}: an ISIN without distribution number ({code.iloc[0]!r})."
        )
    credit = pd.to_datetime(
        credited, format="%m/%d/%Y" if date_order == "month_first" else "%d/%m/%Y"
    )
    if (credit != day).any():
        raise EarningsCreditedFormatError(
            f"{path.name}: credit dates {sorted(set(credited))} disagree with {day.date()}."
        )
    if decimal == ",":
        value = value.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.DataFrame(
        {
            "credit_date": credit,
            "isin": parts[0],
            "distribution_number": parts[1].astype("int64"),
            "label": label.str.upper(),
            "value_brl": value.astype("float64"),
            "approved_on": pd.to_datetime(approved.where(approved != "-"), format="%d/%m/%Y"),
            "issuer": issuer,
            "source_file": path.name,
        }
    )[list(CREDITED_COLUMNS)]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "credit_date": pd.Series(dtype="datetime64[ns]"),
            "isin": pd.Series(dtype="str"),
            "distribution_number": pd.Series(dtype="int64"),
            "label": pd.Series(dtype="str"),
            "value_brl": pd.Series(dtype="float64"),
            "approved_on": pd.Series(dtype="datetime64[ns]"),
            "issuer": pd.Series(dtype="str"),
            "source_file": pd.Series(dtype="str"),
        }
    )
