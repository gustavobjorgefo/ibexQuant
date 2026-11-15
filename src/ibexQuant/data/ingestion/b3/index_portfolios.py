# ibexQuant\src\ibexQuant\data\ingestion\b3\index_portfolios.py

"""
Readers of B3's index portfolio files (ADR 0015, H5.1, H5.2).

Responsibility
--------------
Turn the raw files of ``raw/b3/index_portfolios/`` into member rows: the
workbooks B3 publishes (previews and definitive portfolios, one sheet per
index) and the CSV converted from B3's retroactive Ibovespa series.

Contract
--------
- A workbook says what it is in the title of its ``IBOV`` sheet (cell A1):
  the kind (``Virada`` is the definitive portfolio, ``3ª Prévia`` and
  ``2ª Prévia`` the previews) and the month and year the period starts. The
  file name is not used. A title that cannot be read is an error naming the
  file.
- Every sheet is one index; old sheet codes map to today's through
  :data:`SHEET_ALIASES`. A workbook holding a code and its alias must list
  the same members under both.
- Member rows are those whose first cell is a ticker; the base session comes
  from each sheet's footer (``PREGÃO BASE: 31/08/2017``, or ``Pregão Base:``
  followed by a date cell), and from the ``IBOV`` sheet when a sheet has none.
- The retroactive CSV has ``period``, ``ticker``, ``name``, ``share_class``
  and ``weight_pct``; its rows are ``IBOV`` members without quantity or base
  session.

What does not belong here
-------------------------
- Choosing between sources and resolving tickers — see
  ``ibexQuant.data.universe.constituents``.
- Converting B3's retroactive PDF: done once by a script of the private
  configuration.

Requires ``openpyxl`` (``b3`` extra) for the workbooks.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
import warnings
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import pandas as pd

RAW_PARTS: Final[tuple[str, ...]] = ("b3", "index_portfolios")
WORKBOOKS_DIR: Final[str] = "workbooks"
RETROACTIVE_DIR: Final[str] = "retroactive"
RETROACTIVE_FILE: Final[str] = "ibov_retroactive_2003_2014.csv"
TITLE_SHEET: Final[str] = "IBOV"

SHEET_ALIASES: Final[dict[str, str]] = {
    "IBRX": "IBXX",
    "IBxl": "IBXL",
    "IEE": "IEEX",
    "IGC": "IGCX",
    "ISE": "ISEE",
}
"""Sheet codes of older files (2017–2018) and the codes B3 uses today."""

MEMBER_COLUMNS: Final[tuple[str, ...]] = (
    "index",
    "ticker",
    "name",
    "share_class",
    "theoretical_quantity",
    "weight_pct",
    "base_session",
)
RETROACTIVE_COLUMNS: Final[tuple[str, ...]] = (
    "period",
    "ticker",
    "name",
    "share_class",
    "weight_pct",
)

_TICKER: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9]{4}\d{1,2}$")
_DATE_TEXT: Final[re.Pattern[str]] = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
_TITLE_KIND: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\bvirada\b"), "definitive"),
    (re.compile(r"\b3\s*a\s*previa\b"), "third_preview"),
    (re.compile(r"\b2\s*a\s*previa\b"), "second_preview"),
)
_TITLE_PERIOD: Final[re.Pattern[str]] = re.compile(
    r"\bpara\s+([a-z]+)(?:\s+a\s+[a-z]+)?\s+de\s+(\d{4})\b"
)
_MONTHS: Final[dict[str, int]] = {"janeiro": 1, "maio": 5, "setembro": 9}
_BASE_LABEL: Final[str] = "pregao base"


class PortfolioSource(StrEnum):
    """Where a composition comes from, best first (H5.1)."""

    DEFINITIVE = "definitive"
    THIRD_PREVIEW = "third_preview"
    SECOND_PREVIEW = "second_preview"
    RETROACTIVE = "retroactive"
    LIQUIDITY_RULE = "liquidity_rule"


SOURCE_PRIORITY: Final[tuple[PortfolioSource, ...]] = tuple(PortfolioSource)


class IndexPortfolioFormatError(ValueError):
    """A portfolio file breaks the structure the reader relies on."""


@dataclass(frozen=True, slots=True)
class Workbook:
    """
    The content of one B3 workbook.

    Attributes
    ----------
    path : Path
        The file read.
    source : PortfolioSource
        Definitive portfolio, 3rd or 2nd preview.
    period : str
        The period it is for, ``YYYY-MM``.
    members : pandas.DataFrame
        One row per index and ticker, columns :data:`MEMBER_COLUMNS`.
    """

    path: Path
    source: PortfolioSource
    period: str
    members: pd.DataFrame


# --- Workbooks ---


def read_workbook(path: Path) -> Workbook:
    """
    Read one B3 index portfolio workbook.

    Raises
    ------
    IndexPortfolioFormatError
        If the file has no ``IBOV`` sheet, its title cannot be read, a sheet
        has no base session and the ``IBOV`` sheet none either, or a code and
        its alias list different members.
    """
    import openpyxl
    from openpyxl.utils.exceptions import InvalidFileException

    # B3's files carry print headers openpyxl cannot parse; they hold nothing we read.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        try:
            book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, InvalidFileException) as error:
            raise IndexPortfolioFormatError(
                f"{path.name}: not a readable workbook ({error})."
            ) from error
        try:
            sheets = {
                name: list(book[name].iter_rows(values_only=True)) for name in book.sheetnames
            }
        finally:
            book.close()

    if TITLE_SHEET not in sheets:
        raise IndexPortfolioFormatError(f"{path.name}: no {TITLE_SHEET} sheet.")
    ibov = sheets[TITLE_SHEET]
    title = str(ibov[0][0]) if ibov and ibov[0] and ibov[0][0] is not None else ""
    source, period = parse_title(title, path.name)
    default_base = _base_session(ibov)

    frames: dict[str, pd.DataFrame] = {}
    for name, rows in sheets.items():
        members = list(_members(rows))
        if not members:
            continue
        base = _base_session(rows) or default_base
        if base is None:
            raise IndexPortfolioFormatError(f"{path.name}: sheet {name} has no base session.")
        index = SHEET_ALIASES.get(name, name)
        frame = pd.DataFrame(
            members,
            columns=["ticker", "name", "share_class", "theoretical_quantity", "weight_pct"],
        )
        frame.insert(0, "index", index)
        frame["base_session"] = base
        if index in frames:
            _check_same_members(frames[index], frame, f"{path.name}: sheets {index} and {name}")
            continue
        frames[index] = frame

    members_frame = (
        pd.concat(frames.values(), ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(MEMBER_COLUMNS))
    )
    return Workbook(path, source, period, _typed(members_frame))


def parse_title(title: str, where: str = "title") -> tuple[PortfolioSource, str]:
    """
    Return the kind and period of a workbook from the title of its ``IBOV`` sheet.

    Examples
    --------
    >>> parse_title("3ª Prévia do IBOVESPA para Setembro a Dezembro de 2017")
    (<PortfolioSource.THIRD_PREVIEW: 'third_preview'>, '2017-09')

    Raises
    ------
    IndexPortfolioFormatError
        If the kind or the period cannot be read.
    """
    text = _plain(title)
    kinds = [kind for pattern, kind in _TITLE_KIND if pattern.search(text)]
    if len(kinds) != 1:
        raise IndexPortfolioFormatError(f"{where}: cannot tell the kind from title {title!r}.")
    match = _TITLE_PERIOD.search(text)
    month = _MONTHS.get(match.group(1)) if match else None
    if match is None or month is None:
        raise IndexPortfolioFormatError(f"{where}: cannot tell the period from title {title!r}.")
    return PortfolioSource(kinds[0]), f"{int(match.group(2)):04d}-{month:02d}"


def workbook_paths(folder: Path) -> list[Path]:
    """Return the workbooks of ``folder`` by name, skipping Excel's lock files (``~$``)."""
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")
    )


def _members(rows: Sequence[Sequence[object]]) -> Iterator[tuple[object, ...]]:
    for row in rows:
        if not row or not isinstance(row[0], str):
            continue
        ticker = row[0].strip()
        if not _TICKER.match(ticker):
            continue
        cells = (list(row) + [None] * 5)[:5]
        yield (
            ticker,
            _text(cells[1]),
            _text(cells[2]),
            _number(cells[3]),
            _number(cells[4]),
        )


def _base_session(rows: Sequence[Sequence[object]]) -> pd.Timestamp | None:
    for row in rows:
        for position, cell in enumerate(row):
            if not isinstance(cell, str) or not _plain(cell).startswith(_BASE_LABEL):
                continue
            match = _DATE_TEXT.search(cell)
            if match:
                day, month, year = (int(part) for part in match.groups())
                return pd.Timestamp(year, month, day)
            for following in row[position + 1 :]:
                if isinstance(following, dt.datetime | dt.date):
                    return pd.Timestamp(following).normalize()
    return None


def _check_same_members(first: pd.DataFrame, second: pd.DataFrame, where: str) -> None:
    if set(first["ticker"]) != set(second["ticker"]):
        raise IndexPortfolioFormatError(f"{where} list different members.")


# --- Retroactive series ---


def read_retroactive(path: Path) -> pd.DataFrame:
    """
    Read the CSV of B3's retroactive Ibovespa portfolios.

    Returns
    -------
    pandas.DataFrame
        Columns ``period`` plus :data:`MEMBER_COLUMNS`, index ``IBOV``, no
        quantity and no base session.

    Raises
    ------
    IndexPortfolioFormatError
        If columns are missing, a period or ticker is malformed, or a ticker
        appears twice in one period.
    """
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise IndexPortfolioFormatError(f"{path.name}: not a readable CSV ({error}).") from error
    missing = sorted(set(RETROACTIVE_COLUMNS) - set(table.columns))
    if missing:
        raise IndexPortfolioFormatError(f"{path.name}: missing column(s) {missing}.")
    table = table[list(RETROACTIVE_COLUMNS)].apply(lambda column: column.str.strip())
    bad_period = table[~table["period"].str.fullmatch(r"\d{4}-(01|05|09)")]
    if not bad_period.empty:
        raise IndexPortfolioFormatError(
            f"{path.name}: malformed period(s) {sorted(bad_period['period'].unique())}."
        )
    bad_ticker = table[~table["ticker"].str.fullmatch(_TICKER.pattern)]
    if not bad_ticker.empty:
        raise IndexPortfolioFormatError(
            f"{path.name}: malformed ticker(s) {sorted(bad_ticker['ticker'].unique())}."
        )
    twice = table[table.duplicated(["period", "ticker"])]
    if not twice.empty:
        pairs = sorted(zip(twice["period"], twice["ticker"], strict=True))
        raise IndexPortfolioFormatError(f"{path.name}: repeated (period, ticker) {pairs}.")
    try:
        weights = pd.to_numeric(table["weight_pct"])
    except ValueError as error:
        raise IndexPortfolioFormatError(f"{path.name}: weight_pct is not numeric.") from error
    frame = pd.DataFrame(
        {
            "period": table["period"],
            "index": TITLE_SHEET,
            "ticker": table["ticker"],
            "name": table["name"],
            "share_class": table["share_class"],
            "theoretical_quantity": float("nan"),
            "weight_pct": weights,
            "base_session": pd.NaT,
        }
    )
    return _typed(frame)


# --- Helpers ---


def _plain(text: str) -> str:
    """Lower case, no accents, ordinal marks as letters, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(stripped.lower().split())


def _text(cell: object) -> str:
    return " ".join(str(cell).split()) if cell is not None else ""


def _number(cell: object) -> float:
    if isinstance(cell, int | float):
        return float(cell)
    if isinstance(cell, str) and cell.strip():
        try:
            return float(cell.strip().replace(".", "").replace(",", "."))
        except ValueError:
            return float("nan")
    return float("nan")


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(
        {
            "index": "string",
            "ticker": "string",
            "name": "string",
            "share_class": "string",
            "theoretical_quantity": "float64",
            "weight_pct": "float64",
            "base_session": "datetime64[ns]",
        }
    ).reset_index(drop=True)
