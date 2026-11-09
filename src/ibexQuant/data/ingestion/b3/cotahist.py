# ibexQuant\src\ibexQuant\data\ingestion\b3\cotahist.py

"""
Parser of B3 annual COTAHIST files.

Responsibility
--------------
Turn one ``COTAHIST_AYYYY.ZIP`` into the typed table of ADR 0011 (H1.9),
validating the file's structure on the way, and report the counts that go
into the manifest. :func:`read_cotahist` is pure: it reads one file and
writes nothing.

Contract
--------
- Structure is validated before values are read (H1.3): one header first,
  one trailer last, matching years, a trailer count that follows one of the
  two conventions B3 has used.
- Records are Latin-1; a record that is not 245 bytes gets one UTF-8 repair
  attempt, and is otherwise an error (H1.4).
- Only cash markets ``010`` and ``020`` are kept; every BDI code is kept
  (H1.5). Sessions before 1994-07-01 are dropped; afterwards the currency
  must be ``R$`` (H1.6).
- Prices are per share in BRL; ``volume_shares`` is the share count as
  published (H1.7). The key ``(session_date, ticker, market_type, isin)``
  never repeats (H1.8).

What does not belong here
-------------------------
- Writing partitions and the manifest — see ``bars_daily``.
- Value checks such as OHLC consistency — see
  ``ibexQuant.data.ingestion.validation.check_bars`` (H1.10).
- Adjustment, identifier linking, tradability — phases H2 to H5.
"""

from __future__ import annotations

import datetime as dt
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from ibexQuant.data.ingestion.b3.errors import CotahistFormatError

RECORD_LENGTH: Final[int] = 245
EQUITY_MARKETS: Final[frozenset[bytes]] = frozenset({b"010", b"020"})
FIRST_SESSION: Final[dt.date] = dt.date(1994, 7, 1)
CURRENCY: Final[bytes] = b"R$  "
FILE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^COTAHIST_A(\d{4})\.ZIP$", re.IGNORECASE)

# Layout revision 02 (2020-10-05): (start, end) are the document's 1-based,
# inclusive positions, converted below to Python slices.
_FIELDS: Final[dict[str, tuple[int, int]]] = {
    "session_date": (3, 10),
    "bdi_code": (11, 12),
    "ticker": (13, 24),
    "market_type": (25, 27),
    "company_name": (28, 39),
    "specification": (40, 49),
    "currency": (53, 56),
    "open": (57, 69),
    "high": (70, 82),
    "low": (83, 95),
    "average": (96, 108),
    "close": (109, 121),
    "best_bid": (122, 134),
    "best_ask": (135, 147),
    "trades": (148, 152),
    "volume_shares": (153, 170),
    "traded_value_brl": (171, 188),
    "quotation_factor": (211, 217),
    "isin": (231, 242),
    "distribution_number": (243, 245),
}
_PRICE_FIELDS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "average",
    "best_bid",
    "best_ask",
)
_TEXT_FIELDS: Final[tuple[str, ...]] = ("ticker", "isin", "specification", "company_name")

COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "ticker",
    "isin",
    "market_type",
    "bdi_code",
    "specification",
    "company_name",
    "distribution_number",
    "open",
    "high",
    "low",
    "close",
    "average",
    "best_bid",
    "best_ask",
    "trades",
    "volume_shares",
    "traded_value_brl",
    "quotation_factor",
)
"""Columns of the curated table, in order (ADR 0011, H1.9)."""

KEY: Final[tuple[str, ...]] = ("session_date", "ticker", "market_type", "isin")
"""Columns that identify a row (H1.8)."""


class TrailerConvention(StrEnum):
    """What the trailer's record count includes (H1.3)."""

    ALL_RECORDS = "all_records"
    """Header and trailer included, as the layout says (files up to 2024)."""

    DETAIL_RECORDS = "detail_records"
    """Detail records only, as observed from 2025."""


@dataclass(frozen=True, slots=True)
class CotahistReport:
    """
    Counts produced while reading one COTAHIST file.

    Attributes
    ----------
    year : int
        Year of the file, from its name and header.
    member : str
        Name of the text file inside the ZIP.
    generated_on : datetime.date
        File generation date from the header.
    trailer_convention : TrailerConvention
        Which count convention the trailer follows.
    records_read : int
        Every record in the file, header and trailer included.
    rows_kept : int
        Rows of the returned table.
    rows_repaired : int
        Records fixed by the UTF-8 repair (H1.4), kept or not.
    sessions : int
        Distinct sessions in the returned table.
    first_session, last_session : datetime.date or None
        Range of the returned table; ``None`` when it is empty.
    """

    year: int
    member: str
    generated_on: dt.date
    trailer_convention: TrailerConvention
    records_read: int
    rows_kept: int
    rows_repaired: int
    sessions: int
    first_session: dt.date | None
    last_session: dt.date | None


def year_of(path: Path) -> int:
    """
    Return the year encoded in a COTAHIST file name.

    Parameters
    ----------
    path : Path
        A path whose name is ``COTAHIST_AYYYY.ZIP``, in any letter case.

    Returns
    -------
    int
        The year.

    Raises
    ------
    CotahistFormatError
        If the name does not follow the pattern.
    """
    match = FILE_NAME_PATTERN.match(path.name)
    if match is None:
        raise CotahistFormatError(
            f"{path.name!r} is not a COTAHIST annual file name (COTAHIST_AYYYY.ZIP)."
        )
    return int(match.group(1))


def read_cotahist(path: Path) -> tuple[pd.DataFrame, CotahistReport]:
    """
    Read one annual COTAHIST ZIP into the curated table.

    Parameters
    ----------
    path : Path
        A ``COTAHIST_AYYYY.ZIP`` holding exactly one text file.

    Returns
    -------
    tuple[pandas.DataFrame, CotahistReport]
        The table with :data:`COLUMNS`, sorted by ``(session_date, ticker)``
        with a fresh ``RangeIndex``, and the counts of the read.

    Raises
    ------
    CotahistFormatError
        On any structural problem listed in ADR 0011 (H1.3–H1.8).
    """
    year = year_of(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise CotahistFormatError(
                f"{path.name} must hold exactly one file, found {len(members)}: {members}."
            )
        with archive.open(members[0]) as handle:
            scan = _scan(handle, year=year, source=path.name)

    frame = _to_frame(scan.records, source=path.name)
    sessions = frame["session_date"]
    report = CotahistReport(
        year=year,
        member=members[0],
        generated_on=scan.generated_on,
        trailer_convention=scan.trailer_convention,
        records_read=scan.records_read,
        rows_kept=len(frame),
        rows_repaired=scan.rows_repaired,
        sessions=int(sessions.nunique()),
        first_session=sessions.min().date() if len(frame) else None,
        last_session=sessions.max().date() if len(frame) else None,
    )
    return frame, report


# --- Scan: structure, encoding, market filter ---


@dataclass(frozen=True, slots=True)
class _Scan:
    records: list[bytes]
    generated_on: dt.date
    trailer_convention: TrailerConvention
    records_read: int
    rows_repaired: int


def _scan(lines: Iterable[bytes], *, year: int, source: str) -> _Scan:
    """Walk the file once: validate structure, repair encoding, keep cash markets."""
    kept: list[bytes] = []
    header: bytes | None = None
    trailer: bytes | None = None
    repaired = 0
    number = 0

    for raw in lines:
        number += 1
        record = raw.rstrip(b"\r\n")
        if len(record) != RECORD_LENGTH:
            record = _repair(record, number=number, source=source)
            repaired += 1
        if trailer is not None:
            raise CotahistFormatError(f"{source}: line {number} comes after the trailer.")
        kind = record[:2]
        if number == 1:
            if kind != b"00":
                raise CotahistFormatError(f"{source}: line 1 is not a header (type 00).")
            header = record
            continue
        if kind == b"01":
            if record[24:27] in EQUITY_MARKETS:
                kept.append(record)
        elif kind == b"99":
            trailer = record
        elif kind == b"00":
            raise CotahistFormatError(f"{source}: line {number} is a second header.")
        else:
            raise CotahistFormatError(
                f"{source}: line {number} has unknown record type {kind.decode('latin-1')!r}."
            )

    if header is None:
        raise CotahistFormatError(f"{source}: the file is empty.")
    if trailer is None:
        raise CotahistFormatError(f"{source}: the last line is not a trailer (type 99).")

    for name, record in (("header", header), ("trailer", trailer)):
        if record[2:15] != f"COTAHIST.{year}".encode():
            raise CotahistFormatError(
                f"{source}: the {name} names {record[2:15].decode('latin-1')!r}, "
                f"expected 'COTAHIST.{year}'."
            )

    return _Scan(
        records=kept,
        generated_on=_header_date(header, source=source),
        trailer_convention=_trailer_convention(trailer, number, source=source),
        records_read=number,
        rows_repaired=repaired,
    )


def _repair(record: bytes, *, number: int, source: str) -> bytes:
    """Re-encode a record written in UTF-8 as Latin-1 (H1.4), or fail."""
    try:
        fixed = record.decode("utf-8").encode("latin-1")
    except (UnicodeDecodeError, UnicodeEncodeError):
        fixed = record
    if len(fixed) != RECORD_LENGTH:
        raise CotahistFormatError(
            f"{source}: line {number} is {len(record)} bytes long, expected {RECORD_LENGTH}."
        )
    return fixed


def _header_date(header: bytes, *, source: str) -> dt.date:
    try:
        return dt.datetime.strptime(header[23:31].decode("latin-1"), "%Y%m%d").date()
    except ValueError as error:
        raise CotahistFormatError(f"{source}: invalid generation date in the header.") from error


def _trailer_convention(trailer: bytes, records: int, *, source: str) -> TrailerConvention:
    declared = trailer[31:42]
    if not declared.isdigit():
        raise CotahistFormatError(f"{source}: the trailer count is not a number.")
    count = int(declared)
    if count == records:
        return TrailerConvention.ALL_RECORDS
    if count == records - 2:
        return TrailerConvention.DETAIL_RECORDS
    raise CotahistFormatError(
        f"{source}: the trailer declares {count} records, but the file has {records} "
        f"({records - 2} detail records)."
    )


# --- Frame: fixed-width fields to typed columns ---


def _to_frame(records: list[bytes], *, source: str) -> pd.DataFrame:
    """Decode the kept records into the typed table, applying H1.6–H1.8."""
    matrix = np.frombuffer(b"".join(records), dtype=np.uint8).reshape(-1, RECORD_LENGTH)

    session_number = _digits(matrix, "session_date", source=source)
    after_cut = session_number >= int(FIRST_SESSION.strftime("%Y%m%d"))
    matrix = matrix[after_cut]
    session_number = session_number[after_cut]

    currency = _bytes(matrix, "currency")
    wrong = np.flatnonzero(currency != CURRENCY)
    if wrong.size:
        sample = currency[wrong[0]].decode("latin-1").strip()
        raise CotahistFormatError(
            f"{source}: {wrong.size} record(s) on or after {FIRST_SESSION} are not in R$ "
            f"(first: {sample!r})."
        )

    factor = _digits(matrix, "quotation_factor", source=source)
    if (factor <= 0).any():
        raise CotahistFormatError(f"{source}: quotation factor of zero.")

    columns: dict[str, object] = {
        "session_date": pd.to_datetime(session_number.astype(str), format="%Y%m%d"),
        "market_type": _digits(matrix, "market_type", source=source).astype(np.int16),
        "bdi_code": _digits(matrix, "bdi_code", source=source).astype(np.int16),
        "distribution_number": _digits(matrix, "distribution_number", source=source).astype(
            np.int16
        ),
        "trades": _digits(matrix, "trades", source=source),
        "volume_shares": _digits(matrix, "volume_shares", source=source),
        "traded_value_brl": _digits(matrix, "traded_value_brl", source=source) / 100.0,
        "quotation_factor": factor.astype(np.int32),
    }
    for name in _PRICE_FIELDS:
        columns[name] = _digits(matrix, name, source=source) / 100.0 / factor
    for name in _TEXT_FIELDS:
        columns[name] = pd.array(
            [value.decode("latin-1").strip() for value in _bytes(matrix, name)], dtype="string"
        )

    frame = pd.DataFrame(columns)[list(COLUMNS)]
    repeated = frame.duplicated(subset=list(KEY), keep=False)
    if repeated.any():
        first = frame.loc[repeated, list(KEY)].iloc[0].to_dict()
        raise CotahistFormatError(f"{source}: repeated key {first}.")

    return frame.sort_values(["session_date", "ticker"], kind="stable").reset_index(drop=True)


def _bytes(matrix: npt.NDArray[np.uint8], name: str) -> npt.NDArray[np.bytes_]:
    """Return one field as an array of fixed-width byte strings."""
    start, end = _FIELDS[name]
    width = end - start + 1
    block = np.ascontiguousarray(matrix[:, start - 1 : end])
    return block.view(f"S{width}").ravel()


def _digits(matrix: npt.NDArray[np.uint8], name: str, *, source: str) -> npt.NDArray[np.int64]:
    """Return one numeric field as int64, rejecting anything but ASCII digits."""
    start, end = _FIELDS[name]
    block = matrix[:, start - 1 : end].astype(np.int64) - ord("0")
    if ((block < 0) | (block > 9)).any():
        raise CotahistFormatError(f"{source}: non-numeric value in field {name!r}.")
    powers = 10 ** np.arange(end - start, -1, -1, dtype=np.int64)
    result: npt.NDArray[np.int64] = block @ powers
    return result
