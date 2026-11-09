# ibexQuant\tests\support\cotahist.py

"""
Synthetic COTAHIST records and files for tests.

Builds 245-byte records at the positions of the B3 layout (revision 02), so
tests exercise the parser without ever shipping B3 data (ADR 0010, H0.2).
Prices are given as published, i.e. per quotation lot: a price of 46.70 with
``factor=1000`` means R$ 46.70 per thousand shares.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

RECORD_LENGTH: Final[int] = 245


def _place(record: bytearray, start: int, end: int, value: bytes) -> None:
    """Write ``value`` at the layout's 1-based inclusive positions."""
    width = end - start + 1
    if len(value) != width:
        raise ValueError(f"Field at {start}-{end} needs {width} bytes, got {len(value)}.")
    record[start - 1 : end] = value


def _number(value: int, width: int) -> bytes:
    return str(value).zfill(width).encode("ascii")


def _money(value: float, width: int = 13) -> bytes:
    return _number(round(value * 100), width)


def _text(value: str, width: int) -> bytes:
    return value.encode("latin-1").ljust(width)


def detail(
    *,
    session: str = "20240102",
    ticker: str = "PETR4",
    market: str = "010",
    bdi: str = "02",
    company: str = "PETROBRAS",
    specification: str = "PN      N2",
    currency: str = "R$",
    open_: float = 38.0,
    high: float = 39.0,
    low: float = 37.5,
    close: float = 38.5,
    average: float = 38.2,
    best_bid: float = 38.4,
    best_ask: float = 38.6,
    trades: int = 1_000,
    volume_shares: int = 2_000_000,
    traded_value: float = 76_400_000.0,
    factor: int = 1,
    isin: str = "BRPETRACNPR6",
    distribution: int = 100,
) -> bytes:
    """Return one detail record (type ``01``)."""
    record = bytearray(b" " * RECORD_LENGTH)
    _place(record, 1, 2, b"01")
    _place(record, 3, 10, session.encode("ascii"))
    _place(record, 11, 12, bdi.encode("ascii"))
    _place(record, 13, 24, _text(ticker, 12))
    _place(record, 25, 27, market.encode("ascii"))
    _place(record, 28, 39, _text(company, 12))
    _place(record, 40, 49, _text(specification, 10))
    _place(record, 53, 56, _text(currency, 4))
    for (start, end), price in zip(
        [(57, 69), (70, 82), (83, 95), (96, 108), (109, 121), (122, 134), (135, 147)],
        [open_, high, low, average, close, best_bid, best_ask],
        strict=True,
    ):
        _place(record, start, end, _money(price))
    _place(record, 148, 152, _number(trades, 5))
    _place(record, 153, 170, _number(volume_shares, 18))
    _place(record, 171, 188, _money(traded_value, 18))
    _place(record, 189, 201, _number(0, 13))
    _place(record, 202, 202, b"0")
    _place(record, 203, 210, b"99991231")
    _place(record, 211, 217, _number(factor, 7))
    _place(record, 218, 230, _number(0, 13))
    _place(record, 231, 242, _text(isin, 12))
    _place(record, 243, 245, _number(distribution, 3))
    return bytes(record)


def header(year: int, generated: str = "20250102") -> bytes:
    """Return the header record (type ``00``)."""
    return f"00COTAHIST.{year}BOVESPA {generated}".encode("ascii").ljust(RECORD_LENGTH)


def trailer(year: int, count: int, generated: str = "20250102") -> bytes:
    """Return the trailer record (type ``99``) declaring ``count`` records."""
    text = f"99COTAHIST.{year}BOVESPA {generated}{count:011d}"
    return text.encode("ascii").ljust(RECORD_LENGTH)


def lines_for(year: int, details: Sequence[bytes], *, convention: str = "all") -> list[bytes]:
    """Header, details and a trailer counting ``all`` records or ``detail`` records."""
    count = len(details) + 2 if convention == "all" else len(details)
    return [header(year), *details, trailer(year, count)]


def write_zip(
    folder: Path,
    year: int,
    lines: Sequence[bytes],
    *,
    name: str | None = None,
    member: str | None = None,
    extra_members: Sequence[str] = (),
) -> Path:
    """Write ``lines`` (CRLF-terminated) into ``COTAHIST_A{year}.ZIP`` under ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name or f"COTAHIST_A{year}.ZIP")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member or f"COTAHIST_A{year}.TXT", b"".join(f + b"\r\n" for f in lines))
        for extra in extra_members:
            archive.writestr(extra, b"")
    return path
