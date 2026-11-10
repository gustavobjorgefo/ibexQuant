# ibexQuant\tests\support\reference.py

"""
Helpers for security master tests: valid ISINs, FCA files, override files.

Everything is synthetic; no CVM or B3 file is used (ADR 0010, H0.2).
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

FCA_HEADER = (
    "CNPJ_Companhia;Data_Referencia;Versao;ID_Documento;Nome_Empresarial;Valor_Mobiliario;"
    "Codigo_Negociacao;Mercado"
)


def isin(issuer: str, kind: str = "ACNOR") -> str:
    """Return ``BR`` + issuer (4) + kind (5) + a valid ISO 6166 check digit."""
    body = f"BR{issuer}{kind}"
    if len(body) != 11:
        raise ValueError(f"issuer must have 4 characters and kind 5, got {issuer!r}, {kind!r}")
    digits = "".join(str(int(char, 36)) for char in body)
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 0:
            value = value * 2 - 9 if value * 2 > 9 else value * 2
        total += value
    return body + str((10 - total % 10) % 10)


def write_fca(folder: Path, year: int, rows: Sequence[tuple[str, str]]) -> Path:
    """Write ``fca_cia_aberta_{year}.zip`` with (ticker, cnpj) rows for ``year``."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = [FCA_HEADER]
    lines += [
        f"{cnpj};{year}-01-01;1;1;COMPANY;Ações Ordinárias;{ticker};Bolsa" for ticker, cnpj in rows
    ]
    path = folder / f"fca_cia_aberta_{year}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"fca_cia_aberta_valor_mobiliario_{year}.csv", "\n".join(lines).encode("latin-1")
        )
        archive.writestr(f"fca_cia_aberta_geral_{year}.csv", b"a;b\n")
    return path


def write_links(
    folder: Path,
    links: Sequence[tuple[str, str]] = (),
    rejections: Sequence[tuple[str, str]] = (),
) -> Path:
    """Write ``ticker_links.yaml`` with the given (from, to) pairs."""
    folder.mkdir(parents=True, exist_ok=True)
    text = ["version: 1", "links:"]
    text += [f"  - {{from: {a}, to: {b}, note: reviewed}}" for a, b in links] or []
    text += ["rejections:"]
    text += [f"  - {{from: {a}, to: {b}, note: reviewed}}" for a, b in rejections] or []
    path = folder / "ticker_links.yaml"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path


# --- A small market with the cases of ADR 0012 ---

SESSIONS = ("20240102", "20240103", "20240104", "20240105", "20240108", "20240109")
VVAR, VIIA, BHIA = isin("VVAR"), isin("VIIA"), isin("BHIA")
TIMP, TIMS = isin("TIMP"), isin("TIMS")
PETR = "BRPETRACNPR6"


def rename_market() -> list[bytes]:
    """
    Detail records for six sessions of 2024.

    - VVAR3 (2 sessions) -> VIIA3 (2) -> BHIA3 (2): renames, same CNPJ in the FCA;
    - TIMP3 (3) -> TIMS3 (3): incorporation, different CNPJs;
    - PETR4 (6) and its odd lot PETR4F (2): one security under two tickers.
    """
    from support.cotahist import detail

    day = SESSIONS
    return [
        detail(session=day[0], ticker="VVAR3", isin=VVAR, close=10.0, volume_shares=100),
        detail(session=day[1], ticker="VVAR3", isin=VVAR, close=11.0, volume_shares=100),
        detail(session=day[2], ticker="VIIA3", isin=VIIA, open_=11.0, close=12.0),
        detail(session=day[3], ticker="VIIA3", isin=VIIA, close=13.0),
        detail(session=day[4], ticker="BHIA3", isin=BHIA, open_=13.0, close=14.0),
        detail(session=day[5], ticker="BHIA3", isin=BHIA, close=15.0),
        *[detail(session=d, ticker="TIMP3", isin=TIMP, close=20.0) for d in day[:3]],
        *[detail(session=d, ticker="TIMS3", isin=TIMS, open_=20.0, close=21.0) for d in day[3:]],
        *[detail(session=d, ticker="PETR4", isin=PETR) for d in day],
        *[detail(session=d, ticker="PETR4F", isin=PETR, market="020", bdi="96") for d in day[:2]],
    ]


FCA_ROWS = (
    ("VVAR3", "33.041.260/0652-90"),
    ("VIIA3", "33.041.260/0652-90"),
    ("BHIA3", "33.041.260/0652-90"),
    ("TIMP3", "02.558.115/0001-21"),
    ("TIMS3", "02.421.421/0001-11"),
    ("PETR4", "33.000.167/0001-01"),
)
