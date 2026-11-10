# ibexQuant\src\ibexQuant\data\ingestion\cvm\fca.py

"""
Reader of the securities table of CVM registration forms (FCA).

Responsibility
--------------
Extract ``(reference year, ticker, CNPJ)`` from the
``fca_cia_aberta_valor_mobiliario`` table inside each annual
``fca_cia_aberta_YYYY.zip`` (ADR 0012, H2.6 and H2.8).

Contract
--------
- Only rows with a trading code are returned; the FCA fills it from
  reference year 2018 only.
- Tickers are stripped and upper-cased; repeated rows are dropped.
- The FCA dates (``Data_Inicio_Negociacao`` and similar) are not read: they
  describe the share class, not the ticker.

What does not belong here
-------------------------
- Linking tickers to securities — see ``ibexQuant.data.reference``.
- Downloading: the CVM portal accepts only Brazilian addresses, so files are
  placed in ``raw/cvm/fca/`` by hand.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pandas as pd

from ibexQuant.data.store import DataStore

RAW_PARTS: Final[tuple[str, ...]] = ("cvm", "fca")
FILE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^fca_cia_aberta_(\d{4})\.zip$", re.IGNORECASE
)
MEMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^fca_cia_aberta_valor_mobiliario_\d{4}\.csv$", re.IGNORECASE
)
_REQUIRED: Final[dict[str, str]] = {
    "Data_Referencia": "reference_year",
    "Codigo_Negociacao": "ticker",
    "CNPJ_Companhia": "cnpj",
}
COLUMNS: Final[tuple[str, ...]] = ("reference_year", "ticker", "cnpj")


class FcaFormatError(ValueError):
    """An FCA file lacks the securities table or one of its required columns."""


def fca_files(store: DataStore) -> list[Path]:
    """
    Return the FCA ZIPs in ``raw/cvm/fca/``, by year; empty when there are none.

    Parameters
    ----------
    store : DataStore
        The data store.
    """
    folder = store.raw(*RAW_PARTS)
    if not folder.is_dir():
        return []
    found = [path for path in folder.iterdir() if FILE_NAME_PATTERN.match(path.name)]
    return sorted(found, key=lambda path: path.name.lower())


def read_fca_securities(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Read the ticker-to-CNPJ rows of the given FCA ZIPs.

    Parameters
    ----------
    paths : Iterable[Path]
        ``fca_cia_aberta_YYYY.zip`` files.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`COLUMNS`: ``reference_year`` (int16), ``ticker`` and
        ``cnpj`` (string), sorted and without repeated rows.

    Raises
    ------
    FcaFormatError
        If a ZIP has no securities table, or the table lacks a required column.
    """
    frames = [_read_one(path) for path in paths]
    if not frames:
        return pd.DataFrame({name: pd.Series(dtype="string") for name in COLUMNS}).astype(
            {"reference_year": "int16"}
        )
    table = pd.concat(frames, ignore_index=True).drop_duplicates()
    return table.sort_values(list(COLUMNS), kind="stable").reset_index(drop=True)


def _read_one(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if MEMBER_PATTERN.match(name)]
        if len(members) != 1:
            raise FcaFormatError(
                f"{path.name} must hold one fca_cia_aberta_valor_mobiliario table, "
                f"found {len(members)}."
            )
        raw = archive.read(members[0])
    table = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", dtype=str)
    missing = sorted(set(_REQUIRED) - set(table.columns))
    if missing:
        raise FcaFormatError(f"{path.name}: missing column(s) {missing}.")

    table = table[list(_REQUIRED)].rename(columns=_REQUIRED)
    ticker = table["ticker"].fillna("").str.strip().str.upper()
    cnpj = table["cnpj"].fillna("").str.strip()
    keep = (ticker != "") & (cnpj != "")
    return pd.DataFrame(
        {
            "reference_year": table.loc[keep, "reference_year"].str[:4].astype("int16"),
            "ticker": ticker[keep].astype("string"),
            "cnpj": cnpj[keep].astype("string"),
        }
    )
