# ibexQuant\src\ibexQuant\data\ingestion\cvm\fre.py

"""
Reader of the stock events in the CVM reference forms (FRE, item 17.3).

Responsibility
--------------
Read, from the ``fre_cia_aberta_YYYY.zip`` files in ``raw/cvm/fre/``, the
splits, reverse splits and bonus issues each company declared, as one row per
event and share class with the share counts before and after (ADR 0013).

Contract
--------
- The tables are ``fre_cia_aberta_capital_social_desdobramento_YYYY.csv``
  (common and total preferred counts) and its ``_classe_acao`` companion
  (counts per preferred class). They exist only in forms under ICVM 480,
  delivered up to 2022; later ZIPs do not have them and give no rows.
- Classes are ``ON``, ``PN`` (all preferred shares together) and ``PNA``,
  ``PNB``…; a preferred class letter outside A–H is skipped.
- The same event repeats across forms and versions; one row is kept per
  CNPJ, approval date, type, class and counts.
- Rows with a zero or missing count are not events of that class and are
  dropped.

What does not belong here
-------------------------
- Downloading: CVM serves Brazilian addresses only; files are placed by hand.
- Placing events on ex-sessions — see ``ibexQuant.data.reference``.
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

RAW_PARTS: Final[tuple[str, ...]] = ("cvm", "fre")
FILE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^fre_cia_aberta_(\d{4})\.zip$", re.IGNORECASE
)
EVENTS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^fre_cia_aberta_capital_social_desdobramento_\d{4}\.csv$", re.IGNORECASE
)
CLASSES_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^fre_cia_aberta_capital_social_desdobramento_classe_acao_\d{4}\.csv$", re.IGNORECASE
)
FRE_COLUMNS: Final[tuple[str, ...]] = (
    "cnpj",
    "company",
    "approved_on",
    "kind",
    "share_class",
    "shares_before",
    "shares_after",
    "source_file",
)
_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "CNPJ_Companhia",
    "Nome_Companhia",
    "ID_Capital_Social_Desdobramento",
    "Data_Aprovacao",
    "Tipo_Evento",
    "Quantidade_Acoes_Ordinarias_Antes_Aprovacao",
    "Quantidade_Acoes_Ordinarias_Depois_Aprovacao",
    "Quantidade_Acoes_Preferenciais_Antes_Aprovacao",
    "Quantidade_Acoes_Preferenciais_Depois_Aprovacao",
)
_CLASS_COLUMNS: Final[tuple[str, ...]] = (
    "ID_Capital_Social_Desdobramento",
    "Tipo_Classe_Acao_Preferencial",
    "Quantidade_Acoes_Antes_Aprovacao",
    "Quantidade_Acoes_Depois_Aprovacao",
)
_CLASS_PATTERN: Final[re.Pattern[str]] = re.compile(r"^Preferencial Classe ([A-H])$")


class FreFormatError(ValueError):
    """A reference-form table lacks one of the columns the reader relies on."""


def fre_files(store: DataStore) -> list[Path]:
    """Return the FRE ZIPs in ``raw/cvm/fre/``, by year; empty when there are none."""
    folder = store.raw(*RAW_PARTS)
    if not folder.is_dir():
        return []
    found = [path for path in folder.iterdir() if FILE_NAME_PATTERN.match(path.name)]
    return sorted(found, key=lambda path: path.name.lower())


def read_fre_stock_events(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Read the stock events of the given FRE ZIPs.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`FRE_COLUMNS`, one row per event and class, sorted.

    Raises
    ------
    FreFormatError
        If a table lacks a required column.
    """
    frames = [frame for frame in (_read_one(path) for path in paths) if not frame.empty]
    if not frames:
        return _empty()
    table = pd.concat(frames, ignore_index=True)
    key = ["cnpj", "approved_on", "kind", "share_class", "shares_before", "shares_after"]
    table = table.drop_duplicates(key, keep="last")
    return table.sort_values(["cnpj", "approved_on", "share_class"], kind="stable").reset_index(
        drop=True
    )


def _read_one(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        events_name = next((name for name in names if EVENTS_PATTERN.match(name)), None)
        if events_name is None:
            return _empty()
        events = _csv(archive.read(events_name), _EVENT_COLUMNS, f"{path.name}:{events_name}")
        classes_name = next((name for name in names if CLASSES_PATTERN.match(name)), None)
        classes = (
            _csv(archive.read(classes_name), _CLASS_COLUMNS, f"{path.name}:{classes_name}")
            if classes_name
            else pd.DataFrame(columns=list(_CLASS_COLUMNS))
        )

    base = events[["ID_Capital_Social_Desdobramento", "CNPJ_Companhia", "Nome_Companhia"]]
    base = base.assign(
        approved_on=pd.to_datetime(events["Data_Aprovacao"], format="%Y-%m-%d", errors="coerce"),
        kind=events["Tipo_Evento"].str.strip(),
    )
    common = base.assign(
        share_class="ON",
        shares_before=events["Quantidade_Acoes_Ordinarias_Antes_Aprovacao"],
        shares_after=events["Quantidade_Acoes_Ordinarias_Depois_Aprovacao"],
    )
    preferred = base.assign(
        share_class="PN",
        shares_before=events["Quantidade_Acoes_Preferenciais_Antes_Aprovacao"],
        shares_after=events["Quantidade_Acoes_Preferenciais_Depois_Aprovacao"],
    )
    letters = classes["Tipo_Classe_Acao_Preferencial"].str.strip().str.extract(_CLASS_PATTERN)[0]
    named = classes.loc[letters.notna(), list(_CLASS_COLUMNS)].assign(
        share_class="PN" + letters[letters.notna()]
    )
    by_class = named.merge(base, on="ID_Capital_Social_Desdobramento", how="inner").assign(
        shares_before=lambda frame: frame["Quantidade_Acoes_Antes_Aprovacao"],
        shares_after=lambda frame: frame["Quantidade_Acoes_Depois_Aprovacao"],
    )
    rows = pd.concat([common, preferred, by_class], ignore_index=True)
    rows["shares_before"] = pd.to_numeric(rows["shares_before"], errors="coerce")
    rows["shares_after"] = pd.to_numeric(rows["shares_after"], errors="coerce")
    keep = (rows["shares_before"] > 0) & (rows["shares_after"] > 0) & rows["approved_on"].notna()
    rows = rows[keep]
    return pd.DataFrame(
        {
            "cnpj": rows["CNPJ_Companhia"].str.strip(),
            "company": rows["Nome_Companhia"].str.strip(),
            "approved_on": rows["approved_on"].astype("datetime64[ns]"),
            "kind": rows["kind"],
            "share_class": rows["share_class"],
            "shares_before": rows["shares_before"].astype("float64"),
            "shares_after": rows["shares_after"].astype("float64"),
            "source_file": path.name,
        }
    )


def _csv(raw: bytes, required: tuple[str, ...], where: str) -> pd.DataFrame:
    table = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", dtype=str)
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise FreFormatError(f"{where}: missing column(s) {missing}.")
    return table


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cnpj": pd.Series(dtype="str"),
            "company": pd.Series(dtype="str"),
            "approved_on": pd.Series(dtype="datetime64[ns]"),
            "kind": pd.Series(dtype="str"),
            "share_class": pd.Series(dtype="str"),
            "shares_before": pd.Series(dtype="float64"),
            "shares_after": pd.Series(dtype="float64"),
            "source_file": pd.Series(dtype="str"),
        }
    )
