# ibexQuant\tests\data\ingestion\cvm\test_fre.py

"""Tests for the reader of FRE stock events (ADR 0013)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ibexQuant.data.ingestion.cvm.fre import (
    FRE_COLUMNS,
    FreFormatError,
    fre_files,
    read_fre_stock_events,
)
from ibexQuant.data.store import DataStore

EVENTS_HEADER = (
    "CNPJ_Companhia;Data_Referencia;Versao;ID_Documento;Nome_Companhia;"
    "ID_Capital_Social_Desdobramento;Data_Aprovacao;Tipo_Evento;"
    "Quantidade_Acoes_Ordinarias_Antes_Aprovacao;Quantidade_Acoes_Preferenciais_Antes_Aprovacao;"
    "Quantidade_Total_Acoes_Antes_Aprovacao;Quantidade_Acoes_Ordinarias_Depois_Aprovacao;"
    "Quantidade_Acoes_Preferenciais_Depois_Aprovacao;Quantidade_Total_Acoes_Depois_Aprovacao"
)
CLASSES_HEADER = (
    "CNPJ_Companhia;Data_Referencia;Versao;ID_Documento;Nome_Companhia;"
    "ID_Capital_Social_Desdobramento;Tipo_Classe_Acao_Preferencial;"
    "Quantidade_Acoes_Antes_Aprovacao;Quantidade_Acoes_Depois_Aprovacao"
)
ITSA = "61.532.644/0001-15"
SAUIPE = "00.866.577/0001-80"


def event(
    identifier: int, cnpj: str, day: str, kind: str, counts: tuple[int, int, int, int]
) -> str:
    on_before, pn_before, on_after, pn_after = counts
    before = f"{on_before};{pn_before};{on_before + pn_before}"
    after = f"{on_after};{pn_after};{on_after + pn_after}"
    return f"{cnpj};2016-01-01;1;1;COMPANY;{identifier};{day};{kind};{before};{after}"


def write_fre(folder: Path, year: int, events: list[str], classes: list[str] | None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"fre_cia_aberta_{year}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"fre_cia_aberta_capital_social_desdobramento_{year}.csv",
            "\n".join([EVENTS_HEADER, *events]).encode("latin-1"),
        )
        if classes is not None:
            archive.writestr(
                f"fre_cia_aberta_capital_social_desdobramento_classe_acao_{year}.csv",
                "\n".join([CLASSES_HEADER, *classes]).encode("latin-1"),
            )
        archive.writestr(f"fre_cia_aberta_capital_social_{year}.csv", b"a;b\n")
    return path


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return DataStore.initialize(tmp_path / "data")


def test_events_give_one_row_per_class(store: DataStore) -> None:
    folder = store.raw("cvm", "fre")
    write_fre(
        folder,
        2016,
        [
            event(1, ITSA, "2016-04-29", "Bonificação", (100, 200, 110, 220)),
            event(2, SAUIPE, "2015-06-01", "Grupamento", (0, 1000, 0, 100)),
        ],
        [
            f"{SAUIPE};2016-01-01;1;1;SAUIPE;2;Preferencial Classe A;600;60",
            f"{SAUIPE};2016-01-01;1;1;SAUIPE;2;Preferencial Classe U;400;40",
        ],
    )
    table = read_fre_stock_events(fre_files(store))
    assert list(table.columns) == list(FRE_COLUMNS)
    rows = table[["cnpj", "kind", "share_class", "shares_before", "shares_after"]].values.tolist()
    assert rows == [
        [SAUIPE, "Grupamento", "PN", 1000.0, 100.0],
        [SAUIPE, "Grupamento", "PNA", 600.0, 60.0],
        [ITSA, "Bonificação", "ON", 100.0, 110.0],
        [ITSA, "Bonificação", "PN", 200.0, 220.0],
    ]
    assert table.iloc[0]["approved_on"] == pd.Timestamp("2015-06-01")
    assert set(table["source_file"]) == {"fre_cia_aberta_2016.zip"}


def test_the_same_event_in_several_forms_is_kept_once(store: DataStore) -> None:
    folder = store.raw("cvm", "fre")
    line = event(1, ITSA, "2016-04-29", "Bonificação", (100, 200, 110, 220))
    write_fre(folder, 2016, [line], None)
    write_fre(folder, 2017, [line.replace(";1;1;COMPANY", ";2;5;COMPANY")], None)
    table = read_fre_stock_events(fre_files(store))
    assert len(table) == 2
    assert set(table["source_file"]) == {"fre_cia_aberta_2017.zip"}


def test_forms_without_the_table_give_no_rows(store: DataStore) -> None:
    folder = store.raw("cvm", "fre")
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder / "fre_cia_aberta_2024.zip", "w") as archive:
        archive.writestr("fre_cia_aberta_capital_social_2024.csv", b"a;b\n")
    table = read_fre_stock_events(fre_files(store))
    assert table.empty
    assert list(table.columns) == list(FRE_COLUMNS)


def test_a_missing_column_is_an_error(store: DataStore) -> None:
    folder = store.raw("cvm", "fre")
    path = write_fre(folder, 2016, [], None)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "fre_cia_aberta_capital_social_desdobramento_2016.csv", b"CNPJ_Companhia;Data\n"
        )
    with pytest.raises(FreFormatError, match="Tipo_Evento"):
        read_fre_stock_events([path])


def test_no_folder_lists_no_files(store: DataStore) -> None:
    assert fre_files(store) == []
