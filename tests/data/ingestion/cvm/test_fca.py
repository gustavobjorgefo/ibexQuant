# ibexQuant\tests\data\ingestion\cvm\test_fca.py

"""Tests for :func:`ibexQuant.data.ingestion.cvm.read_fca_securities` (ADR 0012, H2.6)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from support.reference import FCA_HEADER, write_fca

from ibexQuant.data.ingestion.cvm import FcaFormatError, fca_files, read_fca_securities
from ibexQuant.data.store import DataStore


def test_reads_ticker_and_cnpj_by_year(tmp_path: Path) -> None:
    paths = [
        write_fca(tmp_path, 2018, [("VVAR3", "1")]),
        write_fca(tmp_path, 2021, [("VIIA3", "1"), ("PETR4", "2")]),
    ]

    table = read_fca_securities(paths)

    assert table.to_dict("records") == [
        {"reference_year": 2018, "ticker": "VVAR3", "cnpj": "1"},
        {"reference_year": 2021, "ticker": "PETR4", "cnpj": "2"},
        {"reference_year": 2021, "ticker": "VIIA3", "cnpj": "1"},
    ]
    assert table["reference_year"].dtype == "int16"


def test_rows_without_trading_code_are_dropped(tmp_path: Path) -> None:
    path = write_fca(tmp_path, 2015, [("", "1"), ("  ", "2")])
    assert read_fca_securities([path]).empty


def test_tickers_are_normalized_and_repeats_dropped(tmp_path: Path) -> None:
    path = write_fca(tmp_path, 2020, [(" petr4 ", "2"), ("PETR4", "2")])
    assert read_fca_securities([path])["ticker"].tolist() == ["PETR4"]


def test_no_files_gives_an_empty_typed_table() -> None:
    table = read_fca_securities([])
    assert list(table.columns) == ["reference_year", "ticker", "cnpj"]
    assert table.empty


def test_zip_without_the_securities_table(tmp_path: Path) -> None:
    path = tmp_path / "fca_cia_aberta_2020.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fca_cia_aberta_geral_2020.csv", b"a;b\n")
    with pytest.raises(FcaFormatError, match="valor_mobiliario"):
        read_fca_securities([path])


def test_table_without_a_required_column(tmp_path: Path) -> None:
    path = tmp_path / "fca_cia_aberta_2020.zip"
    header = FCA_HEADER.replace("Codigo_Negociacao", "Codigo")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fca_cia_aberta_valor_mobiliario_2020.csv", header.encode("latin-1"))
    with pytest.raises(FcaFormatError, match="Codigo_Negociacao"):
        read_fca_securities([path])


def test_fca_files_lists_annual_zips_only(tmp_path: Path) -> None:
    store = DataStore.initialize(tmp_path / "ibex-data")
    assert fca_files(store) == []
    folder = store.raw("cvm", "fca")
    write_fca(folder, 2019, [])
    write_fca(folder, 2018, [])
    (folder / "notes.txt").write_text("x")
    assert [path.name for path in fca_files(store)] == [
        "fca_cia_aberta_2018.zip",
        "fca_cia_aberta_2019.zip",
    ]
