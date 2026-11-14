# ibexQuant\tests\data\ingestion\b3\test_earnings_credited.py

"""Tests for the reader of B3's daily *Earnings credited* files (ADR 0013, H3.4, H3.10)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ibexQuant.data.ingestion.b3.earnings_credited import (
    CREDITED_COLUMNS,
    EarningsCreditedFormatError,
    credited_files,
    read_earnings_credited,
)
from ibexQuant.data.store import DataStore

ENGLISH = (
    "﻿Dividend credit is the amount received by investors as a return on investments.\n"
    "This table contains information about these earnings.\n"
    "\n"
    "Issuer;ISIN code;Type of earnings;Approval Date;Value (BRL);Credit date\n"
    "BCO SANTANDER (BRASIL) S.A.;BRSANBACNPR5 181;JUROS SOBRE CAPITAL PRÓPRIO;09/10/2026;"
    "0.21004966171;11/09/2026\n"
    "ENJOEI S.A.;BRENJUACNOR9 102;LEILÃO DE FRAÇÕES;-;7.28805931658;11/09/2026\n"
)
PORTUGUESE = (
    "﻿Crédito de proventos é o valor recebido pelos investidores.\n"
    "Esta tabela contém informações sobre esses proventos.\n"
    "\n"
    "Emissor;Código ISIN;Tipo de provento;Data de aprovação;Valor (R$);Data de crédito\n"
    "BICICLETAS MONARK S.A.;BRBMKSACNOR9 169;DIVIDENDO;19/12/2025;10,0581484552;29/06/2026\n"
    "UNITEDHEALTH GROUP INC;BRUNHHBDR000 130;DIVIDENDO;-;1.234,5;29/06/2026\n"
)
EMPTY_DAY = (
    "Emissor;Código ISIN;Tipo de provento;Data de aprovação;Valor (R$);Data de crédito\n"
    "Nenhum resultado\n"
)


def write(store: DataStore, day: str, text: str) -> Path:
    path = store.raw("b3", "earnings_credited", day[:4], f"{day}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return DataStore.initialize(tmp_path / "data")


def test_english_layout_reads_decimal_points_and_month_first_credit_dates(
    store: DataStore,
) -> None:
    table = read_earnings_credited([write(store, "2026-11-09", ENGLISH)])
    assert list(table.columns) == list(CREDITED_COLUMNS)
    row = table.iloc[0]
    assert row["credit_date"] == pd.Timestamp("2026-11-09")
    assert row["isin"] == "BRSANBACNPR5"
    assert row["distribution_number"] == 181
    assert row["label"] == "JUROS SOBRE CAPITAL PRÓPRIO"
    assert row["value_brl"] == pytest.approx(0.21004966171)
    assert row["approved_on"] == pd.Timestamp("2026-10-09")
    assert pd.isna(table.iloc[1]["approved_on"])


def test_portuguese_layout_reads_decimal_commas_and_day_first_dates(store: DataStore) -> None:
    table = read_earnings_credited([write(store, "2026-06-29", PORTUGUESE)])
    assert table["value_brl"].tolist() == pytest.approx([10.0581484552, 1234.5])
    assert table["credit_date"].tolist() == [pd.Timestamp("2026-06-29")] * 2
    assert table.iloc[0]["approved_on"] == pd.Timestamp("2025-12-19")


def test_an_empty_day_gives_no_rows(store: DataStore) -> None:
    assert read_earnings_credited([write(store, "2026-07-04", EMPTY_DAY)]).empty


def test_no_files_give_an_empty_table_with_the_columns() -> None:
    table = read_earnings_credited([])
    assert table.empty
    assert list(table.columns) == list(CREDITED_COLUMNS)


def test_credit_dates_that_disagree_with_the_file_name_are_an_error(store: DataStore) -> None:
    path = write(store, "2026-11-10", ENGLISH)
    with pytest.raises(EarningsCreditedFormatError, match="2026-11-10"):
        read_earnings_credited([path])


def test_a_file_without_the_header_is_an_error(store: DataStore) -> None:
    path = write(store, "2026-11-09", "nothing here\n")
    with pytest.raises(EarningsCreditedFormatError, match="header"):
        read_earnings_credited([path])


def test_a_name_that_is_not_a_date_is_an_error(store: DataStore) -> None:
    path = store.raw("b3", "earnings_credited", "2026", "latest.csv")
    path.parent.mkdir(parents=True)
    path.write_text(ENGLISH, encoding="utf-8")
    with pytest.raises(EarningsCreditedFormatError, match="latest.csv"):
        read_earnings_credited([path])


def test_an_isin_without_distribution_number_is_an_error(store: DataStore) -> None:
    text = ENGLISH.replace("BRSANBACNPR5 181", "BRSANBACNPR5")
    with pytest.raises(EarningsCreditedFormatError, match="distribution number"):
        read_earnings_credited([write(store, "2026-11-09", text)])


def test_files_are_listed_by_date_across_year_folders(store: DataStore) -> None:
    later = write(store, "2027-01-04", EMPTY_DAY)
    earlier = write(store, "2026-12-30", EMPTY_DAY)
    store.raw("b3", "earnings_credited", "2026", "notes.txt").write_text("x")
    assert credited_files(store) == [earlier, later]


def test_no_folder_lists_no_files(store: DataStore) -> None:
    assert credited_files(store) == []
