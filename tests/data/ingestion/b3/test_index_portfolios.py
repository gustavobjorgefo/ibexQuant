# ibexQuant\tests\data\ingestion\b3\test_index_portfolios.py

"""Readers of B3's index portfolio workbooks and retroactive CSV (ADR 0015, H5.2)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from support.index_portfolios import write_workbook

from ibexQuant.data.ingestion.b3.index_portfolios import (
    MEMBER_COLUMNS,
    SOURCE_PRIORITY,
    IndexPortfolioFormatError,
    PortfolioSource,
    parse_title,
    read_retroactive,
    read_workbook,
    workbook_paths,
)

PETR = ("PETR4", "PETROBRAS", "PN  N2", 4_000_000_000, 60.0)
VALE = ("VALE3", "VALE", "ON      NM", 3_000_000_000, 40.0)
ITUB = ("ITUB4", "ITAUUNIBANCO", "PN  ED  N1", 1_000_000, 100.0)


# --- titles ---


@pytest.mark.parametrize(
    ("title", "source", "period"),
    [
        ("3ª Prévia do IBOVESPA para setembro de 2025", PortfolioSource.THIRD_PREVIEW, "2025-09"),
        ("3 ª Prévia do IBOVESPA para maio de 2025", PortfolioSource.THIRD_PREVIEW, "2025-05"),
        (
            "3ª Previa do IBOVESPA para Janeiro a Abril de 2018",
            PortfolioSource.THIRD_PREVIEW,
            "2018-01",
        ),
        (
            "3ª Prévia do IBOVESPA para Setembro a Dezembro de 2017",
            PortfolioSource.THIRD_PREVIEW,
            "2017-09",
        ),
        (
            "3 PREVIA do IBOVESPA para Janeiro a Abril de 2021",
            PortfolioSource.THIRD_PREVIEW,
            "2021-01",
        ),
        ("3º Prévia do IBOVESPA para maio de 2021", PortfolioSource.THIRD_PREVIEW, "2021-05"),
        (
            "Terceira Prévia do IBOVESPA para maio de 2021",
            PortfolioSource.THIRD_PREVIEW,
            "2021-05",
        ),
        (
            "Segunda Prévia do IBOVESPA para maio de 2021",
            PortfolioSource.SECOND_PREVIEW,
            "2021-05",
        ),
        ("Virada do IBOVESPA para janeiro de 2025", PortfolioSource.DEFINITIVE, "2025-01"),
        ("2ª Prévia do IBOVESPA para maio de 2026", PortfolioSource.SECOND_PREVIEW, "2026-05"),
    ],
)
def test_titles_give_kind_and_period(title: str, source: PortfolioSource, period: str) -> None:
    assert parse_title(title) == (source, period)


@pytest.mark.parametrize(
    ("title", "message"),
    [
        ("Carteira do IBOVESPA para maio de 2026", "kind"),
        ("1ª Prévia do IBOVESPA para maio de 2026", "kind"),
        ("13 Prévia do IBOVESPA para maio de 2026", "kind"),
        ("3ª Prévia do IBOVESPA para junho de 2026", "period"),
        ("3ª Prévia do IBOVESPA", "period"),
    ],
)
def test_an_unreadable_title_is_an_error(title: str, message: str) -> None:
    with pytest.raises(IndexPortfolioFormatError, match=f"file.xlsx: cannot tell the {message}"):
        parse_title(title, "file.xlsx")


def test_sources_rank_from_the_definitive_portfolio_down() -> None:
    assert SOURCE_PRIORITY[0] is PortfolioSource.DEFINITIVE
    assert SOURCE_PRIORITY[-1] is PortfolioSource.LIQUIDITY_RULE


# --- workbooks ---


def test_a_new_layout_workbook_is_read(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "any name.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        {"IBOV": [PETR, VALE], "IBXX": [PETR]},
    )
    book = read_workbook(path)
    assert (book.source, book.period, book.path) == (
        PortfolioSource.THIRD_PREVIEW,
        "2024-09",
        path,
    )
    members = book.members
    assert tuple(members.columns) == MEMBER_COLUMNS
    assert members[["index", "ticker"]].values.tolist() == [
        ["IBOV", "PETR4"],
        ["IBOV", "VALE3"],
        ["IBXX", "PETR4"],
    ]
    first = members.iloc[0]
    assert first["share_class"] == "PN N2"
    assert first["theoretical_quantity"] == 4e9
    assert first["weight_pct"] == 60.0
    assert (members["base_session"] == pd.Timestamp("2024-08-28")).all()


def test_an_old_layout_workbook_maps_sheet_codes(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "2017.xlsx",
        "3ª Prévia do IBOVESPA para Setembro a Dezembro de 2017",
        {"IBOV": [PETR], "IBRX": [PETR, VALE], "IBxl": [VALE]},
        base=dt.date(2017, 8, 31),
        old_layout=True,
    )
    members = read_workbook(path).members
    assert sorted(members["index"].unique()) == ["IBOV", "IBXL", "IBXX"]
    assert (members["base_session"] == pd.Timestamp("2017-08-31")).all()


def test_a_sheet_without_footer_takes_the_ibov_base_session(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "f.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        {"IBOV": [PETR], "SMLL": [VALE]},
        no_base_in=frozenset({"SMLL"}),
    )
    members = read_workbook(path).members
    assert (members["base_session"] == pd.Timestamp("2024-08-28")).all()


def test_without_any_base_session_the_read_stops(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "f.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        {"IBOV": [PETR]},
        base=None,
    )
    with pytest.raises(IndexPortfolioFormatError, match="sheet IBOV has no base session"):
        read_workbook(path)


def test_a_code_and_its_alias_with_the_same_members_are_one_index(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "f.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        {"IBOV": [PETR], "IBXX": [PETR, VALE], "IBRX": [VALE, PETR]},
    )
    members = read_workbook(path).members
    assert members[members["index"] == "IBXX"]["ticker"].tolist() == ["PETR4", "VALE3"]


def test_a_code_and_its_alias_with_different_members_stop_the_read(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "f.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        {"IBOV": [PETR], "IBXX": [PETR], "IBRX": [VALE]},
    )
    with pytest.raises(IndexPortfolioFormatError, match="sheets IBXX and IBRX list different"):
        read_workbook(path)


def test_sheets_without_members_are_skipped(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "f.xlsx",
        "Virada do IBOVESPA para janeiro de 2025",
        {"IBOV": [PETR], "IFIX": []},
    )
    assert read_workbook(path).members["index"].unique().tolist() == ["IBOV"]


def test_a_workbook_with_no_member_at_all_is_empty(tmp_path: Path) -> None:
    path = write_workbook(
        tmp_path / "f.xlsx", "Virada do IBOVESPA para janeiro de 2025", {"IBOV": []}
    )
    members = read_workbook(path).members
    assert members.empty
    assert tuple(members.columns) == MEMBER_COLUMNS


def test_a_workbook_without_ibov_sheet_is_refused(tmp_path: Path) -> None:
    path = write_workbook(tmp_path / "f.xlsx", "x", {"SMLL": [PETR]})
    with pytest.raises(IndexPortfolioFormatError, match="f.xlsx: no IBOV sheet"):
        read_workbook(path)


def test_an_empty_title_is_refused_with_the_file_name(tmp_path: Path) -> None:
    path = write_workbook(tmp_path / "f.xlsx", "", {"IBOV": [PETR]})
    with pytest.raises(IndexPortfolioFormatError, match="f.xlsx: cannot tell the kind"):
        read_workbook(path)


def test_a_file_that_is_not_a_workbook_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.xlsx"
    path.write_text("not a zip", encoding="utf-8")
    with pytest.raises(IndexPortfolioFormatError, match="broken.xlsx: not a readable workbook"):
        read_workbook(path)


def test_text_numbers_and_blank_cells_are_read(tmp_path: Path) -> None:
    rows = [("PETR4", None, None, "1.234,5", ""), ("VALE3 ", "VALE", "ON", "n/a", 2)]
    path = write_workbook(
        tmp_path / "f.xlsx",
        "Virada do IBOVESPA para janeiro de 2025",
        {"IBOV": rows},
    )
    members = read_workbook(path).members
    assert members["ticker"].tolist() == ["PETR4", "VALE3"]
    assert members["name"].tolist() == ["", "VALE"]
    assert members["theoretical_quantity"].iloc[0] == 1234.5
    assert members["theoretical_quantity"].isna().iloc[1]
    assert members["weight_pct"].isna().iloc[0]


def test_workbook_paths_skip_lock_files_and_other_files(tmp_path: Path) -> None:
    for name in ("b.xlsx", "a.XLSX", "~$a.xlsx", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "folder.xlsx").mkdir()
    assert [path.name for path in workbook_paths(tmp_path)] == ["a.XLSX", "b.xlsx"]
    assert workbook_paths(tmp_path / "absent") == []


# --- retroactive CSV ---

CSV_HEADER = "period,ticker,name,share_class,weight_pct\n"


def write_csv(tmp_path: Path, body: str, header: str = CSV_HEADER) -> Path:
    path = tmp_path / "ibov_retroactive_2003_2014.csv"
    path.write_text(header + body, encoding="utf-8")
    return path


def test_the_retroactive_csv_is_read(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "2003-01,PETR4,PETROBRAS,PN,10.5\n2013-09,KROT3,KROTON,,0.8\n")
    table = read_retroactive(path)
    assert tuple(table.columns) == ("period", *MEMBER_COLUMNS)
    assert table["index"].unique().tolist() == ["IBOV"]
    assert table["share_class"].tolist() == ["PN", ""]
    assert table["weight_pct"].tolist() == [10.5, 0.8]
    assert table["theoretical_quantity"].isna().all()
    assert table["base_session"].isna().all()


@pytest.mark.parametrize(
    ("body", "header", "message"),
    [
        ("2003-01,PETR4,P,PN,1\n", "period,ticker,name,weight_pct\n", "missing column"),
        ("2003-02,PETR4,P,PN,1\n", CSV_HEADER, "malformed period"),
        ("2003-01,PETR,P,PN,1\n", CSV_HEADER, "malformed ticker"),
        ("2003-01,PETR4,P,PN,1\n2003-01,PETR4,P,PN,2\n", CSV_HEADER, "repeated"),
        ("2003-01,PETR4,P,PN,x\n", CSV_HEADER, "not numeric"),
    ],
)
def test_a_malformed_csv_is_refused(tmp_path: Path, body: str, header: str, message: str) -> None:
    with pytest.raises(IndexPortfolioFormatError, match=message):
        read_retroactive(write_csv(tmp_path, body, header))


def test_an_unreadable_csv_is_refused(tmp_path: Path) -> None:
    with pytest.raises(IndexPortfolioFormatError, match="not a readable CSV"):
        read_retroactive(tmp_path / "absent.csv")
