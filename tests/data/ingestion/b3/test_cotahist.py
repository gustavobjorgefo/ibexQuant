# ibexQuant\tests\data\ingestion\b3\test_cotahist.py

"""
Tests for :func:`ibexQuant.data.ingestion.b3.read_cotahist` (ADR 0011).

Every case is a synthetic file built from records at the layout's positions
(see ``tests/support/cotahist.py``); each one reproduces a finding of the H1
exploration or a rule of the contract.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from support.cotahist import detail, header, lines_for, trailer, write_zip

from ibexQuant.data.ingestion.b3 import (
    CotahistError,
    CotahistFormatError,
    TrailerConvention,
    read_cotahist,
)
from ibexQuant.data.ingestion.b3.cotahist import COLUMNS, year_of

# --- Happy path ---


def test_reads_one_record_into_the_curated_schema(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, lines_for(2024, [detail()]))

    frame, _ = read_cotahist(path)

    assert list(frame.columns) == list(COLUMNS)
    row = frame.iloc[0]
    assert row["session_date"] == pd.Timestamp("2024-01-02")
    assert row["ticker"] == "PETR4"
    assert row["isin"] == "BRPETRACNPR6"
    assert row["market_type"] == 10
    assert row["bdi_code"] == 2
    assert row["specification"] == "PN      N2"
    assert row["company_name"] == "PETROBRAS"
    assert row["distribution_number"] == 100
    assert (row["open"], row["high"], row["low"], row["close"]) == (38.0, 39.0, 37.5, 38.5)
    assert (row["average"], row["best_bid"], row["best_ask"]) == (38.2, 38.4, 38.6)
    assert row["trades"] == 1_000
    assert row["volume_shares"] == 2_000_000
    assert row["traded_value_brl"] == 76_400_000.0
    assert row["quotation_factor"] == 1


def test_column_dtypes(tmp_path: Path) -> None:
    frame, _ = read_cotahist(write_zip(tmp_path, 2024, lines_for(2024, [detail()])))

    assert pd.api.types.is_datetime64_any_dtype(frame["session_date"])
    for column in ("market_type", "bdi_code", "distribution_number"):
        assert frame[column].dtype == "int16"
    for column in ("trades", "volume_shares"):
        assert frame[column].dtype == "int64"
    for column in ("open", "high", "low", "close", "average", "best_bid", "best_ask"):
        assert frame[column].dtype == "float64"
    assert frame["traded_value_brl"].dtype == "float64"
    assert frame["quotation_factor"].dtype == "int32"
    for column in ("ticker", "isin", "specification", "company_name"):
        assert pd.api.types.is_string_dtype(frame[column])


def test_report(tmp_path: Path) -> None:
    details = [
        detail(session="20240103", ticker="VALE3"),
        detail(session="20240102", ticker="PETR4"),
        detail(session="20240102", ticker="PETR4C", market="070"),
    ]
    path = write_zip(tmp_path, 2024, lines_for(2024, details))

    _, report = read_cotahist(path)

    assert report.year == 2024
    assert report.member == "COTAHIST_A2024.TXT"
    assert report.generated_on == dt.date(2025, 1, 2)
    assert report.trailer_convention is TrailerConvention.ALL_RECORDS
    assert report.records_read == 5
    assert report.rows_kept == 2
    assert report.rows_repaired == 0
    assert report.sessions == 2
    assert report.first_session == dt.date(2024, 1, 2)
    assert report.last_session == dt.date(2024, 1, 3)


def test_rows_sorted_by_session_then_ticker(tmp_path: Path) -> None:
    details = [
        detail(session="20240103", ticker="ABEV3", isin="BRABEVACNOR1"),
        detail(session="20240102", ticker="VALE3", isin="BRVALEACNOR0"),
        detail(session="20240102", ticker="PETR4"),
    ]
    frame, _ = read_cotahist(write_zip(tmp_path, 2024, lines_for(2024, details)))

    assert frame["ticker"].tolist() == ["PETR4", "VALE3", "ABEV3"]
    assert frame.index.tolist() == [0, 1, 2]


def test_only_options_gives_an_empty_table(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, lines_for(2024, [detail(market="070")]))

    frame, report = read_cotahist(path)

    assert frame.empty
    assert list(frame.columns) == list(COLUMNS)
    assert (report.sessions, report.first_session, report.last_session) == (0, None, None)


# --- Units (H1.7) ---


def test_quotation_factor_is_applied_row_by_row(tmp_path: Path) -> None:
    details = [
        detail(ticker="PETR4", factor=1, close=46.70),
        detail(ticker="TMCP4", factor=1000, close=46.70, isin="BRTMCPACNPR1"),
    ]
    frame, _ = read_cotahist(write_zip(tmp_path, 2003, lines_for(2003, details)))

    closes = frame.set_index("ticker")["close"]
    assert closes["PETR4"] == pytest.approx(46.70)
    assert closes["TMCP4"] == pytest.approx(0.0467)
    assert frame.set_index("ticker").loc["TMCP4", "quotation_factor"] == 1000


@pytest.mark.parametrize("factor", [10, 100, 10_000, 100_000, 1_000_000])
def test_every_observed_quotation_factor(tmp_path: Path, factor: int) -> None:
    path = write_zip(tmp_path, 2026, lines_for(2026, [detail(factor=factor, open_=50.0)]))
    frame, _ = read_cotahist(path)
    assert frame.loc[0, "open"] == pytest.approx(50.0 / factor)


def test_volume_is_a_share_count_not_divided_by_the_factor(tmp_path: Path) -> None:
    record = detail(factor=1000, volume_shares=5_000, traded_value=233.5)
    frame, _ = read_cotahist(write_zip(tmp_path, 2000, lines_for(2000, [record])))

    assert frame.loc[0, "volume_shares"] == 5_000
    assert frame.loc[0, "traded_value_brl"] == pytest.approx(233.5)


def test_largest_published_values_fit(tmp_path: Path) -> None:
    record = detail(volume_shares=999_999_999_999_999_999, traded_value=9_999_999_999_999.99)
    frame, _ = read_cotahist(write_zip(tmp_path, 2024, lines_for(2024, [record])))

    assert frame.loc[0, "volume_shares"] == 999_999_999_999_999_999
    assert frame.loc[0, "traded_value_brl"] == pytest.approx(9_999_999_999_999.99)


# --- Market filter, no BDI filter (H1.5) ---


@pytest.mark.parametrize("market", ["012", "013", "017", "021", "030", "050", "060", "070", "080"])
def test_non_cash_markets_are_dropped(tmp_path: Path, market: str) -> None:
    details = [detail(), detail(ticker="OTHER", market=market)]
    frame, _ = read_cotahist(write_zip(tmp_path, 2024, lines_for(2024, details)))
    assert frame["ticker"].tolist() == ["PETR4"]


def test_round_and_odd_lot_markets_are_kept(tmp_path: Path) -> None:
    details = [detail(), detail(ticker="PETR4F", market="020", bdi="96")]
    frame, _ = read_cotahist(write_zip(tmp_path, 2024, lines_for(2024, details)))
    assert frame["market_type"].tolist() == [10, 20]


@pytest.mark.parametrize("bdi", ["05", "06", "07", "08", "12", "13", "14", "34", "36"])
def test_every_bdi_code_is_kept(tmp_path: Path, bdi: str) -> None:
    record = detail(ticker="BHIA3", bdi=bdi, isin="BRBHIAACNOR1")
    frame, _ = read_cotahist(write_zip(tmp_path, 2024, lines_for(2024, [record])))
    assert frame.loc[0, "bdi_code"] == int(bdi)


# --- Coverage and currency (H1.6) ---


def test_sessions_before_the_real_are_dropped(tmp_path: Path) -> None:
    details = [
        detail(session="19940630", currency="CR$"),
        detail(session="19940701", currency="R$"),
    ]
    frame, _ = read_cotahist(write_zip(tmp_path, 1994, lines_for(1994, details)))
    assert frame["session_date"].tolist() == [pd.Timestamp("1994-07-01")]


def test_other_currency_after_the_cut_is_an_error(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 1994, lines_for(1994, [detail(session="19940701", currency="US$")]))
    with pytest.raises(CotahistFormatError, match="R\\$"):
        read_cotahist(path)


# --- Key (H1.8) ---


def test_same_ticker_with_two_isins_is_kept(tmp_path: Path) -> None:
    details = [
        detail(ticker="BIOM12F", market="020", bdi="96", isin="BRBIOMN03OR2"),
        detail(ticker="BIOM12F", market="020", bdi="96", isin="BRBIOMR12OR4"),
    ]
    frame, _ = read_cotahist(write_zip(tmp_path, 2022, lines_for(2022, details)))
    assert sorted(frame["isin"]) == ["BRBIOMN03OR2", "BRBIOMR12OR4"]


def test_repeated_key_is_an_error(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, lines_for(2024, [detail(), detail(close=40.0)]))
    with pytest.raises(CotahistFormatError, match="repeated key"):
        read_cotahist(path)


# --- Trailer conventions (H1.3) ---


def test_trailer_counting_detail_records_only(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2025, lines_for(2025, [detail()], convention="detail"))
    _, report = read_cotahist(path)
    assert report.trailer_convention is TrailerConvention.DETAIL_RECORDS


@pytest.mark.parametrize("count", [0, 1, 3, 5, 100])
def test_trailer_count_matching_no_convention(tmp_path: Path, count: int) -> None:
    # 4 records in all, 2 of them detail records: only 4 and 2 are valid counts.
    lines = [header(2024), detail(), detail(ticker="VALE3"), trailer(2024, count)]
    path = write_zip(tmp_path, 2024, lines)
    with pytest.raises(CotahistFormatError, match="trailer declares"):
        read_cotahist(path)


def test_trailer_count_not_a_number(tmp_path: Path) -> None:
    bad = trailer(2024, 3)[:31] + b"ABCDEFGHIJK" + trailer(2024, 3)[42:]
    path = write_zip(tmp_path, 2024, [header(2024), detail(), bad])
    with pytest.raises(CotahistFormatError, match="not a number"):
        read_cotahist(path)


# --- Structure (H1.3) ---


def test_missing_header(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, [detail(), trailer(2024, 2)])
    with pytest.raises(CotahistFormatError, match="line 1 is not a header"):
        read_cotahist(path)


def test_second_header(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, [header(2024), header(2024), trailer(2024, 3)])
    with pytest.raises(CotahistFormatError, match="second header"):
        read_cotahist(path)


def test_missing_trailer(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, [header(2024), detail()])
    with pytest.raises(CotahistFormatError, match="not a trailer"):
        read_cotahist(path)


def test_record_after_the_trailer(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, [*lines_for(2024, [detail()]), detail(ticker="VALE3")])
    with pytest.raises(CotahistFormatError, match="after the trailer"):
        read_cotahist(path)


def test_unknown_record_type(tmp_path: Path) -> None:
    odd = b"07" + detail()[2:]
    path = write_zip(tmp_path, 2024, [header(2024), odd, trailer(2024, 3)])
    with pytest.raises(CotahistFormatError, match="unknown record type '07'"):
        read_cotahist(path)


def test_empty_file(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, [])
    with pytest.raises(CotahistFormatError, match="empty"):
        read_cotahist(path)


@pytest.mark.parametrize("which", ["header", "trailer"])
def test_year_in_header_or_trailer_must_match_the_file(tmp_path: Path, which: str) -> None:
    lines = lines_for(2024, [detail()])
    if which == "header":
        lines[0] = header(2023)
    else:
        lines[-1] = trailer(2023, 3)
    path = write_zip(tmp_path, 2024, lines)
    with pytest.raises(CotahistFormatError, match=f"the {which} names 'COTAHIST.2023'"):
        read_cotahist(path)


def test_invalid_generation_date(tmp_path: Path) -> None:
    lines = lines_for(2024, [detail()])
    lines[0] = header(2024, generated="20241399")
    with pytest.raises(CotahistFormatError, match="generation date"):
        read_cotahist(write_zip(tmp_path, 2024, lines))


def test_zip_with_more_than_one_member(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, lines_for(2024, [detail()]), extra_members=["README.TXT"])
    with pytest.raises(CotahistFormatError, match="exactly one file"):
        read_cotahist(path)


def test_member_name_is_free(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 1994, lines_for(1994, [detail(session="19940801")]), member="X")
    _, report = read_cotahist(path)
    assert report.member == "X"


def test_non_numeric_field(tmp_path: Path) -> None:
    record = bytearray(detail())
    record[147:152] = b"12A45"  # TOTNEG
    path = write_zip(tmp_path, 2024, lines_for(2024, [bytes(record)]))
    with pytest.raises(CotahistFormatError, match="'trades'"):
        read_cotahist(path)


def test_zero_quotation_factor(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, lines_for(2024, [detail(factor=0)]))
    with pytest.raises(CotahistFormatError, match="quotation factor"):
        read_cotahist(path)


# --- Encoding (H1.4) ---


def test_latin1_text_is_decoded(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, lines_for(2024, [detail(company="CIA ENERGÉTI")]))
    frame, _ = read_cotahist(path)
    assert frame.loc[0, "company_name"] == "CIA ENERGÉTI"


def test_utf8_record_is_repaired_and_counted(tmp_path: Path) -> None:
    record = detail(company="FROTA OCEÂNI", ticker="FR0T5L")
    utf8 = record.replace("OCEÂNI".encode("latin-1"), "OCEÂNI".encode())
    assert len(utf8) == 246
    path = write_zip(tmp_path, 2026, lines_for(2026, [detail(), utf8]))

    frame, report = read_cotahist(path)

    assert report.rows_repaired == 1
    assert frame.set_index("ticker").loc["FR0T5L", "company_name"] == "FROTA OCEÂNI"


def test_repaired_record_outside_the_market_filter_is_still_counted(tmp_path: Path) -> None:
    record = detail(company="FROTA OCEÂNI", market="017")
    utf8 = record.replace("OCEÂNI".encode("latin-1"), "OCEÂNI".encode())
    path = write_zip(tmp_path, 2026, lines_for(2026, [detail(), utf8]))

    frame, report = read_cotahist(path)

    assert (report.rows_repaired, report.rows_kept) == (1, 1)


@pytest.mark.parametrize("record", [b"01" + b" " * 240, detail() + b"X"], ids=["short", "long"])
def test_wrong_length_that_utf8_cannot_explain(tmp_path: Path, record: bytes) -> None:
    path = write_zip(tmp_path, 2024, [header(2024), record, trailer(2024, 3)])
    with pytest.raises(CotahistFormatError, match="line 2 is"):
        read_cotahist(path)


def test_invalid_utf8_of_the_wrong_length(tmp_path: Path) -> None:
    record = detail()[:-1] + b"\xff\xfe"
    path = write_zip(tmp_path, 2024, [header(2024), record, trailer(2024, 3)])
    with pytest.raises(CotahistFormatError, match="246 bytes long"):
        read_cotahist(path)


# --- File names ---


@pytest.mark.parametrize(
    "name", ["COTAHIST_A2024.ZIP", "cotahist_a2024.zip", "COTAHIST_A2024.zip"]
)
def test_year_from_name_ignores_case(name: str) -> None:
    assert year_of(Path(name)) == 2024


@pytest.mark.parametrize("name", ["COTAHIST_D02012024.ZIP", "COTAHIST_A24.ZIP", "notes.zip"])
def test_other_names_are_rejected(name: str) -> None:
    with pytest.raises(CotahistFormatError, match="not a COTAHIST annual file name"):
        year_of(Path(name))


def test_errors_share_a_base_class(tmp_path: Path) -> None:
    path = write_zip(tmp_path, 2024, [header(2024), detail()])
    with pytest.raises(CotahistError):
        read_cotahist(path)
    with pytest.raises(ValueError):
        read_cotahist(path)
