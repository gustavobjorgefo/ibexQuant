# ibexQuant\tests\data\ingestion\b3\test_bars_daily.py

"""
Tests for :func:`ibexQuant.data.ingestion.b3.build_bars_daily` (ADR 0011).

Covers finding the raw ZIPs (H1.1), atomic partitions (H1.11), the manifest
(H1.12) and idempotent rebuilds of a replaceable current year (H1.2).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from support.cotahist import detail, header, lines_for, write_zip

from ibexQuant.data.ingestion.b3 import (
    CotahistError,
    CotahistFormatError,
    CotahistNotFoundError,
    build_bars_daily,
    raw_years,
    read_manifest,
)
from ibexQuant.data.ingestion.b3.bars_daily import (
    MANIFEST_SCHEMA_VERSION,
    PARSER_VERSION,
    source_path,
)
from ibexQuant.data.store import DataStore

# --- Fixtures ---


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return DataStore.initialize(tmp_path / "ibex-data")


def raw_folder(store: DataStore) -> Path:
    return store.raw("b3", "cotahist")


def partition(store: DataStore, year: int) -> Path:
    return store.curated("b3", "bars_daily", f"year={year}", "part-0.parquet")


def manifest_path(store: DataStore) -> Path:
    return store.curated("b3", "bars_daily", "manifest.json")


# --- Build ---


def test_build_writes_the_partition(store: DataStore) -> None:
    write_zip(raw_folder(store), 2024, lines_for(2024, [detail(), detail(ticker="VALE3")]))

    build_bars_daily(store, 2024)

    written = pd.read_parquet(partition(store, 2024))
    assert written["ticker"].tolist() == ["PETR4", "VALE3"]
    assert written["market_type"].dtype == "int16"
    assert written["close"].dtype == "float64"


def test_build_returns_and_records_the_manifest_entry(store: DataStore) -> None:
    source = write_zip(raw_folder(store), 2024, lines_for(2024, [detail()], convention="detail"))

    (entry,) = build_bars_daily(store, 2024)

    assert entry.year == 2024
    assert entry.source_file == "COTAHIST_A2024.ZIP"
    assert entry.source_bytes == source.stat().st_size
    assert entry.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert entry.generated_on == "2025-01-02"
    assert entry.trailer_convention == "detail_records"
    assert (entry.records_read, entry.rows_kept, entry.rows_repaired) == (3, 1, 0)
    assert (entry.sessions, entry.first_session, entry.last_session) == (
        1,
        "2024-01-02",
        "2024-01-02",
    )
    assert entry.parser_version == PARSER_VERSION
    assert read_manifest(store) == {2024: entry}


def test_manifest_file_is_plain_json_by_year(store: DataStore) -> None:
    write_zip(raw_folder(store), 2024, lines_for(2024, [detail()]))
    build_bars_daily(store, 2024)

    document = json.loads(manifest_path(store).read_text(encoding="utf-8"))

    assert document["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert list(document["years"]) == ["2024"]
    assert document["years"]["2024"]["source_file"] == "COTAHIST_A2024.ZIP"


def test_empty_year_records_no_session_range(store: DataStore) -> None:
    write_zip(raw_folder(store), 2024, lines_for(2024, [detail(market="070")]))
    (entry,) = build_bars_daily(store, 2024)
    assert (entry.rows_kept, entry.first_session, entry.last_session) == (0, None, None)


def test_build_every_raw_year_by_default(store: DataStore) -> None:
    for year in (2023, 2024):
        write_zip(raw_folder(store), year, lines_for(year, [detail(session=f"{year}0102")]))

    entries = build_bars_daily(store)

    assert [entry.year for entry in entries] == [2023, 2024]
    assert sorted(read_manifest(store)) == [2023, 2024]


def test_build_a_list_of_years(store: DataStore) -> None:
    for year in (2022, 2023, 2024):
        write_zip(raw_folder(store), year, lines_for(year, [detail(session=f"{year}0102")]))

    entries = build_bars_daily(store, [2024, 2022])

    assert [entry.year for entry in entries] == [2024, 2022]
    assert not partition(store, 2023).exists()


def test_rebuilding_keeps_other_years_in_the_manifest(store: DataStore) -> None:
    for year in (2023, 2024):
        write_zip(raw_folder(store), year, lines_for(year, [detail(session=f"{year}0102")]))
    build_bars_daily(store)

    build_bars_daily(store, 2024)

    assert sorted(read_manifest(store)) == [2023, 2024]


# --- Idempotence and the replaceable current year (H1.2) ---


def test_rebuilding_the_same_zip_writes_the_same_rows(store: DataStore) -> None:
    write_zip(raw_folder(store), 2024, lines_for(2024, [detail(), detail(ticker="VALE3")]))
    build_bars_daily(store, 2024)
    first = pd.read_parquet(partition(store, 2024))
    first_entry = read_manifest(store)[2024]

    build_bars_daily(store, 2024)

    pd.testing.assert_frame_equal(pd.read_parquet(partition(store, 2024)), first)
    assert read_manifest(store)[2024].source_sha256 == first_entry.source_sha256


def test_replacing_the_current_year_updates_rows_and_hash(store: DataStore) -> None:
    write_zip(raw_folder(store), 2026, lines_for(2026, [detail(session="20260102")]))
    (old,) = build_bars_daily(store, 2026)

    later = [detail(session="20260102"), detail(session="20260105")]
    write_zip(raw_folder(store), 2026, lines_for(2026, later))
    (new,) = build_bars_daily(store, 2026)

    assert new.source_sha256 != old.source_sha256
    assert new.last_session == "2026-01-05"
    assert len(pd.read_parquet(partition(store, 2026))) == 2


def test_failed_build_leaves_the_previous_partition_untouched(store: DataStore) -> None:
    write_zip(raw_folder(store), 2026, lines_for(2026, [detail(session="20260102")]))
    build_bars_daily(store, 2026)
    before = partition(store, 2026).read_bytes()
    manifest_before = manifest_path(store).read_bytes()

    write_zip(raw_folder(store), 2026, [header(2026), detail(session="20260105")])
    with pytest.raises(CotahistFormatError):
        build_bars_daily(store, 2026)

    assert partition(store, 2026).read_bytes() == before
    assert manifest_path(store).read_bytes() == manifest_before


def test_no_temporary_file_is_left_behind(store: DataStore) -> None:
    write_zip(raw_folder(store), 2024, lines_for(2024, [detail()]))
    build_bars_daily(store, 2024)

    leftovers = list(store.curated("b3", "bars_daily").rglob("*.tmp"))

    assert leftovers == []


def test_a_failing_year_stops_the_build_after_the_years_before_it(store: DataStore) -> None:
    write_zip(raw_folder(store), 2023, lines_for(2023, [detail(session="20230102")]))
    write_zip(raw_folder(store), 2024, [header(2024)])

    with pytest.raises(CotahistError):
        build_bars_daily(store)

    assert partition(store, 2023).exists()
    assert sorted(read_manifest(store)) == [2023]


# --- Finding the raw files (H1.1) ---


def test_raw_years_lists_annual_files_only(store: DataStore) -> None:
    folder = raw_folder(store)
    write_zip(folder, 2024, lines_for(2024, [detail()]))
    write_zip(folder, 1999, lines_for(1999, [detail()]), name="COTAHIST_A1999.zip")
    (folder / "notes.txt").write_text("not a zip")
    (folder / "COTAHIST_D02012024.ZIP").write_bytes(b"")

    assert raw_years(store) == [1999, 2024]


def test_raw_years_without_the_folder(store: DataStore) -> None:
    assert raw_years(store) == []


def test_source_path_ignores_letter_case(store: DataStore) -> None:
    written = write_zip(raw_folder(store), 2005, lines_for(2005, []), name="COTAHIST_A2005.zip")
    assert source_path(store, 2005) == written


def test_missing_year(store: DataStore) -> None:
    write_zip(raw_folder(store), 2024, lines_for(2024, [detail()]))
    with pytest.raises(CotahistNotFoundError, match="COTAHIST_A2023.ZIP"):
        build_bars_daily(store, 2023)


def test_missing_raw_folder(store: DataStore) -> None:
    with pytest.raises(FileNotFoundError):
        build_bars_daily(store, 2024)


def test_two_files_for_one_year_are_ambiguous(store: DataStore) -> None:
    folder = raw_folder(store)
    write_zip(folder, 2024, lines_for(2024, [detail()]), name="COTAHIST_A2024.ZIP")
    write_zip(folder, 2024, lines_for(2024, [detail()]), name="cotahist_a2024.zip")
    if len(list(folder.iterdir())) == 1:
        pytest.skip("case-insensitive file system: both names are the same file")
    with pytest.raises(CotahistFormatError, match="More than one file"):
        build_bars_daily(store, 2024)


# --- Manifest ---


def test_manifest_is_empty_before_any_build(store: DataStore) -> None:
    assert read_manifest(store) == {}


def test_manifest_from_another_schema_version(store: DataStore) -> None:
    path = manifest_path(store)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 99, "years": {}}), encoding="utf-8")
    with pytest.raises(CotahistFormatError, match="schema version 99"):
        read_manifest(store)
