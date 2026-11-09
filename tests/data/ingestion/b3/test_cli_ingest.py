# ibexQuant\tests\data\ingestion\b3\test_cli_ingest.py

"""
Tests for ``ibexquant data ingest cotahist`` (ADR 0011, H1.14).

Calls :func:`ibexQuant.cli.main` in-process with the data root set through
the environment, so no test reads the machine's real configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support.cotahist import detail, header, lines_for, write_zip

from ibexQuant.cli import EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.ingestion.b3 import read_manifest
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module

# --- Fixtures ---


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataStore:
    store = DataStore.initialize(tmp_path / "ibex-data")
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    return store


def add_year(store: DataStore, year: int) -> None:
    write_zip(store.raw("b3", "cotahist"), year, lines_for(year, [detail(session=f"{year}0102")]))


# --- Commands ---


def test_one_year(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    add_year(store, 2024)

    assert main(["data", "ingest", "cotahist", "--year", "2024"]) == EXIT_OK

    out = capsys.readouterr().out
    assert out.startswith("2024: 1 rows, 1 sessions (2024-01-02 to 2024-01-02)")
    assert "trailer all_records" in out
    assert sorted(read_manifest(store)) == [2024]


def test_repeated_year_option(store: DataStore) -> None:
    for year in (2022, 2023, 2024):
        add_year(store, year)

    assert main(["data", "ingest", "cotahist", "--year", "2024", "--year", "2022"]) == EXIT_OK

    assert sorted(read_manifest(store)) == [2022, 2024]


def test_all_years(store: DataStore) -> None:
    for year in (2023, 2024):
        add_year(store, year)

    assert main(["data", "ingest", "cotahist", "--all"]) == EXIT_OK

    assert sorted(read_manifest(store)) == [2023, 2024]


def test_a_failing_year_is_reported_and_the_others_are_built(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    add_year(store, 2023)
    write_zip(store.raw("b3", "cotahist"), 2024, [header(2024)])
    add_year(store, 2025)

    assert main(["data", "ingest", "cotahist", "--all"]) == EXIT_USAGE

    assert "2024: error:" in capsys.readouterr().err
    assert sorted(read_manifest(store)) == [2023, 2025]


def test_missing_year_is_reported(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    add_year(store, 2024)
    assert main(["data", "ingest", "cotahist", "--year", "2023"]) == EXIT_USAGE
    assert "COTAHIST_A2023.ZIP" in capsys.readouterr().err


def test_no_raw_files(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "ingest", "cotahist", "--all"]) == EXIT_USAGE
    assert "no COTAHIST files" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["data", "ingest", "cotahist"],
        ["data", "ingest", "cotahist", "--all", "--year", "2024"],
        ["data", "ingest"],
    ],
    ids=["no-selection", "both-selections", "no-source"],
)
def test_invalid_usage(store: DataStore, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == EXIT_USAGE
