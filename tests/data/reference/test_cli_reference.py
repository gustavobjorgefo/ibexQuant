# ibexQuant\tests\data\reference\test_cli_reference.py

"""
Tests for ``ibexquant data build security-master`` and
``ibexquant data review links`` (ADR 0012, H2.15).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support.cotahist import detail, lines_for, write_zip
from support.reference import (
    FCA_ROWS,
    TIMP,
    TIMS,
    isin,
    rename_market,
    write_fca,
    write_links,
)

from ibexQuant.cli import CANDIDATE_STATUSES, EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.ingestion.b3 import build_bars_daily
from ibexQuant.data.reference.security_master import STATUSES
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, OVERRIDES_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataStore:
    store = DataStore.initialize(tmp_path / "ibex-data")
    write_zip(store.raw("b3", "cotahist"), 2024, lines_for(2024, rename_market()))
    build_bars_daily(store)
    write_fca(store.raw("cvm", "fca"), 2024, FCA_ROWS)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    monkeypatch.delenv(OVERRIDES_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    return store


def use_overrides(monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    monkeypatch.setenv(OVERRIDES_ROOT_ENV_VAR, str(folder))


# --- build security-master ---


def test_build_without_overrides(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "build", "security-master", "--no-overrides"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "spans 7, ISINs 6, securities 4" in out
    assert "cnpj 2" in out
    assert "likely 1" in out
    assert "built without overrides" in out


def test_build_with_overrides(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_links(tmp_path / "overrides", links=[(TIMP, TIMS)])
    use_overrides(monkeypatch, tmp_path / "overrides")

    assert main(["data", "build", "security-master"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "override 1" in out
    assert "built without overrides" not in out


def test_build_needs_overrides_or_the_flag(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "build", "security-master"]) == EXIT_USAGE
    assert "--no-overrides" in capsys.readouterr().err


def test_build_with_a_missing_override_file(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_overrides(monkeypatch, tmp_path / "empty")
    assert main(["data", "build", "security-master"]) == EXIT_USAGE
    assert "ticker_links.yaml" in capsys.readouterr().err


def test_build_with_an_invalid_override(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_links(tmp_path / "overrides", links=[(TIMS, TIMP)])
    use_overrides(monkeypatch, tmp_path / "overrides")
    assert main(["data", "build", "security-master"]) == EXIT_USAGE
    assert "traded before" in capsys.readouterr().err


# --- review links ---


def test_review_lists_every_candidate(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["data", "build", "security-master", "--no-overrides"])
    capsys.readouterr()

    assert main(["data", "review", "links"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "linked_by_cnpj" in out
    assert "superseded" in out
    assert "candidate(s)." in out


def test_review_by_status(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    main(["data", "build", "security-master", "--no-overrides"])
    capsys.readouterr()

    assert main(["data", "review", "links", "--status", "likely"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "TIMP3" in out
    assert "TIMS3" in out
    assert "linked_by_cnpj" not in out
    assert "1 candidate(s)." in out


def test_review_several_statuses(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    main(["data", "build", "security-master", "--no-overrides"])
    capsys.readouterr()

    argv = ["data", "review", "links", "--status", "likely", "--status", "linked_by_cnpj"]
    assert main(argv) == EXIT_OK
    assert "3 candidate(s)." in capsys.readouterr().out


def test_review_as_yaml_round_trips_into_the_override_file(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["data", "build", "security-master", "--no-overrides"])
    capsys.readouterr()

    assert main(["data", "review", "links", "--status", "likely", "--yaml"]) == EXIT_OK
    entries = capsys.readouterr().out
    assert f"from: {TIMP}  # TIMP3, last session 2024-01-04" in entries
    assert f"to: {TIMS}  # TIMS3, first session 2024-01-05" in entries
    assert 'note: "TIMP3 -> TIMS3, price ratio 1.000"' in entries

    folder = tmp_path / "overrides"
    folder.mkdir()
    (folder / "ticker_links.yaml").write_text("version: 1\n" + entries, encoding="utf-8")
    use_overrides(monkeypatch, folder)
    assert main(["data", "build", "security-master"]) == EXIT_OK
    assert "override 1" in capsys.readouterr().out


def test_review_with_no_match(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_links(tmp_path / "overrides", links=[(TIMP, TIMS)])
    use_overrides(monkeypatch, tmp_path / "overrides")
    main(["data", "build", "security-master"])
    capsys.readouterr()

    assert main(["data", "review", "links", "--status", "likely"]) == EXIT_OK
    assert "No candidates." in capsys.readouterr().out


def test_review_yaml_flags_an_implied_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = DataStore.initialize(tmp_path / "ibex-data")
    details = [
        detail(session="20240102", ticker="OLDX3", isin=isin("OLDX"), close=10.0),
        detail(session="20240103", ticker="NEWX3", isin=isin("NEWX"), open_=50.0),
    ]
    write_zip(store.raw("b3", "cotahist"), 2024, lines_for(2024, details))
    build_bars_daily(store)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    main(["data", "build", "security-master", "--no-overrides"])
    capsys.readouterr()

    assert main(["data", "review", "links", "--status", "pending", "--yaml"]) == EXIT_OK
    assert "price ratio 5.000 (implied factor 5: split or exchange ratio?)" in (
        capsys.readouterr().out
    )


def test_cli_statuses_match_the_library() -> None:
    assert CANDIDATE_STATUSES == STATUSES


def test_review_before_building(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "review", "links"]) == EXIT_USAGE
    assert "build security-master" in capsys.readouterr().err
