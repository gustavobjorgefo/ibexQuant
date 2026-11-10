# ibexQuant\tests\data\reference\test_cli_reference.py

"""
Tests for ``ibexquant data build security-master`` and
``ibexquant data review links`` (ADR 0012, H2.15).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support.cotahist import lines_for, write_zip
from support.reference import FCA_ROWS, TIMP, TIMS, rename_market, write_fca, write_links

from ibexQuant.cli import EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.ingestion.b3 import build_bars_daily
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
    assert "pending 1" in out
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


def test_review_pending_only(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    main(["data", "build", "security-master", "--no-overrides"])
    capsys.readouterr()

    assert main(["data", "review", "links", "--pending"]) == EXIT_OK

    out = capsys.readouterr().out
    assert "TIMP3" in out
    assert "TIMS3" in out
    assert "linked_by_cnpj" not in out
    assert "1 candidate(s)." in out


def test_review_with_nothing_pending(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_links(tmp_path / "overrides", links=[(TIMP, TIMS)])
    use_overrides(monkeypatch, tmp_path / "overrides")
    main(["data", "build", "security-master"])
    capsys.readouterr()

    assert main(["data", "review", "links", "--pending"]) == EXIT_OK
    assert "No candidates." in capsys.readouterr().out


def test_review_before_building(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "review", "links"]) == EXIT_USAGE
    assert "build security-master" in capsys.readouterr().err
