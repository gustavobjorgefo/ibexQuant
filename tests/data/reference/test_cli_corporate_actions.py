# ibexQuant\tests\data\reference\test_cli_corporate_actions.py

"""Tests for ``ibexquant data build corporate-actions`` and ``data review events``."""

from __future__ import annotations

from pathlib import Path

import pytest
from support.ledger import PETR, ledger_store

from ibexQuant.cli import EX_SESSION_STATUSES, EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.reference.corporate_actions import EX_SESSION_STATUSES as LIBRARY_STATUSES
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, OVERRIDES_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataStore:
    store = ledger_store(tmp_path)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    monkeypatch.delenv(OVERRIDES_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    return store


def overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str | None) -> None:
    root = tmp_path / "overrides"
    root.mkdir()
    if body is not None:
        (root / "corporate_actions.yaml").write_text("version: 1\n" + body, encoding="utf-8")
    monkeypatch.setenv(OVERRIDES_ROOT_ENV_VAR, str(root))


def test_statuses_mirror_the_library() -> None:
    assert EX_SESSION_STATUSES == LIBRARY_STATUSES


# --- build corporate-actions ---


def test_build_without_overrides_prints_the_summary(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "build", "corporate-actions", "--no-overrides"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "events by source: b3_listed 3, bootstrap 1, inferred 1" in out
    assert "ex-sessions: explained 3, inferred 1, unexplained 1, unexplained_jump 1" in out
    assert "conflicts: 1" in out
    assert "snapshots: 2026-11-10" in out
    assert "built without corporate_actions.yaml" in out


def test_build_reads_the_overrides_root(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overrides(
        tmp_path, monkeypatch, f"accepted:\n  - {{isin: {PETR}, session: 2024-05-03, note: x}}\n"
    )
    assert main(["data", "build", "corporate-actions"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "accepted 1" in out
    assert "built without" not in out


def test_build_errors_exit_with_usage(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overrides(
        tmp_path, monkeypatch, f"accepted:\n  - {{isin: {PETR}, session: 2020-01-02, note: x}}\n"
    )
    assert main(["data", "build", "corporate-actions"]) == EXIT_USAGE
    assert "not in the skeleton" in capsys.readouterr().err


def test_snapshots_line_says_none_without_snapshots(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    for child in sorted(store.raw("b3", "listed_companies").rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    assert main(["data", "build", "corporate-actions", "--no-overrides"]) == EXIT_OK
    assert "snapshots: none" in capsys.readouterr().out


# --- review events ---


def test_review_before_a_build_is_an_error(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "review", "events"]) == EXIT_USAGE
    assert "data build corporate-actions" in capsys.readouterr().err


def test_review_lists_inferred_and_jumps_and_unplaced_events(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["data", "build", "corporate-actions", "--no-overrides"])
    capsys.readouterr()
    assert main(["data", "review", "events"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "inferred" in out and "unexplained_jump" in out
    assert "2 ex-session(s)." in out
    assert "traded on both sides" in out


def test_review_yaml_gives_entries_to_edit(
    store: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["data", "build", "corporate-actions", "--no-overrides"])
    capsys.readouterr()
    assert main(["data", "review", "events", "--yaml"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "rejections:\n  # inferred SPLIT 10: PETR3" in out
    assert f"  # - {{isin: {PETR}, ex_date: 2024-04-30, type: SPLIT, source: inferred" in out
    assert f'  - {{isin: {PETR}, session: 2024-05-03, note: "TODO"}}' in out


def test_review_with_nothing_left(store: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    main(["data", "build", "corporate-actions", "--no-overrides"])
    capsys.readouterr()
    assert main(["data", "review", "events", "--status", "accepted"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "ex-session(s)" not in out
    assert "traded on both sides" in out


def test_review_reports_when_nothing_is_waiting(
    store: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for child in sorted(store.raw("b3", "listed_companies").rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    main(["data", "build", "corporate-actions", "--no-overrides"])
    capsys.readouterr()
    assert main(["data", "review", "events", "--status", "accepted"]) == EXIT_OK
    assert "Nothing to review." in capsys.readouterr().out
