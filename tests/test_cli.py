# ibexQuant\tests\test_cli.py

"""
Tests for the ``ibexquant`` command line (:mod:`ibexQuant.cli`).

Calls :func:`main` in-process with explicit arguments, so no test depends on
the installed console script or on the real configuration of the machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ibexQuant.cli import EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.store import DATA_ROOT_ENV_VAR
from ibexQuant.data.store import config as config_module

# --- Fixtures ---


@pytest.fixture
def unconfigured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make sure neither the environment nor the user's config file is read."""
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")


# --- data init ---


def test_init_creates_every_zone(
    unconfigured: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ibex-data"

    assert main(["data", "init", "--root", str(root)]) == EXIT_OK

    out = capsys.readouterr().out
    assert f"Data root: {root}" in out
    assert out.count("created") == 3
    assert all((root / zone).is_dir() for zone in ("raw", "curated", "derived"))


def test_init_twice_reports_zones_already_present(
    unconfigured: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "ibex-data"
    main(["data", "init", "--root", str(root)])
    capsys.readouterr()

    assert main(["data", "init", "--root", str(root)]) == EXIT_OK

    out = capsys.readouterr().out
    assert out.count("already present") == 3
    assert "created" not in out


def test_init_uses_the_configured_root(
    unconfigured: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "from-env"
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(root))

    assert main(["data", "init"]) == EXIT_OK
    assert (root / "raw").is_dir()


def test_init_without_configuration_fails_with_usage_code(
    unconfigured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "init"]) == EXIT_USAGE
    assert DATA_ROOT_ENV_VAR in capsys.readouterr().err


def test_init_with_relative_root_fails_with_usage_code(
    unconfigured: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "init", "--root", "ibex-data"]) == EXIT_USAGE
    assert capsys.readouterr().err.startswith("error:")


# --- Usage ---


@pytest.mark.parametrize("argv", [[], ["data"], ["unknown"]], ids=["none", "no-command", "bad"])
def test_invalid_usage_exits_with_usage_code(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == EXIT_USAGE
