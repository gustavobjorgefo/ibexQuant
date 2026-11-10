# ibexQuant\tests\data\store\test_config.py

"""
Tests for :func:`ibexQuant.data.store.config.resolve_data_root`.

Covers the resolution order (H0.3), path rules (H0.4) and every error of the
config file. No test reads the real environment or home directory: each one
passes its own ``environ`` and ``config_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ibexQuant.data.store import (
    DATA_ROOT_ENV_VAR,
    OVERRIDES_ROOT_ENV_VAR,
    DataRootConfigError,
    DataRootNotConfiguredError,
    DataStoreError,
    OverridesRootNotConfiguredError,
    resolve_data_root,
    resolve_overrides_root,
)
from ibexQuant.data.store import config as config_module

# --- Fixtures ---


@pytest.fixture
def explicit_root(tmp_path: Path) -> Path:
    return tmp_path / "explicit"


@pytest.fixture
def env_root(tmp_path: Path) -> Path:
    return tmp_path / "from-env"


@pytest.fixture
def file_root(tmp_path: Path) -> Path:
    return tmp_path / "from-file"


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def config_naming(tmp_path: Path, root: Path) -> Path:
    # TOML literal strings ('...') keep Windows backslashes as they are.
    return write_config(tmp_path, f"[data]\nroot = '{root}'\n")


# --- Resolution order ---


def test_explicit_wins_over_environment_and_file(
    tmp_path: Path, explicit_root: Path, env_root: Path, file_root: Path
) -> None:
    resolved = resolve_data_root(
        explicit_root,
        environ={DATA_ROOT_ENV_VAR: str(env_root)},
        config_path=config_naming(tmp_path, file_root),
    )
    assert resolved == explicit_root


def test_environment_wins_over_file(tmp_path: Path, env_root: Path, file_root: Path) -> None:
    resolved = resolve_data_root(
        environ={DATA_ROOT_ENV_VAR: str(env_root)},
        config_path=config_naming(tmp_path, file_root),
    )
    assert resolved == env_root


def test_file_is_used_when_nothing_else_is_set(tmp_path: Path, file_root: Path) -> None:
    resolved = resolve_data_root(environ={}, config_path=config_naming(tmp_path, file_root))
    assert resolved == file_root


def test_empty_environment_variable_counts_as_unset(tmp_path: Path, file_root: Path) -> None:
    resolved = resolve_data_root(
        environ={DATA_ROOT_ENV_VAR: ""}, config_path=config_naming(tmp_path, file_root)
    )
    assert resolved == file_root


def test_explicit_accepts_str(explicit_root: Path) -> None:
    assert resolve_data_root(str(explicit_root), environ={}) == explicit_root


def test_default_config_path_is_used_when_none_given(
    tmp_path: Path, file_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = config_naming(tmp_path, file_root)
    monkeypatch.setattr(config_module, "default_config_path", lambda: path)
    assert resolve_data_root(environ={}) == file_root


def test_default_config_path_lives_in_home() -> None:
    assert config_module.default_config_path() == Path.home() / ".ibexquant" / "config.toml"


# --- Nothing configured ---


@pytest.mark.parametrize(
    "content",
    ["", "[other]\nkey = 1\n", "[data]\nother = 'x'\n"],
    ids=["empty-file", "no-data-table", "no-root-key"],
)
def test_file_without_root_configures_nothing(tmp_path: Path, content: str) -> None:
    with pytest.raises(DataRootNotConfiguredError):
        resolve_data_root(environ={}, config_path=write_config(tmp_path, content))


def test_missing_file_configures_nothing(tmp_path: Path) -> None:
    with pytest.raises(DataRootNotConfiguredError):
        resolve_data_root(environ={}, config_path=tmp_path / "absent.toml")


def test_not_configured_message_lists_every_option(tmp_path: Path) -> None:
    with pytest.raises(DataRootNotConfiguredError) as caught:
        resolve_data_root(environ={}, config_path=tmp_path / "absent.toml")
    message = str(caught.value)
    assert "--root" in message
    assert DATA_ROOT_ENV_VAR in message
    assert "absent.toml" in message


def test_not_configured_is_a_lookup_error(tmp_path: Path) -> None:
    with pytest.raises(LookupError):
        resolve_data_root(environ={}, config_path=tmp_path / "absent.toml")


# --- Path rules ---


@pytest.mark.parametrize("relative", ["ibex-data", "./ibex-data", "../ibex-data"])
def test_relative_root_is_rejected_from_every_source(tmp_path: Path, relative: str) -> None:
    with pytest.raises(DataRootConfigError, match="absolute"):
        resolve_data_root(relative, environ={})
    with pytest.raises(DataRootConfigError, match="absolute"):
        resolve_data_root(environ={DATA_ROOT_ENV_VAR: relative})
    with pytest.raises(DataRootConfigError, match="absolute"):
        resolve_data_root(environ={}, config_path=config_naming(tmp_path, Path(relative)))


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_explicit_root_is_rejected(blank: str) -> None:
    with pytest.raises(DataRootConfigError, match="empty"):
        resolve_data_root(blank, environ={})


def test_blank_root_in_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataRootConfigError, match="empty"):
        resolve_data_root(environ={}, config_path=write_config(tmp_path, "[data]\nroot = ''\n"))


def test_tilde_is_expanded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Path.home() reads HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert resolve_data_root("~/ibex-data", environ={}) == tmp_path / "ibex-data"


def test_path_is_not_checked_on_disk(tmp_path: Path) -> None:
    missing = tmp_path / "does" / "not" / "exist"
    assert resolve_data_root(missing, environ={}) == missing
    assert not missing.exists()


# --- Invalid config file ---


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("[data\nroot = 'x'\n", "not valid TOML"),
        ("data = 'x'\n", "must be a table"),
        ("[data]\nroot = 42\n", "must be a string"),
    ],
    ids=["invalid-toml", "data-not-a-table", "root-not-a-string"],
)
def test_invalid_config_file(tmp_path: Path, content: str, match: str) -> None:
    with pytest.raises(DataRootConfigError, match=match):
        resolve_data_root(environ={}, config_path=write_config(tmp_path, content))


def test_config_errors_are_value_errors_and_store_errors(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[data]\nroot = 42\n")
    with pytest.raises(ValueError):
        resolve_data_root(environ={}, config_path=path)
    with pytest.raises(DataStoreError):
        resolve_data_root(environ={}, config_path=path)


# --- Overrides root (ADR 0012, H2.9) ---


def test_overrides_root_resolution_order(tmp_path: Path) -> None:
    config = write_config(tmp_path, f"[overrides]\nroot = '{tmp_path / 'file'}'\n")
    env = {OVERRIDES_ROOT_ENV_VAR: str(tmp_path / "env")}

    assert resolve_overrides_root(tmp_path / "x", environ=env, config_path=config) == (
        tmp_path / "x"
    )
    assert resolve_overrides_root(environ=env, config_path=config) == tmp_path / "env"
    assert resolve_overrides_root(environ={}, config_path=config) == tmp_path / "file"


def test_overrides_root_is_independent_of_the_data_root(tmp_path: Path) -> None:
    config = write_config(tmp_path, f"[data]\nroot = '{tmp_path}'\n")
    with pytest.raises(OverridesRootNotConfiguredError, match="--no-overrides"):
        resolve_overrides_root(environ={DATA_ROOT_ENV_VAR: str(tmp_path)}, config_path=config)


def test_overrides_root_follows_the_same_path_rules(tmp_path: Path) -> None:
    with pytest.raises(DataRootConfigError, match="absolute"):
        resolve_overrides_root("overrides", environ={})
    with pytest.raises(LookupError):
        resolve_overrides_root(environ={}, config_path=tmp_path / "absent.toml")
