# ibexQuant\src\ibexQuant\data\store\config.py

"""
Resolution of the data root from the machine's configuration.

Responsibility
--------------
Answer one question: *where is the data root on this machine?* The answer
comes, first match wins, from (ADR 0010, H0.3):

1. an explicit argument;
2. the environment variable ``IBEXQUANT_DATA_ROOT``;
3. the key ``root`` of the ``[data]`` table in ``~/.ibexquant/config.toml``.

Contract
--------
- The returned path is absolute (H0.4). A leading ``~`` is expanded; a
  relative path is rejected, because it would depend on the working
  directory.
- An empty environment variable counts as unset.
- Nothing is checked on disk: whether the root exists and holds the expected
  layout is :class:`~ibexQuant.data.store.data_store.DataStore`'s concern.
- There is no default location. Nothing configured raises
  :class:`~ibexQuant.data.store.errors.DataRootNotConfiguredError`.

What does not belong here
-------------------------
- Validating or creating directories — see ``data_store``.
- Any setting other than the data root. New settings are added when a phase
  needs them (the overrides location arrives in H2, see H0.10).
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from ibexQuant.data.store.errors import (
    DataRootConfigError,
    DataRootNotConfiguredError,
    OverridesRootNotConfiguredError,
)

DATA_ROOT_ENV_VAR: Final[str] = "IBEXQUANT_DATA_ROOT"
CONFIG_TABLE: Final[str] = "data"
CONFIG_ROOT_KEY: Final[str] = "root"

OVERRIDES_ROOT_ENV_VAR: Final[str] = "IBEXQUANT_OVERRIDES_ROOT"
OVERRIDES_CONFIG_TABLE: Final[str] = "overrides"


def default_config_path() -> Path:
    """
    Return the location of the per-machine config file.

    A function rather than a constant so that it follows the home directory
    at call time, and so that tests can replace it.

    Returns
    -------
    Path
        ``~/.ibexquant/config.toml``, expanded.
    """
    return Path.home() / ".ibexquant" / "config.toml"


def resolve_data_root(
    explicit: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> Path:
    """
    Resolve the data root, first match wins.

    Parameters
    ----------
    explicit : str or os.PathLike, optional
        Path given directly by the caller. Takes precedence over everything.
    environ : Mapping[str, str], optional
        Environment to read ``IBEXQUANT_DATA_ROOT`` from. Defaults to
        ``os.environ``; tests pass their own mapping.
    config_path : Path, optional
        Config file to read. Defaults to :func:`default_config_path`. A
        missing file is not an error: it simply configures nothing.

    Returns
    -------
    Path
        Absolute path of the data root. Not checked on disk.

    Raises
    ------
    DataRootNotConfiguredError
        If none of the three sources names a data root.
    DataRootConfigError
        If the path found is relative or empty, the config file is not valid
        TOML, or ``[data]`` / ``[data].root`` has the wrong type.
    """
    path = default_config_path() if config_path is None else config_path
    found = _resolve(
        explicit,
        environ=environ,
        config_path=path,
        env_var=DATA_ROOT_ENV_VAR,
        table=CONFIG_TABLE,
        what="data root",
    )
    if found is not None:
        return found
    raise DataRootNotConfiguredError(
        "No data root configured. Use one of:\n"
        "  - an explicit root (DataStore.open(root=...) or --root in the CLI);\n"
        f"  - the environment variable {DATA_ROOT_ENV_VAR};\n"
        f'  - root = "..." under [{CONFIG_TABLE}] in {path}.'
    )


def resolve_overrides_root(
    explicit: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> Path:
    """
    Resolve the folder of the hand-made override files (ADR 0012, H2.9).

    Same order and rules as :func:`resolve_data_root`: explicit argument,
    ``IBEXQUANT_OVERRIDES_ROOT``, then ``[overrides].root`` in the config
    file. The folder lives in the private ``ibexquant-data-config``
    repository, outside the data root (0010, H0.10).

    Returns
    -------
    Path
        Absolute path of the overrides folder. Not checked on disk.

    Raises
    ------
    OverridesRootNotConfiguredError
        If none of the three sources names it.
    DataRootConfigError
        If the path found is relative or empty, or the config file is invalid.
    """
    path = default_config_path() if config_path is None else config_path
    found = _resolve(
        explicit,
        environ=environ,
        config_path=path,
        env_var=OVERRIDES_ROOT_ENV_VAR,
        table=OVERRIDES_CONFIG_TABLE,
        what="overrides root",
    )
    if found is not None:
        return found
    raise OverridesRootNotConfiguredError(
        "No overrides root configured. Use one of:\n"
        "  - an explicit root;\n"
        f"  - the environment variable {OVERRIDES_ROOT_ENV_VAR};\n"
        f'  - root = "..." under [{OVERRIDES_CONFIG_TABLE}] in {path};\n'
        "  - or build without overrides (--no-overrides)."
    )


# --- Helpers ---


def _resolve(
    explicit: str | os.PathLike[str] | None,
    *,
    environ: Mapping[str, str] | None,
    config_path: Path,
    env_var: str,
    table: str,
    what: str,
) -> Path | None:
    """Explicit, then environment, then config file; ``None`` when none is set."""
    if explicit is not None:
        return _absolute(os.fspath(explicit), source=f"the explicit {what}")

    env = os.environ if environ is None else environ
    from_env = env.get(env_var, "")
    if from_env:
        return _absolute(from_env, source=f"environment variable {env_var}")

    from_file = _read_config_value(config_path, table)
    if from_file is not None:
        return _absolute(from_file, source=f"[{table}].{CONFIG_ROOT_KEY} in {config_path}")
    return None


def _absolute(raw: str, *, source: str) -> Path:
    """Expand ``~`` and require an absolute path; ``source`` names it in errors."""
    if not raw.strip():
        raise DataRootConfigError(f"{source} is empty.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise DataRootConfigError(
            f"{source} must be an absolute path, got {raw!r}: a relative root would "
            "depend on the working directory."
        )
    return path


def _read_config_value(path: Path, table_name: str) -> str | None:
    """Return ``[table_name].root`` from ``path``, or ``None`` when absent."""
    if not path.is_file():
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise DataRootConfigError(f"{path} is not valid TOML: {error}") from error

    table = document.get(table_name)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise DataRootConfigError(f"[{table_name}] in {path} must be a table.")

    root = table.get(CONFIG_ROOT_KEY)
    if root is None:
        return None
    if not isinstance(root, str):
        raise DataRootConfigError(
            f"[{table_name}].{CONFIG_ROOT_KEY} in {path} must be a string, "
            f"got {type(root).__name__}."
        )
    return root
