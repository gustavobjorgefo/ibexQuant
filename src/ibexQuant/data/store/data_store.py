# ibexQuant\src\ibexQuant\data\store\data_store.py

"""
The single owner of paths inside the data root.

Responsibility
--------------
:class:`DataStore` validates that the data root exists and holds the three
zones, and returns paths inside them. No other module builds a path into the
data root by hand (ADR 0010, H0.6), so moving the root — or, later, moving it
to the cloud — changes this class only.

Contract
--------
- Three zones, each with its own rule (H0.5): ``raw/`` is immutable,
  ``curated/`` is rebuilt from ``raw/`` plus overrides, ``derived/`` is
  disposable.
- Returned paths are **not created**. A writer creates the directories it
  needs; :meth:`DataStore.initialize` is the only method that touches disk.
- Path parts must be relative and must not contain ``..`` (H0.7), so no caller
  reaches outside its zone by accident.

What does not belong here
-------------------------
- Reading or writing data files — each ingestion module owns its format.
- Locating the data root — see ``config``.
- Cloud storage — deferred (H0.8); paths are ``pathlib.Path``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Self

from ibexQuant.data.store.config import resolve_data_root
from ibexQuant.data.store.errors import (
    DataLayoutError,
    DataRootConfigError,
    DataRootNotFoundError,
)


class Zone(StrEnum):
    """The three zones of the data root; the value is the directory name."""

    RAW = "raw"
    CURATED = "curated"
    DERIVED = "derived"


class DataStore:
    """
    Validated handle on a data root and its three zones.

    Parameters
    ----------
    root : str or os.PathLike
        Absolute path of an existing data root holding every zone.

    Raises
    ------
    DataRootConfigError
        If ``root`` is not absolute.
    DataRootNotFoundError
        If ``root`` does not exist.
    NotADirectoryError
        If ``root`` exists but is not a directory.
    DataLayoutError
        If a zone directory is missing. ``ibexquant data init`` creates them.

    Examples
    --------
    >>> store = DataStore.open()  # doctest: +SKIP
    >>> store.raw("b3", "cotahist", "COTAHIST_A2024.ZIP")  # doctest: +SKIP
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root: Path = _require_absolute(root)
        _require_directory(self._root)
        missing = [zone.value for zone in Zone if not (self._root / zone.value).is_dir()]
        if missing:
            raise DataLayoutError(
                f"Data root {self._root} lacks the zone(s) {', '.join(missing)}. "
                "Run `ibexquant data init` to create them."
            )

    # --- Construction ---

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        config_path: Path | None = None,
    ) -> Self:
        """
        Resolve the data root from the configuration and validate it.

        Parameters
        ----------
        root : str or os.PathLike, optional
            Explicit root; when omitted, the configuration is used.
        environ : Mapping[str, str], optional
            Forwarded to :func:`~ibexQuant.data.store.config.resolve_data_root`.
        config_path : Path, optional
            Forwarded to :func:`~ibexQuant.data.store.config.resolve_data_root`.

        Returns
        -------
        DataStore
            A handle on the validated root.

        Raises
        ------
        DataStoreError
            Any error of resolution (see ``resolve_data_root``) or validation
            (see the class docstring).
        """
        return cls(resolve_data_root(root, environ=environ, config_path=config_path))

    @classmethod
    def initialize(cls, root: str | os.PathLike[str]) -> Self:
        """
        Create the data root and its zones, keeping whatever already exists.

        Idempotent: running it on an initialized root changes nothing.

        Parameters
        ----------
        root : str or os.PathLike
            Absolute path of the data root. Missing parent directories are
            created.

        Returns
        -------
        DataStore
            A handle on the initialized root.

        Raises
        ------
        DataRootConfigError
            If ``root`` is not absolute.
        NotADirectoryError
            If ``root``, or one of its zones, exists as a file.
        """
        path = _require_absolute(root)
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"Data root {path} exists but is not a directory.")
        for zone in Zone:
            zone_path = path / zone.value
            if zone_path.exists() and not zone_path.is_dir():
                raise NotADirectoryError(f"Zone {zone_path} exists but is not a directory.")
            zone_path.mkdir(parents=True, exist_ok=True)
        return cls(path)

    # --- Paths ---

    @property
    def root(self) -> Path:
        """Absolute path of the data root."""
        return self._root

    def zone(self, zone: Zone) -> Path:
        """Return the directory of ``zone``."""
        return self._root / Zone(zone).value

    def raw(self, *parts: str) -> Path:
        """Return a path inside ``raw/`` (immutable downloads). Not created."""
        return _join(self.zone(Zone.RAW), parts)

    def curated(self, *parts: str) -> Path:
        """Return a path inside ``curated/`` (rebuilt tables). Not created."""
        return _join(self.zone(Zone.CURATED), parts)

    def derived(self, *parts: str) -> Path:
        """Return a path inside ``derived/`` (disposable products). Not created."""
        return _join(self.zone(Zone.DERIVED), parts)

    # --- Dunder ---

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={str(self._root)!r})"


# --- Helpers ---


def _require_absolute(root: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and require an absolute path."""
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise DataRootConfigError(f"The data root must be an absolute path, got {str(root)!r}.")
    return path


def _require_directory(path: Path) -> None:
    """Require ``path`` to exist and be a directory."""
    if not path.exists():
        raise DataRootNotFoundError(
            f"Data root {path} does not exist. Run `ibexquant data init` to create it."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Data root {path} exists but is not a directory.")


def _join(base: Path, parts: tuple[str, ...]) -> Path:
    """Join ``parts`` under ``base``, refusing anything that could leave it."""
    for part in parts:
        if not isinstance(part, str):
            raise TypeError(f"Path parts must be strings, got {type(part).__name__}.")
        if not part:
            raise ValueError("Path parts must not be empty.")
        # Both flavours are checked so the rule is the same on every OS: a part
        # that would escape on Windows (a drive such as "C:data", a leading
        # backslash, a backslash-separated "..") is refused on Linux too, where
        # it would merely be an odd file name.
        windows = PureWindowsPath(part)
        if windows.anchor or PurePosixPath(part).anchor:
            raise ValueError(f"Path parts must be relative, got {part!r}.")
        if ".." in windows.parts:
            raise ValueError(f"Path parts must not contain '..', got {part!r}.")
    return base.joinpath(*parts)
