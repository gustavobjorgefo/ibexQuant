# ibexQuant\tests\data\store\test_data_store.py

"""
Tests for :class:`ibexQuant.data.store.DataStore`.

Covers initialization (idempotent, refuses files in the way), validation of
an existing root (H0.5), path building inside each zone and the rules that
keep every path inside its zone (H0.7).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ibexQuant.data.store import (
    DATA_ROOT_ENV_VAR,
    DataLayoutError,
    DataRootConfigError,
    DataRootNotFoundError,
    DataStore,
    DataStoreError,
    Zone,
)

# --- Fixtures ---


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "ibex-data"


@pytest.fixture
def store(root: Path) -> DataStore:
    return DataStore.initialize(root)


# --- Initialize ---


def test_initialize_creates_root_and_every_zone(root: Path) -> None:
    DataStore.initialize(root)
    assert sorted(child.name for child in root.iterdir()) == ["curated", "derived", "raw"]


def test_initialize_creates_missing_parents(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "ibex-data"
    DataStore.initialize(nested)
    assert (nested / "raw").is_dir()


def test_initialize_is_idempotent_and_keeps_content(root: Path) -> None:
    DataStore.initialize(root)
    kept = root / "raw" / "b3" / "file.zip"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"data")

    DataStore.initialize(root)

    assert kept.read_bytes() == b"data"


def test_initialize_completes_a_partial_layout(root: Path) -> None:
    (root / "raw").mkdir(parents=True)
    DataStore.initialize(root)
    assert all((root / zone.value).is_dir() for zone in Zone)


def test_initialize_refuses_root_that_is_a_file(root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        DataStore.initialize(root)


def test_initialize_refuses_zone_that_is_a_file(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "curated").write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        DataStore.initialize(root)


def test_initialize_refuses_relative_root() -> None:
    with pytest.raises(DataRootConfigError):
        DataStore.initialize("ibex-data")


# --- Validation of an existing root ---


def test_constructor_accepts_initialized_root(store: DataStore, root: Path) -> None:
    assert DataStore(root).root == root
    assert DataStore(str(root)).root == root


def test_missing_root(root: Path) -> None:
    with pytest.raises(DataRootNotFoundError, match="ibexquant data init"):
        DataStore(root)


def test_missing_root_is_a_file_not_found_error(root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DataStore(root)


def test_root_that_is_a_file(root: Path) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        DataStore(root)


@pytest.mark.parametrize("missing", list(Zone), ids=[zone.value for zone in Zone])
def test_missing_zone(store: DataStore, root: Path, missing: Zone) -> None:
    (root / missing.value).rmdir()
    with pytest.raises(DataLayoutError, match=missing.value) as caught:
        DataStore(root)
    assert "ibexquant data init" in str(caught.value)


def test_relative_root_is_rejected() -> None:
    with pytest.raises(DataRootConfigError):
        DataStore("ibex-data")


def test_every_error_is_a_store_error(root: Path) -> None:
    with pytest.raises(DataStoreError):
        DataStore(root)


# --- Open ---


def test_open_with_explicit_root(store: DataStore, root: Path) -> None:
    assert DataStore.open(root, environ={}).root == root


def test_open_from_environment(store: DataStore, root: Path, tmp_path: Path) -> None:
    opened = DataStore.open(
        environ={DATA_ROOT_ENV_VAR: str(root)}, config_path=tmp_path / "absent.toml"
    )
    assert opened.root == root


def test_open_from_config_file(store: DataStore, root: Path, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f"[data]\nroot = '{root}'\n", encoding="utf-8")
    assert DataStore.open(environ={}, config_path=config).root == root


def test_open_validates_the_resolved_root(root: Path) -> None:
    with pytest.raises(DataRootNotFoundError):
        DataStore.open(root, environ={})


# --- Paths ---


def test_zone_directories(store: DataStore, root: Path) -> None:
    assert store.zone(Zone.RAW) == root / "raw"
    assert store.zone(Zone.CURATED) == root / "curated"
    assert store.zone(Zone.DERIVED) == root / "derived"


def test_paths_inside_each_zone(store: DataStore, root: Path) -> None:
    assert store.raw("b3", "cotahist", "COTAHIST_A2024.ZIP") == (
        root / "raw" / "b3" / "cotahist" / "COTAHIST_A2024.ZIP"
    )
    assert store.curated("bars_daily", "year=2024") == (
        root / "curated" / "bars_daily" / "year=2024"
    )
    assert store.derived("adjustment_factors.parquet") == (
        root / "derived" / "adjustment_factors.parquet"
    )


def test_no_parts_returns_the_zone(store: DataStore) -> None:
    assert store.raw() == store.zone(Zone.RAW)


def test_parts_may_contain_separators(store: DataStore, root: Path) -> None:
    assert store.raw("b3/cotahist") == root / "raw" / "b3" / "cotahist"


def test_paths_are_not_created(store: DataStore) -> None:
    path = store.curated("bars_daily", "year=2024", "part.parquet")
    assert not path.exists()
    assert not path.parent.exists()


@pytest.mark.parametrize(
    "part",
    ["..", "../outside", "b3/../../outside", "b3\\..\\..\\outside"],
    ids=["dotdot", "dotdot-prefix", "dotdot-middle", "dotdot-backslash"],
)
def test_parent_references_are_refused(store: DataStore, part: str) -> None:
    with pytest.raises(ValueError, match=r"\.\."):
        store.raw(part)


@pytest.mark.parametrize(
    "part",
    ["/etc/passwd", "\\outside", "C:\\outside", "C:outside"],
    ids=["posix-absolute", "windows-root", "windows-absolute", "windows-drive-relative"],
)
def test_absolute_parts_are_refused(store: DataStore, part: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        store.curated(part)


def test_empty_part_is_refused(store: DataStore) -> None:
    with pytest.raises(ValueError, match="empty"):
        store.derived("bars", "")


def test_non_string_part_is_refused(store: DataStore) -> None:
    with pytest.raises(TypeError, match="strings"):
        store.raw(2024)  # type: ignore[arg-type]


# --- Dunder ---


def test_repr_shows_the_root(store: DataStore, root: Path) -> None:
    assert repr(store) == f"DataStore(root={str(root)!r})"
