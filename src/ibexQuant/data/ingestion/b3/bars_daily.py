# ibexQuant\src\ibexQuant\data\ingestion\b3\bars_daily.py

"""
Build of the curated ``bars_daily`` table from the raw COTAHIST files.

Responsibility
--------------
Find each year's ZIP in ``raw/b3/cotahist/``, read it with
:func:`~ibexQuant.data.ingestion.b3.cotahist.read_cotahist`, write the year's
partition and record what was built in the manifest (ADR 0011, H1.11 and
H1.12).

Contract
--------
- One partition per year, ``curated/b3/bars_daily/year=YYYY/part-0.parquet``,
  replaced atomically: a failed build leaves the previous partition as it
  was (H1.11).
- The manifest, ``curated/b3/bars_daily/manifest.json``, holds one entry per
  year with the SHA-256 of the ZIP that produced it (H1.2, H1.12).
- Building a year twice from the same ZIP writes the same rows.

What does not belong here
-------------------------
- Parsing the file format — see ``cotahist``.
- Serving bars to research code — see ``provider``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from ibexQuant.data.ingestion.b3.cotahist import FILE_NAME_PATTERN, read_cotahist, year_of
from ibexQuant.data.ingestion.b3.errors import CotahistFormatError, CotahistNotFoundError
from ibexQuant.data.store import DataStore
from ibexQuant.data.store.atomic import replace_atomically

RAW_PARTS: Final[tuple[str, ...]] = ("b3", "cotahist")
CURATED_PARTS: Final[tuple[str, ...]] = ("b3", "bars_daily")
PARTITION_FILE: Final[str] = "part-0.parquet"
MANIFEST_FILE: Final[str] = "manifest.json"

PARSER_VERSION: Final[int] = 1
"""Bumped whenever a parser change can alter the curated rows."""

MANIFEST_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """
    What was built for one year, and from which file.

    Attributes
    ----------
    year : int
        Year of the partition.
    source_file : str
        Name of the ZIP in ``raw/b3/cotahist/``.
    source_bytes : int
        Size of the ZIP.
    source_sha256 : str
        SHA-256 of the ZIP: identifies the exact download (H1.2).
    generated_on : str
        File generation date from the header, ISO format.
    trailer_convention : str
        ``"all_records"`` or ``"detail_records"`` (H1.3).
    records_read, rows_kept, rows_repaired, sessions : int
        Counts from :class:`~ibexQuant.data.ingestion.b3.cotahist.CotahistReport`.
    first_session, last_session : str or None
        Range of the partition, ISO format.
    parser_version : int
        :data:`PARSER_VERSION` at build time.
    built_at : str
        UTC build time, ISO format.
    """

    year: int
    source_file: str
    source_bytes: int
    source_sha256: str
    generated_on: str
    trailer_convention: str
    records_read: int
    rows_kept: int
    rows_repaired: int
    sessions: int
    first_session: str | None
    last_session: str | None
    parser_version: int
    built_at: str


def raw_years(store: DataStore) -> list[int]:
    """
    Return the years that have a COTAHIST ZIP in the raw zone, ascending.

    Parameters
    ----------
    store : DataStore
        The data store.

    Returns
    -------
    list[int]
        Years found; empty when the folder does not exist yet.
    """
    folder = store.raw(*RAW_PARTS)
    if not folder.is_dir():
        return []
    return sorted(
        {year_of(path) for path in folder.iterdir() if FILE_NAME_PATTERN.match(path.name)}
    )


def source_path(store: DataStore, year: int) -> Path:
    """
    Return the ZIP of ``year`` in the raw zone, whatever the letter case.

    Raises
    ------
    CotahistNotFoundError
        If there is no ZIP for ``year``.
    CotahistFormatError
        If there are two (e.g. ``.zip`` and ``.ZIP``), which would make the
        source ambiguous.
    """
    folder = store.raw(*RAW_PARTS)
    candidates = list(folder.iterdir()) if folder.is_dir() else []
    return pick_source(candidates, year, folder=folder)


def pick_source(candidates: Iterable[Path], year: int, *, folder: Path) -> Path:
    """
    Choose the ZIP of ``year`` among ``candidates``, comparing names without case.

    Separate from :func:`source_path` so the ambiguity rule can be tested on
    any file system: on a case-insensitive one (Windows), two names that
    differ only in case cannot coexist.

    Raises
    ------
    CotahistNotFoundError
        If no candidate is the ZIP of ``year``.
    CotahistFormatError
        If more than one is.
    """
    expected = f"COTAHIST_A{year}.ZIP"
    matches = [path for path in candidates if path.name.upper() == expected]
    if not matches:
        raise CotahistNotFoundError(f"No {expected} in {folder}.")
    if len(matches) > 1:
        names = sorted(path.name for path in matches)
        raise CotahistFormatError(f"More than one file for {year} in {folder}: {names}.")
    return matches[0]


def build_bars_daily(
    store: DataStore, years: int | Iterable[int] | None = None
) -> list[ManifestEntry]:
    """
    Build the curated partitions of ``years`` from their raw ZIPs.

    Each year is read, written and recorded before the next starts, so an
    error leaves the years already built in place.

    Parameters
    ----------
    store : DataStore
        The data store.
    years : int or Iterable[int], optional
        Years to build. Defaults to every year in the raw zone.

    Returns
    -------
    list[ManifestEntry]
        One entry per year built, in the order built.

    Raises
    ------
    CotahistError
        The first year that cannot be found or read stops the build.
    """
    wanted = raw_years(store) if years is None else _as_years(years)
    return [_build_year(store, year) for year in wanted]


def read_manifest(store: DataStore) -> dict[int, ManifestEntry]:
    """
    Return the manifest entries by year; empty when nothing was built yet.

    Raises
    ------
    CotahistFormatError
        If the manifest exists but was written by another schema version.
    """
    path = store.curated(*CURATED_PARTS, MANIFEST_FILE)
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CotahistFormatError(
            f"{path} has schema version {document.get('schema_version')!r}, "
            f"expected {MANIFEST_SCHEMA_VERSION}."
        )
    return {int(year): ManifestEntry(**entry) for year, entry in document["years"].items()}


# --- Helpers ---


def _as_years(years: int | Iterable[int]) -> list[int]:
    return [years] if isinstance(years, int) else list(years)


def _build_year(store: DataStore, year: int) -> ManifestEntry:
    source = source_path(store, year)
    frame, report = read_cotahist(source)

    partition = store.curated(*CURATED_PARTS, f"year={year}", PARTITION_FILE)
    partition.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(
        partition,
        lambda temporary: frame.to_parquet(
            temporary, engine="pyarrow", compression="zstd", index=False
        ),
    )

    entry = ManifestEntry(
        year=year,
        source_file=source.name,
        source_bytes=source.stat().st_size,
        source_sha256=_sha256(source),
        generated_on=report.generated_on.isoformat(),
        trailer_convention=report.trailer_convention.value,
        records_read=report.records_read,
        rows_kept=report.rows_kept,
        rows_repaired=report.rows_repaired,
        sessions=report.sessions,
        first_session=_iso(report.first_session),
        last_session=_iso(report.last_session),
        parser_version=PARSER_VERSION,
        built_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    _write_manifest(store, {**read_manifest(store), year: entry})
    return entry


def _write_manifest(store: DataStore, entries: dict[int, ManifestEntry]) -> None:
    document = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "years": {str(year): asdict(entries[year]) for year in sorted(entries)},
    }
    path = store.curated(*CURATED_PARTS, MANIFEST_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2) + "\n"
    replace_atomically(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(value: dt.date | None) -> str | None:
    return None if value is None else value.isoformat()
