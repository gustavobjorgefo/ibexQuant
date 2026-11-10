# ibexQuant\src\ibexQuant\data\reference\overrides.py

"""
Hand-made link decisions: ``ticker_links.yaml`` (ADR 0012, H2.10).

Responsibility
--------------
Read and validate the *shape* of the override file: version, lists of
``from`` / ``to`` / ``note`` entries, no pair decided twice. Whether the ISINs
exist and the pair is in time order is checked by the build, which has the
spans.

What does not belong here
-------------------------
- Locating the file — see ``ibexQuant.data.store.resolve_overrides_root``.
- Applying the decisions — see ``security_master``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import yaml

OVERRIDES_FILE: Final[str] = "ticker_links.yaml"
SCHEMA_VERSION: Final[int] = 1
_TOP_KEYS: Final[frozenset[str]] = frozenset({"version", "links", "rejections"})
_ENTRY_KEYS: Final[frozenset[str]] = frozenset({"from", "to", "note"})


class OverridesFormatError(ValueError):
    """The override file is missing, unreadable or breaks its schema."""


@dataclass(frozen=True, slots=True)
class IsinPair:
    """One decision: the listing ``from_isin`` continues (or not) as ``to_isin``."""

    from_isin: str
    to_isin: str
    note: str


@dataclass(frozen=True, slots=True)
class LinkOverrides:
    """
    The decisions of one override file.

    Attributes
    ----------
    links, rejections : tuple[IsinPair, ...]
        Pairs to link, and candidates reviewed and not linked.
    sha256 : str or None
        Hash of the file read; ``None`` when building without overrides.
    """

    links: tuple[IsinPair, ...]
    rejections: tuple[IsinPair, ...]
    sha256: str | None

    @classmethod
    def none(cls) -> Self:
        """No decisions: the result of building with ``--no-overrides``."""
        return cls(links=(), rejections=(), sha256=None)


def load_link_overrides(root: Path) -> LinkOverrides:
    """
    Read ``ticker_links.yaml`` from the overrides root.

    Parameters
    ----------
    root : Path
        The overrides root (see ``resolve_overrides_root``).

    Returns
    -------
    LinkOverrides
        The decisions, in file order.

    Raises
    ------
    OverridesFormatError
        If the file does not exist, is not valid YAML, or breaks the schema.
    """
    path = root / OVERRIDES_FILE
    if not path.is_file():
        raise OverridesFormatError(
            f"{path} does not exist. Create it with 'version: 1', or build with --no-overrides."
        )
    raw = path.read_bytes()
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise OverridesFormatError(f"{path} is not valid YAML: {error}") from error

    if not isinstance(document, dict):
        raise OverridesFormatError(f"{path} must be a mapping with 'version: 1'.")
    unknown = sorted(set(document) - _TOP_KEYS)
    if unknown:
        raise OverridesFormatError(f"{path}: unknown key(s) {unknown}.")
    if document.get("version") != SCHEMA_VERSION:
        raise OverridesFormatError(
            f"{path}: version must be {SCHEMA_VERSION}, got {document.get('version')!r}."
        )

    links = _entries(document.get("links"), "links", path)
    rejections = _entries(document.get("rejections"), "rejections", path)
    seen: set[tuple[str, str]] = set()
    for pair in (*links, *rejections):
        key = (pair.from_isin, pair.to_isin)
        if key in seen:
            raise OverridesFormatError(f"{path}: {key[0]} -> {key[1]} is decided twice.")
        seen.add(key)
    return LinkOverrides(
        links=links, rejections=rejections, sha256=hashlib.sha256(raw).hexdigest()
    )


def _entries(value: object, section: str, path: Path) -> tuple[IsinPair, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise OverridesFormatError(f"{path}: '{section}' must be a list.")
    pairs = []
    for index, entry in enumerate(value):
        where = f"{path}: {section}[{index}]"
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise OverridesFormatError(f"{where} must have exactly 'from', 'to' and 'note'.")
        values = {key: entry[key] for key in _ENTRY_KEYS}
        if not all(isinstance(item, str) and item.strip() for item in values.values()):
            raise OverridesFormatError(f"{where}: 'from', 'to' and 'note' must be non-empty text.")
        from_isin, to_isin = values["from"].strip(), values["to"].strip()
        if from_isin == to_isin:
            raise OverridesFormatError(f"{where}: 'from' and 'to' are the same ISIN.")
        pairs.append(IsinPair(from_isin, to_isin, values["note"].strip()))
    return tuple(pairs)
