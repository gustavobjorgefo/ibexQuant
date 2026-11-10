# ibexQuant\tests\data\reference\test_overrides.py

"""Tests for :func:`ibexQuant.data.reference.load_link_overrides` (ADR 0012, H2.10)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from support.reference import write_links

from ibexQuant.data.reference import (
    IsinPair,
    LinkOverrides,
    OverridesFormatError,
    load_link_overrides,
)


def write(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ticker_links.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_links_and_rejections_in_file_order(tmp_path: Path) -> None:
    write_links(tmp_path, links=[("A1", "B1"), ("A2", "B2")], rejections=[("A3", "B3")])

    overrides = load_link_overrides(tmp_path)

    assert overrides.links == (IsinPair("A1", "B1", "reviewed"), IsinPair("A2", "B2", "reviewed"))
    assert overrides.rejections == (IsinPair("A3", "B3", "reviewed"),)


def test_hash_of_the_file(tmp_path: Path) -> None:
    path = write_links(tmp_path, links=[("A1", "B1")])
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_link_overrides(tmp_path).sha256 == expected


def test_version_only_means_no_decisions(tmp_path: Path) -> None:
    write(tmp_path, "version: 1\n")
    overrides = load_link_overrides(tmp_path)
    assert (overrides.links, overrides.rejections) == ((), ())


def test_values_are_stripped(tmp_path: Path) -> None:
    write(tmp_path, "version: 1\nlinks:\n  - {from: ' A1 ', to: 'B1 ', note: ' why '}\n")
    assert load_link_overrides(tmp_path).links == (IsinPair("A1", "B1", "why"),)


def test_none_has_no_hash() -> None:
    assert LinkOverrides.none() == LinkOverrides(links=(), rejections=(), sha256=None)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OverridesFormatError, match="--no-overrides"):
        load_link_overrides(tmp_path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("version: [1\n", "not valid YAML"),
        ("- 1\n", "must be a mapping"),
        ("version: 2\n", "version must be 1"),
        ("version: 1\nlink: []\n", "unknown key"),
        ("version: 1\nlinks: {from: A, to: B, note: x}\n", "must be a list"),
        ("version: 1\nlinks:\n  - {from: A, to: B}\n", "exactly 'from', 'to' and 'note'"),
        ("version: 1\nlinks:\n  - {from: A, to: B, note: x, ratio: 2}\n", "exactly"),
        ("version: 1\nlinks:\n  - {from: A, to: B, note: ''}\n", "non-empty text"),
        ("version: 1\nlinks:\n  - {from: A, to: 7, note: x}\n", "non-empty text"),
        ("version: 1\nlinks:\n  - {from: A, to: A, note: x}\n", "same ISIN"),
        (
            "version: 1\nlinks:\n  - {from: A, to: B, note: x}\n"
            "rejections:\n  - {from: A, to: B, note: y}\n",
            "decided twice",
        ),
    ],
    ids=[
        "invalid-yaml",
        "not-a-mapping",
        "wrong-version",
        "unknown-key",
        "links-not-a-list",
        "missing-note",
        "extra-key",
        "empty-note",
        "not-text",
        "same-isin",
        "decided-twice",
    ],
)
def test_invalid_files(tmp_path: Path, text: str, match: str) -> None:
    write(tmp_path, text)
    with pytest.raises(OverridesFormatError, match=match):
        load_link_overrides(tmp_path)


def test_errors_are_value_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_link_overrides(tmp_path)
