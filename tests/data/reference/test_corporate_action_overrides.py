# ibexQuant\tests\data\reference\test_corporate_action_overrides.py

"""Tests for ``corporate_actions.yaml`` (ADR 0013, H3.11)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ibexQuant.data.reference.corporate_action_overrides import (
    ActionOverrides,
    load_action_overrides,
)
from ibexQuant.data.reference.overrides import OverridesFormatError

VALID = """\
version: 1
events:
  - isin: BRLAMEACNPR6
    last_cum_date: 2022-01-21
    type: CONVERSION
    share_ratio: 0.188964
    target_isin: BRAMERACNOR6
    note: "LAME4 holders received AMER3"
  - isin: BRODERACNPR3
    last_cum_date: 2010-04-30
    type: CASH_DIVIDEND
    cash_amount: 0.12
    note: "from the company's notice"
rejections:
  - isin: BRPETRACNOR9
    ex_date: 2024-04-30
    type: SPLIT
    source: inferred
    note: "price error, not a split"
accepted:
  - isin: BRPETRACNOR9
    session: 2024-05-03
    note: "auction session, no event"
"""


def write(root: Path, text: str) -> Path:
    path = root / "corporate_actions.yaml"
    path.write_text(text, encoding="utf-8")
    return root


def test_a_valid_file_gives_events_rejections_and_accepted(tmp_path: Path) -> None:
    overrides = load_action_overrides(write(tmp_path, VALID))
    events = overrides.events
    assert events[["isin", "event_type", "target_isin", "source"]].values.tolist() == [
        ["BRLAMEACNPR6", "CONVERSION", "BRAMERACNOR6", "manual"],
        ["BRODERACNPR3", "CASH_DIVIDEND", "", "manual"],
    ]
    assert events.iloc[0]["share_ratio"] == 0.188964
    assert events.iloc[0]["last_cum_date"] == pd.Timestamp("2022-01-21")
    assert events.iloc[0]["origin"] == "overrides"
    assert overrides.rejections[0].source == "inferred"
    assert overrides.rejections[0].ex_date == pd.Timestamp("2024-04-30")
    assert overrides.accepted[0].session == pd.Timestamp("2024-05-03")
    assert overrides.sha256 is not None


def test_a_missing_file_means_no_decisions(tmp_path: Path) -> None:
    overrides = load_action_overrides(tmp_path)
    assert overrides.events.empty
    assert overrides.sha256 is None


def test_none_is_empty() -> None:
    overrides = ActionOverrides.none()
    assert overrides.events.empty
    assert overrides.rejections == ()
    assert overrides.accepted == ()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- a\n", "mapping"),
        ("version: 2\n", "version"),
        ("version: 1\nlinks: []\n", "unknown key"),
        ("version: 1\nevents: {}\n", "must be a list"),
        ("version: 1\nevents:\n  - isin: X\n", "isin"),
        ("version: 1\nevents:\n  - just text\n", "mapping"),
        (
            "version: 1\nevents:\n  - {isin: BRPETRACNOR9, last_cum_date: 2024-01-02, "
            "type: BONUS, note: x}\n",
            "type",
        ),
        (
            "version: 1\nevents:\n  - {isin: BRPETRACNOR9, last_cum_date: 2024-01-02, "
            "type: CASH_DIVIDEND, note: x}\n",
            "cash_amount",
        ),
        (
            "version: 1\nevents:\n  - {isin: BRPETRACNOR9, last_cum_date: 2024-01-02, "
            "type: SPLIT, note: x}\n",
            "share_ratio",
        ),
        (
            "version: 1\nevents:\n  - {isin: BRPETRACNOR9, last_cum_date: 2024-01-02, "
            "type: SPIN_OFF, share_ratio: 0.1, note: x}\n",
            "target_isin",
        ),
        (
            "version: 1\nevents:\n  - {isin: BRPETRACNOR9, last_cum_date: someday, "
            "type: SPLIT, share_ratio: 2, note: x}\n",
            "last_cum_date",
        ),
        (
            "version: 1\nevents:\n  - {isin: BRPETRACNOR9, last_cum_date: 2024-01-02, "
            "type: SPLIT, share_ratio: two, note: x}\n",
            "share_ratio",
        ),
        (
            "version: 1\nevents:\n  - {isin: BRPETRACNOR9, last_cum_date: 2024-01-02, "
            "type: SPLIT, share_ratio: 2, note: ''}\n",
            "note",
        ),
        (
            "version: 1\nrejections:\n  - {isin: BRPETRACNOR9, ex_date: 2024-01-02, "
            "type: SPLIT, source: yahoo, note: x}\n",
            "source",
        ),
        ("version: 1\naccepted:\n  - {isin: BRPETRACNOR9, note: x}\n", "session"),
        ("version: 1\naccepted:\n  - {isin: BRPETRACNOR9, session: 2024-01-02}\n", "note"),
        ("version: 1\nevents: [\n", "YAML"),
    ],
)
def test_broken_files_are_errors(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(OverridesFormatError, match=message):
        load_action_overrides(write(tmp_path, text))
