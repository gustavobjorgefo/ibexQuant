# ibexQuant\tests\data\reference\test_corporate_actions.py

"""Tests for the build of the corporate actions ledger (ADR 0013)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from support.ledger import DAY, PETR, VALE, ledger_store

from ibexQuant.data.reference.corporate_action_overrides import (
    ActionOverrides,
    load_action_overrides,
)
from ibexQuant.data.reference.corporate_actions import (
    CONFLICTS_FILE,
    EVENT_COLUMNS,
    EVENTS_FILE,
    EX_SESSIONS_FILE,
    SUMMARY_FILE,
    UNRESOLVED_FILE,
    LedgerBuildError,
    _close,
    build_corporate_actions,
)
from ibexQuant.data.reference.errors import SecurityMasterNotBuiltError
from ibexQuant.data.store import DataStore


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return ledger_store(tmp_path)


@pytest.fixture
def built(store: DataStore) -> DataStore:
    build_corporate_actions(store, ActionOverrides.none())
    return store


def table(store: DataStore, name: str) -> pd.DataFrame:
    return pd.read_parquet(store.curated("b3", name))


def test_events_carry_ex_dates_sources_and_statuses(built: DataStore) -> None:
    events = table(built, EVENTS_FILE)
    assert list(events.columns) == list(EVENT_COLUMNS)
    rows = events.sort_values(["ex_date", "event_type"])[
        ["isin", "ex_date", "event_type", "source", "status", "confirmed_by"]
    ].values.tolist()
    assert rows == [
        [PETR, DAY("2024-04-23"), "CASH_DIVIDEND", "b3_listed", "review", ""],
        [VALE, DAY("2024-04-23"), "CASH_DIVIDEND", "bootstrap", "confirmed", ""],
        [PETR, DAY("2024-04-24"), "CASH_DIVIDEND", "b3_listed", "confirmed", "b3_credited"],
        [PETR, DAY("2024-04-26"), "SPLIT", "b3_listed", "confirmed", ""],
        [PETR, DAY("2024-04-30"), "SPLIT", "inferred", "inferred", ""],
    ]
    split = events[events["source"] == "inferred"].iloc[0]
    assert split["share_ratio"] == 10.0
    assert split["last_cum_date"] == DAY("2024-04-29")
    assert set(events["security_id"]) == {PETR, VALE}
    assert events["event_id"].is_unique


def test_ex_sessions_get_a_status(built: DataStore) -> None:
    ex = table(built, EX_SESSIONS_FILE)
    keys = zip(ex["isin"], ex["session_date"].dt.strftime("%m-%d"), strict=True)
    statuses = dict(zip(keys, ex["status"], strict=True))
    assert statuses == {
        (PETR, "04-24"): "explained",
        (PETR, "04-26"): "explained",
        (PETR, "04-30"): "inferred",
        (PETR, "05-02"): "unexplained",
        (PETR, "05-03"): "unexplained_jump",
        (VALE, "04-23"): "explained",
    }
    assert ex.loc[ex["session_date"] == DAY("2024-04-24"), "event_ids"].iloc[0] != ""


def test_lower_sources_that_disagree_are_conflicts(built: DataStore) -> None:
    conflicts = table(built, CONFLICTS_FILE)
    assert conflicts[["source", "amount", "kept_source", "kept_amounts"]].values.tolist() == [
        ["b3_credited", 0.7, "b3_listed", "0.5"]
    ]


def test_unresolved_rows_and_dropped_bootstrap_are_reported(built: DataStore) -> None:
    unresolved = table(built, UNRESOLVED_FILE)
    assert unresolved["key"].tolist() == ["XPTO3"]
    summary = json.loads(built.curated("b3", SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["bootstrap_dropped"] == {"security_has_b3_cash": 1, "no_ex_session": 1}
    assert summary["ex_sessions"]["inferred"] == 1
    assert summary["unexplained_by_year"]["unexplained_jump"] == {"2024": 1}
    assert summary["conflicts"] == 1
    assert summary["inputs"]["snapshots"] == ["2026-11-10"]
    assert summary["inputs"]["bootstrap"] is not None


def test_rebuilding_gives_the_same_tables(built: DataStore) -> None:
    first = table(built, EVENTS_FILE)
    build_corporate_actions(built, ActionOverrides.none())
    pd.testing.assert_frame_equal(first, table(built, EVENTS_FILE))


def test_a_build_without_sources_lists_the_skeleton(store: DataStore) -> None:
    for path in (
        store.raw("b3", "listed_companies"),
        store.raw("b3", "earnings_credited"),
        store.raw("bootstrap"),
    ):
        for child in sorted(path.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
    summary = build_corporate_actions(store, ActionOverrides.none())
    assert summary.events["source"] == {"inferred": 2}
    assert summary.ex_sessions == {"inferred": 2, "unexplained": 3, "unexplained_jump": 1}
    assert summary.inputs["bootstrap"] is None
    assert table(store, CONFLICTS_FILE).empty


def test_no_bars_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SecurityMasterNotBuiltError, match="bars_daily"):
        build_corporate_actions(DataStore.initialize(tmp_path / "data"), ActionOverrides.none())


@pytest.mark.parametrize(
    ("value", "reference", "expected"),
    [(0.5001, 0.5, True), (0.51, 0.5, False), (math.nan, math.nan, True), (math.nan, 1.0, False)],
)
def test_amounts_match_within_a_relative_tolerance(
    value: float, reference: float, expected: bool
) -> None:
    assert _close(value, reference, 0.005) is expected


# --- Overrides ---


def overrides(tmp_path: Path, body: str) -> ActionOverrides:
    root = tmp_path / "overrides"
    root.mkdir(exist_ok=True)
    (root / "corporate_actions.yaml").write_text("version: 1\n" + body, encoding="utf-8")
    return load_action_overrides(root)


DECISIONS = f"""\
events:
  - {{isin: {PETR}, last_cum_date: 2024-04-30, type: CASH_DIVIDEND, cash_amount: 0.01,
     note: "company notice"}}
rejections:
  - {{isin: {PETR}, ex_date: 2024-04-30, type: SPLIT, source: inferred, note: "bad print"}}
accepted:
  - {{isin: {PETR}, session: 2024-05-03, note: "auction"}}
"""


def test_overrides_add_reject_and_accept(store: DataStore, tmp_path: Path) -> None:
    summary = build_corporate_actions(store, overrides(tmp_path, DECISIONS))
    events = table(store, EVENTS_FILE)
    manual = events[events["source"] == "manual"].iloc[0]
    assert (manual["ex_date"], manual["status"]) == (DAY("2024-05-02"), "confirmed")
    inferred = events[events["source"] == "inferred"].iloc[0]
    assert inferred["status"] == "rejected"
    ex = table(store, EX_SESSIONS_FILE).set_index("session_date")
    assert ex.loc[DAY("2024-05-02"), "status"] == "explained"
    assert ex.loc[DAY("2024-04-30"), "status"] == "unexplained_jump"
    assert ex.loc[DAY("2024-05-03"), "status"] == "accepted"
    assert summary.inputs["overrides"] is not None


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            f"rejections:\n  - {{isin: {PETR}, ex_date: 2024-04-29, type: SPLIT, "
            "source: inferred, note: x}\n",
            "matches no event",
        ),
        (
            f"accepted:\n  - {{isin: {PETR}, session: 2024-04-29, note: x}}\n",
            "not in the skeleton",
        ),
        (f"accepted:\n  - {{isin: {PETR}, session: 2024-04-24, note: x}}\n", "already explained"),
    ],
)
def test_overrides_that_match_nothing_are_errors(
    store: DataStore, tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(LedgerBuildError, match=message):
        build_corporate_actions(store, overrides(tmp_path, body))


def test_an_event_after_the_last_session_has_no_ex_date(store: DataStore) -> None:
    path = store.raw("b3", "listed_companies", "2026-11-10", "supplement", "PETR.json")
    company = json.loads(path.read_text(encoding="utf-8"))
    company[0]["cashDividends"].append(
        {
            "isinCode": PETR,
            "label": "DIVIDENDO",
            "rate": "0,47",
            "lastDatePrior": "21/11/2026",
            "approvedOn": "06/11/2026",
            "paymentDate": "21/12/2026",
            "relatedTo": "Anual/2026",
        }
    )
    path.write_text(json.dumps(company), encoding="utf-8")
    build_corporate_actions(store, ActionOverrides.none())
    events = table(store, EVENTS_FILE)
    future = events[events["last_cum_date"] == DAY("2026-11-21")].iloc[0]
    assert pd.isna(future["ex_date"])
    assert future["placement"] == "not_traded"
    assert events["event_id"].is_unique
