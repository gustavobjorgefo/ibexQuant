# ibexQuant\tests\data\reference\test_corporate_actions.py

"""Tests for the build of the corporate actions ledger (ADR 0013)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ibexQuant.data.reference.corporate_actions import (
    CONFLICTS_FILE,
    EVENT_COLUMNS,
    EVENTS_FILE,
    EX_SESSIONS_FILE,
    SUMMARY_FILE,
    UNRESOLVED_FILE,
    _close,
    build_corporate_actions,
)
from ibexQuant.data.reference.errors import SecurityMasterNotBuiltError
from ibexQuant.data.store import DataStore

PETR, VALE = "BRPETRACNOR9", "BRVALEACNOR0"
DAY = pd.Timestamp

# PETR3 sessions: (day, distribution number, open, close).
PETR_SESSIONS: list[tuple[str, int, float, float]] = [
    ("2024-04-22", 100, 10.0, 10.0),
    ("2024-04-23", 100, 10.0, 10.0),
    ("2024-04-24", 101, 9.5, 10.0),  # dividend, data-com 04-23
    ("2024-04-25", 101, 10.0, 10.0),
    ("2024-04-26", 102, 5.1, 5.1),  # split 2 for 1, data-com 04-25
    ("2024-04-29", 102, 10.2, 10.2),
    ("2024-04-30", 103, 1.02, 1.0),  # no source: open/close 0.1 -> inferred split 10
    ("2024-05-02", 104, 0.99, 1.0),  # no source, small move -> unexplained
    ("2024-05-03", 105, 0.3, 0.3),  # no source, 0.3 -> no common ratio -> jump
]
VALE_SESSIONS: list[tuple[str, int, float, float]] = [
    ("2024-04-22", 200, 60.0, 60.0),
    ("2024-04-23", 201, 58.0, 58.0),  # bootstrap dividend, data-com 04-22
    ("2024-04-24", 201, 58.0, 58.0),
]


def bars_rows(
    isin: str, ticker: str, sessions: list[tuple[str, int, float, float]]
) -> list[dict[str, Any]]:
    return [
        {
            "session_date": DAY(day),
            "ticker": ticker,
            "isin": isin,
            "market_type": 10,
            "distribution_number": number,
            "open": open_,
            "close": close,
            "volume_shares": 1000,
        }
        for day, number, open_, close in sessions
    ]


def master_row(isin: str, ticker: str) -> dict[str, Any]:
    return {
        "security_id": isin,
        "ticker": ticker,
        "isin": isin,
        "market_type": 10,
        "first_session": DAY("2024-04-22"),
        "last_session": DAY("2024-05-03"),
        "sessions": 9,
        "link_rule": "founding",
        "cnpj": "",
    }


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


def history(value: str, last_cum: str) -> dict[str, str]:
    return {
        "typeStock": "ON",
        "dateApproval": "01/04/2024",
        "valueCash": value,
        "ratio": "1",
        "corporateAction": "DIVIDENDO",
        "lastDatePriorEx": last_cum,
        "closingPricePriorExDate": "10,00",
        "quotedPerShares": "1",
    }


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    store = DataStore.initialize(tmp_path / "data")
    bars = pd.DataFrame(
        bars_rows(PETR, "PETR3", PETR_SESSIONS) + bars_rows(VALE, "VALE3", VALE_SESSIONS)
    )
    partition = store.curated("b3", "bars_daily", "year=2024", "part-0.parquet")
    partition.parent.mkdir(parents=True)
    bars.to_parquet(partition, index=False)
    pd.DataFrame([master_row(PETR, "PETR3"), master_row(VALE, "VALE3")]).to_parquet(
        store.curated("b3", "security_master.parquet"), index=False
    )

    snapshot = store.raw("b3", "listed_companies", "2026-11-10")
    company = {
        "tradingName": "PETROBRAS",
        "cashDividends": [],
        "subscriptions": [],
        "stockDividends": [
            {
                "assetIssued": PETR,
                "factor": "100,00000000000",
                "approvedOn": "01/04/2024",
                "isinCode": PETR,
                "label": "DESDOBRAMENTO",
                "lastDatePrior": "25/04/2024",
            }
        ],
    }
    write_json(snapshot / "supplement" / "PETR.json", [company])
    write_json(
        snapshot / "cash_dividends" / "PETR" / "page-001.json",
        {
            "page": {"totalPages": 1},
            "results": [
                history("0,50", "23/04/2024"),
                history("0,10", "22/04/2024"),  # PETR3 traded 04-23 under the same number
            ],
        },
    )

    credited = store.raw("b3", "earnings_credited", "2024", "2024-05-10.csv")
    credited.parent.mkdir(parents=True)
    credited.write_text(
        "Emissor;Código ISIN;Tipo de provento;Data de aprovação;Valor (R$);Data de crédito\n"
        f"PETROBRAS;{PETR} 100;DIVIDENDO;-;0,5001;10/05/2024\n"
        f"PETROBRAS;{PETR} 100;DIVIDENDO;-;0,70;10/05/2024\n",
        encoding="utf-8",
    )

    bootstrap = store.raw("bootstrap", "statusinvest_cash.parquet")
    bootstrap.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["PETR3", "VALE3", "VALE3", "XPTO3"],
            "last_cum_date": [
                DAY("2024-04-23"),
                DAY("2024-04-22"),
                DAY("2024-04-23"),
                DAY("2024-04-22"),
            ],
            "event_type": ["CASH_DIVIDEND"] * 4,
            "cash_amount": [0.5, 2.0, 1.0, 1.0],
            "payment_date": [pd.NaT] * 4,
            "retrieved_on": [DAY("2025-12-18")] * 4,
        }
    ).to_parquet(bootstrap, index=False)
    return store


@pytest.fixture
def built(store: DataStore) -> DataStore:
    build_corporate_actions(store)
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
    build_corporate_actions(built)
    pd.testing.assert_frame_equal(first, table(built, EVENTS_FILE))


def test_a_build_without_sources_lists_the_skeleton(store: DataStore) -> None:
    for path in (
        store.raw("b3", "listed_companies"),
        store.raw("b3", "earnings_credited"),
        store.raw("bootstrap"),
    ):
        for child in sorted(path.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
    summary = build_corporate_actions(store)
    assert summary.events["source"] == {"inferred": 2}
    assert summary.ex_sessions == {"inferred": 2, "unexplained": 3, "unexplained_jump": 1}
    assert summary.inputs["bootstrap"] is None
    assert table(store, CONFLICTS_FILE).empty


def test_no_bars_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SecurityMasterNotBuiltError, match="bars_daily"):
        build_corporate_actions(DataStore.initialize(tmp_path / "data"))


@pytest.mark.parametrize(
    ("value", "reference", "expected"),
    [(0.5001, 0.5, True), (0.51, 0.5, False), (math.nan, math.nan, True), (math.nan, 1.0, False)],
)
def test_amounts_match_within_a_relative_tolerance(
    value: float, reference: float, expected: bool
) -> None:
    assert _close(value, reference, 0.005) is expected
