# ibexQuant\tests\support\ledger.py

"""A small market with every source of the corporate actions ledger (ADR 0013)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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


def ledger_store(tmp_path: Path) -> DataStore:
    """A data store with bars, a master and every ledger source for PETR3 and VALE3."""
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
