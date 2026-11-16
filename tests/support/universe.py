# ibexQuant\tests\support\universe.py

"""A small store for the universe: bars, master, ledger and B3 portfolio files (ADR 0015)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
from support.index_portfolios import Member, write_workbook

from ibexQuant.data.store import DataStore

DAY = pd.Timestamp
FIRST, LAST = "2024-03-01", "2025-02-28"


def span(
    security_id: str,
    ticker: str,
    first: str = FIRST,
    last: str = LAST,
    market_type: int = 10,
) -> dict[str, Any]:
    return {
        "security_id": security_id,
        "ticker": ticker,
        "isin": f"{security_id}ACNOR{0 if market_type == 10 else 9}",
        "market_type": market_type,
        "first_session": DAY(first),
        "last_session": DAY(last),
        "sessions": 1,
        "link_rule": "founding",
        "cnpj": "",
    }


MASTER: list[dict[str, Any]] = [
    span("BRPETR", "PETR4"),
    span("BRPETR", "PETR4F", market_type=20),
    span("BRVALE", "VALE3"),
    span("BROLDX", "OLDX3", last="2024-09-30"),
    span("BROLDX", "NEWX3", first="2024-10-01"),
    span("BRGONE", "GONE3", last="2024-08-20"),
    span("BRLATE", "LATE3", first="2024-09-02"),
    span("BRDUP1", "DUPL3"),
    span("BRDUP2", "DUPL3"),
    span("BRQUIT", "QUIT3", last="2024-10-15"),
    span("BRSTOP", "STOP3", last="2024-12-02"),
    span("BRPAUS", "PAUS3"),
    span("BRFUND", "FUND11"),
]

# Daily traded value (BRL millions) and COTAHIST specification by security.
TRADING: dict[str, tuple[float, str]] = {
    "BRPETR": (100.0, "PN      N2"),
    "BRVALE": (90.0, "ON      NM"),
    "BROLDX": (80.0, "ON      NM"),
    "BRGONE": (70.0, "PNA     N1"),
    "BRLATE": (60.0, "UNT     N2"),
    "BRDUP1": (50.0, "ON"),
    "BRDUP2": (40.0, "ON"),
    "BRQUIT": (30.0, "ON      NM"),
    "BRSTOP": (20.0, "ON      NM"),
    "BRPAUS": (10.0, "ON      NM"),
    "BRFUND": (500.0, "CI"),
}
SPLIT_DAY = "2024-07-01"
# Sessions a security did not trade although listed.
PAUSES: dict[str, list[str]] = {"BRPAUS": ["2024-09-02"]}


def member(ticker: str, weight: float = 1.0) -> Member:
    return (ticker, ticker[:4], "ON  NM", 1000.0, weight)


def bars() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in MASTER:
        if entry["market_type"] != 10:
            continue
        value, specification = TRADING[entry["security_id"]]
        paused = {DAY(day) for day in PAUSES.get(entry["security_id"], [])}
        for day in pd.bdate_range(entry["first_session"], entry["last_session"]):
            if day in paused:
                continue
            # PETR splits 2 for 1 on SPLIT_DAY: half the price, a new distribution number.
            after_split = entry["security_id"] == "BRPETR" and day >= DAY(SPLIT_DAY)
            price = 10.0 if after_split else 20.0
            rows.append(
                {
                    "session_date": day,
                    "ticker": entry["ticker"],
                    "isin": entry["isin"],
                    "market_type": 10,
                    "specification": specification,
                    "distribution_number": 2 if after_split else 1,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume_shares": 1000.0,
                    "traded_value_brl": value * 1e6,
                }
            )
    # An odd-lot row and a row of an ISIN the master does not know: both ignored.
    rows.append(
        {
            "session_date": DAY("2024-05-02"),
            "ticker": "PETR4F",
            "isin": "BRPETRACNOR9",
            "market_type": 20,
            "specification": "PN      N2",
            "distribution_number": 1,
            "open": 20.0,
            "high": 20.0,
            "low": 20.0,
            "close": 20.0,
            "volume_shares": 1.0,
            "traded_value_brl": 1e12,
        }
    )
    rows.append(
        {
            "session_date": DAY("2024-05-02"),
            "ticker": "ALIEN3",
            "isin": "BRALIEN",
            "market_type": 10,
            "specification": "ON",
            "distribution_number": 1,
            "open": 20.0,
            "high": 20.0,
            "low": 20.0,
            "close": 20.0,
            "volume_shares": 1.0,
            "traded_value_brl": 1e12,
        }
    )
    return pd.DataFrame(rows)


def ledger_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "security_id": ["BRQUIT", "BRQUIT", "BRPETR"],
            "ex_date": [DAY("2024-10-16"), DAY("2024-06-01"), DAY("2024-10-16")],
            "last_cum_date": [DAY("2024-10-15"), pd.NaT, DAY("2024-10-15")],
            "event_type": ["CONVERSION", "OTHER", "CASH_DIVIDEND"],
        }
    )


def write_base(store: DataStore) -> DataStore:
    """Bars, master and ledger; no portfolio file."""
    table = bars()
    for year, rows in table.groupby(table["session_date"].dt.year):
        partition = store.curated("b3", "bars_daily", f"year={year}", "part-0.parquet")
        partition.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(partition, index=False)
    pd.DataFrame(MASTER).to_parquet(store.curated("b3", "security_master.parquet"), index=False)
    ledger_events().to_parquet(store.curated("b3", "corporate_actions.parquet"), index=False)
    pd.DataFrame({"security_id": pd.Series(dtype=str)}).to_parquet(
        store.curated("b3", "ex_sessions.parquet"), index=False
    )
    return store


def workbooks(store: DataStore) -> Path:
    return store.raw("b3", "index_portfolios", "workbooks")


def add_workbook(
    store: DataStore, name: str, title: str, sheets: dict[str, list[Member]], base: str
) -> Path:
    return write_workbook(workbooks(store) / name, title, sheets, base=dt.date.fromisoformat(base))


def add_retroactive(store: DataStore, body: str) -> None:
    folder = store.raw("b3", "index_portfolios", "retroactive")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ibov_retroactive_2003_2014.csv").write_text(
        "period,ticker,name,share_class,weight_pct\n" + body, encoding="utf-8"
    )


SEPTEMBER: dict[str, list[Member]] = {
    "IBOV": [member(t) for t in ("PETR4", "VALE3", "OLDX3", "GONE3", "LATE3", "QUIT3")],
    "IBXX": [member(t) for t in ("PETR4", "DUPL3", "STOP3", "PAUS3")],
}


def write_portfolios(store: DataStore) -> DataStore:
    """A 2nd preview, 3rd previews (one twice), a definitive portfolio, the retroactive CSV."""
    third_september = "3ª Prévia do IBOVESPA para setembro de 2024"
    add_workbook(store, "a_2024Q3.xlsx", third_september, SEPTEMBER, "2024-08-28")
    add_workbook(store, "b_2024Q3_copy.xlsx", third_september, SEPTEMBER, "2024-08-28")
    add_workbook(
        store,
        "jan_3rd.xlsx",
        "3ª Prévia do IBOVESPA para janeiro de 2025",
        {"IBOV": [member("PETR4"), member("NEWX3")]},
        "2024-12-27",
    )
    add_workbook(
        store,
        "jan_definitive.xlsx",
        "Virada do IBOVESPA para janeiro de 2025",
        {"IBOV": [member("PETR4"), member("NEWX3"), member("VALE3")]},
        "2025-01-03",
    )
    add_workbook(
        store,
        "may_2nd.xlsx",
        "2ª Prévia do IBOVESPA para maio de 2024",
        {"IBOV": [member("PETR4", 100.0)]},
        "2024-04-16",
    )
    add_retroactive(store, "2003-01,PETR4,PETROBRAS,PN,100\n2024-05,VALE3,VALE,ON,100\n")
    return store
