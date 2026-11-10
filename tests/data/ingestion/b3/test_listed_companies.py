# ibexQuant\tests\data\ingestion\b3\test_listed_companies.py

"""Tests for the reader of B3's listed-company snapshots (ADR 0013, H3.4, H3.10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ibexQuant.data.ingestion.b3.listed_companies import (
    CASH_HISTORY_COLUMNS,
    STOCK_COLUMNS,
    ListedCompaniesFormatError,
    read_listed_companies,
    snapshot_folders,
)
from ibexQuant.data.store import DataStore

MGLU: list[dict[str, Any]] = [
    {
        "tradingName": "MAGAZ LUIZA ",
        "code": "MGLU",
        "cashDividends": [
            {
                "assetIssued": "BRMGLUACNOR2",
                "paymentDate": "08/05/2026",
                "rate": "0,08130019210",
                "relatedTo": "Anual/2025",
                "approvedOn": "23/04/2026",
                "isinCode": "BRMGLUACNOR2",
                "label": "DIVIDENDO",
                "lastDatePrior": "24/04/2026",
                "remarks": "",
            }
        ],
        "stockDividends": [
            {
                "assetIssued": "BRMGLUACNOR2",
                "factor": "0,10000000000",
                "approvedOn": "24/04/2024",
                "isinCode": "BRMGLUACNOR2",
                "label": "GRUPAMENTO",
                "lastDatePrior": "24/05/2024",
                "remarks": "",
            },
            {
                "assetIssued": "",
                "factor": "100,00000000000",
                "approvedOn": "06/07/2012",
                "isinCode": "BRMGLUR14OR8",
                "label": "RESG TOTAL RV",
                "lastDatePrior": "06/07/2012",
                "remarks": "",
            },
        ],
        "subscriptions": [
            {
                "assetIssued": "BRMGLUACNOR2",
                "percentage": "9,57901775290",
                "priceUnit": "1,95000000000",
                "tradingPeriod": "01/02/2024 a 27/02/2024",
                "subscriptionDate": "31/12/9999",
                "approvedOn": "26/01/2024",
                "isinCode": "BRMGLUACNOR2",
                "label": "SUBSCRICAO",
                "lastDatePrior": "31/01/2024",
                "remarks": "",
            }
        ],
    }
]


def history_row(
    value: str, last_cum: str, quoted: str = "1", close: str = "49,34"
) -> dict[str, str]:
    return {
        "typeStock": "ON",
        "dateApproval": "06/08/2026",
        "valueCash": value,
        "ratio": quoted,
        "corporateAction": "DIVIDENDO",
        "lastDatePriorEx": last_cum,
        "dateClosingPricePriorExDate": last_cum,
        "closingPricePriorExDate": close,
        "quotedPerShares": quoted,
        "corporateActionPrice": "1,0",
        "lastDateTimePriorEx": "2026-08-21T00:00:00",
    }


def page(*rows: dict[str, str]) -> dict[str, Any]:
    return {
        "page": {"pageNumber": 1, "pageSize": 120, "totalRecords": len(rows)},
        "results": list(rows),
    }


def write(folder: Path, relative: str, content: object) -> None:
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return DataStore.initialize(tmp_path / "data")


@pytest.fixture
def snapshot(store: DataStore) -> Path:
    folder = store.raw("b3", "listed_companies", "2026-11-10")
    write(folder, "supplement/MGLU.json", MGLU)
    write(folder, "supplement/VVAR.json", b"")
    write(folder, "supplement/ABNB.json", [])
    write(
        folder,
        "cash_dividends/PETR/page-001.json",
        page(
            history_row("0,35477261", "16/04/2025"),
            history_row("0,35477261", "16/04/2025"),
            history_row("1.234,5", "02/04/2001", quoted="1000", close="45.000,00"),
        ),
    )
    write(folder, "cash_dividends/KLBN/page-001.json", page())
    write(folder, "cash_dividends/ALLL/page-001.json", b"")
    return folder


def test_stock_events_keep_label_factor_and_target(snapshot: Path) -> None:
    stock = read_listed_companies([snapshot]).stock
    assert list(stock.columns) == list(STOCK_COLUMNS)
    grouping = stock[stock["label"] == "GRUPAMENTO"].iloc[0]
    assert grouping["isin"] == "BRMGLUACNOR2"
    assert grouping["target_isin"] == "BRMGLUACNOR2"
    assert grouping["factor"] == pytest.approx(0.1)
    assert grouping["last_cum_date"] == pd.Timestamp("2024-05-24")
    assert grouping["snapshot"] == pd.Timestamp("2026-11-10")
    assert stock[stock["label"] == "RESG TOTAL RV"].iloc[0]["target_isin"] == ""


def test_recent_cash_and_subscriptions_are_parsed(snapshot: Path) -> None:
    data = read_listed_companies([snapshot])
    cash = data.cash_recent.iloc[0]
    assert cash["cash_amount"] == pytest.approx(0.0813001921)
    assert cash["payment_date"] == pd.Timestamp("2026-05-08")
    subscription = data.subscriptions.iloc[0]
    assert subscription["percentage"] == pytest.approx(9.5790177529)
    assert subscription["price"] == pytest.approx(1.95)
    assert pd.isna(subscription["subscription_date"])


def test_history_values_are_per_share_and_installments_stay(snapshot: Path) -> None:
    history = read_listed_companies([snapshot]).cash_history
    assert list(history.columns) == list(CASH_HISTORY_COLUMNS)
    assert history["code"].tolist() == ["PETR"] * 3
    old = history[history["last_cum_date"] == pd.Timestamp("2001-04-02")].iloc[0]
    assert old["cash_amount"] == pytest.approx(1.2345)
    assert old["closing_price"] == pytest.approx(45.0)
    assert (history["cash_amount"].round(8) == 0.35477261).sum() == 2


def test_a_row_seen_again_in_a_later_snapshot_is_one_event(
    store: DataStore, snapshot: Path
) -> None:
    later = store.raw("b3", "listed_companies", "2026-11-17")
    write(
        later,
        "cash_dividends/PETR/page-001.json",
        page(history_row("0,35477261", "16/04/2025"), history_row("0,5", "10/11/2026")),
    )
    history = read_listed_companies(snapshot_folders(store)).cash_history
    april = history[history["last_cum_date"] == pd.Timestamp("2025-04-16")]
    assert len(april) == 2
    assert len(history) == 4


def test_snapshot_folders_ignore_other_names(store: DataStore, snapshot: Path) -> None:
    (store.raw("b3", "listed_companies") / "notes").mkdir()
    assert snapshot_folders(store) == [snapshot]


def test_no_raw_folder_lists_no_snapshots(store: DataStore) -> None:
    assert snapshot_folders(store) == []


def test_no_snapshots_give_empty_tables_with_the_columns() -> None:
    data = read_listed_companies([])
    assert data.stock.empty
    assert list(data.cash_history.columns) == list(CASH_HISTORY_COLUMNS)


def test_invalid_json_is_an_error(store: DataStore) -> None:
    folder = store.raw("b3", "listed_companies", "2026-11-10")
    write(folder, "supplement/PETR.json", b"<html>blocked</html>")
    with pytest.raises(ListedCompaniesFormatError, match="PETR.json"):
        read_listed_companies([folder])


@pytest.mark.parametrize(
    ("relative", "content"),
    [("supplement/PETR.json", {"code": "PETR"}), ("cash_dividends/PETR/page-001.json", [1])],
)
def test_unexpected_shapes_are_an_error(store: DataStore, relative: str, content: object) -> None:
    folder = store.raw("b3", "listed_companies", "2026-11-10")
    write(folder, relative, content)
    with pytest.raises(ListedCompaniesFormatError, match="expected"):
        read_listed_companies([folder])
