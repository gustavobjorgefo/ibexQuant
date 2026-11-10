# ibexQuant\tests\data\reference\test_corporate_action_sources.py

"""Tests for the source events of the corporate actions ledger (ADR 0013, H3.4, H3.13)."""

from __future__ import annotations

import pandas as pd
import pytest

from ibexQuant.data.ingestion.b3.listed_companies import (
    CASH_HISTORY_COLUMNS,
    CASH_RECENT_COLUMNS,
    STOCK_COLUMNS,
    SUBSCRIPTION_COLUMNS,
    ListedCompanies,
)
from ibexQuant.data.reference.corporate_action_sources import (
    SOURCE_EVENT_COLUMNS,
    BootstrapFormatError,
    IsinLocator,
    bootstrap_events,
    credited_events,
    listed_events,
)

VVAR, VIIA, BHIA = "BRVVARACNOR1", "BRVIIAACNOR7", "BRBHIAACNOR1"
PETR3, PETR4, SANB11 = "BRPETRACNOR9", "BRPETRACNPR6", "BRSANBCDAM13"
DAY = pd.Timestamp


def span(security: str, ticker: str, isin: str, first: str, last: str) -> dict[str, object]:
    return {
        "security_id": security,
        "ticker": ticker,
        "isin": isin,
        "market_type": 10,
        "first_session": DAY(first),
        "last_session": DAY(last),
        "sessions": 100,
        "link_rule": "founding",
        "cnpj": "",
    }


@pytest.fixture
def locator() -> IsinLocator:
    master = pd.DataFrame(
        [
            span(VVAR, "VVAR3", VVAR, "2015-01-02", "2021-08-13"),
            span(VVAR, "VIIA3", VIIA, "2021-08-16", "2023-09-19"),
            span(VVAR, "BHIA3", BHIA, "2023-09-20", "2026-11-06"),
            span(PETR3, "PETR3", PETR3, "2000-01-03", "2026-11-06"),
            span(PETR4, "PETR4", PETR4, "2000-01-03", "2026-11-06"),
            span(SANB11, "SANB11", SANB11, "2009-10-07", "2026-11-06"),
            {**span(PETR4, "PETR4F", PETR4, "2000-01-03", "2026-11-06"), "market_type": 20},
        ]
    )
    return IsinLocator(master)


def table(columns: tuple[str, ...], rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(columns))
    frame["snapshot"] = DAY("2026-11-10")
    return frame


def listed(
    stock: list[dict[str, object]] | None = None,
    recent: list[dict[str, object]] | None = None,
    history: list[dict[str, object]] | None = None,
    subscriptions: list[dict[str, object]] | None = None,
) -> ListedCompanies:
    return ListedCompanies(
        stock=table(STOCK_COLUMNS, stock or []),
        cash_recent=table(CASH_RECENT_COLUMNS, recent or []),
        cash_history=table(CASH_HISTORY_COLUMNS, history or []),
        subscriptions=table(SUBSCRIPTION_COLUMNS, subscriptions or []),
    )


def history_row(code: str, share_class: str, day: str, amount: float) -> dict[str, object]:
    return {
        "code": code,
        "share_class": share_class,
        "label": "DIVIDENDO",
        "cash_amount": amount,
        "last_cum_date": DAY(day),
        "approved_on": DAY(day),
        "closing_price": 10.0,
    }


# --- Locator ---


def test_locator_finds_the_isin_of_a_renamed_security_on_a_date(locator: IsinLocator) -> None:
    securities = locator.securities_with_prefix("BRBHIAACNOR")
    assert securities == {VVAR}
    assert locator.isin_at(securities, DAY("2019-05-02")) == (VVAR, "")
    assert locator.isin_at(securities, DAY("2022-01-03")) == (VIIA, "")
    assert locator.isin_at(securities, DAY("2010-01-04")) == (None, "no ISIN traded on the date")


def test_locator_reports_several_isins(locator: IsinLocator) -> None:
    securities = locator.securities_with_prefix("BRPETRACN")
    assert locator.isin_at(securities, DAY("2020-01-02")) == (None, "several ISINs")


def test_locator_knows_round_lot_tickers_only(locator: IsinLocator) -> None:
    assert locator.securities_with_ticker("BHIA3") == {VVAR}
    assert locator.securities_with_ticker("PETR4F") == set()


# --- B3 listed companies ---


def test_cash_history_is_placed_by_class_and_date(locator: IsinLocator) -> None:
    events, unresolved = listed_events(
        listed(
            history=[
                history_row("BHIA", "ON", "2019-05-02", 0.1),
                history_row("PETR", "PN", "2020-04-01", 0.2),
                history_row("SANB", "UNT", "2020-04-01", 0.3),
                history_row("BHIA", "ON", "2010-01-04", 0.4),
                history_row("PETR", "PNX", "2020-04-01", 0.5),
            ]
        ),
        locator,
    )
    assert list(events.columns) == list(SOURCE_EVENT_COLUMNS)
    assert events[["isin", "cash_amount"]].values.tolist() == [
        [VVAR, 0.1],
        [PETR4, 0.2],
        [SANB11, 0.3],
    ]
    assert set(events["event_type"]) == {"CASH_DIVIDEND"}
    assert set(events["source"]) == {"b3_listed"}
    assert unresolved["reason"].tolist() == [
        "no ISIN traded on the date",
        "no ISIN for class PNX",
    ]


def test_recent_cash_repeating_the_history_is_one_event_and_installments_stay(
    locator: IsinLocator,
) -> None:
    recent = {
        "code": "PETR",
        "isin": PETR3,
        "label": "DIVIDENDO",
        "cash_amount": 0.35477261,
        "last_cum_date": DAY("2025-04-16"),
        "approved_on": DAY("2025-02-26"),
        "payment_date": DAY("2025-05-20"),
        "related_to": "Anual/2024",
    }
    installment = history_row("PETR", "ON", "2025-04-16", 0.35477261)
    events, _ = listed_events(listed(recent=[recent], history=[installment, installment]), locator)
    assert len(events) == 2
    assert events["origin"].tolist() == ["cash_recent:PETR", "cash_history:PETR"]
    assert events.iloc[0]["payment_date"] == DAY("2025-05-20")


def test_stock_events_become_types_with_ratios_and_targets(locator: IsinLocator) -> None:
    def stock(label: str, factor: float, target: str) -> dict[str, object]:
        return {
            "code": "MGLU",
            "isin": "BRMGLUACNOR2",
            "target_isin": target,
            "label": label,
            "factor": factor,
            "last_cum_date": DAY("2024-05-24"),
            "approved_on": DAY("2024-04-24"),
        }

    events, _ = listed_events(
        listed(
            stock=[
                stock("GRUPAMENTO", 0.1, "BRMGLUACNOR2"),
                stock("CIS RED CAP", 100.0, "BRMGLUACNOR2"),
                stock("CIS RED CAP", 7.215, "BRLOGGACNOR7"),
            ]
        ),
        locator,
    )
    assert events[["event_type", "target_isin"]].values.tolist() == [
        ["REVERSE_SPLIT", ""],
        ["SPIN_OFF", "BRLOGGACNOR7"],
    ]
    assert events["share_ratio"].tolist() == pytest.approx([0.1, 0.07215])


def test_subscriptions_carry_rights_per_share_and_price(locator: IsinLocator) -> None:
    subscription = {
        "code": "MGLU",
        "isin": "BRMGLUACNOR2",
        "target_isin": "BRMGLUACNOR2",
        "label": "SUBSCRICAO",
        "percentage": 9.579,
        "price": 1.95,
        "last_cum_date": DAY("2024-01-31"),
        "approved_on": DAY("2024-01-26"),
        "subscription_date": pd.NaT,
    }
    events, _ = listed_events(listed(subscriptions=[subscription]), locator)
    row = events.iloc[0]
    assert row["event_type"] == "SUBSCRIPTION"
    assert row["share_ratio"] == pytest.approx(0.09579)
    assert row["subscription_price"] == 1.95


def test_events_of_funds_are_left_out(locator: IsinLocator) -> None:
    fund = {
        "code": "HGLG",
        "isin": "BRHGLGCTF004",
        "label": "RENDIMENTO",
        "cash_amount": 1.1,
        "last_cum_date": DAY("2026-09-30"),
        "approved_on": pd.NaT,
        "payment_date": pd.NaT,
        "related_to": "",
    }
    events, _ = listed_events(listed(recent=[fund]), locator)
    assert events.empty


# --- Earnings credited ---


def test_credited_events_keep_the_distribution_number() -> None:
    credited = pd.DataFrame(
        {
            "credit_date": [DAY("2026-11-09")] * 3,
            "isin": ["BRSANBACNPR5", "BRENJUACNOR9", "BRHGLGCTF004"],
            "distribution_number": [181, 102, 300],
            "label": ["JUROS SOBRE CAPITAL PRÓPRIO", "LEILÃO DE FRAÇÕES", "RENDIMENTO"],
            "value_brl": [0.21, 7.28, 1.1],
            "approved_on": [DAY("2026-10-09"), pd.NaT, pd.NaT],
            "issuer": ["SANTANDER", "ENJOEI", "PATRIA LOG"],
            "source_file": ["2026-11-09.csv"] * 3,
        }
    )
    events = credited_events(credited)
    assert events[["isin", "event_type", "source"]].values.tolist() == [
        ["BRSANBACNPR5", "INTEREST_ON_CAPITAL", "b3_credited"]
    ]
    assert events.iloc[0]["distribution_number"] == 181
    assert pd.isna(events.iloc[0]["last_cum_date"])


def test_no_credited_rows_give_no_events() -> None:
    assert credited_events(pd.DataFrame()).empty


# --- Bootstrap ---


def bootstrap(**overrides: object) -> pd.DataFrame:
    row = {
        "ticker": "BHIA3",
        "last_cum_date": DAY("2019-05-02"),
        "event_type": "INTEREST_ON_CAPITAL",
        "cash_amount": 0.05,
        "payment_date": DAY("2019-06-01"),
        "retrieved_on": DAY("2025-12-18"),
    }
    return pd.DataFrame([row | overrides])


def test_bootstrap_tickers_are_placed_through_their_security(locator: IsinLocator) -> None:
    events, unresolved = bootstrap_events(bootstrap(), locator)
    assert events[["isin", "source", "origin"]].values.tolist() == [
        [VVAR, "bootstrap", "bootstrap:BHIA3"]
    ]
    assert unresolved.empty


def test_unknown_bootstrap_tickers_are_unresolved(locator: IsinLocator) -> None:
    events, unresolved = bootstrap_events(bootstrap(ticker="XPTO3"), locator)
    assert events.empty
    assert unresolved["key"].tolist() == ["XPTO3"]


def test_bootstrap_without_a_column_is_an_error(locator: IsinLocator) -> None:
    with pytest.raises(BootstrapFormatError, match="retrieved_on"):
        bootstrap_events(bootstrap().drop(columns="retrieved_on"), locator)


def test_bootstrap_events_must_be_cash(locator: IsinLocator) -> None:
    with pytest.raises(BootstrapFormatError, match="SPLIT"):
        bootstrap_events(bootstrap(event_type="SPLIT"), locator)
