# ibexQuant\tests\data\reference\test_ex_sessions.py

"""Tests for the ex-session skeleton and the matching of events to it (ADR 0013)."""

from __future__ import annotations

import pandas as pd
import pytest
from support.reference import isin

from ibexQuant.data.reference.ex_sessions import (
    EX_SESSION_COLUMNS,
    MatchOutcome,
    distribution_runs,
    find_ex_sessions,
    match_by_date,
    match_by_distribution,
    match_by_ratio,
)

PETR = isin("PETR")
ODER = isin("ODER", "ACNPR")
UNIT = isin("SANB", "CDAM1")
FUND = isin("HGLG", "CTF00")


def bar(
    session: str,
    code: str,
    distribution: int,
    *,
    ticker: str = "PETR3",
    open_: float = 10.0,
    close: float = 10.0,
    market_type: int = 10,
    volume: int = 100,
) -> dict[str, object]:
    return {
        "session_date": pd.Timestamp(session),
        "ticker": ticker,
        "isin": code,
        "market_type": market_type,
        "distribution_number": distribution,
        "open": open_,
        "close": close,
        "volume_shares": volume,
    }


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --- Runs and ex-sessions ---


def test_runs_group_consecutive_sessions_with_one_distribution_number() -> None:
    bars = frame(
        bar("2024-01-02", PETR, 100),
        bar("2024-01-03", PETR, 100),
        bar("2024-01-04", PETR, 101),
        bar("2024-01-05", PETR, 101),
    )
    runs = distribution_runs(bars)
    assert runs[["distribution_number", "first_session", "last_session"]].to_dict("records") == [
        {
            "distribution_number": 100,
            "first_session": pd.Timestamp("2024-01-02"),
            "last_session": pd.Timestamp("2024-01-03"),
        },
        {
            "distribution_number": 101,
            "first_session": pd.Timestamp("2024-01-04"),
            "last_session": pd.Timestamp("2024-01-05"),
        },
    ]


def test_ex_session_carries_the_prices_around_the_change() -> None:
    bars = frame(
        bar("2024-04-25", PETR, 162, close=103.0),
        bar("2024-04-28", PETR, 163, open_=51.89, close=52.0),
    )
    ex = find_ex_sessions(bars)
    assert list(ex.columns) == list(EX_SESSION_COLUMNS)
    row = ex.iloc[0]
    assert row["session_date"] == pd.Timestamp("2024-04-28")
    assert row["previous_session"] == pd.Timestamp("2024-04-25")
    assert (row["from_distribution"], row["to_distribution"]) == (162, 163)
    assert row["previous_close"] == 103.0
    assert row["open_ratio"] == pytest.approx(51.89 / 103.0)


def test_a_skipped_number_is_one_change() -> None:
    bars = frame(bar("2026-01-02", ODER, 140, ticker="ODER4"), bar("2026-03-02", ODER, 146))
    ex = find_ex_sessions(bars)
    assert len(ex) == 1
    assert (ex.iloc[0]["from_distribution"], ex.iloc[0]["to_distribution"]) == (140, 146)


def test_only_round_lot_shares_and_units_are_kept() -> None:
    bars = frame(
        bar("2024-01-02", PETR, 100, market_type=20),
        bar("2024-01-03", PETR, 101, market_type=20),
        bar("2024-01-02", FUND, 100, ticker="HGLG11"),
        bar("2024-01-03", FUND, 101, ticker="HGLG11"),
        bar("2024-01-02", UNIT, 100, ticker="SANB11"),
        bar("2024-01-03", UNIT, 101, ticker="SANB11"),
        bar("2024-01-02", "ACESACON", 100),
        bar("2024-01-03", "ACESACON", 101),
    )
    assert find_ex_sessions(bars)["isin"].tolist() == [UNIT]


def test_two_tickers_of_one_isin_on_a_session_keep_the_larger_volume() -> None:
    bars = frame(
        bar("2024-01-02", PETR, 100, ticker="OLD3", close=9.0, volume=10),
        bar("2024-01-02", PETR, 100, ticker="NEW3", close=10.0, volume=500),
        bar("2024-01-03", PETR, 101, ticker="NEW3", open_=5.0),
    )
    ex = find_ex_sessions(bars)
    assert ex.iloc[0]["previous_close"] == 10.0
    assert ex.iloc[0]["ticker"] == "NEW3"


def test_no_bars_give_an_empty_table_with_the_columns() -> None:
    ex = find_ex_sessions(frame(bar("2024-01-02", PETR, 100)).iloc[:0])
    assert ex.empty
    assert list(ex.columns) == list(EX_SESSION_COLUMNS)


# --- Matching by data-com ---


@pytest.fixture
def runs() -> pd.DataFrame:
    return distribution_runs(
        frame(
            bar("2024-01-02", PETR, 100),
            bar("2024-01-03", PETR, 100),
            bar("2024-01-05", PETR, 101),
            bar("2024-01-08", PETR, 101),
        )
    )


def dates(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values)))


def test_data_com_on_the_last_cum_session_matches_the_change(runs: pd.DataFrame) -> None:
    result = match_by_date(pd.Series([PETR]), dates("2024-01-03"), runs)
    assert result["outcome"].tolist() == [MatchOutcome.MATCHED]
    assert result["ex_session"].tolist() == [pd.Timestamp("2024-01-05")]


def test_data_com_on_a_day_without_trades_matches_the_next_change(runs: pd.DataFrame) -> None:
    result = match_by_date(pd.Series([PETR]), dates("2024-01-04"), runs)
    assert result["outcome"].tolist() == [MatchOutcome.MATCHED]
    assert result["ex_session"].tolist() == [pd.Timestamp("2024-01-05")]


def test_trading_after_the_data_com_under_the_same_number_contradicts(
    runs: pd.DataFrame,
) -> None:
    result = match_by_date(pd.Series([PETR]), dates("2024-01-02"), runs)
    assert result["outcome"].tolist() == [MatchOutcome.NO_EX_SESSION]
    assert result["ex_session"].isna().all()


def test_events_outside_the_trading_history_are_untraded(runs: pd.DataFrame) -> None:
    result = match_by_date(
        pd.Series([PETR, PETR, isin("VALE")]),
        dates("2023-12-29", "2024-01-08", "2024-01-03"),
        runs,
    )
    assert result["outcome"].tolist() == [MatchOutcome.NOT_TRADED] * 3


# --- Matching by distribution number ---


def test_distribution_number_matches_the_change_that_leaves_it() -> None:
    runs = distribution_runs(
        frame(
            bar("2026-01-02", ODER, 140),
            bar("2026-03-02", ODER, 146),
            bar("2026-04-01", ODER, 147),
        )
    )
    result = match_by_distribution(
        pd.Series([ODER, ODER, ODER, ODER]), pd.Series([140, 143, 146, 147]), runs
    )
    assert result["outcome"].tolist() == [
        MatchOutcome.MATCHED,
        MatchOutcome.MATCHED,
        MatchOutcome.MATCHED,
        MatchOutcome.NOT_TRADED,
    ]
    assert result["ex_session"].tolist()[:3] == [
        pd.Timestamp("2026-03-02"),
        pd.Timestamp("2026-03-02"),
        pd.Timestamp("2026-04-01"),
    ]


def test_a_number_before_the_first_trade_is_untraded() -> None:
    runs = distribution_runs(frame(bar("2026-01-02", ODER, 140), bar("2026-03-02", ODER, 146)))
    result = match_by_distribution(pd.Series([ODER]), pd.Series([120]), runs)
    assert result["outcome"].tolist() == [MatchOutcome.NOT_TRADED]


def test_matching_against_no_runs_leaves_every_event_untraded() -> None:
    runs = distribution_runs(frame(bar("2024-01-02", PETR, 100)).iloc[:0])
    by_date = match_by_date(pd.Series([PETR]), dates("2024-01-03"), runs)
    by_number = match_by_distribution(pd.Series([PETR]), pd.Series([100]), runs)
    assert by_date["outcome"].tolist() == [MatchOutcome.NOT_TRADED]
    assert by_number["outcome"].tolist() == [MatchOutcome.NOT_TRADED]


def test_match_results_keep_the_index_of_the_events(runs: pd.DataFrame) -> None:
    isins = pd.Series([PETR, PETR], index=[7, 3])
    result = match_by_date(
        isins, pd.Series(pd.to_datetime(["2024-01-03", "2024-01-02"]), index=[7, 3]), runs
    )
    assert result.index.tolist() == [7, 3]
    assert result["outcome"].tolist() == [MatchOutcome.MATCHED, MatchOutcome.NO_EX_SESSION]


# --- Matching by ratio ---


def test_ratio_events_match_the_first_agreeing_ex_session_in_the_window() -> None:
    ex = find_ex_sessions(
        frame(
            bar("2016-03-01", PETR, 100, close=10.0),
            bar("2016-03-20", PETR, 101, open_=9.8, close=10.0),  # dividend, ratio 0.98
            bar("2016-04-18", PETR, 102, open_=9.05, close=9.1),  # 10% bonus, 0.905
            bar("2016-12-01", PETR, 103, open_=5.0, close=5.0),  # outside the window
        )
    )
    result = match_by_ratio(
        pd.Series([PETR, PETR, PETR, ODER]),
        dates("2016-03-10", "2016-03-10", "2016-05-01", "2016-03-10"),
        pd.Series([1.1, 3.0, 1.1, 1.1]),
        ex,
        window_days=180,
        tolerance=0.1,
    )
    assert result["outcome"].tolist() == [
        MatchOutcome.MATCHED,
        MatchOutcome.NOT_TRADED,
        MatchOutcome.NOT_TRADED,
        MatchOutcome.NOT_TRADED,
    ]
    assert result["ex_session"].iloc[0] == pd.Timestamp("2016-04-18")
