# ibexQuant\tests\data\universe\test_constituents.py

"""Index compositions by period, resolved to securities (ADR 0015, H5.1–H5.5)."""

from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from support.universe import (
    DAY,
    LAST,
    add_retroactive,
    add_workbook,
    member,
    workbooks,
    write_base,
    write_portfolios,
)

from ibexQuant.cli import EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.reference import SecurityMasterError
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, OVERRIDES_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module
from ibexQuant.data.universe import (
    LiquiditySettings,
    PeriodOverride,
    PeriodOverrides,
    UniverseBuildError,
    UniverseNotBuiltError,
    build_universe,
    read_index_constituents,
    read_index_exits,
    read_index_periods,
)
from ibexQuant.data.universe.constituents import CONSTITUENT_COLUMNS, SUMMARY_FILE
from ibexQuant.data.universe.exits import EXIT_COLUMNS


@pytest.fixture
def empty_store(tmp_path: Path) -> DataStore:
    """Bars, master and ledger, no portfolio file yet."""
    return write_base(DataStore.initialize(tmp_path / "data"))


@pytest.fixture
def store(empty_store: DataStore) -> DataStore:
    """Every source: a 2nd preview, 3rd previews, a definitive portfolio, the retroactive CSV."""
    return write_portfolios(empty_store)


SHORT = LiquiditySettings(lookback_months=1)
"""One month of lookback: the fixture's data starts in March 2024."""


def members_of(table: pd.DataFrame, index: str, period: str) -> list[str]:
    rows = table[(table["index"] == index) & (table["period"] == period)]
    return rows["ticker"].tolist()


# --- build ---


def test_the_periods_follow_the_calendar(store: DataStore) -> None:
    build_universe(store, PeriodOverrides.none())
    periods = read_index_periods(store)
    assert periods["period"].tolist() == ["2024-05", "2024-09", "2025-01"]
    assert periods["effective_from"].tolist() == [
        DAY("2024-05-06"),
        DAY("2024-09-02"),
        DAY("2025-01-06"),
    ]
    assert periods["effective_to"].iloc[-1] == DAY(LAST)


def test_overrides_move_a_start(store: DataStore) -> None:
    moved = PeriodOverrides((PeriodOverride("2024-09", DAY("2024-09-09"), "test"),), "hash")
    summary = build_universe(store, moved)
    periods = read_index_periods(store).set_index("period")
    assert periods.loc["2024-09", "effective_from"] == DAY("2024-09-09")
    assert periods.loc["2024-05", "effective_to"] == DAY("2024-09-06")
    assert summary.inputs["index_periods.yaml"] == "hash"


def test_an_override_off_the_calendar_stops_the_build(store: DataStore) -> None:
    moved = PeriodOverrides((PeriodOverride("2024-09", DAY("2024-09-07"), "test"),), "hash")
    with pytest.raises(UniverseBuildError, match="is no session"):
        build_universe(store, moved)


def test_the_best_source_is_kept_per_index_and_period(store: DataStore) -> None:
    summary = build_universe(store, PeriodOverrides.none())
    table = read_index_constituents(store)
    assert tuple(table.columns) == CONSTITUENT_COLUMNS
    sources = table.groupby(["index", "period"])["source"].first().to_dict()
    assert sources == {
        ("IBOV", "2003-01"): "retroactive",
        ("IBOV", "2024-05"): "second_preview",
        ("IBOV", "2024-09"): "third_preview",
        ("IBOV", "2025-01"): "definitive",
        ("IBXX", "2024-09"): "third_preview",
    }
    assert members_of(table, "IBOV", "2025-01") == ["NEWX3", "PETR4", "VALE3"]
    assert summary.superseded == {"retroactive": 1, "third_preview": 1}
    assert summary.compositions == {
        "definitive": 1,
        "retroactive": 1,
        "second_preview": 1,
        "third_preview": 2,
    }


def test_identical_files_count_once_and_the_first_name_is_recorded(store: DataStore) -> None:
    build_universe(store, PeriodOverrides.none())
    table = read_index_constituents(store)
    september = table[table["period"] == "2024-09"]
    assert september["source_file"].unique().tolist() == ["a_2024Q3.xlsx"]
    assert len(september) == 10


def test_files_of_one_kind_that_disagree_stop_the_build(store: DataStore) -> None:
    add_workbook(
        store,
        "c.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        {"IBOV": [member("PETR4")]},
        "2024-08-28",
    )
    with pytest.raises(UniverseBuildError, match="list different members"):
        build_universe(store, PeriodOverrides.none())


def test_every_unreadable_file_is_listed_before_the_build_stops(store: DataStore) -> None:
    add_workbook(store, "bad1.xlsx", "Carteira teórica", {"IBOV": [member("PETR4")]}, "2024-08-28")
    add_workbook(
        store, "bad2.xlsx", "3ª Prévia para junho", {"IBOV": [member("PETR4")]}, "2024-08-28"
    )
    with pytest.raises(UniverseBuildError) as caught:
        build_universe(store, PeriodOverrides.none())
    lines = str(caught.value).splitlines()
    assert lines[0].startswith("bad1.xlsx: cannot tell the kind")
    assert lines[1].startswith("bad2.xlsx: cannot tell the period")


def test_an_unreadable_retroactive_csv_stops_the_build(store: DataStore) -> None:
    add_retroactive(store, "2003-02,PETR4,PETROBRAS,PN,100\n")
    with pytest.raises(UniverseBuildError, match="malformed period"):
        build_universe(store, PeriodOverrides.none())


# --- resolution ---


def test_tickers_resolve_on_the_base_session_or_the_period_start(store: DataStore) -> None:
    build_universe(store, PeriodOverrides.none())
    table = read_index_constituents(store).set_index(["index", "period", "ticker"])
    ids = table["security_id"]
    assert ids[("IBOV", "2024-09", "OLDX3")] == "BROLDX"
    assert ids[("IBOV", "2025-01", "NEWX3")] == "BROLDX"
    # LATE3 starts trading on the period's first session, after the base session.
    assert ids[("IBOV", "2024-09", "LATE3")] == "BRLATE"
    assert ids[("IBOV", "2024-05", "PETR4")] == "BRPETR"


def test_retroactive_rows_resolve_on_the_period_start(store: DataStore) -> None:
    (workbooks(store) / "may_2nd.xlsx").unlink()
    build_universe(store, PeriodOverrides.none())
    table = read_index_constituents(store)
    may = table[table["period"] == "2024-05"]
    assert may[["ticker", "security_id", "source"]].values.tolist() == [
        ["VALE3", "BRVALE", "retroactive"]
    ]
    assert may["base_session"].isna().all()


def test_unresolved_tickers_stay_listed(store: DataStore) -> None:
    summary = build_universe(store, PeriodOverrides.none())
    table = read_index_constituents(store)
    unresolved = table[table["security_id"].isna()]
    # GONE3 left before the base session; DUPL3 is carried by two securities;
    # the 2003 period is outside the calendar.
    assert sorted(unresolved["ticker"]) == ["DUPL3", "GONE3", "PETR4"]
    assert summary.unresolved == {"IBOV": 2, "IBXX": 1}
    assert summary.unresolved_strict == [
        {"index": "IBOV", "period": "2003-01", "ticker": "PETR4"},
        {"index": "IBOV", "period": "2024-09", "ticker": "GONE3"},
        {"index": "IBXX", "period": "2024-09", "ticker": "DUPL3"},
    ]
    assert summary.outside_calendar == ["2003-01"]


# --- liquidity rule ---


def without_may(store: DataStore) -> DataStore:
    (workbooks(store) / "may_2nd.xlsx").unlink()
    shutil.rmtree(store.raw("b3", "index_portfolios", "retroactive"))
    return store


def test_an_ibov_period_without_portfolio_takes_the_liquidity_rule(store: DataStore) -> None:
    summary = build_universe(without_may(store), PeriodOverrides.none(), SHORT)
    table = read_index_constituents(store)
    may = table[(table["index"] == "IBOV") & (table["period"] == "2024-05")]
    # As many as the nearest real portfolio (2024-09, six members); the fund is
    # not eligible, the odd lot and the unknown ISIN are ignored.
    assert sorted(may["security_id"]) == [
        "BRDUP1",
        "BRDUP2",
        "BRGONE",
        "BROLDX",
        "BRPETR",
        "BRVALE",
    ]
    assert set(may["source"]) == {"liquidity_rule"}
    assert set(may["base_session"]) == {DAY("2024-05-01")}
    assert sorted(may["ticker"]) == ["DUPL3", "DUPL3", "GONE3", "OLDX3", "PETR4", "VALE3"]
    assert may["source_file"].tolist() == [""] * 6
    assert summary.liquidity_periods == ["2024-05"]
    assert summary.ibov_gaps == []
    assert summary.compositions["liquidity_rule"] == 1


def test_without_enough_history_the_period_stays_empty(store: DataStore) -> None:
    summary = build_universe(without_may(store), PeriodOverrides.none())
    assert summary.ibov_gaps == ["2024-05"]
    assert summary.liquidity_periods == []
    assert summary.liquidity_overlap == {}


def test_the_rule_is_compared_with_the_real_index(store: DataStore) -> None:
    summary = build_universe(store, PeriodOverrides.none(), SHORT)
    # 2024-09: the rule takes PETR, VALE, OLDX, DUP1, DUP2, QUIT; the index has
    # PETR, VALE, OLDX, GONE, LATE, QUIT: four of six.
    assert summary.liquidity_overlap == {"2024-05": 1.0, "2024-09": 0.6667, "2025-01": 1.0}


# --- exits ---


def test_members_that_stop_trading_inside_the_period_are_exits(store: DataStore) -> None:
    summary = build_universe(store, PeriodOverrides.none())
    exits = read_index_exits(store)
    assert tuple(exits.columns) == EXIT_COLUMNS
    assert exits[["index", "period", "ticker", "reason"]].values.tolist() == [
        ["IBOV", "2024-09", "QUIT3", "conversion"],
        ["IBXX", "2024-09", "STOP3", "no_event"],
    ]
    assert exits["event_id"].iloc[0] == "E1"
    assert pd.isna(exits["event_id"].iloc[1])
    assert exits["last_session"].tolist() == [DAY("2024-10-15"), DAY("2024-12-02")]
    assert exits["effective_to"].tolist() == [DAY("2025-01-03")] * 2
    assert summary.exits == {"conversion": 1, "no_event": 1}


def test_members_absent_on_the_first_session_are_counted(store: DataStore) -> None:
    summary = build_universe(store, PeriodOverrides.none())
    assert summary.absent_at_start == {"IBXX": 1}
    assert summary.absent_at_start_strict == [
        {"index": "IBXX", "period": "2024-09", "ticker": "PAUS3"}
    ]


# --- summary and storage ---


def test_the_summary_is_written_with_the_inputs(store: DataStore) -> None:
    summary = build_universe(store, PeriodOverrides.none())
    written = json.loads(store.curated("b3", SUMMARY_FILE).read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert written["rows"] == summary.rows == 15
    assert (summary.periods, summary.first_period, summary.last_period) == (
        3,
        "2024-05",
        "2025-01",
    )
    assert summary.indices == 2
    assert summary.ibov_gaps == []
    assert sorted(summary.inputs) == [
        "index_periods.yaml",
        "retroactive/ibov_retroactive_2003_2014.csv",
        "workbooks/a_2024Q3.xlsx",
        "workbooks/b_2024Q3_copy.xlsx",
        "workbooks/jan_3rd.xlsx",
        "workbooks/jan_definitive.xlsx",
        "workbooks/may_2nd.xlsx",
    ]
    assert summary.inputs["index_periods.yaml"] is None


def test_periods_without_ibov_are_reported(store: DataStore) -> None:
    (workbooks(store) / "may_2nd.xlsx").unlink()
    shutil.rmtree(store.raw("b3", "index_portfolios", "retroactive"))
    summary = build_universe(store, PeriodOverrides.none())
    assert summary.ibov_gaps == ["2024-05"]


def test_without_portfolio_files_the_tables_are_empty(empty_store: DataStore) -> None:
    summary = build_universe(empty_store, PeriodOverrides.none())
    table = read_index_constituents(empty_store)
    assert table.empty
    assert tuple(table.columns) == CONSTITUENT_COLUMNS
    assert summary.ibov_gaps == ["2024-05", "2024-09", "2025-01"]
    assert summary.compositions == {}


def test_without_bars_the_build_stops(empty_store: DataStore) -> None:
    shutil.rmtree(empty_store.curated("b3", "bars_daily"))
    with pytest.raises(UniverseNotBuiltError, match="ingest cotahist"):
        build_universe(empty_store, PeriodOverrides.none())


def test_without_ledger_the_build_stops(empty_store: DataStore) -> None:
    empty_store.curated("b3", "corporate_actions.parquet").unlink()
    with pytest.raises(UniverseNotBuiltError, match="build corporate-actions"):
        build_universe(empty_store, PeriodOverrides.none())


def test_without_master_the_build_stops(empty_store: DataStore) -> None:
    empty_store.curated("b3", "security_master.parquet").unlink()
    with pytest.raises(SecurityMasterError):
        build_universe(empty_store, PeriodOverrides.none())


def test_reading_before_a_build_says_how_to_build(empty_store: DataStore) -> None:
    with pytest.raises(UniverseNotBuiltError, match="build universe"):
        read_index_constituents(empty_store)
    with pytest.raises(UniverseNotBuiltError, match="build universe"):
        read_index_periods(empty_store)


# --- command line ---


@pytest.fixture
def cli(store: DataStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataStore:
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    monkeypatch.delenv(OVERRIDES_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    return store


def test_build_universe_prints_the_summary(
    cli: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "build", "universe", "--no-overrides"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "periods: 3 (2024-05 to 2025-01)" in out
    assert (
        "compositions by source: definitive 1, retroactive 1, second_preview 1, third_preview 2"
        in out
    )
    assert "set aside for a better source: retroactive 1, third_preview 1" in out
    assert "unresolved tickers by index: IBOV 2, IBXX 1" in out
    assert "  IBOV 2024-09 GONE3" in out
    assert "IBOV periods from the liquidity rule: 0" in out
    assert "IBOV periods without composition: 0\n" in out
    assert "exits inside a period: conversion 1, no_event 1" in out
    assert "members absent on the first session: IBXX 1" in out
    assert "  IBXX 2024-09 PAUS3" in out
    assert "liquidity rule vs" not in out
    assert "compositions outside the calendar: 2003-01" in out
    assert "built without index_periods.yaml" in out


def test_build_universe_reads_index_periods_yaml(
    cli: DataStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "overrides"
    root.mkdir()
    (root / "index_periods.yaml").write_text(
        "version: 1\nperiods:\n  - {period: 2024-09, start: 2024-09-09, note: test}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(OVERRIDES_ROOT_ENV_VAR, str(root))
    assert main(["data", "build", "universe"]) == EXIT_OK
    assert "built without" not in capsys.readouterr().out
    assert read_index_periods(cli).set_index("period").loc["2024-09", "effective_from"] == DAY(
        "2024-09-09"
    )


def test_build_universe_prints_gaps_and_the_overlap(
    cli: DataStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import ibexQuant.data.universe as universe

    (workbooks(cli) / "may_2nd.xlsx").unlink()
    shutil.rmtree(cli.raw("b3", "index_portfolios", "retroactive"))
    # The fixture holds one year of bars: the CLI's twelve-month rule would see none.
    monkeypatch.setattr(universe, "build_universe", partial(build_universe, liquidity=SHORT))
    assert main(["data", "build", "universe", "--no-overrides"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "IBOV periods from the liquidity rule: 1" in out
    assert "liquidity rule vs real IBOV: mean overlap 83.3%, lowest 66.7% over 2 portfolios" in out


def test_build_universe_reports_errors(cli: DataStore, capsys: pytest.CaptureFixture[str]) -> None:
    add_workbook(cli, "bad.xlsx", "Carteira", {"IBOV": [member("PETR4")]}, "2024-08-28")
    assert main(["data", "build", "universe", "--no-overrides"]) == EXIT_USAGE
    assert "cannot tell the kind" in capsys.readouterr().err


def test_review_universe_lists_unresolved_tickers(
    cli: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["data", "build", "universe", "--no-overrides"])
    capsys.readouterr()
    assert main(["data", "review", "universe"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "GONE3" in out and "DUPL3" in out
    assert "3 unresolved ticker(s)." in out
    assert main(["data", "review", "universe", "--index", "IBXX"]) == EXIT_OK
    assert "1 unresolved ticker(s)." in capsys.readouterr().out
    assert main(["data", "review", "universe", "--index", "SMLL"]) == EXIT_OK
    assert "Every ticker of SMLL resolves." in capsys.readouterr().out


def test_review_universe_before_a_build(
    cli: DataStore, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data", "review", "universe"]) == EXIT_USAGE
    assert "build universe" in capsys.readouterr().err
