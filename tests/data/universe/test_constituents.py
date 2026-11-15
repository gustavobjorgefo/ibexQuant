# ibexQuant\tests\data\universe\test_constituents.py

"""Index compositions by period, resolved to securities (ADR 0015, H5.1–H5.5)."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from support.index_portfolios import Member, write_workbook

from ibexQuant.cli import EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.reference import SecurityMasterError
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, OVERRIDES_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module
from ibexQuant.data.universe import (
    PeriodOverride,
    PeriodOverrides,
    UniverseBuildError,
    UniverseNotBuiltError,
    build_universe,
    read_index_constituents,
    read_index_periods,
)
from ibexQuant.data.universe.constituents import CONSTITUENT_COLUMNS, SUMMARY_FILE

DAY = pd.Timestamp
FIRST, LAST = "2024-03-01", "2025-02-28"


def member(ticker: str, weight: float = 1.0) -> Member:
    return (ticker, ticker[:4], "ON  NM", 1000.0, weight)


def span(security_id: str, ticker: str, first: str = FIRST, last: str = LAST) -> dict[str, object]:
    return {
        "security_id": security_id,
        "ticker": ticker,
        "isin": security_id,
        "market_type": 10,
        "first_session": DAY(first),
        "last_session": DAY(last),
        "sessions": 1,
        "link_rule": "founding",
        "cnpj": "",
    }


MASTER = [
    span("BRPETR", "PETR4"),
    span("BRVALE", "VALE3"),
    span("BROLDX", "OLDX3", last="2024-09-30"),
    span("BROLDX", "NEWX3", first="2024-10-01"),
    span("BRGONE", "GONE3", last="2024-08-20"),
    span("BRLATE", "LATE3", first="2024-09-02"),
    span("BRDUP1", "DUPL3"),
    span("BRDUP2", "DUPL3"),
]


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


@pytest.fixture
def empty_store(tmp_path: Path) -> DataStore:
    """Bars and a master, no portfolio file yet."""
    store = DataStore.initialize(tmp_path / "data")
    days = pd.bdate_range(FIRST, LAST)
    for year in (2024, 2025):
        partition = store.curated("b3", "bars_daily", f"year={year}", "part-0.parquet")
        partition.parent.mkdir(parents=True)
        pd.DataFrame({"session_date": days[days.year == year]}).to_parquet(partition, index=False)
    pd.DataFrame(MASTER).to_parquet(store.curated("b3", "security_master.parquet"), index=False)
    return store


@pytest.fixture
def store(empty_store: DataStore) -> DataStore:
    """Every source: a 2nd preview, 3rd previews, a definitive portfolio, the retroactive CSV."""
    september = {
        "IBOV": [member(ticker) for ticker in ("PETR4", "VALE3", "OLDX3", "GONE3", "LATE3")],
        "IBXX": [member("PETR4"), member("DUPL3")],
    }
    add_workbook(
        empty_store,
        "a_2024Q3.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        september,
        "2024-08-28",
    )
    add_workbook(
        empty_store,
        "b_2024Q3_copy.xlsx",
        "3ª Prévia do IBOVESPA para setembro de 2024",
        september,
        "2024-08-28",
    )
    add_workbook(
        empty_store,
        "jan_3rd.xlsx",
        "3ª Prévia do IBOVESPA para janeiro de 2025",
        {"IBOV": [member("PETR4"), member("NEWX3")]},
        "2024-12-27",
    )
    add_workbook(
        empty_store,
        "jan_definitive.xlsx",
        "Virada do IBOVESPA para janeiro de 2025",
        {"IBOV": [member("PETR4"), member("NEWX3"), member("VALE3")]},
        "2025-01-03",
    )
    add_workbook(
        empty_store,
        "may_2nd.xlsx",
        "2ª Prévia do IBOVESPA para maio de 2024",
        {"IBOV": [member("PETR4", 100.0)]},
        "2024-04-16",
    )
    add_retroactive(empty_store, "2003-01,PETR4,PETROBRAS,PN,100\n2024-05,VALE3,VALE,ON,100\n")
    return empty_store


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
    assert len(september) == 7


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


def test_an_unreadable_file_stops_the_build(store: DataStore) -> None:
    add_workbook(store, "bad.xlsx", "Carteira teórica", {"IBOV": [member("PETR4")]}, "2024-08-28")
    with pytest.raises(UniverseBuildError, match="bad.xlsx: cannot tell the kind"):
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


# --- summary and storage ---


def test_the_summary_is_written_with_the_inputs(store: DataStore) -> None:
    summary = build_universe(store, PeriodOverrides.none())
    written = json.loads(store.curated("b3", SUMMARY_FILE).read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert written["rows"] == summary.rows == 12
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


def test_without_bars_the_build_stops(tmp_path: Path) -> None:
    with pytest.raises(UniverseNotBuiltError, match="ingest cotahist"):
        build_universe(DataStore.initialize(tmp_path / "data"), PeriodOverrides.none())


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
    assert "IBOV periods without composition: 0" in out
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
