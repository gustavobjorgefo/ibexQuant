# ibexQuant\tests\data\test_dataset.py

"""B3Dataset: one entry point to the B3 base (ADR 0016)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from support.universe import DAY, write_base, write_portfolios

import ibexQuant.data
from ibexQuant.data import dataset as dataset_module
from ibexQuant.data.adjustment import build_adjustment_factors
from ibexQuant.data.dataset import B3Dataset, DatasetNotReadyError, as_universe
from ibexQuant.data.reference import SecurityNotFoundError
from ibexQuant.data.reference.corporate_action_overrides import ActionOverrides
from ibexQuant.data.reference.corporate_actions import build_corporate_actions
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module
from ibexQuant.data.universe import (
    Index,
    LiquiditySettings,
    PeriodOverrides,
    Tickers,
    build_universe,
)

SHORT = LiquiditySettings(lookback_months=1)
SEPTEMBER = ["BRLATE", "BROLDX", "BRPETR", "BRQUIT", "BRVALE"]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> DataStore:
    """Bars with a rename, a split and an exit; ledger, factors and universe built."""
    store = DataStore.initialize(tmp_path_factory.mktemp("dataset") / "data")
    write_portfolios(write_base(store))
    build_corporate_actions(store, ActionOverrides.none())
    build_adjustment_factors(store)
    build_universe(store, PeriodOverrides.none(), SHORT)
    return store


@pytest.fixture
def ds(built: DataStore) -> B3Dataset:
    return B3Dataset.open(built)


# --- open ---


def test_open_checks_the_base(ds: B3Dataset, built: DataStore) -> None:
    assert ds.store is built
    assert ds.calendar.sessions[0] == DAY("2024-03-01")
    assert ds.provider is ds.provider


def test_open_names_the_missing_table(tmp_path: Path) -> None:
    store = DataStore.initialize(tmp_path / "data")
    with pytest.raises(DatasetNotReadyError, match="ingest cotahist"):
        B3Dataset.open(store)
    write_base(store)
    with pytest.raises(DatasetNotReadyError, match="build universe"):
        B3Dataset.open(store)


def test_open_uses_the_configured_root(
    built: DataStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(built.root))
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    assert B3Dataset.open().store.root == built.root


def test_the_data_package_exports_the_dataset_lazily() -> None:
    assert ibexQuant.data.B3Dataset is B3Dataset
    assert "B3Dataset" in ibexQuant.data.__all__
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        ibexQuant.data.nope  # noqa: B018


# --- panel ---


def test_an_index_panel_has_the_period_members(ds: B3Dataset) -> None:
    panel = ds.panel("IBOV", "2024-09-02", "2024-12-31")
    assert list(panel.symbols) == SEPTEMBER
    assert panel.index[0] == DAY("2024-09-02")
    assert panel.index[-1] == DAY("2024-12-31")
    assert panel.universe_mask.all().all()
    close = panel.field("close")
    # QUIT3 stops trading on 2024-10-15: no price after it, still a member.
    assert close.loc[DAY("2024-10-16") :, "BRQUIT"].isna().all()
    assert close.loc[: DAY("2024-10-15"), "BRQUIT"].notna().all()


def test_warmup_adds_bars_with_the_mask_off(ds: B3Dataset) -> None:
    panel = ds.panel("IBOV", "2024-09-02", "2024-09-30", warmup=5)
    assert panel.index[0] == DAY("2024-08-26")
    assert not panel.universe_mask.loc[: DAY("2024-08-30")].any().any()
    assert panel.universe_mask.loc[DAY("2024-09-02")].all()


def test_prices_are_adjusted_to_the_anchor(ds: B3Dataset) -> None:
    adjusted = ds.panel(["PETR4"], "2024-06-03", "2024-07-31").field("close")["BRPETR"]
    raw = ds.panel(["PETR4"], "2024-06-03", "2024-07-31", adjustment="raw").field("close")
    assert adjusted.unique().tolist() == [10.0]  # the 2-for-1 split removed
    assert raw["BRPETR"].loc[DAY("2024-06-28")] == 20.0
    splits = ds.panel(
        ["PETR4"], "2024-06-03", "2024-06-28", adjustment="splits", anchor="2024-07-31"
    )
    assert splits.field("close")["BRPETR"].unique().tolist() == [10.0]


def test_a_ticker_list_matches_the_provider(ds: B3Dataset) -> None:
    panel = ds.panel(
        ["PETR4", "VALE3"], "2024-09-02", "2024-09-30", fields=["close", "traded_value"]
    )
    bars = ds.provider.get_bars(["PETR4", "VALE3"], "2024-09-02", "2024-09-30", ["close"])
    assert panel.fields == ("close", "traded_value")
    expected = bars["close"].unstack("symbol")
    assert panel.field("close")["BRPETR"].tolist() == expected["PETR4"].tolist()
    assert panel.field("traded_value")["BRVALE"].iloc[0] == 90e6


def test_a_renamed_ticker_keeps_one_column(ds: B3Dataset) -> None:
    panel = ds.panel(["NEWX3"], "2024-09-02", "2024-10-31")
    assert list(panel.symbols) == ["BROLDX"]
    assert panel.field("close")["BROLDX"].notna().all()


def test_universes_combine(ds: B3Dataset) -> None:
    panel = ds.panel(Index("IBXX") - Index("IBOV"), "2024-09-02", "2024-09-30")
    assert list(panel.symbols) == ["BRPAUS", "BRSTOP"]
    # PAUS3 did not trade on 2024-09-02: a real gap, not a missing session.
    assert pd.isna(panel.field("close").loc[DAY("2024-09-02"), "BRPAUS"])


def test_members_without_bars_in_range_are_left_out(ds: B3Dataset) -> None:
    panel = ds.panel(Tickers(["GONE3", "PETR4"]), "2024-09-02", "2024-09-30")
    assert list(panel.symbols) == ["BRPETR"]


def test_bar_findings_are_logged(
    ds: B3Dataset, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def flag_all(bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"warning_x": True, "critical_y": False}, index=bars.index)

    monkeypatch.setattr(dataset_module, "check_bars", flag_all)
    with caplog.at_level(logging.WARNING):
        ds.panel(["PETR4"], "2024-09-02", "2024-09-03")
    assert "check_bars flagged 2 bar(s): warning_x 2" in caplog.text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"adjustment": "dividends"}, "adjustment must be one of"),
        ({"adjustment": "raw", "anchor": "2024-09-30"}, "anchor applies"),
        ({"warmup": -1}, "warmup"),
    ],
)
def test_options_are_checked(ds: B3Dataset, arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ds.panel("IBOV", "2024-09-02", "2024-09-30", **arguments)  # type: ignore[arg-type]


def test_a_range_without_sessions_or_members_is_refused(ds: B3Dataset) -> None:
    with pytest.raises(ValueError, match="No B3 session"):
        ds.panel("IBOV", "2024-09-07", "2024-09-08")
    with pytest.raises(ValueError, match="has no member"):
        ds.panel("NOPE", "2024-09-02", "2024-09-30")


def test_the_universe_argument(ds: B3Dataset) -> None:
    assert as_universe("IBOV") == Index("IBOV")
    assert as_universe(["PETR4", "VALE3"]) == Tickers(["PETR4", "VALE3"])
    spec = Index("IBXX") - Index("IBOV")
    assert as_universe(spec) is spec
    with pytest.raises(TypeError, match="universe must be"):
        as_universe(5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="universe must be"):
        as_universe([1, 2])  # type: ignore[list-item]


# --- lookups ---


def test_constituents_on_a_date(ds: B3Dataset) -> None:
    members = ds.constituents("IBOV", "2024-09-10")
    assert members["ticker"].tolist() == ["GONE3", "LATE3", "OLDX3", "PETR4", "QUIT3", "VALE3"]
    assert set(members["period"]) == {"2024-09"}
    assert members.loc[members["ticker"] == "GONE3", "security_id"].isna().all()
    assert ds.constituents("IBOV", "2024-04-01").empty


def test_rebalance_dates(ds: B3Dataset) -> None:
    assert ds.rebalance_dates("IBOV", "2024-01-01", "2025-12-31").tolist() == [
        DAY("2024-05-06"),
        DAY("2024-09-02"),
        DAY("2025-01-06"),
    ]
    assert ds.rebalance_dates("IBXX", "2024-01-01", "2025-12-31").tolist() == [DAY("2024-09-02")]


def test_events_of_a_security(ds: B3Dataset) -> None:
    events = ds.events("BRPETR")
    assert events["event_type"].tolist() == ["SPLIT"]
    assert ds.events("BRVALE").empty


def test_exits_in_a_range(ds: B3Dataset) -> None:
    assert ds.exits("IBOV", "2024-09-01", "2024-12-31")["ticker"].tolist() == ["QUIT3"]
    assert ds.exits("IBXX", "2024-10-01", "2024-10-31")["ticker"].tolist() == ["STOP3"]
    assert ds.exits("IBOV", "2024-03-01", "2024-04-30").empty


# --- tickers ---


def test_ticker_at_follows_renames_and_exits(ds: B3Dataset) -> None:
    assert ds.ticker_at("BROLDX", "2024-06-03").tolist() == ["OLDX3"]
    tickers = ds.ticker_at(["BROLDX", "BRQUIT", "BRLATE"], "2024-11-01")
    assert tickers.index.name == "security_id"
    assert tickers.iloc[:2].tolist() == ["NEWX3", "QUIT3"]  # QUIT3: its last ticker
    assert tickers.iloc[2] == "LATE3"
    assert pd.isna(ds.ticker_at("BRLATE", "2024-05-01").iloc[0])  # not listed yet
    with pytest.raises(SecurityNotFoundError):
        ds.ticker_at("NOPE", "2024-06-03")


def test_ticker_history(ds: B3Dataset) -> None:
    history = ds.ticker_history(["BROLDX", "BRPETR"])
    assert history["ticker"].tolist() == ["OLDX3", "NEWX3", "PETR4"]
    assert history.loc[0, "last_session"] == DAY("2024-09-30")


def test_label_renames_columns_for_people(ds: B3Dataset) -> None:
    close = ds.panel("IBOV", "2024-09-02", "2024-12-31").field("close")
    labelled = ds.label(close)
    assert list(labelled.columns) == ["LATE3", "NEWX3", "PETR4", "QUIT3", "VALE3"]
    assert labelled.columns.name == "ticker"
    assert list(ds.label(close, as_of="2024-09-02").columns)[1] == "OLDX3"
    assert list(close.columns) == SEPTEMBER  # the original is untouched


def test_label_tells_reused_tickers_apart(ds: B3Dataset) -> None:
    frame = pd.DataFrame([[1.0, 2.0]], index=[DAY("2024-09-02")], columns=["BRDUP1", "BRDUP2"])
    assert list(ds.label(frame).columns) == ["DUPL3 (BRDUP1)", "DUPL3 (BRDUP2)"]


def test_label_keeps_the_id_of_a_security_not_listed_yet(ds: B3Dataset) -> None:
    frame = pd.DataFrame([[1.0]], index=[DAY("2024-05-02")], columns=["BRLATE"])
    assert list(ds.label(frame).columns) == ["BRLATE"]
