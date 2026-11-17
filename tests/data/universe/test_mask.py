# ibexQuant\tests\data\universe\test_mask.py

"""Universe specifications and the membership mask (ADR 0015, H5.6, H5.7)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest
from support.universe import DAY, workbooks, write_base, write_portfolios

from ibexQuant.data.store import DATA_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module
from ibexQuant.data.universe import (
    Index,
    LiquidityRule,
    LiquiditySettings,
    PeriodOverrides,
    Securities,
    Tickers,
    UniverseData,
    UniverseNotBuiltError,
    build_universe,
    universe_mask,
)
from ibexQuant.data.universe.mask import Combined

SHORT = LiquiditySettings(lookback_months=1)
SESSIONS = pd.DatetimeIndex(["2024-04-01", "2024-05-06", "2024-08-30", "2024-09-02", "2025-01-06"])


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    store = write_portfolios(write_base(DataStore.initialize(tmp_path / "data")))
    build_universe(store, PeriodOverrides.none(), SHORT)
    return store


def members(mask: pd.DataFrame, day: str) -> list[str]:
    row = mask.loc[DAY(day)]
    return sorted(row.index[row])


# --- Index ---


def test_index_members_follow_the_period_in_force(store: DataStore) -> None:
    mask = universe_mask(Index("IBOV"), SESSIONS, store)
    assert members(mask, "2024-04-01") == []  # before the first period
    assert members(mask, "2024-05-06") == ["BRPETR"]
    assert members(mask, "2024-08-30") == ["BRPETR"]
    assert members(mask, "2024-09-02") == ["BRLATE", "BROLDX", "BRPETR", "BRQUIT", "BRVALE"]
    assert members(mask, "2025-01-06") == ["BROLDX", "BRPETR", "BRVALE"]


def test_the_mask_shape(store: DataStore) -> None:
    mask = universe_mask(Index("IBOV"), SESSIONS, store)
    assert mask.index.equals(SESSIONS)
    assert mask.index.name == "session_date"
    assert mask.columns.name == "security_id"
    assert list(mask.columns) == sorted(mask.columns)
    assert (mask.dtypes == "bool").all()
    assert mask.any(axis=0).all()  # no column without a member session


def test_fill_false_leaves_liquidity_periods_empty(store: DataStore) -> None:
    (workbooks(store) / "may_2nd.xlsx").unlink()
    shutil.rmtree(store.raw("b3", "index_portfolios", "retroactive"))
    build_universe(store, PeriodOverrides.none(), SHORT)
    filled = universe_mask(Index("IBOV"), SESSIONS, store)
    real = universe_mask(Index("IBOV", fill=False), SESSIONS, store)
    assert len(members(filled, "2024-05-06")) == 6
    assert members(real, "2024-05-06") == []
    assert members(real, "2024-09-02") == members(filled, "2024-09-02")


def test_an_unknown_index_is_empty(store: DataStore) -> None:
    mask = universe_mask(Index("NOPE"), SESSIONS, store)
    assert mask.shape == (len(SESSIONS), 0)


# --- Tickers ---


def test_a_ticker_follows_its_security_across_renames(store: DataStore) -> None:
    mask = universe_mask(Tickers(["NEWX3"]), SESSIONS, store)
    assert list(mask.columns) == ["BROLDX"]
    assert mask["BROLDX"].all()  # OLDX3 before October, NEWX3 after


def test_a_ticker_is_in_only_during_its_life(store: DataStore) -> None:
    mask = universe_mask(Tickers("LATE3"), SESSIONS, store)
    assert mask["BRLATE"].tolist() == [False, False, False, True, True]
    assert Tickers("LATE3").tickers == ("LATE3",)
    assert Tickers(["A", "B", "A"]).tickers == ("A", "B")


def test_an_unknown_ticker_is_logged(store: DataStore, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        mask = universe_mask(Tickers(["PETR4", "ZZZZ3"]), SESSIONS, store)
    assert list(mask.columns) == ["BRPETR"]
    assert "ZZZZ3 is not in the security master" in caplog.text


def test_unknown_tickers_only_give_an_empty_mask(store: DataStore) -> None:
    assert universe_mask(Tickers(["ZZZZ3"]), SESSIONS, store).empty


# --- LiquidityRule ---


def test_the_liquidity_rule_selects_at_each_base_session(store: DataStore) -> None:
    mask = universe_mask(LiquidityRule(top=2, lookback_months=1), SESSIONS, store)
    assert members(mask, "2024-09-02") == ["BRPETR", "BRVALE"]  # the fund is not eligible
    assert members(mask, "2024-04-01") == []


def test_without_history_the_liquidity_rule_is_empty(store: DataStore) -> None:
    assert universe_mask(LiquidityRule(top=2), SESSIONS, store).empty


def test_liquidity_rule_parameters_are_checked() -> None:
    with pytest.raises(ValueError, match="top"):
        LiquidityRule(top=0)
    with pytest.raises(ValueError, match="min_presence"):
        LiquidityRule(top=5, min_presence=2.0)


def test_the_first_period_has_no_base_session(tmp_path: Path) -> None:
    # A period starting on the first session of the data has no base session.
    store = write_portfolios(write_base(DataStore.initialize(tmp_path / "data")))
    build_universe(store, PeriodOverrides.none(), SHORT)
    data = UniverseData(store)
    data.periods.loc[0, "effective_from"] = DAY("2024-03-01")
    mask = LiquidityRule(top=2, lookback_months=1).mask(data, pd.DatetimeIndex(["2024-03-04"]))
    assert mask.empty


# --- Securities ---


def test_securities_are_members_at_every_session(store: DataStore) -> None:
    mask = universe_mask(Securities(["BRVALE", "BRPETR"]), SESSIONS, store)
    assert mask.columns.tolist() == ["BRPETR", "BRVALE"]
    assert mask.index.equals(SESSIONS)
    assert mask.to_numpy().all()


def test_securities_keep_one_of_each_and_take_a_string() -> None:
    assert Securities(["BRVALE", "BRPETR", "BRVALE"]).security_ids == ("BRVALE", "BRPETR")
    assert Securities("BRPETR").security_ids == ("BRPETR",)


def test_securities_do_not_check_the_master(store: DataStore) -> None:
    mask = universe_mask(Securities("BRNONE"), SESSIONS, store)
    assert mask.columns.tolist() == ["BRNONE"]


def test_securities_need_one_id() -> None:
    with pytest.raises(ValueError, match="at least one security_id"):
        Securities([])


def test_securities_combine_with_an_index(store: DataStore) -> None:
    day = pd.DatetimeIndex(["2024-09-02"])
    chosen = Securities(["BRPETR", "BRPAUS"])
    assert members(universe_mask(chosen & Index("IBOV"), day, store), "2024-09-02") == ["BRPETR"]
    assert members(universe_mask(chosen - Index("IBOV"), day, store), "2024-09-02") == ["BRPAUS"]


# --- combinations ---


def test_difference_union_and_intersection(store: DataStore) -> None:
    day = pd.DatetimeIndex(["2024-09-02"])
    ibxx, ibov = Index("IBXX"), Index("IBOV")
    assert members(universe_mask(ibxx - ibov, day, store), "2024-09-02") == ["BRPAUS", "BRSTOP"]
    assert members(universe_mask(ibxx & ibov, day, store), "2024-09-02") == ["BRPETR"]
    union = universe_mask(ibxx | Tickers("FUND11"), day, store)
    assert members(union, "2024-09-02") == ["BRFUND", "BRPAUS", "BRPETR", "BRSTOP"]


def test_an_unknown_operation_is_refused() -> None:
    with pytest.raises(ValueError, match="operation"):
        Combined(Index("IBOV"), Index("IBXX"), "^")


# --- store ---


def test_the_configured_store_is_used_by_default(
    store: DataStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    assert members(universe_mask(Index("IBOV"), SESSIONS), "2024-05-06") == ["BRPETR"]


def test_a_mask_needs_the_universe_built(tmp_path: Path) -> None:
    store = write_base(DataStore.initialize(tmp_path / "data"))
    with pytest.raises(UniverseNotBuiltError, match="build universe"):
        universe_mask(Index("IBOV"), SESSIONS, store)


def test_traded_values_are_read_once(store: DataStore) -> None:
    data = UniverseData(store)
    assert data.traded is data.traded
