# ibexQuant\tests\data\test_download.py

"""``ibexQuant.download``: the provider's bars, wide (ADR 0015, H5.10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from support.ledger import ledger_store

import ibexQuant
from ibexQuant.data.adjustment import build_adjustment_factors
from ibexQuant.data.download import download
from ibexQuant.data.ingestion.b3 import B3Provider
from ibexQuant.data.reference.corporate_action_overrides import ActionOverrides
from ibexQuant.data.reference.corporate_actions import build_corporate_actions
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module

START, END = "2024-04-22", "2024-05-03"
OHLCV = ["open", "high", "low", "close", "volume"]


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    store = ledger_store(tmp_path)
    build_corporate_actions(store, ActionOverrides.none())
    build_adjustment_factors(store)
    return store


def test_one_ticker_gives_flat_columns(store: DataStore) -> None:
    frame = download("PETR3", START, END, store=store)
    assert list(frame.columns) == OHLCV
    assert frame.index.name == "timestamp"
    assert frame.index.is_monotonic_increasing
    assert (frame.dtypes == "float64").all()
    provider = B3Provider(store).get_bars("PETR3", START, END)
    assert frame["close"].tolist() == provider["close"].tolist()


def test_several_tickers_give_field_ticker_columns(store: DataStore) -> None:
    frame = download(["VALE3", "PETR3"], START, END, fields=["close", "volume"], store=store)
    assert frame.columns.names == ["field", "ticker"]
    assert list(frame.columns) == [
        ("close", "VALE3"),
        ("close", "PETR3"),
        ("volume", "VALE3"),
        ("volume", "PETR3"),
    ]
    # VALE3 trades three sessions only: NaN on the others.
    assert frame[("close", "VALE3")].notna().sum() == 3
    assert frame[("close", "PETR3")].notna().all()


def test_a_list_of_one_keeps_the_ticker_level(store: DataStore) -> None:
    frame = download(["PETR3", "PETR3"], START, END, fields=["close"], store=store)
    assert list(frame.columns) == [("close", "PETR3")]


def test_adjustment_modes(store: DataStore) -> None:
    raw = download("PETR3", START, END, adjustment="raw", store=store)
    splits = download(
        "PETR3", START, "2024-04-25", adjustment="splits", anchor="2024-04-29", store=store
    )
    total = download("PETR3", START, END, store=store)
    assert raw["close"].iloc[0] == 10.0
    assert splits["close"].tolist() == [5.0, 5.0, 5.0, 5.0]
    assert total["close"].iloc[0] == pytest.approx(10.0 * 0.95 * 0.5 * 0.1)


def test_traded_value_is_raw(store: DataStore) -> None:
    frame = download("PETR3", START, END, fields=["traded_value", "close"], store=store)
    assert list(frame.columns) == ["close", "traded_value"]
    assert frame["traded_value"].iloc[0] == 1000 * 10.0


def test_an_unknown_ticker_is_left_out(store: DataStore) -> None:
    several = download(["ZZZZ3", "PETR3"], START, END, adjustment="raw", store=store)
    assert set(several.columns.get_level_values("ticker")) == {"PETR3"}
    with pytest.raises(ValueError, match="No B3 data"):
        download("ZZZZ3", START, END, adjustment="raw", store=store)


def test_options_are_checked(store: DataStore) -> None:
    with pytest.raises(ValueError, match="adjustment must be one of"):
        download("PETR3", START, END, adjustment="dividends", store=store)
    with pytest.raises(ValueError, match="anchor applies to adjusted bars only"):
        download("PETR3", START, END, adjustment="raw", anchor=END, store=store)
    with pytest.raises(ValueError, match="Unknown field"):
        download("PETR3", START, END, fields=["vwap"], store=store)


def test_the_package_root_serves_download_lazily(
    store: DataStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    assert ibexQuant.download is download
    assert "download" in ibexQuant.__all__
    frame = ibexQuant.download("PETR3", START, END, adjustment="raw")
    assert len(frame) == 9
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        ibexQuant.nope  # noqa: B018
