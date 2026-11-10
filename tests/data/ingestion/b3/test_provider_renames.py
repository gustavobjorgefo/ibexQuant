# ibexQuant\tests\data\ingestion\b3\test_provider_renames.py

"""
Tests for :class:`B3Provider` following renames (ADR 0012, H2.14).

Uses the six-session market of ``tests/support/reference.py``: VVAR3 ->
VIIA3 -> BHIA3 confirmed by CNPJ, TIMP3 -> TIMS3 pending, PETR4 with its
odd lot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from support.cotahist import detail, lines_for, write_zip
from support.reference import FCA_ROWS, TIMP, TIMS, isin, rename_market, write_fca, write_links

from ibexQuant.data.ingestion.b3 import B3Provider, build_bars_daily
from ibexQuant.data.reference import (
    LinkOverrides,
    SecurityMasterNotBuiltError,
    build_security_master,
    load_link_overrides,
)
from ibexQuant.data.store import DataStore

START, END = "2024-01-01", "2024-12-31"


def built(tmp_path: Path, details: list[bytes]) -> DataStore:
    store = DataStore.initialize(tmp_path / "ibex-data")
    write_zip(store.raw("b3", "cotahist"), 2024, lines_for(2024, details))
    build_bars_daily(store)
    write_fca(store.raw("cvm", "fca"), 2024, FCA_ROWS)
    build_security_master(store, LinkOverrides.none())
    return store


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return built(tmp_path, rename_market())


def closes(bars: pd.DataFrame, symbol: str) -> list[float]:
    return bars.xs(symbol, level="symbol")["close"].tolist()


def test_current_ticker_returns_the_whole_security(store: DataStore) -> None:
    bars = B3Provider(store).get_bars("BHIA3", START, END, adjusted=False)
    assert closes(bars, "BHIA3") == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


def test_old_ticker_returns_the_security_up_to_the_range_end(store: DataStore) -> None:
    bars = B3Provider(store).get_bars("VVAR3", START, END, adjusted=False)
    assert closes(bars, "VVAR3") == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


def test_range_limits_apply_to_the_whole_security(store: DataStore) -> None:
    bars = B3Provider(store).get_bars("VIIA3", "2024-01-03", "2024-01-05", adjusted=False)
    assert closes(bars, "VIIA3") == [11.0, 12.0, 13.0]  # VVAR3's last session, then VIIA3


def test_ticker_resolved_on_its_last_session_up_to_end(store: DataStore) -> None:
    # On 2024-01-03 the security was still VVAR3; BHIA3 did not exist yet.
    bars = B3Provider(store).get_bars(["VVAR3", "BHIA3"], START, "2024-01-03", adjusted=False)
    assert bars.index.get_level_values("symbol").unique().tolist() == ["VVAR3"]


def test_pending_rename_is_not_followed(store: DataStore) -> None:
    bars = B3Provider(store).get_bars("TIMS3", START, END, adjusted=False)
    assert len(bars) == 3


def test_override_makes_it_followed(store: DataStore, tmp_path: Path) -> None:
    write_links(tmp_path / "overrides", links=[(TIMP, TIMS)])
    build_security_master(store, load_link_overrides(tmp_path / "overrides"))
    bars = B3Provider(store).get_bars("TIMS3", START, END, adjusted=False)
    assert closes(bars, "TIMS3") == [20.0, 20.0, 20.0, 21.0, 21.0, 21.0]


def test_odd_lot_is_never_served(store: DataStore) -> None:
    bars = B3Provider(store).get_bars("PETR4", START, END, adjusted=False)
    assert len(bars) == 6


def test_two_tickers_of_the_same_security(store: DataStore) -> None:
    with pytest.raises(ValueError, match="same security"):
        B3Provider(store).get_bars(["VIIA3", "BHIA3"], START, END, adjusted=False)


def test_unknown_ticker_with_others_is_logged(
    store: DataStore, caplog: pytest.LogCaptureFixture
) -> None:
    bars = B3Provider(store).get_bars(["PETR4", "XXXX3"], START, END, adjusted=False)
    assert bars.index.get_level_values("symbol").unique().tolist() == ["PETR4"]
    assert "XXXX3" in caplog.text


def test_only_unknown_tickers(store: DataStore) -> None:
    with pytest.raises(ValueError, match="No B3 data"):
        B3Provider(store).get_bars("XXXX3", START, END, adjusted=False)


def test_reused_ticker_does_not_mix_securities(tmp_path: Path) -> None:
    first, second = isin("ABCB", "R02PR"), isin("ABCB", "R03PR")
    store = built(
        tmp_path,
        [
            detail(session="20240102", ticker="ABCB10", isin=first, close=1.0),
            detail(session="20240108", ticker="ABCB10", isin=second, close=2.0),
        ],
    )
    bars = B3Provider(store).get_bars("ABCB10", START, END, adjusted=False)
    assert closes(bars, "ABCB10") == [2.0]


def test_same_isin_under_two_tickers_on_one_session(tmp_path: Path) -> None:
    code = isin("AALC", "BDR00")
    store = built(
        tmp_path,
        [
            detail(session="20240102", ticker="AALC11B", isin=code, close=1.0),
            detail(session="20240103", ticker="AALC11B", isin=code, close=1.0, volume_shares=5),
            detail(session="20240103", ticker="AALC34", isin=code, close=2.0, volume_shares=50),
        ],
    )
    bars = B3Provider(store).get_bars("AALC34", START, END, adjusted=False)
    assert closes(bars, "AALC34") == [1.0, 2.0]


def test_without_a_master(tmp_path: Path) -> None:
    store = DataStore.initialize(tmp_path / "ibex-data")
    write_zip(store.raw("b3", "cotahist"), 2024, lines_for(2024, [detail()]))
    build_bars_daily(store)
    with pytest.raises(SecurityMasterNotBuiltError, match="build security-master"):
        B3Provider(store).get_bars("PETR4", START, END, adjusted=False)
