# ibexQuant\tests\data\ingestion\b3\test_provider.py

"""
Tests for :class:`ibexQuant.data.ingestion.b3.B3Provider` (ADR 0011, H1.13).

Each test builds a small curated table from synthetic COTAHIST files and
reads it back through the provider, checking the shared provider contract
and the B3-specific rules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from support.cotahist import detail, lines_for, write_zip

from ibexQuant.data.ingestion.b3 import B3Provider, build_bars_daily
from ibexQuant.data.ingestion.provider import OHLCV_FIELDS
from ibexQuant.data.ingestion.validation import check_bars
from ibexQuant.data.store import DataStore

# --- Fixtures ---


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    """A store with 2023 and 2024 built: PETR4 and VALE3, plus odd lot and options."""
    store = DataStore.initialize(tmp_path / "ibex-data")
    folder = store.raw("b3", "cotahist")
    write_zip(
        folder,
        2023,
        lines_for(
            2023,
            [
                detail(session="20231228", close=37.0),
                detail(session="20231228", ticker="VALE3", isin="BRVALEACNOR0", close=77.0),
            ],
        ),
    )
    write_zip(
        folder,
        2024,
        lines_for(
            2024,
            [
                detail(session="20240102", close=38.5, volume_shares=2_000_000),
                detail(session="20240103", close=39.0),
                detail(session="20240102", ticker="VALE3", isin="BRVALEACNOR0", close=78.0),
                detail(session="20240102", ticker="PETR4F", market="020", bdi="96"),
                detail(session="20240102", ticker="PETRA40", market="070", bdi="78"),
                detail(session="20240102", ticker="BHIA3", bdi="07", isin="BRBHIAACNOR1"),
            ],
        ),
    )
    build_bars_daily(store)
    return store


@pytest.fixture
def provider(store: DataStore) -> B3Provider:
    return B3Provider(store)


# --- Shape of the result ---


def test_shape_matches_the_provider_contract(provider: B3Provider) -> None:
    bars = provider.get_bars(["PETR4", "VALE3"], "2023-01-01", "2024-12-31", adjusted=False)

    assert list(bars.index.names) == ["timestamp", "symbol"]
    assert list(bars.columns) == list(OHLCV_FIELDS)
    assert (bars.dtypes == "float64").all()
    assert bars.index.is_monotonic_increasing
    assert not bars.index.duplicated().any()
    assert isinstance(bars.index.get_level_values("timestamp"), pd.DatetimeIndex)
    check_bars(bars)  # accepts the shape


def test_values_and_volume_in_shares(provider: B3Provider) -> None:
    bars = provider.get_bars("PETR4", "2024-01-02", "2024-01-02", adjusted=False)

    row = bars.loc[(pd.Timestamp("2024-01-02"), "PETR4")]
    assert row["close"] == 38.5
    assert row["volume"] == 2_000_000.0


def test_single_symbol_has_the_same_shape(provider: B3Provider) -> None:
    bars = provider.get_bars("PETR4", "2024-01-01", "2024-12-31", adjusted=False)
    assert bars.index.get_level_values("symbol").unique().tolist() == ["PETR4"]


def test_range_spans_years_and_is_inclusive(provider: B3Provider) -> None:
    bars = provider.get_bars("PETR4", "2023-12-28", "2024-01-02", adjusted=False)
    dates = bars.index.get_level_values("timestamp")
    assert dates.tolist() == [pd.Timestamp("2023-12-28"), pd.Timestamp("2024-01-02")]


def test_fields_subset(provider: B3Provider) -> None:
    bars = provider.get_bars("PETR4", "2024-01-01", "2024-12-31", ["close"], adjusted=False)
    assert list(bars.columns) == ["close"]


def test_repeated_symbols_are_served_once(provider: B3Provider) -> None:
    bars = provider.get_bars(["PETR4", "PETR4"], "2024-01-01", "2024-12-31", adjusted=False)
    assert not bars.index.duplicated().any()


# --- B3 rules ---


def test_odd_lot_and_options_are_not_served(
    provider: B3Provider, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        bars = provider.get_bars(
            ["PETR4", "PETR4F", "PETRA40"], "2024-01-01", "2024-12-31", adjusted=False
        )
    assert bars.index.get_level_values("symbol").unique().tolist() == ["PETR4"]


def test_companies_outside_bdi_02_are_served(provider: B3Provider) -> None:
    bars = provider.get_bars("BHIA3", "2024-01-01", "2024-12-31", adjusted=False)
    assert len(bars) == 1


def test_adjusted_prices_are_refused_until_h4(provider: B3Provider) -> None:
    with pytest.raises(ValueError, match="adjusted=False"):
        provider.get_bars("PETR4", "2024-01-01", "2024-12-31")


def test_frequency_is_daily(provider: B3Provider) -> None:
    assert provider.frequency == "1d"


def test_same_ticker_twice_on_one_session_is_refused(tmp_path: Path) -> None:
    store = DataStore.initialize(tmp_path / "ibex-data")
    details = [detail(isin="BRPETRACNPR6"), detail(isin="BRPETRBCNPR9")]
    write_zip(store.raw("b3", "cotahist"), 2024, lines_for(2024, details))
    build_bars_daily(store, 2024)

    with pytest.raises(ValueError, match="More than one round-lot row"):
        B3Provider(store).get_bars("PETR4", "2024-01-01", "2024-12-31", adjusted=False)


# --- Missing data ---


def test_partial_result_logs_the_missing_tickers(
    provider: B3Provider, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        bars = provider.get_bars(["PETR4", "XXXX3"], "2024-01-01", "2024-12-31", adjusted=False)

    assert bars.index.get_level_values("symbol").unique().tolist() == ["PETR4"]
    assert "XXXX3" in caplog.text


def test_no_data_at_all_is_an_error(provider: B3Provider) -> None:
    with pytest.raises(ValueError, match="No B3 data"):
        provider.get_bars("XXXX3", "2024-01-01", "2024-12-31", adjusted=False)


def test_range_without_built_years_is_an_error(provider: B3Provider) -> None:
    with pytest.raises(ValueError, match="No B3 data"):
        provider.get_bars("PETR4", "2010-01-01", "2010-12-31", adjusted=False)


def test_range_inside_a_year_without_sessions(provider: B3Provider) -> None:
    with pytest.raises(ValueError, match="No B3 data"):
        provider.get_bars("PETR4", "2024-06-01", "2024-06-30", adjusted=False)


# --- Arguments ---


def test_start_after_end(provider: B3Provider) -> None:
    with pytest.raises(ValueError, match="after end"):
        provider.get_bars("PETR4", "2024-12-31", "2024-01-01", adjusted=False)


def test_unknown_field(provider: B3Provider) -> None:
    with pytest.raises(ValueError):
        provider.get_bars("PETR4", "2024-01-01", "2024-12-31", ["vwap"], adjusted=False)
