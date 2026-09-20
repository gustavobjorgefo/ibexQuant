# ibexQuant\tests\data\ingestion\yfinance\test_provider.py

"""
Unit tests for YFinanceProvider.

These tests never touch the network — ``yfinance.download`` is
monkeypatched in every test to return controlled, yfinance-shaped data.
For a check against the real, live API instead, run
``scripts/check_yfinance_provider.py`` by hand.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from ibexQuant.data.ingestion.provider import OHLCV_FIELDS
from ibexQuant.data.ingestion.yfinance import provider as provider_module
from ibexQuant.data.ingestion.yfinance.provider import YFinanceProvider

# --- fixtures / helpers ---


def _make_raw_frame(
    dates: pd.DatetimeIndex,
    ticker: str,
    close_values: list[float],
) -> pd.DataFrame:
    """Build a frame shaped like a single-ticker yfinance.download() result."""
    data = {
        "Open": close_values,
        "High": [value + 1 for value in close_values],
        "Low": [value - 1 for value in close_values],
        "Close": close_values,
        "Volume": [1_000.0] * len(close_values),
    }
    frame = pd.DataFrame(data, index=dates)
    frame.columns = pd.MultiIndex.from_product(
        [frame.columns, [ticker]], names=["Price", "Ticker"]
    )
    return frame


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=3, freq="B")


# --- get_bars: shape and schema ---


def test_get_bars_returns_canonical_shape_for_single_symbol(
    monkeypatch: pytest.MonkeyPatch, dates: pd.DatetimeIndex
) -> None:
    raw = _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0])
    monkeypatch.setattr(provider_module.yfinance, "download", lambda **_: raw)

    provider = YFinanceProvider(frequency="1d")
    data = provider.get_bars("PETR4.SA", "2024-01-01", "2024-01-05")

    assert list(data.index.names) == ["timestamp", "symbol"]
    assert tuple(data.columns) == OHLCV_FIELDS
    assert (data.dtypes == "float64").all()
    assert set(data.index.get_level_values("symbol")) == {"PETR4.SA"}


def test_get_bars_single_symbol_matches_list_of_one_shape(
    monkeypatch: pytest.MonkeyPatch, dates: pd.DatetimeIndex
) -> None:
    raw = _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0])
    monkeypatch.setattr(provider_module.yfinance, "download", lambda **_: raw)

    provider = YFinanceProvider(frequency="1d")
    as_str = provider.get_bars("PETR4.SA", "2024-01-01", "2024-01-05")
    as_list = provider.get_bars(["PETR4.SA"], "2024-01-01", "2024-01-05")

    pd.testing.assert_frame_equal(as_str, as_list)


# --- get_bars: multiple symbols ---


def test_get_bars_combines_and_sorts_multiple_symbols(
    monkeypatch: pytest.MonkeyPatch, dates: pd.DatetimeIndex
) -> None:
    frames = {
        "PETR4.SA": _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0]),
        "VALE3.SA": _make_raw_frame(dates, "VALE3.SA", [60.0, 61.0, 62.0]),
    }
    monkeypatch.setattr(provider_module.yfinance, "download", lambda tickers, **_: frames[tickers])

    provider = YFinanceProvider(frequency="1d")
    data = provider.get_bars(["VALE3.SA", "PETR4.SA"], "2024-01-01", "2024-01-05")

    assert set(data.index.get_level_values("symbol")) == {"PETR4.SA", "VALE3.SA"}
    assert data.index.is_monotonic_increasing


# --- get_bars: fields ---


def test_get_bars_filters_and_reorders_requested_fields(
    monkeypatch: pytest.MonkeyPatch, dates: pd.DatetimeIndex
) -> None:
    raw = _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0])
    monkeypatch.setattr(provider_module.yfinance, "download", lambda **_: raw)

    provider = YFinanceProvider(frequency="1d")
    data = provider.get_bars("PETR4.SA", "2024-01-01", "2024-01-05", fields=["volume", "close"])

    assert tuple(data.columns) == ("close", "volume")


# --- get_bars: partial vs. total failure ---


def test_get_bars_skips_symbol_with_no_data_and_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    dates: pd.DatetimeIndex,
    caplog: pytest.LogCaptureFixture,
) -> None:
    good = _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0])

    def fake_download(tickers: str, **_: Any) -> pd.DataFrame:
        return good if tickers == "PETR4.SA" else pd.DataFrame()

    monkeypatch.setattr(provider_module.yfinance, "download", fake_download)

    provider = YFinanceProvider(frequency="1d")
    with caplog.at_level("WARNING"):
        data = provider.get_bars(["PETR4.SA", "DELISTED.SA"], "2024-01-01", "2024-01-05")

    assert set(data.index.get_level_values("symbol")) == {"PETR4.SA"}
    assert "DELISTED.SA" in caplog.text


def test_get_bars_skips_symbol_whose_download_raises(
    monkeypatch: pytest.MonkeyPatch,
    dates: pd.DatetimeIndex,
    caplog: pytest.LogCaptureFixture,
) -> None:
    good = _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0])

    def fake_download(tickers: str, **_: Any) -> pd.DataFrame:
        if tickers == "PETR4.SA":
            return good
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(provider_module.yfinance, "download", fake_download)

    provider = YFinanceProvider(frequency="1d")
    with caplog.at_level("WARNING"):
        data = provider.get_bars(["PETR4.SA", "FLAKY.SA"], "2024-01-01", "2024-01-05")

    assert set(data.index.get_level_values("symbol")) == {"PETR4.SA"}
    assert "FLAKY.SA" in caplog.text


def test_get_bars_raises_when_every_symbol_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module.yfinance, "download", lambda **_: pd.DataFrame())

    provider = YFinanceProvider(frequency="1d")
    with pytest.raises(ValueError, match="No data returned"):
        provider.get_bars(["A.SA", "B.SA"], "2024-01-01", "2024-01-05")


# --- get_bars: data integrity ---


def test_get_bars_deduplicates_repeated_timestamps(
    monkeypatch: pytest.MonkeyPatch, dates: pd.DatetimeIndex
) -> None:
    raw = _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0])
    duplicated = pd.concat([raw, raw.iloc[[0]]])  # repeat the first row
    monkeypatch.setattr(provider_module.yfinance, "download", lambda **_: duplicated)

    provider = YFinanceProvider(frequency="1d")
    data = provider.get_bars("PETR4.SA", "2024-01-01", "2024-01-05")

    assert not data.index.duplicated().any()


# --- get_bars: parameters reach yfinance correctly ---


def test_get_bars_passes_frequency_and_adjusted_to_yfinance(
    monkeypatch: pytest.MonkeyPatch, dates: pd.DatetimeIndex
) -> None:
    raw = _make_raw_frame(dates, "PETR4.SA", [10.0, 11.0, 12.0])
    captured: dict[str, Any] = {}

    def fake_download(**kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return raw

    monkeypatch.setattr(provider_module.yfinance, "download", fake_download)

    provider = YFinanceProvider(frequency="1wk")
    provider.get_bars("PETR4.SA", "2024-01-01", "2024-01-05", adjusted=False)

    assert captured["interval"] == "1wk"
    assert captured["auto_adjust"] is False
