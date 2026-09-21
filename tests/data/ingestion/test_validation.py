# ibexQuant\tests\data\ingestion\test_validation.py

"""
Unit tests for check_bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibexQuant.data.ingestion.validation import check_bars

# --- fixtures / helpers ---


def _make_bars(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a MarketDataProvider-shaped frame from a list of row dicts."""
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.set_index(["timestamp", "symbol"]).sort_index()


def _clean_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": "2024-01-02",
        "symbol": "PETR4.SA",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1_000.0,
    }
    row.update(overrides)
    return row


# --- shape validation ---


def test_check_bars_rejects_non_multiindex_frame() -> None:
    df = pd.DataFrame({"close": [10.0]})
    with pytest.raises(ValueError, match="MultiIndex"):
        check_bars(df)


def test_check_bars_rejects_multiindex_with_wrong_names() -> None:
    df = pd.DataFrame(
        {"close": [10.0]},
        index=pd.MultiIndex.from_tuples([("a", "b")], names=["x", "y"]),
    )
    with pytest.raises(ValueError, match="MultiIndex"):
        check_bars(df)


def test_check_bars_rejects_frame_with_no_ohlcv_columns() -> None:
    df = _make_bars([_clean_row()])[[]]
    with pytest.raises(ValueError, match="none of the expected OHLCV columns"):
        check_bars(df)


# --- clean data ---


def test_check_bars_flags_nothing_for_clean_data() -> None:
    df = _make_bars([_clean_row(), _clean_row(symbol="VALE3.SA")])
    issues = check_bars(df)

    assert not issues.to_numpy().any()


def test_check_bars_output_shares_the_input_index() -> None:
    df = _make_bars([_clean_row()])
    issues = check_bars(df)

    assert issues.index.equals(df.index)


# --- critical rules ---


def test_check_bars_flags_high_below_low() -> None:
    df = _make_bars([_clean_row(high=5.0, low=9.0)])
    issues = check_bars(df)

    assert issues["critical_high_below_low"].iloc[0]


def test_check_bars_flags_high_below_open() -> None:
    df = _make_bars([_clean_row(high=9.0, open=10.0)])
    issues = check_bars(df)

    assert issues["critical_high_below_open"].iloc[0]


def test_check_bars_flags_high_below_close() -> None:
    df = _make_bars([_clean_row(high=9.0, close=10.0)])
    issues = check_bars(df)

    assert issues["critical_high_below_close"].iloc[0]


def test_check_bars_flags_low_above_open() -> None:
    df = _make_bars([_clean_row(low=11.0, open=10.0)])
    issues = check_bars(df)

    assert issues["critical_low_above_open"].iloc[0]


def test_check_bars_flags_low_above_close() -> None:
    df = _make_bars([_clean_row(low=11.0, close=10.0)])
    issues = check_bars(df)

    assert issues["critical_low_above_close"].iloc[0]


def test_check_bars_flags_negative_volume() -> None:
    df = _make_bars([_clean_row(volume=-1.0)])
    issues = check_bars(df)

    assert issues["critical_negative_volume"].iloc[0]


# --- warning rule ---


def test_check_bars_flags_negative_price_as_warning_only() -> None:
    df = _make_bars([_clean_row(open=-1.0, high=11.0, low=-2.0)])
    issues = check_bars(df)

    assert issues["warning_negative_price"].iloc[0]
    assert not issues.filter(like="critical_").iloc[0].any()


# --- missing data ---


def test_check_bars_does_not_flag_nan_rows() -> None:
    df = _make_bars([_clean_row(high=np.nan, low=np.nan)])
    issues = check_bars(df)

    assert not issues.to_numpy().any()


# --- partial fields ---


def test_check_bars_only_evaluates_rules_its_columns_support() -> None:
    df = _make_bars([_clean_row()])[["close"]]
    issues = check_bars(df)

    assert list(issues.columns) == ["warning_negative_price"]
