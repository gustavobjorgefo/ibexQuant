# ibexQuant\tests\data\test_panel.py

"""
Tests for :class:`ibexQuant.data.panel.Panel`.

Covers construction from wide and long formats, every validation error,
immutability under Copy-on-Write, the universe mask and the fingerprint.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pytest

from ibexQuant.data.panel import Panel

SESSIONS: Final[pd.DatetimeIndex] = pd.DatetimeIndex(
    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], name="timestamp"
)
SYMBOLS: Final[list[str]] = ["ITUB4.SA", "PETR4.SA", "VALE3.SA"]


# --- Fixtures ---


def make_wide(
    offset: float = 0.0,
    index: pd.DatetimeIndex = SESSIONS,
    symbols: list[str] = SYMBOLS,
) -> pd.DataFrame:
    values: np.ndarray = np.arange(len(index) * len(symbols), dtype="float64") + offset
    return pd.DataFrame(values.reshape(len(index), len(symbols)), index=index, columns=symbols)


@pytest.fixture
def close() -> pd.DataFrame:
    return make_wide(offset=100.0)


@pytest.fixture
def volume() -> pd.DataFrame:
    return make_wide(offset=1_000.0)


@pytest.fixture
def panel(close: pd.DataFrame, volume: pd.DataFrame) -> Panel:
    return Panel({"close": close, "volume": volume})


@pytest.fixture
def long_bars() -> pd.DataFrame:
    """Long-format bars where VALE3.SA is listed only from the third session."""
    rows: list[tuple[pd.Timestamp, str, float, float]] = []
    for position, timestamp in enumerate(SESSIONS):
        for symbol in ["ITUB4.SA", "PETR4.SA"]:
            rows.append((timestamp, symbol, 10.0 + position, 1_000.0))
        if position >= 2:
            rows.append((timestamp, "VALE3.SA", 60.0 + position, 2_000.0))
    frame = pd.DataFrame(rows, columns=["timestamp", "symbol", "close", "volume"])
    return frame.set_index(["timestamp", "symbol"]).sort_index()


# --- Construction ---


def test_construction_exposes_fields_symbols_and_shape(panel: Panel) -> None:
    assert panel.fields == ("close", "volume")
    assert panel.symbols == tuple(SYMBOLS)
    assert panel.shape == (4, 3)
    assert panel.index.equals(SESSIONS)


def test_construction_sorts_symbols(close: pd.DataFrame) -> None:
    shuffled = close[["VALE3.SA", "ITUB4.SA", "PETR4.SA"]]
    panel = Panel({"close": shuffled})

    assert panel.symbols == tuple(SYMBOLS)
    pd.testing.assert_frame_equal(panel.field("close"), close.rename_axis(columns="symbol"))


def test_construction_names_axes(close: pd.DataFrame) -> None:
    unnamed = close.rename_axis(index=None)
    frame = Panel({"close": unnamed}).field("close")

    assert frame.index.name == "timestamp"
    assert frame.columns.name == "symbol"


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_construction_accepts_naive_and_aware_index(tz: str | None) -> None:
    index = pd.DatetimeIndex(SESSIONS, tz=tz)
    panel = Panel({"close": make_wide(index=index)})

    assert panel.index.tz == (None if tz is None else index.tz)


def test_default_universe_mask_is_all_true(panel: Panel) -> None:
    mask = panel.universe_mask

    assert mask.dtypes.eq(bool).all()
    assert mask.to_numpy().all()
    assert mask.shape == panel.shape


def test_unknown_field_raises_key_error(panel: Panel) -> None:
    with pytest.raises(KeyError, match="Unknown field 'open'"):
        panel.field("open")


# --- Validation errors ---


def test_no_fields_raises() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        Panel({})


def test_non_string_field_name_raises(close: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="Field names must be strings"):
        Panel({1: close})  # type: ignore[dict-item]


def test_non_dataframe_field_raises() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        Panel({"close": pd.Series([1.0], index=SESSIONS[:1])})  # type: ignore[dict-item]


def test_non_datetime_index_raises(close: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        Panel({"close": close.reset_index(drop=True)})


def test_mixed_timezone_values_cannot_form_a_datetime_index(close: pd.DataFrame) -> None:
    mixed = close.copy()
    mixed.index = pd.Index(
        [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03", tz="UTC"), *SESSIONS[2:]],
        dtype="object",
    )
    with pytest.raises(TypeError, match="DatetimeIndex"):
        Panel({"close": mixed})


def test_empty_rows_raise(close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        Panel({"close": close.iloc[0:0]})


def test_empty_columns_raise(close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        Panel({"close": close[[]]})


def test_unsorted_index_raises(close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not sorted"):
        Panel({"close": close.iloc[::-1]})


def test_duplicate_timestamps_raise(close: pd.DataFrame) -> None:
    duplicated = pd.concat([close.iloc[:1], close])
    with pytest.raises(ValueError, match="duplicate timestamps"):
        Panel({"close": duplicated})


def test_duplicate_symbols_raise(close: pd.DataFrame) -> None:
    duplicated = pd.concat([close, close[["PETR4.SA"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate symbols"):
        Panel({"close": duplicated})


def test_non_string_symbol_raises(close: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="non-string symbols"):
        Panel({"close": close.set_axis(["ITUB4.SA", "PETR4.SA", 3], axis=1)})


def test_non_float64_field_raises(close: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="float64"):
        Panel({"close": close.astype("int64")})


def test_misaligned_index_raises(close: pd.DataFrame) -> None:
    shifted = make_wide(index=SESSIONS + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="index does not match"):
        Panel({"close": close, "volume": shifted})


def test_misaligned_symbols_raise(close: pd.DataFrame) -> None:
    other = make_wide(symbols=["ITUB4.SA", "PETR4.SA", "WEGE3.SA"])
    with pytest.raises(ValueError, match=r"missing \['VALE3.SA'\], unexpected \['WEGE3.SA'\]"):
        Panel({"close": close, "volume": other})


# --- Universe mask ---


def test_universe_mask_is_stored_sorted(close: pd.DataFrame) -> None:
    mask = pd.DataFrame(True, index=SESSIONS, columns=SYMBOLS)
    mask.loc[SESSIONS[0], "VALE3.SA"] = False
    panel = Panel({"close": close}, universe_mask=mask[["VALE3.SA", "PETR4.SA", "ITUB4.SA"]])

    assert not panel.universe_mask.loc[SESSIONS[0], "VALE3.SA"]
    assert tuple(panel.universe_mask.columns) == tuple(SYMBOLS)


def test_universe_mask_wrong_type_raises(close: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="universe_mask"):
        Panel({"close": close}, universe_mask=np.ones((4, 3), dtype=bool))  # type: ignore[arg-type]


def test_universe_mask_wrong_index_raises(close: pd.DataFrame) -> None:
    mask = pd.DataFrame(True, index=SESSIONS[:3], columns=SYMBOLS)
    with pytest.raises(ValueError, match="universe_mask index does not match"):
        Panel({"close": close}, universe_mask=mask)


def test_universe_mask_wrong_symbols_raise(close: pd.DataFrame) -> None:
    mask = pd.DataFrame(True, index=SESSIONS, columns=SYMBOLS[:2])
    with pytest.raises(ValueError, match="universe_mask symbols do not match"):
        Panel({"close": close}, universe_mask=mask)


def test_universe_mask_non_bool_raises(close: pd.DataFrame) -> None:
    mask = pd.DataFrame(1.0, index=SESSIONS, columns=SYMBOLS)
    with pytest.raises(ValueError, match="universe_mask must be bool"):
        Panel({"close": close}, universe_mask=mask)


def test_universe_mask_with_missing_values_raises(close: pd.DataFrame) -> None:
    mask = pd.DataFrame(True, index=SESSIONS, columns=SYMBOLS, dtype="object")
    mask.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="universe_mask must be bool"):
        Panel({"close": close}, universe_mask=mask)


def test_with_universe_mask_returns_new_panel(panel: Panel) -> None:
    mask = pd.DataFrame(False, index=SESSIONS, columns=SYMBOLS)
    masked = panel.with_universe_mask(mask)

    assert masked is not panel
    assert not masked.universe_mask.to_numpy().any()
    assert panel.universe_mask.to_numpy().all()
    pd.testing.assert_frame_equal(masked.field("close"), panel.field("close"))


# --- Immutability ---


def test_writing_to_returned_field_does_not_change_panel(panel: Panel) -> None:
    returned = panel.field("close")
    returned.iloc[0, 0] = -1.0

    assert panel.field("close").iloc[0, 0] == 100.0


def test_writing_to_original_input_does_not_change_panel(close: pd.DataFrame) -> None:
    panel = Panel({"close": close})
    close.iloc[0, 0] = -1.0

    assert panel.field("close").iloc[0, 0] == 100.0


def test_writing_to_returned_mask_does_not_change_panel(panel: Panel) -> None:
    returned = panel.universe_mask
    returned.iloc[0, 0] = False

    assert panel.universe_mask.iloc[0, 0]


def test_public_properties_cannot_be_reassigned(panel: Panel) -> None:
    with pytest.raises(AttributeError):
        panel.symbols = ("OTHER",)  # type: ignore[misc]


# --- from_bars ---


def test_from_bars_builds_one_field_per_column(long_bars: pd.DataFrame) -> None:
    panel = Panel.from_bars(long_bars)

    assert panel.fields == ("close", "volume")
    assert panel.symbols == tuple(SYMBOLS)
    assert panel.index.equals(SESSIONS)


def test_from_bars_marks_pre_listing_sessions_as_nan(long_bars: pd.DataFrame) -> None:
    close = Panel.from_bars(long_bars).field("close")

    assert close["VALE3.SA"].iloc[:2].isna().all()
    assert close["VALE3.SA"].iloc[2:].tolist() == [62.0, 63.0]
    assert close[["ITUB4.SA", "PETR4.SA"]].notna().all().all()


def test_from_bars_accepts_universe_mask(long_bars: pd.DataFrame) -> None:
    mask = pd.DataFrame(True, index=SESSIONS, columns=SYMBOLS)
    mask["VALE3.SA"] = False
    panel = Panel.from_bars(long_bars, universe_mask=mask)

    assert not panel.universe_mask["VALE3.SA"].any()


def test_from_bars_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pd.DataFrame"):
        Panel.from_bars([1, 2, 3])  # type: ignore[arg-type]


def test_from_bars_rejects_single_index(long_bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        Panel.from_bars(long_bars.reset_index(level="symbol"))


def test_from_bars_rejects_wrong_level_order(long_bars: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="MultiIndex"):
        Panel.from_bars(long_bars.swaplevel())


def test_from_bars_rejects_duplicate_pairs(long_bars: pd.DataFrame) -> None:
    duplicated = pd.concat([long_bars, long_bars.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        Panel.from_bars(duplicated)


# --- Fingerprint ---


def test_fingerprint_is_sha256_hex(panel: Panel) -> None:
    assert len(panel.fingerprint) == 64
    int(panel.fingerprint, 16)


def test_fingerprint_is_stable_across_column_order(close: pd.DataFrame) -> None:
    ordered = Panel({"close": close})
    shuffled = Panel({"close": close[["VALE3.SA", "PETR4.SA", "ITUB4.SA"]]})

    assert ordered.fingerprint == shuffled.fingerprint


def test_fingerprint_changes_with_values(close: pd.DataFrame) -> None:
    changed = close.copy()
    changed.iloc[-1, -1] += 0.01

    assert Panel({"close": close}).fingerprint != Panel({"close": changed}).fingerprint


def test_fingerprint_changes_with_field_name(close: pd.DataFrame) -> None:
    assert Panel({"close": close}).fingerprint != Panel({"open": close}).fingerprint


def test_fingerprint_changes_with_universe_mask(panel: Panel) -> None:
    mask = pd.DataFrame(True, index=SESSIONS, columns=SYMBOLS)
    mask.iloc[0, 0] = False

    assert panel.with_universe_mask(mask).fingerprint != panel.fingerprint


def test_fingerprint_distinguishes_naive_and_aware_index() -> None:
    naive = Panel({"close": make_wide()})
    aware = Panel({"close": make_wide(index=SESSIONS.tz_localize("UTC"))})

    assert naive.fingerprint != aware.fingerprint


def test_fingerprint_treats_nan_consistently(long_bars: pd.DataFrame) -> None:
    assert Panel.from_bars(long_bars).fingerprint == Panel.from_bars(long_bars).fingerprint


# --- Representation ---


def test_repr_summarizes_panel(panel: Panel) -> None:
    text = repr(panel)

    assert text.startswith("Panel(fields=('close', 'volume'), symbols=3, timestamps=4")
    assert "start=2024-01-02" in text
    assert "end=2024-01-05" in text
