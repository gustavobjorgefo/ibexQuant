# ibexQuant\scripts\check_panel.py

"""
Manual, network-hitting walkthrough of :class:`Panel` built from real data.

Not a test — a plain script for eyeballing how a ``Panel`` is built from the
ingestion pipeline (``get_bars -> check_bars -> align_to_calendar``) and how
its public API behaves: accessors, wide-format operations, universe mask,
immutability, validation and fingerprint. Run it by hand after touching
``ibexQuant.data.panel``; nothing here is enforced by CI.

The return and ranking computations below are illustrations of the wide
format only. They do not follow the observation-time policy for gaps
(ADR 0003); that policy is implemented by the feature kernels.

Usage
-----
python scripts/check_panel.py
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from ibexQuant.data.calendars.calendar import align_to_calendar
from ibexQuant.data.calendars.pandas_market_calendars.calendar import (
    PandasMarketCalendarsCalendar,
)
from ibexQuant.data.ingestion.validation import check_bars
from ibexQuant.data.ingestion.yfinance.provider import YFinanceProvider
from ibexQuant.data.panel import Panel

SYMBOLS: Final[tuple[str, ...]] = ("PETR4.SA", "VALE3.SA", "ITUB4.SA")
START: Final[str] = "2024-01-01"
END: Final[str] = "2024-03-31"

# Simulates an asset that only joins the universe after January.
LATE_ENTRY_SYMBOL: Final[str] = "VALE3.SA"
LATE_ENTRY_DATE: Final[str] = "2024-02-01"

MOMENTUM_WINDOW: Final[int] = 20


def load_aligned_bars() -> pd.DataFrame:
    """
    Run the ingestion pipeline and return aligned long-format bars.

    Returns
    -------
    pd.DataFrame
        Bars indexed by ``MultiIndex(timestamp, symbol)``, aligned to the
        B3 calendar.
    """
    provider = YFinanceProvider(frequency="1d")
    bars = provider.get_bars(list(SYMBOLS), start=START, end=END)

    issues = check_bars(bars)
    print(f"check_bars: {int(issues.any(axis=1).sum())} row(s) flagged")

    aligned = align_to_calendar(bars, calendar=PandasMarketCalendarsCalendar("B3"))
    print(f"align_to_calendar: {len(bars)} -> {len(aligned)} rows")
    return aligned


def show_construction(panel: Panel) -> None:
    """Print the structural accessors of *panel*."""
    print("\n--- 1. Construction: Panel.from_bars() ---")
    print(panel)
    print(f"fields  : {panel.fields}")
    print(f"symbols : {panel.symbols}")
    print(f"shape   : {panel.shape}  (timestamps, symbols)")
    print(f"index   : {panel.index[0]} -> {panel.index[-1]} (tz={panel.index.tz})")


def show_field_access(panel: Panel) -> None:
    """Print one field and the missing values per symbol."""
    print("\n--- 2. Field access: panel.field() ---")
    close = panel.field("close")
    print(close.tail())

    print("\nNaN count per symbol (not observable: pre-listing or real gap):")
    print(close.isna().sum().to_string())

    try:
        panel.field("vwap")
    except KeyError as error:
        print(f"\nunknown field -> KeyError: {error}")


def show_wide_operations(panel: Panel) -> None:
    """Show one time-series and one cross-sectional operation on the wide format."""
    print("\n--- 3. Wide format: time-series (axis 0) and cross-sectional (axis 1) ---")
    close = panel.field("close")

    log_returns = np.log(close / close.shift(1))
    print("log returns, every symbol in one call:")
    print(log_returns.tail(3).round(4))

    momentum = log_returns.rolling(MOMENTUM_WINDOW).sum()
    print(f"\n{MOMENTUM_WINDOW}-session momentum ranked across symbols (1 = strongest):")
    print(momentum.rank(axis=1, ascending=False).tail(3))


def show_universe_mask(panel: Panel) -> Panel:
    """
    Build a point-in-time mask, apply it and show its effect.

    Returns
    -------
    Panel
        A new panel carrying the mask.
    """
    print("\n--- 4. Universe mask: panel.with_universe_mask() ---")
    print("default mask, sessions in the universe per symbol:")
    print(panel.universe_mask.sum().to_string())

    mask = panel.universe_mask
    mask.loc[mask.index < pd.Timestamp(LATE_ENTRY_DATE), LATE_ENTRY_SYMBOL] = False
    masked = panel.with_universe_mask(mask)

    print(f"\nwith {LATE_ENTRY_SYMBOL} entering on {LATE_ENTRY_DATE}:")
    print(masked.universe_mask.sum().to_string())
    print(f"original panel unchanged: {bool(panel.universe_mask.to_numpy().all())}")

    close = masked.field("close")
    momentum = np.log(close / close.shift(MOMENTUM_WINDOW))
    in_universe = momentum.where(masked.universe_mask)
    first_session = masked.index[MOMENTUM_WINDOW]
    print(f"\ncross-sectional rank on {first_session.date()}, universe members only:")
    print(in_universe.loc[[first_session]].rank(axis=1, ascending=False))
    return masked


def show_immutability(panel: Panel) -> None:
    """Show that writes by callers never reach the panel."""
    print("\n--- 5. Immutability (Copy-on-Write) ---")
    close = panel.field("close")
    original = close.iloc[0, 0]
    close.iloc[0, 0] = -1.0
    print(f"returned copy changed to : {close.iloc[0, 0]}")
    print(f"panel still holds        : {panel.field('close').iloc[0, 0]} (was {original})")

    try:
        panel.symbols = ("OTHER.SA",)  # type: ignore[misc]
    except AttributeError as error:
        print(f"reassigning a property -> AttributeError: {error}")


def show_validation(panel: Panel) -> None:
    """Show that an invalid mask is rejected at construction."""
    print("\n--- 6. Validation ---")
    float_mask = panel.universe_mask.astype("float64")
    try:
        panel.with_universe_mask(float_mask)
    except ValueError as error:
        print(f"float mask -> ValueError: {error}")

    short_mask = panel.universe_mask.iloc[:-1]
    try:
        panel.with_universe_mask(short_mask)
    except ValueError as error:
        print(f"short mask -> ValueError: {error}")


def show_fingerprint(panel: Panel, masked: Panel, aligned: pd.DataFrame) -> None:
    """Show that the fingerprint tracks content and nothing else."""
    print("\n--- 7. Fingerprint ---")
    rebuilt = Panel.from_bars(aligned)
    print(f"panel         : {panel.fingerprint[:16]}...")
    print(
        f"rebuilt       : {rebuilt.fingerprint[:16]}...  "
        f"same data -> equal: {rebuilt.fingerprint == panel.fingerprint}"
    )
    print(
        f"masked panel  : {masked.fingerprint[:16]}...  "
        f"different mask -> differs: {masked.fingerprint != panel.fingerprint}"
    )


def main() -> None:
    """Build a Panel from real data and walk through its public API."""
    pd.set_option("display.width", 120)

    aligned = load_aligned_bars()
    panel = Panel.from_bars(aligned)

    show_construction(panel)
    show_field_access(panel)
    show_wide_operations(panel)
    masked = show_universe_mask(panel)
    show_immutability(panel)
    show_validation(panel)
    show_fingerprint(panel, masked, aligned)


if __name__ == "__main__":
    main()
