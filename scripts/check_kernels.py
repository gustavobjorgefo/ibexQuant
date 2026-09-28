# ibexQuant\scripts\check_kernels.py

"""
Manual, network-hitting walkthrough of every feature kernel on real data.

Not a test — a plain script for eyeballing each kernel on real Yahoo Finance
prices: its warm-up (leading NaN per asset), its latest values and, for one
kernel per family, agreement between batch and incremental execution. Run it
by hand after touching ``ibexQuant.features.kernels``; nothing here is
enforced by CI.

Moving averages of adjusted prices are not scale invariant (ADR 0005); they
are also shown as ``close / average`` ratios, the form valid for features.

Usage
-----
python scripts/check_kernels.py
"""

from __future__ import annotations

import math
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
from ibexQuant.features.kernels import (
    EwmaVolatilityState,
    IncrementalState,
    RollingZScoreState,
    SmaState,
    alpha_from_decay,
    alpha_from_halflife,
    alpha_from_span,
    cs_rank,
    cs_zscore,
    ema,
    ewma_volatility,
    exponential_lookback,
    historical_volatility,
    lag,
    log_returns,
    rolling_std,
    rolling_sum,
    rolling_zscore,
    simple_returns,
    sma,
)

SYMBOLS: Final[tuple[str, ...]] = ("PETR4.SA", "ITUB4.SA", "VALE3.SA", "SUZB3.SA")
START: Final[str] = "2023-01-01"
END: Final[str] = "2025-12-31"

WINDOW: Final[int] = 20
EMA_SPAN: Final[int] = 20
RISKMETRICS_DECAY: Final[float] = 0.94
TAIL_ROWS: Final[int] = 3

# Simulates an asset that only joins the universe in 2024.
LATE_ENTRY_SYMBOL: Final[str] = "SUZB3.SA"
LATE_ENTRY_DATE: Final[str] = "2024-01-01"


# --- Data ---


def load_panel() -> Panel:
    """
    Download, validate and align bars, and build a Panel.

    Returns
    -------
    Panel
        Daily bars for :data:`SYMBOLS` between :data:`START` and :data:`END`.
    """
    provider = YFinanceProvider(frequency="1d")
    bars = provider.get_bars(list(SYMBOLS), start=START, end=END)

    issues = check_bars(bars)
    print(f"check_bars: {int(issues.any(axis=1).sum())} row(s) flagged")

    aligned = align_to_calendar(bars, calendar=PandasMarketCalendarsCalendar("B3"))
    panel = Panel.from_bars(aligned)
    print(panel)
    return panel


# --- Display helpers ---


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def show(title: str, frame: pd.DataFrame, decimals: int = 4) -> None:
    """Print leading NaN per symbol (warm-up) and the latest rows of *frame*."""
    leading_nan: pd.Series = frame.notna().cummax().eq(False).sum()
    print(f"\n--- {title} ---")
    print(f"leading NaN (warm-up): {leading_nan.to_dict()}")
    print(frame.tail(TAIL_ROWS).round(decimals).to_string())


def compare_with_state(
    title: str,
    batch: pd.DataFrame,
    values: pd.DataFrame,
    state: IncrementalState,
) -> None:
    """Feed one column into a fresh *state* and compare its last value with *batch*."""
    symbol: str = SYMBOLS[0]
    streamed: float = math.nan
    for value in values[symbol]:
        streamed = state.update(float(value))
    batched: float = float(batch[symbol].iloc[-1])
    print(
        f"{title:<28} {symbol}  batch={batched:.10f}  incremental={streamed:.10f}  "
        f"diff={abs(batched - streamed):.1e}"
    )


# --- Walkthrough ---


def show_exponential_parameters() -> None:
    """Show alpha conversions and the warm-up they imply."""
    section("0. Exponential parameters (ADR 0008, D1.6-D1.7)")
    for label, alpha in [
        (f"span {EMA_SPAN}", alpha_from_span(EMA_SPAN)),
        ("halflife 10", alpha_from_halflife(10)),
        (f"decay {RISKMETRICS_DECAY} (RiskMetrics)", alpha_from_decay(RISKMETRICS_DECAY)),
    ]:
        print(f"{label:<28} alpha={alpha:.4f}  lookback={exponential_lookback(alpha)} bars")


def show_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Show return kernels and return log returns for later sections."""
    section("1. Returns")
    show("simple_returns", simple_returns(close))
    returns = log_returns(close)
    show("log_returns", returns)
    return returns


def show_windows(close: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Show fixed-window kernels."""
    section(f"2. Fixed windows (window = {WINDOW})")
    show(f"rolling_sum of log returns ({WINDOW}-bar momentum)", rolling_sum(returns, WINDOW))
    show("rolling_std of log returns", rolling_std(returns, WINDOW))
    average = sma(close, WINDOW)
    show("sma of close (price level, not scale invariant)", average, decimals=2)
    show("close / sma (scale invariant)", close / average)


def show_exponentials(close: pd.DataFrame) -> None:
    """Show exponential kernels."""
    section("3. Exponential kernels")
    alpha = alpha_from_span(EMA_SPAN)
    average = ema(close, alpha)
    show(f"ema of close, span {EMA_SPAN} (price level)", average, decimals=2)
    show("close / ema (scale invariant)", close / average)


def show_volatility(returns: pd.DataFrame) -> None:
    """Show volatility kernels, per period and annualized."""
    section("4. Volatility (per period; annualized = x sqrt(252))")
    historical = historical_volatility(returns, WINDOW)
    riskmetrics = ewma_volatility(returns, alpha_from_decay(RISKMETRICS_DECAY))
    show(f"historical_volatility, {WINDOW} bars", historical)
    show("ewma_volatility, RiskMetrics 0.94", riskmetrics)
    print("\nlatest annualized volatility:")
    annualized = pd.DataFrame(
        {
            "historical": historical.iloc[-1] * np.sqrt(252),
            "riskmetrics": riskmetrics.iloc[-1] * np.sqrt(252),
        }
    )
    print(annualized.round(4).to_string())


def show_lag_and_normalization(close: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Show lag (with the log-return identity) and the rolling z-score."""
    section("5. Lag and normalization")
    show("lag of close, 1 observation", lag(close, 1), decimals=2)

    log_close = np.log(close)
    identity_gap = (returns - (log_close - lag(log_close, 1))).abs().max().max()
    print(f"\nmax |log_returns - (log p - lag(log p, 1))| = {identity_gap:.1e}")

    zscore = rolling_zscore(close, WINDOW)
    show(f"rolling_zscore of close, {WINDOW} bars", zscore)
    bound = (WINDOW - 1) / math.sqrt(WINDOW)
    print(f"max |z| observed = {zscore.abs().max().max():.3f}  (bound = {bound:.3f})")


def show_cross_sectional(panel: Panel, returns: pd.DataFrame) -> None:
    """Show cross-sectional kernels with and without a universe mask."""
    section("6. Cross-sectional")
    momentum = rolling_sum(returns, WINDOW)
    show(f"cs_rank of {WINDOW}-bar momentum (0 = weakest, 1 = strongest)", cs_rank(momentum))
    show(f"cs_zscore of {WINDOW}-bar momentum", cs_zscore(momentum))

    mask = panel.universe_mask
    mask.loc[mask.index < pd.Timestamp(LATE_ENTRY_DATE), LATE_ENTRY_SYMBOL] = False
    masked = panel.with_universe_mask(mask)
    ranked = cs_rank(momentum, masked.universe_mask)

    last_2023 = ranked.loc[ranked.index < pd.Timestamp(LATE_ENTRY_DATE)].iloc[[-1]]
    print(f"\ncs_rank with {LATE_ENTRY_SYMBOL} outside the universe until {LATE_ENTRY_DATE}:")
    print(last_2023.round(4).to_string())


def show_parity(close: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Compare batch and incremental execution on the last row, one kernel per family."""
    section("7. Batch vs incremental (last value, full history)")
    compare_with_state("sma", sma(close, WINDOW), close, SmaState(WINDOW))
    compare_with_state(
        "rolling_zscore", rolling_zscore(close, WINDOW), close, RollingZScoreState(WINDOW)
    )
    alpha = alpha_from_decay(RISKMETRICS_DECAY)
    compare_with_state(
        "ewma_volatility",
        ewma_volatility(returns, alpha),
        returns,
        EwmaVolatilityState(alpha),
    )


def main() -> None:
    """Download real data and walk through every kernel."""
    pd.set_option("display.width", 140)

    panel = load_panel()
    close = panel.field("close")

    show_exponential_parameters()
    returns = show_returns(close)
    show_windows(close, returns)
    show_exponentials(close)
    show_volatility(returns)
    show_lag_and_normalization(close, returns)
    show_cross_sectional(panel, returns)
    show_parity(close, returns)


if __name__ == "__main__":
    main()
