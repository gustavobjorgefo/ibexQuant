# ibexQuant\src\ibexQuant\data\universe\liquidity.py

"""
Liquidity ranking of B3 shares (ADR 0015, H5.6, H5.9).

Responsibility
--------------
Read the raw daily traded value of every round-lot security from the bars and
select, at a base session, the most traded shares over a lookback window.
This is the rule that stands in for the Ibovespa where B3 has no portfolio
(before 2003, May 2014 to May 2017), and the ``LiquidityRule`` universe.

Contract
--------
- Traded value is raw BRL, never adjusted (H5.9).
- Only shares and units are eligible: the first word of the COTAHIST
  specification is ``ON``, ``PN`` (any class letter) or ``UNT``. Funds, BDRs,
  receipts and rights are not.
- Ranking uses only sessions in ``(base − lookback, base]``: nothing after the
  base session. A security qualifies when it traded on at least
  ``min_presence`` of those sessions; qualified securities rank by the median
  daily traded value over the window, a session without trading counting as
  zero, ties broken by ``security_id``.
- A window that starts before the first session of the data has no ranking.
- The base session of a period is :data:`BASE_OFFSET` sessions before its
  start, as B3's 3rd preview is computed a few sessions ahead.

What does not belong here
-------------------------
- B3's negotiability index, the price floor and the free-float weighting of
  the methodology: the rule is a selection proxy, and its gap to the real
  index is measured and reported (``constituents``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from ibexQuant.data.ingestion.b3.bars_daily import CURATED_PARTS as BARS_PARTS
from ibexQuant.data.ingestion.b3.bars_daily import PARTITION_FILE
from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.store import DataStore
from ibexQuant.data.universe.errors import UniverseNotBuiltError

ROUND_LOT_MARKET: Final[int] = 10
BASE_OFFSET: Final[int] = 3
"""Sessions between the base session of a period and its start."""

ELIGIBLE_SPECIFICATION: Final[str] = r"^(?:ON|PN[A-Z]?|UNT)\b"
TRADED_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "security_id",
    "traded_value",
    "eligible",
)


@dataclass(frozen=True, slots=True)
class LiquiditySettings:
    """
    Parameters of the liquidity rule.

    Attributes
    ----------
    lookback_months : int
        Length of the window before the base session.
    min_presence : float
        Share of the window's sessions a security must have traded, in (0, 1].
    """

    lookback_months: int = 12
    min_presence: float = 0.95

    def __post_init__(self) -> None:
        if self.lookback_months < 1:
            raise ValueError(f"lookback_months must be at least 1, got {self.lookback_months}.")
        if not 0 < self.min_presence <= 1:
            raise ValueError(f"min_presence must be in (0, 1], got {self.min_presence}.")


IBOV_LIQUIDITY: Final[LiquiditySettings] = LiquiditySettings()
"""Ibovespa-like defaults: three portfolios of lookback, 95% presence (H5.9)."""


@dataclass(frozen=True, slots=True)
class TradedValues:
    """
    Daily traded value by security, sorted by session.

    Attributes
    ----------
    table : pandas.DataFrame
        Columns :data:`TRADED_COLUMNS`, one row per session and security that
        traded.
    calendar : pandas.DatetimeIndex
        Every session of the bars.
    """

    table: pd.DataFrame
    calendar: pd.DatetimeIndex


# --- Reading ---


def read_traded_values(store: DataStore, master: SecurityMaster) -> TradedValues:
    """
    Read the raw traded value of every round-lot security.

    Rows whose ISIN is not in the master are left out. A security trading
    under two tickers on one session counts its most traded row, as the
    provider does.

    Raises
    ------
    UniverseNotBuiltError
        If there is no ``bars_daily`` partition.
    """
    folder = store.curated(*BARS_PARTS)
    paths = sorted(folder.glob(f"year=*/{PARTITION_FILE}")) if folder.is_dir() else []
    if not paths:
        raise UniverseNotBuiltError(
            "No bars_daily partition. Run `ibexquant data ingest cotahist --all` first."
        )
    columns = ["session_date", "isin", "market_type", "specification", "traded_value_brl"]
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    rows = pd.concat(frames, ignore_index=True)
    calendar = pd.DatetimeIndex(rows["session_date"].unique()).sort_values()

    round_lot = rows[rows["market_type"] == ROUND_LOT_MARKET]
    table = master.table
    security_of = dict(
        zip(table["isin"].astype(str), table["security_id"].astype(str), strict=True)
    )
    traded = pd.DataFrame(
        {
            "session_date": pd.to_datetime(round_lot["session_date"]).astype("datetime64[ns]"),
            "security_id": round_lot["isin"].astype(str).map(security_of),
            "traded_value": round_lot["traded_value_brl"].astype("float64"),
            "eligible": round_lot["specification"]
            .astype(str)
            .str.strip()
            .str.contains(ELIGIBLE_SPECIFICATION, regex=True),
        }
    ).dropna(subset=["security_id"])
    traded = traded.groupby(["session_date", "security_id"], as_index=False, sort=True).agg(
        traded_value=("traded_value", "max"), eligible=("eligible", "any")
    )
    traded = traded.astype({"security_id": "string", "eligible": "bool"})
    return TradedValues(traded[list(TRADED_COLUMNS)].reset_index(drop=True), calendar)


# --- Selection ---


def base_session(calendar: pd.DatetimeIndex, start: pd.Timestamp) -> pd.Timestamp | None:
    """Return the session :data:`BASE_OFFSET` sessions before ``start``, if the data has it."""
    position = int(calendar.searchsorted(start)) - BASE_OFFSET
    return calendar[position] if position >= 0 else None


def select_by_liquidity(
    traded: TradedValues,
    base: pd.Timestamp,
    top: int,
    settings: LiquiditySettings = IBOV_LIQUIDITY,
) -> list[str]:
    """
    Return the ``top`` most traded eligible securities up to ``base``.

    Parameters
    ----------
    traded : TradedValues
        The output of :func:`read_traded_values`.
    base : pandas.Timestamp
        Last session the ranking may see.
    top : int
        Number of securities to keep; fewer if fewer qualify.
    settings : LiquiditySettings
        Lookback and minimum presence.

    Returns
    -------
    list of str
        ``security_id`` values, most traded first; empty when the window
        starts before the data.

    Raises
    ------
    ValueError
        If ``top`` is not positive.
    """
    if top < 1:
        raise ValueError(f"top must be at least 1, got {top}.")
    calendar = traded.calendar
    window_start = base - pd.DateOffset(months=settings.lookback_months)
    if calendar.empty or window_start < calendar[0]:
        return []
    sessions = calendar[(calendar > window_start) & (calendar <= base)]
    if sessions.empty:
        return []
    table = traded.table
    days = table["session_date"]
    window = table.iloc[
        int(days.searchsorted(sessions[0], side="left")) : int(
            days.searchsorted(sessions[-1], side="right")
        )
    ]
    window = window[window["eligible"]]
    if window.empty:
        return []

    count = len(sessions)
    presence = window.groupby("security_id", sort=True)["traded_value"].size()
    qualified = presence[presence >= settings.min_presence * count].index
    values = window[window["security_id"].isin(qualified)]
    medians = values.groupby("security_id", sort=True)["traded_value"].agg(
        lambda column: _median_with_zeros(column.to_numpy(dtype="float64"), count)
    )
    ranked = medians.reset_index().sort_values(
        ["traded_value", "security_id"], ascending=[False, True], kind="stable"
    )
    return ranked["security_id"].astype(str).head(top).tolist()


def _median_with_zeros(values: np.ndarray, count: int) -> float:
    """Median of ``values`` padded with zeros up to ``count`` sessions."""
    padded = np.concatenate([values, np.zeros(count - len(values))])
    return float(np.median(padded))
