# ibexQuant\src\ibexQuant\data\universe\exits.py

"""
Index members that stop trading inside their period (ADR 0015, H5.8).

Responsibility
--------------
List, for every index composition, the members whose security's last
round-lot session falls before the end of the period, with the reason the
ledger gives for it.

Contract
--------
- A member exits when the last session of its security (over every ticker it
  carried) is before ``effective_to``; ``last_session`` is that session, which
  may precede the period's start for a security that stopped before it.
- The reason is the ledger event of type ``CONVERSION``, ``EXCHANGE``,
  ``REDEMPTION`` or ``OTHER`` closest to the last session within
  :data:`EVENT_WINDOW_DAYS` (by last cum date, or ex date when there is
  none); ``no_event`` when the ledger has none.
- Exits are recorded, not priced: valuing a holding that stops trading is the
  backtest engine's decision (pending item P10).

What does not belong here
-------------------------
- Prices or returns of exiting securities.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from ibexQuant.data.reference.event_types import EventType
from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.universe.liquidity import ROUND_LOT_MARKET

EXIT_COLUMNS: Final[tuple[str, ...]] = (
    "index",
    "period",
    "ticker",
    "security_id",
    "effective_to",
    "last_session",
    "reason",
    "event_id",
)
EXIT_EVENT_TYPES: Final[dict[str, str]] = {
    EventType.CONVERSION.value: "conversion",
    EventType.EXCHANGE.value: "exchange",
    EventType.REDEMPTION.value: "redemption",
    EventType.OTHER.value: "other",
}
NO_EVENT: Final[str] = "no_event"
EVENT_WINDOW_DAYS: Final[int] = 31


def last_sessions(master: SecurityMaster) -> pd.Series:
    """Return the last round-lot session of every security, by ``security_id``."""
    table = master.table
    round_lot = table[table["market_type"] == ROUND_LOT_MARKET]
    last = round_lot.groupby(round_lot["security_id"].astype(str))["last_session"].max()
    return pd.to_datetime(last).astype("datetime64[ns]")


def find_exits(
    constituents: pd.DataFrame,
    periods: pd.DataFrame,
    last: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    List the members that stop trading before their period ends.

    Parameters
    ----------
    constituents : pandas.DataFrame
        The constituents table.
    periods : pandas.DataFrame
        The periods table.
    last : pandas.Series
        Last session by ``security_id`` (:func:`last_sessions`).
    events : pandas.DataFrame
        The ledger events (``read_ledger``); may be empty.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`EXIT_COLUMNS`, sorted by index, period and ticker.
    """
    members = constituents.dropna(subset=["security_id"]).merge(
        periods[["period", "effective_to"]], on="period", how="inner"
    )
    members = members.assign(last_session=members["security_id"].astype(str).map(last))
    exits = members[members["last_session"] < members["effective_to"]].copy()
    exits = exits.reset_index(drop=True)

    reason = pd.Series(NO_EVENT, index=exits.index, dtype="object")
    event_id = pd.Series(None, index=exits.index, dtype="object")
    candidates = events[events["event_type"].astype(str).isin(EXIT_EVENT_TYPES)]
    if not exits.empty and not candidates.empty:
        dated = candidates.assign(
            event_date=pd.to_datetime(candidates["last_cum_date"]).fillna(
                pd.to_datetime(candidates["ex_date"])
            ),
            security_id=candidates["security_id"].astype(str),
        )
        probe = exits[["security_id", "last_session"]].astype({"security_id": str})
        probe = probe.reset_index(names="row")
        hits = probe.merge(dated, on="security_id", how="inner")
        hits = hits.assign(gap=(hits["event_date"] - hits["last_session"]).abs())
        hits = hits[hits["gap"] <= pd.Timedelta(days=EVENT_WINDOW_DAYS)]
        nearest = hits.sort_values(["row", "gap", "event_id"]).drop_duplicates("row")
        reason.loc[nearest["row"]] = nearest["event_type"].astype(str).map(EXIT_EVENT_TYPES).values
        event_id.loc[nearest["row"]] = nearest["event_id"].astype(str).values
    exits["reason"] = reason
    exits["event_id"] = event_id
    exits = exits.sort_values(["index", "period", "ticker"]).reset_index(drop=True)
    return exits[list(EXIT_COLUMNS)].astype(
        {
            "index": "string",
            "period": "string",
            "ticker": "string",
            "security_id": "string",
            "effective_to": "datetime64[ns]",
            "last_session": "datetime64[ns]",
            "reason": "string",
            "event_id": "string",
        }
    )
