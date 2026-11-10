# ibexQuant\src\ibexQuant\data\reference\ex_sessions.py

"""
The ex-session skeleton of the corporate actions ledger.

Responsibility
--------------
Find, in ``bars_daily``, every session on which a round-lot share or unit
traded under a new ``distribution_number`` (ADR 0013, H3.2), and match
source events to those sessions by data-com or by distribution number
(H3.3).

Contract
--------
- Only round-lot rows (market type 10) whose ISIN type (characters 7–8) is
  ``AC``, ``UN`` or ``CD`` are read; pre-1995 internal codes are not ISINs
  and are ignored.
- When two tickers of one ISIN traded on one session, the row with the
  larger volume represents the session (as in ADR 0012, H2.14).
- A *run* is a stretch of consecutive traded sessions of one ISIN under one
  distribution number. Every run after an ISIN's first is an ex-session.
- Matching never guesses: an event is ``matched`` to one ex-session,
  ``no_ex_session`` when its ISIN traded on both sides of the data-com under
  one number, or ``not_traded`` when the trading history cannot place it.

What does not belong here
-------------------------
- Reading sources, merging events or deciding statuses — see the ledger build.
- Prices adjustment (H4).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

import pandas as pd

ROUND_LOT: Final[int] = 10
SKELETON_TYPES: Final[frozenset[str]] = frozenset({"AC", "UN", "CD"})
"""ISIN types (characters 7–8) whose ex-sessions are listed (H3.2)."""

BAR_COLUMNS: Final[tuple[str, ...]] = (
    "session_date",
    "ticker",
    "isin",
    "market_type",
    "distribution_number",
    "open",
    "close",
    "volume_shares",
)
"""Columns of ``bars_daily`` the skeleton reads."""

RUN_COLUMNS: Final[tuple[str, ...]] = (
    "isin",
    "distribution_number",
    "first_session",
    "last_session",
    "ticker",
    "first_open",
    "last_close",
)
EX_SESSION_COLUMNS: Final[tuple[str, ...]] = (
    "isin",
    "ticker",
    "session_date",
    "previous_session",
    "from_distribution",
    "to_distribution",
    "previous_close",
    "open",
    "open_ratio",
)
MATCH_COLUMNS: Final[tuple[str, ...]] = ("outcome", "ex_session")
_RUN_DTYPES: Final[dict[str, str]] = {
    "isin": "str",
    "distribution_number": "int64",
    "first_session": "datetime64[ns]",
    "last_session": "datetime64[ns]",
    "ticker": "str",
    "first_open": "float64",
    "last_close": "float64",
}


class MatchOutcome(StrEnum):
    """Result of placing one event on the skeleton (H3.3)."""

    MATCHED = "matched"
    NO_EX_SESSION = "no_ex_session"
    NOT_TRADED = "not_traded"


# --- Skeleton ---


def distribution_runs(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Group each ISIN's traded sessions into runs of one distribution number.

    Parameters
    ----------
    bars : pandas.DataFrame
        Rows of ``bars_daily`` with at least :data:`BAR_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`RUN_COLUMNS`, sorted by ``isin`` and ``first_session``.
        ``ticker`` and ``first_open`` describe the run's first session,
        ``last_close`` its last.
    """
    sessions = _sessions(bars)
    if sessions.empty:
        return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in _RUN_DTYPES.items()})
    isin = sessions["isin"]
    number = sessions["distribution_number"]
    starts = (isin != isin.shift()) | (number != number.shift())
    run_id = starts.cumsum()
    grouped = sessions.groupby(run_id, sort=False)
    runs = pd.DataFrame(
        {
            "isin": grouped["isin"].first(),
            "distribution_number": grouped["distribution_number"].first().astype("int64"),
            "first_session": grouped["session_date"].first(),
            "last_session": grouped["session_date"].last(),
            "ticker": grouped["ticker"].first(),
            "first_open": grouped["open"].first().astype("float64"),
            "last_close": grouped["close"].last().astype("float64"),
        }
    )
    return runs.reset_index(drop=True)


def find_ex_sessions(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Return every session on which an ISIN traded under a new distribution number.

    Parameters
    ----------
    bars : pandas.DataFrame
        Rows of ``bars_daily`` with at least :data:`BAR_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`EX_SESSION_COLUMNS`. ``open_ratio`` is the session's
        open over the previous traded session's close.
    """
    return ex_sessions_from_runs(distribution_runs(bars))


def ex_sessions_from_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Return the ex-sessions of :func:`distribution_runs` output."""
    if runs.empty:
        return pd.DataFrame({name: pd.Series(dtype=object) for name in EX_SESSION_COLUMNS})
    previous = runs.shift()
    changed = runs["isin"] == previous["isin"]
    ex = pd.DataFrame(
        {
            "isin": runs["isin"],
            "ticker": runs["ticker"],
            "session_date": runs["first_session"],
            "previous_session": previous["last_session"],
            "from_distribution": previous["distribution_number"],
            "to_distribution": runs["distribution_number"],
            "previous_close": previous["last_close"],
            "open": runs["first_open"],
        }
    )[changed]
    ex = ex.astype({"from_distribution": "int64", "to_distribution": "int64"})
    ex["open_ratio"] = ex["open"] / ex["previous_close"]
    return ex[list(EX_SESSION_COLUMNS)].reset_index(drop=True)


# --- Matching ---


def match_by_date(isins: pd.Series, last_cum_dates: pd.Series, runs: pd.DataFrame) -> pd.DataFrame:
    """
    Place events given by ISIN and data-com on the skeleton (H3.3).

    An event matches the change whose previous traded session is on or
    before the data-com and whose session is after it.

    Parameters
    ----------
    isins, last_cum_dates : pandas.Series
        One element per event, aligned.
    runs : pandas.DataFrame
        Output of :func:`distribution_runs`.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`MATCH_COLUMNS`, aligned with the inputs:
        ``outcome`` (:class:`MatchOutcome`) and ``ex_session`` (NaT unless
        matched).
    """
    events = _events(isins, last_cum_dates.astype("datetime64[ns]"), "key")
    # The run in force on the data-com: the latest one that started on or before it.
    current = _asof(events, runs, "first_session")
    following = _next_runs(runs)
    current = current.join(following, on="run_index")

    traded_after = current["last_session"] > current["key"]
    has_next = current["next_first_session"].notna()
    outcome = pd.Series(MatchOutcome.NOT_TRADED.value, index=current.index, dtype=object)
    outcome[current["run_index"].notna() & traded_after] = MatchOutcome.NO_EX_SESSION.value
    matched = current["run_index"].notna() & ~traded_after & has_next
    outcome[matched] = MatchOutcome.MATCHED.value
    return _result(outcome, current["next_first_session"].where(matched), isins.index)


def match_by_distribution(
    isins: pd.Series, numbers: pd.Series, runs: pd.DataFrame
) -> pd.DataFrame:
    """
    Place events given by ISIN and distribution number on the skeleton (H3.3).

    An event with number ``n`` matches the change from ``a`` to ``b`` with
    ``a <= n < b``.

    Parameters
    ----------
    isins, numbers : pandas.Series
        One element per event, aligned.
    runs : pandas.DataFrame
        Output of :func:`distribution_runs`.

    Returns
    -------
    pandas.DataFrame
        As :func:`match_by_date`; ``no_ex_session`` does not occur.
    """
    events = _events(isins, numbers.astype("int64"), "key")
    current = _asof(events, runs, "distribution_number")
    current = current.join(_next_runs(runs), on="run_index")
    matched = current["run_index"].notna() & current["next_first_session"].notna()
    outcome = pd.Series(MatchOutcome.NOT_TRADED.value, index=current.index, dtype=object)
    outcome[matched] = MatchOutcome.MATCHED.value
    return _result(outcome, current["next_first_session"].where(matched), isins.index)


# --- Helpers ---


def _sessions(bars: pd.DataFrame) -> pd.DataFrame:
    isin = bars["isin"].astype(str)
    keep = (bars["market_type"] == ROUND_LOT) & isin.str[6:8].isin(SKELETON_TYPES)
    rows = bars.loc[keep, list(BAR_COLUMNS)].assign(isin=isin[keep])
    rows = rows.sort_values(
        ["isin", "session_date", "volume_shares"], ascending=[True, True, False], kind="stable"
    )
    rows = rows.drop_duplicates(["isin", "session_date"], keep="first")
    return rows.reset_index(drop=True)


def _events(isins: pd.Series, keys: pd.Series, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"isin": isins.astype(str).to_numpy(), name: keys.to_numpy(), "order": range(len(isins))}
    )


def _asof(events: pd.DataFrame, runs: pd.DataFrame, on: str) -> pd.DataFrame:
    """Attach to each event the last run (per ISIN) whose ``on`` is <= the event key."""
    indexed = runs.assign(run_index=range(len(runs)))
    left = events.sort_values("key", kind="stable")
    right = indexed.sort_values(on, kind="stable").rename(columns={on: "run_key"})
    right["run_key"] = right["run_key"].astype(left["key"].dtype)
    right["isin"] = right["isin"].astype(left["isin"].dtype)
    joined = pd.merge_asof(
        left,
        right[["isin", "run_key", "run_index", "last_session"]],
        left_on="key",
        right_on="run_key",
        by="isin",
        direction="backward",
    )
    return joined.sort_values("order").set_index("order")


def _next_runs(runs: pd.DataFrame) -> pd.DataFrame:
    following = runs["first_session"].shift(-1).where(runs["isin"].shift(-1) == runs["isin"])
    return pd.DataFrame({"next_first_session": following.to_numpy()}, index=range(len(runs)))


def _result(outcome: pd.Series, ex_session: pd.Series, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome": outcome.map(MatchOutcome).to_numpy(),
            "ex_session": pd.to_datetime(ex_session).to_numpy(),
        },
        index=index,
    )
