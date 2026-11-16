# ibexQuant\src\ibexQuant\data\dataset.py

"""
One entry point to the B3 base for research and backtests (ADR 0016).

Responsibility
--------------
Assemble, in one call, the ``Panel`` of a universe over a range: the members' bars by
``security_id`` from the provider, aligned to B3's own sessions, adjusted to an anchor,
with warmup history and the point-in-time universe mask. Around it, the lookups a study
needs: members on a date, period starts, events, exits and tickers.

Contract
--------
- Panel symbols are ``security_id`` values (H6.2); ``ticker_at``, ``ticker_history`` and
  ``label`` turn them into tickers for people.
- ``universe`` is a ``Universe``, an index code (``Index``) or a list of tickers
  (``Tickers``) (H6.3).
- ``warmup`` counts bars before ``start``; their mask is ``False`` (H6.4).
- One adjustment per panel; ``"raw"`` serves bars as traded (H6.5).
- A security gets a column when it is a member on at least one session of
  ``[start, end]``; its bars run from the first warmup bar to ``end`` (H6.6).
- Bars come from ``B3Provider.get_security_bars`` and are aligned to
  ``CotahistCalendar`` (H6.7, H6.8): a ``NaN`` inside the range is a real gap.
- ``check_bars`` findings are logged, never fixed here (ADR 0003).

What does not belong here
-------------------------
- Executing orders, holding positions, crediting cash, valuing exits: the backtest
  engine (P5, P6, P9–P12).
- Building any table: the ``ibexquant data`` commands do that.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final, Self

import pandas as pd

from ibexQuant.data.calendars.calendar import align_to_calendar
from ibexQuant.data.calendars.cotahist.calendar import CotahistCalendar, CotahistCalendarError
from ibexQuant.data.ingestion.b3.provider import ROUND_LOT_MARKET, B3Provider
from ibexQuant.data.ingestion.provider import DateLike
from ibexQuant.data.ingestion.validation import check_bars
from ibexQuant.data.panel import Panel
from ibexQuant.data.reference.corporate_actions import read_ledger
from ibexQuant.data.reference.errors import SecurityMasterError
from ibexQuant.data.reference.security_master import SecurityMaster
from ibexQuant.data.store import DataStore
from ibexQuant.data.universe.calendar import assign_periods
from ibexQuant.data.universe.constituents import read_index_exits
from ibexQuant.data.universe.errors import UniverseError
from ibexQuant.data.universe.mask import Index, Tickers, Universe, UniverseData, universe_mask

logger: logging.Logger = logging.getLogger(__name__)

RAW: Final[str] = "raw"
ADJUSTMENTS: Final[tuple[str, ...]] = (RAW, "splits", "total_return")
TICKER_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "security_id",
    "ticker",
    "first_session",
    "last_session",
)


class DatasetNotReadyError(FileNotFoundError):
    """A table the dataset needs was never built; the message names the command."""


class B3Dataset:
    """
    The B3 historical base as one object.

    Use :meth:`open` rather than the constructor: it checks the base first.

    Parameters
    ----------
    store : DataStore
        The data store holding the B3 base.

    Examples
    --------
    >>> ds = B3Dataset.open()  # doctest: +SKIP
    >>> panel = ds.panel("IBOV", "2019-01-01", "2019-12-31", warmup=252)  # doctest: +SKIP
    >>> ds.label(panel.field("close")).tail()  # doctest: +SKIP
    """

    def __init__(self, store: DataStore) -> None:
        self._store: DataStore = store
        self._calendar: CotahistCalendar = CotahistCalendar(store)
        self._provider: B3Provider = B3Provider(store)
        self._universe: UniverseData | None = None
        self._master: SecurityMaster | None = None

    @classmethod
    def open(cls, store: DataStore | None = None) -> Self:
        """
        Open the base and check the tables every call needs.

        Parameters
        ----------
        store : DataStore, optional
            Defaults to the configured data root (``DataStore.open()``).

        Raises
        ------
        DatasetNotReadyError
            If the bars, the security master or the universe were never built.
        """
        dataset = cls(store or DataStore.open())
        try:
            dataset.calendar.sessions  # noqa: B018 - reads and caches the sessions
            dataset._universe_data()
        except (CotahistCalendarError, SecurityMasterError, UniverseError) as error:
            raise DatasetNotReadyError(str(error)) from error
        return dataset

    # --- properties ---

    @property
    def store(self) -> DataStore:
        """The data store the dataset reads."""
        return self._store

    @property
    def calendar(self) -> CotahistCalendar:
        """B3 sessions as recorded in the base."""
        return self._calendar

    @property
    def provider(self) -> B3Provider:
        """The provider the bars come from."""
        return self._provider

    # --- panel ---

    def panel(
        self,
        universe: Universe | str | Sequence[str],
        start: DateLike,
        end: DateLike,
        *,
        adjustment: str = "total_return",
        anchor: DateLike | None = None,
        warmup: int = 0,
        fields: Sequence[str] | None = None,
    ) -> Panel:
        """
        Build the panel of a universe over ``[start, end]``.

        Parameters
        ----------
        universe : Universe, str or sequence of str
            A universe specification, an index code or a list of tickers.
        start, end : str or pandas.Timestamp
            Range of decisions; the mask is ``False`` outside it.
        adjustment : {"total_return", "splits", "raw"}, default "total_return"
            Events removed by the adjustment; ``"raw"`` for bars as traded.
        anchor : str or pandas.Timestamp, optional
            Date the adjusted prices are relative to; defaults to ``end``.
        warmup : int, default 0
            Bars before ``start`` to load as history.
        fields : sequence of str, optional
            ``open``, ``high``, ``low``, ``close``, ``volume``, ``traded_value``;
            OHLCV by default.

        Returns
        -------
        Panel
            Symbols are ``security_id`` values, the index the sessions from the
            first warmup bar to ``end``.

        Raises
        ------
        TypeError
            If ``universe`` is not a ``Universe``, a string or a sequence of strings.
        ValueError
            If ``adjustment`` is unknown, ``anchor`` is given with ``"raw"``,
            ``warmup`` is negative, the range has no session, or the universe has no
            member with bars in it.
        """
        spec = as_universe(universe)
        if adjustment not in ADJUSTMENTS:
            raise ValueError(f"adjustment must be one of {list(ADJUSTMENTS)}, got {adjustment!r}.")
        if adjustment == RAW and anchor is not None:
            raise ValueError(
                "anchor applies to adjusted bars only; drop it or pick an adjustment."
            )
        if warmup < 0:
            raise ValueError(f"warmup must be zero or positive, got {warmup}.")

        decisions = self._calendar.valid_sessions(start, end)
        if decisions.empty:
            raise ValueError(f"No B3 session between {start} and {end}.")
        first = self._calendar.offset(decisions[0], warmup)
        last = decisions[-1]

        members = universe_mask(spec, decisions, data=self._universe_data())
        if members.columns.empty:
            raise ValueError(
                f"{spec} has no member between {decisions[0].date()} and {last.date()}."
            )

        if adjustment == RAW:
            bars = self._provider.get_security_bars(
                list(members.columns), first, last, fields, adjusted=False
            )
        else:
            bars = self._provider.get_security_bars(
                list(members.columns), first, last, fields, anchor=anchor, adjustment=adjustment
            )
        self._log_findings(bars)
        aligned = align_to_calendar(bars, self._calendar, start=first, end=last)
        panel = Panel.from_bars(aligned)
        mask = members.reindex(index=panel.index, columns=list(panel.symbols), fill_value=False)
        return panel.with_universe_mask(mask.astype(bool))

    # --- lookups ---

    def constituents(self, index: str, as_of: DateLike) -> pd.DataFrame:
        """
        Return the members of ``index`` in the period in force on ``as_of``.

        Returns
        -------
        pandas.DataFrame
            ``security_id``, ``ticker`` (as in the portfolio), ``source`` and
            ``period``; empty when the index has no composition then. Members
            whose ticker did not resolve keep an empty ``security_id``.
        """
        data = self._universe_data()
        day = pd.DatetimeIndex([pd.Timestamp(as_of).normalize()])
        period = assign_periods(day, data.periods).iloc[0]
        table = data.constituents
        rows = table[
            (table["index"] == index) & (table["period"] == ("" if pd.isna(period) else period))
        ]
        columns = ["security_id", "ticker", "source", "period"]
        members: pd.DataFrame = rows[columns].sort_values("ticker").reset_index(drop=True)
        return members

    def rebalance_dates(self, index: str, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        """Return the starts of the periods with a composition of ``index`` in the range."""
        data = self._universe_data()
        held = set(data.constituents.loc[data.constituents["index"] == index, "period"])
        periods = data.periods[data.periods["period"].isin(held)]
        starts = pd.DatetimeIndex(periods["effective_from"])
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        return starts[(starts >= first) & (starts <= last)]

    def events(self, security_id: str) -> pd.DataFrame:
        """Return the ledger events of a security, by ex-date."""
        events, _ = read_ledger(self._store)
        rows = events[events["security_id"].astype(str) == security_id]
        return rows.sort_values("ex_date", kind="stable").reset_index(drop=True)

    def exits(self, index: str, start: DateLike, end: DateLike) -> pd.DataFrame:
        """Return the members of ``index`` that stop trading in a period overlapping the range."""
        data = self._universe_data()
        first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        periods = data.periods[
            (data.periods["effective_from"] <= last) & (data.periods["effective_to"] >= first)
        ]
        table = read_index_exits(self._store)
        rows = table[(table["index"] == index) & table["period"].isin(periods["period"])]
        return rows.reset_index(drop=True)

    # --- tickers ---

    def ticker_at(self, security_ids: str | Sequence[str], as_of: DateLike) -> pd.Series:
        """
        Return the ticker of each security on ``as_of``.

        A security that stopped trading before ``as_of`` gets its last ticker; one
        not listed yet gets ``<NA>``.

        Raises
        ------
        SecurityNotFoundError
            If a ``security_id`` is not in the master.
        """
        ids = [security_ids] if isinstance(security_ids, str) else list(security_ids)
        day = pd.Timestamp(as_of).normalize()
        master = self._security_master()
        tickers: list[str | None] = []
        for security in ids:
            spans = master.spans(security)
            spans = spans[
                (spans["market_type"] == ROUND_LOT_MARKET) & (spans["first_session"] <= day)
            ]
            if spans.empty:
                tickers.append(None)
                continue
            current = spans[spans["last_session"] >= day]
            chosen = (
                current
                if not current.empty
                else spans[spans["last_session"] == spans["last_session"].max()]
            )
            tickers.append(str(chosen.sort_values("first_session").iloc[-1]["ticker"]))
        return pd.Series(
            tickers, index=pd.Index(ids, name="security_id"), dtype="string", name="ticker"
        )

    def ticker_history(self, security_ids: str | Sequence[str]) -> pd.DataFrame:
        """
        Return every round-lot ticker each security carried, in time order.

        Raises
        ------
        SecurityNotFoundError
            If a ``security_id`` is not in the master.
        """
        ids = [security_ids] if isinstance(security_ids, str) else list(security_ids)
        master = self._security_master()
        frames = []
        for security in ids:
            spans = master.spans(security)
            frames.append(spans[spans["market_type"] == ROUND_LOT_MARKET])
        table = pd.concat(frames, ignore_index=True)
        return table[list(TICKER_HISTORY_COLUMNS)].reset_index(drop=True)

    def label(self, frame: pd.DataFrame, as_of: DateLike | None = None) -> pd.DataFrame:
        """
        Return a copy of ``frame`` with its ``security_id`` columns renamed to tickers.

        Parameters
        ----------
        frame : pandas.DataFrame
            Columns are ``security_id`` values (a ``Panel`` field, a mask, weights).
        as_of : str or pandas.Timestamp, optional
            Date of the tickers; defaults to the frame's last index value.

        Notes
        -----
        Two columns with the same ticker on ``as_of`` (a reused ticker) are told apart
        by their ``security_id`` in parentheses.
        """
        day = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(frame.index.max())
        tickers = self.ticker_at([str(column) for column in frame.columns], day)
        names = tickers.fillna(pd.Series(tickers.index, index=tickers.index, dtype="string"))
        repeated = names.duplicated(keep=False)
        names = names.where(
            ~repeated, names + " (" + names.index.to_series().astype("string") + ")"
        )
        labelled = frame.copy()
        labelled.columns = pd.Index(names.tolist(), name="ticker")
        return labelled

    # --- helpers ---

    def _universe_data(self) -> UniverseData:
        if self._universe is None:
            self._universe = UniverseData(self._store)
        return self._universe

    def _security_master(self) -> SecurityMaster:
        if self._master is None:
            self._master = self._universe_data().master
        return self._master

    @staticmethod
    def _log_findings(bars: pd.DataFrame) -> None:
        findings = check_bars(bars)
        flagged = findings[findings.any(axis=1)]
        if not flagged.empty:
            counts = ", ".join(
                f"{rule} {int(count)}" for rule, count in flagged.sum().items() if count
            )
            logger.warning("check_bars flagged %d bar(s): %s", len(flagged), counts)


def as_universe(universe: Universe | str | Sequence[str]) -> Universe:
    """
    Turn the ``universe`` argument into a ``Universe`` (H6.3).

    Raises
    ------
    TypeError
        If it is neither a ``Universe``, a string nor a sequence of strings.
    """
    if isinstance(universe, Universe):
        return universe
    if isinstance(universe, str):
        return Index(universe)
    if isinstance(universe, Sequence) and all(isinstance(item, str) for item in universe):
        return Tickers(list(universe))
    raise TypeError(
        f"universe must be a Universe, an index code or a list of tickers, got {universe!r}."
    )
