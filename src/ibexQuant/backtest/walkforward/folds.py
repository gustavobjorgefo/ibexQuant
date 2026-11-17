# ibexQuant\src\ibexQuant\backtest\walkforward\folds.py

"""
Layer 4 of the walk-forward: folds and their data (ADR 0017, B1.7, B1.8).

Responsibility
--------------
Assemble the three layers below into :class:`Fold` objects (a window, its eligible
securities and the ones dropped) and load each fold's training and test panels.

Contract
--------
- One fold per window of the schedule, or per piece when the universe rule cuts it; the
  pieces are built by ``schedule.window``, with the schedule's own rule.
- Members without a ``security_id`` are dropped as ``unresolved``; then each eligibility
  rule, in order, sees the securities the previous ones kept.
- A fold without an eligible security raises :class:`WalkForwardError`.
- ``Fold.load`` returns a training panel of the fold's securities over the training
  window, anchored at its end, and a test panel of the fold's universe over the decision
  sessions, anchored at the last one; both with ``warmup`` bars before them.

What does not belong here
-------------------------
- Fitting and signals: the strategy's.
- Joining the folds into one simulation: the engines'.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from ibexQuant.backtest.walkforward.eligibility import EligibilityRule, HasHistory, StillTrading
from ibexQuant.backtest.walkforward.errors import WalkForwardError
from ibexQuant.backtest.walkforward.schedule import MONTH, WalkForwardSchedule, Window
from ibexQuant.backtest.walkforward.universe import Selection, UniverseRule
from ibexQuant.data.calendars.calendar import TradingCalendar
from ibexQuant.data.dataset import B3Dataset
from ibexQuant.data.panel import Panel
from ibexQuant.data.universe import Securities, Universe

UNRESOLVED: Final[str] = "unresolved"
EXCLUDED_COLUMNS: Final[tuple[str, ...]] = ("security_id", "ticker", "reason")
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "train_start",
    "decision_start",
    "test_start",
    "test_end",
    "index",
    "period",
    "source",
    "candidates",
    "securities",
)
RAW: Final[str] = "raw"


@dataclass(frozen=True)
class FoldData:
    """
    The panels of one fold.

    Attributes
    ----------
    train : Panel
        The fold's securities over the training window, with ``warmup`` bars.
    test : Panel
        The fold's universe over its decision sessions, with ``warmup`` bars.
    """

    train: Panel
    test: Panel


@dataclass(frozen=True, eq=False)
class Fold:
    """
    One window, the securities it fits and trades, and the candidates it dropped.

    Attributes
    ----------
    window : Window
        Training, decision and test sessions; a piece when the universe rule cut it.
    securities : tuple of str
        Eligible ``security_id`` values.
    universe : Universe
        The test panel's universe.
    excluded : pandas.DataFrame
        ``security_id``, ``ticker`` and ``reason`` of each candidate dropped.
    warmup : int
        Bars of history before each range.
    index, period, source : str or None
        The index period used, when there is one.
    """

    window: Window
    securities: tuple[str, ...]
    universe: Universe
    excluded: pd.DataFrame = field(repr=False)
    warmup: int
    index: str | None = None
    period: str | None = None
    source: str | None = None

    def load(
        self,
        ds: B3Dataset,
        fields: Sequence[str] | None = None,
        adjustment: str = "total_return",
    ) -> FoldData:
        """
        Return the training and test panels of the fold.

        Parameters
        ----------
        ds : B3Dataset
            The dataset.
        fields : sequence of str, optional
            Fields of the panels; OHLCV by default.
        adjustment : {"total_return", "splits", "raw"}, default "total_return"
            As in ``ds.panel``; adjusted panels are anchored at the end of their range.
        """
        window = self.window
        adjusted = adjustment != RAW
        train = ds.panel(
            Securities(self.securities),
            window.train_start,
            window.train_end,
            adjustment=adjustment,
            anchor=window.train_end if adjusted else None,
            warmup=self.warmup,
            fields=fields,
        )
        test = ds.panel(
            self.universe,
            window.decision_start,
            window.decision_end,
            adjustment=adjustment,
            anchor=window.decision_end if adjusted else None,
            warmup=self.warmup,
            fields=fields,
        )
        return FoldData(train=train, test=test)


class WalkForward:
    """
    The four layers together: windows, universe, eligibility and data.

    Parameters
    ----------
    schedule : WalkForwardSchedule
        Layer 1.
    universe : UniverseRule
        Layer 2.
    eligibility : sequence of EligibilityRule, default (HasHistory(), StillTrading())
        Layer 3, applied in order; empty keeps every candidate.
    warmup : int
        Bars of history the strategy needs before each range.

    Raises
    ------
    ValueError
        If ``warmup`` is negative.

    Examples
    --------
    >>> walk = WalkForward(schedule, IndexComposition("IBXX"), warmup=252)  # doctest: +SKIP
    >>> folds = walk.folds(ds)  # doctest: +SKIP
    >>> data = folds[0].load(ds)  # doctest: +SKIP
    """

    def __init__(
        self,
        schedule: WalkForwardSchedule,
        universe: UniverseRule,
        eligibility: Sequence[EligibilityRule] = (HasHistory(), StillTrading()),
        *,
        warmup: int,
    ) -> None:
        if warmup < 0:
            raise ValueError(f"warmup must be zero or positive, got {warmup}.")
        self.schedule: WalkForwardSchedule = schedule
        self.universe: UniverseRule = universe
        self.eligibility: tuple[EligibilityRule, ...] = tuple(eligibility)
        self.warmup: int = warmup

    # --- folds ---

    def folds(self, ds: B3Dataset) -> list[Fold]:
        """
        Return the folds, in order.

        Raises
        ------
        WalkForwardError
            If a layer cannot build a window, or a fold has no eligible security.
        """
        calendar = ds.calendar
        return [
            self._fold(ds, piece)
            for window in self.schedule.windows(calendar)
            for piece in self._pieces(window, calendar)
        ]

    @staticmethod
    def summary(folds: Sequence[Fold]) -> pd.DataFrame:
        """Return one row per fold: dates, index period, candidates, securities and drops."""
        rows = [
            {
                "train_start": fold.window.train_start,
                "decision_start": fold.window.decision_start,
                "test_start": fold.window.test_start,
                "test_end": fold.window.test_end,
                "index": fold.index,
                "period": fold.period,
                "source": fold.source,
                "candidates": len(fold.securities) + len(fold.excluded),
                "securities": len(fold.securities),
                **fold.excluded["reason"].value_counts().to_dict(),
            }
            for fold in folds
        ]
        frame = pd.DataFrame(rows, columns=[*SUMMARY_COLUMNS, *_reasons_of(folds)])
        reasons = frame.columns[len(SUMMARY_COLUMNS) :]
        frame[reasons] = frame[reasons].fillna(0).astype(int)
        return frame

    def __repr__(self) -> str:
        return (
            f"WalkForward({self.schedule!r}, {self.universe!r}, "
            f"eligibility={list(self.eligibility)!r}, warmup={self.warmup})"
        )

    # --- helpers ---

    def _pieces(self, window: Window, calendar: TradingCalendar) -> list[Window]:
        cuts = self.universe.cuts(window)
        if not cuts:
            return [window]
        starts = [window.test_start.to_period(MONTH), *cuts]
        ends = [month - 1 for month in cuts] + [window.test_end.to_period(MONTH)]
        return [
            self.schedule.window(first, last, calendar)
            for first, last in zip(starts, ends, strict=True)
        ]

    def _fold(self, ds: B3Dataset, window: Window) -> Fold:
        selection = self.universe.select(ds, window)
        kept = list(selection.security_ids)
        dropped: list[pd.DataFrame] = []
        if selection.unresolved:
            dropped.append(
                pd.DataFrame(
                    {"security_id": pd.NA, "ticker": selection.unresolved, "reason": UNRESOLVED}
                )
            )
        for rule in self.eligibility:
            if not kept:
                break
            reasons = rule.apply(ds, window, kept, self.warmup)
            out = reasons[reasons.notna()]
            if not out.empty:
                tickers = ds.ticker_at(list(out.index), window.decision_start)
                dropped.append(
                    pd.DataFrame(
                        {
                            "security_id": list(out.index),
                            "ticker": tickers.tolist(),
                            "reason": out.tolist(),
                        }
                    )
                )
            kept = [security for security in kept if pd.isna(reasons[security])]
        excluded = _excluded(dropped)
        if not kept:
            counts = ", ".join(
                f"{reason} {count}" for reason, count in excluded["reason"].value_counts().items()
            )
            raise WalkForwardError(
                f"No eligible security in the window testing {window.test_start.date()} to "
                f"{window.test_end.date()}: {len(excluded)} candidate(s) dropped ({counts})."
            )
        return Fold(
            window=window,
            securities=tuple(kept),
            universe=_test_universe(kept, selection),
            excluded=excluded,
            warmup=self.warmup,
            index=selection.index,
            period=selection.period,
            source=selection.source,
        )


def _reasons_of(folds: Sequence[Fold]) -> list[str]:
    """Every drop reason of the folds, in order of first appearance."""
    reasons = (str(reason) for fold in folds for reason in fold.excluded["reason"])
    return list(dict.fromkeys(reasons))


def _excluded(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame({column: pd.Series(dtype="string") for column in EXCLUDED_COLUMNS})
    table = pd.concat(frames, ignore_index=True)
    return table.astype("string")[list(EXCLUDED_COLUMNS)]


def _test_universe(securities: list[str], selection: Selection) -> Universe:
    fixed = Securities(securities)
    return fixed if selection.universe is None else fixed & selection.universe
