# ibexQuant\tests\backtest\walkforward\test_folds.py

"""Layer 4 of the walk-forward: folds and their data (ADR 0017, B1.7, B1.8)."""

from __future__ import annotations

import pandas as pd
import pytest

from ibexQuant.backtest import equal_weight, simulate_weights
from ibexQuant.backtest.walkforward import (
    HasHistory,
    IndexComposition,
    MinimumPresence,
    StaticUniverse,
    StillTrading,
    WalkForward,
    WalkForwardError,
    WalkForwardSchedule,
)
from ibexQuant.data.dataset import B3Dataset
from ibexQuant.data.universe import Index, Securities, Tickers


def schedule(start: str, end: str, test_months: int = 1) -> WalkForwardSchedule:
    return WalkForwardSchedule(start, end, train_months=2, test_months=test_months)


def excluded(fold_excluded: pd.DataFrame) -> list[tuple[object, str, str]]:
    return list(fold_excluded.itertuples(index=False, name=None))


# --- folds ---


def test_index_folds(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-11"), IndexComposition("IBOV"), warmup=0)
    october, november = walk.folds(ds)
    assert october.securities == ("BROLDX", "BRPETR", "BRQUIT", "BRVALE")
    assert excluded(october.excluded) == [
        (pd.NA, "GONE3", "unresolved"),
        ("BRLATE", "LATE3", "short_history"),
    ]
    assert october.universe == Securities(["BROLDX", "BRPETR", "BRQUIT", "BRVALE"])
    assert (october.index, october.period, october.source) == ("IBOV", "2024-09", "third_preview")
    # In November LATE3 has its history and QUIT3 has stopped; NEWX3 is OLDX3 renamed.
    assert november.securities == ("BRLATE", "BROLDX", "BRPETR", "BRVALE")
    assert excluded(november.excluded)[1:] == [("BRQUIT", "QUIT3", "stopped_trading")]
    assert november.excluded.dtypes.eq("string").all()


def test_split_cuts_at_the_quadrimester(ds: B3Dataset) -> None:
    walk = WalkForward(
        schedule("2024-08", "2024-09", test_months=2), IndexComposition("IBOV"), warmup=0
    )
    august, september = walk.folds(ds)
    assert (august.period, september.period) == ("2024-05", "2024-09")
    assert (august.window.test_start, august.window.test_end) == (
        pd.Timestamp("2024-08-01"),
        pd.Timestamp("2024-08-30"),
    )
    # The piece of September has its own decision session and training window.
    assert september.window.decision_start == pd.Timestamp("2024-08-30")
    assert september.window.train_start == pd.Timestamp("2024-07-01")
    assert september.window.test_end == pd.Timestamp("2024-09-30")


def test_hold_keeps_the_first_composition(ds: B3Dataset) -> None:
    walk = WalkForward(
        schedule("2024-08", "2024-09", test_months=2),
        IndexComposition("IBOV", at_rebalance="hold"),
        warmup=0,
    )
    (fold,) = walk.folds(ds)
    assert fold.period == "2024-05"
    assert fold.securities == ("BRPETR",)
    assert fold.window.test_end == pd.Timestamp("2024-09-30")


def test_static_folds_keep_their_mask(ds: B3Dataset) -> None:
    walk = WalkForward(
        schedule("2024-10", "2024-10"), StaticUniverse(["PETR4", "VALE3"]), warmup=0
    )
    (fold,) = walk.folds(ds)
    assert fold.securities == ("BRPETR", "BRVALE")
    assert fold.universe == Securities(["BRPETR", "BRVALE"]) & Tickers(["PETR4", "VALE3"])
    assert fold.excluded.empty
    assert fold.excluded.columns.tolist() == ["security_id", "ticker", "reason"]
    assert fold.index is None


def test_rules_run_in_order_on_what_is_left(ds: B3Dataset) -> None:
    rules = (StillTrading(), MinimumPresence(1.0), HasHistory())
    walk = WalkForward(schedule("2024-10", "2024-10"), IndexComposition("IBXX"), rules, warmup=0)
    (fold,) = walk.folds(ds)
    assert excluded(fold.excluded) == [
        (pd.NA, "DUPL3", "unresolved"),
        ("BRPAUS", "PAUS3", "low_presence"),
    ]
    assert fold.securities == ("BRPETR", "BRSTOP")


def test_no_rule_keeps_every_candidate(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-10"), IndexComposition("IBOV"), (), warmup=0)
    (fold,) = walk.folds(ds)
    assert len(fold.securities) == 5
    assert excluded(fold.excluded) == [(pd.NA, "GONE3", "unresolved")]


def test_a_fold_without_a_security_raises(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-10"), StaticUniverse(["LATE3"]), warmup=0)
    message = (
        r"No eligible security in the window testing 2024-10-01 to 2024-10-31: "
        r"1 candidate\(s\) dropped \(short_history 1\)"
    )
    with pytest.raises(WalkForwardError, match=message):
        walk.folds(ds)


def test_rules_stop_when_nothing_is_left(ds: B3Dataset) -> None:
    walk = WalkForward(
        schedule("2024-10", "2024-10"),
        StaticUniverse(["LATE3"]),
        (HasHistory(), MinimumPresence(1.0)),
        warmup=0,
    )
    with pytest.raises(WalkForwardError, match=r"\(short_history 1\)"):
        walk.folds(ds)


# --- data ---


def test_load(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-10"), IndexComposition("IBOV"), warmup=5)
    (fold,) = walk.folds(ds)
    data = fold.load(ds, fields=["close"])
    window = fold.window
    assert data.train.symbols == fold.securities
    assert data.train.index[5] == window.train_start
    assert data.train.index[-1] == window.train_end
    assert not data.train.universe_mask.iloc[:5].to_numpy().any()
    assert data.test.index[5] == window.decision_start
    assert data.test.index[-1] == window.decision_end
    assert data.test.universe_mask.iloc[5:].to_numpy().all()


def test_load_gives_the_decisions(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-10"), IndexComposition("IBOV"), warmup=5)
    (fold,) = walk.folds(ds)
    data = fold.load(ds, fields=["close"])
    assert data.decisions[0] == fold.window.decision_start
    assert data.decisions[-1] == fold.window.decision_end
    assert len(data.decisions) == len(data.test.index) - 5


def test_decisions_without_bars(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-10"), IndexComposition("IBOV"), warmup=5)
    (fold,) = walk.folds(ds)
    data = fold.load(ds, fields=["close"])
    assert fold.decisions(ds.calendar).equals(data.decisions)
    assert fold.load_test(ds, fields=["close"]).index.equals(data.test.index)
    assert fold.load_train(ds, fields=["close"]).index.equals(data.train.index)


def test_load_raw(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-10"), StaticUniverse(["PETR4"]), warmup=0)
    (fold,) = walk.folds(ds)
    data = fold.load(ds, fields=["close"], adjustment="raw")
    assert data.train.symbols == data.test.symbols == ("BRPETR",)


# --- one simulation over the folds ---


def test_bars_cover_the_run(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-11"), IndexComposition("IBOV"), warmup=0)
    folds = walk.folds(ds)
    bars = WalkForward.bars(ds, folds)
    assert set(bars.symbols) == {s for fold in folds for s in fold.securities}
    assert bars.fields == ("open", "close")
    assert bars.index[0] == folds[0].window.decision_start
    assert bars.index[-1] == folds[-1].window.test_end
    with pytest.raises(ValueError, match="No fold to load bars for"):
        WalkForward.bars(ds, [])


def test_an_equal_weight_walk_forward(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2025-01"), IndexComposition("IBOV"), warmup=0)
    folds = walk.folds(ds)
    pieces = []
    for fold in folds:
        data = fold.load(ds, fields=["close"])
        pieces.append(equal_weight(data.test.universe_mask).loc[data.decisions])
    weights = pd.concat(pieces)
    result = simulate_weights(WalkForward.bars(ds, folds), weights, execution="next_close")
    assert result.returns.index[0] == folds[0].window.test_start
    assert result.returns.index[-1] == folds[-1].window.test_end
    assert result.returns.notna().all()
    total = result.contributions.sum(axis=1) - result.costs
    assert (result.returns - total).abs().max() < 1e-12
    # QUIT3 stops on 2024-10-15 inside October's fold: cash from the next session on.
    assert result.weights.loc["2024-10-16":, "BRQUIT"].eq(0).all()


def test_e1_repeats_the_loop_of_notebook_0002(ds: B3Dataset) -> None:
    # Equal weight bought at the close of each IBOV period's first session and held (B1.22).
    end = ds.calendar.sessions[-1]
    edges = [*ds.rebalance_dates("IBOV", "2024-01-01", end), end]
    pieces, rows = [], {}
    for entry, exit_ in zip(edges[:-1], edges[1:], strict=True):
        panel = ds.panel(Index("IBOV"), entry, exit_, fields=["close"])
        close = panel.field("close")
        held = close.columns[panel.universe_mask.loc[entry] & close.loc[entry].notna()]
        value = (close[held].ffill() / close.loc[entry, held]).mean(axis=1)
        pieces.append(value.pct_change().iloc[1:])
        rows[ds.calendar.offset(entry, 1)] = pd.Series(1 / len(held), index=held)
    reference = pd.concat(pieces)
    weights = pd.DataFrame(rows).T.fillna(0.0)
    bars = ds.panel(Securities(list(weights.columns)), weights.index[0], end, fields=["close"])
    result = simulate_weights(bars, weights, execution="next_close")
    compared = pd.concat([reference, result.returns], axis=1, join="inner")
    assert len(compared) == len(reference)
    assert (compared.iloc[:, 0] - compared.iloc[:, 1]).abs().max() < 1e-12


# --- summary, checks and repr ---


def test_summary(ds: B3Dataset) -> None:
    walk = WalkForward(schedule("2024-10", "2024-11"), IndexComposition("IBOV"), warmup=0)
    summary = WalkForward.summary(walk.folds(ds))
    assert summary.columns.tolist() == [
        "train_start",
        "decision_start",
        "test_start",
        "test_end",
        "index",
        "period",
        "source",
        "candidates",
        "securities",
        "unresolved",
        "short_history",
        "stopped_trading",
    ]
    assert summary["candidates"].tolist() == [6, 6]
    assert summary["securities"].tolist() == [4, 4]
    assert summary["short_history"].tolist() == [1, 0]
    assert summary["stopped_trading"].tolist() == [0, 1]


def test_summary_of_nothing() -> None:
    assert WalkForward.summary([]).empty


def test_warmup_is_checked() -> None:
    with pytest.raises(ValueError, match="warmup must be zero or positive, got -1"):
        WalkForward(schedule("2024-10", "2024-10"), IndexComposition("IBOV"), warmup=-1)


def test_repr() -> None:
    walk = WalkForward(schedule("2024-10", "2024-10"), IndexComposition("IBOV"), warmup=10)
    assert repr(walk) == (
        "WalkForward(WalkForwardSchedule(2024-10 to 2024-10, train=2m, test=1m, step=1m, "
        "expanding=False), IndexComposition('IBOV', at_rebalance='split'), "
        "eligibility=[HasHistory(cover='train'), StillTrading()], warmup=10)"
    )
