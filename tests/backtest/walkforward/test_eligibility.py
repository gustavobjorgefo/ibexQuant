# ibexQuant\tests\backtest\walkforward\test_eligibility.py

"""Layer 3 of the walk-forward: eligibility rules (ADR 0017, B1.6)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from ibexQuant.backtest.walkforward import HasHistory, MinimumPresence, StillTrading, Window
from ibexQuant.data.dataset import B3Dataset

SEPTEMBER = ["BRLATE", "BROLDX", "BRPETR", "BRQUIT", "BRVALE"]


def dropped(reasons: pd.Series) -> dict[str, str]:
    return {str(security): str(reason) for security, reason in reasons.dropna().items()}


# --- HasHistory ---


def test_history_covers_the_training_window(
    ds: B3Dataset, window_for: Callable[[str], Window]
) -> None:
    # Training of October 2024 starts on 2024-08-01; LATE3 lists on 2024-09-02.
    reasons = HasHistory().apply(ds, window_for("2024-10"), SEPTEMBER, warmup=0)
    assert reasons.index.tolist() == SEPTEMBER
    assert reasons.dtype == "string"
    assert dropped(reasons) == {"BRLATE": "short_history"}
    # Training of November starts on 2024-09-02, LATE3's first session.
    assert dropped(HasHistory().apply(ds, window_for("2024-11"), SEPTEMBER, warmup=0)) == {}


def test_history_counts_the_warmup_in_bars(
    ds: B3Dataset, window_for: Callable[[str], Window]
) -> None:
    window = window_for("2024-11")
    assert dropped(HasHistory().apply(ds, window, SEPTEMBER, warmup=1)) == {
        "BRLATE": "short_history"
    }
    # The warmup reaches before the base: nobody has it.
    reasons = HasHistory().apply(ds, window, SEPTEMBER, warmup=200)
    assert set(dropped(reasons).values()) == {"short_history"}
    assert len(dropped(reasons)) == len(SEPTEMBER)


def test_history_before_the_decision(ds: B3Dataset, window_for: Callable[[str], Window]) -> None:
    # The decision of October is 2024-09-30; LATE3 has 20 sessions before it.
    window = window_for("2024-10")
    assert dropped(HasHistory(cover="test").apply(ds, window, SEPTEMBER, warmup=20)) == {}
    assert dropped(HasHistory(cover="test").apply(ds, window, SEPTEMBER, warmup=21)) == {
        "BRLATE": "short_history"
    }


# --- StillTrading ---


def test_still_trading(ds: B3Dataset, window_for: Callable[[str], Window]) -> None:
    # QUIT3 stops on 2024-10-15: traded at October's decision, gone at November's.
    assert dropped(StillTrading().apply(ds, window_for("2024-10"), SEPTEMBER, warmup=0)) == {}
    assert dropped(StillTrading().apply(ds, window_for("2024-11"), SEPTEMBER, warmup=0)) == {
        "BRQUIT": "stopped_trading"
    }


def test_spans_join_every_ticker(ds: B3Dataset, window_for: Callable[[str], Window]) -> None:
    # OLDX3 became NEWX3 on 2024-10-01: one security, history from 2024-03-01.
    reasons = HasHistory().apply(ds, window_for("2024-12"), ["BROLDX"], warmup=100)
    assert dropped(reasons) == {}


# --- MinimumPresence ---


def test_presence_over_the_training_window(
    ds: B3Dataset, window_for: Callable[[str], Window]
) -> None:
    # PAUS3 has no bar on 2024-09-02, inside October's training window (43 sessions).
    window = window_for("2024-10")
    securities = ["BRPAUS", "BRPETR"]
    assert dropped(MinimumPresence(1.0).apply(ds, window, securities, warmup=0)) == {
        "BRPAUS": "low_presence"
    }
    assert dropped(MinimumPresence(0.95).apply(ds, window, securities, warmup=0)) == {}


def test_presence_before_the_decision(ds: B3Dataset, window_for: Callable[[str], Window]) -> None:
    window = window_for("2024-10")
    rule = MinimumPresence(1.0, cover="test")
    assert dropped(rule.apply(ds, window, ["BRPAUS"], warmup=5)) == {}
    assert dropped(rule.apply(ds, window, ["BRPAUS"], warmup=25)) == {"BRPAUS": "low_presence"}


def test_presence_checks_its_arguments() -> None:
    for fraction in (0.0, 1.5):
        with pytest.raises(ValueError, match=r"fraction must be in \(0, 1\]"):
            MinimumPresence(fraction)
    with pytest.raises(ValueError, match="cover must be one of"):
        MinimumPresence(0.9, cover="all")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cover must be one of"):
        HasHistory(cover="all")  # type: ignore[arg-type]


def test_reprs() -> None:
    assert repr(HasHistory()) == "HasHistory(cover='train')"
    assert repr(StillTrading()) == "StillTrading()"
    assert repr(MinimumPresence(0.95, cover="test")) == "MinimumPresence(0.95, cover='test')"
