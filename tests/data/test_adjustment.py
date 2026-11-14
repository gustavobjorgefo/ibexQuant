# ibexQuant\tests\data\test_adjustment.py

"""Factors and anchored adjustment (ADR 0014)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ibexQuant.data.adjustment import (
    FACTOR_COLUMNS,
    REJECTED_COLUMNS,
    AdjustmentMode,
    adjust,
    compute_factors,
    target_closes_from,
)
from ibexQuant.data.reference.corporate_actions import EVENT_COLUMNS

DAY = pd.Timestamp
SEC, OTHER, TARGET = "BRAAAAACNOR1", "BRBBBBACNOR2", "BRCCCCACNOR3"


# --- Fixtures ---


def factor(
    ex_date: str,
    f_total: float,
    share_ratio: float = 1.0,
    security: str = SEC,
) -> dict[str, Any]:
    return {
        "security_id": security,
        "isin": security,
        "ex_date": DAY(ex_date),
        "previous_close": 10.0,
        "cash_value": 0.0,
        "share_ratio": share_ratio,
        "f_total": f_total,
        "f_splits": 1.0 / share_ratio,
        "event_ids": "e",
    }


def factors(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(FACTOR_COLUMNS))


def bars(closes: dict[str, float], security: str = SEC) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": [DAY(day) for day in closes],
            "security_id": security,
            "open": list(closes.values()),
            "high": list(closes.values()),
            "low": list(closes.values()),
            "close": list(closes.values()),
            "volume": 100.0,
        }
    )


# A dividend of 1.0 on a close of 10.0 (f = 0.9), then a 2-for-1 split.
SERIES = {
    "2024-01-02": 10.0,
    "2024-01-03": 9.0,  # ex-dividend
    "2024-01-04": 9.0,
    "2024-01-05": 4.5,  # ex-split
    "2024-01-08": 4.6,
}
LEDGER = factors(factor("2024-01-03", 0.9), factor("2024-01-05", 0.5, share_ratio=2.0))


# --- adjust ---


def test_bars_at_and_after_the_anchor_stay_raw() -> None:
    out = adjust(bars(SERIES), LEDGER, DAY("2024-01-05"))
    after = out["session_date"] >= DAY("2024-01-05")
    assert out.loc[after, "close"].tolist() == [4.5, 4.6]


def test_factors_multiply_every_price_before_their_ex_date() -> None:
    out = adjust(bars(SERIES), LEDGER, DAY("2024-01-08"))
    assert np.allclose(out["close"], [10.0 * 0.9 * 0.5, 9.0 * 0.5, 9.0 * 0.5, 4.5, 4.6])
    assert np.allclose(out["open"], out["close"])
    assert np.allclose(out["high"], out["close"]) and np.allclose(out["low"], out["close"])


def test_returns_do_not_depend_on_the_anchor() -> None:
    early = adjust(bars(SERIES), LEDGER, DAY("2024-01-05"))["close"]
    late = adjust(bars(SERIES), LEDGER, DAY("2024-12-31"))["close"]
    first = early.pct_change().iloc[1:4]
    assert np.allclose(first, late.pct_change().iloc[1:4])


def test_the_return_across_a_dividend_reinvests_at_the_theoretical_price() -> None:
    out = adjust(bars(SERIES), LEDGER, DAY("2024-01-03"))["close"]
    assert out.iloc[1] / out.iloc[0] == pytest.approx(9.0 / (10.0 - 1.0))


def test_splits_mode_removes_only_share_events() -> None:
    out = adjust(bars(SERIES), LEDGER, DAY("2024-01-08"), AdjustmentMode.SPLITS)
    assert np.allclose(out["close"], [5.0, 4.5, 4.5, 4.5, 4.6])


def test_an_event_after_the_anchor_changes_nothing() -> None:
    before = adjust(bars(SERIES), LEDGER.iloc[:1], DAY("2024-01-04"))
    with_later = adjust(bars(SERIES), LEDGER, DAY("2024-01-04"))
    pd.testing.assert_frame_equal(before, with_later)


def test_volume_takes_the_share_ratios_only() -> None:
    out = adjust(bars(SERIES), LEDGER, DAY("2024-01-08"))
    assert out["volume"].tolist() == [200.0, 200.0, 200.0, 100.0, 100.0]


def test_raw_mode_and_empty_bars_are_returned_as_they_are() -> None:
    raw = bars(SERIES)
    pd.testing.assert_frame_equal(adjust(raw, LEDGER, DAY("2024-01-08"), "raw"), raw)
    empty = raw.iloc[0:0]
    assert adjust(empty, LEDGER, DAY("2024-01-08")).empty


def test_securities_are_adjusted_independently() -> None:
    both = pd.concat([bars(SERIES), bars(SERIES, security=OTHER)], ignore_index=True)
    out = adjust(both, LEDGER, DAY("2024-01-08"))
    assert out.loc[out["security_id"] == OTHER, "close"].tolist() == list(SERIES.values())
    assert out.loc[out["security_id"] == SEC, "close"].iloc[0] == pytest.approx(4.5)


def test_columns_that_are_absent_are_not_required() -> None:
    only_close = bars(SERIES)[["session_date", "security_id", "close"]]
    out = adjust(only_close, LEDGER, DAY("2024-01-08"))
    assert list(out.columns) == ["session_date", "security_id", "close"]


def test_an_unknown_mode_is_an_error() -> None:
    with pytest.raises(ValueError, match="adjusted"):
        adjust(bars(SERIES), LEDGER, DAY("2024-01-08"), "adjusted")


# --- compute_factors ---


def event(
    event_type: str,
    *,
    cash: float | None = None,
    ratio: float | None = None,
    price: float | None = None,
    target: str | None = None,
    session: str = "2024-01-03",
    status: str = "confirmed",
    isin: str = SEC,
    event_id: str = "e1",
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(EVENT_COLUMNS)
    row.update(
        event_id=event_id,
        security_id=isin,
        isin=isin,
        ex_date=DAY(session),
        event_type=event_type,
        cash_amount=cash,
        share_ratio=ratio,
        subscription_price=price,
        target_isin=target,
        status=status,
        ex_session=DAY(session) if status != "review" else pd.NaT,
    )
    return row


def ledger(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(EVENT_COLUMNS))


EX = pd.DataFrame(
    {
        "isin": [SEC, SEC],
        "session_date": [DAY("2024-01-03"), DAY("2024-01-10")],
        "previous_close": [10.0, np.nan],
    }
)
SECURITY_OF = {SEC: SEC, OTHER: SEC, TARGET: TARGET}
NO_TARGETS = target_closes_from(
    pd.DataFrame(
        {
            "session_date": pd.Series(dtype="datetime64[ns]"),
            "isin": pd.Series(dtype="object"),
            "market_type": pd.Series(dtype="int64"),
            "close": pd.Series(dtype="float64"),
        }
    ),
    set(),
)


def one(events: pd.DataFrame, targets: pd.DataFrame = NO_TARGETS) -> pd.Series:
    kept, rejected, _ = compute_factors(events, EX, SECURITY_OF, targets)
    assert rejected.empty, rejected
    assert len(kept) == 1
    return kept.iloc[0]


def test_cash_events_on_one_session_are_summed_gross() -> None:
    row = one(
        ledger(
            event("CASH_DIVIDEND", cash=0.6, event_id="a"),
            event("INTEREST_ON_CAPITAL", cash=0.4, event_id="b"),
        )
    )
    assert row["cash_value"] == pytest.approx(1.0)
    assert row["f_total"] == pytest.approx(0.9)
    assert row["f_splits"] == 1.0
    assert row["event_ids"] == "a,b"


def test_share_events_divide_by_the_ratio() -> None:
    row = one(ledger(event("STOCK_BONUS", ratio=1.1)))
    assert row["f_total"] == pytest.approx(1 / 1.1)
    assert row["f_splits"] == pytest.approx(1 / 1.1)


def test_cash_and_split_on_one_session_combine() -> None:
    row = one(ledger(event("CASH_DIVIDEND", cash=1.0, event_id="a"), event("SPLIT", ratio=2.0)))
    assert row["f_total"] == pytest.approx(0.9 / 2.0)
    assert row["f_splits"] == pytest.approx(0.5)


def test_a_subscription_right_is_worth_its_theoretical_value() -> None:
    row = one(ledger(event("SUBSCRIPTION", ratio=0.25, price=8.0)))
    value = 0.25 * (10.0 - 8.0) / 1.25
    assert row["cash_value"] == pytest.approx(value)
    assert row["f_total"] == pytest.approx(1 - value / 10.0)


def test_a_right_above_the_price_is_worth_nothing() -> None:
    row = one(ledger(event("SUBSCRIPTION", ratio=0.25, price=12.0)))
    assert row["f_total"] == 1.0


def test_a_spin_off_is_valued_at_the_targets_first_close_in_time() -> None:
    market = pd.DataFrame(
        {
            "session_date": [DAY("2024-01-03"), DAY("2024-01-04"), DAY("2024-01-04")],
            "isin": [SEC, TARGET, TARGET],
            "market_type": [10, 10, 20],
            "close": [9.0, 5.0, 4.0],
        }
    )
    targets = target_closes_from(market, {TARGET})
    row = one(ledger(event("SPIN_OFF", ratio=0.2, target=TARGET)), targets)
    assert row["cash_value"] == pytest.approx(1.0)
    assert row["f_total"] == pytest.approx(0.9)


def test_a_spin_off_target_priced_too_late_rejects_the_session() -> None:
    days = pd.bdate_range("2024-01-03", periods=8)
    market = pd.DataFrame(
        {
            "session_date": list(days) + [days[-1]],
            "isin": [SEC] * 8 + [TARGET],
            "market_type": 10,
            "close": [9.0] * 8 + [5.0],
        }
    )
    targets = target_closes_from(market, {TARGET})
    kept, rejected, _ = compute_factors(
        ledger(event("SPIN_OFF", ratio=0.2, target=TARGET)), EX, SECURITY_OF, targets
    )
    assert kept.empty
    assert rejected["reason"].tolist() == ["no_target_price"]


def test_conversions_inside_the_chain_are_share_events_others_are_ignored() -> None:
    kept, _, ignored = compute_factors(
        ledger(
            event("CONVERSION", ratio=0.5, target=OTHER, event_id="a"),
            event("EXCHANGE", ratio=3.0, target=TARGET, event_id="b"),
            event("REDEMPTION", event_id="c"),
            event("OTHER", event_id="d"),
        ),
        EX,
        SECURITY_OF,
        NO_TARGETS,
    )
    assert kept["f_total"].tolist() == [pytest.approx(2.0)]
    assert kept["event_ids"].tolist() == ["a"]
    assert ignored == {"EXCHANGE": 1, "OTHER": 1, "REDEMPTION": 1}


def test_impossible_factors_are_rejected_with_the_reason() -> None:
    kept, rejected, _ = compute_factors(
        ledger(
            event("CASH_DIVIDEND", cash=10.0, event_id="a"),
            event("CASH_DIVIDEND", cash=0.1, session="2024-01-10", event_id="b"),
        ),
        EX,
        SECURITY_OF,
        NO_TARGETS,
    )
    assert kept.empty
    assert list(rejected.columns) == list(REJECTED_COLUMNS)
    assert rejected["reason"].tolist() == ["value_exceeds_price", "no_previous_close"]


def test_review_events_and_empty_ledgers_give_no_factor() -> None:
    kept, rejected, ignored = compute_factors(
        ledger(event("CASH_DIVIDEND", cash=0.1, status="review")), EX, SECURITY_OF, NO_TARGETS
    )
    assert kept.empty and rejected.empty and ignored == {}
    assert list(kept.columns) == list(FACTOR_COLUMNS)
