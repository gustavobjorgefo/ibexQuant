# ibexQuant\tests\data\reference\test_event_types.py

"""Tests for the label tables of the corporate actions ledger (ADR 0013, H3.5)."""

from __future__ import annotations

import math

import pytest

from ibexQuant.data.reference.event_types import (
    CREDITED_LABELS,
    EventType,
    StockFactorError,
    UnknownLabelError,
    event_type,
    stock_event,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("DIVIDENDO", EventType.CASH_DIVIDEND),
        (" juros sobre capital próprio ", EventType.INTEREST_ON_CAPITAL),
        ("RESTITUIÇÃO DE CAPITAL", EventType.RETURN_OF_CAPITAL),
        ("LEILÃO DE FRAÇÕES", None),
    ],
)
def test_credited_labels_map_to_types(label: str, expected: EventType | None) -> None:
    assert event_type(label, CREDITED_LABELS, "credited") == expected


def test_an_unknown_label_names_the_source() -> None:
    with pytest.raises(UnknownLabelError, match="credited label 'BONUS'"):
        event_type("BONUS", CREDITED_LABELS, "credited")


@pytest.mark.parametrize(
    ("label", "factor", "other", "expected"),
    [
        ("DESDOBRAMENTO", 100.0, False, (EventType.SPLIT, 2.0)),
        ("DESDOBRAMENTO", 300.0, False, (EventType.SPLIT, 4.0)),
        ("BONIFICACAO", 30.0, False, (EventType.STOCK_BONUS, 1.3)),
        ("GRUPAMENTO", 0.1, False, (EventType.REVERSE_SPLIT, 0.1)),
        ("CIS RED CAP QTD", 92.61731232092, False, (EventType.REVERSE_SPLIT, 0.9261731232092)),
        ("CIS RED CAP", 7.215075382, True, (EventType.SPIN_OFF, 0.07215075382)),
        ("REST CAP ACOES", 33.2373453, True, (EventType.SPIN_OFF, 0.332373453)),
        ("INCORPORACAO", 100.0, True, (EventType.CONVERSION, 1.0)),
        ("GRUPAMENTO", 0.01, True, (EventType.CONVERSION, 0.01)),
    ],
)
def test_stock_events_give_type_and_share_ratio(
    label: str, factor: float, other: bool, expected: tuple[EventType, float]
) -> None:
    kind, ratio = stock_event(label, factor, other)
    assert kind == expected[0]
    assert ratio == pytest.approx(expected[1])


def test_a_redemption_has_no_ratio() -> None:
    kind, ratio = stock_event("RESG TOTAL RV", 100.0, False)
    assert kind == EventType.REDEMPTION
    assert math.isnan(ratio)


def test_keeping_the_share_is_not_an_event() -> None:
    kind, ratio = stock_event("CIS RED CAP", 100.0, False)
    assert kind is None
    assert math.isnan(ratio)


def test_keeping_the_share_with_another_factor_is_an_error() -> None:
    with pytest.raises(StockFactorError, match="INCORPORACAO"):
        stock_event("INCORPORACAO", 50.0, False)


def test_an_unknown_stock_combination_is_an_error() -> None:
    with pytest.raises(UnknownLabelError, match="another asset"):
        stock_event("DESDOBRAMENTO", 100.0, True)
