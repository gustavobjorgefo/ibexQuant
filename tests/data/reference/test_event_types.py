# ibexQuant\tests\data\reference\test_event_types.py

"""Tests for the label tables of the corporate actions ledger (ADR 0013, H3.5)."""

from __future__ import annotations

import pytest

from ibexQuant.data.reference.event_types import (
    CREDITED_LABELS,
    EventType,
    UnknownLabelError,
    event_type,
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
