# ibexQuant\src\ibexQuant\data\reference\event_types.py

"""
Event types of the corporate actions ledger and the B3 labels that map to them.

Responsibility
--------------
Name every kind of event the ledger records (ADR 0013, H3.5) and translate
each label B3 publishes into one of them, so that an unknown label stops the
build instead of being guessed.

Contract
--------
- A label maps to an :class:`EventType`, or to ``None`` when it is
  deliberately not an event (a fraction auction pays for rounding, not for a
  right).
- Lookups are exact on the upper-cased, stripped label; accents matter.

What does not belong here
-------------------------
- Reading files — see the B3 readers in ``ibexQuant.data.ingestion.b3``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class EventType(StrEnum):
    """Kinds of event in the ledger (H3.5)."""

    CASH_DIVIDEND = "CASH_DIVIDEND"
    INTEREST_ON_CAPITAL = "INTEREST_ON_CAPITAL"
    INCOME = "INCOME"
    RETURN_OF_CAPITAL = "RETURN_OF_CAPITAL"
    SPLIT = "SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    STOCK_BONUS = "STOCK_BONUS"
    SUBSCRIPTION = "SUBSCRIPTION"
    EXCHANGE = "EXCHANGE"
    CONVERSION = "CONVERSION"
    REDEMPTION = "REDEMPTION"
    OTHER = "OTHER"


CASH_TYPES: Final[frozenset[EventType]] = frozenset(
    {
        EventType.CASH_DIVIDEND,
        EventType.INTEREST_ON_CAPITAL,
        EventType.INCOME,
        EventType.RETURN_OF_CAPITAL,
        EventType.REDEMPTION,
    }
)
"""Types that carry ``cash_amount``."""

CREDITED_LABELS: Final[dict[str, EventType | None]] = {
    "DIVIDENDO": EventType.CASH_DIVIDEND,
    "JUROS SOBRE CAPITAL PRÓPRIO": EventType.INTEREST_ON_CAPITAL,
    "RENDIMENTO": EventType.INCOME,
    "AMORTIZAÇÃO": EventType.RETURN_OF_CAPITAL,
    "RESTITUIÇÃO DE CAPITAL": EventType.RETURN_OF_CAPITAL,
    "RESGATE RENDA VARIÁVEL": EventType.REDEMPTION,
    "LEILÃO DE FRAÇÕES": None,
}
"""Labels of the *Earnings credited* report (65 files, 2026-06 to 2026-11)."""


class UnknownLabelError(KeyError):
    """A source label has no entry in the mapping; read it once and add it."""


def event_type(label: str, mapping: dict[str, EventType | None], source: str) -> EventType | None:
    """
    Return the event type of a label, or ``None`` for labels that are not events.

    Parameters
    ----------
    label : str
        The label as published.
    mapping : dict
        One of the label tables of this module.
    source : str
        Source name, for the error message.

    Raises
    ------
    UnknownLabelError
        If the label is not in the mapping.
    """
    key = label.strip().upper()
    if key not in mapping:
        raise UnknownLabelError(f"Unknown {source} label {label!r}; map it in event_types.py.")
    return mapping[key]
