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
- B3's stock-event factor is a percentage, except for ``GRUPAMENTO``, where
  it is the multiplier itself (H3.5). :func:`stock_event` turns a label, a
  factor and whether the holder received another asset into a type and a
  share ratio.

What does not belong here
-------------------------
- Reading files — see the B3 readers in ``ibexQuant.data.ingestion.b3``.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    SPIN_OFF = "SPIN_OFF"
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


LISTED_CASH_LABELS: Final[dict[str, EventType | None]] = {
    "DIVIDENDO": EventType.CASH_DIVIDEND,
    "JRS CAP PROPRIO": EventType.INTEREST_ON_CAPITAL,
    "RENDIMENTO": EventType.INCOME,
    "REST CAP DIN": EventType.RETURN_OF_CAPITAL,
}
"""Cash labels of B3's listed-company service (snapshot 2026-11-10)."""

SUBSCRIPTION_LABELS: Final[dict[str, EventType | None]] = {
    "SUBSCRICAO": EventType.SUBSCRIPTION,
    "PRIORIDADE SUBS": EventType.SUBSCRIPTION,
    "SUBS C/ RENUNC": EventType.SUBSCRIPTION,
}
"""Subscription labels of B3's listed-company service (snapshot 2026-11-10)."""


@dataclass(frozen=True, slots=True)
class _StockRule:
    event: EventType | None
    ratio: str  # "plus_percent", "percent", "multiplier", "none" or "keep"


_KEEP: Final[_StockRule] = _StockRule(None, "keep")
STOCK_RULES: Final[dict[tuple[str, bool], _StockRule]] = {
    # (label, holder received another asset) -> rule
    ("DESDOBRAMENTO", False): _StockRule(EventType.SPLIT, "plus_percent"),
    ("BONIFICACAO", False): _StockRule(EventType.STOCK_BONUS, "plus_percent"),
    ("GRUPAMENTO", False): _StockRule(EventType.REVERSE_SPLIT, "multiplier"),
    ("CIS RED CAP QTD", False): _StockRule(EventType.REVERSE_SPLIT, "percent"),
    ("REST CAP C/ RED", False): _StockRule(EventType.REVERSE_SPLIT, "percent"),
    ("CIS RED CAP", False): _KEEP,
    ("INCORPORACAO", False): _KEEP,
    ("REST CAP ACOES", False): _KEEP,
    ("RESG TOTAL RV", False): _StockRule(EventType.REDEMPTION, "none"),
    ("CIS RED CAP", True): _StockRule(EventType.SPIN_OFF, "percent"),
    ("CIS RED CAP QTD", True): _StockRule(EventType.SPIN_OFF, "percent"),
    ("REST CAP ACOES", True): _StockRule(EventType.SPIN_OFF, "percent"),
    ("BONIFICACAO", True): _StockRule(EventType.SPIN_OFF, "percent"),
    ("INCORPORACAO", True): _StockRule(EventType.CONVERSION, "percent"),
    ("GRUPAMENTO", True): _StockRule(EventType.CONVERSION, "multiplier"),
}
"""Stock-event labels of B3's listed-company service (H3.5, snapshot 2026-11-10)."""


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


class StockFactorError(ValueError):
    """A stock event that should leave holdings unchanged carries another factor."""


def stock_event(label: str, factor: float, other_asset: bool) -> tuple[EventType | None, float]:
    """
    Return the event type and share ratio of a B3 stock event (H3.5).

    Parameters
    ----------
    label : str
        B3's label (``DESDOBRAMENTO``, ``GRUPAMENTO``...).
    factor : float
        B3's factor, already parsed.
    other_asset : bool
        Whether the holders received an asset other than the event's ISIN.

    Returns
    -------
    tuple[EventType | None, float]
        The type (``None`` for a row that only says the holder keeps the
        share) and the share ratio: shares held after per share held before,
        or units of the other asset per share. ``nan`` when not applicable.

    Raises
    ------
    UnknownLabelError
        If the label and asset combination has no rule.
    StockFactorError
        If a row that keeps the share has a factor other than 100.
    """
    key = (label.strip().upper(), other_asset)
    if key not in STOCK_RULES:
        side = "another asset" if other_asset else "the same asset"
        raise UnknownLabelError(
            f"Unknown stock label {label!r} with {side}; map it in event_types.py."
        )
    rule = STOCK_RULES[key]
    if rule.ratio == "keep":
        if abs(factor - 100.0) > 1e-9:
            raise StockFactorError(f"{label} on the same asset with factor {factor}, not 100.")
        return None, float("nan")
    ratio = {
        "plus_percent": 1.0 + factor / 100.0,
        "percent": factor / 100.0,
        "multiplier": factor,
        "none": float("nan"),
    }[rule.ratio]
    return rule.event, ratio
