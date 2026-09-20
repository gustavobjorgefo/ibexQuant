# ibexQuant\tests\data\ingestion\test_provider.py

"""
Unit tests for the MarketDataProvider abstract contract.

Exercises the concrete, shared behavior implemented on the ABC itself
(``_resolve_fields``) through a minimal stub subclass, since the ABC
cannot be instantiated directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from ibexQuant.data.ingestion.provider import OHLCV_FIELDS, DateLike, MarketDataProvider


class _StubProvider(MarketDataProvider):
    """Minimal concrete provider used only to exercise base-class behavior."""

    def get_bars(
        self,
        symbols: str | Sequence[str],
        start: DateLike,
        end: DateLike,
        fields: Sequence[str] | None = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        return pd.DataFrame()


# --- construction ---


def test_market_data_provider_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        MarketDataProvider(frequency="1d")  # type: ignore[abstract]


def test_frequency_is_exposed_as_a_read_only_property() -> None:
    provider = _StubProvider(frequency="1d")
    assert provider.frequency == "1d"


# --- _resolve_fields ---


def test_resolve_fields_defaults_to_all_ohlcv_fields() -> None:
    provider = _StubProvider(frequency="1d")
    assert provider._resolve_fields(None) == OHLCV_FIELDS


def test_resolve_fields_preserves_canonical_order_regardless_of_input_order() -> None:
    provider = _StubProvider(frequency="1d")
    assert provider._resolve_fields(["volume", "close"]) == ("close", "volume")


def test_resolve_fields_rejects_unknown_field() -> None:
    provider = _StubProvider(frequency="1d")
    with pytest.raises(ValueError, match="Unknown field"):
        provider._resolve_fields(["not_a_field"])
