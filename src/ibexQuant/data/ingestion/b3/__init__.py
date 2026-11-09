# ibexQuant\src\ibexQuant\data\ingestion\b3\__init__.py

"""
B3 historical data: COTAHIST ingestion and the raw daily-bar provider (ADR 0011).

Requires the ``b3`` extra (``pyarrow``) to write and read the curated table.
"""

from __future__ import annotations

from ibexQuant.data.ingestion.b3.bars_daily import (
    ManifestEntry,
    build_bars_daily,
    raw_years,
    read_manifest,
)
from ibexQuant.data.ingestion.b3.cotahist import CotahistReport, TrailerConvention, read_cotahist
from ibexQuant.data.ingestion.b3.errors import (
    CotahistError,
    CotahistFormatError,
    CotahistNotFoundError,
)
from ibexQuant.data.ingestion.b3.provider import B3Provider

__all__ = [
    "B3Provider",
    "CotahistError",
    "CotahistFormatError",
    "CotahistNotFoundError",
    "CotahistReport",
    "ManifestEntry",
    "TrailerConvention",
    "build_bars_daily",
    "raw_years",
    "read_cotahist",
    "read_manifest",
]
