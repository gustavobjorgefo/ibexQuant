# ibexQuant\src\ibexQuant\data\store\__init__.py

"""
Location and layout of the data root (ADR 0010).

Public API: :func:`resolve_data_root`, :class:`DataStore`, :class:`Zone` and
the :class:`DataStoreError` family.
"""

from __future__ import annotations

from ibexQuant.data.store.config import (
    DATA_ROOT_ENV_VAR,
    OVERRIDES_ROOT_ENV_VAR,
    resolve_data_root,
    resolve_overrides_root,
)
from ibexQuant.data.store.data_store import DataStore, Zone
from ibexQuant.data.store.errors import (
    DataLayoutError,
    DataRootConfigError,
    DataRootNotConfiguredError,
    DataRootNotFoundError,
    DataStoreError,
    OverridesRootNotConfiguredError,
)

__all__ = [
    "DATA_ROOT_ENV_VAR",
    "DataLayoutError",
    "DataRootConfigError",
    "DataRootNotConfiguredError",
    "DataRootNotFoundError",
    "DataStore",
    "DataStoreError",
    "OVERRIDES_ROOT_ENV_VAR",
    "OverridesRootNotConfiguredError",
    "Zone",
    "resolve_data_root",
    "resolve_overrides_root",
]
