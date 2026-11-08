# ibexQuant\src\ibexQuant\data\store\errors.py

"""
Exceptions raised while locating and validating the data root.

Every exception derives from :class:`DataStoreError`, so a caller (the CLI,
a notebook) can handle the whole family in one clause. Each one also derives
from the closest built-in type, so code that already catches
``FileNotFoundError`` or ``ValueError`` keeps working.

See ``docs/decisions/0010-data-root-and-storage-layout.md``.
"""

from __future__ import annotations


class DataStoreError(Exception):
    """Base class of every error raised by ``ibexQuant.data.store``."""


class DataRootNotConfiguredError(DataStoreError, LookupError):
    """No explicit argument, environment variable or config entry names a data root."""


class DataRootConfigError(DataStoreError, ValueError):
    """The data root is configured, but the configuration itself is invalid."""


class DataRootNotFoundError(DataStoreError, FileNotFoundError):
    """The configured data root does not exist on disk."""


class DataLayoutError(DataStoreError):
    """The data root exists but lacks a zone directory."""
