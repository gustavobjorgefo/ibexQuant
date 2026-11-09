# ibexQuant\src\ibexQuant\data\ingestion\b3\errors.py

"""
Exceptions raised while reading COTAHIST files (ADR 0011).

Both concrete errors derive from :class:`CotahistError`, so the command line
can stop one year and carry on with the next in a single clause; each also
derives from the closest built-in type.
"""

from __future__ import annotations


class CotahistError(Exception):
    """Base class of every error raised while reading a COTAHIST file."""


class CotahistNotFoundError(CotahistError, FileNotFoundError):
    """No COTAHIST ZIP exists in the raw zone for the requested year."""


class CotahistFormatError(CotahistError, ValueError):
    """A COTAHIST file breaks the structure the parser relies on (H1.3–H1.8)."""
