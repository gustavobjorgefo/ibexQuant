# ibexQuant\src\ibexQuant\data\reference\errors.py

"""
Exceptions of the security master (ADR 0012).

All derive from :class:`SecurityMasterError`; each also derives from the
closest built-in type.
"""

from __future__ import annotations


class SecurityMasterError(Exception):
    """Base class of every error raised by ``ibexQuant.data.reference``."""


class SecurityMasterNotBuiltError(SecurityMasterError, FileNotFoundError):
    """The master, or the bars it is built from, does not exist yet."""


class SecurityMasterBuildError(SecurityMasterError, ValueError):
    """An override contradicts the spans: unknown ISIN, wrong order, or a broken chain."""


class SecurityNotFoundError(SecurityMasterError, LookupError):
    """No span carries the ticker (or the security) on the requested date."""
