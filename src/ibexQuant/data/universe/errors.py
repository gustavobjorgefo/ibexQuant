# ibexQuant\src\ibexQuant\data\universe\errors.py

"""Errors of ``ibexQuant.data.universe`` (ADR 0015)."""

from __future__ import annotations


class UniverseError(Exception):
    """Base class of every error raised by ``ibexQuant.data.universe``."""


class UniverseNotBuiltError(UniverseError, FileNotFoundError):
    """The universe tables, or the tables they are built from, do not exist yet."""


class UniverseBuildError(UniverseError, ValueError):
    """The raw portfolio files contradict each other or cannot be read."""
