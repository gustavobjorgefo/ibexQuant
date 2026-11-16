# ibexQuant\src\ibexQuant\__init__.py

"""
ibexQuant: event-driven research, backtest and live trading for B3.

The package root stays light: ``import ibexQuant`` loads nothing heavy, and
``ibexQuant.download`` imports the data layer on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ibexQuant.data.download import download

__all__ = ["download"]


def __getattr__(name: str) -> object:
    # Lazy: the data layer needs the `b3` extra (pyarrow), which a plain import must not.
    if name == "download":
        from ibexQuant.data.download import download

        return download
    raise AttributeError(f"module 'ibexQuant' has no attribute {name!r}")
