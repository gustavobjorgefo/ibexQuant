# ibexQuant\src\ibexQuant\data\__init__.py

"""
Data layer: storage, ingestion, reference tables, adjustment, universe and panels.

``from ibexQuant.data import B3Dataset`` is the entry point to the B3 base (ADR 0016); it
is loaded on first use, so importing a light submodule never pulls the whole base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ibexQuant.data.dataset import B3Dataset

__all__ = ["B3Dataset"]


def __getattr__(name: str) -> object:
    # Lazy: the dataset needs the `b3` extra (pyarrow, pyyaml), which light imports must not.
    if name == "B3Dataset":
        from ibexQuant.data.dataset import B3Dataset

        return B3Dataset
    raise AttributeError(f"module 'ibexQuant.data' has no attribute {name!r}")
