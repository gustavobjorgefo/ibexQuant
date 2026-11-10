# ibexQuant\src\ibexQuant\data\store\atomic.py

"""
Atomic replacement of files inside the data root.

Every build that rewrites a curated table goes through
:func:`replace_atomically`, so that a failure halfway never leaves a
half-written file where a complete one used to be (ADR 0011, H1.11).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def replace_atomically(target: Path, write: Callable[[Path], object]) -> None:
    """
    Write ``target`` through a temporary file in the same folder, then rename.

    Parameters
    ----------
    target : Path
        File to create or replace. Its folder must exist.
    write : Callable[[Path], object]
        Writes the complete content to the path it receives.
    """
    temporary = target.with_name(target.name + ".tmp")
    try:
        write(temporary)
        # os.replace is atomic on the same file system, on Windows included.
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
