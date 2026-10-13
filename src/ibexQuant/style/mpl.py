# ibexQuant\src\ibexQuant\style\mpl.py

"""
The ibexQuant visual identity, applied to matplotlib.

Provides:

- :func:`register_fonts` — makes the bundled IBM Plex fonts known to
  matplotlib.

Nothing in this module runs on import: importing it never changes
matplotlib's state. Every effect happens only when a function is called.

What does not belong here
-------------------------
The values of the identity (colors, sizes, weights), which live in
:mod:`ibexQuant.style.tokens`; analysis charts, which live in
:mod:`ibexQuant.plotting`.
"""

from __future__ import annotations

from functools import cache

from matplotlib import font_manager

from ibexQuant.style.fonts import font_files


@cache
def register_fonts() -> None:
    """
    Register the bundled IBM Plex fonts with matplotlib.

    Adds every bundled font file to matplotlib's font manager for the current
    process, so text drawn in ``IBM Plex Sans`` or ``IBM Plex Mono`` uses the
    bundled files on every machine instead of falling back to the default
    font. The registration lasts for the process and is not written to
    matplotlib's font cache.

    Runs once per process: later calls return immediately, so callers invoke
    it freely before drawing.

    Raises
    ------
    FileNotFoundError
        If a bundled font file is missing.
    """
    for path in font_files():
        font_manager.fontManager.addfont(str(path))
