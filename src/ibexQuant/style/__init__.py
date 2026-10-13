# ibexQuant\src\ibexQuant\style\__init__.py

"""
Visual identity: the ibexQuant house style.

- :mod:`ibexQuant.style.tokens` — colors, typography and geometry, as plain
  Python constants. The single source of truth, independent of any renderer.
- :mod:`ibexQuant.style.fonts` — the bundled IBM Plex font files.
- :mod:`ibexQuant.style.mpl` — the identity applied to matplotlib; requires
  the ``plot`` extra.

Importing this package changes nothing in any plotting library: the style is
applied only when asked for.
"""

from __future__ import annotations
