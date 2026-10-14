# ibexQuant\src\ibexQuant\style\mpl.py

"""
The ibexQuant visual identity, applied to matplotlib.

Provides:

- :func:`register_fonts` — makes the bundled IBM Plex fonts known to
  matplotlib.
- :func:`rc_params` — the matplotlib settings derived from the tokens.
- :func:`ibex_style` — a context manager: inside it, matplotlib draws in the
  house style; on exit, the previous settings return.
- :func:`apply_ibex_style` — the same settings applied for the rest of the
  session, for notebooks.
- :func:`style_axes` — the axes rules applied to an existing ``Axes``.

Why both settings and :func:`style_axes`
----------------------------------------
matplotlib reads most settings when an artist is **created**, not when it is
drawn. An ``Axes`` created outside the style keeps default spines, grid and
fonts even if the style is applied before drawing on it. Plotting functions
therefore draw inside :func:`ibex_style` **and** call :func:`style_axes` on
the ``Axes`` they receive, so a chart follows the identity wherever its
``Axes`` came from.

Usage
-----
In a notebook, once at the top::

    apply_ibex_style()

In library code, which must never change the caller's settings::

    with ibex_style():
        figure, ax = plt.subplots()
        ...

Nothing in this module runs on import: importing it never changes
matplotlib's state.

What does not belong here
-------------------------
The values of the identity (colors, sizes, weights), which live in
:mod:`ibexQuant.style.tokens`; analysis charts, which live in
:mod:`ibexQuant.plotting`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Final, Literal

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager
from matplotlib.axes import Axes

from ibexQuant.style import tokens
from ibexQuant.style.fonts import font_files

# Tick marks on the x axis only: on the y axis the grid lines already mark
# the values.
X_TICK_LENGTH: Final[float] = 3.0
Y_TICK_LENGTH: Final[float] = 0.0

TitleLocation = Literal["left", "center", "right"]

TITLE_LOCATIONS: Final[tuple[TitleLocation, ...]] = ("left", "center", "right")


# --- Fonts ---


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


# --- Settings ---


def rc_params() -> mpl.RcParams:
    """
    Return the matplotlib settings (rcParams) of the house style.

    Every value comes from :mod:`ibexQuant.style.tokens`; this function only
    translates tokens into matplotlib's vocabulary. Building a ``RcParams``
    validates every value, so a token matplotlib cannot use fails here.

    Returns
    -------
    matplotlib.RcParams
        A new mapping on every call, safe to modify.
    """
    settings: dict[str, object] = {
        # Text
        # Plex alone, with no fallback family: matplotlib would look up every
        # weight in the fallback too and warn for each one it lacks. Plex
        # covers every character the charts use (digits, %, the minus sign).
        "font.family": [tokens.FONT_FAMILY],
        "font.size": tokens.AXIS_LABEL.size,
        "text.color": tokens.TEXT,
        # Figure
        "figure.figsize": tokens.FIGURE_FULL_WIDTH,
        "figure.dpi": tokens.SCREEN_DPI,
        "figure.facecolor": tokens.BACKGROUND,
        "figure.titlesize": tokens.FIGURE_TITLE.size,
        "figure.titleweight": tokens.FIGURE_TITLE.weight,
        # Axes
        "axes.facecolor": tokens.BACKGROUND,
        "axes.edgecolor": tokens.AXIS,
        "axes.linewidth": tokens.AXIS_LINE_WIDTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.labelsize": tokens.AXIS_LABEL.size,
        "axes.labelweight": tokens.AXIS_LABEL.weight,
        "axes.labelcolor": tokens.AXIS_LABEL.color,
        "axes.titlesize": tokens.PANEL_TITLE.size,
        "axes.titleweight": tokens.PANEL_TITLE.weight,
        "axes.titlecolor": tokens.PANEL_TITLE.color,
        "axes.titlelocation": "left",
        "axes.prop_cycle": cycler(color=list(tokens.CATEGORICAL)),
        # Grid
        "grid.color": tokens.GRID,
        "grid.linewidth": tokens.GRID_LINE_WIDTH,
        # Ticks
        "xtick.color": tokens.AXIS,
        "ytick.color": tokens.AXIS,
        "xtick.labelcolor": tokens.TICK_LABEL.color,
        "ytick.labelcolor": tokens.TICK_LABEL.color,
        "xtick.labelsize": tokens.TICK_LABEL.size,
        "ytick.labelsize": tokens.TICK_LABEL.size,
        "xtick.major.width": tokens.AXIS_LINE_WIDTH,
        "ytick.major.width": tokens.AXIS_LINE_WIDTH,
        "xtick.major.size": X_TICK_LENGTH,
        "ytick.major.size": Y_TICK_LENGTH,
        # Legend
        "legend.frameon": False,
        "legend.fontsize": tokens.LEGEND.size,
        "legend.labelcolor": tokens.LEGEND.color,
        # Lines
        "lines.linewidth": tokens.SERIES_LINE_WIDTH,
        # Export: TrueType fonts embedded in PDF and PostScript instead of
        # matplotlib's default Type 3, which PDF tools render and search poorly.
        "savefig.dpi": tokens.EXPORT_DPI,
        "savefig.facecolor": tokens.BACKGROUND,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    return mpl.RcParams(settings)


@contextmanager
def ibex_style() -> Iterator[None]:
    """
    Draw in the house style inside a ``with`` block.

    On exit every matplotlib setting returns to its previous value, so
    library code can use the style without side effects on the caller.

    Yields
    ------
    None

    Examples
    --------
    >>> with ibex_style():
    ...     figure, ax = plt.subplots()
    ...     ax.plot(returns.index, equity)
    """
    register_fonts()
    with mpl.rc_context(rc_params()):
        yield


def apply_ibex_style() -> None:
    """
    Apply the house style to matplotlib for the rest of the session.

    Meant for notebooks and scripts, where wrapping every cell in
    :func:`ibex_style` would be friction. Library code must use
    :func:`ibex_style` instead, which restores the caller's settings.
    """
    register_fonts()
    mpl.rcParams.update(rc_params())


# --- Axes ---


def style_axes(ax: Axes, *, grid: bool = True) -> Axes:
    """
    Apply the house axes rules to an existing ``Axes``.

    Covers what matplotlib fixes when an ``Axes`` is created: spines, grid,
    tick marks and labels, axis labels, titles and the color sequence of
    series drawn afterwards. Series already drawn keep their color.
    Idempotent.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to style, wherever it was created.
    grid : bool, default True
        Draw the horizontal grid. Charts whose fill or color carries the
        reading (drawdown areas, heatmaps) turn it off.

    Returns
    -------
    matplotlib.axes.Axes
        *ax*, styled.
    """
    register_fonts()
    family = [tokens.FONT_FAMILY]

    ax.set_facecolor(tokens.BACKGROUND)
    ax.set_axisbelow(True)
    # Lines drawn from now on follow the house sequence; lines already drawn
    # keep their color.
    ax.set_prop_cycle(color=list(tokens.CATEGORICAL))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(tokens.AXIS)
        ax.spines[side].set_linewidth(tokens.AXIS_LINE_WIDTH)

    ax.grid(False)
    if grid:
        ax.yaxis.grid(True, color=tokens.GRID, linewidth=tokens.GRID_LINE_WIDTH)

    ax.tick_params(
        axis="both",
        color=tokens.AXIS,
        width=tokens.AXIS_LINE_WIDTH,
        labelcolor=tokens.TICK_LABEL.color,
        labelsize=tokens.TICK_LABEL.size,
        labelfontfamily=family,
    )
    ax.tick_params(axis="x", length=X_TICK_LENGTH)
    ax.tick_params(axis="y", length=Y_TICK_LENGTH)

    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set(
            fontfamily=family,
            fontsize=tokens.AXIS_LABEL.size,
            fontweight=tokens.AXIS_LABEL.weight,
            color=tokens.AXIS_LABEL.color,
        )

    # Re-setting each title keeps its text and fixes its font, so a title set
    # later through set_title inherits the house family.
    for location in TITLE_LOCATIONS:
        ax.set_title(
            ax.get_title(loc=location),
            loc=location,
            fontfamily=family,
            fontsize=tokens.PANEL_TITLE.size,
            fontweight=tokens.PANEL_TITLE.weight,
            color=tokens.PANEL_TITLE.color,
        )
    return ax
