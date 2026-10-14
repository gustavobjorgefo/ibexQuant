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
- :func:`outside_legend` — a legend in one row above the plot area, for
  charts whose data leaves no room for the legend inside.
- :func:`fit_time_axis` — the house time axis: series flush with the left
  edge, with room after their last value.
- :func:`finish_figure` — the figure furniture that signs a chart: title and
  subtitle at the top left, source at the bottom left, the ibexQuant
  signature at the bottom right.
- :func:`save_figure` — export, with one rule for format, resolution and
  size.

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

import math
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Final, Literal

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.transforms import offset_copy

from ibexQuant._validation import require_integer, require_real
from ibexQuant.style import tokens
from ibexQuant.style.fonts import font_files

# Tick marks on the x axis only: on the y axis the grid lines already mark
# the values.
X_TICK_LENGTH: Final[float] = 3.0
Y_TICK_LENGTH: Final[float] = 0.0

TitleLocation = Literal["left", "center", "right"]

TITLE_LOCATIONS: Final[tuple[TitleLocation, ...]] = ("left", "center", "right")

# Identifiers of the furniture texts, so finish_figure can find and replace
# its own texts when called again.
TITLE_ID: Final[str] = "ibexquant-title"
SUBTITLE_ID: Final[str] = "ibexquant-subtitle"
SOURCE_ID: Final[str] = "ibexquant-source"
SIGNATURE_ID: Final[str] = "ibexquant-signature"
FURNITURE_IDS: Final[frozenset[str]] = frozenset({TITLE_ID, SUBTITLE_ID, SOURCE_ID, SIGNATURE_ID})

POINTS_PER_INCH: Final[float] = 72.0

# Formats chosen by file extension. Anything else is refused rather than
# passed to matplotlib, which would pick a default silently.
EXPORT_FORMATS: Final[frozenset[str]] = frozenset({".png", ".pdf", ".svg"})

# matplotlib's code for embedding TrueType fonts in PDF and PostScript, instead
# of its default Type 3, which PDF tools render and search poorly.
TRUETYPE_FONT_TYPE: Final[int] = 42


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
        # The house legend sits inside the plot area on an opaque background
        # without border, so grid lines and series pass behind it instead of
        # crossing its text.
        "legend.frameon": True,
        "legend.facecolor": tokens.BACKGROUND,
        "legend.edgecolor": "none",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.fontsize": tokens.LEGEND.size,
        "legend.labelcolor": tokens.LEGEND.color,
        # Lines
        "lines.linewidth": tokens.SERIES_LINE_WIDTH,
        # Export
        "savefig.dpi": tokens.EXPORT_DPI,
        "savefig.facecolor": tokens.BACKGROUND,
        "pdf.fonttype": TRUETYPE_FONT_TYPE,
        "ps.fonttype": TRUETYPE_FONT_TYPE,
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


def fit_time_axis(ax: Axes, *, end_padding: float = tokens.SERIES_END_PADDING) -> Axes:
    """
    Fit the x axis to the data: flush at the start, with room at the end.

    matplotlib pads both ends of an axis equally; a time series reads better
    starting at the left edge, where its history begins, and ending a little
    before the right edge, so its last value does not run into the frame.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose series span the x range; call after drawing them.
    end_padding : float, default ``tokens.SERIES_END_PADDING``
        Room after the last value, as a fraction of the x range.

    Returns
    -------
    matplotlib.axes.Axes
        *ax*, with its x limits set.

    Raises
    ------
    ValueError
        If *ax* has no data, its data spans a single x value, or
        *end_padding* is negative.
    """
    end_padding = require_real("end_padding", end_padding, above=-1.0)
    if end_padding < 0.0:
        raise ValueError(f"end_padding must be >= 0, got {end_padding}.")
    start, end = (float(value) for value in ax.dataLim.intervalx)
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        raise ValueError("ax has no data spanning an x range to fit.")
    ax.set_xlim(start, end + (end - start) * end_padding)
    return ax


def outside_legend(ax: Axes) -> Legend:
    """
    Place the legend in one row above the plot area, aligned right.

    Outside the plot area no series or grid line can cross the labels, and
    on the right it shares the line of a left-aligned panel title instead of
    pushing the plot down.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose labelled series form the legend.

    Returns
    -------
    matplotlib.legend.Legend
        The legend, styled from the tokens.

    Raises
    ------
    ValueError
        If no series of *ax* has a label.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        raise ValueError("ax has no labelled series to show in a legend.")
    register_fonts()
    legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=len(handles),
        frameon=False,
        borderaxespad=tokens.LEGEND_OFFSET,
        prop={"family": [tokens.FONT_FAMILY], "size": tokens.LEGEND.size},
        labelcolor=tokens.LEGEND.color,
    )
    return legend


# --- Figure ---


def finish_figure(
    figure: Figure,
    *,
    title: str,
    subtitle: str | None = None,
    source: str | None = None,
    signature: bool = True,
) -> Figure:
    """
    Add the house furniture to a figure and reserve room for it.

    Places the title and optional subtitle at the top left, the source at the
    bottom left and the ibexQuant signature at the bottom right, each a fixed
    distance in points from the figure edge, so they hold the same position
    whatever the figure size. The figure switches to matplotlib's constrained
    layout, restricted to the band between header and footer, so axes,
    labels and legends never collide with the furniture.

    Calling it again replaces the furniture instead of stacking a second set.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to finish.
    title : str
        What the chart shows, e.g. ``"Growth of 1"``.
    subtitle : str, optional
        The context needed to read it, e.g. the period or the universe.
    source : str, optional
        Where the data comes from, e.g. ``"B3 via Yahoo Finance"``; written
        as ``"Source: ..."``.
    signature : bool, default True
        Sign the figure with ``ibexQuant`` at the bottom right.

    Returns
    -------
    matplotlib.figure.Figure
        *figure*, finished.

    Raises
    ------
    ValueError
        If *title* is empty.
    """
    if not title.strip():
        raise ValueError("title must not be empty: every house figure says what it shows.")
    register_fonts()

    for existing in [text for text in figure.texts if text.get_gid() in FURNITURE_IDS]:
        existing.remove()

    margin = tokens.FIGURE_MARGIN
    header = margin + _line_height(tokens.FIGURE_TITLE)
    _add_furniture(figure, TITLE_ID, title, tokens.FIGURE_TITLE, x=0.0, top=margin)
    if subtitle:
        top = header + tokens.TITLE_GAP
        _add_furniture(figure, SUBTITLE_ID, subtitle, tokens.FIGURE_SUBTITLE, x=0.0, top=top)
        header = top + _line_height(tokens.FIGURE_SUBTITLE)

    footer = 0.0
    if source:
        line = tokens.SOURCE_PREFIX + source
        _add_furniture(figure, SOURCE_ID, line, tokens.SOURCE, x=0.0, bottom=margin)
        footer = margin + _line_height(tokens.SOURCE)
    if signature:
        _add_furniture(
            figure, SIGNATURE_ID, tokens.SIGNATURE_TEXT, tokens.SIGNATURE, x=1.0, bottom=margin
        )
        footer = max(footer, margin + _line_height(tokens.SIGNATURE))

    # The layout fills the band left between header and footer; a margin
    # below the header keeps the first axes title off the figure title.
    height = figure.get_figheight() * POINTS_PER_INCH
    top = 1.0 - (header + margin) / height
    bottom = footer / height
    if bottom >= top:
        raise ValueError("figure is too short for its title, source and signature.")
    figure.set_layout_engine("constrained", rect=(0.0, bottom, 1.0, top - bottom))
    return figure


def save_figure(figure: Figure, path: str | Path, *, dpi: int | None = None) -> Path:
    """
    Export a figure to PNG, PDF or SVG, chosen by the file extension.

    The figure is saved at its own size, without cropping: a chart finished
    with :func:`finish_figure` already has its margins, and keeping the size
    fixed is what lets a report place figures at a known width. PDF and
    PostScript embed the fonts as TrueType even when the figure is saved
    outside :func:`ibex_style`. Missing parent directories are created.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Destination; the extension (``.png``, ``.pdf`` or ``.svg``) sets the
        format.
    dpi : int, optional
        Resolution of PNG output; defaults to the export resolution of the
        tokens (300). PDF and SVG are vector formats and ignore it.

    Returns
    -------
    Path
        The absolute path written.

    Raises
    ------
    ValueError
        If the extension is not one of the supported formats, or *dpi* is
        not a positive integer.
    """
    destination = Path(path).resolve()
    if destination.suffix.lower() not in EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported figure format {destination.suffix!r}; "
            f"use one of {sorted(EXPORT_FORMATS)}."
        )
    resolution = tokens.EXPORT_DPI if dpi is None else require_integer("dpi", dpi, minimum=1)

    register_fonts()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Font embedding is read at save time, so it is set here: a figure saved
    # outside ibex_style still embeds TrueType fonts.
    embedding = {"pdf.fonttype": TRUETYPE_FONT_TYPE, "ps.fonttype": TRUETYPE_FONT_TYPE}
    with mpl.rc_context(mpl.RcParams(embedding)):
        figure.savefig(
            destination,
            dpi=resolution,
            facecolor=tokens.BACKGROUND,
            metadata={"Creator": tokens.SIGNATURE_TEXT},
        )
    return destination


# --- Helpers ---


def _line_height(role: tokens.TextRole) -> float:
    """Height of one line of text in *role*, in points."""
    return role.size * tokens.LINE_SPACING


def _add_furniture(
    figure: Figure,
    gid: str,
    text: str,
    role: tokens.TextRole,
    *,
    x: float,
    top: float | None = None,
    bottom: float | None = None,
) -> None:
    """Place one furniture text *top* points below the top or *bottom* above the bottom."""
    # Offsets in points from the figure edge, so the furniture keeps its
    # distance from the edges whatever the figure size.
    horizontal = tokens.FIGURE_MARGIN if x == 0.0 else -tokens.FIGURE_MARGIN
    if top is not None:
        y, vertical, alignment = 1.0, -top, "top"
    else:
        y, vertical, alignment = 0.0, bottom or 0.0, "bottom"
    transform = offset_copy(
        figure.transFigure, fig=figure, x=horizontal, y=vertical, units="points"
    )
    figure.text(
        x,
        y,
        text,
        transform=transform,
        horizontalalignment="left" if x == 0.0 else "right",
        verticalalignment=alignment,
        fontfamily=[tokens.FONT_FAMILY],
        fontsize=role.size,
        fontweight=role.weight,
        fontstyle=role.style,
        color=role.color,
        gid=gid,
    )
