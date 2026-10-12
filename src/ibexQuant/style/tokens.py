# ibexQuant\src\ibexQuant\style\tokens.py

"""
Design tokens: the single source of truth of the ibexQuant visual identity.

Tokens are plain Python constants, with no dependency on any plotting
library, so charts (matplotlib today) and future reports (HTML, LaTeX) read
the same values. They are grouped in four layers:

- **Palette** — every color the identity uses, named by what it is
  (``IBEX_BLUE``, ``STEEL``).
- **Semantic colors** — what charts actually use, named by role
  (``STRATEGY``, ``BENCHMARK``, ``NEGATIVE``). A chart never says "blue": it
  says "strategy", so a palette change is one edit here.
- **Typography** — font families and one :class:`TextRole` per kind of text.
- **Geometry** — line widths, figure sizes and resolution.

Contract
--------
Colors are ``#RRGGBB`` strings; sizes are points; figure sizes are inches.
Font weights are numeric (400 regular, 500 medium, 600 semibold, 700 bold),
the convention shared by CSS and matplotlib.

What does not belong here
-------------------------
Applying the tokens to a renderer (rcParams, figure finishing, export), which
lives in the renderer-specific modules of this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

# --- Palette: brand ---

IBEX_BLUE: Final[str] = "#0B3D91"
IBEX_NAVY: Final[str] = "#0B1E3B"
IBEX_SKY: Final[str] = "#4F86C6"
IBEX_TINT: Final[str] = "#DCE6F4"

# --- Palette: neutrals, darkest to lightest ---

INK: Final[str] = "#1F2933"
SLATE: Final[str] = "#4B5563"
STEEL: Final[str] = "#6B7280"
SILVER: Final[str] = "#9CA3AF"
ASH: Final[str] = "#D1D5DB"
CLOUD: Final[str] = "#F5F7FA"
WHITE: Final[str] = "#FFFFFF"

# --- Palette: accents ---

TERRACOTTA: Final[str] = "#C8553D"
GOLD: Final[str] = "#D9A441"
TAUPE: Final[str] = "#9C755F"
COOL_GRAY: Final[str] = "#A4A9B3"

# --- Semantic colors ---

STRATEGY: Final[str] = IBEX_BLUE
BENCHMARK: Final[str] = STEEL
# Blue and terracotta differ in lightness as well as hue, so gains and losses
# stay distinct in grayscale print and under red-green color blindness, which
# the usual green and red do not.
POSITIVE: Final[str] = IBEX_BLUE
NEGATIVE: Final[str] = TERRACOTTA

TEXT: Final[str] = INK
SECONDARY_TEXT: Final[str] = STEEL
AXIS: Final[str] = SLATE
REFERENCE_LINE: Final[str] = SLATE
GRID: Final[str] = ASH
BACKGROUND: Final[str] = WHITE
PAGE_BACKGROUND: Final[str] = CLOUD

# Series beyond the strategy and the benchmark, in order. The set was chosen
# to stay distinguishable under every type of color blindness; a sixth color
# collided with these, so a chart shows at most five series and splits or
# highlights beyond that.
CATEGORICAL: Final[tuple[str, ...]] = (IBEX_BLUE, IBEX_SKY, GOLD, TAUPE, COOL_GRAY)
MAX_CATEGORICAL_SERIES: Final[int] = len(CATEGORICAL)

# Color scales, as their anchor colors from low to high.
DIVERGING_SCALE: Final[tuple[str, str, str]] = (NEGATIVE, WHITE, POSITIVE)
SEQUENTIAL_SCALE: Final[tuple[str, str]] = (WHITE, IBEX_BLUE)

# --- Typography ---

FONT_FAMILY: Final[str] = "IBM Plex Sans"
MONO_FONT_FAMILY: Final[str] = "IBM Plex Mono"

FontWeight = Literal[400, 500, 600, 700]
FontStyle = Literal["normal", "italic"]


@dataclass(frozen=True)
class TextRole:
    """
    How one kind of text looks.

    Attributes
    ----------
    size : float
        Font size, in points.
    weight : {400, 500, 600, 700}
        Numeric font weight.
    color : str
        ``#RRGGBB`` color.
    style : {"normal", "italic"}
        Font style.
    """

    size: float
    weight: FontWeight
    color: str
    style: FontStyle = "normal"


FIGURE_TITLE: Final[TextRole] = TextRole(size=13.0, weight=600, color=IBEX_BLUE)
FIGURE_SUBTITLE: Final[TextRole] = TextRole(size=10.0, weight=400, color=SECONDARY_TEXT)
PANEL_TITLE: Final[TextRole] = TextRole(size=10.0, weight=500, color=TEXT)
AXIS_LABEL: Final[TextRole] = TextRole(size=9.0, weight=400, color=TEXT)
TICK_LABEL: Final[TextRole] = TextRole(size=8.5, weight=400, color=AXIS)
LEGEND: Final[TextRole] = TextRole(size=8.5, weight=400, color=TEXT)
ANNOTATION: Final[TextRole] = TextRole(size=8.0, weight=400, color=TEXT)
SOURCE: Final[TextRole] = TextRole(size=7.5, weight=400, color=SECONDARY_TEXT, style="italic")
SIGNATURE: Final[TextRole] = TextRole(size=8.0, weight=600, color=IBEX_BLUE)

SIGNATURE_TEXT: Final[str] = "ibexQuant"

# --- Geometry ---

# Line widths, in points. The benchmark is thinner than the strategy and drawn
# behind it, so hierarchy reads from width and order, not only from color.
STRATEGY_LINE_WIDTH: Final[float] = 1.8
BENCHMARK_LINE_WIDTH: Final[float] = 1.3
SERIES_LINE_WIDTH: Final[float] = 1.6
REFERENCE_LINE_WIDTH: Final[float] = 0.8
AXIS_LINE_WIDTH: Final[float] = 0.7
GRID_LINE_WIDTH: Final[float] = 0.6

DRAWDOWN_FILL_ALPHA: Final[float] = 0.18

# Figure sizes, in inches. Provisional: once a report template exists they
# derive from its text width, so that point sizes hold on paper.
FIGURE_FULL_WIDTH: Final[tuple[float, float]] = (10.0, 5.0)
FIGURE_HALF_WIDTH: Final[tuple[float, float]] = (5.0, 3.5)
FIGURE_TEARSHEET: Final[tuple[float, float]] = (11.0, 15.5)

SCREEN_DPI: Final[int] = 110
EXPORT_DPI: Final[int] = 300
