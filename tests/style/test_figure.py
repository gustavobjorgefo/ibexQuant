# ibexQuant\tests\style\test_figure.py

"""Tests for figure finishing and export in :mod:`ibexQuant.style.mpl`."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.text import Text

from ibexQuant.style import tokens
from ibexQuant.style.mpl import (
    SIGNATURE_ID,
    SOURCE_ID,
    SUBTITLE_ID,
    TITLE_ID,
    finish_figure,
    ibex_style,
    save_figure,
)


@pytest.fixture(autouse=True)
def isolated_settings() -> Iterator[None]:
    """Restore matplotlib's settings after each test."""
    with mpl.rc_context():
        yield


@pytest.fixture
def figure() -> Figure:
    """A small styled figure with one panel."""
    with ibex_style():
        figure = Figure(figsize=(6.0, 4.0))
        ax = figure.subplots()
        ax.plot([1.0, 1.1, 1.05, 1.2])
        ax.set_title("Growth of 1")
    return figure


def render(figure: Figure) -> RendererBase:
    """Draw *figure* on an Agg canvas and return its renderer."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return canvas.get_renderer()


def furniture(figure: Figure) -> dict[str, Text]:
    """The furniture texts of a figure, by identifier."""
    return {str(text.get_gid()): text for text in figure.texts if text.get_gid()}


# --- Furniture ---


def test_full_furniture_is_placed(figure: Figure) -> None:
    finish_figure(figure, title="Strategy", subtitle="2019–2025", source="B3 via Yahoo Finance")

    texts = furniture(figure)

    assert set(texts) == {TITLE_ID, SUBTITLE_ID, SOURCE_ID, SIGNATURE_ID}
    assert texts[TITLE_ID].get_text() == "Strategy"
    assert texts[SUBTITLE_ID].get_text() == "2019–2025"
    assert texts[SOURCE_ID].get_text() == "Source: B3 via Yahoo Finance"
    assert texts[SIGNATURE_ID].get_text() == "ibexQuant"


@pytest.mark.parametrize(
    ("gid", "role"),
    [
        (TITLE_ID, tokens.FIGURE_TITLE),
        (SUBTITLE_ID, tokens.FIGURE_SUBTITLE),
        (SOURCE_ID, tokens.SOURCE),
        (SIGNATURE_ID, tokens.SIGNATURE),
    ],
)
def test_each_text_follows_its_role(figure: Figure, gid: str, role: tokens.TextRole) -> None:
    finish_figure(figure, title="Strategy", subtitle="Period", source="B3")

    text = furniture(figure)[gid]

    assert text.get_fontfamily() == [tokens.FONT_FAMILY]
    assert text.get_fontsize() == role.size
    assert text.get_fontweight() == role.weight
    assert text.get_fontstyle() == role.style
    assert to_hex(text.get_color()) == role.color.lower()


def test_header_is_left_aligned_and_signature_right_aligned(figure: Figure) -> None:
    finish_figure(figure, title="Strategy", source="B3")
    texts = furniture(figure)

    assert texts[TITLE_ID].get_horizontalalignment() == "left"
    assert texts[SOURCE_ID].get_horizontalalignment() == "left"
    assert texts[SIGNATURE_ID].get_horizontalalignment() == "right"


def test_optional_furniture_can_be_left_out(figure: Figure) -> None:
    finish_figure(figure, title="Strategy", signature=False)

    assert set(furniture(figure)) == {TITLE_ID}


def test_finishing_again_replaces_the_furniture(figure: Figure) -> None:
    finish_figure(figure, title="First", subtitle="Old subtitle", source="B3")

    finish_figure(figure, title="Second")
    texts = furniture(figure)

    assert set(texts) == {TITLE_ID, SIGNATURE_ID}
    assert texts[TITLE_ID].get_text() == "Second"


def test_figure_texts_not_owned_by_the_furniture_are_kept(figure: Figure) -> None:
    note = figure.text(0.5, 0.5, "note")

    finish_figure(figure, title="Strategy")

    assert note in figure.texts


# --- Layout ---


def test_furniture_never_overlaps_the_axes(figure: Figure) -> None:
    finish_figure(figure, title="Strategy", subtitle="Period", source="B3")
    renderer = render(figure)
    texts = furniture(figure)
    panel = figure.axes[0].get_tightbbox(renderer)
    assert panel is not None

    assert texts[SUBTITLE_ID].get_window_extent(renderer).y0 > panel.y1
    assert texts[SOURCE_ID].get_window_extent(renderer).y1 < panel.y0
    assert texts[SIGNATURE_ID].get_window_extent(renderer).y1 < panel.y0


def test_furniture_stays_inside_the_figure(figure: Figure) -> None:
    finish_figure(figure, title="Strategy", source="B3")
    renderer = render(figure)
    canvas = figure.bbox

    for text in furniture(figure).values():
        extent = text.get_window_extent(renderer)
        assert canvas.x0 <= extent.x0 and extent.x1 <= canvas.x1
        assert canvas.y0 <= extent.y0 and extent.y1 <= canvas.y1


def test_furniture_keeps_its_distance_from_the_edge_at_any_size() -> None:
    # The offsets are in points: the title sits the same distance from the
    # top edge on a small and on a large figure.
    distances = []
    for size in ((5.0, 3.0), (11.0, 8.0)):
        figure = Figure(figsize=size)
        figure.subplots()
        finish_figure(figure, title="Strategy")
        extent = furniture(figure)[TITLE_ID].get_window_extent(render(figure))
        distances.append((figure.bbox.y1 - extent.y1) / figure.dpi * 72.0)

    assert distances[0] == pytest.approx(distances[1], abs=0.5)
    assert distances[0] == pytest.approx(tokens.FIGURE_MARGIN, abs=0.5)


def test_layout_switches_to_constrained(figure: Figure) -> None:
    finish_figure(figure, title="Strategy")

    assert isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine)


def test_empty_title_is_rejected(figure: Figure) -> None:
    with pytest.raises(ValueError, match="title must not be empty"):
        finish_figure(figure, title="  ")


def test_figure_too_short_for_its_furniture_is_rejected() -> None:
    figure = Figure(figsize=(6.0, 0.8))
    figure.subplots()

    with pytest.raises(ValueError, match="too short"):
        finish_figure(figure, title="Strategy", subtitle="Period", source="B3")


# --- Export ---


@pytest.mark.parametrize("extension", [".png", ".pdf", ".svg"])
def test_supported_formats_are_written(figure: Figure, tmp_path: Path, extension: str) -> None:
    finish_figure(figure, title="Strategy")

    written = save_figure(figure, tmp_path / f"chart{extension}")

    assert written == (tmp_path / f"chart{extension}").resolve()
    assert written.stat().st_size > 0


def test_png_uses_the_export_resolution_at_the_figure_size(figure: Figure, tmp_path: Path) -> None:
    written = save_figure(figure, tmp_path / "chart.png")

    # PNG header: width and height as 4-byte big-endian integers.
    header = written.read_bytes()[16:24]
    width, height = int.from_bytes(header[:4], "big"), int.from_bytes(header[4:], "big")

    assert (width, height) == (6 * tokens.EXPORT_DPI, 4 * tokens.EXPORT_DPI)


def test_png_resolution_can_be_set(figure: Figure, tmp_path: Path) -> None:
    written = save_figure(figure, tmp_path / "chart.png", dpi=100)

    assert int.from_bytes(written.read_bytes()[16:20], "big") == 600


def test_pdf_embeds_plex_as_truetype(figure: Figure, tmp_path: Path) -> None:
    # Saved outside ibex_style, where matplotlib's default is Type 3 fonts.
    finish_figure(figure, title="Strategy", source="B3")
    assert mpl.rcParams["pdf.fonttype"] == 3

    content = save_figure(figure, tmp_path / "chart.pdf").read_bytes()

    assert b"/FontFile2" in content
    assert b"Type3" not in content
    assert re.search(rb"/BaseFont /[A-Z]{6}\+IBMPlexSans", content)


def test_missing_directories_are_created(figure: Figure, tmp_path: Path) -> None:
    written = save_figure(figure, tmp_path / "reports" / "0001" / "chart.png")

    assert written.is_file()


def test_unsupported_formats_are_rejected(figure: Figure, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported figure format '.jpg'"):
        save_figure(figure, tmp_path / "chart.jpg")
    with pytest.raises(ValueError, match="Unsupported figure format ''"):
        save_figure(figure, tmp_path / "chart")


@pytest.mark.parametrize("dpi", [0, -10])
def test_invalid_resolution_is_rejected(figure: Figure, tmp_path: Path, dpi: int) -> None:
    with pytest.raises(ValueError, match="dpi"):
        save_figure(figure, tmp_path / "chart.png", dpi=dpi)
