# ibexQuant\tests\style\test_mpl.py

"""Tests for the house style in :mod:`ibexQuant.style.mpl`."""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates
import pandas as pd
import pytest
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from ibexQuant.style import tokens
from ibexQuant.style.mpl import (
    apply_ibex_style,
    fit_time_axis,
    ibex_style,
    outside_legend,
    rc_params,
    style_axes,
)


@pytest.fixture(autouse=True)
def isolated_settings() -> Iterator[None]:
    """Restore matplotlib's settings after each test, whatever it changed."""
    with mpl.rc_context():
        yield


def resolved_file(text: object) -> str:
    """The font file matplotlib uses for a text artist."""
    properties = text.get_fontproperties()  # type: ignore[attr-defined]
    return Path(font_manager.findfont(properties)).name


def default_axes() -> Axes:
    """An Axes created with matplotlib's defaults, outside the house style."""
    with mpl.rc_context(mpl.rcParamsDefault):
        return Figure().subplots()


# --- Settings ---


def test_settings_are_valid_and_come_from_the_tokens() -> None:
    settings = rc_params()

    assert isinstance(settings, mpl.RcParams)
    assert settings["font.family"] == [tokens.FONT_FAMILY]
    assert settings["axes.titlecolor"] == tokens.PANEL_TITLE.color
    assert settings["grid.color"] == tokens.GRID
    assert settings["axes.prop_cycle"].by_key()["color"] == list(tokens.CATEGORICAL)
    assert settings["axes.spines.top"] is False
    assert settings["axes.titlelocation"] == "left"


def test_settings_are_a_fresh_copy_on_every_call() -> None:
    first = rc_params()
    first["grid.color"] = "#000000"

    assert rc_params()["grid.color"] == tokens.GRID


def test_exported_fonts_are_embedded_as_truetype() -> None:
    settings = rc_params()

    assert settings["pdf.fonttype"] == 42
    assert settings["ps.fonttype"] == 42


# --- Applying the style ---


def test_importing_the_module_changes_nothing() -> None:
    code = (
        "import matplotlib as mpl\n"
        "before = dict(mpl.rcParams)\n"
        "import ibexQuant.style.mpl\n"
        "print(dict(mpl.rcParams) == before)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "True"


def test_context_manager_applies_and_restores() -> None:
    before = mpl.rcParams["grid.color"]

    with ibex_style():
        assert mpl.rcParams["grid.color"] == tokens.GRID
        assert mpl.rcParams["font.family"] == [tokens.FONT_FAMILY]

    assert mpl.rcParams["grid.color"] == before


def test_context_manager_restores_after_an_error() -> None:
    before = mpl.rcParams["axes.titlelocation"]

    with pytest.raises(RuntimeError), ibex_style():
        raise RuntimeError

    assert mpl.rcParams["axes.titlelocation"] == before


def test_apply_changes_the_session_settings() -> None:
    apply_ibex_style()

    assert mpl.rcParams["grid.color"] == tokens.GRID
    assert mpl.rcParams["axes.titlelocation"] == "left"


def test_axes_created_inside_the_style_follow_it() -> None:
    with ibex_style():
        ax = Figure().subplots()
        (line,) = ax.plot([0.0, 1.0])
        ax.set_title("Growth of 1")

    assert not ax.spines["top"].get_visible()
    assert to_hex(line.get_color()) == tokens.CATEGORICAL[0].lower()
    assert ax.get_title(loc="left") == "Growth of 1"


def test_text_renders_in_plex_without_font_warnings(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"), ibex_style():
        figure = Figure()
        ax = figure.subplots()
        ax.plot([0.0, 1.0])
        ax.set_title("Panel")
        figure.canvas.draw()

    assert resolved_file(ax.get_yticklabels()[0]) == "IBMPlexSans-Regular.ttf"
    assert not [record for record in caplog.records if "findfont" in record.getMessage()]


# --- Styling an existing Axes ---


def test_style_axes_fixes_an_axes_created_outside_the_style() -> None:
    ax = default_axes()
    assert ax.spines["top"].get_visible()

    style_axes(ax)

    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert to_hex(ax.spines["left"].get_edgecolor()) == tokens.AXIS.lower()
    assert ax.get_axisbelow() is True


def test_style_axes_draws_the_grid_on_y_only() -> None:
    ax = default_axes()
    ax.plot([0.0, 1.0])

    style_axes(ax)

    assert all(line.get_visible() for line in ax.yaxis.get_gridlines())
    assert not any(line.get_visible() for line in ax.xaxis.get_gridlines())
    assert to_hex(ax.yaxis.get_gridlines()[0].get_color()) == tokens.GRID.lower()


def test_style_axes_can_turn_the_grid_off() -> None:
    ax = default_axes()
    ax.plot([0.0, 1.0])

    style_axes(ax, grid=False)

    assert not any(line.get_visible() for line in ax.yaxis.get_gridlines())


def test_style_axes_sets_tick_labels_in_the_house_font() -> None:
    ax = default_axes()
    ax.plot([0.0, 1.0])

    style_axes(ax)
    label = ax.get_yticklabels()[0]

    assert label.get_fontfamily() == [tokens.FONT_FAMILY]
    assert label.get_fontsize() == tokens.TICK_LABEL.size
    assert to_hex(label.get_color()) == tokens.TICK_LABEL.color.lower()
    assert resolved_file(label) == "IBMPlexSans-Regular.ttf"


def test_style_axes_sets_titles_and_labels() -> None:
    ax = default_axes()
    ax.set_title("Drawdown", loc="left")
    ax.set_ylabel("Return")

    style_axes(ax)
    # A title set afterwards keeps the house font.
    ax.set_title("Drawdown (strategy)", loc="left")
    title = ax.get_title(loc="left")

    assert title == "Drawdown (strategy)"
    for text in (ax.title, ax.yaxis.label):
        assert text.get_fontfamily() == [tokens.FONT_FAMILY]
    assert resolved_file(ax.yaxis.label) == "IBMPlexSans-Regular.ttf"


def test_style_axes_colors_series_drawn_afterwards() -> None:
    ax = default_axes()
    (before,) = ax.plot([0.0, 1.0])

    style_axes(ax)
    (after,) = ax.plot([1.0, 0.0])

    assert to_hex(after.get_color()) == tokens.CATEGORICAL[0].lower()
    assert to_hex(before.get_color()) != tokens.CATEGORICAL[0].lower()


def test_style_axes_is_idempotent_and_returns_the_axes() -> None:
    ax = default_axes()

    assert style_axes(style_axes(ax)) is ax
    assert not ax.spines["top"].get_visible()


# --- Legends ---


def labelled_axes() -> Axes:
    """A styled Axes with two labelled series and a panel title."""
    with ibex_style():
        ax = Figure().subplots()
        ax.plot([1.0, 1.2, 1.1], label="Benchmark")
        ax.plot([1.0, 1.3, 1.5], label="Strategy")
        ax.set_title("Growth of 1")
    return ax


def test_inside_legend_has_an_opaque_borderless_background() -> None:
    # Grid lines and series pass behind the box instead of crossing the text.
    with ibex_style():
        legend = labelled_axes().legend(loc="upper left")

    frame = legend.get_frame()
    assert legend.get_frame_on()
    assert to_hex(frame.get_facecolor()) == tokens.BACKGROUND.lower()
    assert frame.get_alpha() in (None, 1.0)
    assert frame.get_edgecolor()[3] == 0.0


def test_outside_legend_sits_in_one_row_above_the_plot_area() -> None:
    ax = labelled_axes()

    legend = outside_legend(ax)
    canvas = FigureCanvasAgg(ax.figure)  # type: ignore[arg-type]
    canvas.draw()
    renderer = canvas.get_renderer()
    box = legend.get_window_extent(renderer)
    plot_area = ax.get_window_extent(renderer)

    labels = [text.get_window_extent(renderer) for text in legend.get_texts()]

    assert box.y0 >= plot_area.y1
    # Right-aligned with the plot area, inset only by the legend padding.
    assert plot_area.x1 - 10.0 <= box.x1 <= plot_area.x1
    # One row: labels at the same height (within a pixel, since font
    # rasterization differs across platforms), left to right. A second row
    # would sit a full line, about 15 pixels, lower.
    assert labels[0].y0 == pytest.approx(labels[1].y0, abs=1.0)
    assert labels[0].x1 < labels[1].x0
    assert not legend.get_frame_on()


def test_outside_legend_uses_the_house_font() -> None:
    legend = outside_legend(labelled_axes())

    for text in legend.get_texts():
        assert text.get_fontfamily() == [tokens.FONT_FAMILY]
        assert text.get_fontsize() == tokens.LEGEND.size
        assert to_hex(text.get_color()) == tokens.LEGEND.color.lower()


def test_outside_legend_needs_labelled_series() -> None:
    ax = Figure().subplots()
    ax.plot([1.0, 2.0])

    with pytest.raises(ValueError, match="no labelled series"):
        outside_legend(ax)


# --- Time axis ---


def test_time_axis_is_flush_at_the_start_with_room_at_the_end() -> None:
    ax = Figure().subplots()
    ax.plot([0.0, 50.0, 100.0], [1.0, 1.2, 1.1])

    fit_time_axis(ax)

    assert ax.get_xlim() == pytest.approx((0.0, 100.0 * (1 + tokens.SERIES_END_PADDING)))


def test_time_axis_works_on_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    ax = Figure().subplots()
    ax.plot(dates, range(100))

    fit_time_axis(ax, end_padding=0.0)

    start, end = ax.get_xlim()
    assert mpl.dates.num2date(start).date() == dates[0].date()
    assert mpl.dates.num2date(end).date() == dates[-1].date()


def test_time_axis_needs_data() -> None:
    with pytest.raises(ValueError, match="no data"):
        fit_time_axis(Figure().subplots())


def test_time_axis_rejects_negative_padding() -> None:
    ax = Figure().subplots()
    ax.plot([0.0, 1.0])

    with pytest.raises(ValueError, match="end_padding"):
        fit_time_axis(ax, end_padding=-0.1)
