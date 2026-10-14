# ibexQuant\tests\style\test_tokens.py

"""Tests for :mod:`ibexQuant.style.tokens`."""

from __future__ import annotations

import dataclasses
import itertools
import re
import subprocess
import sys
from typing import Final

import pytest

from ibexQuant.style import tokens

HEX_COLOR: Final[re.Pattern[str]] = re.compile(r"#[0-9A-F]{6}")

# WCAG 2.1 minimum contrast: 4.5 for text (1.4.3), 3.0 for graphical objects
# such as lines and areas (1.4.11).
TEXT_CONTRAST: Final[float] = 4.5
GRAPHIC_CONTRAST: Final[float] = 3.0


def relative_luminance(color: str) -> float:
    """WCAG 2.1 relative luminance of a ``#RRGGBB`` color."""
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    """WCAG 2.1 contrast ratio between two colors, from 1 to 21."""
    lighter, darker = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def color_tokens() -> dict[str, str]:
    """Every string token that is a color, by name."""
    return {
        name: value
        for name, value in vars(tokens).items()
        if name.isupper() and isinstance(value, str) and value.startswith("#")
    }


# --- Format ---


def test_every_color_is_an_uppercase_hex_string() -> None:
    colors = color_tokens()

    assert len(colors) > 20
    for name, value in colors.items():
        assert HEX_COLOR.fullmatch(value), f"{name} = {value!r}"


def test_text_roles_use_palette_colors() -> None:
    palette = set(color_tokens().values())
    roles = [value for value in vars(tokens).values() if isinstance(value, tokens.TextRole)]

    assert len(roles) == 9
    for role in roles:
        assert role.color in palette


def test_text_roles_are_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        tokens.FIGURE_TITLE.size = 20.0  # type: ignore[misc]


# --- Legibility ---


@pytest.mark.parametrize("name", ["TEXT", "SECONDARY_TEXT", "AXIS"])
def test_text_colors_are_readable_on_the_background(name: str) -> None:
    assert contrast_ratio(getattr(tokens, name), tokens.BACKGROUND) >= TEXT_CONTRAST


@pytest.mark.parametrize("name", ["STRATEGY", "BENCHMARK", "POSITIVE", "NEGATIVE"])
def test_chart_lines_stand_out_from_the_background(name: str) -> None:
    assert contrast_ratio(getattr(tokens, name), tokens.BACKGROUND) >= GRAPHIC_CONTRAST


def test_gains_and_losses_differ_in_lightness() -> None:
    # Distinct in grayscale print and under red-green color blindness; the
    # previous green and red pair had a ratio of about 1.1.
    assert contrast_ratio(tokens.POSITIVE, tokens.NEGATIVE) >= 2.0


def test_text_roles_are_readable_on_the_background() -> None:
    roles = [value for value in vars(tokens).values() if isinstance(value, tokens.TextRole)]

    for role in roles:
        assert contrast_ratio(role.color, tokens.BACKGROUND) >= TEXT_CONTRAST


def test_reference_line_reads_as_a_grid_line() -> None:
    # Heavier than the grid, it would read as a second axis next to the x axis.
    assert tokens.REFERENCE_LINE == tokens.GRID
    assert tokens.REFERENCE_LINE_WIDTH == tokens.GRID_LINE_WIDTH


def test_grid_stays_in_the_background() -> None:
    # The grid supports reading values; it must be much fainter than any line.
    assert contrast_ratio(tokens.GRID, tokens.BACKGROUND) < 1.6


# --- Series ---


def test_categorical_sequence_has_at_most_five_distinct_colors() -> None:
    assert len(tokens.CATEGORICAL) == tokens.MAX_CATEGORICAL_SERIES == 5
    assert len(set(tokens.CATEGORICAL)) == len(tokens.CATEGORICAL)


def test_categorical_sequence_starts_with_the_brand() -> None:
    assert tokens.CATEGORICAL[0] == tokens.IBEX_BLUE


def test_no_two_categorical_colors_are_near_identical() -> None:
    for first, second in itertools.combinations(tokens.CATEGORICAL, 2):
        channels = [abs(int(first[i : i + 2], 16) - int(second[i : i + 2], 16)) for i in (1, 3, 5)]
        assert max(channels) > 40, f"{first} and {second}"


def test_scales_run_from_losses_to_gains() -> None:
    assert tokens.DIVERGING_SCALE == (tokens.NEGATIVE, tokens.WHITE, tokens.POSITIVE)
    assert tokens.SEQUENTIAL_SCALE[0] == tokens.WHITE


# --- Hierarchy ---


def test_strategy_line_is_heavier_than_the_benchmark() -> None:
    assert tokens.STRATEGY_LINE_WIDTH > tokens.BENCHMARK_LINE_WIDTH > tokens.GRID_LINE_WIDTH


def test_title_outranks_every_other_text() -> None:
    roles = [value for value in vars(tokens).values() if isinstance(value, tokens.TextRole)]

    assert all(tokens.FIGURE_TITLE.size >= role.size for role in roles)
    assert all(tokens.SOURCE.size <= role.size for role in roles)


# --- Independence from renderers ---


def test_tokens_import_no_plotting_library() -> None:
    # A fresh interpreter, so modules imported by other tests do not count.
    code = (
        "import sys, ibexQuant.style.tokens; "
        "print(any(name.split('.')[0] in {'matplotlib', 'plotly'} for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
