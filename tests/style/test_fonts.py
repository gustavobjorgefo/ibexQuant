# ibexQuant\tests\style\test_fonts.py

"""Tests for :mod:`ibexQuant.style.fonts` and font registration in matplotlib."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib import font_manager

from ibexQuant.style import fonts, tokens
from ibexQuant.style.mpl import register_fonts

# Every (family, weight, style) the identity's text roles can ask for, and the
# bundled file that must serve it.
EXPECTED_FILES: dict[tuple[str, int, str], str] = {
    (tokens.FONT_FAMILY, 400, "normal"): "IBMPlexSans-Regular.ttf",
    (tokens.FONT_FAMILY, 400, "italic"): "IBMPlexSans-Italic.ttf",
    (tokens.FONT_FAMILY, 500, "normal"): "IBMPlexSans-Medium.ttf",
    (tokens.FONT_FAMILY, 600, "normal"): "IBMPlexSans-SemiBold.ttf",
    (tokens.FONT_FAMILY, 700, "normal"): "IBMPlexSans-Bold.ttf",
    (tokens.MONO_FONT_FAMILY, 400, "normal"): "IBMPlexMono-Regular.ttf",
}


def resolved_file(family: str, weight: int, style: str) -> str:
    """The file matplotlib picks for a font request, without default fallback."""
    properties = font_manager.FontProperties(family=family, weight=weight, style=style)
    return Path(font_manager.findfont(properties, fallback_to_default=False)).name


# --- Bundled files ---


def test_every_declared_font_file_is_bundled() -> None:
    files = fonts.font_files()

    assert len(files) == 6
    for path in files:
        assert path.is_file()
        assert path.parent.parent == fonts.FONT_DIRECTORY
        # A TrueType file starts with the version tag 0x00010000.
        assert path.read_bytes()[:4] == b"\x00\x01\x00\x00"


def test_each_family_ships_its_open_font_license() -> None:
    for path in fonts.LICENSE_FILES:
        text = path.read_text(encoding="utf-8")
        assert "SIL OPEN FONT LICENSE Version 1.1" in text
        assert "IBM" in text


def test_missing_font_file_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fonts, "FONT_FILES", (tmp_path / "IBMPlexSans-Regular.ttf",))

    with pytest.raises(FileNotFoundError, match=r"IBMPlexSans-Regular\.ttf.*Reinstall"):
        fonts.font_files()


def test_locating_fonts_imports_no_plotting_library() -> None:
    code = (
        "import sys, ibexQuant.style.fonts; "
        "print(any(name.split('.')[0] == 'matplotlib' for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


# --- Registration in matplotlib ---


def test_bundled_files_declare_the_token_families() -> None:
    families = {
        font_manager.ttfFontProperty(font_manager.get_font(path)).name
        for path in fonts.font_files()
    }

    assert families == {tokens.FONT_FAMILY, tokens.MONO_FONT_FAMILY}


@pytest.mark.parametrize(
    ("font_request", "expected"), EXPECTED_FILES.items(), ids=list(EXPECTED_FILES.values())
)
def test_every_text_role_resolves_to_its_plex_file(
    font_request: tuple[str, int, str], expected: str
) -> None:
    # Compared by file name: on a machine with Plex installed system-wide,
    # matplotlib may serve the same style from the system copy.
    register_fonts()

    assert resolved_file(*font_request) == expected


def test_text_roles_only_ask_for_bundled_styles() -> None:
    roles = [value for value in vars(tokens).values() if isinstance(value, tokens.TextRole)]

    for role in roles:
        assert (tokens.FONT_FAMILY, role.weight, role.style) in EXPECTED_FILES


def test_registration_runs_once_per_process() -> None:
    register_fonts()
    registered = len(font_manager.fontManager.ttflist)

    register_fonts()

    # A file may yield more than one entry (matplotlib also registers legacy
    # family names such as "IBM Plex Sans SmBld"); a second call adds none.
    assert len(font_manager.fontManager.ttflist) == registered


def test_registration_reaches_a_fresh_process() -> None:
    # Without registration, matplotlib would not know the bundled files; the
    # check runs in a clean interpreter so earlier tests cannot mask it.
    code = (
        "from pathlib import Path\n"
        "from matplotlib import font_manager\n"
        "from ibexQuant.style.mpl import register_fonts\n"
        "register_fonts()\n"
        "properties = font_manager.FontProperties(family='IBM Plex Sans', weight=600)\n"
        "print(Path(font_manager.findfont(properties, fallback_to_default=False)).name)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "IBMPlexSans-SemiBold.ttf"
