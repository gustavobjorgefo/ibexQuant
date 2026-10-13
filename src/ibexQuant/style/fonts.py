# ibexQuant\src\ibexQuant\style\fonts.py

"""
Font files bundled with ibexQuant.

The identity's typefaces, IBM Plex Sans and IBM Plex Mono, ship inside the
package so that every machine renders charts with the same fonts. Without
them a plotting library falls back silently to its default font (DejaVu Sans
in matplotlib), and a chart drawn in CI would differ from one drawn locally.

Only the styles the identity uses are bundled: Plex Sans Regular, Italic,
Medium, SemiBold and Bold, and Plex Mono Regular. Both families are licensed
under the SIL Open Font License 1.1, whose text ships next to the files as
the license requires.

This module only locates the files; it depends on no plotting library, so a
future report renderer (e.g. LaTeX) can point at the same files.

What does not belong here
-------------------------
Registering the fonts with a renderer, in the renderer-specific modules
(:func:`ibexQuant.style.mpl.register_fonts` for matplotlib).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

FONT_DIRECTORY: Final[Path] = Path(__file__).resolve().parent / "fonts"

SANS_DIRECTORY: Final[Path] = FONT_DIRECTORY / "ibm_plex_sans"
MONO_DIRECTORY: Final[Path] = FONT_DIRECTORY / "ibm_plex_mono"

# Listed explicitly rather than globbed, so a missing file fails loudly
# instead of silently dropping a weight.
FONT_FILES: Final[tuple[Path, ...]] = (
    SANS_DIRECTORY / "IBMPlexSans-Regular.ttf",
    SANS_DIRECTORY / "IBMPlexSans-Italic.ttf",
    SANS_DIRECTORY / "IBMPlexSans-Medium.ttf",
    SANS_DIRECTORY / "IBMPlexSans-SemiBold.ttf",
    SANS_DIRECTORY / "IBMPlexSans-Bold.ttf",
    MONO_DIRECTORY / "IBMPlexMono-Regular.ttf",
)

LICENSE_FILES: Final[tuple[Path, ...]] = (
    SANS_DIRECTORY / "OFL.txt",
    MONO_DIRECTORY / "OFL.txt",
)


def font_files() -> tuple[Path, ...]:
    """
    Return the bundled font files, after checking that all of them exist.

    Returns
    -------
    tuple of Path
        Absolute paths of every bundled font file.

    Raises
    ------
    FileNotFoundError
        If any declared font file is missing, e.g. from a broken install.
    """
    missing = [path.name for path in FONT_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Bundled font files are missing from {FONT_DIRECTORY}: {missing}. "
            "Reinstall ibexQuant."
        )
    return FONT_FILES
