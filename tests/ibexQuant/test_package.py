from __future__ import annotations

import ibexQuant


def test_package_is_importable() -> None:
    """Sanity check that the package installs and imports correctly."""
    assert ibexQuant is not None
