# ibexQuant\src\ibexQuant\cli.py

"""
Command-line entry point: ``ibexquant``.

Responsibility
--------------
Parse arguments, call the library, print the outcome and translate errors
into exit codes. Every command is a thin wrapper: the work lives in the
library, so everything the CLI does can also be done from Python.

Commands
--------
``ibexquant data init [--root PATH]``
    Create the data root and its zones (ADR 0010).

Exit codes
----------
- ``0`` — success.
- ``2`` — invalid usage or a :class:`~ibexQuant.data.store.DataStoreError`
  (the configuration or the data root needs fixing, not the code).

What does not belong here
-------------------------
Any logic beyond argument handling and output formatting.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Final

from ibexQuant.data.store import DataStore, DataStoreError, Zone, resolve_data_root

EXIT_OK: Final[int] = 0
EXIT_USAGE: Final[int] = 2


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the ``ibexquant`` command line.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (see the module docstring).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except DataStoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE


# --- Commands ---


def _data_init(args: argparse.Namespace) -> int:
    """Create the data root and its zones, reporting what already existed."""
    root = resolve_data_root(args.root)
    existed = {zone: (root / zone.value).is_dir() for zone in Zone}
    store = DataStore.initialize(root)
    print(f"Data root: {store.root}")
    for zone in Zone:
        status = "already present" if existed[zone] else "created"
        print(f"  {zone.value + '/':<10}{status}")
    return EXIT_OK


# --- Parser ---


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser; one sub-parser per command group."""
    parser = argparse.ArgumentParser(prog="ibexquant", description="ibexQuant command line.")
    groups = parser.add_subparsers(dest="group", required=True, metavar="GROUP")

    data = groups.add_parser("data", help="Manage the historical data base.")
    data_commands = data.add_subparsers(dest="command", required=True, metavar="COMMAND")

    init = data_commands.add_parser("init", help="Create the data root and its zones.")
    init.add_argument(
        "--root",
        help="Data root to initialize. Defaults to the configured one "
        "(IBEXQUANT_DATA_ROOT or ~/.ibexquant/config.toml).",
    )
    init.set_defaults(handler=_data_init)

    return parser
