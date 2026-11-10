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
``ibexquant data ingest cotahist (--year YEAR ... | --all)``
    Build ``curated/b3/bars_daily`` from the raw COTAHIST files (ADR 0011).
    A year that fails is reported and skipped; the others are still built.
``ibexquant data build security-master [--no-overrides]``
    Build the security master from ``bars_daily``, the CVM forms and the
    override file (ADR 0012).
``ibexquant data review links [--pending]``
    Print the rename candidates with the prices around each change.

Exit codes
----------
- ``0`` — success.
- ``2`` — invalid usage, a :class:`~ibexQuant.data.store.DataStoreError`
  (the configuration or the data root needs fixing, not the code), or at
  least one year that could not be ingested.

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


def _data_ingest_cotahist(args: argparse.Namespace) -> int:
    """Build the requested years, one at a time, reporting each."""
    # Imported here so that `ibexquant data init` works without the `b3` extra.
    from ibexQuant.data.ingestion.b3 import CotahistError, build_bars_daily, raw_years

    store = DataStore.open()
    years = raw_years(store) if args.all else sorted(set(args.year))
    if not years:
        print("error: no COTAHIST files in raw/b3/cotahist/.", file=sys.stderr)
        return EXIT_USAGE

    failed = 0
    for year in years:
        try:
            (entry,) = build_bars_daily(store, year)
        except CotahistError as error:
            failed += 1
            print(f"{year}: error: {error}", file=sys.stderr)
            continue
        print(
            f"{year}: {entry.rows_kept} rows, {entry.sessions} sessions "
            f"({entry.first_session} to {entry.last_session}), "
            f"trailer {entry.trailer_convention}, repaired {entry.rows_repaired}, "
            f"sha256 {entry.source_sha256[:12]}"
        )
    return EXIT_USAGE if failed else EXIT_OK


def _data_build_security_master(args: argparse.Namespace) -> int:
    """Build the master and print its summary."""
    from ibexQuant.data.ingestion.cvm import FcaFormatError
    from ibexQuant.data.reference import (
        LinkOverrides,
        OverridesFormatError,
        SecurityMasterError,
        build_security_master,
        load_link_overrides,
    )
    from ibexQuant.data.store import resolve_overrides_root

    store = DataStore.open()
    try:
        overrides = (
            LinkOverrides.none()
            if args.no_overrides
            else load_link_overrides(resolve_overrides_root())
        )
        summary = build_security_master(store, overrides)
    except (OverridesFormatError, FcaFormatError, SecurityMasterError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE

    print(f"spans {summary.spans}, ISINs {summary.isins}, securities {summary.securities}")
    print("links: " + ", ".join(f"{rule} {count}" for rule, count in summary.links.items()))
    print(
        "candidates: "
        + ", ".join(f"{status} {count}" for status, count in summary.candidates.items())
    )
    print(f"ambiguous pre-1995 codes: {summary.ambiguous_internal_codes}")
    print(f"ambiguous issuer codes: {len(summary.ambiguous_issuers)}")
    print(f"identifier changes vs. previous build: {summary.identifier_changes}")
    if summary.inputs["ticker_links.yaml"] is None:
        print("built without overrides")
    return EXIT_OK


def _data_review_links(args: argparse.Namespace) -> int:
    """Print the candidates, optionally only the pending ones."""
    import pandas as pd

    from ibexQuant.data.reference import SecurityMasterError, read_candidates

    try:
        candidates = read_candidates(DataStore.open())
    except SecurityMasterError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    if args.pending:
        candidates = candidates[candidates["status"] == "pending"]
    if candidates.empty:
        print("No candidates.")
        return EXIT_OK
    shown = candidates[
        [
            "status",
            "from_ticker",
            "from_isin",
            "from_last_session",
            "to_ticker",
            "to_isin",
            "to_first_session",
            "gap_sessions",
            "old_close",
            "new_open",
            "ratio",
        ]
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(shown.to_string(index=False))
    print(f"\n{len(shown)} candidate(s).")
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

    ingest = data_commands.add_parser("ingest", help="Build curated tables from raw files.")
    sources = ingest.add_subparsers(dest="source", required=True, metavar="SOURCE")
    cotahist = sources.add_parser("cotahist", help="B3 annual COTAHIST files.")
    which = cotahist.add_mutually_exclusive_group(required=True)
    which.add_argument("--year", type=int, action="append", help="Year to build (repeatable).")
    which.add_argument("--all", action="store_true", help="Every year found in raw/.")
    cotahist.set_defaults(handler=_data_ingest_cotahist)

    build = data_commands.add_parser("build", help="Build reference tables.")
    targets = build.add_subparsers(dest="target", required=True, metavar="TARGET")
    master = targets.add_parser("security-master", help="Permanent security identifiers.")
    master.add_argument(
        "--no-overrides",
        action="store_true",
        help="Build without ticker_links.yaml (candidates stay pending).",
    )
    master.set_defaults(handler=_data_build_security_master)

    review = data_commands.add_parser("review", help="Review decisions waiting for a human.")
    subjects = review.add_subparsers(dest="subject", required=True, metavar="SUBJECT")
    links = subjects.add_parser("links", help="Rename candidates of the security master.")
    links.add_argument("--pending", action="store_true", help="Only the pending candidates.")
    links.set_defaults(handler=_data_review_links)

    return parser
