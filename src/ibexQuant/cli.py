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
``ibexquant data review links [--status STATUS ...] [--yaml]``
    Print the rename candidates with the prices around each change, or, with
    ``--yaml``, ready-to-paste ``ticker_links.yaml`` entries for them.
``ibexquant data build corporate-actions [--no-overrides]``
    Build the corporate actions ledger from ``bars_daily``, the security
    master, B3's raw files, the CVM reference forms and the override file
    (ADR 0013).
``ibexquant data build adjustment-factors``
    Compute one adjustment factor per ex-session of the ledger into
    ``derived/b3`` (ADR 0014).
``ibexquant data review events [--status STATUS ...] [--yaml]``
    Print the ex-sessions waiting for review (by default inferred splits and
    unexplained jumps) with the prices around them, or, with ``--yaml``,
    ready-to-edit ``corporate_actions.yaml`` entries.

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
from typing import TYPE_CHECKING, Final

from ibexQuant.data.store import DataStore, DataStoreError, Zone, resolve_data_root

if TYPE_CHECKING:
    import pandas as pd

EXIT_OK: Final[int] = 0
EXIT_USAGE: Final[int] = 2

# Mirrors ibexQuant.data.reference.security_master.STATUSES (a test keeps them equal);
# importing it here would load pandas for every command, `data init` included.
CANDIDATE_STATUSES: Final[tuple[str, ...]] = (
    "linked_by_cnpj",
    "linked_by_override",
    "rejected",
    "likely",
    "pending",
    "conflict",
    "superseded",
)
# Mirrors ibexQuant.data.reference.corporate_actions.EX_SESSION_STATUSES (a test keeps
# them equal).
EX_SESSION_STATUSES: Final[tuple[str, ...]] = (
    "explained",
    "inferred",
    "unexplained_jump",
    "unexplained",
    "accepted",
)
REVIEWED_BY_DEFAULT: Final[tuple[str, ...]] = ("inferred", "unexplained_jump")


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
    """Print the candidates, as a table or as override entries to paste."""
    import pandas as pd

    from ibexQuant.data.reference import SecurityMasterError, read_candidates

    try:
        candidates = read_candidates(DataStore.open())
    except SecurityMasterError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    if args.status:
        candidates = candidates[candidates["status"].isin(args.status)]
    if candidates.empty:
        print("No candidates.")
        return EXIT_OK

    if args.yaml:
        print(_as_yaml(candidates))
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
            "implied_factor",
        ]
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(shown.to_string(index=False))
    print(f"\n{len(shown)} candidate(s).")
    return EXIT_OK


def _as_yaml(candidates: pd.DataFrame) -> str:
    """Render candidates as ``links:`` entries of ``ticker_links.yaml``, to review and paste."""
    import pandas as pd

    lines = ["links:"]
    for record in candidates.to_dict("records"):
        last = str(record["from_last_session"])[:10]
        first = str(record["to_first_session"])[:10]
        old, new = record["from_ticker"], record["to_ticker"]
        note = f"{old} -> {new}, price ratio {record['ratio']:.3f}"
        if pd.notna(record["implied_factor"]):
            note += f" (implied factor {record['implied_factor']:g}: split or exchange ratio?)"
        lines += [
            f"  - from: {record['from_isin']}  # {old}, last session {last}",
            f"    to: {record['to_isin']}  # {new}, first session {first}",
            f'    note: "{note}"',
        ]
    return "\n".join(lines)


def _data_build_corporate_actions(args: argparse.Namespace) -> int:
    """Build the ledger and print its summary."""
    from ibexQuant.data.ingestion.b3.earnings_credited import EarningsCreditedFormatError
    from ibexQuant.data.ingestion.b3.listed_companies import ListedCompaniesFormatError
    from ibexQuant.data.ingestion.cvm import FreFormatError
    from ibexQuant.data.reference import OverridesFormatError, SecurityMasterError
    from ibexQuant.data.reference.corporate_action_overrides import (
        ActionOverrides,
        load_action_overrides,
    )
    from ibexQuant.data.reference.corporate_actions import (
        LedgerBuildError,
        build_corporate_actions,
    )
    from ibexQuant.data.reference.event_types import StockFactorError, UnknownLabelError
    from ibexQuant.data.store import resolve_overrides_root

    store = DataStore.open()
    try:
        overrides = (
            ActionOverrides.none()
            if args.no_overrides
            else load_action_overrides(resolve_overrides_root())
        )
        summary = build_corporate_actions(store, overrides)
    except (
        OverridesFormatError,
        SecurityMasterError,
        LedgerBuildError,
        UnknownLabelError,
        StockFactorError,
        ListedCompaniesFormatError,
        EarningsCreditedFormatError,
        FreFormatError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE

    def line(title: str, counts: dict[str, int]) -> None:
        print(f"{title}: " + ", ".join(f"{key} {count}" for key, count in counts.items()))

    line("events by source", summary.events["source"])
    line("events by status", summary.events["status"])
    line("ex-sessions", summary.ex_sessions)
    print(f"conflicts: {summary.conflicts}")
    line("unresolved", summary.unresolved)
    snapshots = summary.inputs["snapshots"]
    names = ", ".join(str(name) for name in snapshots) if isinstance(snapshots, list) else ""
    print(f"snapshots: {names or 'none'}; FRE files: {summary.inputs['fre_files']}")
    if summary.inputs["overrides"] is None:
        print("built without corporate_actions.yaml")
    return EXIT_OK


def _data_build_adjustment_factors(args: argparse.Namespace) -> int:
    """Compute the adjustment factors and print their summary."""
    from ibexQuant.data.adjustment import build_adjustment_factors
    from ibexQuant.data.reference import SecurityMasterError
    from ibexQuant.data.reference.corporate_actions import LedgerNotBuiltError

    try:
        summary = build_adjustment_factors(DataStore.open())
    except (SecurityMasterError, LedgerNotBuiltError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE

    def counts(values: dict[str, int]) -> str:
        return ", ".join(f"{key} {count}" for key, count in values.items()) or "none"

    print(f"factors: {summary.factors}")
    print(f"rejected: {counts(summary.rejected)}")
    print(f"ignored events: {counts(summary.ignored)}")
    print(f"ex-sessions kept raw: {counts(summary.unexplained)}")
    return EXIT_OK


def _data_review_events(args: argparse.Namespace) -> int:
    """Print the ex-sessions to review, as a table or as override entries to edit."""
    import pandas as pd

    from ibexQuant.data.reference.corporate_actions import LedgerNotBuiltError, read_ledger

    try:
        events, ex = read_ledger(DataStore.open())
    except LedgerNotBuiltError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    statuses = args.status or list(REVIEWED_BY_DEFAULT)
    shown = ex[ex["status"].isin(statuses)]
    unplaced = events[events["status"] == "review"]
    if shown.empty and unplaced.empty:
        print("Nothing to review.")
        return EXIT_OK

    if args.yaml:
        print(_events_as_yaml(shown, events))
        return EXIT_OK

    columns = [
        "status",
        "ticker",
        "isin",
        "previous_session",
        "session_date",
        "previous_close",
        "open",
        "open_ratio",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        if not shown.empty:
            print(shown[columns].to_string(index=False))
            print(f"\n{len(shown)} ex-session(s).")
        if not unplaced.empty:
            print("\nEvents whose ISIN traded on both sides of the data-com without a change:")
            print(
                unplaced[
                    ["isin", "last_cum_date", "event_type", "cash_amount", "source"]
                ].to_string(index=False)
            )
    return EXIT_OK


def _events_as_yaml(ex: pd.DataFrame, events: pd.DataFrame) -> str:
    """Render ex-sessions as ``corporate_actions.yaml`` entries, to review and edit."""
    inferred = events[events["status"] == "inferred"].set_index(["isin", "ex_date"])
    rejections, accepted = ["rejections:"], ["accepted:"]
    for record in ex.to_dict("records"):
        day = str(record["session_date"])[:10]
        where = (
            f"{record['ticker']}, previous close {record['previous_close']:g} on "
            f"{str(record['previous_session'])[:10]}, open {record['open']:g} "
            f"(ratio {record['open_ratio']:.4f})"
        )
        if record["status"] == "inferred":
            ratio = inferred.loc[(record["isin"], record["session_date"]), "share_ratio"]
            kind = inferred.loc[(record["isin"], record["session_date"]), "event_type"]
            rejections += [
                f"  # inferred {kind} {float(ratio):g}: {where}",
                f"  # - {{isin: {record['isin']}, ex_date: {day}, type: {kind}, "
                'source: inferred, note: "TODO"}',
            ]
        else:
            accepted += [
                f"  # {where}",
                f'  - {{isin: {record["isin"]}, session: {day}, note: "TODO"}}',
            ]
    parts = [block for block in (rejections, accepted) if len(block) > 1]
    return "\n".join("\n".join(block) for block in parts)


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
    actions = targets.add_parser("corporate-actions", help="The corporate actions ledger.")
    actions.add_argument(
        "--no-overrides",
        action="store_true",
        help="Build without corporate_actions.yaml.",
    )
    actions.set_defaults(handler=_data_build_corporate_actions)
    factors = targets.add_parser(
        "adjustment-factors", help="Price adjustment factors from the ledger."
    )
    factors.set_defaults(handler=_data_build_adjustment_factors)

    review = data_commands.add_parser("review", help="Review decisions waiting for a human.")
    subjects = review.add_subparsers(dest="subject", required=True, metavar="SUBJECT")
    links = subjects.add_parser("links", help="Rename candidates of the security master.")
    links.add_argument(
        "--status",
        action="append",
        choices=CANDIDATE_STATUSES,
        help="Only candidates with this status (repeatable).",
    )
    links.add_argument(
        "--yaml", action="store_true", help="Print ticker_links.yaml entries instead of a table."
    )
    links.set_defaults(handler=_data_review_links)
    review_events = subjects.add_parser("events", help="Ex-sessions of the ledger to review.")
    review_events.add_argument(
        "--status",
        action="append",
        choices=EX_SESSION_STATUSES,
        help="Only ex-sessions with this status (repeatable; default: "
        + ", ".join(REVIEWED_BY_DEFAULT)
        + ").",
    )
    review_events.add_argument(
        "--yaml",
        action="store_true",
        help="Print corporate_actions.yaml entries instead of a table.",
    )
    review_events.set_defaults(handler=_data_review_events)

    return parser
