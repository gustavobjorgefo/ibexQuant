# ibexQuant\scripts\check_b3_provider.py

"""
Manual check of the curated B3 base and of B3Provider (ADR 0011).

Not a test — a plain script for eyeballing what the local B3 base holds after
``ibexquant data ingest cotahist``. It reads the machine's real data root
(``IBEXQUANT_DATA_ROOT`` or ``~/.ibexquant/config.toml``), so nothing here
runs in CI or is asserted against a known-good value. Run it by hand after a
(re)ingestion or after touching the parser or the provider.

It prints:

- the manifest, one line per year (rows, sessions, range, trailer
  convention, repaired records, source hash);
- sessions per ticker since 2000 for a few reference tickers;
- the first and last sessions of one ticker in one year, to compare with an
  independent source;
- the ``check_bars`` findings over everything it loaded.

Usage
-----
python scripts/check_b3_provider.py
python scripts/check_b3_provider.py --tickers PETR4 ITUB4 --year 2023
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Sequence
from typing import Final

from ibexQuant.data.ingestion.b3 import B3Provider, read_manifest
from ibexQuant.data.ingestion.validation import check_bars
from ibexQuant.data.store import DataStore

# PETR4 and VALE3 cover the whole period; the three Casas Bahia tickers are
# one company under three names, which H2 links into one series.
_DEFAULT_TICKERS: Final[tuple[str, ...]] = ("PETR4", "VALE3", "BHIA3", "VIIA3", "VVAR3")
_DEFAULT_START: Final[str] = "2000-01-01"
_DEFAULT_YEAR: Final[int] = 2024


def main(argv: Sequence[str] | None = None) -> None:
    """Print the manifest, coverage per ticker, a sample year and check_bars findings."""
    args = _parse(argv)
    store = DataStore.open()

    print("--- Manifest ---")
    print("year  rows      sessions  first       last        trailer          repaired  sha256")
    for year, entry in read_manifest(store).items():
        print(
            f"{year}  {entry.rows_kept:<9} {entry.sessions:<9} {entry.first_session}  "
            f"{entry.last_session}  {entry.trailer_convention:<16} {entry.rows_repaired:<9} "
            f"{entry.source_sha256[:12]}"
        )

    bars = B3Provider(store).get_bars(
        list(args.tickers), _DEFAULT_START, f"{args.end_year}-12-31", adjusted=False
    )

    print(f"\n--- Sessions per ticker since {_DEFAULT_START} ---")
    print(bars.groupby(level="symbol").size())

    sample = args.tickers[0]
    year_bars = bars.xs(sample, level="symbol").loc[str(args.year)]
    print(f"\n--- {sample}: first and last sessions of {args.year} ---")
    print(year_bars.head(3))
    print(year_bars.tail(3))

    print("\n--- check_bars findings (rows flagged per rule) ---")
    print(check_bars(bars).sum())


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual check of the curated B3 base.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(_DEFAULT_TICKERS),
        help="Tickers to load; the first one is sampled by year.",
    )
    parser.add_argument("--year", type=int, default=_DEFAULT_YEAR, help="Year to sample.")
    parser.add_argument(
        "--end-year",
        type=int,
        default=dt.date.today().year,
        help="Last year loaded (default: the current year).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
