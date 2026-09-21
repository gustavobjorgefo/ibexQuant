# ibexQuant\scripts\check_data_pipeline.py

"""
Manual, network-hitting check of the full ingestion pipeline.

Not a test — a plain script for visually eyeballing get_bars(),
check_bars(), and align_to_calendar() working together against real
Yahoo Finance data. Run it by hand after touching any of the three;
nothing here is enforced by CI or asserted against a known-good value.

Usage
-----
python scripts/check_data_pipeline.py
"""

from __future__ import annotations

from ibexQuant.data.calendars.calendar import align_to_calendar
from ibexQuant.data.calendars.pandas_market_calendars.calendar import (
    PandasMarketCalendarsCalendar,
)
from ibexQuant.data.ingestion.validation import check_bars
from ibexQuant.data.ingestion.yfinance.provider import YFinanceProvider

_SYMBOLS: tuple[str, ...] = ("PETR4.SA", "VALE3.SA", "ITUB4.SA")


def main() -> None:
    """Run get_bars -> check_bars -> align_to_calendar and print each step."""
    provider = YFinanceProvider(frequency="1d")
    data = provider.get_bars(list(_SYMBOLS), start="2024-01-01", end="2024-01-31")

    print("--- 1. get_bars() ---")
    print(f"{len(data)} rows, {data.index.get_level_values('symbol').nunique()} symbols")
    print(data.head())

    print("\n--- 2. check_bars() ---")
    issues = check_bars(data)
    flagged = issues[issues.any(axis=1)]
    if flagged.empty:
        print("no issues found")
    else:
        print(f"{len(flagged)} row(s) flagged:")
        print(flagged)
    print("\nissue counts per rule:")
    print(issues.sum())

    print("\n--- 3. align_to_calendar() ---")
    b3 = PandasMarketCalendarsCalendar("B3")
    aligned = align_to_calendar(data, calendar=b3)
    gaps = aligned[aligned["close"].isna()]
    print(f"{len(aligned)} rows after alignment (was {len(data)} before)")
    if gaps.empty:
        print("no real gaps revealed — every valid B3 session in range has data")
    else:
        print(f"{len(gaps)} real gap(s) found (valid session, no data):")
        print(gaps)


if __name__ == "__main__":
    main()
