# ibexQuant\scripts\check_yfinance_provider.py

"""
Manual, network-hitting check of YFinanceProvider.

Not a test — this is a plain script for visually eyeballing what
YFinanceProvider actually returns from the live Yahoo Finance API. Run it
by hand after touching the provider, or after a yfinance upgrade, since
nothing here is enforced by CI or asserted against a known-good value.

Usage
-----
python scripts/check_yfinance_provider.py
"""

from __future__ import annotations

from ibexQuant.data.ingestion.provider import OHLCV_FIELDS
from ibexQuant.data.ingestion.yfinance.provider import YFinanceProvider

_SYMBOLS: tuple[str, ...] = ("PETR4.SA", "VALE3.SA", "ITUB4.SA")


def main() -> None:
    """Download a couple weeks of data for three B3 tickers and print it."""
    provider = YFinanceProvider(frequency="1d")

    data = provider.get_bars(
        symbols=list(_SYMBOLS),
        start="2024-01-01",
        end="2024-01-15",
    )

    print("--- YFinanceProvider manual check ---")
    print(data)
    print("\ndtypes:")
    print(data.dtypes)
    print("\nrows per symbol:")
    print(data.groupby(level="symbol").size())
    print("\ncolumns match OHLCV_FIELDS:", set(data.columns) == set(OHLCV_FIELDS))


if __name__ == "__main__":
    main()
