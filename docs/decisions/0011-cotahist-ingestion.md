# 0011 — COTAHIST ingestion

- Status: Proposed
- Date: 2026-11-09
- Phase: H1

## Context

B3 publishes one file per year with every quote of every instrument traded on
the exchange: `COTAHIST_AYYYY.ZIP`, from the "Cotações históricas" page. The
format is fixed-width text described in *Layout do arquivo – Cotações
Históricas*, revision 02 of 2020-10-05 (kept in the private
`ibexquant-data-config` repository, `docs/b3/`; see 0010, H0.1).

H1 turns those files into a typed daily-bar table in `curated/` and serves it
through a `MarketDataProvider`, with raw (unadjusted) prices only. Adjustment
is H4; identifiers that survive ticker changes are H2.

### What the files actually contain

Before this record, every annual file from 1994 to 2026 was profiled (33 files,
23.8 million records), plus the B3 demo file and samples of 2000 and 2024. The
findings below are the evidence behind each decision; a parser that follows
the layout document alone would get several of them wrong.

| Finding | Evidence | Decision |
|---|---|---|
| Every record is 245 bytes, Latin-1, CRLF | All years | H1.3 |
| One record in 2026 is 246 bytes: a company name written in UTF-8 (`FROTA OCEÂNICA`, market 017) | 2026, line 2,072,135 | H1.4 |
| The trailer counts header and trailer, as the layout says, up to 2024; from 2025 it counts detail records only | Trailer vs. records in every year | H1.3 |
| The 2025 file first obtained was a partial download (last session 2025-05-09) | Profile; replaced by the complete file | H1.2 |
| `FATCOT` varies per row on the same day; values 1, 10, 100, 1,000, 10,000, 100,000 and 1,000,000 all occur, the last in 2016, 2017 and 2026 | All years | H1.7 |
| `QUATOT` is a count of shares, not of quotation lots: `VOLTOT ≈ PREMED / FATCOT × QUATOT` in every sampled row | Samples 2000 and 2024 | H1.7 |
| Currency is `CR$` before the Real (57,049 equity rows in 1994) and `R$` afterwards | 1994 | H1.6 |
| ISIN is never blank; before 1995-05-15 the field holds an exchange-internal code (layout) | All years | H1.9 |
| `(session, ticker, market)` repeats: `ICP 4` in 1996 and `BIOM12F` on five days of 2022 — each time two securities with different ISINs (a share and a receipt or warrant) | Anomaly report | H1.8 |
| Companies in distress leave BDI 02: `BHIA3` trades under BDI 07 (extrajudicial recovery) and `RNEW11` under 08 (judicial recovery) in 2024 | Sample 2024 | H1.5 |
| BDI codes 13, 34, 35, 36 and market 021 are missing from the 2020 layout; 13 is Fiagro, 34 BDR (`DRN`), 36 BDR of ETF (`DRE`) | Samples, profile | H1.5 |
| `ESPECI` mixes share class, governance segment and event markers (`PN EJ`, `ON EDJ N2`, `PN *`) | Samples | H1.9 |

## Decision

### Source files

- **H1.1 — Annual files only.** The source is `COTAHIST_AYYYY.ZIP` from B3,
  stored as downloaded in `raw/b3/cotahist/`. File names are matched without
  regard to case (`.zip` and `.ZIP` both occur). Daily and monthly files are
  not used: the annual file of the current year is updated by B3 and holds
  the same records.
- **H1.2 — Closed years are immutable; the current year is replaced.** This
  refines H0.5 for a source that is still being written. A year is closed
  once B3 has published its final file. Until then, refreshing the current
  year means replacing its ZIP in `raw/` with a new download, as often as
  wanted. The manifest (H1.12) records which file — by SHA-256 — produced the
  curated data, so a number can always be traced to the exact download.

### Parsing

- **H1.3 — Structure is validated before any value is read.** For each file:
  - exactly one header (`00`) first and one trailer (`99`) last;
  - the year in the header (`COTAHIST.AAAA`) equals the year in the file name;
  - the trailer count equals either all records (header and trailer
    included, as in the layout) or detail records only (as observed from
    2025). Any other count is an error. The convention found is recorded in
    the manifest;
  - every other record is a detail record (`01`).

  A file that fails any of these is not written to `curated/`.
- **H1.4 — Encoding: Latin-1, with one repair.** Records are decoded as
  Latin-1. A record whose length is not 245 bytes is decoded as UTF-8 and
  re-encoded as Latin-1; if that yields 245 bytes, the record is kept and
  counted as repaired in the manifest. Otherwise the file is rejected. The
  rule is narrow on purpose: it fixes the one defect observed, and anything
  else still fails loudly.
- **H1.5 — Market filter on ingestion, no BDI filter.** Only detail records
  of market types `010` (cash, round lot) and `020` (cash, odd lot) are kept;
  options, forwards, futures, exercises, auctions (`017`) and the unlisted
  `021` are dropped. **Every BDI code is kept**, with `bdi_code` as a column.
  A company under judicial or extrajudicial recovery moves from BDI 02 to
  05–08 exactly when it falls; filtering on BDI here would remove it from
  the base at that moment and plant a survivorship bias. What is tradable
  is a universe decision (H5), made on top of complete data.
- **H1.6 — Coverage starts on 1994-07-01.** Records before the Real are
  dropped (0010, H0.11). After the cut, a currency other than `R$` is an
  error, not a filter.
- **H1.7 — Units.** Prices are stored per share in BRL:
  `field / 100 / FATCOT`, row by row. `volume_shares` is `QUATOT` as is,
  because it already counts shares. `traded_value_brl` is `VOLTOT / 100`.
  `FATCOT` is kept as a column for audit.
- **H1.8 — Key.** A row is identified by
  `(session_date, ticker, market_type, isin)`. A repeated key is an error.
  The ISIN is part of the key because the same ticker can carry two
  securities on the same day (H1 evidence: 1996, 2022).

### The table

- **H1.9 — Schema of `curated/b3/bars_daily`.**

  | Column | Type | Source (layout position) |
  |---|---|---|
  | `session_date` | `timestamp`, date at midnight, no time zone | `DATA DO PREGÃO` (3–10) |
  | `ticker` | `string` | `CODNEG` (13–24), stripped |
  | `isin` | `string` | `CODISI` (231–242); exchange-internal code before 1995-05-15 |
  | `market_type` | `int16` | `TPMERC` (25–27) |
  | `bdi_code` | `int16` | `CODBDI` (11–12) |
  | `specification` | `string` | `ESPECI` (40–49), stripped, as published |
  | `company_name` | `string` | `NOMRES` (28–39), stripped |
  | `distribution_number` | `int16` | `DISMES` (243–245) |
  | `open`, `high`, `low`, `close`, `average` | `float64` | `PREABE`, `PREMAX`, `PREMIN`, `PREULT`, `PREMED` |
  | `best_bid`, `best_ask` | `float64` | `PREOFC`, `PREOFV` |
  | `trades` | `int64` | `TOTNEG` (148–152) |
  | `volume_shares` | `int64` | `QUATOT` (153–170) |
  | `traded_value_brl` | `float64` | `VOLTOT` (171–188) |
  | `quotation_factor` | `int32` | `FATCOT` (211–217) |

  `specification` is stored as published; splitting it into class, segment
  and event marker is left to H2 and H3, which are the phases that need it.
  Fields that only describe options and forwards (`PRAZOT`, `PREEXE`,
  `INDOPC`, `DATVEN`, `PTOEXE`) and the currency (constant after H1.6) are
  not stored.
- **H1.10 — Values are reported, never corrected.** Ingestion validates
  structure (H1.3–H1.8) and nothing else. Value checks (OHLC consistency,
  negative values) belong to `check_bars`, which reports and never edits
  (see `ibexQuant.data.ingestion.validation`). Exchange data is stored as the
  exchange published it.
- **H1.11 — One partition per year, written atomically.**
  `curated/b3/bars_daily/year=YYYY/part-0.parquet`, sorted by
  `(session_date, ticker)`, zstd-compressed. A build writes to a temporary
  file in the same directory and renames it over the old one, so a failed
  build never leaves a half-written year. Building a year twice from the
  same ZIP produces the same rows.
- **H1.12 — Manifest.** `curated/b3/bars_daily/manifest.json` holds one entry
  per year: source file name, size and SHA-256, header generation date,
  trailer convention, records read, rows kept, rows repaired, sessions,
  first and last session, parser version and build time. The manifest is
  what answers "which download produced this number?" (H1.2).

### Access

- **H1.13 — `B3Provider`.** A `MarketDataProvider` with frequency `"1d"`
  over `bars_daily`:
  - symbols are tickers as B3 writes them (`PETR4`, no `.SA` suffix);
  - only market `010` is served: the odd-lot market (`020`) trades the same
    security under a different ticker (`PETR4F`) and is kept in `curated/`
    for liquidity studies, not for bars;
  - `volume` is `volume_shares`;
  - `adjusted=True` raises `ValueError`, as the contract requires of a
    provider that cannot honor it, until H4;
  - if a requested ticker has two rows on the same session in market `010`,
    `get_bars` raises `ValueError` naming them, since the contract forbids
    duplicate `(timestamp, symbol)` pairs. None was observed.
- **H1.14 — Command line.** `ibexquant data ingest cotahist --year 2024`
  builds one year (the option repeats); `--all` builds every year found in
  `raw/`. Each year prints a summary of its manifest entry. A year that fails
  is reported and skipped, its previous partition untouched; the others are
  still built, and the command exits with code 2.

### Dependencies

- **H1.15 — `pyarrow` only.** The `b3` optional extra adds `pyarrow`, used to
  write and read the partitions (pandas reads them with filters on `year`
  and `ticker`). `duckdb`, announced for H1 in 0010 (H0.9), is deferred until
  a query needs more than filtered reads; this amends H0.9.

### Interface

```python
from ibexQuant.data.ingestion.b3 import B3Provider, build_bars_daily, read_cotahist
from ibexQuant.data.store import DataStore

store = DataStore.open()

frame, report = read_cotahist(store.raw("b3", "cotahist", "COTAHIST_A2024.ZIP"))
(entry,) = build_bars_daily(store, 2024)  # writes the partition + manifest entry
entries = build_bars_daily(store)  # every year found in raw/, one entry each

provider = B3Provider(store)
bars = provider.get_bars(["PETR4", "VALE3"], "2024-01-01", "2024-12-31", adjusted=False)
```

- `read_cotahist(path)` is pure: it reads one ZIP and returns the filtered
  table (H1.9 schema) plus a `CotahistReport` with the counts that go into
  the manifest. It writes nothing.
- `build_bars_daily(store, years=None)` takes one year, several, or none
  (every year in `raw/`). For each, in order, it resolves the ZIP in `raw/`,
  calls `read_cotahist`, writes the partition (H1.11) and the manifest entry
  (H1.12). It returns one entry per year built; the first error stops it,
  leaving the years already built in place.

| Condition | Error |
|---|---|
| No ZIP for the requested year in `raw/b3/cotahist/` | `CotahistNotFoundError` (`FileNotFoundError`) |
| ZIP with no member, or more than one | `CotahistFormatError` (`ValueError`) |
| Header/trailer missing, misplaced or with the wrong year; unknown record type | `CotahistFormatError` |
| Trailer count matches neither convention | `CotahistFormatError` |
| Record not 245 bytes after the UTF-8 repair | `CotahistFormatError`, with the line number |
| Currency other than `R$` on or after 1994-07-01 | `CotahistFormatError`, with the line number |
| Repeated `(session_date, ticker, market_type, isin)` | `CotahistFormatError`, with the key |
| `adjusted=True` in `B3Provider.get_bars` | `ValueError` |
| Duplicate session for one ticker in market `010` | `ValueError` |

`CotahistNotFoundError` and `CotahistFormatError` derive from a common
`CotahistError`.

## Alternatives considered

- **Keep only BDI 02 (round lot).** The common shortcut, and what the previous
  project effectively did. Rejected (H1.5): it drops companies in recovery
  exactly when they fall.
- **Fail on the 246-byte record.** Would make 2026 unreadable because of a
  single auction record outside the market filter. Rejected in favor of a
  narrow, counted repair (H1.4).
- **Drop records with a repeated `(session, ticker, market)`.** Would silently
  choose one of two real securities. Rejected; the ISIN joins the key (H1.8).
- **Store prices as integer cents.** Exact, but every consumer would divide
  by 100 and by `FATCOT`. Rejected: float64 has 15–16 significant digits,
  far beyond the 13 digits of the layout, and the provider contract is
  float64.
- **Skip the manifest.** Rejected: with a replaceable current year (H1.2),
  the manifest is the only link from a curated number to its download.
- **DuckDB from the start.** Deferred (H1.15): filtered parquet reads are all
  H1 needs.

## Consequences

- Building every year takes minutes and needs only `raw/`; `curated/` can
  always be deleted and rebuilt.
- `bars_daily` is an exact, typed copy of the cash market as B3 published it:
  no adjustment, no identifier linking, no tradability filter. Each of those
  is a later phase reading this table.
- Backtests can switch from yfinance to `B3Provider` for unadjusted research
  as soon as H1 lands; adjusted research waits for H4.
- Tests use synthetic fixed-width records built in code, never B3 files
  (0010, H0.2).
