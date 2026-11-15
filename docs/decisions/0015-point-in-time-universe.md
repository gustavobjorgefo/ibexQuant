# 0015 — Point-in-time universe

- Status: Proposed
- Date: 2026-11-15
- Phase: H5

## Context

A backtest that picks its stocks from today's index list only trades the survivors. ADR 0004
gave the `Panel` a boolean universe mask (timestamp × symbol) and left pending item P3: the
index composition by effective date and a point-in-time liquidity filter. H5 builds both,
on top of the security master (ADR 0012), the ledger (ADR 0013) and the adjusted provider
(ADR 0014).

The universe is pluggable (D4): an index name, a list of tickers, or a rule, and
combinations of them. Index weights are not used (D4); the theoretical quantity and
weight are stored because they cost nothing. The composition of each period is a proxy for
selecting assets: later filters of the strategy (minimum history, liquidity) remove new or
leaving assets from trading decisions.

H5 also adds a one-call download for simple work on one or a few tickers, the base's
equivalent of `yfinance.download`.

### What the data shows

| Finding | Evidence | Decision |
|---|---|---|
| B3 revises every index portfolio each four months (January–April, May–August, September–December). The start of validity has had three rules: the first business day of the month up to May 2011; the first Monday (or the next session) from September 2011 (Ofício Circular 007/2011-DT); the second Monday from January 2027 (Ofício Circular 068/2026-VPC, 27/10/2026). A period ends on the session before the next one starts | *Histórico de Adequações Metodológicas*; *Manual de Definições e Procedimentos dos Índices da B3*, 2.2; OC 068/2026-VPC | H5.3 |
| The rule reproduces all 21 starts of the hand-kept table 2017-09 to 2024-05. Three hand-kept ends were wrong: 2018-12-28 (the next period started on 2019-01-07), 2022-12-30 and 2023-12-29 (no trading on either day) | Hand-kept table against the B3 calendar | H5.3 |
| Members and theoretical quantities of a new portfolio are set by the calculations of the 3rd preview; the definitive portfolio ("Virada") follows it | Manual, 2.3 | H5.1 |
| The B3 workbooks hold one sheet per index (23 in 2017, 40 in 2026). The title says what the file is ("3ª Prévia do IBOVESPA para setembro de 2025", "Virada do IBOVESPA para janeiro de 2025", "2ª Prévia ... maio de 2026"), not the file name: the January 2025 file saved as a 3rd preview is the Virada. Headers and the base-session footer changed format over time (`COD.`/`CÓDIGO`; `PREGÃO BASE: 31/08/2017` / `Pregão Base:` with a date cell). Sheet codes changed too (`IBRX`, `IBxl`, `IEE`, `IGC`, `ISE` in older files) | 3rd previews 2017-09 to 2026-09, the January 2025 Virada, the May 2026 2nd preview | H5.1, H5.2 |
| No 3rd preview of May 2026 was published for download; the 2nd preview (79 IBOV members, as many as the 3rd preview announced) stands in until B3 sends it | B3 news page; 2nd preview of 14/04/2026 | H5.1 |
| B3's retroactive Ibovespa under the 2013 methodology lists members and weights of 34 portfolios, January 2003 to January 2014. B3 calls it illustrative: the index of that time used other rules | *Série retroativa do Ibovespa*, B3 | H5.1 |
| The retroactive tables convert cleanly: 1,870 rows, 34 portfolios of 37 to 73 members, each adding up to 100% within 0.05 point; three rows lack the share class (`KROT3`, `VCPA3`) | Conversion of the PDF | H5.2 |
| Sheet codes: `IBRX`, `IBxl`, `IEE`, `IGC`, `ISE` in the 2017–2018 files; `IBXX`, `IBXL`, `IEEX`, `IGCX`, `ISEE` from 2024 | Sheets of the files at hand | H5.2 |
| No composition exists for May 2014 to May 2017 (10 portfolios) nor before 2003 | Above | H5.6 |

## Decision

### Sources

- **H5.1 — One composition per index and period**, from the best source available:
  `definitive` > `third_preview` > `second_preview` > `retroactive` > `liquidity_rule`.
  The source is recorded on every row. A better file added later (the 3rd preview of May
  2026) replaces the stand-in on the next build; nothing is edited by hand.

- **H5.2 — Raw files as downloaded**, in the data root:
  - `raw/b3/index_portfolios/workbooks/*.xlsx`: B3 previews and definitive portfolios, any
    file name. The reader takes the kind (definitive, 3rd or 2nd preview) and the period
    from the title of the IBOV sheet, the base session from the footer, and every sheet as
    one index. Sheet codes are mapped to today's codes through one alias table, checked
    against the files: `IBRX` → `IBXX`, `IBxl` → `IBXL`, `IEE` → `IEEX`, `IGC` →
    `IGCX`, `ISE` → `ISEE`. A file with both an alias and its code must list the same
    members under both, or the build stops. A title it cannot read stops the build with the file name.
  - `raw/b3/index_portfolios/retroactive/`: B3's retroactive series as published (PDF)
    and its portfolios converted once to `ibov_retroactive_2003_2014.csv` (`period`,
    `ticker`, `name`, `share_class`, `weight_pct`) by a converter kept in the private
    configuration, which checks that every portfolio adds up to 100% within 0.1 point. The
    build reads the CSV (source `retroactive`, index `IBOV`): it needs no PDF library, and
    a 77-page table is converted once, not on every build.
  - B3's methodology documents live with the private configuration
    (`references/b3_index_methodology/`), not in this repository.

### Calendar

- **H5.3 — Periods from rules.** `index_periods` is computed from the B3 session calendar
  and a list of rule versions, each with the first period it applies to:

  | From period | Start of validity |
  |---|---|
  | (all before) | first session of January, May or September |
  | 2011-09 | first Monday, or the next session |
  | 2027-01 | second Monday, or the next session |

  A period ends on the session before the next period starts. A period is labelled by the
  year and month it starts: `2025-09`. A one-off change by B3 goes into
  `index_periods.yaml` in the overrides root (period, start, note). The calendar serves
  the base; exact publication dates for a rebalance trade are checked against B3's notices,
  outside it.

### Constituents

- **H5.4 — Curated table.** `curated/b3/index_constituents.parquet`, one row per index,
  period and ticker: `index`, `period`, `ticker`, `security_id`, `name`, `share_class`,
  `theoretical_quantity`, `weight_pct`, `base_session`, `source`, `source_file`; and
  `curated/b3/index_periods.parquet`: `period`, `effective_from`, `effective_to`,
  `rule`. Both are rebuilt from raw by `ibexquant data build universe`.

- **H5.5 — Tickers to securities.** Each ticker resolves to a `security_id` with the
  security master on the base session (`resolve(ticker, as_of=base_session)`), falling
  back to the period's start. An unresolved ticker is listed with index and period; for
  `IBOV` and `IBXX` the list must be empty to close the phase. Fixes go to
  `ticker_links.yaml` (H2), not to this module.

### The mask

- **H5.6 — Universe specification.** A universe is one of:
  - `Index("IBOV")`: members of the period in force at each session;
  - `Tickers(["PETR4", ...])`: fixed list, resolved per session through the security
    master, so a renamed ticker stays in;
  - `LiquidityRule(top=..., lookback=..., min_presence=...)`: at each period's base
    session, the securities ranked by median daily traded value over the lookback, with a
    minimum share of sessions traded, using only data up to that session;

  and the combinations `a | b`, `a & b`, `a - b` (for example `Index("IBXX") -
  Index("IBOV")`). Before 2003 and from May 2014 to May 2017, `Index("IBOV")` takes the
  `liquidity_rule` source with the IBOV defaults (H5.9), unless the caller asks for
  index data only (`fill=False`), which leaves those periods empty.

- **H5.7 — Mask = membership.** `universe_mask(universe, sessions)` returns a boolean
  frame, sessions × `security_id`, true when the security belongs to the universe in that
  session's period. Whether it traded that day is the panel's business (missing bars);
  the provider's labelling (ticker in force at the anchor, ADR 0014) is applied when the
  mask joins a `Panel`.

- **H5.8 — Exits are recorded, not priced.** A member whose last session falls inside its
  period is listed in `curated/b3/index_exits.parquet` with the last session and the
  reason, taken from the ledger when there is one (conversion, exchange, redemption,
  other) or `no_event`. How a backtest values a holding that stops trading is the backtest
  engine's decision (ADR 0014, H4.2; pending item P10).

### Liquidity

- **H5.9 — Traded value is raw.** The provider serves `traded_value` (BRL), never
  adjusted: money traded is money traded. The IBOV-like defaults for `LiquidityRule` are a
  12-month lookback (three portfolios, as the Ibovespa methodology does), 95% presence and
  the number of members of the nearest real IBOV portfolio; the gap between the rule and
  the real index is measured on 2017–2026, where both exist, and reported with the build.

### Download

- **H5.10 — `ibexQuant.download`.** `download(tickers, start, end, adjustment="total_return",
  anchor=None, fields=None)` opens the configured data root and calls `B3Provider`:
  one ticker gives a frame indexed by session with `open`, `high`, `low`, `close`,
  `volume` (and `traded_value` if asked); several tickers give columns `(field, ticker)`,
  as `yfinance.download` does. `adjustment="raw"` serves raw bars. It is a convenience:
  research with a universe goes through the dataset API of H6.

## Alternatives considered

- **A hand-kept table of effective dates.** Rejected: the rule reproduces it, and the table
  had three wrong ends. The table becomes a test.
- **Definitive portfolio only.** Not available before 2025 in the files at hand; the 3rd
  preview sets the members by methodology, so it is the source of record when the
  definitive is missing.
- **Only the liquidity rule before 2017.** Rejected for 2003–2014: B3's retroactive
  portfolios apply one methodology uniformly and are closer to the index than any rule of
  ours. The rule stays for the gaps.
- **Backtests only from 2017-09.** Kept as an option (`fill=False`), not the default.
- **Daily composition from the b3-data-collector.** Deferred: not needed for the base now.
- **Weights in the mask.** Rejected (D4).

## Consequences

- Universes are survivorship-free by construction: every security ever in a portfolio is in
  the table, delisted or not, and the mask follows each period.
- Each row says how good it is (`source`); research can drop `retroactive` or
  `liquidity_rule` periods when it needs real compositions only.
- Within a period the mask does not follow B3's intra-period changes (exclusions, spin-off
  inclusions); the error is small for a selection proxy and is documented.
- New pending item **P10**: valuation of holdings that stop trading inside a period
  (delisting returns). Resolve in: backtest engine design.
- Closing criteria: `IBOV` and `IBXX` resolve every ticker; every period from 2003 has a
  composition with its source; members traded on the period's first session or are in the
  exits table; the liquidity rule's overlap with the real IBOV is reported.
