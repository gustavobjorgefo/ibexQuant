# Changelog

## [Unreleased]
### Added
- Initial project skeleton.
- `ibexQuant.data.store`: data root resolution (`resolve_data_root`), `DataStore` with the
  `raw/`, `curated/` and `derived/` zones, and the `DataStoreError` family (ADR 0010).
- `ibexquant` command line with `ibexquant data init`.
- `ibexQuant.data.ingestion.b3`: COTAHIST parser (`read_cotahist`), curated `bars_daily` build
  with a per-year manifest (`build_bars_daily`, `read_manifest`) and `B3Provider`, a raw
  daily-bar `MarketDataProvider` over the B3 base (ADR 0011).
- `ibexquant data ingest cotahist (--year YEAR ... | --all)`.
- `b3` optional extra (`pyarrow`), included in `research`.
- `ibexQuant.data.reference`: security master (`build_security_master`, `SecurityMaster` with
  `resolve`, `spans` and `ticker_at`), rename candidates with review statuses and
  `implied_factor`, and `ticker_links.yaml` overrides (`load_link_overrides`) (ADR 0012).
- `ibexQuant.data.ingestion.cvm`: FCA securities reader (`read_fca_securities`) used to
  confirm renames by CNPJ.
- `B3Provider(follow_renames=True)`: a ticker returns the full history of its security.
- `ibexquant data build security-master [--no-overrides]` and
  `ibexquant data review links [--status ...] [--yaml]`.
- Overrides root resolution (`resolve_overrides_root`, `IBEXQUANT_OVERRIDES_ROOT`).
- `pyyaml` in the `b3` extra.
- `ibexQuant.data.reference`: corporate actions ledger (`build_corporate_actions`,
  `read_ledger`): ex-sessions from COTAHIST distribution numbers, events from B3's
  listed-company service, the CVM reference forms and *Earnings credited*, price-inferred
  splits, and `corporate_actions.yaml` overrides (`load_action_overrides`) (ADR 0013).
- `ibexQuant.data.ingestion.b3`: readers of B3's listed-company snapshots
  (`read_listed_companies`, cash history under current and old trading names) and of the
  *Earnings credited* files (`read_earnings_credited`).
- `ibexQuant.data.ingestion.cvm`: stock events of the reference forms (`read_fre_stock_events`).
- `ibexquant data build corporate-actions [--no-overrides]` and
  `ibexquant data review events [--status ...] [--yaml]`.
- `ibexQuant.data.adjustment`: anchored price adjustment (`build_adjustment_factors`,
  `read_adjustment_factors`, `adjust`, `AdjustmentMode`): one factor per ex-session from the
  ledger (gross cash, spin-offs, subscription rights, share ratios), applied relative to an
  anchor date (ADR 0014); `ibexquant data build adjustment-factors`.
- `B3Provider.get_bars` serves adjusted bars by default (`total_return` or `splits`, `anchor`
  defaulting to `end`) and warns about unexplained jumps in the range.
- `corporate_actions.yaml`: `value_fraction` for spin-offs whose target never trades.
- `ibexQuant.data.universe`: index periods from the B3 calendar and three rule versions
  (`index_periods`, `assign_periods`, `index_periods.yaml` overrides) and index compositions
  by period from B3's workbooks and retroactive series, resolved to securities
  (`build_universe`, `read_index_constituents`, `read_index_periods`) (ADR 0015);
  `ibexquant data build universe [--no-overrides]` and `ibexquant data review universe`.
- `B3Dataset` (`from ibexQuant.data import B3Dataset`): `panel(universe, start, end,
  adjustment, anchor, warmup, fields)` builds a `Panel` with `security_id` columns, B3
  sessions and the point-in-time mask; lookups `constituents`, `rebalance_dates`, `events`,
  `exits`; tickers for people with `ticker_at`, `ticker_history` and `label` (ADR 0016).
- `universe_mask(..., data=...)` reuses tables already read.
- Notebook `research/backtests/0002_ibov_point_in_time.ipynb`: equal-weight IBOV,
  point-in-time against today's members and BOVA11; survivorship gap of 9.2% a year on B3
  from 2003 (ADR 0016, H6.10).
- `CotahistCalendar`: B3 sessions as recorded in `bars_daily`, read on first use and kept in
  memory, with `offset` to count sessions back (ADR 0016, H6.8).
- `B3Provider.get_security_bars`: daily bars by `security_id`, through the same read and
  adjustment path as `get_bars` (ADR 0016, H6.7).
- `ibexQuant.download(tickers, start, end, adjustment=..., anchor=..., fields=...)`: the B3
  provider's daily bars, wide like `yfinance.download` (one ticker: fields; a list:
  `(field, ticker)` columns), loaded lazily from the package root (ADR 0015, H5.10).
- E1 position engine (ADR 0017, B1.9–B1.16): `simulate_weights(bars, weights,
  execution="next_open" | "next_close", cost_bps=0.0)` simulates target weights decided at
  closes, with drift between rebalances, signed units, suspensions kept, exits as cash at no
  cost and a linear cost; `BacktestResult` gives `returns`, `contributions`, `weights` (with
  `cash`), `turnover` and `costs`, all over the previous equity, plus `equity`, exposures and
  `summary`. `equal_weight(mask)`. `WalkForward.bars(ds, folds)` loads one panel for the
  run, and `FoldData.decisions` gives each fold's decision sessions.
- `ibexQuant.backtest.walkforward`, layers 2 to 4 (ADR 0017, B1.5–B1.8): universe rules
  (`UniverseRule`, `StaticUniverse`, `IndexComposition` with `at_rebalance="split"` or
  `"hold"`), eligibility rules (`EligibilityRule`, `HasHistory`, `StillTrading`,
  `MinimumPresence`) and `WalkForward`, whose `folds(ds)` gives `Fold` objects with their
  excluded candidates and `load(ds)` gives their `FoldData`; `WalkForward.summary(folds)`.
- `B3Dataset.members(universe, start, end)`: the membership mask `ds.panel` uses, without
  bars (ADR 0017, B1.8).
- `ibexQuant.backtest.walkforward`, layer 1 (ADR 0017, B1.4): `WalkForwardSchedule(start,
  end, train_months=..., test_months=..., step_months=None, expanding=False)` cuts any
  `TradingCalendar` into `Window` objects (training, decision and test sessions), with
  `window(first, last, calendar)` for one range of test months, `frame(calendar)` and
  `tiles`; `WalkForwardError`.
- `Securities([...])`: a universe of fixed `security_id` values, members at every session
  (ADR 0017, B1.8).
- `B3Dataset.composition(index, period)`: the members of one index period by its label,
  before B3's effective date if a study wants it; `constituents` reads through it
  (ADR 0017, B1.8).
- Universe mask (ADR 0015): `Index`, `Tickers` and `LiquidityRule` specifications,
  combined with `|`, `&` and `-`, and `universe_mask` (sessions × `security_id`). IBOV
  periods without a B3 portfolio take the liquidity rule (`liquidity_rule` source); the
  rule's overlap with the real index, members that stop trading inside a period
  (`index_exits.parquet`, `read_index_exits`) and members absent on a period's first
  session are reported by `ibexquant data build universe`.
- `B3Provider.get_bars(..., fields=[..., "traded_value"])`: raw traded value in BRL,
  never adjusted.
- `ibexQuant.data.ingestion.b3`: readers of B3's index portfolio workbooks and of the
  retroactive Ibovespa CSV (`read_workbook`, `read_retroactive`); `openpyxl` joins the `b3`
  extra.

### Changed
- `B3Provider` logs the requested symbols without data in a range as one warning per
  request instead of one per symbol.

### Fixed
- The corporate actions build no longer fails on a base without ex-sessions.
- B3 cash history counted once when two codes of one chain (`ITAU`, `ITUB`) return it.
