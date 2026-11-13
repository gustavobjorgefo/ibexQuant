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
