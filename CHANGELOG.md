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
