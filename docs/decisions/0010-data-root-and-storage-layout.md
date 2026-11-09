# 0010 — Data root and storage layout

- Status: Accepted
- Date: 2026-11-08
- Phase: H0

## Context

ibexQuant is about to get its own historical data base for B3 equities,
replacing yfinance as the research source for long backtests. It is built
in phases H0–H6 (H = historical data track):

| Phase | Scope                                                         |
|-------|---------------------------------------------------------------|
| H0    | Data root, storage layout, `DataStore` (this record)          |
| H1    | COTAHIST ingestion into daily bars                            |
| H2    | Security master: permanent identifiers across ticker changes  |
| H3    | Corporate actions ledger                                      |
| H4    | Anchored price adjustment                                     |
| H5    | Point-in-time universe (resolves P3)                          |
| H6    | `B3Dataset` API                                               |

The base is built from files that cannot live in this repository:

- they are large (COTAHIST alone covers every trading day since 1994);
- B3 restricts redistribution of its market data;
- part of the corporate actions history comes from secondary sources
  (StatusInvest, yfinance) whose terms do not allow republishing.

The repository is public; the data, the hand-made corrections and the
strategies built on top are private. The code must therefore find the data
by configuration, and nothing in the repository may depend on a particular
machine.

In the previous project, paths such as
`C:\Users\...\OneDrive\...\data\b3_data` were written inside each script.
Moving a folder broke every script that pointed at it, and the same
directory was described in several places.

## Decision

### Where things live

- **H0.1 — Four places, four owners.**

  | Place                                   | Content                                                | Versioning |
  |-----------------------------------------|--------------------------------------------------------|------------|
  | `ibexQuant` (public repository)         | Code, schemas, ADRs, small synthetic test fixtures     | Public git |
  | `ibexquant-data-config` (private repo)  | Override files, one-off bootstrap converters           | Private git |
  | Data root (local disk, later cloud)     | `raw/`, `curated/`, `derived/`                         | None — backups of `raw/` only |
  | `ibexquant-research` (private repo)     | Strategies and research notebooks                      | Private git |

- **H0.2 — The public repository never contains real data.** Tests use
  synthetic fixtures generated in code or committed as small files under
  `tests/`. `.gitignore` blocks data formats everywhere else.

### Finding the data root

- **H0.3 — Resolution order.** The data root is resolved, first match wins,
  from:
  1. an explicit argument (`DataStore.open(root=...)`, `--root` in the CLI);
  2. the environment variable `IBEXQUANT_DATA_ROOT`;
  3. the key `root` of the `[data]` table in `~/.ibexquant/config.toml`.

  Nothing found raises `DataRootNotConfiguredError`, whose message lists the
  three options. There is no default location: a silent default would write
  gigabytes to a place the user never chose.
- **H0.4 — Absolute paths only.** A relative root is rejected, because it
  would make the data root depend on the working directory of whoever runs
  the code. A leading `~` is expanded. An empty environment variable counts
  as unset.

### The layout

- **H0.5 — Three zones with different rules.**

  | Zone       | Content                                                    | Rule |
  |------------|------------------------------------------------------------|------|
  | `raw/`     | Files exactly as downloaded (ZIP, CSV, JSON, XLSX)         | Immutable: written once, never edited. The only zone worth backing up. |
  | `curated/` | Typed, normalized tables (parquet)                         | Rebuilt from `raw/` plus overrides; never edited by hand. |
  | `derived/` | Computed products (adjustment factors, caches)             | Disposable: deleting it loses nothing but time. |

  Each zone can be reproduced from the zones above it and the overrides.
  That is what makes the base auditable: any number in `curated/` traces
  back to a file in `raw/` or to a hand-made decision in an override file.
- **H0.6 — `DataStore` is the only owner of paths.** No other module builds
  a path into the data root by hand. `DataStore` validates the root and the
  layout and returns paths inside a zone; it never reads or writes data
  itself. Moving the data root, or later moving it to the cloud, changes
  one class.
- **H0.7 — Paths stay inside their zone.** Path parts given to `DataStore`
  must be relative and must not contain `..`, so that no caller can reach
  outside the data root by accident.

### Scope of this phase

- **H0.8 — Local filesystem first.** `DataStore` returns `pathlib.Path`.
  Cloud storage (S3 through `fsspec`) is deferred until the data outgrows
  the local disk. When it comes, the return type changes from `Path` to a
  storage-agnostic path, and this record is superseded. Designing for S3 now
  would add an abstraction with a single implementation.
- **H0.9 — Dependencies arrive with the phase that uses them.** H0 needs
  only the standard library (`pathlib`, `tomllib`, `argparse`). `pyarrow`
  and `duckdb` arrive in H1, `pyyaml` in H2, each as part of an optional
  `b3` extra, so the core package keeps depending only on pandas and numpy.
- **H0.10 — Overrides are located in H2.** The override files live in the
  private `ibexquant-data-config` repository, not in the data root, because
  they are hand-written decisions that deserve version history. How
  ibexQuant finds them (a second configured path) is decided in H2, the
  first phase that reads them.
- **H0.11 — Track-wide decisions recorded here.** Two decisions apply to
  every later phase:
  - corporate actions are stored in a canonical ibexQuant format; StatusInvest
    and yfinance are converted once into a frozen bootstrap file, after
    which only official sources (B3, CVM) feed the base;
  - coverage starts on 1994-07-01, the start of the Real. Earlier records
    are dropped on ingestion, without currency conversion; research is
    expected to start in 2000 or later.

### Interface

```python
from ibexQuant.data.store import DataStore, Zone, resolve_data_root

root = resolve_data_root()  # Path, absolute; not checked on disk
store = DataStore.open()  # resolve + validate root and layout
store = DataStore.initialize(root)  # create root and zones, idempotent

store.root  # Path
store.zone(Zone.RAW)  # Path to raw/
store.raw("b3", "cotahist", "COTAHIST_A2024.ZIP")  # Path inside raw/
store.curated("bars_daily", "year=2024")  # Path inside curated/
store.derived("adjustment_factors.parquet")  # Path inside derived/
```

```text
ibexquant data init [--root PATH]
```

- `resolve_data_root(explicit=None, *, environ=None, config_path=None)`:
  `environ` and `config_path` default to `os.environ` and
  `~/.ibexquant/config.toml`; they are parameters so tests never touch the
  real environment or home directory.
- Returned paths are **not created**: a writer creates the directories it
  needs (`path.parent.mkdir(parents=True, exist_ok=True)`). `DataStore`
  never creates anything outside `initialize`.
- `ibexquant data init` prints each zone as created or already present and
  exits with status 0; on any `DataStoreError` it prints the message and
  exits with status 2.

| Condition                                                     | Error |
|---------------------------------------------------------------|-------|
| No explicit argument, environment variable or config entry     | `DataRootNotConfiguredError` (`LookupError`) |
| Relative root; `config.toml` not valid TOML; `[data].root` not a string | `DataRootConfigError` (`ValueError`) |
| Root does not exist                                            | `DataRootNotFoundError` (`FileNotFoundError`) |
| Root exists but is not a directory                             | `NotADirectoryError` |
| Root exists but a zone directory is missing                    | `DataLayoutError`, suggesting `ibexquant data init` |
| Path part empty, absolute, or containing `..`                  | `ValueError` |
| Path part not a string                                         | `TypeError` |

All errors raised by the module derive from `DataStoreError`, so a caller
can catch the whole family in one clause.

## Alternatives considered

- **A default data root** (e.g. `~/ibex-data`). Rejected: a missing
  configuration would silently write a large base to a place the user never
  chose, and a second machine would quietly use an empty base.
- **Only an environment variable.** Simple, but environment variables are
  easy to lose between terminals, IDEs and scheduled jobs. The config file
  gives a persistent per-machine setting; the variable still overrides it.
- **Paths as module-level constants.** The approach of the previous project;
  every script repeats them and breaks when a folder moves.
- **`fsspec` from the start.** Rejected for now (H0.8): it would abstract
  over one implementation and add a dependency before any data exists.
- **Overrides inside the data root.** Rejected (H0.10): the data root is
  not versioned, and hand-made decisions are exactly what needs history.
- **A single flat folder.** Rejected: mixing immutable downloads with
  rebuildable tables makes it impossible to know what is safe to delete and
  what must be backed up.

## Consequences

- A new machine needs one setting (`IBEXQUANT_DATA_ROOT` or the config file)
  and `ibexquant data init` before any ingestion.
- Tests run anywhere, including CI, without data: every test builds its own
  data root under a temporary directory.
- `pyproject.toml` gains a `[project.scripts]` entry, `ibexquant`, the
  entry point for every data command of the following phases.
- Later phases add sub-commands (`ingest`, `build`, `audit`, `review`)
  to the same CLI and their own tables to the zones, without changing the
  layout.

## Amendments

- **2026-11-09** — H0.9 named `duckdb` as an H1 dependency. H1 needed only
  filtered parquet reads, so 0011 (H1.15) defers `duckdb` until a query needs
  more; `pyarrow` alone forms the `b3` extra. H0.5 is refined, not changed,
  by 0011 (H1.2): a closed year's raw file is immutable, while the current
  year's is replaced on each refresh, with the manifest recording which
  download built the curated data.
