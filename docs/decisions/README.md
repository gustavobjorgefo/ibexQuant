# Architecture Decision Records

This directory records the architectural decisions of ibexQuant: what was
decided, why, and which alternatives were rejected. Each record answers the
question a future reader is most likely to ask — not only *what* the code
does, but *why it is not done differently*.

## Conventions

- One file per **topic**, named `NNNN-short-title.md`, numbered sequentially.
- Every record follows the same structure: Context, Decision, Alternatives
  considered, Consequences.
- Status values:
  - **Proposed** — under discussion, not yet binding.
  - **Accepted** — binding; code must follow it.
  - **Superseded by NNNN** — replaced; kept for history, never deleted.
- A decision is never edited to say something different. To change it,
  write a new record that supersedes the old one.
- **What deserves a record.** Only decisions whose rationale is not visible
  in the code: if, months later, someone would ask "why not the simpler
  alternative?" and the code does not answer, write a record. Conventions
  that explain themselves belong in docstrings.
- **At most one record per phase**, grouping that phase's decisions as
  sections, unless a single decision is large enough to stand alone.
- Corrections that do not change a decision (e.g. a missing item in a list)
  are added as a dated **Amendments** section instead of a new record.
- Individual rules inside a record keep the identifiers used during the
  design discussion (e.g. `A5`, `D0.2a`) so they can be referenced from
  code reviews and docstrings.

## Index

| #    | Title                                                     | Status   | Phase |
|------|-----------------------------------------------------------|----------|-------|
| 0001 | [Feature engineering architecture](0001-feature-engineering-architecture.md) | Accepted | —     |
| 0002 | [Panel: the input container for features](0002-panel.md)  | Accepted | 0     |
| 0003 | [Missing data policy](0003-missing-data-policy.md)        | Accepted | 0     |
| 0004 | [Universe mask](0004-universe-mask.md)                    | Accepted | 0     |
| 0005 | [Adjusted prices and scale invariance](0005-adjusted-prices.md) | Accepted | 0 |
| 0006 | [Legacy code](0006-legacy-code.md)                        | Accepted | 0     |
| 0007 | [Minimum Python version](0007-minimum-python-version.md)  | Accepted | 0     |
| 0008 | [Kernel conventions](0008-kernel-conventions.md)          | Accepted | 1     |
| 0009 | [Performance analytics](0009-performance-analytics.md)    | Accepted | —     |

## Pending items

Open questions deliberately deferred. Each one names where it will be
resolved; when it is, the resolving record is linked here.

| #  | Pending item                                                        | Resolve in            |
|----|---------------------------------------------------------------------|-----------------------|
| P1 | Timestamp convention for the live engine, compatible with `Panel` (see 0002) | Live engine design |
| P2 | `SessionsSinceLastObservation` node, exposing absence to the model (see 0003) | Phase 2 |
| P3 | Universe module: Ibovespa composition by effective date and point-in-time liquidity filter (see 0004) | After Phase 3 |
| P4 | Unadjusted prices in `Panel` for features that are not scale invariant (see 0005) | When needed |
| P5 | Backtest runner loads data from `lookback` sessions before the backtest start, using `FeatureSet.lookback` and the trading calendar (see 0008) | Backtest engine design |
| P6 | Backtest engine produces the trades table (`entry_time`, `exit_time`, result per trade) and positions, enabling exposure and turnover metrics from holdings (see 0009) | Backtest engine design |
| P7 | `analytics/inference.py`: normality, significance of the mean, Probabilistic Sharpe Ratio; scipy as an optional `stats` extra (see 0009) | After plotting |
| P8 | Deflated Sharpe Ratio and multiple-testing corrections, which need the number of strategies tried (see 0009) | Phase 5 |

## Roadmap

### Feature engineering

| Phase | Scope                                                | Status      |
|-------|------------------------------------------------------|-------------|
| 0     | Foundations and contracts                            | Done        |
| 1     | Kernels (batch + incremental)                        | Done        |
| 2     | `Feature` abstraction and concrete nodes             | Not started |
| 3     | `FeatureSet`: DAG, batch compute, cache, `to_matrix` | Not started |
| 4     | Labels with `t1`                                     | Not started |
| 5     | Leakage-aware validation (purging, embargo)          | Not started |
| 6     | ML integration (fitted preprocessing, sample weights)| Not started |
| 7     | Streaming execution and parity tests                 | Not started |

### Research and reporting

Parallel to feature engineering: evaluating what a strategy did, from numbers
to charts and reports (see 0009). Neither track depends on the other.

| Part      | Scope                                                        | Status      |
|-----------|--------------------------------------------------------------|-------------|
| Analytics | Returns, metrics, distribution, trades, relative, rolling, summary | In progress |
| Style     | Visual identity: tokens, fonts, figure finishing and export  | Not started |
| Plotting  | Analysis charts and tearsheet                                | Not started |
| Reports   | Composed documents (PDF/HTML) in the house identity          | Not started |