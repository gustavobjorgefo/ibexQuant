# 0001 — Feature engineering architecture

- Status: Accepted
- Date: 2026-09-27
- Phase: — (cross-cutting)

## Context

The previous ibexQuant project computed features through a chain of
`BaseTransform` subclasses. Each transform received a single asset's bar
DataFrame, appended one or more columns and returned it. The design was
abandoned for the following reasons:

1. **Contract violated by its own implementations.** The base class forbade
   in-place modification, yet every subclass assigned `df[col] = ...` on the
   input and returned the same object.
2. **Full recomputation as the only execution mode.** Every call recomputed
   the whole history. Walk-forward validation with *k* refits cost
   O(k·N) per feature; bar-by-bar execution cost O(N²) in total.
3. **Warm-up not machine-readable.** The number of bars required before a
   valid value was documented in prose only, so no engine could query it.
4. **Implicit, string-based dependencies.** A volatility transform depended
   on a `"return"` column that some earlier transform had to create, in the
   right order. Nothing resolved the order or deduplicated shared inputs.
5. **Duplicated layers.** Transforms were thin wrappers over pure functions,
   each repeating the same validation boilerplate.
6. **Single-asset by construction.** Cross-sectional features (ranks,
   cross-asset z-scores, betas) could not be expressed at all.
7. **No place for fitted operations.** A stateless contract cannot express
   operations that learn parameters (scalers, PCA), which must be fitted on
   training data only.

## Decision

- **A1 — Declarative definitions.** A feature is an immutable `Feature`
  object describing its inputs, parameters and lookback. It computes nothing
  when constructed. Dependencies are references to other `Feature` objects,
  never column names.
- **A2 — Dependency graph.** A set of features forms a DAG that is resolved
  automatically: topological ordering, deduplication of identical nodes by
  their identity key, and propagation of lookback from inputs to outputs.
- **A3 — Two execution modes, one definition.** Every feature supports
  batch execution (vectorized over the full history) and incremental
  execution (constant cost per new bar). A parity test asserting that both
  modes produce the same values is mandatory for every node. Incremental
  execution is implemented from Phase 7, but the interface anticipates it
  from the start.
- **A4 — Wide format.** Computation operates on one DataFrame per field,
  indexed by timestamp with one column per symbol, so that time-series
  operations run over the whole universe at once and cross-sectional
  operations are row-wise.
- **A5 — Strict causality.** A feature value at time *t* depends only on
  data available up to *t*. Consequently features are computed once over the
  full history and cached; cross-validation only slices the resulting matrix.
- **A6 — Fitted preprocessing outside the graph.** Operations that learn
  parameters from data (scaling, PCA, feature selection, the model itself)
  are not features. They are fitted inside each validation fold, on the
  training portion only.
- **A7 — Labels are separate.** Labels live in their own module, which is
  the only place allowed to look ahead in time. Every label carries `t1`,
  the timestamp at which its outcome is fully determined.
- **A8 — Package layout and dependency direction.**

  ```
  src/ibexQuant/
  ├── data/
  │   ├── ingestion/          # existing: providers, validation, calendar
  │   ├── engine/             # existing: data handlers for the engine
  │   └── panel.py            # Panel (see 0002)
  ├── features/
  │   ├── kernels/            # pure math, batch + incremental pairs
  │   ├── core/               # Feature, graph, FeatureSet, cache, stream
  │   └── nodes/              # concrete Feature definitions
  ├── labels/                 # the only look-ahead module
  └── validation/             # purging, embargo, splitters
  ```

  Imports may only follow these directions:

  ```
  features ──► data
  labels   ──► data
  validation      (depends on neither; receives t1 and indices only)
  ```

  `features` never imports `labels`, which structurally prevents future
  information from reaching a feature.
- **A9 — Paired kernels.** The batch and incremental kernels of the same
  computation live in the same file, so that changing one keeps the other
  in view.
- **A10 — Decide before implementing.** Each phase follows the cycle:
  discuss, record the decision here, then implement with tests.

## Alternatives considered

- **Keep the transform chain and optimize it.** Rejected: the problems are
  structural (string dependencies, single-asset scope, no incremental mode),
  not performance details.
- **Ad-hoc precomputation in research scripts.** Fast to start, but the live
  engine would need a second implementation of every feature with no
  guarantee of equality — the classic source of backtest/live divergence.
- **Adopt an external framework (e.g. Zipline's Pipeline API).** Proven
  design, and a direct inspiration for A1–A4, but it couples the project to
  a framework's data model, calendar and maintenance cycle. The concepts are
  adopted; the dependency is not.

## Consequences

- Walk-forward and cross-validation become cheap: features are computed
  once, folds only index into a cached matrix.
- The engine can query the total warm-up required by a feature set.
- Cross-sectional features become first-class citizens.
- Every node requires two implementations (batch and incremental) plus a
  parity test. This is deliberate extra work traded for backtest/live
  consistency.
- Labels, validation and fitted preprocessing need their own modules and
  decisions (Phases 4–6).