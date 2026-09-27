# 0003 — Missing data policy

- Status: Accepted
- Date: 2026-09-27
- Phase: 0

## Context

When long-format bars become wide (see 0002), three different situations
all appear as `NaN`:

1. **The asset does not exist** — before its listing or after delisting.
   There was no row in long format; the wide index created one.
2. **Real gap** — the asset exists and the market was open, but there is no
   bar: suspension, auction without trades, provider failure. After
   `align_to_calendar`, a `NaN` inside a symbol's range is always this case,
   never a non-trading day.
3. **Outside the universe** — the data exists and is valid, but the asset
   was not part of the investable universe at that date. This does not show
   up as `NaN` at all; it is handled by the universe mask (see 0004).

The absence of data is itself information: at that moment there was no
price and no trading. A policy must preserve it rather than hide it.

Example used throughout this record — an asset suspended for two sessions:

```
session   close
d1        30.00
d2        30.30
d3        NaN     ← suspended
d4        NaN     ← suspended
d5        31.50
```

## Decision

- **D0.2a — Observation time (policy C).** Time-series computations run over
  each asset's own valid observations. Gaps are skipped, never filled:

  ```
  d2   ln(30.30 / 30.00) = 0.0100
  d3   NaN     (asset not observable)
  d4   NaN     (asset not observable)
  d5   ln(31.50 / 30.30) = 0.0388   (one observation spanning three sessions)
  ```

  A rolling window of *n* means the last *n* observations of that asset.
  `NaN` in a feature output means "not observable at this timestamp", and
  rows carrying it are dropped by `to_matrix`, so no signal is ever produced
  for an asset that could not be traded.
- Situations 1 and 2 are handled identically: an asset before its listing
  simply has no observations, and its warm-up starts naturally at its first
  bar.
- **D0.2b — No reset on long gaps.** The computation does not restart after
  a long absence. Illiquid assets are handled by a liquidity criterion in
  the universe mask (see 0004), which keeps them out of the model's
  observations.
- **D0.2d — Fast path.** Kernels check each column for gaps. Columns without
  `NaN` use the direct vectorized computation; only columns with gaps go
  through the per-column observation-time path (drop gaps, compute, reindex).
- Cross-sectional operations naturally exclude assets whose value is `NaN`
  at a given timestamp.

## Alternatives considered

- **Policy A — propagate `NaN`.** Nothing is done. The d5 return becomes
  `NaN` because it depends on d4, so the +3.9% move on resumption is lost.
  A `rolling(20)` computation with `min_periods=20` then stays `NaN` for 20
  more sessions: a two-session suspension erases a month of features.
  Rejected.
- **Policy B — forward-fill prices.** d3 and d4 receive 30.30, producing two
  zero returns. The resumption move survives, but the zeros are fictitious
  (the market was closed for this asset, the price did not stay still), they
  bias volatility downwards, and on d3–d4 the model sees features as if the
  asset were trading. It also erases the information that a suspension
  happened, and contradicts the ingestion rule of never filling `NaN`.
  Rejected.
- **Restart history after *N* sessions without data.** Considered to avoid
  windows spanning months for illiquid assets. Rejected as unnecessary: such
  assets are removed from the universe by the liquidity criterion, so their
  distorted features are never consumed.

## Consequences

- The policy maps directly onto incremental execution (Phase 7): a symbol
  with no new bar simply does not update its state. Policies A and B would
  require resetting states or synthesizing fake bars, respectively.
- A return following a gap spans several sessions but is counted as a
  single observation. The model cannot know this unless absence is exposed
  explicitly — pending item **P2**: a `SessionsSinceLastObservation` node
  (1 on a normal session, 3 on d5 in the example above).
- Kernels need two code paths (vectorized and per-column). The per-column
  path runs only for columns that actually contain gaps.