# 0008 — Kernel conventions

- Status: Accepted
- Date: 2026-09-28
- Phase: 1

## Context

Kernels are the numerical layer every feature is built on (ADR 0001). The
conventions they follow propagate to the whole system, and several of them
are not obvious from the code alone: they answer a "why not the simpler
alternative?" that a future reader is likely to ask.

This record covers only those. Self-explanatory conventions live in the
kernels' docstrings: DataFrame-in, DataFrame-out signatures; log returns for
features and simple returns for portfolio aggregation; `ddof=1` by default;
strict `min_periods = window` for fixed windows; annualization outside
kernels.

## Decision

### D1.2 — One observation-time helper, with a fast path by interior gaps

The missing data policy of ADR 0003 is implemented once, in
`kernels/_observation_time.py`, and every time-series kernel runs through it:

- Columns **without interior gaps** are computed together in one vectorized
  call. An interior gap is a `NaN` lying between two valid values. Leading
  `NaN` (before listing) and trailing `NaN` (after delisting) do not change
  the result of a rolling or recursive computation, so assets listed
  mid-sample stay on the fast path.
- Columns **with interior gaps** are computed on their valid observations
  only and reindexed onto the full index.
- The output is masked to `NaN` wherever the input is `NaN`, whatever the
  computation produced there. This is required, not defensive: pandas'
  `ewm().mean()` carries the last value forward through `NaN` rows, which
  would report a feature for an asset that is not observable.

Cross-sectional kernels do not use the helper; they operate row-wise and
pandas already skips `NaN` within a row.

### D1.6 — Exponential kernels declare a finite lookback by tolerance

A fixed window depends only on its last *n* bars. An exponentially weighted
computation depends on the entire history: after *L* bars, the starting
observation still weighs `(1 - alpha) ** L`. For a span-20 EMA, that is
13.5% after 20 bars.

Two consequences follow if nothing is done:

1. **The value depends on where the data starts.** An EMA started in 2015
   and one started in 2016 differ in early 2016, so a causal feature would
   not have a well-defined value at a given date.
2. **Streaming cannot match batch.** Warming up with `lookback` bars would
   leave the stream up to 13.5% away from the batch value.

Rule:

```
lookback = ceil( ln(tolerance) / ln(1 - alpha) )
```

- The kernel emits `NaN` until `lookback + 1` observations are available,
  exactly like a fixed window.
- `tolerance` is a parameter of every exponential kernel, defaulting to
  `DEFAULT_EXPONENTIAL_TOLERANCE = 1e-3`: the starting point influences at
  most 0.1% of the value.

| Tolerance | Starting-point weight | EMA span 20 | RiskMetrics decay 0.94 |
|-----------|-----------------------|-------------|------------------------|
| 1e-2      | 1%                    | 47 bars     | 75 bars                |
| **1e-3**  | **0.1%**              | **70 bars** | **112 bars**           |
| 1e-4      | 0.01%                 | 93 bars     | 149 bars               |

A smaller tolerance requires more bars; it is affordable at intraday
frequencies, where bars are plentiful.

Consequence for streaming parity (Phase 7): with a finite tolerance, parity
is verified by two tests — a stream fed the full history must match batch
exactly, and a stream warmed up with only `lookback` bars must match batch
within the tolerance.

### D1.7 — `alpha` has a single meaning

Every exponential kernel takes one smoothing factor, `alpha`, always in the
pandas sense — the weight of the **newest** observation:

```
ewma_t = alpha * x_t + (1 - alpha) * ewma_{t-1}
```

- Always computed with `adjust=False`, which is exactly this recursion and
  therefore identical to the incremental state.
- Other parameterizations are converted explicitly: `alpha_from_span`,
  `alpha_from_halflife`, `alpha_from_decay`.

The previous project passed `alpha=0.94` to its EWMA volatility. In
RiskMetrics, 0.94 is the **decay** (the weight of the previous value); the
corresponding `alpha` is 0.06. Passed to pandas as `alpha`, 0.94 yields a
volatility driven almost entirely by the last return, with no error raised.
One meaning for `alpha`, plus named conversions, removes the trap. Feature
nodes (Phase 2) may expose `span` or `halflife`, which read better.

### D1.8 — EWMA volatility is RiskMetrics

```
sigma2_t = decay * sigma2_{t-1} + (1 - decay) * r_t ** 2
```

The mean is assumed to be zero. For daily returns the mean is negligible
relative to the dispersion, so estimating it adds noise rather than
information; this is the market standard for short-horizon risk. pandas'
`ewm().std()`, which demeans, is a different estimator and is not used. The
recursion is trivial, giving exact parity with the incremental state.

### D1.11 — Incremental states: scalar, `NaN` in, `NaN` out

Incremental states are implemented in Phase 1, next to their batch kernels,
with parity tests at kernel level. Only the orchestration across the feature
graph (`FeatureStream`) waits for Phase 7.

- One state per asset, updated with scalars.
- While fewer than `lookback + 1` observations have been seen, `update`
  returns `NaN`.
- **A `NaN` input is not an observation**: the state does not change and
  returns `NaN`. This reproduces in streaming what the batch helper does with
  interior gaps, and makes warm-up `NaN` from an upstream node transparent to
  the next one.

## Alternatives considered

- **Per-kernel gap handling.** Every kernel repeating drop-compute-reindex
  would eventually diverge. Rejected in favor of D1.2.
- **Fast path only for columns without any `NaN`.** Correct but slower:
  every asset listed mid-sample would take the per-column path for no
  benefit.
- **Emitting biased early EWMA values** (previous project). Rejected: values
  would depend on the data start, and streaming parity would be impossible.
- **Lookback equal to the span.** Leaves 13.5% starting-point weight for a
  span-20 EMA.
- **Accepting `span`, `halflife`, `alpha` and `decay` in every kernel.**
  Four mutually exclusive parameters per kernel, with validation of their
  combinations, instead of one number and three small conversion functions.
- **Vectorized states over all assets.** Faster, but each asset would sit at
  a different stage of its own buffer because of gaps. Scalar states suffice
  at the current universe size; revisit in Phase 7 if needed.

## Consequences

- No kernel implements gap handling itself; new time-series kernels only
  provide the vectorized computation.
- Every feature, exponential or not, has the same contract: after its
  lookback, its value does not depend on how much history precedes it.
- Exponential features lose their first `lookback` bars (70 for a span-20
  EMA at the default tolerance).
- The backtest runner must load data starting `lookback` sessions before the
  backtest start — pending item **P5**.