# 0005 — Adjusted prices and scale invariance

- Status: Accepted
- Date: 2026-09-27
- Phase: 0

## Context

Research data comes from Yahoo Finance with corporate-action adjustment.
Back-adjustment multiplies **every price before an event** (dividend, split)
by a factor. For a price observed at time *s* and all events after *s*:

```
P_adjusted(s) = P_raw(s) · F,   F = product of the factors of every event after s
```

This has two consequences:

1. **Price levels contain future information.** The adjusted price shown
   for January 2024 depends on dividends paid after January 2024. A feature
   such as "close above 30" evaluated in 2024 uses information from later
   years.
2. **Adjusted history changes over time.** Every new event rewrites the whole
   adjusted history. Downloading the same period twice may yield different
   values.

However, every price inside a window ending at *t* is multiplied by the same
factor contributed by events after *t*. Any feature that is **invariant to
multiplying all its inputs by a constant** is therefore unaffected: the
factor cancels.

Invariant (safe): returns, log returns, `close / sma`, rolling z-scores,
volatility of returns, ranks, ratios between prices of the same asset.

Not invariant (unsafe on adjusted data): price levels, price thresholds,
absolute differences between prices (`ema_12 - ema_26` in currency units),
traded value in currency.

## Decision

- **D0.3a — Scale invariance rule.** Every feature computed on adjusted
  prices must be scale invariant. Each node documents whether it is, and
  this is a review criterion for new nodes.
- **D0.3b — Content fingerprint.** `Panel` exposes a fingerprint (a hash of
  its content). The feature cache (Phase 3) keys entries by
  `(feature.key, panel.fingerprint)`, never by the feature key alone, so
  that re-downloaded data with rewritten history never hits a stale cache.

## Alternatives considered

- **Use unadjusted prices only.** Returns would jump at every dividend and
  split, requiring separate event handling in every return-based feature.
- **Allow any feature and document the risk.** Rejected: the bias is silent
  and produces optimistic backtests without any visible error.

## Consequences

- Scale-dependent features need unadjusted prices, which `Panel` does not
  carry yet — pending item **P4**. Longer historical backtests will likely
  bring a different data source that addresses this.
- Differences such as MACD must be expressed in normalized form
  (e.g. divided by price or by volatility) to be valid on adjusted data.
- Computing the fingerprint has a cost proportional to the data size; the
  exact method is decided in Phase 3.