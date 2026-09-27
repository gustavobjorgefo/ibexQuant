# 0004 — Universe mask

- Status: Accepted
- Date: 2026-09-27
- Phase: 0

## Context

Using data from assets that were not investable at a given date introduces
survivorship bias: selecting today's index constituents and backtesting
them over the past uses the knowledge that they would later belong to the
index.

In the previous project the universe was always the Ibovespa, and the whole
dataset was rebuilt at every four-month rebalancing. This avoided the bias
but repeated work at each rebalancing and restarted every feature's warm-up
for assets entering the index.

## Decision

- **D0.2c — A boolean mask carried by `Panel`.** `panel.universe_mask` is a
  wide boolean DataFrame with the same index and columns as the fields.
  `True` means the asset belongs to the universe at that timestamp.
- The mask is applied **only** in two places:
  - **cross-sectional nodes** — a rank or z-score at time *t* considers only
    assets in the universe at *t*;
  - **`to_matrix`** — rows outside the universe do not become observations.
- The mask is **never** applied to time-series computation. Features are
  computed over each asset's full history, so an asset entering the universe
  arrives with its features already warmed up.
- Initially the mask is entirely `True` (a static universe, with its
  survivorship bias acknowledged). The interface exists from the start so
  that a point-in-time universe can be introduced without changing any
  contract.
- The future universe (pending item **P3**) combines:
  - index composition by **effective date** of each portfolio, not by the
    publication date of its preview;
  - a **point-in-time liquidity criterion**, computed over a trailing window
    (e.g. average daily traded value over the previous *n* sessions).
- **Pre-filtering symbols before building `Panel` is allowed only as an
  optimization equivalent to an all-`False` mask column**, i.e. removing
  assets that fail the point-in-time criterion at every date. Because
  time-series features are computed per asset and cross-sectional features
  already exclude out-of-universe assets, removing such a column changes no
  other value.
- **Filtering by a statistic computed over the whole sample** (e.g. average
  liquidity from 2015 to 2024) **is forbidden**: in 2016 it uses liquidity
  that only materialized later, which is look-ahead bias.

## Alternatives considered

- **Rebuild the dataset at each rebalancing (previous approach).** Correct,
  but repetitive, and every newly included asset starts with cold features.
- **Drop out-of-universe rows from the data.** Removes the history needed
  to warm up features before an asset enters the universe.
- **Apply the mask inside time-series features.** Same problem: every entry
  into the universe would behave like a new listing.

## Consequences

- Research can load the full history once and still respect point-in-time
  universes.
- Cross-sectional nodes must accept and honor the mask.
- The universe module itself (composition parsing, liquidity rules) is
  deferred (P3); until then results carry survivorship bias and should be
  read accordingly.