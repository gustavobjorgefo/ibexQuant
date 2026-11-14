# 0014 — Anchored price adjustment

- Status: Accepted
- Date: 2026-11-14
- Phase: H4

## Context

H3 records every ex-session of every security with the events placed on it
(ADR 0013). A raw series still jumps at each of them. Research needs
adjusted prices, and the usual back-adjustment has the two flaws ADR 0005
describes: price levels carry events that happen after the observation, and
the whole adjusted history changes with every new event.

H4 adjusts **relative to an anchor date** chosen by the caller, normally the
moment a decision is taken (the end of a training window). At the anchor the
adjusted price equals the raw price; going back, only the events between the
observation and the anchor enter the factor. Nothing after the anchor is
used, and the same anchor always gives the same numbers.

The cash a holder receives (net of tax, on the payment date) is a matter for
a portfolio simulation, not for the price series. Simple backtests that do
not simulate a portfolio read adjusted prices only; a backtest engine may
instead execute on raw prices and credit the cash. Both must be possible,
and H4 delivers the first.

### What the data shows

The exploration used the first full ledger build (2026-11-10: 29,920
placed events, 26,879 ex-sessions) and the B3 snapshot of the same date.

| Finding | Evidence | Decision |
|---|---|---|
| B3 does not restate history: its closing price before each ex-date equals the COTAHIST previous close (median ratio 1.000, 78% within 2%), whatever splits came later; `PETR`, `VALE`, `ITUB`, `BBDC`, `MGLU`, `WEGE` events before a later 1:4, 1:8 or 13.2 total ratio show the same ratio 1.000. Amounts are per share as held on the data-com | 27,157 B3 cash events matched to their ex-session | H4.2 |
| 3,873 ex-sessions carry more than one cash event (1,705 dividend + JCP on the same day; 58 with three kinds) | Same build | H4.2 |
| Cash per share is below 6.3% of the previous close in 90% of the sessions and below 23.6% in 99%; 83 sessions exceed 50% and 49 reach or exceed the price itself. Most of these are pre-2010 penny stocks with a reverse split on the same session (`BSLI`, `CEPE`, `UGPA`, `CALI`: 1,000:1), where the amount is per new share; the others are unexplained (`ILMD3` 2010, `BPAR3` 2018) | Same build | H4.3 |
| On the 89 sessions that combine cash and a share event, reading the cash as per old share fits the opening price better in 38, per new share in 21, and both alike in 30 | Same build | H4.2, H4.3 |
| Subscriptions: 438 events, all with a price; in 196 the price is at or above the previous close, so the right is worth nothing | Same build | H4.2 |
| Spin-offs (130), conversions (39) and redemptions (18) are rare; conversions and exchanges hand the holder a different security, whose series continues only if H2 chained the two | Same build | H4.2 |
| External check, yearly total returns 2021–2025 of `PETR4`, `VALE3`, `ITUB4`, `BBDC4`, `WEGE3`, `ITSA4`, `MGLU3` against Yahoo Finance and ProfitPro: Yahoo matches to the basis point where both have the same events, and misses events B3 has (`PETR4` 2022: one of two dividends on one ex-date, −812 bp; `MGLU3` 2024). ProfitPro uses JCP net of withholding: with net JCP, 29 of 35 years agree within 15 bp. The check found a defect of the base (a chain's cash history placed twice through its old code, `ITAU` → `ITUB`: +580 to +2,435 bp a year), fixed in H3.10, and the `XPart` spin-off of 2021 (B3 adjustment −17.54% for `ITUB4`), whose target never traded | Comparison of 2026-11-14 | H4.2, H4.3 |

## Decision

### The factor of an ex-session

- **H4.1 — One factor per ex-session.** All events placed on one ex-session
  of a security combine into one factor. Only ledger events with status
  `confirmed` or `inferred` are used; `review` events (no ex-session) are
  not. Ex-sessions without events (`unexplained`, `unexplained_jump`) have
  no factor: the raw move stays in the adjusted series (H4.9).

- **H4.2 — The factor.** Let `P` be the security's raw close on the last
  session before the ex-session (`previous_close` in `ex_sessions`), and all
  amounts per share as held on the data-com. The value handed to the holder
  per share is

  ```
  V = sum of cash amounts                      (dividend, JCP, income, return of capital)
    + sum of r · P_target                      (spin-offs: r target shares per share)
    + sum of r · max(P − S, 0) / (1 + r)       (subscriptions: r new shares at price S)
  ```

  and the share ratio `R` is the product of the share ratios of the splits,
  reverse splits and bonus issues on the session (new shares per old share;
  1 if none). Then

  ```
  f_total  = (1 − V / P) / R
  f_splits = 1 / R
  ```

  - JCP enters **gross**: the market discounts the gross amount on the
    ex-session, which is also what B3 applies. The 15% withheld from
    individuals is the cash side's concern (H4.11).
  - `P_target` is the target's raw close on the ex-session; if the target
    does not trade that day, its first close within the next 5 sessions;
    otherwise the spin-off is not applied (H4.3). A spin-off whose target
    never trades (`XPart`, merged into XP on the day it was created) carries
    instead a `value_fraction` in `corporate_actions.yaml`, B3's published
    adjustment for the class, and its value is `value_fraction · P`.
  - The subscription term is the theoretical value of the right, from the
    theoretical ex-rights price `(P + r·S) / (1 + r)`.
  - Conversions and exchanges into a security of the **same** H2 chain are
    share events with `R = r`: the series continues, so the ratio must be
    removed like a split. Into another security, and redemptions, they are
    terminal: the old series ends, the new one starts with its own history
    and enters research once it has enough bars. They get no factor here;
    their effect on returns at delisting belongs to the universe and
    backtest phases (H5, H6).
  - `OTHER` events get no factor and are reported.
  - Ex-sessions without events are not corrected with their observed
    price ratio, even when it is small: a correction would hide the gap the
    H3 review exists to close. They stay raw and are counted (H4.8).

- **H4.3 — A factor must be possible.** An ex-session whose value `V`
  reaches or exceeds `P`, whose `P` is missing (first session of the
  series), or whose spin-off target has no price is **not applied** and is
  listed with the reason. Its cash is not reinterpreted (per new share, per
  1,000 shares): those cases are old and few, and a guess would hide them.
  The fix is an entry in `corporate_actions.yaml` (ADR 0013, H3.11) with
  the right amount, and the rebuild applies it.

### The anchored adjustment

- **H4.4 — Formula.** For a bar of session `s` and an anchor `A`:

  ```
  P_adj(s; A) = P_raw(s) · prod{ f_e : s < ex_e ≤ A }
  ```

  The same product multiplies `open`, `high`, `low` and `close`. The bar of
  the ex-session itself is already ex and is not multiplied by its own
  factor. For `s ≥ A` the product is empty and the bar is raw.

- **H4.5 — Modes.** `raw` (no factor), `splits` (`f_splits`: share events
  only, so price levels stay comparable across splits but drop at every
  dividend) and `total_return` (`f_total`: a holder who reinvests every
  value received, gross, in the same share at the theoretical ex price
  `P − V`). `total_return` is the default for research.

- **H4.6 — Volume.** `volume` (shares) is multiplied by the share ratios
  between `s` and `A`, `prod{ R_e : s < ex_e ≤ A }`, in every mode except
  `raw`, so that turnover stays comparable across splits. It is never
  adjusted for value events. Traded value in currency is never adjusted
  (and is not served by the provider).

- **H4.7 — Anchor rules.** The anchor defaults to `end`. An anchor before
  `end` is an error: bars after it would carry unadjusted jumps. An anchor
  after `end` is allowed: it is classic back-adjustment as of that date,
  and the caller takes responsibility for the events after `end` it
  brings in (ADR 0005).

### Storage and API

- **H4.8 — Factor table.** `derived/b3/adjustment_factors.parquet`, one
  row per applied ex-session: `security_id`, `isin`, `ex_date`,
  `previous_close`, `cash_value`, `share_ratio`, `f_total`, `f_splits`,
  `event_ids`. Not applied ex-sessions go to
  `derived/b3/adjustment_rejected.parquet` (`security_id`, `isin`,
  `ex_date`, `reason`, `event_ids`), and a summary
  `derived/b3/adjustment_factors.json` holds counts per reason, the events
  that never take a factor by type, the ledger's unexplained ex-sessions by
  status (so the extent of what stays raw is always visible), and the
  SHA-256 of the ledger's `corporate_actions.parquet` and
  `ex_sessions.parquet` they were built from. The table is derived and
  disposable; `ibexquant data build adjustment-factors` rebuilds it in
  seconds.

- **H4.9 — Provider.** `B3Provider.get_bars(symbols, start, end, fields,
  adjusted=True, *, anchor=None, adjustment="total_return")`.
  - `adjusted=False` serves raw bars, as today; passing `anchor` or
    `adjustment` with it is an error.
  - `adjusted=True` applies `adjustment` (`"splits"` or `"total_return"`)
    at `anchor` (default `end`). Factors are joined by `security_id`, so a
    ticker followed across renames (H2.14) is adjusted along its whole
    chain.
  - If the factor table is missing, or was built from another ledger than
    the one on disk, the provider raises a specific error that names the
    command to run. A stale table is never used silently.
  - When the served range contains an `unexplained_jump` ex-session of a
    requested security, the provider logs a warning naming it.

- **H4.10 — Pure function.** `adjust(bars, factors, anchor, mode)` works
  on a long frame (`session_date`, `security_id`, OHLCV) and does the
  arithmetic of H4.4–H4.6; the provider only reads, maps tickers and
  reshapes. It is vectorised: a reverse cumulative product of the factors
  per security, cut at the anchor, looked up by `searchsorted`.

### Cash and the backtest

- **H4.11 — Adjusted prices are self-sufficient.** A `total_return` series
  already includes every value handed to holders, reinvested gross; a
  backtest that only reads adjusted prices needs nothing else. The
  alternative — execute on raw prices and credit the cash, JCP net of the
  15% withheld, on the payment date — is the backtest engine's option,
  chosen per backtest, and is recorded as pending item P9. The two must
  never be combined: raw execution with cash credited, or adjusted prices
  alone.

## Alternatives considered

- **Store adjusted prices.** Rejected (ADR 0005): every new event rewrites
  history, and the anchor would be fixed at build time.
- **Back-adjust to the last session (Yahoo, Economatica).** That is anchor
  = today. It remains available (an anchor after `end`), but it is not the
  default: it puts PETR4 of 2005 in cents and leaks future events.
- **Net JCP in the factor.** Rejected: the price discounts the gross
  amount, so a net factor leaves a residual move on every JCP ex-session
  and makes the series differ from the market's. The net amount belongs to
  the cash side (H4.11).
- **Additive (difference) adjustment for cash.** Keeps price differences
  exact but can drive old prices negative and breaks return invariance.
  Multiplicative factors keep returns exact and prices positive.
- **Reinterpret impossible cash amounts automatically** (per new share when
  a reverse split shares the session). Rejected for now (H4.3): it fixes
  most of the 49 cases but guesses on the rest, all of them old penny
  stocks; a reviewed override is cheap and explicit.
- **Factors computed on the fly from the ledger.** Equivalent, but every
  call would redo the joins and the spin-off lookups; the derived table
  keeps the provider fast and makes the factors inspectable.

## Consequences

- Returns are identical for every anchor: for any `A ≥ s+1`, the return
  from `s` to `s+1` does not depend on `A`. This is the property the tests
  guarantee, together with: adjusted = raw at and after the anchor; a
  synthetic 1:10 split disappears in `splits` mode; an event after the
  anchor changes nothing; the adjusted return across a dividend ex-session
  equals `P_ex / (P − D)`, with `P_ex` the raw close of the ex-session.
- Features computed on adjusted prices stay scale invariant (ADR 0005),
  and price levels near the anchor are the real ones, which serves the
  scale-dependent features of P4.
- Unexplained ex-sessions remain visible as raw moves in adjusted series;
  the warning in H4.9 points to them, and the H3 review is how they are
  closed.
- The external check (evidence above) is repeatable with any source that
  publishes adjusted and raw prices; agreement to a few basis points a year
  where conventions match is the acceptance criterion. After the fixes, the base agrees
  with ProfitPro (net JCP) within 15 bp in 31 of 35 years, `ITUB4` 2021
  included (−17.14% in both); the rest are under 90 bp.
- New pending item **P9**: backtest engine option to execute on raw prices
  and credit event cash (JCP net of withholding) on the payment date
  (H4.11). Resolve in: backtest engine design.
