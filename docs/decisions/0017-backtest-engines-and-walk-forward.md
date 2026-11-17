# 0017 — Backtest engines, walk-forward schedule and the vectorized engine

- Status: Proposed
- Date: 2026-11-17
- Phase: B1

## Context

H0 to H6 built a survivorship-free B3 base and one call to read it, `ds.panel(...)` (ADR
0016). Notebook 0002 already runs a backtest by hand: one panel per IBOV period, an
equal-weight portfolio held until the next rebalance. This record turns that loop into
code, in two pieces:

1. a **walk-forward** in layers: the training and test windows, the universe of each
   window (a fixed list, or an index such as the IBrX 100, `IBXX`), the securities with
   enough history, and the data of each window;
2. the **simplest engine**, vectorized, for fast and shallow studies in notebooks.

It also fixes the map of engines the project will grow into, so the simple engine does not
take decisions that belong to the others.

### How a study runs

The universe is an index's member list, not its weights: it controls the sample and keeps
survivorship out. B3 revises the portfolios every four months (January–April, May–August,
September–December). A strategy is fitted on some months and traded on the next ones, out
of sample; the window then moves on. With 12 months of training and 1 of test: for a test
in January, the members of the January–April portfolio are fitted on January–December of
the year before and traded in January; February, March and April use the same members;
May takes the new portfolio. The durations are the study's choice (18 and 3, 36 and 6),
and the index is optional: a study may trade a fixed list of tickers.

### What the base gives

| Finding | Evidence | Decision |
|---|---|---|
| A period starts on the first Monday of January, May or September (second Monday from 2027), not on the 1st | ADR 0015, H5.3 | B1.5 |
| The members of a new portfolio are set by the 3rd preview, published before the month turns; the definitive portfolio changes weights, not members | Manual of B3 indices, 2.3; ADR 0015, H5.1 | B1.5 |
| `IBXX` compositions come from B3's workbooks, from 2017-09; the retroactive series and the liquidity rule cover `IBOV` only | ADR 0015, H5.1, H5.6 | B1.5 |
| After a member stops trading its bars are `NaN` to the end of the panel; the exit is listed in `index_exits` | ADR 0015, H5.8; ADR 0016, H6.6 | B1.12 |
| Returns of `total_return` prices do not depend on the anchor; levels (targets, stops) do | ADR 0005, 0014 | B1.15, B1.20 |
| `ds.ticker_history` gives each security's first and last session | ADR 0016, H6.2 | B1.6 |

## Decision

### Engines

- **B1.1 — Three levels, two holding models.** Engines differ in how much of the
  simulation runs bar by bar:

  | Level | What runs before the loop | What runs bar by bar | Use | Reference |
  |---|---|---|---|---|
  | **E1 — vectorized** | features, signals, target weights or orders | nothing: array operations, a loop only over rebalances | fast, shallow studies in a notebook | Hilpisch |
  | **E2 — precomputed signals, event loop** | features and signals | portfolio construction, orders, fills, cash, events | research with realistic accounting (P6, P9, P11, P12) | Halls-Moore, HangukQuant |
  | **E3 — full event-driven** | nothing | market data event → features → signal → portfolio → order → fill | the backtest and the live system share the code; live swaps the data and execution adapters | Halls-Moore, *Inside the Black Box* |

  Independent of the level, a strategy holds positions in one of two ways:

  - **Position**: positions carry across bars, with P&L and allocation tracked over time.
    The bar index is any ordered time key: sessions now, intraday timestamps or event bars
    (volume, dollar) later. No engine of this model assumes one bar per day.
  - **Day-trade on daily bars**: every position opens and closes inside one session. It
    never carries a position across a close, so it never meets an event, an exit or an
    overnight valuation; it produces daily P&L only. It is a separate engine, never a flag
    of the position engine. (Day trading on intraday bars is the position model on
    intraday bars.)

  This record designs E1 for both models and the walk-forward. E2 and E3 get their
  own records.

- **B1.2 — Package.** `ibexQuant.backtest`:

  | Module | Content |
  |---|---|
  | `walkforward/` | one module per layer (B1.3): `schedule.py`, `universe.py`, `eligibility.py`, `folds.py` |
  | `vectorized.py` | E1 position engine: `simulate_weights`, `BacktestResult` |
  | `daytrade.py` | E1 day-trade engine: `DayTradeOrders`, `simulate_day_trades` |
  | `weights.py` | target-weight helpers, starting with `equal_weight` |

  Strategies, their parameters and their fitting live outside the package (in notebooks
  now, in the private repository when they are real). The engines never fit anything.

### Walk-forward

- **B1.3 — Four layers, each usable alone.** The walk-forward is not one object that does
  everything. Each layer answers one question and knows nothing of the layers above it:

  | Layer | Question | Objects | Needs |
  |---|---|---|---|
  | 1. Time | when is it fitted, when is it traded? | `WalkForwardSchedule` → `Window` | a calendar |
  | 2. Universe | which securities may it trade in a window? | `UniverseRule`: `StaticUniverse`, `IndexComposition` | the dataset |
  | 3. Eligibility | which of them have enough data? | `EligibilityRule`: `HasHistory`, `StillTrading`, `MinimumPresence` | the dataset |
  | 4. Assembly and data | the folds, and the bars of each | `WalkForward` → `Fold` → `FoldData` | the three above |

  A study can stop at any layer: windows alone for a hand-made loop, a fixed list of
  tickers without any index, or the full stack. A new rule (a liquidity filter, a sector
  universe, a rule read from the private repository) is a new class of one layer, never a
  flag of another.

  ```python
  from ibexQuant.backtest.walkforward import (
      HasHistory,
      IndexComposition,
      StaticUniverse,
      StillTrading,
      WalkForward,
      WalkForwardSchedule,
  )

  # Layer 1 alone: the windows.
  schedule = WalkForwardSchedule(start="2019-01", end="2026-10", train_months=18, test_months=3)
  windows = schedule.windows(ds.calendar)

  # All layers: windows, index universe, eligibility, data.
  walk = WalkForward(
      schedule,
      universe=IndexComposition("IBXX"),  # or StaticUniverse(["PETR4", "VALE3"])
      eligibility=(HasHistory(cover="train"), StillTrading()),
      warmup=252,
  )
  folds = walk.folds(ds)
  data = folds[0].load(ds)  # FoldData(train, test)
  WalkForward.summary(folds)  # one row per fold, for inspection
  ```

- **B1.4 — Layer 1: time.** `WalkForwardSchedule(start, end, train_months, test_months,
  step_months=None, expanding=False)` holds no data. A test window is the sessions of
  `test_months` calendar months; its training window is the `train_months` calendar months
  before it (rolling, or from the first training start with `expanding=True`); windows
  start every `step_months` (default `test_months`, so test windows tile the range without
  overlap). `start` and `end` are the first and last test months. Durations are free: 12
  and 1, 18 and 3, 36 and 6. `windows(calendar)` maps them to sessions of any
  `TradingCalendar`; the schedule never counts bars, so the same schedule serves intraday or
  event bars later.

  Decisions are taken at the close and executed in the next session. The decisions of a
  test window are the session before each test session: from the last session before the
  window (the *decision session*, where the strategy is fitted and the first orders are
  formed) to the second to last test session. The last test session's decision belongs to
  the next window.

  | `Window` field | Meaning |
  |---|---|
  | `train_start`, `train_end` | training range; `train_end` is the decision session |
  | `decision_start`, `decision_end` | sessions whose close makes the window's orders |
  | `test_start`, `test_end` | sessions where those orders are executed and held |

- **B1.5 — Layer 2: universe rules.** A `UniverseRule` (abstract) answers two questions
  about a window: `select(ds, window)`, the `Selection` of candidates, and `cuts(window)`,
  the months where the window must be cut because the universe changes (none by default).

  | `Selection` field | Meaning |
  |---|---|
  | `security_ids` | candidate `security_id` values |
  | `unresolved` | tickers of members with no `security_id` (dropped as `unresolved`, B1.7) |
  | `universe` | a `Universe` whose own mask the test panel keeps, or none |
  | `index`, `period`, `source` | the index period used, when there is one |

  - `StaticUniverse(universe)` takes any universe argument of `ds.panel` (a ticker list,
    an index code, `Index`, `LiquidityRule`, combinations). The candidates are its members
    on any session of the test window (`ds.members`, B1.8), and the test panel keeps the
    universe's own mask, so a point-in-time `Index("IBOV")` follows B3's effective dates.
    It never cuts a window. With a ticker list no index is involved.
  - `IndexComposition(index, at_rebalance="split")` takes the members of the index period
    of the quadrimester that contains the test window (January–April, May–August,
    September–December), by label, from `ds.composition` (B1.8). It applies from the first
    session of the quadrimester, not from B3's effective date (the first Monday): the
    members of the 3rd preview are known before the month turns.

    | `at_rebalance` | Behaviour |
    |---|---|
    | `"split"` | a test window that crosses the start of a quadrimester is cut there; each piece is a fold with its own decision session, training window and composition |
    | `"hold"` | the composition at the window's start is kept to its end |

    With `"split"`, a security that leaves the index is sold on the first session of the
    new period, and one that joins is fitted before it is traded: the rule of the study.
    The next windows keep the schedule's grid. With 12 and 1 nothing is cut; with 18 and 3
    from January, the window April–June becomes April and May–June.

    A period without a composition of the index raises `WalkForwardError` naming it (for
    `IBXX`, any period before 2017-09): the study picks a start, the rule does not invent
    members.

- **B1.6 — Layer 3: eligibility rules.** An `EligibilityRule` (abstract) has one method,
  `apply(ds, window, security_ids, warmup)`, which returns for each security the reason it
  is dropped, or nothing when it is kept. It reads data up to the decision session only:
  what was known when the fold was decided. Rules are applied in order, each to the
  securities the previous ones kept; the default is the strict pair
  `(HasHistory(cover="train"), StillTrading())`, and an empty sequence keeps every
  candidate.

  | Rule | Keeps a security when | Reason when dropped |
  |---|---|---|
  | `HasHistory(cover="train")` | its first session is on or before the first warmup bar before the training window: `warmup` bars plus the whole training window | `short_history` |
  | `HasHistory(cover="test")` | it has `warmup` bars before the decision session (looser; for studies that fit nothing) | `short_history` |
  | `StillTrading()` | its last session is on or after the decision session | `stopped_trading` |
  | `MinimumPresence(fraction, cover="train")` | it traded on at least `fraction` of the sessions covered (the training window, or the `warmup` bars before the decision session) | `low_presence` |

  History is the date of the first session; how often a security traded is
  `MinimumPresence`'s question, not `HasHistory`'s. First and last sessions come from
  `ds.ticker_history` (earliest first and latest last over every ticker a security
  carried); presence counts the sessions with a close in the raw bars.

  **What this layer is not.** It decides once per fold whether a security has the data to
  be fitted and traded. A selection that changes bar by bar from the strategy's own
  results (the top 20 by rolling Sharpe ratio, say) is part of the strategy, not of the
  walk-forward: in E1 it is a mask the strategy computes with its weights; in E2 and E3 it
  is a stateful component updated at each bar, designed with them.

- **B1.7 — Layer 4: folds.** `WalkForward(schedule, universe, eligibility, warmup)`
  assembles one `Fold` per window, or per piece of a window the universe rule cuts. The
  pieces are built by `schedule.window` (B1.4), with the schedule's own rule.

  | `Fold` field | Meaning |
  |---|---|
  | `window` | the `Window` (B1.4), cut if split |
  | `securities` | eligible `security_id` values |
  | `universe` | the test panel's `Universe`: `Securities(securities)`, intersected with the selection's universe when it has one |
  | `excluded` | a frame of the candidates dropped: `security_id`, `ticker` (on the decision session; the portfolio's ticker when unresolved) and `reason` (`unresolved`, then the first rule that dropped it) |
  | `index`, `period`, `source` | from the selection; empty for a static universe |
  | `warmup` | bars of history before each range |

  A fold without an eligible security raises `WalkForwardError` naming the window and
  counting the drops by reason: an empty fold would turn into cash without a word.
  `WalkForward.summary(folds)` gives one row per fold (dates, period, source, candidates,
  securities, drops by reason) for inspection.

  The training universe is the fold's securities: the composition known at the decision,
  fitted on its members' past.

- **B1.8 — Data per fold, and three additions to the data API.**
  `fold.load(ds, fields=None, adjustment="total_return")` returns `FoldData(train, test)`:

  - `train`: `Securities(fold.securities)` over `[train_start, train_end]` with `warmup`
    bars, anchored at `train_end`;
  - `test`: `fold.universe` over `[decision_start, decision_end]` with `warmup` bars,
    anchored at `decision_end`.

  The test panel is anchored at its last decision, so the adjusted levels inside it already
  carry events of later sessions of the window. Features that are scale invariant (returns,
  ratios, ranks; ADR 0005) are not affected; a strategy that uses price levels inside the
  test window needs an anchor per decision, which is E2's job (P12).

  The data API gains `Securities(security_ids)`, a `Universe` of fixed `security_id`
  values, true on every session (combinable like the others);
  `B3Dataset.composition(index, period)`, the members of one period by its label, with the
  columns of `constituents`; and `B3Dataset.members(universe, start, end)`, the membership
  mask of a universe argument over the sessions of a range, as `ds.panel` computes it.

### E1 position engine

- **B1.9 — Inputs.**

  ```python
  from ibexQuant.backtest import simulate_weights

  result = simulate_weights(
      bars,  # Panel with open and close, total_return
      weights,  # decision session × security_id
      execution="next_open",  # or "next_close"
      cost_bps=0.0,
  )
  ```

  `weights` holds target fractions of equity, indexed by decision session. A row is a
  rebalance; a session without a row holds the units and lets them drift with prices.
  Values are signed (a negative weight is a short), `NaN` is zero, the cash is what is left
  (`1 - sum`), and the gross exposure is not capped. A daily-rebalanced strategy gives a
  row every session. The engine applies the one-session lag itself (B1.4): a strategy never
  shifts its signals, so it cannot forget to.

  `bars` covers every security the weights name, from the first decision to the last
  session to value. With a walk-forward, it is one panel of `Securities` over the union of
  the folds' securities. Returns do not depend on the anchor, so one panel serves the whole
  run.

- **B1.10 — Accounting.** Initial equity 1, fractional units, cash at 0%. On an execution
  session the holdings are valued at the execution price (`open` or `close`), the new units
  are `weight × equity / price`, and the cost is taken from cash. Between executions the
  units are constant and the values are computed for the whole segment at once: the only
  loop is over rebalances.

- **B1.11 — Execution conventions.** `next_open`: a decision at the close of `d` trades at
  the open of the next session, so the overnight return belongs to the old holdings and the
  intraday return to the new ones. `next_close`: trades at the close of the next session.

- **B1.12 — Securities that do not trade.**

  | Case | Rule |
  |---|---|
  | No bar on the execution session, bars later (suspension) | cannot trade: units kept, target ignored for that rebalance; valued at the last close |
  | No bar after its last session in `bars` (exit) | the holding becomes cash at the last close on its last session; later targets are zero (P10 for E1) |
  | Target on a security with no bar yet | `ValueError` |

  The member still trades on its last session: nothing tells the strategy it is the last.
  An announced tender offer excluded by hand is the strategy's choice, not the engine's.

- **B1.13 — Costs.** `cost_bps` per side on the traded notional (`|Δunits| × price`),
  default zero. Fees, slippage and impact models are E2's.

- **B1.14 — Result.** `BacktestResult` with `returns` (simple, by session, from the first
  execution), `weights` (held at each close, after drift), `turnover` and `costs` by session.
  `returns` feeds `ibexQuant.analytics` as is. There is no trades table: a trade needs an
  order and a fill (E2, P6).

- **B1.15 — Adjusted prices, on purpose.** E1 simulates on `total_return` prices: dividends
  are reinvested in the security, there is no tick or lot, and returns are exact for a
  fractional holder. Raw execution with event cash (P9), tick and lot (P11) and levels
  across ex-sessions (P12) are E2's.

- **B1.16 — `equal_weight(mask)`.** `1 / n` over the true cells of each row, zero
  elsewhere; rows with no true cell are all cash.

### E1 day-trade engine

- **B1.17 — Orders.** `DayTradeOrders` holds frames indexed by decision session (the close
  of `d − 1`, for session `d`), columns `security_id`:

  | Field | Meaning |
  |---|---|
  | `buy_limit` | limit price to buy; `NaN` = no order |
  | `sell_limit` | limit price to sell short; `NaN` = no order |
  | `size` | notional of each order as a fraction of equity at the close of `d − 1` |
  | `buy_stop`, `sell_stop` | optional stop levels of each side; `NaN` = no stop |

  `entry="open"` replaces the limits with a `side` frame (+1, −1, 0): the position opens at
  the open of `d`.

- **B1.18 — Fills.** A buy limit fills when the low of `d` is strictly below it, at the
  limit price; a sell limit when the high is strictly above it, at the limit price. Touching
  the limit is not a fill, and a limit crossed by the open still fills at the limit, not at
  the open: both are conservative. Both sides can fill on the same session: they are two
  independent trades.

- **B1.19 — Exits.** Every position closes at the close of `d`. A stop closes it before: a
  long stop is hit when the low reaches it (`low ≤ stop`), a short stop when the high does
  (`high ≥ stop`), at the stop price. With a limit entry the fill is taken to come first.
  There are no profit targets: with daily bars nobody knows whether the target or the stop
  came first, and taking the stop is the only order that is never optimistic.

- **B1.20 — Levels on the adjusted panel.** Limits and stops computed at the close of
  `d − 1` and compared with the bars of `d` in the same `total_return` panel are on the same
  scale even when `d` is an ex-session: the adjustment removes the jump. This is P12 for E1.
  On raw bars the level would need the event's factor.

- **B1.21 — Result.** Daily returns (P&L over equity at the close of `d − 1`, costs per
  side included) and a trades table with `security_id`, `side`, `entry_time` and
  `exit_time` (both `d`), `entry_price`, `exit_price`, `exit_reason` (`close`, `stop`) and
  `return`, which `ibexQuant.analytics.trades` reads.

### Acceptance

- **B1.22 — Notebook 0003.** With zero cost and the rebalance rule of notebook 0002 (one
  row of equal weights per IBOV period, executed at the close of its first session), E1
  reproduces the point-in-time line of notebook 0002 (daily returns within 1e-10). Then the
  same notebook runs an `IBXX` walk-forward with equal weight and monthly windows, and turns
  on costs, measuring the effect of each change.

## Alternatives considered

- **One engine with a day-trade flag.** Rejected: positions carried across bars and a P&L
  that never becomes a position have different accounting; a flag would put conditionals in
  every step of the position engine.
- **One schedule object with the index built in.** The first draft of this record. It
  forced every study through an index and fixed durations that divide the quadrimester;
  replaced by layers (B1.3).
- **`hold` as the only behaviour at a rebalance.** Simple, but a member that leaves the
  index would stay traded until the window ends, against the study's rule. Kept as an
  option; `split` is the default.
- **An `effective` option in `IndexComposition`** (the quadrimester's first session or
  B3's effective date). Dropped before implementation: a cut at B3's date falls on a
  Monday inside the month, which the time layer does not build; and a study that follows
  B3's dates already has `StaticUniverse(Index(...))`, whose mask changes on them.
- **Bar-by-bar selection in the walk-forward** (top-N by rolling Sharpe ratio, as in the
  first prototype's `UniverseEligibility`). Rejected here: it depends on the strategy's
  results and changes inside a fold. It belongs to the strategy (E1) or to a stateful
  component of the event loop (E2, E3).
- **Point-in-time membership over the training window.** Rejected: the decision is taken
  with the composition known at the decision, and the strategy is fitted on those members'
  past. Members join an index after gaining liquidity, so the training sample leans to
  securities that did well; the bias is known and the test window never sees it.
- **Warmup only before the test window as the default.** A member traded without its full
  training history is fitted on less data than the others; stricter is simpler. Kept as
  `HasHistory(cover="test")` for studies that fit nothing.
- **Weights as a constant mix (daily rebalance by default).** Rejected: a holder sees
  drift; rebalancing every day is one row per session.
- **The strategy shifts its own signals** (`position.shift(1)`). Rejected: the most common
  leakage of vectorized backtests. The engine owns the lag.
- **Forward-filled value after an exit** (notebook 0002). Same value, but the weight stays
  taken by a security that no longer trades. Replaced by cash at the last close (B1.12).
- **Raw prices and the tick in E1.** Rejected: E1 is for quick answers; exact accounting
  is E2's, and both read the same base.

## Consequences

- **P5** resolved: a fold loads `warmup` bars before each training and test range, and
  `HasHistory` requires them (B1.6, B1.8).
- **P10** resolved for E1: an exit is cash at the last close (B1.12). E2 keeps it, with raw
  prices.
- **P6, P9, P11, P12** stay with E2; P12 for levels inside a vectorized study is covered by
  the adjusted panel (B1.20). **P1** moves to E3. **P13** stays before E2's first study;
  an E1 study from 2016 crosses none of the jumps it lists.
- New pending item **P14**: cash yield. Uninvested cash earns 0% in E1; E2 credits the CDI
  (series to be added to the base).
- `IBXX` test windows start in 2017-09 at the earliest; an earlier start raises (B1.5).
  The training window may reach before it: it reads bars, not compositions.
- Implementation steps:

  | Step | Scope |
  |---|---|
  | B1.a | `Securities` and `B3Dataset.composition` (B1.8); `B3Dataset.members` comes with B1.c |
  | B1.b | Layer 1: `WalkForwardSchedule`, `Window` (B1.4) |
  | B1.c | Layers 2 to 4: universe and eligibility rules, `WalkForward`, `Fold`, `FoldData` (B1.5–B1.8) |
  | B1.d | `simulate_weights`, `BacktestResult`, `equal_weight` (B1.9–B1.16) |
  | B1.e | Notebook 0003: acceptance against 0002, `IBXX` walk-forward, costs (B1.22) |
  | B1.f | `simulate_day_trades` (B1.17–B1.21) |

- Closing criteria: tests on the synthetic base for each layer alone (free durations,
  rolling and expanding windows; static and index universes, both `at_rebalance` options,
  the composition gap, unresolved members; every eligibility rule; an empty fold), for the position engine (drift, both execution
  conventions, suspension, exit, costs) and for the day-trade engine (strict fills, both
  sides, stops, ex-session on the adjusted panel); notebook 0003 meets B1.22; 100% coverage.
