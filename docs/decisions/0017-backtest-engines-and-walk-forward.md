# 0017 — Backtest engines, walk-forward schedule and the vectorized engine

- Status: Proposed
- Date: 2026-11-17
- Phase: B1

## Context

H0 to H6 built a survivorship-free B3 base and one call to read it, `ds.panel(...)` (ADR
0016). Notebook 0002 already runs a backtest by hand: one panel per IBOV period, an
equal-weight portfolio held until the next rebalance. This record turns that loop into
code, in two pieces:

1. a **walk-forward schedule** that cuts history into training and test windows and gives
   each window its universe, taken from an index (IBrX 100 first, `IBXX`);
2. the **simplest engine**, vectorized, for fast and shallow studies in notebooks.

It also fixes the map of engines the project will grow into, so the simple engine does not
take decisions that belong to the others.

### How a study runs

The universe is an index's member list, not its weights: it controls the sample and keeps
survivorship out. B3 revises the portfolios every four months (January–April, May–August,
September–December). A strategy is fitted on 12 months and traded on the next month, out
of sample; the window moves one month at a time. For a test in January, the members of
the January–April portfolio are fitted on January–December of the year before and traded
in January; February, March and April use the same members; May takes the new portfolio.

### What the base gives

| Finding | Evidence | Decision |
|---|---|---|
| A period starts on the first Monday of January, May or September (second Monday from 2027), not on the 1st | ADR 0015, H5.3 | B1.4 |
| The members of a new portfolio are set by the 3rd preview, published before the month turns; the definitive portfolio changes weights, not members | Manual of B3 indices, 2.3; ADR 0015, H5.1 | B1.4 |
| `IBXX` compositions come from B3's workbooks, from 2017-09; the retroactive series and the liquidity rule cover `IBOV` only | ADR 0015, H5.1, H5.6 | B1.4 |
| After a member stops trading its bars are `NaN` to the end of the panel; the exit is listed in `index_exits` | ADR 0015, H5.8; ADR 0016, H6.6 | B1.12 |
| Returns of `total_return` prices do not depend on the anchor; levels (targets, stops) do | ADR 0005, 0014 | B1.15, B1.20 |
| `ds.ticker_history` gives each security's first and last session | ADR 0016, H6.2 | B1.5 |

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

  This record designs E1 for both models and the walk-forward schedule. E2 and E3 get their
  own records.

- **B1.2 — Package.** `ibexQuant.backtest`:

  | Module | Content |
  |---|---|
  | `schedule.py` | `WalkForwardSchedule`, `Window`, `WindowData` |
  | `vectorized.py` | E1 position engine: `simulate_weights`, `BacktestResult` |
  | `daytrade.py` | E1 day-trade engine: `DayTradeOrders`, `simulate_day_trades` |
  | `weights.py` | target-weight helpers, starting with `equal_weight` |

  Strategies, their parameters and their fitting live outside the package (in notebooks
  now, in the private repository when they are real). The engines never fit anything.

### Walk-forward schedule

- **B1.3 — Windows on calendar months.** A test window is the sessions of `test_months`
  calendar months; its training window is the `train_months` calendar months before it
  (rolling; `expanding=True` keeps the first training start). The schedule steps by
  `test_months`. Defaults: 12 and 1.

  ```python
  from ibexQuant.backtest import WalkForwardSchedule

  schedule = WalkForwardSchedule(
      index="IBXX",
      start="2019-01",  # first test month
      end="2026-10",  # last test month
      train_months=12,
      test_months=1,
      warmup=252,  # bars of history before the training window
  )
  windows = schedule.windows(ds)  # list[Window]
  schedule.summary(ds)  # one row per window, for inspection
  ```

  The schedule is a specification: it holds no data, and `windows(ds)` reads the
  dataset. Windows are calendar ranges over B3 sessions; only `warmup` counts bars, so the
  same schedule serves intraday or event bars later.

- **B1.4 — Universe per window: the quadrimester's composition, from the 1st.** The
  universe of a test window is the composition of the index period of the quadrimester that
  contains it (January–April, May–August, September–December), labelled by its first month
  (`2025-05`). It applies from the first session of that month, not from B3's effective
  date: the members are those of the 3rd preview, known before the month turns. The
  composition is the best source of the period (H5.1); its source is shown in the window.

  `test_months` must divide 4 (1, 2 or 4), so that a test window never crosses two
  quadrimesters. A window whose period has no composition of the index raises
  `ScheduleError` naming the period (for `IBXX`, any period before 2017-09): the study picks
  a start, the schedule does not invent members.

- **B1.5 — Eligibility: full history or out.** A member is traded in a window only if its
  first session is on or before the first warmup bar before the training window (the
  `warmup` bars plus the whole training window), and its last session is on or after the
  decision session (B1.6). Members that fail are listed in the window with the reason
  (`short_history`, `stopped_trading`) and are neither fitted nor traded, even though they
  are in the index. The training universe is the test window's universe: the composition
  known at the decision, fitted on its members' past (B1.7).

- **B1.6 — Decisions at the close, execution in the test window.** A signal uses data up to
  the close of a session and is executed in the next session. The decisions of a test
  window are therefore the session before each test session: from the last session before
  the test window (the *decision session*, where the strategy is fitted and the first
  orders are formed) to the second to last test session. The last test session's decision
  belongs to the next window.

  | `Window` field | Meaning |
  |---|---|
  | `index`, `period`, `source` | index, quadrimester composition used, its source |
  | `universe` | eligible `security_id` values |
  | `excluded` | members left out, with the reason (B1.5) |
  | `train_start`, `train_end` | training range; `train_end` is the decision session |
  | `decision_start`, `decision_end` | sessions whose close makes the window's orders |
  | `test_start`, `test_end` | sessions where those orders are executed and held |
  | `warmup` | bars of history before each range |

- **B1.7 — Data per window.** `window.load(ds, fields=None, adjustment="total_return")`
  returns `WindowData(train, test)`, two panels of `Securities(window.universe)` (B1.8):

  - `train`: `[train_start, train_end]` with `warmup` bars, anchored at `train_end`;
  - `test`: `[decision_start, decision_end]` with `warmup` bars, anchored at
    `decision_end`.

  The mask is true on every session of each range: eligibility was settled by the schedule.
  The test panel is anchored at its last decision, so the adjusted levels inside it already
  carry events of later sessions of the window. Features that are scale invariant (returns,
  ratios, ranks; ADR 0005) are not affected; a strategy that uses price levels inside the
  test window needs an anchor per decision, which is E2's job (P12).

- **B1.8 — Two additions to the data API.** `Securities(security_ids)`, a `Universe` of
  fixed `security_id` values, true on every session (combinable like the others); and
  `B3Dataset.composition(index, period)`, the members of one period by its label, with the
  columns of `constituents`.

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
  row every session. The engine applies the one-session lag itself (B1.6): a strategy never
  shifts its signals, so it cannot forget to.

  `bars` covers every security the weights name, from the first decision to the last
  session to value. With the schedule, it is one panel of `Securities` over the union of
  the windows' universes. Returns do not depend on the anchor, so one panel serves the whole
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
- **Universe from B3's effective date (`Index` mask).** Rejected for the schedule: the
  members are known from the 3rd preview, and calendar months keep windows and universes
  aligned. `Index` stays for studies that follow the real index.
- **Point-in-time membership over the training window.** Rejected: the decision is taken
  with the composition known at the decision, and the strategy is fitted on those members'
  past. Members join an index after gaining liquidity, so the training sample leans to
  securities that did well; the bias is known and the test window never sees it.
- **Warmup only before the test window.** Rejected: a member traded without its full
  training history is fitted on less data than the others. Stricter is simpler.
- **Weights as a constant mix (daily rebalance by default).** Rejected: a holder sees
  drift; rebalancing every day is one row per session.
- **The strategy shifts its own signals** (`position.shift(1)`). Rejected: the most common
  leakage of vectorized backtests. The engine owns the lag.
- **Forward-filled value after an exit** (notebook 0002). Same value, but the weight stays
  taken by a security that no longer trades. Replaced by cash at the last close (B1.12).
- **Raw prices and the tick in E1.** Rejected: E1 is for quick answers; exact accounting
  is E2's, and both read the same base.

## Consequences

- **P5** resolved: the schedule loads `warmup` bars before each training and test range,
  and eligibility requires them (B1.5, B1.7).
- **P10** resolved for E1: an exit is cash at the last close (B1.12). E2 keeps it, with raw
  prices.
- **P6, P9, P11, P12** stay with E2; P12 for levels inside a vectorized study is covered by
  the adjusted panel (B1.20). **P1** moves to E3. **P13** stays before E2's first study;
  an E1 study from 2016 crosses none of the jumps it lists.
- New pending item **P14**: cash yield. Uninvested cash earns 0% in E1; E2 credits the CDI
  (series to be added to the base).
- `IBXX` test windows start in 2017-09 at the earliest; an earlier start raises (B1.4).
  The training window may reach before it: it reads bars, not compositions.
- Implementation steps:

  | Step | Scope |
  |---|---|
  | B1.a | `Securities` and `B3Dataset.composition` (B1.8) |
  | B1.b | `WalkForwardSchedule`, `Window`, `WindowData` (B1.3–B1.7) |
  | B1.c | `simulate_weights`, `BacktestResult`, `equal_weight` (B1.9–B1.16) |
  | B1.d | Notebook 0003: acceptance against 0002, `IBXX` walk-forward, costs (B1.22) |
  | B1.e | `simulate_day_trades` (B1.17–B1.21) |

- Closing criteria: tests on the synthetic base for windows (quadrimester boundaries, the
  composition gap, both exclusion reasons), for the position engine (drift, both execution
  conventions, suspension, exit, costs) and for the day-trade engine (strict fills, both
  sides, stops, ex-session on the adjusted panel); notebook 0003 meets B1.22; 100% coverage.
