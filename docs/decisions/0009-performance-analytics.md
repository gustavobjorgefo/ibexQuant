# 0009 — Performance analytics

- Status: Accepted
- Date: 2026-10-09
- Phase: — (Research and reporting track)

## Context

Besides backtesting and live trading, ibexQuant needs to evaluate what a
strategy did: performance and risk metrics, the shape of its returns, the
quality of its trades, and eventually charts and reports with a consistent
identity. The previous project had this logic spread across
`analysis/performance.py`, `performance/performance.py` and
`visualization/*_plots.py`, mixing computation, colors and file export in
the same functions (see 0006).

Performance analytics is not a phase of the feature engineering roadmap: it
does not depend on features, labels or validation, and nothing there depends
on it. It is the first part of a parallel track, **Research and reporting**
(analytics, style, plotting, reports), and this record numbers its decisions
`AN1`, `AN2`, ... so they do not collide with phase identifiers.

This record covers the decisions whose rationale is not visible in the code.
Self-explanatory conventions live in the docstrings: `ddof=1`, the layout of
the monthly returns table, period labels at the calendar period end,
`min_count=1` when compounding by period, the `B3_SESSIONS_PER_YEAR`
constant, and the use of standard pandas operations throughout.

## Decision

### AN1 — Three packages, one direction of dependency

```
plotting ──► analytics
plotting ──► style
analytics, style: depend on nothing else in ibexQuant
```

- **`analytics`** turns returns and trades into numbers. It depends only on
  pandas and NumPy and never draws.
- **`style`** holds the visual identity (colors, fonts, sizes, figure
  finishing). It knows nothing about finance; its tokens are pure Python, so
  a future HTML or LaTeX report can use them without matplotlib.
- **`plotting`** draws analysis charts by combining the other two. It never
  computes a metric itself.

matplotlib is an optional extra (`plot`), required only by `style`'s
matplotlib layer and by `plotting`. Numbers are consumed by code that never
draws — walk-forward selection, validation, live monitoring — and none of it
must install a plotting library.

### AN2 — The input is periodic simple returns; analytics never produce them

Return analytics consume periodic simple returns (`0.01` is 1%), as a
`pd.Series` or a `pd.DataFrame` with one column per series (e.g. strategy
and benchmark), indexed by a strictly increasing `DatetimeIndex`.

- **Shape is preserved.** Transformations return the shape they receive;
  scalar metrics return a `float` for a Series and one value per column for
  a DataFrame. Each column is measured over its own observed period.
- **Simple, not log, returns**, because they aggregate across assets, as
  for portfolio aggregation in kernels (0008).
- **Producing returns is the backtest's job.** Analytics start where a
  return series exists, so they work today with research notebooks and
  later with any engine whose result exposes its returns.

### AN3 — Missing values only at the edges of each column

Leading and trailing `NaN` are allowed per column, so series of different
lengths share one DataFrame (a benchmark that starts a year later does not
shorten the strategy). **Interior `NaN` raises**, naming the column and the
first date.

A strategy is never unobservable mid-history: out of the market it earns
cash. A hole in its returns means a bug upstream. Filling it with zero would
hide the bug; dropping rows silently cuts other columns, which is what the
previous project's `returns.dropna()` did to a DataFrame of strategy and
benchmark.

### AN4 — Empty input, undefined values and zero dispersion

- **Empty return input raises `ValueError`.** Asking for the Sharpe ratio of
  nothing is a slicing bug upstream; returning `NaN` would let it propagate
  silently to the final table.
- **A metric that is undefined for a valid input is `NaN`, never `inf`.**
  The volatility of one observation, a Sharpe ratio over zero volatility, a
  Sortino ratio without downside, a Calmar ratio without drawdown, a profit
  factor without losses. In a DataFrame, one undefined column does not
  break the others; in walk-forward, a fold where the strategy stayed in
  cash is a result, not an exception.
- **Zero dispersion is detected exactly** (`max == min`), as in kernels
  (commit `099aa3d`). The floating-point mean of a repeated value is not
  always that value, which leaves ~1e-17 of spurious deviation and would turn
  a ratio over zero risk into ~1e16 instead of `NaN`.
- **No minimum length is imposed.** Annualizing a few sessions is
  arithmetically correct and statistically meaningless; any threshold would
  be an opinion buried in a low-level function. Flagging short samples is
  the job of the summary or report.
- **A period without trades is a valid outcome.** Unlike returns, which
  exist on every session, trades may legitimately not exist in a window.
  Counts and sums are zero; metrics that need at least one trade are `NaN`.

### AN5 — Explicit annualization; CAGR by sessions

- `periods_per_year` is a **required keyword**, with no default. The
  frequency is never assumed; `B3_SESSIONS_PER_YEAR = 252` exists so the
  number is not scattered, and callers pass it explicitly.
- **CAGR counts time in observed periods**, `(1 + total)^(N / n) - 1`, not in
  calendar days. It is consistent with every other annualized metric, which
  scale by periods, and matches the reference implementation used in tests.
- The annualized arithmetic mean (`mean · N`) is not provided: two numbers
  called "annual return" with different values confuse the reader. CAGR is
  the annual return.

### AN6 — The type of the risk-free rate decides how it is read

- A **`float` is an annual rate**, converted to the period geometrically,
  `(1 + rf)^(1 / N) - 1`, so that compounding it over a year gives back the
  annual rate. The previous project divided linearly (`rf / N`).
- A **`pd.Series` holds periodic returns** already (e.g. the daily CDI),
  subtracted as given; it must cover every timestamp where returns are
  observed.
- The default is **0%**. The risk-free rate is generic: the CDI is the
  natural Brazilian choice, but any rate or series can be passed.

Confusing an annual rate with a periodic one is a classic error; letting the
type decide removes the ambiguity without a mode parameter.

### AN7 — Sharpe and Sortino formulas

- **Sharpe** is `mean(r - rf) / std(r - rf) · √N`: both the arithmetic mean
  and the sample standard deviation are taken over excess returns (Lo,
  2002). With a constant rate this equals the deviation of raw returns; with
  a varying series such as the CDI it does not.
- **Sortino** divides the mean excess return by the downside deviation
  `√mean(min(r - rf, 0)²)`, averaged over **all** periods, periods above the
  target contributing zero (Sortino and Price, 1994). The target is the same
  `risk_free` as Sharpe's, so the two ratios are directly comparable.

### AN8 — Losses are always negative numbers

Drawdowns, maximum drawdown, value at risk, conditional value at risk, worst
period, average loss, gross loss and worst trade are all reported as
negative numbers; a 95% VaR of `-0.021` is a 2.1% loss. Ratios that divide
by a loss (payoff ratio, profit factor, Calmar) use its absolute value and
are positive.

A reader of an ibexQuant table never has to ask which sign convention a row
uses.

### AN9 — The drawdown's first peak is the initial capital

`drawdown = V_t / max(V_0, ..., V_t) - 1`, with `V_0 = 1` the capital
before the first return. A loss in the first period is already a drawdown.
The direct implementation, `equity / equity.cummax() - 1`, starts the peak
at the first equity value and reports zero there.

### AN10 — Distribution statistics

- **Skewness and kurtosis are the bias-adjusted sample estimates** computed
  by pandas, the same as Excel's `SKEW` and `KURT`. The direct (population)
  formulas underestimate both in small samples. Kurtosis is in excess of the
  normal's, so a normal distribution gives 0. Below 3 (skewness) or 4
  (kurtosis) observations, or for constant returns, they are `NaN`.
- **Quantiles interpolate linearly** between neighbouring observations, the
  pandas and NumPy default and the convention of the reference libraries.
  CVaR is the mean of the returns at or below the VaR.
- **A return of exactly zero is neither positive nor negative.** In daily
  returns zero usually means no position, and counting it either way biases
  the ratio of gains and losses.
- Distribution statistics are **per period, never annualized**, and live in
  their own module, separate from the annualized metrics.

### AN11 — Trades: the round trip is the unit

- **One trade is one round trip**: opened, closed, one result. A day trade
  and a five-day swing trade are measured alike; the difference between
  styles shows up as a metric, the average holding period, not as a branch
  in the code.
- **Outcome metrics take a Series of per-trade results**, in chronological
  order of exit. The results may be returns or PnL; every metric reports in
  the unit it receives. Summing PnL is exact; summing returns is meaningful
  mainly with constant position size, which is documented on the metrics
  that sum.
- **Timing metrics take a trades table** with `entry_time` and `exit_time`,
  plus the sessions of the period. Trades are matched to sessions by date,
  so intraday times work.
- **Holding periods count sessions inclusively**: a day trade lasts one
  session, and that session counts as exposed. Overlapping trades count each
  session once.
- **A trade with a result of exactly zero is neither a win nor a loss**: it
  is left out of win and loss rates and averages, and it breaks a winning or
  losing streak.
- Trade metrics take one strategy at a time: different strategies rarely
  have the same number of trades, so trades do not align as DataFrame
  columns.

### AN12 — Reference libraries are test oracles, never dependencies

empyrical and scipy are `dev` dependencies, used only in tests, where the
ibexQuant result must match theirs under the same convention. No module in
`src/` imports them. This is the stance of 0001 applied to analytics: the
concepts are adopted, the dependency is not. Their conventions sometimes
differ from ours, and some of them style their own charts, which conflicts
with a house identity.

## Alternatives considered

- **Metrics and charts in one package.** Simpler to start, but either the
  core requires matplotlib or every plotting import must be local. Rejected
  for AN1.
- **Style inside the performance plotting module.** The identity serves
  feature diagnostics and reports too; those would have to import from a
  performance module. Rejected for AN1.
- **Reusing feature kernels for rolling metrics.** Kernels are causal, run in
  observation time (0003) and carry incremental states; strategy returns have
  no gaps and metrics are ex post. Importing those semantics would couple two
  unrelated contracts for a one-line computation.
- **Accepting the equity curve as the canonical input.** Returns are
  independent of the initial capital and align trivially with a benchmark and
  a risk-free series; the equity curve is derived from them.
- **Filling interior gaps with zero, or dropping rows.** Hides bugs and, for
  DataFrames, shortens every column to the shortest one. Rejected for AN3.
- **Always raising, or always returning `NaN`, for undefined metrics.** The
  first breaks a whole table over one constant column; the second turns
  slicing bugs into silent `NaN`. Rejected for the split in AN4.
- **A default of 252 for `periods_per_year`.** Silently wrong for any other
  frequency. Rejected for AN5.
- **A mode parameter for the risk-free rate** (`rf_is_annual=True`). One
  more parameter to get wrong, where the type already carries the meaning.
- **Sortino over negative periods only** (the standard deviation of losses).
  Ignores how often losses happen and is centered on their own mean, not on
  the target. Rejected for AN7.
- **VaR as a positive loss**, common in bank risk reports. Rejected for
  AN8's single sign convention.
- **Population skewness and kurtosis** (scipy's default). Biased in small
  samples and different from what a reader checking in Excel will find.
- **Quantile at the nearest observed return.** Always an actual period, but
  differs from the reference libraries, losing the parity tests.
- **Trade metrics over a table with an `on="return" | "pnl"` parameter.**
  Forces building a table with times to compute, say, a profit factor from
  a list of results. Rejected for AN11's split.
- **Excluding flat trades from streaks without breaking them.** Three
  losses, a flat trade and two losses would count as a streak of five. The
  flat trade is neither outcome, so it ends the run.
- **empyrical, quantstats or pyfolio as dependencies.** Rejected for AN12.

## Consequences

- Any consumer of numbers — notebooks, walk-forward, validation, live
  monitoring — uses `ibexQuant.analytics` with only pandas and NumPy
  installed.
- The backtest engine owes analytics two things: a return series and,
  for trade metrics, a trades table with `entry_time`, `exit_time` and a
  result per trade. Positions, which would enable exposure and turnover
  metrics from holdings, are pending item **P6**.
- Every analytics function validates its input against one contract; a
  malformed series fails at the first call, with the column and date of the
  problem.
- Sharpe, Sortino, volatility, CAGR, Calmar, maximum drawdown, VaR, CVaR,
  skewness and kurtosis are tested for parity with empyrical or scipy;
  changing a convention breaks a test, not a report.
- Hypothesis tests (normality, significance of the mean, Probabilistic
  Sharpe Ratio) need distribution functions from scipy and return a
  statistic with a p-value; they belong in a separate inference module with
  its own optional extra — pending item **P7**.
- Corrections for multiple testing (Deflated Sharpe Ratio) need the number
  of strategies tried during research, which analytics do not know — pending
  item **P8**, in Phase 5.
- The modules still to come in this track (`relative`, `rolling`,
  `summary`) apply these conventions; a decision that is genuinely new there
  gets its own record.