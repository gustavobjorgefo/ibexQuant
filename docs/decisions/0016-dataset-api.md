# 0016 — Dataset API

- Status: Proposed
- Date: 2026-11-16
- Phase: H6

## Context

H1 to H5 built the B3 base piece by piece: raw daily bars and `B3Provider` (ADR 0011), the
security master (0012), the corporate actions ledger (0013), anchored adjustment (0014) and
the point-in-time universe (0015). Each piece has its own reader. Research and, later, the
backtest engine need them together: the bars of every security in a universe, adjusted to
the decision date, aligned to the trading calendar, with the universe mask, in one `Panel`
(ADR 0002).

H6 adds that one call, `B3Dataset.panel(...)`, and the lookups around it (members, period
starts, events, exits, tickers). It also runs the whole base end to end in a notebook that
measures survivorship bias. The backtest engine and the walk-forward schedule are designed
separately, in their own record, after H6.

### What the base gives

| Finding | Evidence | Decision |
|---|---|---|
| The universe mask is keyed by `security_id`; tickers change (`VIIA3` → `BHIA3`, same `security_id` `BRVVARACNOR1`) and are reused by unrelated companies | ADR 0012, 0015; security master | H6.2 |
| `B3Provider.get_bars` labels each security with the ticker asked for; a universe asks for no ticker | ADR 0012, H2.14 | H6.2, H6.7 |
| Adjustment factors and the ledger are keyed by `security_id` | ADR 0013, 0014 | H6.7 |
| The sessions of `bars_daily` are B3's own trading days, 1994-07 to the last ingestion; the index periods of H5 are computed from them | ADR 0011, 0015 (H5.3) | H6.8 |
| `pandas_market_calendars` ships its own holiday rules and can disagree with the days B3 actually traded | `PandasMarketCalendarsCalendar` docstring | H6.8 |
| Features need history before the first decision (P5); its length is counted in bars | ADR 0008, P5 | H6.4 |

## Decision

### H6.1 Entry point

`B3Dataset` lives in `ibexQuant.data.dataset` and is exported as
`from ibexQuant.data import B3Dataset` (loaded on first use, like `ibexQuant.download`).
`B3Dataset.open(store=None)` opens the configured data root and checks that the tables it
needs exist (bars, master, ledger, factors, universe), naming the command that builds a
missing one.

```python
from ibexQuant.data import B3Dataset

ds = B3Dataset.open()

panel = ds.panel(
    universe="IBOV",  # a Universe, an index code or a list of tickers
    start="2019-01-01",
    end="2019-12-31",
    adjustment="total_return",  # "raw" | "splits" | "total_return"
    anchor=None,  # defaults to end
    warmup=252,  # bars before start, mask False there
    fields=None,  # OHLCV by default; "traded_value" on request
)
```

### H6.2 Columns are `security_id`; tickers are a lookup

The `Panel`'s symbols are `security_id` values. Features, models and the mask never depend
on a name; a rename inside the window does not split a series, and a reused ticker never
joins two companies.

Tickers are read from the master when a person or an order needs them:

- `ds.ticker_at(security_ids, as_of)` → the ticker of each security on `as_of` (its last
  ticker if it stopped trading before), as a `Series` indexed by `security_id`.
- `ds.ticker_history(security_ids)` → every ticker each security carried, with first and
  last session.
- `ds.label(frame, as_of=None)` → a copy of a `Panel` field or any frame with `security_id`
  columns, renamed to tickers on `as_of` (default: the frame's last session), for plots and
  reports.

Orders take the ticker valid on the day of execution (`ticker_at`), never a label from the
start of the window.

### H6.3 The `universe` argument

| Given | Becomes | Example |
|---|---|---|
| `Universe` | itself | `Index("IBXX") - Index("IBOV")` |
| `str` | `Index(code)` | `"IBOV"` |
| sequence of `str` | `Tickers([...])` | `["PETR4", "VALE3"]` |

Anything else is a `TypeError`.

### H6.4 Warmup in bars

`warmup` is a number of bars of the panel's frequency (sessions for daily bars), not a
calendar duration: the bars before `start` are loaded and the mask is `False` there, so
features see history and no decision is taken on it. The data side of P5 is resolved here;
the backtest runner still chooses `warmup` from `FeatureSet.lookback`.

### H6.5 One adjustment per `Panel`

`adjustment` and `anchor` apply to the whole panel; `anchor` defaults to `end`, as in the
provider. A strategy decides on adjusted prices and orders at raw prices: the backtest
engine asks for a second, `"raw"` panel for execution (P9, P11). The two never share a
`Panel`, so a field can never be read with the wrong meaning.

A price level a strategy derives from the decision panel (a target, a stop, a limit) is
already a raw price: at the anchor, adjusted equals raw, and every older bar is expressed in
the anchor's money. It needs only rounding to the tick (P11). A level that stays alive past
an ex-session (an open position's target or stop, or an order placed after the close of the
last cum session for the ex-session) is rescaled by that event's factor, `level × f`, known
in advance from the ledger; that is the engine's job (P12).

### H6.6 Which securities

Every `security_id` that is a member on at least one session of `[start, end]` gets a
column, with its bars from the first warmup bar to `end`. After a security stops trading
its fields are `NaN`, as the `Panel` contract already says; valuing a holding at its exit
stays pending item P10, and `ds.exits` lists those cases.

### H6.7 Bars by `security_id` through the provider

`B3Provider` gains an internal reader of bars by `security_id`, used by `get_bars` (after
resolving tickers) and by `B3Dataset`. Raw rows, renames and adjustment go through one code
path; `get_bars` keeps its contract.

### H6.8 Calendar from COTAHIST

A new `CotahistCalendar(store)` implements `TradingCalendar` with the sessions of
`bars_daily`: the days B3 traded, the same calendar the index periods use. The panel is
aligned to it, so a missing bar inside the range is a real gap, never a holiday. The
sessions are read from the partitions on first use (only the `session_date` column) and kept
in memory for the life of the object: no file to maintain, never stale after an ingestion.
If reading ever becomes a cost (the live engine), an `ingest cotahist` output can replace it
behind the same class. It covers sessions up to the last ingestion only; projecting future
sessions (for the live engine) is not its job. `PandasMarketCalendarsCalendar` stays for other markets and the yfinance
path.

### H6.9 Lookups

| Method | Returns | From |
|---|---|---|
| `ds.constituents(index, as_of)` | members on `as_of`: `security_id`, ticker, source, period | `index_constituents.parquet` |
| `ds.rebalance_dates(index, start, end)` | period starts (`effective_from`) in the range | `index_periods.parquet` |
| `ds.events(security_id)` | the security's ledger events | ledger (ADR 0013) |
| `ds.exits(index, start, end)` | members that stop trading inside a period | `index_exits.parquet` |

### H6.10 Notebook 0002

`research/backtests/0002_ibov_point_in_time.ipynb` compares, January 2003 to the last
session, three equal-weight portfolios rebalanced at each IBOV period start:

1. **Point-in-time**: the members of each period (`Index("IBOV")`).
2. **Survivors**: today's members applied to the whole past (the classic mistake).
3. **Benchmark**: IBOV through BOVA11 where available.

The gap between 1 and 2 is the size of survivorship bias on B3. The notebook builds one
`ds.panel(...)` per period in a plain loop, with the anchor at each decision date; that loop
is the sketch the walk-forward schedule will formalize.

## Alternatives considered

- **Ticker columns, labelled at the end of the window.** Readable, but a rename inside the
  window changes the label with the window, and a reused ticker is ambiguous. Rejected for
  the panel; kept as the `label` step for reports.
- **Warmup as a duration (`"252D"`).** Ambiguous between calendar and trading days, and
  meaningless for intraday bars. Rejected.
- **Raw and adjusted fields in one `Panel`** (`close`, `close_raw`). One object, but nothing
  stops a feature from reading the wrong one. Rejected (H6.5).
- **`pandas_market_calendars` for B3.** Its holidays are maintained by hand and can disagree
  with COTAHIST. Rejected for the B3 base.
- **A second bar reader inside the dataset.** Faster to write; two code paths for prices.
  Rejected (H6.7).

## Consequences

- One call gives research the same data the engine will use: survivorship-free, adjusted to
  the decision date, aligned to B3's real sessions.
- Reports and plots go through `ds.label` or `ds.ticker_at`; a column name in a panel is
  never shown to a person as is.
- P5 is resolved on the data side; the runner's choice of `warmup` stays with the backtest
  engine design.
- New pending item **P12**: price levels kept across an ex-session (targets, stops, pending
  orders) are rescaled by the event's factor on the ex-session. Resolve in: backtest engine
  design.
- A day-trade engine on daily bars needs none of P9, P10 or P12 for its positions (always
  flat at the close), but its features still read the adjusted decision panel, and limit
  orders placed for an ex-session are rescaled as in H6.5. It is a separate engine (roadmap).
- Closing criteria: `ds.panel` reproduces `B3Provider.get_bars` for a ticker list (same
  numbers, `security_id` columns); end-to-end tests on a synthetic base with a rename, a
  split, a dividend, an exit and two periods; notebook 0002 runs end to end with one
  `ds.panel` call per period and reports the survivorship gap.
