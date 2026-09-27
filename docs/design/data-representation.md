# Data representation across the system

This document explains why ibexQuant uses the wide per-field format of
`Panel` for feature computation, how that format relates to the other
representations in the system, and where it stops being adequate.

It complements [ADR 0002](../decisions/0002-panel.md), which records the
decision itself. This document explains how the decision fits each stage:
research, backtesting and live trading.

## The candidate formats

The same bar data can be organized in four ways:

| Format              | Structure                                   | Typical use                                     |
|---------------------|---------------------------------------------|-------------------------------------------------|
| Long                | `MultiIndex(timestamp, symbol)` x fields    | Ingestion and storage, ML model input, alphalens |
| Dict per asset      | `{"PETR4.SA": OHLCV DataFrame}`             | Most tutorials; the previous ibexQuant project  |
| Wide per field      | `{"close": timestamp x symbol}` (`Panel`)   | Zipline Pipeline, formulaic alphas              |
| 3D array            | time x symbol x field (numpy, xarray)       | Heavy numerical research                        |

None is best at everything. The right choice for each stage depends on the
operations that stage performs most often.

## One system, three representations

ibexQuant uses each format where it is strongest:

```
ingestion (long)  ->  Panel (wide)  ->  feature computation (wide)  ->  to_matrix (long)  ->  model
   storage,           single              time-series and                 one row per
   validation         conversion          cross-sectional operations      observation
```

- **Long** at ingestion: natural for storage, validation and calendar
  alignment, where each row is one bar of one asset.
- **Wide** for computation: see below.
- **Long** again at the model boundary: estimators and factor-evaluation
  tools expect one row per `(timestamp, symbol)` observation.

## Research: why wide wins for computation

Almost every quantitative feature is built from two kinds of operations:

- **Along time, within one asset**: returns, moving averages, volatility,
  lags.
- **Across assets, within one instant**: ranks, cross-sectional z-scores,
  sector neutralization.

The wide format is the only one where both are native, vectorized
operations:

```python
close.rolling(20).std()  # time-series: every symbol at once (axis 0)
momentum.rank(axis=1, pct=True)  # cross-sectional: every timestamp at once (axis 1)
```

In the alternatives, one of the two is always expensive:

- **Long** requires a `groupby` for both: by symbol for time-series, by
  timestamp for cross-sectional. It is slower, more verbose, and forgetting
  the `groupby` silently mixes assets inside a rolling window.
- **Dict per asset** handles time-series naturally, but cross-sectional
  operations require assembling a wide frame on the fly — which is what the
  previous project ended up doing by hand.

## Backtesting: serves both styles

**Vectorized backtests.** Signals, weights and returns are matrices of the
same shape, so simulation becomes matrix algebra:

```python
weights = signal.div(signal.abs().sum(axis=1), axis=0)  # normalize each row
portfolio_returns = (weights.shift(1) * returns).sum(axis=1)  # trade on the next bar
```

**Event-driven backtests.** `Panel` is the replay source: each row is a
snapshot of the whole market at one instant, which is exactly the event the
engine emits.

```python
for timestamp in panel.index:
    snapshot = close.loc[timestamp]  # every symbol at this bar
```

In this mode, however, features are **not** computed from `Panel`. They are
computed by the incremental states of `FeatureStream`, because the purpose of
an event-driven backtest is to reproduce live conditions, and live trading
has no `Panel`.

## Live trading: a deliberately small role

In live trading, bars arrive one at a time. Using `Panel` as the working
format would mean building a new immutable panel on every bar and
recomputing features over the full history — the O(N^2) cost that motivated
the current architecture (see [ADR 0001](../decisions/0001-feature-engineering-architecture.md)).

`Panel` is used only to **warm up** the incremental states:

```
startup:   Panel (last `lookback` bars)  ->  FeatureStream.warm_up()
each bar:  new bars  ->  FeatureStream.update()  ->  model  ->  orders
```

Nothing else in the live path depends on `Panel`.

## Known limitations

The wide format assumes a **common clock**: every asset has bars on the same
timestamps. This holds for daily bars and for fixed-interval intraday bars.
It is the source of the three limitations below, none of which applies to
the current scope.

### Asynchronous data

Tick data and information-driven bars (volume bars, dollar bars — López de
Prado, *Advances in Financial Machine Learning*, ch. 2) close at different
instants for each asset. A panel of such data would be almost entirely
`NaN`.

**Path if needed:** compute time-series features per asset on its own clock,
and resample to a common clock only for cross-sectional operations.

### Multiple frequencies

Prices are daily, fundamentals quarterly, macro data monthly. Forcing them
into one daily panel requires repeating the last known value of each slow
series on every session.

This is semantically different from the forward-fill rejected for price gaps
([ADR 0003](../decisions/0003-missing-data-policy.md)): a quarterly earnings
figure genuinely remains the latest known value until the next report,
whereas a suspended asset's price does not remain its last trade.

**Path if needed:** one `Panel` per frequency, combined only at `to_matrix`
through a point-in-time as-of join.

### Memory at intraday frequencies

The wide format stores every cell, including `NaN`. At the current scale
this is negligible; at one-minute bars it is not:

| Scenario                                                   | Approximate size |
|------------------------------------------------------------|------------------|
| 90 assets, 20 years of daily sessions, 5 `float64` fields  | ~18 MB           |
| Same universe and period, one-minute bars                  | ~7 GB            |

**Path if needed:** process in time chunks, or move computation to a
columnar engine such as polars. Feature contracts are unaffected, because
kernels receive DataFrames, not `Panel`.

## Summary

For the current scope — daily bars, an Ibovespa-sized universe, research,
vectorized backtesting and preparation for event-driven simulation — the
wide per-field format is the most adequate choice:

- it is the only format in which time-series and cross-sectional operations
  are both native;
- it turns vectorized backtesting into matrix algebra;
- it provides a natural replay source for event-driven simulation;
- it stays out of the live critical path, where it would be inadequate.

Its limitations are known and contained by the architecture: kernels depend
on DataFrames rather than on `Panel`, and streaming does not depend on
`Panel` beyond warm-up.
