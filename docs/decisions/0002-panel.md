# 0002 — Panel: the input container for features

- Status: Accepted
- Date: 2026-09-27
- Phase: 0

## Context

The ingestion layer returns OHLCV bars in **long format**: a DataFrame
indexed by `MultiIndex(timestamp, symbol)`, sorted ascending, columns
`open`, `high`, `low`, `close`, `volume` as `float64`, with missing data as
`NaN` and never filled. The standard pipeline is:

```
provider.get_bars() → check_bars() → align_to_calendar()
```

Feature computation (0001, A4) requires **wide format**: one DataFrame per
field, timestamp × symbol. The conversion must happen in exactly one place,
and the result must satisfy invariants every feature can rely on.

The previous project already converged on this shape informally: data was
stored as `dict[str, DataFrame]` per ticker and, for speed, wide `close`
DataFrames were built by hand.

The name follows the econometric term *panel data*: observations of several
entities (assets) over several periods (timestamps).

## Decision

- **D0.1a — A dedicated class.** `Panel`, in `data/panel.py`, is built from
  the aligned long-format frame:

  ```python
  aligned = align_to_calendar(provider.get_bars(symbols, start, end), calendar)
  panel = Panel.from_bars(aligned)

  panel.field("close")  # wide DataFrame: timestamp x symbol
  panel.symbols
  panel.index
  ```

- **D0.1b — Name.** The class is named `Panel`.
- **D0.1c — Immutability.** A `Panel` never changes after construction:
  - attributes are private and exposed only through read-only accessors;
  - accessors return shallow copies (`copy(deep=False)`), which cost nothing
    under pandas Copy-on-Write and guarantee that any write by a caller
    triggers a real copy instead of reaching the `Panel`'s internal data;
  - the project therefore **requires pandas ≥ 3.0**, where Copy-on-Write is
    the default. If an earlier version is ever needed, Copy-on-Write must be
    enabled explicitly.
- **Invariants validated at construction:**
  - every field shares exactly the same index and the same columns;
  - the index is monotonic increasing and has no duplicates;
  - **D0.4** — the index is either entirely timezone-naive or entirely
    timezone-aware, never mixed. `Panel` does not impose which: daily bars
    use naive session dates, intraday bars will use tz-aware timestamps.
    Features are timezone-agnostic;
  - **D0.5** — every field is `float64`.
- `Panel` also carries the universe mask (see 0004) and exposes a content
  fingerprint (see 0005).

### Interface

- **Two construction paths.** `Panel(fields, universe_mask=None)` receives
  wide DataFrames directly (tests, notebooks). `Panel.from_bars(bars,
  universe_mask=None)` unstacks the long-format pipeline output and calls the
  constructor, so all validation lives in one place.
- **Public API.**

  ```python
  panel.field("close")  # wide DataFrame, shallow copy
  panel.fields  # tuple[str, ...], in the order given
  panel.symbols  # tuple[str, ...], sorted
  panel.index  # DatetimeIndex (immutable in pandas)
  panel.shape  # (n_timestamps, n_symbols)
  panel.universe_mask  # wide bool DataFrame, shallow copy
  panel.fingerprint  # str, memoized on first access
  panel.with_universe_mask(m)  # returns a NEW Panel
  ```

  Fields are accessed only through `field(name)`, never as attributes: one
  access path, no dynamic `__getattr__`, full support from type checkers.
- **Free field names.** Any string is accepted as a field name, not only
  OHLCV, so that unadjusted prices (P4) or other data fit without changes.
- **Validation at construction.**

  | Condition                                                        | Error        |
  |------------------------------------------------------------------|--------------|
  | Field or mask not a `DataFrame`; index not a `DatetimeIndex`; field name or symbol not a string | `TypeError` |
  | No fields; zero timestamps or zero symbols                       | `ValueError` |
  | Fields with different index or symbols                           | `ValueError` |
  | Index unsorted or with duplicates; duplicated symbols            | `ValueError` |
  | Field not `float64` (never cast silently)                        | `ValueError` |
  | Mask with different index or symbols, or not `bool` (which also excludes `NaN`) | `ValueError` |

  Homogeneity of the index (D0.4) needs no dedicated check: a
  `DatetimeIndex` cannot mix naive and tz-aware values.
- **Canonical order.** Columns are sorted by symbol, the index is named
  `timestamp` and the column axis `symbol`. Two panels built from the same
  data in different column orders are identical and share a fingerprint.
- **Default mask.** All `True` when no mask is given (see 0004).
- **Fingerprint.** SHA-256 over the index dtype, symbols, field names, each
  field's values and the mask (row hashes from
  `pd.util.hash_pandas_object`). The index dtype is included so that naive
  and tz-aware indices with equal wall-clock values differ. Memoizing it does
  not break immutability, because nothing observable changes.
- **Out of scope for now.** Slicing by period, source metadata (frequency,
  adjustment) and equality between panels. Each is added when a phase needs
  it.

## Alternatives considered

- **A plain `dict[str, DataFrame]`.** Rejected: nothing enforces that fields
  are aligned. A misaligned field surfaces much later as an unexplained `NaN`
  inside some feature.
- **Keep long format for computation.** Rejected: time-series operations
  would require a `groupby("symbol")` on every node and cross-sectional
  operations a `groupby("timestamp")`, both slower and more verbose than
  operating on wide frames.
- **`dataclass(frozen=True)` alone.** Rejected as insufficient: it blocks
  attribute reassignment but not mutation of a DataFrame held by an
  attribute (`panel.close.iloc[0, 0] = x`). Immutability is shallow; the
  Copy-on-Write accessors close that gap.
- **Other names.** `BarPanel` (more specific), `MarketData` (too vague),
  `DataCube` (uncommon in finance), `FieldMatrix` (describes the
  implementation rather than the concept), `Universe` (would collide with
  the universe mask). `Panel` was preferred for being short and standard.
- **`float32`.** Halves memory, but introduces numerical differences that
  complicate parity tests and brings no benefit at the current universe
  size and bar frequency.

## Consequences

- Every feature receives aligned, validated, immutable inputs; kernels do
  not repeat input validation.
- `pyproject.toml` pins `pandas >= 3.0`.
- The wide format assumes a common clock across assets. Its known
  limitations (asynchronous data, multiple frequencies, intraday memory)
  and how `Panel` fits research, backtesting and live trading are described
  in [Data representation across the system](../design/data-representation.md).
- Supporting both naive and tz-aware indices keeps daily and intraday data
  usable. When the live engine is designed, its timestamp convention must be
  compatible with this contract (pending item **P1**).\
