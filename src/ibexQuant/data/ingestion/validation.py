# ibexQuant\src\ibexQuant\data\ingestion\validation.py

"""
Value-level sanity auditing for OHLCV bar data.

Responsibility
--------------
``check_bars`` audits the *values* inside an already-shaped OHLCV
DataFrame (as returned by any :class:`MarketDataProvider`) and reports
what looks wrong. It never corrects, drops, or fills anything — it only
reports, leaving the decision of what to do about a finding to the
caller. This mirrors a deliberate split seen across quant data
literature: automating the *notification* of a data problem is
tractable, automating its *correction* is not — that always requires
context only a human (or a downstream, purpose-specific model) has.

Two severities, and why they're different
-------------------------------------------
- ``critical_*`` columns: violations of definitions that are true by
  construction, for any market — e.g. a candle's ``high`` can never be
  below its ``low``. There is no legitimate scenario where these should
  fire; a ``True`` here always means the row is broken.
- ``warning_*`` columns: values that are unusual but not universally
  invalid. Negative prices are the concrete example: equities should
  never trade negative, but WTI crude oil futures famously did, for a
  single day, in April 2020. Treating negative price as a hard error
  would silently discard real, correct data for some instruments. It is
  reported, not escalated to critical, so the caller — who knows the
  asset class — makes the call.

What deliberately does not belong here
---------------------------------------
- Spike detection (values that are statistically unusual relative to
  recent volatility). This requires a lookback window and a threshold,
  both of which are judgment calls with no universally correct default,
  and doing it well needs state this stateless, per-call function does
  not have. Left as a future, separate concern.
- Counting or reporting missing (``NaN``) values. It is a single line
  the caller can compute directly (``df.isna().sum()``) whenever they
  need it; it does not carry the same investigative weight as an actual
  wrong value, so it is not dignified with a place in this function's
  contract. Relatedly: a ``NaN`` on either side of any comparison below
  evaluates to ``False`` by construction (`numpy`'s comparison
  semantics), so missing data never triggers a critical or warning
  column here — it is neither flagged nor silently treated as valid,
  it is simply out of scope.
- Trading calendar alignment (deciding whether an absence is a holiday
  or a real gap). See ``ibexQuant.data.ingestion.calendar`` instead.
- Fixing, dropping, or filling anything. This function's output is
  read-only information about the input; the input itself is never
  modified.
"""

from __future__ import annotations

import pandas as pd

from ibexQuant.data.ingestion.provider import OHLCV_FIELDS, validate_bars_shape

_PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")


def check_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit an OHLCV DataFrame for value-level sanity issues.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by ``MultiIndex(timestamp, symbol)``, as returned by any
        :class:`MarketDataProvider`. Does not need every OHLCV field —
        a rule is only evaluated when the columns it needs are present.

    Returns
    -------
    pd.DataFrame
        Same index as ``df``. One boolean column per applicable rule,
        named ``critical_*`` or ``warning_*``; ``True`` marks a row that
        violates that rule. Only rules whose required columns are
        present in ``df`` appear as columns here.

    Raises
    ------
    ValueError
        If ``df`` is not indexed by ``MultiIndex(timestamp, symbol)``,
        or contains none of :data:`ibexQuant.data.ingestion.provider.OHLCV_FIELDS`
        — both indicate a frame that was never shaped by a
        ``MarketDataProvider`` in the first place, not a data-quality
        finding.

    Examples
    --------
    >>> issues = check_bars(bars)
    >>> issues[issues.any(axis=1)]  # every row with at least one finding
    >>> issues.sum()  # a count per rule
    """
    validate_bars_shape(df)

    checks: dict[str, pd.Series] = {}
    open_, high, low, close, volume = (
        df[field] if field in df.columns else None for field in OHLCV_FIELDS
    )

    if high is not None and low is not None:
        checks["critical_high_below_low"] = high < low
    if high is not None and open_ is not None:
        checks["critical_high_below_open"] = high < open_
    if high is not None and close is not None:
        checks["critical_high_below_close"] = high < close
    if low is not None and open_ is not None:
        checks["critical_low_above_open"] = low > open_
    if low is not None and close is not None:
        checks["critical_low_above_close"] = low > close
    if volume is not None:
        checks["critical_negative_volume"] = volume < 0

    price_columns = [field for field in _PRICE_FIELDS if field in df.columns]
    if price_columns:
        checks["warning_negative_price"] = (df[price_columns] < 0).any(axis=1)

    if not checks:
        raise ValueError(
            f"df has none of the expected OHLCV columns {OHLCV_FIELDS}; nothing to check."
        )

    return pd.DataFrame(checks, index=df.index)
