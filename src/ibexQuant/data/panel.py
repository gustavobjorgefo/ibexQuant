# ibexQuant\src\ibexQuant\data\panel.py

"""
Immutable wide-format container of bar data for feature computation.

This module provides :class:`Panel`, the single input every feature in
``ibexQuant.features`` receives. It holds one wide DataFrame per field
(timestamp x symbol) plus a boolean universe mask of the same shape.

Responsibilities
----------------
- Convert the long-format output of the ingestion pipeline
  (``get_bars -> check_bars -> align_to_calendar``) into wide format, in
  exactly one place (:meth:`Panel.from_bars`).
- Enforce, once and at construction, the invariants every feature relies
  on: all fields share the same index and symbols, the index is a sorted
  ``DatetimeIndex`` without duplicates, and every field is ``float64``.
- Guarantee immutability: accessors return shallow copies which, under
  pandas Copy-on-Write (default since pandas 3.0), never let a caller's
  write reach the internal data.
- Expose a content fingerprint used by the feature cache.

Contract
--------
- ``NaN`` in a field means "not observable at this timestamp" (asset not
  yet listed, delisted, or a real gap). It is never filled here.
- The universe mask marks which assets are investable at each timestamp.
  It is consumed only by cross-sectional features and by the conversion
  to a model matrix, never by time-series features.
- The index may be timezone-naive (daily session dates) or timezone-aware
  (intraday instants); a ``DatetimeIndex`` cannot mix both.

What does not belong here
-------------------------
- Calendar alignment and data quality checks — see
  ``ibexQuant.data.ingestion``.
- Building the universe (index composition, liquidity rules) — the mask is
  received ready-made.
- Any feature computation, slicing by period, or metadata about the data
  source (frequency, adjustment).

See ``docs/decisions/0002-panel.md`` for the rationale behind this design.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import cached_property
from typing import Final, Self, cast

import numpy as np
import pandas as pd

TIMESTAMP_LEVEL: Final[str] = "timestamp"
SYMBOL_LEVEL: Final[str] = "symbol"
FIELD_DTYPE: Final[np.dtype] = np.dtype("float64")
MASK_DTYPE: Final[np.dtype] = np.dtype("bool")

# Separates variable-length names inside the fingerprint, so that e.g.
# fields ("ab", "c") and ("a", "bc") cannot produce the same byte stream.
_FINGERPRINT_SEPARATOR: Final[bytes] = b"\x00"


class Panel:
    """
    Immutable collection of wide bar DataFrames sharing one index and symbols.

    Each field (e.g. ``"close"``) is a DataFrame indexed by timestamp with one
    column per symbol. All fields share exactly the same index and the same
    columns; columns are stored sorted by symbol, the index is named
    ``"timestamp"`` and the columns axis ``"symbol"``.

    Parameters
    ----------
    fields : Mapping[str, pd.DataFrame]
        Field name to wide DataFrame. Field names are free (not restricted to
        OHLCV). Every DataFrame must have a sorted ``DatetimeIndex`` without
        duplicates, unique string column labels, and ``float64`` columns.
        Column order does not matter; columns are sorted on construction.
    universe_mask : pd.DataFrame, optional
        Boolean DataFrame with the same index and symbols as the fields.
        ``True`` means the asset belongs to the universe at that timestamp.
        Defaults to all ``True``.

    Raises
    ------
    TypeError
        If a field or the mask is not a DataFrame, if an index is not a
        ``DatetimeIndex``, or if a field name or symbol label is not a
        string.
    ValueError
        If no field is given, if a field is empty, if fields are not
        aligned, if an index is unsorted or has duplicates, if symbols are
        duplicated, if a field is not ``float64``, or if the mask does not
        match the fields' shape or is not ``bool``.

    See Also
    --------
    Panel.from_bars : Build a panel from the long-format ingestion output.

    Examples
    --------
    >>> panel = Panel.from_bars(aligned_bars)
    >>> close = panel.field("close")
    >>> panel.symbols
    ('ITUB4.SA', 'PETR4.SA', 'VALE3.SA')
    """

    def __init__(
        self,
        fields: Mapping[str, pd.DataFrame],
        universe_mask: pd.DataFrame | None = None,
    ) -> None:
        normalized_fields: dict[str, pd.DataFrame] = self._normalize_fields(fields)
        reference: pd.DataFrame = next(iter(normalized_fields.values()))

        self._fields: dict[str, pd.DataFrame] = normalized_fields
        # Guaranteed by _normalize_frame, which rejects any other index type.
        self._index: pd.DatetimeIndex = cast(pd.DatetimeIndex, reference.index)
        self._symbols: tuple[str, ...] = tuple(reference.columns)
        self._universe_mask: pd.DataFrame = self._normalize_universe_mask(universe_mask, reference)

    # --- Construction ---

    @classmethod
    def from_bars(
        cls,
        bars: pd.DataFrame,
        universe_mask: pd.DataFrame | None = None,
    ) -> Self:
        """
        Build a panel from long-format bars.

        Parameters
        ----------
        bars : pd.DataFrame
            Indexed by ``MultiIndex(timestamp, symbol)`` with no duplicate
            pairs, as produced by the ingestion pipeline. Every column
            becomes a field.
        universe_mask : pd.DataFrame, optional
            Forwarded to the constructor. Defaults to all ``True``.

        Returns
        -------
        Panel
            One field per column of *bars*. Timestamps where a symbol has no
            row (e.g. before its listing) become ``NaN``.

        Raises
        ------
        TypeError
            If *bars* is not a DataFrame.
        ValueError
            If *bars* is not indexed by ``MultiIndex(timestamp, symbol)`` or
            contains duplicate ``(timestamp, symbol)`` pairs, plus any error
            raised by the constructor.
        """
        if not isinstance(bars, pd.DataFrame):
            raise TypeError(f"Expected bars to be a pd.DataFrame, got {type(bars)!r}.")

        expected_levels: list[str] = [TIMESTAMP_LEVEL, SYMBOL_LEVEL]
        if not isinstance(bars.index, pd.MultiIndex) or list(bars.index.names) != expected_levels:
            raise ValueError(
                f"Expected bars indexed by MultiIndex{tuple(expected_levels)}, "
                f"got index names {list(bars.index.names)}."
            )

        if bars.index.has_duplicates:
            raise ValueError("bars contains duplicate (timestamp, symbol) pairs.")

        fields: dict[str, pd.DataFrame] = {
            name: bars[name].unstack(SYMBOL_LEVEL).sort_index() for name in bars.columns
        }
        return cls(fields, universe_mask)

    def with_universe_mask(self, universe_mask: pd.DataFrame) -> Self:
        """
        Return a new panel with the same fields and a different universe mask.

        Parameters
        ----------
        universe_mask : pd.DataFrame
            Boolean DataFrame with the same index and symbols as this panel.

        Returns
        -------
        Panel
            A new instance; this panel is left unchanged.

        Raises
        ------
        TypeError
            If *universe_mask* is not a DataFrame or its index is not a
            ``DatetimeIndex``.
        ValueError
            If *universe_mask* does not match this panel's shape or is not
            ``bool``.
        """
        return type(self)(self._fields, universe_mask)

    # --- Read-only accessors ---

    def field(self, name: str) -> pd.DataFrame:
        """
        Return the wide DataFrame of one field.

        Parameters
        ----------
        name : str
            Field name, one of :attr:`fields`.

        Returns
        -------
        pd.DataFrame
            Timestamp x symbol. A shallow copy: writing to it never affects
            this panel.

        Raises
        ------
        KeyError
            If *name* is not a field of this panel.
        """
        if name not in self._fields:
            raise KeyError(f"Unknown field '{name}'. Available fields: {self.fields}.")
        return self._fields[name].copy(deep=False)

    @property
    def fields(self) -> tuple[str, ...]:
        """tuple[str, ...]: Field names, in the order they were given."""
        return tuple(self._fields)

    @property
    def symbols(self) -> tuple[str, ...]:
        """tuple[str, ...]: Symbols, sorted ascending."""
        return self._symbols

    @property
    def index(self) -> pd.DatetimeIndex:
        """pd.DatetimeIndex: Shared timestamp index, sorted ascending."""
        # pandas Index objects are immutable; no copy is needed.
        return self._index

    @property
    def shape(self) -> tuple[int, int]:
        """tuple[int, int]: ``(number of timestamps, number of symbols)``."""
        return len(self._index), len(self._symbols)

    @property
    def universe_mask(self) -> pd.DataFrame:
        """pd.DataFrame: Boolean universe mask, timestamp x symbol (shallow copy)."""
        return self._universe_mask.copy(deep=False)

    @cached_property
    def fingerprint(self) -> str:
        """
        str: SHA-256 hex digest identifying this panel's content.

        Covers field names, symbols, index (values and dtype), every field's
        values and the universe mask. Independent of the column order given
        at construction.
        Computed on first access and memoized; memoization changes nothing
        observable, so immutability is preserved.
        """
        digest = hashlib.sha256()
        # The dtype distinguishes naive from tz-aware indices with equal wall-clock values.
        digest.update(str(self._index.dtype).encode())
        digest.update(_FINGERPRINT_SEPARATOR.join(s.encode() for s in self._symbols))
        for name in sorted(self._fields):
            digest.update(_FINGERPRINT_SEPARATOR + name.encode() + _FINGERPRINT_SEPARATOR)
            digest.update(self._hash_frame(self._fields[name]))
        digest.update(_FINGERPRINT_SEPARATOR + b"universe_mask" + _FINGERPRINT_SEPARATOR)
        digest.update(self._hash_frame(self._universe_mask))
        return digest.hexdigest()

    # --- Representation ---

    def __repr__(self) -> str:
        n_timestamps, n_symbols = self.shape
        return (
            f"{type(self).__name__}("
            f"fields={self.fields}, "
            f"symbols={n_symbols}, "
            f"timestamps={n_timestamps}, "
            f"start={self._index[0]}, "
            f"end={self._index[-1]})"
        )

    # --- Validation helpers ---

    @classmethod
    def _normalize_fields(cls, fields: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        if not fields:
            raise ValueError("Panel requires at least one field.")

        non_string_names: list[object] = [name for name in fields if not isinstance(name, str)]
        if non_string_names:
            raise TypeError(f"Field names must be strings, got {non_string_names}.")

        normalized: dict[str, pd.DataFrame] = {
            name: cls._normalize_frame(frame, label=f"field '{name}'")
            for name, frame in fields.items()
        }

        for name, frame in normalized.items():
            if not all(dtype == FIELD_DTYPE for dtype in frame.dtypes):
                raise ValueError(
                    f"Field '{name}' must be {FIELD_DTYPE}; got dtypes "
                    f"{sorted(set(map(str, frame.dtypes)))}."
                )

        reference_name, reference = next(iter(normalized.items()))
        for name, frame in normalized.items():
            cls._require_aligned(
                frame,
                reference,
                label=f"field '{name}'",
                reference_label=f"field '{reference_name}'",
            )
        return normalized

    @classmethod
    def _normalize_universe_mask(
        cls,
        universe_mask: pd.DataFrame | None,
        reference: pd.DataFrame,
    ) -> pd.DataFrame:
        if universe_mask is None:
            return pd.DataFrame(True, index=reference.index, columns=reference.columns)

        mask: pd.DataFrame = cls._normalize_frame(universe_mask, label="universe_mask")
        cls._require_aligned(mask, reference, label="universe_mask", reference_label="fields")

        # A bool dtype cannot hold NaN, so this check also rejects missing values.
        if not all(dtype == MASK_DTYPE for dtype in mask.dtypes):
            raise ValueError(
                f"universe_mask must be {MASK_DTYPE}; got dtypes "
                f"{sorted(set(map(str, mask.dtypes)))}."
            )
        return mask

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
        """
        Validate one wide frame and return it with sorted, named axes.

        Under Copy-on-Write the returned object is a lazy copy: later writes
        to the caller's original frame never reach it.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Expected {label} to be a pd.DataFrame, got {type(frame)!r}.")

        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(
                f"Expected {label} to have a pd.DatetimeIndex, got {type(frame.index)!r}."
            )

        if frame.empty:
            raise ValueError(
                f"{label} is empty: shape {frame.shape}. "
                f"Panel requires at least one timestamp and one symbol."
            )

        if frame.index.has_duplicates:
            raise ValueError(f"{label} has duplicate timestamps.")

        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{label} index is not sorted ascending.")

        if frame.columns.has_duplicates:
            raise ValueError(f"{label} has duplicate symbols.")

        non_string_symbols: list[object] = [s for s in frame.columns if not isinstance(s, str)]
        if non_string_symbols:
            raise TypeError(f"{label} has non-string symbols: {non_string_symbols}.")

        return frame.sort_index(axis=1).rename_axis(index=TIMESTAMP_LEVEL, columns=SYMBOL_LEVEL)

    @staticmethod
    def _require_aligned(
        frame: pd.DataFrame,
        reference: pd.DataFrame,
        *,
        label: str,
        reference_label: str,
    ) -> None:
        if not frame.index.equals(reference.index):
            raise ValueError(f"{label} index does not match the index of {reference_label}.")

        if not frame.columns.equals(reference.columns):
            missing: list[str] = sorted(set(reference.columns) - set(frame.columns))
            extra: list[str] = sorted(set(frame.columns) - set(reference.columns))
            raise ValueError(
                f"{label} symbols do not match {reference_label}: "
                f"missing {missing}, unexpected {extra}."
            )

    @staticmethod
    def _hash_frame(frame: pd.DataFrame) -> bytes:
        row_hashes: pd.Series = pd.util.hash_pandas_object(frame, index=True)
        return row_hashes.to_numpy().tobytes()
