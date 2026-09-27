# 0006 — Legacy code

- Status: Accepted
- Date: 2026-09-27
- Phase: 0

## Context

A previous ibexQuant project exists with its own data handlers, transforms
(`data/transforms/`) and Series-only feature functions (`features/*.py`).
That project is frozen; it has no live engine in use. The current project
was started from scratch.

## Decision

- **D0.6 — No migration.** The current project is built without any code
  carried over from the previous one. No compatibility layers, adapters or
  deprecation shims are written.
- The previous project is kept strictly as a **reference**: its problems
  are recorded as the context of 0001, and useful numerical logic may be
  re-derived when writing new kernels, always under the current contracts
  and coding standard.

## Alternatives considered

- **Coexistence with deprecation until the streaming engine exists.** Only
  relevant if the old transforms were running in production. They are not.

## Consequences

- The `features/` package starts empty; there is no path conflict between
  old and new modules.
- Old functions are not reused as-is: they accept only `pd.Series`, while
  the new kernels operate on wide DataFrames (0001, A4) and follow the
  observation-time policy (0003).