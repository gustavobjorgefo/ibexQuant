# 0007 — Minimum Python version

- Status: Accepted
- Date: 2026-09-27
- Phase: 0

## Context

The project declared `requires-python = ">=3.11"`, while mypy was configured
with `python_version = "3.12"`. Code using 3.12-only features would pass type
checking and fail on the declared minimum version.

The first attempt to fix this aligned mypy down to 3.11. It failed: NumPy 2.5
requires Python 3.12 and its type stubs use the `type` statement (PEP 695),
which only exists from Python 3.12. In a Python 3.12 environment, pip
installs NumPy 2.5, and mypy targeting 3.11 cannot even parse those stubs:

```
numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12 and greater
```

The configuration was therefore inconsistent: it promised 3.11 support while
the development environment resolved dependencies that exist only for 3.12.
The only interpreter actually used by the project is 3.12.

## Decision

- The minimum supported version is **Python 3.12**.
- The settings that express it must always agree:
  - `requires-python = ">=3.12"`
  - `[tool.ruff] target-version = "py312"`
  - `[tool.mypy] python_version = "3.12"`
  - `python-version: "3.12"` in `.github/workflows/ci.yml`

## Alternatives considered

- **Keep 3.11 and point mypy at 3.12.** Removes the error but brings back the
  inconsistency: code using 3.12-only features would pass type checking and
  fail on 3.11.
- **Keep 3.11 and pin `numpy<2.5`.** Holds back a core dependency only to
  support an interpreter the project does not use.
- **Keep 3.11 and test it in CI with a separate environment.** The correct
  way to genuinely support two versions, but it adds maintenance with no
  current benefit for a personal framework.

## Consequences

- Python 3.12 syntax (PEP 695 generics and `type` aliases, among others) may
  be used in the codebase.
- Raising the minimum version again follows the same rule: change all
  settings together and record it here.

## Amendments

- **2026-09-28** — The CI workflow was missing from the list of settings.
  It still installed Python 3.11, so dependency resolution failed on the
  first push after this decision. It now installs 3.12, the minimum
  supported version, so that CI validates the oldest version where
  incompatibilities appear first.
