# ibexQuant\tests\data\test_adjustment_store.py

"""Factor build, stale-table check, adjusted provider and CLI (ADR 0014)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from support.ledger import PETR, VALE, ledger_store

from ibexQuant.cli import EXIT_OK, EXIT_USAGE, main
from ibexQuant.data.adjustment import (
    DERIVED_PARTS,
    SUMMARY_FILE,
    AdjustmentFactorsNotBuiltError,
    StaleAdjustmentFactorsError,
    build_adjustment_factors,
    read_adjustment_factors,
)
from ibexQuant.data.ingestion.b3 import B3Provider
from ibexQuant.data.reference.corporate_action_overrides import (
    ActionOverrides,
    load_action_overrides,
)
from ibexQuant.data.reference.corporate_actions import build_corporate_actions
from ibexQuant.data.store import DATA_ROOT_ENV_VAR, DataStore
from ibexQuant.data.store import config as config_module

START, END = "2024-04-22", "2024-05-03"
SPIN_OFF_YAML = (
    "version: 1\nevents:\n"
    f"  - {{isin: {VALE}, last_cum_date: 2024-04-22, type: SPIN_OFF, share_ratio: 0.1,\n"
    f"     target_isin: {PETR}, note: test}}\n"
)


@pytest.fixture
def ledger(tmp_path: Path) -> DataStore:
    store = ledger_store(tmp_path)
    build_corporate_actions(store, ActionOverrides.none())
    return store


@pytest.fixture
def built(ledger: DataStore) -> DataStore:
    build_adjustment_factors(ledger)
    return ledger


# --- build and read ---


def test_the_build_writes_one_factor_per_explained_ex_session(built: DataStore) -> None:
    factors = read_adjustment_factors(built)
    assert factors["ex_date"].dt.strftime("%m-%d").tolist() == ["04-24", "04-26", "04-30"]
    assert factors["f_total"].round(6).tolist() == [0.95, 0.5, 0.1]
    assert factors["f_splits"].tolist() == [1.0, 0.5, 0.1]
    summary = json.loads(built.derived(*DERIVED_PARTS, SUMMARY_FILE).read_text("utf-8"))
    assert summary["factors"] == 3
    assert summary["unexplained"] == {"unexplained": 2, "unexplained_jump": 1}
    assert set(summary["ledger"]) == {"corporate_actions.parquet", "ex_sessions.parquet"}


def test_a_spin_off_is_priced_from_the_targets_bars(tmp_path: Path, ledger: DataStore) -> None:
    root = tmp_path / "overrides"
    root.mkdir()
    (root / "corporate_actions.yaml").write_text(SPIN_OFF_YAML, encoding="utf-8")
    build_corporate_actions(ledger, load_action_overrides(root))
    build_adjustment_factors(ledger)
    factors = read_adjustment_factors(ledger)
    spin = factors[factors["security_id"] == VALE].iloc[0]
    assert spin["cash_value"] == pytest.approx(0.1 * 10.0)
    assert spin["f_total"] == pytest.approx(1 - 1.0 / 60.0)


def test_a_spin_off_without_target_bars_is_rejected(tmp_path: Path, ledger: DataStore) -> None:
    root = tmp_path / "overrides"
    root.mkdir()
    (root / "corporate_actions.yaml").write_text(SPIN_OFF_YAML, encoding="utf-8")
    build_corporate_actions(ledger, load_action_overrides(root))
    ledger.curated("b3", "bars_daily", "year=2024", "part-0.parquet").unlink()
    summary = build_adjustment_factors(ledger)
    assert summary.rejected == {"no_target_price": 1}


def test_factors_must_be_built_first(ledger: DataStore) -> None:
    with pytest.raises(AdjustmentFactorsNotBuiltError, match="adjustment-factors"):
        read_adjustment_factors(ledger)


def test_a_changed_ledger_makes_the_factors_stale(built: DataStore) -> None:
    path = built.curated("b3", "ex_sessions.parquet")
    pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    with pytest.raises(StaleAdjustmentFactorsError, match="adjustment-factors"):
        read_adjustment_factors(built)


# --- provider ---


def test_adjusted_bars_are_total_return_at_the_end_by_default(built: DataStore) -> None:
    bars = B3Provider(built).get_bars("PETR3", START, END)
    close = bars["close"].tolist()
    assert close[0] == pytest.approx(10.0 * 0.95 * 0.5 * 0.1)
    assert close[-1] == 0.3
    assert bars["volume"].iloc[0] == 1000 * 2 * 10


def test_splits_mode_and_an_anchor_after_end(built: DataStore) -> None:
    bars = B3Provider(built).get_bars(
        "PETR3", START, "2024-04-25", anchor="2024-04-29", adjustment="splits"
    )
    assert bars["close"].tolist() == [5.0, 5.0, 5.0, 5.0]


def test_follow_renames_off_is_adjusted_through_the_isin(built: DataStore) -> None:
    bars = B3Provider(built, follow_renames=False).get_bars(
        "PETR3", START, "2024-04-23", anchor=END
    )
    assert bars["close"].round(6).tolist() == [0.475, 0.475]


def test_unexplained_jumps_in_range_are_logged(
    built: DataStore, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        B3Provider(built).get_bars("PETR3", START, END)
    assert "Unexplained jump kept in adjusted bars: PETR3 on 2024-05-03" in caplog.text


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"anchor": "2024-04-30"}, "before end"),
        ({"adjustment": "raw"}, "adjusted=False"),
        ({"adjusted": False, "anchor": END}, "adjusted bars only"),
        ({"adjusted": False, "adjustment": "splits"}, "adjusted bars only"),
    ],
)
def test_inconsistent_options_are_errors(
    built: DataStore, options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        B3Provider(built).get_bars("PETR3", START, END, **options)  # type: ignore[arg-type]


def test_raw_bars_need_no_factors(ledger: DataStore) -> None:
    bars = B3Provider(ledger).get_bars("PETR3", START, END, adjusted=False)
    assert bars["close"].iloc[0] == 10.0


# --- CLI ---


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "absent.toml")
    return tmp_path


def test_cli_builds_the_factors(
    ledger: DataStore,
    environment: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(ledger.root))
    assert main(["data", "build", "adjustment-factors"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "factors: 3" in out
    assert "rejected: none" in out
    assert "ex-sessions kept raw: unexplained 2, unexplained_jump 1" in out
    assert read_adjustment_factors(ledger)["security_id"].unique().tolist() == [PETR]


def test_cli_without_a_ledger_is_a_usage_error(
    environment: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = DataStore.initialize(environment / "data")
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(store.root))
    assert main(["data", "build", "adjustment-factors"]) == EXIT_USAGE
    assert "corporate-actions" in capsys.readouterr().err
