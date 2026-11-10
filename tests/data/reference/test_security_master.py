# ibexQuant\tests\data\reference\test_security_master.py

"""
Tests for :func:`ibexQuant.data.reference.build_security_master` and
:class:`ibexQuant.data.reference.SecurityMaster` (ADR 0012).

The main fixture is a six-session market (``tests/support/reference.py``)
holding the cases found in the H2 exploration: a chain of renames confirmed
by CNPJ, an incorporation across CNPJs, and a share with an odd-lot ticker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from support.cotahist import detail, lines_for, write_zip
from support.reference import (
    BHIA,
    FCA_ROWS,
    PETR,
    TIMP,
    TIMS,
    VIIA,
    VVAR,
    isin,
    rename_market,
    write_fca,
    write_links,
)

from ibexQuant.data.ingestion.b3 import build_bars_daily
from ibexQuant.data.reference import (
    LinkOverrides,
    SecurityMaster,
    SecurityMasterBuildError,
    SecurityMasterNotBuiltError,
    SecurityNotFoundError,
    build_security_master,
    implied_factor,
    is_isin,
    load_link_overrides,
    read_candidates,
)
from ibexQuant.data.store import DataStore

# --- Fixtures ---


def market(tmp_path: Path, details: list[bytes], *, fca: bool = True) -> DataStore:
    store = DataStore.initialize(tmp_path / "ibex-data")
    write_zip(store.raw("b3", "cotahist"), 2024, lines_for(2024, details))
    build_bars_daily(store)
    if fca:
        write_fca(store.raw("cvm", "fca"), 2024, FCA_ROWS)
    return store


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    return market(tmp_path, rename_market())


def build(store: DataStore, tmp_path: Path, **pairs: list[tuple[str, str]]) -> None:
    folder = tmp_path / "overrides"
    write_links(folder, **pairs)
    build_security_master(store, load_link_overrides(folder))


def security_of(master: SecurityMaster, ticker: str) -> str:
    rows = master.table
    return str(rows.loc[rows["ticker"] == ticker, "security_id"].iloc[0])


def status_of(store: DataStore, old: str, new: str) -> str:
    candidates = read_candidates(store)
    row = candidates[(candidates["from_isin"] == old) & (candidates["to_isin"] == new)]
    return str(row["status"].iloc[0])


# --- ISIN check digit ---


@pytest.mark.parametrize("code", ["BRPETRACNPR6", "BRVALEACNOR0", "BRBHIAACNOR1", "US0378331005"])
def test_valid_isins(code: str) -> None:
    assert is_isin(code)


@pytest.mark.parametrize(
    "code", ["BRPETRACNPR7", "ACESACON", "BRPETRACNPRX", "12PETRACNPR6", "BRPETRA-NPR6", ""]
)
def test_invalid_isins(code: str) -> None:
    assert not is_isin(code)


# --- Implied split factor (hint for review) ---


@pytest.mark.parametrize(
    ("ratio", "factor"),
    [(4.994, 5.0), (0.254, 0.25), (19.63, 20.0), (0.0503, 0.05), (2.0, 2.0), (0.001005, 0.001)],
)
def test_ratio_close_to_a_common_factor(ratio: float, factor: float) -> None:
    assert implied_factor(ratio) == factor


@pytest.mark.parametrize("ratio", [1.0, 1.02, 7.0, 0.6, 1500.0, 0.0, -2.0, float("nan")])
def test_ratio_without_a_common_factor(ratio: float) -> None:
    assert implied_factor(ratio) is None


def test_candidates_carry_the_implied_factor(tmp_path: Path) -> None:
    old, new = isin("OLDX"), isin("NEWX")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="OLDX3", isin=old, close=10.0),
            detail(session="20240103", ticker="NEWX3", isin=new, open_=50.0),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    row = read_candidates(store).iloc[0]
    assert row["ratio"] == 5.0
    assert row["implied_factor"] == 5.0
    assert row["status"] == "pending"  # a hint, never a decision


# --- Rule 1: same ISIN ---


def test_odd_lot_ticker_is_the_same_security(store: DataStore) -> None:
    build_security_master(store, LinkOverrides.none())
    master = SecurityMaster.load(store)
    assert security_of(master, "PETR4F") == security_of(master, "PETR4") == PETR


def test_ticker_format_change_with_the_same_isin(tmp_path: Path) -> None:
    code = isin("ACES")
    store = market(
        tmp_path,
        [
            detail(session="19980313", ticker="ACE 3", isin=code),
            detail(session="19980316", ticker="ACES3", isin=code),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    table = SecurityMaster.load(store).table.set_index("ticker")
    assert table.loc["ACE 3", "security_id"] == table.loc["ACES3", "security_id"] == code
    assert table.loc["ACE 3", "link_rule"] == "founding"
    assert table.loc["ACES3", "link_rule"] == "same_isin"


# --- Rule 2: pre-1995 code to ISIN ---


def test_internal_code_continues_as_the_isin_of_its_ticker(tmp_path: Path) -> None:
    code = isin("ACES")
    store = market(
        tmp_path,
        [
            detail(session="19950110", ticker="ACE 3", isin="ACESACON"),
            detail(session="19950316", ticker="ACE 3", isin=code),
        ],
        fca=False,
    )
    summary = build_security_master(store, LinkOverrides.none())
    table = SecurityMaster.load(store).table.set_index("isin")
    assert table.loc[code, "security_id"] == "ACESACON"
    assert table.loc[code, "link_rule"] == "internal_code"
    assert summary.links["internal_code"] == 1


def test_internal_code_with_two_successors_is_left_alone(tmp_path: Path) -> None:
    store = market(
        tmp_path,
        [
            detail(session="19950110", ticker="ACE 5", isin="ACESS1ON"),
            detail(session="19950614", ticker="ACE 5", isin=isin("ACES", "D01PR")),
            detail(session="19950803", ticker="ACE 5", isin=isin("ACES", "A02PR")),
        ],
        fca=False,
    )
    summary = build_security_master(store, LinkOverrides.none())
    assert summary.ambiguous_internal_codes == 1
    assert SecurityMaster.load(store).table["security_id"].nunique() == 3


# --- Rule 3: adjacent renames ---


def test_renames_confirmed_by_cnpj_form_one_security(store: DataStore) -> None:
    summary = build_security_master(store, LinkOverrides.none())
    master = SecurityMaster.load(store)

    assert security_of(master, "VVAR3") == security_of(master, "VIIA3") == VVAR
    assert security_of(master, "BHIA3") == VVAR
    assert summary.links["cnpj"] == 2
    assert status_of(store, VVAR, VIIA) == "linked_by_cnpj"
    assert status_of(store, VIIA, BHIA) == "linked_by_cnpj"


def test_link_rule_and_cnpj_per_span(store: DataStore) -> None:
    build_security_master(store, LinkOverrides.none())
    table = SecurityMaster.load(store).table.set_index("ticker")
    assert table.loc["VVAR3", "link_rule"] == "founding"
    assert table.loc["VIIA3", "link_rule"] == "cnpj"
    assert table.loc["BHIA3", "cnpj"] == "33.041.260/0652-90"


def test_different_cnpjs_wait_for_review_as_likely(store: DataStore) -> None:
    # TIMP3 -> TIMS3: one partner each, and the price continues (20.0 -> 20.0).
    summary = build_security_master(store, LinkOverrides.none())
    master = SecurityMaster.load(store)

    assert security_of(master, "TIMP3") != security_of(master, "TIMS3")
    assert status_of(store, TIMP, TIMS) == "likely"
    assert (summary.candidates["likely"], summary.candidates["pending"]) == (1, 0)


def test_price_jump_keeps_a_pair_pending(tmp_path: Path) -> None:
    old, new = isin("OLDX"), isin("NEWX")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="OLDX3", isin=old, close=10.0),
            detail(session="20240103", ticker="NEWX3", isin=new, open_=30.0),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    assert status_of(store, old, new) == "pending"


def test_listing_with_two_partners_is_not_likely(tmp_path: Path) -> None:
    old, first, second = isin("OLDX"), isin("NEWA"), isin("NEWB")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="OLDX3", isin=old, close=10.0),
            detail(session="20240103", ticker="NEWA3", isin=first, open_=10.0),
            detail(session="20240104", ticker="NEWB3", isin=second, open_=10.0),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    assert status_of(store, old, first) == "pending"
    assert status_of(store, old, second) == "pending"


@pytest.mark.parametrize("kind", ["CTF00", "BDR00"], ids=["fund-quota", "bdr"])
def test_fund_quotas_and_bdrs_are_not_candidates(tmp_path: Path, kind: str) -> None:
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="AAAA11", isin=isin("AAAA", kind)),
            detail(session="20240103", ticker="BBBB11", isin=isin("BBBB", kind)),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    assert read_candidates(store).empty


def test_units_are_candidates(tmp_path: Path) -> None:
    old, new = isin("AAAA", "UNT00"), isin("BBBB", "UNT00")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="AAAA11", isin=old),
            detail(session="20240103", ticker="BBBB11", isin=new),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    assert status_of(store, old, new) == "likely"


def test_pairs_with_an_already_linked_side_are_superseded(store: DataStore) -> None:
    build_security_master(store, LinkOverrides.none())
    assert status_of(store, VVAR, TIMS) == "superseded"
    assert status_of(store, TIMP, BHIA) == "superseded"


def test_without_fca_every_rename_is_pending(tmp_path: Path) -> None:
    store = market(tmp_path, rename_market(), fca=False)
    summary = build_security_master(store, LinkOverrides.none())
    assert summary.links["cnpj"] == 0
    assert SecurityMaster.load(store).table["security_id"].nunique() == 6


def test_candidate_prices_around_the_change(store: DataStore) -> None:
    build_security_master(store, LinkOverrides.none())
    candidates = read_candidates(store)
    row = candidates[(candidates["from_isin"] == VVAR) & (candidates["to_isin"] == VIIA)].iloc[0]
    assert row["from_ticker"] == "VVAR3"
    assert row["to_ticker"] == "VIIA3"
    assert row["from_last_session"] == pd.Timestamp("2024-01-03")
    assert row["to_first_session"] == pd.Timestamp("2024-01-04")
    assert row["gap_sessions"] == 1
    assert (row["old_close"], row["new_open"], row["ratio"]) == (11.0, 11.0, 1.0)


def test_only_listings_of_the_same_kind_are_candidates(store: DataStore) -> None:
    build_security_master(store, LinkOverrides.none())
    candidates = read_candidates(store)
    assert PETR not in set(candidates["from_isin"]) | set(candidates["to_isin"])


def test_gap_longer_than_five_sessions_is_not_a_candidate(tmp_path: Path) -> None:
    old, new = isin("AAAA"), isin("BBBB")
    days = ["20240102", "20240103", "20240104", "20240105", "20240108", "20240109", "20240110"]
    details = [detail(session=d, ticker="FILL3", isin=isin("FILL", "ACNPR")) for d in days]
    details += [
        detail(session=days[0], ticker="AAAA3", isin=old),
        detail(session=days[6], ticker="BBBB3", isin=new),
    ]
    store = market(tmp_path, details, fca=False)
    build_security_master(store, LinkOverrides.none())
    assert read_candidates(store).empty


def test_two_confirmed_partners_at_the_same_distance_conflict(tmp_path: Path) -> None:
    old, first, second = isin("OLDX"), isin("NEWA"), isin("NEWB")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="OLDX3", isin=old),
            detail(session="20240103", ticker="NEWA3", isin=first),
            detail(session="20240103", ticker="NEWB3", isin=second),
        ],
        fca=False,
    )
    write_fca(store.raw("cvm", "fca"), 2024, [("OLDX3", "1"), ("NEWA3", "1"), ("NEWB3", "1")])
    summary = build_security_master(store, LinkOverrides.none())
    assert summary.candidates["conflict"] == 2
    assert SecurityMaster.load(store).table["security_id"].nunique() == 3


def test_ambiguous_issuer_codes_are_reported_and_unused(tmp_path: Path) -> None:
    store = market(tmp_path, rename_market(), fca=False)
    write_fca(store.raw("cvm", "fca"), 2024, [("VVAR3", "1"), ("VIIA3", "1")])
    write_fca(store.raw("cvm", "fca"), 2025, [])
    write_fca(store.raw("cvm", "fca"), 2023, [("VVAR3", "2")])
    # 2023 FCA names VVAR3 under another CNPJ, but VVAR3 did not trade in 2023.
    summary = build_security_master(store, LinkOverrides.none())
    assert summary.ambiguous_issuers == []

    write_fca(store.raw("cvm", "fca"), 2024, [("VVAR3", "1"), ("VVAR3", "2"), ("VIIA3", "1")])
    summary = build_security_master(store, LinkOverrides.none())
    assert summary.ambiguous_issuers == ["VVAR"]
    assert status_of(store, VVAR, VIIA) == "pending"


# --- Rule 4: overrides ---


def test_override_links_across_cnpjs(store: DataStore, tmp_path: Path) -> None:
    build(store, tmp_path, links=[(TIMP, TIMS)])
    master = SecurityMaster.load(store)
    assert security_of(master, "TIMS3") == TIMP
    assert master.table.set_index("ticker").loc["TIMS3", "link_rule"] == "override"
    assert status_of(store, TIMP, TIMS) == "linked_by_override"


def test_override_rejects_a_confirmed_candidate(store: DataStore, tmp_path: Path) -> None:
    build(store, tmp_path, rejections=[(VIIA, BHIA)])
    master = SecurityMaster.load(store)
    assert security_of(master, "BHIA3") == BHIA
    assert status_of(store, VIIA, BHIA) == "rejected"


def test_override_links_listings_that_are_not_candidates(tmp_path: Path) -> None:
    old, new = isin("AAAA"), isin("BBBB")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="AAAA3", isin=old),
            detail(session="20240301", ticker="BBBB3", isin=new),
        ],
        fca=False,
    )
    build(store, tmp_path, links=[(old, new)])
    assert SecurityMaster.load(store).resolve("BBBB3", "2024-03-01") == old


@pytest.mark.parametrize(
    ("links", "match"),
    [
        ([("BRXXXXACNOR1", TIMS)], "unknown ISIN"),
        ([(TIMS, TIMP)], "traded before"),
        ([(VVAR, VIIA), (VVAR, TIMS)], "chain"),
        ([(TIMP, BHIA), (VIIA, BHIA)], "chain"),
    ],
    ids=["unknown-isin", "wrong-order", "two-successors", "two-predecessors"],
)
def test_invalid_override_links(
    store: DataStore, tmp_path: Path, links: list[tuple[str, str]], match: str
) -> None:
    with pytest.raises(SecurityMasterBuildError, match=match):
        build(store, tmp_path, links=links)


def test_conversion_into_a_listing_that_already_traded_cannot_be_linked(tmp_path: Path) -> None:
    pn, on = isin("PCAR", "ACNPR"), isin("PCAR")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="PCAR3", isin=on),
            detail(session="20240103", ticker="PCAR4", isin=pn),
            detail(session="20240104", ticker="PCAR3", isin=on),
        ],
        fca=False,
    )
    with pytest.raises(SecurityMasterBuildError, match="conversion"):
        build(store, tmp_path, links=[(pn, on)])


def test_rejection_of_a_pair_that_is_not_a_candidate(store: DataStore, tmp_path: Path) -> None:
    with pytest.raises(SecurityMasterBuildError, match="not a candidate"):
        build(store, tmp_path, rejections=[(PETR, TIMS)])


# --- Outputs ---


def test_master_columns_and_types(store: DataStore) -> None:
    build_security_master(store, LinkOverrides.none())
    table = SecurityMaster.load(store).table
    assert list(table.columns) == [
        "security_id",
        "ticker",
        "isin",
        "market_type",
        "first_session",
        "last_session",
        "sessions",
        "link_rule",
        "cnpj",
    ]
    assert table["market_type"].dtype == "int16"
    assert table["sessions"].dtype == "int64"
    assert len(table) == 7  # VVAR3, VIIA3, BHIA3, TIMP3, TIMS3, PETR4, PETR4F


def test_summary_counts_and_inputs(store: DataStore) -> None:
    summary = build_security_master(store, LinkOverrides.none())

    # Securities: the VVAR3 chain, TIMP3, TIMS3 (pending) and PETR4.
    assert (summary.spans, summary.isins, summary.securities) == (7, 6, 4)
    assert summary.inputs["ticker_links.yaml"] is None
    assert summary.inputs["bars_daily/manifest.json"] is not None
    assert summary.inputs["cvm/fca/fca_cia_aberta_2024.zip"] is not None

    document = json.loads(store.curated("b3", "security_master.json").read_text("utf-8"))
    assert document["schema_version"] == 1
    assert document["securities"] == summary.securities


def test_override_file_hash_is_recorded(store: DataStore, tmp_path: Path) -> None:
    build(store, tmp_path, links=[(TIMP, TIMS)])
    document = json.loads(store.curated("b3", "security_master.json").read_text("utf-8"))
    assert len(document["inputs"]["ticker_links.yaml"]) == 64


def test_rebuilding_reports_identifier_changes(store: DataStore, tmp_path: Path) -> None:
    first = build_security_master(store, LinkOverrides.none())
    assert first.identifier_changes == 0

    unchanged = build_security_master(store, LinkOverrides.none())
    assert unchanged.identifier_changes == 0

    folder = tmp_path / "overrides"
    write_links(folder, links=[(TIMP, TIMS)])
    changed = build_security_master(store, load_link_overrides(folder))
    assert changed.identifier_changes == 1  # TIMS3 now carries TIMP's identifier


def test_build_without_bars(tmp_path: Path) -> None:
    store = DataStore.initialize(tmp_path / "ibex-data")
    with pytest.raises(SecurityMasterNotBuiltError, match="ingest cotahist"):
        build_security_master(store, LinkOverrides.none())


# --- SecurityMaster queries ---


@pytest.fixture
def master(store: DataStore) -> SecurityMaster:
    build_security_master(store, LinkOverrides.none())
    return SecurityMaster.load(store)


def test_resolve(master: SecurityMaster) -> None:
    assert master.resolve("BHIA3", "2024-01-09") == VVAR
    assert master.resolve("VVAR3", "2024-01-02") == VVAR
    assert master.resolve("PETR4F", "2024-01-02") == PETR


def test_resolve_outside_the_ticker_span(master: SecurityMaster) -> None:
    with pytest.raises(SecurityNotFoundError, match="VVAR3"):
        master.resolve("VVAR3", "2024-01-09")
    with pytest.raises(LookupError):
        master.resolve("XXXX3", "2024-01-02")


def test_resolve_ticker_shared_by_two_securities_on_one_day(tmp_path: Path) -> None:
    store = market(
        tmp_path,
        [
            detail(ticker="ABCB10", isin=isin("ABCB", "R02PR")),
            detail(ticker="ABCB10", isin=isin("ABCB", "R03PR")),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    with pytest.raises(ValueError, match="ABCB10"):
        SecurityMaster.load(store).resolve("ABCB10", "2024-01-02")


def test_spans_in_time_order(master: SecurityMaster) -> None:
    spans = master.spans(VVAR)
    assert spans["ticker"].tolist() == ["VVAR3", "VIIA3", "BHIA3"]


def test_spans_of_unknown_security(master: SecurityMaster) -> None:
    with pytest.raises(SecurityNotFoundError):
        master.spans("BRXXXXACNOR1")


def test_ticker_at(master: SecurityMaster) -> None:
    assert master.ticker_at(VVAR, "2024-01-02") == "VVAR3"
    assert master.ticker_at(VVAR, "2024-01-05") == "VIIA3"
    assert master.ticker_at(PETR, "2024-01-02") == "PETR4"  # never the odd lot


def test_ticker_at_outside_any_listing(master: SecurityMaster) -> None:
    with pytest.raises(SecurityNotFoundError):
        master.ticker_at(TIMP, "2024-01-09")


def test_ticker_at_prefers_the_newer_ticker_when_two_overlap(tmp_path: Path) -> None:
    code = isin("AALC", "BDR00")
    store = market(
        tmp_path,
        [
            detail(session="20240102", ticker="AALC11B", isin=code),
            detail(session="20240103", ticker="AALC11B", isin=code),
            detail(session="20240103", ticker="AALC34", isin=code),
        ],
        fca=False,
    )
    build_security_master(store, LinkOverrides.none())
    assert SecurityMaster.load(store).ticker_at(code, "2024-01-03") == "AALC34"


def test_load_before_building(tmp_path: Path) -> None:
    store = DataStore.initialize(tmp_path / "ibex-data")
    with pytest.raises(SecurityMasterNotBuiltError, match="build security-master"):
        SecurityMaster.load(store)
    with pytest.raises(FileNotFoundError):
        read_candidates(store)


def test_master_table_requires_its_columns() -> None:
    with pytest.raises(ValueError, match="security_id"):
        SecurityMaster(pd.DataFrame({"ticker": ["PETR4"]}))


def test_missing_bars_manifest_is_recorded_as_none(store: DataStore) -> None:
    store.curated("b3", "bars_daily", "manifest.json").unlink()
    summary = build_security_master(store, LinkOverrides.none())
    assert summary.inputs["bars_daily/manifest.json"] is None
