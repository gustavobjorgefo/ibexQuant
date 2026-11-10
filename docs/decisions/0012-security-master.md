# 0012 — Security master

- Status: Proposed
- Date: 2026-11-09
- Phase: H2

## Context

H1 stores every quote under the ticker and ISIN B3 published that day. A
ticker is not an identity: the same company trades as `VVAR3`, then `VIIA3`,
then `BHIA3`, and one ticker (`ABCB10`) has carried twenty different
subscription receipts. Features computed per ticker restart at every rename,
and a universe built from tickers silently loses companies that changed name.

H2 gives every listing a permanent `security_id`, so that a renamed security
is one series. Research code keeps using tickers (guide decision D5): the
identifier is the key underneath.

### What the data shows

The exploration covered the round-lot market from 1994 to 2026 (4.1 million
rows, 6,091 tickers, 6,842 ISINs, 7,737 ticker–ISIN spans) and the CVM
registration forms (FCA) from 2010 to 2026.

| Finding | Evidence | Decision |
|---|---|---|
| Mechanical ticker changes keep the ISIN: the 1998-03-16 move from `ACE 3` to `ACES3` style, BDRs `AAPL11B` → `AAPL34` (2013), funds `ALMI11B` → `ALMI11` (2017) | 890 ISINs under more than one ticker | H2.4, rule 1 |
| Corporate renames change the ISIN, because the issuer code (ISIN characters 3–6) follows the ticker root: `BRVVAR…` → `BRVIIA…` → `BRBHIA…` | VVAR3/VIIA3/BHIA3, BVMF3/B3SA3, KROT3/COGN3, BTOW3/AMER3, TIMP3/TIMS3 | H2.4, rule 3 |
| Each of those renames is adjacent: the old ticker's last session is the session before the new ticker's first; the ISIN type and kind (characters 7–11, e.g. `ACNOR`) are equal; `distribution_number` restarts at 100 | Same six cases | H2.4, rule 3 |
| Before 1995-05-15 the ISIN field holds an exchange-internal code (`ACESACON`); the real ISIN replaces it under the same ticker | 8% of codes fail the ISIN check digit, all of this kind | H2.4, rule 2 |
| A ticker can be reused by unrelated securities: `ABCB10` and `ABCB2` carried 20 receipts and rights each | 932 tickers with more than one ISIN | H2.2 |
| Conversion into a security that already traded is not a rename: `PCAR3` existed since 2000 when `PCAR4` was converted into it (2020); `SUZB5` into `SUZB3` (2017) | Overlapping spans | H2.5 |
| The CNPJ in the FCA confirms five of the six renames as the same legal entity; `TIMP3` → `TIMS3` is an incorporation by another entity | FCA 2018–2026 | H2.3, H2.6 |
| The FCA fills `Codigo_Negociacao` only from reference year 2018; 2010–2017 have the field empty in every row | 0 of 4,676 rows before 2018 | H2.6 |
| FCA dates (`Data_Inicio_Negociacao`) describe the share class, not the ticker: `BHIA3` is shown as traded since 2018-11-26, the date of `VVAR3`'s segment migration | FCA vs. COTAHIST | H2.6 |
| The CVM portal refuses connections from outside Brazil | Cloudflare block page from Japan; works through a Brazilian address | H2.8 |
| A first build that let fund quotas and BDRs be rename candidates left 1,950 pairs pending: 799 fund quotas and 544 BDRs (listings that happened to end and start in the same week), 599 shares | First build on the real base | H2.4, rule 3 |
| Among the share pairs, those with one partner on each side and a continuous price (new first open / old last close between 0.8 and 1.25) are 44, and read as known renames: ITAU4 → ITUB4, PRGA3 → BRFS3, CPNE5 → BRKM5, KROT3 → COGN3 (missing from the 2018 FCA). The others show ratios such as 0.003 or 850 | Same build | H2.7, H2.15 |

## Decision

### What a security is

- **H2.1 — A security is the continuous economic position of a holder.**
  Two listings are the same security when a holder of the first became a
  holder of the second without trading: a rename, a change of ticker format,
  or a reorganization that exchanged one share for another. The legal entity
  may change (`TIMP3` → `TIMS3`); the exchange ratio, when it is not 1, is a
  corporate action recorded in H3, not a reason to break the series.
- **H2.2 — The ticker is a label, the ISIN is evidence, neither is the
  identity.** A ticker is resolved to a security only together with a date.
- **H2.3 — The CNPJ identifies the company, not the security.** `PCAR3` and
  `PCAR4` share a CNPJ and are two securities. A matching CNPJ can confirm a
  link between two adjacent listings of the same kind; it never creates one
  on its own.

### Building the links

- **H2.4 — Spans, then four rules.** A *span* is the run of sessions of
  one `(ticker, isin, market_type)` in `bars_daily`, both cash markets.
  Spans are joined into securities by:
  1. **Same ISIN** — spans that share an ISIN are one security
     (link rule `same_isin`). Covers the 1998 format change, BDR and fund
     ticker changes, and the odd-lot ticker (`PETR4F`) of each share.
  2. **Internal code to ISIN** — a pre-1995 code (not a valid ISIN) is
     linked to the ISIN its round-lot ticker carries next, when there is
     exactly one such ISIN (`internal_code`). No gap limit: illiquid shares
     went weeks without trading around the change. Codes with two possible
     successors (rights, receipts) are reported and left unlinked.
  3. **Adjacent renames** — span A ends, span B starts within 5 sessions
     after it, B never traded before A's last session, their ISINs have the
     same type and kind (characters 7–11), and the type is a share or a unit
     (`AC`, `UN`). Fund quotas and BDRs keep their ISIN when their ticker
     changes, so rule 1 covers them; as candidates they only produced
     coincidences. Such a pair is a
     *candidate*. It becomes a link (`cnpj`) when both listings map to the
     same CNPJ (H2.6); otherwise it waits for a decision in the overrides.
     Confirmed candidates are linked closest first (smallest gap), so a
     listing confirmed with two partners keeps the nearer one; two partners
     at the same distance are a conflict, and neither is linked.
  4. **Overrides** — a reviewed decision in `ticker_links.yaml` links or
     rejects a candidate, or links any two listings the rules did not pair
     (`override`). Overrides are applied first and win over rules 2 and 3.
- **H2.5 — A security is a chain, never a tree.** Each span has at most one
  predecessor and one successor across different ISINs. A conversion into a
  security that already traded (`PCAR4` → `PCAR3`) or a merger of two
  securities into one is never a link: it is a corporate action of H3
  (holders of the old security receive the new one at a ratio). An override
  that would create a merge or a cycle is an error.
- **H2.6 — CNPJ from the FCA, extended through the issuer code.** The FCA
  table of securities gives `(reference year, ticker, CNPJ)` from 2018. A
  ticker is mapped to the ISIN it carried in that year (from `bars_daily`),
  and the ISIN's issuer code (characters 3–6) to the CNPJ. The issuer code
  then confirms candidates in years before 2018. An issuer code that maps to
  more than one CNPJ is not used, and is reported. FCA dates are never used:
  ticker dates come from COTAHIST.
- **H2.7 — Unreviewed candidates stay unlinked.** A wrong link merges two
  companies' prices; a missing link only restarts a series. Until reviewed,
  a candidate is reported, and the two spans remain two securities. A
  candidate that reads as a rename — among the undecided pairs, neither
  listing has another partner, and the price continues (new first open /
  old last close between 0.8 and 1.25) — is marked `likely` and listed
  first for review; it is still not linked without a decision. A rename
  made together with a split or an exchange ratio jumps by that factor
  and stays `pending`; when the ratio is within 5% of a common factor (2,
  3, 4, 5, 8, 10, 20, 25, 50, 100, 1,000 or their inverses), the candidate
  carries it as `implied_factor`, a hint for review and never a decision —
  16 of the 135 one-to-one share pairs of the first build had one, some
  real (TIET3 → AESB3, 5; BELG3 → ARCE3, 20), some coincidences. H3
  re-evaluates these candidates against the corporate actions it records:
  a ratio explained by an event at the change reads as a continuing price.

### Sources and overrides

- **H2.8 — CVM files, downloaded by hand.** `raw/cvm/fca/fca_cia_aberta_YYYY.zip`,
  as downloaded from the CVM open data portal. The portal accepts only
  Brazilian addresses, so downloads are manual (browser or a Brazilian
  connection). The current year's file is replaced on refresh, as in 0011
  (H1.2); the build summary records the SHA-256 of every file used. The
  company register (`cad_cia_aberta.csv`) carries no ticker and is not used.
- **H2.9 — Overrides location (resolves 0010, H0.10).** The overrides root
  is resolved like the data root (H0.3): explicit argument, environment
  variable `IBEXQUANT_OVERRIDES_ROOT`, or `[overrides].root` in
  `~/.ibexquant/config.toml`. Building without overrides requires saying so
  (`--no-overrides`), so that a missing setting never yields a silently
  different master.
- **H2.10 — `ticker_links.yaml`.** Links and rejections are written by ISIN
  pair, which is unambiguous where tickers are not:

  ```yaml
  version: 1
  links:
    - from: BRTIMPACNOR1   # last ISIN of the old listing
      to: BRTIMSACNOR5     # first ISIN of the new listing
      note: TIM Participações incorporated by TIM S.A., 1:1 exchange
  rejections:
    - from: BRXXXXACNOR1
      to: BRYYYYACNOR2
      note: unrelated company that listed the next day
  ```

  Both ISINs must exist in the spans, `to` must start after `from` ends, and
  every entry needs a `note`. An entry that breaks a rule is an error, never
  skipped.

### Identifiers

- **H2.11 — The `security_id` is the security's founding ISIN.** It is the
  ISIN (or pre-1995 code) of the security's earliest span. It is derived
  from `raw/` plus the overrides, so the master can be rebuilt anywhere and
  always gives the same identifiers (0010, H0.5); no hidden registry is
  kept. It changes only when a new link adds an earlier span to a security;
  every build reports those changes against the previous master.

### Output and access

- **H2.12 — Tables.** In `curated/b3/`:
  - `security_master.parquet`: one row per span — `security_id`, `ticker`,
    `isin`, `market_type`, `first_session`, `last_session`, `sessions`,
    `link_rule` (`founding`, `same_isin`, `internal_code`, `cnpj`,
    `override`), `cnpj` (nullable);
  - `link_candidates.parquet`: every rule-3 candidate with its status, the
    prices around the change (old last close, new first open, their ratio)
    and the `implied_factor` (H2.7). Statuses: `linked_by_cnpj`, `linked_by_override`, `rejected`,
    `likely` (reads as a rename, waits for review), `pending` (waits for
    review, low plausibility), `conflict` (two confirmed partners at the
    same distance), `superseded` (one side is already linked to another
    listing — no review needed);
  - `security_master.json`: counts, pending candidates, identifier changes,
    and the SHA-256 of every input (the bars manifest, each FCA ZIP, the
    overrides file).
- **H2.13 — `SecurityMaster`.** Loads the tables and answers:
  - `resolve(ticker, as_of)` → the `security_id` of the span carrying that
    ticker on that date; outside every span of the ticker, an error;
  - `spans(security_id)` → its tickers and ISINs over time;
  - `ticker_at(security_id, as_of)` → the round-lot ticker on that date.
- **H2.14 — `B3Provider` follows renames.** `B3Provider(store,
  follow_renames=True)`, the new default, resolves each requested ticker to
  its security on the last session up to `end` on which it traded, and
  returns the whole security's round-lot bars in `[start, end]`, labeled
  with the requested ticker. `get_bars("BHIA3", "2015-01-01", "2024-12-31")`
  therefore returns VVAR3, VIIA3 and BHIA3 as one `BHIA3` series. Two
  requested tickers that resolve to the same security raise `ValueError`.
  When a security trades under two round-lot tickers on one session (a
  ticker change that kept the ISIN), the row with the larger volume
  represents the session.
  `follow_renames=False` keeps the H1 behavior. Following renames requires
  a built master; without one, an error says how to build it.
- **H2.15 — Command line.**
  - `ibexquant data build security-master [--no-overrides]` builds the
    tables and prints the summary;
  - `ibexquant data review links [--status STATUS ...] [--yaml]` prints the
    candidates with the prices around each change; with `--yaml`, it prints
    them as `links:` entries of `ticker_links.yaml`, to be checked, trimmed
    and pasted (`--status likely --yaml` is the usual review).

### Dependencies

- **H2.16 — `pyyaml`** joins the `b3` extra, as announced in 0010 (H0.9).

## Alternatives considered

- **Same legal entity (CNPJ) as the identity.** Rejected by decision
  (H2.1): `TIMP3` → `TIMS3` would end one series and start another, although
  every holder simply kept a position. The two choices differ only in
  reorganizations across entities.
- **ISIN as the identity.** Rejected: the six best-known renames all changed
  ISIN (H2.4, rule 3).
- **Automatic links for every adjacent candidate.** Rejected (H2.7): an
  unrelated listing that starts the day after a delisting would merge two
  companies. The CNPJ or a human confirms.
- **An append-only registry of opaque identifiers (`SEC-000001`).**
  Rejected (H2.11): it is state that cannot be rebuilt from `raw/`, needs
  its own versioning, and buys stability only in the rare case of a link
  that adds earlier history — a case the build reports anyway.
- **Linking conversions into existing securities.** Rejected (H2.5): it
  would merge two price histories that overlapped in time.
- **Using FCA dates.** Rejected (H2.6): they describe the share class, and
  contradict COTAHIST for every rename observed.

## Consequences

- Research can ask for `BHIA3` from 2015 and get one series; a universe of
  tickers taken from an index composition (H5) resolves to securities on
  the composition date.
- The first build will report pending candidates. Their number decides how
  much review the base needs; companies that renamed and delisted before
  2018 can only be linked by review.
- Corporate actions between securities (conversions, mergers, exchange
  ratios) are left to H3, which reads the master.
- The provider keeps the shared `MarketDataProvider` contract: tickers in,
  tickers out.
