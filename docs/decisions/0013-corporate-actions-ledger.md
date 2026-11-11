# 0013 — Corporate actions ledger

- Status: Proposed
- Date: 2026-11-10
- Phase: H3

## Context

H1 stores raw prices and H2 joins listings into securities. A raw series
still jumps at every dividend, split, bonus issue and exchange: `MGLU3`
opens at a quarter of its previous close on 2020-10-14, and a backtest that
reads that as a −75% day is wrong. H4 adjusts prices, and it needs, for each
security, every ex-date with what happened on it. H3 records that ledger.

Two goals shape the sources. The ledger must rest on official data, and the
third-party data used by the old pipeline (StatusInvest, yfinance) must be
avoided wherever official data exists. It turned out to exist everywhere the
ledger needs it (B3 back to 1995, the CVM forms for stock events), so the
ledger uses no third-party data at all.

### What the data shows

The exploration covered the round-lot shares and units of `bars_daily`
(1995–2026, 26,879 changes of distribution number over AC, UN and CD
ISINs), 65 daily *Earnings credited* files (2026-06-29 to 2026-11-06), the
B3 listed-company data of four issuers (WEGE, ITSA, MGLU, PETR), two
StatusInvest snapshots and the yfinance splits of the old IBrX work.

| Finding | Evidence | Decision |
|---|---|---|
| A security's `distribution_number` changes exactly on the first session it trades without a right. Every PETR3 cash event in B3's history (56 data-com dates, 2007–2026) is the last session before a change | PETR3: 56 of 56 | H3.2 |
| Stock events land on the same changes, with the expected price ratio: WEGE3 split 2021 (open / previous close 0.515), MGLU3 split 2020 (0.255), MGLU3 reverse split 2024 (9.72), PETR3/PETR4 split 2008 (0.504), PETR reverse split 2000 (101) | 12 of 12 events since 2000 | H3.2, H3.5 |
| An illiquid share can skip several distributions between two trades (`ODER4` from 140 to 146) | 1,143 changes of more than one number | H3.3 |
| B3's listed-company service gives, per issuer, the full history of stock events (back to 1994), recent cash events and subscriptions; a second call gives the paginated cash history (PETROBRAS: 343 records, back to 2007) with the data-com, the value and the closing price on the data-com | WEGE, ITSA, MGLU, PETR | H3.4, H3.10 |
| The stock-event `factor` is a percentage for splits and bonus issues (WEGE3 split `100` = ×2, MGLU3 split `300` = ×4, WEGE3 bonus `30` = ×1.3, ITSA bonus `2` = ×1.02) and a multiplier for reverse splits (MGLU3 `0.1` = 10 to 1) | Factors against price ratios | H3.5 |
| B3 lists one cash event twice when it is paid in two installments (PETR 2025-04-16, 0.35477261 twice) | Cash history | H3.8 |
| *Earnings credited* carries ISIN and distribution number (`BRPETRACNOR9 213`), no data-com; its value equals B3's rate (JCP gross). Its layout changed between downloads: English header, decimal point and MM/DD dates, then Portuguese header, decimal comma and DD/MM | 204 of 273 share rows fall inside a change; the rest belong to issuers that did not trade | H3.3, H3.4 |
| StatusInvest gives only cash events, relabels history with today's ticker (`COGN3` from 2009, `MOTV3` from 2008), changes history between downloads (184 vs. 65 rows differ between two snapshots) and is nearly empty before 2007 | Two snapshots | H3.4 |
| yfinance covers only the tickers it was asked for, mixes bonuses with splits and duplicates events (`AMER3`: four 100-to-1 reverse splits in five weeks of 2024, one in reality) | 272 rows, 85 tickers | H3.4 |
| Changes with a large price jump (open / previous close below 0.7 or above 1.4) that neither StatusInvest nor yfinance explains: 77 in 2000, 13 to 30 a year recently | Exploration | H3.6 |
| The first full download (880 issuer codes, 2026-11-10) answered for 781, including long-delisted issuers (SADIA, BANESPA, ARACRUZ, UNIBANCO); roots of renamed issuers answer empty, and their history sits under today's code (`BHIA` from 1997, `MOTV` from 2003). A root can be reused (`TRPL` now answers for a real-estate fund) | Snapshot 2026-11-10 | H3.3, H3.4 |
| The cash history holds 31,133 rows from 1995 for 610 codes, with share class (`ON`, `PN`, `PNA`…, `UNT`) and no ISIN | Snapshot | H3.13 |
| Rows quoted per 1,000 shares (`quotedPerShares` 1000, most events up to 2007) give value and closing price per 1,000 shares: the closing price is 1,000 times COTAHIST's, and the published yield matches. StatusInvest copied these values as per share | 1,926 rows against COTAHIST; 169 against StatusInvest | H3.5 |
| Every issuer whose trading name contains `/` (`KLABIN S/A`, `AMBEV S/A`, `SADIA S/A`) returned an empty cash history | 16 issuers | H3.10 |
| Stock events have nine labels; 247 rows deliver another asset (`assetIssued` ≠ `isinCode`): spin-offs (`MRVE3` holders received 7.215% of a `LOGG3`, `BRAP` 33.24% of a `VALE3`), mergers (`INCORPORACAO`, `LIGH` → `LIGT`) and bonuses in another class. A row whose asset is the ISIN itself with factor 100 only says the holder keeps the share | 1,032 rows, 1978–2026 | H3.5, H3.7 |
| The first build left 562 unexplained jumps and hid small bonus issues inside ordinary moves: BBDC4 opened at about 0.91 of its previous close every year from 2010 to 2021, ITUB4 in 2013–2016, ITSA4 in 2012–2016, with no event in any source. B3's stock list also lacks PETR's 2005 split and VALE's 2004 and 2006 splits; the inferred splits found them (31 of 34 found in yfinance on the same date) | First real build | H3.14 |
| The CVM reference forms (FRE, item 17.3) list splits, reverse splits and bonus issues with CNPJ, approval date and share counts before and after, per class; the table exists only in forms under ICVM 480 (ZIPs up to 2022). It holds every missing Bradesco, Itaú and Itaúsa bonus; counts give approximate ratios (Bradesco 2021: 1.0982 common, 1.0932 preferred) | FRE 2016–2022: 510 events | H3.14 |
| Placing FRE events on the first ex-session after approval whose price ratio agrees: 263 of 582 company-ISIN pairs placed (123 confirming B3, 75 explaining unexplained sessions, 36 confirming inferred splits, 29 explaining jumps); the rest are events before listing or classes that never traded | Simulation on the real base | H3.14 |
| Holders of a delisted share received another existing share at a ratio: `LAME4` → `AMER3` (0.188964), `SOMA3` → `AZZA3`, `PCAR4` → `PCAR3`, `VIVT4` → `VIVT3`, `CESP6` → `AURE3`; the H2 pairs of the old IBrX notes that are true renames are all linked | Old IBrX notes against the security master | H3.7 |

## Decision

### What the ledger records

- **H3.1 — An event is something that happened to holders of one ISIN on
  one ex-date.** The `ex_date` is the first B3 session on which the security
  traded, or could have traded, without the right; the `last_cum_date`
  (data-com) is the session before it. Events are kept per ISIN and carry
  the `security_id` of H2, so a renamed security has one event history.
  Prices in the ledger are in BRL per share of that ISIN, as of the event.

### The skeleton

- **H3.2 — Every change of distribution number is an ex-session.** The
  build lists, per round-lot ISIN of type AC, UN or CD, the sessions on
  which `distribution_number` differs from the previous traded session
  (`ex_sessions`), with the prices around the change. This list comes from
  COTAHIST alone, covers delisted securities, and is the reference every
  source is checked against: an event explains an ex-session, and an
  ex-session that no event explains is reported (H3.6).
- **H3.3 — Events are matched to ex-sessions by interval.** An event with a
  data-com `d` on ISIN `i` matches the change of `i` whose previous traded
  session is on or before `d` and whose session is after `d`. An event that
  carries a distribution number `n` instead of a date (*Earnings credited*)
  matches the change from `a` to `b` with `a ≤ n < b`. An event whose ISIN
  traded on both sides of its data-com under one distribution number
  contradicts COTAHIST and goes to review (`no_ex_session`); an event whose
  ISIN did not trade around the data-com is kept, unmatched.

### Sources

- **H3.4 — Sources, in priority order.**
  1. `manual` — reviewed entries in `corporate_actions.yaml` (H3.11).
  2. `b3_listed` — B3's listed-company service: stock events, subscriptions
     and recent cash events (`GetListedSupplementCompany`), and the cash
     history from 1995 (`GetListedCashDividends`), one JSON snapshot per
     issuer and download (H3.10). It covers delisted issuers too.
  3. `cvm_fre` — stock events of the CVM reference forms (H3.14).
  4. `b3_credited` — the daily *Earnings credited* files; cash events from
     2026-06 on, and the check of values credited.
  5. `inferred` — stock events read from an ex-session's price ratio (H3.6).

  StatusInvest and yfinance are not sources. A frozen StatusInvest
  bootstrap was planned as a fifth source for issuers B3 does not describe;
  the first full B3 download covered delisted issuers back to 1995, and the
  gaps it left (roots that answer empty) are closed by fetching the old
  trading name or by `corporate_actions.yaml`, so the bootstrap was dropped
  before release. Every source keeps its raw files in `raw/`, as
  downloaded; the build never fetches anything.

### Event types and amounts

- **H3.5 — Types and the share ratio.** `event_type` is one of:
  `CASH_DIVIDEND`, `INTEREST_ON_CAPITAL` (JCP, gross), `INCOME`
  (*rendimento*, the monetary update B3 lists beside a dividend),
  `RETURN_OF_CAPITAL`, `SPLIT`, `REVERSE_SPLIT`, `STOCK_BONUS`,
  `SPIN_OFF`, `SUBSCRIPTION`, `EXCHANGE` (H3.7), `CONVERSION` (H3.7),
  `REDEMPTION` and `OTHER`. Cash events carry `cash_amount` (BRL per
  share); a value quoted per 1,000 shares is divided by 1,000. Stock events
  carry `share_ratio`, the number of shares held after the event per share
  held before. B3's factor `f` is a percentage, except for `GRUPAMENTO`,
  where it is the multiplier itself:
  - same asset: `DESDOBRAMENTO` → `SPLIT`, `BONIFICACAO` → `STOCK_BONUS`,
    both `1 + f / 100`; `GRUPAMENTO` → `REVERSE_SPLIT`, `f`; `CIS RED CAP
    QTD` and `REST CAP C/ RED` (fewer shares after a capital reduction) →
    `REVERSE_SPLIT`, `f / 100`; `CIS RED CAP`, `INCORPORACAO` and
    `REST CAP ACOES` with `f = 100` are not events (the holder keeps the
    share), with another `f` an error; `RESG TOTAL RV` → `REDEMPTION`;
  - another asset (`target_isin`): `CIS RED CAP`, `CIS RED CAP QTD`,
    `REST CAP ACOES` and `BONIFICACAO` → `SPIN_OFF` with `f / 100` units of
    the target per share, whose value H4 takes from the target's price;
    `INCORPORACAO` → `CONVERSION`, `f / 100`; `GRUPAMENTO` → `CONVERSION`,
    `f`. Subscriptions carry the
  percentage and the subscription price as published; whether they adjust
  prices is decided in H4. Fraction auctions (*leilão de frações*) are not
  events: they pay for rounding, not for a right. Each B3 label maps to one
  type in a table in the code; an unknown label is an error, so a new label
  is read once and mapped.

### Explaining the skeleton

- **H3.6 — Each ex-session gets a status.** Let `r` be the open over the
  previous traded close.
  - `explained` — at least one event matches it (H3.3);
  - `inferred` — no event matches, `r` is outside [0.7, 1.4], and `r` is
    within 5% of a common share ratio (2, 3, 4, 5, 8, 10, 20, 25, 50, 100,
    1,000 or their inverses; H2.7). The build records a `SPLIT` or
    `REVERSE_SPLIT` with `source = inferred`, status `inferred`, and uses
    it. A missing split turns a security's return into a −50% day; a
    distribution change at a common ratio is strong evidence, and the
    review lists every inferred event, so a wrong one is rejected in the
    overrides;
  - `unexplained_jump` — no event matches, `r` is outside [0.7, 1.4], and
    no common ratio fits: waits for review;
  - `unexplained` — no event matches and `r` is within [0.7, 1.4]: a small
    event the sources do not describe (a dividend of a delisted issuer, a
    subscription). Not adjusted, counted per year in the summary;
  - `accepted` — reviewed and marked in the overrides as needing no event.

### Between securities

- **H3.7 — Exchanges and conversions.** Two events relate two ISINs:
  - `EXCHANGE` — the old ISIN's holders received the new ISIN at a ratio,
    and H2 linked them into one security (a rename with a ratio, H2.1). It
    sits on the new ISIN's first session and carries `share_ratio`.
  - `CONVERSION` — the holders received a different security at a ratio
    (`LAME4` → `AMER3`, `PCAR4` → `PCAR3`). It sits on the old security's
    last session plus one and carries `share_ratio` and `target_isin`. A
    conversion ends the old series; it does not adjust the target's prices.
    Its use is to carry a position from one security to another in a
    backtest.
  B3 publishes conversions as stock events (H3.5); others come from the
  overrides. The build proposes them: every H2 candidate
  that is `pending` or `superseded`, with its price ratio and
  `implied_factor`, is listed for review as a possible exchange or
  conversion.

### Building the ledger

- **H3.8 — Merging sources.** Within one source, rows are distinct events
  (two installments of one dividend are two events). Across sources, events
  of one ISIN, ex-date and type are matched one to one by amount, within
  0.5% for cash and 1% for share ratios; the higher-priority source wins and
  the others are kept as confirmations. Amounts that differ beyond the
  tolerance are written to the conflicts table with both values, and the
  higher-priority one is used. `b3_listed` snapshots are merged by keeping
  every event ever seen; an event whose amount changed between snapshots
  keeps the latest amount, and the change is reported.
- **H3.9 — Tables.** In `curated/b3/`:
  - `corporate_actions.parquet`: one row per event — `event_id` (hash of
    ISIN, ex-date, type, amount and source row), `security_id`, `isin`,
    `ex_date`, `last_cum_date`, `event_type`, `cash_amount`, `share_ratio`,
    `subscription_price`, `target_isin`, `payment_date`, `approved_on`,
    `source`, `origin`, `label`, `status` (`confirmed`, `inferred`,
    `rejected`, or `review` when the ISIN traded on both sides of the
    data-com without a change, H3.3), `placement`, `ex_session`,
    `confirmed_by` (other sources that agree);
  - `ex_sessions.parquet`: the skeleton with its status (H3.6) and the
    matching `event_id`s;
  - `corporate_action_conflicts.parquet`;
  - `corporate_action_unresolved.parquet`: source rows that could not be
    placed on an ISIN (H3.13), with the reason;
  - `corporate_actions.json`: counts per source, type, status and year,
    unexplained ex-sessions per year, the inputs used and the SHA-256 of
    the override file.

### Raw inputs

- **H3.10 — Layout.**
  - `raw/b3/listed_companies/YYYY-MM-DD/supplement/<CODE>.json` and
    `raw/b3/listed_companies/YYYY-MM-DD/cash_dividends/<CODE>/page-NNN.json`:
    one folder per download, the responses as received. `<CODE>` is the
    issuer code (the ticker root, `PETR`). A download covers every root seen
    in COTAHIST since 2000, so the snapshot itself records which issuers B3
    still describes; an empty answer is saved too.
  - `raw/b3/earnings_credited/YYYY/YYYY-MM-DD.csv`, one file per day as
    published.
  The downloads are made by the B3 data collector (outside this
  repository); IbexQuant reads the files.

### Placing B3's cash history

- **H3.13 — From share class to ISIN.** A cash-history row names the issuer
  code and the share class. The class maps to ISIN characters 7–11 (`ON` →
  `ACNOR`, `PN` → `ACNPR`, `PNA` → `ACNPA`, … `UNT` → `CDAM`); the ISINs
  in `bars_daily` that start with `BR`, the code and those characters are the
  class's ISINs today, and the security master gives the ISIN of that
  security in force on the row's data-com (renamed issuers report their old
  history under the new code). A row that resolves to no ISIN, or to more
  than one, is reported and not used.

### The CVM reference forms

- **H3.14 — FRE stock events.** `raw/cvm/fre/fre_cia_aberta_YYYY.zip`, placed
  by hand like the FCA. Each event of `capital_social_desdobramento` (and
  its `_classe_acao` companion) gives, per share class, the ratio of shares
  after to shares before; a ratio within 0.5% of a whole percentage or of a
  common factor is read as that value. The class maps to the company's
  round-lot ISINs through the CNPJ of the security master (`ON` → `ACNOR`,
  `PNA` → `ACNPA`, `PN` → the preferred ISINs no class row names; units
  follow when common and preferred ratios agree). The event goes ex on the
  first ex-session of the ISIN within 180 days of approval whose price
  ratio agrees: `|log(open_ratio × share_ratio)|` at most 0.1 and at most
  half of `|log(share_ratio)|`, so a 10% bonus is not taken for an ordinary
  move. Events with no such session are reported as unresolved and not
  used. The forms after 2022 (Resolution CVM 80) no longer carry the table;
  B3's weekly snapshots cover events from then on.

### Overrides and commands

- **H3.11 — `corporate_actions.yaml`**, beside `ticker_links.yaml` in the
  overrides root (H2.9), optional: `events` (add an event of any type,
  exchanges and conversions (H3.7) included, by ISIN, data-com, type,
  amount and, for spin-offs, exchanges and conversions, `target_isin`),
  `rejections` (mark a source event or an inferred one as wrong, by ISIN,
  ex-date, type and source), `accepted` (ex-sessions that need no event).
  Every entry has a note. A rejection or accepted ex-session that matches
  nothing stops the build, so the file never carries stale decisions.
- **H3.12 — Command line.**
  - `ibexquant data build corporate-actions [--no-overrides]` builds the
    tables and prints the summary;
  - `ibexquant data review events [--status STATUS ...] [--yaml]` prints
    ex-sessions and events awaiting review with the prices around them; with
    `--yaml`, as entries of `corporate_actions.yaml`.

## Alternatives considered

- **Ex-dates from the sources' dates alone.** Rejected (H3.2): B3 dates are
  data-com, StatusInvest relabels tickers, *Earnings credited* has no date
  at all, and none of them covers every security. The distribution number
  is in every COTAHIST row since 1995.
- **StatusInvest and yfinance as sources.** Rejected (H3.4): they revise
  history, cover only what they were asked, and duplicate events. A frozen
  StatusInvest bootstrap limited to issuers without B3 history was built
  and then removed: B3 and the CVM forms left nothing it was needed for.
- **Inferred splits only after review.** Rejected (H3.6): unlike a wrong
  link in H2, which merges two companies, a missing split corrupts every
  return across it, and the review list is the same either way.
- **Price-based cash amounts for delisted issuers (`1 − open / previous
  close`).** Rejected: the overnight move is of the same size as most
  dividends, so the amount would be noise. These ex-sessions stay
  `unexplained` and are counted.
- **Adjusting the target of a conversion.** Rejected (H3.7, H2.5): the
  target traded before the conversion, at its own prices.

## Consequences

- H4 reads one table: per security, the events with `ex_date`,
  `cash_amount` and `share_ratio`.
- The ledger is as complete as B3's service for listed issuers, as
  complete as COTAHIST's distribution numbers for the dates of every
  security, and incomplete in amounts for issuers delisted before the first
  B3 download whose root B3 no longer answers for; those are fetched under
  their old trading name or entered in `corporate_actions.yaml`.
- Weekly B3 downloads keep the ledger current; a missed week loses nothing,
  because the cash history is cumulative.
- Units are ISIN type `CD`, not `UN` (`BRSANBCDAM13`); H2's adjacent-rename
  rule (H2.4, rule 3) does not consider them. The build reports CD
  ex-sessions that start a new ISIN, so a missed unit rename shows up here;
  a change to H2 is a separate record.
