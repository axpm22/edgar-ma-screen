# Data Inventory

Everything here is free and public. No CRSP, Compustat, SDC, or any licensed
source. Total on disk: ~9.5 GB, of which 8.5 GB is the raw download cache.

---

## The spine

**`data/deal.duckdb`** (447 MB) — the main warehouse.

| Table | Rows | What it is |
|---|---|---|
| `panel` | 4,123,449 | company × ISO week. One row per listed company per week |
| `universe` | 15,325 | point-in-time listing spans (`listed`, `delisted`) |
| `deals` | 2,456 | DEFM14A merger proxies, episode-collapsed |
| `fundamentals` | 2,041,665 | XBRL facts (superseded by `fund2.duckdb`) |
| `insider_trans` | 2,949,427 | Form 4 transactions, with 10b5-1 flag post-2023 |
| `insider_value` | 278,878 | dollar-weighted insider activity |
| `company_sic` | 12,839 | CIK → SIC industry code |
| `fts_events` | 16,813 | EDGAR full-text hits on strategic-alternatives phrases |
| `xwalk_name` | 15,181 | normalised company name → CIK |
| `xwalk_domain` | 3,588 | CIK → website (from Wikidata) |

**Universe construction is the survivorship fix.** Membership derives from
periodic filing activity, not index constituents — a company is in the panel
for exactly the weeks it filed 10-Ks or 10-Qs. Acquired and delisted companies
stay in for the weeks they existed, which is where the positive labels live.

---

## Signal sources

| Database | Rows | Contents |
|---|---|---|
| **`fund2.duckdb`** (212 MB) | 3,643,395 | XBRL facts, **25 tags** — balance sheet, income statement, cash flow, PP&E, preferred stock. Supersedes `deal.fundamentals` |
| **`forms2.duckdb`** (87 MB) | 1,399,656 | Form events, **8 families**: `sc13d`, `sc13g`, `form8k`, `def14a`, `s4`, `late`, `shelf`, `raise`. Supersedes `forms.duckdb` |
| **`items.duckdb`** (30 MB) | 729,740 | 8-K item codes (1.01 material agreement, 5.02 officer change, 4.01 auditor change, …). Item 3.01 excluded — delisting notice, leaks the outcome |
| **`activist.duckdb`** (3 MB) | 43,709 | 13D filer identity; repeat filers across many targets are professional activists |
| **`float.duckdb`** (4 MB) | 58,627 | `EntityPublicFloat` from XBRL frames. **Survivorship-free market cap** — comes from the company's own 10-K cover, so acquired companies retain history |
| **`ct.duckdb`** (7 MB) | 120,144 | Certificate Transparency — novel hostnames per week. Only 2,448 of 15,325 companies resolve to a domain |
| **`tender.duckdb`** (1 MB) | 616 | Tender-offer targets, resolved from SEC-HEADER `SUBJECT COMPANY`. Held out as an independent validation label |

---

## Feature matrices

| File | Shape | Notes |
|---|---|---|
| `features.parquet` | 4,123,449 × 86 | Target model. 72 active features; industry-relative columns present but excluded (measured −1.6pp) |
| `buyer_features.parquet` | 4,123,449 × 93 | Buyer model. Adds shelf/raise rolls and capacity measures |
| `lm_scores.parquet` | 3,101 × 13 | Loughran-McDonald sentiment on 8-K documents (null result) |

### Feature families

- **Fundamentals (11)** — size, leverage, liquidity, margins, goodwill, intangibles
- **Literature (11)** — Palepu's growth-resource mismatch, Ambrose & Megginson's tangible assets and blank-cheque preferred, ROA, FCF, dividend payout
- **Market (3)** — public float level, valuation, growth
- **Insider (9)** — discretionary trade counts, blackout flag, dollar-weighted flows
- **Form counts (7)** — 13D, 13G, 8-K, S-4, DEF 14A, late filings, new-activist flag
- **8-K items (13)** — decomposed event types
- **Strategic-alternatives text (4)** — "reviewing strategic alternatives", "letter of intent", "unsolicited"
- **Activist + peer (4)** — activist reach and recency, peer deals in the trailing 13 weeks, sector deal intensity
- **Per-company z-scores (5)** — counts normalised against the firm's own trailing 104-week baseline
- **Deltas (3)** — acceleration rather than level
- **Buyer-only (7)** — shelf registrations, prospectus supplements, dry powder, debt headroom

---

## Labels

Three distinct label sets, and the distinction matters more than anything else
in this project.

**1. Raw proxy filers (2,456).** Every DEFM14A. **Contaminated** — a merger
proxy is filed by whoever's shareholders vote, so buyers in stock deals file
one too. Teledyne after acquiring FLIR, Newmont after Newcrest, Dow after
DowDuPont are all in here as if they were targets.

**2. Verified targets (1,664).** Proxy filers whose periodic filings *stopped*
within 270 days. A target disappears; an acquirer does not. 581 survivors
removed, 200 too recent to classify. **This is the correct target label.**

**3. Buyers (S-4 filers).** An S-4 registers securities issued to pay for an
acquisition, so the filer is the buyer. 3.00% weekly label rate.

---

## Raw cache (8.5 GB, gitignored)

| Path | Size | Contents |
|---|---|---|
| `data/raw/sec` | 6.8 GB | EDGAR indexes, financial-statement ZIPs, insider ZIPs, submissions JSON, 8-K documents |
| `data/raw/sec_hdr` | 1.7 GB | Filing-header range requests (tender-offer subject resolution) |
| `data/raw/lm` | 6 MB | Loughran-McDonald master dictionary |
| `data/raw/wikidata`, `data/raw/uspto` | <1 MB | CIK→domain crosswalk; abandoned USPTO attempt |

Content-addressed by URL hash, so re-runs make zero requests and a parser
change never triggers a re-download.

---

## Known gaps

- **No prices, returns, or volume.** No free source retains price history for
  delisted companies, which is exactly where the positives are. Blocks
  momentum, illiquidity, and the abnormal-volume clock.
- **CT covers 16% of companies** — gated by domain resolution via Wikidata.
- **USPTO abandoned** — the Open Data Portal now requires ID.me (government ID
  and SSN).
- **200 deals unclassifiable** — too close to the panel edge to see whether the
  company stopped filing.
- **Rumour dates unmeasured** — labels sit on the proxy date, which lands
  40–70 days after the real announcement.
