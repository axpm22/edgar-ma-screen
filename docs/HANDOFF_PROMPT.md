# Handoff prompt

Paste everything below the line into a fresh Claude Code session in this repo.

---

I have a working M&A prediction pipeline built entirely from free SEC filings.
Read `docs/DATA.md` first — it inventories every database, table and feature
family. Then read `README.md` for current results.

## Where things stand

Two models, cross-validated across **eleven** test years (2015–2025) with two
seeds each, on a panel running from 2012:

| Model | Precision @25/wk | Lift | Label |
|---|---|---|---|
| Target | 12.28% | 6.91× | company stopped filing within 270d of its merger proxy |
| Buyer | 31.72% | 12.21× | filed an S-4 within 12 months |

Both operating-companies-only (SPACs excluded) and buyer excludes
self-referential features. **Compare years by lift, not precision** — deal base
rates halved across the panel, which moves precision without moving skill.

Both exclude SPACs. Both numbers are *after* removing self-referential
features. Earlier, higher numbers in this repo's history are retracted.

## Hard-won rules — violate these and the result is worthless

1. **Never evaluate with global top-k over company-weeks.** One company
   occupies the top 100 eighty-four times. Use `screen.weekly_precision`,
   which ranks within each week.
2. **Never split randomly over rows.** A company contributes ~550 correlated
   rows and one deal spans its whole label window. Use `splits.grouped`
   (by company) or `splits.by_time`.
3. **Never use the test period for early stopping.** Three-way temporal split:
   train, validation, then a test set touched once.
4. **Right-censoring is real.** With a 52-week label and a panel ending
   2026-07-27, rows after 2025-07-28 cannot have observed outcomes. Truncate
   the test window or you understate performance by ~8pp.
5. **Watch for self-referential features.** `s4_52w` predicting a future-S-4
   label is autocorrelation. `goodwill_to_assets` accumulates *from* past
   deals. Anything that encodes the label's own history must be reported
   separately.
6. **Cluster standard errors by company.** Naive errors measured 4× too small
   on this panel.
7. **±2pp is the noise band** at two seeds. Do not narrate a 1pp difference as
   a finding.
8. **Compute discipline:** 2 threads (already pinned in `deal/__init__.py`),
   one model per process, `gc.collect()` between fits. Multi-fit single
   processes have been OOM-killed twice. Tell me the fit count and expected
   runtime before launching anything over ~5 minutes.

## What I want

### 1. Stress-test what exists

Attack both models. `scripts/audit.py` runs data-integrity checks and
`scripts/stress_suite.py` runs permutation, embargo and seed tests — extend
them rather than starting over. Specifically:

- Run the permutation test on the **buyer** model. It has never had one.
- Re-run the clustered-SE hazard model on **verified-target labels**. The
  existing inference (including the ROA-vs-Palepu finding) was computed on
  contaminated labels and may not survive.
- Check whether the size effect (`log_assets`, a top predictor) was partly
  the model spotting *acquirers* rather than targets.
- Test the target model against `tender.duckdb` — 616 tender-offer targets it
  has never seen, an independent label.

### 2. Find improvements

Things measured and rejected, so don't repeat them: industry-relative ratios
(−1.6pp), Loughran-McDonald sentiment (null at ≥6 months), Palepu and
Ambrose–Megginson variables (+0.08pp), shelf/prospectus buyer features
(+1.65pp, inside noise), trimming the feature set (hurts), shortening the
training window (hurts).

The one family that consistently earns its place in **both** models is
peer/activist — sector consolidation timing. That is where I would look.

### 3. Propose and test new features

All free sources. Candidates I have not built:

- **Form 144** — notice of intent to sell restricted stock. Filing a 144 and
  then *not* selling is the sharpest available version of the insider-silence
  signal, which is already significant at z=+5.0.
- **13F institutional ownership** — free quarterly, and I have *no* ownership
  data at all. Concentration is a documented predictor.
- **10-K year-over-year text similarity** ("Lazy Prices", Cohen/Malloy/Nguyen
  2020) — the filing URLs are already in the cache.
- **Filing-agent identity** — the SEC header names the filer agent. M&A
  counsel appearing on a company's filings is close to a direct observation.
- **H-1B/LCA disclosure data (DOL)** — free, quarterly, and the legal route to
  detecting a hiring freeze.
- **Short interest (FINRA)** — free, twice monthly.

Scope any new source with a few hundred documents before committing to a large
download. That is how the sentiment idea was killed for 20 minutes of work
instead of two days.

### 4. Graphs

`scripts/make_charts.py` has the house style. Read the `dataviz` skill before
writing chart code, and validate any palette with its script rather than
eyeballing. **Every figure currently in `docs/figures/` is stale** — they show
retracted numbers and must be regenerated from `data/*.json`.

Figures worth having:
- Precision curve by list size, both models on one axis
- Target vs buyer lift, showing buying is easier to predict
- Feature-family ablation for both models, with the noise band drawn
- Permutation null distribution against the real score
- Deal rate by company-size decile (the hump that breaks linear models)
- Year-by-year CV spread — the honest picture of regime dependence

## How to work

Show me the number that kills an idea as readily as the one that confirms it.
Four errors have already been caught this way — right-censoring, a metric
measuring three companies, SPAC contamination, and acquirers mislabelled as
targets — and each would have produced a confident wrong answer. Assume there
is a fifth.

When something improves the score, check it isn't the label leaking before
you tell me it worked.
