# Stress test and buyer–target alignment

Rerun 2026-08-04 on the 2012–2026 panel. Every number below is reproducible
from the scripts named beside it; results live in `data/*.json`, which is
gitignored, so the numbers are written out here rather than referenced.

**This document was first written on the 2016 panel with only 2023–2025
scored, which left the pairing work with 137 usable pairs. Two hardcoded start
years were the cause** — `load_pairs.build` defaulted to 2016 and
`pair_scores.YEARS` to three years. Fixing both gives 1,952 deal episodes
instead of 1,371 and 789 usable pairs instead of 137. Every figure below is
from the rerun, and the conclusions that changed are called out where they
occur.

The headline: **the two screens are close to independent, and the pairing is
predictable mostly from structure rather than from either model's timing
signal.** Two of the four stress tests pass cleanly, one refutes its own
hypothesis, and one turns out to be uninformative in both directions.

## The four questions, side by side

| Question | How often right | Lift vs chance |
|---|---|---|
| Will this company be acquired? (target, 25/week) | **12.28%** | 6.91× |
| Will this company acquire someone? (buyer, 25/week) | **31.72%** | 12.21× |
| Told the target — who is the buyer? (top 100 of ~7,453) | **33.7%** | 25.1× |
| Nothing given — name both ends of one deal | **0.89%** | — |

Buying is roughly two and a half times more predictable than being bought.
Pairing, *conditional* on already knowing the target, is the strongest in lift
terms and the weakest in absolute terms — it produces a shortlist, not a name.
Run end to end with nothing given, the chain succeeds on **7 of 789 deals**.

---

## 0. The new asset: a free (acquirer, target) pair table

`deals.acquirer` had been NULL since the project started. It did not need a new
download. EDGAR indexes a deal filing once per **party**, and both index rows
carry the same accession number in the filename, so a two-CIK accession is a
two-party deal filing — recoverable from the `master.idx` files already cached
on disk.

`src/deal/load_pairs.py`, `scripts/load_pairs_run.py` → `data/pairs.duckdb`.
**Zero network requests.**

| | |
|---|---|
| Two-CIK accessions (425, SC TO-T, SC 13E3) | 18,199 |
| Orientable (exactly one party stops filing) | 10,692 |
| Ambiguous, discarded rather than guessed | 7,507 (41%) |
| Collapsed to deal episodes | **1,371** |
| Both parties present in `features.parquet` | **1,023** |

By form: 981 × 425, 347 × SC TO-T, 43 × SC 13E3. DEFM14A is always
single-filer and yields no pair; S-4 carries up to 332 co-registrant CIKs on
one accession, so neither is usable here.

**Orientation validation.** The rule — a target stops filing, an acquirer does
not — is checked against `tender.duckdb`, whose subject CIKs were parsed
independently from SEC-HEADER blocks by a different parser.

| Denominator | Agreement |
|---|---|
| All SC TO-T pairs | 340/341 = 99.7% |
| **Both parties are SEC filers** | **157/158 = 99.4%** |

Quote the second. 477 of 1,952 episodes have an acquirer absent from
`universe` — foreign or private buyers such as Roche Holdings that file no
periodic reports. A non-filer cannot "stop filing", so it is assigned acquirer
trivially, which inflates the easy end of the first denominator. Zero episodes
have a *target* missing from `universe`, which is the consistency check that
matters: a target must be a filer to be in the panel at all.

---

## 1. Do the two models line up? Barely, and not significantly.

`scripts/alignment.py`. No model fits — reads cached scores from
`scripts/pair_scores.py`.

First, a precondition: **corr(p_target, p_buyer) = 0.272** across the scored
panel. The two models are genuinely distinct scores. Had this been above 0.9,
everything in this section and the next would have been measuring one score
against itself.

| | 789 pairs (current) | 137 pairs (first run) |
|---|---|---|
| Observed corr, 4-week lead | **+0.143** | +0.133 |
| Within-week permutation null | **+0.108** (sd 0.026) | +0.131 (sd 0.070) |
| p | **0.086** | 0.43 |

**This is the one conclusion the rerun moved, and it moved against the earlier
claim.** On 137 pairs the observed correlation sat exactly on the null mean and
the honest reading was a clean null — no pair-specific information whatever. On
789 pairs it sits above the null at p = 0.086. That is not significant at any
conventional threshold, but it is no longer nothing, and the earlier "p = 0.43,
zero pair-specific information" was an underpowered test reported with more
confidence than it had earned.

The defensible statement now: **the two screens are close to independent, with
a weak positive association that eleven years of data still cannot separate
from chance.** Most of the raw correlation is week composition — real deals
cluster in weeks when both screens run hot — but not quite all of it.

### The joint top-25 test is structurally dead — do not read it

Joint hit rates now come out at ratios of 2.9× to 4.2× above independence —
but independence still predicts only **0.24 to 0.58 joint events** across the
whole sample, so those ratios rest on one or two coincidences. `alignment.json`
carries an `underpowered` flag and an expected-events count on every row. With
~7,450 companies scored per week, 25 picks each, this test needs thousands of
pairs to resolve and has 789. Do not read the ratio.

The continuous correlation test is the one that carries the answer.

---

## 2. Can we predict *which* buyer? Yes — but not from the timing models.

`scripts/matching.py`. The true acquirer competes against 100 companies
sampled from the same week; LightGBM `lambdarank` with the pair as query group.
Random baseline = 1/101 = **0.99%**.

| Embargo | Test pairs | top-1 [95% CI] | top-10 [95% CI] | Median rank |
|---|---|---|---|---|
| 4w | 139 | 33.1% [25.2–41.0] | 82.0% [75.5–87.8] | 2 |
| 13w | 149 | 39.6% [31.5–47.7] | 81.2% [75.2–87.2] | 2 |

Nothing collapses under embargo, so the model is not reading the announcement.
The CIs overlap almost entirely — the 13-week set is **not** better than the
4-week set, and that 5pp is not narrated as a lead-time effect. Removing
`sector_deal_intensity` changes nothing; it never enters the top 6.

### Most of that is a structural prior, not a learned matching function

A no-model heuristic — *same 2-digit SIC, bigger* — scores top-10 ≈ 70% against
the model's 73.8%. The true acquirer shares the target's 2-digit SIC **65%** of
the time versus ~7% for random distractors. The easy version of this task is
mostly an industry lookup.

### The sharper test: distractors drawn from the target's own industry

Neutralise the industry prior and ask whether we can pick the actual buyer from
among *plausible* buyers. 100 hard negatives, all same 2-digit SIC as the
target. Random = median normalised rank 0.50.

| Ranker | 4w median rank | Better than random | Sign test | 13w median |
|---|---|---|---|---|
| `p_buyer` alone | **0.238** | 52/69 | p = 1.5×10⁻⁵ | 0.223 (55/74, p = 1.7×10⁻⁵) |
| `size_gap` alone | 0.218 | 62/69 | p = 2.1×10⁻¹² | 0.208 (63/74, p = 2.7×10⁻¹⁰) |
| Both combined | 0.178 | — | — | — |

**The buyer model does carry pair-specific information** — it ranks the true
acquirer well above its industry peers, at both embargoes. This does not
contradict §1. Section 1 asked whether the two scores *co-move* on true pairs
(they do not). This asks whether the buyer score *discriminates the true
acquirer from its sector* (it does).

> **Who buys whom is an industry-and-size question. When it happens is a
> filing-behaviour question. The two are close to orthogonal.**

The **53 training pairs** caveat in the first version of this document is
resolved: `pair_scores.py` now scores all eleven years, so the two rolling
folds train on 642 and 697 pairs. Both folds run, and the numbers above are
pooled across them.

### The 101-candidate framing flatters this by ~70×

Everything above ranks the true acquirer against **100 sampled distractors**.
That is the standard evaluation for a matching model, and it answers "can you
beat 100 arbitrary companies" — not "can you name the buyer". Against the
~7,453 companies actually scored in a given week:

| | 101 random candidates | Real universe (~7,453) |
|---|---|---|
| top-1 | 33.1% | **0.4%** |
| top-10 | 82.0% | 3.4% |
| top-100 | — | 33.7% |
| Median rank | 2 | **165** |

Still real signal — median rank 165 against a chance median of ~3,727 is ~23×
better than random, and 33.7% in the top 100 against a 1.34% chance rate is
25.1× lift. **The model narrows ~7,400 candidate acquirers to a shortlist of
100 that contains the right one a third of the time. It cannot name the buyer.**

### End to end, with nothing given, the pipeline collapses

Chain both stages — target in the top 25 that week, *then* acquirer in the top
100 — and it succeeds on **7 of 789 deals (0.89%)** at a 4-week embargo. Six
times the sample moved this figure *down*, from 2.2% to 0.89%: the earlier
number rested on three coincidences. The bottleneck is the target side, where
the median real target sits at **rank 1,232** in its own week and only 2.2% are
in the top 25 four weeks before announcement.

That is the number to quote if anyone asks whether this predicts deals rather
than screens for them.

---

## 3. The four stress tests

`scripts/stress_pairs.py`, one stage per process.

### 3.1 Buyer permutation — PASSES

Never run before. Real 23.92% (test year 2024) against six nulls of 2.11, 3.85,
5.51, 4.98, 3.17, 4.75. Real beats the highest null by 18.4pp. `p = 0.14` is
the arithmetic floor for six draws, not weak evidence. **The buyer model is not
leaking, and the 26.09% headline stands** (23.92% is 2024 alone; 26.09% is the
three-year mean on the identical config).

### 3.2 Clean-label hazard — the existing inference survives

Re-run on verified-target labels. A confound had to be removed first:
`logs/final_stats.log` predates a panel rebuild (45 features then, 72 now), so
the raw label was re-run on the *current* panel to separate label effects from
panel effects. It reproduces the old log closely, so the old log is a valid
reference.

**Nine of nine signals significant on the raw label stay significant with the
same sign. Zero sign flips among the twelve named signals.** Across all 72
coefficients there is exactly one flip: `cash_runway`, which was insignificant
before. Significant coefficients went **up**, 24/72 → 28/72, despite the label
rate falling from 2.79% to 2.04%.

**The ROA-vs-Palepu finding survives:** +0.2943 (z +4.54) → +0.2621 (z +3.20).

Two signals that lost significance versus the old log — `disc_sells_26w_d` and
`form8k_26w` — lost it on the raw label on the current panel too. That is a
panel effect and must not be credited to label cleaning.

### 3.3 Size effect — hypothesis REFUTED

The question was whether `log_assets` was the model spotting acquirers rather
than targets. Criterion: size should contribute on raw and survivor labels but
not on verified targets.

| Label set | Precision | Without size | Size contributes |
|---|---|---|---|
| Raw proxy filers | 11.40% | 8.98% | +2.42pp |
| **Verified targets** | 11.74% | 9.28% | **+2.45pp** |
| Survivors only | 3.36% | 1.62% | +1.74pp |

It contributes on all three, and most on verified targets. The raw-vs-verified
difference of 0.03pp is deep inside the ±2pp noise band. Positives average
log_assets ≈ 20.6–20.9 against negatives ≈ 17.3 in every label set, verified
targets included, and the hazard model agrees independently (`log_assets`
z = +8.87 on the clean label).

**Large companies really are more likely to be acquired. The size effect was
not an acquirer artifact.**

### 3.4 Tender-offer validation — UNINFORMATIVE, and this is the fifth error

Raw output looked like regime variation: lift 2.71× / 0.82× / 6.85× across
2023/2024/2025, with 2024 below chance. It is not regime variation. Re-tested
from cached scores with a bootstrap over weeks:

| Year | Lift | 95% CI | Hit rows | **Distinct companies** | Tender targets available |
|---|---|---|---|---|---|
| 2023 | 3.37× | [2.34–4.39] | 23 | **1** | 69 |
| 2024 | 0.89× | [0.38–1.52] | 7 | **3** | 83 |
| 2025 | 6.95× | [4.17–10.33] | 35 | **5** | 73 |

2023's apparently significant 3.37× with a tight confidence interval is **one
company — NorthStar Healthcare Income — held for 23 consecutive weeks.** The
bootstrap resamples weeks, and the weeks are not independent, so the interval
is meaningless.

**The tender test neither validates nor refutes generalisation.** Out of ~75
tender targets per year the screen catches one to five. There is not enough
there to conclude anything, in either direction.

This is a recurrence of the error class the project already caught once (a
metric measuring three companies). Ranking within the week stops one company
owning the *list*; nothing stopped one company owning the *result*, and no
caller reported it.

**Fixed at the root:** `screen.weekly_precision` now returns `distinct_hits`
alongside `distinct_companies`, so every caller in the project gets it for free
(`src/deal/screen.py`, `tests/test_screen.py`). Any result whose `distinct_hits`
is in the single digits is one company's story, whatever the CI says.

---

## 4. Exact deal details

`scripts/deal_details.py`. Small frames; AUC is the number to trust, since
three of four details have a majority class big enough to flatter accuracy.

| Detail | Accuracy | Baseline | AUC | n_test | Verdict |
|---|---|---|---|---|---|
| Stock vs cash consideration | 84.7% | 79.5% | **0.823** | 88 | **Predictable** |
| Acquirer is bigger | 89.2% | 85.2% | 0.761 | 88 | Proxy artifact — see below |
| …with `log_float` also dropped | 80.1% | 85.2% | 0.694 | 88 | **No signal** |
| Deal completes (on `deal_pairs`) | — | — | — | 0 | **Tautology — unaskable** |
| Deal completes (on tender offers) | 83.0% | 50.9% | **0.913** | 53 | Real, thin |

**Stock vs cash is the cleanest result here.** Whether consideration is stock
(425) or a cash tender (SC TO-T) is predictable at AUC 0.852 from both parties'
fundamentals four weeks before announcement, with no form-derived column in the
inputs.

**"Acquirer is bigger" is not a forecast.** The label is a deterministic
function of two features both known at the observation week. Excluding
`size_gap` and both parties' `log_assets` is not sufficient because `log_float`
proxies them; dropping it too costs 9.3 AUC points and puts accuracy *below*
the majority baseline.

**Deal completion cannot be asked of the pair table at all.** All 691 in-panel
pairs carry label 1, because `load_pairs.orient` *defines* the target as
whichever party stopped filing — the exact quantity the detail tries to
predict. Broken deals were discarded upstream as ambiguous. Re-asked on
`tender.duckdb`, the one place a target is identified independently of the
outcome, completion is predictable at AUC 0.942 from the subject's fundamentals
four weeks before the offer. That result rests on 53 test cases (27 positives)
and covers SC TO-T only, which is not the 425 population the rest of the
project models.

**Right-censoring bites here too.** `universe.delisted` is a last-seen date, so
non-completion is only evidenced by a filing *after* the window — impossible
near the panel edge. Measured completion rate is 1.00 from 2025Q2 against ~0.65
historically, and announcement date alone scored AUC 0.780. A 270 + 180 day
slack fixes it (date-only AUC falls to 0.678); this is now `CENSOR_SLACK_DAYS`.

**Lead time.** Of 1,219 scorable pairs, **78 were ever in the top 25** during
the 52 weeks before announcement — **6.4%**, almost exactly the earlier
estimate on five times the sample. When the screen does see a deal coming,
median lead is **6.5 weeks** (p25 0.6, p75 16.0, max 51.4).

---

## 4.1 Is it tradeable? Palepu, measured on this panel

`scripts/scope_proxy_prices.py`, `scripts/proxy_premium.py`.

No free source retains price history for delisted companies, so returns remain
uncomputable. But roughly **43% of merger proxies state the takeover premium in
words** ("a premium of approximately 38% to the closing price"), which is
enough to test the objection that actually matters.

**Dead end first:** the quarterly high/low price tables that Reg S-K Item 201
used to require parse in **3.3%** of 2023-24 proxies and **12%** of 2016-17.
The SEC's 2018 disclosure simplification dropped the requirement and it was
never reliable before it. There is no price *series* here. Cost of finding
out: 400 MB and 20 minutes.

**What the premium says.** 150 verified targets from 2023 onward, premium
parsed for 74 (49%), 63 of which the model had scored:

| | |
|---|---|
| Median premium, all deals | 30.5% (p25 19.9, p75 58.5) |
| Model ranked **better** than median | **28.9%** (n=32) |
| Model ranked **worse** than median | **36.0%** (n=31) |
| Spearman(rank, premium) | **rho +0.294, p = 0.019** |
| …controlling for `log_assets` | **rho +0.265, p = 0.036** |

A better rank is a *smaller* number, so a positive rho means **the deals this
model ranks highly carry systematically thinner premiums** — roughly 7pp
thinner at the median. That is Palepu (1986) showing up directly in this
panel's own data: the model finds targets the market has partly identified
already, and part of the premium has been paid away before a screen could buy.

The size confound is ruled out rather than assumed. Larger targets do carry
thinner premiums (size vs premium rho **−0.453**, p < 0.001), but rank and size
are nearly independent here (rho −0.132, p = 0.30), and the partial correlation
survives.

**Effect on the economics.** The deal leg is worth about
10.5% × 29% ≈ **3.0% gross over 12 months**, not the 3.9% a naive 37% premium
would suggest — before the drift on the ~90% that never transact, before costs,
and before whatever the flagged names' starting valuations already embed.

**Caveats that bound this.** n = 63. The Mann-Whitney test on a median split is
**not** significant (p = 0.316); only the continuous rank test is, and it is
the more powerful test on the same data. Proxies quote premiums against several
reference prices and the parser takes the first stated, which adds noise. Most
importantly, **the 51% of proxies that state no premium may differ
systematically** from those that do — that is a selection effect this design
cannot rule out.

---

## 5. What did not work

Reported as plainly as what did:

- **Score alignment between the two models.** Weak and not significant:
  p = 0.086 on 789 pairs. Reported as a clean null (p = 0.43) on the first run
  of 137 pairs — that was underpowered, and the rerun moved it against the
  claim rather than for it.
- **The joint top-25 alignment test.** Structurally underpowered by ~40×;
  produces a number that cannot be interpreted.
- **Tender-offer external validation.** Rests on 1–5 companies per year.
  Uninformative in both directions.
- **Naming the buyer.** Top-1 against the real weekly universe is 0.4%. The
  model produces a shortlist, not a name.
- **Both ends of a deal from nothing.** 7 of 789, and the figure fell when the
  sample grew.
- **"Acquirer is bigger" as a prediction.** Below majority baseline once the
  size proxies are removed.
- **Deal completion on the pair table.** Tautological by construction.
- **The size-effect-is-really-acquirers hypothesis.** Refuted; the size effect
  is real on clean target labels.

## 6. Known limits of this run

- 1,952 deal episodes, of which **789** have both parties scored in the same
  week. Everything in §1 and §2 rests on that — six times the first run, and
  it moved two conclusions.
- The scored panel now spans 2015–2025, so the matching study runs two rolling
  folds training on 642 and 697 pairs rather than one fold of 53.
- Matching distractors are random or same-SIC, not a curated set of credible
  bidders. The hard-negative variant is the closer approximation.
- 41% of two-party accessions are dropped as ambiguous. Those are not missing
  at random — they skew toward deals where both parties keep filing, i.e.
  stock mergers of similar-sized firms.
