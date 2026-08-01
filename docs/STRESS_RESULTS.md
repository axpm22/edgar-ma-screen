# Stress test and buyer–target alignment

Run 2026-08-01. Every number below is reproducible from the scripts named
beside it; results live in `data/*.json`, which is gitignored, so the numbers
are written out here rather than referenced.

The headline: **the two models do not line up, but the pairing is still
predictable — from structure rather than from either model's timing signal.**
Two of the four stress tests pass cleanly, one refutes its own hypothesis, and
one turns out to be uninformative in both directions.

## The three questions, side by side

| Question | How often right | Lift vs chance |
|---|---|---|
| Will this company be acquired? (target, 25/week) | **11.7%** | 7.0× |
| Will this company acquire someone? (buyer, 25/week) | **23.9%** | 13.2× |
| Told the target — who is the buyer? (top 100 of ~7,138) | **34.3%** | 24.5× |
| Nothing given — name both ends of one deal | **2.2%** | — |

Buying is roughly twice as predictable as being bought. Pairing, *conditional*
on already knowing the target, is the strongest of the three in lift terms and
the weakest in absolute terms — it produces a shortlist, not a name. Run
end-to-end with nothing given, the chain succeeds on 3 of 137 deals. Details in
§2 and §2.1.

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

Quote the second. 327 of 1,371 episodes have an acquirer absent from
`universe` — foreign or private buyers such as Roche Holdings that file no
periodic reports. A non-filer cannot "stop filing", so it is assigned acquirer
trivially, which inflates the easy end of the first denominator. Zero episodes
have a *target* missing from `universe`, which is the consistency check that
matters: a target must be a filer to be in the panel at all.

---

## 1. Do the two models line up? No.

`scripts/alignment.py`. No model fits — reads cached scores from
`scripts/pair_scores.py`.

First, a precondition: **corr(p_target, p_buyer) = 0.245** across 966,670
company-weeks. The two models are genuinely distinct scores. Had this been
above 0.9, everything in this section and the next would have been measuring
one score against itself.

| | |
|---|---|
| Observed corr(target score, buyer score) on 137 real pairs, 4-week lead | **+0.133** |
| Within-week permutation null | **+0.131** (sd 0.070) |
| p | **0.43** |

The entire correlation is week composition. Real deals cluster in weeks when
both screens run hot — common market and time factors. Hold the week fixed,
reshuffle which acquirer is paired with which target, and the same correlation
reappears. **There is no pair-specific information in the two scores jointly:
knowing this target is hot tells you nothing about which buyer's score is hot.**

**The naive test would have said the opposite.** Raw Spearman gives
rho = +0.240, p = 0.005 — a publishable-looking result. Under the within-week
permutation it goes to p = 0.134. This is the too-small-naive-error failure
mode landing on a metric that had never been subjected to it.

### The joint top-25 test is structurally dead — do not read it

`agreement_ratio = 0.00×` appears in `data/alignment.json` at every lead. It
means nothing. With ~7,100 companies scored per week, 25 picks each, and 130
pairs, independence predicts **0.1 joint hits**; observing zero is the modal
outcome under the null. The design can only reject agreement ratios above
~4×–23×, and needs roughly 5,500 pairs to be informative against 130. Written
up unguarded it reads as "the screens actively anti-align", which is a
confident wrong answer. The JSON now carries an `underpowered` flag and an
expected-events count so it cannot be misread later.

The continuous correlation test is the one that carries the answer.

---

## 2. Can we predict *which* buyer? Yes — but not from the timing models.

`scripts/matching.py`. The true acquirer competes against 100 companies
sampled from the same week; LightGBM `lambdarank` with the pair as query group.
Random baseline = 1/101 = **0.99%**.

| Embargo | Test pairs | top-1 [95% CI] | top-10 [95% CI] | Median rank |
|---|---|---|---|---|
| 4w | 84 | 28.6% [19.0–38.1] | 73.8% [64.3–83.3] | 3 |
| 13w | 89 | 33.7% [24.7–42.7] | 71.9% [62.9–82.0] | 2 |

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

Caveats not to drop: the trained model rests on **53 training pairs** — the
2023 fold has none, because `p_buyer` only exists in scored test years, a
limitation of how `pair_scores.py` was built. The hard-negative rows above are
untrained heuristics, which is why they are more trustworthy here than the
53-pair fit.

### The 101-candidate framing flatters this by ~70×

Everything above ranks the true acquirer against **100 sampled distractors**.
That is the standard evaluation for a matching model, and it answers "can you
beat 100 arbitrary companies" — not "can you name the buyer". Against the
~7,138 companies actually scored in a given week:

| | 101 random candidates | Real universe (7,138) |
|---|---|---|
| top-1 | 28.6% | **0.0%** |
| top-10 | 73.8% | 1.5% |
| top-100 | — | 34.3% |
| Median rank | 3 | **202** |

Still real signal — median rank 202 against a chance median of 3,569 is ~18×
better than random, and 34.3% in the top 100 against a 1.4% chance rate is
24.5× lift. **The model narrows ~7,000 candidate acquirers to a shortlist of
100 that contains the right one a third of the time. It cannot name the buyer.**

### End to end, with nothing given, the pipeline collapses

Chain both stages — target in the top 25 that week, *then* acquirer in the top
100 — and it succeeds on **3 of 137 deals (2.2%)** at a 4-week embargo, 2 of
136 at 13 weeks. The bottleneck is the target side: the median real target sits
at **rank 1,216** in its own week, and only 2.9% are in the top 25 four weeks
before announcement.

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
| Stock vs cash consideration | 84.1% | 79.5% | **0.852** | 88 | **Predictable** |
| Acquirer is bigger | 88.1% | 85.2% | 0.756 | 88 | Proxy artifact — see below |
| …with `log_float` also dropped | 83.5% | 85.2% | 0.663 | 88 | **No signal** |
| Deal completes (on `deal_pairs`) | — | — | — | 0 | **Tautology — unaskable** |
| Deal completes (on tender offers) | 85.8% | 50.9% | **0.942** | 53 | Real, thin |

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

**Lead time.** Of 246 scorable pairs, **16 were ever in the top 25** during the
52 weeks before announcement — 6.5%. When the screen does see a deal coming,
median lead is **9.3 weeks** (p25 0.9, p75 13.8, max 49.1).

---

## 5. What did not work

Reported as plainly as what did:

- **Score alignment between the two models.** Null, p = 0.43. The screens carry
  no joint information about a specific transaction.
- **The joint top-25 alignment test.** Structurally underpowered by ~40×;
  produces a number that cannot be interpreted.
- **Tender-offer external validation.** Rests on 1–5 companies per year.
  Uninformative in both directions.
- **"Acquirer is bigger" as a prediction.** Below majority baseline once the
  size proxies are removed.
- **Deal completion on the pair table.** Tautological by construction.
- **The size-effect-is-really-acquirers hypothesis.** Refuted; the size effect
  is real on clean target labels.

## 6. Known limits of this run

- 1,023 usable pairs, of which only ~130–137 have both parties scored in the
  same week. Everything in §1 and §2 rests on that.
- `pair_scores.parquet` covers 2023–2025 only, which caps the matching study at
  a single training fold of 53 pairs. Scoring `p_buyer` across the full
  2016–2025 panel is the single change that would most improve §2.
- Matching distractors are random or same-SIC, not a curated set of credible
  bidders. The hard-negative variant is the closer approximation.
- 41% of two-party accessions are dropped as ambiguous. Those are not missing
  at random — they skew toward deals where both parties keep filing, i.e.
  stock mergers of similar-sized firms.
