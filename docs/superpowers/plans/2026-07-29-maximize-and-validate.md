# Maximize and Validate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Search the feature/hyperparameter space for maximum screen precision, confirm the winner is not a fluke via repeated out-of-sample evaluation, and subject it to statistical tests that would expose a spurious result.

**Architecture:** Three-way temporal split. All searching happens on a validation period; the 2024+ test period is touched exactly once, at the end. The winner is then re-evaluated across rolling-origin folds and multiple seeds, and finally attacked with a permutation test, bootstrap confidence intervals, and a clustered-standard-error hazard model.

**Tech Stack:** Python 3.11+, DuckDB, Polars, LightGBM, statsmodels, numpy.

---

## Global Constraints

- **The 2024+ test period is off-limits during search.** Every configuration decision is made on validation (2022–2023) only. Reporting a test score for a config chosen on that same test score is selection bias, and it is the single easiest way to produce a number that evaporates.
- Splits are temporal or grouped by company — never random over rows.
- Evaluation is per-week top-N screen precision.
- Every reported final number carries a confidence interval.
- A result that fails the permutation test is reported as failed, not re-tuned until it passes.
- No feature may be added during the final evaluation phase.

---

## The traps this plan exists to avoid

**1. Test-set mining.** Trying 40 configs and reporting the best test score inflates that score by the maximum of 40 draws from noise. Fixed by the three-way split.

**2. Single-split luck.** One train/test boundary is one draw. A config that wins at the 2024 boundary may lose at 2023 or 2025. Fixed by rolling-origin folds.

**3. Seed luck.** LightGBM bags rows and features stochastically. Fixed by averaging over seeds and reporting spread.

**4. A signal that is really noise.** If labels are shuffled and the model still "works", the pipeline leaks. Fixed by a permutation test — the strongest single check available.

**5. Inference without clustering.** A company contributes ~550 correlated rows; naive standard errors ran **4× too small** when measured earlier. Fixed by clustering on company.

---

## File Structure

```
src/deal/
  evaluate_robust.py   # bootstrap CI, permutation test, rolling-origin folds
scripts/
  search_config.py     # phase 1: search on VALIDATION only
  final_validate.py    # phase 2: repeated eval + statistical tests on TEST
docs/
  FINAL_RESULTS.md     # the single report
```

---

### Task 1: Robust evaluation primitives

**Files:** Create `src/deal/evaluate_robust.py`, `tests/test_evaluate_robust.py`

**Produces:** `bootstrap_precision(df, p, n_per_week, n_boot, seed) -> dict`,
`permutation_test(train, test, cols, n_perm, seed) -> dict`,
`rolling_origin(df, cutoffs) -> list[tuple]`

- [ ] Step 1: write failing tests
- [ ] Step 2: run, confirm failure
- [ ] Step 3: implement
- [ ] Step 4: run, confirm pass

### Task 2: Configuration search on validation

**Files:** Create `scripts/search_config.py`

Search axes: feature subsets (all / no-z / no-deltas / novel-heavy), horizon
label (13/26/39/52 weeks), `num_leaves`, `min_data_in_leaf`, `learning_rate`,
recency half-life, universe restriction. Score on validation precision@25.

- [ ] Step 1: implement grid over validation only
- [ ] Step 2: run, record ranked table
- [ ] Step 3: freeze the winner

### Task 3: Repeated final evaluation

**Files:** Create `scripts/final_validate.py`

Winner evaluated across 4 rolling-origin cutoffs × 5 seeds, plus grouped
90/10, reporting mean ± SD.

- [ ] Step 1: implement
- [ ] Step 2: run

### Task 4: Statistical tests

Permutation test (labels shuffled within week), bootstrap CI on precision@25,
clustered-SE hazard model on the winning feature set, nested controls-vs-novel
increment with inference.

- [ ] Step 1: implement
- [ ] Step 2: run
- [ ] Step 3: write `docs/FINAL_RESULTS.md`

---

## Success criteria, declared in advance

- Precision@25 of **20%+** on held-out test, with a bootstrap CI whose lower
  bound clears 15%.
- Stable across rolling-origin folds: SD under a third of the mean.
- Permutation test: real score outside the entire null distribution.
- At least three novel signals surviving clustered-SE inference at p<0.05.

**If the honest number lands below 20%, that is the reported number.** Tuning
until the target appears is how the 20% becomes fiction.
