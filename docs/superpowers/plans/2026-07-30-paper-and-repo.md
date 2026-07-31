# Paper Outline and Public Repo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a laptop directory into a public GitHub repo containing a paper outline, the figures that support it, and a pipeline a stranger can clone and run.

**Architecture:** The repo is the primary artifact and the paper lives inside it. Three layers: the paper outline (`PAPER.md`) written as bullets against evidence that already exists; the figures, regenerated from measured numbers with every stale figure corrected; and the reproducibility shell — git, README, pinned dependencies, one command that rebuilds everything. Nothing here re-runs the model. Every number is already measured.

**Tech Stack:** git, Python 3.11+, matplotlib, existing `deal` package.

---

## Global Constraints

- **`data/` is 9.2 GB and never enters git.** `.gitignore` excludes it before the first commit.
- **Every number in the paper and every figure comes from a measured run.** No number is retyped from memory; each traces to a file in `data/*.json` or a log.
- **The headline is 13.81% / 5.65× (operating companies, 3-year mean).** The 27.58% all-companies figure appears only labelled as including de-SPACs.
- **Current figures are stale.** `hit_rate.png`, `shuffle.png`, `stability.png`, `nested.png` and `mistakes.png` all carry the retracted 27.5% single-split numbers and must be regenerated.
- Do not commit `.venv/`, `logs/`, `__pycache__/`, or `*.duckdb`.
- Repo name: `ma-signals`. CRSP work, if it ever happens, goes in a separate private repo — WRDS forbids redistribution.

---

## What the evidence actually supports

Written first so the paper outline cannot drift from it.

| Claim | Evidence | Source |
|---|---|---|
| Screen beats chance | 13.81% vs 2.44% base, **5.65× lift**, 3-year mean | `data/select_cv.json` |
| Not luck | Permutation null max 5.89%, real 21.40% | `data/stress_results.json` |
| Not leakage | Flat under 8/16-week embargo | stress log |
| Not SPAC detection | Holds at 13.81% with all 1,707 SPACs removed | `select_cv.json` |
| Regime-dependent | 11.40–15.15% across 2023/24/25 | `select_cv.json` |
| **ROA sign flips vs Palepu** | β=+0.35, z=+5.25 clustered | `final_stats.json` |
| Literature adds ~nothing | +0.08pp | ablation |
| Industry-relative ratios hurt | −1.6pp both universes | `data/ind_*.json` |
| Sentiment is null | p>0.30 all categories at ≥6 months | `data/lm_scores.parquet` |
| Behaviour beats accounting | form counts +5.43pp, only family clearing noise | ablation |

---

## File Structure

```
ma-signals/                     # public repo root
  README.md                     # result, then reproduce steps
  PAPER.md                      # the outline (Task 2)
  LICENSE                       # MIT
  .gitignore
  pyproject.toml                # already exists; add pins
  requirements-lock.txt
  Makefile                      # one command per pipeline stage
  src/deal/                     # existing package, unchanged
  scripts/                      # existing scripts, unchanged
  tests/
  docs/
    figures/                    # regenerated (Task 1)
    FINAL_RESULTS.md            # corrected (Task 3)
```

---

### Task 1: Regenerate every figure against measured numbers

Do this first: the paper outline references figures, and five of the eight
currently show retracted numbers.

**Files:**
- Modify: `scripts/make_charts.py`
- Create: `docs/figures/cv_years.png`, `docs/figures/universes.png`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: `data/select_cv.json`, `data/stress_results.json`.
- Produces: eight PNGs in `docs/figures/`, all traceable to a measured source.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_charts.py
import json
from pathlib import Path


def test_no_chart_hardcodes_a_retracted_number():
    """27.52 and 30.94 came from the contaminated single split and were
    retracted. They must not survive anywhere in the chart source."""
    src = Path("scripts/make_charts.py").read_text()
    for stale in ("27.52", "30.94", "29.66", "24.10", "43.00"):
        assert stale not in src, f"retracted value {stale} still in charts"


def test_headline_numbers_match_the_measured_run():
    rows = json.loads(Path("data/select_cv.json").read_text())
    nospac = [r for r in rows
              if r["stage"] == "nospac" and r["subset"] == "A_all"]
    mean = sum(r["prec"] for r in nospac) / len(nospac)
    assert 13.0 < mean < 14.5, f"operating-only mean drifted to {mean}"


def test_every_declared_figure_exists():
    for f in ("funnel", "hit_rate", "size_hump", "shuffle", "stability",
              "signals", "nested", "cv_years", "universes"):
        assert Path(f"docs/figures/{f}.png").exists(), f"missing {f}.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_charts.py -v`
Expected: FAIL — `retracted value 27.52 still in charts`

- [ ] **Step 3: Rewrite the stale charts**

Replace `fig_hit_rate`, `fig_shuffle`, `fig_stability`, `fig_nested` and
`fig_mistakes` in `scripts/make_charts.py` with these, and add two new ones:

```python
def fig_hit_rate():
    """Source: data/curve_nospac.json, produced by Step 3a. Only the N=25
    value (13.81%) was measured during development; the rest of the curve is
    generated fresh rather than assumed."""
    import json
    from pathlib import Path as _P
    curve = json.loads(_P("data/curve_nospac.json").read_text())
    ns = [f"Top {n}" for n in curve["ns"]]
    hit = curve["precision"]
    base = curve["base"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(ns, hit, color=BLUE, width=0.6)
    _label(ax, bars)
    ax.axhline(base, color=ORANGE, linewidth=2, linestyle="--")
    ax.text(len(ns) - 0.55, base + 1.0,
            f"if you picked at random: {base:.1f}%", ha="right",
            fontsize=9, color=ORANGE,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.set_ylim(0, max(hit) * 1.35)
    _style(ax, "share later acquired, within 12 months")
    ax.set_title("How often the shortlist is right (real companies only)")
    fig.tight_layout(); fig.savefig(OUT / "hit_rate.png"); plt.close(fig)


def fig_cv_years():
    """Source: select_cv.json. The spread is the point, not the mean."""
    yrs = ["2023", "2024", "2025"]
    allc = [32.62, 21.40, 28.73]
    oper = [15.15, 11.40, 14.87]
    x = range(3)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    b1 = ax.bar([i - 0.2 for i in x], allc, width=0.38, color=BLUE,
                label="including de-SPACs")
    b2 = ax.bar([i + 0.2 for i in x], oper, width=0.38, color=AQUA,
                label="operating companies only")
    _label(ax, b1, dy=0.5); _label(ax, b2, dy=0.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(yrs)
    ax.set_ylim(0, 40)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax, "hit rate")
    ax.set_xlabel("tested on this year", fontsize=9)
    ax.set_title("Performance varies more by year than by anything else")
    fig.tight_layout(); fig.savefig(OUT / "cv_years.png"); plt.close(fig)


def fig_universes():
    """Source: select_cv.json. What SPACs were worth."""
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bars = ax.bar(["Including\nde-SPACs", "Operating\ncompanies only"],
                  [9.40, 5.65], color=[BLUE, AQUA], width=0.45)
    for b, v in zip(bars, [9.40, 5.65]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}x",
                ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 11)
    _style(ax, "lift over base rate")
    ax.set_title("Blank-cheque shells were a third of the apparent skill")
    fig.tight_layout(); fig.savefig(OUT / "universes.png"); plt.close(fig)


def fig_shuffle():
    """Source: stress_results.json permutation stage."""
    null = [5.89, 2.51, 4.02, 1.77, 3.30]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(1, 6), null, color=BLUE, width=0.55)
    ax.axhline(21.40, color=ORANGE, linewidth=2.5)
    ax.text(3, 19.3, "the real model: 21.4%", ha="center", fontsize=10,
            color=ORANGE, fontweight="bold")
    ax.text(3, 8.5, "5 rebuilds on scrambled answers\nbest of them: 5.9%",
            ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0, 24); ax.set_xticks(range(1, 6))
    _style(ax, "hit rate")
    ax.set_xlabel("scrambled run", fontsize=9)
    ax.set_title("The scramble test: nothing fake survives it")
    fig.tight_layout(); fig.savefig(OUT / "shuffle.png"); plt.close(fig)


def fig_stability():
    """Source: stress_results.json seed stage (clean three-way split)."""
    seeds = [20.31, 22.04, 18.58]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bars = ax.bar(range(1, 4), seeds, color=BLUE, width=0.5)
    _label(ax, bars, "{:.1f}", dy=0.35)
    ax.set_ylim(0, 27); ax.set_xticks(range(1, 4))
    _style(ax, "hit rate")
    ax.set_xlabel("repeat run, different random seed", fontsize=9)
    ax.set_title("Rerunning the same setup moves it by about 2 points")
    fig.tight_layout(); fig.savefig(OUT / "stability.png"); plt.close(fig)


def fig_nested():
    """Source: ablation. Only form counts clear the noise bar."""
    fams = ["Form-filing counts", "Activist + peer deals", "8-K item codes",
            "Deltas", "Per-company z-scores", "Literature variables",
            "Insider trading", "Market value"]
    vals = [5.43, 2.45, 2.15, 1.13, 0.68, 0.08, -0.15, -1.55]
    cols = [AQUA if v > 3 else BLUE if v > 0 else ORANGE for v in vals]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.barh(fams[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + (0.12 if v >= 0 else -0.12),
                b.get_y() + b.get_height() / 2, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-2.6, 6.6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("percentage points contributed (2 seeds; +/-2pp is noise)",
                  fontsize=9)
    ax.set_title("Only one family of signals clearly earns its place",
                 loc="left")
    fig.tight_layout(); fig.savefig(OUT / "nested.png"); plt.close(fig)
```

Delete `fig_mistakes` entirely — its three bars are all retracted numbers, and
the mistakes belong in prose rather than a chart.

Update the `__main__` block:

```python
if __name__ == "__main__":
    for fn in (fig_hit_rate, fig_size_hump, fig_shuffle, fig_stability,
               fig_signals, fig_nested, fig_cv_years, fig_universes,
               fig_funnel):
        fn()
        print(f"  {fn.__name__}")
    print(f"figures -> {OUT}")
```

- [ ] **Step 3a: Measure the precision curve (do not assume it)**

Only the top-25 figure was measured during development. Generate the rest —
**6 fits, ~5 minutes, 2 threads.**

```bash
cat > /tmp/curve.py <<'PYEOF'
import datetime as dt, gc, json, sys
import numpy as np, polars as pl
sys.path.insert(0, "scripts")
from final_stats import HORIZON, relabel
from select_cv import load_all, split, spac_ciks, PARAMS, ROUNDS
from deal import screen
import lightgbm as lgb

df, cols = load_all()
df = df.filter(~pl.col("cik").is_in(spac_ciks()))
NS = [10, 25, 50, 100]
acc = {n: [] for n in NS}
bases = []
for yr in (2023, 2024, 2025):
    tr, va, te = split(df, yr)
    if not te.height or not te["y"].sum():
        continue
    d = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                    label=tr["y"].to_pandas())
    dv = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                     label=va["y"].to_pandas())
    b = lgb.train(PARAMS, d, num_boost_round=ROUNDS, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    p = np.asarray(b.predict(te.select(cols).to_pandas().astype("float32")))
    for n in NS:
        acc[n].append(screen.weekly_precision(te, p, n)["precision"] * 100)
    bases.append(float(te["y"].mean() * 100))
    del b, d, dv, p, tr, va, te
    gc.collect()
out = {"ns": NS, "precision": [float(np.mean(acc[n])) for n in NS],
       "base": float(np.mean(bases))}
json.dump(out, open("data/curve_nospac.json", "w"), indent=1)
print(out)
PYEOF
.venv/bin/python -u /tmp/curve.py
```

Expected: the N=25 entry lands near **13.81%**, confirming it reproduces. If
it does not, stop — something has drifted since the CV run.



Run: `.venv/bin/python scripts/make_charts.py && .venv/bin/python -m pytest tests/test_charts.py -v`
Expected: 9 figures printed, 3 passed

- [ ] **Step 5: Look at the two new figures**

Open `docs/figures/cv_years.png` and `docs/figures/nested.png`. Check the
legend does not sit on a bar and no label is clipped. The validator checks
colour, not layout.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_charts.py tests/test_charts.py docs/figures/
git commit -m "fix: regenerate figures against measured multi-year numbers"
```

---

### Task 2: The paper outline

**Files:**
- Create: `PAPER.md`
- Test: `tests/test_paper.py`

**Interfaces:**
- Consumes: figures from Task 1, numbers from `data/*.json`.
- Produces: the outline every later document quotes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paper.py
from pathlib import Path


def test_paper_leads_with_the_defensible_number():
    t = Path("PAPER.md").read_text()
    i_oper, i_all = t.index("13.81"), t.index("27.58")
    assert i_oper < i_all, "all-companies figure must not precede operating-only"


def test_paper_states_the_negative_results():
    t = Path("PAPER.md").read_text().lower()
    for claim in ("sentiment", "industry-relative", "palepu"):
        assert claim in t, f"missing {claim}"


def test_paper_references_only_existing_figures():
    t = Path("PAPER.md").read_text()
    import re
    for f in re.findall(r"figures/([a-z_]+\.png)", t):
        assert Path(f"docs/figures/{f}").exists(), f"missing figure {f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_paper.py -v`
Expected: FAIL — `FileNotFoundError: PAPER.md`

- [ ] **Step 3: Write `PAPER.md`**

```markdown
# Behaviour Beats Balance Sheets: Predicting Acquisitions from Public Filings

**Working outline.** Every number traces to a measured run; sources in
brackets. Figures live in `docs/figures/`.

---

## 1. Introduction

- Takeover-target prediction dates to Palepu (1986) and is considered largely
  settled: targets are predictable, and the prediction does not earn
  abnormal returns.
- Almost all of that literature uses licensed data (CRSP, Compustat, SDC), so
  results cannot be independently reproduced.
- **This paper asks a narrower question:** how far can you get on free public
  filings alone, and does *behavioural* filing data add anything beyond the
  accounting variables the literature settled on?
- Three contributions:
  1. A fully reproducible pipeline — 11.6M EDGAR filings to a 4.1M
     company-week panel, all free, all in the accompanying repo.
  2. **Filing behaviour outperforms accounting fundamentals.** Form-filing
     counts contribute +5.43pp; the classic literature variables +0.08pp.
  3. **Palepu's central hypothesis inverts on modern data.** Profitability
     predicts acquisition *positively* (β=+0.35, z=+5.25 clustered), not
     negatively.

## 2. Data

- EDGAR master indexes 2016–2026: **11,591,580 filings** → 15,325 companies.
- Labels: DEFM14A merger proxies, episode-collapsed → **2,456 deals**.
- Panel: **4,123,449 company-weeks**, one row per listed company per week.
- Feature families: fundamentals (25 XBRL tags), public float, insider
  transactions (3.1M Form 4 records), form-filing counts (1.4M events), 8-K
  item codes, full-text search phrases, activist identity, peer deals.
- *Figure:* `figures/funnel.png` — pipeline scale.
- **Survivorship:** universe derived from filing activity, not index
  membership, so delisted and acquired companies stay in for exactly the
  weeks they existed.

## 3. Method

- Discrete-time hazard framing: one row per company-week, label = acquired
  within 52 weeks. Censoring is encoded by row structure.
- LightGBM rather than logit — the size relationship is hump-shaped
  (*figure:* `figures/size_hump.png`) and a linear coefficient ranks
  mega-caps first, the least acquirable decile.
- Evaluation is **per-week top-N screen precision**, not global top-k. A
  global ranking puts one company in the top 100 eighty-four times.
- Splits are temporal or grouped by company, never random over rows.

## 4. Results

- **Operating companies: 13.81% precision at top-25/week, 5.65× lift**
  (3-year mean; range 11.40–15.15%). [select_cv.json]
- Including de-SPACs: 27.58%, 9.40× — reported separately because
  blank-cheque vehicles exist to merge. *Figure:* `figures/universes.png`.
- *Figure:* `figures/hit_rate.png` — precision by list size.
- *Figure:* `figures/cv_years.png` — the year-to-year spread exceeds every
  other source of variation.
- *Figure:* `figures/signals.png` — 13 signals significant under
  company-clustered standard errors.

## 5. What works, and what does not

- *Figure:* `figures/nested.png` — leave-one-family-out.
- **Form-filing counts (+5.43pp) are the only family clearly clearing the
  noise bar.** Everything else is inside ±2pp.
- **Negative result 1 — the literature adds +0.08pp.** Palepu's
  growth-resource mismatch, Ambrose & Megginson's tangible assets and
  blank-cheque preferred: all present, all inert for prediction.
- **Negative result 2 — industry-relative ratios cost 1.6pp** in both
  universes. The literature found they help *linear* models, which cannot
  condition on industry otherwise; a tree learns that interaction natively,
  so pre-computing it only adds noise. **The advice is model-class-specific
  and does not transfer.**
- **Negative result 3 — sentiment is null.** Loughran-McDonald scores on
  3,101 8-K documents show no difference at ≥6 months before announcement
  (all p>0.30). The apparent near-deal effect is document composition:
  merger 8-Ks are 38% longer and denser in *every* category, including
  strong modals, which is the opposite of hedging.
- **Synthesis: companies leak through what they are obliged to file, not
  through how they write.** A specific phrase ("reviewing strategic
  alternatives", z=+7.73) carries signal; diffuse tone across the same
  documents carries none.

## 6. Palepu revisited

- Palepu (1986): takeovers discipline inefficient management, so
  profitability should predict *negatively*.
- Measured here: **ROA β=+0.35, z=+5.25** with company-clustered errors.
  Sign reversed, strongly significant.
- Supporting: `fcf_to_assets` positive (+2.12) while `ocf_to_assets` is
  negative (−2.68) — those differ by capex, so conditional on cash
  generation, firms that need not spend it are the targets. That is the
  leveraged-buyout screen.
- **Interpretation:** deal flow is now driven by financial buyers acquiring
  cash generators, not raiders fixing underperformers. The 1986 target
  archetype no longer describes the market.

## 7. Robustness

- Permutation test: 5 refits on labels shuffled within week. Null max 5.89%
  against 21.40% real. *Figure:* `figures/shuffle.png`.
- Embargo: performance flat with 8 and 16 weeks blanked before each deal,
  so the model is not reading post-announcement filings.
- Seeds: ±2pp. *Figure:* `figures/stability.png`.
- SPAC exclusion: holds with all 1,707 blank-cheque companies removed.
- Company-level check: 71 of 188 flagged companies acquired (37.77%), higher
  than the row-level figure — repetition biases *down*, because acquired
  companies exit the panel.
- Data audit: no duplicate rows, no null or infinite features, no positive
  at or after its own announcement date.

## 8. Four errors caught, and what each would have cost

| Error | Effect if missed |
|---|---|
| Right-censoring — final year unable to have outcomes | understated by 7.9pp |
| Global top-k measuring 3 companies, not 100 | metric meaningless |
| SPAC contamination — 18.5% of labels | a third of apparent skill |
| DEFM14A dated 40–70 days after announcement | post-announcement leakage |

## 9. Limitations

- Wrong ~86% of the time on any individual company. A screen, not a forecast.
- Regime-dependent: 11.40–15.15% by year.
- **No returns.** No free source retains price history for delisted
  companies, so economic significance is untested — and Palepu's finding
  that prediction does not earn abnormal returns remains unchallenged here.
- Labels restricted to DEFM14A, omitting tender offers (~27%, non-randomly).
- Rumour dates unmeasured.

## 10. Reproducibility

- Every input free. No licensed data anywhere.
- `make all` rebuilds the panel from EDGAR; `make paper` regenerates figures.
- Repo: `github.com/<user>/ma-signals`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_paper.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add PAPER.md tests/test_paper.py
git commit -m "docs: paper outline against measured evidence"
```

---

### Task 3: Repo shell — git, gitignore, README, Makefile

**Files:**
- Create: `.gitignore`, `README.md`, `Makefile`, `LICENSE`, `requirements-lock.txt`
- Modify: `docs/FINAL_RESULTS.md`

**Interfaces:**
- Consumes: `PAPER.md`, `docs/figures/`.
- Produces: a repo a stranger can clone and run.

- [ ] **Step 1: Initialise git and exclude the data**

`data/` is 9.2 GB. Write `.gitignore` **before** the first `git add`, or the
objects land in history permanently.

```bash
cd /Users/albanm/Claude/Projects/RandomBSQuant
git init
cat > .gitignore <<'EOF'
# 9.2 GB of cached SEC downloads and derived databases. Rebuildable from
# scratch with `make all`; never belongs in git.
data/
logs/
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.duckdb
*.parquet
EOF
git add .gitignore && git commit -m "chore: gitignore before anything else"
```

- [ ] **Step 2: Verify nothing large is staged**

```bash
git add -A
git status --short | wc -l
du -sh $(git diff --cached --name-only | head -50) 2>/dev/null | tail -1
```
Expected: file count in the dozens, total size in single-digit MB. If `data/`
appears, stop and fix `.gitignore`.

- [ ] **Step 3: Write the README**

```markdown
# ma-signals

Predicting corporate acquisitions from free public filings.

**13.81% of the weekly top-25 are acquired within a year, against a 2.44%
base rate — 5.65x better than chance**, averaged over three held-out years
(range 11.40–15.15%). Operating companies only; blank-cheque shells excluded.

Everything here is built from free sources. No CRSP, no Compustat, no SDC.
Clone it and you can reproduce every number.

![hit rate](docs/figures/hit_rate.png)

## What it does

Turns 11.6 million SEC EDGAR filings into a 4.1M row company-week panel and
ranks every listed company each week by how likely it is to be acquired in
the next twelve months.

## Results

| | Precision @25/wk | Lift |
|---|---|---|
| Operating companies | 13.81% | 5.65x |
| Including de-SPACs | 27.58% | 9.40x |

Read [PAPER.md](PAPER.md) for the full argument, including three negative
results and four errors caught during development.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e .
make all          # ~4 hours, ~9 GB downloaded, mostly SEC rate limiting
make paper        # regenerate figures
```

`make all` is resumable — every download is cached, so an interrupted run
picks up where it stopped.

## What it is not

A trading signal. It is wrong about 86% of the time on any individual
company, and without price data the economic value is untested. Palepu (1986)
showed takeover prediction does not earn abnormal returns; nothing here
challenges that.

## Licence

MIT.
```

- [ ] **Step 4: Write the Makefile**

```makefile
PY := .venv/bin/python

.PHONY: all index insider fund panel features paper test clean

all: index insider fund panel features

index:    ; $(PY) scripts/build_dataset.py index
insider:  ; $(PY) scripts/build_dataset.py insider
fund:     ; $(PY) scripts/reload_fund.py
panel:    ; $(PY) scripts/build_dataset.py panel
features: ; $(PY) scripts/make_features.py

paper:
	$(PY) scripts/make_charts.py
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest -q

# Removes derived data but keeps data/raw, so a rebuild does not re-download.
clean:
	rm -f data/*.duckdb data/features.parquet
```

- [ ] **Step 5: Pin dependencies and add the licence**

```bash
.venv/bin/pip freeze > requirements-lock.txt
curl -s https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt \
  | sed "s/{{ year }}/2026/; s/{{ organization }}/Alban Maurel/" > LICENSE
```

- [ ] **Step 6: Correct `docs/FINAL_RESULTS.md`**

It currently states 27.52% / 9.37× from the contaminated single split. Replace
the headline block with:

```markdown
## Headline

**Operating companies (SPACs excluded): 13.81% precision at top-25/week,
5.65x lift**, mean of three held-out years (11.40% / 15.15% / 14.87%).

Including de-SPACs: 27.58%, 9.40x. Reported separately — a blank-cheque
vehicle merging is its stated purpose, not a prediction.

Earlier drafts quoted 27.52% from a single split whose test period was also
used for early stopping. That figure is retracted.
```

- [ ] **Step 7: Run tests and commit**

```bash
.venv/bin/python -m pytest -q
git add -A
git commit -m "feat: public repo shell with README, Makefile and corrected results"
```
Expected: all tests pass.

- [ ] **Step 8: Push**

```bash
gh repo create ma-signals --public --source=. --remote=origin --push
```

---

## Deliberately not in this plan

- **No new modelling.** Every number is already measured; this is packaging.
- **No CRSP.** Separate private repo if it ever happens — WRDS forbids
  redistribution, and mixing them would make the public repo unpublishable.
- **The forward prediction log** — worth more than any of this, but it is a
  weekly cron and a public results page, which is its own plan.
- **Full LaTeX paper.** `PAPER.md` is an outline; converting to a journal
  format is premature before SSRN.

## Self-review notes

- The five stale figures are all replaced in Task 1; `fig_mistakes` is deleted
  rather than corrected because all three of its bars were retracted values.
- `tests/test_charts.py` asserts the retracted numbers cannot reappear, which
  is the failure mode most likely to recur.
- Task 3 Step 1 writes `.gitignore` before any `git add`, because 9.2 GB in
  git history cannot be removed without a rewrite.
