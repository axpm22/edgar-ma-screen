# Model Stress Test and Buyer–Target Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attack both existing models with the four unrun stress tests, then build the
project's first (acquirer, target) pair table and measure whether the two models line
up on real deals — and what deal details are predictable before announcement.

**Architecture:** EDGAR double-indexes a deal filing under both parties, so the same
accession number appears in `master.idx` under two CIKs. The indexes are already
cached on disk, which makes a free (acquirer, target) pair table possible with zero
downloads. Orientation ("which of the two is the target") uses the project's existing
discriminator — a target stops filing, an acquirer does not — validated against
`tender.duckdb`, whose subject CIKs were parsed independently from SEC headers.
That pair table is the asset every alignment and detail question needs.

**Tech Stack:** Python 3, polars, duckdb, LightGBM, statsmodels, matplotlib.
Existing package `src/deal`, existing scripts in `scripts/`.

## Global Constraints

Copied verbatim from `docs/HANDOFF_PROMPT.md`. Every task's requirements include these.

1. **Never evaluate with global top-k over company-weeks.** Use `screen.weekly_precision`, which ranks within each week.
2. **Never split randomly over rows.** Use `splits.grouped` (by company) or `splits.by_time`.
3. **Never use the test period for early stopping.** Three-way temporal split: train, validation, then a test set touched once.
4. **Right-censoring is real.** Panel ends 2026-07-27; with a 52-week label, rows after 2025-07-28 cannot have observed outcomes. Truncate the test window.
5. **Watch for self-referential features.** Anything that encodes the label's own history must be reported separately.
6. **Cluster standard errors by company.** Naive errors measured 4× too small.
7. **±2pp is the noise band** at two seeds. Do not narrate a 1pp difference as a finding.
8. **Compute discipline:** 2 threads (pinned in `deal/__init__.py`), one model per process, `gc.collect()` between fits. Multi-fit single processes have been OOM-killed twice.

Additional constraints for this plan:

9. **Every new script writes its numbers to `data/<name>.json`.** Charts read JSON, never re-fit.
10. **Every new pair-level or matching result must be reported at embargoes 0, 4 and 13 weeks.** A result that collapses under embargo was reading the announcement.
11. **Measured runtime budget:** one LightGBM fit at 25% sample / 200 rounds = **11s** on this machine (8 cores, 16 GB). Any task expecting more than ~30 fits states its fit count in its own header comment.
12. **Existing measured-and-rejected ideas must not be re-tested:** industry-relative ratios (−1.6pp), Loughran-McDonald sentiment (null), Palepu and Ambrose–Megginson variables (+0.08pp), shelf/prospectus buyer features (+1.65pp), trimming the feature set (hurts), shortening the training window (hurts).

## File Structure

| File | Responsibility |
|---|---|
| `src/deal/load_pairs.py` | Parse cached `master.idx`, group by accession, orient, episode-collapse. Pure functions + a `build(con)` entry point. |
| `scripts/load_pairs_run.py` | Driver: builds `data/pairs.duckdb`, prints orientation validation. |
| `tests/test_load_pairs.py` | Orientation and collapse logic, on hand-built rows. No network, no big files. |
| `scripts/pair_scores.py` | Fit target and buyer models once per test year, write `data/pair_scores.parquet` (`cik`, `week`, `p_target`, `p_buyer`). Every downstream task reads this instead of re-fitting. |
| `scripts/stress_pairs.py` | Stage-per-invocation stress tests: `buyerperm`, `hazard`, `size`, `tender`. Appends to `data/stress_pairs.json`. |
| `scripts/alignment.py` | Do the two scores line up on real pairs? Joint precision, embargo decay. |
| `scripts/matching.py` | Rank the true acquirer against sampled counterfactual buyers. |
| `scripts/deal_details.py` | Predict deal structure, relative size, and completion-vs-termination. |
| `scripts/make_charts.py` | Extend with the six figures. Existing house style. |
| `docs/STRESS_RESULTS.md` | Written last, from the JSON. |

---

### Task 1: Free (acquirer, target) pair table

**Files:**
- Create: `src/deal/load_pairs.py`
- Create: `scripts/load_pairs_run.py`
- Test: `tests/test_load_pairs.py`

**Interfaces:**
- Consumes: `deal.universe.parse_master_idx`, `deal.fetch.cache_path`, `deal.config.IDX_URL`, `deal.universe.quarters`.
- Produces:
  - `ACCESSION_RE` — compiled regex matching `\d{10}-\d{2}-\d{6}`.
  - `accession_of(filename: str) -> str | None`
  - `group_filings(rows: list[dict], forms: set[str]) -> dict[str, dict]` — accession → `{"ciks": set[str], "date": dt.date, "form": str}`.
  - `orient(ciks: set[str], date: dt.date, delisted: dict[str, dt.date | None], survive_days: int = 270) -> tuple[str, str] | None` — returns `(target_cik, acquirer_cik)` or `None` when ambiguous.
  - `collapse(pairs: list[dict]) -> list[dict]` — episode-collapse repeated filings for the same (target, acquirer) within 365 days.
  - `build(con, delisted: dict, start_year: int = 2016, end_year: int = 2026) -> dict` — creates table `deal_pairs`, returns counts.
  - Table `deal_pairs(target_cik VARCHAR, acquirer_cik VARCHAR, first_ts DATE, last_ts DATE, form VARCHAR, n_filings INTEGER, PRIMARY KEY (target_cik, acquirer_cik, first_ts))`.

**Background — why this works, for the implementer**

`master.idx` is a pipe-delimited quarterly index of every EDGAR filing, already cached
under `data/raw/sec` (43 quarters, zero network needed). A deal filing is indexed once
per party, with the *same accession number* in the filename. Measured on this cache:

| Form | Accessions | With exactly 2 CIKs |
|---|---|---|
| 425 | 44,899 | 17,532 |
| SC TO-T | 616 | 611 |
| SC 13E3 | 439 | 56 |

18,199 two-CIK accessions in total; 13,852 have both parties present in
`features.parquet`. `DEFM14A` is always single-filer and yields no pair — do not
include it. `S-4` co-registrant counts run up to 332 CIKs on one accession (a bank
registering many subsidiaries), so `S-4` is excluded here too.

Orientation uses the discriminator already established in `src/deal/clean_labels.py`:
a target stops filing within 270 days, an acquirer does not. Validated against the 191
SC TO-T pairs whose true subject CIK was parsed independently from SEC headers into
`tender.duckdb`: **187/191 = 97.9% agreement.**

- [ ] **Step 1: Write the failing test**

Create `tests/test_load_pairs.py`:

```python
import datetime as dt

from deal import load_pairs


def test_accession_of_extracts_from_filename():
    fn = "edgar/data/1659587/0001659587-17-000009.txt"
    assert load_pairs.accession_of(fn) == "0001659587-17-000009"
    assert load_pairs.accession_of("edgar/data/1/no-accession.txt") is None


def test_group_filings_keeps_only_wanted_forms_and_earliest_date():
    rows = [
        {"cik": "1", "form": "425", "file_date": dt.date(2020, 3, 5),
         "filename": "a/0000000001-20-000001.txt"},
        {"cik": "2", "form": "425", "file_date": dt.date(2020, 3, 4),
         "filename": "b/0000000001-20-000001.txt"},
        {"cik": "3", "form": "10-K", "file_date": dt.date(2020, 3, 4),
         "filename": "c/0000000002-20-000002.txt"},
    ]
    g = load_pairs.group_filings(rows, {"425"})
    assert set(g) == {"0000000001-20-000001"}
    assert g["0000000001-20-000001"]["ciks"] == {"1", "2"}
    # Earliest date across both index rows -- the announcement, not the last amendment.
    assert g["0000000001-20-000001"]["date"] == dt.date(2020, 3, 4)


def test_orient_picks_the_party_that_stops_filing():
    d = dt.date(2020, 1, 1)
    delisted = {"T": dt.date(2020, 6, 1), "A": dt.date(2026, 1, 1)}
    assert load_pairs.orient({"T", "A"}, d, delisted) == ("T", "A")


def test_orient_returns_none_when_both_or_neither_stop():
    d = dt.date(2020, 1, 1)
    both = {"X": dt.date(2020, 6, 1), "Y": dt.date(2020, 7, 1)}
    assert load_pairs.orient({"X", "Y"}, d, both) is None
    neither = {"X": dt.date(2026, 1, 1), "Y": dt.date(2026, 1, 1)}
    assert load_pairs.orient({"X", "Y"}, d, neither) is None


def test_orient_needs_exactly_two_parties():
    d = dt.date(2020, 1, 1)
    assert load_pairs.orient({"X"}, d, {"X": None}) is None
    assert load_pairs.orient({"X", "Y", "Z"}, d, {}) is None


def test_collapse_merges_repeat_filings_within_a_year():
    rows = [
        {"target_cik": "T", "acquirer_cik": "A", "date": dt.date(2020, 1, 1),
         "form": "425"},
        {"target_cik": "T", "acquirer_cik": "A", "date": dt.date(2020, 4, 1),
         "form": "425"},
        # More than 365 days later: a separate episode.
        {"target_cik": "T", "acquirer_cik": "A", "date": dt.date(2022, 6, 1),
         "form": "425"},
    ]
    out = load_pairs.collapse(rows)
    assert len(out) == 2
    assert out[0]["first_ts"] == dt.date(2020, 1, 1)
    assert out[0]["last_ts"] == dt.date(2020, 4, 1)
    assert out[0]["n_filings"] == 2
    assert out[1]["n_filings"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_load_pairs.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `AttributeError: module 'deal.load_pairs' has no attribute ...`

- [ ] **Step 3: Write the implementation**

Create `src/deal/load_pairs.py`:

```python
"""(Acquirer, target) pairs, free, from the already-cached EDGAR indexes.

`deals.acquirer` has been NULL since the project started, because master.idx
carries no counterparty column. It does not need one: EDGAR indexes a deal
filing once per PARTY, and both rows carry the same accession number in the
filename. An accession appearing under exactly two CIKs is a two-party deal
filing, and the two CIKs are the two parties -- recoverable with zero network
requests from files already on disk.

Measured on the 43 cached quarters: 18,199 two-CIK accessions, of which 13,852
have both parties present in features.parquet.

Which of the two is the target uses the discriminator from clean_labels.py --
a target stops filing, an acquirer does not. That rule is checkable here rather
than merely asserted: tender.duckdb's subject CIKs were parsed independently
from SEC-HEADER blocks, and on the 191 SC TO-T pairs where both sources speak,
the rule agrees 187 times (97.9%).

DEFM14A is excluded: it is always single-filer and yields no pair. S-4 is
excluded too -- one accession carries up to 332 co-registrant CIKs when a bank
registers its subsidiaries, so "exactly two" does not identify a deal there.
"""
import datetime as dt
import re

from . import config, fetch, universe

# Forms EDGAR indexes under both parties. 425 = business-combination
# communication, SC TO-T = third-party tender offer, SC 13E3 = going private.
PAIR_FORMS = {"425", "SC TO-T", "SC 13E3"}

# Same window clean_labels uses: deals close 3-9 months after announcement.
SURVIVE_DAYS = 270

# Repeat 425s across one deal are one episode, not fifty. A year apart means a
# genuinely separate approach.
EPISODE_DAYS = 365

ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")

SCHEMA = """
CREATE TABLE IF NOT EXISTS deal_pairs (
    target_cik   VARCHAR,
    acquirer_cik VARCHAR,
    first_ts     DATE,
    last_ts      DATE,
    form         VARCHAR,
    n_filings    INTEGER,
    PRIMARY KEY (target_cik, acquirer_cik, first_ts)
);
"""


def accession_of(filename: str) -> str | None:
    m = ACCESSION_RE.search(filename.rsplit("/", 1)[-1])
    return m.group(1) if m else None


def group_filings(rows: list[dict], forms: set[str]) -> dict[str, dict]:
    """accession -> {ciks, date, form}. Date is the EARLIEST index row.

    Both parties' index rows normally share a date, but amendments and late
    acceptance can differ by a day; the earliest is the announcement.
    """
    out: dict[str, dict] = {}
    for r in rows:
        if r["form"] not in forms:
            continue
        acc = accession_of(r["filename"])
        if not acc:
            continue
        rec = out.setdefault(acc, {"ciks": set(), "date": r["file_date"],
                                   "form": r["form"]})
        rec["ciks"].add(r["cik"])
        rec["date"] = min(rec["date"], r["file_date"])
    return out


def orient(ciks: set[str], date: dt.date, delisted: dict[str, dt.date | None],
           survive_days: int = SURVIVE_DAYS) -> tuple[str, str] | None:
    """(target, acquirer), or None when the rule cannot separate them.

    Ambiguity is common and is returned as None rather than guessed: 4,678 of
    14,155 in-universe pairs have both or neither party stopping. Guessing
    would put acquirers in the target column, which is the exact error this
    project already caught once.
    """
    if len(ciks) != 2:
        return None
    a, b = sorted(ciks)
    cutoff = date + dt.timedelta(days=survive_days)

    def stops(cik: str) -> bool:
        d = delisted.get(cik)
        return bool(d and d <= cutoff)

    sa, sb = stops(a), stops(b)
    if sa == sb:
        return None
    return (a, b) if sa else (b, a)


def collapse(pairs: list[dict], episode_days: int = EPISODE_DAYS) -> list[dict]:
    """One row per (target, acquirer) episode, not per filing."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for p in pairs:
        by_key.setdefault((p["target_cik"], p["acquirer_cik"]), []).append(p)

    out = []
    for (t, a), group in by_key.items():
        group.sort(key=lambda p: p["date"])
        cur = None
        for p in group:
            if cur and (p["date"] - cur["last_ts"]).days <= episode_days:
                cur["last_ts"] = p["date"]
                cur["n_filings"] += 1
                continue
            if cur:
                out.append(cur)
            cur = {"target_cik": t, "acquirer_cik": a, "first_ts": p["date"],
                   "last_ts": p["date"], "form": p["form"], "n_filings": 1}
        if cur:
            out.append(cur)
    return sorted(out, key=lambda r: (r["target_cik"], r["acquirer_cik"],
                                      r["first_ts"]))


def index_rows(start_year: int = 2016, end_year: int = 2026) -> list[dict]:
    """Every cached quarterly index row. Cache-only: no network requests."""
    rows = []
    for y, q in universe.quarters(start_year, end_year):
        p = fetch.cache_path("sec", config.IDX_URL.format(year=y, q=q))
        if not p.exists():
            continue
        rows.extend(universe.parse_master_idx(p.read_bytes()))
    return rows


def build(con, delisted: dict[str, dt.date | None],
          start_year: int = 2016, end_year: int = 2026) -> dict:
    """Create and fill deal_pairs. Returns counts for the caller to print."""
    con.execute(SCHEMA)
    con.execute("DELETE FROM deal_pairs")

    grouped = group_filings(index_rows(start_year, end_year), PAIR_FORMS)
    two_party = {k: v for k, v in grouped.items() if len(v["ciks"]) == 2}

    oriented, ambiguous = [], 0
    for v in two_party.values():
        o = orient(v["ciks"], v["date"], delisted)
        if o is None:
            ambiguous += 1
            continue
        oriented.append({"target_cik": o[0], "acquirer_cik": o[1],
                         "date": v["date"], "form": v["form"]})

    episodes = collapse(oriented)
    if episodes:
        con.executemany(
            "INSERT OR IGNORE INTO deal_pairs VALUES "
            "($target_cik, $acquirer_cik, $first_ts, $last_ts, $form, "
            "$n_filings)", episodes)
    return {"accessions": len(grouped), "two_party": len(two_party),
            "oriented": len(oriented), "ambiguous": ambiguous,
            "episodes": len(episodes)}


def validate_orientation(con, tender_con) -> dict:
    """Check the orientation rule against independently-parsed SC TO-T subjects.

    tender.duckdb's CIKs came from the SUBJECT COMPANY block of the SEC header,
    a different file and a different parser. Agreement is evidence the rule
    works; disagreement is a reason not to ship it.
    """
    truth = {r[0] for r in tender_con.execute(
        "SELECT DISTINCT cik FROM tender_offers").fetchall()}
    rows = con.execute(
        "SELECT target_cik, acquirer_cik FROM deal_pairs "
        "WHERE form = 'SC TO-T'").fetchall()
    checked = agree = 0
    for target, acquirer in rows:
        # Only pairs where exactly one side is a known subject are informative.
        if (target in truth) == (acquirer in truth):
            continue
        checked += 1
        agree += target in truth
    return {"checked": checked, "agree": agree,
            "pct": 100.0 * agree / checked if checked else 0.0}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_load_pairs.py -v
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Write the driver**

Create `scripts/load_pairs_run.py`:

```python
"""Build data/pairs.duckdb from the cached EDGAR indexes. No network.

    .venv/bin/python scripts/load_pairs_run.py

Runtime ~40s, dominated by parsing 43 quarterly index files.
"""
import duckdb

from deal import load_pairs

con = duckdb.connect("data/pairs.duckdb")
meta = duckdb.connect("data/deal.duckdb", read_only=True)
delisted = {r[0]: r[1] for r in
            meta.execute("SELECT cik, delisted FROM universe").fetchall()}

counts = load_pairs.build(con, delisted)
print(f"two-party accessions {counts['two_party']:,}  "
      f"oriented {counts['oriented']:,}  ambiguous {counts['ambiguous']:,}")
print(f"collapsed to {counts['episodes']:,} deal episodes")

tender = duckdb.connect("data/tender.duckdb", read_only=True)
v = load_pairs.validate_orientation(con, tender)
print(f"orientation vs tender.duckdb subjects: "
      f"{v['agree']}/{v['checked']} = {v['pct']:.1f}%")

print("\nby form:")
for form, n in con.execute(
        "SELECT form, count(*) FROM deal_pairs GROUP BY 1 ORDER BY 2 DESC"
).fetchall():
    print(f"  {form:<10} {n:>6,}")

print("\nby year:")
for yr, n in con.execute(
        "SELECT year(first_ts), count(*) FROM deal_pairs GROUP BY 1 ORDER BY 1"
).fetchall():
    print(f"  {yr}  {n:>5,}")

top = con.execute("""
    SELECT acquirer_cik, count(*) n FROM deal_pairs
    GROUP BY 1 ORDER BY 2 DESC LIMIT 5
""").fetchall()
names = {r[0]: r[1] for r in
         meta.execute("SELECT cik, name FROM universe").fetchall()}
print("\nmost acquisitive:")
for cik, n in top:
    print(f"  {names.get(cik, cik):<40} {n:>3} deals")
con.close()
```

- [ ] **Step 6: Run the driver and sanity-check the output**

```bash
.venv/bin/python scripts/load_pairs_run.py 2>&1 | tee logs/pairs.log
```

Expected: orientation validation at or above 95%. **If it comes in below 90%, stop
and report — the whole pairing branch of this plan rests on it.** The "most
acquisitive" list should read like real serial acquirers, not like filing agents;
if a single CIK holds hundreds of deals it is an agent CIK and needs excluding.

- [ ] **Step 7: Commit**

```bash
git add src/deal/load_pairs.py scripts/load_pairs_run.py tests/test_load_pairs.py logs/pairs.log
git commit -m "feat: free (acquirer, target) pair table from cached EDGAR indexes"
```

---

### Task 2: Cached model scores for both models

**Files:**
- Create: `scripts/pair_scores.py`

**Interfaces:**
- Consumes: `data/features.parquet`, `data/buyer_features.parquet`, `scripts/final_stats.HORIZON`, `scripts/select_cv.{PARAMS, ROUNDS, SEEDS, split, spac_ciks}`, `deal.features.FEATURE_COLS`, `deal.feat_buyer.{BUYER_COLS, SELF_REFERENTIAL}`.
- Produces: `data/pair_scores.parquet` with columns `cik`, `week`, `test_year`, `p_target`, `p_buyer`. Tasks 5, 6 and 7 all read this file and never re-fit.

**Why this task exists:** three downstream analyses need the same scores. Fitting
once and caching costs 6 fits (~70s); re-fitting in each script costs 18 and risks
the three analyses silently disagreeing about which model they measured.

Fit count: 3 test years × 2 models × 2 seeds = **12 fits, ~2.5 minutes.**
(`p_target` and `p_buyer` are 2-seed mean predictions.)

- [ ] **Step 1: Write the script**

Create `scripts/pair_scores.py`:

```python
"""Score both models on the three test years, once, and cache to parquet.

Three downstream analyses (alignment, matching, deal details) need per
company-week target and buyer scores. Fitting in each of them would cost three
times the compute and would let them drift apart on model config. This is the
single place the two models are defined for pair work.

Both models use the honest configuration: SPACs excluded, and for the buyer
model the self-referential features (s4_52w, goodwill_to_assets,
intangibles_to_assets) dropped -- they identify acquisitive firms rather than
imminent acquisitions.

    .venv/bin/python scripts/pair_scores.py

6 fits, ~2 minutes.
"""
import gc
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, relabel  # noqa: E402
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks  # noqa: E402

from deal import feat_buyer, features  # noqa: E402

YEARS = (2023, 2024, 2025)
OUT = "data/pair_scores.parquet"


def _fit_predict(tr, va, te, cols):
    """Mean prediction over seeds. One booster alive at a time."""
    acc = np.zeros(te.height)
    for s in SEEDS:
        p = {**PARAMS, "bagging_seed": s, "feature_fraction_seed": s,
             "data_random_seed": s}
        dtr = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                          label=tr["y"].to_pandas())
        dva = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                          label=va["y"].to_pandas())
        b = lgb.train(p, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        acc += np.asarray(
            b.predict(te.select(cols).to_pandas().astype("float32")))
        del b, dtr, dva
        gc.collect()
    return acc / len(SEEDS)


def _target_frame(spac):
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    df = relabel(raw, HORIZON).select(["cik", "week", "y"] + cols)
    del raw
    gc.collect()
    return df.filter(~pl.col("cik").is_in(spac)), cols


def _buyer_frame(spac):
    df = pl.read_parquet("data/buyer_features.parquet").filter(
        ~pl.col("cik").is_in(spac))
    self_ref = set(feat_buyer.SELF_REFERENTIAL)
    base = [c for c in features.FEATURE_COLS
            if c in df.columns and df[c].std() and df[c].std() > 0]
    extra = [c for c in feat_buyer.BUYER_COLS
             if df[c].std() and df[c].std() > 0]
    cols = [c for c in base + extra if c not in self_ref]
    return df.select(["cik", "week", "y"] + cols), cols


def main() -> None:
    spac = spac_ciks()
    out = []

    for name, loader in (("p_target", _target_frame), ("p_buyer", _buyer_frame)):
        df, cols = loader(spac)
        print(f"{name}: {df.height:,} rows, {len(cols)} features, "
              f"label rate {df['y'].mean() * 100:.2f}%", flush=True)
        for yr in YEARS:
            tr, va, te = split(df, yr)
            if not te.height:
                continue
            p = _fit_predict(tr, va, te, cols)
            out.append(te.select(["cik", "week"]).with_columns([
                pl.lit(yr).cast(pl.Int32).alias("test_year"),
                pl.Series(name, p),
            ]))
            print(f"  {yr}: {te.height:,} scored", flush=True)
            del tr, va, te
            gc.collect()
        del df
        gc.collect()

    tgt = pl.concat([o for o in out if "p_target" in o.columns])
    buy = pl.concat([o for o in out if "p_buyer" in o.columns])
    merged = tgt.join(buy, on=["cik", "week", "test_year"], how="inner")
    merged.write_parquet(OUT)
    print(f"\nwrote {OUT}: {merged.height:,} rows")
    print(merged.select(["p_target", "p_buyer"]).describe())
    # The two scores must not be near-duplicates -- if they are, "do they line
    # up" is a question about one model, not two.
    r = np.corrcoef(merged["p_target"].to_numpy(),
                    merged["p_buyer"].to_numpy())[0, 1]
    print(f"corr(p_target, p_buyer) = {r:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/pair_scores.py 2>&1 | tee logs/pair_scores.log
```

Expected: `data/pair_scores.parquet` written, roughly 1.0–1.5M rows.
Report the printed correlation. **If `corr(p_target, p_buyer) > 0.9`, say so
prominently** — it would mean the two models are one model and every alignment
result downstream is measuring a single score against itself.

- [ ] **Step 3: Commit**

```bash
git add scripts/pair_scores.py logs/pair_scores.log
git commit -m "feat: cache target and buyer scores for pair analysis"
```

---

### Task 3: The four unrun stress tests

**Files:**
- Create: `scripts/stress_pairs.py`

**Interfaces:**
- Consumes: `data/features.parquet`, `data/buyer_features.parquet`, `data/tender.duckdb`, `data/pairs.duckdb` (Task 1), `deal.clean_labels`, `scripts/select_cv.{PARAMS, ROUNDS, SEEDS, split, spac_ciks}`.
- Produces: `data/stress_pairs.json`, appended one record per stage. Record shape: `{"test": str, "label": str, ...numbers}`.

**Structure:** one stage per process invocation, matching `scripts/stress_suite.py`.
Do not add a stage that runs all four — that is what got OOM-killed twice.

Fit counts per stage: `buyerperm` 8, `size` 12, `tender` 6, `hazard` 0 fits but one
clustered logit on ~1.2M rows. **Total ~26 fits plus the logit, ~12 minutes.**

- [ ] **Step 1: Write the script**

Create `scripts/stress_pairs.py`:

```python
"""The four stress tests the handoff lists as never run. ONE per invocation.

    .venv/bin/python scripts/stress_pairs.py buyerperm
    .venv/bin/python scripts/stress_pairs.py hazard
    .venv/bin/python scripts/stress_pairs.py size
    .venv/bin/python scripts/stress_pairs.py tender
    .venv/bin/python scripts/stress_pairs.py report

Each appends to data/stress_pairs.json and exits, because LightGBM's arena
memory is not reliably returned to the OS and running several fits' worth of
stages in one process has been killed twice.
"""
import datetime as dt
import gc
import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL, relabel  # noqa: E402
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks  # noqa: E402

from deal import clean_labels, feat_buyer, features, screen  # noqa: E402

RESULTS = Path("data/stress_pairs.json")
YEARS = (2023, 2024, 2025)


def record(test: str, label: str, **kw) -> None:
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    rows.append({"test": test, "label": label, **kw})
    RESULTS.write_text(json.dumps(rows, indent=1, default=str))
    print(f"  {label:<38} " + "  ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in kw.items()), flush=True)


def fit_pred(tr, va, te, cols, seed):
    p = {**PARAMS, "bagging_seed": seed, "feature_fraction_seed": seed,
         "data_random_seed": seed}
    dtr = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                      label=tr["y"].to_pandas())
    dva = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                      label=va["y"].to_pandas())
    b = lgb.train(p, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    out = np.asarray(b.predict(te.select(cols).to_pandas().astype("float32")))
    del b, dtr, dva
    gc.collect()
    return out


def seeded(tr, va, te, cols):
    vals = [screen.weekly_precision(te, fit_pred(tr, va, te, cols, s),
                                    N_EVAL)["precision"] * 100 for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))


def buyer_frame():
    """Buyer features, SPACs out, self-referential columns out."""
    df = pl.read_parquet("data/buyer_features.parquet").filter(
        ~pl.col("cik").is_in(spac_ciks()))
    self_ref = set(feat_buyer.SELF_REFERENTIAL)
    base = [c for c in features.FEATURE_COLS
            if c in df.columns and df[c].std() and df[c].std() > 0]
    extra = [c for c in feat_buyer.BUYER_COLS
             if df[c].std() and df[c].std() > 0]
    cols = [c for c in base + extra if c not in self_ref]
    return df.select(["cik", "week", "y"] + cols), cols


# --------------------------------------------------------------------------- #

def t_buyerperm():
    """Permutation test on the BUYER model. It has never had one.

    Labels shuffle within week, so the null keeps each week's positive count
    and only destroys the feature-label link. 8 fits.
    """
    df, cols = buyer_frame()
    tr, va, te = split(df, 2024)
    real, sd = seeded(tr, va, te, cols)
    record("buyerperm", "buyer real (2024)", real=real, sd=sd,
           base=float(te["y"].mean() * 100), n_feat=len(cols))
    null = []
    for k in range(6):
        sh = tr.with_columns(
            pl.col("y").shuffle(seed=900 + k).over("week").alias("y"))
        p = fit_pred(sh, va, te, cols, 11)
        null.append(screen.weekly_precision(te, p, N_EVAL)["precision"] * 100)
        print(f"    null {k + 1}: {null[-1]:.2f}%", flush=True)
        del sh, p
        gc.collect()
    a = np.array(null)
    record("buyerperm", "buyer null distribution", null_mean=float(a.mean()),
           null_max=float(a.max()), null_sd=float(a.std()),
           p_value=float((np.sum(a >= real) + 1) / (len(a) + 1)),
           beats_all=bool(real > a.max()))


def t_hazard():
    """Clustered-SE hazard model on VERIFIED-TARGET labels.

    The existing inference -- including the ROA-vs-Palepu finding -- was
    computed on raw proxy-filer labels, which contain 581 acquirers and
    survivors. Whether those coefficients survive the clean label is unknown,
    and a coefficient that flips sign is a real finding either way.

    No LightGBM fits; one clustered logit.
    """
    import statsmodels.api as sm
    from final_stats import CONTROLS

    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    panel_end = raw["week"].max()

    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
    con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
    counts = clean_labels.build(con, panel_end)
    print(f"  proxy filings classified: {counts}", flush=True)

    con.register("f", raw.select(["cik", "week"]).to_arrow())
    lab = con.execute(f"""
        SELECT f.cik, f.week, CASE WHEN EXISTS (
            SELECT 1 FROM deals_clean d
            WHERE d.cik = f.cik AND d.outcome = 'target'
              AND f.week <  d.agreement_date
              AND f.week >= d.agreement_date - INTERVAL {HORIZON} WEEK
        ) THEN 1 ELSE 0 END AS yh FROM f
    """).pl()
    j = raw.select(["cik", "week"]).join(lab, on=["cik", "week"], how="left")
    df = raw.with_columns(j["yh"].fill_null(0).cast(pl.Int8).alias("y")) \
            .select(["cik", "week", "y"] + cols)
    del raw, lab, j
    gc.collect()

    tr = df.filter(pl.col("week") < dt.date(2024, 1, 1)).sample(
        fraction=0.3, seed=7)
    del df
    gc.collect()
    X = sm.add_constant(tr.select(cols).to_pandas())
    fit = sm.Logit(tr["y"].to_pandas(), X).fit(
        disp=False, maxiter=300, cov_type="cluster",
        cov_kwds={"groups": tr["cik"].to_pandas()})
    tv = fit.tvalues.drop("const")
    novel = [n for n in tv.index if n not in CONTROLS and abs(tv[n]) > 1.96]
    record("hazard", "verified-target labels", n_rows=tr.height,
           label_rate=float(tr["y"].mean() * 100),
           novel_significant=len(novel))
    for n in sorted(tv.index, key=lambda x: -abs(tv[x]))[:15]:
        record("hazard", f"  {n}", beta=float(fit.params[n]), z=float(tv[n]))


def t_size():
    """Was log_assets the model spotting ACQUIRERS rather than targets?

    log_assets is a top predictor of a label built from merger proxies, and
    merger proxies are filed by buyers too. If the size effect is really an
    acquirer effect, then dropping the 581 known survivors from the positive
    class should flatten it. Three label sets, four configurations. 12 fits.
    """
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    panel_end = raw["week"].max()
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
    con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
    clean_labels.build(con, panel_end)
    con.register("f", raw.select(["cik", "week"]).to_arrow())

    def labelled(outcome_sql):
        lab = con.execute(f"""
            SELECT f.cik, f.week, CASE WHEN EXISTS (
                SELECT 1 FROM deals_clean d WHERE d.cik = f.cik
                  AND {outcome_sql}
                  AND f.week <  d.agreement_date
                  AND f.week >= d.agreement_date - INTERVAL {HORIZON} WEEK
            ) THEN 1 ELSE 0 END AS yh FROM f
        """).pl()
        j = raw.select(["cik", "week"]).join(lab, on=["cik", "week"],
                                             how="left")
        return raw.with_columns(
            j["yh"].fill_null(0).cast(pl.Int8).alias("y")
        ).select(["cik", "week", "y"] + cols).filter(
            ~pl.col("cik").is_in(spac_ciks()))

    size_cols = ["log_assets", "log_float"]
    for name, sql in (("raw proxy filers", "1=1"),
                      ("verified targets", "d.outcome = 'target'"),
                      ("survivors only", "d.outcome = 'survivor'")):
        df = labelled(sql)
        tr, va, te = split(df, 2024)
        if not te.height or not te["y"].sum():
            del df
            gc.collect()
            continue
        # Mean log_assets of positives vs negatives -- the effect itself,
        # before any model is involved.
        pos = te.filter(pl.col("y") == 1)["log_assets"].mean()
        neg = te.filter(pl.col("y") == 0)["log_assets"].mean()
        m, sd = seeded(tr, va, te, cols)
        m2, _ = seeded(tr, va, te, [c for c in cols if c not in size_cols])
        record("size", name, prec=m, sd=sd, without_size=m2,
               size_contributes=m - m2,
               pos_log_assets=float(pos or 0.0),
               neg_log_assets=float(neg or 0.0),
               base=float(te["y"].mean() * 100))
        del df, tr, va, te
        gc.collect()


def t_tender():
    """The target model against 616 tender-offer targets it has never seen.

    Tender offers were excluded from training because master.idx cannot tell
    bidder from target. That exclusion is what makes them a real held-out
    label: no shareholder vote, frequently hostile, usually cash -- a deal type
    the model has not been shown. 6 fits.
    """
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/tender.duckdb' AS t (READ_ONLY)")
    con.register("f", raw.select(["cik", "week"]).to_arrow())
    tlab = con.execute(f"""
        SELECT f.cik, f.week, CASE WHEN EXISTS (
            SELECT 1 FROM t.tender_offers o WHERE o.cik = f.cik
              AND f.week <  o.public_ts
              AND f.week >= o.public_ts - INTERVAL {HORIZON} WEEK
        ) THEN 1 ELSE 0 END AS yh FROM f
    """).pl()
    keys = raw.select(["cik", "week"])
    tender_y = keys.join(tlab, on=["cik", "week"], how="left")["yh"] \
                   .fill_null(0).cast(pl.Int8)

    # Train on the PROXY label, test on the TENDER label. Any company that is
    # a tender target must not also be a training positive, or this is not
    # held out -- so proxy positives keep their own label for training and the
    # tender label is only ever used on the test side.
    train_df = relabel(raw, HORIZON).select(["cik", "week", "y"] + cols)
    test_df = raw.with_columns(tender_y.alias("y")).select(
        ["cik", "week", "y"] + cols)
    del raw
    gc.collect()

    spac = spac_ciks()
    train_df = train_df.filter(~pl.col("cik").is_in(spac))
    test_df = test_df.filter(~pl.col("cik").is_in(spac))

    for yr in YEARS:
        tr, va, _ = split(train_df, yr)
        _, _, te = split(test_df, yr)
        if not te.height or not te["y"].sum():
            continue
        m, sd = seeded(tr, va, te, cols)
        base = float(te["y"].mean() * 100)
        record("tender", f"tender targets {yr}", prec=m, sd=sd, base=base,
               lift=m / base if base else 0.0,
               positives=int(te["y"].sum()))
        del tr, va, te
        gc.collect()


STAGES = {"buyerperm": t_buyerperm, "hazard": t_hazard, "size": t_size,
          "tender": t_tender}


def report():
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    cur = None
    for r in rows:
        if r["test"] != cur:
            cur = r["test"]
            print(f"\n=== {cur.upper()} ===")
        rest = {k: v for k, v in r.items() if k not in ("test", "label")}
        print(f"  {r['label']:<38} " + "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in rest.items()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    report() if cmd == "report" else STAGES[cmd]()
```

- [ ] **Step 2: Run the four stages, each as its own process**

```bash
for s in buyerperm hazard size tender; do .venv/bin/python scripts/stress_pairs.py $s 2>&1 | tee -a logs/stress_pairs.log; done
```

Expected, and what each result means:
- `buyerperm`: real should beat every null draw. If the null tops the real score,
  **the buyer model is leaking and the 26.09% headline is void.**
- `hazard`: report which coefficients change sign or lose significance against
  the contaminated-label run in `logs/final_stats.log`. A flip is the finding.
- `size`: compare `size_contributes` across the three label sets. If size
  contributes on "raw proxy filers" and "survivors only" but not on "verified
  targets", the size effect was an acquirer effect.
- `tender`: lift materially above 1× is the strongest external validation in the
  project. Lift near 1× means the model learned DEFM14A filers, not deal prep.

- [ ] **Step 3: Commit**

```bash
git add scripts/stress_pairs.py data/stress_pairs.json logs/stress_pairs.log
git commit -m "test: permutation on buyer model, clean-label hazard, size decomposition, tender validation"
```

---

### Task 4: Do the two models line up on real deals?

**Files:**
- Create: `scripts/alignment.py`

**Interfaces:**
- Consumes: `data/pair_scores.parquet` (Task 2), `data/pairs.duckdb` (Task 1).
- Produces: `data/alignment.json`.

**The question:** on a week where a real deal was about to be announced, was the
target in the target model's top 25 *and* the acquirer in the buyer model's top 25?
Two independent screens agreeing on a real pair is a different and stronger claim
than either screen's own precision.

**The trap to avoid:** joint hit rate will be low simply because two low-precision
screens rarely intersect. The number that carries information is joint hit rate
**against the product of the two marginals** — that ratio says whether the screens
agree more than chance, which is the actual "do they line up" question.

**Embargo is mandatory here.** Features at the announcement week already contain
the announcement. Report weeks −1, −4, −13 and −26 before `first_ts`.

**Sample size, measured.** `deal_pairs` holds 1,371 episodes, 1,023 with both
parties in `features.parquet`. `pair_scores.parquet` only covers test years
2023–2025, so alignment can only use pairs whose observation week falls in a
scored year — roughly **231 pairs**, and fewer at longer leads, since a 26-week
lead pushes early-2023 announcements out of the scored range. Print `n_pairs` at
every lead and **do not compare two leads whose `n_pairs` differ by more than
about 20%** — that is a composition change, not a lead-time effect.

No fits — reads cached scores. **Runtime under a minute.**

- [ ] **Step 1: Write the script**

Create `scripts/alignment.py`:

```python
"""Do the target screen and the buyer screen agree on real deals?

Each model's own precision says how often its list is right. It says nothing
about whether the two lists point at the same transaction. This measures that
directly: for each real (target, acquirer) pair, was the target in the target
model's top 25 that week AND the acquirer in the buyer model's top 25?

The raw joint rate is uninteresting on its own -- two sparse lists intersect
rarely whatever they encode. The informative quantity is

    joint observed / (target marginal x buyer marginal)

which is 1.0 when the two screens agree only as often as independent screens
would, and above 1.0 when they genuinely co-fire on the same deal.

Everything is reported at several lead times. A result that only appears at
lead 1 is the models reading the announcement, not predicting it.

    .venv/bin/python scripts/alignment.py

No model fits.
"""
import json

import duckdb
import numpy as np
import polars as pl

LEADS = (1, 4, 13, 26)
TOP_N = 25
OUT = "data/alignment.json"


def main() -> None:
    scores = pl.read_parquet("data/pair_scores.parquet")

    # Within-week rank for each model. rank 1 = highest score.
    ranked = scores.with_columns([
        pl.col("p_target").rank("ordinal", descending=True)
          .over("week").alias("r_target"),
        pl.col("p_buyer").rank("ordinal", descending=True)
          .over("week").alias("r_buyer"),
    ])
    week_size = ranked.group_by("week").len().rename({"len": "n_week"})
    ranked = ranked.join(week_size, on="week")

    con = duckdb.connect("data/pairs.duckdb", read_only=True)
    pairs = pl.from_arrow(con.execute(
        "SELECT target_cik, acquirer_cik, first_ts FROM deal_pairs").arrow())
    con.close()

    out = []
    for lead in LEADS:
        # The observation week: `lead` weeks before the announcement.
        obs = pairs.with_columns(
            (pl.col("first_ts").cast(pl.Date)
             - pl.duration(weeks=lead)).dt.truncate("1w").alias("week"))

        t = obs.join(
            ranked.select(["cik", "week", "r_target", "n_week"]),
            left_on=["target_cik", "week"], right_on=["cik", "week"],
            how="inner")
        both = t.join(
            ranked.select(["cik", "week", "r_buyer"]),
            left_on=["acquirer_cik", "week"], right_on=["cik", "week"],
            how="inner")
        if not both.height:
            continue

        hit_t = both["r_target"] <= TOP_N
        hit_b = both["r_buyer"] <= TOP_N
        joint = float((hit_t & hit_b).mean())
        pt, pb = float(hit_t.mean()), float(hit_b.mean())
        expected = pt * pb
        # What a random pair would score, given each week's universe size.
        chance = float((TOP_N / both["n_week"]).mean()) ** 2

        rec = {
            "lead_weeks": lead,
            "n_pairs": both.height,
            "target_in_top25": 100 * pt,
            "buyer_in_top25": 100 * pb,
            "joint": 100 * joint,
            "expected_if_independent": 100 * expected,
            "agreement_ratio": joint / expected if expected else 0.0,
            "chance_joint": 100 * chance,
            "joint_lift_vs_chance": joint / chance if chance else 0.0,
        }
        out.append(rec)
        print(f"lead {lead:>2}w  n={both.height:>5}  "
              f"target {100 * pt:>5.1f}%  buyer {100 * pb:>5.1f}%  "
              f"joint {100 * joint:>5.2f}%  "
              f"(independent would give {100 * expected:.2f}%, "
              f"ratio {rec['agreement_ratio']:.2f}x)", flush=True)

    # Rank correlation between the two scores across real pairs: do good
    # targets get matched to good buyers, beyond the top-25 cutoff?
    lead4 = pairs.with_columns(
        (pl.col("first_ts").cast(pl.Date)
         - pl.duration(weeks=4)).dt.truncate("1w").alias("week"))
    j = lead4.join(ranked.select(["cik", "week", "p_target"]),
                   left_on=["target_cik", "week"],
                   right_on=["cik", "week"], how="inner") \
             .join(ranked.select(["cik", "week", "p_buyer"]),
                   left_on=["acquirer_cik", "week"],
                   right_on=["cik", "week"], how="inner")
    corr = float(np.corrcoef(j["p_target"].to_numpy(),
                             j["p_buyer"].to_numpy())[0, 1]) if j.height else 0.0
    print(f"\ncorr(target score, buyer score) across {j.height} real pairs "
          f"at 4w lead: {corr:+.3f}")

    json.dump({"leads": out, "pair_score_corr_4w": corr, "n_corr": j.height},
              open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/alignment.py 2>&1 | tee logs/alignment.log
```

Report `agreement_ratio` at each lead. **A ratio near 1.0 at every lead is a
negative result and must be reported as one** — it would mean the two screens
carry no joint information about a specific transaction, only about each party
separately. That is a publishable finding, not a failure to hide.

- [ ] **Step 3: Commit**

```bash
git add scripts/alignment.py data/alignment.json logs/alignment.log
git commit -m "feat: measure whether target and buyer screens co-fire on real deals"
```

---

### Task 5: Can we predict WHICH buyer? A matching model

**Files:**
- Create: `scripts/matching.py`

**Interfaces:**
- Consumes: `data/pairs.duckdb`, `data/pair_scores.parquet`, `data/features.parquet`, `data/buyer_features.parquet`, `deal.duckdb:company_sic`.
- Produces: `data/matching.json`.

**Design.** For each real pair `(T, A, first_ts)`, build a candidate set for the
observation week: the true acquirer `A` plus `K = 100` companies sampled from the
same week's universe. Train a LightGBM `lambdarank` with the *pair* as the query
group, so the objective is exactly the thing being measured — ranking the true
acquirer above the counterfactuals.

**Feature rules, and these are where this goes wrong if rushed:**
- Any feature constant across a pair's candidates (anything computed on `T` alone,
  including `p_target`) contributes nothing to within-pair ranking. Include only
  candidate-varying features and *interactions* between `T` and the candidate.
- Split by time, not randomly: a random split would put the same acquirer's
  other deals on both sides.
- Report at embargo 4 and 13 weeks. `sector_deal_intensity` and `peer_deal_13w`
  can encode this very deal.
- Random baseline for K=100 is 1/101 = 0.99%. Quote it beside every number.

**Measured pair counts, which drive the evaluation design.** `deal_pairs` holds
1,371 episodes; **1,023 have both parties in `features.parquet`** and are usable
here. A single 2024 cutoff leaves only 128 test pairs — too thin to compare two
embargoes, since top-1 on 128 pairs carries roughly ±6pp of sampling error.
So use **rolling-origin folds at 2023-01-01 and 2024-01-01, pooling the test
ranks** (231 pairs), matching the pattern already in
`deal.evaluate_robust.rolling_origin`, and **report a bootstrap CI over pairs**
next to every accuracy. A difference between embargoes that is inside the CI is
not a finding.

Fit count: 2 embargoes × 2 folds × 2 seeds = **8 fits on a small frame, ~2 minutes.**

- [ ] **Step 1: Write the script**

Create `scripts/matching.py`:

```python
"""Given a target about to be acquired, can we name the acquirer?

Framed as ranking rather than classification. For each real (target, acquirer)
pair, the true acquirer competes against 100 companies sampled from the same
week's universe, and the model ranks all 101. LightGBM's lambdarank with the
PAIR as query group optimises exactly that ordering.

Two design points carry the result:

1. Features constant within a candidate set cannot rank anything. The target's
   own attributes -- including its target-model score -- are identical across
   all 101 candidates, so they are excluded and only candidate attributes and
   target-candidate INTERACTIONS are used. Getting this wrong produces a model
   that looks trained and ranks at chance.

2. Sector-timing features can encode the deal being predicted. Everything is
   reported at 4- and 13-week embargoes; if accuracy collapses between them the
   model was reading the announcement.

Random baseline with 100 distractors is 1/101 = 0.99%.

    .venv/bin/python scripts/matching.py

4 fits, ~2 minutes.
"""
import gc
import json

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

K_NEG = 100
EMBARGOES = (4, 13)
SEEDS = (11, 22)
OUT = "data/matching.json"

RANK_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 5, 10],
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbosity": -1, "num_threads": 2,
}
ROUNDS = 300

# Candidate-varying attributes only.
CAND_COLS = ["log_assets", "log_float", "cash_to_assets", "leverage",
             "fcf_to_assets", "acq_capacity", "dry_powder", "debt_headroom",
             "shelf_52w", "raise_52w", "shelf_new", "form8k_26w",
             "sector_deal_intensity", "p_buyer"]


def build_candidates(rng, pairs, panel, sic):
    """One block of K_NEG+1 rows per pair. Label 1 on the true acquirer.

    The embargo is already baked into `pairs.obs_week` by the caller.
    """
    by_week = {w: g for w, g in
               panel.partition_by("week", as_dict=True, include_key=True).items()}
    blocks = []
    for row in pairs.iter_rows(named=True):
        wk = row["obs_week"]
        pool = by_week.get((wk,)) or by_week.get(wk)
        if pool is None or pool.height < K_NEG + 1:
            continue
        if row["acquirer_cik"] not in pool["cik"]:
            continue
        truth = pool.filter(pl.col("cik") == row["acquirer_cik"])
        negs = pool.filter(pl.col("cik") != row["acquirer_cik"])
        idx = rng.choice(negs.height, size=K_NEG, replace=False)
        block = pl.concat([truth, negs[idx]])
        blocks.append(block.with_columns([
            pl.Series("rel", [1] + [0] * K_NEG).cast(pl.Int8),
            pl.lit(row["target_cik"]).alias("target_cik"),
            pl.lit(row["first_ts"]).alias("first_ts"),
            pl.lit(f"{row['target_cik']}|{row['first_ts']}").alias("qid"),
        ]))
    if not blocks:
        return None
    df = pl.concat(blocks)

    # Target-candidate interactions: the only features that can express "these
    # two fit together" rather than "this candidate is acquisitive".
    t = sic.rename({"cik": "target_cik", "sic": "t_sic"})
    df = df.join(t, on="target_cik", how="left")
    df = df.join(sic.rename({"sic": "c_sic"}), on="cik", how="left")
    tgt_size = panel.select(["cik", "week", "log_assets"]).rename(
        {"cik": "target_cik", "log_assets": "t_log_assets"})
    df = df.join(tgt_size, left_on=["target_cik", "week"],
                 right_on=["target_cik", "week"], how="left")
    return df.with_columns([
        (pl.col("t_sic").str.slice(0, 2)
         == pl.col("c_sic").str.slice(0, 2)).cast(pl.Int8).alias("same_sic2"),
        (pl.col("t_sic") == pl.col("c_sic")).cast(pl.Int8).alias("same_sic4"),
        (pl.col("log_assets") - pl.col("t_log_assets")).alias("size_gap"),
    ]).with_columns(
        pl.col("size_gap").abs().alias("abs_size_gap")
    ).drop_nulls(subset=["size_gap"])


def true_ranks(model, te, cols):
    """Rank of the true acquirer within each pair's 101 candidates."""
    p = np.asarray(model.predict(te.select(cols).to_pandas().astype("float32")))
    d = te.select(["qid", "rel"]).with_columns(pl.Series("p", p))
    return (d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("qid").alias("r"))
        .filter(pl.col("rel") == 1)["r"].to_numpy())


def summarise(ranks, n_boot=2000, seed=20260801):
    """Accuracy plus a bootstrap CI over pairs.

    The test set is ~231 pairs pooled across two folds, so a top-1 difference
    of a few points between embargoes is inside sampling error. Reporting the
    CI is what stops that being narrated as a finding.
    """
    rng = np.random.default_rng(seed)
    boot = rng.choice(ranks, size=(n_boot, len(ranks)), replace=True)
    return {
        "n_pairs": int(len(ranks)),
        "top1": float(100 * np.mean(ranks <= 1)),
        "top5": float(100 * np.mean(ranks <= 5)),
        "top10": float(100 * np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "top1_ci_lo": float(np.percentile(100 * np.mean(boot <= 1, axis=1), 2.5)),
        "top1_ci_hi": float(np.percentile(100 * np.mean(boot <= 1, axis=1), 97.5)),
        "top10_ci_lo": float(np.percentile(100 * np.mean(boot <= 10, axis=1), 2.5)),
        "top10_ci_hi": float(np.percentile(100 * np.mean(boot <= 10, axis=1), 97.5)),
    }


def main() -> None:
    feats = pl.read_parquet("data/buyer_features.parquet")
    scores = pl.read_parquet("data/pair_scores.parquet").select(
        ["cik", "week", "p_buyer"])
    panel = feats.join(scores, on=["cik", "week"], how="inner")
    keep = ["cik", "week"] + [c for c in CAND_COLS if c in panel.columns]
    panel = panel.select(keep)
    del feats
    gc.collect()

    con = duckdb.connect("data/deal.duckdb", read_only=True)
    sic = pl.from_arrow(con.execute(
        "SELECT cik, sic FROM company_sic").arrow())
    con.close()
    con = duckdb.connect("data/pairs.duckdb", read_only=True)
    all_pairs = pl.from_arrow(con.execute(
        "SELECT target_cik, acquirer_cik, first_ts FROM deal_pairs").arrow())
    con.close()

    cols = [c for c in CAND_COLS if c in panel.columns] + \
           ["same_sic2", "same_sic4", "size_gap", "abs_size_gap"]

    results = []
    for emb in EMBARGOES:
        rng = np.random.default_rng(20260801)
        pairs = all_pairs.with_columns(
            (pl.col("first_ts").cast(pl.Date)
             - pl.duration(weeks=emb)).dt.truncate("1w").alias("obs_week"))
        cand = build_candidates(rng, pairs, panel, sic)
        if cand is None:
            print(f"embargo {emb}w: no candidate blocks")
            continue

        # Rolling-origin folds, pooled. A single 2024 cutoff leaves 128 test
        # pairs, too few to tell two embargoes apart.
        import datetime as dt
        pooled, train_pairs, imp = [], 0, []
        for cut in (dt.date(2023, 1, 1), dt.date(2024, 1, 1)):
            end = dt.date(cut.year + 1, 1, 1) if cut.year == 2023 else None
            tr = cand.filter(pl.col("first_ts") < cut).sort("qid")
            te = cand.filter(pl.col("first_ts") >= cut)
            if end is not None:
                te = te.filter(pl.col("first_ts") < end)
            te = te.sort("qid")
            if not tr.height or not te.height:
                continue
            train_pairs = max(train_pairs, int(tr["qid"].n_unique()))
            per_seed = []
            for s in SEEDS:
                g = tr.group_by("qid", maintain_order=True).len()["len"].to_list()
                d = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                                label=tr["rel"].to_pandas(), group=g)
                m = lgb.train({**RANK_PARAMS, "bagging_seed": s,
                               "feature_fraction_seed": s,
                               "data_random_seed": s}, d,
                              num_boost_round=ROUNDS)
                per_seed.append(true_ranks(m, te, cols))
                if s == SEEDS[0] and not imp:
                    imp = sorted(zip(m.feature_name(),
                                     m.feature_importance("gain")),
                                 key=lambda t: -t[1])[:6]
                del m, d
                gc.collect()
            # Average the rank across seeds, then pool folds.
            pooled.append(np.mean(per_seed, axis=0))
            del tr, te
            gc.collect()

        if not pooled:
            print(f"embargo {emb}w: empty split")
            continue
        rec = {"embargo_weeks": emb, "train_pairs": train_pairs,
               "random_baseline_top1": 100 / (K_NEG + 1),
               "top_features": [n for n, _ in imp],
               **summarise(np.concatenate(pooled))}
        results.append(rec)
        print(f"embargo {emb:>2}w  test pairs {rec['n_pairs']:>4}  "
              f"top1 {rec['top1']:>5.1f}% "
              f"[{rec['top1_ci_lo']:.1f}-{rec['top1_ci_hi']:.1f}]  "
              f"top10 {rec['top10']:>5.1f}% "
              f"[{rec['top10_ci_lo']:.1f}-{rec['top10_ci_hi']:.1f}]  "
              f"MRR {rec['mrr']:.3f}  median rank {rec['median_rank']:.0f} "
              f"(random top1 = {rec['random_baseline_top1']:.2f}%)",
              flush=True)
        print(f"           top features: {', '.join(rec['top_features'])}",
              flush=True)
        del cand, tr, te
        gc.collect()

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/matching.py 2>&1 | tee logs/matching.log
```

Expected: top-1 above 0.99% is signal; the honest headline is top-10 and median
rank. **If top-1 at 13-week embargo is at or below 1%, report the model as failing
at 13 weeks** — that is the result, and it bounds how early the pairing is knowable.
Check `top_features`: if `sector_deal_intensity` dominates, the model may be finding
"a deal is happening in this sector" rather than "this buyer for this target", so
re-run once with that column removed from `CAND_COLS` and report both.

- [ ] **Step 3: Commit**

```bash
git add scripts/matching.py data/matching.json logs/matching.log
git commit -m "feat: rank the true acquirer against counterfactual buyers"
```

---

### Task 6: Exact deal details

**Files:**
- Create: `scripts/deal_details.py`

**Interfaces:**
- Consumes: `data/pairs.duckdb`, `data/pair_scores.parquet`, `data/features.parquet`, `deal.duckdb:{deals, universe, company_sic}`, `deal.clean_labels`.
- Produces: `data/deal_details.json`.

**Four details, each free and each with an honest baseline:**

1. **Structure — stock vs cash.** `form = '425'` or an S-4 exists for the acquirer
   near the deal ⇒ stock consideration; `SC TO-T` with no 425 ⇒ cash tender.
   Baseline: the majority class rate.
2. **Relative size.** Predict `sign(log_assets(A) − log_assets(T))` and the size
   ratio bucket. Baseline: always predicting "acquirer is bigger".
3. **Completion vs termination.** From `clean_labels`: a proxy filer that keeps
   filing 270 days later is a deal that did not complete. Never tested in this
   project, and it is the most decision-relevant detail here.
4. **Lead time.** For targets the model ranked in the top 25, how many weeks
   before `first_ts` did the rank first cross the threshold? Descriptive, no model.

Fit count: 3 small classifiers × 2 seeds = **6 fits on frames of a few thousand
rows, under a minute.**

- [ ] **Step 1: Write the script**

Create `scripts/deal_details.py`:

```python
"""Beyond "a deal": what can be said about WHICH deal, before it is announced?

Four details, each with the baseline it has to beat stated next to it, because
three of the four have a majority class big enough to look like skill.

  structure    stock (425/S-4) vs cash tender (SC TO-T). Baseline: majority.
  size order   is the acquirer bigger than the target? Baseline: majority.
  completion   does the deal close, or does the target keep filing? This is
               the one nobody has measured here, and the one that matters --
               a screen that flags deals which then break is worth less than
               its precision suggests.
  lead time    descriptive: how many weeks before announcement the target
               first entered the top 25.

    .venv/bin/python scripts/deal_details.py

6 fits on small frames, under a minute.
"""
import datetime as dt
import gc
import json

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

SEEDS = (11, 22)
CUT = dt.date(2024, 1, 1)
OUT = "data/deal_details.json"

PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 15, "min_data_in_leaf": 30, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbosity": -1, "num_threads": 2,
}
ROUNDS = 200


def fit_score(df, cols, ycol):
    """Time-split accuracy and AUC against the majority-class baseline."""
    tr = df.filter(pl.col("first_ts") < CUT)
    te = df.filter(pl.col("first_ts") >= CUT)
    if tr.height < 50 or te.height < 20 or te[ycol].n_unique() < 2:
        return None
    majority = max(float(te[ycol].mean()), 1 - float(te[ycol].mean()))
    accs, aucs = [], []
    for s in SEEDS:
        d = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                        label=tr[ycol].to_pandas())
        m = lgb.train({**PARAMS, "bagging_seed": s, "feature_fraction_seed": s,
                       "data_random_seed": s}, d, num_boost_round=ROUNDS)
        p = np.asarray(m.predict(te.select(cols).to_pandas().astype("float32")))
        y = te[ycol].to_numpy()
        accs.append(float(np.mean((p > 0.5).astype(int) == y)))
        order = np.argsort(p)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, len(p) + 1)
        n1, n0 = y.sum(), len(y) - y.sum()
        aucs.append(float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
                    if n1 and n0 else 0.5)
        del m, d
        gc.collect()
    return {"n_train": tr.height, "n_test": te.height,
            "accuracy": 100 * float(np.mean(accs)),
            "majority_baseline": 100 * majority,
            "auc": float(np.mean(aucs)),
            "positive_rate": 100 * float(te[ycol].mean())}


def main() -> None:
    con = duckdb.connect("data/pairs.duckdb", read_only=True)
    pairs = pl.from_arrow(con.execute(
        "SELECT target_cik, acquirer_cik, first_ts, form, n_filings "
        "FROM deal_pairs").arrow())
    con.close()

    feats = pl.read_parquet("data/features.parquet")
    meta = duckdb.connect("data/deal.duckdb", read_only=True)
    sic = pl.from_arrow(meta.execute(
        "SELECT cik, sic FROM company_sic").arrow())
    uni = pl.from_arrow(meta.execute(
        "SELECT cik, delisted FROM universe").arrow())
    meta.close()

    # Observation week: 4 weeks before announcement, for both parties.
    obs = pairs.with_columns(
        (pl.col("first_ts").cast(pl.Date)
         - pl.duration(weeks=4)).dt.truncate("1w").alias("week"))

    side_cols = ["log_assets", "log_float", "cash_to_assets", "leverage",
                 "fcf_to_assets", "operating_margin", "revenue_growth",
                 "goodwill_to_assets", "form8k_26w", "sc13d_52w",
                 "sector_deal_intensity", "activist_reach"]
    side = feats.select(["cik", "week"] + [c for c in side_cols
                                           if c in feats.columns])

    df = (obs
          .join(side.rename({c: f"t_{c}" for c in side.columns
                             if c not in ("cik", "week")})
                    .rename({"cik": "target_cik"}),
                on=["target_cik", "week"], how="inner")
          .join(side.rename({c: f"a_{c}" for c in side.columns
                             if c not in ("cik", "week")})
                    .rename({"cik": "acquirer_cik"}),
                on=["acquirer_cik", "week"], how="inner")
          .join(sic.rename({"cik": "target_cik", "sic": "t_sic"}),
                on="target_cik", how="left")
          .join(sic.rename({"cik": "acquirer_cik", "sic": "a_sic"}),
                on="acquirer_cik", how="left"))

    df = df.with_columns([
        (pl.col("t_sic").str.slice(0, 2)
         == pl.col("a_sic").str.slice(0, 2)).cast(pl.Int8).alias("same_sic2"),
        (pl.col("a_log_assets") - pl.col("t_log_assets")).alias("size_gap"),
    ])
    cols = [c for c in df.columns
            if c.startswith(("t_", "a_")) and c not in ("t_sic", "a_sic")] \
        + ["same_sic2"]
    cols = [c for c in cols if df[c].dtype.is_numeric()]

    out = {}

    # 1. Structure. 425 => securities issued => stock consideration.
    d1 = df.with_columns((pl.col("form") == "425").cast(pl.Int8).alias("yv"))
    out["structure_stock_vs_cash"] = fit_score(
        d1, [c for c in cols if c != "size_gap"], "yv")

    # 2. Size order. size_gap is the answer, so it cannot be an input.
    d2 = df.with_columns((pl.col("size_gap") > 0).cast(pl.Int8).alias("yv"))
    out["acquirer_is_bigger"] = fit_score(
        d2, [c for c in cols if c not in ("size_gap", "a_log_assets",
                                          "t_log_assets")], "yv")

    # 3. Completion. A target that is still filing 270 days later did not get
    # acquired -- the deal broke, or it was never a target.
    d3 = (df.join(uni.rename({"cik": "target_cik"}), on="target_cik",
                  how="left")
            .with_columns(
                (pl.col("delisted").cast(pl.Date)
                 <= pl.col("first_ts").cast(pl.Date)
                 + pl.duration(days=270)).fill_null(False)
                .cast(pl.Int8).alias("yv")))
    out["deal_completes"] = fit_score(d3, cols, "yv")

    # 4. Lead time, descriptive.
    scores = pl.read_parquet("data/pair_scores.parquet")
    ranked = scores.with_columns(
        pl.col("p_target").rank("ordinal", descending=True)
          .over("week").alias("r"))
    lead = []
    for row in pairs.iter_rows(named=True):
        hits = ranked.filter(
            (pl.col("cik") == row["target_cik"])
            & (pl.col("week") < row["first_ts"])
            & (pl.col("week") >= row["first_ts"] - dt.timedelta(weeks=52))
            & (pl.col("r") <= 25))
        if hits.height:
            lead.append((row["first_ts"] - hits["week"].max()).days / 7.0)
    out["lead_time_weeks"] = {
        "n_pairs_ever_flagged": len(lead),
        "n_pairs_total": pairs.height,
        "flagged_pct": 100.0 * len(lead) / max(pairs.height, 1),
        "median": float(np.median(lead)) if lead else None,
        "p25": float(np.percentile(lead, 25)) if lead else None,
        "p75": float(np.percentile(lead, 75)) if lead else None,
    }

    for k, v in out.items():
        if v is None:
            print(f"{k}: insufficient data")
            continue
        if "accuracy" in v:
            print(f"{k:<28} acc {v['accuracy']:>5.1f}%  "
                  f"(majority {v['majority_baseline']:.1f}%)  "
                  f"AUC {v['auc']:.3f}  n_test {v['n_test']}", flush=True)
        else:
            print(f"{k:<28} {v}", flush=True)

    json.dump(out, open(OUT, "w"), indent=2, default=str)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/deal_details.py 2>&1 | tee logs/deal_details.log
```

For each detail, compare `accuracy` against `majority_baseline` and treat
anything inside 2pp of the baseline as no signal. **AUC is the number to trust
here** — accuracy on a 90/10 split flatters a model that predicts one class.

- [ ] **Step 3: Commit**

```bash
git add scripts/deal_details.py data/deal_details.json logs/deal_details.log
git commit -m "feat: predict deal structure, size order, completion and lead time"
```

---

### Task 7: Figures

**Files:**
- Modify: `scripts/make_charts.py`
- Create: `docs/figures/` outputs (gitignored or committed per existing convention)

**REQUIRED SUB-SKILL:** Read the `dataviz` skill before writing any chart code, and
validate the palette with its script rather than by eye.

**Every figure in `docs/figures/` is currently stale** — they show retracted numbers.
Regenerate all of them from `data/*.json`; never hard-code a number into a chart.

Figures, in priority order:

1. Precision curve by list size, both models on one axis.
2. Target vs buyer lift — buying is easier to predict than being bought.
3. Feature-family ablation for both models, **with the ±2pp noise band drawn**.
4. Permutation null distribution against the real score, for both models
   (the buyer null comes from Task 3's `buyerperm`).
5. Deal rate by company-size decile — the hump that breaks linear models.
6. Year-by-year CV spread — the honest picture of regime dependence.
7. **New:** alignment ratio by lead time (Task 4).
8. **New:** matching top-k accuracy vs the 0.99% random baseline, at both
   embargoes (Task 5).

- [ ] **Step 1: Read the dataviz skill and the existing house style**

```bash
sed -n '1,80p' scripts/make_charts.py
```

- [ ] **Step 2: Add the eight figures, each reading from `data/*.json`**

- [ ] **Step 3: Regenerate everything and check every figure opens**

```bash
.venv/bin/python scripts/make_charts.py 2>&1 | tee logs/charts.log && ls -la docs/figures/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/make_charts.py docs/figures logs/charts.log
git commit -m "docs: regenerate all figures from current results"
```

---

### Task 8: Write up

**Files:**
- Create: `docs/STRESS_RESULTS.md`
- Modify: `README.md` (the results table is stale — it quotes 13.81% / 5.65×, retracted)
- Modify: `docs/DATA.md` (add `pairs.duckdb` to the inventory)

- [ ] **Step 1: Write `docs/STRESS_RESULTS.md`**

Sections, each stating the number that would have killed the claim:
- What the pair table is and its 97.9% orientation validation
- The four stress tests, with the contaminated-label comparison for the hazard model
- Alignment: do the screens co-fire, and the agreement ratio by lead
- Matching: top-k against the 0.99% baseline, and the embargo decay
- Deal details: which of the four beat their baseline and which did not
- **A "what we tried that did not work" section.** Anything that came in at
  baseline goes here explicitly.

- [ ] **Step 2: Fix the stale README numbers**

Replace the results table with the current verified-target figures and remove
the retracted 13.81% / 5.65× claim.

- [ ] **Step 3: Add `pairs.duckdb` to `docs/DATA.md`**

- [ ] **Step 4: Commit**

```bash
git add docs/STRESS_RESULTS.md README.md docs/DATA.md
git commit -m "docs: stress-test results and buyer-target alignment findings"
```

---

## Execution notes

**Dependency order.** Tasks 1, 2 and 3 are mutually independent — Task 2 reads only
the feature parquets, not the pair table. Tasks 4, 5 and 6 each need both Task 1's
`pairs.duckdb` and Task 2's `pair_scores.parquet`. Tasks 7 and 8 are last.

**Parallel-safe groups:**
- Group A (immediately, concurrent): Task 1, Task 2, Task 3
- Group B (after Tasks 1 and 2, concurrent): Task 4, Task 5, Task 6
- Group C: Task 7, then Task 8

**Concurrency ceiling: 3 processes.** 8 cores with 2 threads pinned per process,
16 GB RAM, and `features.parquet` costs roughly 2–3 GB resident per process.

**`data/` and `logs/` are both gitignored, repo-wide.** Every
`git add ... logs/*.log` and `git add data/*.json` in this plan errors on an
ignored path. **Commit scripts and docs only.** Results live on disk and are
regenerable; that means Task 8's write-up must embed the actual numbers in the
markdown rather than pointing at a JSON file no reader will have.

**Total compute: about 40 model fits plus one clustered logit, ~25 minutes wall
clock** at the measured 11s per fit, plus index parsing and I/O.

**The standing instruction from the handoff, which applies to every task above:**
show the number that kills an idea as readily as the one that confirms it. Four
errors have already been caught that way. Assume there is a fifth — and note that
this plan's most likely candidate is the matching model in Task 5, where a feature
that is constant within a candidate set, or a sector-timing feature that encodes
the announcement, would both produce a confident wrong answer.
