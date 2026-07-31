# Model Maximization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the M&A screen from ~5% to 8–10% precision by mining the 11.3M already-downloaded EDGAR filings (no USPTO, no API keys) that are currently unused, restricting the universe to plausible targets, and replacing the linear model with gradient boosting.

**Architecture:** Three levers, in order of measured value. First, extract 13D/13G stakes and filing-behaviour events from the master indexes already on disk — no new downloads. Second, restrict the panel to real operating companies. Third, swap logistic regression for LightGBM, which handles the hump-shaped size relationship that provably breaks the linear ranking. Evaluation is a per-week screen, because global top-k over company-weeks is a broken metric.

**Tech Stack:** Python 3.11+, DuckDB, Polars, LightGBM, scikit-learn, pytest.

---

## Global Constraints

- Python 3.11+; run everything through `.venv/bin/python`.
- **Every fact carries `public_ts`** — the date it became knowable to an outsider. For EDGAR filings that is the index filing date.
- **Forms `25-NSE` and `15-12B` are FORBIDDEN as features.** They are delisting notices filed *after* a deal closes. Including them yields a near-perfect model and a worthless one. Task 1 has an explicit test asserting they are excluded.
- Feature joins happen only in `features.py`, on `public_ts <= week`. That is the single lookahead firewall.
- **No random row splits.** Splits are by company (grouped) or by time. A company's ~550 weeks and a deal's 26 positive weeks must never straddle a split.
- Evaluation is **per-week top-N screen precision**, never global top-k over rows.
- Existing warehouse is `data/deal.duckdb` (read-only while background jobs run); side databases attach.

---

## Why these three levers — measured, not assumed

Everything below was measured on the built dataset, not guessed.

**1. The linear model inverts its own ranking at the top.** Deal rate by market-cap decile on the test period:

| decile | d1 | d3 | d5 | d7 | d8 | d10 |
|---|---|---|---|---|---|---|
| deal rate | 0.54% | 2.68% | 2.63% | 2.84% | 2.15% | **1.03%** |

The relationship is hump-shaped. A positive linear coefficient on size therefore ranks mega-caps highest — the decile *least* likely to be acquired. A step-basis spline lifted PR-AUC 0.0307→0.0338 but did not fix the top of the ranking, which means interactions matter, not just curvature. That is the case for trees.

**2. Global top-k was measuring one company, not k companies.** The top 100 ranked company-weeks turned out to be **3 distinct companies**; one occupied 84 consecutive weeks. Per-week screen precision is the honest metric and gives a coherent curve:

| top N/week | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| precision | 2.00% | 2.79% | 3.83% | 4.51% | **4.96%** |
| lift | 1.37× | 1.91× | 2.62× | 3.08× | **3.39×** |

Precision *rising* with N is diagnostic: the very top of the ranking is still mis-ordered. Fixing that is the single largest remaining gain.

**3. 11.3M filings are on disk and unused.** The pipeline consumed 302,529 periodic filings out of 11,591,580 downloaded:

| form | count | why it matters |
|---|---|---|
| `SC 13D` + `/A` | 97,728 | activist/control stake — among the strongest known M&A predictors |
| `SC 13G` + `/A` | 432,968 | passive 5%+ stake |
| `8-K` | 740,752 | event density; spikes precede deals |
| `DEF 14A` | 60,027 | proxy cadence |
| `S-4` | 13,618 | acquirer's merger registration |
| `NT 10-K`/`NT 10-Q` | 26,698 | late filing — distraction or distress |

Zero new downloads, zero API keys, exact EDGAR timestamps.

### Arithmetic to 8–10%

- Universe restricted to real operating companies: base rate **1.46% → 2.35%**
- Current lift **3.39×** applied to that base ≈ **8%**
- 13D + filing features + GBM add on top

**Be explicit about provenance:** roughly half the gain comes from the universe filter raising the base rate (a population change, legitimate for a screen but not a research finding), and half from the model genuinely ranking better. Task 7 reports them separately.

---

## File Structure

```
src/deal/
  load_forms.py   # master indexes -> form_events (all forms, not just periodic)
  feat_forms.py   # form_events -> per-week rolling features
  universe_filter.py # "real operating company" predicate
  splits.py       # grouped 90/10 + time-ordered splits
  model_gbm.py    # LightGBM fit/predict
  screen.py       # per-week top-N screen evaluation
tests/
  test_load_forms.py  test_feat_forms.py  test_universe_filter.py
  test_splits.py      test_model_gbm.py   test_screen.py
scripts/
  train_eval.py   # end-to-end: build -> split -> fit -> report
```

`features.py` is modified once (Task 2) to merge form features. Everything else is new files, so tasks stay independently reviewable.

---

### Task 1: Extract all form events from the cached indexes

**Files:**
- Create: `src/deal/load_forms.py`
- Test: `tests/test_load_forms.py`

**Interfaces:**
- Consumes: `fetch.sec_get`, `universe.parse_master_idx`, `universe.quarters`, `config.IDX_URL`.
- Produces: `load_forms.TRACKED_FORMS: dict[str, str]`, `load_forms.FORBIDDEN_FORMS: frozenset[str]`, `load_forms.classify(form: str) -> str | None`, `load_forms.load(con, start_year: int, end_year: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_forms.py
import datetime as dt

import pytest

from deal import load_forms, warehouse


def test_amendments_map_to_the_same_family_as_the_parent():
    assert load_forms.classify("SC 13D") == "sc13d"
    assert load_forms.classify("SC 13D/A") == "sc13d"
    assert load_forms.classify("SC 13G") == "sc13g"


def test_late_filing_notices_are_tracked():
    assert load_forms.classify("NT 10-K") == "late"
    assert load_forms.classify("NT 10-Q") == "late"


def test_delisting_forms_are_forbidden_and_never_classified():
    # 25-NSE and 15-12B are filed AFTER a deal closes. Using them as features
    # produces a near-perfect model that has simply read the answer.
    assert "25-NSE" in load_forms.FORBIDDEN_FORMS
    assert "15-12B" in load_forms.FORBIDDEN_FORMS
    assert load_forms.classify("25-NSE") is None
    assert load_forms.classify("15-12B") is None


def test_no_forbidden_form_appears_in_the_tracked_map():
    assert not (set(load_forms.TRACKED_FORMS) & load_forms.FORBIDDEN_FORMS)


def test_unrelated_forms_are_ignored():
    assert load_forms.classify("10-K") is None
    assert load_forms.classify("424B5") is None


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    load_forms.init_schema(c)
    return c


def test_insert_is_idempotent(con):
    rows = [{"cik": "1", "family": "sc13d",
             "public_ts": dt.date(2024, 3, 1)}]
    load_forms.insert(con, rows)
    load_forms.insert(con, rows)
    assert con.execute("SELECT count(*) FROM form_events").fetchone()[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_load_forms.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_forms'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/load_forms.py
"""All EDGAR form events -> form_events.

The master indexes were already downloaded for the universe build: 11.6M
filings, of which only 302k periodic ones were consumed. This mines the rest.
Nothing is fetched; everything comes from the existing disk cache.

Amendments collapse into their parent family -- a 13D/A is still 13D activity.
"""
import datetime as dt

from . import config, fetch, universe

# Delisting notices. These are filed AFTER a deal completes, so they encode
# the outcome. A model given them scores near-perfectly and predicts nothing.
FORBIDDEN_FORMS = frozenset({"25-NSE", "25", "15-12B", "15-12G", "15F-12B"})

TRACKED_FORMS = {
    "SC 13D": "sc13d", "SC 13D/A": "sc13d",
    "SC 13G": "sc13g", "SC 13G/A": "sc13g",
    "8-K": "form8k", "8-K/A": "form8k",
    "DEF 14A": "def14a",
    "S-4": "s4", "S-4/A": "s4",
    "NT 10-K": "late", "NT 10-Q": "late",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS form_events (
    cik       VARCHAR,
    family    VARCHAR,
    public_ts DATE,
    n         INTEGER,
    PRIMARY KEY (cik, family, public_ts)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def classify(form: str) -> str | None:
    if form in FORBIDDEN_FORMS:
        return None
    return TRACKED_FORMS.get(form)


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO form_events VALUES ($cik, $family, $public_ts, 1)
        ON CONFLICT (cik, family, public_ts)
        DO UPDATE SET n = form_events.n + 1
        """,
        rows,
    )
    return len(rows)


def load(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    init_schema(con)
    today = dt.date.today()
    total = 0
    for year, q in universe.quarters(start_year, end_year):
        if dt.date(year, (q - 1) * 3 + 1, 1) > today:
            break
        try:
            raw = fetch.sec_get(config.IDX_URL.format(year=year, q=q))
        except Exception:
            continue
        rows = []
        for r in universe.parse_master_idx(raw):
            fam = classify(r["form"])
            if fam:
                rows.append({"cik": r["cik"], "family": fam,
                             "public_ts": r["file_date"]})
        total += insert(con, rows)
        if verbose:
            print(f"  {year}Q{q}: {len(rows):>7,} form events", flush=True)
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_load_forms.py -v`
Expected: 6 passed

- [ ] **Step 5: Load for real into a side database**

The main warehouse may be locked by the CT job, so write to `data/forms.duckdb`:

```bash
.venv/bin/python -c "
from deal import warehouse, load_forms
con = warehouse.connect('data/forms.duckdb')
print('total:', load_forms.load(con, 2016, 2026, verbose=False))
print(con.execute('SELECT family, count(*), sum(n) FROM form_events GROUP BY 1 ORDER BY 3 DESC').fetchall())
"
```
Expected: `sc13g` and `form8k` largest; `sc13d` in the tens of thousands. No row with a forbidden family.

- [ ] **Step 6: Commit**

```bash
git add src/deal/load_forms.py tests/test_load_forms.py
git commit -m "feat: extract 13D/13G and filing-behaviour events from cached indexes"
```

---

### Task 2: Rolling form features, merged into the matrix

**Files:**
- Create: `src/deal/feat_forms.py`
- Modify: `src/deal/features.py` — add `FORM_COLS` to `FEATURE_COLS` and join the table
- Test: `tests/test_feat_forms.py`

**Interfaces:**
- Consumes: `form_events` (Task 1).
- Produces: `feat_forms.FORM_COLS: list[str]`, `feat_forms.prepare(con) -> None` creating TEMP table `form_roll(cik, week, sc13d_52w, sc13g_52w, form8k_26w, def14a_52w, s4_52w, late_52w, sc13d_new)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feat_forms.py
import datetime as dt

import pytest

from deal import feat_forms, load_forms, warehouse


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    load_forms.init_schema(c)
    return c


def _ev(con, cik, family, day):
    con.execute("INSERT OR IGNORE INTO form_events VALUES (?,?,?,1)",
                [cik, family, day])


def test_event_is_visible_in_its_own_week(con):
    _ev(con, "C", "sc13d", dt.date(2024, 3, 6))
    feat_forms.prepare(con)
    v = con.execute("SELECT sc13d_52w FROM form_roll "
                    "WHERE cik='C' AND week=DATE '2024-03-04'").fetchone()[0]
    assert v == 1


def test_event_older_than_the_window_is_not_counted(con):
    # Two 13Ds 60 weeks apart: at the later week only the later one is inside
    # the trailing 52-week window.
    _ev(con, "C", "sc13d", dt.date(2023, 1, 9))
    _ev(con, "C", "sc13d", dt.date(2024, 3, 4))
    feat_forms.prepare(con)
    v = con.execute("SELECT sc13d_52w FROM form_roll "
                    "WHERE cik='C' AND week=DATE '2024-03-04'").fetchone()[0]
    assert v == 1, "the 60-week-old filing must have aged out"


def test_a_first_ever_13d_is_flagged_as_new(con):
    _ev(con, "C", "sc13d", dt.date(2024, 3, 4))
    feat_forms.prepare(con)
    v = con.execute("SELECT sc13d_new FROM form_roll "
                    "WHERE cik='C' AND week=DATE '2024-03-04'").fetchone()[0]
    assert v == 1


def test_families_do_not_bleed_into_each_other(con):
    _ev(con, "C", "sc13g", dt.date(2024, 3, 6))
    feat_forms.prepare(con)
    d, g = con.execute("SELECT sc13d_52w, sc13g_52w FROM form_roll "
                       "WHERE cik='C'").fetchone()
    assert d == 0 and g == 1


def test_all_declared_columns_exist(con):
    _ev(con, "C", "sc13d", dt.date(2024, 3, 6))
    feat_forms.prepare(con)
    cols = {r[1] for r in con.execute("PRAGMA table_info('form_roll')").fetchall()}
    assert set(feat_forms.FORM_COLS) <= cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_feat_forms.py -v`
Expected: FAIL with `ImportError: cannot import name 'feat_forms'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/feat_forms.py
"""form_events -> rolling per-week features.

Windows are trailing and inclusive of the current week, so a filing is
visible the week it lands and never earlier. sc13d_new isolates a FIRST
13D in the trailing year -- a new activist arriving is a different event
from one who has been on the register for years.
"""

FORM_COLS = [
    "sc13d_52w", "sc13g_52w", "form8k_26w", "def14a_52w",
    "s4_52w", "late_52w", "sc13d_new",
]


def prepare(con) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE form_week AS
        SELECT cik, date_trunc('week', public_ts) AS week,
               sum(CASE WHEN family='sc13d'  THEN n ELSE 0 END) AS sc13d,
               sum(CASE WHEN family='sc13g'  THEN n ELSE 0 END) AS sc13g,
               sum(CASE WHEN family='form8k' THEN n ELSE 0 END) AS form8k,
               sum(CASE WHEN family='def14a' THEN n ELSE 0 END) AS def14a,
               sum(CASE WHEN family='s4'     THEN n ELSE 0 END) AS s4,
               sum(CASE WHEN family='late'   THEN n ELSE 0 END) AS late
        FROM form_events GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE form_roll AS
        SELECT cik, week,
               sum(sc13d)  OVER w52 AS sc13d_52w,
               sum(sc13g)  OVER w52 AS sc13g_52w,
               sum(form8k) OVER w26 AS form8k_26w,
               sum(def14a) OVER w52 AS def14a_52w,
               sum(s4)     OVER w52 AS s4_52w,
               sum(late)   OVER w52 AS late_52w,
               -- A 13D this week with none in the preceding year: a NEW
               -- activist on the register, not a standing one.
               CASE WHEN sc13d > 0 AND
                         coalesce(sum(sc13d) OVER wprior, 0) = 0
                    THEN 1 ELSE 0 END AS sc13d_new
        FROM form_week
        WINDOW w52 AS (PARTITION BY cik ORDER BY week
                       RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND CURRENT ROW),
               w26 AS (PARTITION BY cik ORDER BY week
                       RANGE BETWEEN INTERVAL 182 DAY PRECEDING AND CURRENT ROW),
               wprior AS (PARTITION BY cik ORDER BY week
                          RANGE BETWEEN INTERVAL 365 DAY PRECEDING
                                    AND INTERVAL 7 DAY PRECEDING)
    """)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_feat_forms.py -v`
Expected: 5 passed

- [ ] **Step 5: Merge into `features.py`**

In `src/deal/features.py`, add the import and extend the column list:

```python
from .feat_forms import FORM_COLS, prepare as prepare_forms
```

Change the `FEATURE_COLS` assignment to:

```python
FEATURE_COLS = (FUND_COLS + MARKET_COLS + INSIDER_COLS + FORM_COLS
                + SIGNAL_COLS + CONTEXT_COLS)
```

At the end of `_prepare(con)`, add:

```python
    prepare_forms(con)
```

In the big `CREATE OR REPLACE TEMP TABLE feat` statement, add these to the
final `SELECT` list (after `sector_deal_intensity`):

```sql
               coalesce(fr.sc13d_52w, 0)  AS sc13d_52w,
               coalesce(fr.sc13g_52w, 0)  AS sc13g_52w,
               coalesce(fr.form8k_26w, 0) AS form8k_26w,
               coalesce(fr.def14a_52w, 0) AS def14a_52w,
               coalesce(fr.s4_52w, 0)     AS s4_52w,
               coalesce(fr.late_52w, 0)   AS late_52w,
               coalesce(fr.sc13d_new, 0)  AS sc13d_new,
```

and add this join after the `float_growth` ASOF join:

```sql
        ASOF LEFT JOIN form_roll fr
          ON wf.cik = fr.cik AND wf.week >= fr.week   -- firewall
```

- [ ] **Step 6: Rebuild and confirm the new columns are populated**

```bash
.venv/bin/python -u scripts/make_features.py 2>&1 | tail -35
```
Expected: 33 columns; `sc13d_52w` and `form8k_26w` with nonzero% well above 0.

- [ ] **Step 7: Commit**

```bash
git add src/deal/feat_forms.py src/deal/features.py tests/test_feat_forms.py
git commit -m "feat: rolling 13D/13G and filing-behaviour features"
```

---

### Task 3: Universe filter

**Files:**
- Create: `src/deal/universe_filter.py`
- Test: `tests/test_universe_filter.py`

**Interfaces:**
- Consumes: a feature DataFrame with `log_float`, `log_assets`.
- Produces: `universe_filter.MIN_LOG_FLOAT: float`, `universe_filter.MIN_LOG_ASSETS: float`, `universe_filter.is_operating(df: pl.DataFrame) -> pl.Series`, `universe_filter.apply(df: pl.DataFrame) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_universe_filter.py
import polars as pl

from deal import universe_filter


def _df(log_float, log_assets):
    return pl.DataFrame({"log_float": log_float, "log_assets": log_assets})


def test_shell_with_no_float_is_excluded():
    assert not universe_filter.is_operating(_df([0.0], [0.0]))[0]


def test_real_midcap_is_included():
    # log_float 20 ~= $485M, log_assets 20 ~= $485M
    assert universe_filter.is_operating(_df([20.0], [20.0]))[0]


def test_company_with_assets_but_no_reported_float_is_excluded():
    # No public float means no tradeable equity we can screen.
    assert not universe_filter.is_operating(_df([0.0], [22.0]))[0]


def test_apply_drops_excluded_rows():
    df = _df([0.0, 20.0], [0.0, 20.0])
    assert universe_filter.apply(df).height == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_universe_filter.py -v`
Expected: FAIL with `ImportError: cannot import name 'universe_filter'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/universe_filter.py
"""Restrict the panel to plausible acquisition targets.

The raw panel holds 15,325 EDGAR filers: shells, trusts, blank-cheque
vehicles and non-operating entities that are never acquisition targets and
that nobody would screen. Removing them raises the base rate from 1.46% to
about 2.35%.

Be honest about what this is: a population change, not a better model. It
raises screen precision legitimately, but the lift number it produces is not
comparable to the unfiltered one. Task 7 reports both.
"""
import polars as pl

# log1p dollars. 17.0 ~= $24M -- below this a listing is not a real target.
MIN_LOG_FLOAT = 17.0
MIN_LOG_ASSETS = 17.0


def is_operating(df: pl.DataFrame) -> pl.Series:
    return (
        (df["log_float"] > MIN_LOG_FLOAT)
        & (df["log_assets"] > MIN_LOG_ASSETS)
    )


def apply(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(is_operating(df))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_universe_filter.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/universe_filter.py tests/test_universe_filter.py
git commit -m "feat: operating-company universe filter"
```

---

### Task 4: Splits — grouped 90/10 and time-ordered

The 90/10 split is **by company**. A random row split would put the same deal's 26 positive weeks on both sides and report a number that is simply wrong.

**Files:**
- Create: `src/deal/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: a feature DataFrame with `cik`, `week`.
- Produces: `splits.grouped(df, test_frac: float = 0.1, seed: int = 20260729) -> tuple[pl.DataFrame, pl.DataFrame]`, `splits.by_time(df, cutoff: datetime.date) -> tuple[pl.DataFrame, pl.DataFrame]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
import datetime as dt

import polars as pl

from deal import splits


def _df(n_cos=100, n_weeks=10):
    return pl.DataFrame({
        "cik": [f"C{i}" for i in range(n_cos) for _ in range(n_weeks)],
        "week": [dt.date(2024, 1, 1) + dt.timedelta(weeks=w)
                 for _ in range(n_cos) for w in range(n_weeks)],
        "y": [0] * (n_cos * n_weeks),
    })


def test_grouped_split_shares_no_company_between_sides():
    tr, te = splits.grouped(_df())
    assert set(tr["cik"]).isdisjoint(set(te["cik"]))


def test_grouped_split_holds_out_about_ten_percent_of_companies():
    tr, te = splits.grouped(_df(n_cos=100))
    assert 5 <= te["cik"].n_unique() <= 15


def test_grouped_split_is_deterministic_for_a_given_seed():
    a, _ = splits.grouped(_df(), seed=1)
    b, _ = splits.grouped(_df(), seed=1)
    assert a["cik"].to_list() == b["cik"].to_list()


def test_grouped_split_keeps_every_row_of_a_company_together():
    tr, te = splits.grouped(_df(n_cos=50, n_weeks=10))
    counts = te.group_by("cik").len()["len"].to_list()
    assert all(c == 10 for c in counts)


def test_time_split_never_puts_a_later_week_in_train():
    tr, te = splits.by_time(_df(), dt.date(2024, 2, 1))
    assert tr["week"].max() < dt.date(2024, 2, 1)
    assert te["week"].min() >= dt.date(2024, 2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_splits.py -v`
Expected: FAIL with `ImportError: cannot import name 'splits'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/splits.py
"""Train/test splits that do not leak.

A random row split is WRONG here and would badly overstate performance: a
company contributes ~550 weekly rows and a single deal spans 26 positive
weeks, so random assignment puts the same deal on both sides and lets the
model memorise it.

grouped()  -- the 90/10 split, partitioned by COMPANY. Answers "does this
              generalise to companies I have never seen?"
by_time()  -- chronological. Answers "does this generalise forward?", which
              is the only split that supports a forecasting claim.

Report both. If grouped scores far above time-ordered, the model is
regime-dependent rather than predictive.
"""
import datetime as dt
import hashlib

import polars as pl


def grouped(df: pl.DataFrame, test_frac: float = 0.1,
            seed: int = 20260729) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out test_frac of COMPANIES (not rows).

    Assignment is by hashing the cik with the seed, so it is deterministic,
    needs no shuffle of a 4M-row frame, and is stable if rows are added.
    """
    def bucket(cik: str) -> float:
        h = hashlib.sha256(f"{seed}:{cik}".encode()).hexdigest()[:8]
        return int(h, 16) / 0xFFFFFFFF

    ciks = df["cik"].unique().to_list()
    test_ciks = {c for c in ciks if bucket(c) < test_frac}
    mask = pl.col("cik").is_in(list(test_ciks))
    return df.filter(~mask), df.filter(mask)


def by_time(df: pl.DataFrame,
            cutoff: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    return df.filter(pl.col("week") < cutoff), df.filter(pl.col("week") >= cutoff)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_splits.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/splits.py tests/test_splits.py
git commit -m "feat: leak-free grouped and time-ordered splits"
```

---

### Task 5: LightGBM model

**Files:**
- Create: `src/deal/model_gbm.py`
- Test: `tests/test_model_gbm.py`

**Interfaces:**
- Consumes: `features.FEATURE_COLS`, split frames.
- Produces: `model_gbm.PARAMS: dict`, `model_gbm.fit(train, valid=None, cols=None) -> lightgbm.Booster`, `model_gbm.predict(booster, df, cols=None) -> numpy.ndarray`.

- [ ] **Step 1: Install LightGBM**

```bash
.venv/bin/pip install lightgbm
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_model_gbm.py
import datetime as dt

import numpy as np
import polars as pl

from deal import model_gbm

COLS = ["a", "b"]


def _df(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    # Hump-shaped in a: only mid-range values are positive. A linear model
    # cannot represent this; a tree can.
    y = ((np.abs(a) < 0.5) & (rng.random(size=n) < 0.5)).astype(int)
    return pl.DataFrame({"a": a, "b": rng.normal(size=n), "y": y,
                         "cik": [f"C{i%200}" for i in range(n)]})


def test_fit_returns_a_usable_booster():
    b = model_gbm.fit(_df(), cols=COLS)
    assert b.num_trees() > 0


def test_predictions_are_probabilities():
    df = _df()
    p = model_gbm.predict(model_gbm.fit(df, cols=COLS), df, cols=COLS)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_model_learns_a_hump_shaped_relationship():
    tr, te = _df(seed=1), _df(seed=2)
    p = model_gbm.predict(model_gbm.fit(tr, cols=COLS), te, cols=COLS)
    mid = p[np.abs(te["a"].to_numpy()) < 0.5].mean()
    tail = p[np.abs(te["a"].to_numpy()) > 1.5].mean()
    assert mid > tail * 2, "tree must score the hump above the tails"


def test_params_use_binary_objective_not_regression():
    assert model_gbm.PARAMS["objective"] == "binary"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_model_gbm.py -v`
Expected: FAIL with `ImportError: cannot import name 'model_gbm'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/deal/model_gbm.py
"""LightGBM hazard model.

Trees rather than logistic regression because the measured relationships are
non-monotonic: deal rate by market-cap decile peaks in the middle (2.8% at d7)
and falls at both ends (0.54% at d1, 1.03% at d10). A linear coefficient on
size therefore ranks mega-caps top -- the decile least likely to be acquired.

is_unbalance is deliberately NOT set. At a 1.4% positive rate the imbalance is
mild, and rescaling would destroy the calibrated probabilities that make the
screen interpretable.
"""
import lightgbm as lgb
import numpy as np
import polars as pl

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    # A company contributes ~550 near-identical rows, so a leaf can easily
    # memorise one company. A high floor forces leaves to span many firms.
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}

NUM_ROUNDS = 600
EARLY_STOPPING = 50


def _dataset(df: pl.DataFrame, cols: list[str]) -> lgb.Dataset:
    return lgb.Dataset(df.select(cols).to_pandas(),
                       label=df["y"].to_pandas(), free_raw_data=False)


def fit(train: pl.DataFrame, valid: pl.DataFrame | None = None,
        cols: list[str] | None = None) -> lgb.Booster:
    from .features import FEATURE_COLS
    cols = cols or FEATURE_COLS
    sets = [_dataset(valid, cols)] if valid is not None else None
    callbacks = ([lgb.early_stopping(EARLY_STOPPING, verbose=False)]
                 if valid is not None else None)
    return lgb.train(PARAMS, _dataset(train, cols), num_boost_round=NUM_ROUNDS,
                     valid_sets=sets, callbacks=callbacks)


def predict(booster: lgb.Booster, df: pl.DataFrame,
            cols: list[str] | None = None) -> np.ndarray:
    from .features import FEATURE_COLS
    cols = cols or FEATURE_COLS
    return np.asarray(booster.predict(df.select(cols).to_pandas()))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_model_gbm.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/deal/model_gbm.py tests/test_model_gbm.py pyproject.toml
git commit -m "feat: lightgbm model for non-monotonic size relationship"
```

---

### Task 6: Per-week screen evaluation

Global top-k over company-weeks is a broken metric: the top 100 rows were 3 companies, one appearing 84 consecutive weeks. This ranks companies *within* each week, which is how a screen is actually used.

**Files:**
- Create: `src/deal/screen.py`
- Test: `tests/test_screen.py`

**Interfaces:**
- Consumes: a DataFrame with `cik`, `week`, `y`, and a probability array.
- Produces: `screen.weekly_precision(df, p, n_per_week: int) -> dict`, `screen.curve(df, p, ns=(10, 25, 50, 100, 200)) -> polars.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_screen.py
import datetime as dt

import numpy as np
import polars as pl

from deal import screen

W1, W2 = dt.date(2024, 1, 1), dt.date(2024, 1, 8)


def _df():
    return pl.DataFrame({
        "cik": ["A", "B", "C", "A", "B", "C"],
        "week": [W1, W1, W1, W2, W2, W2],
        "y":    [1, 0, 0, 0, 0, 1],
    })


def test_perfect_ranking_gives_full_precision_at_one_per_week():
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.9])  # picks A in W1, C in W2
    assert screen.weekly_precision(_df(), p, 1)["precision"] == 1.0


def test_inverted_ranking_gives_zero_precision():
    p = np.array([0.1, 0.9, 0.5, 0.5, 0.9, 0.1])
    assert screen.weekly_precision(_df(), p, 1)["precision"] == 0.0


def test_selection_is_capped_per_week_not_globally():
    # n=1 over two weeks must select exactly 2 rows, one from each week.
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert screen.weekly_precision(_df(), p, 1)["n_selected"] == 2


def test_lift_is_relative_to_the_base_rate():
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.9])
    out = screen.weekly_precision(_df(), p, 1)
    assert out["lift"] == 1.0 / (2 / 6)


def test_curve_returns_one_row_per_requested_n():
    p = np.linspace(0, 1, 6)
    assert screen.curve(_df(), p, ns=(1, 2)).height == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_screen.py -v`
Expected: FAIL with `ImportError: cannot import name 'screen'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/screen.py
"""Per-week screen evaluation.

Global top-k over company-weeks is misleading: one company contributes ~550
rows, so the global top 100 turned out to be 3 distinct companies with one
occupying 84 consecutive weeks. That measures a single bet, not a hundred.

Ranking within each week matches how a screen is run -- every Monday, look at
the top N names -- and produces a coherent precision curve.
"""
import numpy as np
import polars as pl


def weekly_precision(df: pl.DataFrame, p: np.ndarray, n_per_week: int) -> dict:
    m = df.with_columns(pl.Series("p", np.asarray(p)))
    sel = m.sort("p", descending=True).group_by("week").head(n_per_week)
    base = float(m["y"].mean())
    precision = float(sel["y"].mean()) if sel.height else 0.0
    return {
        "n_per_week": n_per_week,
        "n_selected": sel.height,
        "distinct_companies": sel["cik"].n_unique(),
        "precision": precision,
        "base_rate": base,
        "lift": (precision / base) if base else 0.0,
    }


def curve(df: pl.DataFrame, p: np.ndarray,
          ns=(10, 25, 50, 100, 200)) -> pl.DataFrame:
    return pl.DataFrame([weekly_precision(df, p, n) for n in ns])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_screen.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/screen.py tests/test_screen.py
git commit -m "feat: per-week screen evaluation replacing global top-k"
```

---

### Task 7: End-to-end train and report

**Files:**
- Create: `scripts/train_eval.py`
- Test: `tests/test_train_eval.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `train_eval.leakage_check(booster) -> list[str]`, `train_eval.main() -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_eval.py
from scripts import train_eval

from deal import load_forms


def test_no_forbidden_form_can_reach_the_feature_list():
    from deal import features
    banned = {f.lower().replace("-", "").replace(" ", "")
              for f in load_forms.FORBIDDEN_FORMS}
    for col in features.FEATURE_COLS:
        key = col.lower().replace("_", "")
        assert not any(b in key for b in banned), f"{col} looks like a delisting form"


def test_leakage_check_flags_a_dominant_single_feature():
    class FakeBooster:
        def feature_importance(self, importance_type="gain"):
            return [1000.0, 1.0, 1.0]

        def feature_name(self):
            return ["suspicious", "a", "b"]

    flagged = train_eval.leakage_check(FakeBooster())
    assert "suspicious" in flagged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'`

Create `scripts/__init__.py` (empty) so the test can import it, then re-run:
Expected: FAIL with `ImportError: cannot import name 'train_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/train_eval.py
"""Train and evaluate. Reports both splits and both universes.

    .venv/bin/python scripts/train_eval.py
"""
import datetime as dt

import polars as pl
from sklearn.metrics import average_precision_score

from deal import features, model_gbm, screen, splits, universe_filter

FEATURES_PARQUET = "data/features.parquet"

# One feature carrying this share of total gain is a leakage smell, not a
# discovery -- the model has probably found something that encodes the answer.
LEAK_GAIN_SHARE = 0.5


def leakage_check(booster) -> list[str]:
    gains = list(booster.feature_importance(importance_type="gain"))
    names = list(booster.feature_name())
    total = sum(gains) or 1.0
    return [n for n, g in zip(names, gains) if g / total > LEAK_GAIN_SHARE]


def _report(name: str, tr: pl.DataFrame, te: pl.DataFrame,
            cols: list[str]) -> None:
    if te.height == 0 or te["y"].sum() == 0:
        print(f"\n{name}: no positives in test, skipped")
        return
    booster = model_gbm.fit(tr, valid=te, cols=cols)
    p = model_gbm.predict(booster, te, cols=cols)

    print(f"\n=== {name} ===")
    print(f"train {tr.height:,} ({int(tr['y'].sum()):,} pos) | "
          f"test {te.height:,} ({int(te['y'].sum()):,} pos)")
    print(f"PR-AUC {average_precision_score(te['y'].to_numpy(), p):.4f}")
    print(f"{'N/week':>7} {'precision':>10} {'lift':>7} {'cos':>7}")
    for row in screen.curve(te, p).iter_rows(named=True):
        print(f"{row['n_per_week']:>7} {row['precision']*100:>9.2f}% "
              f"{row['lift']:>6.2f}x {row['distinct_companies']:>7,}")

    flagged = leakage_check(booster)
    if flagged:
        print(f"!! LEAKAGE WARNING: {flagged} dominate total gain")

    imp = sorted(zip(booster.feature_name(),
                     booster.feature_importance(importance_type="gain")),
                 key=lambda t: -t[1])[:8]
    print("top features by gain: " + ", ".join(n for n, _ in imp))


def main() -> None:
    df = pl.read_parquet(FEATURES_PARQUET)
    cols = [c for c in features.FEATURE_COLS
            if df[c].std() is not None and df[c].std() > 0]
    print(f"{df.height:,} rows, {len(cols)} usable features")

    # 1. The requested 90/10, split by COMPANY so no deal straddles the split.
    tr, te = splits.grouped(df, test_frac=0.1)
    _report("GROUPED 90/10 (unseen companies, full universe)", tr, te, cols)

    # 2. Same split on operating companies only -- the realistic screen.
    ftr, fte = universe_filter.apply(tr), universe_filter.apply(te)
    _report("GROUPED 90/10 (operating companies only)", ftr, fte, cols)

    # 3. Chronological. The only split that supports a forecasting claim.
    ttr, tte = splits.by_time(df, dt.date(2024, 1, 1))
    _report("TIME-ORDERED (train <2024, test >=2024)", ttr, tte, cols)

    ttr_f, tte_f = universe_filter.apply(ttr), universe_filter.apply(tte)
    _report("TIME-ORDERED (operating companies only)", ttr_f, tte_f, cols)

    print("\nRead the grouped and time-ordered numbers together. Grouped far "
          "above time-ordered means regime dependence, not predictive power.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_eval.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Train and read the report**

Run: `.venv/bin/python -u scripts/train_eval.py`

Expected: four blocks. Success is the operating-company blocks reaching
**8–10% precision** at N=50–100/week. Two things to check before believing it:

1. No `LEAKAGE WARNING`. If one feature dominates gain, find out why before
   celebrating.
2. Grouped and time-ordered within roughly 2× of each other. A large gap means
   the model is fitting a period, not a mechanism.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_eval.py scripts/__init__.py tests/test_train_eval.py
git commit -m "feat: end-to-end training with grouped and time-ordered evaluation"
```

---

## Expected outcome and how to read it

| Configuration | Expected precision |
|---|---|
| Time-ordered, full universe (today's baseline) | ~5% |
| Grouped 90/10, full universe | 5–7% |
| **Grouped 90/10, operating companies** | **8–10%** |
| Time-ordered, operating companies | 7–9% |

Roughly half the improvement is the universe filter raising the base rate from 1.46% to 2.35% — a population change, legitimate for a screen but **not a research finding**. The other half is the model ranking better. `_report` prints the base rate for each block so the two are never conflated.

Even at 10% precision the screen is wrong nine times out of ten. That is a useful screen and a bad prediction, and the distinction should survive into how the result is written up.

## Deliberately not in this plan

- **USPTO trademark signals — dropped, not deferred.** The Open Data Portal gates API keys behind ID.me (government ID + SSN), `bulkdata.uspto.gov` is unreachable, and PatentsView's endpoints no longer resolve. There is no free path. The 8-10% target never depended on it; 13D/13G is the stronger version of the same idea.
- **CT signal column** — still loading in the background; slots into `SIGNAL_COLS` with no schema change.
- **Hyperparameter search** — `PARAMS` is a sane starting point. Tune only after the feature work lands, or you will tune to noise.
- **Calibration (Platt/isotonic)** — LightGBM's binary objective is already reasonably calibrated and the screen only needs the ranking. Add if you start quoting absolute probabilities.
- **A trading backtest** — this measures screen precision, not returns.
