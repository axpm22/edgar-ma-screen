# M&A Pre-Announcement Signal Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a discrete-time hazard model over a company-week panel that predicts acquisition announcements from public registry signals, and run it forward weekly with an immutable prediction log.

**Architecture:** One point-in-time panel (company × week) is the spine. Each signal ingester writes an event table stamped with the date the fact became *public*. A feature builder joins signals onto the panel with a strict as-of cutoff. The model is a pooled logistic regression on that panel — which is exactly a discrete-time hazard model — and evaluation compares its lead time against the abnormal-volume window rather than chasing accuracy.

**Tech Stack:** Python 3.11+, DuckDB, Polars, statsmodels, scikit-learn (metrics only), httpx, pytest.

---

## Global Constraints

- Python 3.11+.
- **Every fact carries a `public_ts`: the moment it became knowable to an outsider, not the moment it happened.** A USPTO assignment executed in March and recorded in June has `public_ts` = June. Using the execution date is lookahead and voids every downstream result.
- **Wayback-derived facts use the *later* snapshot bound.** A change observed between snapshots at t1 and t2 is stamped t2. Midpoint or t1 is lookahead.
- All timestamps are UTC dates stored as `DATE`. Weeks are ISO weeks, identified by their **Monday** date.
- Warehouse is `data/deal.duckdb`. Raw pulls cache to `data/raw/` as Parquet and are never re-fetched.
- No resampling, SMOTE, or class weighting. The positive rate is ~0.09% and calibrated probabilities are the deliverable — resampling destroys them.
- Model evaluation uses PR-AUC and lift-at-k. **Never report ROC-AUC** — under 0.09% positives it reads ~0.9 for a useless model.
- The live prediction table is append-only. No `UPDATE`, no `DELETE`, ever.

---

## Modeling Logic — read before Task 1

The three questions in the request map to three parts of this plan. This section is the "why"; the tasks are the "how".

### Why a hazard model and not a classifier

The naive framing is "classify each company as acquired / not acquired in the next 6 months." That throws away two things you need:

1. **Right-censoring.** Most companies are never acquired inside the sample window. A company observed for 3 years and not acquired is not a negative — it's a company that survived 3 years. Binary classification treats those identically to a company observed for 3 weeks.
2. **Timing.** You care *when*. The headline result of this project is lead time, and a classifier has no notion of it.

The correct model is a **discrete-time hazard model**: for each company-week, the probability of announcement in that week given survival until then. The convenient part — and the reason this needs no survival library — is that a discrete-time hazard model *is* a pooled logistic regression on the company-week panel, with duration/calendar controls as covariates. `statsmodels.Logit` gives you the whole thing.

```
P(announce in week t | survived to t) = logit⁻¹(β·x_{i,t} + γ·calendar_t)
```

Each company contributes one row per week until it is acquired or the panel ends. Acquired companies contribute a final row with `y=1` and then leave. That row structure is what encodes censoring — nothing else is required.

### The imbalance is not a problem to fix

~1,750 companies × 52 weeks ≈ 91,000 company-weeks per year against ~80 announcements. That's a 0.09% positive rate, and the instinct is to rebalance. Don't.

Rebalancing changes the intercept and destroys probability calibration, and calibrated probability is the entire output here — "this company has a 4% chance of announcing in the next quarter, versus a 0.5% base rate" is the product. An 8x lift on a calibrated model is a finding; an 8x lift on a resampled model is an artifact of the resampling.

At 91k rows/year the full panel fits in memory, so there is no compute argument for sampling either. Use every row.

Evaluate with **PR-AUC** and **lift at the top k**. ROC-AUC is actively misleading at this imbalance because the huge true-negative pool inflates it.

### The four lookahead traps

These sink most alt-data projects, and three of them are subtle.

1. **Recording lag.** USPTO assignments carry an execution date and a recording date. Only the recording date is public knowledge. Same idea for any filing with a retroactive effective date.
2. **Wayback interval ambiguity.** You know the page changed *between* two snapshots. Stamping it at the earlier bound claims knowledge you didn't have.
3. **Rumor vs. definitive agreement.** If Bloomberg reports "exploring a sale" three weeks before the 8-K, the market knew at the rumor. Scoring lead time against the 8-K inflates every result by that gap. This is the single largest source of fake alpha in M&A prediction work.
4. **Survivorship in the universe.** Building the historical panel from today's index constituents silently drops every company that was delisted — including the acquired ones, which are your positives. The universe must be point-in-time.

Trap 3 has an expensive fix (a news archive license) and a cheap one used here: label on definitive-agreement dates from EDGAR, then **hand-collect rumor dates for a random 50-deal subsample** to measure the gap distribution, and report lead time both ways. You don't need rumor dates for 800 deals; you need to know how big the correction is.

### What "see what happens" has to mean

A forward run is only worth anything if you cannot retroactively touch it. Two rules make it evidence rather than anecdote:

- **Append-only prediction log.** Every weekly run writes predictions plus the exact feature vector, and nothing ever mutates them. If you can recompute a past prediction, you will eventually cheat without noticing.
- **Pre-registered evaluation.** The success criteria go into `EVALUATION.md` and get committed *before* the first live run. Otherwise you will rationalise whatever comes out.

Six months of timestamped, immutable, out-of-sample predictions is worth more than any backtest — and it is the single best artifact to show an interviewer, because backtests are cheap and forward logs are not.

---

## File Structure

```
src/deal/
  config.py      # universe params, signal windows, thresholds
  warehouse.py   # DuckDB connection + schema
  panel.py       # point-in-time universe -> company-week panel
  labels.py      # EDGAR deal labels + rumor-gap subsample
  sig_ct.py      # certificate transparency events
  sig_uspto.py   # trademark assignments + intent-to-use filings
  sig_form4.py   # discretionary insider blackout
  features.py    # signals -> as-of feature matrix (the lookahead firewall)
  model.py       # discrete-time hazard fit + predict
  evaluate.py    # PR-AUC, lift@k, lead time vs abnormal volume
  live.py        # weekly run, append-only prediction log
tests/
  test_panel.py  test_labels.py   test_sig_ct.py
  test_sig_uspto.py  test_sig_form4.py  test_features.py
  test_model.py  test_evaluate.py test_live.py
EVALUATION.md    # pre-registered success criteria, committed before live run
```

`features.py` is the only module allowed to join a signal to a panel row. Every as-of-date rule lives there, so there is exactly one place where lookahead can enter and exactly one place to test for it.

---

### Task 1: Point-in-time universe and company-week panel

Survivorship is the failure that silently invalidates everything downstream, so it is Task 1.

**Files:**
- Create: `src/deal/config.py`, `src/deal/warehouse.py`, `src/deal/panel.py`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `warehouse.connect(path: str = "data/deal.duckdb") -> duckdb.DuckDBPyConnection`, `warehouse.init_schema(con) -> None`, `panel.iso_monday(d: datetime.date) -> datetime.date`, `panel.build(con, start: date, end: date) -> int` (rows inserted into `panel`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel.py
import datetime as dt

import pytest

from deal import panel, warehouse


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    return c


def test_iso_monday_snaps_to_start_of_week():
    # 2025-03-13 is a Thursday; its ISO week starts Monday 2025-03-10.
    assert panel.iso_monday(dt.date(2025, 3, 13)) == dt.date(2025, 3, 10)
    assert panel.iso_monday(dt.date(2025, 3, 10)) == dt.date(2025, 3, 10)


def test_company_absent_before_it_listed(con):
    con.execute(
        "INSERT INTO universe VALUES ('CIK1','LateCo','2025-02-01',NULL)"
    )
    panel.build(con, dt.date(2025, 1, 1), dt.date(2025, 3, 1))
    weeks = [r[0] for r in con.execute(
        "SELECT week FROM panel WHERE cik='CIK1' ORDER BY week"
    ).fetchall()]
    assert min(weeks) >= dt.date(2025, 1, 27)  # first week on/after listing


def test_company_present_until_delisting_not_dropped(con):
    # The acquired company MUST stay in the panel up to its exit. Dropping
    # delisted names is exactly the survivorship bug that deletes positives.
    con.execute(
        "INSERT INTO universe VALUES ('CIK2','GoneCo','2025-01-01','2025-02-10')"
    )
    panel.build(con, dt.date(2025, 1, 1), dt.date(2025, 3, 1))
    weeks = [r[0] for r in con.execute(
        "SELECT week FROM panel WHERE cik='CIK2'"
    ).fetchall()]
    assert len(weeks) > 0
    assert max(weeks) <= dt.date(2025, 2, 10)


def test_panel_is_idempotent(con):
    con.execute("INSERT INTO universe VALUES ('CIK3','Co','2025-01-01',NULL)")
    first = panel.build(con, dt.date(2025, 1, 1), dt.date(2025, 2, 1))
    panel.build(con, dt.date(2025, 1, 1), dt.date(2025, 2, 1))
    total = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    assert total == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deal'`

- [ ] **Step 3: Write config, warehouse, and panel**

```python
# src/deal/config.py
"""Constants for the M&A precursor project."""

EDGAR_BASE = "https://www.sec.gov"
EDGAR_FTS = "https://efts.sec.gov/LATEST/search-index?q="
# SEC requires a descriptive UA with contact info on every request.
EDGAR_UA = "RandomBSQuant research your.email@example.com"

CRT_SH = "https://crt.sh"
USPTO_TSDR = "https://tsdrapi.uspto.gov/ts/cd"

# Horizon the hazard model predicts over, in weeks.
HORIZON_WEEKS = 26

# A Form 4 filer is "regularly active" if they filed at least this many
# discretionary transactions in the trailing year. Below this, a gap in
# their filings is noise rather than a blackout.
FORM4_MIN_PRIOR = 4
FORM4_LOOKBACK_WEEKS = 52
```

```python
# src/deal/warehouse.py
"""DuckDB warehouse. Rebuild-from-scratch is cheap; there are no migrations."""
from pathlib import Path

import duckdb

SCHEMA = """
-- Point-in-time listing spans. delisted may be NULL (still listed).
CREATE TABLE IF NOT EXISTS universe (
    cik        VARCHAR PRIMARY KEY,
    name       VARCHAR,
    listed     DATE,
    delisted   DATE
);

CREATE TABLE IF NOT EXISTS panel (
    cik        VARCHAR,
    week       DATE,          -- ISO week, identified by its Monday
    PRIMARY KEY (cik, week)
);

CREATE TABLE IF NOT EXISTS deals (
    cik            VARCHAR,
    agreement_date DATE,      -- definitive agreement, from EDGAR
    rumor_date     DATE,      -- hand-collected subsample only, else NULL
    acquirer       VARCHAR,
    PRIMARY KEY (cik, agreement_date)
);

-- Every signal lands here. public_ts is when an OUTSIDER could have known.
CREATE TABLE IF NOT EXISTS signals (
    cik        VARCHAR,
    kind       VARCHAR,       -- 'ct' | 'tm_assign' | 'tm_itu' | 'form4_gap'
    public_ts  DATE,
    value      DOUBLE,
    PRIMARY KEY (cik, kind, public_ts)
);

-- Append-only. Never UPDATE, never DELETE.
CREATE TABLE IF NOT EXISTS predictions (
    run_ts     TIMESTAMP,
    cik        VARCHAR,
    week       DATE,
    prob       DOUBLE,
    features   VARCHAR        -- JSON of the exact vector used
);
"""


def connect(path: str = "data/deal.duckdb") -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)
```

```python
# src/deal/panel.py
"""Company-week panel over a point-in-time universe.

A company appears from the first ISO week on or after it listed, through the
week containing its delisting. Companies that left the sample MUST remain --
they are where the positives live, and dropping them is the survivorship bug
that quietly deletes the outcome you are modelling.
"""
import datetime as dt


def iso_monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def build(con, start: dt.date, end: dt.date) -> int:
    rows = con.execute(
        "SELECT cik, listed, delisted FROM universe"
    ).fetchall()

    out = []
    for cik, listed, delisted in rows:
        first = iso_monday(max(listed, start))
        if listed > first:            # listing mid-week: start the next week
            first += dt.timedelta(days=7)
        last = iso_monday(min(delisted or end, end))
        week = first
        while week <= last:
            out.append({"cik": cik, "week": week})
            week += dt.timedelta(days=7)

    if not out:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO panel VALUES ($cik, $week)", out
    )
    return len(out)
```

Add `src/deal/__init__.py` (empty) and:

```toml
# pyproject.toml
[project]
name = "deal"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["duckdb", "polars", "httpx", "statsmodels", "scikit-learn", "numpy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/deal"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e . && pytest tests/test_panel.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/deal/ tests/test_panel.py
git commit -m "feat: point-in-time company-week panel"
```

---

### Task 2: Deal labels from EDGAR

**Files:**
- Create: `src/deal/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `warehouse`, `panel.iso_monday`.
- Produces: `labels.parse_filing(raw: dict) -> dict | None`, `labels.label_panel(con, horizon_weeks: int) -> int` which adds column `y` to `panel` (1 if an agreement falls in the next `horizon_weeks`, else 0).

A merger proxy (`DEFM14A`) is the cleanest marker of a public-target deal; an 8-K Item 1.01 catches the rest. Both carry an exact EDGAR filing date, which is genuinely public the day it lands.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels.py
import datetime as dt

import pytest

from deal import labels, panel, warehouse


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    return c


def test_parse_filing_uses_filing_date_not_period():
    # period_of_report is when the event happened; filing date is when the
    # world found out. Only the latter is knowable in real time.
    raw = {
        "cik": "0000123",
        "form": "DEFM14A",
        "file_date": "2025-06-10",
        "period_of_report": "2025-05-01",
    }
    assert labels.parse_filing(raw)["agreement_date"] == dt.date(2025, 6, 10)


def test_parse_filing_ignores_unrelated_forms():
    assert labels.parse_filing({"cik": "1", "form": "10-K",
                                "file_date": "2025-01-01"}) is None


def test_label_marks_weeks_inside_the_horizon(con):
    con.execute("INSERT INTO universe VALUES ('C','Co','2025-01-01',NULL)")
    panel.build(con, dt.date(2025, 1, 1), dt.date(2025, 4, 1))
    con.execute("INSERT INTO deals VALUES ('C','2025-03-03',NULL,'Acq')")
    labels.label_panel(con, horizon_weeks=4)

    y_close = con.execute(
        "SELECT y FROM panel WHERE cik='C' AND week=DATE '2025-02-10'"
    ).fetchone()[0]
    y_far = con.execute(
        "SELECT y FROM panel WHERE cik='C' AND week=DATE '2025-01-06'"
    ).fetchone()[0]
    assert y_close == 1   # 3 weeks before the agreement
    assert y_far == 0     # 8 weeks before -- outside a 4-week horizon


def test_label_never_marks_weeks_after_the_agreement(con):
    con.execute("INSERT INTO universe VALUES ('C','Co','2025-01-01',NULL)")
    panel.build(con, dt.date(2025, 1, 1), dt.date(2025, 4, 1))
    con.execute("INSERT INTO deals VALUES ('C','2025-02-03',NULL,'Acq')")
    labels.label_panel(con, horizon_weeks=8)
    after = con.execute(
        "SELECT coalesce(sum(y),0) FROM panel "
        "WHERE cik='C' AND week > DATE '2025-02-03'"
    ).fetchone()[0]
    assert after == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL with `ImportError: cannot import name 'labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/labels.py
"""Deal labels from EDGAR.

DEFM14A (merger proxy) and 8-K Item 1.01 (material definitive agreement) are
the two forms that mark a public-target deal. The label is the EDGAR FILING
date -- period_of_report describes when the event occurred and is not what an
outsider could have known at the time.
"""
import datetime as dt

import httpx

from . import config

DEAL_FORMS = {"DEFM14A", "8-K"}


def parse_filing(raw: dict) -> dict | None:
    if raw.get("form") not in DEAL_FORMS:
        return None
    return {
        "cik": raw["cik"].lstrip("0") or "0",
        "agreement_date": dt.date.fromisoformat(raw["file_date"]),
        "rumor_date": None,
        "acquirer": raw.get("acquirer"),
    }


def fetch_filings(cik: str) -> list[dict]:
    """Submissions feed for one CIK. SEC requires a descriptive User-Agent."""
    r = httpx.get(
        f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json",
        headers={"User-Agent": config.EDGAR_UA},
        timeout=30,
    )
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    return [
        {"cik": cik, "form": f, "file_date": d}
        for f, d in zip(recent["form"], recent["filingDate"])
    ]


def label_panel(con, horizon_weeks: int) -> int:
    """Add y=1 to panel rows within horizon_weeks BEFORE an agreement date.

    Rows at or after the agreement are 0: once the deal is public there is
    nothing left to predict, and marking them would be pure lookahead.
    """
    con.execute("ALTER TABLE panel ADD COLUMN IF NOT EXISTS y TINYINT")
    con.execute("UPDATE panel SET y = 0")
    con.execute(
        """
        UPDATE panel SET y = 1
        WHERE EXISTS (
            SELECT 1 FROM deals d
            WHERE d.cik = panel.cik
              AND panel.week < d.agreement_date
              AND panel.week >= d.agreement_date - INTERVAL (?) WEEK
        )
        """,
        [horizon_weeks],
    )
    return con.execute("SELECT sum(y) FROM panel").fetchone()[0] or 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_labels.py -v`
Expected: 4 passed

- [ ] **Step 5: Record the rumor-gap subsample protocol**

Create `docs/rumor-gap-protocol.md`:

```markdown
# Rumor-gap subsample

Lead time measured against the definitive-agreement date overstates the
signal by however long the deal was publicly rumoured beforehand.

Protocol:
1. Draw 50 deals at random from `deals` with a fixed seed (`seed=20260728`).
2. For each, search a news archive for the earliest public report of a
   sale process. Record it in `deals.rumor_date`.
3. Report the distribution of `agreement_date - rumor_date`.
4. Report every lead-time result twice: against agreement date, and
   discounted by the median gap.

50 is enough to estimate the gap distribution. Collecting it for all deals
buys precision that changes no conclusion.
```

- [ ] **Step 6: Commit**

```bash
git add src/deal/labels.py tests/test_labels.py docs/rumor-gap-protocol.md
git commit -m "feat: EDGAR deal labels with horizon windowing"
```

---

### Task 3: Certificate Transparency signal

**Files:**
- Create: `src/deal/sig_ct.py`
- Test: `tests/test_sig_ct.py`

**Interfaces:**
- Consumes: `warehouse`.
- Produces: `sig_ct.parse_entry(entry: dict) -> dict`, `sig_ct.novel_names(entries: list[dict], known: set[str]) -> list[dict]`, `sig_ct.load(con, cik: str, domain: str) -> int`.

CT logs are append-only and a cert is logged at issuance, so `public_ts` is unambiguous — this is the cleanest timestamp in the project.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sig_ct.py
import datetime as dt

from deal import sig_ct

ENTRY = {
    "name_value": "checkout.example.com\nwww.example.com",
    "not_before": "2025-04-01T00:00:00",
    "entry_timestamp": "2025-04-02T11:30:00",
}


def test_public_ts_is_log_entry_time_not_cert_validity_start():
    # not_before can be backdated; the log entry is when it became visible.
    assert sig_ct.parse_entry(ENTRY)["public_ts"] == dt.date(2025, 4, 2)


def test_parse_splits_multi_san_certs():
    assert sig_ct.parse_entry(ENTRY)["names"] == [
        "checkout.example.com", "www.example.com"
    ]


def test_novel_names_ignores_already_seen_hosts():
    known = {"www.example.com"}
    out = sig_ct.novel_names([ENTRY], known)
    assert len(out) == 1
    assert out[0]["name"] == "checkout.example.com"


def test_novel_names_deduplicates_within_a_batch():
    entries = [ENTRY, {**ENTRY, "entry_timestamp": "2025-04-05T09:00:00"}]
    out = sig_ct.novel_names(entries, set())
    names = [o["name"] for o in out]
    assert len(names) == len(set(names))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sig_ct.py -v`
Expected: FAIL with `ImportError: cannot import name 'sig_ct'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/sig_ct.py
"""Certificate Transparency -> signals(kind='ct').

Every publicly-trusted TLS cert is written to an append-only CT log at
issuance, so a company standing up infrastructure for a merged brand or an
integration endpoint leaves a timestamped public trace it cannot retract.

The signal is the count of NOVEL hostnames appearing in a week. Renewals of
known hosts are noise; first appearances are the event.
"""
import datetime as dt

import httpx

from . import config


def parse_entry(entry: dict) -> dict:
    return {
        "names": [n.strip() for n in entry["name_value"].split("\n") if n.strip()],
        # entry_timestamp is when the log accepted it. not_before is issuer-
        # controlled and can predate visibility, so it must not be used.
        "public_ts": dt.datetime.fromisoformat(entry["entry_timestamp"]).date(),
    }


def novel_names(entries: list[dict], known: set[str]) -> list[dict]:
    seen = set(known)
    out = []
    for entry in sorted(entries, key=lambda e: e["entry_timestamp"]):
        parsed = parse_entry(entry)
        for name in parsed["names"]:
            if name in seen:
                continue
            seen.add(name)
            out.append({"name": name, "public_ts": parsed["public_ts"]})
    return out


def fetch(domain: str) -> list[dict]:
    r = httpx.get(
        f"{config.CRT_SH}/",
        params={"q": f"%.{domain}", "output": "json"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def load(con, cik: str, domain: str) -> int:
    # crt.sh returns the domain's full history on every call, and novel_names
    # walks it in log order -- so novelty is computed from scratch each run and
    # the result is identical whether this is the first run or the fiftieth.
    # No persisted host list, nothing to keep in sync.
    fresh = novel_names(fetch(domain), set())
    if not fresh:
        return 0

    weekly: dict[dt.date, int] = {}
    for item in fresh:
        weekly[item["public_ts"]] = weekly.get(item["public_ts"], 0) + 1

    con.executemany(
        "INSERT OR IGNORE INTO signals VALUES ($cik, 'ct', $public_ts, $value)",
        [{"cik": cik, "public_ts": d, "value": float(n)} for d, n in weekly.items()],
    )
    return len(fresh)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sig_ct.py -v`
Expected: 4 passed

- [ ] **Step 5: Smoke-test against crt.sh**

Run:
```bash
python -c "
from deal import sig_ct
e = sig_ct.fetch('github.com')
print('entries', len(e))
print('novel', len(sig_ct.novel_names(e, set())))
"
```
Expected: thousands of entries, novel count nonzero. crt.sh is slow — a 120s timeout is deliberate, not generous.

- [ ] **Step 6: Commit**

```bash
git add src/deal/sig_ct.py tests/test_sig_ct.py
git commit -m "feat: certificate transparency novel-hostname signal"
```

---

### Task 4: USPTO trademark signal

**Files:**
- Create: `src/deal/sig_uspto.py`
- Test: `tests/test_sig_uspto.py`

**Interfaces:**
- Consumes: `warehouse`.
- Produces: `sig_uspto.parse_assignment(raw: dict) -> dict`, `sig_uspto.load_assignments(con, cik: str, rows: list[dict]) -> int`.

The USPTO Trademark Assignment Dataset records mergers, name changes, **security interest agreements**, and their releases — a lien release on IP is the pre-sale-cleanup tell, and it arrives as structured bulk data.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sig_uspto.py
import datetime as dt

import pytest

from deal import sig_uspto, warehouse

RAW = {
    "execution_date": "2025-03-01",
    "recorded_date": "2025-06-15",
    "conveyance_text": "RELEASE BY SECURED PARTY",
}


def test_public_ts_is_recorded_date_not_execution_date():
    # Execution is private until the USPTO records it. Using execution_date
    # would hand the model months of information it could not have had.
    assert sig_uspto.parse_assignment(RAW)["public_ts"] == dt.date(2025, 6, 15)


def test_lien_release_is_flagged():
    assert sig_uspto.parse_assignment(RAW)["kind"] == "tm_release"


def test_security_interest_grant_is_distinct_from_release():
    grant = {**RAW, "conveyance_text": "SECURITY INTEREST"}
    assert sig_uspto.parse_assignment(grant)["kind"] == "tm_lien"


def test_plain_assignment_is_classified_as_assign():
    other = {**RAW, "conveyance_text": "ASSIGNS THE ENTIRE INTEREST"}
    assert sig_uspto.parse_assignment(other)["kind"] == "tm_assign"


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    return c


def test_load_writes_one_signal_row_per_assignment(con):
    assert sig_uspto.load_assignments(con, "C", [RAW]) == 1
    kind = con.execute("SELECT kind FROM signals WHERE cik='C'").fetchone()[0]
    assert kind == "tm_release"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sig_uspto.py -v`
Expected: FAIL with `ImportError: cannot import name 'sig_uspto'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/sig_uspto.py
"""USPTO trademark assignments -> signals.

The assignment record captures mergers, name changes, security interests and
their releases. A company clearing liens off its IP is tidying its balance
sheet, which is standard pre-sale housekeeping.

ponytail: substring matching on conveyance_text, not a taxonomy. The field is
a short controlled-ish phrase and three buckets is all the model consumes.
Upgrade to the full USPTO conveyance code list only if a bucket proves noisy.
"""
import datetime as dt


def parse_assignment(raw: dict) -> dict:
    text = raw.get("conveyance_text", "").upper()
    if "RELEASE" in text:
        kind = "tm_release"
    elif "SECURITY INTEREST" in text:
        kind = "tm_lien"
    else:
        kind = "tm_assign"
    return {
        # recorded_date is when the USPTO published it. execution_date is
        # when the parties signed, which nobody outside the deal could see.
        "public_ts": dt.date.fromisoformat(raw["recorded_date"]),
        "kind": kind,
    }


def load_assignments(con, cik: str, rows: list[dict]) -> int:
    parsed = [parse_assignment(r) for r in rows]
    if not parsed:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO signals VALUES ($cik, $kind, $public_ts, 1.0)",
        [{"cik": cik, **p} for p in parsed],
    )
    return len(parsed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sig_uspto.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/sig_uspto.py tests/test_sig_uspto.py
git commit -m "feat: USPTO trademark assignment and lien-release signal"
```

---

### Task 5: Form 4 discretionary blackout signal

The absence signal. Insiders stop *discretionary* selling once a deal is live, while pre-scheduled 10b5-1 sales keep executing — so the tell is a gap in discretionary filings by someone who was previously regular.

**Files:**
- Create: `src/deal/sig_form4.py`
- Test: `tests/test_sig_form4.py`

**Interfaces:**
- Consumes: `warehouse`.
- Produces: `sig_form4.is_discretionary(raw: dict) -> bool`, `sig_form4.blackout_weeks(filings: list[dict], min_prior: int, lookback_weeks: int) -> list[dict]`, `sig_form4.load(con, cik: str, filings: list[dict]) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sig_form4.py
import datetime as dt

from deal import sig_form4


def _filing(day: str, plan: bool = False) -> dict:
    return {"file_date": day, "is_10b5_1": plan}


def test_10b5_1_transactions_are_not_discretionary():
    # Scheduled plan sales execute regardless of what management knows, so
    # they carry no information. Only discretionary trades do.
    assert sig_form4.is_discretionary(_filing("2025-01-06", plan=True)) is False
    assert sig_form4.is_discretionary(_filing("2025-01-06")) is True


def test_no_blackout_flagged_for_an_irregular_filer():
    # Two prior filings is not a pattern; its absence is not evidence.
    filings = [_filing("2025-01-06"), _filing("2025-02-03")]
    assert sig_form4.blackout_weeks(filings, min_prior=4, lookback_weeks=52) == []


def test_blackout_flagged_after_a_regular_filer_goes_quiet():
    filings = [_filing(f"2025-0{m}-06") for m in range(1, 7)]  # 6 monthly
    out = sig_form4.blackout_weeks(filings, min_prior=4, lookback_weeks=52)
    assert out, "a regular filer going silent should raise a blackout"
    assert all(o["public_ts"] > dt.date(2025, 6, 6) for o in out)


def test_plan_only_activity_still_counts_as_a_blackout():
    # Discretionary stops, scheduled continues -- that is the exact pattern.
    filings = [_filing(f"2025-0{m}-06") for m in range(1, 7)]
    filings += [_filing(f"2025-0{m}-20", plan=True) for m in range(7, 10)]
    assert sig_form4.blackout_weeks(filings, min_prior=4, lookback_weeks=52)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sig_form4.py -v`
Expected: FAIL with `ImportError: cannot import name 'sig_form4'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/sig_form4.py
"""Form 4 discretionary blackout -> signals(kind='form4_gap').

You can suppress an announcement; you cannot announce a non-event. Once a deal
is live, insiders stop trading at their own discretion while pre-scheduled
10b5-1 sales carry on. A regular discretionary filer going quiet is therefore
a harder signal to conceal than anything the company publishes.

Caveat for the sample window: the 10b5-1 checkbox on Form 4 only became
mandatory in 2023. Before that, plan status has to be inferred and this
signal should be treated as unavailable rather than guessed.
"""
import datetime as dt

MEDIAN_GAP_MULTIPLE = 3.0


def is_discretionary(raw: dict) -> bool:
    return not raw.get("is_10b5_1", False)


def blackout_weeks(filings: list[dict], min_prior: int, lookback_weeks: int) -> list[dict]:
    disc = sorted(
        dt.date.fromisoformat(f["file_date"])
        for f in filings
        if is_discretionary(f)
    )
    if len(disc) < min_prior:
        return []

    gaps = [(b - a).days for a, b in zip(disc, disc[1:])]
    if not gaps:
        return []
    typical = sorted(gaps)[len(gaps) // 2]
    threshold = max(typical * MEDIAN_GAP_MULTIPLE, 21)

    # Silence is only observable in arrears: flag the week at which the gap
    # first EXCEEDS the threshold, never the week the filer went quiet.
    last = disc[-1]
    flagged = last + dt.timedelta(days=int(threshold))
    horizon = last + dt.timedelta(weeks=lookback_weeks)

    out, week = [], flagged
    while week <= horizon:
        out.append({"public_ts": week, "value": 1.0})
        week += dt.timedelta(days=7)
    return out


def load(con, cik: str, filings: list[dict]) -> int:
    rows = blackout_weeks(filings, min_prior=4, lookback_weeks=52)
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO signals VALUES ($cik, 'form4_gap', $public_ts, $value)",
        [{"cik": cik, **r} for r in rows],
    )
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sig_form4.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/sig_form4.py tests/test_sig_form4.py
git commit -m "feat: Form 4 discretionary blackout signal"
```

---

### Task 6: Feature assembly — the lookahead firewall

Every as-of rule lives here so there is exactly one place lookahead can enter and exactly one place to test for it. The adversarial test in Step 1 is the most important test in the project.

**Files:**
- Create: `src/deal/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `panel` (with `y`), `signals`.
- Produces: `features.FEATURE_COLS: list[str]`, `features.build(con, decay_weeks: int = 8) -> polars.DataFrame` with columns `cik, week, y` plus `FEATURE_COLS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import datetime as dt

import pytest

from deal import features, panel, warehouse


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    c.execute("INSERT INTO universe VALUES ('C','Co','2025-01-01',NULL)")
    panel.build(c, dt.date(2025, 1, 1), dt.date(2025, 4, 1))
    c.execute("ALTER TABLE panel ADD COLUMN IF NOT EXISTS y TINYINT")
    c.execute("UPDATE panel SET y = 0")
    return c


def test_future_signal_does_not_leak_into_an_earlier_week(con):
    """THE test. A signal public in March must be invisible in January."""
    con.execute("INSERT INTO signals VALUES ('C','ct','2025-03-10',5.0)")
    df = features.build(con)
    jan = df.filter(
        (df["cik"] == "C") & (df["week"] == dt.date(2025, 1, 6))
    )
    assert jan["ct"][0] == 0.0


def test_signal_is_visible_in_the_week_it_becomes_public(con):
    con.execute("INSERT INTO signals VALUES ('C','ct','2025-03-10',5.0)")
    df = features.build(con)
    wk = df.filter(
        (df["cik"] == "C") & (df["week"] == dt.date(2025, 3, 10))
    )
    assert wk["ct"][0] == 5.0


def test_signal_decays_and_does_not_persist_forever(con):
    con.execute("INSERT INTO signals VALUES ('C','ct','2025-01-06',8.0)")
    df = features.build(con, decay_weeks=2)
    fresh = df.filter(
        (df["cik"] == "C") & (df["week"] == dt.date(2025, 1, 6))
    )
    late = df.filter(
        (df["cik"] == "C") & (df["week"] == dt.date(2025, 3, 24))
    )
    assert fresh["ct"][0] == 8.0
    assert late["ct"][0] < 0.05 * fresh["ct"][0]  # 11 weeks on, ~5 half-lives


def test_all_declared_feature_columns_exist(con):
    df = features.build(con)
    assert set(features.FEATURE_COLS) <= set(df.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/features.py
"""Signals -> as-of feature matrix.

This is the only module permitted to join a signal onto a panel row. The join
condition `s.public_ts <= p.week` is the lookahead firewall: a fact is
available in week t only if it was public on or before the Monday of week t.

Signals decay exponentially rather than persisting, because a cert issued
eight months ago says little about this week.
"""
import polars as pl

FEATURE_COLS = ["ct", "tm_release", "tm_lien", "tm_assign", "form4_gap"]

_HALF_LIFE_SCALE = 0.5


def build(con, decay_weeks: int = 8) -> pl.DataFrame:
    rows = con.execute(
        f"""
        SELECT p.cik, p.week, p.y, s.kind,
               sum(s.value * pow(?, date_diff('day', s.public_ts, p.week) / 7.0
                                    / ?)) AS val
        FROM panel p
        LEFT JOIN signals s
          ON s.cik = p.cik
         AND s.public_ts <= p.week      -- the firewall
        GROUP BY p.cik, p.week, p.y, s.kind
        """,
        [_HALF_LIFE_SCALE, decay_weeks],
    ).fetchall()

    long = pl.DataFrame(
        [
            {"cik": c, "week": w, "y": int(y or 0),
             "kind": k or "_none", "val": float(v or 0.0)}
            for c, w, y, k, v in rows
        ],
        schema={"cik": pl.Utf8, "week": pl.Date, "y": pl.Int8,
                "kind": pl.Utf8, "val": pl.Float64},
    )

    wide = long.pivot(on="kind", index=["cik", "week", "y"],
                      values="val", aggregate_function="sum")

    for col in FEATURE_COLS:
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(col))
    return wide.with_columns(
        [pl.col(c).fill_null(0.0) for c in FEATURE_COLS]
    ).select(["cik", "week", "y", *FEATURE_COLS])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/features.py tests/test_features.py
git commit -m "feat: as-of feature assembly with lookahead firewall"
```

---

### Task 7: Discrete-time hazard model and evaluation

**Files:**
- Create: `src/deal/model.py`, `src/deal/evaluate.py`
- Test: `tests/test_model.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `features.build`, `features.FEATURE_COLS`.
- Produces: `model.fit(df: polars.DataFrame) -> statsmodels result`, `model.predict(res, df) -> polars.Series`, `model.split_by_time(df, cutoff: date) -> tuple[pl.DataFrame, pl.DataFrame]`, `evaluate.pr_auc(y, p) -> float`, `evaluate.lift_at_k(y, p, k: int) -> float`, `evaluate.lead_time_days(signal_ts, announce_ts, volume_ts) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import datetime as dt

import polars as pl

from deal import model


def _frame(n: int = 400) -> pl.DataFrame:
    # ct carries real signal; the rest are noise.
    return pl.DataFrame({
        "cik": [f"C{i%40}" for i in range(n)],
        "week": [dt.date(2025, 1, 6) + dt.timedelta(weeks=i % 40) for i in range(n)],
        "y": [1 if i % 10 == 0 else 0 for i in range(n)],
        "ct": [5.0 if i % 10 == 0 else 0.0 for i in range(n)],
        "tm_release": [0.0] * n,
        "tm_lien": [0.0] * n,
        "tm_assign": [0.0] * n,
        "form4_gap": [0.0] * n,
    })


def test_split_by_time_never_puts_a_later_week_in_train():
    train, test = model.split_by_time(_frame(), dt.date(2025, 6, 1))
    assert train["week"].max() < dt.date(2025, 6, 1)
    assert test["week"].min() >= dt.date(2025, 6, 1)


def test_fit_recovers_a_positive_coefficient_on_the_real_signal():
    res = model.fit(_frame())
    assert res.params["ct"] > 0


def test_predictions_are_probabilities():
    df = _frame()
    p = model.predict(model.fit(df), df)
    assert p.min() >= 0.0 and p.max() <= 1.0
```

```python
# tests/test_evaluate.py
import datetime as dt

import numpy as np

from deal import evaluate


def test_pr_auc_is_near_one_for_a_perfect_ranker():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluate.pr_auc(y, p) > 0.99


def test_lift_at_k_beats_one_when_the_ranking_works():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.zeros(90), np.ones(10)])
    assert evaluate.lift_at_k(y, p, k=10) == 10.0


def test_lead_time_reports_signal_ahead_of_the_volume_clock():
    out = evaluate.lead_time_days(
        signal_ts=dt.date(2025, 1, 1),
        announce_ts=dt.date(2025, 4, 1),
        volume_ts=dt.date(2025, 3, 1),
    )
    assert out["signal_lead"] == 90
    assert out["volume_lead"] == 31
    assert out["signal_is_upstream"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py tests/test_evaluate.py -v`
Expected: FAIL with `ImportError: cannot import name 'model'`

- [ ] **Step 3: Write the model**

```python
# src/deal/model.py
"""Discrete-time hazard model.

A discrete-time hazard model IS a pooled logistic regression over the
company-week panel: P(announce in week t | survived to t). Censoring is
encoded by the row structure -- a company contributes rows until it exits --
so no survival library is needed and none is used.

ponytail: statsmodels.Logit, not lifelines or a GBM. The panel is small, the
output that matters is a calibrated probability plus interpretable
coefficients, and a linear model gives both for free. Reach for gradient
boosting only if the linear fit leaves obvious structure on the table.
"""
import datetime as dt

import numpy as np
import polars as pl
import statsmodels.api as sm

from .features import FEATURE_COLS


def split_by_time(df: pl.DataFrame, cutoff: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Time-ordered split. Random splits leak: the same deal's weeks would
    land on both sides, and the model would learn that deal from itself."""
    return df.filter(pl.col("week") < cutoff), df.filter(pl.col("week") >= cutoff)


def _design(df: pl.DataFrame):
    x = df.select(FEATURE_COLS).to_pandas()
    # Calendar control: deal flow is seasonal and the baseline hazard is not
    # flat. Quarter dummies absorb that without a full duration spline.
    quarters = df.select(
        (pl.col("week").dt.month() - 1) // 3
    ).to_series().to_numpy()
    for q in (1, 2, 3):
        x[f"q{q}"] = (quarters == q).astype(float)
    return sm.add_constant(x, has_constant="add")


def fit(df: pl.DataFrame):
    y = df["y"].to_pandas()
    # Cluster by company: a firm's weeks are serially correlated, and naive
    # standard errors would be far too small.
    groups = df["cik"].to_pandas()
    return sm.Logit(y, _design(df)).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": groups}
    )


def predict(res, df: pl.DataFrame) -> pl.Series:
    return pl.Series("prob", np.asarray(res.predict(_design(df))))
```

- [ ] **Step 4: Write the evaluation**

```python
# src/deal/evaluate.py
"""Evaluation for a 0.09%-positive panel.

PR-AUC and lift, never ROC-AUC: with a true-negative pool this large, ROC-AUC
reads around 0.9 for a model with no practical value.

lead_time_days is the headline. Abnormal trading volume marks the moment the
leak reached the market. A signal that fires BEFORE that window is genuinely
upstream of the leak rather than another echo of it -- and that holds as a
finding even if precision is poor.
"""
import datetime as dt

import numpy as np
from sklearn.metrics import average_precision_score


def pr_auc(y, p) -> float:
    return float(average_precision_score(np.asarray(y), np.asarray(p)))


def lift_at_k(y, p, k: int) -> float:
    y, p = np.asarray(y), np.asarray(p)
    base = y.mean()
    if base == 0:
        return 0.0
    top = y[np.argsort(-p)[:k]]
    return float(top.mean() / base)


def lead_time_days(signal_ts: dt.date, announce_ts: dt.date,
                   volume_ts: dt.date) -> dict:
    signal_lead = (announce_ts - signal_ts).days
    volume_lead = (announce_ts - volume_ts).days
    return {
        "signal_lead": signal_lead,
        "volume_lead": volume_lead,
        "signal_is_upstream": signal_lead > volume_lead,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_model.py tests/test_evaluate.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/deal/model.py src/deal/evaluate.py tests/test_model.py tests/test_evaluate.py
git commit -m "feat: discrete-time hazard model with PR-AUC and lead-time evaluation"
```

---

### Task 8: Continuous weekly feed with append-only log

**Files:**
- Create: `src/deal/live.py`, `EVALUATION.md`
- Test: `tests/test_live.py`

**Interfaces:**
- Consumes: `features.build`, `model.fit`, `model.predict`.
- Produces: `live.run_week(con, res, run_ts: datetime) -> int`, `live.assert_append_only(con, before: int) -> None`.

**Note on the abnormal-volume clock:** `evaluate.lead_time_days` needs a `volume_ts`. Options data is the sharper input but costs money; abnormal daily equity volume is free and well established as a leak marker. Use equity volume, and say so in the writeup.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live.py
import datetime as dt
import json

import polars as pl
import pytest

from deal import live, model, panel, warehouse


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    c.execute("INSERT INTO universe VALUES ('C','Co','2025-01-01',NULL)")
    panel.build(c, dt.date(2025, 1, 1), dt.date(2025, 3, 1))
    c.execute("ALTER TABLE panel ADD COLUMN IF NOT EXISTS y TINYINT")
    c.execute("UPDATE panel SET y = 0")
    return c


def _fitted():
    n = 400
    df = pl.DataFrame({
        "cik": [f"C{i%40}" for i in range(n)],
        "week": [dt.date(2025, 1, 6) + dt.timedelta(weeks=i % 40) for i in range(n)],
        "y": [1 if i % 10 == 0 else 0 for i in range(n)],
        "ct": [5.0 if i % 10 == 0 else 0.0 for i in range(n)],
        "tm_release": [0.0] * n, "tm_lien": [0.0] * n,
        "tm_assign": [0.0] * n, "form4_gap": [0.0] * n,
    })
    return model.fit(df)


def test_run_week_records_the_feature_vector_it_used(con):
    live.run_week(con, _fitted(), dt.datetime(2025, 2, 3, 12, 0))
    feats = con.execute("SELECT features FROM predictions LIMIT 1").fetchone()[0]
    assert "ct" in json.loads(feats)


def test_rerunning_appends_and_never_overwrites(con):
    live.run_week(con, _fitted(), dt.datetime(2025, 2, 3, 12, 0))
    first = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    live.run_week(con, _fitted(), dt.datetime(2025, 2, 10, 12, 0))
    second = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert second > first


def test_assert_append_only_raises_if_history_shrank(con):
    live.run_week(con, _fitted(), dt.datetime(2025, 2, 3, 12, 0))
    n = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    con.execute("DELETE FROM predictions")
    with pytest.raises(AssertionError):
        live.assert_append_only(con, before=n)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live.py -v`
Expected: FAIL with `ImportError: cannot import name 'live'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/live.py
"""Weekly forward run with an immutable prediction log.

A forward log is only evidence if it cannot be rewritten. Predictions are
appended with the exact feature vector used, and nothing ever mutates them --
if past predictions were recomputable, they would eventually get recomputed
with a better model and the out-of-sample record would quietly become
in-sample.

ponytail: weekly batch, no scheduler, no queue. The underlying signals move
at daily-to-weekly cadence, so a cron line invoking this module is the whole
orchestration layer. Add real infrastructure only if the cadence changes.
"""
import datetime as dt
import json

from . import features
from .features import FEATURE_COLS
from .model import predict


def run_week(con, res, run_ts: dt.datetime) -> int:
    df = features.build(con)
    week = df["week"].max()
    current = df.filter(df["week"] == week)
    if current.height == 0:
        return 0

    probs = predict(res, current)
    rows = [
        {
            "run_ts": run_ts,
            "cik": current["cik"][i],
            "week": current["week"][i],
            "prob": float(probs[i]),
            "features": json.dumps(
                {c: float(current[c][i]) for c in FEATURE_COLS}
            ),
        }
        for i in range(current.height)
    ]
    con.executemany(
        "INSERT INTO predictions VALUES ($run_ts, $cik, $week, $prob, $features)",
        rows,
    )
    return len(rows)


def assert_append_only(con, before: int) -> None:
    """Guard for the weekly job: the log may only ever grow."""
    now = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert now >= before, (
        f"prediction log shrank from {before} to {now} -- the forward "
        f"record has been tampered with and is no longer out-of-sample"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_live.py -v`
Expected: 3 passed

- [ ] **Step 5: Pre-register the evaluation**

Create `EVALUATION.md`. **Commit this before the first live run** — criteria written afterwards are criteria fitted to the result.

```markdown
# Pre-registered evaluation

Committed before the first forward run. Not to be edited afterwards; revisions
go in a dated appendix.

## Live period
26 weeks from the first run recorded in `predictions`.

## Primary
Median `signal_lead - volume_lead` across deals announced in the live period.
Success: signal fires at least 14 days before the abnormal-volume window.

## Secondary
- PR-AUC on live weeks, versus a base-rate-only null model.
- Lift at k=20 on the top-ranked companies each week.

## Declared in advance
- Expected positives in 26 weeks: roughly 40 announcements across the universe.
- A null result is a publishable result and will be reported as one.
- No feature may be added, and no threshold retuned, during the live period.
  Changes start a new live period with a new pre-registration.
```

- [ ] **Step 6: Add the weekly cron line**

```bash
# Run Mondays at 06:00 UTC. This is the entire orchestration layer.
0 6 * * 1 cd /path/to/repo && python -m deal.live >> logs/live.log 2>&1
```

Add `src/deal/__main__.py`:

```python
# src/deal/__main__.py
"""Entry point for the weekly cron run."""
import datetime as dt
import pickle
from pathlib import Path

from . import live, warehouse

if __name__ == "__main__":
    con = warehouse.connect()
    warehouse.init_schema(con)
    before = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    with Path("data/model.pkl").open("rb") as fh:
        res = pickle.load(fh)
    n = live.run_week(con, res, dt.datetime.now(dt.UTC))
    live.assert_append_only(con, before)
    print(f"{dt.datetime.now(dt.UTC).isoformat()} wrote {n} predictions")
```

- [ ] **Step 7: Run the whole suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/deal/live.py src/deal/__main__.py tests/test_live.py EVALUATION.md
git commit -m "feat: weekly forward run with append-only prediction log"
```

---

## Deliberately not in this plan

- **Hiring-freeze signal** — it needs either the Wayback careers scraper or a Revelio/WRDS licence, and it is the one Tier-1 signal whose data source is not settled. Add it as Task 9 once you know which. The `signals` table already accommodates it with no schema change.
- **Backtesting a trading strategy** — this measures lead time, not returns. A backtester before a validated signal turns the project into a bot that never ships a paper.
- **Gradient boosting** — the linear fit has to leave visible structure on the table first.
- **Options-implied leak clock** — abnormal equity volume is free and adequate; buy options data only if a referee asks.

## Sources

- [USPTO Trademark Assignment Dataset](https://www.uspto.gov/ip-policy/economic-research/research-datasets/trademark-assignment-dataset)
- [USPTO TSDR API](https://developer.uspto.gov/swagger/tsdr-api-v1)
- [crt.sh Certificate Transparency search](https://crt.sh)
- [Certificate Transparency log list](https://certificate.transparency.dev/logs/)
