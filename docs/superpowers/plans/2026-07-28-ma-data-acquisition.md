# M&A Signal Data Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Acquire every input the M&A precursor model needs — point-in-time universe, deal labels, fundamentals, insider transactions, USPTO assignments, and Certificate Transparency — into a local DuckDB warehouse.

**Architecture:** Bulk-first. Almost every source publishes a periodic bulk dump; per-company API loops are used only for gap-filling. All raw bytes cache to disk before parsing, so a re-parse never re-fetches. One entity crosswalk maps CIK ↔ name variants ↔ domain, and every non-SEC source joins through it.

**Tech Stack:** Python 3.11+, DuckDB, Polars, httpx, psycopg (crt.sh), pytest.

---

## Global Constraints

- Python 3.11+.
- **SEC allows at most 10 requests/second and requires a User-Agent with real contact info.** Exceeding it gets the IP banned. Every SEC call goes through `fetch.sec_get`, never raw `httpx`.
- **Raw bytes cache to `data/raw/<source>/<key>` before parsing.** A parser change must never trigger a re-download.
- Every fact carries `public_ts` — the date it became knowable to an outsider. For bulk datasets this is the filing/recording date, never the execution or period date.
- Warehouse is `data/deal.duckdb`, sharing the schema from the model plan (`2026-07-28-ma-precursor-model.md`). This plan adds tables; it does not alter existing ones.
- All bulk downloads are idempotent: re-running a loader inserts nothing new.
- Company names normalise through exactly one function, `crosswalk.normalize_name`. No ad-hoc string cleaning anywhere else.

---

## Source Inventory

| Source | Access | Bulk? | Cost | Keys on |
|---|---|---|---|---|
| EDGAR full-index | `sec.gov/Archives/edgar/full-index/` | Yes, quarterly | Free | CIK |
| Deal labels (DEFM14A, 8-K, SC TO-T, SC 13E3) | Same index | Yes | Free | CIK |
| Financial Statement Data Sets | `sec.gov/files/dera/data/...` | Yes, quarterly ZIP | Free | CIK |
| Insider Transactions Data Sets | `sec.gov/files/structureddata/data/...` | Yes, quarterly ZIP | Free | CIK |
| USPTO trademark assignments | `bulkdata.uspto.gov` | Yes, annual | Free | **Assignee name** |
| USPTO patent assignments | `bulkdata.uspto.gov` | Yes, annual | Free | **Assignee name** |
| Certificate Transparency | crt.sh Postgres | Yes, SQL | Free | **Domain** |
| Wikidata (CIK↔website) | SPARQL | Yes | Free | CIK + domain |
| Prices/volume | Stooq, or CRSP via WRDS | Yes | Free / licensed | Ticker |

The three sources that key on something other than CIK are why Task 3 exists and why it is the critical path.

---

## File Structure

```
src/deal/
  fetch.py       # rate-limited HTTP + disk cache. Everything downloads through here.
  universe.py    # EDGAR full-index -> point-in-time universe
  crosswalk.py   # CIK <-> normalized name <-> domain. CRITICAL PATH.
  load_deals.py  # deal labels from the index
  load_fund.py   # Financial Statement Data Sets -> fundamentals
  load_insider.py# Insider Transactions Data Sets -> form4 events
  load_uspto.py  # trademark + patent assignment bulk -> signals
  load_ct.py     # crt.sh Postgres -> signals
  coverage.py    # per-source coverage report. Run after every load.
tests/
  test_fetch.py  test_universe.py  test_crosswalk.py  test_load_deals.py
  test_load_fund.py  test_load_insider.py  test_load_uspto.py
  test_load_ct.py  test_coverage.py
```

---

### Task 1: Rate-limited fetcher with disk cache

Everything downloads through this. Getting the SEC rate limit wrong once costs you an IP ban, so it is centralised and tested first.

**Files:**
- Create: `src/deal/fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `config.EDGAR_UA`.
- Produces: `fetch.cache_path(source: str, key: str) -> pathlib.Path`, `fetch.cached(source: str, key: str, fetcher: Callable[[], bytes]) -> bytes`, `fetch.sec_get(url: str) -> bytes`, `fetch.RateLimiter(per_second: float)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch.py
import time

import pytest

from deal import fetch


def test_cached_calls_fetcher_once_then_serves_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    calls = []

    def fetcher() -> bytes:
        calls.append(1)
        return b"payload"

    assert fetch.cached("src", "k", fetcher) == b"payload"
    assert fetch.cached("src", "k", fetcher) == b"payload"
    assert len(calls) == 1, "second call must read disk, not refetch"


def test_cache_key_is_filesystem_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    # Keys come from URLs; slashes must not create directories.
    p = fetch.cache_path("src", "2024/QTR1/master.idx")
    assert "/" not in p.name


def test_rate_limiter_spaces_calls(monkeypatch):
    limiter = fetch.RateLimiter(per_second=20.0)
    start = time.monotonic()
    for _ in range(4):
        limiter.wait()
    # 4 calls at 20/s must take at least 3 intervals of 0.05s.
    assert time.monotonic() - start >= 0.15


def test_sec_rate_limit_is_at_most_ten_per_second():
    assert fetch.SEC_LIMITER.per_second <= 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetch.py -v`
Expected: FAIL with `ImportError: cannot import name 'fetch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/fetch.py
"""Rate-limited HTTP with a disk cache.

Every download in this project goes through here. Two reasons: the SEC bans
IPs that exceed 10 requests/second, and a parser bug should never cost a
re-download of several GB.
"""
import hashlib
import threading
import time
from pathlib import Path
from typing import Callable

import httpx

from . import config

CACHE_ROOT = Path("data/raw")


class RateLimiter:
    """Spaces calls to at most per_second. Thread-safe."""

    def __init__(self, per_second: float):
        self.per_second = per_second
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = max(now, self._next) + self._interval


# SEC's published ceiling is 10/s. Sitting at 8 leaves headroom for any
# other process on the same IP.
SEC_LIMITER = RateLimiter(per_second=8.0)


def cache_path(source: str, key: str) -> Path:
    # Keys are URLs or URL fragments; hash them so slashes and query strings
    # cannot escape into the directory structure.
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return CACHE_ROOT / source / digest


def cached(source: str, key: str, fetcher: Callable[[], bytes]) -> bytes:
    path = cache_path(source, key)
    if path.exists():
        return path.read_bytes()
    data = fetcher()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def sec_get(url: str) -> bytes:
    def _go() -> bytes:
        SEC_LIMITER.wait()
        r = httpx.get(url, headers={"User-Agent": config.EDGAR_UA},
                      timeout=120, follow_redirects=True)
        r.raise_for_status()
        return r.content

    return cached("sec", url, _go)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetch.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/fetch.py tests/test_fetch.py
git commit -m "feat: rate-limited SEC fetcher with disk cache"
```

---

### Task 2: Point-in-time universe from EDGAR full-index

The survivorship fix, and it is free. A company was listed and reporting in quarter Q if it filed a 10-K or 10-Q in quarter Q. Deriving the universe from filing activity gives point-in-time membership with no index-constituent licence and no survivorship bias — delisted and acquired companies appear in exactly the quarters they actually existed.

**Files:**
- Create: `src/deal/universe.py`
- Test: `tests/test_universe.py`

**Interfaces:**
- Consumes: `fetch.sec_get`, `warehouse`.
- Produces: `universe.parse_master_idx(raw: bytes) -> list[dict]`, `universe.quarters(start_year: int, end_year: int) -> list[tuple[int, int]]`, `universe.build(con, start_year: int, end_year: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_universe.py
import datetime as dt

import pytest

from deal import universe, warehouse

IDX = b"""Description:           Master Index
CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
320193|Apple Inc.|10-Q|2024-02-02|edgar/data/320193/x.txt
789019|Microsoft Corp|8-K|2024-01-30|edgar/data/789019/y.txt
1018724|Amazon.com Inc|10-K|2024-02-02|edgar/data/1018724/z.txt
"""


def test_parse_strips_the_header_block():
    rows = universe.parse_master_idx(IDX)
    assert len(rows) == 3
    assert rows[0]["cik"] == "320193"


def test_parse_keeps_filing_date_as_a_date():
    assert universe.parse_master_idx(IDX)[0]["file_date"] == dt.date(2024, 2, 2)


def test_quarters_are_inclusive_of_both_years():
    q = universe.quarters(2023, 2024)
    assert (2023, 1) in q and (2024, 4) in q
    assert len(q) == 8


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    return c


def test_only_periodic_filers_enter_the_universe(con):
    # An 8-K alone does not establish a reporting company; a 10-K/10-Q does.
    rows = universe.parse_master_idx(IDX)
    universe.upsert(con, rows)
    ciks = {r[0] for r in con.execute("SELECT cik FROM universe").fetchall()}
    assert ciks == {"320193", "1018724"}
    assert "789019" not in ciks


def test_listing_span_widens_as_more_quarters_load(con):
    universe.upsert(con, [{"cik": "1", "name": "Co", "form": "10-Q",
                           "file_date": dt.date(2024, 2, 1)}])
    universe.upsert(con, [{"cik": "1", "name": "Co", "form": "10-Q",
                           "file_date": dt.date(2022, 5, 1)}])
    listed, delisted = con.execute(
        "SELECT listed, delisted FROM universe WHERE cik='1'"
    ).fetchone()
    assert listed == dt.date(2022, 5, 1)
    assert delisted == dt.date(2024, 2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_universe.py -v`
Expected: FAIL with `ImportError: cannot import name 'universe'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/universe.py
"""Point-in-time universe from EDGAR quarterly master indexes.

A company was a listed reporting company in quarter Q if it filed a periodic
report in Q. Building membership this way means acquired and delisted names
stay in the panel for exactly the quarters they existed -- which is the whole
survivorship problem solved with a free file.

delisted is the LAST observed periodic filing. For a company still filing it
will track the end of the sample, which is the correct open-interval
behaviour for the panel builder.
"""
import datetime as dt

from . import fetch

PERIODIC_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"}
IDX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"


def parse_master_idx(raw: bytes) -> list[dict]:
    out = []
    for line in raw.decode("latin-1").splitlines():
        parts = line.split("|")
        if len(parts) != 5 or not parts[0].strip().isdigit():
            continue          # header, separator, or malformed row
        cik, name, form, date, _ = parts
        out.append({
            "cik": cik.strip().lstrip("0") or "0",
            "name": name.strip(),
            "form": form.strip(),
            "file_date": dt.date.fromisoformat(date.strip()),
        })
    return out


def quarters(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [(y, q) for y in range(start_year, end_year + 1) for q in (1, 2, 3, 4)]


def upsert(con, rows: list[dict]) -> int:
    periodic = [r for r in rows if r["form"] in PERIODIC_FORMS]
    if not periodic:
        return 0
    con.executemany(
        """
        INSERT INTO universe VALUES ($cik, $name, $file_date, $file_date)
        ON CONFLICT (cik) DO UPDATE SET
            listed   = least(universe.listed, excluded.listed),
            delisted = greatest(universe.delisted, excluded.delisted)
        """,
        [{"cik": r["cik"], "name": r["name"], "file_date": r["file_date"]}
         for r in periodic],
    )
    return len(periodic)


def build(con, start_year: int, end_year: int) -> int:
    total = 0
    for year, q in quarters(start_year, end_year):
        raw = fetch.sec_get(IDX_URL.format(year=year, q=q))
        total += upsert(con, parse_master_idx(raw))
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_universe.py -v`
Expected: 6 passed

- [ ] **Step 5: Load one real quarter and sanity-check**

Run:
```bash
python -c "
from deal import warehouse, universe
con = warehouse.connect(); warehouse.init_schema(con)
raw = universe.__dict__['fetch'].sec_get(
    universe.IDX_URL.format(year=2024, q=1))
rows = universe.parse_master_idx(raw)
print('index rows', len(rows))
print('periodic  ', universe.upsert(con, rows))
print('companies ', con.execute('SELECT count(*) FROM universe').fetchone()[0])
"
```
Expected: index rows in the hundreds of thousands, companies in the ~6,000–8,000 range for one quarter.

- [ ] **Step 6: Commit**

```bash
git add src/deal/universe.py tests/test_universe.py
git commit -m "feat: point-in-time universe from EDGAR master indexes"
```

---

### Task 3: Entity crosswalk — the critical path

USPTO keys on assignee name. Certificate Transparency keys on domain. EDGAR keys on CIK. Nothing joins without this table, and **its match rate is the error floor for every non-SEC signal.**

The failure mode to watch: name matching succeeds more often for large, well-known companies. That makes missing data non-random and biases the USPTO and CT signals toward large caps. Task 9 measures it; this task makes it measurable.

**Files:**
- Create: `src/deal/crosswalk.py`
- Test: `tests/test_crosswalk.py`

**Interfaces:**
- Consumes: `universe`, `fetch`.
- Produces: `crosswalk.normalize_name(name: str) -> str`, `crosswalk.build_names(con) -> int`, `crosswalk.load_domains(con, rows: list[dict]) -> int`, `crosswalk.match_rate(con, table: str, column: str) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crosswalk.py
import pytest

from deal import crosswalk, warehouse


def test_normalize_strips_corporate_suffixes():
    assert crosswalk.normalize_name("Apple Inc.") == "APPLE"
    assert crosswalk.normalize_name("Microsoft Corporation") == "MICROSOFT"
    assert crosswalk.normalize_name("Alphabet Inc") == "ALPHABET"


def test_normalize_strips_leading_the_and_punctuation():
    assert crosswalk.normalize_name("The Walt Disney Co.") == "WALT DISNEY"
    assert crosswalk.normalize_name("Amazon.com, Inc.") == "AMAZONCOM"


def test_normalize_collapses_whitespace_and_case():
    assert crosswalk.normalize_name("  ACME   widgets  LLC ") == "ACME WIDGETS"


def test_normalize_is_idempotent():
    once = crosswalk.normalize_name("Cisco Systems, Inc.")
    assert crosswalk.normalize_name(once) == once


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    crosswalk.init_schema(c)
    return c


def test_build_names_indexes_every_universe_company(con):
    con.execute("INSERT INTO universe VALUES ('1','Cisco Systems, Inc.','2020-01-01',NULL)")
    con.execute("INSERT INTO universe VALUES ('2','Oracle Corp','2020-01-01',NULL)")
    assert crosswalk.build_names(con) == 2
    norm = {r[0] for r in con.execute("SELECT norm_name FROM xwalk_name").fetchall()}
    assert norm == {"CISCO SYSTEMS", "ORACLE"}


def test_match_rate_reports_the_fraction_that_resolved(con):
    con.execute("INSERT INTO universe VALUES ('1','Oracle Corp','2020-01-01',NULL)")
    crosswalk.build_names(con)
    con.execute("CREATE TABLE probe (assignee VARCHAR)")
    con.executemany("INSERT INTO probe VALUES (?)",
                    [["ORACLE"], ["UNKNOWNCO"]])
    assert crosswalk.match_rate(con, "probe", "assignee") == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_crosswalk.py -v`
Expected: FAIL with `ImportError: cannot import name 'crosswalk'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/crosswalk.py
"""CIK <-> normalized name <-> domain.

Every non-SEC source joins through this table, so its match rate bounds the
coverage of every non-SEC signal. Report that rate rather than assuming it.

ponytail: exact match on a normalized string, not fuzzy matching. Fuzzy
matching on company names produces confident false positives ("DELTA AIR" vs
"DELTA APPAREL"), and a false join injects another company's signal into your
panel -- strictly worse than a missing row. Add blocked fuzzy matching only
after measuring how much the exact matcher actually misses.
"""
import re

SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LLC", "LP", "LTD", "LIMITED", "PLC", "SA", "NV", "AG", "HOLDINGS",
    "HOLDING", "GROUP", "THE",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS xwalk_name (
    norm_name VARCHAR PRIMARY KEY,
    cik       VARCHAR
);
CREATE TABLE IF NOT EXISTS xwalk_domain (
    cik    VARCHAR,
    domain VARCHAR,
    PRIMARY KEY (cik, domain)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def normalize_name(name: str) -> str:
    up = name.upper()
    up = re.sub(r"[^A-Z0-9 ]", "", up)          # drop punctuation entirely
    tokens = [t for t in up.split() if t]
    while tokens and tokens[0] in SUFFIXES:      # leading "THE"
        tokens.pop(0)
    while tokens and tokens[-1] in SUFFIXES:     # trailing INC / CORP / ...
        tokens.pop()
    return " ".join(tokens)


def build_names(con) -> int:
    rows = con.execute("SELECT cik, name FROM universe").fetchall()
    out = [{"norm_name": normalize_name(n), "cik": c} for c, n in rows]
    out = [r for r in out if r["norm_name"]]
    if not out:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO xwalk_name VALUES ($norm_name, $cik)", out
    )
    return len(out)


def load_domains(con, rows: list[dict]) -> int:
    """rows: [{'cik': ..., 'domain': ...}] from Wikidata or filing cover pages."""
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO xwalk_domain VALUES ($cik, $domain)", rows
    )
    return len(rows)


def match_rate(con, table: str, column: str) -> float:
    """Fraction of distinct values in table.column that resolve to a CIK."""
    total, hit = con.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE x.cik IS NOT NULL)
        FROM (SELECT DISTINCT {column} AS v FROM {table}) t
        LEFT JOIN xwalk_name x ON x.norm_name = t.v
        """
    ).fetchone()
    return (hit / total) if total else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_crosswalk.py -v`
Expected: 7 passed

- [ ] **Step 5: Load domains from Wikidata**

Wikidata carries both CIK (property P5531) and official website (P856) for most US-listed companies. Run this SPARQL at `https://query.wikidata.org/sparql` with `Accept: text/csv`, save to `data/raw/wikidata/cik_domain.csv`:

```sparql
SELECT ?cik ?website WHERE {
  ?company wdt:P5531 ?cik .
  ?company wdt:P856 ?website .
}
```

```bash
curl -H "Accept: text/csv" \
  --data-urlencode 'query=SELECT ?cik ?website WHERE { ?company wdt:P5531 ?cik . ?company wdt:P856 ?website . }' \
  https://query.wikidata.org/sparql -o data/raw/wikidata/cik_domain.csv
```

- [ ] **Step 6: Hand-validate 50 matches**

Sample 50 rows from `xwalk_name` at seed 20260728 and eyeball them. **Any false positive is worse than a miss** — a wrong join injects a different company's certs and trademarks into your panel. If you find more than one bad match in 50, tighten `SUFFIXES` before proceeding.

```bash
python -c "
from deal import warehouse
con = warehouse.connect()
for r in con.execute(
    'SELECT x.norm_name, u.name FROM xwalk_name x '
    'JOIN universe u USING (cik) USING SAMPLE 50 ROWS (bernoulli, 20260728)'
).fetchall():
    print(r)
"
```

- [ ] **Step 7: Commit**

```bash
git add src/deal/crosswalk.py tests/test_crosswalk.py
git commit -m "feat: CIK/name/domain crosswalk with match-rate reporting"
```

---

### Task 4: Deal labels

**Files:**
- Create: `src/deal/load_deals.py`
- Test: `tests/test_load_deals.py`

**Interfaces:**
- Consumes: `universe.parse_master_idx`, `fetch.sec_get`, `warehouse`.
- Produces: `load_deals.DEAL_FORMS: set[str]`, `load_deals.extract(rows: list[dict]) -> list[dict]`, `load_deals.load(con, start_year: int, end_year: int) -> int`.

`DEFM14A` is a merger proxy and is unambiguous. `SC TO-T` is a third-party tender offer. `SC 13E3` is a going-private transaction. All three mark a live public-target deal on their EDGAR filing date. Plain `8-K` is deliberately excluded here — it fires on everything, and Item-level detail is not in the index.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_deals.py
import datetime as dt

import pytest

from deal import load_deals, warehouse

ROWS = [
    {"cik": "1", "name": "TargetCo", "form": "DEFM14A",
     "file_date": dt.date(2024, 3, 1)},
    {"cik": "2", "name": "OtherCo", "form": "10-K",
     "file_date": dt.date(2024, 3, 1)},
    {"cik": "3", "name": "TenderCo", "form": "SC TO-T",
     "file_date": dt.date(2024, 4, 1)},
]


def test_extract_keeps_only_deal_forms():
    out = load_deals.extract(ROWS)
    assert {d["cik"] for d in out} == {"1", "3"}


def test_extract_uses_the_edgar_filing_date():
    out = load_deals.extract(ROWS)
    assert out[0]["agreement_date"] == dt.date(2024, 3, 1)


def test_plain_8k_is_not_a_deal_form():
    # 8-K fires on hundreds of unrelated events; the index has no Item detail.
    assert "8-K" not in load_deals.DEAL_FORMS


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    return c


def test_first_deal_form_wins_when_a_company_files_several(con):
    # A target files DEFM14A then amendments; the earliest is the label.
    rows = [
        {"cik": "1", "name": "T", "form": "DEFM14A",
         "file_date": dt.date(2024, 5, 1)},
        {"cik": "1", "name": "T", "form": "DEFM14A",
         "file_date": dt.date(2024, 3, 1)},
    ]
    load_deals.insert(con, load_deals.extract(rows))
    dates = [r[0] for r in con.execute(
        "SELECT agreement_date FROM deals WHERE cik='1'").fetchall()]
    assert min(dates) == dt.date(2024, 3, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_deals.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_deals'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/load_deals.py
"""Deal labels from EDGAR quarterly indexes.

Three forms mark a live public-target deal, all timestamped by EDGAR on the
day they land:
  DEFM14A  -- merger proxy
  SC TO-T  -- third-party tender offer
  SC 13E3  -- going-private transaction

Plain 8-K is excluded. It would catch Item 1.01 material agreements, but the
master index carries no Item detail, so including it would add far more noise
than signal.
"""
from . import fetch, universe

DEAL_FORMS = {"DEFM14A", "SC TO-T", "SC 13E3"}


def extract(rows: list[dict]) -> list[dict]:
    return [
        {
            "cik": r["cik"],
            "agreement_date": r["file_date"],
            "rumor_date": None,
            "acquirer": None,
        }
        for r in rows
        if r["form"] in DEAL_FORMS
    ]


def insert(con, deals: list[dict]) -> int:
    if not deals:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO deals VALUES "
        "($cik, $agreement_date, $rumor_date, $acquirer)",
        deals,
    )
    return len(deals)


def load(con, start_year: int, end_year: int) -> int:
    total = 0
    for year, q in universe.quarters(start_year, end_year):
        raw = fetch.sec_get(universe.IDX_URL.format(year=year, q=q))
        total += insert(con, extract(universe.parse_master_idx(raw)))
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load_deals.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/deal/load_deals.py tests/test_load_deals.py
git commit -m "feat: deal labels from EDGAR deal forms"
```

---

### Task 5: Fundamentals from Financial Statement Data Sets

One quarterly ZIP carries every XBRL numeric fact from every filer — replacing ~7,000 per-company API calls per quarter with a single download.

**Files:**
- Create: `src/deal/load_fund.py`
- Test: `tests/test_load_fund.py`

**Interfaces:**
- Consumes: `fetch`, `warehouse`.
- Produces: `load_fund.TAGS: dict[str, str]`, `load_fund.parse_num(raw: bytes, sub: dict[str, dict]) -> list[dict]`, `load_fund.load_quarter(con, year: int, q: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_fund.py
import datetime as dt

import pytest

from deal import load_fund, warehouse

SUB = {"0001-24-000001": {"cik": "320193", "filed": dt.date(2024, 2, 2)}}

NUM = (
    b"adsh\ttag\tversion\tddate\tqtrs\tuom\tvalue\tfootnote\n"
    b"0001-24-000001\tAssets\tus-gaap\t20231231\t0\tUSD\t352755000000\t\n"
    b"0001-24-000001\tCashAndCashEquivalentsAtCarryingValue\tus-gaap"
    b"\t20231231\t0\tUSD\t29965000000\t\n"
    b"0001-24-000001\tIrrelevantTag\tus-gaap\t20231231\t0\tUSD\t1\t\n"
)


def test_parse_keeps_only_tags_we_model():
    out = load_fund.parse_num(NUM, SUB)
    assert {r["tag"] for r in out} == {"Assets", "CashAndCashEquivalentsAtCarryingValue"}


def test_public_ts_is_the_filing_date_not_the_period_end():
    # ddate 20231231 is the period; the world saw it on the 2024-02-02 filing.
    out = load_fund.parse_num(NUM, SUB)
    assert all(r["public_ts"] == dt.date(2024, 2, 2) for r in out)


def test_rows_for_unknown_submissions_are_dropped():
    assert load_fund.parse_num(NUM, {}) == []


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    load_fund.init_schema(c)
    return c


def test_insert_is_idempotent(con):
    rows = load_fund.parse_num(NUM, SUB)
    load_fund.insert(con, rows)
    load_fund.insert(con, rows)
    assert con.execute("SELECT count(*) FROM fundamentals").fetchone()[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_fund.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_fund'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/load_fund.py
"""SEC Financial Statement Data Sets -> fundamentals.

One quarterly ZIP holds every XBRL numeric fact from every filer. sub.txt maps
accession -> CIK and filing date; num.txt holds the facts. Joining them gives
the control variables (size, cash, leverage, growth) with a single download
per quarter instead of thousands of API calls.
"""
import datetime as dt
import io
import zipfile

from . import fetch

URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{q}.zip"

# Only the tags the model's control block consumes.
TAGS = {
    "Assets": "assets",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "Liabilities": "liabilities",
    "StockholdersEquity": "equity",
    "Revenues": "revenue",
    "ResearchAndDevelopmentExpense": "rnd",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals (
    cik       VARCHAR,
    tag       VARCHAR,
    public_ts DATE,
    value     DOUBLE,
    PRIMARY KEY (cik, tag, public_ts)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def parse_sub(raw: bytes) -> dict[str, dict]:
    out = {}
    lines = raw.decode("latin-1").splitlines()
    header = lines[0].split("\t")
    i_adsh, i_cik, i_filed = (header.index(c) for c in ("adsh", "cik", "filed"))
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) <= max(i_adsh, i_cik, i_filed):
            continue
        out[f[i_adsh]] = {
            "cik": f[i_cik].lstrip("0") or "0",
            "filed": dt.datetime.strptime(f[i_filed], "%Y%m%d").date(),
        }
    return out


def parse_num(raw: bytes, sub: dict[str, dict]) -> list[dict]:
    lines = raw.decode("latin-1").splitlines()
    header = lines[0].split("\t")
    i_adsh, i_tag, i_val = (header.index(c) for c in ("adsh", "tag", "value"))

    out = []
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) <= max(i_adsh, i_tag, i_val):
            continue
        if f[i_tag] not in TAGS:
            continue
        meta = sub.get(f[i_adsh])
        if meta is None:
            continue
        try:
            value = float(f[i_val])
        except ValueError:
            continue
        out.append({
            "cik": meta["cik"],
            "tag": f[i_tag],
            # The filing date, never ddate: the period end is when the number
            # describes, the filing is when anyone could read it.
            "public_ts": meta["filed"],
            "value": value,
        })
    return out


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO fundamentals VALUES "
        "($cik, $tag, $public_ts, $value)",
        rows,
    )
    return len(rows)


def load_quarter(con, year: int, q: int) -> int:
    blob = fetch.sec_get(URL.format(year=year, q=q))
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        sub = parse_sub(z.read("sub.txt"))
        rows = parse_num(z.read("num.txt"), sub)
    return insert(con, rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load_fund.py -v`
Expected: 5 passed

- [ ] **Step 5: Load one real quarter**

Run:
```bash
python -c "
from deal import warehouse, load_fund
con = warehouse.connect(); warehouse.init_schema(con); load_fund.init_schema(con)
print('rows', load_fund.load_quarter(con, 2024, 1))
print(con.execute('SELECT tag, count(*) FROM fundamentals GROUP BY tag').fetchall())
"
```
Expected: tens of thousands of rows spread across all six tags. The ZIP is ~50MB and caches, so re-runs are instant.

- [ ] **Step 6: Commit**

```bash
git add src/deal/load_fund.py tests/test_load_fund.py
git commit -m "feat: fundamentals from SEC financial statement data sets"
```

---

### Task 6: Insider transactions from DERA bulk

**Files:**
- Create: `src/deal/load_insider.py`
- Test: `tests/test_load_insider.py`

**Interfaces:**
- Consumes: `fetch`, `warehouse`.
- Produces: `load_insider.parse_transactions(sub_raw: bytes, trans_raw: bytes) -> list[dict]`, `load_insider.load_quarter(con, year: int, q: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_insider.py
import datetime as dt

import pytest

from deal import load_insider, warehouse

SUB = (
    b"ACCESSION_NUMBER\tISSUERCIK\tFILING_DATE\n"
    b"0001-24-01\t0000320193\t02-FEB-2024\n"
    b"0001-24-02\t0000320193\t09-FEB-2024\n"
)

TRANS = (
    b"ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tEQUITY_SWAP_INVOLVED\n"
    b"0001-24-01\t01-FEB-2024\tS\t1000\t0\n"
    b"0001-24-02\t08-FEB-2024\tS\t2000\t0\n"
)


def test_parse_uses_the_filing_date_not_the_transaction_date():
    # TRANS_DATE is when the insider traded; FILING_DATE (up to 2 business
    # days later) is when the market could see it.
    out = load_insider.parse_transactions(SUB, TRANS)
    assert out[0]["public_ts"] == dt.date(2024, 2, 2)


def test_parse_handles_the_dera_date_format():
    out = load_insider.parse_transactions(SUB, TRANS)
    assert out[1]["public_ts"] == dt.date(2024, 2, 9)


def test_parse_strips_leading_zeros_from_cik():
    out = load_insider.parse_transactions(SUB, TRANS)
    assert all(r["cik"] == "320193" for r in out)


def test_transactions_without_a_submission_are_dropped():
    orphan = (
        b"ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tEQUITY_SWAP_INVOLVED\n"
        b"0001-24-99\t01-FEB-2024\tS\t1000\t0\n"
    )
    assert load_insider.parse_transactions(SUB, orphan) == []


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    load_insider.init_schema(c)
    return c


def test_insert_is_idempotent(con):
    rows = load_insider.parse_transactions(SUB, TRANS)
    load_insider.insert(con, rows)
    load_insider.insert(con, rows)
    assert con.execute("SELECT count(*) FROM insider_trans").fetchone()[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_insider.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_insider'`

- [ ] **Step 3: Confirm the archive layout before writing the loader**

The DERA insider archive's exact filenames and the 10b5-1 column name must be read from the archive itself, not assumed:

```bash
curl -s -A "RandomBSQuant research your.email@example.com" \
  -o /tmp/ins.zip \
  "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2024q1_form345.zip"
unzip -l /tmp/ins.zip
head -1 <(unzip -p /tmp/ins.zip NONDERIV_TRANS.tsv)
```

Expected: members including `SUBMISSION.tsv`, `REPORTINGOWNER.tsv`, `NONDERIV_TRANS.tsv`. Note the exact header names — the code below uses `ACCESSION_NUMBER`, `ISSUERCIK`, `FILING_DATE`, `TRANS_DATE`, `TRANS_CODE`, `TRANS_SHARES`. If the header differs, adjust the `COLS` mapping in Step 4 and nothing else.

- [ ] **Step 4: Write minimal implementation**

```python
# src/deal/load_insider.py
"""SEC Insider Transactions Data Sets -> insider_trans.

DERA publishes Forms 3/4/5 flattened into TSVs, one ZIP per quarter. That
replaces parsing roughly half a million ownership XML documents a year.

Transaction code 'S' is an open-market sale and 'P' an open-market purchase;
those are the discretionary ones. Codes like 'A' (grant) and 'F' (tax
withholding) are automatic and carry no information about what management
knows.
"""
import datetime as dt
import io
import zipfile

from . import fetch

URL = ("https://www.sec.gov/files/structureddata/data/"
       "insider-transactions-data-sets/{year}q{q}_form345.zip")

DISCRETIONARY_CODES = {"S", "P"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS insider_trans (
    accession     VARCHAR,
    cik           VARCHAR,
    public_ts     DATE,
    trans_code    VARCHAR,
    shares        DOUBLE,
    discretionary BOOLEAN,
    PRIMARY KEY (accession, public_ts, trans_code, shares)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def _dera_date(raw: str) -> dt.date:
    # DERA writes dates as 02-FEB-2024.
    return dt.datetime.strptime(raw.strip(), "%d-%b-%Y").date()


def _rows(raw: bytes) -> tuple[list[str], list[list[str]]]:
    lines = raw.decode("latin-1").splitlines()
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def parse_transactions(sub_raw: bytes, trans_raw: bytes) -> list[dict]:
    head, body = _rows(sub_raw)
    i_acc, i_cik, i_filed = (head.index(c) for c in
                             ("ACCESSION_NUMBER", "ISSUERCIK", "FILING_DATE"))
    subs = {
        f[i_acc]: {
            "cik": f[i_cik].lstrip("0") or "0",
            "filed": _dera_date(f[i_filed]),
        }
        for f in body
        if len(f) > max(i_acc, i_cik, i_filed)
    }

    head, body = _rows(trans_raw)
    i_acc, i_code, i_sh = (head.index(c) for c in
                           ("ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES"))

    out = []
    for f in body:
        if len(f) <= max(i_acc, i_code, i_sh):
            continue
        meta = subs.get(f[i_acc])
        if meta is None:
            continue
        try:
            shares = float(f[i_sh])
        except ValueError:
            continue
        code = f[i_code].strip()
        out.append({
            "accession": f[i_acc],
            "cik": meta["cik"],
            # FILING_DATE, not TRANS_DATE: the trade is private until filed.
            "public_ts": meta["filed"],
            "trans_code": code,
            "shares": shares,
            "discretionary": code in DISCRETIONARY_CODES,
        })
    return out


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO insider_trans VALUES "
        "($accession, $cik, $public_ts, $trans_code, $shares, $discretionary)",
        rows,
    )
    return len(rows)


def load_quarter(con, year: int, q: int) -> int:
    blob = fetch.sec_get(URL.format(year=year, q=q))
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        rows = parse_transactions(
            z.read("SUBMISSION.tsv"), z.read("NONDERIV_TRANS.tsv")
        )
    return insert(con, rows)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_load_insider.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/deal/load_insider.py tests/test_load_insider.py
git commit -m "feat: insider transactions from DERA bulk data sets"
```

---

### Task 7: USPTO assignments

Joins on assignee name, so its coverage is capped by Task 3's match rate.

**Files:**
- Create: `src/deal/load_uspto.py`
- Test: `tests/test_load_uspto.py`

**Interfaces:**
- Consumes: `crosswalk.normalize_name`, `warehouse`.
- Produces: `load_uspto.classify(conveyance: str) -> str`, `load_uspto.parse_rows(records: list[dict]) -> list[dict]`, `load_uspto.load(con, records: list[dict]) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_uspto.py
import datetime as dt

import pytest

from deal import crosswalk, load_uspto, warehouse

REC = {
    "assignee": "Oracle Corporation",
    "recorded_date": "2024-06-15",
    "execution_date": "2024-03-01",
    "conveyance_text": "RELEASE BY SECURED PARTY",
}


def test_classify_distinguishes_release_from_grant():
    assert load_uspto.classify("RELEASE BY SECURED PARTY") == "tm_release"
    assert load_uspto.classify("SECURITY INTEREST") == "tm_lien"
    assert load_uspto.classify("ASSIGNS THE ENTIRE INTEREST") == "tm_assign"


def test_parse_uses_recorded_date_as_public_ts():
    # Execution is private between the parties until the USPTO records it.
    assert load_uspto.parse_rows([REC])[0]["public_ts"] == dt.date(2024, 6, 15)


def test_parse_normalizes_the_assignee_for_joining():
    assert load_uspto.parse_rows([REC])[0]["norm_name"] == "ORACLE"


def test_parse_skips_records_with_no_recorded_date():
    assert load_uspto.parse_rows([{**REC, "recorded_date": ""}]) == []


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    crosswalk.init_schema(c)
    c.execute("INSERT INTO universe VALUES ('1','Oracle Corp','2020-01-01',NULL)")
    crosswalk.build_names(c)
    return c


def test_load_joins_through_the_crosswalk_to_a_cik(con):
    assert load_uspto.load(con, [REC]) == 1
    cik, kind = con.execute(
        "SELECT cik, kind FROM signals WHERE kind LIKE 'tm_%'"
    ).fetchone()
    assert cik == "1"
    assert kind == "tm_release"


def test_unmatched_assignees_are_dropped_not_guessed(con):
    # A wrong join injects another company's signal -- worse than a miss.
    assert load_uspto.load(con, [{**REC, "assignee": "Nonexistent Widgets"}]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_uspto.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_uspto'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/load_uspto.py
"""USPTO assignment records -> signals.

The assignment dataset records mergers, name changes, security interests and
their releases. A company clearing liens off its IP is doing standard pre-sale
housekeeping.

Records key on assignee NAME, so every row joins through xwalk_name. Rows that
do not match are dropped rather than guessed: a false join puts another
company's filings into your panel, which is strictly worse than a gap.
"""
import datetime as dt

from .crosswalk import normalize_name


def classify(conveyance: str) -> str:
    text = conveyance.upper()
    if "RELEASE" in text:
        return "tm_release"
    if "SECURITY INTEREST" in text:
        return "tm_lien"
    return "tm_assign"


def parse_rows(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        recorded = (r.get("recorded_date") or "").strip()
        if not recorded:
            continue
        out.append({
            "norm_name": normalize_name(r["assignee"]),
            # recorded_date is publication; execution_date is not public.
            "public_ts": dt.date.fromisoformat(recorded),
            "kind": classify(r.get("conveyance_text", "")),
        })
    return out


def load(con, records: list[dict]) -> int:
    rows = parse_rows(records)
    if not rows:
        return 0
    con.execute("CREATE OR REPLACE TEMP TABLE _uspto "
                "(norm_name VARCHAR, public_ts DATE, kind VARCHAR)")
    con.executemany(
        "INSERT INTO _uspto VALUES ($norm_name, $public_ts, $kind)", rows
    )
    con.execute(
        """
        INSERT OR IGNORE INTO signals
        SELECT x.cik, u.kind, u.public_ts, 1.0
        FROM _uspto u JOIN xwalk_name x ON x.norm_name = u.norm_name
        """
    )
    return con.execute(
        "SELECT count(*) FROM _uspto u "
        "JOIN xwalk_name x ON x.norm_name = u.norm_name"
    ).fetchone()[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_load_uspto.py -v`
Expected: 7 passed

- [ ] **Step 5: Download the bulk assignment datasets**

```bash
mkdir -p data/raw/uspto
curl -o data/raw/uspto/tm_assignment.zip \
  "https://bulkdata.uspto.gov/data/trademark/assignment/economic-research/tm_assignment_economics_2024.zip"
curl -o data/raw/uspto/pat_assignment.zip \
  "https://bulkdata.uspto.gov/data/patent/assignment/economic-research/pat_assignment_economics_2024.zip"
unzip -l data/raw/uspto/tm_assignment.zip
```

Expected: CSVs including an assignment table and an assignee table. Note the exact column names for assignee, recorded date, and conveyance text — they feed the `records` dicts passed to `load()`.

- [ ] **Step 6: Report the match rate before trusting the signal**

```bash
python -c "
from deal import warehouse, crosswalk
con = warehouse.connect()
con.execute('CREATE OR REPLACE TABLE probe AS SELECT DISTINCT norm_name AS a FROM _uspto')
print('assignee match rate:', crosswalk.match_rate(con, 'probe', 'a'))
"
```

**Below ~0.5 means the normaliser is the bottleneck, not the data.** Match rate is also almost certainly higher for large caps, so record it — Task 9 turns that into a reported bias, not a silent one.

- [ ] **Step 7: Commit**

```bash
git add src/deal/load_uspto.py tests/test_load_uspto.py
git commit -m "feat: USPTO assignment loader joining through the crosswalk"
```

---

### Task 8: Certificate Transparency via Postgres

**Files:**
- Create: `src/deal/load_ct.py`
- Test: `tests/test_load_ct.py`

**Interfaces:**
- Consumes: `crosswalk` (xwalk_domain), `warehouse`.
- Produces: `load_ct.QUERY: str`, `load_ct.novel_by_week(rows: list[tuple]) -> list[dict]`, `load_ct.query_domain(domain: str, retries: int = 3) -> list[tuple]`, `load_ct.load(con, cik: str, domain: str) -> int`.

crt.sh terminates long-running queries via its replication system. The documented workaround is to retry — the first attempt primes the cache and the second completes. That is a normal code path here, not an error path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_ct.py
import datetime as dt

import pytest

from deal import load_ct, warehouse


ROWS = [
    ("www.example.com", dt.datetime(2024, 3, 1, 10, 0)),
    ("www.example.com", dt.datetime(2024, 3, 8, 10, 0)),   # renewal
    ("checkout.example.com", dt.datetime(2024, 3, 8, 11, 0)),
    ("api.example.com", dt.datetime(2024, 3, 9, 11, 0)),
]


def test_only_first_appearances_count_as_novel():
    out = load_ct.novel_by_week(ROWS)
    total = sum(r["value"] for r in out)
    assert total == 3, "the www renewal must not count twice"


def test_novel_names_bucket_into_iso_weeks():
    out = {r["public_ts"]: r["value"] for r in load_ct.novel_by_week(ROWS)}
    # 2024-03-08 and 2024-03-09 are both in the week starting Monday 03-04.
    assert out[dt.date(2024, 3, 4)] == 2
    assert out[dt.date(2024, 2, 26)] == 1


def test_rows_are_processed_in_log_order_regardless_of_input_order():
    shuffled = [ROWS[2], ROWS[0], ROWS[3], ROWS[1]]
    assert load_ct.novel_by_week(shuffled) == load_ct.novel_by_week(ROWS)


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    return c


def test_insert_writes_ct_signals(con):
    load_ct.insert(con, "C", load_ct.novel_by_week(ROWS))
    kinds = {r[0] for r in con.execute("SELECT kind FROM signals").fetchall()}
    assert kinds == {"ct"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load_ct.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_ct'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/load_ct.py
"""Certificate Transparency -> signals(kind='ct').

crt.sh exposes its Postgres read replica publicly, which is far better than
the JSON endpoint for bulk work. Long queries are killed by the replication
system; the documented fix is to retry, because the first attempt primes the
cache and the second completes. Retrying is the normal path here.

The signal is NOVEL hostnames per ISO week. Renewals of known hosts are
noise -- first appearances are the event.
"""
import datetime as dt

import psycopg

DSN = "host=crt.sh port=5432 dbname=certwatch user=guest connect_timeout=30"

QUERY = """
SELECT ci.NAME_VALUE, le.ENTRY_TIMESTAMP
FROM certificate_identity ci
JOIN ct_log_entry le ON le.CERTIFICATE_ID = ci.CERTIFICATE_ID
WHERE ci.NAME_VALUE ILIKE %s
ORDER BY le.ENTRY_TIMESTAMP
"""


def iso_monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def novel_by_week(rows: list[tuple]) -> list[dict]:
    seen: set[str] = set()
    weekly: dict[dt.date, int] = {}
    for name, ts in sorted(rows, key=lambda r: r[1]):
        host = name.strip().lower()
        if host in seen:
            continue
        seen.add(host)
        week = iso_monday(ts.date())
        weekly[week] = weekly.get(week, 0) + 1
    return [{"public_ts": w, "value": float(n)} for w, n in sorted(weekly.items())]


def query_domain(domain: str, retries: int = 3) -> list[tuple]:
    last: Exception | None = None
    for _ in range(retries):
        try:
            with psycopg.connect(DSN) as conn, conn.cursor() as cur:
                cur.execute(QUERY, (f"%.{domain}",))
                return cur.fetchall()
        except psycopg.Error as exc:
            last = exc      # replication killed it; the retry hits a warm cache
    raise RuntimeError(f"crt.sh query failed after {retries} attempts") from last


def insert(con, cik: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO signals VALUES ($cik, 'ct', $public_ts, $value)",
        [{"cik": cik, **r} for r in rows],
    )
    return len(rows)


def load(con, cik: str, domain: str) -> int:
    return insert(con, cik, novel_by_week(query_domain(domain)))
```

Add `psycopg[binary]` to `pyproject.toml` dependencies.

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e . && pytest tests/test_load_ct.py -v`
Expected: 5 passed

- [ ] **Step 5: Smoke-test the live connection**

Run:
```bash
python -c "
from deal import load_ct
rows = load_ct.query_domain('github.com')
print('cert identities', len(rows))
print('novel weeks    ', len(load_ct.novel_by_week(rows)))
"
```
Expected: tens of thousands of rows. If it hangs past ~60s the replica killed the query — the retry loop handles it, so let it run.

- [ ] **Step 6: Commit**

```bash
git add src/deal/load_ct.py tests/test_load_ct.py pyproject.toml
git commit -m "feat: certificate transparency loader via crt.sh postgres"
```

---

### Task 9: Coverage report

Missing data here is not random. USPTO and CT coverage depend on name and domain matching, which works better for large well-known companies — so a naive read of the results would attribute to the *signal* what is really an artifact of *coverage*. This task makes that visible, and it runs after every load.

**Files:**
- Create: `src/deal/coverage.py`
- Test: `tests/test_coverage.py`

**Interfaces:**
- Consumes: `warehouse`, all loaded tables.
- Produces: `coverage.by_source(con) -> polars.DataFrame`, `coverage.by_size_decile(con, source: str) -> polars.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage.py
import pytest

from deal import coverage, warehouse


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    for i in range(10):
        c.execute(f"INSERT INTO universe VALUES ('{i}','Co{i}','2020-01-01',NULL)")
    # Only 3 of 10 companies have any ct signal.
    for i in range(3):
        c.execute(f"INSERT INTO signals VALUES ('{i}','ct','2024-01-01',1.0)")
    return c


def test_by_source_reports_the_fraction_of_companies_covered(con):
    df = coverage.by_source(con)
    row = df.filter(df["kind"] == "ct")
    assert row["companies"][0] == 3
    assert abs(row["pct_universe"][0] - 0.3) < 1e-9


def test_by_source_lists_a_source_with_no_rows_as_zero(con):
    df = coverage.by_source(con)
    row = df.filter(df["kind"] == "tm_release")
    assert row["companies"][0] == 0


def test_by_size_decile_returns_one_row_per_populated_decile(con):
    df = coverage.by_size_decile(con, "ct")
    assert df.height > 0
    assert set(df.columns) >= {"decile", "covered", "total"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coverage.py -v`
Expected: FAIL with `ImportError: cannot import name 'coverage'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/deal/coverage.py
"""Per-source coverage. Run after every load.

Coverage gaps here are not random: USPTO and CT both join on name or domain,
and those match better for large, well-known companies. Unreported, that
turns a coverage artifact into a fake size effect in the results. Reported,
it is a caveat.
"""
import polars as pl

KINDS = ["ct", "tm_release", "tm_lien", "tm_assign", "form4_gap"]


def by_source(con) -> pl.DataFrame:
    total = con.execute("SELECT count(*) FROM universe").fetchone()[0] or 1
    rows = []
    for kind in KINDS:
        n = con.execute(
            "SELECT count(DISTINCT cik) FROM signals WHERE kind = ?", [kind]
        ).fetchone()[0]
        rows.append({"kind": kind, "companies": n, "pct_universe": n / total})
    return pl.DataFrame(rows)


def by_size_decile(con, source: str) -> pl.DataFrame:
    """Coverage split by asset decile. A steep gradient here means the signal
    is partly a size proxy and must be reported as such."""
    rows = con.execute(
        """
        WITH size AS (
            SELECT u.cik,
                   coalesce(max(f.value), 0) AS assets
            FROM universe u
            LEFT JOIN fundamentals f
              ON f.cik = u.cik AND f.tag = 'Assets'
            GROUP BY u.cik
        ),
        ranked AS (
            SELECT cik, ntile(10) OVER (ORDER BY assets) AS decile FROM size
        )
        SELECT r.decile,
               count(*) FILTER (WHERE s.cik IS NOT NULL) AS covered,
               count(*)                                  AS total
        FROM ranked r
        LEFT JOIN (SELECT DISTINCT cik FROM signals WHERE kind = ?) s
          ON s.cik = r.cik
        GROUP BY r.decile ORDER BY r.decile
        """,
        [source],
    ).fetchall()
    return pl.DataFrame(
        [{"decile": d, "covered": c, "total": t} for d, c, t in rows],
        schema={"decile": pl.Int64, "covered": pl.Int64, "total": pl.Int64},
    )
```

`by_size_decile` reads the `fundamentals` table from Task 5. If Task 5 has not run, every company lands in one asset bucket and the deciles are uninformative — run Task 5 first.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coverage.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/deal/coverage.py tests/test_coverage.py
git commit -m "feat: per-source and by-size coverage reporting"
```

---

## Load order

Dependencies are real here — Task 3 must run before 7 and 8, or every join drops.

```bash
python -c "
from deal import warehouse, universe, crosswalk, load_deals, load_fund, load_insider
con = warehouse.connect(); warehouse.init_schema(con)
crosswalk.init_schema(con); load_fund.init_schema(con); load_insider.init_schema(con)
universe.build(con, 2015, 2026)      # ~44 index files
crosswalk.build_names(con)           # must precede USPTO and CT
load_deals.load(con, 2015, 2026)
for y in range(2015, 2027):
    for q in (1, 2, 3, 4):
        load_fund.load_quarter(con, y, q)
        load_insider.load_quarter(con, y, q)
"
```

Rough first-run cost: EDGAR indexes ~1GB, financial statement sets ~50MB × 44, insider sets ~30MB × 44. Call it 5GB on disk and a few hours, almost all of it SEC rate limiting. Everything caches, so the second run is minutes.

crt.sh is the slow one and is not in the batch above: one query per domain, retries expected. Run it overnight against `xwalk_domain` and expect a meaningful fraction of domains to return nothing.

## Deliberately not in this plan

- **Price and volume data.** One source decision (free Stooq vs. CRSP through WRDS) that depends on university access, and it is the only input with a licensed option. Add once that is settled — it feeds `ret_12m`, `idio_vol`, `amihud_illiq`, and the abnormal-volume clock.
- **Hiring signal.** Same open question as in the model plan: Wayback scraper or Revelio. The `signals` table takes it with no schema change.
- **Fuzzy name matching.** Exact-match coverage has to be measured before anything more elaborate is justified.
- **Incremental refresh.** Full reload is a few minutes warm. Build incremental loading when that stops being true.

## Sources

- [SEC EDGAR full-index](https://www.sec.gov/Archives/edgar/full-index/)
- [SEC Insider Transactions Data Sets](https://www.sec.gov/newsroom/whats-new/osd-announcement-081222-form-345-data-sets)
- [USPTO Trademark Assignment Dataset](https://www.uspto.gov/ip-policy/economic-research/research-datasets/trademark-assignment-dataset)
- [USPTO bulk data](https://bulkdata.uspto.gov/)
- [crt.sh direct database access](https://groups.google.com/g/crtsh/c/sUmV0mBz8bQ)
- [Wikidata SPARQL endpoint](https://query.wikidata.org/)
