# Prediction-Market Informed Flow & Liquidity–Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Polymarket/Kalshi dataset that answers two linked questions — do identifiable wallet clusters trade ahead of price, and does more liquidity make a contract's forecast better or noisier — and ship it as a single paper.

**Architecture:** One local DuckDB warehouse fed by three ingesters (Polymarket Data API for trades, Polygon logs via HyperSync for USDC funding edges, Kalshi public API for the cross-listed venue). Four analysis modules read that warehouse and emit result tables. No services, no scheduler, no cloud — a laptop and a Parquet directory.

**Tech Stack:** Python 3.11+, DuckDB, Polars, `hypersync` (Envio), `httpx`, `scipy`/`statsmodels`, `pytest`.

---

## Part A — Idea Build-Out and Ranking

This section is the assessment you asked for. Part B is the executable plan for the winner.

### The ranking

| # | Idea | Feasibility | Novelty | Paper strength | Rank |
|---|------|-------------|---------|----------------|------|
| 1+4 | **Polymarket informed flow *merged with* liquidity-vs-accuracy** | High | Moderate–high | Strong | **1** |
| 2 | Wayback ToS/privacy-policy diffing as M&A precursor | Moderate | Very high | Weak (likely null) | **3** |
| 3 | Bollen 2011 redo, LLM sentiment on prediction markets | High | Low | Moderate | **2** (as a chapter, not a paper) |

**The single most useful finding: ideas 1 and 4 are the same paper.** They run off one dataset, one ingestion pipeline, one cleaning pass. Idea 1 supplies the mechanism (concentrated informed flow), idea 4 supplies the market-level consequence (does that flow help or hurt aggregate accuracy). Separately they're two thin papers; together they're one with an actual causal story. Build them as one project. That is Part B.

---

### Idea 1 — Polymarket on-chain clustering → information diffusion

**What survives contact with the data**

Your three proposed clustering signals do not have equal value:

- **Gas-payer relationships — dead.** Polymarket users trade through proxy wallets (Gnosis Safe), and Polymarket's relayer pays gas for essentially all of them. The gas-payer graph is one giant hub. Drop this signal; don't spend a day rediscovering it.
- **Funding-source overlap — works, with a caveat.** Proxy wallets are funded by USDC/USDC.e transfers on Polygon. The naive version also produces hubs, because half of retail funds from the same Coinbase and Binance hot wallets. It becomes informative only after you strip high-degree nodes. The rule that makes this work: an address funding more than ~20 distinct proxies is infrastructure (exchange, bridge, relayer), not a person. Everything below that threshold is a genuine link.
- **Co-trading patterns — works, and is the strongest signal.** Same contract, same side, within a short window, repeated across unrelated markets. Repetition across *unrelated* markets is what separates a real link from two people reacting to the same news.

**The killer constraint on the venue comparison.** You proposed comparing propagation speed against "Kalshi's more centralized, KYC'd order flow." Kalshi publishes trades but not trader identity, and as a centralized exchange it never will. So there is no Kalshi-side cluster to compare against. Trying to force the symmetry is how this project dies at month two.

Reframe: the venue comparison is about **price discovery, not traders**. Both venues list overlapping contracts (Fed decisions, CPI prints, elections, sports). Standard microstructure gives you the tool — Hasbrouck information share and the Gonzalo–Granger common-factor decomposition tell you which venue's price moves first and how much of permanent price innovation originates on each. That is a well-posed, defensible question with an established method, and it uses only public data from both sides. It is strictly better than the version you proposed because it's answerable.

**Where the real risk is.** Not data access — that's fine. It's that "informed" is unobservable and clusters have no ground truth. You can never prove two wallets are one person. Do not try. Instead, make the claim falsifiable through outcomes: does cluster *X*'s net flow at time *t* predict the resolution outcome, out of sample? Resolution is a hard label. That converts an unverifiable identity claim into a testable forecasting claim, and it's the difference between a paper and a blog post.

**Verdict:** high feasibility, free data, modest compute. Novelty is moderate — there is journalistic work on Polymarket whales (the 2024 French whale) and academic work on Polymarket efficiency, but systematic clustering plus cross-venue lead-lag is genuinely underexplored.

---

### Idea 4 — IEM at scale (folded into the above)

**Your framing has an identification problem that is fatal as stated.** Comparing IEM's poll-beating edge (measured 1988–2004) against Polymarket's (2024–2026) confounds "liquidity degraded the signal" with "polling got much worse." Response rates collapsed from roughly a third in the late 1990s to low single digits today. Any difference you measure across those eras is mostly the polls changing, not the markets. That's why the question is still open — not because nobody tried.

Also note the data has holes: IEM publishes historical data for 1998–present, but much of 1988–1996 is not accessible. And 538 was shut down by ABC in March 2025, so its archive is static (still on GitHub, still usable for the historical window); Silver Bulletin is the live successor.

**The fix — go cross-sectional, not cross-era.** Compare accuracy across *contracts within the same cycle* that differ in liquidity. Same election, same news environment, same polling regime, varying depth. That removes the era confound entirely and needs only the Polymarket data you're already pulling for idea 1.

Two econometric traps to design around from day one, because retrofitting them is painful:

1. **Endogeneity.** Volume is highest on contracts that are close to 50/50 and near resolution — exactly where Brier scores are naturally worst. Regress accuracy on liquidity without controlling for horizon and for distance from 0.5 and you will "discover" that liquidity destroys accuracy. It doesn't; you measured the control.
2. **Selection.** Contracts that attract volume are not random. Prefer within-contract variation over time to pure cross-section where you can.

**Verdict:** as originally framed, low chance of a clean result. Reframed cross-sectionally, it's a strong second half to the idea-1 paper and costs almost no extra engineering.

---

### Idea 3 — Bollen 2011 redo with LLM sentiment

**Feasibility is better than expected, novelty is worse.**

Data: X's academic tier died in 2023 and full-archive search is Enterprise-only at $42k+/month, so the literal replication on 2008 tweets is impossible. Third-party archive resellers index the full history at roughly $150 per million tweets, which is affordable — but for a paper you cannot verify their sampling, which is a real methodological weakness a referee will find. Free and verifiable alternatives: Reddit dumps, the Bluesky firehose (fully open, though a skewed population), and GDELT for news text.

**The trap that sinks most papers in this genre: LLM lookahead.** A model whose training cutoff postdates the event already knows the outcome. Asking it to score 2024 election sentiment is not forecasting, it's recall. This is the single most important design constraint, and it must be handled up front:
- restrict to events *after* the scoring model's cutoff, and
- run a leakage probe — ask the model to predict the outcome given no text at all. If it beats chance, your signal is contaminated and every downstream result is void.

**Novelty is the real problem.** "LLM sentiment beats lexicon sentiment" is a crowded literature — Lopez-Lira & Tang (2023) is the canonical version and is heavily cited, and the leakage critique is already well known. Swapping the target from equities to prediction markets is a modest twist, not a new question. Your instinct that prediction markets are a cleaner sentiment-to-price pipeline is correct and it is the best thing about this idea — the event is unambiguous, so you're spared the "which factor model" argument entirely.

**Verdict:** very buildable, cheap, low risk — and low ceiling. High chance of being duplicative. Best use: a signal layer *inside* the idea-1 paper ("does public sentiment lead or lag concentrated informed flow?"), where it's a genuinely novel comparison, rather than a standalone paper competing with a crowded field.

---

### Idea 2 — Wayback ToS/privacy-policy diffing as M&A precursor

This is the most fun idea and the one most likely to produce nothing. Both are worth saying plainly.

**The mechanism claim is backwards.** Legal teams overwhelmingly rewrite ToS and privacy policies *after* close, not before. Pre-close, the parties are separate legal entities and HSR gun-jumping rules specifically prohibit the kind of integration and data-sharing that "compliance harmonization" would imply. A target's counsel doing what you describe would be creating antitrust exposure.

**There is a narrower version that is real.** Companies preparing for a sale sometimes add a clause permitting transfer of personal data "in the event of a merger, acquisition, or sale of assets." A company that lacks that clause and adds it may be preparing the ground. That's a specific, testable, single-clause hypothesis — far better than your proposed aggregate text-length and complexity metrics, which will be swamped by noise.

**Three problems, in order of severity:**
1. **Statistical power.** Public tech targets number in the low hundreds per year globally, and the interesting cases are often private companies with no tradeable security. Rare events plus a weak signal means you need an effect size you almost certainly don't have.
2. **Common shocks.** GDPR, CCPA, and the DSA each caused near-universal simultaneous policy rewrites. These will dominate any complexity time series and mimic the signal across your entire panel at once.
3. **Sampling.** Wayback crawl cadence is irregular and sparse for mid-caps — gaps of months are routine, which destroys the event-time precision the whole design depends on.

**The competitor you must cite.** *Lazy Prices* (Cohen, Malloy & Nguyen, 2020) showed that year-over-year language changes in 10-K and 10-Q filings predict returns. It's the direct precedent, it's well cited, and it explains why nobody has done your version: EDGAR filings are mandatory, timestamped, structured, and complete, while ToS pages are none of those. Your "zero competition" reading is right about the fact and wrong about the reason — the space is empty because the data is worse, not because nobody thought of it.

**Verdict:** cheap to try, genuinely original, low probability of a positive result. Good weekend project or a null-result note. Bad flagship, and a bad interview story if it produces nothing — "I built a scraper and found no effect" is a much weaker close than a working signal.

---

### On the interview-story question

Your instinct is right that original data engineering beats applying known edge logic. But the thing that makes idea 1+4 the strongest story isn't the on-chain scraping — it's that you found a question that is *answerable with public data and has a falsifiable outcome label*. Resolution outcomes are the hard ground truth almost no sentiment or M&A project can offer. Lead with that.

---

## Part B — Implementation Plan (Idea 1 + 4 merged)

Ideas 2 and 3 are deliberately **not** planned to task level here. Each is a separate project with its own plan; folding them in now would triple the scope before the core question has a single validated data point. If you want either after Task 1 clears, ask for its own plan.

## Global Constraints

- Python 3.11+ — required for `tomllib` and typing syntax used below.
- All data lands in `data/warehouse.duckdb`; all raw pulls cached as Parquet under `data/raw/`. Never re-hit an API for data already on disk.
- No API keys required for any source in this plan. Kalshi market data and Polymarket Data API are both unauthenticated for reads.
- Polymarket migrated CTF Exchange contracts on **2026-04-28**; the legacy subgraph is unsupported. Read on-chain data via HyperSync, never the old subgraph.
- Every analysis module writes a result table to DuckDB and returns a DataFrame. No module prints, plots, or writes to stdout — plotting is a separate concern.
- Timestamps are UTC epoch seconds (int64) everywhere. No naive datetimes crossing a module boundary.
- Money is USDC with 6 decimals. Store integer micro-USDC (`int64`), never float dollars.

---

## File Structure

```
src/pmflow/
  config.py        # constants: contract addresses, API bases, thresholds
  warehouse.py     # DuckDB connection + schema DDL
  ingest_poly.py   # Polymarket Data API -> trades table
  ingest_funding.py# Polygon USDC logs via HyperSync -> transfers table
  ingest_kalshi.py # Kalshi public API -> kalshi_trades, kalshi_candles
  cluster.py       # funding + co-trading graph -> wallet clusters
  skill.py         # per-cluster forecast skill vs resolution
  leadlag.py       # Hasbrouck information share, Polymarket vs Kalshi
  liquidity.py     # accuracy ~ liquidity cross-sectional regression
tests/
  test_config.py
  test_warehouse.py
  test_ingest_poly.py
  test_ingest_funding.py
  test_cluster.py
  test_skill.py
  test_leadlag.py
  test_liquidity.py
data/
  raw/             # cached Parquet, gitignored
  warehouse.duckdb # gitignored
```

Each ingester owns exactly one source. Each analysis module owns exactly one research claim. `cluster.py` is the only module that knows about graph structure; `skill.py` consumes its output and knows nothing about how clusters were formed.

---

### Task 1: Kill-gate spike — prove the data exists

**Why this is first:** every downstream task assumes wallet-attributed trades and non-degenerate funding edges. If either fails, the project is dead and you want to know in a day, not a month. This task is deliberately throwaway except for `config.py`.

**Files:**
- Create: `src/pmflow/config.py`
- Create: `scripts/spike_datacheck.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.POLY_DATA_API: str`, `config.USDC_POLYGON: str`, `config.USDC_E_POLYGON: str`, `config.HUB_DEGREE_THRESHOLD: int`, `config.KALSHI_API: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pmflow import config


def test_addresses_are_checksummed_hex():
    for addr in (config.USDC_POLYGON, config.USDC_E_POLYGON):
        assert addr.startswith("0x")
        assert len(addr) == 42


def test_hub_threshold_is_sane():
    # A funder touching more than this many proxies is infrastructure, not a person.
    assert 5 <= config.HUB_DEGREE_THRESHOLD <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pmflow'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pmflow/config.py
"""Constants for the prediction-market flow project."""

POLY_DATA_API = "https://data-api.polymarket.com"
POLY_CLOB_API = "https://clob.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# Polygon token contracts. USDC.e is the bridged legacy token; native USDC
# is the newer one. Polymarket collateral has used both, so index both.
USDC_E_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_POLYGON = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

# A funding address that pays more than this many distinct proxy wallets is
# an exchange hot wallet, bridge, or relayer -- not a person. Edges through
# these nodes are noise and get stripped before clustering.
HUB_DEGREE_THRESHOLD = 20

# Polymarket migrated CTF Exchange contracts on this block; the legacy
# subgraph stopped being maintained afterwards.
CTF_MIGRATION_DATE = "2026-04-28"
```

Add `src/pmflow/__init__.py` (empty) and make the package importable:

```toml
# pyproject.toml
[project]
name = "pmflow"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["duckdb", "polars", "httpx", "hypersync", "statsmodels", "scipy", "networkx"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pmflow"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e . && pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Write the spike script**

```python
# scripts/spike_datacheck.py
"""Throwaway. Answers three yes/no questions before we build anything.

1. Does the Polymarket Data API return per-wallet trades?
2. After stripping hubs, do funding edges connect distinct proxies?
3. Does Kalshi return public trades with no auth?

Run: python scripts/spike_datacheck.py
"""
import collections
import httpx
from pmflow import config


def check_poly_trades() -> int:
    r = httpx.get(f"{config.POLY_DATA_API}/trades", params={"limit": 500}, timeout=30)
    r.raise_for_status()
    trades = r.json()
    wallets = {t.get("proxyWallet") or t.get("maker") for t in trades}
    wallets.discard(None)
    print(f"poly trades={len(trades)} distinct wallets={len(wallets)}")
    print(f"  sample keys: {sorted(trades[0].keys())}")
    return len(wallets)


def check_kalshi_trades() -> int:
    r = httpx.get(f"{config.KALSHI_API}/markets/trades", params={"limit": 100}, timeout=30)
    r.raise_for_status()
    trades = r.json().get("trades", [])
    print(f"kalshi trades={len(trades)}")
    if trades:
        print(f"  sample keys: {sorted(trades[0].keys())}")
        has_identity = any(k for k in trades[0] if "user" in k.lower() or "addr" in k.lower())
        print(f"  exposes trader identity: {has_identity}  (expected: False)")
    return len(trades)


if __name__ == "__main__":
    n_wallets = check_poly_trades()
    n_kalshi = check_kalshi_trades()
    print()
    print("GATE 1 (poly wallet attribution):", "PASS" if n_wallets > 50 else "FAIL")
    print("GATE 2 (kalshi public trades):   ", "PASS" if n_kalshi > 0 else "FAIL")
    print()
    print("If GATE 1 fails, this project does not work. Stop here.")
```

- [ ] **Step 6: Run the spike and read the output**

Run: `python scripts/spike_datacheck.py`
Expected: both gates PASS; `distinct wallets` well above 50 out of 500 trades.

**If GATE 1 fails, stop and report — do not proceed to Task 2.** The whole plan rests on wallet-attributed trades.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/pmflow/ tests/test_config.py scripts/spike_datacheck.py
git commit -m "feat: config constants and data-availability kill-gate spike"
```

---

### Task 2: Warehouse schema

**Files:**
- Create: `src/pmflow/warehouse.py`
- Test: `tests/test_warehouse.py`

**Interfaces:**
- Consumes: `config`.
- Produces: `warehouse.connect(path: str = "data/warehouse.duckdb") -> duckdb.DuckDBPyConnection`, `warehouse.init_schema(con) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_warehouse.py
from pmflow import warehouse


def test_init_schema_creates_all_tables(tmp_path):
    con = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(con)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"poly_trades", "transfers", "kalshi_trades", "markets"} <= tables


def test_init_schema_is_idempotent(tmp_path):
    con = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(con)
    warehouse.init_schema(con)  # must not raise
    assert con.execute("SELECT count(*) FROM poly_trades").fetchone()[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_warehouse.py -v`
Expected: FAIL with `ImportError: cannot import name 'warehouse'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pmflow/warehouse.py
"""DuckDB warehouse. One file, no migrations -- drop and rebuild is cheap."""
from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    market_id     VARCHAR PRIMARY KEY,
    question      VARCHAR,
    slug          VARCHAR,
    end_ts        BIGINT,
    resolved_ts   BIGINT,
    outcome       TINYINT     -- 1 = YES resolved true, 0 = NO, NULL = open
);

CREATE TABLE IF NOT EXISTS poly_trades (
    trade_id      VARCHAR PRIMARY KEY,
    market_id     VARCHAR,
    wallet        VARCHAR,     -- proxy wallet (Gnosis Safe), not the EOA
    side          TINYINT,     -- 1 = buy YES, 0 = sell YES
    price_micro   BIGINT,      -- price in micro-USDC, 0..1_000_000
    size_micro    BIGINT,      -- size in micro-USDC of collateral
    ts            BIGINT
);

CREATE TABLE IF NOT EXISTS transfers (
    tx_hash       VARCHAR,
    from_addr     VARCHAR,
    to_addr       VARCHAR,
    amount_micro  BIGINT,
    ts            BIGINT,
    PRIMARY KEY (tx_hash, from_addr, to_addr, amount_micro)
);

CREATE TABLE IF NOT EXISTS kalshi_trades (
    trade_id      VARCHAR PRIMARY KEY,
    ticker        VARCHAR,
    price_micro   BIGINT,
    size          BIGINT,
    ts            BIGINT
);
"""


def connect(path: str = "data/warehouse.duckdb") -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_warehouse.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/pmflow/warehouse.py tests/test_warehouse.py
git commit -m "feat: duckdb warehouse schema"
```

---

### Task 3: Polymarket trade ingester

**Files:**
- Create: `src/pmflow/ingest_poly.py`
- Test: `tests/test_ingest_poly.py`

**Interfaces:**
- Consumes: `config`, `warehouse`.
- Produces: `ingest_poly.normalize_trade(raw: dict) -> dict | None`, `ingest_poly.fetch_trades(limit: int, offset: int) -> list[dict]`, `ingest_poly.load(con, pages: int) -> int` (returns rows inserted).

The parsing logic is tested against a fixture, not the live API — network tests are flaky and slow. `fetch_trades` is a thin uncovered shim by design; `normalize_trade` holds all the logic and all the tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_poly.py
from pmflow import ingest_poly

RAW = {
    "transactionHash": "0xabc",
    "conditionId": "0xmarket1",
    "proxyWallet": "0xWALLET1",
    "side": "BUY",
    "outcomeIndex": 0,
    "price": "0.63",
    "size": "125.5",
    "timestamp": 1750000000,
}


def test_normalize_converts_money_to_micro_ints():
    out = ingest_poly.normalize_trade(RAW)
    assert out["price_micro"] == 630_000
    assert out["size_micro"] == 125_500_000
    assert isinstance(out["price_micro"], int)


def test_normalize_lowercases_wallet():
    # Polygon addresses arrive in mixed checksum case; joins need one casing.
    assert ingest_poly.normalize_trade(RAW)["wallet"] == "0xwallet1"


def test_normalize_maps_sell_side_to_zero():
    out = ingest_poly.normalize_trade({**RAW, "side": "SELL"})
    assert out["side"] == 0


def test_normalize_flips_side_for_no_outcome():
    # outcomeIndex 1 is the NO token: buying NO is economically selling YES.
    # Without this flip every NO trade is counted backwards.
    out = ingest_poly.normalize_trade({**RAW, "outcomeIndex": 1})
    assert out["side"] == 0
    assert out["price_micro"] == 370_000  # 1 - 0.63, expressed as YES price


def test_normalize_rejects_row_missing_wallet():
    assert ingest_poly.normalize_trade({**RAW, "proxyWallet": None}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_poly.py -v`
Expected: FAIL with `ImportError: cannot import name 'ingest_poly'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pmflow/ingest_poly.py
"""Polymarket Data API -> poly_trades.

Everything is normalised to the YES side so that price and direction are
comparable across trades. A NO-token buy is a YES sell at (1 - price).
"""
from decimal import Decimal

import httpx

from . import config

MICRO = 1_000_000


def normalize_trade(raw: dict) -> dict | None:
    wallet = raw.get("proxyWallet")
    if not wallet:
        return None

    price = Decimal(str(raw["price"]))
    is_buy = str(raw.get("side", "")).upper() == "BUY"

    # outcomeIndex 1 == NO token. Restate as the equivalent YES-side trade.
    if int(raw.get("outcomeIndex", 0)) == 1:
        price = Decimal(1) - price
        is_buy = not is_buy

    return {
        "trade_id": f'{raw["transactionHash"]}:{raw.get("conditionId")}:{raw["timestamp"]}',
        "market_id": raw.get("conditionId"),
        "wallet": wallet.lower(),
        "side": 1 if is_buy else 0,
        "price_micro": int(price * MICRO),
        "size_micro": int(Decimal(str(raw["size"])) * MICRO),
        "ts": int(raw["timestamp"]),
    }


def fetch_trades(limit: int = 500, offset: int = 0) -> list[dict]:
    r = httpx.get(
        f"{config.POLY_DATA_API}/trades",
        params={"limit": limit, "offset": offset},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def load(con, pages: int = 10, limit: int = 500) -> int:
    rows = []
    for page in range(pages):
        batch = fetch_trades(limit=limit, offset=page * limit)
        if not batch:
            break
        rows.extend(r for r in (normalize_trade(t) for t in batch) if r)
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO poly_trades VALUES "
        "($trade_id, $market_id, $wallet, $side, $price_micro, $size_micro, $ts)",
        rows,
    )
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_poly.py -v`
Expected: 5 passed

- [ ] **Step 5: Smoke-test against the live API**

Run:
```bash
python -c "
from pmflow import warehouse, ingest_poly
con = warehouse.connect(); warehouse.init_schema(con)
print('inserted', ingest_poly.load(con, pages=2))
print(con.execute('SELECT count(*), count(DISTINCT wallet) FROM poly_trades').fetchone())
"
```
Expected: a nonzero insert count and a distinct-wallet count well above 1.

- [ ] **Step 6: Commit**

```bash
git add src/pmflow/ingest_poly.py tests/test_ingest_poly.py
git commit -m "feat: polymarket trade ingester with YES-side normalisation"
```

---

### Task 4: Funding-graph ingester

**Files:**
- Create: `src/pmflow/ingest_funding.py`
- Test: `tests/test_ingest_funding.py`

**Interfaces:**
- Consumes: `config`, `warehouse`, `poly_trades` table (for the target wallet set).
- Produces: `ingest_funding.decode_transfer(log: dict) -> dict`, `ingest_funding.load(con, from_block: int, to_block: int) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_funding.py
from pmflow import ingest_funding

# ERC-20 Transfer(address indexed from, address indexed to, uint256 value)
LOG = {
    "transaction_hash": "0xdead",
    "topics": [
        "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
        "0x000000000000000000000000AAAA000000000000000000000000000000000001",
        "0x000000000000000000000000BBBB000000000000000000000000000000000002",
    ],
    "data": "0x0000000000000000000000000000000000000000000000000000000005f5e100",
    "block_timestamp": 1750000000,
}


def test_decode_extracts_addresses_from_topics():
    out = ingest_funding.decode_transfer(LOG)
    assert out["from_addr"] == "0xaaaa000000000000000000000000000000000001"
    assert out["to_addr"] == "0xbbbb000000000000000000000000000000000002"


def test_decode_parses_amount_as_micro_usdc():
    # 0x05f5e100 == 100_000_000 == 100 USDC at 6 decimals
    assert ingest_funding.decode_transfer(LOG)["amount_micro"] == 100_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_funding.py -v`
Expected: FAIL with `ImportError: cannot import name 'ingest_funding'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pmflow/ingest_funding.py
"""Polygon USDC/USDC.e Transfer logs -> transfers.

Read via HyperSync rather than an RPC provider: the legacy Polymarket
subgraph was retired at the 2026-04-28 CTF migration, and plain RPC
log queries get throttled long before you finish a full-history scan.
"""
import hypersync
from hypersync import BlockField, LogField, TransactionField

from . import config

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def _addr_from_topic(topic: str) -> str:
    # Topics are 32 bytes; an address is the low 20. Strip 0x + 24 hex chars.
    return "0x" + topic[-40:].lower()


def decode_transfer(log: dict) -> dict:
    return {
        "tx_hash": log["transaction_hash"],
        "from_addr": _addr_from_topic(log["topics"][1]),
        "to_addr": _addr_from_topic(log["topics"][2]),
        "amount_micro": int(log["data"], 16),
        "ts": int(log["block_timestamp"]),
    }


async def load(con, from_block: int, to_block: int) -> int:
    client = hypersync.HypersyncClient(
        hypersync.ClientConfig(url="https://polygon.hypersync.xyz")
    )
    query = hypersync.Query(
        from_block=from_block,
        to_block=to_block,
        logs=[
            hypersync.LogSelection(
                address=[config.USDC_POLYGON, config.USDC_E_POLYGON],
                topics=[[TRANSFER_TOPIC]],
            )
        ],
        field_selection=hypersync.FieldSelection(
            log=[LogField.TRANSACTION_HASH, LogField.TOPIC0, LogField.TOPIC1,
                 LogField.TOPIC2, LogField.DATA, LogField.BLOCK_NUMBER],
            block=[BlockField.NUMBER, BlockField.TIMESTAMP],
        ),
    )
    res = await client.get(query)

    ts_by_block = {b.number: b.timestamp for b in res.data.blocks}
    rows = [
        decode_transfer({
            "transaction_hash": log.transaction_hash,
            "topics": [log.topic0, log.topic1, log.topic2],
            "data": log.data,
            "block_timestamp": ts_by_block[log.block_number],
        })
        for log in res.data.logs
    ]
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO transfers VALUES "
        "($tx_hash, $from_addr, $to_addr, $amount_micro, $ts)",
        rows,
    )
    return len(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest_funding.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/pmflow/ingest_funding.py tests/test_ingest_funding.py
git commit -m "feat: polygon USDC funding-graph ingester via hypersync"
```

---

### Task 5: Wallet clustering

**Files:**
- Create: `src/pmflow/cluster.py`
- Test: `tests/test_cluster.py`

**Interfaces:**
- Consumes: `poly_trades`, `transfers`.
- Produces: `cluster.funding_edges(con) -> list[tuple[str, str]]`, `cluster.cotrade_edges(con, window_s: int, min_shared: int) -> list[tuple[str, str]]`, `cluster.build(con, window_s: int = 300, min_shared: int = 3) -> dict[str, int]` mapping wallet -> cluster_id.

The hub-stripping rule is the load-bearing piece here: without it the funding graph collapses into one component containing every retail wallet on the exchange.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cluster.py
import pytest

from pmflow import cluster, warehouse


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    return c


def test_funding_edges_link_wallets_sharing_a_small_funder(con):
    con.executemany(
        "INSERT INTO transfers VALUES ($tx, $f, $t, $a, $ts)",
        [
            {"tx": "0x1", "f": "0xfunder", "t": "0xw1", "a": 100, "ts": 1},
            {"tx": "0x2", "f": "0xfunder", "t": "0xw2", "a": 100, "ts": 2},
        ],
    )
    assert cluster.funding_edges(con) == [("0xw1", "0xw2")]


def test_funding_edges_ignores_hub_funders(con):
    # An exchange hot wallet funds everyone; it must not link anyone.
    rows = [
        {"tx": f"0x{i}", "f": "0xbinance", "t": f"0xw{i}", "a": 100, "ts": i}
        for i in range(50)
    ]
    con.executemany("INSERT INTO transfers VALUES ($tx, $f, $t, $a, $ts)", rows)
    assert cluster.funding_edges(con) == []


def test_cotrade_edges_require_repetition_across_distinct_markets(con):
    # Two wallets on the same side of ONE market is coincidence, not a link.
    con.executemany(
        "INSERT INTO poly_trades VALUES ($id, $m, $w, $s, $p, $sz, $ts)",
        [
            {"id": "a", "m": "m1", "w": "0xw1", "s": 1, "p": 1, "sz": 1, "ts": 100},
            {"id": "b", "m": "m1", "w": "0xw2", "s": 1, "p": 1, "sz": 1, "ts": 110},
        ],
    )
    assert cluster.cotrade_edges(con, window_s=300, min_shared=3) == []


def test_build_merges_transitive_links(con):
    con.executemany(
        "INSERT INTO transfers VALUES ($tx, $f, $t, $a, $ts)",
        [
            {"tx": "0x1", "f": "0xfa", "t": "0xw1", "a": 1, "ts": 1},
            {"tx": "0x2", "f": "0xfa", "t": "0xw2", "a": 1, "ts": 2},
            {"tx": "0x3", "f": "0xfb", "t": "0xw2", "a": 1, "ts": 3},
            {"tx": "0x4", "f": "0xfb", "t": "0xw3", "a": 1, "ts": 4},
        ],
    )
    m = cluster.build(con)
    assert m["0xw1"] == m["0xw2"] == m["0xw3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cluster.py -v`
Expected: FAIL with `ImportError: cannot import name 'cluster'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pmflow/cluster.py
"""Wallet clustering from funding overlap + repeated co-trading.

Two wallets are linked if they share a *non-hub* funding source, or if they
repeatedly trade the same side of the same market within a short window
across several distinct markets. Connected components are the clusters.

ponytail: connected components, not Louvain. Components are deterministic
and need no resolution parameter to defend in a referee report. Switch to
Louvain only if components turn out to be too coarse to interpret.
"""
import networkx as nx

from .config import HUB_DEGREE_THRESHOLD


def funding_edges(con) -> list[tuple[str, str]]:
    """Wallet pairs sharing a funder that pays fewer than HUB_DEGREE_THRESHOLD wallets."""
    return [
        (a, b)
        for a, b in con.execute(
            """
            WITH funder_degree AS (
                SELECT from_addr, count(DISTINCT to_addr) AS deg
                FROM transfers GROUP BY from_addr
            ),
            small AS (
                SELECT t.from_addr, t.to_addr
                FROM transfers t JOIN funder_degree d USING (from_addr)
                WHERE d.deg < ?
            )
            SELECT DISTINCT a.to_addr, b.to_addr
            FROM small a JOIN small b
              ON a.from_addr = b.from_addr AND a.to_addr < b.to_addr
            """,
            [HUB_DEGREE_THRESHOLD],
        ).fetchall()
    ]


def cotrade_edges(con, window_s: int = 300, min_shared: int = 3) -> list[tuple[str, str]]:
    """Wallet pairs that co-traded the same side within window_s, in >= min_shared
    DISTINCT markets. The distinct-market requirement is what separates a real
    link from two people reacting to the same headline."""
    return [
        (a, b)
        for a, b in con.execute(
            """
            SELECT x.wallet, y.wallet
            FROM poly_trades x JOIN poly_trades y
              ON x.market_id = y.market_id
             AND x.side = y.side
             AND x.wallet < y.wallet
             AND abs(x.ts - y.ts) <= ?
            GROUP BY x.wallet, y.wallet
            HAVING count(DISTINCT x.market_id) >= ?
            """,
            [window_s, min_shared],
        ).fetchall()
    ]


def build(con, window_s: int = 300, min_shared: int = 3) -> dict[str, int]:
    g = nx.Graph()
    g.add_edges_from(funding_edges(con))
    g.add_edges_from(cotrade_edges(con, window_s, min_shared))
    return {
        wallet: cid
        for cid, component in enumerate(nx.connected_components(g))
        for wallet in component
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cluster.py -v`
Expected: 4 passed

- [ ] **Step 5: Sanity-check cluster sizes on real data**

Run:
```bash
python -c "
import collections
from pmflow import warehouse, cluster
con = warehouse.connect()
m = cluster.build(con)
sizes = collections.Counter(m.values())
print('clusters:', len(sizes), 'largest:', max(sizes.values(), default=0))
"
```
Expected: many clusters, largest well under half of all wallets. **If one cluster holds most wallets, hub-stripping failed** — raise `HUB_DEGREE_THRESHOLD` scrutiny before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/pmflow/cluster.py tests/test_cluster.py
git commit -m "feat: wallet clustering via funding overlap and repeated co-trading"
```

---

### Task 6: Cluster forecast skill

This is the task that converts an unverifiable identity claim into a testable one. Clusters are scored against resolution outcomes, which are hard labels.

**Files:**
- Create: `src/pmflow/skill.py`
- Test: `tests/test_skill.py`

**Interfaces:**
- Consumes: `poly_trades`, `markets`, `cluster.build`.
- Produces: `skill.brier(pred: float, outcome: int) -> float`, `skill.cluster_scores(con, clusters: dict[str, int], horizon_s: int) -> polars.DataFrame` with columns `cluster_id, n_trades, n_markets, mean_brier, edge_vs_market`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skill.py
import pytest

from pmflow import skill, warehouse


def test_brier_is_zero_for_perfect_confident_call():
    assert skill.brier(1.0, 1) == 0.0


def test_brier_is_one_for_confidently_wrong_call():
    assert skill.brier(1.0, 0) == 1.0


def test_brier_at_coinflip():
    assert skill.brier(0.5, 1) == 0.25


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    warehouse.init_schema(c)
    c.execute("INSERT INTO markets VALUES ('m1','q','s',1000,1000,1)")
    return c


def test_cluster_scores_excludes_trades_after_the_horizon_cutoff(con):
    # A trade 10s before resolution is not a forecast. Only trades at least
    # horizon_s before the end may count, or the whole result is lookahead.
    con.executemany(
        "INSERT INTO poly_trades VALUES ($id,$m,$w,$s,$p,$sz,$ts)",
        [
            {"id": "late", "m": "m1", "w": "0xw1", "s": 1,
             "p": 990_000, "sz": 1, "ts": 990},
        ],
    )
    df = skill.cluster_scores(con, {"0xw1": 0}, horizon_s=3600)
    assert len(df) == 0


def test_cluster_scores_rewards_an_early_correct_cluster(con):
    con.executemany(
        "INSERT INTO poly_trades VALUES ($id,$m,$w,$s,$p,$sz,$ts)",
        [
            {"id": "early", "m": "m1", "w": "0xw1", "s": 1,
             "p": 200_000, "sz": 1, "ts": 100},
        ],
    )
    df = skill.cluster_scores(con, {"0xw1": 0}, horizon_s=100)
    # Bought YES at 0.20 and YES resolved true -> big positive edge.
    assert df["edge_vs_market"][0] == pytest.approx(0.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill.py -v`
Expected: FAIL with `ImportError: cannot import name 'skill'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pmflow/skill.py
"""Score wallet clusters against resolution outcomes.

Clusters have no ground truth -- you can never prove two wallets are one
person. Resolution outcomes DO have ground truth. So the claim under test is
never "these wallets are the same entity", it is "this cluster's flow
predicts the outcome", which is falsifiable.

horizon_s exists purely to prevent lookahead: a trade placed seconds before
resolution carries no information and would inflate every score.
"""
import polars as pl

MICRO = 1_000_000


def brier(pred: float, outcome: int) -> float:
    return (pred - outcome) ** 2


def cluster_scores(con, clusters: dict[str, int], horizon_s: int = 3600) -> pl.DataFrame:
    rows = con.execute(
        """
        SELECT t.wallet, t.market_id, t.side, t.price_micro, t.size_micro, m.outcome
        FROM poly_trades t
        JOIN markets m USING (market_id)
        WHERE m.outcome IS NOT NULL
          AND t.ts <= m.resolved_ts - ?
        """,
        [horizon_s],
    ).fetchall()

    recs = []
    for wallet, market_id, side, price_micro, size_micro, outcome in rows:
        cid = clusters.get(wallet)
        if cid is None:
            continue
        price = price_micro / MICRO
        # The cluster's implied forecast: buying YES asserts P(YES) > price.
        implied = price if side == 1 else 1 - price
        realised = outcome if side == 1 else 1 - outcome
        recs.append({
            "cluster_id": cid,
            "market_id": market_id,
            "brier": brier(implied, realised),
            # Edge = what the position actually paid off versus what it cost.
            "edge": realised - implied,
            "size_micro": size_micro,
        })

    if not recs:
        return pl.DataFrame(schema={
            "cluster_id": pl.Int64, "n_trades": pl.UInt32, "n_markets": pl.UInt32,
            "mean_brier": pl.Float64, "edge_vs_market": pl.Float64,
        })

    return (
        pl.DataFrame(recs)
        .group_by("cluster_id")
        .agg(
            pl.len().alias("n_trades"),
            pl.col("market_id").n_unique().alias("n_markets"),
            pl.col("brier").mean().alias("mean_brier"),
            pl.col("edge").mean().alias("edge_vs_market"),
        )
        .sort("edge_vs_market", descending=True)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/pmflow/skill.py tests/test_skill.py
git commit -m "feat: cluster forecast skill scored against resolution outcomes"
```

---

### Task 7: Liquidity vs accuracy regression (idea 4)

**Files:**
- Create: `src/pmflow/liquidity.py`
- Test: `tests/test_liquidity.py`

**Interfaces:**
- Consumes: `poly_trades`, `markets`.
- Produces: `liquidity.contract_panel(con, horizon_s: int) -> polars.DataFrame`, `liquidity.fit(panel) -> statsmodels result`.

**The two controls are the whole task.** Volume concentrates on contracts that are near 50/50 and near resolution — exactly where Brier scores are naturally worst. Omit `dist_from_half` and `log_horizon` and the regression will report that liquidity destroys accuracy, which is an artifact, not a finding.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_liquidity.py
import polars as pl

from pmflow import liquidity


def test_fit_includes_both_confounder_controls():
    # Without these controls the liquidity coefficient is uninterpretable.
    panel = pl.DataFrame({
        "market_id": [f"m{i}" for i in range(60)],
        "brier": [0.1 + (i % 7) * 0.02 for i in range(60)],
        "log_volume": [float(i % 11) for i in range(60)],
        "dist_from_half": [(i % 5) * 0.1 for i in range(60)],
        "log_horizon": [float(i % 9) for i in range(60)],
    })
    res = liquidity.fit(panel)
    assert "dist_from_half" in res.params.index
    assert "log_horizon" in res.params.index
    assert "log_volume" in res.params.index


def test_contract_panel_drops_unresolved_markets(warehouse_con=None):
    from pmflow import warehouse
    import tempfile, pathlib
    d = tempfile.mkdtemp()
    con = warehouse.connect(str(pathlib.Path(d) / "t.duckdb"))
    warehouse.init_schema(con)
    con.execute("INSERT INTO markets VALUES ('open','q','s',9999,NULL,NULL)")
    con.execute(
        "INSERT INTO poly_trades VALUES ('t','open','0xw',1,500000,1000000,10)"
    )
    assert len(liquidity.contract_panel(con)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_liquidity.py -v`
Expected: FAIL with `ImportError: cannot import name 'liquidity'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/pmflow/liquidity.py
"""Does more liquidity make a contract's forecast better or noisier?

Cross-sectional WITHIN one cycle, deliberately. Comparing IEM (1988-2004)
against Polymarket (2024+) would confound liquidity with the collapse in
polling response rates over the same period -- that confound is why the
question is still open, and going cross-sectional sidesteps it entirely.
"""
import numpy as np
import polars as pl
import statsmodels.api as sm

MICRO = 1_000_000

CONTROLS = ["log_volume", "dist_from_half", "log_horizon"]


def contract_panel(con, horizon_s: int = 86_400) -> pl.DataFrame:
    """One row per resolved market: accuracy of its price at `horizon_s`
    before resolution, plus liquidity and the two confound controls."""
    rows = con.execute(
        """
        WITH snap AS (
            SELECT t.market_id,
                   last(t.price_micro ORDER BY t.ts) AS price_micro,
                   m.outcome,
                   m.resolved_ts,
                   min(t.ts) AS first_ts
            FROM poly_trades t
            JOIN markets m USING (market_id)
            WHERE m.outcome IS NOT NULL
              AND t.ts <= m.resolved_ts - ?
            GROUP BY t.market_id, m.outcome, m.resolved_ts
        ),
        vol AS (
            SELECT market_id, sum(size_micro) AS volume_micro
            FROM poly_trades GROUP BY market_id
        )
        SELECT s.market_id, s.price_micro, s.outcome, v.volume_micro,
               s.resolved_ts - s.first_ts AS lifespan_s
        FROM snap s JOIN vol v USING (market_id)
        """,
        [horizon_s],
    ).fetchall()

    if not rows:
        return pl.DataFrame(schema={
            "market_id": pl.Utf8, "brier": pl.Float64, "log_volume": pl.Float64,
            "dist_from_half": pl.Float64, "log_horizon": pl.Float64,
        })

    return pl.DataFrame([
        {
            "market_id": mid,
            "brier": (price_micro / MICRO - outcome) ** 2,
            "log_volume": float(np.log1p(volume_micro / MICRO)),
            # Control 1: contracts near 50/50 have mechanically worse Brier.
            "dist_from_half": abs(price_micro / MICRO - 0.5),
            # Control 2: long-lived contracts are forecast further out.
            "log_horizon": float(np.log1p(max(lifespan_s, 0))),
        }
        for mid, price_micro, outcome, volume_micro, lifespan_s in rows
    ])


def fit(panel: pl.DataFrame):
    x = sm.add_constant(panel.select(CONTROLS).to_pandas())
    y = panel["brier"].to_pandas()
    return sm.OLS(y, x).fit(cov_type="HC3")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_liquidity.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/pmflow/liquidity.py tests/test_liquidity.py
git commit -m "feat: liquidity vs forecast accuracy with endogeneity controls"
```

---

### Task 8: Cross-venue lead-lag (Polymarket vs Kalshi)

**Files:**
- Create: `src/pmflow/ingest_kalshi.py`
- Create: `src/pmflow/leadlag.py`
- Test: `tests/test_leadlag.py`

**Interfaces:**
- Consumes: `poly_trades`, `kalshi_trades`.
- Produces: `ingest_kalshi.load(con, ticker: str) -> int`, `leadlag.align(con, market_id: str, ticker: str, bar_s: int) -> polars.DataFrame`, `leadlag.information_share(aligned: polars.DataFrame, lags: int) -> tuple[float, float]`.

Kalshi publishes no trader identity, so this is the only venue comparison the data supports. It answers "which venue's price moves first", not "which venue's traders are informed".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_leadlag.py
import numpy as np
import polars as pl

from pmflow import leadlag


def test_information_share_sums_to_one():
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(size=500))
    aligned = pl.DataFrame({
        "poly_price": common + rng.normal(scale=0.01, size=500),
        "kalshi_price": common + rng.normal(scale=0.01, size=500),
    })
    poly_share, kalshi_share = leadlag.information_share(aligned, lags=5)
    assert abs(poly_share + kalshi_share - 1.0) < 1e-9


def test_leading_venue_gets_the_larger_share():
    # Kalshi is Polymarket delayed by 3 bars, plus noise -> poly must lead.
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.normal(size=500))
    aligned = pl.DataFrame({
        "poly_price": common,
        "kalshi_price": np.concatenate([np.zeros(3), common[:-3]])
                        + rng.normal(scale=0.001, size=500),
    })
    poly_share, kalshi_share = leadlag.information_share(aligned, lags=5)
    assert poly_share > kalshi_share
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_leadlag.py -v`
Expected: FAIL with `ImportError: cannot import name 'leadlag'`

- [ ] **Step 3: Write the Kalshi ingester**

```python
# src/pmflow/ingest_kalshi.py
"""Kalshi public trade feed -> kalshi_trades.

Kalshi market data needs no authentication. It also exposes no trader
identity -- maker and taker are anonymous -- which is why the venue
comparison in leadlag.py is price-based rather than trader-based.
"""
import httpx

from . import config

MICRO = 1_000_000


def normalize_trade(raw: dict) -> dict:
    return {
        "trade_id": raw["trade_id"],
        "ticker": raw["ticker"],
        # Kalshi quotes YES in integer cents; restate as micro-USDC.
        "price_micro": int(raw["yes_price"]) * 10_000,
        "size": int(raw["count"]),
        "ts": int(raw["created_time_ts"]) if "created_time_ts" in raw
              else int(raw["created_time"]),
    }


def load(con, ticker: str, pages: int = 20) -> int:
    rows, cursor = [], None
    for _ in range(pages):
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = httpx.get(f"{config.KALSHI_API}/markets/trades", params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("trades", [])
        if not batch:
            break
        rows.extend(normalize_trade(t) for t in batch)
        cursor = payload.get("cursor")
        if not cursor:
            break
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO kalshi_trades VALUES "
        "($trade_id, $ticker, $price_micro, $size, $ts)",
        rows,
    )
    return len(rows)
```

- [ ] **Step 4: Write the lead-lag implementation**

```python
# src/pmflow/leadlag.py
"""Which venue discovers price first?

Hasbrouck (1995) information share on a cross-listed contract. Two prices
tracking one latent value: whichever venue's innovations explain more of the
permanent component is doing the price discovery.

ponytail: reduced-form VAR residual decomposition, not a full VECM. The
cointegrating vector for the same contract on two venues is known a priori
to be (1, -1) -- there is nothing to estimate, so estimating it adds a
standard error for free. Upgrade to a full VECM only if a referee asks.
"""
import numpy as np
import polars as pl
from statsmodels.tsa.api import VAR

MICRO = 1_000_000


def align(con, market_id: str, ticker: str, bar_s: int = 60) -> pl.DataFrame:
    """Last price in each bar_s bucket on both venues, inner-joined on bucket."""
    rows = con.execute(
        """
        WITH p AS (
            SELECT (ts / ?)::BIGINT AS bucket,
                   last(price_micro ORDER BY ts) AS poly_price
            FROM poly_trades WHERE market_id = ? GROUP BY 1
        ),
        k AS (
            SELECT (ts / ?)::BIGINT AS bucket,
                   last(price_micro ORDER BY ts) AS kalshi_price
            FROM kalshi_trades WHERE ticker = ? GROUP BY 1
        )
        SELECT p.bucket, p.poly_price, k.kalshi_price
        FROM p JOIN k USING (bucket) ORDER BY p.bucket
        """,
        [bar_s, market_id, bar_s, ticker],
    ).fetchall()
    return pl.DataFrame(
        [
            {"bucket": b, "poly_price": pp / MICRO, "kalshi_price": kp / MICRO}
            for b, pp, kp in rows
        ],
        schema={"bucket": pl.Int64, "poly_price": pl.Float64, "kalshi_price": pl.Float64},
    )


def information_share(aligned: pl.DataFrame, lags: int = 5) -> tuple[float, float]:
    """Return (poly_share, kalshi_share), summing to 1.

    Hasbrouck bounds are order-dependent when residuals correlate; we report
    the midpoint of the two Cholesky orderings, which is the convention.
    """
    prices = aligned.select(["poly_price", "kalshi_price"]).to_numpy()
    diffs = np.diff(prices, axis=0)

    res = VAR(diffs).fit(maxlags=lags)
    omega = np.cov(res.resid, rowvar=False)

    def share_given_order(order: list[int]) -> float:
        chol = np.linalg.cholesky(omega[np.ix_(order, order)])
        # Equal-weight the permanent-component loadings: for two venues
        # tracking one value the row sums of the Cholesky factor give the
        # contribution of each venue's innovation to the common factor.
        contrib = chol.sum(axis=0) ** 2
        s = contrib / contrib.sum()
        return s[order.index(0)]

    poly = 0.5 * (share_given_order([0, 1]) + share_given_order([1, 0]))
    return poly, 1.0 - poly
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_leadlag.py -v`
Expected: 2 passed

- [ ] **Step 6: Run the whole suite**

Run: `pytest -v`
Expected: all tests pass, no failures.

- [ ] **Step 7: Commit**

```bash
git add src/pmflow/ingest_kalshi.py src/pmflow/leadlag.py tests/test_leadlag.py
git commit -m "feat: kalshi ingester and cross-venue Hasbrouck information share"
```

---

## What is deliberately not in this plan

- **Idea 2 (ToS diffing)** and **idea 3 (LLM sentiment)** — separate projects, separate plans. Ask if you want either after Task 1 clears.
- **Backtesting or execution** — this is a measurement paper, not a strategy. Adding a backtester before the signal is measured is how the project turns into a trading bot that never ships a paper.
- **A scheduler / live pipeline** — everything is batch over a fixed window. Add live ingestion only if the paper needs an out-of-sample period you don't have yet.
- **Plots** — result tables land in DuckDB; charting is one notebook at the end, not a module.

## Sources

- [Polymarket subgraph & data API docs](https://docs.polymarket.com/developers/subgraph/overview)
- [Polymarket magic proxy builder (proxy wallet / relayer)](https://github.com/Polymarket/magic-proxy-builder-example)
- [Kalshi historical data docs](https://docs.kalshi.com/getting_started/historical_data)
- [Kalshi API guide 2026](https://pm.wiki/learn/kalshi-api)
- [Iowa Electronic Markets historical data 1988–1996](https://iemweb.biz.uiowa.edu/markets/historical-data-1988-1996)
- [IEM research page](https://iemweb.biz.uiowa.edu/research)
- [X/Twitter API pricing 2026](https://postproxy.dev/blog/x-api-pricing-2026/)
- [Twitter academic research API status](https://xcrop.io/blog/twitter-academic-research-api-guide)
