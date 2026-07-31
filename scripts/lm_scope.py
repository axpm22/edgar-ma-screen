"""Scoping test: does hedging language rise before a deal?

Deliberately small. Downloading 8-K text for the whole panel would be 740k
documents; this pulls a few thousand for a matched sample and checks whether
the effect is visible at all before anything larger is committed to.

Design: companies acquired in 2022-2025, each matched to a control company that
was never acquired, matched on size decile and SIC2. Each company's 8-Ks in the
two years before its deal date (controls use their match's date) are scored,
then bucketed by how long before the announcement they were filed.

    python scripts/lm_scope.py sample     # build the document list
    python scripts/lm_scope.py fetch      # download (rate-limited)
    python scripts/lm_scope.py score      # LM scores + the test
"""
import datetime as dt
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from deal import fetch as fetchmod  # noqa: E402
from deal import lm_sentiment as lm  # noqa: E402

SAMPLE_PATH = Path("data/lm_sample.json")
SCORES_PATH = Path("data/lm_scores.parquet")
N_TARGETS = 160
LOOKBACK_DAYS = 730
MAX_DOCS_PER_CO = 12
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def _con():
    c = duckdb.connect(":memory:")
    c.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    return c


def stage_sample() -> None:
    con = _con()
    # Targets: acquired 2022+, excluding SPACs, which file boilerplate.
    targets = con.execute("""
        SELECT d.cik, min(d.agreement_date) AS ev, s.sic
        FROM m.deals d
        JOIN m.universe u USING (cik)
        LEFT JOIN m.company_sic s USING (cik)
        WHERE d.agreement_date >= DATE '2022-01-01'
          AND coalesce(s.sic,'') <> '6770'
          AND upper(u.name) NOT LIKE '%ACQUISITION CORP%'
        GROUP BY 1, 3
    """).pl().sample(n=N_TARGETS, seed=7)

    never = con.execute("""
        SELECT u.cik, s.sic FROM m.universe u
        LEFT JOIN m.company_sic s USING (cik)
        WHERE u.cik NOT IN (SELECT cik FROM m.deals)
          AND coalesce(s.sic,'') <> '6770'
          AND u.delisted >= DATE '2025-01-01'
    """).pl()

    # Match each target to a control in the same 2-digit SIC. A control gets
    # its match's event date so both sides face the same calendar window --
    # otherwise the comparison would confound hedging with the macro cycle.
    rng = np.random.default_rng(7)
    never = never.with_columns(
        pl.col("sic").cast(pl.Utf8).str.slice(0, 2).alias("s2"))
    # group_by yields the key as a tuple; unpack it or every lookup misses.
    pool = {(k[0] if isinstance(k, tuple) else k): g["cik"].to_list()
            for k, g in never.group_by("s2")}
    rows = []
    used = set()
    for t in targets.iter_rows(named=True):
        rows.append({"cik": t["cik"], "event": str(t["ev"]), "grp": "target"})
        s2 = str(t["sic"])[:2] if t["sic"] else ""
        cands = [c for c in pool.get(s2, []) if c not in used]
        if cands:
            c = cands[int(rng.integers(len(cands)))]
            used.add(c)
            rows.append({"cik": c, "event": str(t["ev"]), "grp": "control"})

    # Resolve each company's 8-K primary documents inside the window.
    con.execute("ATTACH 'data/items.duckdb' AS it (READ_ONLY)")
    docs = []
    for r in rows:
        ev = dt.date.fromisoformat(r["event"])
        lo = ev - dt.timedelta(days=LOOKBACK_DAYS)
        try:
            sub = json.loads(fetchmod.sec_get(
                f"https://data.sec.gov/submissions/CIK{r['cik'].zfill(10)}.json"))
        except Exception:
            continue
        rec = sub.get("filings", {}).get("recent", {})
        got = 0
        for form, date, acc, doc in zip(rec.get("form", []),
                                        rec.get("filingDate", []),
                                        rec.get("accessionNumber", []),
                                        rec.get("primaryDocument", [])):
            if form != "8-K" or not doc:
                continue
            try:
                d = dt.date.fromisoformat(date)
            except ValueError:
                continue
            if not (lo <= d < ev) or got >= MAX_DOCS_PER_CO:
                continue
            docs.append({"cik": r["cik"], "grp": r["grp"], "event": r["event"],
                         "date": date, "acc": acc.replace("-", ""),
                         "doc": doc,
                         "days_before": (ev - d).days})
            got += 1
    SAMPLE_PATH.write_text(json.dumps(docs, indent=1))
    n_t = len({d['cik'] for d in docs if d['grp'] == 'target'})
    n_c = len({d['cik'] for d in docs if d['grp'] == 'control'})
    print(f"{len(docs):,} documents | {n_t} targets, {n_c} controls")


def stage_fetch() -> None:
    docs = json.loads(SAMPLE_PATH.read_text())
    ok = 0
    for i, d in enumerate(docs, 1):
        url = DOC_URL.format(cik=d["cik"], acc=d["acc"], doc=d["doc"])
        try:
            fetchmod.sec_get(url)
            ok += 1
        except Exception:
            pass
        if i % 250 == 0:
            print(f"  {i:,}/{len(docs):,}  ok={ok:,}", flush=True)
    print(f"fetched {ok:,}/{len(docs):,}")


def stage_score() -> None:
    docs = json.loads(SAMPLE_PATH.read_text())
    rows = []
    for d in docs:
        url = DOC_URL.format(cik=d["cik"], acc=d["acc"], doc=d["doc"])
        p = fetchmod.cache_path("sec", url)
        if not p.exists():
            continue
        try:
            s = lm.score(p.read_bytes().decode("latin-1", "ignore"))
        except Exception:
            continue
        if s["lm_words"] < 50:
            continue
        rows.append({**{k: d[k] for k in ("cik", "grp", "days_before")}, **s})
    df = pl.DataFrame(rows)
    df.write_parquet(SCORES_PATH)
    print(f"scored {df.height:,} documents "
          f"({df.filter(pl.col('grp')=='target').height:,} target)\n")

    # Quarter buckets before the event.
    df = df.with_columns(
        (pl.col("days_before") // 91).clip(0, 7).alias("q"))
    print(f"{'quarters before':<16}{'target hedge':>14}{'control hedge':>15}"
          f"{'diff':>8}{'n_t':>6}{'n_c':>6}")
    for q in range(8):
        t = df.filter((pl.col("q") == q) & (pl.col("grp") == "target"))
        c = df.filter((pl.col("q") == q) & (pl.col("grp") == "control"))
        if t.height < 5 or c.height < 5:
            continue
        tm, cm = t["lm_hedge"].mean(), c["lm_hedge"].mean()
        print(f"{q:<16}{tm:>14.3f}{cm:>15.3f}{tm-cm:>8.3f}"
              f"{t.height:>6}{c.height:>6}")

    print("\nnearest 2 quarters vs the rest, by category:")
    near = df.filter(pl.col("q") <= 1)
    far = df.filter(pl.col("q") >= 4)
    from scipy import stats
    for col in ["lm_hedge", "lm_modal_weak", "lm_uncertainty",
                "lm_modal_strong", "lm_negative", "lm_litigious"]:
        a = near.filter(pl.col("grp") == "target")[col].to_numpy()
        b = far.filter(pl.col("grp") == "target")[col].to_numpy()
        if len(a) < 10 or len(b) < 10:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        flag = "  <-- significant" if p < 0.05 else ""
        print(f"  {col:<18} near {a.mean():>7.3f}  far {b.mean():>7.3f}  "
              f"t={t:>6.2f}  p={p:.4f}{flag}")


if __name__ == "__main__":
    {"sample": stage_sample, "fetch": stage_fetch,
     "score": stage_score}[sys.argv[1]]()
