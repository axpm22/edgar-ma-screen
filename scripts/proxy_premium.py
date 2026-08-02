"""Do the deals this model finds carry thinner premiums than the ones it misses?

That is Palepu (1986) restated as a testable question on this panel. If the
market has already identified a target, the takeover premium is partly paid
away before a screen could buy it, and a model that predicts targets earns
nothing. If flagged and missed deals carry the same premium, the objection
weakens.

The premium is free: roughly 43% of merger proxies state it in words
("a premium of approximately 38% to the closing price"). The quarterly price
tables do NOT parse -- 3.3% of 2023-24 proxies, see scope_proxy_prices.py --
so this is a premium distribution, never a price series.

Bytes are kept down with a 3MB range request: the premium statement sits at
median byte 0.35MB and 100% of them land inside the first 3MB, against a 4MB
mean document.

    .venv/bin/python scripts/proxy_premium.py [n_docs]
"""
import datetime as dt
import json
import re
import sys

import duckdb
import httpx
import numpy as np
import polars as pl

from deal import clean_labels, config, fetch, universe

ACC = re.compile(r"(\d{10}-\d{2}-\d{6})")
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"[ \t\xa0]+")
PREMIUM = re.compile(
    r"premium\s+of\s+(?:approximately\s+)?(\d{1,3}(?:\.\d)?)\s*%", re.I)
# Range applies to the GZIPPED stream: 1MB compressed decompresses to roughly
# 6MB, and the premium statement sits at median byte 0.35MB decompressed. A
# 3MB cap pulled ~18MB of text per document for no benefit.
DOC_BYTES = 1_000_000
TIMEOUT = 45


def strip_html(raw: bytes) -> str:
    t = raw.decode("latin-1", "ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    return WS.sub(" ", TAG.sub(" ", t).replace("&nbsp;", " "))


def head_of(cik: str, accession: str) -> bytes | None:
    """First DOC_BYTES of the primary document. Cached, so re-runs are free."""
    a = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a}"
    idx = json.loads(fetch.sec_get(f"{base}/index.json"))
    items = [i for i in idx["directory"]["item"]
             if i["name"].lower().endswith((".htm", ".html", ".txt"))]
    if not items:
        return None
    items.sort(key=lambda i: -int(i.get("size") or 0))
    url = f"{base}/{items[0]['name']}"

    def _go() -> bytes:
        fetch.SEC_LIMITER.wait()
        r = httpx.get(url, headers={"User-Agent": config.EDGAR_UA,
                                    "Range": f"bytes=0-{DOC_BYTES}"},
                      timeout=TIMEOUT)
        r.raise_for_status()
        return r.content

    return fetch.cached("sec_head", url, _go)


def verified_targets():
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
    con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
    clean_labels.build(con, pl.read_parquet(
        "data/features.parquet", columns=["week"])["week"].max())
    return con.execute("""
        SELECT cik, agreement_date FROM deals_clean
        WHERE outcome = 'target' AND agreement_date >= DATE '2023-01-01'
        ORDER BY agreement_date""").fetchall()


def model_rank(ranked, cik, when):
    """Best (lowest) within-week rank in the 52 weeks before the proxy."""
    h = ranked.filter((pl.col("cik") == cik) & (pl.col("week") < when)
                      & (pl.col("week") >= when - dt.timedelta(weeks=52)))
    return int(h["rk"].min()) if h.height else None


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    idx = {}
    for y, q in universe.quarters(2023, 2026):
        p = fetch.cache_path("sec", config.IDX_URL.format(year=y, q=q))
        if not p.exists():
            continue
        for r in universe.parse_master_idx(p.read_bytes()):
            if r["form"] == "DEFM14A":
                idx.setdefault(r["cik"], []).append(r)

    targets = [t for t in verified_targets() if t[0] in idx]
    step = max(1, len(targets) // limit)
    targets = targets[::step][:limit]
    print(f"{len(targets)} verified targets with a DEFM14A, 2023+\n", flush=True)

    sc = pl.read_parquet("data/pair_scores.parquet")
    ranked = sc.with_columns(
        pl.col("p_target").rank("ordinal", descending=True)
        .over("week").alias("rk")).select(["cik", "week", "rk"])

    out, nbytes = [], 0
    for cik, when in targets:
        cand = min(idx[cik], key=lambda r: abs((r["file_date"] - when).days))
        m = ACC.search(cand["filename"])
        if not m:
            continue
        try:
            raw = head_of(cik, m.group(1))
        except Exception as e:
            print(f"  {cik:<10} FETCH {type(e).__name__}", flush=True)
            continue
        if not raw:
            continue
        nbytes += len(raw)
        hits = [float(x) for x in PREMIUM.findall(strip_html(raw))
                if 0 < float(x) < 400]
        if not hits:
            continue
        out.append({"cik": cik, "date": when,
                    # Several reference prices are quoted; the first stated is
                    # conventionally against the unaffected close.
                    "premium": hits[0],
                    "rank": model_rank(ranked, cik, when)})
        print(f"  {cik:<10} {when}  premium {hits[0]:>5.1f}%  "
              f"rank {out[-1]['rank']}", flush=True)

    df = pl.DataFrame(out)
    print(f"\n{'='*62}")
    print(f"fetched {nbytes/1e6:.0f} MB | premium parsed for "
          f"{df.height}/{len(targets)} ({100*df.height/max(len(targets),1):.0f}%)")
    if not df.height:
        return
    p = df["premium"].to_numpy()
    print(f"premium: median {np.median(p):.1f}%  p25 {np.percentile(p,25):.1f}%"
          f"  p75 {np.percentile(p,75):.1f}%")

    known = df.filter(pl.col("rank").is_not_null())
    if known.height >= 10:
        r = known["rank"].to_numpy()
        pr = known["premium"].to_numpy()
        cut = np.median(r)
        hi, lo = pr[r <= cut], pr[r > cut]
        print(f"\nscored by the model: {known.height} deals "
              f"(median best rank {cut:.0f})")
        print(f"  model ranked BETTER than median: median premium "
              f"{np.median(hi):.1f}%  (n={len(hi)})")
        print(f"  model ranked WORSE  than median: median premium "
              f"{np.median(lo):.1f}%  (n={len(lo)})")
        from scipy import stats
        u = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        sp = stats.spearmanr(r, pr)
        print(f"  Mann-Whitney p={u.pvalue:.3f}   "
              f"Spearman(rank, premium) rho={sp.statistic:+.3f} "
              f"p={sp.pvalue:.3f}")
        print("\n  Palepu would predict better-ranked deals carry THINNER"
              " premiums\n  (positive rho, since a better rank is a SMALLER"
              " number).")
    json.dump([{**r, "date": str(r["date"])} for r in out],
              open("data/proxy_premium.json", "w"), indent=1)
    print("\nwrote data/proxy_premium.json")


if __name__ == "__main__":
    main()
