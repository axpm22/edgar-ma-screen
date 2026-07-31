"""Comprehensive stress suite. ONE test per process invocation.

Earlier attempts ran every test in a single process and were killed by the OS
twice. Python does not reliably return LightGBM's arena memory to the system,
so the fix is structural: each test runs as its own process, appends a line to
data/stress_results.json, and exits. The driver is a shell loop.

    python scripts/stress_suite.py <test>     # one of the TESTS keys
    python scripts/stress_suite.py report     # print everything collected

Every configuration is measured on the CLEAN split: train ends before the
validation year, early stopping uses the validation year, and the test year is
never seen during training.
"""
import datetime as dt
import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL, relabel  # noqa: E402

from deal import features, screen  # noqa: E402

RESULTS = Path("data/stress_results.json")
SAMPLE = 0.25
ROUNDS = 200
SEEDS = (11, 22)

PARAMS = {
    "objective": "binary", "metric": "average_precision",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbosity": -1, "num_threads": 2,
}


def record(test: str, label: str, **kw) -> None:
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    rows.append({"test": test, "label": label, **kw})
    RESULTS.write_text(json.dumps(rows, indent=1, default=str))
    print(f"  {label:<34} " +
          "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in kw.items()), flush=True)


def load(test_year: int = 2024):
    """Clean split: train < (test_year-1), validate (test_year-1), test year."""
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    df = relabel(raw, HORIZON).select(["cik", "week", "y"] + cols)
    del raw
    gc.collect()
    safe = df["week"].max() - dt.timedelta(weeks=HORIZON)
    va_start = dt.date(test_year - 1, 1, 1)
    te_start = dt.date(test_year, 1, 1)
    te_end = min(dt.date(test_year, 12, 31), safe)
    tr = df.filter(pl.col("week") < va_start).sample(fraction=SAMPLE, seed=1)
    va = df.filter((pl.col("week") >= va_start) & (pl.col("week") < te_start))
    te = df.filter((pl.col("week") >= te_start) & (pl.col("week") <= te_end))
    del df
    gc.collect()
    return tr, va, te, cols


def fit_pred(tr, va, te, cols, seed, params=None, rounds=ROUNDS):
    p = {**PARAMS, **(params or {}), "bagging_seed": seed,
         "feature_fraction_seed": seed, "data_random_seed": seed}
    dtr = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                      label=tr["y"].to_pandas())
    dva = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                      label=va["y"].to_pandas())
    b = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    pred = np.asarray(b.predict(te.select(cols).to_pandas().astype("float32")))
    del b, dtr, dva
    gc.collect()
    return pred


def seeded(tr, va, te, cols, params=None):
    vals = []
    for s in SEEDS:
        pr = fit_pred(tr, va, te, cols, s, params)
        vals.append(screen.weekly_precision(te, pr, N_EVAL)["precision"] * 100)
        del pr
        gc.collect()
    a = np.array(vals)
    return float(a.mean()), float(a.std())


# --------------------------------------------------------------------------- #

def t_baseline():
    tr, va, te, cols = load()
    m, sd = seeded(tr, va, te, cols)
    record("baseline", "clean split 2024", mean=m, sd=sd, n_feat=len(cols),
           base=float(te["y"].mean() * 100))


def t_cv():
    """Time-series CV: the same design rolled across three test years."""
    for yr in (2023, 2024, 2025):
        tr, va, te, cols = load(yr)
        if not te.height or not te["y"].sum():
            continue
        m, sd = seeded(tr, va, te, cols)
        record("cv", f"test year {yr}", mean=m, sd=sd,
               base=float(te["y"].mean() * 100), rows=te.height)
        del tr, va, te
        gc.collect()


def t_permutation():
    tr, va, te, cols = load()
    real, _ = seeded(tr, va, te, cols)
    null = []
    for k in range(5):
        sh = tr.with_columns(pl.col("y").shuffle(seed=700 + k)
                             .over("week").alias("y"))
        pr = fit_pred(sh, va, te, cols, 11)
        null.append(screen.weekly_precision(te, pr, N_EVAL)["precision"] * 100)
        del sh, pr
        gc.collect()
    null = np.array(null)
    record("permutation", "scrambled labels", real=real,
           null_mean=float(null.mean()), null_max=float(null.max()),
           beats_all=bool(real > null.max()))


def t_ablation():
    tr, va, te, cols = load()
    from deal.feat_literature import LIT_COLS
    from deal.feat_items import ITEM_COLS
    full, full_sd = seeded(tr, va, te, cols)
    record("ablation", "ALL", mean=full, sd=full_sd)
    fam = {
        "z-scores": features.ZSCORE_COLS, "literature": LIT_COLS,
        "8-K items": ITEM_COLS, "deltas": features.DELTA_COLS,
        "strategic text": features.FTS_COLS, "form counts": features.FORM_COLS,
        "insider": features.INSIDER_COLS, "market": features.MARKET_COLS,
        "activist+peer": features.ACTIVIST_COLS + features.PEER_COLS,
    }
    for name, block in fam.items():
        keep = [c for c in cols if c not in block]
        if len(keep) == len(cols):
            continue
        m, sd = seeded(tr, va, te, keep)
        record("ablation", f"without {name}", mean=m, sd=sd,
               contributes=full - m,
               verdict=("HELPS" if full - m > 2 * full_sd else
                        "HURTS" if full - m < -2 * full_sd else "noise"))


def t_embargo():
    import duckdb
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    base = raw.select(["cik", "week"] + cols)
    del raw
    gc.collect()
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    keys = base.select(["cik", "week"])
    con.register("f", keys.to_arrow())
    safe = base["week"].max() - dt.timedelta(weeks=HORIZON)
    for emb in (0, 8, 16):
        lab = con.execute(f"""
            SELECT f.cik, f.week, CASE WHEN EXISTS (
              SELECT 1 FROM m.deals d WHERE d.cik=f.cik
                AND f.week <  d.agreement_date - INTERVAL {emb} WEEK
                AND f.week >= d.agreement_date - INTERVAL {emb + HORIZON} WEEK)
              THEN 1 ELSE 0 END AS yh FROM f""").pl()
        j = keys.join(lab, on=["cik", "week"], how="left")
        d = base.with_columns(j["yh"].fill_null(0).cast(pl.Int8).alias("y"))
        tr = d.filter(pl.col("week") < dt.date(2023, 1, 1)).sample(
            fraction=SAMPLE, seed=1)
        va = d.filter((pl.col("week") >= dt.date(2023, 1, 1))
                      & (pl.col("week") < dt.date(2024, 1, 1)))
        te = d.filter((pl.col("week") >= dt.date(2024, 1, 1))
                      & (pl.col("week") <= safe))
        m, sd = seeded(tr, va, te, cols)
        record("embargo", f"{emb}w embargo", mean=m, sd=sd,
               base=float(te["y"].mean() * 100))
        del lab, j, d, tr, va, te
        gc.collect()


def t_overfit():
    tr, va, te, cols = load()
    pr_te = fit_pred(tr, va, te, cols, 11)
    pr_tr = fit_pred(tr, va, tr, cols, 11)
    record("overfit", "train vs test",
           train_prec=screen.weekly_precision(tr, pr_tr, N_EVAL)["precision"] * 100,
           test_prec=screen.weekly_precision(te, pr_te, N_EVAL)["precision"] * 100)


def t_company():
    tr, va, te, cols = load()
    pr = fit_pred(tr, va, te, cols, 11)
    m = te.with_columns(pl.Series("p", pr))
    sel = m.sort("p", descending=True).group_by("week").head(N_EVAL)
    hit = sel.filter(pl.col("y") == 1)["cik"].n_unique()
    tot = te.filter(pl.col("y") == 1)["cik"].n_unique()
    record("company", "company-level",
           row_precision=float(sel["y"].mean() * 100),
           company_precision=100.0 * hit / sel["cik"].n_unique(),
           distinct_flagged=sel["cik"].n_unique(),
           recall=100.0 * hit / tot)


def t_spac():
    import duckdb
    tr, va, te, cols = load()
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    spac = [r[0] for r in con.execute("""
        SELECT DISTINCT u.cik FROM m.universe u
        LEFT JOIN m.company_sic s USING (cik)
        WHERE s.sic='6770' OR upper(u.name) LIKE '%ACQUISITION CORP%'
    """).fetchall()]
    for tag, t2, e2 in [("SPACs in", tr, te),
                        ("SPACs excluded", tr.filter(~pl.col("cik").is_in(spac)),
                         te.filter(~pl.col("cik").is_in(spac)))]:
        m, sd = seeded(t2, va, e2, cols)
        record("spac", tag, mean=m, sd=sd, base=float(e2["y"].mean() * 100))


TESTS = {"baseline": t_baseline, "cv": t_cv, "permutation": t_permutation,
         "ablation": t_ablation, "embargo": t_embargo, "overfit": t_overfit,
         "company": t_company, "spac": t_spac}


def report():
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    cur = None
    for r in rows:
        if r["test"] != cur:
            cur = r["test"]
            print(f"\n=== {cur.upper()} ===")
        rest = {k: v for k, v in r.items() if k not in ("test", "label")}
        print(f"  {r['label']:<34} " + "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in rest.items()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        report()
    else:
        TESTS[cmd]()
