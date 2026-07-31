"""Feature selection + time-series CV, on both universes.

The full 72-feature set is mostly ballast: leave-one-family-out showed only the
EDGAR form counts clearly clearing the noise bar, while market features scored
negative. This trims to candidate subsets and cross-validates each across three
test years, separately for all companies and for operating companies only
(SPACs excluded), because those are different problems with different answers.

One stage per process invocation -- LightGBM's arena memory is not reliably
returned to the OS, and running every stage in one process was killed twice.

    python scripts/select_cv.py sets      # subsets x 3 test years
    python scripts/select_cv.py nospac    # best subsets, SPACs excluded
    python scripts/select_cv.py window    # does older training data still help?
    python scripts/select_cv.py report
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
from deal.feat_items import ITEM_COLS  # noqa: E402
from deal.feat_literature import LIT_COLS  # noqa: E402

OUT = Path("data/select_cv.json")
SAMPLE = 0.25
ROUNDS = 200
SEEDS = (11, 22)
YEARS = (2023, 2024, 2025)

PARAMS = {
    "objective": "binary", "metric": "average_precision",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbosity": -1, "num_threads": 2,
}


def subsets(cols):
    """Candidate sets, ordered by how much the ablation said each family gave."""
    ctx = features.CONTEXT_COLS
    forms = features.FORM_COLS
    actpeer = features.ACTIVIST_COLS + features.PEER_COLS
    drop_neg = set(features.MARKET_COLS + features.INSIDER_COLS + LIT_COLS)
    core = set(forms + ITEM_COLS + actpeer + features.ZSCORE_COLS
               + features.DELTA_COLS + ctx + features.FTS_COLS)
    return {
        "A_all": cols,
        "B_no_market": [c for c in cols if c not in set(features.MARKET_COLS)],
        "C_drop_negatives": [c for c in cols if c not in drop_neg],
        "D_core": [c for c in cols if c in core],
        "E_minimal": [c for c in cols if c in set(forms + actpeer + ctx)],
    }


def record(**kw):
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    rows.append(kw)
    OUT.write_text(json.dumps(rows, indent=1, default=str))
    print("  " + "  ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in kw.items()), flush=True)


def load_all():
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    df = relabel(raw, HORIZON).select(["cik", "week", "y"] + cols)
    del raw
    gc.collect()
    return df, cols


def split(df, test_year, train_from=None):
    safe = df["week"].max() - dt.timedelta(weeks=HORIZON)
    va0, te0 = dt.date(test_year - 1, 1, 1), dt.date(test_year, 1, 1)
    te1 = min(dt.date(test_year, 12, 31), safe)
    tr = df.filter(pl.col("week") < va0)
    if train_from:
        tr = tr.filter(pl.col("week") >= train_from)
    return (tr.sample(fraction=SAMPLE, seed=1),
            df.filter((pl.col("week") >= va0) & (pl.col("week") < te0)),
            df.filter((pl.col("week") >= te0) & (pl.col("week") <= te1)))


def seeded(tr, va, te, cols):
    vals, lifts = [], []
    for s in SEEDS:
        p = {**PARAMS, "bagging_seed": s, "feature_fraction_seed": s,
             "data_random_seed": s}
        dtr = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                          label=tr["y"].to_pandas())
        dva = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                          label=va["y"].to_pandas())
        b = lgb.train(p, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        pr = np.asarray(b.predict(te.select(cols).to_pandas().astype("float32")))
        r = screen.weekly_precision(te, pr, N_EVAL)
        vals.append(r["precision"] * 100)
        lifts.append(r["lift"])
        del b, dtr, dva, pr
        gc.collect()
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(lifts))


def spac_ciks():
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    out = [r[0] for r in con.execute("""
        SELECT DISTINCT u.cik FROM m.universe u
        LEFT JOIN m.company_sic s USING (cik)
        WHERE s.sic='6770' OR upper(u.name) LIKE '%ACQUISITION CORP%'
    """).fetchall()]
    con.close()
    return out


def stage_sets():
    df, cols = load_all()
    for name, cs in subsets(cols).items():
        for yr in YEARS:
            tr, va, te = split(df, yr)
            if not te.height or not te["y"].sum():
                continue
            m, sd, lift = seeded(tr, va, te, cs)
            record(stage="sets", subset=name, n=len(cs), year=yr,
                   prec=m, sd=sd, lift=lift)
            del tr, va, te
            gc.collect()


def stage_nospac():
    df, cols = load_all()
    spac = spac_ciks()
    df = df.filter(~pl.col("cik").is_in(spac))
    gc.collect()
    for name in ("A_all", "C_drop_negatives", "D_core", "E_minimal"):
        cs = subsets(cols)[name]
        for yr in YEARS:
            tr, va, te = split(df, yr)
            if not te.height or not te["y"].sum():
                continue
            m, sd, lift = seeded(tr, va, te, cs)
            record(stage="nospac", subset=name, n=len(cs), year=yr,
                   prec=m, sd=sd, lift=lift, base=float(te["y"].mean() * 100))
            del tr, va, te
            gc.collect()


def stage_window():
    """Is older data still useful, or has the market moved on?"""
    df, cols = load_all()
    cs = subsets(cols)["C_drop_negatives"]
    for label, frm in [("2016+ (all history)", None),
                       ("2019+ only", dt.date(2019, 1, 1)),
                       ("2021+ only", dt.date(2021, 1, 1))]:
        tr, va, te = split(df, 2024, train_from=frm)
        m, sd, lift = seeded(tr, va, te, cs)
        record(stage="window", train_from=label, rows=tr.height,
               prec=m, sd=sd, lift=lift)
        del tr, va, te
        gc.collect()


def report():
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for stage in ("sets", "nospac", "window"):
        rs = [r for r in rows if r["stage"] == stage]
        if not rs:
            continue
        print(f"\n=== {stage.upper()} ===")
        if stage == "window":
            for r in rs:
                print(f"  {r['train_from']:<22} {r['prec']:>6.2f}% +/-{r['sd']:.2f}"
                      f"  lift {r['lift']:.2f}x  rows={r['rows']:,}")
            continue
        by = {}
        for r in rs:
            by.setdefault(r["subset"], []).append(r)
        print(f"  {'subset':<20}{'n':>4}  " +
              "".join(f"{y:>9}" for y in YEARS) + "     MEAN")
        for name, group in by.items():
            d = {g["year"]: g for g in group}
            cells = "".join(f"{d[y]['prec']:>8.2f}%" if y in d else f"{'-':>9}"
                            for y in YEARS)
            mean = np.mean([g["prec"] for g in group])
            print(f"  {name:<20}{group[0]['n']:>4}  {cells}  {mean:>8.2f}%")


def stage_best():
    """Confirm the winning combination: all features, training from 2019.

    Trimming turned out to HURT the operating-company model badly, and the
    window test showed 2019+ beats all-history by ~6pp -- older data really is
    from a different market.
    """
    df, cols = load_all()
    spac = spac_ciks()
    for universe, d in (("all", df), ("no_spac", df.filter(~pl.col("cik").is_in(spac)))):
        for yr in YEARS:
            tr, va, te = split(d, yr, train_from=dt.date(2019, 1, 1))
            if not te.height or not te["y"].sum():
                continue
            m, sd, lift = seeded(tr, va, te, cols)
            record(stage="best", universe=universe, year=yr, prec=m, sd=sd,
                   lift=lift, base=float(te["y"].mean() * 100))
            del tr, va, te
            gc.collect()


if __name__ == "__main__":
    {"sets": stage_sets, "nospac": stage_nospac, "window": stage_window,
     "best": stage_best, "report": report}[sys.argv[1]]()


