"""Phase 2b: folds, permutation test, nested increment, clustered inference.

Split out from final_validate.py because holding several boosters and their
Datasets alive at once exhausted memory. Each model is released before the
next is built.

    .venv/bin/python scripts/final_stats.py
"""
import datetime as dt
import gc
import json

import duckdb
import numpy as np
import polars as pl
import statsmodels.api as sm

from deal import features, model_gbm, screen

PARQUET = "data/features.parquet"
TEST_START = dt.date(2024, 1, 1)
N_EVAL = 25
HORIZON = 52
FOLD_CUTOFFS = [dt.date(y, 1, 1) for y in (2022, 2023, 2024, 2025)]
N_PERM = 6

CONTROLS = ["log_assets", "log_float", "float_to_assets", "cash_to_assets",
            "leverage", "revenue_growth", "asset_growth", "float_growth",
            "rnd_intensity", "operating_margin", "net_margin",
            "goodwill_to_assets", "intangibles_to_assets", "lt_debt_to_assets",
            "sector_deal_intensity", "age_weeks", "quarter", "year"]


def relabel(df: pl.DataFrame, h: int) -> pl.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    keys = df.select(["cik", "week"])
    con.register("f", keys.to_arrow())
    lab = con.execute(f"""
        SELECT f.cik, f.week, CASE WHEN EXISTS (
                 SELECT 1 FROM m.deals d WHERE d.cik = f.cik
                   AND f.week <  d.agreement_date
                   AND f.week >= d.agreement_date - INTERVAL {h} WEEK
               ) THEN 1 ELSE 0 END AS yh
        FROM f
    """).pl()
    j = keys.join(lab, on=["cik", "week"], how="left")
    con.close()
    return df.with_columns(j["yh"].fill_null(0).cast(pl.Int8).alias("y"))


def score(tr, te, cols):
    """Fit, score, release. Returns (precision, lift)."""
    b = model_gbm.fit(tr, valid=te, cols=cols)
    p = model_gbm.predict(b, te, cols)
    r = screen.weekly_precision(te, p, N_EVAL)
    del b
    gc.collect()
    return r["precision"], r["lift"]


def main() -> None:
    raw = pl.read_parquet(PARQUET)
    usable = [c for c in features.FEATURE_COLS
              if raw[c].std() is not None and raw[c].std() > 0]
    cols = [c for c in usable if not c.endswith("_z")]   # stage-A winner
    df = relabel(raw, HORIZON)
    del raw
    gc.collect()

    tr = df.filter(pl.col("week") < TEST_START)
    te = df.filter(pl.col("week") >= TEST_START)
    print(f"{len(cols)} features | horizon {HORIZON}w | "
          f"test base {te['y'].mean()*100:.2f}%\n", flush=True)

    out = {}

    print("=== 3. ROLLING-ORIGIN FOLDS ===", flush=True)
    fold_p = []
    for c in FOLD_CUTOFFS:
        end = dt.date(c.year + 1, c.month, c.day)
        ftr = df.filter(pl.col("week") < c)
        fte = df.filter((pl.col("week") >= c) & (pl.col("week") < end))
        if not fte.height or not fte["y"].sum():
            continue
        p, l = score(ftr, fte, cols)
        fold_p.append(p)
        print(f"  cutoff {c}: prec@25 {p*100:>6.2f}%  lift {l:>5.2f}x  "
              f"(base {fte['y'].mean()*100:.2f}%)", flush=True)
        del ftr, fte
        gc.collect()
    fold_p = np.array(fold_p)
    out["fold_mean"] = float(fold_p.mean())
    out["fold_sd"] = float(fold_p.std())
    out["fold_cv"] = float(fold_p.std() / fold_p.mean())
    print(f"  MEAN {fold_p.mean()*100:.2f}% (SD {fold_p.std()*100:.2f})  "
          f"SD/mean {out['fold_cv']:.3f}\n", flush=True)

    print("=== 4. PERMUTATION TEST (labels shuffled within week) ===", flush=True)
    real, _ = score(tr, te, cols)
    print(f"  real: {real*100:.2f}%", flush=True)
    null = []
    for k in range(N_PERM):
        sh = tr.with_columns(pl.col("y").shuffle(seed=1000 + k)
                             .over("week").alias("y"))
        p, _ = score(sh, te, cols)
        null.append(p)
        print(f"  null {k+1}: {p*100:.2f}%", flush=True)
        del sh
        gc.collect()
    null = np.array(null)
    out.update({"real": float(real), "null_mean": float(null.mean()),
                "null_max": float(null.max()),
                "p_value": float((np.sum(null >= real) + 1) / (N_PERM + 1)),
                "beats_all_null": bool(real > null.max())})
    print(f"  null mean {null.mean()*100:.2f}%  max {null.max()*100:.2f}%  "
          f"p={out['p_value']:.4f}  beats all: {out['beats_all_null']}\n",
          flush=True)

    print("=== 5. NESTED: controls vs controls+novel ===", flush=True)
    ctl = [c for c in CONTROLS if c in cols]
    pc, lc = score(tr, te, ctl)
    out["controls_only"] = float(pc)
    print(f"  controls only     {pc*100:>6.2f}%  lift {lc:>5.2f}x", flush=True)
    print(f"  controls + novel  {real*100:>6.2f}%", flush=True)
    print(f"  increment {(real-pc)*100:+.2f}pp "
          f"({100*(real-pc)/pc:+.1f}% relative)\n", flush=True)

    print("=== 6. HAZARD MODEL, SEs CLUSTERED BY COMPANY ===", flush=True)
    print("  (naive SEs measured ~4x too small on this panel)", flush=True)
    sub = tr.sample(fraction=0.3, seed=7)
    X = sm.add_constant(sub.select(cols).to_pandas())
    fit = sm.Logit(sub["y"].to_pandas(), X).fit(
        disp=False, maxiter=300, cov_type="cluster",
        cov_kwds={"groups": sub["cik"].to_pandas()})
    tv = fit.tvalues.drop("const")
    novel_sig = [n for n in tv.index if n not in CONTROLS and abs(tv[n]) > 1.96]
    out["novel_significant"] = len(novel_sig)
    print(f"  novel signals significant at p<0.05: {len(novel_sig)}", flush=True)
    for n in sorted(novel_sig, key=lambda x: -abs(tv[x]))[:12]:
        print(f"    {n:<24} beta {fit.params[n]:+.4f}  z {tv[n]:+7.2f}",
              flush=True)

    json.dump(out, open("data/final_stats.json", "w"), indent=2)
    print("\nwrote data/final_stats.json", flush=True)


if __name__ == "__main__":
    main()
