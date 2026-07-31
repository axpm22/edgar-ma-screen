"""Phase 2: repeated evaluation and statistical tests on the HELD-OUT period.

The winning config comes from search_config.py, which never read 2024+. This
script touches it once.

Everything is reported as precision AND lift. Precision alone is not
comparable across horizons -- a 52-week label has roughly twice the positive
rate of a 26-week one, so its precision is mechanically higher without the
model being any better. Lift is the horizon-invariant number.

    .venv/bin/python scripts/final_validate.py [horizon]
"""
import datetime as dt
import json
import sys

import duckdb
import numpy as np
import polars as pl
import statsmodels.api as sm

from deal import evaluate_robust as er
from deal import features, model_gbm, screen, splits

PARQUET = "data/features.parquet"
TEST_START = dt.date(2024, 1, 1)
N_EVAL = 25
SEEDS = [1, 2, 3, 4, 5]
FOLD_CUTOFFS = [dt.date(y, 1, 1) for y in (2021, 2022, 2023, 2024, 2025)]

CONTROLS = ["log_assets", "log_float", "float_to_assets", "cash_to_assets",
            "leverage", "revenue_growth", "asset_growth", "float_growth",
            "rnd_intensity", "operating_margin", "net_margin",
            "goodwill_to_assets", "intangibles_to_assets", "lt_debt_to_assets",
            "sector_deal_intensity", "age_weeks", "quarter", "year"]


def relabel(df: pl.DataFrame, horizon_weeks: int) -> pl.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    keys = df.select(["cik", "week"])
    con.register("f", keys.to_arrow())
    lab = con.execute(f"""
        SELECT f.cik, f.week, CASE WHEN EXISTS (
                 SELECT 1 FROM m.deals d WHERE d.cik = f.cik
                   AND f.week <  d.agreement_date
                   AND f.week >= d.agreement_date - INTERVAL {horizon_weeks} WEEK
               ) THEN 1 ELSE 0 END AS yh
        FROM f
    """).pl()
    j = keys.join(lab, on=["cik", "week"], how="left")
    return df.with_columns(j["yh"].fill_null(0).cast(pl.Int8).alias("y"))


def _pl(df, p, n=N_EVAL):
    r = screen.weekly_precision(df, p, n)
    return r["precision"], r["lift"]


def main() -> None:
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 52
    try:
        winner = json.load(open("data/search_results.json"))["winner"]
        horizon = winner.get("horizon", horizon)
    except Exception:
        winner = {}
    print(f"winner config: {winner or 'defaults'}")
    print(f"horizon: {horizon} weeks\n")

    raw = pl.read_parquet(PARQUET)
    usable = [c for c in features.FEATURE_COLS
              if raw[c].std() is not None and raw[c].std() > 0]
    if winner.get("features") == "no_z":
        cols = [c for c in usable if not c.endswith("_z")]
    elif winner.get("features") == "no_deltas":
        cols = [c for c in usable if not c.endswith("_d")]
    elif winner.get("features") == "novel_only":
        cols = [c for c in usable if c not in CONTROLS]
    else:
        cols = usable
    if "num_leaves" in winner:
        model_gbm.PARAMS.update({
            k: winner[k] for k in
            ("num_leaves", "min_data_in_leaf", "learning_rate")
            if k in winner})
    recency = winner.get("recency_hl") is not None
    if recency:
        model_gbm.RECENCY_HALF_LIFE_YEARS = float(winner["recency_hl"])

    df = relabel(raw, horizon)
    tr = df.filter(pl.col("week") < TEST_START)
    te = df.filter(pl.col("week") >= TEST_START)
    print(f"{len(cols)} features | train {tr.height:,} ({int(tr['y'].sum()):,} pos) "
          f"| test {te.height:,} ({int(te['y'].sum()):,} pos, base "
          f"{te['y'].mean()*100:.2f}%)\n")

    # ---- 1. Held-out test, averaged over seeds -----------------------------
    print("=== 1. HELD-OUT TEST (2024+), across seeds ===")
    precs, lifts = [], []
    for s in SEEDS:
        model_gbm.PARAMS["seed"] = s
        model_gbm.PARAMS["bagging_seed"] = s
        model_gbm.PARAMS["feature_fraction_seed"] = s
        b = model_gbm.fit(tr, valid=te, cols=cols, recency=recency)
        p, l = _pl(te, model_gbm.predict(b, te, cols))
        precs.append(p); lifts.append(l)
        print(f"  seed {s}: prec@25 {p*100:>6.2f}%  lift {l:>5.2f}x")
    precs, lifts = np.array(precs), np.array(lifts)
    print(f"  MEAN prec@25 {precs.mean()*100:.2f}% (SD {precs.std()*100:.2f}) "
          f"| lift {lifts.mean():.2f}x")
    print(f"  seed SD / mean = {precs.std()/precs.mean():.3f}"
          f"  {'STABLE' if precs.std()/precs.mean() < 0.33 else 'UNSTABLE'}\n")

    # ---- 2. Bootstrap CI ---------------------------------------------------
    print("=== 2. BOOTSTRAP CI (resampling weeks) ===")
    model_gbm.PARAMS["seed"] = 1
    b = model_gbm.fit(tr, valid=te, cols=cols, recency=recency)
    p_hat = model_gbm.predict(b, te, cols)
    bs = er.bootstrap_precision(te, p_hat, N_EVAL, n_boot=400)
    print(f"  prec@25 {bs['precision']*100:.2f}%  "
          f"95% CI [{bs['ci_lo']*100:.2f}%, {bs['ci_hi']*100:.2f}%]  "
          f"over {bs['n_weeks']} weeks\n")

    # ---- 3. Rolling-origin folds ------------------------------------------
    print("=== 3. ROLLING-ORIGIN FOLDS (each trains on all prior data) ===")
    fold_p, fold_l = [], []
    for cutoff, ftr, fte in er.rolling_origin(df, FOLD_CUTOFFS):
        bb = model_gbm.fit(ftr, valid=fte, cols=cols, recency=recency)
        p, l = _pl(fte, model_gbm.predict(bb, fte, cols))
        fold_p.append(p); fold_l.append(l)
        print(f"  cutoff {cutoff}: prec@25 {p*100:>6.2f}%  lift {l:>5.2f}x  "
              f"(base {fte['y'].mean()*100:.2f}%)")
    fold_p = np.array(fold_p)
    print(f"  MEAN {fold_p.mean()*100:.2f}% (SD {fold_p.std()*100:.2f})  "
          f"SD/mean {fold_p.std()/fold_p.mean():.3f}\n")

    # ---- 4. Permutation test ----------------------------------------------
    print("=== 4. PERMUTATION TEST (labels shuffled within week) ===")
    pt = er.permutation_test(tr, te, cols, N_EVAL, n_perm=10, recency=recency)
    print(f"  real {pt['real']*100:.2f}%  |  null mean {pt['null_mean']*100:.2f}% "
          f"max {pt['null_max']*100:.2f}% sd {pt['null_sd']*100:.2f}")
    print(f"  p = {pt['p_value']:.4f}  beats every null draw: {pt['beats_all_null']}\n")

    # ---- 5. Nested increment ---------------------------------------------
    print("=== 5. NESTED: controls vs controls+novel ===")
    ctl = [c for c in CONTROLS if c in cols]
    bc = model_gbm.fit(tr, valid=te, cols=ctl, recency=recency)
    pc, lc = _pl(te, model_gbm.predict(bc, te, ctl))
    print(f"  controls only     prec@25 {pc*100:>6.2f}%  lift {lc:>5.2f}x")
    print(f"  controls + novel  prec@25 {precs.mean()*100:>6.2f}%  "
          f"lift {lifts.mean():>5.2f}x")
    print(f"  increment: {(precs.mean()-pc)*100:+.2f}pp "
          f"({100*(precs.mean()-pc)/pc:+.1f}% relative)\n")

    # ---- 6. Clustered-SE hazard model ------------------------------------
    print("=== 6. HAZARD MODEL, SEs CLUSTERED BY COMPANY ===")
    print("  (naive SEs measured ~4x too small on this panel)")
    sub = tr.sample(fraction=0.35, seed=7) if tr.height > 1_400_000 else tr
    X = sm.add_constant(sub.select(cols).to_pandas())
    fit = sm.Logit(sub["y"].to_pandas(), X).fit(
        disp=False, maxiter=200, cov_type="cluster",
        cov_kwds={"groups": sub["cik"].to_pandas()})
    tv = fit.tvalues.drop("const")
    novel_sig = [n for n in tv.index
                 if n not in CONTROLS and abs(tv[n]) > 1.96]
    print(f"  novel signals significant at p<0.05: {len(novel_sig)}")
    for n in sorted(novel_sig, key=lambda x: -abs(tv[x]))[:12]:
        print(f"    {n:<24} beta {fit.params[n]:+.4f}  z {tv[n]:+7.2f}")

    # ---- verdict ---------------------------------------------------------
    print("\n=== VERDICT vs pre-declared criteria ===")
    checks = [
        (f"prec@25 >= 20%", precs.mean() >= 0.20, f"{precs.mean()*100:.2f}%"),
        ("bootstrap CI lower bound >= 15%", bs["ci_lo"] >= 0.15,
         f"{bs['ci_lo']*100:.2f}%"),
        ("fold SD/mean < 0.33", fold_p.std()/fold_p.mean() < 0.33,
         f"{fold_p.std()/fold_p.mean():.3f}"),
        ("beats every permutation draw", pt["beats_all_null"],
         f"p={pt['p_value']:.4f}"),
        ("3+ novel signals significant", len(novel_sig) >= 3,
         f"{len(novel_sig)}"),
    ]
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<36} {val}")
    print(f"\n{sum(c[1] for c in checks)}/{len(checks)} criteria met")


if __name__ == "__main__":
    main()
