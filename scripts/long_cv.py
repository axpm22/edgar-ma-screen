"""Seven test years instead of three: is the model regime-dependent, or was
three years just too few to tell?

Three test years is not a data limit, it is a choice. The panel runs 2016-2026
and every year from 2016 on carries 114-218 verified targets. What genuinely
rules out early years is WARM-UP, not volume:

  public float      0% of rows populated in 2016, 47% in 2017, 66% from 2018
  activist + peer  30% in 2016, then ~62-68%
  form counts      88% in 2016, reaching 98% only by 2022

Rolling 52- and 104-week features cannot be valid before the panel has that
much history behind them, so the earliest honest test year is 2019.

Expanding-window CV then has a defect of its own: test 2019 trains on ~1.5
usable years while test 2025 trains on ~6.5. Differences across folds confound
REGIME with TRAINING-SET SIZE. So both designs are run:

  expanding  train on everything before the validation year (what the project
             does now) -- more data late, confounded
  fixed      always train on FIXED_YEARS years before validation -- less data
             late, but every fold is comparable

If the spread stays wide under the fixed window, it is genuinely regime. If it
narrows, part of what looked like regime dependence was sample size.

    .venv/bin/python scripts/long_cv.py        # 28 fits, ~6 min

One process, one booster alive at a time.
"""
import datetime as dt
import gc
import json
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL  # noqa: E402
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks  # noqa: E402

from deal import clean_labels, features, screen  # noqa: E402

YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
FIXED_YEARS = 3          # fold-comparable training window
OUT = "data/long_cv.json"


def verified_frame():
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
    con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
    clean_labels.build(con, raw["week"].max())
    con.register("f", raw.select(["cik", "week"]).to_arrow())
    lab = con.execute(f"""
        SELECT f.cik, f.week, CASE WHEN EXISTS(
            SELECT 1 FROM deals_clean d WHERE d.cik = f.cik
              AND d.outcome = 'target'
              AND f.week <  d.agreement_date
              AND f.week >= d.agreement_date - INTERVAL {HORIZON} WEEK)
            THEN 1 ELSE 0 END AS yh FROM f""").pl()
    j = raw.select(["cik", "week"]).join(lab, on=["cik", "week"], how="left")
    df = (raw.with_columns(j["yh"].fill_null(0).cast(pl.Int8).alias("y"))
          .select(["cik", "week", "y"] + cols)
          .filter(~pl.col("cik").is_in(spac_ciks())))
    del raw, lab, j
    gc.collect()
    return df, cols


def run(tr, va, te, cols):
    vals, hits = [], []
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
        hits.append(r["distinct_hits"])
        del b, dtr, dva, pr
        gc.collect()
    return float(np.mean(vals)), float(np.std(vals)), int(np.mean(hits))


def main():
    df, cols = verified_frame()
    print(f"{df.height:,} rows | {len(cols)} features | "
          f"label rate {df['y'].mean()*100:.2f}%\n", flush=True)

    out = []
    for design in ("expanding", "fixed"):
        print(f"=== {design.upper()} WINDOW"
              f"{f' ({FIXED_YEARS}y)' if design == 'fixed' else ''} ===",
              flush=True)
        for yr in YEARS:
            frm = (dt.date(yr - 1 - FIXED_YEARS, 1, 1)
                   if design == "fixed" else None)
            tr, va, te = split(df, yr, train_from=frm)
            if not te.height or not te["y"].sum() or tr.height < 50_000:
                print(f"  {yr}  skipped (train rows {tr.height:,})", flush=True)
                continue
            m, sd, dh = run(tr, va, te, cols)
            base = float(te["y"].mean() * 100)
            out.append({"design": design, "year": yr, "prec": m, "sd": sd,
                        "base": base, "lift": m / base if base else 0.0,
                        "distinct_hits": dh, "train_rows": tr.height,
                        "test_rows": te.height})
            print(f"  {yr}  {m:>6.2f}% +/-{sd:.2f}  lift {m/base:>5.2f}x  "
                  f"base {base:.2f}%  hits from {dh} companies  "
                  f"train {tr.height/1000:.0f}k", flush=True)
            del tr, va, te
            gc.collect()
        v = np.array([r["prec"] for r in out if r["design"] == design])
        if len(v):
            print(f"  MEAN {v.mean():.2f}%  SD {v.std():.2f}  "
                  f"range {v.min():.1f}-{v.max():.1f}\n", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    for design in ("expanding", "fixed"):
        v = np.array([r["prec"] for r in out if r["design"] == design])
        if len(v) > 1:
            print(f"{design:<10} n={len(v)}  mean {v.mean():.2f}%  "
                  f"SD {v.std():.2f}  SD/mean {v.std()/v.mean():.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
