"""Acquirer-side model: who will BUY in the next 12 months?

Label = files an S-4 within the horizon. An S-4 registers securities issued
to pay for an acquisition, so the filer is the buyer. It is noisier than the
target label -- S-4 also covers debt exchange offers -- so the serial-acquirer
variant (>=2 lifetime S-4s) is reported alongside it.
"""
import datetime as dt, gc, json, sys
import duckdb, lightgbm as lgb, numpy as np, polars as pl
sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks
from deal import features, screen

raw = pl.read_parquet("data/features.parquet")
cols = [c for c in features.FEATURE_COLS
        if raw[c].std() is not None and raw[c].std() > 0]
con = duckdb.connect(":memory:")
con.execute("ATTACH 'data/forms.duckdb' AS fm (READ_ONLY)")
con.register("f", raw.select(["cik", "week"]).to_arrow())

for variant, having in (("any S-4", 1), ("serial acquirer (2+ S-4s)", 2)):
    lab = con.execute(f"""
        WITH acq AS (
            SELECT cik, public_ts FROM fm.form_events WHERE family='s4'
              AND cik IN (SELECT cik FROM fm.form_events WHERE family='s4'
                          GROUP BY cik HAVING count(*) >= {having})
        )
        SELECT f.cik, f.week, CASE WHEN EXISTS (
            SELECT 1 FROM acq a WHERE a.cik = f.cik
              AND a.public_ts >  f.week
              AND a.public_ts <= f.week + INTERVAL {HORIZON} WEEK
        ) THEN 1 ELSE 0 END AS yh FROM f
    """).pl()
    j = raw.select(["cik", "week"]).join(lab, on=["cik", "week"], how="left")
    df = raw.with_columns(j["yh"].fill_null(0).cast(pl.Int8).alias("y")) \
            .select(["cik", "week", "y"] + cols)
    spac = spac_ciks()
    df = df.filter(~pl.col("cik").is_in(spac))     # de-SPACs are not real buyers
    gc.collect()
    print(f"=== ACQUIRER: {variant} (SPACs excluded) ===", flush=True)
    out = []
    for yr in (2023, 2024, 2025):
        tr, va, te = split(df, yr)
        if not te.height or not te["y"].sum():
            continue
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
            vals.append(r["precision"] * 100); lifts.append(r["lift"])
            if s == SEEDS[0] and yr == 2024:
                imp = sorted(zip(b.feature_name(), b.feature_importance("gain")),
                             key=lambda t: -t[1])[:6]
                top = ", ".join(n for n, _ in imp)
            del b, dtr, dva, pr; gc.collect()
        out.append((yr, float(np.mean(vals)), float(np.mean(lifts)),
                    float(te["y"].mean() * 100)))
        print(f"  {yr}  {out[-1][1]:>6.2f}% +/-{np.std(vals):.2f}  "
              f"lift {out[-1][2]:.2f}x  base {out[-1][3]:.2f}%", flush=True)
        del tr, va, te; gc.collect()
    if out:
        print(f"  MEAN  {np.mean([o[1] for o in out]):>6.2f}%  "
              f"lift {np.mean([o[2] for o in out]):.2f}x", flush=True)
        print(f"  top by gain: {top}\n", flush=True)
    json.dump(out, open(f"data/acq_{having}.json", "w"), indent=1)
