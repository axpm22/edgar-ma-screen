"""Rebuild labels as genuine targets only, then re-run the CV."""
import datetime as dt, gc, json, sys
import duckdb, lightgbm as lgb, numpy as np, polars as pl
sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks
from deal import features, screen, clean_labels

raw = pl.read_parquet("data/features.parquet")
cols = [c for c in features.FEATURE_COLS
        if raw[c].std() is not None and raw[c].std() > 0]
panel_end = raw["week"].max()

con = duckdb.connect(":memory:")
con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
counts = clean_labels.build(con, panel_end)
print("proxy filings classified:", counts, flush=True)
targets = clean_labels.target_ciks(con)
print(f"genuine targets: {len(targets):,} companies\n", flush=True)

# Relabel: positive only if a CONFIRMED target has a deal within the horizon.
con.register("f", raw.select(["cik", "week"]).to_arrow())
lab = con.execute(f"""
    SELECT f.cik, f.week, CASE WHEN EXISTS (
        SELECT 1 FROM deals_clean d
        WHERE d.cik = f.cik AND d.outcome = 'target'
          AND f.week <  d.agreement_date
          AND f.week >= d.agreement_date - INTERVAL {HORIZON} WEEK
    ) THEN 1 ELSE 0 END AS yh FROM f
""").pl()
j = raw.select(["cik", "week"]).join(lab, on=["cik", "week"], how="left")
df = raw.with_columns(j["yh"].fill_null(0).cast(pl.Int8).alias("y")) \
        .select(["cik", "week", "y"] + cols)
del raw; gc.collect()

spac = spac_ciks()
for uni in ("all", "nospac"):
    d = df if uni == "all" else df.filter(~pl.col("cik").is_in(spac))
    out = []
    for yr in (2023, 2024, 2025):
        tr, va, te = split(d, yr)
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
            del b, dtr, dva, pr; gc.collect()
        out.append((yr, float(np.mean(vals)), float(np.std(vals)),
                    float(np.mean(lifts)), float(te["y"].mean() * 100)))
        print(f"  {uni:<7} {yr}  {out[-1][1]:>6.2f}% +/-{out[-1][2]:.2f}  "
              f"lift {out[-1][3]:.2f}x  base {out[-1][4]:.2f}%", flush=True)
        del tr, va, te; gc.collect()
    if out:
        print(f"  MEAN {uni:<7} {np.mean([o[1] for o in out]):>6.2f}%  "
              f"lift {np.mean([o[3] for o in out]):.2f}x\n", flush=True)
    json.dump(out, open(f"data/clean_{uni}.json", "w"), indent=1)
