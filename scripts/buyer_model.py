"""Buyer model, built and stress-tested like the target model.

    python scripts/buyer_model.py build     # buyer features -> parquet
    python scripts/buyer_model.py cv        # CV, with and without self-ref
    python scripts/buyer_model.py ablate    # which families are signal?
"""
import datetime as dt, gc, json, sys
import duckdb, lightgbm as lgb, numpy as np, polars as pl
sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks
from deal import features, screen, feat_buyer

OUT = "data/buyer_features.parquet"


def _label_and_features():
    raw = pl.read_parquet("data/features.parquet")
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/forms2.duckdb' AS f2 (READ_ONLY)")
    con.execute("CREATE TEMP VIEW form_events AS SELECT * FROM f2.form_events")
    feat_buyer.prepare(con)
    con.register("p", raw.select(["cik", "week"]).to_arrow())
    # ASOF join keeps the firewall: only rolls known on or before the week.
    br = con.execute("""
        SELECT p.cik, p.week,
               coalesce(b.shelf_52w,0) AS shelf_52w,
               coalesce(b.raise_52w,0) AS raise_52w,
               coalesce(b.shelf_new,0) AS shelf_new,
               coalesce(b.raise_burst,0) AS raise_burst
        FROM p ASOF LEFT JOIN buyer_roll b
          ON p.cik = b.cik AND p.week >= b.week
    """).pl()
    lab = con.execute(f"""
        SELECT p.cik, p.week, CASE WHEN EXISTS (
            SELECT 1 FROM form_events a WHERE a.cik=p.cik AND a.family='s4'
              AND a.public_ts >  p.week
              AND a.public_ts <= p.week + INTERVAL {HORIZON} WEEK
        ) THEN 1 ELSE 0 END AS yh FROM p
    """).pl()
    keys = raw.select(["cik", "week"])
    df = raw.with_columns(
        keys.join(lab, on=["cik", "week"], how="left")["yh"]
            .fill_null(0).cast(pl.Int8).alias("y"))
    df = df.join(br, on=["cik", "week"], how="left")
    df = feat_buyer.add(df)
    del raw; gc.collect()
    return df


def stage_build():
    df = _label_and_features()
    df.write_parquet(OUT)
    print(f"{df.height:,} rows | buyer label rate {df['y'].mean()*100:.2f}%")
    for c in feat_buyer.BUYER_COLS:
        nz = 100.0 * (df[c] != 0).sum() / df.height
        print(f"  {c:<16} nonzero {nz:>6.2f}%  mean {df[c].mean():>10.4f}")


def _cols(df):
    base = [c for c in features.FEATURE_COLS
            if c in df.columns and df[c].std() and df[c].std() > 0]
    return base + [c for c in feat_buyer.BUYER_COLS
                   if df[c].std() and df[c].std() > 0]


def _run(df, cols, tag):
    res = []
    for yr in (2023, 2024, 2025):
        tr, va, te = split(df, yr)
        if not te.height or not te["y"].sum():
            continue
        v, l = [], []
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
            v.append(r["precision"] * 100); l.append(r["lift"])
            del b, dtr, dva, pr; gc.collect()
        res.append((np.mean(v), np.mean(l)))
        del tr, va, te; gc.collect()
    m, lf = np.mean([r[0] for r in res]), np.mean([r[1] for r in res])
    print(f"  {tag:<34} {m:>6.2f}%  lift {lf:>5.2f}x   n={len(cols)}", flush=True)
    return m, lf


def stage_cv():
    df = pl.read_parquet(OUT).filter(~pl.col("cik").is_in(spac_ciks()))
    cols = _cols(df)
    honest = [c for c in cols if c not in set(feat_buyer.SELF_REFERENTIAL)]
    print("=== BUYER MODEL (SPACs excluded) ===")
    _run(df, cols, "all features")
    _run(df, honest, "WITHOUT self-referential")
    _run(df, [c for c in honest if c not in set(feat_buyer.BUYER_COLS)],
         "  ...and without new buyer feats")


def stage_ablate():
    df = pl.read_parquet(OUT).filter(~pl.col("cik").is_in(spac_ciks()))
    cols = [c for c in _cols(df) if c not in set(feat_buyer.SELF_REFERENTIAL)]
    base, _ = _run(df, cols, "BASELINE (honest features)")
    print()
    fams = {
        "new buyer feats": feat_buyer.BUYER_COLS,
        "shelf/raise only": ["shelf_52w", "raise_52w", "shelf_new",
                             "raise_burst"],
        "capacity only": ["dry_powder", "debt_headroom", "acq_capacity"],
        "form counts": features.FORM_COLS,
        "insider": features.INSIDER_COLS,
        "market": features.MARKET_COLS,
        "peer/activist": features.PEER_COLS + features.ACTIVIST_COLS,
    }
    for name, block in fams.items():
        keep = [c for c in cols if c not in set(block)]
        if len(keep) == len(cols):
            continue
        m, _ = _run(df, keep, f"without {name}")
        d = base - m
        print(f"      contributes {d:+.2f}pp"
              f"{'   SIGNAL' if abs(d) > 2 else '   (noise)'}", flush=True)


if __name__ == "__main__":
    {"build": stage_build, "cv": stage_cv, "ablate": stage_ablate}[sys.argv[1]]()
