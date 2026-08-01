"""The four stress tests the handoff lists as never run. ONE per invocation.

    .venv/bin/python scripts/stress_pairs.py buyerperm
    .venv/bin/python scripts/stress_pairs.py hazard
    .venv/bin/python scripts/stress_pairs.py size
    .venv/bin/python scripts/stress_pairs.py tender
    .venv/bin/python scripts/stress_pairs.py report

Each appends to data/stress_pairs.json and exits, because LightGBM's arena
memory is not reliably returned to the OS and running several fits' worth of
stages in one process has been killed twice.
"""
import datetime as dt
import gc
import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL, relabel  # noqa: E402
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks  # noqa: E402

from deal import clean_labels, feat_buyer, features, screen  # noqa: E402

RESULTS = Path("data/stress_pairs.json")
YEARS = (2023, 2024, 2025)


def record(test: str, label: str, **kw) -> None:
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    rows.append({"test": test, "label": label, **kw})
    RESULTS.write_text(json.dumps(rows, indent=1, default=str))
    print(f"  {label:<38} " + "  ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in kw.items()), flush=True)


def fit_pred(tr, va, te, cols, seed):
    p = {**PARAMS, "bagging_seed": seed, "feature_fraction_seed": seed,
         "data_random_seed": seed}
    dtr = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                      label=tr["y"].to_pandas())
    dva = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                      label=va["y"].to_pandas())
    b = lgb.train(p, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    out = np.asarray(b.predict(te.select(cols).to_pandas().astype("float32")))
    del b, dtr, dva
    gc.collect()
    return out


def seeded(tr, va, te, cols):
    vals = [screen.weekly_precision(te, fit_pred(tr, va, te, cols, s),
                                    N_EVAL)["precision"] * 100 for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))


def buyer_frame():
    """Buyer features, SPACs out, self-referential columns out."""
    df = pl.read_parquet("data/buyer_features.parquet").filter(
        ~pl.col("cik").is_in(spac_ciks()))
    self_ref = set(feat_buyer.SELF_REFERENTIAL)
    base = [c for c in features.FEATURE_COLS
            if c in df.columns and df[c].std() and df[c].std() > 0]
    extra = [c for c in feat_buyer.BUYER_COLS
             if df[c].std() and df[c].std() > 0]
    cols = [c for c in base + extra if c not in self_ref]
    return df.select(["cik", "week", "y"] + cols), cols


# --------------------------------------------------------------------------- #

def t_buyerperm():
    """Permutation test on the BUYER model. It has never had one.

    Labels shuffle within week, so the null keeps each week's positive count
    and only destroys the feature-label link. 8 fits.
    """
    df, cols = buyer_frame()
    tr, va, te = split(df, 2024)
    real, sd = seeded(tr, va, te, cols)
    record("buyerperm", "buyer real (2024)", real=real, sd=sd,
           base=float(te["y"].mean() * 100), n_feat=len(cols))
    null = []
    for k in range(6):
        sh = tr.with_columns(
            pl.col("y").shuffle(seed=900 + k).over("week").alias("y"))
        p = fit_pred(sh, va, te, cols, 11)
        null.append(screen.weekly_precision(te, p, N_EVAL)["precision"] * 100)
        print(f"    null {k + 1}: {null[-1]:.2f}%", flush=True)
        del sh, p
        gc.collect()
    a = np.array(null)
    record("buyerperm", "buyer null distribution", null_mean=float(a.mean()),
           null_max=float(a.max()), null_sd=float(a.std()),
           p_value=float((np.sum(a >= real) + 1) / (len(a) + 1)),
           beats_all=bool(real > a.max()))


def t_hazard():
    """Clustered-SE hazard model on VERIFIED-TARGET labels.

    The existing inference -- including the ROA-vs-Palepu finding -- was
    computed on raw proxy-filer labels, which contain 581 acquirers and
    survivors. Whether those coefficients survive the clean label is unknown,
    and a coefficient that flips sign is a real finding either way.

    No LightGBM fits; one clustered logit.
    """
    import statsmodels.api as sm
    from final_stats import CONTROLS

    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    panel_end = raw["week"].max()

    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
    con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
    counts = clean_labels.build(con, panel_end)
    print(f"  proxy filings classified: {counts}", flush=True)

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
    del raw, lab, j
    gc.collect()

    tr = df.filter(pl.col("week") < dt.date(2024, 1, 1)).sample(
        fraction=0.3, seed=7)
    del df
    gc.collect()
    X = sm.add_constant(tr.select(cols).to_pandas())
    fit = sm.Logit(tr["y"].to_pandas(), X).fit(
        disp=False, maxiter=300, cov_type="cluster",
        cov_kwds={"groups": tr["cik"].to_pandas()})
    tv = fit.tvalues.drop("const")
    novel = [n for n in tv.index if n not in CONTROLS and abs(tv[n]) > 1.96]
    record("hazard", "verified-target labels", n_rows=tr.height,
           label_rate=float(tr["y"].mean() * 100),
           novel_significant=len(novel))
    for n in sorted(tv.index, key=lambda x: -abs(tv[x]))[:15]:
        record("hazard", f"  {n}", beta=float(fit.params[n]), z=float(tv[n]))


def t_size():
    """Was log_assets the model spotting ACQUIRERS rather than targets?

    log_assets is a top predictor of a label built from merger proxies, and
    merger proxies are filed by buyers too. If the size effect is really an
    acquirer effect, then dropping the 581 known survivors from the positive
    class should flatten it. Three label sets, four configurations. 12 fits.
    """
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    panel_end = raw["week"].max()
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
    con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
    clean_labels.build(con, panel_end)
    con.register("f", raw.select(["cik", "week"]).to_arrow())

    def labelled(outcome_sql):
        lab = con.execute(f"""
            SELECT f.cik, f.week, CASE WHEN EXISTS (
                SELECT 1 FROM deals_clean d WHERE d.cik = f.cik
                  AND {outcome_sql}
                  AND f.week <  d.agreement_date
                  AND f.week >= d.agreement_date - INTERVAL {HORIZON} WEEK
            ) THEN 1 ELSE 0 END AS yh FROM f
        """).pl()
        j = raw.select(["cik", "week"]).join(lab, on=["cik", "week"],
                                             how="left")
        return raw.with_columns(
            j["yh"].fill_null(0).cast(pl.Int8).alias("y")
        ).select(["cik", "week", "y"] + cols).filter(
            ~pl.col("cik").is_in(spac_ciks()))

    size_cols = ["log_assets", "log_float"]
    for name, sql in (("raw proxy filers", "1=1"),
                      ("verified targets", "d.outcome = 'target'"),
                      ("survivors only", "d.outcome = 'survivor'")):
        df = labelled(sql)
        tr, va, te = split(df, 2024)
        if not te.height or not te["y"].sum():
            del df
            gc.collect()
            continue
        # Mean log_assets of positives vs negatives -- the effect itself,
        # before any model is involved.
        pos = te.filter(pl.col("y") == 1)["log_assets"].mean()
        neg = te.filter(pl.col("y") == 0)["log_assets"].mean()
        m, sd = seeded(tr, va, te, cols)
        m2, _ = seeded(tr, va, te, [c for c in cols if c not in size_cols])
        record("size", name, prec=m, sd=sd, without_size=m2,
               size_contributes=m - m2,
               pos_log_assets=float(pos or 0.0),
               neg_log_assets=float(neg or 0.0),
               base=float(te["y"].mean() * 100))
        del df, tr, va, te
        gc.collect()


def t_tender():
    """The target model against 616 tender-offer targets it has never seen.

    Tender offers were excluded from training because master.idx cannot tell
    bidder from target. That exclusion is what makes them a real held-out
    label: no shareholder vote, frequently hostile, usually cash -- a deal type
    the model has not been shown. 6 fits.
    """
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/tender.duckdb' AS t (READ_ONLY)")
    con.register("f", raw.select(["cik", "week"]).to_arrow())
    tlab = con.execute(f"""
        SELECT f.cik, f.week, CASE WHEN EXISTS (
            SELECT 1 FROM t.tender_offers o WHERE o.cik = f.cik
              AND f.week <  o.public_ts
              AND f.week >= o.public_ts - INTERVAL {HORIZON} WEEK
        ) THEN 1 ELSE 0 END AS yh FROM f
    """).pl()
    keys = raw.select(["cik", "week"])
    tender_y = keys.join(tlab, on=["cik", "week"], how="left")["yh"] \
                   .fill_null(0).cast(pl.Int8)

    # Train on the PROXY label, test on the TENDER label. Any company that is
    # a tender target must not also be a training positive, or this is not
    # held out -- so proxy positives keep their own label for training and the
    # tender label is only ever used on the test side.
    train_df = relabel(raw, HORIZON).select(["cik", "week", "y"] + cols)
    test_df = raw.with_columns(tender_y.alias("y")).select(
        ["cik", "week", "y"] + cols)
    del raw
    gc.collect()

    spac = spac_ciks()
    train_df = train_df.filter(~pl.col("cik").is_in(spac))
    test_df = test_df.filter(~pl.col("cik").is_in(spac))

    for yr in YEARS:
        tr, va, _ = split(train_df, yr)
        _, _, te = split(test_df, yr)
        if not te.height or not te["y"].sum():
            continue
        m, sd = seeded(tr, va, te, cols)
        base = float(te["y"].mean() * 100)
        record("tender", f"tender targets {yr}", prec=m, sd=sd, base=base,
               lift=m / base if base else 0.0,
               positives=int(te["y"].sum()))
        del tr, va, te
        gc.collect()


STAGES = {"buyerperm": t_buyerperm, "hazard": t_hazard, "size": t_size,
          "tender": t_tender}


def report():
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    cur = None
    for r in rows:
        if r["test"] != cur:
            cur = r["test"]
            print(f"\n=== {cur.upper()} ===")
        rest = {k: v for k, v in r.items() if k not in ("test", "label")}
        print(f"  {r['label']:<38} " + "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in rest.items()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    report() if cmd == "report" else STAGES[cmd]()
