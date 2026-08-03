"""Accuracy and feature-value report for both models. ONE stage per process.

    .venv/bin/python scripts/feature_report.py accuracy   # year-on-year tables
    .venv/bin/python scripts/feature_report.py ablation   # leave-one-family-out
    .venv/bin/python scripts/feature_report.py solo       # each family ALONE
    .venv/bin/python scripts/feature_report.py report     # print everything

Every stage appends to data/feature_report.json and exits. LightGBM's arena
memory is not reliably returned to the OS and this project has been OOM-killed
twice by multi-fit single processes.

Two things this reports that the project has not reported before:

  distinct_hits -- how many separate companies a precision number rests on. A
      tender-offer lift of 3.37x with a tight CI turned out to be one company
      held for 23 consecutive weeks.
  solo          -- each family fitted ALONE. Leave-one-out only shows what a
      family adds ON TOP of everything else, so a family that is genuinely
      predictive but redundant reads as worthless. Solo separates "carries no
      signal" from "carries signal someone else already carries".
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
from final_stats import HORIZON, N_EVAL  # noqa: E402
from select_cv import PARAMS, ROUNDS, SEEDS, spac_ciks  # noqa: E402

from deal import clean_labels, feat_buyer, features, screen  # noqa: E402
from deal.feat_items import ITEM_COLS  # noqa: E402
from deal.feat_literature import LIT_COLS  # noqa: E402

OUT = Path("data/feature_report.json")
SAMPLE = 0.25
ABLATE_YEARS = (2023, 2024, 2025)


def record(**kw) -> None:
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    rows.append(kw)
    OUT.write_text(json.dumps(rows, indent=1, default=str))


def families(df) -> dict:
    """Feature families present in this frame."""
    f = {
        "fundamentals": features.FUND_COLS,
        "literature": LIT_COLS,
        "market": features.MARKET_COLS,
        "insider": features.INSIDER_COLS,
        "form counts": features.FORM_COLS,
        "8-K items": ITEM_COLS,
        "strategic text": features.FTS_COLS,
        "activist+peer": features.ACTIVIST_COLS + features.PEER_COLS,
        "z-scores": features.ZSCORE_COLS,
        "deltas": features.DELTA_COLS,
        "context": features.CONTEXT_COLS,
        "cert transparency": features.SIGNAL_COLS,
    }
    if any(c in df.columns for c in feat_buyer.BUYER_COLS):
        f["buyer capacity"] = feat_buyer.BUYER_COLS
    return {k: [c for c in v if c in df.columns] for k, v in f.items()}


# --------------------------------------------------------------- data --------

def _panel_years(df) -> list[int]:
    """Test years with a usable window, derived from the panel, not hard-coded."""
    safe = df["week"].max() - dt.timedelta(weeks=HORIZON)
    first = df["week"].min().year + 3        # 2y warm-up + 1 validation year
    return [y for y in range(first, safe.year + 1)
            if dt.date(y, 1, 1) <= safe]


def target_frame():
    """Verified targets: proxy filers that actually stopped filing."""
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if c in raw.columns and raw[c].std() and raw[c].std() > 0]
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
            AND f.week < d.agreement_date
            AND f.week >= d.agreement_date - INTERVAL {HORIZON} WEEK)
          THEN 1 ELSE 0 END AS y FROM f""").pl()
    y = raw.select(["cik", "week"]).join(lab, on=["cik", "week"],
                                         how="left")["y"].fill_null(0)
    df = raw.with_columns(y.cast(pl.Int8).alias("y")).select(
        ["cik", "week", "y"] + cols)
    del raw
    gc.collect()
    return df, cols


def buyer_frame():
    """Files an S-4 within the horizon. Self-referential columns dropped."""
    df = pl.read_parquet("data/buyer_features.parquet")
    self_ref = set(feat_buyer.SELF_REFERENTIAL)
    cols = [c for c in features.FEATURE_COLS + feat_buyer.BUYER_COLS
            if c in df.columns and c not in self_ref
            and df[c].std() and df[c].std() > 0]
    return df.select(["cik", "week", "y"] + cols), cols


# --------------------------------------------------------------- fitting -----

def split(df, year):
    safe = df["week"].max() - dt.timedelta(weeks=HORIZON)
    va0, te0 = dt.date(year - 1, 1, 1), dt.date(year, 1, 1)
    te1 = min(dt.date(year, 12, 31), safe)
    return (df.filter(pl.col("week") < va0).sample(fraction=SAMPLE, seed=1),
            df.filter((pl.col("week") >= va0) & (pl.col("week") < te0)),
            df.filter((pl.col("week") >= te0) & (pl.col("week") <= te1)))


def evaluate(tr, va, te, cols):
    """Mean over seeds. One booster alive at a time."""
    if tr.height < 40_000 or not te.height or not te["y"].sum():
        return None
    p, l, h = [], [], []
    for s in SEEDS:
        prm = {**PARAMS, "bagging_seed": s, "feature_fraction_seed": s,
               "data_random_seed": s}
        dtr = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                          label=tr["y"].to_pandas())
        dva = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                          label=va["y"].to_pandas())
        b = lgb.train(prm, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        pred = np.asarray(
            b.predict(te.select(cols).to_pandas().astype("float32")))
        r = screen.weekly_precision(te, pred, N_EVAL)
        p.append(r["precision"] * 100)
        l.append(r["lift"])
        h.append(r["distinct_hits"])
        del b, dtr, dva, pred
        gc.collect()
    return {"prec": float(np.mean(p)), "sd": float(np.std(p)),
            "lift": float(np.mean(l)), "distinct_hits": int(np.mean(h)),
            "base": float(te["y"].mean() * 100), "n_test": te.height,
            "n_train": tr.height, "n_feat": len(cols)}


# --------------------------------------------------------------- stages ------

def stage_accuracy():
    """Year-on-year, both models, with and without SPACs."""
    spac = set(spac_ciks())
    for model, loader in (("target", target_frame), ("buyer", buyer_frame)):
        full, cols = loader()
        years = _panel_years(full)
        print(f"\n=== {model.upper()}  years {years[0]}-{years[-1]} ===",
              flush=True)
        for uni in ("all", "nospac"):
            df = full if uni == "all" else full.filter(
                ~pl.col("cik").is_in(list(spac)))
            print(f"  -- {uni} ({df.height:,} rows, "
                  f"label {df['y'].mean()*100:.2f}%)", flush=True)
            for yr in years:
                tr, va, te = split(df, yr)
                r = evaluate(tr, va, te, cols)
                del tr, va, te
                gc.collect()
                if r is None:
                    print(f"     {yr}  (insufficient)", flush=True)
                    continue
                record(stage="accuracy", model=model, universe=uni,
                       year=yr, **r)
                print(f"     {yr}  {r['prec']:>6.2f}% +/-{r['sd']:.2f}  "
                      f"lift {r['lift']:>5.2f}x  base {r['base']:.2f}%  "
                      f"hits from {r['distinct_hits']:>2} cos", flush=True)
            if uni == "nospac":
                del df
            gc.collect()
        del full
        gc.collect()


def _family_stage(kind):
    """kind='ablation' (drop one) or 'solo' (keep only one)."""
    spac = set(spac_ciks())
    for model, loader in (("target", target_frame), ("buyer", buyer_frame)):
        full, cols = loader()
        df = full.filter(~pl.col("cik").is_in(list(spac)))
        del full
        gc.collect()
        fams = families(df)
        ctx = [c for c in features.CONTEXT_COLS if c in df.columns]
        print(f"\n=== {model.upper()} {kind} (SPACs excluded) ===", flush=True)

        base = _mean_over_years(df, cols)
        record(stage=kind, model=model, family="ALL", prec=base,
               n_feat=len(cols))
        print(f"  {'ALL FEATURES':<22} {base:>6.2f}%   n={len(cols)}",
              flush=True)

        for name, block in fams.items():
            if not block:
                continue
            if kind == "ablation":
                use = [c for c in cols if c not in set(block)]
            else:
                # Context alone is not a family test -- every solo run keeps
                # it so the comparison is "family + context" vs "context".
                use = [c for c in cols if c in set(block) | set(ctx)]
            if not use or len(use) == len(cols):
                continue
            m = _mean_over_years(df, use)
            if m is None:
                continue
            record(stage=kind, model=model, family=name, prec=m,
                   n_feat=len(use), delta=base - m if kind == "ablation"
                   else m - base)
            if kind == "ablation":
                d = base - m
                verdict = ("HELPS" if d > 2 else
                           "HURTS" if d < -2 else "noise")
                print(f"  without {name:<14} {m:>6.2f}%  "
                      f"contributes {d:+5.2f}pp  {verdict}", flush=True)
            else:
                print(f"  only {name:<17} {m:>6.2f}%  "
                      f"({m - base:+5.2f}pp vs all)  n={len(use)}", flush=True)
        del df
        gc.collect()


def _mean_over_years(df, cols):
    vals = []
    for yr in ABLATE_YEARS:
        tr, va, te = split(df, yr)
        r = evaluate(tr, va, te, cols)
        del tr, va, te
        gc.collect()
        if r:
            vals.append(r["prec"])
    return float(np.mean(vals)) if vals else None


def report():
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    acc = [r for r in rows if r["stage"] == "accuracy"]
    if acc:
        for model in ("target", "buyer"):
            for uni in ("all", "nospac"):
                sub = sorted([r for r in acc if r["model"] == model
                              and r["universe"] == uni], key=lambda r: r["year"])
                if not sub:
                    continue
                v = np.array([r["prec"] for r in sub])
                print(f"\n=== {model.upper()} / {uni} ===")
                print(f"{'year':<6}{'prec':>8}{'lift':>8}{'base':>8}"
                      f"{'cos':>6}")
                for r in sub:
                    print(f"{r['year']:<6}{r['prec']:>7.2f}%{r['lift']:>7.2f}x"
                          f"{r['base']:>7.2f}%{r['distinct_hits']:>6}")
                print(f"{'MEAN':<6}{v.mean():>7.2f}%   SD {v.std():.2f}   "
                      f"range {v.min():.1f}-{v.max():.1f}")
    for kind in ("ablation", "solo"):
        sub = [r for r in rows if r["stage"] == kind]
        if not sub:
            continue
        print(f"\n=== {kind.upper()} ===")
        for model in ("target", "buyer"):
            ms = [r for r in sub if r["model"] == model]
            if not ms:
                continue
            print(f"  -- {model}")
            for r in sorted(ms, key=lambda r: -(r.get("delta") or 99)):
                d = r.get("delta")
                print(f"     {r['family']:<20} {r['prec']:>6.2f}%"
                      + (f"  {d:+6.2f}pp" if d is not None else ""))


STAGES = {"accuracy": stage_accuracy,
          "ablation": lambda: _family_stage("ablation"),
          "solo": lambda: _family_stage("solo"),
          "report": report}

if __name__ == "__main__":
    STAGES[sys.argv[1]]()
