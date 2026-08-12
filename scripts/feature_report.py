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
# Family stages run on EVERY test year, paired. Three years produced a
# literature-only result that eleven years reversed by 5.25pp.
# One seed, because pairing across 11 years already averages seed noise
# and two seeds would double a 570-fit stage for little gain.
FAMILY_SEEDS = (11,)


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


def evaluate(tr, va, te, cols, seeds=SEEDS):
    """Mean over seeds. One booster alive at a time."""
    if tr.height < 40_000 or not te.height or not te["y"].sum():
        return None
    p, l, h = [], [], []
    for s in seeds:
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
        years = _panel_years(df)
        print(f"\n=== {model.upper()} {kind} (SPACs excluded, "
              f"{len(years)} years, {len(FAMILY_SEEDS)} seed) ===", flush=True)

        base = _per_year(df, cols, years, FAMILY_SEEDS)
        bm = float(np.mean(list(base.values())))
        record(stage=kind, model=model, family="ALL", prec=bm,
               n_feat=len(cols), per_year=base)
        print(f"  {'ALL FEATURES':<22} {bm:>6.2f}%   n={len(cols)}",
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
            got = _per_year(df, use, years, FAMILY_SEEDS)
            if not got:
                continue
            m = float(np.mean(list(got.values())))
            st = _paired(base, got)          # positive = family helps
            record(stage=kind, model=model, family=name, prec=m,
                   n_feat=len(use), **st)
            flag = "SIGNIFICANT" if st.get("significant") else "not sig"
            print(f"  {'without' if kind == 'ablation' else 'only':<7} "
                  f"{name:<18} {m:>6.2f}%  "
                  f"{'contributes' if kind == 'ablation' else 'vs all'} "
                  f"{st['delta'] if kind == 'ablation' else -st['delta']:+6.2f}pp"
                  f" +/-{st['se']:.2f}  {st['years_positive']}/{st['n_years']}"
                  f"  {flag}", flush=True)
        del df
        gc.collect()


def _per_year(df, cols, years, seeds):
    """Precision for each test year. Returns {year: precision}."""
    out = {}
    for yr in years:
        tr, va, te = split(df, yr)
        r = evaluate(tr, va, te, cols, seeds=seeds)
        del tr, va, te
        gc.collect()
        if r:
            out[yr] = r["prec"]
    return out


def _paired(base: dict, other: dict) -> dict:
    """Paired year-by-year comparison.

    The families were first compared on three years, and that was not enough:
    literature-only looked equal to the full 72-feature model on 2023-2025
    (9.83% vs 9.31%) and came in 5.25pp WORSE across eleven. Two noisy tails
    pointing opposite ways produced a clean-looking result from nothing.

    Pairing by year cancels the regime effect, which is the largest source of
    variance here, so the SD of the DIFFERENCE is far smaller than the SD of
    either level. That is what makes eleven years decisive where three were not.
    """
    yrs = sorted(set(base) & set(other))
    d = np.array([base[y] - other[y] for y in yrs])
    if not len(d):
        return {}
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    return {"delta": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
            "se": se, "n_years": len(d),
            "years_positive": int((d > 0).sum()),
            # Significant when the mean difference clears two standard errors.
            "significant": bool(abs(d.mean()) > 2 * se) if se else False}


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


def stage_macro():
    """Do macro regime variables help? Two different questions.

    1. WITHIN-WEEK RANKING. Expected to be ~0 and that is not a weak result:
       every macro column is identical for all companies in a week, and the
       screen ranks within weeks, so the main effect cannot move precision by
       construction. Only interactions can.

    2. BETWEEN-YEAR. The model scores 19.2% in 2021 and 6.7% in 2022 and no
       company-level feature explains that. Here we correlate each year's
       precision against that year's mean credit spread, VIX and yield curve.
       A regime variable that says WHEN the screen works beats one that
       marginally reorders a week.
    """
    from deal import feat_macro

    spac = set(spac_ciks())
    df, cols = target_frame()
    df = df.filter(~pl.col("cik").is_in(list(spac)))
    df = feat_macro.add(df)
    years = _panel_years(df)
    macro = [c for c in feat_macro.MACRO_COLS
             if c in df.columns and df[c].std() and df[c].std() > 0]
    print(f"=== MACRO: within-week ranking ({len(macro)} cols) ===", flush=True)

    per_year = {}
    for tag, use in (("without macro", cols), ("with macro", cols + macro)):
        vals = []
        for yr in years:
            tr, va, te = split(df, yr)
            r = evaluate(tr, va, te, use)
            del tr, va, te
            gc.collect()
            if r:
                vals.append(r["prec"])
                per_year.setdefault(tag, {})[yr] = r["prec"]
        m = float(np.mean(vals)) if vals else 0.0
        record(stage="macro", variant=tag, mean=m, n_feat=len(use),
               per_year=per_year.get(tag, {}))
        print(f"  {tag:<16} {m:>6.2f}%   n={len(use)}", flush=True)

    base = per_year.get("without macro", {})
    print("\n=== MACRO: does it explain WHEN the screen works? ===", flush=True)
    yr_macro = (feat_macro.add(
        pl.DataFrame({"week": sorted(df["week"].unique().to_list())}))
        .with_columns(pl.col("week").dt.year().alias("yr"))
        .group_by("yr").mean())
    rows = [(y, p, yr_macro.filter(pl.col("yr") == y)) for y, p in
            sorted(base.items())]
    rows = [(y, p, g) for y, p, g in rows if g.height]
    prec = np.array([p for _, p, _ in rows])
    print(f"  {'year':<6}{'prec':>8}{'credit':>9}{'vix':>8}{'curve':>8}")
    for y, p, g in rows:
        print(f"  {y:<6}{p:>7.2f}%{g['mac_credit_spread'][0]:>9.2f}"
              f"{g['mac_vix'][0]:>8.1f}{g['mac_yield_curve'][0]:>8.2f}")
    for name in ("mac_credit_spread", "mac_vix", "mac_yield_curve",
                 "mac_unrate", "mac_pres_rep"):
        x = np.array([g[name][0] for _, _, g in rows], dtype=float)
        if x.std() == 0 or len(x) < 4:
            continue
        r = float(np.corrcoef(x, prec)[0, 1])
        record(stage="macro_year", variable=name, corr=r, n_years=len(x))
        print(f"  corr(year precision, {name:<20}) = {r:+.3f}  n={len(x)}",
              flush=True)


STAGES = {"accuracy": stage_accuracy,
          "ablation": lambda: _family_stage("ablation"),
          "solo": lambda: _family_stage("solo"),
          "macro": stage_macro,
          "report": report}

if __name__ == "__main__":
    STAGES[sys.argv[1]]()
