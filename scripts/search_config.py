"""Phase 1: configuration search on VALIDATION only.

The 2024+ test period is never read here. Every decision is made on
2022-2023 validation, so the test set stays unspent for phase 2.

    .venv/bin/python scripts/search_config.py
"""
import datetime as dt
import itertools
import json

import duckdb
import polars as pl

from deal import features, model_gbm, screen

PARQUET = "data/features.parquet"
TRAIN_END = dt.date(2022, 1, 1)
VALID_END = dt.date(2024, 1, 1)   # test period starts here and is untouched

N_EVAL = 25


def relabel(df: pl.DataFrame, horizon_weeks: int) -> pl.DataFrame:
    """Label joined on (cik, week) -- never positionally, since a DuckDB query
    over a registered Arrow table gives no row-order guarantee."""
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
    joined = keys.join(lab, on=["cik", "week"], how="left")
    return df.with_columns(joined["yh"].fill_null(0).cast(pl.Int8).alias("y"))


def main() -> None:
    df = pl.read_parquet(PARQUET)
    usable = [c for c in features.FEATURE_COLS
              if df[c].std() is not None and df[c].std() > 0]

    zc = [c for c in usable if c.endswith("_z")]
    dc = [c for c in usable if c.endswith("_d")]
    controls = [c for c in ["log_assets", "log_float", "float_to_assets",
                            "cash_to_assets", "leverage", "revenue_growth",
                            "asset_growth", "float_growth", "rnd_intensity",
                            "operating_margin", "net_margin",
                            "goodwill_to_assets", "intangibles_to_assets",
                            "lt_debt_to_assets", "sector_deal_intensity",
                            "age_weeks", "quarter", "year"] if c in usable]

    FEATURE_SETS = {
        "all": usable,
        "no_z": [c for c in usable if c not in zc],
        "no_deltas": [c for c in usable if c not in dc],
        "novel_only": [c for c in usable if c not in controls],
    }
    HORIZONS = [13, 26, 39, 52]
    LEAVES = [31, 63, 127]
    MIN_LEAF = [200, 500, 2000]
    LR = [0.03, 0.05]
    RECENCY = [None, 2.0, 4.0, 8.0]

    results = []
    # Stage A: coarse -- feature set x horizon, defaults elsewhere.
    print("stage A: feature set x horizon", flush=True)
    for fs_name, cols in FEATURE_SETS.items():
        for h in HORIZONS:
            lab = relabel(df, h)
            tr = lab.filter(pl.col("week") < TRAIN_END)
            va = lab.filter((pl.col("week") >= TRAIN_END)
                            & (pl.col("week") < VALID_END))
            b = model_gbm.fit(tr, valid=va, cols=cols)
            pr = screen.weekly_precision(
                va, model_gbm.predict(b, va, cols), N_EVAL)["precision"]
            results.append({"stage": "A", "features": fs_name, "horizon": h,
                            "precision": pr})
            print(f"  {fs_name:<11} h{h:<3} prec@25 {pr*100:>6.2f}%", flush=True)

    best_a = max(r for r in results if r["stage"] == "A")
    best_a = sorted([r for r in results if r["stage"] == "A"],
                    key=lambda r: -r["precision"])[0]
    print(f"\nstage A winner: {best_a['features']} h{best_a['horizon']} "
          f"({best_a['precision']*100:.2f}%)\n", flush=True)

    cols = FEATURE_SETS[best_a["features"]]
    lab = relabel(df, best_a["horizon"])
    tr = lab.filter(pl.col("week") < TRAIN_END)
    va = lab.filter((pl.col("week") >= TRAIN_END) & (pl.col("week") < VALID_END))

    # Stage B: hyperparameters + recency, on the stage-A winner.
    print("stage B: hyperparameters", flush=True)
    base = dict(model_gbm.PARAMS)
    for leaves, min_leaf, lr, hl in itertools.product(LEAVES, MIN_LEAF, LR, RECENCY):
        model_gbm.PARAMS.update({"num_leaves": leaves,
                                 "min_data_in_leaf": min_leaf,
                                 "learning_rate": lr})
        w = None
        if hl is not None:
            model_gbm.RECENCY_HALF_LIFE_YEARS = hl
        b = model_gbm.fit(tr, valid=va, cols=cols, recency=hl is not None)
        pr = screen.weekly_precision(
            va, model_gbm.predict(b, va, cols), N_EVAL)["precision"]
        results.append({"stage": "B", "features": best_a["features"],
                        "horizon": best_a["horizon"], "num_leaves": leaves,
                        "min_data_in_leaf": min_leaf, "learning_rate": lr,
                        "recency_hl": hl, "precision": pr})
        print(f"  leaves={leaves:<4} minleaf={min_leaf:<5} lr={lr} "
              f"recency={hl}  prec@25 {pr*100:>6.2f}%", flush=True)
    model_gbm.PARAMS.clear(); model_gbm.PARAMS.update(base)

    ranked = sorted(results, key=lambda r: -r["precision"])
    print("\n=== TOP 10 ON VALIDATION ===")
    for r in ranked[:10]:
        print(f"  {r['precision']*100:>6.2f}%  {r}")

    with open("data/search_results.json", "w") as fh:
        json.dump({"ranked": ranked, "winner": ranked[0]}, fh, indent=2,
                  default=str)
    print(f"\nwinner written to data/search_results.json")


if __name__ == "__main__":
    main()
