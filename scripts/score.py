"""Train-and-persist the scorer, then score companies.

    .venv/bin/python scripts/score.py train           # fit + persist + calibrate
    .venv/bin/python scripts/score.py top --n 20      # highest-scoring names now
    .venv/bin/python scripts/score.py cik 320193      # score one company
"""
import argparse
import datetime as dt
import gc
import sys

import duckdb
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL, TEST_START, relabel  # noqa: E402

from deal import features, model_gbm, scorer, screen  # noqa: E402

PARQUET = "data/features.parquet"


def _cols(df: pl.DataFrame) -> list[str]:
    usable = [c for c in features.FEATURE_COLS
              if df[c].std() is not None and df[c].std() > 0]
    return [c for c in usable if not c.endswith("_z")]   # validated config


def _names(ciks: list[str]) -> dict[str, str]:
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    rows = con.execute(
        "SELECT cik, name FROM m.universe WHERE cik IN "
        f"({','.join(repr(c) for c in ciks)})").fetchall()
    con.close()
    return dict(rows)


def cmd_train() -> None:
    raw = pl.read_parquet(PARQUET)
    cols = _cols(raw)
    df = relabel(raw, HORIZON)
    del raw
    gc.collect()

    safe_end = df["week"].max() - dt.timedelta(weeks=HORIZON)
    tr = df.filter(pl.col("week") < TEST_START)
    te = df.filter((pl.col("week") >= TEST_START) & (pl.col("week") <= safe_end))
    print(f"train {tr.height:,} | calibration test {te.height:,} "
          f"(base {te['y'].mean()*100:.2f}%)")

    booster = model_gbm.fit(tr, valid=te, cols=cols)
    p = model_gbm.predict(booster, te, cols)

    # Calibrate the bands on held-out data: the hit rate quoted for "top 25"
    # is what top-25 actually delivered, not an assertion.
    curve = {str(n): screen.weekly_precision(te, p, n)["precision"]
             for n in scorer.BANDS}
    meta = {"horizon_weeks": HORIZON, "base_rate": float(te["y"].mean()),
            "trained_through": str(tr["week"].max()),
            "precision_curve": curve}
    scorer.save(booster, cols, meta)

    print("\ncalibrated bands (measured on held-out test):")
    for n in scorer.BANDS:
        print(f"  top {n:>3}/week -> {curve[str(n)]*100:>6.2f}% had a deal "
              f"within {HORIZON} weeks")
    print(f"\nsaved {scorer.MODEL_PATH}")


def _latest_week(df: pl.DataFrame) -> pl.DataFrame:
    """Active cross-section, not one literal week -- see scorer.asof_cross_section."""
    return scorer.asof_cross_section(df)


def cmd_top(n: int) -> None:
    booster, cols, meta = scorer.load()
    df = pl.read_parquet(PARQUET)
    week = _latest_week(df)
    ranked = scorer.score_week(booster, cols, meta, week)
    top = ranked.head(n)
    names = _names(top["cik"].to_list())

    print(f"as of {week['week'].max()}  |  {week.height:,} active companies scored")
    print(f"horizon: {meta['horizon_weeks']} weeks   base rate "
          f"{meta['base_rate']*100:.2f}%\n")
    print(f"{'rank':>4} {'score':>6} {'prob':>7} {'lift':>6}  company")
    print("-" * 74)
    for r in top.iter_rows(named=True):
        nm = names.get(r["cik"], f"CIK {r['cik']}")[:38]
        print(f"{r['rank']:>4} {r['score']:>6.1f} {r['prob']*100:>6.2f}% "
              f"{r['lift']:>5.1f}x  {nm}")
    b = scorer.band_for_rank(1, meta)
    print(f"\nhistorically, {b['band']} names had a "
          f"{b['historical_hit_rate']*100:.1f}% chance of a deal in "
          f"{meta['horizon_weeks']} weeks")


def cmd_cik(cik: str) -> None:
    booster, cols, meta = scorer.load()
    df = pl.read_parquet(PARQUET)
    week = _latest_week(df)
    ranked = scorer.score_week(booster, cols, meta, week)
    row = ranked.filter(pl.col("cik") == cik)
    if not row.height:
        print(f"CIK {cik} not in the latest week ({week['week'][0]}).")
        return
    r = row.row(0, named=True)
    band = scorer.band_for_rank(r["rank"], meta)
    nm = _names([cik]).get(cik, f"CIK {cik}")

    print(f"{nm}   (CIK {cik})")
    print(f"week of {r['week']}\n")
    print(f"  SCORE       {r['score']:.1f} / 100")
    print(f"  rank        {r['rank']:,} of {ranked.height:,} companies")
    print(f"  probability {r['prob']*100:.2f}%  ({r['lift']:.1f}x base rate)")
    print(f"  band        {band['band']} -> historically "
          f"{band['historical_hit_rate']*100:.1f}% had a deal within "
          f"{meta['horizon_weeks']} weeks\n")

    src = week.filter(pl.col("cik") == cik)
    print("  what is driving this score:")
    for name, val in scorer.explain(booster, cols, src):
        raw = src[name][0]
        arrow = "raises" if val > 0 else "lowers"
        print(f"    {arrow:>6}  {name:<24} (value {raw:>10.3f})  "
              f"contribution {val:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "top", "cik"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train()
    elif a.cmd == "top":
        cmd_top(a.n)
    else:
        cmd_cik(a.target)
