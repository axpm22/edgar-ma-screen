"""Train and evaluate. Compares model variants on both splits.

    .venv/bin/python scripts/train_eval.py
"""
import datetime as dt

import duckdb
import polars as pl
from sklearn.metrics import average_precision_score

from deal import features, model_gbm, screen, splits, universe_filter

FEATURES_PARQUET = "data/features.parquet"
LEAK_GAIN_SHARE = 0.5
HORIZONS = (13, 26, 52)


def leakage_check(booster) -> list[str]:
    gains = list(booster.feature_importance(importance_type="gain"))
    names = list(booster.feature_name())
    total = sum(gains) or 1.0
    return [n for n, g in zip(names, gains) if g / total > LEAK_GAIN_SHARE]


def horizon_labels(df: pl.DataFrame) -> dict[str, pl.Series]:
    """Relabel the same rows at several horizons, in DuckDB for speed.

    Labels are JOINED back on (cik, week), never taken positionally: a DuckDB
    query over a registered Arrow table gives no row-order guarantee, and
    assigning labels by position silently shuffles them against the features.
    """
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    keys = df.select(["cik", "week"])
    con.register("f", keys.to_arrow())
    out = {}
    for h in HORIZONS:
        lab = con.execute(f"""
            SELECT f.cik, f.week,
                   CASE WHEN EXISTS (
                     SELECT 1 FROM m.deals d WHERE d.cik = f.cik
                       AND f.week <  d.agreement_date
                       AND f.week >= d.agreement_date - INTERVAL {h} WEEK
                   ) THEN 1 ELSE 0 END AS yh
            FROM f
        """).pl()
        joined = keys.join(lab, on=["cik", "week"], how="left")
        out[f"h{h}"] = joined["yh"].fill_null(0).cast(pl.Int8)
    return out


def _curve(name: str, te: pl.DataFrame, p, extra: str = "") -> float:
    base = te["y"].mean()
    rows = screen.curve(te, p).to_dicts()
    best = max(r["precision"] for r in rows)
    line = "  ".join(f"N{r['n_per_week']}={r['precision']*100:.2f}%"
                     for r in rows)
    print(f"  {name:<22} PR-AUC {average_precision_score(te['y'].to_numpy(), p):.4f}  "
          f"| {line} {extra}")
    return best


def compare(label: str, tr: pl.DataFrame, te: pl.DataFrame,
            cols: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(f"train {tr.height:,} ({int(tr['y'].sum()):,} pos) | "
          f"test {te.height:,} ({int(te['y'].sum()):,} pos) | "
          f"base {te['y'].mean()*100:.2f}%")

    b = model_gbm.fit(tr, valid=te, cols=cols)
    _curve("binary", te, model_gbm.predict(b, te, cols))
    flagged = leakage_check(b)
    if flagged:
        print(f"  !! LEAKAGE WARNING: {flagged}")
    imp = sorted(zip(b.feature_name(), b.feature_importance("gain")),
                 key=lambda t: -t[1])[:8]
    print(f"  top by gain: {', '.join(n for n, _ in imp)}")

    br = model_gbm.fit(tr, valid=te, cols=cols, recency=True)
    _curve("binary + recency", te, model_gbm.predict(br, te, cols))

    try:
        bl = model_gbm.fit_rank(tr, valid=te, cols=cols)
        _curve("lambdarank", te, model_gbm.predict(bl, te, cols))
    except Exception as exc:
        print(f"  lambdarank failed: {type(exc).__name__}: {exc}")

    try:
        labs = horizon_labels(tr)
        ens = model_gbm.fit_horizon_ensemble(tr, labs, cols=cols)
        _curve("horizon ensemble", te,
               model_gbm.predict_ensemble(ens, te, cols),
               extra=f"({'/'.join(labs)})")
    except Exception as exc:
        print(f"  ensemble failed: {type(exc).__name__}: {exc}")


def main() -> None:
    df = pl.read_parquet(FEATURES_PARQUET)
    cols = [c for c in features.FEATURE_COLS
            if df[c].std() is not None and df[c].std() > 0]
    print(f"{df.height:,} rows, {len(cols)} usable features")
    dropped = sorted(set(features.FEATURE_COLS) - set(cols))
    if dropped:
        print(f"dropped (zero variance): {dropped}")

    tr, te = splits.grouped(df, test_frac=0.1)
    compare("GROUPED 90/10 (unseen companies)", tr, te, cols)

    ttr, tte = splits.by_time(df, dt.date(2024, 1, 1))
    compare("TIME-ORDERED (train <2024, test >=2024)", ttr, tte, cols)

    compare("TIME-ORDERED (operating companies only)",
            universe_filter.apply(ttr), universe_filter.apply(tte), cols)

    print("\nRead grouped and time-ordered together. Grouped far above "
          "time-ordered means regime dependence, not predictive power.")


if __name__ == "__main__":
    main()
