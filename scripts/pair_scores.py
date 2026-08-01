"""Score both models on the three test years, once, and cache to parquet.

Three downstream analyses (alignment, matching, deal details) need per
company-week target and buyer scores. Fitting in each of them would cost three
times the compute and would let them drift apart on model config. This is the
single place the two models are defined for pair work.

Both models use the honest configuration: SPACs excluded, and for the buyer
model the self-referential features (s4_52w, goodwill_to_assets,
intangibles_to_assets) dropped -- they identify acquisitive firms rather than
imminent acquisitions.

    .venv/bin/python scripts/pair_scores.py

6 fits, ~2 minutes.
"""
import gc
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, relabel  # noqa: E402
from select_cv import PARAMS, ROUNDS, SEEDS, split, spac_ciks  # noqa: E402

from deal import feat_buyer, features  # noqa: E402

YEARS = (2023, 2024, 2025)
OUT = "data/pair_scores.parquet"


def _fit_predict(tr, va, te, cols):
    """Mean prediction over seeds. One booster alive at a time."""
    acc = np.zeros(te.height)
    for s in SEEDS:
        p = {**PARAMS, "bagging_seed": s, "feature_fraction_seed": s,
             "data_random_seed": s}
        dtr = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                          label=tr["y"].to_pandas())
        dva = lgb.Dataset(va.select(cols).to_pandas().astype("float32"),
                          label=va["y"].to_pandas())
        b = lgb.train(p, dtr, num_boost_round=ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        acc += np.asarray(
            b.predict(te.select(cols).to_pandas().astype("float32")))
        del b, dtr, dva
        gc.collect()
    return acc / len(SEEDS)


def _target_frame(spac):
    raw = pl.read_parquet("data/features.parquet")
    cols = [c for c in features.FEATURE_COLS
            if raw[c].std() is not None and raw[c].std() > 0]
    df = relabel(raw, HORIZON).select(["cik", "week", "y"] + cols)
    del raw
    gc.collect()
    return df.filter(~pl.col("cik").is_in(spac)), cols


def _buyer_frame(spac):
    df = pl.read_parquet("data/buyer_features.parquet").filter(
        ~pl.col("cik").is_in(spac))
    self_ref = set(feat_buyer.SELF_REFERENTIAL)
    base = [c for c in features.FEATURE_COLS
            if c in df.columns and df[c].std() and df[c].std() > 0]
    extra = [c for c in feat_buyer.BUYER_COLS
             if df[c].std() and df[c].std() > 0]
    cols = [c for c in base + extra if c not in self_ref]
    return df.select(["cik", "week", "y"] + cols), cols


def main() -> None:
    spac = spac_ciks()
    out = []

    for name, loader in (("p_target", _target_frame), ("p_buyer", _buyer_frame)):
        df, cols = loader(spac)
        print(f"{name}: {df.height:,} rows, {len(cols)} features, "
              f"label rate {df['y'].mean() * 100:.2f}%", flush=True)
        for yr in YEARS:
            tr, va, te = split(df, yr)
            if not te.height:
                continue
            p = _fit_predict(tr, va, te, cols)
            out.append(te.select(["cik", "week"]).with_columns([
                pl.lit(yr).cast(pl.Int32).alias("test_year"),
                pl.Series(name, p),
            ]))
            print(f"  {yr}: {te.height:,} scored", flush=True)
            del tr, va, te
            gc.collect()
        del df
        gc.collect()

    tgt = pl.concat([o for o in out if "p_target" in o.columns])
    buy = pl.concat([o for o in out if "p_buyer" in o.columns])
    merged = tgt.join(buy, on=["cik", "week", "test_year"], how="inner")
    merged.write_parquet(OUT)
    print(f"\nwrote {OUT}: {merged.height:,} rows")
    print(merged.select(["p_target", "p_buyer"]).describe())
    # The two scores must not be near-duplicates -- if they are, "do they line
    # up" is a question about one model, not two.
    r = np.corrcoef(merged["p_target"].to_numpy(),
                    merged["p_buyer"].to_numpy())[0, 1]
    print(f"corr(p_target, p_buyer) = {r:.3f}")


if __name__ == "__main__":
    main()
