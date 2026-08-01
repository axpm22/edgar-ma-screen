"""Given a target about to be acquired, can we name the acquirer?

Framed as ranking rather than classification. For each real (target, acquirer)
pair, the true acquirer competes against 100 companies sampled from the same
week's universe, and the model ranks all 101. LightGBM's lambdarank with the
PAIR as query group optimises exactly that ordering.

Two design points carry the result:

1. Features constant within a candidate set cannot rank anything. The target's
   own attributes -- including its target-model score -- are identical across
   all 101 candidates, so they are excluded and only candidate attributes and
   target-candidate INTERACTIONS are used. Getting this wrong produces a model
   that looks trained and ranks at chance.

2. Sector-timing features can encode the deal being predicted. Everything is
   reported at 4- and 13-week embargoes; if accuracy collapses between them the
   model was reading the announcement.

Random baseline with 100 distractors is 1/101 = 0.99%.

    .venv/bin/python scripts/matching.py            # full feature set
    .venv/bin/python scripts/matching.py no_sector  # sector_deal_intensity out

Both variants are kept in data/matching.json, keyed by (embargo, variant).

4 fits per variant, ~2 minutes.
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

K_NEG = 100
EMBARGOES = (4, 13)
SEEDS = (11, 22)
OUT = Path("data/matching.json")

RANK_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 5, 10],
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbosity": -1, "num_threads": 2,
}
ROUNDS = 300

# Candidate-varying attributes only.
CAND_COLS = ["log_assets", "log_float", "cash_to_assets", "leverage",
             "fcf_to_assets", "acq_capacity", "dry_powder", "debt_headroom",
             "shelf_52w", "raise_52w", "shelf_new", "form8k_26w",
             "sector_deal_intensity", "p_buyer"]

# Rolling-origin cutoffs. pair_scores only covers 2023-01-02..2025-07-28, so
# the 2023 fold has no training pairs; it is attempted and reported anyway
# rather than silently dropped.
CUTS = (dt.date(2023, 1, 1), dt.date(2024, 1, 1))


def build_candidates(rng, pairs, panel, sic):
    """One block of K_NEG+1 rows per pair. Label 1 on the true acquirer.

    The embargo is already baked into `pairs.obs_week` by the caller.
    """
    by_week = {w: g for w, g in
               panel.partition_by("week", as_dict=True, include_key=True).items()}
    blocks = []
    for row in pairs.iter_rows(named=True):
        wk = row["obs_week"]
        # polars 1.x keys partitions by 1-tuple; older versions by scalar.
        # `a or b` is not usable here -- bool(DataFrame) raises.
        pool = by_week.get((wk,))
        if pool is None:
            pool = by_week.get(wk)
        if pool is None or pool.height < K_NEG + 1:
            continue
        if row["acquirer_cik"] not in pool["cik"]:
            continue
        truth = pool.filter(pl.col("cik") == row["acquirer_cik"])
        negs = pool.filter(pl.col("cik") != row["acquirer_cik"])
        idx = rng.choice(negs.height, size=K_NEG, replace=False)
        block = pl.concat([truth, negs[idx]])
        blocks.append(block.with_columns([
            pl.Series("rel", [1] + [0] * K_NEG).cast(pl.Int8),
            pl.lit(row["target_cik"]).alias("target_cik"),
            pl.lit(row["first_ts"]).alias("first_ts"),
            pl.lit(f"{row['target_cik']}|{row['first_ts']}").alias("qid"),
        ]))
    if not blocks:
        return None
    df = pl.concat(blocks)

    # Target-candidate interactions: the only features that can express "these
    # two fit together" rather than "this candidate is acquisitive".
    t = sic.rename({"cik": "target_cik", "sic": "t_sic"})
    df = df.join(t, on="target_cik", how="left")
    df = df.join(sic.rename({"sic": "c_sic"}), on="cik", how="left")
    tgt_size = panel.select(["cik", "week", "log_assets"]).rename(
        {"cik": "target_cik", "log_assets": "t_log_assets"})
    df = df.join(tgt_size, left_on=["target_cik", "week"],
                 right_on=["target_cik", "week"], how="left")
    return df.with_columns([
        (pl.col("t_sic").str.slice(0, 2)
         == pl.col("c_sic").str.slice(0, 2)).cast(pl.Int8).alias("same_sic2"),
        (pl.col("t_sic") == pl.col("c_sic")).cast(pl.Int8).alias("same_sic4"),
        (pl.col("log_assets") - pl.col("t_log_assets")).alias("size_gap"),
    ]).with_columns(
        pl.col("size_gap").abs().alias("abs_size_gap")
    ).drop_nulls(subset=["size_gap"])


def true_ranks(model, te, cols):
    """Rank of the true acquirer within each pair's 101 candidates."""
    p = np.asarray(model.predict(te.select(cols).to_pandas().astype("float32")))
    d = te.select(["qid", "rel"]).with_columns(pl.Series("p", p))
    return (d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("qid").alias("r"))
        .filter(pl.col("rel") == 1)["r"].to_numpy())


def summarise(ranks, n_boot=2000, seed=20260801):
    """Accuracy plus a bootstrap CI over pairs.

    The test set is thin -- a couple of hundred pairs -- so a top-1 difference
    of a few points between embargoes is inside sampling error. Reporting the
    CI is what stops that being narrated as a finding.
    """
    rng = np.random.default_rng(seed)
    boot = rng.choice(ranks, size=(n_boot, len(ranks)), replace=True)
    return {
        "n_pairs": int(len(ranks)),
        "top1": float(100 * np.mean(ranks <= 1)),
        "top5": float(100 * np.mean(ranks <= 5)),
        "top10": float(100 * np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "top1_ci_lo": float(np.percentile(100 * np.mean(boot <= 1, axis=1), 2.5)),
        "top1_ci_hi": float(np.percentile(100 * np.mean(boot <= 1, axis=1), 97.5)),
        "top10_ci_lo": float(np.percentile(100 * np.mean(boot <= 10, axis=1), 2.5)),
        "top10_ci_hi": float(np.percentile(100 * np.mean(boot <= 10, axis=1), 97.5)),
    }


def save(results, variant):
    """Merge into data/matching.json; the two variants are separate runs."""
    old = json.loads(OUT.read_text()) if OUT.exists() else []
    keys = {(r["embargo_weeks"], r["variant"]) for r in results}
    keep = [r for r in old if (r["embargo_weeks"], r.get("variant")) not in keys]
    OUT.write_text(json.dumps(keep + results, indent=2))
    print(f"wrote {OUT} ({variant})")


def main(variant: str = "full") -> None:
    cand_cols = [c for c in CAND_COLS
                 if variant != "no_sector" or c != "sector_deal_intensity"]

    feats = pl.read_parquet("data/buyer_features.parquet")
    scores = pl.read_parquet("data/pair_scores.parquet").select(
        ["cik", "week", "p_buyer"])
    panel = feats.join(scores, on=["cik", "week"], how="inner")
    keep = ["cik", "week"] + [c for c in cand_cols if c in panel.columns]
    panel = panel.select(keep)
    del feats
    gc.collect()
    print(f"panel {panel.height:,} company-weeks, "
          f"{panel['week'].min()}..{panel['week'].max()}", flush=True)

    con = duckdb.connect("data/deal.duckdb", read_only=True)
    sic = pl.from_arrow(con.execute(
        "SELECT cik, sic FROM company_sic").arrow())
    con.close()
    con = duckdb.connect("data/pairs.duckdb", read_only=True)
    all_pairs = pl.from_arrow(con.execute(
        "SELECT target_cik, acquirer_cik, first_ts FROM deal_pairs").arrow())
    con.close()

    cols = [c for c in cand_cols if c in panel.columns] + \
           ["same_sic2", "same_sic4", "size_gap", "abs_size_gap"]

    results = []
    for emb in EMBARGOES:
        rng = np.random.default_rng(20260801)
        pairs = all_pairs.with_columns(
            (pl.col("first_ts").cast(pl.Date)
             - pl.duration(weeks=emb)).dt.truncate("1w").alias("obs_week"))
        cand = build_candidates(rng, pairs, panel, sic)
        if cand is None:
            print(f"embargo {emb}w: no candidate blocks")
            continue
        # Sanity: block count must be in the hundreds. Single digits means the
        # week keys or the obs_week truncation did not line up with the panel.
        n_blocks = cand["qid"].n_unique()
        print(f"embargo {emb}w: {n_blocks} candidate blocks, "
              f"{cand.height:,} rows, {int(cand['rel'].sum())} positives, "
              f"first_ts {cand['first_ts'].min()}..{cand['first_ts'].max()}",
              flush=True)

        # Rolling-origin folds, pooled.
        pooled, train_pairs, imp = [], 0, []
        for cut in CUTS:
            end = dt.date(cut.year + 1, 1, 1) if cut.year == 2023 else None
            tr = cand.filter(pl.col("first_ts") < cut).sort("qid")
            te = cand.filter(pl.col("first_ts") >= cut)
            if end is not None:
                te = te.filter(pl.col("first_ts") < end)
            te = te.sort("qid")
            if not tr.height or not te.height:
                print(f"  fold {cut}: skipped "
                      f"(train {tr['qid'].n_unique()} pairs, "
                      f"test {te['qid'].n_unique()} pairs)", flush=True)
                continue
            print(f"  fold {cut}: train {tr['qid'].n_unique()} pairs, "
                  f"test {te['qid'].n_unique()} pairs", flush=True)
            train_pairs = max(train_pairs, int(tr["qid"].n_unique()))
            per_seed = []
            for s in SEEDS:
                g = tr.group_by("qid", maintain_order=True).len()["len"].to_list()
                d = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                                label=tr["rel"].to_pandas(), group=g)
                m = lgb.train({**RANK_PARAMS, "bagging_seed": s,
                               "feature_fraction_seed": s,
                               "data_random_seed": s}, d,
                              num_boost_round=ROUNDS)
                per_seed.append(true_ranks(m, te, cols))
                if s == SEEDS[0] and not imp:
                    imp = sorted(zip(m.feature_name(),
                                     m.feature_importance("gain")),
                                 key=lambda t: -t[1])[:6]
                del m, d
                gc.collect()
            # Average the rank across seeds, then pool folds.
            pooled.append(np.mean(per_seed, axis=0))
            del tr, te
            gc.collect()

        if not pooled:
            print(f"embargo {emb}w: empty split")
            continue
        rec = {"embargo_weeks": emb, "variant": variant,
               "n_blocks": int(n_blocks), "train_pairs": train_pairs,
               "random_baseline_top1": 100 / (K_NEG + 1),
               "top_features": [n for n, _ in imp],
               **summarise(np.concatenate(pooled))}
        results.append(rec)
        print(f"embargo {emb:>2}w  test pairs {rec['n_pairs']:>4}  "
              f"top1 {rec['top1']:>5.1f}% "
              f"[{rec['top1_ci_lo']:.1f}-{rec['top1_ci_hi']:.1f}]  "
              f"top10 {rec['top10']:>5.1f}% "
              f"[{rec['top10_ci_lo']:.1f}-{rec['top10_ci_hi']:.1f}]  "
              f"MRR {rec['mrr']:.3f}  median rank {rec['median_rank']:.0f} "
              f"(random top1 = {rec['random_baseline_top1']:.2f}%)",
              flush=True)
        print(f"           top features: {', '.join(rec['top_features'])}",
              flush=True)
        del cand
        gc.collect()

    save(results, variant)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
