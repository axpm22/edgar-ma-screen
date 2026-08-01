"""Do the target screen and the buyer screen agree on real deals?

Each model's own precision says how often its list is right. It says nothing
about whether the two lists point at the same transaction. This measures that
directly: for each real (target, acquirer) pair, was the target in the target
model's top 25 that week AND the acquirer in the buyer model's top 25?

The raw joint rate is uninteresting on its own -- two sparse lists intersect
rarely whatever they encode. The informative quantity is

    joint observed / (target marginal x buyer marginal)

which is 1.0 when the two screens agree only as often as independent screens
would, and above 1.0 when they genuinely co-fire on the same deal.

Everything is reported at several lead times. A result that only appears at
lead 1 is the models reading the announcement, not predicting it.

    .venv/bin/python scripts/alignment.py

No model fits.
"""
import json

import duckdb
import numpy as np
import polars as pl

LEADS = (1, 4, 13, 26)
TOP_N = 25
OUT = "data/alignment.json"


def main() -> None:
    scores = pl.read_parquet("data/pair_scores.parquet")

    # Within-week rank for each model. rank 1 = highest score.
    ranked = scores.with_columns([
        pl.col("p_target").rank("ordinal", descending=True)
          .over("week").alias("r_target"),
        pl.col("p_buyer").rank("ordinal", descending=True)
          .over("week").alias("r_buyer"),
    ])
    week_size = ranked.group_by("week").len().rename({"len": "n_week"})
    ranked = ranked.join(week_size, on="week")

    con = duckdb.connect("data/pairs.duckdb", read_only=True)
    pairs = pl.from_arrow(con.execute(
        "SELECT target_cik, acquirer_cik, first_ts FROM deal_pairs").arrow())
    con.close()

    out = []
    for lead in LEADS:
        # The observation week: `lead` weeks before the announcement.
        obs = pairs.with_columns(
            (pl.col("first_ts").cast(pl.Date)
             - pl.duration(weeks=lead)).dt.truncate("1w").alias("week"))

        t = obs.join(
            ranked.select(["cik", "week", "r_target", "n_week"]),
            left_on=["target_cik", "week"], right_on=["cik", "week"],
            how="inner")
        both = t.join(
            ranked.select(["cik", "week", "r_buyer"]),
            left_on=["acquirer_cik", "week"], right_on=["cik", "week"],
            how="inner")
        if not both.height:
            continue

        hit_t = both["r_target"] <= TOP_N
        hit_b = both["r_buyer"] <= TOP_N
        joint = float((hit_t & hit_b).mean())
        pt, pb = float(hit_t.mean()), float(hit_b.mean())
        expected = pt * pb
        # What a random pair would score, given each week's universe size.
        chance = float((TOP_N / both["n_week"]).mean()) ** 2

        # DEVIATION from plan: the expected joint EVENT COUNT, not just the
        # rate. agreement_ratio is uninterpretable without it. At n~130 with
        # ~3% marginals, independence predicts <1 joint hit, so an observed
        # zero -- and a ratio of 0.00 -- is what independence looks like, not
        # evidence the screens anti-align. Anything under ~5 expected events
        # means the ratio measures nothing and must not be narrated.
        exp_events = both.height * expected
        rec = {
            "lead_weeks": lead,
            "n_pairs": both.height,
            "target_in_top25": 100 * pt,
            "buyer_in_top25": 100 * pb,
            "joint": 100 * joint,
            "expected_if_independent": 100 * expected,
            "expected_joint_events": exp_events,
            "underpowered": bool(exp_events < 5),
            "agreement_ratio": joint / expected if expected else 0.0,
            "chance_joint": 100 * chance,
            "joint_lift_vs_chance": joint / chance if chance else 0.0,
        }
        out.append(rec)
        print(f"lead {lead:>2}w  n={both.height:>5}  "
              f"target {100 * pt:>5.1f}%  buyer {100 * pb:>5.1f}%  "
              f"joint {100 * joint:>5.2f}%  "
              f"(independent would give {100 * expected:.2f}%, "
              f"ratio {rec['agreement_ratio']:.2f}x)"
              + (f"  [UNDERPOWERED: independence predicts only "
                 f"{exp_events:.2f} joint events]" if exp_events < 5 else ""),
              flush=True)

    # Rank correlation between the two scores across real pairs: do good
    # targets get matched to good buyers, beyond the top-25 cutoff?
    lead4 = pairs.with_columns(
        (pl.col("first_ts").cast(pl.Date)
         - pl.duration(weeks=4)).dt.truncate("1w").alias("week"))
    j = lead4.join(ranked.select(["cik", "week", "p_target"]),
                   left_on=["target_cik", "week"],
                   right_on=["cik", "week"], how="inner") \
             .join(ranked.select(["cik", "week", "p_buyer"]),
                   left_on=["acquirer_cik", "week"],
                   right_on=["cik", "week"], how="inner")
    pt_v, pb_v = j["p_target"].to_numpy(), j["p_buyer"].to_numpy()
    corr = float(np.corrcoef(pt_v, pb_v)[0, 1]) if j.height else 0.0

    # DEVIATION from plan: a permutation null for that correlation. Constraint
    # 6 -- naive errors here measured 4x too small. Real pairs cluster in weeks
    # where BOTH screens run hot, so a positive correlation appears with no
    # pair-specific information at all. Reassigning acquirers among pairs seen
    # in the SAME week holds that composition fixed and isolates the part that
    # is about the actual transaction.
    null, wk = [], j["week"].to_numpy()
    if j.height:
        rng = np.random.default_rng(0)
        idx = np.arange(j.height)
        for _ in range(2000):
            perm = idx.copy()
            for w in np.unique(wk):
                m = np.flatnonzero(wk == w)
                if len(m) > 1:
                    perm[m] = rng.permutation(m)
            null.append(np.corrcoef(pt_v, pb_v[perm])[0, 1])
    null = np.array(null) if null else np.zeros(1)
    p_perm = float(np.mean(np.abs(null) >= abs(corr)))

    print(f"\ncorr(target score, buyer score) across {j.height} real pairs "
          f"at 4w lead: {corr:+.3f}")
    print(f"within-week permutation null: mean {null.mean():+.3f} "
          f"sd {null.std():.3f}  ->  p = {p_perm:.3f}")

    json.dump({"leads": out, "pair_score_corr_4w": corr, "n_corr": j.height,
               "corr_perm_null_mean": float(null.mean()),
               "corr_perm_null_sd": float(null.std()),
               "corr_perm_p": p_perm},
              open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
