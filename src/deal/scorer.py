"""Score a company's probability of being in a deal within 12 months.

Three things make a score useful rather than decorative:

1. A RANK, because raw probabilities from a 2.9%-base-rate model are all small
   and look alarming or reassuring for the wrong reasons.
2. An EMPIRICAL band -- what fraction of companies historically at this rank
   actually had a deal. That comes from the held-out test period, so the
   number quoted to a user is measured, not asserted.
3. A REASON. LightGBM's pred_contrib gives per-feature contributions, so a
   score can say "high because a new activist filed and the company disclosed
   a strategic review" instead of just "87".

The model is the validated configuration: 52-week horizon, no per-company
z-scores, LightGBM binary. See docs/FINAL_RESULTS.md.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

MODEL_PATH = Path("data/scorer.txt")
META_PATH = Path("data/scorer_meta.json")

# Rank buckets, expressed as top-N per week. The precision attached to each is
# measured on the held-out test period, not assumed.
BANDS = [10, 25, 50, 100, 200, 500]


def save(booster: lgb.Booster, cols: list[str], meta: dict) -> None:
    booster.save_model(str(MODEL_PATH))
    META_PATH.write_text(json.dumps({**meta, "cols": cols}, indent=2,
                                    default=str))


def load() -> tuple[lgb.Booster, list[str], dict]:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    meta = json.loads(META_PATH.read_text())
    return booster, meta["cols"], meta


def band_for_rank(rank: int, meta: dict) -> dict:
    """Empirical hit rate for a company at this within-week rank."""
    curve = meta.get("precision_curve", {})
    for n in BANDS:
        if rank <= n and str(n) in curve:
            return {"band": f"top {n}", "historical_hit_rate": curve[str(n)]}
    return {"band": f"outside top {BANDS[-1]}",
            "historical_hit_rate": meta.get("base_rate", 0.0)}


def asof_cross_section(df: pl.DataFrame, asof: "dt.date | None" = None,
                       stale_weeks: int = 26) -> pl.DataFrame:
    """One row per still-active company: its latest row at or before `asof`.

    Taking a single literal week does not work. Each company's panel ends at
    its last periodic filing, so the final week of the panel contains only the
    handful of firms that happened to file that week -- 28 of 14,680. This
    instead takes each company's most recent observation and drops any company
    whose newest data is more than `stale_weeks` old, which is the practical
    definition of "still listed and reporting".
    """
    import datetime as dt  # local: only needed for the default

    asof = asof or df["week"].max()
    upto = df.filter(pl.col("week") <= asof)
    latest = (upto.sort("week")
              .group_by("cik", maintain_order=False)
              .last())
    cutoff = asof - dt.timedelta(weeks=stale_weeks)
    return latest.filter(pl.col("week") >= cutoff)


def score_week(booster: lgb.Booster, cols: list[str], meta: dict,
               week_df: pl.DataFrame) -> pl.DataFrame:
    """Score a cross-section of companies and rank them against each other."""
    p = np.asarray(booster.predict(week_df.select(cols).to_pandas()))
    base = meta.get("base_rate", 0.0294)
    out = week_df.select(["cik", "week"]).with_columns([
        pl.Series("prob", p),
        pl.Series("lift", p / base if base else p),
    ])
    out = out.sort("prob", descending=True).with_row_index("rank", offset=1)
    n = out.height
    return out.with_columns(
        (100.0 * (1.0 - (pl.col("rank") - 1) / max(n - 1, 1))).alias("score")
    )


def explain(booster: lgb.Booster, cols: list[str], row: pl.DataFrame,
            top_k: int = 6) -> list[tuple[str, float]]:
    """Per-feature contributions to this row's score, largest first.

    pred_contrib returns len(cols)+1 values; the final one is the base value.
    """
    contrib = booster.predict(row.select(cols).to_pandas(), pred_contrib=True)
    vals = np.asarray(contrib)[0][:-1]
    order = np.argsort(-np.abs(vals))[:top_k]
    return [(cols[i], float(vals[i])) for i in order]
