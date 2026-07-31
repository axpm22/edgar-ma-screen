"""Robust evaluation: bootstrap CIs, permutation tests, rolling-origin folds.

A single train/test boundary is one draw from a noisy process, and a config
chosen by its test score has already spent that test set. Everything here
exists to separate real signal from the maximum of many noisy draws.

The permutation test is the strongest single check: shuffle the labels and
refit. If the model still scores well, the pipeline is leaking rather than
predicting.
"""
import datetime as dt

import numpy as np
import polars as pl

from . import model_gbm, screen


def bootstrap_precision(df: pl.DataFrame, p: np.ndarray, n_per_week: int = 25,
                        n_boot: int = 400, seed: int = 20260729) -> dict:
    """Bootstrap a CI for per-week screen precision.

    Resampling is over WEEKS, not rows: rows within a week are selected
    jointly by the screen, so resampling rows independently would understate
    the variance.
    """
    m = df.with_columns(pl.Series("p", np.asarray(p)))
    sel = m.sort("p", descending=True).group_by("week").head(n_per_week)
    per_week = (sel.group_by("week")
                .agg(pl.col("y").sum().alias("hits"), pl.len().alias("n")))
    hits = per_week["hits"].to_numpy().astype(float)
    ns = per_week["n"].to_numpy().astype(float)

    rng = np.random.default_rng(seed)
    n_weeks = len(hits)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_weeks, n_weeks)
        draws[i] = hits[idx].sum() / max(ns[idx].sum(), 1.0)
    point = hits.sum() / max(ns.sum(), 1.0)
    return {
        "precision": float(point),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "n_weeks": int(n_weeks),
    }


def permutation_test(train: pl.DataFrame, test: pl.DataFrame, cols: list[str],
                     n_per_week: int = 25, n_perm: int = 10,
                     seed: int = 20260729, recency: bool = True) -> dict:
    """Refit on label-shuffled training data to build a null distribution.

    Labels are shuffled WITHIN week, preserving each week's positive count, so
    the null keeps the panel's time structure and only destroys the
    feature-label link.
    """
    b = model_gbm.fit(train, valid=test, cols=cols, recency=recency)
    real = screen.weekly_precision(test, model_gbm.predict(b, test, cols),
                                   n_per_week)["precision"]

    null = []
    for k in range(n_perm):
        shuffled = train.with_columns(
            pl.col("y").shuffle(seed=seed + k).over("week").alias("y"))
        bp = model_gbm.fit(shuffled, cols=cols, recency=recency)
        null.append(screen.weekly_precision(
            test, model_gbm.predict(bp, test, cols), n_per_week)["precision"])

    null = np.array(null)
    return {
        "real": float(real),
        "null_mean": float(null.mean()),
        "null_max": float(null.max()),
        "null_sd": float(null.std()),
        # With n_perm permutations the smallest attainable p is 1/(n_perm+1).
        "p_value": float((np.sum(null >= real) + 1) / (n_perm + 1)),
        "beats_all_null": bool(real > null.max()),
    }


def rolling_origin(df: pl.DataFrame,
                   cutoffs: list[dt.date]) -> list[tuple[dt.date, pl.DataFrame, pl.DataFrame]]:
    """Expanding-window folds: train on everything before the cutoff, test on
    the following 12 months. A config that only works at one boundary is a
    coincidence."""
    out = []
    for c in cutoffs:
        end = dt.date(c.year + 1, c.month, c.day)
        tr = df.filter(pl.col("week") < c)
        te = df.filter((pl.col("week") >= c) & (pl.col("week") < end))
        if te.height and te["y"].sum():
            out.append((c, tr, te))
    return out
