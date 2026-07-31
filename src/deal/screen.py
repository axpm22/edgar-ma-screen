"""Per-week screen evaluation.

Global top-k over company-weeks is misleading: one company contributes ~550
rows, so the global top 100 turned out to be 3 distinct companies with one
occupying 84 consecutive weeks. That measures a single bet, not a hundred.

Ranking within each week matches how a screen is run -- every Monday, look at
the top N names -- and produces a coherent precision curve.
"""
import numpy as np
import polars as pl


def weekly_precision(df: pl.DataFrame, p: np.ndarray, n_per_week: int) -> dict:
    m = df.with_columns(pl.Series("p", np.asarray(p)))
    sel = m.sort("p", descending=True).group_by("week").head(n_per_week)
    base = float(m["y"].mean())
    precision = float(sel["y"].mean()) if sel.height else 0.0
    return {
        "n_per_week": n_per_week,
        "n_selected": sel.height,
        "distinct_companies": sel["cik"].n_unique(),
        "precision": precision,
        "base_rate": base,
        "lift": (precision / base) if base else 0.0,
    }


def curve(df: pl.DataFrame, p: np.ndarray,
          ns=(10, 25, 50, 100, 200)) -> pl.DataFrame:
    return pl.DataFrame([weekly_precision(df, p, n) for n in ns])
