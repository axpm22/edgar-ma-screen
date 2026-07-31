"""Train/test splits that do not leak.

A random row split is WRONG here and would badly overstate performance: a
company contributes ~550 weekly rows and a single deal spans 26 positive
weeks, so random assignment puts the same deal on both sides and lets the
model memorise it.

grouped()  -- the 90/10 split, partitioned by COMPANY. Answers "does this
              generalise to companies I have never seen?"
by_time()  -- chronological. Answers "does this generalise forward?", which is
              the only split that supports a forecasting claim.

Report both. If grouped scores far above time-ordered, the model is
regime-dependent rather than predictive.
"""
import datetime as dt
import hashlib

import polars as pl


def grouped(df: pl.DataFrame, test_frac: float = 0.1,
            seed: int = 20260729) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out test_frac of COMPANIES (not rows).

    Assignment hashes the cik with the seed: deterministic, needs no shuffle
    of a 4M-row frame, and stable if rows are added later.
    """
    def bucket(cik: str) -> float:
        h = hashlib.sha256(f"{seed}:{cik}".encode()).hexdigest()[:8]
        return int(h, 16) / 0xFFFFFFFF

    test_ciks = [c for c in df["cik"].unique().to_list() if bucket(c) < test_frac]
    mask = pl.col("cik").is_in(test_ciks)
    return df.filter(~mask), df.filter(mask)


def by_time(df: pl.DataFrame,
            cutoff: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    return df.filter(pl.col("week") < cutoff), df.filter(pl.col("week") >= cutoff)
