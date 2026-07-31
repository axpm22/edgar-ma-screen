"""Restrict the panel to plausible acquisition targets.

The raw panel holds 15,325 EDGAR filers: shells, trusts, blank-cheque vehicles
and non-operating entities that are never acquisition targets and that nobody
would screen. Removing them raises the base rate from 1.46% to about 2.35%.

Be honest about what this is: a population change, not a better model. It
raises screen precision legitimately, but the lift number it produces is not
comparable to the unfiltered one.
"""
import polars as pl

# log1p dollars. 17.0 ~= $24M -- below this a listing is not a real target.
MIN_LOG_FLOAT = 17.0
MIN_LOG_ASSETS = 17.0


def is_operating(df: pl.DataFrame) -> pl.Series:
    return (df["log_float"] > MIN_LOG_FLOAT) & (df["log_assets"] > MIN_LOG_ASSETS)


def apply(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(is_operating(df))
