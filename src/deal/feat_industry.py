"""Industry-relative features.

The takeover-prediction literature is consistent on this and I had ignored it:
raw accounting ratios underperform *industry-relative* ones. Barnes tested the
comparison directly on UK targets; Cremers, Nair & John (2009) put
book-to-market at the centre of their model; Palepu's "industry disturbance"
and Powell's sector controls are the same instinct.

The logic is simple. A 40% debt ratio means something different for a utility
than for a software firm, and a raw threshold cannot express that. Subtracting
the industry-week median lets one split say "levered *for its sector*".

Everything here is contemporaneous cross-sectional arithmetic -- the median at
week t uses only week-t data -- so no lookahead is introduced. Industries are
2-digit SIC, which keeps buckets populated; 4-digit leaves too many companies
alone in a group with a median equal to themselves.
"""
import polars as pl

# Variables the literature relativises. Each becomes rel_<name>.
RELATIVE_BASE = [
    "float_to_assets",   # book-to-market -- Cremers et al.'s central variable
    "log_assets",        # size relative to sector, not in absolute dollars
    "log_float",
    "roa",               # "inefficient management", judged against peers
    "leverage",
    "cash_to_assets",
    "revenue_growth",
    "tangible_ratio",    # Ambrose & Megginson, sector-adjusted
    "fcf_to_assets",
]

REL_COLS = [f"rel_{c}" for c in RELATIVE_BASE] + ["ind_hhi", "ind_size"]

_MIN_PEERS = 5


def add(df: pl.DataFrame) -> pl.DataFrame:
    """Attach industry-relative deviations plus two industry descriptors."""
    if "sic" not in df.columns:
        return df.with_columns([pl.lit(0.0).alias(c) for c in REL_COLS])

    # 2-digit SIC: enough companies per bucket for a median to mean anything.
    d = df.with_columns(
        pl.col("sic").cast(pl.Utf8).str.slice(0, 2).alias("sic2"))

    present = [c for c in RELATIVE_BASE if c in d.columns]
    med = (d.group_by(["sic2", "week"])
             .agg([pl.col(c).median().alias(f"__m_{c}") for c in present]
                  + [pl.len().alias("__n"),
                     pl.col("log_assets").mean().alias("ind_size")]))
    d = d.join(med, on=["sic2", "week"], how="left")

    # A median computed from fewer than _MIN_PEERS companies is mostly the
    # company itself, so the deviation would be structurally near zero.
    enough = pl.col("__n") >= _MIN_PEERS
    d = d.with_columns([
        pl.when(enough).then(pl.col(c) - pl.col(f"__m_{c}"))
          .otherwise(0.0).alias(f"rel_{c}")
        for c in present
    ])
    for c in RELATIVE_BASE:
        if f"rel_{c}" not in d.columns:
            d = d.with_columns(pl.lit(0.0).alias(f"rel_{c}"))

    # Herfindahl concentration by assets. Cremers et al. use industry
    # concentration; a fragmented sector consolidates, a concentrated one
    # cannot without attracting a regulator.
    hhi = (d.with_columns(pl.col("assets").fill_null(0.0).clip(0, None))
             .group_by(["sic2", "week"])
             .agg((((pl.col("assets") / pl.col("assets").sum().clip(1, None))
                    ** 2).sum()).alias("ind_hhi")))
    d = d.join(hhi, on=["sic2", "week"], how="left")

    drop = [c for c in d.columns if c.startswith("__")] + ["sic2"]
    return d.drop(drop).with_columns(
        [pl.col(c).fill_null(0.0) for c in REL_COLS])
