"""Features taken from the established takeover-target literature.

These are the variables prior work found predictive, reconstructed from free
XBRL data. They belong in the model for two reasons: they should help, and
they raise the bar the novel filing-based signals have to clear. A finding
that survives against Palepu's and Ambrose & Megginson's variables is worth
far more than one that only beats a naive baseline.

Palepu (1986), "Predicting takeover targets":
  * growth-resource mismatch -- high growth with thin financial means invites
    a bidder who can correct the imbalance
  * management inefficiency  -- poor returns invite replacement
  * dividend payout, operating cash flow to assets, free cash flow to sales
  * size, market-to-book, industry disturbance (already in the model)

Ambrose & Megginson (1992), asset and ownership structure:
  * tangible assets POSITIVELY related to bid likelihood -- collateral
  * blank-cheque preferred stock, the one defense that actually deterred bids
"""
import polars as pl

LIT_COLS = [
    "tangible_ratio", "pref_stock_ratio", "roa", "ocf_to_assets",
    "fcf_to_assets", "dividend_payout", "capex_intensity",
    "palepu_mismatch", "cash_runway", "debt_wall", "buyback_halt",
]

_EPS = 1e5


def _safe(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den.abs() > _EPS).then(num / den.abs()).otherwise(None)


def add(df: pl.DataFrame) -> pl.DataFrame:
    a = pl.col("assets")
    out = df.with_columns([
        _safe(pl.col("ppe"), a).alias("tangible_ratio"),
        _safe(pl.col("pref_stock"), a).alias("pref_stock_ratio"),
        _safe(pl.col("net_income"), a).alias("roa"),
        _safe(pl.col("ocf"), a).alias("ocf_to_assets"),
        _safe(pl.col("ocf") - pl.col("capex"), a).alias("fcf_to_assets"),
        _safe(pl.col("dividends"), a).alias("dividend_payout"),
        _safe(pl.col("capex"), a).alias("capex_intensity"),
        # Cash divided by the burn rate, in years. Only meaningful when
        # operating cash flow is NEGATIVE -- a profitable firm is not running
        # out of anything, so it gets the capped value rather than a nonsense
        # ratio.
        pl.when(pl.col("ocf") < 0)
          .then((pl.col("cash") / pl.col("ocf").abs()).clip(0, 10))
          .otherwise(10.0).alias("cash_runway"),
        _safe(pl.col("lt_debt_current"), a).alias("debt_wall"),
    ])

    # Palepu's mismatch: growing fast on thin resources. Built as a continuous
    # score rather than his dummy -- a tree can find its own threshold, and a
    # hard cut throws information away.
    out = out.with_columns(
        (pl.col("revenue_growth").fill_null(0)
         - pl.col("ocf_to_assets").fill_null(0)
         + pl.col("leverage").fill_null(0)).alias("palepu_mismatch")
    )

    # Buybacks stop during deal negotiations. Another ABSENCE signal, which is
    # the shape that has worked best in this model so far.
    out = out.with_columns(
        ((pl.col("buyback_ly").fill_null(0) > 0)
         & (pl.col("buyback").fill_null(0) == 0)).cast(pl.Int8)
        .alias("buyback_halt")
    )
    return out
