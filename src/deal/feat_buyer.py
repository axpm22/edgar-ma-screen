"""Buyer-side features: who is preparing to acquire?

Built to the same discipline as the target model, with one addition the target
side did not need — an explicit split between features that describe what a
company IS and features that describe what it is about to DO.

That split matters here more than anywhere else in the project. The first
buyer model scored 16x lift with `s4_52w` as its top feature, while the label
was "files an S-4 next year". Trailing S-4 count predicting future S-4 count
is close to autocorrelation: it identifies acquisitive companies rather than
imminent acquisitions. Goodwill and intangibles have the same defect from the
other direction -- both accumulate FROM past deals.

SELF_REFERENTIAL names those. They are legitimate predictors of "is this an
acquisitive firm" and illegitimate as evidence the model forecasts events, so
every result is reported with and without them.

Genuinely forward-looking buyer signals, none of which encode past deals:

  shelf     S-3 registration -- dry powder. A shelf lets a company issue
            securities at short notice, which is what a stock-funded bid
            needs, and it is filed BEFORE any target is named.
  raise     424B prospectus supplements and free writing prospectuses --
            an offering actually executed, i.e. the powder is now cash.
  capacity  balance-sheet room to pay: cash, low leverage, free cash flow.
  cadence   8-K and proxy activity relative to the firm's own baseline.
"""
import polars as pl

# Trailing windows over the new form families.
BUYER_FORM_COLS = ["shelf_52w", "raise_52w", "shelf_new", "raise_burst"]

# Derived capacity measures.
BUYER_CAP_COLS = ["dry_powder", "debt_headroom", "acq_capacity"]

BUYER_COLS = BUYER_FORM_COLS + BUYER_CAP_COLS

# Predictors of "is an acquisitive company", not "is about to acquire".
# Reported separately; never counted as forecasting skill.
SELF_REFERENTIAL = ["s4_52w", "goodwill_to_assets", "intangibles_to_assets"]


def prepare(con) -> None:
    """Rolling shelf/raise counts, mirroring feat_forms.prepare."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE buyer_week AS
        SELECT cik, date_trunc('week', public_ts) AS week,
               sum(CASE WHEN family='shelf' THEN n ELSE 0 END) AS shelf,
               sum(CASE WHEN family='raise' THEN n ELSE 0 END) AS raise_n
        FROM form_events WHERE family IN ('shelf','raise')
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE buyer_roll AS
        SELECT cik, week,
               sum(shelf)   OVER w52 AS shelf_52w,
               sum(raise_n) OVER w52 AS raise_52w,
               -- A FIRST shelf in two years is the interesting event: a
               -- company newly giving itself the ability to issue paper.
               CASE WHEN shelf > 0
                     AND coalesce(sum(shelf) OVER wprior, 0) = 0
                    THEN 1 ELSE 0 END AS shelf_new,
               -- Offering activity well above the firm's own two-year norm.
               CASE WHEN sum(raise_n) OVER w26
                         > 2 * coalesce(avg(raise_n) OVER w104, 0) * 26
                    THEN 1 ELSE 0 END AS raise_burst
        FROM buyer_week
        WINDOW w52 AS (PARTITION BY cik ORDER BY week
                       RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND CURRENT ROW),
               w26 AS (PARTITION BY cik ORDER BY week
                       RANGE BETWEEN INTERVAL 182 DAY PRECEDING AND CURRENT ROW),
               w104 AS (PARTITION BY cik ORDER BY week
                        RANGE BETWEEN INTERVAL 730 DAY PRECEDING AND CURRENT ROW),
               wprior AS (PARTITION BY cik ORDER BY week
                          RANGE BETWEEN INTERVAL 730 DAY PRECEDING
                                    AND INTERVAL 7 DAY PRECEDING)
    """)


def add(df: pl.DataFrame) -> pl.DataFrame:
    """Balance-sheet capacity to fund a deal."""
    have = set(df.columns)
    cash = pl.col("cash_to_assets") if "cash_to_assets" in have else pl.lit(0.0)
    lev = pl.col("leverage") if "leverage" in have else pl.lit(0.0)
    fcf = pl.col("fcf_to_assets") if "fcf_to_assets" in have else pl.lit(0.0)

    out = df.with_columns([
        cash.alias("dry_powder"),
        # Room to borrow. Leverage is winsorised, so 1.0 is a sane ceiling.
        (1.0 - lev.clip(0, 1)).alias("debt_headroom"),
    ])
    # Combined: cash on hand, borrowing room, and cash generation. A buyer
    # needs at least one of the three.
    out = out.with_columns(
        (pl.col("dry_powder") + pl.col("debt_headroom") + fcf)
        .alias("acq_capacity")
    )
    for c in BUYER_COLS:
        if c not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(c))
    return out.with_columns([pl.col(c).fill_null(0.0) for c in BUYER_COLS])
