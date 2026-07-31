"""All sources -> as-of feature matrix.

This is the only module permitted to join anything onto a panel row. Every
join is on `public_ts <= week` (ASOF or a trailing window), which is the
lookahead firewall: a fact is visible in week t only if it was public on or
before that Monday. One module, one rule, one place to test it.

Design note on normalisation. Raw counts are largely a size proxy -- a mega-cap
files more 8-Ks every single week, so `form8k_26w` was substantially restating
`log_assets`. Every count feature therefore also gets a *_z variant: the count
minus that company's own trailing mean, over its own trailing SD. That turns
"this is a big company" into "this is unusual for this company", which is the
thing actually worth predicting on.

USPTO trademark signals are deliberately absent: the Open Data Portal gates API
keys behind ID.me (government ID + SSN), and 13D/13G from EDGAR is the same
"a registry reveals preparation" idea -- already on disk, better documented,
and exactly timestamped.
"""
import polars as pl

from .feat_industry import REL_COLS
from .feat_industry import add as add_industry
from .feat_literature import LIT_COLS
from .feat_literature import add as add_literature
from .feat_items import ITEM_COLS
from .feat_items import prepare as prepare_items
from .feat_forms import FORM_COLS
from .feat_forms import prepare as prepare_forms

# --- fundamentals -----------------------------------------------------------
FUND_COLS = [
    "log_assets", "cash_to_assets", "leverage", "rnd_intensity",
    "revenue_growth", "asset_growth",
    # Added tags: goodwill signals prior acquisitiveness, margins give
    # profitability, and a loss-making firm is a different kind of target.
    "goodwill_to_assets", "intangibles_to_assets", "operating_margin",
    "net_margin", "lt_debt_to_assets",
]

# --- market (SEC public float; no free daily source covers delistings) ------
MARKET_COLS = ["log_float", "float_to_assets", "float_growth"]

# --- insider ----------------------------------------------------------------
INSIDER_COLS = [
    "disc_52w", "disc_sells_26w", "disc_buys_26w", "plan_trades_26w",
    "weeks_since_disc", "disc_blackout",
    # Dollar-weighted: a CEO selling $40M and an officer selling $8k are not
    # the same event, which the count features could not distinguish.
    "usd_sold_26w_scaled", "usd_bought_26w_scaled", "usd_net_26w_scaled",
]

# --- strategic intent (EDGAR full-text search) ------------------------------
# The only family where the company states intent rather than leaving a trace.
FTS_COLS = ["sa_review_52w", "sa_sale_52w", "sa_unsolicited_52w", "sa_loi_52w"]

# --- registry / network -----------------------------------------------------
SIGNAL_COLS = ["ct"]
ACTIVIST_COLS = ["activist_reach", "activist_recent"]
PEER_COLS = ["peer_deal_13w", "sector_deal_intensity"]

# --- per-company normalised variants ---------------------------------------
ZSCORE_BASE = ["form8k_26w", "sc13g_52w", "sc13d_52w", "disc_sells_26w",
               "plan_trades_26w"]
ZSCORE_COLS = [f"{c}_z" for c in ZSCORE_BASE]

# --- deltas (acceleration beats level for precursor signals) ----------------
DELTA_BASE = ["form8k_26w", "disc_sells_26w", "log_float"]
DELTA_COLS = [f"{c}_d" for c in DELTA_BASE]

CONTEXT_COLS = ["age_weeks", "quarter", "year"]

# REL_COLS deliberately excluded -- measured -1.6pp in both universes.
# The literature found industry-relative ratios beat raw ones using LINEAR
# models, which cannot condition on industry any other way. A tree learns
# that interaction natively, so pre-computing it only adds 11 noisy
# dimensions. Kept importable for the write-up.
FEATURE_COLS = (FUND_COLS + LIT_COLS + MARKET_COLS + INSIDER_COLS + FORM_COLS + ITEM_COLS
                + FTS_COLS + SIGNAL_COLS + ACTIVIST_COLS + PEER_COLS
                + ZSCORE_COLS + DELTA_COLS + CONTEXT_COLS)

_DECAY_HALF_LIFE_WEEKS = 8
_MIN_DENOM = 1e5
_MIN_PRIOR_TRADES = 4
_WINSOR = (0.01, 0.99)
_RATIO_COLS = [
    "cash_to_assets", "leverage", "rnd_intensity", "revenue_growth",
    "asset_growth", "float_to_assets", "float_growth",
    "goodwill_to_assets", "intangibles_to_assets", "operating_margin",
    "net_margin", "lt_debt_to_assets",
    "usd_sold_26w_scaled", "usd_bought_26w_scaled", "usd_net_26w_scaled",
    "tangible_ratio", "pref_stock_ratio", "roa", "ocf_to_assets",
    "fcf_to_assets", "dividend_payout", "capex_intensity",
    "palepu_mismatch", "debt_wall",
]

# Trailing window for the per-company baseline behind the *_z features.
_BASELINE_WEEKS = 104


def _table_exists(con, name: str) -> bool:
    return bool(con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [name]).fetchone()[0])


def _prepare(con) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE fund_wide AS
        SELECT cik, public_ts,
               max(CASE WHEN tag='Assets' THEN value END) AS assets,
               max(CASE WHEN tag='CashAndCashEquivalentsAtCarryingValue'
                        THEN value END) AS cash,
               max(CASE WHEN tag='Liabilities' THEN value END) AS liabilities,
               max(CASE WHEN tag='Revenues' THEN value END) AS revenue,
               max(CASE WHEN tag='ResearchAndDevelopmentExpense'
                        THEN value END) AS rnd,
               max(CASE WHEN tag='Goodwill' THEN value END) AS goodwill,
               max(CASE WHEN tag='IntangibleAssetsNetExcludingGoodwill'
                        THEN value END) AS intangibles,
               max(CASE WHEN tag='OperatingIncomeLoss' THEN value END) AS op_inc,
               max(CASE WHEN tag='NetIncomeLoss' THEN value END) AS net_inc,
               max(CASE WHEN tag='LongTermDebtNoncurrent'
                        THEN value END) AS lt_debt,
               max(CASE WHEN tag='PropertyPlantAndEquipmentNet'
                        THEN value END) AS ppe,
               max(CASE WHEN tag='PreferredStockValue'
                        THEN value END) AS pref_stock,
               max(CASE WHEN tag='NetCashProvidedByUsedInOperatingActivities'
                        THEN value END) AS ocf,
               max(CASE WHEN tag='PaymentsToAcquirePropertyPlantAndEquipment'
                        THEN value END) AS capex,
               max(CASE WHEN tag='PaymentsOfDividendsCommonStock'
                        THEN value END) AS dividends,
               max(CASE WHEN tag='PaymentsForRepurchaseOfCommonStock'
                        THEN value END) AS buyback,
               max(CASE WHEN tag='LongTermDebtCurrent'
                        THEN value END) AS lt_debt_current
        FROM fundamentals GROUP BY cik, public_ts
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE fund_growth AS
        SELECT a.*, b.assets AS assets_ly, b.revenue AS revenue_ly,
               b.buyback AS buyback_ly, b.capex AS capex_ly
        FROM fund_wide a
        ASOF LEFT JOIN fund_wide b
          ON a.cik = b.cik AND a.public_ts - INTERVAL 365 DAY >= b.public_ts
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ins_week AS
        SELECT cik, date_trunc('week', public_ts) AS week,
               count(*) FILTER (WHERE discretionary AND trans_code='S') AS sells,
               count(*) FILTER (WHERE discretionary AND trans_code='P') AS buys,
               count(*) FILTER (WHERE NOT discretionary)                AS plan
        FROM insider_trans GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE float_growth AS
        SELECT a.cik, a.public_ts, a.value AS float_val, b.value AS float_val_ly
        FROM public_float a
        ASOF LEFT JOIN public_float b
          ON a.cik = b.cik AND a.public_ts - INTERVAL 365 DAY >= b.public_ts
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE sic_intensity AS
        SELECT s.sic, year(d.agreement_date) + 1 AS year, count(*) AS n
        FROM deals d JOIN company_sic s USING (cik)
        GROUP BY 1, 2
    """)
    # Peer effect with real timing: consolidation cascades run in months, and a
    # SIC+year count cannot see that. The window ends one week back so a
    # company's own deal can never enter its own feature.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE peer_roll AS
        SELECT sic, week,
               sum(n) OVER (PARTITION BY sic ORDER BY week
                            RANGE BETWEEN INTERVAL 91 DAY PRECEDING
                                      AND INTERVAL 7 DAY PRECEDING) AS peer_deal_13w
        FROM (
            SELECT s.sic, date_trunc('week', d.agreement_date) AS week,
                   count(*) AS n
            FROM deals d JOIN company_sic s USING (cik)
            GROUP BY 1, 2
        )
    """)
    prepare_forms(con)
    if _table_exists(con, 'item_events'):
        prepare_items(con)

    # Optional sources: an absent table becomes all-zero columns rather than a
    # crash, so the matrix builds before every loader has finished.
    if _table_exists(con, "insider_value"):
        con.execute("""
            CREATE OR REPLACE TEMP TABLE insval_roll AS
            SELECT cik, week,
                   sum(usd_sold)   OVER w AS usd_sold_26w,
                   sum(usd_bought) OVER w AS usd_bought_26w
            FROM insider_value
            WINDOW w AS (PARTITION BY cik ORDER BY week
                         RANGE BETWEEN INTERVAL 182 DAY PRECEDING AND CURRENT ROW)
        """)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE insval_roll AS SELECT "
                    "''::VARCHAR cik, DATE '1970-01-01' week, "
                    "0.0::DOUBLE usd_sold_26w, 0.0::DOUBLE usd_bought_26w")

    if _table_exists(con, "fts_events"):
        con.execute("""
            CREATE OR REPLACE TEMP TABLE fts_roll AS
            SELECT cik, week,
                   sum(sa_review)      OVER w AS sa_review_52w,
                   sum(sa_sale)        OVER w AS sa_sale_52w,
                   sum(sa_unsolicited) OVER w AS sa_unsolicited_52w,
                   sum(sa_loi)         OVER w AS sa_loi_52w
            FROM (
                SELECT cik, date_trunc('week', public_ts) AS week,
                       sum(CASE WHEN family='sa_review' THEN n ELSE 0 END) sa_review,
                       sum(CASE WHEN family='sa_sale' THEN n ELSE 0 END) sa_sale,
                       sum(CASE WHEN family='sa_unsolicited' THEN n ELSE 0 END) sa_unsolicited,
                       sum(CASE WHEN family='sa_loi' THEN n ELSE 0 END) sa_loi
                FROM fts_events GROUP BY 1, 2
            )
            WINDOW w AS (PARTITION BY cik ORDER BY week
                         RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND CURRENT ROW)
        """)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE fts_roll AS SELECT "
                    "''::VARCHAR cik, DATE '1970-01-01' week, "
                    "0.0::DOUBLE sa_review_52w, 0.0::DOUBLE sa_sale_52w, "
                    "0.0::DOUBLE sa_unsolicited_52w, 0.0::DOUBLE sa_loi_52w")

    if _table_exists(con, "activist_events"):
        con.execute("""
            CREATE OR REPLACE TEMP TABLE act_roll AS
            SELECT cik, date_trunc('week', public_ts) AS week,
                   max(filer_targets)::DOUBLE AS activist_reach
            FROM activist_events GROUP BY 1, 2
        """)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE act_roll AS SELECT "
                    "''::VARCHAR cik, DATE '1970-01-01' week, "
                    "0.0::DOUBLE activist_reach")


def build(con, decay_weeks: int = _DECAY_HALF_LIFE_WEEKS) -> pl.DataFrame:
    _prepare(con)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE feat AS
        WITH base AS (
            SELECT p.cik, p.week, p.y,
                   date_diff('week', u.listed, p.week) AS age_weeks,
                   quarter(p.week) AS quarter, year(p.week) AS year, c.sic
            FROM panel p
            JOIN universe u USING (cik)
            LEFT JOIN company_sic c USING (cik)
        ),
        ins_roll AS (
            SELECT cik, week,
                   sum(sells) OVER w26 AS disc_sells_26w,
                   sum(buys)  OVER w26 AS disc_buys_26w,
                   sum(plan)  OVER w26 AS plan_trades_26w,
                   sum(sells + buys) OVER w52 AS disc_52w,
                   max(CASE WHEN sells+buys > 0 THEN week END) OVER w26
                       AS last_disc_week
            FROM ins_week
            WINDOW w26 AS (PARTITION BY cik ORDER BY week
                           RANGE BETWEEN INTERVAL 182 DAY PRECEDING AND CURRENT ROW),
                   w52 AS (PARTITION BY cik ORDER BY week
                           RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND CURRENT ROW)
        )
        SELECT b.cik, b.week, b.y, b.age_weeks, b.quarter, b.year, b.sic,
               f.assets, f.cash, f.liabilities, f.revenue, f.rnd,
               f.goodwill, f.intangibles, f.op_inc, f.net_inc, f.lt_debt,
               f.assets_ly, f.revenue_ly,
               f.ppe, f.pref_stock, f.ocf, f.capex, f.dividends, f.buyback,
               f.lt_debt_current, f.buyback_ly, f.capex_ly,
               f.net_inc AS net_income,
               coalesce(ir.disc_sells_26w, 0)  AS disc_sells_26w,
               coalesce(ir.disc_buys_26w, 0)   AS disc_buys_26w,
               coalesce(ir.plan_trades_26w, 0) AS plan_trades_26w,
               coalesce(ir.disc_52w, 0)        AS disc_52w,
               date_diff('week', ir.last_disc_week, b.week) AS weeks_since_disc,
               coalesce(si.n, 0)               AS sector_deal_intensity,
               coalesce(pr.peer_deal_13w, 0)   AS peer_deal_13w,
               fl.float_val, fl.float_val_ly,
               coalesce(fr.sc13d_52w, 0)  AS sc13d_52w,
               coalesce(fr.sc13g_52w, 0)  AS sc13g_52w,
               coalesce(fr.form8k_26w, 0) AS form8k_26w,
               coalesce(fr.def14a_52w, 0) AS def14a_52w,
               coalesce(fr.s4_52w, 0)     AS s4_52w,
               coalesce(fr.late_52w, 0)   AS late_52w,
               coalesce(fr.sc13d_new, 0)  AS sc13d_new,
               coalesce(itm.i_officer_change_52w, 0) AS i_officer_change_52w,
               coalesce(itm.i_auditor_change_52w, 0) AS i_auditor_change_52w,
               coalesce(itm.i_nonreliance_52w, 0) AS i_nonreliance_52w,
               coalesce(itm.i_exit_costs_52w, 0) AS i_exit_costs_52w,
               coalesce(itm.i_impairment_52w, 0) AS i_impairment_52w,
               coalesce(itm.i_security_rights_52w, 0) AS i_security_rights_52w,
               coalesce(itm.i_bylaw_change_52w, 0) AS i_bylaw_change_52w,
               coalesce(itm.i_vote_results_52w, 0) AS i_vote_results_52w,
               coalesce(itm.i_agmt_termination_52w, 0) AS i_agmt_termination_52w,
               coalesce(itm.i_reg_fd_52w, 0) AS i_reg_fd_52w,
               coalesce(itm.i_other_events_52w, 0) AS i_other_events_52w,
               coalesce(iv.usd_sold_26w, 0)   AS usd_sold_26w,
               coalesce(iv.usd_bought_26w, 0) AS usd_bought_26w,
               coalesce(ft.sa_review_52w, 0)      AS sa_review_52w,
               coalesce(ft.sa_sale_52w, 0)        AS sa_sale_52w,
               coalesce(ft.sa_unsolicited_52w, 0) AS sa_unsolicited_52w,
               coalesce(ft.sa_loi_52w, 0)         AS sa_loi_52w,
               coalesce(ar.activist_reach, 0)     AS activist_reach
        FROM base b
        ASOF LEFT JOIN fund_growth f
          ON b.cik = f.cik AND b.week >= f.public_ts        -- firewall
        ASOF LEFT JOIN ins_roll ir
          ON b.cik = ir.cik AND b.week >= ir.week            -- firewall
        ASOF LEFT JOIN float_growth fl
          ON b.cik = fl.cik AND b.week >= fl.public_ts       -- firewall
        ASOF LEFT JOIN form_roll fr
          ON b.cik = fr.cik AND b.week >= fr.week            -- firewall
        ASOF LEFT JOIN item_roll itm
          ON b.cik = itm.cik AND b.week >= itm.week      -- firewall
        ASOF LEFT JOIN insval_roll iv
          ON b.cik = iv.cik AND b.week >= iv.week            -- firewall
        ASOF LEFT JOIN fts_roll ft
          ON b.cik = ft.cik AND b.week >= ft.week            -- firewall
        ASOF LEFT JOIN act_roll ar
          ON b.cik = ar.cik AND b.week >= ar.week            -- firewall
        LEFT JOIN sic_intensity si ON si.sic = b.sic AND si.year = b.year
        ASOF LEFT JOIN peer_roll pr
          ON b.sic = pr.sic AND b.week >= pr.week            -- firewall
    """)

    sig_rows = con.execute(f"""
        SELECT p.cik, p.week,
               sum(s.value * pow(0.5,
                   date_diff('day', s.public_ts, p.week) / 7.0 / {decay_weeks}))
        FROM panel p
        JOIN signals s ON s.cik = p.cik AND s.public_ts <= p.week  -- firewall
        WHERE s.kind = 'ct'
        GROUP BY p.cik, p.week
    """).fetchall()

    df = con.execute("SELECT * FROM feat").pl().sort(["cik", "week"])

    floor = pl.when(pl.col("assets") > _MIN_DENOM).then(pl.col("assets"))
    rev_f = pl.when(pl.col("revenue").abs() > _MIN_DENOM).then(pl.col("revenue").abs())
    rev_ly = pl.when(pl.col("revenue_ly").abs() > _MIN_DENOM).then(pl.col("revenue_ly").abs())
    ast_ly = pl.when(pl.col("assets_ly").abs() > _MIN_DENOM).then(pl.col("assets_ly").abs())
    float_f = pl.when(pl.col("float_val") > _MIN_DENOM).then(pl.col("float_val"))
    float_ly = pl.when(pl.col("float_val_ly").abs() > _MIN_DENOM).then(pl.col("float_val_ly").abs())

    df = df.with_columns([
        pl.col("assets").log1p().alias("log_assets"),
        (pl.col("cash") / floor).alias("cash_to_assets"),
        (pl.col("liabilities") / floor).alias("leverage"),
        (pl.col("rnd") / rev_f).alias("rnd_intensity"),
        ((pl.col("revenue") - pl.col("revenue_ly")) / rev_ly).alias("revenue_growth"),
        ((pl.col("assets") - pl.col("assets_ly")) / ast_ly).alias("asset_growth"),
        (pl.col("goodwill") / floor).alias("goodwill_to_assets"),
        (pl.col("intangibles") / floor).alias("intangibles_to_assets"),
        (pl.col("op_inc") / rev_f).alias("operating_margin"),
        (pl.col("net_inc") / rev_f).alias("net_margin"),
        (pl.col("lt_debt") / floor).alias("lt_debt_to_assets"),
        pl.col("float_val").log1p().alias("log_float"),
        (pl.col("float_val") / floor).alias("float_to_assets"),
        ((pl.col("float_val") - pl.col("float_val_ly")) / float_ly).alias("float_growth"),
        # Dollar insider flow scaled by market value, so it is comparable
        # across a $200M company and a $200B one.
        (pl.col("usd_sold_26w") / float_f).alias("usd_sold_26w_scaled"),
        (pl.col("usd_bought_26w") / float_f).alias("usd_bought_26w_scaled"),
        ((pl.col("usd_sold_26w") - pl.col("usd_bought_26w")) / float_f)
        .alias("usd_net_26w_scaled"),
        pl.col("weeks_since_disc").fill_null(999).alias("weeks_since_disc"),
        (pl.col("activist_reach") > 0).cast(pl.Int8).alias("activist_recent"),
    ])

    df = df.with_columns(
        ((pl.col("disc_52w") >= _MIN_PRIOR_TRADES)
         & (pl.col("disc_sells_26w") + pl.col("disc_buys_26w") == 0))
        .cast(pl.Int8).alias("disc_blackout")
    )

    sig = pl.DataFrame(
        [{"cik": c, "week": w, "ct": float(v)} for c, w, v in sig_rows],
        schema={"cik": pl.Utf8, "week": pl.Date, "ct": pl.Float64})
    df = (df.join(sig, on=["cik", "week"], how="left") if sig.height
          else df.with_columns(pl.lit(0.0).alias("ct")))

    # Per-company z-scores and deltas. Both use ONLY trailing windows inside a
    # company's own history, so neither can see the future.
    df = df.sort(["cik", "week"]).with_columns([
        pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in ZSCORE_BASE
    ])
    # Literature variables (Palepu 1986; Ambrose & Megginson 1992). These
    # run after the base ratios because palepu_mismatch consumes
    # revenue_growth and leverage.
    df = add_literature(df)
    # Industry-relative deviations need the base ratios to exist first.
    df = add_industry(df)

    df = df.with_columns([
        ((pl.col(c) - pl.col(c).rolling_mean(_BASELINE_WEEKS, min_samples=8))
         / (pl.col(c).rolling_std(_BASELINE_WEEKS, min_samples=8) + 1e-6))
        .over("cik").alias(f"{c}_z")
        for c in ZSCORE_BASE
    ])
    df = df.with_columns([
        (pl.col(c).cast(pl.Float64) - pl.col(c).cast(pl.Float64).shift(26))
        .over("cik").alias(f"{c}_d")
        for c in DELTA_BASE
    ])

    for col in FEATURE_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Order matters: infs (division by a zero denominator) map to null FIRST,
    # then get filled. Filling first would leave infs and poison every fit.
    df = df.with_columns([
        pl.col(c).cast(pl.Float64)
        .replace([float("inf"), float("-inf")], None)
        .fill_nan(None).fill_null(0.0).alias(c)
        for c in FEATURE_COLS
    ])

    # REL_COLS are differences of ratios computed BEFORE winsorising, so
    # they inherit the raw tails and must be clipped too.
    clip_cols = _RATIO_COLS + ZSCORE_COLS + DELTA_COLS + REL_COLS
    lo_hi = {c: (df[c].quantile(_WINSOR[0]), df[c].quantile(_WINSOR[1]))
             for c in clip_cols}
    df = df.with_columns([
        pl.col(c).clip(lo_hi[c][0], lo_hi[c][1]).alias(c)
        for c in clip_cols
        if lo_hi[c][0] is not None and lo_hi[c][1] is not None
    ])
    return df.select(["cik", "week", "y", *FEATURE_COLS])
