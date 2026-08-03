"""Macro regime features: credit conditions, volatility, and who is president.

READ THIS BEFORE EXPECTING THESE TO RAISE PRECISION.

Every column here is CONSTANT ACROSS COMPANIES within a given week. The screen
is evaluated by ranking companies WITHIN each week (screen.weekly_precision),
and a feature identical for all 8,000 companies in a week cannot change their
relative order. The main effect on precision@25 is therefore exactly zero, by
construction, not by weakness of the signal.

They can only move the metric through INTERACTIONS -- a tree splitting on
credit spread and then on company size, i.e. "when financing is expensive, the
kind of company that gets bought changes". That is a real mechanism and worth
testing, but it is a second-order effect and the panel already carries `year`
and `quarter`, with which these are heavily collinear.

What macro variables are genuinely FOR here is the between-year question: the
model scores 19.2% in 2021 and 6.7% in 2022, and nothing in the feature set
explains why. A regime variable that predicts WHEN the screen works is more
useful than one that marginally reorders a single week.

LOOKAHEAD, and why USREC is not here:
  NBER dates recessions retrospectively -- the committee declared the COVID
  recession's April 2020 trough in July 2021, fifteen months later. Using a
  recession flag at week W is knowledge nobody had at W. It is excluded rather
  than lagged, because the lag is irregular and unknowable in advance.

  Daily market series (BAA10Y, T10Y2Y, VIXCLS) are known same-day and are
  safe. UNRATE is published in the first week of the FOLLOWING month, so it
  carries a one-month lag here.
"""
import datetime as dt

import polars as pl

from . import fetch

FRED = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id={sid}&cosd={start}&coed={end}")

# Daily, market-priced, known same day.
DAILY = {
    "mac_credit_spread": "BAA10Y",   # Baa corporate over 10y Treasury
    "mac_yield_curve": "T10Y2Y",     # 10y minus 2y
    "mac_vix": "VIXCLS",             # implied volatility
}
# Monthly, published with a lag.
MONTHLY = {"mac_unrate": "UNRATE"}
MONTHLY_LAG_DAYS = 35

MACRO_COLS = (list(DAILY) + list(MONTHLY)
              + ["mac_pres_rep", "mac_credit_spread_d26", "mac_vix_d26"])

# Party holding the presidency, by inauguration date. Public record.
# 1 = Republican, 0 = Democrat.
PRESIDENTS = [
    (dt.date(2009, 1, 20), 0),
    (dt.date(2017, 1, 20), 1),
    (dt.date(2021, 1, 20), 0),
    (dt.date(2025, 1, 20), 1),
]


def president_republican(d: dt.date) -> int:
    party = PRESIDENTS[0][1]
    for start, p in PRESIDENTS:
        if d >= start:
            party = p
    return party


def _series(sid: str, start: dt.date, end: dt.date) -> pl.DataFrame:
    url = FRED.format(sid=sid, start=start, end=end)
    raw = fetch.cached("fred", url, lambda: _get(url))
    df = pl.read_csv(raw, null_values=[".", ""])
    cols = df.columns
    return (df.rename({cols[0]: "date", cols[1]: "value"})
            .with_columns([pl.col("date").str.to_date(),
                           pl.col("value").cast(pl.Float64)])
            .drop_nulls())


def _get(url: str) -> bytes:
    import httpx
    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.content


def build(weeks: list[dt.date]) -> pl.DataFrame:
    """One row per week. Values are the last observation ON OR BEFORE the week.

    Backward as-of join, never forward: a Monday's row may only see data
    published by that Monday.
    """
    lo, hi = min(weeks), max(weeks)
    out = pl.DataFrame({"week": sorted(set(weeks))}).with_columns(
        pl.col("week").cast(pl.Date))

    for name, sid in {**DAILY, **MONTHLY}.items():
        s = _series(sid, lo - dt.timedelta(days=400), hi)
        if name in MONTHLY:
            s = s.with_columns(
                (pl.col("date")
                 + pl.duration(days=MONTHLY_LAG_DAYS)).alias("date"))
        s = s.rename({"value": name}).sort("date")
        out = out.sort("week").join_asof(
            s, left_on="week", right_on="date", strategy="backward").drop("date")

    out = out.with_columns(
        pl.col("week").map_elements(president_republican,
                                    return_dtype=pl.Int8).alias("mac_pres_rep"))
    # Change matters more than level: a widening spread is the signal, and a
    # level is nearly collinear with `year`.
    return out.with_columns([
        (pl.col("mac_credit_spread")
         - pl.col("mac_credit_spread").shift(26)).alias("mac_credit_spread_d26"),
        (pl.col("mac_vix") - pl.col("mac_vix").shift(26)).alias("mac_vix_d26"),
    ]).with_columns([pl.col(c).fill_null(strategy="forward").fill_null(0.0)
                     for c in MACRO_COLS])


def add(df: pl.DataFrame) -> pl.DataFrame:
    """Attach macro columns to a company-week frame."""
    macro = build(df["week"].unique().to_list())
    return df.join(macro, on="week", how="left").with_columns(
        [pl.col(c).fill_null(0.0) for c in MACRO_COLS])
