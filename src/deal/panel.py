"""Company-week panel and deal labelling.

A company appears from the first ISO week on or after it listed, through the
week containing its last periodic filing. Companies that left the sample MUST
remain -- they are where the positives live, and dropping them is the
survivorship bug that silently deletes the outcome being modelled.
"""
import datetime as dt

# DEFM14A only. The target files its own merger proxy, so the CIK is
# unambiguously the target. SC TO-T and SC 13E3 are filed by bidders and
# affiliates too, and master.idx carries no filer-vs-subject distinction, so
# including them would label acquirers as targets.
# ponytail: this drops tender offers (no shareholder vote, so no proxy) and
# with them roughly a third of deals. Recovering them means fetching the
# filing index page per SC TO-T to read SUBJECT COMPANY -- about 2,200
# requests. Worth doing only if the sample proves too small.
CLEAN_DEAL_FORM = "DEFM14A"

# Two deal filings by one company inside this window are one episode
# (proxy plus amendments), not two deals.
EPISODE_DAYS = 365


def iso_monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def collapse_episodes(dates: list[dt.date], window_days: int = EPISODE_DAYS) -> list[dt.date]:
    """Keep the first date of each cluster of filings."""
    out: list[dt.date] = []
    for d in sorted(dates):
        if not out or (d - out[-1]).days > window_days:
            out.append(d)
    return out


def rebuild_deals(con, rows: list[dict]) -> int:
    """Replace `deals` with DEFM14A-only, episode-collapsed labels.

    rows: [{'cik','form','file_date'}] straight from parse_master_idx.
    """
    by_cik: dict[str, list[dt.date]] = {}
    for r in rows:
        if r["form"] != CLEAN_DEAL_FORM:
            continue
        by_cik.setdefault(r["cik"], []).append(r["file_date"])

    out = [
        {"cik": cik, "agreement_date": d, "rumor_date": None, "acquirer": None}
        for cik, dates in by_cik.items()
        for d in collapse_episodes(dates)
    ]
    con.execute("DELETE FROM deals")
    if out:
        con.executemany(
            "INSERT OR IGNORE INTO deals VALUES "
            "($cik, $agreement_date, $rumor_date, $acquirer)",
            out,
        )
    return len(out)


def build(con, start: dt.date, end: dt.date) -> int:
    rows = con.execute("SELECT cik, listed, delisted FROM universe").fetchall()

    out = []
    for cik, listed, delisted in rows:
        first = iso_monday(max(listed, start))
        if listed > first:  # listed mid-week: start the following week
            first += dt.timedelta(days=7)
        last = iso_monday(min(delisted or end, end))
        week = first
        while week <= last:
            out.append({"cik": cik, "week": week})
            week += dt.timedelta(days=7)

    if not out:
        return 0
    # Columns named, not positional: label() adds `y` via ALTER TABLE, so on
    # any REBUILD the table has three columns and a positional insert fails
    # with a BinderException. The pipeline only ever worked from empty.
    con.executemany(
        "INSERT OR IGNORE INTO panel (cik, week) VALUES ($cik, $week)", out)
    return len(out)


def label(con, horizon_weeks: int) -> int:
    """y = 1 for panel weeks within horizon_weeks BEFORE an agreement date.

    Weeks at or after the agreement are 0: once the deal is public there is
    nothing left to predict and marking them would be pure lookahead.
    """
    con.execute("ALTER TABLE panel ADD COLUMN IF NOT EXISTS y TINYINT")
    con.execute("UPDATE panel SET y = 0")
    con.execute(
        """
        UPDATE panel SET y = 1
        WHERE EXISTS (
            SELECT 1 FROM deals d
            WHERE d.cik = panel.cik
              AND panel.week < d.agreement_date
              AND panel.week >= d.agreement_date - INTERVAL (?) WEEK
        )
        """,
        [horizon_weeks],
    )
    return con.execute("SELECT coalesce(sum(y), 0) FROM panel").fetchone()[0]
