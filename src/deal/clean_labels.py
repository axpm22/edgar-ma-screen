"""Separate genuine acquisition targets from other merger-proxy filers.

A DEFM14A is filed by whoever's shareholders vote, which is not the same as
"was acquired". In a cash deal only the target votes, but in a stock deal the
acquirer's shareholders vote too because the acquirer is issuing shares, so
the BUYER files a proxy. Terminated deals leave a proxy behind with no
acquisition at all.

Measured on this panel: 581 of 2,445 proxy filers (23.8%) were still filing
periodic reports years later -- Teledyne after acquiring FLIR, Newmont after
Newcrest, Dow after DowDuPont. Those are buyers and survivors sitting in a
target label set.

The discriminator is simple and needs no new data: a target stops filing.
An acquirer does not.

Caveat at the recent edge -- a company acquired within SURVIVE_DAYS of the
panel end cannot be distinguished from a survivor, because neither has had
time to stop filing. Those are excluded from both classes rather than guessed.
"""
import datetime as dt

SURVIVE_DAYS = 270          # deals typically close 3-9 months after the proxy

SCHEMA = """
CREATE TABLE IF NOT EXISTS deals_clean (
    cik            VARCHAR,
    agreement_date DATE,
    outcome        VARCHAR,   -- 'target' | 'survivor' | 'undetermined'
    PRIMARY KEY (cik, agreement_date)
);
"""


def build(con, panel_end: dt.date, survive_days: int = SURVIVE_DAYS) -> dict:
    """Classify every proxy filing. Returns counts by outcome."""
    con.execute(SCHEMA)
    con.execute("DELETE FROM deals_clean")
    con.execute(
        f"""
        INSERT INTO deals_clean
        SELECT d.cik, d.agreement_date,
               CASE
                 -- Too close to the panel edge for absence to mean anything.
                 WHEN d.agreement_date > DATE '{panel_end}'
                      - INTERVAL {survive_days} DAY THEN 'undetermined'
                 -- Stopped filing soon after the vote: acquired.
                 WHEN u.delisted <= d.agreement_date
                      + INTERVAL {survive_days} DAY THEN 'target'
                 ELSE 'survivor'
               END
        FROM deals d JOIN universe u USING (cik)
        """
    )
    rows = con.execute(
        "SELECT outcome, count(*) FROM deals_clean GROUP BY 1"
    ).fetchall()
    return dict(rows)


def target_ciks(con) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT cik FROM deals_clean WHERE outcome = 'target'"
    ).fetchall()]


def survivor_ciks(con) -> list[str]:
    """Proxy filers that were NOT acquired -- mostly acquirers in stock deals
    and parties to terminated transactions."""
    return [r[0] for r in con.execute(
        "SELECT DISTINCT cik FROM deals_clean WHERE outcome = 'survivor'"
    ).fetchall()]
