"""Deal labels from EDGAR quarterly indexes.

Three forms mark a live public-target deal, all timestamped by EDGAR the day
they land:
  DEFM14A  -- merger proxy
  SC TO-T  -- third-party tender offer
  SC 13E3  -- going-private transaction

Plain 8-K is excluded: the master index carries no Item detail, so it would
add far more noise than signal.
"""
import datetime as dt

from . import fetch, universe

DEAL_FORMS = {"DEFM14A", "SC TO-T", "SC 13E3"}


def extract(rows: list[dict]) -> list[dict]:
    return [
        {
            "cik": r["cik"],
            "agreement_date": r["file_date"],
            "rumor_date": None,
            "acquirer": None,
        }
        for r in rows
        if r["form"] in DEAL_FORMS
    ]


def insert(con, deals: list[dict]) -> int:
    if not deals:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO deals VALUES "
        "($cik, $agreement_date, $rumor_date, $acquirer)",
        deals,
    )
    return len(deals)


def load(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    total = 0
    today = dt.date.today()
    for year, q in universe.quarters(start_year, end_year):
        if dt.date(year, (q - 1) * 3 + 1, 1) > today:
            continue
        try:
            raw = fetch.sec_get(universe.IDX_URL.format(year=year, q=q))
        except Exception:
            continue
        n = insert(con, extract(universe.parse_master_idx(raw)))
        total += n
        if verbose and n:
            print(f"  {year}Q{q}: {n:>4} deal filings")
    return total
