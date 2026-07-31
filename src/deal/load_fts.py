"""EDGAR full-text search -> strategic-intent events.

Every other signal in this project is an inference from a trace. This one is a
company stating its intent in plain language: a "review of strategic
alternatives" is corporate for "we may sell ourselves."

EDGAR full-text search covers 2001+, needs no key, and caps results at 10,000
per query -- so queries are chunked by month, which keeps every chunk far
below the cap (~80 hits/month for the main phrase).
"""
import datetime as dt
import json

from . import config, fetch

FTS = "https://efts.sec.gov/LATEST/search-index"

# Ordered strongest to weakest. Each becomes its own feature family, because a
# formal strategic review is a much louder signal than a passing mention.
PHRASES = {
    "strategic alternatives": "sa_review",
    "exploring a sale": "sa_sale",
    "unsolicited proposal": "sa_unsolicited",
    "letter of intent": "sa_loi",
}

FORMS = "8-K"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fts_events (
    cik       VARCHAR,
    family    VARCHAR,
    public_ts DATE,
    n         INTEGER,
    PRIMARY KEY (cik, family, public_ts)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def _months(start_year: int, end_year: int):
    d = dt.date(start_year, 1, 1)
    today = dt.date.today()
    while d <= today and d.year <= end_year:
        nxt = dt.date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
        yield d, min(nxt - dt.timedelta(days=1), today)
        d = nxt


def parse_hits(raw: bytes, family: str) -> list[dict]:
    payload = json.loads(raw)
    out = []
    for hit in payload.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        ciks = src.get("ciks") or []
        date = src.get("file_date")
        if not ciks or not date:
            continue
        for cik in ciks:
            out.append({
                "cik": str(cik).lstrip("0") or "0",
                "family": family,
                "public_ts": dt.date.fromisoformat(date),
            })
    return out


def fetch_month(phrase: str, lo: dt.date, hi: dt.date) -> bytes:
    url = (f'{FTS}?q=%22{phrase.replace(" ", "+")}%22&forms={FORMS}'
           f"&dateRange=custom&startdt={lo}&enddt={hi}")
    return fetch.sec_get(url)


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO fts_events VALUES ($cik, $family, $public_ts, 1)
        ON CONFLICT (cik, family, public_ts)
        DO UPDATE SET n = fts_events.n + 1
        """,
        rows,
    )
    return len(rows)


def load(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    init_schema(con)
    total = 0
    for phrase, family in PHRASES.items():
        n = 0
        for lo, hi in _months(start_year, end_year):
            try:
                rows = parse_hits(fetch_month(phrase, lo, hi), family)
            except Exception:
                continue
            n += insert(con, rows)
        total += n
        if verbose:
            print(f"  {family:<16} {n:>7,} hits", flush=True)
    return total
