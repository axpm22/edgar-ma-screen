"""All EDGAR form events -> form_events.

The master indexes were already downloaded for the universe build: 11.6M
filings, of which only 302k periodic ones were consumed. This mines the rest.
Nothing is fetched; everything comes from the existing disk cache.

Amendments collapse into their parent family -- a 13D/A is still 13D activity.
"""
import datetime as dt

from . import config, fetch, universe

# Delisting notices. These are filed AFTER a deal completes, so they encode
# the outcome. A model given them scores near-perfectly and predicts nothing.
FORBIDDEN_FORMS = frozenset({"25-NSE", "25", "15-12B", "15-12G", "15F-12B",
                             "15F-12G", "25-NSE/A"})

TRACKED_FORMS = {
    "SC 13D": "sc13d", "SC 13D/A": "sc13d",
    "SC 13G": "sc13g", "SC 13G/A": "sc13g",
    "8-K": "form8k", "8-K/A": "form8k",
    "DEF 14A": "def14a",
    "S-4": "s4", "S-4/A": "s4",
    "NT 10-K": "late", "NT 10-Q": "late",
    # --- buyer-side financing capacity -------------------------------------
    # A shelf registration is dry powder: it lets a company issue securities
    # at short notice, which is what a stock-funded acquisition needs.
    "S-3": "shelf", "S-3ASR": "shelf", "S-3/A": "shelf",
    # A prospectus supplement means an offering actually happened.
    "424B5": "raise", "424B2": "raise", "424B3": "raise",
    "FWP": "raise",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS form_events (
    cik       VARCHAR,
    family    VARCHAR,
    public_ts DATE,
    n         INTEGER,
    PRIMARY KEY (cik, family, public_ts)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def classify(form: str) -> str | None:
    if form in FORBIDDEN_FORMS:
        return None
    return TRACKED_FORMS.get(form)


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO form_events VALUES ($cik, $family, $public_ts, 1)
        ON CONFLICT (cik, family, public_ts)
        DO UPDATE SET n = form_events.n + 1
        """,
        rows,
    )
    return len(rows)


def load(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    init_schema(con)
    today = dt.date.today()
    total = 0
    for year, q in universe.quarters(start_year, end_year):
        if dt.date(year, (q - 1) * 3 + 1, 1) > today:
            break
        try:
            raw = fetch.sec_get(config.IDX_URL.format(year=year, q=q))
        except Exception:
            continue
        rows = []
        for r in universe.parse_master_idx(raw):
            fam = classify(r["form"])
            if fam:
                rows.append({"cik": r["cik"], "family": fam,
                             "public_ts": r["file_date"]})
        total += insert(con, rows)
        if verbose:
            print(f"  {year}Q{q}: {len(rows):>7,} form events", flush=True)
    return total
