"""8-K item codes -> item_events.

`form8k_26w` is the top feature by tree gain but the WEAKEST significant
signal under clustered inference (z=+2.33). That gap is what a raw count looks
like: trees love a high-cardinality integer to split on, but "filed six 8-Ks"
carries little independent information. The item codes say what those filings
were actually about.

The submissions API exposes an `items` field per filing. Companies with long
histories page their older filings into extra JSON files listed under
`filings.files`, which are fetched too -- otherwise coverage silently
collapses to roughly the last three years for active filers.
"""
import datetime as dt
import json

from . import fetch

SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE = "https://data.sec.gov/submissions/{name}"

# Items worth modelling. Anything not listed is ignored rather than lumped
# into a catch-all, which would just recreate the blunt count.
TRACKED_ITEMS = {
    "1.01": "i_material_agmt",     # material definitive agreement
    "1.02": "i_agmt_termination",
    "2.05": "i_exit_costs",        # restructuring / exit
    "2.06": "i_impairment",
    "3.03": "i_security_rights",   # rights plans live here
    "4.01": "i_auditor_change",
    "4.02": "i_nonreliance",       # restatement
    "5.01": "i_control_change",
    "5.02": "i_officer_change",    # departures / appointments
    "5.03": "i_bylaw_change",
    "5.07": "i_vote_results",
    "7.01": "i_reg_fd",
    "8.01": "i_other_events",
}

# Item 3.01 is a delisting/listing-deficiency notice. Same leakage class as
# form 25-NSE: it is filed when a deal is already closing.
FORBIDDEN_ITEMS = frozenset({"3.01"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS item_events (
    cik       VARCHAR,
    family    VARCHAR,
    public_ts DATE,
    n         INTEGER,
    PRIMARY KEY (cik, family, public_ts)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def classify_item(code: str) -> str | None:
    code = code.strip()
    if code in FORBIDDEN_ITEMS:
        return None
    return TRACKED_ITEMS.get(code)


def parse_block(block: dict, cik: str, start_year: int) -> list[dict]:
    forms = block.get("form", [])
    dates = block.get("filingDate", [])
    items = block.get("items", [""] * len(forms))
    out = []
    for form, date, item in zip(forms, dates, items):
        if form not in ("8-K", "8-K/A") or not item:
            continue
        try:
            d = dt.date.fromisoformat(date)
        except ValueError:
            continue
        if d.year < start_year:
            continue
        for code in str(item).split(","):
            fam = classify_item(code)
            if fam:
                out.append({"cik": cik, "family": fam, "public_ts": d})
    return out


def fetch_company(cik: str, start_year: int) -> list[dict]:
    raw = fetch.sec_get(SUBMISSIONS.format(cik10=cik.zfill(10)))
    payload = json.loads(raw)
    rows = parse_block(payload.get("filings", {}).get("recent", {}),
                       cik, start_year)
    # Older filings are paged out into separate archive files.
    for extra in payload.get("filings", {}).get("files", []):
        name = extra.get("name")
        if not name:
            continue
        try:
            rows += parse_block(json.loads(fetch.sec_get(
                ARCHIVE.format(name=name))), cik, start_year)
        except Exception:
            continue
    return rows


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO item_events VALUES ($cik, $family, $public_ts, 1)
        ON CONFLICT (cik, family, public_ts)
        DO UPDATE SET n = item_events.n + 1
        """,
        rows,
    )
    return len(rows)
