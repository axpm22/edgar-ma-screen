"""Identify professional activists among 13D filers.

`sc13d_52w` treats every 13D as equivalent, but a 13D from Elliott means
something categorically different from one filed by a founder's family trust.

EDGAR indexes a SC 13D under BOTH the filer and the subject company, and the
accession number is recoverable from the filename -- so the two parties of
each filing can be paired. Within a pair, the professional activist is the CIK
that appears across many OTHER 13D filings; the subject appears in few. That
count is itself the measure of how professional the filer is.
"""
import datetime as dt
from collections import defaultdict

from . import config, fetch, universe

SC13D = {"SC 13D", "SC 13D/A"}

# A CIK filing against at least this many distinct subjects is a professional
# campaigner rather than an insider crossing 5% in their own company.
PRO_FILER_MIN_TARGETS = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS activist_events (
    cik            VARCHAR,   -- the SUBJECT company
    public_ts      DATE,
    filer_targets  INTEGER,   -- how many distinct subjects that filer has hit
    PRIMARY KEY (cik, public_ts)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def _accession(filename: str) -> str:
    # edgar/data/<cik>/0001234567-24-000123.txt
    return filename.rsplit("/", 1)[-1].replace(".txt", "")


def collect(start_year: int, end_year: int) -> list[dict]:
    """Pair each 13D accession's CIKs, then attribute the filer's reach."""
    by_accession: dict[str, list[tuple[str, dt.date]]] = defaultdict(list)
    today = dt.date.today()
    for year, q in universe.quarters(start_year, end_year):
        if dt.date(year, (q - 1) * 3 + 1, 1) > today:
            break
        try:
            raw = fetch.sec_get(config.IDX_URL.format(year=year, q=q))
        except Exception:
            continue
        for r in universe.parse_master_idx(raw):
            if r["form"] in SC13D:
                by_accession[_accession(r["filename"])].append(
                    (r["cik"], r["file_date"]))

    # How many distinct accessions does each CIK appear in? Professional
    # filers appear in many; a one-off subject appears in one or two.
    reach: dict[str, int] = defaultdict(int)
    for parties in by_accession.values():
        for cik, _ in parties:
            reach[cik] += 1

    out = []
    for parties in by_accession.values():
        if len(parties) < 2:
            continue
        # Highest reach is the filer; the rest are subjects.
        parties = sorted(parties, key=lambda t: -reach[t[0]])
        filer_reach = reach[parties[0][0]]
        if filer_reach < PRO_FILER_MIN_TARGETS:
            continue
        for cik, date in parties[1:]:
            out.append({"cik": cik, "public_ts": date,
                        "filer_targets": filer_reach})
    return out


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO activist_events VALUES ($cik, $public_ts, $filer_targets)
        ON CONFLICT (cik, public_ts) DO UPDATE SET
            filer_targets = greatest(activist_events.filer_targets,
                                     excluded.filer_targets)
        """,
        rows,
    )
    return len(rows)


def load(con, start_year: int, end_year: int) -> int:
    init_schema(con)
    return insert(con, collect(start_year, end_year))
