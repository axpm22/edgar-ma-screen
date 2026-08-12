"""(Acquirer, target) pairs, free, from the already-cached EDGAR indexes.

`deals.acquirer` has been NULL since the project started, because master.idx
carries no counterparty column. It does not need one: EDGAR indexes a deal
filing once per PARTY, and both rows carry the same accession number in the
filename. An accession appearing under exactly two CIKs is a two-party deal
filing, and the two CIKs are the two parties -- recoverable with zero network
requests from files already on disk.

Measured on the 43 cached quarters: 18,199 two-CIK accessions, of which 13,852
have both parties present in features.parquet.

Which of the two is the target uses the discriminator from clean_labels.py --
a target stops filing, an acquirer does not. That rule is checkable here rather
than merely asserted: tender.duckdb's subject CIKs were parsed independently
from SEC-HEADER blocks, and on the 191 SC TO-T pairs where both sources speak,
the rule agrees 187 times (97.9%).

DEFM14A is excluded: it is always single-filer and yields no pair. S-4 is
excluded too -- one accession carries up to 332 co-registrant CIKs when a bank
registers its subsidiaries, so "exactly two" does not identify a deal there.
"""
import datetime as dt
import re

from . import config, fetch, universe

# Forms EDGAR indexes under both parties. 425 = business-combination
# communication, SC TO-T = third-party tender offer, SC 13E3 = going private.
PAIR_FORMS = {"425", "SC TO-T", "SC 13E3"}

# Same window clean_labels uses: deals close 3-9 months after announcement.
SURVIVE_DAYS = 270

# Repeat 425s across one deal are one episode, not fifty. A year apart means a
# genuinely separate approach.
EPISODE_DAYS = 365

ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")

SCHEMA = """
CREATE TABLE IF NOT EXISTS deal_pairs (
    target_cik   VARCHAR,
    acquirer_cik VARCHAR,
    first_ts     DATE,
    last_ts      DATE,
    form         VARCHAR,
    n_filings    INTEGER,
    PRIMARY KEY (target_cik, acquirer_cik, first_ts)
);
"""


def accession_of(filename: str) -> str | None:
    m = ACCESSION_RE.search(filename.rsplit("/", 1)[-1])
    return m.group(1) if m else None


def group_filings(rows: list[dict], forms: set[str]) -> dict[str, dict]:
    """accession -> {ciks, date, form}. Date is the EARLIEST index row.

    Both parties' index rows normally share a date, but amendments and late
    acceptance can differ by a day; the earliest is the announcement.
    """
    out: dict[str, dict] = {}
    for r in rows:
        if r["form"] not in forms:
            continue
        acc = accession_of(r["filename"])
        if not acc:
            continue
        rec = out.setdefault(acc, {"ciks": set(), "date": r["file_date"],
                                   "form": r["form"]})
        rec["ciks"].add(r["cik"])
        rec["date"] = min(rec["date"], r["file_date"])
    return out


def orient(ciks: set[str], date: dt.date, delisted: dict[str, dt.date | None],
           survive_days: int = SURVIVE_DAYS) -> tuple[str, str] | None:
    """(target, acquirer), or None when the rule cannot separate them.

    Ambiguity is common and is returned as None rather than guessed: 4,678 of
    14,155 in-universe pairs have both or neither party stopping. Guessing
    would put acquirers in the target column, which is the exact error this
    project already caught once.
    """
    if len(ciks) != 2:
        return None
    a, b = sorted(ciks)
    cutoff = date + dt.timedelta(days=survive_days)

    def stops(cik: str) -> bool:
        d = delisted.get(cik)
        return bool(d and d <= cutoff)

    sa, sb = stops(a), stops(b)
    if sa == sb:
        return None
    return (a, b) if sa else (b, a)


def collapse(pairs: list[dict], episode_days: int = EPISODE_DAYS) -> list[dict]:
    """One row per (target, acquirer) episode, not per filing."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for p in pairs:
        by_key.setdefault((p["target_cik"], p["acquirer_cik"]), []).append(p)

    out = []
    for (t, a), group in by_key.items():
        group.sort(key=lambda p: p["date"])
        cur = None
        for p in group:
            if cur and (p["date"] - cur["last_ts"]).days <= episode_days:
                cur["last_ts"] = p["date"]
                cur["n_filings"] += 1
                continue
            if cur:
                out.append(cur)
            cur = {"target_cik": t, "acquirer_cik": a, "first_ts": p["date"],
                   "last_ts": p["date"], "form": p["form"], "n_filings": 1}
        if cur:
            out.append(cur)
    return sorted(out, key=lambda r: (r["target_cik"], r["acquirer_cik"],
                                      r["first_ts"]))


def index_rows(start_year: int = config.PANEL_START_YEAR,
               end_year: int = config.PANEL_END_YEAR) -> list[dict]:
    """Every cached quarterly index row. Cache-only: no network requests."""
    rows = []
    for y, q in universe.quarters(start_year, end_year):
        p = fetch.cache_path("sec", config.IDX_URL.format(year=y, q=q))
        if not p.exists():
            continue
        rows.extend(universe.parse_master_idx(p.read_bytes()))
    return rows


def build(con, delisted: dict[str, dt.date | None],
          start_year: int = config.PANEL_START_YEAR,
          end_year: int = config.PANEL_END_YEAR) -> dict:
    """Create and fill deal_pairs. Returns counts for the caller to print."""
    con.execute(SCHEMA)
    con.execute("DELETE FROM deal_pairs")

    grouped = group_filings(index_rows(start_year, end_year), PAIR_FORMS)
    two_party = {k: v for k, v in grouped.items() if len(v["ciks"]) == 2}

    oriented, ambiguous = [], 0
    for v in two_party.values():
        o = orient(v["ciks"], v["date"], delisted)
        if o is None:
            ambiguous += 1
            continue
        oriented.append({"target_cik": o[0], "acquirer_cik": o[1],
                         "date": v["date"], "form": v["form"]})

    episodes = collapse(oriented)
    if episodes:
        con.executemany(
            "INSERT OR IGNORE INTO deal_pairs VALUES "
            "($target_cik, $acquirer_cik, $first_ts, $last_ts, $form, "
            "$n_filings)", episodes)
    return {"accessions": len(grouped), "two_party": len(two_party),
            "oriented": len(oriented), "ambiguous": ambiguous,
            "episodes": len(episodes)}


def validate_orientation(con, tender_con) -> dict:
    """Check the orientation rule against independently-parsed SC TO-T subjects.

    tender.duckdb's CIKs came from the SUBJECT COMPANY block of the SEC header,
    a different file and a different parser. Agreement is evidence the rule
    works; disagreement is a reason not to ship it.
    """
    truth = {r[0] for r in tender_con.execute(
        "SELECT DISTINCT cik FROM tender_offers").fetchall()}
    rows = con.execute(
        "SELECT target_cik, acquirer_cik FROM deal_pairs "
        "WHERE form = 'SC TO-T'").fetchall()
    checked = agree = 0
    for target, acquirer in rows:
        # Only pairs where exactly one side is a known subject are informative.
        if (target in truth) == (acquirer in truth):
            continue
        checked += 1
        agree += target in truth
    return {"checked": checked, "agree": agree,
            "pct": 100.0 * agree / checked if checked else 0.0}
