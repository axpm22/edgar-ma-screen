"""SC TO-T tender offers -> an INDEPENDENT validation label.

These were excluded from training labels because EDGAR indexes a tender offer
under both the bidder and the target, and master.idx gives no way to tell them
apart. That exclusion is what makes them valuable now: tender offers are a
structurally different deal type -- no shareholder vote, frequently hostile,
usually cash -- that the model has never seen.

If a model trained only on negotiated mergers also ranks tender-offer targets
highly, it learned something about deal preparation in general rather than
about DEFM14A filers in particular. That is a stronger claim than any
resampling of the training labels can support.

The target is read from the SEC-HEADER block, which names SUBJECT COMPANY and
FILED BY explicitly. Only the first few KB of each submission are needed, so
these are Range requests rather than full downloads.
"""
import datetime as dt
import re

import httpx

from . import config, fetch

ARCHIVE = "https://www.sec.gov/Archives/{filename}"
HEADER_BYTES = 8192

_SUBJECT = re.compile(
    r"SUBJECT COMPANY:.*?CENTRAL INDEX KEY:\s*(\d+)", re.S)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tender_offers (
    cik        VARCHAR,   -- the SUBJECT (target)
    public_ts  DATE,
    PRIMARY KEY (cik, public_ts)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def parse_subject(header: str) -> str | None:
    """Return the subject company's CIK, or None if the header lacks one."""
    m = _SUBJECT.search(header)
    if not m:
        return None
    return m.group(1).lstrip("0") or "0"


def fetch_header(filename: str) -> str:
    """First few KB only -- the SEC-HEADER block sits at the top of the file."""
    def _go() -> bytes:
        fetch.SEC_LIMITER.wait()
        r = httpx.get(ARCHIVE.format(filename=filename),
                      headers={"User-Agent": config.EDGAR_UA,
                               "Range": f"bytes=0-{HEADER_BYTES}"},
                      timeout=60)
        r.raise_for_status()
        return r.content

    return fetch.cached("sec_hdr", filename, _go).decode("latin-1", "ignore")


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO tender_offers VALUES ($cik, $public_ts)", rows)
    return len(rows)


def collect(index_rows: list[dict], forms=("SC TO-T",)) -> list[dict]:
    """index_rows: parse_master_idx output. One header fetch per accession."""
    seen: set[str] = set()
    out = []
    for r in index_rows:
        if r["form"] not in forms:
            continue
        fn = r["filename"]
        if fn in seen:
            continue
        seen.add(fn)
        try:
            cik = parse_subject(fetch_header(fn))
        except Exception:
            continue
        if cik:
            out.append({"cik": cik, "public_ts": r["file_date"]})
    return out
