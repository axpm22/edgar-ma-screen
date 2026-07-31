"""CIK -> SIC industry code, from the cached financial-statement sub.txt files.

Sector is a mandatory control: M&A arrives in industry waves, so without it a
model happily attributes to its signals what is really "this sector was
consolidating that year". The code is already sitting in sub.txt, which is
1.8MB and already on disk, so this costs one cheap re-read of the cache.
"""
import datetime as dt
import io
import zipfile

from . import config, fetch

SCHEMA = """
CREATE TABLE IF NOT EXISTS company_sic (
    cik VARCHAR PRIMARY KEY,
    sic VARCHAR
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def parse_sub_sic(fh) -> list[dict]:
    header = fh.readline().decode("latin-1").rstrip("\n").split("\t")
    i_cik, i_sic = header.index("cik"), header.index("sic")
    out = []
    for raw in fh:
        f = raw.decode("latin-1").rstrip("\n").split("\t")
        if len(f) <= max(i_cik, i_sic):
            continue
        sic = f[i_sic].strip()
        if not sic:
            continue
        out.append({"cik": f[i_cik].lstrip("0") or "0", "sic": sic})
    return out


def load(con, start_year: int, end_year: int) -> int:
    init_schema(con)
    today = dt.date.today()
    total = 0
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            if dt.date(y, (q - 1) * 3 + 1, 1) > today:
                continue
            try:
                blob = fetch.sec_get(config.FUND_URL.format(year=y, q=q))
                with zipfile.ZipFile(io.BytesIO(blob)) as z, z.open("sub.txt") as fh:
                    rows = parse_sub_sic(fh)
            except Exception:
                continue
            if rows:
                con.executemany(
                    "INSERT OR IGNORE INTO company_sic VALUES ($cik, $sic)", rows
                )
                total += len(rows)
    return total
