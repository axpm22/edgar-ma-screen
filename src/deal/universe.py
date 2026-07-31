"""Point-in-time universe from EDGAR quarterly master indexes.

A company was a listed reporting company in quarter Q if it filed a periodic
report in Q. Building membership this way means acquired and delisted names
stay in the panel for exactly the quarters they existed -- the survivorship
problem solved with a free file.
"""
import datetime as dt

from . import config, fetch

PERIODIC_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"}
IDX_URL = config.IDX_URL


def parse_master_idx(raw: bytes) -> list[dict]:
    out = []
    for line in raw.decode("latin-1").splitlines():
        parts = line.split("|")
        if len(parts) != 5 or not parts[0].strip().isdigit():
            continue  # header, separator, or malformed row
        cik, name, form, date, filename = parts
        try:
            filed = dt.date.fromisoformat(date.strip())
        except ValueError:
            continue
        out.append({
            "cik": cik.strip().lstrip("0") or "0",
            "name": name.strip(),
            "form": form.strip(),
            "file_date": filed,
            # Needed to pair the two parties of a SC 13D: the accession
            # number is only recoverable from the filename.
            "filename": filename.strip(),
        })
    return out


def quarters(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [(y, q) for y in range(start_year, end_year + 1) for q in (1, 2, 3, 4)]


def upsert(con, rows: list[dict]) -> int:
    periodic = [r for r in rows if r["form"] in PERIODIC_FORMS]
    if not periodic:
        return 0
    con.executemany(
        """
        INSERT INTO universe VALUES ($cik, $name, $file_date, $file_date)
        ON CONFLICT (cik) DO UPDATE SET
            listed   = least(universe.listed, excluded.listed),
            delisted = greatest(universe.delisted, excluded.delisted)
        """,
        [{"cik": r["cik"], "name": r["name"], "file_date": r["file_date"]}
         for r in periodic],
    )
    return len(periodic)


def build(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    total = 0
    today = dt.date.today()
    for year, q in quarters(start_year, end_year):
        # Skip quarters that have not happened yet.
        if dt.date(year, (q - 1) * 3 + 1, 1) > today:
            continue
        try:
            raw = fetch.sec_get(IDX_URL.format(year=year, q=q))
        except Exception as exc:
            if verbose:
                print(f"  {year}Q{q}: SKIP ({type(exc).__name__})")
            continue
        n = upsert(con, parse_master_idx(raw))
        total += n
        if verbose:
            print(f"  {year}Q{q}: {n:>7,} periodic filings")
    return total
