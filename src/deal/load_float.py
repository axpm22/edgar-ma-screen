"""SEC XBRL Frames API -> public float (market capitalisation proxy).

No free daily price source covers delisted companies, and delisted companies
are exactly where the positives live -- every acquired target leaves the
market. EntityPublicFloat sidesteps that entirely: it comes from the company's
own 10-K cover page, so an acquired company's history is as complete as anyone
else's. That makes it survivorship-free where a price feed would not be.

The cost is frequency. Public float is an annual number measured at the fiscal
second-quarter instant, so this gives slow-moving size and valuation, not
returns or volume.

Point-in-time: the measurement instant is NOT when the number became public.
It appears in the 10-K, filed months later. The instant is therefore lagged by
PUBLICATION_LAG_DAYS, chosen to clear the filing deadline with margin.
"""
import datetime as dt
import json

from . import fetch

FRAMES = "https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/{period}.json"

# Public float is measured at the fiscal Q2 instant and disclosed on the 10-K
# cover, filed after fiscal year end (60-90 days depending on filer status).
# From a mid-year instant that is roughly 7-9 months. 270 days clears it with
# margin -- deliberately conservative, because market cap is a slow control
# and a stale value costs far less than a leaked one.
PUBLICATION_LAG_DAYS = 270

SCHEMA = """
CREATE TABLE IF NOT EXISTS public_float (
    cik       VARCHAR,
    end_date  DATE,
    public_ts DATE,
    value     DOUBLE,
    PRIMARY KEY (cik, end_date)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def parse_frame(raw: bytes) -> list[dict]:
    payload = json.loads(raw)
    out = []
    for r in payload.get("data", []):
        try:
            end = dt.date.fromisoformat(r["end"])
        except (KeyError, ValueError):
            continue
        out.append({
            "cik": str(r["cik"]).lstrip("0") or "0",
            "end_date": end,
            "public_ts": end + dt.timedelta(days=PUBLICATION_LAG_DAYS),
            "value": float(r["val"]),
        })
    return out


def load(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    init_schema(con)
    today = dt.date.today()
    total = 0
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            if dt.date(y, (q - 1) * 3 + 1, 1) > today:
                continue
            period = f"CY{y}Q{q}I"
            try:
                rows = parse_frame(fetch.sec_get(FRAMES.format(period=period)))
            except Exception:
                continue
            if rows:
                con.executemany(
                    "INSERT OR IGNORE INTO public_float VALUES "
                    "($cik, $end_date, $public_ts, $value)",
                    rows,
                )
                total += len(rows)
            if verbose:
                print(f"  {period}: {len(rows):>5,}", flush=True)
    return total
