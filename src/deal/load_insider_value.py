"""Form 4 transactions in DOLLARS, aggregated per company-week.

The existing insider_trans table counts transactions. A CEO dumping $40M and
an officer selling $8,000 count the same there, which throws away most of the
information. NONDERIV_TRANS carries TRANS_PRICEPERSHARE alongside
TRANS_SHARES, so the dollar value is available for free.

Aggregating to company-week during the parse keeps this compact -- roughly
600k rows instead of 3.1M -- and it is the grain the features need anyway.
"""
import datetime as dt
import io
import zipfile

from . import config, fetch
from .load_insider import DISCRETIONARY_CODES, is_10b5_1

SCHEMA = """
CREATE TABLE IF NOT EXISTS insider_value (
    cik        VARCHAR,
    week       DATE,
    usd_sold   DOUBLE,
    usd_bought DOUBLE,
    PRIMARY KEY (cik, week)
);
"""


def init_schema(con) -> None:
    con.execute(SCHEMA)


def _iso_monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def _dera_date(raw: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(raw.strip(), "%d-%b-%Y").date()
    except ValueError:
        return None


def parse_quarter(sub_raw: bytes, trans_raw: bytes) -> list[dict]:
    lines = sub_raw.decode("latin-1").splitlines()
    head = lines[0].split("\t")
    i_acc, i_cik = head.index("ACCESSION_NUMBER"), head.index("ISSUERCIK")
    i_filed = head.index("FILING_DATE")
    i_plan = head.index("AFF10B5ONE") if "AFF10B5ONE" in head else -1

    subs = {}
    need = max(i_acc, i_cik, i_filed, i_plan)
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) <= need:
            continue
        filed = _dera_date(f[i_filed])
        if filed is None:
            continue
        subs[f[i_acc]] = {
            "cik": f[i_cik].lstrip("0") or "0",
            "week": _iso_monday(filed),
            "plan": is_10b5_1(f[i_plan]) if i_plan >= 0 else False,
        }

    lines = trans_raw.decode("latin-1").splitlines()
    head = lines[0].split("\t")
    i_acc = head.index("ACCESSION_NUMBER")
    i_code = head.index("TRANS_CODE")
    i_sh = head.index("TRANS_SHARES")
    i_px = head.index("TRANS_PRICEPERSHARE")

    agg: dict[tuple[str, dt.date], dict] = {}
    need = max(i_acc, i_code, i_sh, i_px)
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) <= need:
            continue
        meta = subs.get(f[i_acc])
        if meta is None or meta["plan"]:
            continue  # scheduled plan trades carry no information
        code = f[i_code].strip()
        if code not in DISCRETIONARY_CODES:
            continue
        try:
            usd = float(f[i_sh]) * float(f[i_px])
        except ValueError:
            continue
        key = (meta["cik"], meta["week"])
        rec = agg.setdefault(key, {"cik": key[0], "week": key[1],
                                   "usd_sold": 0.0, "usd_bought": 0.0})
        rec["usd_sold" if code == "S" else "usd_bought"] += usd
    return list(agg.values())


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO insider_value VALUES ($cik, $week, $usd_sold, $usd_bought)
        ON CONFLICT (cik, week) DO UPDATE SET
            usd_sold   = insider_value.usd_sold   + excluded.usd_sold,
            usd_bought = insider_value.usd_bought + excluded.usd_bought
        """,
        rows,
    )
    return len(rows)


def load(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    init_schema(con)
    today = dt.date.today()
    total = 0
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            if dt.date(y, (q - 1) * 3 + 1, 1) > today:
                continue
            try:
                blob = fetch.sec_get(config.INSIDER_URL.format(year=y, q=q))
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    rows = parse_quarter(z.read("SUBMISSION.tsv"),
                                         z.read("NONDERIV_TRANS.tsv"))
            except Exception:
                continue
            total += insert(con, rows)
            if verbose:
                print(f"  {y}Q{q}: {len(rows):>6,} company-weeks", flush=True)
    return total
