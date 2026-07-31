"""SEC Insider Transactions Data Sets -> insider_trans.

DERA publishes Forms 3/4/5 flattened into TSVs, one ZIP per quarter -- which
replaces parsing roughly half a million ownership XML documents a year.

Two fields make the blackout signal work:
  TRANS_CODE   'S' open-market sale, 'P' open-market purchase. Those are the
               discretionary ones. 'A' (grant), 'F' (tax withholding) and
               'M' (option exercise) are automatic and carry no information
               about what management knows.
  AFF10B5ONE   the 10b5-1 affirmation. A trade under a pre-scheduled plan
               executes regardless of what the insider knows, so it is not
               discretionary either.

AFF10B5ONE is written inconsistently across quarters ('1', 'true', '0',
'false', empty), so it is parsed permissively.
"""
import datetime as dt
import io
import zipfile

from . import config, fetch

DISCRETIONARY_CODES = {"S", "P"}
_TRUTHY = {"1", "TRUE", "Y", "YES"}


def _dera_date(raw: str) -> dt.date | None:
    try:
        return dt.datetime.strptime(raw.strip(), "%d-%b-%Y").date()
    except ValueError:
        return None


def is_10b5_1(raw: str) -> bool:
    return raw.strip().upper() in _TRUTHY


def parse_sub(fh) -> dict[str, dict]:
    header = fh.readline().decode("latin-1").rstrip("\n").split("\t")
    i_acc = header.index("ACCESSION_NUMBER")
    i_filed = header.index("FILING_DATE")
    i_cik = header.index("ISSUERCIK")
    # AFF10B5ONE only exists from 2023 onward -- the Form 4 10b5-1 checkbox
    # became mandatory then. Earlier quarters carry no plan information, so
    # plan status is recorded as unknown (False) rather than guessed, and the
    # discretionary split should be treated as clean only post-2023.
    i_plan = header.index("AFF10B5ONE") if "AFF10B5ONE" in header else -1

    out = {}
    need = max(i_acc, i_filed, i_cik, i_plan)
    for raw in fh:
        f = raw.decode("latin-1").rstrip("\n").split("\t")
        if len(f) <= need:
            continue
        filed = _dera_date(f[i_filed])
        if filed is None:
            continue
        out[f[i_acc]] = {
            "cik": f[i_cik].lstrip("0") or "0",
            "filed": filed,
            "plan": is_10b5_1(f[i_plan]) if i_plan >= 0 else False,
            "plan_known": i_plan >= 0,
        }
    return out


def stream_trans(fh, sub: dict[str, dict]):
    header = fh.readline().decode("latin-1").rstrip("\n").split("\t")
    i_acc = header.index("ACCESSION_NUMBER")
    i_code = header.index("TRANS_CODE")
    i_sh = header.index("TRANS_SHARES")

    for raw in fh:
        f = raw.decode("latin-1").rstrip("\n").split("\t")
        if len(f) <= max(i_acc, i_code, i_sh):
            continue
        meta = sub.get(f[i_acc])
        if meta is None:
            continue
        try:
            shares = float(f[i_sh])
        except ValueError:
            continue
        code = f[i_code].strip()
        yield {
            "accession": f[i_acc],
            "cik": meta["cik"],
            # FILING_DATE, not TRANS_DATE: the trade is private until filed.
            "public_ts": meta["filed"],
            "trans_code": code,
            "shares": shares,
            "discretionary": code in DISCRETIONARY_CODES and not meta["plan"],
        }


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO insider_trans VALUES "
        "($accession, $cik, $public_ts, $trans_code, $shares, $discretionary)",
        rows,
    )
    return len(rows)


def load_quarter(con, year: int, q: int) -> int:
    blob = fetch.sec_get(config.INSIDER_URL.format(year=year, q=q))
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open("SUBMISSION.tsv") as fh:
            sub = parse_sub(fh)
        with z.open("NONDERIV_TRANS.tsv") as fh:
            rows = list(stream_trans(fh, sub))
    return insert(con, rows)
