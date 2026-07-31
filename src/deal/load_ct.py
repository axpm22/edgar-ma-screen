"""Certificate Transparency -> signals(kind='ct').

crt.sh exposes its Postgres read replica publicly, which beats the JSON
endpoint for bulk work. Two operational facts learned the hard way:

  * The pooler runs in statement-pooling mode and rejects transaction blocks,
    so the connection must be autocommit.
  * `certificate_identity` has been superseded by a full-text index on
    `certificate`; queries go through identities()/plainto_tsquery now.

The signal is NOVEL hostnames per ISO week -- a hostname's FIRST appearance.
Renewals of known hosts are noise; first appearances are the event.
"""
import datetime as dt

import psycopg

DSN = "host=crt.sh port=5432 dbname=certwatch user=guest connect_timeout=30"

QUERY = """
SELECT min(le.ENTRY_TIMESTAMP) AS first_seen,
       x509_commonName(c.CERTIFICATE) AS host
FROM certificate c
JOIN ct_log_entry le ON le.CERTIFICATE_ID = c.ID
WHERE plainto_tsquery('certwatch', %s) @@ identities(c.CERTIFICATE)
GROUP BY 2
"""

STATEMENT_TIMEOUT_MS = 180_000


def iso_monday(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def novel_by_week(rows: list[tuple]) -> list[dict]:
    """rows: [(first_seen_ts, hostname)] -> weekly counts of first appearances."""
    weekly: dict[dt.date, int] = {}
    seen: set[str] = set()
    for ts, host in sorted(rows, key=lambda r: r[0]):
        if host is None or host in seen:
            continue
        seen.add(host)
        week = iso_monday(ts.date())
        weekly[week] = weekly.get(week, 0) + 1
    return [{"public_ts": w, "value": float(n)} for w, n in sorted(weekly.items())]


def query_domain(domain: str, retries: int = 2) -> list[tuple]:
    """crt.sh kills long queries via its replication system; the documented
    workaround is to retry, since the first attempt primes the cache."""
    last: Exception | None = None
    for _ in range(retries):
        try:
            with psycopg.connect(DSN, autocommit=True) as conn, conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
                cur.execute(QUERY, (domain,))
                return cur.fetchall()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"crt.sh failed for {domain}") from last


def rows_for(cik: str, domain: str) -> list[dict]:
    return [{"cik": cik, **r} for r in novel_by_week(query_domain(domain))]


def insert(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO signals VALUES ($cik, 'ct', $public_ts, $value)",
        rows,
    )
    return len(rows)
