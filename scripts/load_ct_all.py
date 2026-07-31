"""Fetch Certificate Transparency history for every crosswalked domain.

crt.sh is a free shared service, so concurrency is deliberately low. Results
stream into a Parquet-friendly staging table as they arrive, making the job
resumable -- a domain already present is skipped on the next run.

    python scripts/load_ct_all.py [max_workers]
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from deal import load_ct, warehouse

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def main() -> None:
    # Separate DB file: DuckDB allows one writer, and the main warehouse
    # must stay writable for feature rebuilds while this long job runs.
    con = warehouse.connect("data/ct.duckdb")
    con.execute("ATTACH IF NOT EXISTS 'data/deal.duckdb' AS main_db (READ_ONLY)")
    con.execute("CREATE TABLE IF NOT EXISTS signals (cik VARCHAR, kind VARCHAR, public_ts DATE, value DOUBLE, PRIMARY KEY (cik, kind, public_ts))")
    con.execute("""
        CREATE TABLE IF NOT EXISTS ct_done (
            cik VARCHAR, domain VARCHAR, n_weeks INTEGER, ok BOOLEAN,
            PRIMARY KEY (cik, domain)
        )
    """)

    todo = con.execute("""
        SELECT d.cik, d.domain
        FROM main_db.xwalk_domain d
        JOIN main_db.universe u USING (cik)
        LEFT JOIN ct_done t ON t.cik = d.cik AND t.domain = d.domain
        WHERE t.cik IS NULL
        -- companies with a deal first: they carry the signal being tested
        ORDER BY (d.cik IN (SELECT cik FROM main_db.deals)) DESC, d.cik
    """).fetchall()
    print(f"{len(todo):,} domains to fetch, {WORKERS} workers", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(load_ct.rows_for, cik, dom): (cik, dom)
                   for cik, dom in todo}
        for fut in as_completed(futures):
            cik, dom = futures[fut]
            try:
                rows = fut.result()
                load_ct.insert(con, rows)
                con.execute("INSERT OR IGNORE INTO ct_done VALUES (?,?,?,?)",
                            [cik, dom, len(rows), True])
            except Exception:
                con.execute("INSERT OR IGNORE INTO ct_done VALUES (?,?,?,?)",
                            [cik, dom, 0, False])
            done += 1
            if done % 25 == 0:
                print(f"  {done:,}/{len(todo):,}", flush=True)

    ok, bad = con.execute(
        "SELECT count(*) FILTER (WHERE ok), count(*) FILTER (WHERE NOT ok) "
        "FROM ct_done"
    ).fetchone()
    sig = con.execute("SELECT count(*) FROM signals WHERE kind='ct'").fetchone()[0]
    print(f"done: {ok:,} ok, {bad:,} failed, {sig:,} ct signal rows", flush=True)


if __name__ == "__main__":
    main()
