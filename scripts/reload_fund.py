"""Re-scan the cached quarterly ZIPs with the expanded tag set.

Writes to a side database: deal.duckdb is held read-only by the 8-K item
job, and DuckDB allows one writer.
"""
import time
from deal import load_fund, warehouse

con = warehouse.connect("data/fund2.duckdb")
con.execute("""CREATE TABLE IF NOT EXISTS fundamentals (
    cik VARCHAR, tag VARCHAR, public_ts DATE, value DOUBLE,
    PRIMARY KEY (cik, tag, public_ts))""")
t0 = time.time()
n = load_fund.load_all(con, 2016, 2026)
print(f"\n{n:,} facts in {time.time()-t0:.0f}s")
for t, c, co in con.execute(
        "SELECT tag, count(*), count(DISTINCT cik) FROM fundamentals "
        "GROUP BY 1 ORDER BY 2 DESC").fetchall():
    print(f"  {t:<46} {c:>8,} rows {co:>6,} cos")
