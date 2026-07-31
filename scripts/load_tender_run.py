import datetime as dt
from deal import config, fetch, load_tender, universe, warehouse

con = warehouse.connect("data/tender.duckdb")
load_tender.init_schema(con)
rows = []
for y, q in universe.quarters(2016, 2026):
    if dt.date(y, (q-1)*3+1, 1) > dt.date.today():
        break
    try:
        rows += universe.parse_master_idx(fetch.sec_get(config.IDX_URL.format(year=y, q=q)))
    except Exception:
        continue
print(f"index rows {len(rows):,}", flush=True)
found = load_tender.collect(rows)
print(f"tender offers with a resolved subject: {load_tender.insert(con, found):,}", flush=True)
print("distinct target companies:",
      con.execute("SELECT count(DISTINCT cik) FROM tender_offers").fetchone()[0])
for y, n in con.execute(
        "SELECT year(public_ts), count(*) FROM tender_offers GROUP BY 1 ORDER BY 1").fetchall():
    print(f"  {y}  {n:>4}")
