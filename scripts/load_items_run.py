"""Fetch 8-K item codes for every company in the universe."""
import duckdb
from deal import config, load_items, warehouse

con = warehouse.connect("data/items.duckdb")
load_items.init_schema(con)
src = duckdb.connect(":memory:")
src.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
ciks = [r[0] for r in src.execute(
    "SELECT DISTINCT cik FROM m.universe ORDER BY cik").fetchall()]
done = {r[0] for r in con.execute("SELECT DISTINCT cik FROM item_events").fetchall()}
todo = [c for c in ciks if c not in done]
print(f"{len(todo):,} companies to fetch", flush=True)

total = 0
for i, cik in enumerate(todo, 1):
    try:
        total += load_items.insert(con, load_items.fetch_company(cik, config.PANEL_START_YEAR))
    except Exception:
        pass
    if i % 500 == 0:
        print(f"  {i:,}/{len(todo):,}  events {total:,}", flush=True)
print(f"item events: {total:,}", flush=True)
for fam, n in con.execute(
        "SELECT family, sum(n) FROM item_events GROUP BY 1 ORDER BY 2 DESC").fetchall():
    print(f"  {fam:<22} {n:>9,}", flush=True)
