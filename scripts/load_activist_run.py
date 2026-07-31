"""Activist identification from 13D filer/subject pairing."""
from deal import load_activist, warehouse

con = warehouse.connect("data/activist.duckdb")
n = load_activist.load(con, 2016, 2026)
print(f"activist events: {n:,}")
print("subjects:", con.execute("SELECT count(DISTINCT cik) FROM activist_events").fetchone()[0])
print("reach distribution:")
for lo, cnt in con.execute(
    "SELECT filer_targets/50*50 AS bucket, count(*) FROM activist_events "
    "GROUP BY 1 ORDER BY 1 LIMIT 8").fetchall():
    print(f"  filer_targets ~{int(lo):>4}+  {cnt:>6,}")
