"""Build data/pairs.duckdb from the cached EDGAR indexes. No network.

    .venv/bin/python scripts/load_pairs_run.py

Runtime ~40s, dominated by parsing 43 quarterly index files.
"""
import duckdb

from deal import load_pairs

con = duckdb.connect("data/pairs.duckdb")
meta = duckdb.connect("data/deal.duckdb", read_only=True)
delisted = {r[0]: r[1] for r in
            meta.execute("SELECT cik, delisted FROM universe").fetchall()}

counts = load_pairs.build(con, delisted)
print(f"two-party accessions {counts['two_party']:,}  "
      f"oriented {counts['oriented']:,}  ambiguous {counts['ambiguous']:,}")
print(f"collapsed to {counts['episodes']:,} deal episodes")

tender = duckdb.connect("data/tender.duckdb", read_only=True)
v = load_pairs.validate_orientation(con, tender)
print(f"orientation vs tender.duckdb subjects: "
      f"{v['agree']}/{v['checked']} = {v['pct']:.1f}%")

print("\nby form:")
for form, n in con.execute(
        "SELECT form, count(*) FROM deal_pairs GROUP BY 1 ORDER BY 2 DESC"
).fetchall():
    print(f"  {form:<10} {n:>6,}")

print("\nby year:")
for yr, n in con.execute(
        "SELECT year(first_ts), count(*) FROM deal_pairs GROUP BY 1 ORDER BY 1"
).fetchall():
    print(f"  {yr}  {n:>5,}")

top = con.execute("""
    SELECT acquirer_cik, count(*) n FROM deal_pairs
    GROUP BY 1 ORDER BY 2 DESC LIMIT 5
""").fetchall()
names = {r[0]: r[1] for r in
         meta.execute("SELECT cik, name FROM universe").fetchall()}
print("\nmost acquisitive:")
for cik, n in top:
    print(f"  {names.get(cik, cik):<40} {n:>3} deals")
con.close()
