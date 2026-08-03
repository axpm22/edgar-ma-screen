"""EntityPublicFloat from the XBRL frames API -> data/float.duckdb.

No committed runner existed for this database either; it was built ad hoc.

Starts one year BEFORE the panel. Public float is measured at the fiscal-Q2
instant and only becomes public on the 10-K cover months later, so load_float
stamps public_ts = end_date + 270 days. Covering panel weeks from January of
PANEL_START_YEAR therefore needs frames from the preceding year.

Coverage is thin before 2011 -- 702 companies in CY2009 against 4,834 in
CY2011 -- because small filers were not required to tag until 2011. That is
the reason the panel starts in 2012 rather than 2009.

    .venv/bin/python scripts/load_float_run.py
"""
from deal import config, load_float, warehouse

con = warehouse.connect("data/float.duckdb")
n = load_float.load(con, config.PANEL_START_YEAR - 1, config.PANEL_END_YEAR)
print(f"\n{n:,} float observations")
for yr, c, cos in con.execute(
        "SELECT year(public_ts), count(*), count(DISTINCT cik) "
        "FROM public_float GROUP BY 1 ORDER BY 1").fetchall():
    print(f"  {yr}  {c:>6,} rows  {cos:>6,} companies")
con.close()
