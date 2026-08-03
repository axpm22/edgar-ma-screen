"""Mine every tracked form family out of the cached EDGAR indexes.

Writes data/forms2.duckdb. No network: the master indexes were already
downloaded for the universe build, and this reads the ~11M filings the
universe build ignored.

There was no committed runner for this database -- it was built ad hoc, which
is why extending the panel to 2012 could not simply be re-run. This is that
runner. data/forms.duckdb is the superseded 6-family version; forms2 is a
strict superset (identical on the six shared families, plus shelf and raise),
so only this one is maintained.

    .venv/bin/python scripts/load_forms_run.py
"""
from deal import config, load_forms, warehouse

con = warehouse.connect("data/forms2.duckdb")
n = load_forms.load(con, config.PANEL_START_YEAR, config.PANEL_END_YEAR)
print(f"\n{n:,} form events")
for fam, c, lo, hi in con.execute(
        "SELECT family, count(*), min(public_ts), max(public_ts) "
        "FROM form_events GROUP BY 1 ORDER BY 1").fetchall():
    print(f"  {fam:<10} {c:>9,}  {lo} .. {hi}")
con.close()
