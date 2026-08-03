"""Load the three new data sources. Run: python scripts/load_extras.py <stage>"""
import sys

from deal import (config, load_fts, load_fund, load_insider_value,
                  warehouse)

START, END = config.PANEL_START_YEAR, config.PANEL_END_YEAR


def stage_fts(con):
    print("[fts] EDGAR full-text search", flush=True)
    print(f"[fts] {load_fts.load(con, START, END):,} rows", flush=True)


def stage_insider_value(con):
    print("[insider_value] Form 4 dollar values", flush=True)
    print(f"[insider_value] {load_insider_value.load(con, START, END, verbose=False):,} rows", flush=True)


def stage_fund_extra(con):
    print("[fund] reloading with 12 tags", flush=True)
    print(f"[fund] {load_fund.load_all(con, START, END, verbose=False):,} new facts", flush=True)


STAGES = {"fts": stage_fts, "insider_value": stage_insider_value,
          "fund": stage_fund_extra}

if __name__ == "__main__":
    stage = sys.argv[1]
    con = warehouse.connect("data/deal.duckdb")
    warehouse.init_schema(con)
    STAGES[stage](con)
    con.close()
    print(f"[{stage}] done", flush=True)
