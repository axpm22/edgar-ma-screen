"""Build the full dataset. Resumable -- every download caches to data/raw/.

    python scripts/build_dataset.py <stage>

Stages:
    index    EDGAR master indexes -> universe + crosswalk
    insider  Insider Transactions Data Sets -> insider_trans
    fund     Financial Statement Data Sets -> fundamentals
    sic      CIK -> SIC industry code (re-reads cached sub.txt)
    panel    clean DEFM14A labels -> panel -> y
    all      every stage in dependency order
"""
import datetime as dt
import sys
import time

from deal import (config, crosswalk, fetch, load_fund, load_insider, load_sic,
                  panel, universe, warehouse)

START_YEAR, END_YEAR = 2016, 2026
HORIZON_WEEKS = config.HORIZON_WEEKS


def _quarters():
    today = dt.date.today()
    for y in range(START_YEAR, END_YEAR + 1):
        for q in (1, 2, 3, 4):
            if dt.date(y, (q - 1) * 3 + 1, 1) > today:
                return
            yield y, q


def _all_index_rows():
    rows = []
    for y, q in _quarters():
        try:
            raw = fetch.sec_get(config.IDX_URL.format(year=y, q=q))
        except Exception:
            continue
        rows.extend(universe.parse_master_idx(raw))
    return rows


def stage_index(con) -> None:
    print(f"[index] EDGAR master indexes {START_YEAR}-{END_YEAR}", flush=True)
    n = universe.build(con, START_YEAR, END_YEAR)
    companies = con.execute("SELECT count(*) FROM universe").fetchone()[0]
    print(f"[index] {n:,} periodic filings, {companies:,} companies", flush=True)
    print(f"[index] {crosswalk.build_names(con):,} crosswalk names", flush=True)


def stage_insider(con) -> None:
    print("[insider] Form 3/4/5 data sets", flush=True)
    total = 0
    for y, q in _quarters():
        try:
            n = load_insider.load_quarter(con, y, q)
        except Exception as exc:
            print(f"  {y}Q{q}: SKIP ({type(exc).__name__}: {exc})", flush=True)
            continue
        total += n
        print(f"  {y}Q{q}: {n:>8,} transactions", flush=True)
    print(f"[insider] {total:,} new rows", flush=True)


def stage_fund(con) -> None:
    print("[fund] Financial statement data sets", flush=True)
    t0 = time.time()
    n = load_fund.load_all(con, START_YEAR, END_YEAR)
    print(f"[fund] {n:,} facts in {time.time()-t0:.0f}s", flush=True)


def stage_sic(con) -> None:
    n = load_sic.load(con, START_YEAR, END_YEAR)
    distinct = con.execute("SELECT count(*) FROM company_sic").fetchone()[0]
    print(f"[sic] {n:,} rows -> {distinct:,} companies with a SIC", flush=True)


def stage_panel(con) -> None:
    print("[panel] rebuilding clean DEFM14A labels", flush=True)
    n = panel.rebuild_deals(con, _all_index_rows())
    print(f"[panel] {n:,} clean deals", flush=True)

    rows = panel.build(con, dt.date(START_YEAR, 1, 1), dt.date.today())
    print(f"[panel] {rows:,} company-weeks", flush=True)

    pos = panel.label(con, HORIZON_WEEKS)
    total = con.execute("SELECT count(*) FROM panel").fetchone()[0]
    print(f"[panel] {pos:,} positive weeks ({100*pos/total:.3f}% of {total:,})",
          flush=True)


def stage_all(con) -> None:
    for fn in (stage_index, stage_insider, stage_fund, stage_sic, stage_panel):
        fn(con)


STAGES = {
    "index": stage_index, "insider": stage_insider, "fund": stage_fund,
    "sic": stage_sic, "panel": stage_panel, "all": stage_all,
}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    con = warehouse.connect()
    warehouse.init_schema(con)
    load_sic.init_schema(con)
    STAGES[stage](con)
    con.close()
    print(f"[{stage}] done", flush=True)
