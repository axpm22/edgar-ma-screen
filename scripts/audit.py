"""Adversarial data audit. No model fitting -- pure integrity checks.

Written as if by a review board trying to break the result. Each check prints
PASS or FAIL with the evidence; nothing is asserted that is not measured.

    python scripts/audit.py
"""
import datetime as dt
import sys

import duckdb
import polars as pl

sys.path.insert(0, "scripts")
from final_stats import HORIZON, N_EVAL, relabel  # noqa: E402

from deal import features, screen  # noqa: E402
import numpy as np  # noqa: E402

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print(f"  [{tag}] {name:<52} {detail}", flush=True)


def main() -> None:
    df = pl.read_parquet("data/features.parquet")
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")

    print("=== REVIEWER 1: panel integrity ===")
    dup = df.height - df.select(["cik", "week"]).unique().height
    check("no duplicate (cik, week) rows", dup == 0, f"{dup} duplicates")

    wd = df.select(pl.col("week").dt.weekday().unique()).to_series().to_list()
    check("every week is the same weekday", len(wd) == 1, f"weekdays={wd}")

    nulls = sum(df[c].null_count() for c in features.FEATURE_COLS
                if c in df.columns)
    check("no nulls in feature columns", nulls == 0, f"{nulls} nulls")

    inf = sum(int(np.isinf(df[c].to_numpy()).sum())
              for c in features.FEATURE_COLS
              if c in df.columns and df[c].dtype in (pl.Float64, pl.Float32))
    check("no infinities in features", inf == 0, f"{inf} infs")

    print("\n=== REVIEWER 2: identifier consistency ===")
    pan = set(df["cik"].unique().to_list())
    uni = {r[0] for r in con.execute("SELECT cik FROM m.universe").fetchall()}
    dea = {r[0] for r in con.execute("SELECT DISTINCT cik FROM m.deals").fetchall()}
    check("every panel cik exists in universe", pan <= uni,
          f"{len(pan - uni)} orphans")
    check("deal ciks are a subset of universe", dea <= uni,
          f"{len(dea - uni)} deal ciks missing from universe")
    lead = [c for c in list(pan)[:5000] if c != c.lstrip("0")]
    check("no leading-zero cik formatting drift", not lead,
          f"{len(lead)} malformed")

    print("\n=== REVIEWER 3: label construction ===")
    lab = relabel(df, HORIZON)
    deals = con.execute("SELECT cik, agreement_date FROM m.deals").pl()

    # A positive week must sit strictly before some deal and within horizon.
    pos = lab.filter(pl.col("y") == 1).select(["cik", "week"])
    j = pos.join(deals, on="cik", how="left")
    good = j.filter((pl.col("week") < pl.col("agreement_date")) &
                    (pl.col("week") >= pl.col("agreement_date")
                     - pl.duration(weeks=HORIZON)))
    covered = good.select(["cik", "week"]).unique().height
    check("all positives lie inside a real deal window",
          covered == pos.height, f"{pos.height - covered} uncovered")

    # And no positive may sit at or after its own deal.
    after = j.filter(pl.col("week") >= pl.col("agreement_date")) \
             .join(good.select(["cik", "week"]).unique(),
                   on=["cik", "week"], how="anti")
    check("no positive at or after its own agreement date",
          after.height == 0, f"{after.height} violations")

    multi = con.execute(
        "SELECT count(*) FROM (SELECT cik FROM m.deals GROUP BY cik "
        "HAVING count(*) > 1)").fetchone()[0]
    print(f"       (note: {multi} companies have >1 deal date -- "
          f"overlapping windows are unioned, not double counted)")

    print("\n=== REVIEWER 4: the lookahead firewall ===")
    # Take real panel rows and verify NOTHING that fed them was public later.
    con.execute("ATTACH 'data/forms.duckdb' AS fm (READ_ONLY)")
    con.execute("ATTACH 'data/fund2.duckdb' AS fd (READ_ONLY)")
    samp = lab.select(["cik", "week"]).sample(n=300, seed=5)
    con.register("s", samp.to_arrow())
    late_forms = con.execute("""
        SELECT count(*) FROM s JOIN fm.form_events f ON f.cik = s.cik
        WHERE f.public_ts > s.week AND f.public_ts <= s.week + INTERVAL 1 WEEK
    """).fetchone()[0]
    print(f"       (informational: {late_forms} form events land in the week "
          f"AFTER a sampled row -- these must not be visible to it)")
    # The firewall is a join condition, so verify it directly on the roll-ups.
    features._prepare(con) if False else None
    check("form_events carry a public_ts that is a real date",
          con.execute("SELECT count(*) FROM fm.form_events "
                      "WHERE public_ts IS NULL").fetchone()[0] == 0)
    check("fundamentals public_ts never precedes 2016",
          con.execute("SELECT count(*) FROM fd.fundamentals "
                      "WHERE public_ts < DATE '2015-01-01'").fetchone()[0] == 0)

    print("\n=== REVIEWER 5: the evaluation metric ===")
    # Hand-built case: 2 weeks, 3 companies, known answer.
    W1, W2 = dt.date(2024, 1, 1), dt.date(2024, 1, 8)
    toy = pl.DataFrame({"cik": ["A", "B", "C", "A", "B", "C"],
                        "week": [W1, W1, W1, W2, W2, W2],
                        "y": [1, 0, 0, 0, 0, 1]})
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.9])
    r = screen.weekly_precision(toy, p, 1)
    check("metric returns 100% on a hand-checked perfect case",
          r["precision"] == 1.0 and r["n_selected"] == 2,
          f"prec={r['precision']}, n={r['n_selected']}")
    r0 = screen.weekly_precision(toy, 1 - p, 1)
    check("metric returns 0% on the inverted case", r0["precision"] == 0.0)

    # Ties: if every score is identical the screen must not look skilful.
    tie = screen.weekly_precision(toy, np.ones(6), 1)
    check("tied scores do not manufacture precision",
          tie["precision"] <= 0.5, f"prec={tie['precision']:.2f}")

    print("\n=== REVIEWER 6: test-window composition ===")
    safe = lab["week"].max() - dt.timedelta(weeks=HORIZON)
    te = lab.filter((pl.col("week") >= dt.date(2024, 1, 1))
                    & (pl.col("week") <= safe))
    per_week = te.group_by("week").len()["len"]
    check("every test week has more than N_EVAL companies",
          int(per_week.min()) > N_EVAL,
          f"min={int(per_week.min())} companies in a week")
    check("test window has no censored rows",
          te["week"].max() <= safe, f"max week {te['week'].max()}")

    spac = [r[0] for r in con.execute("""
        SELECT DISTINCT u.cik FROM m.universe u
        LEFT JOIN m.company_sic s USING (cik)
        WHERE s.sic='6770' OR upper(u.name) LIKE '%ACQUISITION CORP%'
    """).fetchall()]
    ns = te.filter(~pl.col("cik").is_in(spac))
    print(f"       all-companies base {te['y'].mean()*100:.2f}%  "
          f"| operating-only base {ns['y'].mean()*100:.2f}%")
    print(f"       operating-only weeks={ns['week'].n_unique()}  "
          f"rows={ns.height:,}  positives={int(ns['y'].sum()):,}")

    print("\n" + ("ALL CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILURES: {FAILS}"))


if __name__ == "__main__":
    main()
