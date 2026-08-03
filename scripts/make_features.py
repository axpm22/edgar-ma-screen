"""Assemble the feature matrix and write it to Parquet.

    python scripts/make_features.py

Writes data/features.parquet and prints a summary: row counts, positive rate,
per-feature coverage, and a lookahead assertion.
"""
import polars as pl

from deal import features, warehouse

OUT = "data/features.parquet"


def main() -> None:
    # Main warehouse read-only: the CT job holds it while it runs.
    # Side databases attach so every source is visible in one query.
    con = warehouse.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("ATTACH 'data/float.duckdb' AS f (READ_ONLY)")
    con.execute("ATTACH 'data/fund2.duckdb' AS fd (READ_ONLY)")
    con.execute("ATTACH 'data/items.duckdb' AS it (READ_ONLY)")
    # forms2, not forms: identical on the six shared families and adds shelf
    # and raise, so the old 6-family database is dead weight. Verified equal
    # family-by-family before switching.
    con.execute("ATTACH 'data/forms2.duckdb' AS fm (READ_ONLY)")
    con.execute("ATTACH 'data/activist.duckdb' AS av (READ_ONLY)")
    con.execute("ATTACH 'data/ct.duckdb' AS ctdb (READ_ONLY)")
    con.execute("USE m")
    con.execute("CREATE OR REPLACE TEMP VIEW public_float AS SELECT * FROM f.public_float")
    # Expanded 25-tag reload lives in its own database.
    con.execute("CREATE OR REPLACE TEMP VIEW fundamentals AS SELECT * FROM fd.fundamentals")
    con.execute("CREATE OR REPLACE TEMP VIEW item_events AS SELECT * FROM it.item_events")
    con.execute("CREATE OR REPLACE TEMP VIEW form_events AS SELECT * FROM fm.form_events")
    con.execute("CREATE OR REPLACE TEMP VIEW activist_events AS SELECT * FROM av.activist_events")
    con.execute("CREATE OR REPLACE TEMP VIEW signals AS SELECT * FROM ctdb.signals")
    print("building feature matrix...", flush=True)
    df = features.build(con)

    df.write_parquet(OUT)
    print(f"\nwrote {OUT}  ({df.height:,} rows x {df.width} cols)\n")

    pos = int(df["y"].sum())
    print(f"positives     {pos:,}")
    print(f"positive rate {100 * pos / df.height:.4f}%")
    print(f"companies     {df['cik'].n_unique():,}")
    print(f"weeks         {df['week'].min()} .. {df['week'].max()}\n")

    print(f"{'feature':<24} {'nonzero%':>9} {'mean':>14} {'std':>14}")
    print("-" * 65)
    for col in features.FEATURE_COLS:
        s = df[col]
        nz = 100.0 * (s != 0).sum() / df.height
        print(f"{col:<24} {nz:>8.2f}% {s.mean():>14.4f} {s.std():>14.4f}")

    # A positive week must never sit at or after its own agreement date.
    bad = con.execute("""
        SELECT count(*) FROM panel p JOIN deals d USING (cik)
        WHERE p.y = 1 AND p.week >= d.agreement_date
          AND NOT EXISTS (
              SELECT 1 FROM deals d2
              WHERE d2.cik = p.cik AND p.week < d2.agreement_date
          )
    """).fetchone()[0]
    print(f"\nlookahead check (positives on/after their deal): {bad}")
    assert bad == 0, "label leakage: positive weeks at or after announcement"
    print("PASS")


if __name__ == "__main__":
    main()
