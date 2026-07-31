"""8-K item codes -> rolling per-week features.

`form8k_26w` is a raw count: it says a company filed six 8-Ks, not what any of
them were about. That is why it ranks near the top on tree gain and near the
bottom on clustered inference. Item codes say what actually happened.

TWO FAMILIES ARE EXCLUDED AS LEAKAGE:

  Item 5.01 "change in control"    -- filed when control HAS changed. The deal
                                      already happened.
  Item 1.01 "material definitive
            agreement"             -- this is where a merger agreement itself
                                      is filed. Our label is the DEFM14A date,
                                      which lands 40-70 days AFTER that 8-K, so
                                      a 1.01 inside the prediction window is
                                      frequently the announcement being
                                      predicted.

Both would raise the score and destroy the claim. They are dropped at the
feature layer rather than filtered downstream, so no caller can reintroduce
them by accident.
"""

LEAKY_ITEMS = frozenset({"i_control_change", "i_material_agmt"})

ITEM_FAMILIES = [
    "i_officer_change",     # 5.02 departures and appointments
    "i_auditor_change",     # 4.01
    "i_nonreliance",        # 4.02 restatement
    "i_exit_costs",         # 2.05 restructuring
    "i_impairment",         # 2.06
    "i_security_rights",    # 3.03 where rights plans live
    "i_bylaw_change",       # 5.03
    "i_vote_results",       # 5.07
    "i_agmt_termination",   # 1.02 a deal falling apart
    "i_reg_fd",             # 7.01 selective-disclosure cures
    "i_other_events",       # 8.01 catch-all, often strategic news
]

ITEM_COLS = [f"{f}_52w" for f in ITEM_FAMILIES]


def prepare(con) -> None:
    """Build TEMP TABLE item_roll(cik, week, <family>_52w ...)."""
    sums = ",\n".join(
        f"               sum(CASE WHEN family='{f}' THEN n ELSE 0 END) AS {f}"
        for f in ITEM_FAMILIES)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE item_week AS
        SELECT cik, date_trunc('week', public_ts) AS week,
{sums}
        FROM item_events
        WHERE family NOT IN ({','.join(repr(x) for x in sorted(LEAKY_ITEMS))})
        GROUP BY 1, 2
    """)
    rolls = ",\n".join(
        f"               sum({f}) OVER w52 AS {f}_52w" for f in ITEM_FAMILIES)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE item_roll AS
        SELECT cik, week,
{rolls}
        FROM item_week
        WINDOW w52 AS (PARTITION BY cik ORDER BY week
                       RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND CURRENT ROW)
    """)
