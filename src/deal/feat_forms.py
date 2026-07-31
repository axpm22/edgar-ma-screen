"""form_events -> rolling per-week features.

Windows are trailing and inclusive of the current week, so a filing is visible
the week it lands and never earlier. sc13d_new isolates a FIRST 13D in the
trailing year -- a new activist arriving is a different event from one who has
been sitting on the register for years.
"""

FORM_COLS = [
    "sc13d_52w", "sc13g_52w", "form8k_26w", "def14a_52w",
    "s4_52w", "late_52w", "sc13d_new",
]


def prepare(con) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE form_week AS
        SELECT cik, date_trunc('week', public_ts) AS week,
               sum(CASE WHEN family='sc13d'  THEN n ELSE 0 END) AS sc13d,
               sum(CASE WHEN family='sc13g'  THEN n ELSE 0 END) AS sc13g,
               sum(CASE WHEN family='form8k' THEN n ELSE 0 END) AS form8k,
               sum(CASE WHEN family='def14a' THEN n ELSE 0 END) AS def14a,
               sum(CASE WHEN family='s4'     THEN n ELSE 0 END) AS s4,
               sum(CASE WHEN family='late'   THEN n ELSE 0 END) AS late
        FROM form_events GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE form_roll AS
        SELECT cik, week,
               sum(sc13d)  OVER w52 AS sc13d_52w,
               sum(sc13g)  OVER w52 AS sc13g_52w,
               sum(form8k) OVER w26 AS form8k_26w,
               sum(def14a) OVER w52 AS def14a_52w,
               sum(s4)     OVER w52 AS s4_52w,
               sum(late)   OVER w52 AS late_52w,
               -- A 13D this week with none in the preceding year: a NEW
               -- activist on the register, not a standing one.
               CASE WHEN sc13d > 0
                     AND coalesce(sum(sc13d) OVER wprior, 0) = 0
                    THEN 1 ELSE 0 END AS sc13d_new
        FROM form_week
        WINDOW w52 AS (PARTITION BY cik ORDER BY week
                       RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND CURRENT ROW),
               w26 AS (PARTITION BY cik ORDER BY week
                       RANGE BETWEEN INTERVAL 182 DAY PRECEDING AND CURRENT ROW),
               wprior AS (PARTITION BY cik ORDER BY week
                          RANGE BETWEEN INTERVAL 365 DAY PRECEDING
                                    AND INTERVAL 7 DAY PRECEDING)
    """)
