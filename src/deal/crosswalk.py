"""CIK <-> normalized name <-> domain.

Every non-SEC source joins through this table, so its match rate bounds the
coverage of every non-SEC signal. Report that rate rather than assuming it.

ponytail: exact match on a normalized string, not fuzzy matching. Fuzzy
matching produces confident false positives ("DELTA AIR" vs "DELTA APPAREL"),
and a false join injects another company's signal into the panel -- strictly
worse than a missing row. Add blocked fuzzy matching only after measuring
what the exact matcher misses.
"""
import re

SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LLC", "LP", "LTD", "LIMITED", "PLC", "SA", "NV", "AG", "HOLDINGS",
    "HOLDING", "GROUP", "THE",
}


def normalize_name(name: str) -> str:
    up = name.upper()
    up = re.sub(r"[^A-Z0-9 ]", "", up)
    tokens = [t for t in up.split() if t]
    while tokens and tokens[0] in SUFFIXES:
        tokens.pop(0)
    while tokens and tokens[-1] in SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def build_names(con) -> int:
    rows = con.execute("SELECT cik, name FROM universe").fetchall()
    out = [{"norm_name": normalize_name(n), "cik": c} for c, n in rows]
    out = [r for r in out if r["norm_name"]]
    if not out:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO xwalk_name VALUES ($norm_name, $cik)", out
    )
    return len(out)


def load_domains(con, rows: list[dict]) -> int:
    if not rows:
        return 0
    con.executemany(
        "INSERT OR IGNORE INTO xwalk_domain VALUES ($cik, $domain)", rows
    )
    return len(rows)


def match_rate(con, table: str, column: str) -> float:
    """Fraction of distinct values in table.column that resolve to a CIK."""
    total, hit = con.execute(
        f"""
        SELECT count(*), count(*) FILTER (WHERE x.cik IS NOT NULL)
        FROM (SELECT DISTINCT {column} AS v FROM {table}) t
        LEFT JOIN xwalk_name x ON x.norm_name = t.v
        """
    ).fetchone()
    return (hit / total) if total else 0.0
