"""DuckDB warehouse. Rebuild-from-scratch is cheap; there are no migrations."""
from pathlib import Path

import duckdb

SCHEMA = """
-- Point-in-time listing spans, derived from periodic filing activity.
CREATE TABLE IF NOT EXISTS universe (
    cik        VARCHAR PRIMARY KEY,
    name       VARCHAR,
    listed     DATE,
    delisted   DATE
);

CREATE TABLE IF NOT EXISTS panel (
    cik        VARCHAR,
    week       DATE,
    PRIMARY KEY (cik, week)
);

CREATE TABLE IF NOT EXISTS deals (
    cik            VARCHAR,
    agreement_date DATE,
    rumor_date     DATE,
    acquirer       VARCHAR,
    PRIMARY KEY (cik, agreement_date)
);

-- Every signal lands here. public_ts is when an OUTSIDER could have known.
CREATE TABLE IF NOT EXISTS signals (
    cik        VARCHAR,
    kind       VARCHAR,
    public_ts  DATE,
    value      DOUBLE,
    PRIMARY KEY (cik, kind, public_ts)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    cik       VARCHAR,
    tag       VARCHAR,
    public_ts DATE,
    value     DOUBLE,
    PRIMARY KEY (cik, tag, public_ts)
);

CREATE TABLE IF NOT EXISTS insider_trans (
    accession     VARCHAR,
    cik           VARCHAR,
    public_ts     DATE,
    trans_code    VARCHAR,
    shares        DOUBLE,
    discretionary BOOLEAN,
    PRIMARY KEY (accession, public_ts, trans_code, shares)
);

CREATE TABLE IF NOT EXISTS xwalk_name (
    norm_name VARCHAR PRIMARY KEY,
    cik       VARCHAR
);

CREATE TABLE IF NOT EXISTS xwalk_domain (
    cik    VARCHAR,
    domain VARCHAR,
    PRIMARY KEY (cik, domain)
);

-- Append-only. Never UPDATE, never DELETE.
CREATE TABLE IF NOT EXISTS predictions (
    run_ts     TIMESTAMP,
    cik        VARCHAR,
    week       DATE,
    prob       DOUBLE,
    features   VARCHAR
);
"""


def connect(path: str = "data/deal.duckdb") -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)
