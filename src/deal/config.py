"""Constants for the M&A precursor project."""
import os

# The SEC requires a descriptive User-Agent carrying real contact details on
# every request, and rate-limits or blocks requests without one. Set
# EDGAR_UA in your environment before running any loader; the default is a
# placeholder so a personal address is never committed.
EDGAR_UA = os.environ.get(
    "EDGAR_UA", "ma-signals research your.email@example.com")

IDX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
FUND_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{q}.zip"
INSIDER_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{year}q{q}_form345.zip"
)

CRT_DSN = "host=crt.sh port=5432 dbname=certwatch user=guest connect_timeout=30"

# Horizon the hazard model predicts over, in weeks.
HORIZON_WEEKS = 26

# First year of the panel. Was 2016, duplicated across five loader scripts;
# one constant now, because five literals drift and a half-extended warehouse
# is worse than an un-extended one.
#
# 2012 is a floor set by XBRL phase-in, not by availability. The DERA
# financial-statement sets nominally start 2009Q2, but 2010Q1 is 5.3 MB
# against 2012Q1's 85.5 MB, and dei:EntityPublicFloat covers 702 companies in
# CY2009 against 4,834 by CY2011 -- small filers were not required to tag
# until 2011. Pulling earlier buys years whose coverage is biased toward large
# companies, which lands squarely on log_assets, the model's top feature.
PANEL_START_YEAR = 2012
PANEL_END_YEAR = 2026
