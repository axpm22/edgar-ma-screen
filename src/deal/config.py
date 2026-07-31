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
