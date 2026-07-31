"""SEC Financial Statement Data Sets -> fundamentals.

One quarterly ZIP holds every XBRL numeric fact from every filer. sub.txt maps
accession -> CIK, SIC and filing date; num.txt holds the facts.

num.txt is ~490MB uncompressed per quarter and there are ~44 of them. Parsing
that in Python line-by-line takes hours; DuckDB scans the same files natively
in a fraction of the time, so the members are extracted to a scratch file and
handed to DuckDB rather than iterated here.
"""
import datetime as dt
import io
import tempfile
import zipfile
from pathlib import Path

from . import config, fetch

# Only the tags the model's control block consumes.
TAGS = (
    # Balance sheet
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "CashAndCashEquivalentsAtCarryingValue",
    "StockholdersEquity",
    "Goodwill",
    "IntangibleAssetsNetExcludingGoodwill",
    "InventoryNet",
    # Ambrose & Megginson (1992): acquisition likelihood is POSITIVELY
    # related to tangible assets -- they are collateral for a leveraged bid.
    "PropertyPlantAndEquipmentNet",
    # Their one defense that actually deterred bids was blank-check preferred.
    "PreferredStockValue",
    "AccountsReceivableNetCurrent",
    # Debt. LongTermDebtNoncurrent alone understates leverage -- the current
    # portion and short-term borrowings are real obligations and were missing.
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "ShortTermBorrowings",
    # Income statement
    "Revenues",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "ResearchAndDevelopmentExpense",
    "DepreciationDepletionAndAmortization",
    # Cash flow. Free cash flow is what PE and LBO buyers screen on, and
    # take-privates are a large share of deal flow -- this was the single
    # biggest hole in the financial data.
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsOfDividendsCommonStock",
    # Share count: its decline is a buyback proxy needing no price data.
    "CommonStockSharesOutstanding",
)


def load_quarter(con, year: int, q: int) -> int:
    blob = fetch.sec_get(config.FUND_URL.format(year=year, q=q))
    tags = "(" + ", ".join(f"'{t}'" for t in TAGS) + ")"

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for member in ("sub.txt", "num.txt"):
                (tmpdir / member).write_bytes(z.read(member))

        before = con.execute("SELECT count(*) FROM fundamentals").fetchone()[0]
        con.execute(f"""
            INSERT OR IGNORE INTO fundamentals
            SELECT n.cik, n.tag, n.public_ts, n.value FROM (
                SELECT s.cik::VARCHAR AS raw_cik,
                       ltrim(s.cik::VARCHAR, '0') AS cik,
                       num.tag AS tag,
                       strptime(s.filed::VARCHAR, '%Y%m%d')::DATE AS public_ts,
                       num.value AS value
                FROM read_csv('{tmpdir/"num.txt"}', delim='\t', header=true,
                              quote='', escape='',
                              ignore_errors=true, sample_size=-1) num
                JOIN read_csv('{tmpdir/"sub.txt"}', delim='\t', header=true,
                              quote='', escape='',
                              ignore_errors=true, sample_size=-1) s
                  ON s.adsh = num.adsh
                WHERE num.tag IN {tags}
                  AND num.value IS NOT NULL
                  -- segments/coreg populated means a business-unit or
                  -- subsidiary breakdown, which would double-count against
                  -- the consolidated total.
                  AND (num.segments IS NULL OR num.segments = '')
                  AND (num.coreg   IS NULL OR num.coreg   = '')
            ) n
            WHERE n.cik <> '' AND n.public_ts IS NOT NULL
        """)
        after = con.execute("SELECT count(*) FROM fundamentals").fetchone()[0]
    return after - before


def load_all(con, start_year: int, end_year: int, verbose: bool = True) -> int:
    today = dt.date.today()
    total = 0
    for y in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            if dt.date(y, (q - 1) * 3 + 1, 1) > today:
                continue
            try:
                n = load_quarter(con, y, q)
            except Exception as exc:
                if verbose:
                    print(f"  {y}Q{q}: SKIP ({type(exc).__name__}: {exc})",
                          flush=True)
                continue
            total += n
            if verbose:
                print(f"  {y}Q{q}: {n:>8,} facts", flush=True)
    return total
