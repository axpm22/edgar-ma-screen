"""Scope: does the merger proxy carry usable price history for the target?

No free source retains price history for delisted companies, which is exactly
where every positive label lives. But the target files its own price history:
Reg S-K Item 201 requires market-price disclosure, so a DEFM14A normally
carries a "Market Price of Common Stock" table -- quarterly high/low, usually
two years -- plus the per-share merger consideration.

If that parses reliably, the missing half of a survivorship-free price panel
comes free from EDGAR. If it does not, this dies here for an hour of work
rather than a fortnight, the way the sentiment idea did.

    .venv/bin/python scripts/scope_proxy_prices.py [n_docs]

Downloads are content-addressed, so a re-run makes zero requests.
"""
import json
import re
import sys

import numpy as np

from deal import fetch, universe
from deal import config

ACC = re.compile(r"(\d{10}-\d{2}-\d{6})")
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"[ \t\xa0]+")

# The section heading, in the several forms issuers actually use.
SECTION = re.compile(
    r"(MARKET\s+PRICE|PRICE\s+RANGE|MARKET\s+FOR\s+(?:THE\s+)?(?:REGISTRANT|COMPANY)"
    r"|COMPARATIVE\s+(?:PER\s+SHARE\s+)?MARKET|MARKET\s+AND\s+DIVIDEND)",
    re.I)

# "First Quarter   12.34   10.11"  /  "Q1 2023  $12.34  $10.11"
QUARTER_ROW = re.compile(
    r"(first|second|third|fourth)\s+quarter[^\n]{0,120}?"
    r"\$?\s*(\d{1,5}\.\d{2})[^\n]{0,40}?\$?\s*(\d{1,5}\.\d{2})", re.I)

# The consideration: "$34.50 in cash", "merger consideration of $34.50"
OFFER = re.compile(
    r"(?:\$\s*(\d{1,4}\.\d{2})\s+in\s+cash(?:\s*,?\s*without\s+interest)?"
    r"|merger\s+consideration\s+of\s+\$\s*(\d{1,4}\.\d{2})"
    r"|right\s+to\s+receive\s+\$\s*(\d{1,4}\.\d{2}))", re.I)


def strip_html(raw: bytes) -> str:
    t = raw.decode("latin-1", "ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = TAG.sub(" ", t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&")
         .replace("&#160;", " ").replace("&#8217;", "'"))
    return WS.sub(" ", t)


def primary_doc(cik: str, accession: str) -> bytes | None:
    """The main .htm, not the whole submission -- exhibits dominate the bytes."""
    a = accession.replace("-", "")
    idx = json.loads(fetch.sec_get(
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a}/index.json"))
    items = [i for i in idx["directory"]["item"]
             if i["name"].lower().endswith((".htm", ".html", ".txt"))]
    if not items:
        return None
    items.sort(key=lambda i: -int(i.get("size") or 0))
    return fetch.sec_get(
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{a}/"
        f"{items[0]['name']}")


def scope(rows, limit):
    stats = {"tried": 0, "fetched": 0, "has_section": 0, "quarters": 0,
             "has_offer": 0, "both": 0, "bytes": 0}
    prem, nq = [], []
    for r in rows[:limit]:
        m = ACC.search(r["filename"])
        if not m:
            continue
        stats["tried"] += 1
        try:
            raw = primary_doc(r["cik"], m.group(1))
        except Exception as e:
            print(f"  {r['name'][:34]:<34} FETCH {type(e).__name__}", flush=True)
            continue
        if not raw:
            continue
        stats["fetched"] += 1
        stats["bytes"] += len(raw)
        txt = strip_html(raw)

        sec = bool(SECTION.search(txt))
        stats["has_section"] += sec
        qs = QUARTER_ROW.findall(txt)
        # High/low must be ordered and plausible, or it is some other table.
        qs = [(float(h), float(lo)) for _, h, lo in qs
              if float(h) >= float(lo) > 0 and float(h) < 5000]
        off = OFFER.search(txt)
        price = next((float(g) for g in off.groups() if g), None) if off else None

        if len(qs) >= 4:
            stats["quarters"] += 1
            nq.append(len(qs))
        if price:
            stats["has_offer"] += 1
        if len(qs) >= 4 and price:
            stats["both"] += 1
            last_high = qs[-1][0]
            if 0 < last_high < 5000:
                prem.append(100 * (price / last_high - 1))
        print(f"  {r['name'][:34]:<34} {len(raw)/1e6:>5.1f}MB  "
              f"section={'Y' if sec else 'n'}  quarters={len(qs):>2}  "
              f"offer={'$%.2f' % price if price else '--':>8}", flush=True)
    return stats, prem, nq


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rows = []
    for y, q in universe.quarters(2023, 2024):
        p = fetch.cache_path("sec", config.IDX_URL.format(year=y, q=q))
        if p.exists():
            rows += [r for r in universe.parse_master_idx(p.read_bytes())
                     if r["form"] == "DEFM14A"]
    # Spread across the two years rather than taking one quarter's filings.
    rows = rows[::max(1, len(rows) // limit)]
    print(f"{len(rows)} DEFM14A sampled from 2023-2024, scoping {limit}\n")

    stats, prem, nq = scope(rows, limit)
    f = max(stats["fetched"], 1)
    print(f"\n{'='*66}")
    print(f"fetched              {stats['fetched']}/{stats['tried']}"
          f"   {stats['bytes']/1e6:.0f} MB "
          f"({stats['bytes']/f/1e6:.1f} MB/doc)")
    print(f"price section found  {stats['has_section']:>3}  "
          f"{100*stats['has_section']/f:>5.1f}%")
    print(f">=4 quarterly H/L    {stats['quarters']:>3}  "
          f"{100*stats['quarters']/f:>5.1f}%"
          f"   median {np.median(nq) if nq else 0:.0f} rows")
    print(f"offer price found    {stats['has_offer']:>3}  "
          f"{100*stats['has_offer']/f:>5.1f}%")
    print(f"BOTH (usable row)    {stats['both']:>3}  "
          f"{100*stats['both']/f:>5.1f}%")
    if prem:
        p = np.array(prem)
        print(f"\ncrude premium (offer vs last quarterly high), n={len(p)}")
        print(f"  median {np.median(p):+.1f}%   p25 {np.percentile(p,25):+.1f}%"
              f"   p75 {np.percentile(p,75):+.1f}%")
        print(f"  negative (offer below the quarter high): "
              f"{100*np.mean(p<0):.0f}%")
    print(f"\nprojected for all 2,456 proxies: "
          f"{2456*stats['bytes']/f/1e9:.1f} GB, "
          f"~{2456*2/8/60:.0f} min of requests")


if __name__ == "__main__":
    main()
