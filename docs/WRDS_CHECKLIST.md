# WRDS day-pass plan

Summer restriction lifts **2026-09-23**. This is what to do with a day pass
before then, and what to leave for Fall.

**The short answer to "can we get it in one day":** the *download* is easily a
day's work. The *linkage* is what would burn the day, because CRSP has no CIK
and this project is keyed entirely on CIK. Do the linkage and the small,
high-value tables on the day pass; leave the bulk daily file for Fall.

---

## The one thing to grab first

**`crsp.dsedelist` — the delisting file. ~30k rows. Seconds to download.**

It is the highest-value table in this entire plan and almost nobody reaches for
it first:

- `dlstcd` **200–299 = merger**. That is an **independent M&A label** for the
  whole project — built by CRSP from exchange records, with no dependence on
  DEFM14A filings, no contamination from stock-deal buyers filing proxies, and
  no 270-day "did it stop filing" inference. It would replace the single
  weakest link in the current pipeline.
- `dlret` is the **delisting return** — what a holder actually earned on the
  final day. That is the takeover premium, measured, for exactly the companies
  no free source retains.

One small table answers both the label question and the economic-value
question. Get this even if you get nothing else.

Caveats to handle: `dlret` carries documented missing-value codes (-55, -66,
-88, -99) that must not be read as returns; and `dlstcd` 300–399 (exchange),
400–499 (liquidation) and 500+ (dropped for cause) are **not** acquisitions.

---

## Verify these exist (5 minutes, do it first)

Institutional subscriptions vary. Confirm all four before planning around them:

| Table | What it is | Rough size |
|---|---|---|
| `crsp.dsedelist` | delisting codes + returns | ~30k rows |
| `crsp.ccmxpf_lnkhist` | CCM link history, GVKEY ↔ PERMNO | ~100k rows |
| `comp.company` | GVKEY → **CIK** | ~50k rows |
| `crsp.msf` / `crsp.dsf` | monthly / daily stock file | ~1M / ~22M rows |

CCM is often a **separate product** from CRSP and Compustat. If it is not in
the subscription, the CIK linkage has no clean path and everything below
changes — which is precisely why this check comes first.

---

## The linkage, which is the part that silently fails

CRSP is keyed on PERMNO. This project is keyed on CIK. The bridge:

```sql
SELECT c.cik, l.lpermno AS permno, l.linkdt, l.linkenddt
FROM comp.company c
JOIN crsp.ccmxpf_lnkhist l USING (gvkey)
WHERE l.linktype IN ('LC', 'LU')     -- valid research links only
  AND l.linkprim IN ('P', 'C')       -- primary security only
```

Then join prices on `permno` **and** `date BETWEEN linkdt AND
COALESCE(linkenddt, CURRENT_DATE)`.

Four ways this goes wrong, all silent:

1. **Dropping the date range.** A PERMNO is only that GVKEY's for a window.
   Ignoring it attaches the wrong company's prices across mergers and
   re-listings — and the errors land disproportionately on acquired companies,
   i.e. every positive label.
2. **Ignoring `linkprim`.** Multiple share classes produce duplicate rows and
   silently double-count.
3. **Assuming one CIK → one PERMNO.** It is one-to-many over time.
4. **`linkenddt` NULL means still active,** not "no link".

**Sanity check before trusting any of it:** the panel has 15,325 CIKs, of which
**7,520 (49%) have already stopped filing.** If the linked set is missing most
of those, the link is survivorship-broken and everything downstream inherits
it. Expect well under 15,325 matches — many EDGAR filers were never
exchange-listed — but the *acquired* ones must survive.

---

## Day-pass sequence (~2–3 hours, not a full day)

1. Verify the four tables above exist. **5 min.**
2. Pull `comp.company` and `crsp.ccmxpf_lnkhist` in full. Tiny. **10 min.**
3. Build and validate the CIK → PERMNO map offline. Check coverage against the
   1,664 verified targets specifically, not against the universe average.
   **30 min.**
4. Pull `crsp.dsedelist` in full. **5 min.**
5. Pull `crsp.msf` (monthly) for the matched PERMNOs, 2015-01 onward. ~1M rows,
   comfortably a single query. **30 min.**
6. Spot-check ten known deals end to end — CIK, PERMNO, delisting code,
   delisting return, and the premium already parsed from the merger proxy.
   Agreement across two independent sources is the real validation. **30 min.**

That leaves `crsp.dsf` (daily, ~22M rows, ~1 GB as parquet) for Fall. Daily
resolution is only needed for a proper event study; monthly returns already
support momentum, valuation and illiquidity features.

---

## What each piece unlocks

| Data | Unlocks |
|---|---|
| `dsedelist` codes | **An independent label.** Removes the DEFM14A proxy-date problem, the 270-day inference, and the buyer-files-a-proxy contamination in one move. |
| `dsedelist` returns | The premium actually earned, for delisted companies. Answers "is this tradeable" directly. |
| `msf` | Momentum, valuation, illiquidity — the blocked feature families. |
| `dsf` | Daily event study: cumulative abnormal returns around announcement, flagged vs missed. |

---

## Calibrate the expectation

Adding price features is worth testing, but this project's track record on new
feature families is: industry-relative **−1.6pp**, sentiment **null**, Palepu
**+0.08pp**, shelf/prospectus **+1.65pp**. Only form counts (**+5.43pp**) and
peer/activist have cleared the noise band.

And the strongest price-based predictor in the literature is **pre-bid runup**,
which is closest to the label and the most likely to be a leak rather than a
signal. Embargo it the way `stress_pairs.py` embargoes everything else.

**The bigger prize is measurement, not prediction.** The independent label and
the delisting returns settle questions the current data cannot answer at all.
A model improvement is speculative; those two are not.
