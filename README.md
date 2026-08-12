# edgar-ma-screen

Predicting corporate acquisitions from free SEC filings.

**Of the 25 companies this screen flags each week, 10.73% are acquired within
twelve months — against a 1.67% base rate. That is 6.42× chance**, on operating
companies with verified-target labels, averaged over two clean held-out years.

Every input is free and public. No CRSP, no Compustat, no SDC Platinum. Clone
the repo and you can rebuild every number in it.

`PAPER.md` is the authoritative write-up. Where this README and the paper ever
disagree, the paper is right.

---

## Results

Held-out testing, mean of 2023 and 2024, top 25 companies per week, verified-target
labels:

| Universe | Precision | Lift |
|---|---|---|
| **Operating companies** | **10.73%** | **6.42×** |
| Including de-SPACs | 14.95% | 7.06× |

*Base rate 1.67% on operating companies.*

The two rows answer different questions. A blank-cheque vehicle merging is its
stated purpose rather than a prediction, so **the operating-company row is the
defensible one** and the number to quote.

The two clean years are 9.73% (2023) and 11.74% (2024) — a spread wide enough
that neither should be quoted alone. 2025 is excluded: a deal announced after
about October 2025 has not had 270 days to show whether the filer stopped
filing, so genuine deals in that window get labelled zero and the model is
punished for correct predictions.

## The label was wrong, and fixing it changed every number

A DEFM14A merger proxy is filed by whoever's shareholders vote — which in a
stock deal includes the **buyer**. An earlier draft treated every proxy filer as
a target and reported 13.81% precision at 5.65× lift. Classifying each filer on
whether it was still filing periodic reports 270 days later found that **23.8%
were acquirers or terminated deals**, not targets. Teledyne after FLIR. Newmont
after Newcrest. Dow after DowDuPont.

Precision fell, but so did the base rate, because the remaining positives are
fewer and purer. **Lift rose, 5.94× to 6.42×** — stripping a quarter of the
positives improved discrimination, which is what removing noise looks like.

If you find the old numbers quoted anywhere in this repo's history, they are
superseded. `PAPER.md` §4 is the correction.

## Is it real?

**Caveat first:** the robustness suite below was measured before the label fix
and has not been re-derived. Given the direction of the fix — lift up, base
rate down — the conclusions should hold, but the levels should not be quoted.
It is listed so the gap is visible rather than hidden.

| Test | Result |
|---|---|
| Permutation — refit on labels shuffled within week | every null draw beaten; null maxes at 5.89% against 21.40% real |
| Embargo — blank the 8 and 16 weeks before each deal | 20.02% → 14.31% → 14.46%: the first step costs ~28% of the edge, then it is flat |
| Clean three-way split — test period never used for early stopping | holds |
| Company-level rather than row-level | 23.78% vs 23.02% row-level, so repetition is not inflating the headline |
| Seed stability, 5 seeds | 22.34% ± 1.19 |
| Data audit | no duplicate rows, no nulls or infinities, no label at or after its own announcement |

The embargo row is the one that matters, and it does not say what an earlier
version of this README said. Performance is **not** flat from zero to 8 weeks —
a quarter of the edge lives in the final two months, which is exactly where
genuine deal preparation shows up. What rules out leakage is the *second* step:
flat from 8 to 16 weeks. A model reading post-announcement residue would keep
collapsing as the window moved back. It does not.

## What it does

Builds a **4,123,449**-row company-week panel covering **14,680** companies and
**1,664** verified acquisition targets, 2016 to 2026, then ranks every listed
company each week. 302,529 periodic filings define the universe; 2,949,427
Form 4 insider transactions, 2,041,665 XBRL fundamental facts and 1,167,814 form
events supply the features.

Signals come from filing behaviour rather than accounting: 13D and 13G stakes,
activist identity, 8-K item codes, proxy activity, insider trading stopping, a
sector peer being acquired, and specific disclosure language. Removing every
accounting variable from the forty-year takeover literature costs **0.08
percentage points**. Removing one feature — the count of forms filed in the last
26 weeks — costs **5.43 points**.

Survivorship is handled by deriving the universe from filing activity rather
than index membership, so delisted and acquired companies remain in the panel
for exactly the weeks they existed.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -e .
export EDGAR_UA="your-project your.email@example.com"   # the SEC requires this

make all      # ~4 hours, ~9 GB downloaded, mostly SEC rate limiting
make paper    # regenerate figures
make test
```

`make all` is resumable. Every download is content-addressed to disk, so an
interrupted run picks up where it stopped and a re-run makes zero requests.

The SEC rate-limits to 10 requests/second and requires a User-Agent carrying
real contact details; `EDGAR_UA` supplies it and requests will be refused
without one.

## What it is not

**A trading signal.** It is wrong about 89% of the time on any individual
company, and without price data the economic value is untested. Palepu (1986)
showed takeover-prediction models predict targets without earning abnormal
returns, and nothing here challenges that.

Think of it as a way to narrow seven thousand companies to twenty-five worth
reading about.

## Known limitations

- Labels come from DEFM14A merger proxies, which omits tender offers — roughly
  27% of deals, and not at random, since hostile and cash bids skew that way.
  The defensible claim is "predicts negotiated mergers well and tender offers
  weakly", not "predicts M&A".
- Only two clean test years survive the label fix.
- Rumour dates are unmeasured, so lead time is measured against the proxy
  filing rather than the first public report.
- No returns, because no free source retains price history for delisted
  companies — which is precisely where the positive observations are.

## Further reading

`PAPER.md` carries the full argument: three negative results (sentiment
analysis, industry-relative ratios, and the classic literature variables all
fail to help), a second model for predicting *acquirers* rather than targets,
and seven errors caught during development, each of which would have produced a
confident wrong answer.

## Licence

MIT.
