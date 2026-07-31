# edgar-acquisition-signals

Predicting corporate acquisitions from free SEC filings.

**Of the 25 companies this screen flags each week, 13.8% are acquired within
twelve months — against a 2.4% base rate. That is 5.65× better than chance**,
averaged over three held-out years.

Every input is free and public. No CRSP, no Compustat, no SDC Platinum. Clone
the repo and you can rebuild every number in it.

---

## Results

Held-out testing, mean of 2023 / 2024 / 2025, top 25 companies per week:

| Universe | Precision | Base rate | Lift |
|---|---|---|---|
| **Operating companies** | **13.81%** | 2.44% | **5.65×** |
| Including de-SPACs | 27.58% | 2.93% | 9.40× |

The two rows answer different questions. A blank-cheque vehicle merging is its
stated purpose rather than a prediction, so **the operating-company row is the
defensible one** and the number to quote. Excluding all 1,707 SPAC-like
companies costs about a third of the apparent skill and the result survives.

Performance is regime-dependent: **11.40% to 15.15%** across the three test
years. That spread is five to ten times larger than seed-to-seed noise, so no
single year should be quoted on its own.

## Is it real?

| Test | Result |
|---|---|
| Permutation — refit on labels shuffled within week | null tops out at 5.89% against 21.40% real |
| Embargo — blank the 8 and 16 weeks before each deal | performance flat, so it is not reading post-announcement filings |
| Clean three-way split — test period never used for early stopping | holds |
| SPACs removed from training and test | holds at 13.81% |
| Company-level rather than row-level | 37.77% — repetition biases the headline *down*, not up |
| Data audit | no duplicate rows, no nulls or infinities, no label at or after its own announcement |

## What it does

Turns **11,591,580** EDGAR filing records into a **4,123,449**-row company-week
panel covering **15,325** companies and **2,456** acquisitions from 2016 to the
present, then ranks every listed company each week.

Signals come from filing behaviour rather than accounting: 13D and 13G stakes,
activist identity, 8-K item codes, proxy activity, insider trading stopping, a
sector peer being acquired, and specific disclosure language.

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

**A trading signal.** It is wrong about 86% of the time on any individual
company, and without price data the economic value is untested. Palepu (1986)
showed takeover-prediction models predict targets without earning abnormal
returns, and nothing here challenges that.

Think of it as a way to narrow seven thousand companies to twenty-five worth
reading about.

## Known limitations

- Labels come from DEFM14A merger proxies, which omits tender offers — roughly
  27% of deals, and not at random, since hostile and cash bids skew that way.
- Rumour dates are unmeasured, so lead time is measured against the proxy
  filing rather than the first public report.
- No returns, because no free source retains price history for delisted
  companies — which is precisely where the positive observations are.

## Further reading

`PAPER.md` carries the full argument, including three negative results
(sentiment analysis, industry-relative ratios, and the classic literature
variables all fail to help) and the four errors caught during development, each
of which would have produced a confident wrong answer.

## Licence

MIT.
