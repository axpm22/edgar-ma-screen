# edgar-acquisition-signals

Predicting corporate acquisitions from free SEC filings.

**Of the 25 companies this screen flags each week, 12.3% are acquired within
twelve months — against a 1.7% base rate. That is 6.9× better than chance**,
averaged over eleven held-out years.

Every input is free and public. No CRSP, no Compustat, no SDC Platinum. Clone
the repo and you can rebuild every number in it.

---

## Results

Held-out testing, mean of eleven test years (2015–2025), top 25 companies per
week. Two models, two questions.

| Model | Universe | Precision | Base rate | Lift |
|---|---|---|---|---|
| **Target** — will it be acquired? | operating companies | **12.28%** | 1.66% | **6.91×** |
| Target | including de-SPACs | 13.29% | 1.79% | 6.87× |
| **Buyer** — will it acquire? | operating companies | **31.72%** | 2.87% | **12.21×** |
| Buyer | including de-SPACs | 32.86% | 3.16% | 10.69× |

**Buying is roughly twice as predictable as being bought.** Target labels are
verified — a merger proxy is filed by whoever's shareholders vote, so 581 of
2,445 proxy filers turned out to be acquirers and survivors; only companies
that actually stopped filing count as targets.

**Quote lift, not precision, when comparing across years.** Deal rates move a
lot: the buyer base rate fell from 3.22% (2015–19) to 1.87% (2022–25), which
drags precision down 14pp while lift is unchanged. Excluding SPACs likewise
moves precision by about a point while leaving lift flat (+0.04× target), so
it changes the population, not the skill.

Performance is regime-dependent — target lift spans 4.8× to 9.3× across the
eleven years — so no single year should be quoted on its own.

## Is it real?

| Test | Result |
|---|---|
| Permutation, target — refit on labels shuffled within week | null tops out well below real |
| Permutation, buyer — never run before | real 23.9% against a null max of 5.5% |
| Embargo — blank the 8 and 16 weeks before each deal | flat, so it is not reading post-announcement filings |
| Clean three-way split — test period never used for early stopping | holds |
| SPACs removed from training and test | holds; lift unchanged |
| Hazard model on verified-target labels, SEs clustered by company | 9 of 9 signals keep sign and significance |
| Is the size effect really spotting *acquirers*? | no — size contributes equally on verified-target labels |
| Tender offers as an independent label | **uninformative** — rests on 1–5 companies a year |
| Data audit | no duplicate rows, no nulls or infinities, no label at or after its own announcement |

Every precision figure now reports `distinct_hits`, the number of separate
companies behind it. That check exists because the tender-offer validation
produced a 3.37× lift with a tight confidence interval from **one company held
for 23 consecutive weeks**.

## What it does

Turns EDGAR filing records into a **5,967,094**-row company-week panel covering
**19,021** companies and **3,188** acquisitions from 2012 to the present, then
ranks every listed company each week.

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

**A trading signal.** It is wrong about 88% of the time on any individual
company, and without price data the economic value is untested. Palepu (1986)
showed takeover-prediction models predict targets without earning abnormal
returns, and nothing here challenges that.

Think of it as a way to narrow seven thousand companies to twenty-five worth
reading about.

**A way to name the buyer.** Given a target, the model puts the true acquirer
in a shortlist of 100 about a third of the time (median rank ~200 of ~7,100),
which is real signal — but its top-1 accuracy against the full universe is
**0.0%**. Chained end to end with nothing given, flagging both sides of the
same deal succeeds on **3 of 137** cases. Most of what makes pairing work is
structural rather than predictive: the true acquirer shares the target's
2-digit SIC 65% of the time against ~7% for random companies.

## Known limitations

- Labels come from DEFM14A merger proxies, which omits tender offers — roughly
  27% of deals, and not at random, since hostile and cash bids skew that way.
- Rumour dates are unmeasured. Measured on the pair table, the merger proxy
  lands a median **84 days after** the deal is already public, so the label is
  systematically late relative to the tradeable event.
- No returns, because no free source retains price history for delisted
  companies — which is precisely where the positive observations are. Merger
  proxies were scoped as a substitute and do not carry usable price history:
  quarterly high/low tables parse in only 3.3% of recent filings, since the
  SEC's 2018 disclosure simplification dropped the requirement.
- Recall is low. The screen flags only about 6.5% of deals in the year before
  announcement; when it does fire, the median lead is 70 days.
- Macro regime variables (credit spread, VIX, yield curve, president's party)
  do not help. They are constant within a week, and the screen ranks within
  weeks, so their main effect on precision is zero by construction; the best
  between-year correlation is r = +0.35 on eleven observations.

## Further reading

`PAPER.md` carries the full argument, including three negative results
(sentiment analysis, industry-relative ratios, and the classic literature
variables all fail to help) and the four errors caught during development, each
of which would have produced a confident wrong answer.

## Licence

MIT.
