import datetime as dt

import numpy as np
import polars as pl

from deal import screen


def _panel(rows):
    return pl.DataFrame(
        {"cik": [r[0] for r in rows],
         "week": [dt.date(2024, 1, 1) + dt.timedelta(weeks=r[1]) for r in rows],
         "y": [r[2] for r in rows]})


def test_distinct_hits_exposes_a_result_resting_on_one_company():
    """One company held across many weeks can carry a whole precision number.

    This is the tender-offer failure: 23 hit-rows, lift 3.37x, tight CI, and
    every hit was the same firm. Precision alone cannot show that.
    """
    rows = [("SAME", w, 1) for w in range(10)]          # one firm, ten weeks
    rows += [("OTHER%d" % w, w, 0) for w in range(10)]
    df = _panel(rows)
    # Score the positives top so they are all selected.
    p = np.array([1.0] * 10 + [0.0] * 10)
    r = screen.weekly_precision(df, p, n_per_week=1)
    assert r["precision"] == 1.0
    assert r["n_selected"] == 10
    assert r["distinct_hits"] == 1


def test_distinct_hits_counts_separate_companies():
    rows = [("A", 0, 1), ("B", 1, 1), ("C", 2, 1)]
    rows += [("X%d" % w, w, 0) for w in range(3)]
    df = _panel(rows)
    p = np.array([1.0] * 3 + [0.0] * 3)
    r = screen.weekly_precision(df, p, n_per_week=1)
    assert r["distinct_hits"] == 3


def test_distinct_hits_is_zero_when_nothing_hits():
    df = _panel([("A", 0, 0), ("B", 0, 0)])
    r = screen.weekly_precision(df, np.array([1.0, 0.0]), n_per_week=1)
    assert r["distinct_hits"] == 0
    assert r["precision"] == 0.0
