import datetime as dt

from deal import load_pairs


def test_accession_of_extracts_from_filename():
    fn = "edgar/data/1659587/0001659587-17-000009.txt"
    assert load_pairs.accession_of(fn) == "0001659587-17-000009"
    assert load_pairs.accession_of("edgar/data/1/no-accession.txt") is None


def test_group_filings_keeps_only_wanted_forms_and_earliest_date():
    rows = [
        {"cik": "1", "form": "425", "file_date": dt.date(2020, 3, 5),
         "filename": "a/0000000001-20-000001.txt"},
        {"cik": "2", "form": "425", "file_date": dt.date(2020, 3, 4),
         "filename": "b/0000000001-20-000001.txt"},
        {"cik": "3", "form": "10-K", "file_date": dt.date(2020, 3, 4),
         "filename": "c/0000000002-20-000002.txt"},
    ]
    g = load_pairs.group_filings(rows, {"425"})
    assert set(g) == {"0000000001-20-000001"}
    assert g["0000000001-20-000001"]["ciks"] == {"1", "2"}
    # Earliest date across both index rows -- the announcement, not the last amendment.
    assert g["0000000001-20-000001"]["date"] == dt.date(2020, 3, 4)


def test_orient_picks_the_party_that_stops_filing():
    d = dt.date(2020, 1, 1)
    delisted = {"T": dt.date(2020, 6, 1), "A": dt.date(2026, 1, 1)}
    assert load_pairs.orient({"T", "A"}, d, delisted) == ("T", "A")


def test_orient_returns_none_when_both_or_neither_stop():
    d = dt.date(2020, 1, 1)
    both = {"X": dt.date(2020, 6, 1), "Y": dt.date(2020, 7, 1)}
    assert load_pairs.orient({"X", "Y"}, d, both) is None
    neither = {"X": dt.date(2026, 1, 1), "Y": dt.date(2026, 1, 1)}
    assert load_pairs.orient({"X", "Y"}, d, neither) is None


def test_orient_needs_exactly_two_parties():
    d = dt.date(2020, 1, 1)
    assert load_pairs.orient({"X"}, d, {"X": None}) is None
    assert load_pairs.orient({"X", "Y", "Z"}, d, {}) is None


def test_collapse_merges_repeat_filings_within_a_year():
    rows = [
        {"target_cik": "T", "acquirer_cik": "A", "date": dt.date(2020, 1, 1),
         "form": "425"},
        {"target_cik": "T", "acquirer_cik": "A", "date": dt.date(2020, 4, 1),
         "form": "425"},
        # More than 365 days later: a separate episode.
        {"target_cik": "T", "acquirer_cik": "A", "date": dt.date(2022, 6, 1),
         "form": "425"},
    ]
    out = load_pairs.collapse(rows)
    assert len(out) == 2
    assert out[0]["first_ts"] == dt.date(2020, 1, 1)
    assert out[0]["last_ts"] == dt.date(2020, 4, 1)
    assert out[0]["n_filings"] == 2
    assert out[1]["n_filings"] == 1
