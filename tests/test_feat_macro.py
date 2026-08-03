import datetime as dt

from deal import feat_macro


def test_president_party_by_inauguration():
    assert feat_macro.president_republican(dt.date(2016, 6, 1)) == 0
    assert feat_macro.president_republican(dt.date(2017, 1, 19)) == 0
    assert feat_macro.president_republican(dt.date(2017, 1, 20)) == 1
    assert feat_macro.president_republican(dt.date(2020, 12, 1)) == 1
    assert feat_macro.president_republican(dt.date(2021, 1, 20)) == 0
    assert feat_macro.president_republican(dt.date(2025, 6, 1)) == 1


def test_recession_flag_is_not_a_feature():
    """NBER dates recessions retrospectively -- USREC at week W is knowledge
    nobody had at W. If it ever appears in MACRO_COLS that is lookahead."""
    assert not any("rec" in c.lower() for c in feat_macro.MACRO_COLS)
    assert "USREC" not in feat_macro.DAILY.values()
    assert "USREC" not in feat_macro.MONTHLY.values()


def test_monthly_series_carry_a_publication_lag():
    """UNRATE for month M is published in the first week of M+1."""
    assert feat_macro.MONTHLY_LAG_DAYS >= 30
