import datetime as dt

import pytest

from deal import load_forms, warehouse


def test_amendments_map_to_the_same_family_as_the_parent():
    assert load_forms.classify("SC 13D") == "sc13d"
    assert load_forms.classify("SC 13D/A") == "sc13d"
    assert load_forms.classify("SC 13G") == "sc13g"


def test_late_filing_notices_are_tracked():
    assert load_forms.classify("NT 10-K") == "late"
    assert load_forms.classify("NT 10-Q") == "late"


def test_delisting_forms_are_forbidden_and_never_classified():
    # 25-NSE and 15-12B are filed AFTER a deal closes. Using them as features
    # produces a near-perfect model that has simply read the answer.
    assert "25-NSE" in load_forms.FORBIDDEN_FORMS
    assert "15-12B" in load_forms.FORBIDDEN_FORMS
    assert load_forms.classify("25-NSE") is None
    assert load_forms.classify("15-12B") is None


def test_no_forbidden_form_appears_in_the_tracked_map():
    assert not (set(load_forms.TRACKED_FORMS) & load_forms.FORBIDDEN_FORMS)


def test_unrelated_forms_are_ignored():
    assert load_forms.classify("10-K") is None
    assert load_forms.classify("424B5") is None


@pytest.fixture
def con(tmp_path):
    c = warehouse.connect(str(tmp_path / "t.duckdb"))
    load_forms.init_schema(c)
    return c


def test_insert_is_idempotent(con):
    rows = [{"cik": "1", "family": "sc13d", "public_ts": dt.date(2024, 3, 1)}]
    load_forms.insert(con, rows)
    load_forms.insert(con, rows)
    assert con.execute("SELECT count(*) FROM form_events").fetchone()[0] == 1
