"""Guards on the paper and its figures.

The failure mode these exist for is not a crash: it is a retracted number
quietly reappearing. 27.52% survived its own retraction in FINAL_RESULTS.md
and five committed figures, so the check is mechanical.
"""
import json
import re
from pathlib import Path

PAPER = Path("PAPER.md")
CHARTS = Path("scripts/make_charts.py")

# From the single split whose test period was also used for early stopping.
RETRACTED = ("27.52", "30.94", "29.66", "24.10", "43.00")
# Prose wraps, so the disclaimer often sits on a neighbouring line.
_WINDOW = 260
_DISCLAIMERS = ("retract", "understated", "censor", "stale")


def _quoted_as_live(text: str, value: str) -> str | None:
    """Return the first occurrence of `value` with no disclaimer nearby."""
    for m in re.finditer(re.escape(value), text):
        near = text[max(0, m.start() - _WINDOW):m.end() + _WINDOW].lower()
        if not any(d in near for d in _DISCLAIMERS):
            return text[max(0, m.start() - 90):m.end() + 90].replace("\n", " ")
    return None


def test_no_chart_hardcodes_a_retracted_number():
    src = CHARTS.read_text()
    for stale in RETRACTED:
        bad = _quoted_as_live(src, stale)
        assert bad is None, f"retracted value {stale} back in charts: {bad}"


def test_paper_leads_with_the_defensible_number():
    """Operating-company, verified-target number comes before all-companies."""
    t = PAPER.read_text()
    assert t.index("10.73") < t.index("14.95"), \
        "all-companies figure must not precede the operating-only headline"


def test_headline_is_the_clean_label_result():
    """13.81% was measured with 581 acquirers in the positive class."""
    for line in PAPER.read_text().splitlines():
        if "13.81" in line:
            assert any(w in line.lower() for w in
                       ("earlier draft", "proxy filers", "contaminated")), \
                f"13.81 quoted as live: {line.strip()}"


def test_paper_only_cites_retracted_numbers_as_retracted():
    t = PAPER.read_text()
    for stale in RETRACTED:
        bad = _quoted_as_live(t, stale)
        assert bad is None, f"{stale} quoted as live: ...{bad}..."


def test_paper_states_the_negative_results():
    t = PAPER.read_text().lower()
    for claim in ("sentiment", "industry-relative", "palepu"):
        assert claim in t, f"missing negative result: {claim}"


def test_paper_references_only_existing_figures():
    for f in re.findall(r"docs/figures/([a-z_]+\.png)", PAPER.read_text()):
        assert Path(f"docs/figures/{f}").exists(), f"missing figure {f}"


def test_headline_matches_the_measured_run():
    """2025 is excluded: its deals cannot be classified target-vs-survivor."""
    rows = [r for r in json.loads(Path("data/clean_nospac.json").read_text())
            if r[0] != 2025]
    mean = sum(r[1] for r in rows) / len(rows)
    assert 10.0 < mean < 11.5, f"operating-only mean drifted to {mean:.2f}"


def test_measured_curve_reproduces_the_headline():
    curve = json.loads(Path("data/curve_clean.json").read_text())
    at25 = curve["precision"][curve["ns"].index(25)]
    assert abs(at25 - 10.73) < 0.5, f"curve N=25 is {at25}, expected ~10.73"
    assert curve["precision"] == sorted(curve["precision"], reverse=True), \
        "operating-company curve should decline monotonically with list size"


def test_labels_are_classified_not_assumed():
    """A DEFM14A filer is only a target if it stopped filing afterwards."""
    src = Path("src/deal/clean_labels.py").read_text()
    assert "survivor" in src and "SURVIVE_DAYS" in src
