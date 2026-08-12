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
    assert t.index("12.28") < t.index("13.29"), \
        "all-companies figure must not precede the operating-only headline"


def test_headline_is_the_clean_label_result():
    """13.81% was measured with 749 acquirers in the positive class."""
    for line in PAPER.read_text().splitlines():
        if "13.81" in line:
            assert any(w in line.lower() for w in
                       ("earlier draft", "early draft", "proxy filers",
                        "contaminated")), \
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
    """The paper's headline must match what the accuracy stage actually wrote.

    Pinned to the eleven-year operating-company mean. The previous version of
    this guard was pinned to a three-year mean of ~10.73, and kept passing
    against a stale data file after the panel was rebuilt -- a guard that
    protects a superseded number is worse than none.
    """
    p = Path("data/feature_report.json")
    if not p.exists():
        return                       # results are gitignored; skip on a clone
    rows = [r for r in json.loads(p.read_text())
            if r.get("stage") == "accuracy" and r.get("model") == "target"
            and r.get("universe") == "nospac"]
    if not rows:
        return
    mean = sum(r["prec"] for r in rows) / len(rows)
    assert 11.0 < mean < 13.5, f"operating-only mean drifted to {mean:.2f}"
    assert len(rows) >= 10, f"expected ~11 test years, got {len(rows)}"


def test_measured_curve_reproduces_the_headline():
    p = Path("data/curve_final.json")
    if not p.exists():
        return
    curve = json.loads(p.read_text())["target"]["curve"]
    at25 = next(c["prec"] for c in curve if c["n"] == 25)
    assert abs(at25 - 12.65) < 0.8, f"curve N=25 is {at25}, expected ~12.65"
    prec = [c["prec"] for c in curve]
    assert prec == sorted(prec, reverse=True), \
        "operating-company curve should decline monotonically with list size"


def test_labels_are_classified_not_assumed():
    """A DEFM14A filer is only a target if it stopped filing afterwards."""
    src = Path("src/deal/clean_labels.py").read_text()
    assert "survivor" in src and "SURVIVE_DAYS" in src
