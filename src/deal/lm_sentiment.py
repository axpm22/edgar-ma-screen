"""Loughran-McDonald sentiment scoring for SEC filing text.

General-purpose sentiment lexicons are actively wrong on financial text --
"liability", "tax", "cost" and "depreciation" all score negative in Harvard IV
while being neutral accounting vocabulary. Loughran & McDonald built their word
lists from 10-K filings specifically to fix that.

The hypothesis being tested: a board that has begun exploring a sale hedges its
language before it is required to disclose anything. "Will" becomes "may".
Weak-modal and uncertainty word rates should therefore rise in the quarters
before an announcement.

Modal is coded 1=strong (will, must, definitely), 2=moderate (should, would),
3=weak (may, could, might, possibly). Weak modal is the one the hypothesis
predicts.
"""
import re
from functools import lru_cache
from pathlib import Path

DICT_PATH = Path("data/raw/lm/master.csv")

CATEGORIES = ["negative", "positive", "uncertainty", "litigious",
              "constraining", "modal_strong", "modal_moderate", "modal_weak"]

SCORE_COLS = [f"lm_{c}" for c in CATEGORIES] + ["lm_words", "lm_hedge"]

_TOKEN = re.compile(r"[A-Za-z']+")
# Filings are HTML; tags and entities would otherwise be counted as words.
_TAG = re.compile(r"<[^>]+>")
_ENT = re.compile(r"&[a-z]+;|&#\d+;")


@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, set[str]]:
    """Word -> category sets, read once and cached."""
    import pandas as pd
    d = pd.read_csv(DICT_PATH)
    w = d["Word"].astype(str).str.lower()
    out = {
        "negative": set(w[d["Negative"] > 0]),
        "positive": set(w[d["Positive"] > 0]),
        "uncertainty": set(w[d["Uncertainty"] > 0]),
        "litigious": set(w[d["Litigious"] > 0]),
        "constraining": set(w[d["Constraining"] > 0]),
        "modal_strong": set(w[d["Modal"] == 1]),
        "modal_moderate": set(w[d["Modal"] == 2]),
        "modal_weak": set(w[d["Modal"] == 3]),
    }
    return out


def strip_html(raw: str) -> str:
    return _ENT.sub(" ", _TAG.sub(" ", raw))


def score(text: str) -> dict[str, float]:
    """Category rates per 1,000 words, plus a combined hedging measure."""
    lex = load_lexicon()
    tokens = [t.lower() for t in _TOKEN.findall(strip_html(text))]
    n = len(tokens)
    if n < 50:                      # too short for a rate to mean anything
        return {c: 0.0 for c in SCORE_COLS}

    out = {}
    counts = {}
    for cat, words in lex.items():
        counts[cat] = sum(1 for t in tokens if t in words)
        out[f"lm_{cat}"] = 1000.0 * counts[cat] / n
    out["lm_words"] = float(n)
    # Hedging: weak modals plus uncertainty, net of strong modals. A board
    # that has stopped committing says "may" and stops saying "will".
    out["lm_hedge"] = (out["lm_modal_weak"] + out["lm_uncertainty"]
                       - out["lm_modal_strong"])
    return out
