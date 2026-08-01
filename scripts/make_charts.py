"""Charts for PAPER.md.

Every number here is measured, and the docstring on each function names the
file it came from. Nothing is illustrative.

Five figures in the previous version carried numbers from the contaminated
single split (27.52% and friends), which was retracted once the test period
turned out to have been used for early stopping. Those are gone. The headline
throughout is the operating-company result: 13.81% at top-25/week, 5.65x.

Palette is the validated light-mode categorical set (slots 1-3), which passes
lightness, chroma, colourblind separation and normal-vision separation. Slot 3
(aqua) sits below 3:1 contrast on this surface, so the relief rule applies and
every bar carries a visible direct label.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRID = "#e2e1dd"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRID, "figure.dpi": 200,
})


def _style(ax, ylabel=""):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _label(ax, bars, fmt="{:.1f}%", dy=0.4):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=9, color=INK)


def _cv(stage, subset="A_all"):
    rows = json.loads((DATA / "select_cv.json").read_text())
    return [r for r in rows
            if r.get("stage") == stage and r.get("subset") == subset]


def fig_funnel():
    """Source: measured counts from deal.duckdb, forms.duckdb and
    logs/index.log. Not the README's 11.6M, which counts every master-index
    row including forms the pipeline never reads. The label count is verified
    targets, not proxy filers -- 581 of those turned out to be acquirers."""
    labels = ["Insider\ntransactions", "XBRL\nfacts", "Form\nevents",
              "Periodic filings\n(universe)", "Company-weeks\n(panel)",
              "Verified targets\n(labels)"]
    vals = [2_949_427, 2_041_665, 1_167_814, 302_529, 4_123_449, 1_664]
    cols = [BLUE, BLUE, BLUE, BLUE, AQUA, ORANGE]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.4, f"{v:,}",
                ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(1e3, 2e8)
    _style(ax, "records (log scale)")
    ax.tick_params(axis="x", labelsize=8.2)
    ax.set_title("Four free SEC sources into one panel")
    fig.tight_layout(); fig.savefig(OUT / "funnel.png"); plt.close(fig)


def fig_hit_rate():
    """Source: data/curve_clean.json -- VERIFIED-TARGET labels, operating
    companies, 2023-24. 2025 is excluded: deals after ~Oct 2025 cannot yet be
    classified target-vs-survivor, so genuine deals get labelled 0."""
    curve = json.loads((DATA / "curve_clean.json").read_text())
    ns = [f"Top {n}" for n in curve["ns"]]
    hit, base = curve["precision"], curve["base"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(ns, hit, color=BLUE, width=0.6)
    _label(ax, bars)
    ax.axhline(base, color=ORANGE, linewidth=2, linestyle="--")
    ax.text(len(ns) - 0.55, base + 0.45,
            f"picking at random: {base:.2f}%", ha="right",
            fontsize=9, color=ORANGE,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.set_ylim(0, max(hit) * 1.32)
    _style(ax, "share acquired within 12 months")
    ax.set_xlabel("companies flagged per week", fontsize=9)
    ax.set_title("Precision falls monotonically with list size")
    fig.tight_layout(); fig.savefig(OUT / "hit_rate.png"); plt.close(fig)


def fig_labels():
    """Source: select_cv.json (contaminated) vs clean_all/clean_nospac.json.
    Both restricted to 2023-24 so the comparison is like-for-like.

    The whole point of the chart: precision falls in both universes, but lift
    RISES for operating companies. Fewer, purer positives."""
    old_all = sum(r["lift"] for r in _cv("sets") if r["year"] != 2025) / 2
    old_op = sum(r["lift"] for r in _cv("nospac") if r["year"] != 2025) / 2
    ca = json.loads((DATA / "clean_all.json").read_text())
    cn = json.loads((DATA / "clean_nospac.json").read_text())
    new_all = sum(r[3] for r in ca if r[0] != 2025) / 2
    new_op = sum(r[3] for r in cn if r[0] != 2025) / 2

    fig, ax = plt.subplots(figsize=(7, 3.7))
    x = [0, 1]
    b1 = ax.bar([i - 0.2 for i in x], [old_all, old_op], width=0.38,
                color=ORANGE, label="proxy filers (contaminated)")
    b2 = ax.bar([i + 0.2 for i in x], [new_all, new_op], width=0.38,
                color=AQUA, label="verified targets")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.16,
                    f"{b.get_height():.2f}x", ha="center", fontsize=9,
                    color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["Including de-SPACs", "Operating companies only"])
    ax.set_ylim(0, 11.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax, "lift over base rate")
    ax.set_title("Removing 581 acquirers made the finding stronger",
                 loc="left")
    fig.tight_layout(); fig.savefig(OUT / "labels.png"); plt.close(fig)


def fig_acquirer():
    """Source: clean_nospac.json vs acq_1.json / acq_2.json. All SPAC-free.
    Acquirer label = files an S-4 within 12 months."""
    cn = json.loads((DATA / "clean_nospac.json").read_text())
    tgt = sum(r[3] for r in cn if r[0] != 2025) / 2
    a1 = json.loads((DATA / "acq_1.json").read_text())
    a2 = json.loads((DATA / "acq_2.json").read_text())
    m1 = sum(r[2] for r in a1) / len(a1)
    m2 = sum(r[2] for r in a2) / len(a2)
    names = ["Will be\nacquired", "Will buy\n(any S-4)",
             "Will buy\n(serial acquirer)"]
    vals = [tgt, m1, m2]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    bars = ax.bar(names, vals, color=[AQUA, BLUE, BLUE], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}x",
                ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 21)
    _style(ax, "lift over base rate")
    ax.set_title("Buying is far more predictable than being bought")
    fig.tight_layout(); fig.savefig(OUT / "acquirer.png"); plt.close(fig)


def fig_cv_years():
    """Source: clean_all.json / clean_nospac.json -- verified-target labels.
    2025 is shown but hatched: it is a censoring artifact, not a bad year."""
    ca = {r[0]: r[1] for r in json.loads((DATA / "clean_all.json").read_text())}
    cn = {r[0]: r[1]
          for r in json.loads((DATA / "clean_nospac.json").read_text())}
    yrs = sorted(cn)
    x = range(len(yrs))
    fig, ax = plt.subplots(figsize=(7, 3.7))
    b1 = ax.bar([i - 0.2 for i in x], [ca[y] for y in yrs], width=0.38,
                color=BLUE, label="including de-SPACs")
    b2 = ax.bar([i + 0.2 for i in x], [cn[y] for y in yrs], width=0.38,
                color=AQUA, label="operating companies only")
    for bars in (b1, b2):          # mark the unusable year
        bars[yrs.index(2025)].set_hatch("///")
        bars[yrs.index(2025)].set_alpha(0.45)
    _label(ax, b1, dy=0.35); _label(ax, b2, dy=0.35)
    ax.set_xticks(list(x)); ax.set_xticklabels([str(y) for y in yrs])
    ax.set_ylim(0, 22)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax, "precision at top-25/week")
    ax.set_xlabel("held-out test year", fontsize=9)
    ax.text(2, 12.4, "outcomes not\nyet observable", ha="center",
            fontsize=8.5, color=INK2, style="italic")
    ax.set_title("2025 is unusable: its deals cannot be classified yet",
                 loc="left")
    fig.tight_layout(); fig.savefig(OUT / "cv_years.png"); plt.close(fig)


def fig_universes():
    """Source: clean_all.json / clean_nospac.json, 2023-24 mean lift."""
    ca = json.loads((DATA / "clean_all.json").read_text())
    cn = json.loads((DATA / "clean_nospac.json").read_text())
    a = sum(r[3] for r in ca if r[0] != 2025) / 2
    n = sum(r[3] for r in cn if r[0] != 2025) / 2
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bars = ax.bar(["Including\nde-SPACs", "Operating\ncompanies only"],
                  [a, n], color=[BLUE, AQUA], width=0.45)
    for b, v in zip(bars, [a, n]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}x",
                ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 9)
    _style(ax, "lift over base rate")
    ax.set_title("On clean labels the two universes nearly converge")
    fig.tight_layout(); fig.savefig(OUT / "universes.png"); plt.close(fig)


def fig_size_hump():
    """Source: data/curve_clean.json -- deal rate by public-float decile on
    VERIFIED-TARGET labels at the 52-week horizon.

    The previous version read `y` straight from features.parquet, which stores
    a 26-WEEK label (1.413% positive; the 52-week rate is 2.769%). It was
    captioned 'within 12 months' and was not. Every model script calls
    relabel(raw, 52) first, so only the figure was wrong."""
    rate = json.loads((DATA / "curve_clean.json").read_text())["size_deciles"]
    labels = ["smallest", "2", "3", "4", "5", "6", "7", "8", "9", "largest"]
    cols = [AQUA if 3 <= i <= 7 else BLUE for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(labels, rate, color=cols, width=0.68)
    _label(ax, bars, "{:.2f}%", dy=0.06)
    ax.set_ylim(0, max(rate) * 1.28)
    _style(ax, "share acquired within 12 months")
    ax.set_xlabel("public float, smallest to largest (tenths)", fontsize=9)
    ax.set_title("Mid-caps get bought; giants and micro-caps don't")
    fig.tight_layout(); fig.savefig(OUT / "size_hump.png"); plt.close(fig)


def fig_signals():
    """Source: data/hazard_clean.json -- logit on 959,421 rows with
    VERIFIED-TARGET labels, SEs clustered by company. Nineteen non-control
    signals clear |z|>1.96; the ten largest are shown."""
    rows = [
        ("A known activist is on the register", 7.99),
        ("A rival in the same industry was just bought", 6.79),
        ('Company disclosed a "strategic review"', 5.92),
        ("Unusual proxy activity", 5.59),
        ("Insiders who normally trade have gone quiet", 5.12),
        ('Mention of a "letter of intent"', 4.71),
        ("Auditor was changed", -4.35),
        ("Filed an S-4 (i.e. is itself a buyer)", -3.99),
        ("A new activist just filed", 3.76),
        ("Profitability (return on assets)", 3.18),
    ][::-1]
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    cols = [ORANGE if v < 0 else BLUE for _, v in rows]
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows],
                   color=cols, height=0.62)
    for b, (_, v) in zip(bars, rows):
        ax.text(v + (0.18 if v >= 0 else -0.18),
                b.get_y() + b.get_height() / 2, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-7.0, 10.0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_xlabel("z-statistic, standard errors clustered by company "
                  "(|z|>2 = unlikely to be chance)", fontsize=9)
    ax.set_title("What carries information, and in which direction", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "signals.png"); plt.close(fig)


def fig_nested():
    """Source: stress_results.json ablation stage -- leave-one-family-out,
    2 seeds, 2024 test year."""
    fams = ["Form-filing counts", "Activist + peer deals", "8-K item codes",
            "Deltas", "Per-company z-scores", "Literature variables",
            "Insider trading", "Market value"]
    vals = [5.43, 2.45, 2.15, 1.13, 0.68, 0.08, -0.15, -1.55]
    cols = [AQUA if v > 3 else BLUE if v > 0 else ORANGE for v in vals]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.barh(fams[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + (0.12 if v >= 0 else -0.12),
                b.get_y() + b.get_height() / 2, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-2.6, 6.6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("percentage points lost when the family is removed "
                  "(2 seeds; +/-2pp is noise)", fontsize=9)
    ax.set_title("Only one family of signals clearly earns its place",
                 loc="left")
    fig.tight_layout(); fig.savefig(OUT / "nested.png"); plt.close(fig)


def fig_embargo():
    """Source: stress_results.json embargo stage. The README calls this
    'flat'; it is not flat between 0 and 8 weeks, and the distinction is the
    whole leakage argument."""
    labels = ["No embargo", "Blank the final\n8 weeks",
              "Blank the final\n16 weeks"]
    vals = [20.02, 14.31, 14.46]
    base = 2.95
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    bars = ax.bar(labels, vals, color=[BLUE, AQUA, AQUA], width=0.5)
    _label(ax, bars, dy=0.35)
    ax.axhline(base, color=ORANGE, linewidth=2, linestyle="--")
    ax.text(2.42, base + 0.7, f"base rate: {base:.1f}%", ha="right",
            fontsize=9, color=ORANGE,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.annotate("", xy=(0.72, 22.0), xytext=(0.28, 22.0),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.text(0.5, 22.6, "-5.7pp", fontsize=9, color=INK2, ha="center")
    ax.annotate("", xy=(1.72, 22.0), xytext=(1.28, 22.0),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2))
    ax.text(1.5, 22.6, "flat", fontsize=9, color=INK2, ha="center")
    ax.set_ylim(0, 25)
    _style(ax, "precision at top-25/week")
    ax.set_title("A quarter of the edge lives in the last two months")
    fig.tight_layout(); fig.savefig(OUT / "embargo.png"); plt.close(fig)


def fig_featureset():
    """Source: select_cv.json, subsets A_all / C_drop_negatives / D_core /
    E_minimal, mean of three test years, both universes."""
    subs = [("E_minimal", 14), ("D_core", 37),
            ("C_drop_negatives", 49), ("A_all", 72)]
    allc, oper = [], []
    for s, _ in subs:
        allc.append(sum(r["prec"] for r in _cv("sets", s)) / 3)
        oper.append(sum(r["prec"] for r in _cv("nospac", s)) / 3)
    xs = [n for _, n in subs]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(xs, allc, marker="o", ms=7, lw=2, color=BLUE,
            label="including de-SPACs")
    ax.plot(xs, oper, marker="o", ms=7, lw=2, color=AQUA,
            label="operating companies only")
    for x, v in zip(xs, allc):
        ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=INK)
    for x, v in zip(xs, oper):
        ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color=INK)
    ax.set_xticks(xs)
    ax.set_ylim(0, 34)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    _style(ax, "precision at top-25/week")
    ax.set_xlabel("features in the model", fontsize=9)
    ax.set_title("Shells are easy to predict with 14 features; "
                 "real companies are not", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "featureset.png"); plt.close(fig)


FIGURES = (fig_funnel, fig_labels, fig_hit_rate, fig_cv_years,
           fig_universes, fig_acquirer, fig_size_hump, fig_signals,
           fig_nested, fig_embargo, fig_featureset)

if __name__ == "__main__":
    for fn in FIGURES:
        fn()
        print(f"  {fn.__name__}")
    print(f"{len(FIGURES)} figures -> {OUT}")
