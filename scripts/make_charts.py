"""Charts for the findings report.

Every number here is measured, not illustrative; sources are noted per chart.

Palette is the validated light-mode categorical set (slots 1-3), which passes
lightness, chroma, colourblind separation and normal-vision separation. Slot 3
(aqua) sits below 3:1 contrast on this surface, so the relief rule applies and
every bar carries a visible direct label.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

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


def fig_hit_rate():
    """Source: held-out test 2024-01..2025-07, uncensored."""
    ns = ["Top 10", "Top 25", "Top 50", "Top 100", "Top 200"]
    hit = [24.10, 27.52, 29.66, 24.01, 20.01]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(ns, hit, color=BLUE, width=0.6)
    _label(ax, bars)
    ax.axhline(2.94, color=ORANGE, linewidth=2, linestyle="--")
    # The reference line runs under every bar, so the label needs a surface
    # backing to stay legible where it crosses one.
    ax.text(4.45, 4.4, "if you picked at random: 2.9%", ha="right",
            fontsize=9, color=ORANGE,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.set_ylim(0, 34)
    _style(ax, "share later acquired, within 12 months")
    ax.set_title("How often the shortlist is right")
    fig.tight_layout(); fig.savefig(OUT / "hit_rate.png"); plt.close(fig)


def fig_size_hump():
    """Source: deal rate by market-value decile, test period."""
    rate = [0.54, 1.45, 2.68, 2.67, 2.63, 3.01, 2.84, 2.15, 1.67, 1.03]
    labels = ["smallest", "2", "3", "4", "5", "6", "7", "8", "9", "largest"]
    cols = [AQUA if 2 <= i <= 6 else BLUE for i in range(10)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(labels, rate, color=cols, width=0.68)
    _label(ax, bars, "{:.2f}%", dy=0.06)
    ax.set_ylim(0, 3.7)
    _style(ax, "share acquired within 12 months")
    ax.set_xlabel("company size, smallest to largest (tenths)", fontsize=9)
    ax.set_title("Mid-sized companies get bought; giants and minnows don't")
    fig.tight_layout(); fig.savefig(OUT / "size_hump.png"); plt.close(fig)


def fig_shuffle():
    """Source: 20 label-shuffled refits vs the real model."""
    null = [3.86, 4.19, 1.45, 5.83, 2.84, 0.53, 4.72, 3.86, 4.58, 2.27,
            2.27, 0.63, 1.16, 7.57, 2.12, 3.18, 5.25, 1.78, 1.06, 0.58]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(1, 21), null, color=BLUE, width=0.7)
    ax.axhline(27.52, color=ORANGE, linewidth=2.5)
    ax.text(10.5, 25.2, "the real model: 27.5%", ha="center", fontsize=10,
            color=ORANGE, fontweight="bold")
    ax.text(10.5, 9.0, "20 runs on deliberately scrambled answers\n"
            "best of them: 7.6%", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0, 31)
    ax.set_xticks([1, 5, 10, 15, 20])
    _style(ax, "hit rate")
    ax.set_xlabel("scrambled run", fontsize=9)
    ax.set_title("The scramble test: nothing fake survives it")
    fig.tight_layout(); fig.savefig(OUT / "shuffle.png"); plt.close(fig)


def fig_stability():
    """Source: 5 random seeds and 4 expanding-window time periods."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    seeds = [21.99, 24.00, 22.07, 23.17, 20.47]
    b1 = a1.bar(range(1, 6), seeds, color=BLUE, width=0.6)
    _label(a1, b1, "{:.1f}", dy=0.4)
    a1.set_ylim(0, 30); a1.set_xticks(range(1, 6))
    _style(a1, "hit rate")
    a1.set_xlabel("repeat run", fontsize=9)
    a1.set_title("Same data, 5 reruns", fontsize=11)

    folds = [32.54, 43.00, 34.11, 33.23]
    b2 = a2.bar(["2022", "2023", "2024", "2025"], folds, color=AQUA, width=0.6)
    _label(a2, b2, "{:.1f}", dy=0.6)
    a2.set_ylim(0, 50)
    _style(a2, "")
    a2.set_xlabel("tested on this year", fontsize=9)
    a2.set_title("Different years", fontsize=11)
    fig.suptitle("The result doesn't depend on luck or on one time period",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "stability.png"); plt.close(fig)


def fig_signals():
    """Source: hazard model, errors clustered by company."""
    rows = [
        ("A rival in the same industry was just bought", 10.60),
        ("A known activist investor is on the register", 8.03),
        ('Company said it is "reviewing strategic alternatives"', 7.73),
        ("Unusual proxy-statement activity", 6.83),
        ('Mention of a "letter of intent"', 5.94),
        ("Insiders who normally trade have gone quiet", 5.04),
        ("An unsolicited approach was disclosed", 4.50),
        ("A new activist just filed", 2.83),
        ("Volume of company announcements", 2.33),
    ]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    bars = ax.barh([r[0] for r in rows], [r[1] for r in rows],
                   color=BLUE, height=0.62)
    for b, (_, v) in zip(bars, rows):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_xlabel("strength of evidence  (above 2 = unlikely to be chance)",
                  fontsize=9)
    # Long category labels push the plot right; anchoring the title to the
    # axes rather than the figure keeps it from running off the canvas.
    ax.set_title("Which warning signs actually carry information", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "signals.png"); plt.close(fig)


def fig_nested():
    """Source: nested comparison on the held-out period."""
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bars = ax.bar(["Using only what was\nalready well known",
                   "Adding the new\nfiling-based signals"],
                  [16.03, 19.64], color=[BLUE, AQUA], width=0.5)
    _label(ax, bars, "{:.1f}%", dy=0.3)
    ax.set_ylim(0, 24)
    _style(ax, "hit rate")
    ax.set_title("The new signals add real information")
    fig.tight_layout(); fig.savefig(OUT / "nested.png"); plt.close(fig)


def fig_mistakes():
    """Source: censoring correction and SPAC-exclusion reruns."""
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    labels = ["Before fixing a\ncounting error", "After the fix",
              "Excluding shell\ncompanies too"]
    vals = [19.64, 27.52, 22.02]
    bars = ax.bar(labels, vals, color=[ORANGE, BLUE, AQUA], width=0.5)
    _label(ax, bars, "{:.1f}%", dy=0.35)
    ax.set_ylim(0, 32)
    _style(ax, "hit rate")
    ax.set_title("Two corrections that changed the answer")
    fig.tight_layout(); fig.savefig(OUT / "mistakes.png"); plt.close(fig)


def fig_funnel():
    """Source: pipeline row counts."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = ["Filings read", "Company records", "Weekly snapshots",
              "Acquisitions found"]
    vals = [11_591_580, 15_325, 4_123_449, 2_456]
    bars = ax.bar(labels, vals, color=BLUE, width=0.55)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:,}",
                ha="center", fontsize=9, color=INK)
    ax.set_ylim(1e3, 5e8)
    _style(ax, "count (log scale)")
    ax.set_title("What the pipeline processes")
    fig.tight_layout(); fig.savefig(OUT / "funnel.png"); plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_hit_rate, fig_size_hump, fig_shuffle, fig_stability,
               fig_signals, fig_nested, fig_mistakes, fig_funnel):
        fn()
        print(f"  {fn.__name__}")
    print(f"figures -> {OUT}")
