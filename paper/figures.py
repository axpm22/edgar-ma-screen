"""Figures for the regional-bank deposit paper. Every value is sourced in the paper's
reference list; nothing here is interpolated or modeled. Run: python3 figures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444", "figure.dpi": 200, "savefig.bbox": "tight",
})
INK, ACCENT, MUTED = "#1f2d3d", "#a33a2c", "#9aa5b1"


def fig1():
    """Failures by asset size — 2023 vs 2008. FDIC receivership data."""
    banks = ["IndyMac\n2008", "Signature\n2023", "SVB\n2023",
             "First Republic\n2023", "Washington Mutual\n2008"]
    assets = [32, 110, 209, 229, 307]
    colors = [MUTED, ACCENT, ACCENT, ACCENT, MUTED]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    bars = ax.barh(banks, assets, color=colors, height=0.62)
    for b, v in zip(bars, assets):
        ax.text(v + 5, b.get_y() + b.get_height() / 2, f"${v}B", va="center", fontsize=9)
    ax.set_xlim(0, 355)
    ax.set_xlabel("Total assets at failure ($ billions)")
    ax.set_title("Figure 1. Three of the four largest U.S. bank failures ever\n"
                 "happened in eight weeks of 2023", loc="left", pad=10)
    ax.xaxis.grid(True, color="#e6e8eb", lw=0.7)
    ax.set_axisbelow(True)
    fig.savefig("fig1_failures.png")


def fig2():
    """The round trip: uninsured deposits as a share of total domestic deposits."""
    labels = ["Q4 2019\n(pre-pandemic)", "Q2 2024\n(post-crisis trough)", "Q4 2025"]
    vals = [43.3, 40.7, 43.3]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(labels, vals, color=ACCENT, lw=2, marker="o", ms=7, zorder=3)
    ax.axhline(43.3, color=INK, ls=(0, (4, 3)), lw=0.9, zorder=1)
    for x, v, dy in zip(labels, vals, (-20, -22, 12)):
        ax.annotate(f"{v}%", (x, v), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=10, color=INK)
    ax.text(1.0, 43.48, "pre-pandemic level", fontsize=8.2, color=INK,
            style="italic", ha="center")
    ax.set_ylim(39.5, 44.6)
    ax.set_ylabel("Uninsured share of domestic deposits (%)")
    ax.set_title("Figure 2. Depositors returned to the exact uninsured concentration\n"
                 "the 2023 failures were supposed to have taught them to avoid",
                 loc="left", pad=10)
    ax.yaxis.grid(True, color="#e6e8eb", lw=0.7)
    ax.set_axisbelow(True)
    fig.savefig("fig2_uninsured.png")


def fig3():
    """Emergency liquidity drawn in March 2023, against the prior record."""
    labels = ["Discount window\nweek to\nMar 8, 2023",
              "Discount window\n2008 peak\n(prior record)",
              "Discount window\nweek to\nMar 15, 2023",
              "BTFP\npeak\n(Jan 2024)"]
    vals = [4.58, 111, 152.85, 165]
    colors = [MUTED, MUTED, ACCENT, "#c98a7d"]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"${v:g}B", ha="center", fontsize=9)
    ax.set_ylim(0, 195)
    ax.set_ylabel("Outstanding ($ billions)")
    ax.set_title("Figure 3. The March 2023 draw broke the 2008 record in one week —\n"
                 "but the BTFP peak is not a stress reading", loc="left", pad=10)
    ax.yaxis.grid(True, color="#e6e8eb", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.2)
    fig.savefig("fig3_liquidity.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("wrote fig1_failures.png fig2_uninsured.png fig3_liquidity.png")
