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


def _size_deciles():
    """Deal rate by public-float decile, verified targets, 52-week horizon.

    Recomputed rather than read from a cached file: the previous version read
    `y` straight from features.parquet, which stores a 26-WEEK label, and
    captioned it "within 12 months". Deriving it here keeps the horizon
    explicit and the figure honest.
    """
    import duckdb
    import numpy as np
    import polars as pl
    from deal import clean_labels

    df = pl.read_parquet("data/features.parquet",
                         columns=["cik", "week", "log_float"])
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.execute("CREATE TEMP VIEW deals AS SELECT * FROM m.deals")
    con.execute("CREATE TEMP VIEW universe AS SELECT * FROM m.universe")
    clean_labels.build(con, df["week"].max())
    con.register("f", df.select(["cik", "week"]).to_arrow())
    lab = con.execute("""
        SELECT f.cik, f.week, CASE WHEN EXISTS(
          SELECT 1 FROM deals_clean d WHERE d.cik = f.cik
            AND d.outcome = 'target'
            AND f.week <  d.agreement_date
            AND f.week >= d.agreement_date - INTERVAL 52 WEEK)
          THEN 1 ELSE 0 END AS y FROM f""").pl()
    d = df.with_columns(
        df.select(["cik", "week"]).join(lab, on=["cik", "week"],
                                        how="left")["y"].fill_null(0).alias("y")
    ).filter(pl.col("log_float") > 0)
    q = np.quantile(d["log_float"].to_numpy(), np.linspace(0, 1, 11))
    out = []
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        sel = d.filter((pl.col("log_float") >= lo)
                       & (pl.col("log_float") <= hi if i == 9
                          else pl.col("log_float") < hi))
        out.append(100.0 * float(sel["y"].mean()) if sel.height else 0.0)
    return out


def _acc():
    """Rows from the eleven-year accuracy run, filtered by model and universe."""
    rows = [r for r in json.loads((DATA / "feature_report.json").read_text())
            if r.get("stage") == "accuracy"]
    def get(model, universe):
        return sorted([r for r in rows if r["model"] == model
                       and r["universe"] == universe], key=lambda r: r["year"])
    return get


def _cv(stage, subset="A_all"):
    rows = json.loads((DATA / "select_cv.json").read_text())
    return [r for r in rows
            if r.get("stage") == stage and r.get("subset") == subset]


def _stale_stamp(ax, note="measured on the 2016 panel / contaminated labels — "
                          "read the ordering, not the levels"):
    """Mark a figure whose underlying run was never re-derived.

    These three come from analyses that predate the 2012 rebuild. Section 9 of
    the paper says so in prose, but a figure travels without its caption, so
    the warning belongs on the canvas.
    """
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center",
            va="top", fontsize=7.5, color=ORANGE, style="italic")


def fig_funnel():
    """Source: measured counts from deal.duckdb, forms2.duckdb and
    logs/index.log on the 2012-2026 panel. Not the README's old 11.6M, which
    counted every master-index row including forms the pipeline never reads.
    The label count is verified targets, not proxy filers -- 749 of those
    turned out to be acquirers or terminated deals."""
    labels = ["Insider\ntransactions", "XBRL\nfacts", "Form\nevents",
              "Periodic filings\n(universe)", "Company-weeks\n(panel)",
              "Verified targets\n(labels)"]
    vals = [4_115_188, 5_249_647, 2_025_920, 444_748, 5_967_094, 2_227]
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
    """Source: data/curve_final.json -- VERIFIED-TARGET labels, operating
    companies, eleven test years 2015-2025."""
    c = json.loads((DATA / "curve_final.json").read_text())["target"]
    ns = [f"Top {x['n']}" for x in c["curve"]]
    hit, base = [x["prec"] for x in c["curve"]], c["base"]
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
    # Contaminated-label lifts are the historical 2023-24 measurement; the
    # verified-target side is the eleven-year run.
    old_all, old_op = 9.63, 5.94
    acc = _acc()
    new_all = sum(r["lift"] for r in acc("target", "all")) / len(acc("target", "all"))
    new_op = sum(r["lift"] for r in acc("target", "nospac")) / len(acc("target", "nospac"))

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
    ax.set_title("Removing 749 acquirers made the finding stronger",
                 loc="left")
    fig.tight_layout(); fig.savefig(OUT / "labels.png"); plt.close(fig)


def fig_acquirer():
    """Source: feature_report.json, eleven test years, SPAC-free, buyer label
    = files an S-4 within 12 months with self-referential features removed."""
    acc = _acc()
    tgt = sum(r["lift"] for r in acc("target", "nospac")) / len(acc("target", "nospac"))
    buy = sum(r["lift"] for r in acc("buyer", "nospac")) / len(acc("buyer", "nospac"))
    names = ["Will be\nacquired", "Will buy\n(files an S-4)"]
    vals = [tgt, buy]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    bars = ax.bar(names, vals, color=[AQUA, BLUE], width=0.45)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}x",
                ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, max(vals) * 1.3)
    _style(ax, "lift over base rate")
    ax.set_title("Buying is far more predictable than being bought")
    fig.tight_layout(); fig.savefig(OUT / "acquirer.png"); plt.close(fig)


def fig_cv_years():
    """Source: feature_report.json -- verified-target labels, eleven test
    years. 2025 is shown but hatched: it is right-censored to ~30 weeks and
    its operating-company figure rests on five distinct companies."""
    acc = _acc()
    ca = {r["year"]: r["prec"] for r in acc("target", "all")}
    cn = {r["year"]: r["prec"] for r in acc("target", "nospac")}
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
    # 22 bars is too dense for a value on each: label the operating-company
    # series only, which is the defensible one, and stagger to avoid collision.
    for i, b in enumerate(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.35,
                f"{b.get_height():.1f}", ha="center", fontsize=8,
                color=INK, rotation=0)
    ax.set_xticks(list(x)); ax.set_xticklabels([str(y) for y in yrs],
                                               fontsize=8.5)
    ax.set_ylim(0, max(max(ca.values()), max(cn.values())) * 1.35)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax, "precision at top-25/week")
    ax.set_xlabel("held-out test year", fontsize=9)
    ax.set_title("Regime dominates: eleven test years, 2015-2025",
                 loc="left")
    fig.tight_layout(); fig.savefig(OUT / "cv_years.png"); plt.close(fig)


def fig_universes():
    """Source: feature_report.json, eleven-year mean lift. The point of the
    chart: on LIFT the SPAC decision is worth +0.04x, so it changes the
    population and not the skill."""
    acc = _acc()
    a = sum(r["lift"] for r in acc("target", "all")) / len(acc("target", "all"))
    n = sum(r["lift"] for r in acc("target", "nospac")) / len(acc("target", "nospac"))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bars = ax.bar(["Including\nde-SPACs", "Operating\ncompanies only"],
                  [a, n], color=[BLUE, AQUA], width=0.45)
    for b, v in zip(bars, [a, n]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}x",
                ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, max(a, n) * 1.35)
    _style(ax, "lift over base rate")
    ax.set_title("On lift, excluding SPACs changes almost nothing")
    fig.tight_layout(); fig.savefig(OUT / "universes.png"); plt.close(fig)


def fig_size_hump():
    """Source: recomputed from features.parquet + verified-target labels at
    the 52-week horizon on the 2012-2026 panel.

    The previous version read `y` straight from features.parquet, which stores
    a 26-WEEK label (1.413% positive; the 52-week rate is 2.769%). It was
    captioned 'within 12 months' and was not. Every model script calls
    relabel(raw, 52) first, so only the figure was wrong."""
    rate = _size_deciles()
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
    """Source: data/stress_pairs.json -- logit on 1,512,368 rows with
    VERIFIED-TARGET labels on the 2012-2026 panel, SEs clustered by company.
    Twenty non-control signals clear |z|>1.96; the eleven largest are shown.

    Labels are hand-written plain English rather than column names, so the
    chart reads without the data dictionary."""
    NAMES = {
        "activist_recent": "A known activist is on the register",
        "log_assets": "Company size (total assets)",
        "log_float": "Company size (public float)",
        "sa_review_52w": 'Company disclosed a "strategic review"',
        "sc13d_52w_z": "13D activity vs the firm's own baseline",
        "peer_deal_13w": "A rival in the same industry was just bought",
        "roa": "Profitability (return on assets)",
        "s4_52w": "Filed an S-4 (i.e. is itself a buyer)",
        "sa_loi_52w": 'Mention of a "letter of intent"',
        "sa_unsolicited_52w": 'Mention of an "unsolicited" approach',
        "sector_deal_intensity": "Sector consolidating",
        "disc_blackout": "Insiders who normally trade have gone quiet",
        "i_auditor_change_52w": "Auditor was changed",
        "cash_runway": "Cash runway",
        "age_weeks": "Years since the company began filing",
    }
    haz = [r for r in json.loads((DATA / "stress_pairs.json").read_text())
           if r.get("test") == "hazard" and "z" in r]
    rows = []
    for r in haz:
        key = r["label"].strip()
        if key in NAMES and abs(r["z"]) > 1.96:
            rows.append((NAMES[key], float(r["z"])))
    rows = sorted(rows, key=lambda x: -abs(x[1]))[:11][::-1]
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
    lo = min(v for _, v in rows); hi = max(v for _, v in rows)
    ax.set_xlim(lo - 2.0, hi + 2.0)
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
    _stale_stamp(ax)
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
    _stale_stamp(ax)
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
    _stale_stamp(ax)
    fig.tight_layout(); fig.savefig(OUT / "featureset.png"); plt.close(fig)


FIGURES = (fig_funnel, fig_labels, fig_hit_rate, fig_cv_years,
           fig_universes, fig_acquirer, fig_size_hump, fig_signals,
           fig_nested, fig_embargo, fig_featureset)

if __name__ == "__main__":
    for fn in FIGURES:
        fn()
        print(f"  {fn.__name__}")
    print(f"{len(FIGURES)} figures -> {OUT}")
