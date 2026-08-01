"""Figures for docs/STRESS_RESULTS.md.

Same house style as make_charts.py: validated light categorical palette,
slots 1-3. Slot 3 (aqua) sits below 3:1 contrast on this surface, so the
relief rule applies and every mark carries a visible direct label.

Every number is measured. Values come from data/*.json where a stage wrote
one, and are otherwise recomputed here from data/pair_scores.parquet and
data/pairs.duckdb -- no model fits, ~15s. Nothing is illustrative.

    .venv/bin/python scripts/make_stress_charts.py
"""
import json
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data")
CACHE = DATA / "pair_compare.json"

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


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


# ------------------------------------------------------------------ measure --

def measure() -> dict:
    """Recompute everything the stage JSONs do not already hold. No fits."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())

    sc = pl.read_parquet("data/pair_scores.parquet")
    con = duckdb.connect("data/pairs.duckdb", read_only=True)
    pairs = pl.from_arrow(con.execute(
        "SELECT target_cik, acquirer_cik, first_ts FROM deal_pairs").arrow())
    con.close()
    con = duckdb.connect("data/deal.duckdb", read_only=True)
    sic = {r[0]: r[1] for r in
           con.execute("SELECT cik, sic FROM company_sic").fetchall()}
    con.close()

    out = {}

    # --- precision curve + per-year, both models, from the cached scores ----
    tl = _label_target(sc)
    bl = _label_buyer(sc)
    for tag, lab in (("target", tl), ("buyer", bl)):
        d = sc.with_columns(lab.alias("y"))
        out[f"curve_{tag}"] = [
            _prec(d, f"p_{tag}", n) for n in (10, 25, 50, 100, 200)]
        out[f"years_{tag}"] = [
            _prec(d.filter(pl.col("test_year") == yr), f"p_{tag}", 25)
            for yr in (2023, 2024, 2025)]

    # --- tender: lift looks fine, distinct companies do not ----------------
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/tender.duckdb' AS t (READ_ONLY)")
    con.register("f", sc.select(["cik", "week"]).to_arrow())
    tlab = con.execute("""
        SELECT f.cik, f.week, CASE WHEN EXISTS(
          SELECT 1 FROM t.tender_offers o WHERE o.cik = f.cik
            AND f.week < o.public_ts
            AND f.week >= o.public_ts - INTERVAL 52 WEEK)
          THEN 1 ELSE 0 END AS y FROM f""").pl()
    d = sc.join(tlab, on=["cik", "week"], how="left").with_columns(
        pl.col("y").fill_null(0))
    out["tender"] = []
    for yr in (2023, 2024, 2025):
        m = d.filter(pl.col("test_year") == yr)
        sel = m.sort("p_target", descending=True).group_by("week").head(25)
        hits = sel.filter(pl.col("y") == 1)
        base = float(m["y"].mean())
        out["tender"].append({
            "year": yr,
            "lift": float(sel["y"].mean()) / base if base else 0.0,
            "distinct_hits": int(hits["cik"].n_unique()),
            "available": int(m.filter(pl.col("y") == 1)["cik"].n_unique()),
        })

    # --- matching in the REAL universe, both directions ---------------------
    bf = pl.read_parquet("data/buyer_features.parquet",
                         columns=["cik", "week", "log_assets"])
    panel = (bf.join(sc, on=["cik", "week"], how="inner")
             .with_columns(pl.col("cik").replace_strict(sic, default="")
                           .str.slice(0, 2).alias("sic2")))
    byw = {(w[0] if isinstance(w, tuple) else w): g
           for w, g in panel.partition_by("week", as_dict=True).items()}
    obs = pairs.with_columns(
        (pl.col("first_ts").cast(pl.Date) - pl.duration(weeks=4))
        .dt.truncate("1w").alias("w"))
    fwd, rev, tgt_rank = [], [], []
    for r in obs.iter_rows(named=True):
        pool = byw.get(r["w"])
        if pool is None:
            continue
        ck = pool["cik"].to_numpy()
        if r["acquirer_cik"] not in ck or r["target_cik"] not in ck:
            continue
        ia = int(np.where(ck == r["acquirer_cik"])[0][0])
        it = int(np.where(ck == r["target_cik"])[0][0])
        la = pool["log_assets"].to_numpy()
        s2c = pool["sic2"].to_numpy()
        pt, pb = pool["p_target"].to_numpy(), pool["p_buyer"].to_numpy()
        tgt_rank.append(int((pt >= pt[it]).sum()))
        ts = sic.get(r["target_cik"], "")[:2]
        m1 = (s2c == ts) & (ts != "")
        s = m1 * 10 + (la - la[it]) / max(la.std(), 1e-9) \
            + 2 * pb / max(pb.std(), 1e-9)
        fwd.append(int((s >= s[ia]).sum()))
        as_ = sic.get(r["acquirer_cik"], "")[:2]
        m2 = (s2c == as_) & (as_ != "")
        s = m2 * 10 + (la[ia] - la) / max(la.std(), 1e-9) \
            + 2 * pt / max(pt.std(), 1e-9)
        rev.append(int((s >= s[it]).sum()))
    f, rv, tr = np.array(fwd), np.array(rev), np.array(tgt_rank)
    out["universe_median"] = float(panel.group_by("week").len()["len"].median())
    for tag, a in (("fwd", f), ("rev", rv)):
        out[f"match_{tag}"] = {
            "n": int(len(a)),
            "top1": float(100 * np.mean(a <= 1)),
            "top10": float(100 * np.mean(a <= 10)),
            "top100": float(100 * np.mean(a <= 100)),
            "median_rank": float(np.median(a)),
        }
    out["target_in_top25"] = float(100 * np.mean(tr <= 25))
    out["end_to_end"] = float(100 * np.mean((tr <= 25) & (f <= 100)))
    out["end_to_end_n"] = int(((tr <= 25) & (f <= 100)).sum())

    # --- alignment: observed correlation vs within-week permutation null ----
    j = (obs.join(sc.select(["cik", "week", "p_target"]),
                  left_on=["target_cik", "w"], right_on=["cik", "week"],
                  how="inner")
         .join(sc.select(["cik", "week", "p_buyer"]),
               left_on=["acquirer_cik", "w"], right_on=["cik", "week"],
               how="inner"))
    t, b = j["p_target"].to_numpy(), j["p_buyer"].to_numpy()
    wk = j["w"].to_numpy()
    rng = np.random.default_rng(42)
    null = []
    for _ in range(2000):
        bb = b.copy()
        for w in np.unique(wk):
            mm = wk == w
            if mm.sum() > 1:
                bb[mm] = rng.permutation(bb[mm])
        null.append(float(np.corrcoef(t, bb)[0, 1]))
    out["align"] = {"observed": float(np.corrcoef(t, b)[0, 1]),
                    "null": null, "n": int(j.height)}

    CACHE.write_text(json.dumps(out, indent=1))
    return out


def _label_target(sc):
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    con.register("f", sc.select(["cik", "week"]).to_arrow())
    lab = con.execute("""
        SELECT f.cik, f.week, CASE WHEN EXISTS(
          SELECT 1 FROM m.deals d WHERE d.cik = f.cik
            AND f.week < d.agreement_date
            AND f.week >= d.agreement_date - INTERVAL 52 WEEK)
          THEN 1 ELSE 0 END AS y FROM f""").pl()
    return sc.select(["cik", "week"]).join(
        lab, on=["cik", "week"], how="left")["y"].fill_null(0)


def _label_buyer(sc):
    con = duckdb.connect(":memory:")
    con.execute("ATTACH 'data/forms2.duckdb' AS f2 (READ_ONLY)")
    con.register("f", sc.select(["cik", "week"]).to_arrow())
    lab = con.execute("""
        SELECT f.cik, f.week, CASE WHEN EXISTS(
          SELECT 1 FROM f2.form_events a WHERE a.cik = f.cik AND a.family='s4'
            AND a.public_ts > f.week
            AND a.public_ts <= f.week + INTERVAL 52 WEEK)
          THEN 1 ELSE 0 END AS y FROM f""").pl()
    return sc.select(["cik", "week"]).join(
        lab, on=["cik", "week"], how="left")["y"].fill_null(0)


def _prec(d, col, n):
    sel = d.sort(col, descending=True).group_by("week").head(n)
    base = float(d["y"].mean())
    p = float(sel["y"].mean()) if sel.height else 0.0
    return {"n": n, "precision": 100 * p, "base": 100 * base,
            "lift": p / base if base else 0.0,
            "distinct_hits": int(sel.filter(pl.col("y") == 1)["cik"].n_unique())}


# ------------------------------------------------------------------ figures --

def fig_three_questions(M):
    """Source: measure() -- curve_target/curve_buyer at n=25, match_fwd,
    end_to_end. The four questions are not the same question; the lift column
    is the only thing comparable across them."""
    names = ["Will this company\nbe acquired?",
             "Will this company\nacquire someone?",
             "Told the target —\nwho is the buyer?",
             "Nothing given —\nname both ends"]
    vals = [M["curve_target"][1]["precision"], M["curve_buyer"][1]["precision"],
            M["match_fwd"]["top100"], M["end_to_end"]]
    subs = ["top 25 / week", "top 25 / week",
            f"top 100 of {M['universe_median']:.0f}", "both stages chained"]
    lifts = [M["curve_target"][1]["lift"], M["curve_buyer"][1]["lift"],
             M["match_fwd"]["top100"] / (100 * 100 / M["universe_median"]), None]
    cols = [BLUE, ORANGE, AQUA, INK2]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, vals, color=cols, height=0.6)
    for i, b in enumerate(bars):
        lift = f"   ({lifts[i]:.1f}× lift)" if lifts[i] else ""
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2,
                f"{vals[i]:.1f}%{lift}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    # The list size belongs with the question, not floating on the bar -- as an
    # overlay it clipped at the bar edge.
    ax.set_yticklabels([f"{n}\n({s})" for n, s in zip(names, subs)],
                       fontsize=9)
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_xlabel("how often the answer is right (%)", fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Buying is twice as predictable as being bought.\n"
                 "Pairing them, end to end, is not predictable at all.",
                 loc="left", pad=12)
    _save(fig, "s1_three_questions.png")


def fig_curves(M):
    """Source: measure() -- curve_target / curve_buyer, precision at each list
    size on the cached 2023-2025 scores."""
    ns = [c["n"] for c in M["curve_target"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for tag, col, lbl in (("buyer", ORANGE, "Buyer  (files an S-4)"),
                          ("target", BLUE, "Target  (merger proxy)")):
        v = [c["precision"] for c in M[f"curve_{tag}"]]
        ax.plot(ns, v, "-o", color=col, linewidth=2, markersize=7, label=lbl)
        ax.annotate(f"{v[1]:.1f}%", (ns[1], v[1]), textcoords="offset points",
                    xytext=(6, 8), fontsize=9, color=INK)
    base = M["curve_target"][0]["base"]
    ax.axhline(base, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(200, base + 0.5, f"base rate {base:.1f}%", ha="right", fontsize=8,
            color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("companies flagged per week", fontsize=9)
    _style(ax, "precision (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Precision falls as the list grows — for both models",
                 loc="left", pad=10)
    _save(fig, "s2_precision_curve.png")


def fig_years(M):
    """Source: measure() -- years_target / years_buyer at top 25/week.
    Regime dependence is far larger than seed noise (+/-2pp)."""
    yrs = [2023, 2024, 2025]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(3)
    for i, (tag, col, lbl) in enumerate(
            (("target", BLUE, "Target"), ("buyer", ORANGE, "Buyer"))):
        v = [c["precision"] for c in M[f"years_{tag}"]]
        b = ax.bar(x + (i - 0.5) * 0.38, v, width=0.36, color=col, label=lbl)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.4,
                    f"{r.get_height():.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yrs])
    _style(ax, "precision @ top 25/week (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Year-to-year spread dwarfs seed noise — quote the mean, "
                 "never one year", loc="left", pad=10)
    _save(fig, "s3_year_spread.png")


def fig_tender(M):
    """Source: measure() -- tender. The left panel is what the stress suite
    reported; the right panel is why it means nothing. Two panels, not two
    y-axes."""
    yrs = [t["year"] for t in M["tender"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    b = a1.bar([str(y) for y in yrs], [t["lift"] for t in M["tender"]],
               color=AQUA, width=0.55)
    for r in b:
        a1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.12,
                f"{r.get_height():.2f}×", ha="center", fontsize=9.5, color=INK)
    a1.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    # Left of the bars: at the right it collided with 2024's value label.
    a1.text(-0.42, 1.10, "chance", ha="left", fontsize=8, color=INK2)
    _style(a1, "lift vs base rate")
    a1.set_title("What it looked like", loc="left", fontsize=11)

    b = a2.bar([str(y) for y in yrs], [t["distinct_hits"] for t in M["tender"]],
               color=ORANGE, width=0.55)
    for i, r in enumerate(b):
        a2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.12,
                f"{int(r.get_height())} of {M['tender'][i]['available']}",
                ha="center", fontsize=9.5, color=INK)
    _style(a2, "distinct companies behind the number")
    a2.set_ylim(0, max(t["distinct_hits"] for t in M["tender"]) * 1.45)
    a2.set_title("What it rested on", loc="left", fontsize=11)
    fig.suptitle("The tender-offer validation is uninformative: 2023's 3.4× "
                 "lift is ONE company held 23 weeks",
                 x=0.007, ha="left", fontsize=12, fontweight="bold")
    _save(fig, "s4_tender_trap.png")


def fig_matching(M):
    """Source: data/matching.json (101 sampled distractors) vs measure()
    match_fwd (the real ~7,100-company week). Same model, same pairs."""
    mj = json.loads((DATA / "matching.json").read_text())
    m4 = next(r for r in mj if r["embargo_weeks"] == 4
              and r.get("variant", "full") == "full")
    labels = ["top-1", "top-10"]
    sampled = [m4["top1"], m4["top10"]]
    real = [M["match_fwd"]["top1"], M["match_fwd"]["top10"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(2)
    b1 = ax.bar(x - 0.19, sampled, width=0.36, color=AQUA,
                label="vs 100 sampled distractors")
    b2 = ax.bar(x + 0.19, real, width=0.36, color=BLUE,
                label=f"vs the real week (~{M['universe_median']:.0f} companies)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2,
                    f"{r.get_height():.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _style(ax, "true acquirer found (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("The 101-candidate setup flatters matching by ~70×\n"
                 f"median rank 3 of 101  →  "
                 f"{M['match_fwd']['median_rank']:.0f} of "
                 f"{M['universe_median']:.0f}",
                 loc="left", pad=12)
    _save(fig, "s5_matching_universe.png")


def fig_alignment(M):
    """Source: measure() -- align. 2000 within-week permutations: reshuffle
    which acquirer pairs with which target, holding the week fixed."""
    a = M["align"]
    null = np.array(a["null"])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.hist(null, bins=40, color=GRID, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(a["observed"], color=ORANGE, linewidth=2.5)
    ax.axvline(float(null.mean()), color=INK2, linewidth=1,
               linestyle=(0, (4, 3)))
    top = ax.get_ylim()[1]
    ax.text(a["observed"], top * 0.96, f"  observed {a['observed']:+.3f}",
            color=ORANGE, fontsize=9.5, va="top")
    ax.text(float(null.mean()), top * 0.60,
            f"null mean {null.mean():+.3f}  ", color=INK2, fontsize=9,
            va="top", ha="right")
    p = float(np.mean(null >= a["observed"]))
    ax.set_xlabel("corr(target score, buyer score) across real pairs",
                  fontsize=9)
    _style(ax, "permutation draws")
    ax.set_title("The two screens carry no pair-specific information\n"
                 f"observed sits on the null mean — p = {p:.2f}, "
                 f"n = {a['n']} pairs", loc="left", pad=12)
    _save(fig, "s6_alignment_null.png")


if __name__ == "__main__":
    print("measuring (no model fits)...", flush=True)
    M = measure()
    print(f"  universe {M['universe_median']:.0f}/week | "
          f"{M['match_fwd']['n']} pairs | "
          f"end-to-end {M['end_to_end']:.2f}%", flush=True)
    fig_three_questions(M)
    fig_curves(M)
    fig_years(M)
    fig_tender(M)
    fig_matching(M)
    fig_alignment(M)
