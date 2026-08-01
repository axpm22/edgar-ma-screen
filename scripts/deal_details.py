"""Beyond "a deal": what can be said about WHICH deal, before it is announced?

Four details, each with the baseline it has to beat stated next to it, because
three of the four have a majority class big enough to look like skill.

  structure    stock (425/S-4) vs cash tender (SC TO-T). Baseline: majority.
  size order   is the acquirer bigger than the target? Baseline: majority.
  completion   does the deal close, or does the target keep filing? This is
               the one nobody has measured here, and the one that matters --
               a screen that flags deals which then break is worth less than
               its precision suggests.
  lead time    descriptive: how many weeks before announcement the target
               first entered the top 25.

    .venv/bin/python scripts/deal_details.py

12 fits on frames of a few hundred rows, under a minute.

Deviations from the plan's code, all for correctness, all cheap:

1. Completion cannot be asked of `deal_pairs` at all: `load_pairs.orient` calls
   target whichever party stopped filing within 270 days, so 691 of 691 pairs
   carry label 1 and the broken deals are precisely the ones dropped upstream
   as ambiguous. The pair-based version is kept, reporting its own tautology,
   and the question is re-asked on `tender.duckdb`, whose subject CIKs were
   parsed from SEC headers rather than inferred from the outcome
   (`deal_completes_tender`). Right-censoring is truncated there and in the
   pair version (global constraint 4): `universe.delisted` is a LAST-SEEN
   date, so an unresolved recent deal reads as "completed" rather than as
   missing.
2. `acquirer_is_bigger` gets a no-model reference and a no-float variant. The
   label is a deterministic function of two features that are both observable
   at the observation week, so this is not forecasting -- and log_float is a
   near-perfect stand-in for the dropped log_assets. Both numbers are needed
   before any AUC here can be read as skill.
3. Every detail prints an explicit leak check of its own feature list.
4. Lead time reports a second denominator: pair_scores.parquet only covers
   2023-2025, so pairs announced before that can never be flagged and do not
   belong in a "how often were we early" rate.
"""
import datetime as dt
import gc
import json

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

SEEDS = (11, 22)
CUT = dt.date(2024, 1, 1)
SURVIVE_DAYS = 270
# A completion is evidenced by absence, a NON-completion only by a filing that
# lands after the 270-day window. Filings are quarterly at best, so near the
# panel edge that evidence cannot exist and every recent deal reads as
# "completed" -- measured: the tender completion rate is 1.00 from 2025Q2 on,
# against ~0.65 historically. Two quarters of slack buys back the evidence.
CENSOR_SLACK_DAYS = 180
MIN_TEST = 50  # below this a detail is untestable, not merely uncertain
OUT = "data/deal_details.json"

PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "num_leaves": 15, "min_data_in_leaf": 30, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbosity": -1, "num_threads": 2,
}
ROUNDS = 200


def leak_check(name: str, cols: list[str], banned: set[str]) -> dict:
    """Verify the exclusions actually took effect, rather than assuming."""
    present = sorted(c for c in cols if c in banned)
    print(f"  leak-check {name}: {len(cols)} features, "
          f"banned {sorted(banned)} -> "
          f"{'PRESENT: ' + ', '.join(present) if present else 'none present'}",
          flush=True)
    return {"n_features": len(cols), "banned": sorted(banned),
            "banned_present": present}


def auc_of(y, p):
    """Rank AUC. 0.5 when either class is empty."""
    order = np.argsort(p)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) \
        if n1 and n0 else 0.5


def fit_score(df, cols, ycol):
    """Time-split accuracy and AUC against the majority-class baseline."""
    tr = df.filter(pl.col("first_ts") < CUT)
    te = df.filter(pl.col("first_ts") >= CUT)
    if tr.height < 50 or te.height < 20 or te[ycol].n_unique() < 2:
        return None
    majority = max(float(te[ycol].mean()), 1 - float(te[ycol].mean()))
    accs, aucs = [], []
    for s in SEEDS:
        d = lgb.Dataset(tr.select(cols).to_pandas().astype("float32"),
                        label=tr[ycol].to_pandas())
        m = lgb.train({**PARAMS, "bagging_seed": s, "feature_fraction_seed": s,
                       "data_random_seed": s}, d, num_boost_round=ROUNDS)
        p = np.asarray(m.predict(te.select(cols).to_pandas().astype("float32")))
        y = te[ycol].to_numpy()
        accs.append(float(np.mean((p > 0.5).astype(int) == y)))
        aucs.append(auc_of(y, p))
        del m, d
        gc.collect()
    acc = 100 * float(np.mean(accs))
    return {"n_train": tr.height, "n_test": te.height,
            "accuracy": acc,
            "majority_baseline": 100 * majority,
            "beats_baseline_by": acc - 100 * majority,
            "auc": float(np.mean(aucs)),
            "auc_sd": float(np.std(aucs)),
            "positive_rate": 100 * float(te[ycol].mean()),
            # Two verdict flags so the write-up cannot quietly drop them.
            "underpowered": te.height < MIN_TEST,
            "no_signal": acc - 100 * majority < 2.0}


def main() -> None:
    con = duckdb.connect("data/pairs.duckdb", read_only=True)
    pairs = pl.from_arrow(con.execute(
        "SELECT target_cik, acquirer_cik, first_ts, form, n_filings "
        "FROM deal_pairs").arrow())
    con.close()

    feats = pl.read_parquet("data/features.parquet")
    panel_end = feats["week"].max()
    meta = duckdb.connect("data/deal.duckdb", read_only=True)
    sic = pl.from_arrow(meta.execute(
        "SELECT cik, sic FROM company_sic").arrow())
    uni = pl.from_arrow(meta.execute(
        "SELECT cik, delisted FROM universe").arrow())
    meta.close()

    # Observation week: 4 weeks before announcement, for both parties.
    obs = pairs.with_columns(
        (pl.col("first_ts").cast(pl.Date)
         - pl.duration(weeks=4)).dt.truncate("1w").alias("week"))

    side_cols = ["log_assets", "log_float", "cash_to_assets", "leverage",
                 "fcf_to_assets", "operating_margin", "revenue_growth",
                 "goodwill_to_assets", "form8k_26w", "sc13d_52w",
                 "sector_deal_intensity", "activist_reach"]
    side = feats.select(["cik", "week"] + [c for c in side_cols
                                           if c in feats.columns])
    t_side = side.rename({c: f"t_{c}" for c in side.columns
                          if c not in ("cik", "week")}) \
                 .rename({"cik": "target_cik"})

    df = (obs
          .join(t_side, on=["target_cik", "week"], how="inner")
          .join(side.rename({c: f"a_{c}" for c in side.columns
                             if c not in ("cik", "week")})
                    .rename({"cik": "acquirer_cik"}),
                on=["acquirer_cik", "week"], how="inner")
          .join(sic.rename({"cik": "target_cik", "sic": "t_sic"}),
                on="target_cik", how="left")
          .join(sic.rename({"cik": "acquirer_cik", "sic": "a_sic"}),
                on="acquirer_cik", how="left"))

    df = df.with_columns([
        (pl.col("t_sic").str.slice(0, 2)
         == pl.col("a_sic").str.slice(0, 2)).cast(pl.Int8).alias("same_sic2"),
        (pl.col("a_log_assets") - pl.col("t_log_assets")).alias("size_gap"),
    ])
    cols = [c for c in df.columns
            if c.startswith(("t_", "a_")) and c not in ("t_sic", "a_sic")] \
        + ["same_sic2"]
    cols = [c for c in cols if df[c].dtype.is_numeric()]

    print(f"pairs {pairs.height:,}  both parties in panel {df.height:,}  "
          f"panel_end {panel_end}", flush=True)

    out = {"n_pairs_total": pairs.height, "n_pairs_in_panel": df.height,
           "panel_end": str(panel_end), "obs_lead_weeks": 4}

    # 1. Structure. 425 => securities issued => stock consideration.
    # SC 13E3 (43 episodes) sits in the 0 class with SC TO-T: going-private
    # deals are overwhelmingly cash.
    d1 = df.with_columns((pl.col("form") == "425").cast(pl.Int8).alias("yv"))
    c1 = [c for c in cols if c != "size_gap"]
    r1 = fit_score(d1, c1, "yv")
    out["structure_stock_vs_cash"] = {
        **(r1 or {}),
        # The label is read off `form`; nothing form-derived may be an input.
        **leak_check("structure", c1, {"form", "n_filings", "last_ts",
                                       "size_gap"})}

    # 2. Size order. size_gap is the answer, so it cannot be an input.
    d2 = df.with_columns((pl.col("size_gap") > 0).cast(pl.Int8).alias("yv"))
    c2 = [c for c in cols if c not in ("size_gap", "a_log_assets",
                                       "t_log_assets")]
    r2 = fit_score(d2, c2, "yv")
    out["acquirer_is_bigger"] = {
        **(r2 or {}),
        **leak_check("size order", c2, {"size_gap", "a_log_assets",
                                        "t_log_assets"})}

    # DEVIATION: the label is a deterministic function of two features that
    # are both known at the observation week, and log_float proxies for the
    # log_assets we dropped. Both references below are needed before any AUC
    # here is called skill.
    te2 = d2.filter(pl.col("first_ts") >= CUT)
    if te2.height:
        gap = d2["size_gap"].to_numpy()
        fgap = (d2["a_log_float"] - d2["t_log_float"]).to_numpy()
        ok = ~(np.isnan(gap) | np.isnan(fgap))
        out["acquirer_is_bigger"]["float_rule_accuracy"] = 100 * float(np.mean(
            ((te2["a_log_float"] - te2["t_log_float"]).to_numpy() > 0)
            .astype(int) == te2["yv"].to_numpy()))
        out["acquirer_is_bigger"]["corr_size_gap_float_gap"] = float(
            np.corrcoef(gap[ok], fgap[ok])[0, 1]) if ok.sum() > 2 else None
    c2b = [c for c in c2 if c not in ("a_log_float", "t_log_float")]
    r2b = fit_score(d2, c2b, "yv")
    out["acquirer_is_bigger_no_float"] = {
        **(r2b or {}),
        **leak_check("size order, no float", c2b,
                     {"size_gap", "a_log_assets", "t_log_assets",
                      "a_log_float", "t_log_float"})}

    # 3. Completion. A target that is still filing 270 days later did not get
    # acquired -- the deal broke, or it was never a target.
    d3 = (df.join(uni.rename({"cik": "target_cik"}), on="target_cik",
                  how="left")
            .with_columns(
                (pl.col("delisted").cast(pl.Date)
                 <= pl.col("first_ts").cast(pl.Date)
                 + pl.duration(days=SURVIVE_DAYS)).fill_null(False)
                .cast(pl.Int8).alias("yv")))
    # DEVIATION (global constraint 4): `delisted` is a last-seen date, so a
    # deal whose 270-day window runs past the panel end is not observed, it is
    # censored -- and censoring here pushes the label to 1, not to 0.
    last_observable = panel_end - dt.timedelta(days=SURVIVE_DAYS
                                               + CENSOR_SLACK_DAYS)
    d3c = d3.filter(pl.col("first_ts").cast(pl.Date) <= last_observable)
    print(f"  completion: censoring cutoff {last_observable}, "
          f"{d3.height - d3c.height} of {d3.height} pairs dropped", flush=True)
    banned3 = {"form", "n_filings", "last_ts", "delisted"}
    r3 = fit_score(d3c, cols, "yv")
    out["deal_completes"] = {
        **(r3 or {}),
        "censor_cutoff": str(last_observable),
        "n_censored_dropped": d3.height - d3c.height,
        "label_positive_rate_all_rows": 100 * float(d3c["yv"].mean()),
        "n_label_classes": int(d3c["yv"].n_unique()),
        **leak_check("completion", cols, banned3)}
    r3u = fit_score(d3, cols, "yv")
    out["deal_completes_uncensored"] = {
        **(r3u or {}),
        "label_positive_rate_all_rows": 100 * float(d3["yv"].mean()),
        "note": "plan's version, untruncated"}

    # 3b. DEVIATION, and the reason for it: on deal_pairs the completion label
    # is a TAUTOLOGY. load_pairs.orient designates as target whichever party
    # stopped filing within 270 days, so every target in the table completed by
    # construction and the negative class does not exist -- broken deals were
    # dropped as "ambiguous" upstream. tender.duckdb is the one place the
    # target is known independently of the outcome: its subject CIKs were
    # parsed from SEC-HEADER blocks, not inferred from who stopped filing. So
    # the question is asked there instead, on the subject's own features.
    tender_cut = last_observable
    tcon = duckdb.connect(":memory:")
    tcon.execute("ATTACH 'data/tender.duckdb' AS t (READ_ONLY)")
    tcon.execute("ATTACH 'data/deal.duckdb' AS m (READ_ONLY)")
    tend = tcon.execute(f"""
        SELECT o.cik AS target_cik, o.public_ts AS first_ts,
               CASE WHEN u.delisted <= o.public_ts
                    + INTERVAL {SURVIVE_DAYS} DAY THEN 1 ELSE 0 END AS yv
        FROM t.tender_offers o JOIN m.universe u USING (cik)
        WHERE o.public_ts <= DATE '{tender_cut}'
    """).pl()
    tcon.close()
    d3t = (tend.with_columns(
        (pl.col("first_ts").cast(pl.Date)
         - pl.duration(weeks=4)).dt.truncate("1w").alias("week"))
        .join(t_side, on=["target_cik", "week"], how="inner"))
    tcols = [c for c in d3t.columns
             if c.startswith("t_") and d3t[c].dtype.is_numeric()]
    r3t = fit_score(d3t, tcols, "yv")
    # What the announcement DATE alone scores on the same test slice. The
    # censoring above was found this way and would have been reported as a
    # 0.94-AUC model otherwise, so the check stays in.
    tte = d3t.filter(pl.col("first_ts") >= CUT)
    date_auc = auc_of(tte["yv"].to_numpy(),
                      tte["first_ts"].cast(pl.Date).to_numpy()
                      .astype("datetime64[D]").astype(float)) \
        if tte.height else 0.5
    out["deal_completes_tender"] = {
        **(r3t or {}),
        "n_offers_observable": tend.height,
        "n_in_panel": d3t.height,
        "censor_cutoff": str(tender_cut),
        "label_positive_rate_all_rows": 100 * float(d3t["yv"].mean()),
        "date_only_auc": max(date_auc, 1 - date_auc),
        **leak_check("completion (tender subjects)", tcols, {"delisted", "yv"})}

    # 4. Lead time, descriptive.
    scores = pl.read_parquet("data/pair_scores.parquet")
    ranked = scores.with_columns(
        pl.col("p_target").rank("ordinal", descending=True)
          .over("week").alias("r"))
    lead, scorable = [], 0
    for row in pairs.iter_rows(named=True):
        window = ranked.filter(
            (pl.col("cik") == row["target_cik"])
            & (pl.col("week") < row["first_ts"])
            & (pl.col("week") >= row["first_ts"] - dt.timedelta(weeks=52)))
        if not window.height:
            continue  # DEVIATION: pair_scores covers 2023-2025 only.
        scorable += 1
        hits = window.filter(pl.col("r") <= 25)
        if hits.height:
            lead.append((row["first_ts"] - hits["week"].max()).days / 7.0)
    out["lead_time_weeks"] = {
        "n_pairs_ever_flagged": len(lead),
        "n_pairs_total": pairs.height,
        "n_pairs_scorable": scorable,
        "flagged_pct": 100.0 * len(lead) / max(pairs.height, 1),
        "flagged_pct_of_scorable": 100.0 * len(lead) / max(scorable, 1),
        "median": float(np.median(lead)) if lead else None,
        "p25": float(np.percentile(lead, 25)) if lead else None,
        "p75": float(np.percentile(lead, 75)) if lead else None,
        "max": float(np.max(lead)) if lead else None,
    }

    print()
    for k, v in out.items():
        if not isinstance(v, dict):
            continue
        if k == "lead_time_weeks":
            print(f"{k:<28} {v}", flush=True)
            continue
        if "accuracy" not in v:
            why = ("single label class -- unmeasurable"
                   if v.get("n_label_classes") == 1
                   or v.get("label_positive_rate_all_rows") in (0.0, 100.0)
                   else "too few rows")
            print(f"{k:<28} NO RESULT: {why} "
                  f"(positive rate "
                  f"{v.get('label_positive_rate_all_rows', float('nan')):.1f}%)",
                  flush=True)
            continue
        flags = []
        if v["underpowered"]:
            flags.append(f"UNDERPOWERED n_test<{MIN_TEST}")
        if v["no_signal"]:
            flags.append("NO SIGNAL (within 2pp of baseline)")
        if v["auc"] > 0.95:
            flags.append("AUC>0.95 -- suspect leakage")
        if v.get("date_only_auc", 0) > v["auc"] - 0.05:
            flags.append(f"date alone scores {v['date_only_auc']:.3f} "
                         "-- reading the calendar")
        print(f"{k:<28} acc {v['accuracy']:>5.1f}%  "
              f"(majority {v['majority_baseline']:.1f}%)  "
              f"AUC {v['auc']:.3f}  n_train {v['n_train']}  "
              f"n_test {v['n_test']}"
              + ("  << " + "; ".join(flags) if flags else ""), flush=True)

    json.dump(out, open(OUT, "w"), indent=2, default=str)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
