"""LightGBM models.

Trees rather than logistic regression because the measured relationships are
non-monotonic: deal rate by market-cap decile peaks in the middle (2.84% at d7)
and falls at both ends (0.54% at d1, 1.03% at d10). A linear coefficient on
size ranks mega-caps top -- the decile least likely to be acquired, which is
exactly why the linear model's top-100 precision was zero.

Two objectives are provided:

  fit()       binary logloss. Calibrated probabilities, interpretable.
  fit_rank()  lambdarank with the WEEK as the query group. Evaluation is
              per-week ranking, so this trains the thing being measured. It
              targets the residual defect the binary model still shows --
              precision at N=10 sitting below N=25, i.e. the few highest picks
              each week being mis-ordered.

is_unbalance is deliberately NOT set. At a 1.4% positive rate the imbalance is
mild, and rescaling would destroy the calibrated probabilities.
"""
import datetime as dt

import lightgbm as lgb
import numpy as np
import polars as pl

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    # A company contributes ~550 near-identical rows, so a leaf can easily
    # memorise one company. A high floor forces leaves to span many firms.
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 2,
}

RANK_PARAMS = {
    **PARAMS,
    "objective": "lambdarank",
    "metric": "map",
    # Only the head of each week's ranking matters -- that is what the screen
    # reads -- so truncate the gradient there instead of over all ~7k rows.
    "lambdarank_truncation_level": 50,
    "eval_at": [25],
}

NUM_ROUNDS = 600
EARLY_STOPPING = 50

# Deal regimes shift with rates and antitrust posture, so older years get less
# weight. Half-life in years; 4 keeps a decade relevant without letting 2016
# count equally with 2025.
RECENCY_HALF_LIFE_YEARS = 4.0


def _cols(cols: list[str] | None) -> list[str]:
    if cols is not None:
        return cols
    from .features import FEATURE_COLS
    return FEATURE_COLS


def recency_weights(df: pl.DataFrame,
                    half_life_years: float = RECENCY_HALF_LIFE_YEARS) -> np.ndarray:
    """Exponential decay in calendar age, normalised to mean 1."""
    latest = df["week"].max()
    age_years = np.array(
        [(latest - w).days / 365.25 for w in df["week"].to_list()])
    w = np.power(0.5, age_years / half_life_years)
    return w / w.mean()


def _dataset(df: pl.DataFrame, cols: list[str],
             weight: np.ndarray | None = None) -> lgb.Dataset:
    return lgb.Dataset(df.select(cols).to_pandas(), label=df["y"].to_pandas(),
                       weight=weight, free_raw_data=False)


def fit(train: pl.DataFrame, valid: pl.DataFrame | None = None,
        cols: list[str] | None = None, recency: bool = False) -> lgb.Booster:
    cols = _cols(cols)
    w = recency_weights(train) if recency else None
    sets = [_dataset(valid, cols)] if valid is not None else None
    callbacks = ([lgb.early_stopping(EARLY_STOPPING, verbose=False)]
                 if valid is not None else None)
    return lgb.train(PARAMS, _dataset(train, cols, w),
                     num_boost_round=NUM_ROUNDS, valid_sets=sets,
                     callbacks=callbacks)


def _rank_dataset(df: pl.DataFrame, cols: list[str]) -> lgb.Dataset:
    """Query groups are weeks, so the objective optimises within-week order.

    lambdarank requires rows contiguous by group, hence the sort.
    """
    d = df.sort("week")
    groups = d.group_by("week", maintain_order=True).len()["len"].to_list()
    ds = lgb.Dataset(d.select(cols).to_pandas(), label=d["y"].to_pandas(),
                     group=groups, free_raw_data=False)
    return ds


def fit_rank(train: pl.DataFrame, valid: pl.DataFrame | None = None,
             cols: list[str] | None = None) -> lgb.Booster:
    cols = _cols(cols)
    sets = [_rank_dataset(valid, cols)] if valid is not None else None
    callbacks = ([lgb.early_stopping(EARLY_STOPPING, verbose=False)]
                 if valid is not None else None)
    return lgb.train(RANK_PARAMS, _rank_dataset(train, cols),
                     num_boost_round=NUM_ROUNDS, valid_sets=sets,
                     callbacks=callbacks)


def predict(booster: lgb.Booster, df: pl.DataFrame,
            cols: list[str] | None = None) -> np.ndarray:
    return np.asarray(booster.predict(df.select(_cols(cols)).to_pandas()))


def fit_horizon_ensemble(train: pl.DataFrame, labels: dict[str, pl.Series],
                         cols: list[str] | None = None) -> dict[str, lgb.Booster]:
    """One booster per horizon label.

    Different features lead at different ranges -- a strategic-alternatives
    disclosure is near-term, a valuation discount is not -- so a short-horizon
    and a long-horizon model disagree usefully and their average ranks better
    than either.
    """
    cols = _cols(cols)
    out = {}
    for name, y in labels.items():
        d = train.with_columns(y.alias("y"))
        out[name] = fit(d, cols=cols)
    return out


def predict_ensemble(boosters: dict[str, lgb.Booster], df: pl.DataFrame,
                     cols: list[str] | None = None) -> np.ndarray:
    """Average of per-horizon rank-normalised scores.

    Raw probabilities are not comparable across horizons (a 52-week model
    predicts a higher base rate than a 13-week one), so each model's scores
    are converted to ranks before averaging.
    """
    ranks = []
    for b in boosters.values():
        p = predict(b, df, cols)
        r = np.argsort(np.argsort(p)) / max(len(p) - 1, 1)
        ranks.append(r)
    return np.mean(ranks, axis=0)
