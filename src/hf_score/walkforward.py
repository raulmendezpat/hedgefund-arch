from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

H_MS = 3_600_000


@dataclass(frozen=True)
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end: pd.Timestamp  # exclusive; test_start minus the embargo


def monthly_folds(ts: pd.Series, *, first_test: str, embargo_bars: int) -> list[Fold]:
    """Expanding-window monthly folds. Training rows must have *exited* before train_end:
    the caller purges on exit_ts, the embargo covers label overlap at the boundary."""
    start = pd.Timestamp(first_test, tz="UTC")
    last = ts.max()
    folds = []
    m = start
    while m <= last:
        nxt = m + pd.offsets.MonthBegin(1)
        folds.append(Fold(test_start=m, test_end=nxt, train_end=m - pd.Timedelta(hours=embargo_bars)))
        m = nxt
    return folds


def default_model() -> HistGradientBoostingRegressor:
    # Shallow, heavily regularised: the target is noisy and candidates on consecutive bars overlap.
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=400,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=7,
    )


def run_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    exit_ts: pd.Series,
    *,
    first_test: str,
    embargo_bars: int,
    model_factory: Callable[[], object] = default_model,
    target_clip: float | None = 3.0,
    min_train_rows: int = 5_000,
) -> pd.Series:
    """Out-of-sample predictions for every row whose ts falls in a test fold (NaN elsewhere)."""
    pred = pd.Series(np.nan, index=X.index)
    for f in monthly_folds(ts, first_test=first_test, embargo_bars=embargo_bars):
        test = (ts >= f.test_start) & (ts < f.test_end)
        if not test.any():
            continue
        # Purge: a training label must be fully realized before the embargoed boundary.
        train = (exit_ts < f.train_end) & y.notna()
        if train.sum() < min_train_rows:
            continue
        yt = y[train].clip(-target_clip, target_clip) if target_clip else y[train]
        model = model_factory()
        model.fit(X[train], yt)
        pred[test] = model.predict(X[test])
    return pred


def run_walkforward_grouped(
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    exit_ts: pd.Series,
    groups: pd.Series,
    *,
    min_train_rows: int = 2_000,
    **kwargs,
) -> pd.Series:
    """One independent walk-forward model per group (e.g. per symbol). Groups or folds with
    too little history get no prediction rather than a fallback, so their gaps stay visible."""
    pred = pd.Series(np.nan, index=X.index)
    for _, idx in groups.groupby(groups).groups.items():
        pred[idx] = run_walkforward(
            X.loc[idx], y.loc[idx], ts.loc[idx], exit_ts.loc[idx], min_train_rows=min_train_rows, **kwargs
        )
    return pred


def categorical_model(cat_index: int) -> Callable[[], HistGradientBoostingRegressor]:
    """Global model that can split on a categorical column (integer codes) such as the symbol."""

    def factory() -> HistGradientBoostingRegressor:
        m = default_model()
        m.set_params(categorical_features=[cat_index])
        return m

    return factory


def monthly_ic(df: pd.DataFrame, score: str, outcome: str, *, min_rows: int = 50) -> pd.Series:
    """Spearman IC of score vs outcome per calendar month (months with < min_rows are skipped).
    Always returns a float Series, possibly empty."""
    vals = {}
    for month, g in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        if len(g) > min_rows:
            ic = spearmanr(g[score], g[outcome]).statistic
            if np.isfinite(ic):
                vals[month] = float(ic)
    return pd.Series(vals, dtype=float)


def evaluate_scores(df: pd.DataFrame, score_cols: list[str], *, outcome: str = "net_R", gross: str = "gross_R") -> dict:
    """Per-score OOS diagnostics on rows that have a prediction."""
    out = {}
    month = df["ts"].dt.strftime("%Y-%m")
    for col in score_cols:
        d = df[df[col].notna()]
        m = month[d.index]
        ic_m = monthly_ic(d, col, outcome)
        icg_m = monthly_ic(d, col, gross)
        q = pd.qcut(d[col].rank(method="first"), 5, labels=False)
        quint = d.groupby(q).agg(n=(outcome, "size"), net_R=(outcome, "mean"), gross_R=(gross, "mean"))
        taken = d[d[col] > 0.0]
        monthly_take = taken.groupby(m[taken.index])[outcome].sum() if len(taken) else pd.Series(dtype=float)
        out[col] = {
            "rows": int(len(d)),
            "ic_net_mean": float(ic_m.mean()),
            "ic_net_pos_share": float((ic_m > 0).mean()),
            "ic_gross_mean": float(icg_m.mean()),
            "ic_gross_pos_share": float((icg_m > 0).mean()),
            "quintile_net_R": [round(float(v), 4) for v in quint["net_R"]],
            "quintile_gross_R": [round(float(v), 4) for v in quint["gross_R"]],
            "top_minus_bottom_net_R": float(quint["net_R"].iloc[-1] - quint["net_R"].iloc[0]),
            "take_score_gt_0": {
                "n": int(len(taken)),
                "share": float(len(taken) / max(len(d), 1)),
                "mean_net_R": float(taken[outcome].mean()) if len(taken) else float("nan"),
                "months_positive_share": float((monthly_take > 0).mean()) if len(monthly_take) else float("nan"),
            },
        }
    return out
