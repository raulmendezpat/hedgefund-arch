from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_NUMERIC = [
    "abs_weight",
    "signed_weight",
    "strength",
    "base_weight",
    "p_win",
    "competitive_score",
    "post_ml_score",
    "portfolio_breadth",
    "portfolio_avg_pwin",
    "portfolio_avg_atrp",
    "portfolio_avg_strength",
    "portfolio_conviction",
    "portfolio_regime_scale_applied",
    "execution_target_weight",
    "cluster_target_weight",
    "adx",
    "atrp",
    "bb_width",
    "ml_position_size_mult",
]

FEATURE_CATEGORICAL = [
    "symbol",
    "strategy_id",
    "side",
    "engine",
    "registry_symbol",
    "portfolio_regime",
]

TARGET_WIN = "y_win_3"
TARGET_RET = "ret_net_3"


def _build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, FEATURE_NUMERIC),
            ("cat", cat_pipe, FEATURE_CATEGORICAL),
        ],
        remainder="drop",
    )


def _build_classifier(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", _build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=300,
                    min_samples_leaf=50,
                    l2_regularization=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_regressor(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", _build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=300,
                    min_samples_leaf=50,
                    l2_regularization=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _safe_auc(y_true: pd.Series, y_score) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    if y_true.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_rmse(y_true: pd.Series, y_pred) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0.0).astype(float)
    try:
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return float("nan")


def _period_from_ts(ts_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(pd.to_numeric(ts_series, errors="coerce"), unit="ms", utc=True)
    out = pd.Series(index=ts_series.index, dtype="object")
    out.loc[dt.dt.year == 2024] = "2024"
    out.loc[dt.dt.year == 2025] = "2025"
    out.loc[dt.dt.year == 2026] = "2026_ytd"
    return out.fillna("other")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train selector models by temporal chunks.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--holdout-run", default="replay_bad_30d_regime_guard_v1")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, low_memory=False)

    for c in FEATURE_NUMERIC:
        if c not in df.columns:
            df[c] = 0.0
    for c in FEATURE_CATEGORICAL:
        if c not in df.columns:
            df[c] = "missing"

    df["period_chunk"] = _period_from_ts(df["ts"])
    train_df = df[df["run"].astype(str) != str(args.holdout_run)].copy()
    valid_df = df[df["run"].astype(str) == str(args.holdout_run)].copy()

    if len(train_df) == 0 or len(valid_df) == 0:
        raise SystemExit("Train/holdout quedó vacío.")

    out_base = Path(args.out_dir) / "selector_models_time_v1"
    out_base.mkdir(parents=True, exist_ok=True)

    registry = {
        "feature_numeric": FEATURE_NUMERIC,
        "feature_categorical": FEATURE_CATEGORICAL,
        "target_win": TARGET_WIN,
        "target_ret": TARGET_RET,
        "holdout_run": str(args.holdout_run),
        "periods": {},
    }

    feature_cols = FEATURE_NUMERIC + FEATURE_CATEGORICAL

    print("=== TRAIN PERIOD COUNTS ===")
    print(train_df["period_chunk"].value_counts(dropna=False).to_string())

    for period in ["2024", "2025", "2026_ytd"]:
        tr = train_df[train_df["period_chunk"] == period].copy()
        va = valid_df.copy()

        if len(tr) == 0:
            print(f"{period}: skipped (0 train rows)")
            continue

        y_tr_win = pd.to_numeric(tr[TARGET_WIN], errors="coerce").fillna(0).astype(int)
        if y_tr_win.nunique() < 2:
            print(f"{period}: skipped (single class in y_win_3)")
            continue

        X_tr = tr[feature_cols].copy()
        X_va = va[feature_cols].copy()

        y_va_win = pd.to_numeric(va[TARGET_WIN], errors="coerce").fillna(0).astype(int)
        y_tr_ret = pd.to_numeric(tr[TARGET_RET], errors="coerce").fillna(0.0).astype(float)
        y_va_ret = pd.to_numeric(va[TARGET_RET], errors="coerce").fillna(0.0).astype(float)

        clf = _build_classifier(args.random_state)
        reg = _build_regressor(args.random_state)

        clf.fit(X_tr, y_tr_win)
        reg.fit(X_tr, y_tr_ret)

        p_va = clf.predict_proba(X_va)[:, 1]
        r_va = reg.predict(X_va)

        period_dir = out_base / period
        period_dir.mkdir(parents=True, exist_ok=True)

        clf_path = period_dir / f"{period}_pwin.pkl"
        reg_path = period_dir / f"{period}_retnet.pkl"

        with open(clf_path, "wb") as f:
            pickle.dump(clf, f)
        with open(reg_path, "wb") as f:
            pickle.dump(reg, f)

        registry["periods"][period] = {
            "train_rows": int(len(tr)),
            "valid_rows": int(len(va)),
            "pwin_model_path": str(clf_path),
            "retnet_model_path": str(reg_path),
            "valid_auc_on_holdout": _safe_auc(y_va_win, p_va),
            "valid_rmse_on_holdout": _safe_rmse(y_va_ret, r_va),
        }

        print(
            f"{period}: train_rows={len(tr)} valid_rows={len(va)} "
            f"holdout_auc={registry['periods'][period]['valid_auc_on_holdout']} "
            f"holdout_rmse={registry['periods'][period]['valid_rmse_on_holdout']}"
        )

    registry_path = out_base / "selector_model_registry_time_v1.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)

    print("\nSaved ->", registry_path)


if __name__ == "__main__":
    main()
