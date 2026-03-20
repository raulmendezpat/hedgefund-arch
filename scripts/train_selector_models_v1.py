from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
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
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, FEATURE_NUMERIC),
            ("cat", categorical_pipe, FEATURE_CATEGORICAL),
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


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    if y_true.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0.0).astype(float)
    try:
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return float("nan")


def _make_dirs(base_dir: Path) -> dict[str, Path]:
    model_dir = base_dir / "selector_models_v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": model_dir,
        "global": model_dir / "global",
        "local": model_dir / "local",
    }


def _ensure_dirs(dirs: dict[str, Path]) -> None:
    dirs["global"].mkdir(parents=True, exist_ok=True)
    dirs["local"].mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train selector models v1 (global + local experts).")
    ap.add_argument("--dataset", required=True, help="Path to selector_training_dataset_v1.csv")
    ap.add_argument("--out-dir", default="artifacts", help="Artifacts output base dir")
    ap.add_argument("--min-local-rows", type=int, default=600, help="Minimum rows to train local expert")
    ap.add_argument("--train-end-ts", type=int, default=None, help="Optional train cutoff timestamp in ms")
    ap.add_argument("--valid-start-ts", type=int, default=None, help="Optional validation start timestamp in ms")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, low_memory=False)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")

    for c in FEATURE_NUMERIC:
        if c not in df.columns:
            df[c] = 0.0
    for c in FEATURE_CATEGORICAL:
        if c not in df.columns:
            df[c] = "missing"

    if args.train_end_ts is not None:
        train_df = df[df["ts"] <= int(args.train_end_ts)].copy()
    else:
        q = df["ts"].dropna().quantile(0.80)
        train_df = df[df["ts"] <= q].copy()

    if args.valid_start_ts is not None:
        valid_df = df[df["ts"] >= int(args.valid_start_ts)].copy()
    else:
        q = df["ts"].dropna().quantile(0.80)
        valid_df = df[df["ts"] > q].copy()

    if len(train_df) == 0 or len(valid_df) == 0:
        raise SystemExit("Train/valid split quedó vacío. Ajusta timestamps o dataset.")

    feature_cols = FEATURE_NUMERIC + FEATURE_CATEGORICAL
    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()

    y_train_win = pd.to_numeric(train_df[TARGET_WIN], errors="coerce").fillna(0).astype(int)
    y_valid_win = pd.to_numeric(valid_df[TARGET_WIN], errors="coerce").fillna(0).astype(int)

    y_train_ret = pd.to_numeric(train_df[TARGET_RET], errors="coerce").fillna(0.0).astype(float)
    y_valid_ret = pd.to_numeric(valid_df[TARGET_RET], errors="coerce").fillna(0.0).astype(float)

    dirs = _make_dirs(Path(args.out_dir))
    _ensure_dirs(dirs)

    registry: dict[str, object] = {
        "feature_numeric": FEATURE_NUMERIC,
        "feature_categorical": FEATURE_CATEGORICAL,
        "target_win": TARGET_WIN,
        "target_ret": TARGET_RET,
        "min_local_rows": int(args.min_local_rows),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "models": {
            "global": {},
            "local_by_symbol": {},
        },
    }

    # Global models
    global_clf = _build_classifier(args.random_state)
    global_reg = _build_regressor(args.random_state)

    global_clf.fit(X_train, y_train_win)
    global_reg.fit(X_train, y_train_ret)

    p_valid_global = global_clf.predict_proba(X_valid)[:, 1]
    r_valid_global = global_reg.predict(X_valid)

    global_auc = _safe_auc(y_valid_win, p_valid_global)
    global_rmse = _safe_rmse(y_valid_ret, r_valid_global)

    global_clf_path = dirs["global"] / "global_pwin_model.pkl"
    global_reg_path = dirs["global"] / "global_retnet_model.pkl"

    with open(global_clf_path, "wb") as f:
        pickle.dump(global_clf, f)
    with open(global_reg_path, "wb") as f:
        pickle.dump(global_reg, f)

    registry["models"]["global"] = {
        "pwin_model_path": str(global_clf_path),
        "retnet_model_path": str(global_reg_path),
        "valid_auc": global_auc,
        "valid_rmse": global_rmse,
    }

    print("=== GLOBAL MODELS ===")
    print("train_rows:", len(train_df))
    print("valid_rows:", len(valid_df))
    print("valid_auc:", global_auc)
    print("valid_rmse:", global_rmse)

    # Local experts by symbol
    print("\n=== LOCAL EXPERTS ===")
    symbol_counts = train_df.groupby("symbol").size().sort_values(ascending=False)

    for symbol, n_rows in symbol_counts.items():
        if int(n_rows) < int(args.min_local_rows):
            print(f"{symbol}: skipped (rows={int(n_rows)} < min_local_rows={args.min_local_rows})")
            continue

        tr_sym = train_df[train_df["symbol"] == symbol].copy()
        va_sym = valid_df[valid_df["symbol"] == symbol].copy()

        if len(tr_sym) < int(args.min_local_rows):
            print(f"{symbol}: skipped after filter")
            continue
        if len(va_sym) == 0:
            print(f"{symbol}: skipped (no validation rows)")
            continue
        if pd.to_numeric(tr_sym[TARGET_WIN], errors='coerce').fillna(0).astype(int).nunique() < 2:
            print(f"{symbol}: skipped (train target_win single class)")
            continue

        X_tr = tr_sym[feature_cols].copy()
        X_va = va_sym[feature_cols].copy()

        y_tr_win = pd.to_numeric(tr_sym[TARGET_WIN], errors="coerce").fillna(0).astype(int)
        y_va_win = pd.to_numeric(va_sym[TARGET_WIN], errors="coerce").fillna(0).astype(int)

        y_tr_ret = pd.to_numeric(tr_sym[TARGET_RET], errors="coerce").fillna(0.0).astype(float)
        y_va_ret = pd.to_numeric(va_sym[TARGET_RET], errors="coerce").fillna(0.0).astype(float)

        local_clf = _build_classifier(args.random_state)
        local_reg = _build_regressor(args.random_state)

        local_clf.fit(X_tr, y_tr_win)
        local_reg.fit(X_tr, y_tr_ret)

        p_va = local_clf.predict_proba(X_va)[:, 1]
        r_va = local_reg.predict(X_va)

        auc = _safe_auc(y_va_win, p_va)
        rmse = _safe_rmse(y_va_ret, r_va)

        sym_slug = str(symbol).lower()
        clf_path = dirs["local"] / f"{sym_slug}_pwin_model.pkl"
        reg_path = dirs["local"] / f"{sym_slug}_retnet_model.pkl"

        with open(clf_path, "wb") as f:
            pickle.dump(local_clf, f)
        with open(reg_path, "wb") as f:
            pickle.dump(local_reg, f)

        registry["models"]["local_by_symbol"][symbol] = {
            "train_rows": int(len(tr_sym)),
            "valid_rows": int(len(va_sym)),
            "pwin_model_path": str(clf_path),
            "retnet_model_path": str(reg_path),
            "valid_auc": auc,
            "valid_rmse": rmse,
        }

        print(
            f"{symbol}: train_rows={len(tr_sym)} valid_rows={len(va_sym)} "
            f"valid_auc={auc} valid_rmse={rmse}"
        )

    registry_path = dirs["base"] / "selector_model_registry_v1.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)

    print("\nSaved registry ->", registry_path)


if __name__ == "__main__":
    main()
