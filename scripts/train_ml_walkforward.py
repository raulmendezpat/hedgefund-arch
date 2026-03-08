from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "strength",
    "base_weight",
    "competitive_score",
    "adx",
    "atr",
    "atrp",
    "rsi",
    "ema_gap",
    "ema_gap_pct",
    "bb_width",
    "bb_span",
    "bb_pos",
    "range_expansion",
    "close",
    "volume",
]

CATEGORICAL_FEATURES = [
    "strategy_id",
    "symbol",
    "side_raw",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Train walk-forward global ML classifier")
    ap.add_argument("--input", required=True, help="CSV dataset exported by hf_pipeline_alloc.py --ml-export-features")
    ap.add_argument("--output-model", required=True, help="Path to save final trained .joblib model")
    ap.add_argument("--output-metrics", required=True, help="Path to save walk-forward metrics JSON")
    ap.add_argument("--target-col", default="y_win", help="Target column")
    ap.add_argument("--min-train-rows", type=int, default=1000, help="Minimum rows required")
    ap.add_argument("--n-folds", type=int, default=4, help="Number of walk-forward folds")
    ap.add_argument("--min-train-frac", type=float, default=0.50, help="Minimum train fraction for first fold")
    return ap.parse_args()


def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_logloss(y_true, y_prob):
    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return None


def build_model(feature_cols_num, feature_cols_cat):
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, feature_cols_num),
            ("cat", categorical_pipe, feature_cols_cat),
        ],
        remainder="drop",
    )

    clf = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    return clf


def main():
    args = parse_args()

    in_path = Path(args.input)
    out_model = Path(args.output_model)
    out_metrics = Path(args.output_metrics)

    if not in_path.exists():
        raise SystemExit(f"No existe input dataset: {in_path}")

    df = pd.read_csv(in_path)

    if args.target_col not in df.columns:
        raise SystemExit(f"No existe target column: {args.target_col}")
    if "ts" not in df.columns:
        raise SystemExit("El dataset no contiene columna ts")

    df = df[df[args.target_col].notna()].copy()
    df = df[df["side_raw"].astype(str) != "flat"].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    if len(df) < int(args.min_train_rows):
        raise SystemExit(f"Dataset insuficiente para entrenar: {len(df)} filas")

    feature_cols_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    feature_cols_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X_all = df[feature_cols_num + feature_cols_cat].copy()
    y_all = df[args.target_col].astype(int).to_numpy()

    n = len(df)
    n_folds = max(2, int(args.n_folds))
    min_train_frac = float(args.min_train_frac)

    start_test = int(n * min_train_frac)
    if start_test >= n - n_folds:
        start_test = max(1, n // 2)

    test_block = max(1, (n - start_test) // n_folds)

    folds = []
    oos_parts = []

    for i in range(n_folds):
        train_end = start_test + i * test_block
        test_start = train_end
        test_end = start_test + (i + 1) * test_block if i < n_folds - 1 else n

        if train_end <= 0 or test_start >= n or test_end <= test_start:
            continue

        X_train = X_all.iloc[:train_end].copy()
        y_train = y_all[:train_end]
        X_test = X_all.iloc[test_start:test_end].copy()
        y_test = y_all[test_start:test_end]

        if len(X_train) == 0 or len(X_test) == 0:
            continue
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = build_model(feature_cols_num, feature_cols_cat)
        model.fit(X_train, y_train)

        p_train = model.predict_proba(X_train)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        pred_train = (p_train >= 0.5).astype(int)
        pred_test = (p_test >= 0.5).astype(int)

        fold_metrics = {
            "fold": i + 1,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_start_ts": int(df.iloc[0]["ts"]),
            "train_end_ts": int(df.iloc[train_end - 1]["ts"]),
            "test_start_ts": int(df.iloc[test_start]["ts"]),
            "test_end_ts": int(df.iloc[test_end - 1]["ts"]),
            "train_positive_rate": float(np.mean(y_train)),
            "test_positive_rate": float(np.mean(y_test)),
            "train_accuracy": float(accuracy_score(y_train, pred_train)),
            "test_accuracy": float(accuracy_score(y_test, pred_test)),
            "train_auc": safe_auc(y_train, p_train),
            "test_auc": safe_auc(y_test, p_test),
            "train_logloss": safe_logloss(y_train, p_train),
            "test_logloss": safe_logloss(y_test, p_test),
        }
        folds.append(fold_metrics)

        part = df.iloc[test_start:test_end][["ts", "strategy_id", "symbol", "side_raw"]].copy()
        part["y_true"] = y_test
        part["p_win_oos"] = p_test
        part["fold"] = i + 1
        oos_parts.append(part)

    if not folds:
        raise SystemExit("No se pudieron construir folds walk-forward válidos")

    final_model = build_model(feature_cols_num, feature_cols_cat)
    final_model.fit(X_all, y_all)

    oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    overall_oos_auc = safe_auc(oos_df["y_true"].to_numpy(), oos_df["p_win_oos"].to_numpy()) if not oos_df.empty else None
    overall_oos_logloss = safe_logloss(oos_df["y_true"].to_numpy(), oos_df["p_win_oos"].to_numpy()) if not oos_df.empty else None
    overall_oos_accuracy = float(accuracy_score(oos_df["y_true"].to_numpy(), (oos_df["p_win_oos"].to_numpy() >= 0.5).astype(int))) if not oos_df.empty else None

    metrics = {
        "input_rows_total": int(len(df)),
        "target_col": args.target_col,
        "n_folds": int(n_folds),
        "min_train_frac": float(min_train_frac),
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "overall_positive_rate": float(np.mean(y_all)),
        "walkforward": {
            "oos_rows": int(len(oos_df)),
            "oos_auc": overall_oos_auc,
            "oos_logloss": overall_oos_logloss,
            "oos_accuracy": overall_oos_accuracy,
            "folds": folds,
        },
    }

    artifact = {
        "model": final_model,
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "target_col": args.target_col,
        "walkforward_metrics": metrics,
    }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, out_model)

    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if not oos_df.empty:
        oos_out = out_metrics.with_suffix(".oos.csv")
        oos_df.to_csv(oos_out, index=False)
        print(f"Saved OOS preds -> {oos_out}")

    print(f"Saved model   -> {out_model}")
    print(f"Saved metrics -> {out_metrics}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
