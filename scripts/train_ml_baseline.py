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
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
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

META_COLUMNS = [
    "ts",
    "ts_utc",
    "engine",
    "registry_symbol",
    "side_final",
    "selected_by_opportunity_selector",
    "post_ml_score",
    "p_win",
    "label_horizon",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train baseline ML classifier for opportunity win probability")
    ap.add_argument("--input", required=True, help="CSV dataset exported by hf_pipeline_alloc.py --ml-export-features")
    ap.add_argument("--output-model", required=True, help="Path to save trained .joblib model")
    ap.add_argument("--output-metrics", required=True, help="Path to save training metrics JSON")
    ap.add_argument("--target-col", default="y_win", help="Target column")
    ap.add_argument("--min-train-rows", type=int, default=200, help="Minimum train rows required")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Temporal train fraction")
    return ap.parse_args()


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_model = Path(args.output_model)
    out_metrics = Path(args.output_metrics)

    if not in_path.exists():
        raise SystemExit(f"No existe input dataset: {in_path}")

    df = pd.read_csv(in_path)

    if args.target_col not in df.columns:
        raise SystemExit(f"No existe target column: {args.target_col}")

    required = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES + [args.target_col, "ts"]) if c in df.columns]
    if "ts" not in required:
        raise SystemExit("El dataset no contiene columna ts")

    keep_cols = list(dict.fromkeys(required + [c for c in META_COLUMNS if c in df.columns]))
    df = df[keep_cols].copy()

    df = df[df[args.target_col].notna()].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    if len(df) < args.min_train_rows:
        raise SystemExit(f"Dataset insuficiente para entrenar: {len(df)} filas")

    feature_cols_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    feature_cols_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    X = df[feature_cols_num + feature_cols_cat].copy()
    y = df[args.target_col].astype(int).to_numpy()

    split_idx = int(len(df) * float(args.train_frac))
    split_idx = max(1, min(split_idx, len(df) - 1))

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y[:split_idx]
    y_test = y[split_idx:]

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

    clf.fit(X_train, y_train)

    p_train = clf.predict_proba(X_train)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    pred_train = (p_train >= 0.5).astype(int)
    pred_test = (p_test >= 0.5).astype(int)

    metrics = {
        "input_rows_total": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_col": args.target_col,
        "numeric_features": feature_cols_num,
        "categorical_features": feature_cols_cat,
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "train_auc": safe_auc(y_train, p_train),
        "test_auc": safe_auc(y_test, p_test),
        "train_logloss": safe_logloss(y_train, p_train),
        "test_logloss": safe_logloss(y_test, p_test),
    }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": clf,
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "target_col": args.target_col,
        "train_frac": float(args.train_frac),
        "metrics": metrics,
    }

    joblib.dump(artifact, out_model)

    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model   -> {out_model}")
    print(f"Saved metrics -> {out_metrics}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
