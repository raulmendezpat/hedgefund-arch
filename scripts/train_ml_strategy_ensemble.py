from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
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
    "symbol",
    "side_raw",
]

META_COLUMNS = [
    "ts",
    "ts_utc",
    "strategy_id",
    "engine",
    "registry_symbol",
    "side_final",
    "selected_by_opportunity_selector",
    "post_ml_score",
    "p_win",
    "label_horizon",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Train per-strategy ensemble ML models")
    ap.add_argument("--input", required=True, help="CSV dataset from ml feature export")
    ap.add_argument("--output-model-registry", required=True, help="Output .joblib registry path")
    ap.add_argument("--output-metrics", required=True, help="Output metrics JSON")
    ap.add_argument("--target-col", default="y_win", help="Target column")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Temporal train fraction")
    ap.add_argument("--min-strategy-rows", type=int, default=80, help="Minimum rows per strategy")
    ap.add_argument("--min-class-count", type=int, default=10, help="Minimum positive and negative count")
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

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    hgb = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("hgb", hgb),
        ],
        voting="soft",
    )

    clf = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", ensemble),
    ])

    return clf


def main():
    args = parse_args()

    in_path = Path(args.input)
    out_model = Path(args.output_model_registry)
    out_metrics = Path(args.output_metrics)

    if not in_path.exists():
        raise SystemExit(f"No existe input dataset: {in_path}")

    df = pd.read_csv(in_path)

    if args.target_col not in df.columns:
        raise SystemExit(f"No existe target column: {args.target_col}")

    if "strategy_id" not in df.columns:
        raise SystemExit("El dataset no contiene strategy_id")

    df = df[df[args.target_col].notna()].copy()
    df = df[df["side_raw"].astype(str) != "flat"].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    feature_cols_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    feature_cols_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    registry = {}
    metrics = {
        "input_rows_total": int(len(df)),
        "target_col": args.target_col,
        "train_frac": float(args.train_frac),
        "strategies": {},
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
    }

    for strategy_id, g in df.groupby("strategy_id", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)

        if len(g) < int(args.min_strategy_rows):
            metrics["strategies"][str(strategy_id)] = {
                "status": "skipped",
                "reason": f"rows<{args.min_strategy_rows}",
                "rows": int(len(g)),
            }
            continue

        y_all = g[args.target_col].astype(int).to_numpy()
        pos = int((y_all == 1).sum())
        neg = int((y_all == 0).sum())

        if pos < int(args.min_class_count) or neg < int(args.min_class_count):
            metrics["strategies"][str(strategy_id)] = {
                "status": "skipped",
                "reason": f"class_count<{args.min_class_count}",
                "rows": int(len(g)),
                "positive_count": pos,
                "negative_count": neg,
            }
            continue

        X = g[feature_cols_num + feature_cols_cat].copy()
        y = y_all

        split_idx = int(len(g) * float(args.train_frac))
        split_idx = max(1, min(split_idx, len(g) - 1))

        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        model = build_model(feature_cols_num, feature_cols_cat)
        model.fit(X_train, y_train)

        p_train = model.predict_proba(X_train)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        pred_train = (p_train >= 0.5).astype(int)
        pred_test = (p_test >= 0.5).astype(int)

        strategy_metrics = {
            "status": "trained",
            "rows": int(len(g)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_count": pos,
            "negative_count": neg,
            "train_positive_rate": float(np.mean(y_train)),
            "test_positive_rate": float(np.mean(y_test)),
            "train_accuracy": float(accuracy_score(y_train, pred_train)),
            "test_accuracy": float(accuracy_score(y_test, pred_test)),
            "train_auc": safe_auc(y_train, p_train),
            "test_auc": safe_auc(y_test, p_test),
            "train_logloss": safe_logloss(y_train, p_train),
            "test_logloss": safe_logloss(y_test, p_test),
        }

        metrics["strategies"][str(strategy_id)] = strategy_metrics

        registry[str(strategy_id)] = {
            "model": model,
            "feature_cols_num": feature_cols_num,
            "feature_cols_cat": feature_cols_cat,
            "target_col": args.target_col,
            "strategy_id": str(strategy_id),
            "metrics": strategy_metrics,
        }

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "type": "strategy_model_registry",
        "models": registry,
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "target_col": args.target_col,
        "train_frac": float(args.train_frac),
        "metrics": metrics,
    }

    joblib.dump(artifact, out_model)

    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model registry -> {out_model}")
    print(f"Saved metrics        -> {out_metrics}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
