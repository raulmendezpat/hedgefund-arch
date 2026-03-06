from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from hf.engines.ml_filter import FEATURE_COLUMNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to exported ML features CSV")
    ap.add_argument("--target", default="y_win_3", choices=["y_win_1", "y_win_3", "y_win_6", "y_win_12"])
    ap.add_argument("--out-model", default="artifacts/ml_baseline.pkl")
    ap.add_argument("--out-metrics", default="artifacts/ml_baseline_metrics.json")
    ap.add_argument("--train-end", default=None, help="Optional max ts for train split (inclusive)")
    ap.add_argument("--symbol", default=None, help="Optional symbol filter, e.g. BTC/USDT:USDT or SOL/USDT:USDT")
    ap.add_argument("--side", default=None, choices=["long", "short"], help="Optional side_raw filter")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV vacío")

    need_cols = ["ts", "symbol", "side_raw", args.target] + FEATURE_COLUMNS
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltan columnas requeridas: {missing}")

    if args.symbol:
        df = df[df["symbol"] == args.symbol].copy()

    if args.side:
        df = df[df["side_raw"] == args.side].copy()

    if df.empty:
        raise SystemExit("El filtro symbol/side dejó el dataset vacío")

    df = df.dropna(subset=[args.target]).copy()
    df[args.target] = df[args.target].astype(int)

    if len(df) < 200:
        raise SystemExit(f"Muy pocas filas para entrenar: {len(df)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[args.target].copy()

    if y.nunique() < 2:
        raise SystemExit("El target tiene una sola clase después de filtrar")

    if args.train_end is not None:
        train_mask = df["ts"].astype("int64") <= int(args.train_end)
        if train_mask.sum() == 0 or (~train_mask).sum() == 0:
            raise SystemExit("Split temporal inválido: train o test quedó vacío")
    else:
        cutoff = int(df["ts"].quantile(0.8))
        train_mask = df["ts"].astype("int64") <= cutoff
        if train_mask.sum() == 0 or (~train_mask).sum() == 0:
            raise SystemExit("No se pudo construir split temporal 80/20")

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[~train_mask]
    y_test = y.loc[~train_mask]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise SystemExit("Train o test quedó con una sola clase")

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    yhat_test = (p_test >= 0.5).astype(int)

    metrics = {
        "target": args.target,
        "symbol_filter": args.symbol,
        "side_filter": args.side,
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "train_auc": float(roc_auc_score(y_train, p_train)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "test_accuracy_0.5": float(accuracy_score(y_test, yhat_test)),
        "feature_columns": list(FEATURE_COLUMNS),
    }

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_model, "wb") as f:
        pickle.dump(model, f)

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("saved model ->", args.out_model)
    print("saved metrics ->", args.out_metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
