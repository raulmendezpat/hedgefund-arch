from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error


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


def _local_weight(n_rows: int, k: float = 800.0, max_w: float = 0.70) -> float:
    w = float(n_rows) / float(n_rows + k)
    return max(0.0, min(float(max_w), w))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score selector dataset using global + local ensemble.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", default="results/selector_scored_dataset_v1.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, low_memory=False)
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))

    feature_cols = registry["feature_numeric"] + registry["feature_categorical"]

    with open(registry["models"]["global"]["pwin_model_path"], "rb") as f:
        global_pwin = pickle.load(f)
    with open(registry["models"]["global"]["retnet_model_path"], "rb") as f:
        global_retnet = pickle.load(f)

    X = df[feature_cols].copy()

    df["pred_global_pwin"] = global_pwin.predict_proba(X)[:, 1]
    df["pred_global_retnet"] = global_retnet.predict(X)

    df["pred_local_pwin"] = np.nan
    df["pred_local_retnet"] = np.nan
    df["local_weight"] = 0.0

    local_by_symbol = registry["models"]["local_by_symbol"]

    for symbol, meta in local_by_symbol.items():
        mask = df["symbol"].astype(str) == str(symbol)
        if not mask.any():
            continue

        with open(meta["pwin_model_path"], "rb") as f:
            local_pwin = pickle.load(f)
        with open(meta["retnet_model_path"], "rb") as f:
            local_retnet = pickle.load(f)

        X_sym = df.loc[mask, feature_cols].copy()

        df.loc[mask, "pred_local_pwin"] = local_pwin.predict_proba(X_sym)[:, 1]
        df.loc[mask, "pred_local_retnet"] = local_retnet.predict(X_sym)

        w = _local_weight(int(meta.get("train_rows", 0)))
        df.loc[mask, "local_weight"] = float(w)

    df["pred_local_pwin"] = pd.to_numeric(df["pred_local_pwin"], errors="coerce")
    df["pred_local_retnet"] = pd.to_numeric(df["pred_local_retnet"], errors="coerce")

    df["pwin_final"] = (
        (1.0 - df["local_weight"]) * df["pred_global_pwin"]
        + df["local_weight"] * df["pred_local_pwin"].fillna(df["pred_global_pwin"])
    )

    df["retnet_final"] = (
        (1.0 - df["local_weight"]) * df["pred_global_retnet"]
        + df["local_weight"] * df["pred_local_retnet"].fillna(df["pred_global_retnet"])
    )

    df["selector_score"] = df["pwin_final"] * np.maximum(df["retnet_final"], 0.0)
    df["selector_accept"] = (
        (pd.to_numeric(df["retnet_final"], errors="coerce").fillna(0.0) > 0.0)
        & (pd.to_numeric(df["pwin_final"], errors="coerce").fillna(0.0) >= 0.50)
    ).astype(int)

    # Quick diagnostics
    print("=== GLOBAL DIAGNOSTICS ===")
    print("global_auc:", _safe_auc(df["y_win_3"], df["pred_global_pwin"].values))
    print("global_rmse:", _safe_rmse(df["ret_net_3"], df["pred_global_retnet"].values))

    print("\n=== ENSEMBLE DIAGNOSTICS ===")
    print("ensemble_auc:", _safe_auc(df["y_win_3"], df["pwin_final"].values))
    print("ensemble_rmse:", _safe_rmse(df["ret_net_3"], df["retnet_final"].values))

    print("\nselector_accept rate:", float(df["selector_accept"].mean()))
    print("\nselector_accept by symbol x side:")
    print(
        df.groupby(["symbol", "side"])["selector_accept"]
        .mean()
        .sort_values()
        .to_string()
    )

    print("\nmean realized ret_net_3 by selector_accept:")
    print(df.groupby("selector_accept")["ret_net_3"].mean().to_string())

    print("\ny_win_3 by selector_accept:")
    print(df.groupby("selector_accept")["y_win_3"].mean().to_string())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("\nSaved ->", out_path)


if __name__ == "__main__":
    main()
