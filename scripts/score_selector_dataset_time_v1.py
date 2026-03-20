from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Score dataset with temporal ensemble selector.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", default="results/selector_scored_time_v1.csv")
    ap.add_argument("--w-2024", type=float, default=0.15)
    ap.add_argument("--w-2025", type=float, default=0.35)
    ap.add_argument("--w-2026", type=float, default=0.50)
    ap.add_argument("--pwin-min", type=float, default=0.52)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, low_memory=False)
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))

    feature_cols = registry["feature_numeric"] + registry["feature_categorical"]
    weights = {
        "2024": float(args.w_2024),
        "2025": float(args.w_2025),
        "2026_ytd": float(args.w_2026),
    }

    pred_p_cols = []
    pred_r_cols = []

    for period in ["2024", "2025", "2026_ytd"]:
        meta = registry["periods"].get(period)
        if not meta:
            continue

        with open(meta["pwin_model_path"], "rb") as f:
            p_model = pickle.load(f)
        with open(meta["retnet_model_path"], "rb") as f:
            r_model = pickle.load(f)

        p_col = f"pred_pwin_{period}"
        r_col = f"pred_retnet_{period}"

        X = df[feature_cols].copy()
        df[p_col] = p_model.predict_proba(X)[:, 1]
        df[r_col] = r_model.predict(X)

        pred_p_cols.append((p_col, weights[period]))
        pred_r_cols.append((r_col, weights[period]))

    if not pred_p_cols or not pred_r_cols:
        raise SystemExit("No se cargaron modelos temporales.")

    w_sum = sum(w for _, w in pred_p_cols)
    df["pwin_final"] = 0.0
    df["retnet_final"] = 0.0

    for col, w in pred_p_cols:
        df["pwin_final"] += (w / w_sum) * pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col, w in pred_r_cols:
        df["retnet_final"] += (w / w_sum) * pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["selector_score"] = df["pwin_final"] * df["retnet_final"].clip(lower=0.0)
    df["selector_accept"] = (
        (pd.to_numeric(df["retnet_final"], errors="coerce").fillna(0.0) > 0.0)
        & (pd.to_numeric(df["pwin_final"], errors="coerce").fillna(0.0) >= float(args.pwin_min))
    ).astype(int)

    print("=== ENSEMBLE DIAGNOSTICS ===")
    print("ensemble_auc:", _safe_auc(df["y_win_3"], df["pwin_final"].values))
    print("ensemble_rmse:", _safe_rmse(df["ret_net_3"], df["retnet_final"].values))
    print("accept_rate:", float(df["selector_accept"].mean()))
    print("win_rate_all:", float(df["y_win_3"].mean()))
    print("win_rate_accept:", float(df.loc[df["selector_accept"] == 1, "y_win_3"].mean()))
    print("ret_net_3_all:", float(df["ret_net_3"].mean()))
    print("ret_net_3_accept:", float(df.loc[df["selector_accept"] == 1, "ret_net_3"].mean()))

    print("\n=== BY RUN ===")
    g = df.groupby("run").agg(
        rows=("run", "size"),
        accept_rate=("selector_accept", "mean"),
        win_rate_all=("y_win_3", "mean"),
        win_rate_accept=("y_win_3", lambda s: float(s[df.loc[s.index, 'selector_accept'] == 1].mean()) if (df.loc[s.index, 'selector_accept'] == 1).any() else float("nan")),
        ret_net_all=("ret_net_3", "mean"),
        ret_net_accept=("ret_net_3", lambda s: float(s[df.loc[s.index, 'selector_accept'] == 1].mean()) if (df.loc[s.index, 'selector_accept'] == 1).any() else float("nan")),
    ).reset_index()
    print(g.to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("\nSaved ->", out_path)


if __name__ == "__main__":
    main()
