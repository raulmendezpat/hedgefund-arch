from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import timedelta

import pandas as pd

import importlib.util

_train_path = Path(__file__).resolve().parent / "train_pwin_ml.py"
_spec = importlib.util.spec_from_file_location("train_pwin_ml_module", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_train_mod)

prepare_dataset = _train_mod.prepare_dataset
train_best_model = _train_mod.train_best_model
DEFAULT_NUMERIC = _train_mod.DEFAULT_NUMERIC
DEFAULT_CATEGORICAL = _train_mod.DEFAULT_CATEGORICAL

WINDOWS = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "12m": 365,
    "24m": 730,
}


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def filter_window(df: pd.DataFrame, ts_col: str, days: int) -> pd.DataFrame:
    x = df.copy()
    x[ts_col] = pd.to_datetime(x[ts_col], utc=True, errors="coerce")
    x = x.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    if x.empty:
        return x
    end_ts = x[ts_col].max()
    start_ts = end_ts - pd.Timedelta(days=int(days))
    return x[x[ts_col] >= start_ts].copy().reset_index(drop=True)


def metric_score(result: dict) -> tuple:
    auc = result.get("roc_auc", None)
    auc = -1.0 if auc is None else float(auc)
    return (
        bool(result.get("recommended_apply", False)),
        auc,
        float(result.get("balanced_accuracy", 0.0) or 0.0),
        float(result.get("accuracy", 0.0) or 0.0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-csv", required=True)
    ap.add_argument("--attribution-csv", required=True)
    ap.add_argument("--horizon", default="24h", choices=["1h", "4h", "12h", "24h", "48h"])
    ap.add_argument("--out-dir", default="artifacts/pwin_ml_multiwindow")
    ap.add_argument("--min-rows-symbol", type=int, default=1500)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--min-accuracy", type=float, default=0.54)
    ap.add_argument("--min-balanced-accuracy", type=float, default=0.53)
    ap.add_argument("--min-auc", type=float, default=0.56)
    args = ap.parse_args()

    candidates_df = load_csv(args.candidates_csv)
    attribution_df = load_csv(args.attribution_csv)

    full_df = prepare_dataset(candidates_df, attribution_df, args.horizon)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_rows = []
    best_registry = {
        "horizon": str(args.horizon),
        "source_candidates_csv": str(args.candidates_csv),
        "source_attribution_csv": str(args.attribution_csv),
        "windows": {},
        "best_by_symbol": {},
    }

    for win_name, days in WINDOWS.items():
        df_win = filter_window(full_df, "ts", days)
        if df_win.empty:
            continue

        win_rows = []
        symbols = sorted(df_win["symbol"].astype(str).unique().tolist())

        for symbol in symbols:
            try:
                bundle, result = train_best_model(
                    df_all=df_win,
                    target_symbol=symbol,
                    numeric_cols=list(DEFAULT_NUMERIC),
                    categorical_cols=list(DEFAULT_CATEGORICAL),
                    min_rows_symbol=int(args.min_rows_symbol),
                    test_frac=float(args.test_frac),
                    min_accuracy=float(args.min_accuracy),
                    min_bal_acc=float(args.min_balanced_accuracy),
                    min_auc=float(args.min_auc),
                )

                safe_symbol = symbol.replace("/", "_").replace(":", "_")
                model_path = str(Path(args.out_dir) / f"pwin_model_{safe_symbol}_{win_name}.joblib")

                import joblib
                joblib.dump(bundle, model_path)

                row = {
                    "window": win_name,
                    "window_days": int(days),
                    **result,
                    "model_path": model_path,
                }
                win_rows.append(row)
                all_rows.append(row)

            except Exception as e:
                all_rows.append({
                    "window": win_name,
                    "window_days": int(days),
                    "target_key": symbol,
                    "error": repr(e),
                })

        win_df = pd.DataFrame(win_rows)
        if not win_df.empty:
            best_registry["windows"][win_name] = {}
            for _, r in win_df.iterrows():
                best_registry["windows"][win_name][r["target_key"]] = {
                    "model_type": r["model_type"],
                    "model_path": r["model_path"],
                    "recommended_apply": bool(r["recommended_apply"]),
                    "accuracy": float(r["accuracy"]),
                    "balanced_accuracy": float(r["balanced_accuracy"]),
                    "roc_auc": None if pd.isna(r["roc_auc"]) else float(r["roc_auc"]),
                    "brier": None if pd.isna(r["brier"]) else float(r["brier"]),
                    "fallback_mode": r["fallback_mode"],
                }

    leaderboard = pd.DataFrame(all_rows)
    leaderboard_csv = str(Path(args.out_dir) / "pwin_ml_multiwindow_leaderboard.csv")
    leaderboard.to_csv(leaderboard_csv, index=False)

    if not leaderboard.empty:
        valid = leaderboard[leaderboard["error"].isna()] if "error" in leaderboard.columns else leaderboard.copy()
        if not valid.empty:
            best_rows = []
            for symbol, g in valid.groupby("target_key", sort=True):
                g = g.copy()
                g["_score"] = g.apply(
                    lambda r: (
                        int(bool(r.get("recommended_apply", False))),
                        -1.0 if pd.isna(r.get("roc_auc")) else float(r.get("roc_auc")),
                        float(r.get("balanced_accuracy", 0.0) or 0.0),
                        float(r.get("accuracy", 0.0) or 0.0),
                    ),
                    axis=1,
                )
                best_idx = max(g.index, key=lambda i: g.loc[i, "_score"])
                r = g.loc[best_idx]
                best_registry["best_by_symbol"][symbol] = {
                    "window": r["window"],
                    "model_type": r["model_type"],
                    "model_path": r["model_path"],
                    "recommended_apply": bool(r["recommended_apply"]),
                    "accuracy": float(r["accuracy"]),
                    "balanced_accuracy": float(r["balanced_accuracy"]),
                    "roc_auc": None if pd.isna(r["roc_auc"]) else float(r["roc_auc"]),
                    "brier": None if pd.isna(r["brier"]) else float(r["brier"]),
                    "fallback_mode": r["fallback_mode"],
                }
                best_rows.append(r)

            best_df = pd.DataFrame(best_rows).sort_values(
                ["recommended_apply", "roc_auc", "balanced_accuracy", "accuracy"],
                ascending=[False, False, False, False],
            )
        else:
            best_df = pd.DataFrame()
    else:
        best_df = pd.DataFrame()

    best_csv = str(Path(args.out_dir) / "pwin_ml_best_by_symbol.csv")
    best_df.to_csv(best_csv, index=False)

    registry_json = str(Path(args.out_dir) / "pwin_ml_multiwindow_registry.json")
    Path(registry_json).write_text(json.dumps(best_registry, indent=2), encoding="utf-8")

    print("saved:", leaderboard_csv)
    print("saved:", best_csv)
    print("saved:", registry_json)

    print("\n=== BEST BY SYMBOL ===")
    if best_df.empty:
        print("(empty)")
    else:
        cols = [c for c in [
            "target_key", "window", "model_type", "accuracy",
            "balanced_accuracy", "roc_auc", "recommended_apply"
        ] if c in best_df.columns]
        print(best_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
