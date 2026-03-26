from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hf_core.pwin_v2.dataset import build_runtime_trade_dataset
from hf_core.pwin_v2.trainer import train_scope_model, save_artifact


def _safe_name(x: str) -> str:
    return (
        str(x)
        .replace("/", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _metrics_row_from_artifact(artifact) -> dict:
    row = dict(getattr(artifact, "metrics", {}) or artifact["metrics"])
    row["model_name"] = getattr(artifact, "model_name", None) or artifact["model_name"]
    row["scope"] = getattr(artifact, "scope", None) or artifact["scope"]
    row["key_value"] = getattr(artifact, "key_value", None) or artifact["key_value"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-rows-scope", type=int, default=120)
    ap.add_argument("--min-auc", type=float, default=0.54)
    ap.add_argument("--min-balanced-accuracy", type=float, default=0.52)
    ap.add_argument("--min-accuracy", type=float, default=0.52)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_df = pd.read_csv(args.candidates)
    trades_df = pd.read_csv(args.trades)

    bundle = build_runtime_trade_dataset(
        candidates_df=candidates_df,
        trades_df=trades_df,
    )
    df = bundle.df.copy()

    summary_rows = []
    leaderboard_frames = []
    prediction_frames = []

    registry = {
        "type": "pwin_v2_registry",
        "source_candidates": str(args.candidates),
        "source_trades": str(args.trades),
        "models": {},
        "global_fallback": {},
    }

    global_artifact, global_lb, global_preds = train_scope_model(
        df_scope=df,
        scope_name="global",
        key_value="global",
        num_cols=bundle.numeric_features,
        cat_cols=bundle.categorical_features,
        min_rows=max(120, int(args.min_rows_scope)),
        min_auc=float(args.min_auc),
        min_bal_acc=float(args.min_balanced_accuracy),
        min_acc=float(args.min_accuracy),
    )
    global_path = out_dir / "pwin_v2_global.joblib"
    save_artifact(str(global_path), global_artifact)

    registry["global_fallback"] = {
        "artifact_path": str(global_path),
        "model_name": global_artifact.model_name,
        "recommended_apply": bool(global_artifact.metrics.get("recommended_apply", False)),
        "roc_auc_valid": global_artifact.metrics.get("roc_auc_valid"),
        "balanced_accuracy_valid": global_artifact.metrics.get("balanced_accuracy_valid"),
        "accuracy_valid": global_artifact.metrics.get("accuracy_valid"),
    }

    summary_rows.append(_metrics_row_from_artifact(global_artifact))
    leaderboard_frames.append(global_lb.assign(scope_key="global"))
    prediction_frames.append(global_preds.assign(scope_key="global"))

    keys = (
        df[["symbol", "side"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(["symbol", "side"])
        .to_dict(orient="records")
    )

    for item in keys:
        symbol = str(item["symbol"])
        side = str(item["side"]).lower()
        if side not in {"long", "short"}:
            continue

        key = f"{symbol}|{side}"
        scope_df = df[df["symbol"].eq(symbol) & df["side"].eq(side)].copy()
        if len(scope_df) < int(args.min_rows_scope):
            registry["models"][key] = {
                "artifact_path": None,
                "model_name": None,
                "recommended_apply": False,
                "skip_reason": f"insufficient_rows:{len(scope_df)}",
                "rows": int(len(scope_df)),
            }
            continue

        artifact, lb, preds = train_scope_model(
            df_scope=scope_df,
            scope_name="symbol_side",
            key_value=key,
            num_cols=bundle.numeric_features,
            cat_cols=bundle.categorical_features,
            min_rows=int(args.min_rows_scope),
            min_auc=float(args.min_auc),
            min_bal_acc=float(args.min_balanced_accuracy),
            min_acc=float(args.min_accuracy),
        )

        art_path = out_dir / f"pwin_v2_{_safe_name(symbol)}_{side}.joblib"
        save_artifact(str(art_path), artifact)

        registry["models"][key] = {
            "artifact_path": str(art_path),
            "model_name": artifact.model_name,
            "recommended_apply": bool(artifact.metrics.get("recommended_apply", False)),
            "roc_auc_valid": artifact.metrics.get("roc_auc_valid"),
            "balanced_accuracy_valid": artifact.metrics.get("balanced_accuracy_valid"),
            "accuracy_valid": artifact.metrics.get("accuracy_valid"),
            "rows_train": artifact.metrics.get("train_rows"),
            "rows_valid": artifact.metrics.get("valid_rows"),
            "rows_test": artifact.metrics.get("test_rows"),
        }

        summary_rows.append(_metrics_row_from_artifact(artifact))
        leaderboard_frames.append(lb.assign(scope_key=key))
        prediction_frames.append(preds.assign(scope_key=key))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["scope", "roc_auc_valid", "balanced_accuracy_valid", "accuracy_valid"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    leaderboard_df = pd.concat(leaderboard_frames, ignore_index=True) if leaderboard_frames else pd.DataFrame()
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    summary_csv = out_dir / "pwin_v2_summary.csv"
    leaderboard_csv = out_dir / "pwin_v2_leaderboard.csv"
    preds_csv = out_dir / "pwin_v2_predictions.csv"
    registry_json = out_dir / "pwin_v2_registry.json"

    summary_df.to_csv(summary_csv, index=False)
    leaderboard_df.to_csv(leaderboard_csv, index=False)
    predictions_df.to_csv(preds_csv, index=False)
    registry_json.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("saved:", summary_csv)
    print("saved:", leaderboard_csv)
    print("saved:", preds_csv)
    print("saved:", registry_json)

    print("\n=== TOP SUMMARY ===")
    cols = [c for c in [
        "scope",
        "key_value",
        "model_name",
        "roc_auc_valid",
        "balanced_accuracy_valid",
        "accuracy_valid",
        "roc_auc_test",
        "balanced_accuracy_test",
        "accuracy_test",
        "recommended_apply",
    ] if c in summary_df.columns]
    print(summary_df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
