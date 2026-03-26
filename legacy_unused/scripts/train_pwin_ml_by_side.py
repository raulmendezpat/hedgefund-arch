from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from scripts.train_pwin_ml import (
    DEFAULT_CATEGORICAL,
    DEFAULT_NUMERIC,
    prepare_dataset,
    train_best_model,
)


def _safe_symbol_name(symbol: str) -> str:
    return (
        str(symbol)
        .replace("/", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace("-", "_")
    )


def _is_model_obj(x) -> bool:
    return hasattr(x, "predict_proba") or hasattr(x, "predict")


def _flatten_summary(summary: dict) -> dict:
    out = dict(summary or {})
    metrics = out.get("metrics")
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if k not in out:
                out[k] = v
    return out


def _extract_model_and_summary(raw_result):
    """
    Casos observados/relevantes:
    - tuple(dict_with_model, ...)
    - tuple(model_obj, dict_summary)
    - dict_with_model
    """
    model_obj = None
    summary = {}

    if raw_result is None:
        return model_obj, summary

    if isinstance(raw_result, dict):
        summary = dict(raw_result)
        maybe_model = summary.get("model")
        if _is_model_obj(maybe_model):
            model_obj = maybe_model
        return model_obj, summary

    if isinstance(raw_result, (tuple, list)):
        for item in raw_result:
            if _is_model_obj(item) and model_obj is None:
                model_obj = item
                continue

            if isinstance(item, dict):
                d = dict(item)
                maybe_model = d.get("model")
                if model_obj is None and _is_model_obj(maybe_model):
                    model_obj = maybe_model
                # quedarse con el dict más informativo
                if len(d) > len(summary):
                    summary = d

        return model_obj, summary

    return model_obj, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-csv", required=True)
    ap.add_argument("--attribution-csv", required=True)
    ap.add_argument("--horizon", default="24h")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-rows-symbol-side", type=int, default=800)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--min-accuracy", type=float, default=0.54)
    ap.add_argument("--min-balanced-accuracy", type=float, default=0.53)
    ap.add_argument("--min-auc", type=float, default=0.56)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("DEBUG loading candidates:", args.candidates_csv)
    print("DEBUG loading attribution:", args.attribution_csv)

    candidates_df = pd.read_csv(str(args.candidates_csv))
    attribution_df = pd.read_csv(str(args.attribution_csv))

    print("DEBUG candidates_rows:", len(candidates_df))
    print("DEBUG attribution_rows:", len(attribution_df))

    df = prepare_dataset(
        candidates_df=candidates_df,
        attribution_df=attribution_df,
        horizon=str(args.horizon),
    ).copy()

    print("DEBUG prepared_rows:", len(df))
    print("DEBUG prepared_cols_sample:", list(df.columns)[:20])

    if "side" not in df.columns:
        raise ValueError("Prepared dataset does not contain 'side'")

    df["side"] = df["side"].astype(str).str.lower().str.strip()
    df = df[df["side"].isin(["long", "short"])].copy()

    print("DEBUG prepared_rows_after_side_filter:", len(df))
    print("DEBUG side_counts:")
    print(df["side"].value_counts(dropna=False).to_string())

    summary_rows = []
    registry = {
        "horizon": str(args.horizon),
        "candidates_csv": str(args.candidates_csv),
        "attribution_csv": str(args.attribution_csv),
        "models": {},
    }

    for side in ["long", "short"]:
        df_side = df[df["side"].eq(side)].copy()
        print(f"\nDEBUG side={side} rows={len(df_side)}")
        if df_side.empty:
            continue

        for symbol in sorted(df_side["symbol"].dropna().astype(str).unique()):
            g = df_side[df_side["symbol"].eq(symbol)].copy()
            print(f"DEBUG symbol={symbol} side={side} rows={len(g)}")

            if len(g) < int(args.min_rows_symbol_side):
                print(f"DEBUG skip_insufficient_rows symbol={symbol} side={side}")
                continue

            raw_result = train_best_model(
                df_all=df_side,
                target_symbol=str(symbol),
                numeric_cols=list(DEFAULT_NUMERIC),
                categorical_cols=list(DEFAULT_CATEGORICAL),
                min_rows_symbol=int(args.min_rows_symbol_side),
                test_frac=float(args.test_frac),
                min_accuracy=float(args.min_accuracy),
                min_bal_acc=float(args.min_balanced_accuracy),
                min_auc=float(args.min_auc),
            )

            print("DEBUG raw_result_type:", type(raw_result).__name__)

            model_obj, summary = _extract_model_and_summary(raw_result)
            summary = _flatten_summary(summary)

            target_key = f"{symbol}|{side}"
            safe_name = _safe_symbol_name(symbol)
            model_path = out_dir / f"pwin_model_{safe_name}_{side}.joblib"

            if model_obj is not None:
                joblib.dump(model_obj, model_path)
                print("DEBUG saved_model:", model_path)
            else:
                print("DEBUG no_model_obj_for:", target_key)

            # quitar objetos no serializables del summary
            summary.pop("model", None)

            summary["target_key"] = target_key
            summary["target_symbol"] = str(symbol)
            summary["side"] = side
            summary["model_path"] = str(model_path) if model_obj is not None else None

            registry["models"][target_key] = {
                "path": str(model_path) if model_obj is not None else None,
                "target_key": target_key,
                "target_symbol": str(symbol),
                "side": side,
                "model_type": summary.get("model_type"),
                "accuracy": summary.get("accuracy"),
                "balanced_accuracy": summary.get("balanced_accuracy"),
                "roc_auc": summary.get("roc_auc"),
                "brier": summary.get("brier"),
                "recommended_apply": summary.get("recommended_apply", False),
                "fallback_mode": summary.get("fallback_mode", "symbol"),
                "feature_numeric": summary.get("feature_numeric"),
                "feature_categorical": summary.get("feature_categorical"),
            }

            summary_rows.append(summary)

            print(
                "TRAINED:",
                target_key,
                "model_type=", summary.get("model_type"),
                "auc=", summary.get("roc_auc"),
                "recommended_apply=", summary.get("recommended_apply"),
                "path=", str(model_path) if model_obj is not None else None,
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "pwin_ml_by_side_summary.csv"
    registry_json = out_dir / "pwin_ml_by_side_registry.json"

    summary_df.to_csv(summary_csv, index=False)
    registry_json.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("saved:", summary_csv)
    print("saved:", registry_json)

    print("\n=== SUMMARY ===")
    if summary_df.empty:
        print("(empty)")
    else:
        cols = [c for c in [
            "target_key",
            "side",
            "model_type",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "brier",
            "recommended_apply",
            "fallback_mode",
            "model_path",
        ] if c in summary_df.columns]

        sort_cols = [c for c in ["recommended_apply", "roc_auc", "balanced_accuracy", "accuracy"] if c in summary_df.columns]
        if sort_cols:
            print(summary_df[cols].sort_values(sort_cols, ascending=[False] * len(sort_cols)).to_string(index=False))
        else:
            print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
