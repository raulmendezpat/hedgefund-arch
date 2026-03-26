from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


LEADERBOARD_CSV = "artifacts/pwin_ml_multiwindow_v1/pwin_ml_multiwindow_leaderboard.csv"
FALLBACK_JSON = "artifacts/symbol_similarity/symbol_fallback_map.json"
OUT_JSON = "artifacts/pwin_ml_multiwindow_v1/pwin_ml_operational_registry.json"


def compute_model_score(row: pd.Series) -> float:
    auc = float(row.get("roc_auc", 0.0) or 0.0)
    bacc = float(row.get("balanced_accuracy", 0.0) or 0.0)
    acc = float(row.get("accuracy", 0.0) or 0.0)
    return (0.50 * auc) + (0.30 * bacc) + (0.20 * acc)


def main() -> None:
    df = pd.read_csv(LEADERBOARD_CSV)

    if "error" in df.columns:
        df = df[df["error"].isna()].copy()

    df["recommended_apply"] = df["recommended_apply"].fillna(False).astype(bool)
    df["roc_auc"] = pd.to_numeric(df["roc_auc"], errors="coerce")
    df["balanced_accuracy"] = pd.to_numeric(df["balanced_accuracy"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["model_score"] = df.apply(compute_model_score, axis=1)

    fallback_map = {}
    p_fallback = Path(FALLBACK_JSON)
    if p_fallback.exists():
        fallback_map = json.loads(p_fallback.read_text(encoding="utf-8"))

    registry = {
        "version": 1,
        "selection_rule": {
            "max_windows_per_symbol": 3,
            "score_formula": {
                "roc_auc": 0.50,
                "balanced_accuracy": 0.30,
                "accuracy": 0.20,
            },
            "only_recommended_apply": True,
        },
        "symbols": {},
    }

    best_rows = []

    for symbol, g in df.groupby("target_key", sort=True):
        g = g.copy()
        g = g[g["recommended_apply"].eq(True)].copy()

        if g.empty:
            fb = fallback_map.get(symbol, {})
            registry["symbols"][symbol] = {
                "mode": "fallback_only",
                "windows": [],
                "fallback_symbol": fb.get("fallback_symbol"),
                "fallback_similarity_score": fb.get("similarity_score"),
                "transfer_scale": 0.85 if fb.get("similarity_score", 0.0) and float(fb.get("similarity_score", 0.0)) >= 0.70 else 0.0,
            }
            continue

        g = g.sort_values(["model_score", "roc_auc", "balanced_accuracy", "accuracy"], ascending=[False, False, False, False]).head(3).copy()

        total = float(g["model_score"].sum())
        if total <= 0.0:
            g["weight"] = 1.0 / float(len(g))
        else:
            g["weight"] = g["model_score"] / total

        windows = []
        for _, r in g.iterrows():
            windows.append({
                "window": str(r["window"]),
                "window_days": int(r["window_days"]),
                "model_type": str(r["model_type"]),
                "model_path": str(r["model_path"]),
                "weight": float(r["weight"]),
                "accuracy": float(r["accuracy"]),
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "roc_auc": None if pd.isna(r["roc_auc"]) else float(r["roc_auc"]),
                "brier": None if pd.isna(r["brier"]) else float(r["brier"]),
                "fallback_mode": str(r["fallback_mode"]),
            })

        fb = fallback_map.get(symbol, {})
        registry["symbols"][symbol] = {
            "mode": "multiwindow_blend",
            "windows": windows,
            "fallback_symbol": fb.get("fallback_symbol"),
            "fallback_similarity_score": fb.get("similarity_score"),
            "transfer_scale": 0.85 if fb.get("similarity_score", 0.0) and float(fb.get("similarity_score", 0.0)) >= 0.70 else 0.0,
        }

        best_rows.append({
            "symbol": symbol,
            "n_windows": len(windows),
            "fallback_symbol": fb.get("fallback_symbol"),
            "fallback_similarity_score": fb.get("similarity_score"),
            "windows": " | ".join([f'{w["window"]}:{w["model_type"]}:{w["weight"]:.3f}' for w in windows]),
        })

    p_out = Path(OUT_JSON)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("saved:", OUT_JSON)
    print("\n=== OPERATIONAL REGISTRY SUMMARY ===")
    if best_rows:
        print(pd.DataFrame(best_rows).sort_values("symbol").to_string(index=False))
    else:
        print("(empty)")


if __name__ == "__main__":
    main()
