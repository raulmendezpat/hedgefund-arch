from __future__ import annotations

import json
from pathlib import Path


CANDIDATE_PATH_KEYS = [
    "path",
    "model_path",
    "artifact_path",
    "joblib_path",
]


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return bool(x)


def _extract_metrics(entry: dict) -> dict:
    metrics = {}
    for k in ["accuracy", "balanced_accuracy", "roc_auc", "brier", "recommended_apply", "fallback_mode", "model_type"]:
        if k in entry:
            metrics[k] = entry.get(k)
    return metrics


def _extract_path(entry: dict) -> str | None:
    for k in CANDIDATE_PATH_KEYS:
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main() -> None:
    src = Path("artifacts/pwin_ml_by_side_v1/pwin_ml_by_side_registry.json")
    out = Path("artifacts/pwin_ml_by_side_v1/pwin_ml_by_side_operational_registry.json")

    if not src.exists():
        raise SystemExit(f"Missing registry: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    models = dict(data.get("models", {}) or {})

    out_models = {}

    for target_key, entry in sorted(models.items()):
        entry = dict(entry or {})
        path = _extract_path(entry)
        rec = _as_bool(entry.get("recommended_apply", False))

        # Si recommended_apply no viene en el entry, intentamos inferir desde all_results/metrics ya normalizadas
        if not rec and "metrics" in entry and isinstance(entry["metrics"], dict):
            rec = _as_bool(entry["metrics"].get("recommended_apply", False))

        if not rec:
            continue
        if not path:
            print(f"SKIP no_path: {target_key}")
            continue

        out_models[target_key] = {
            "path": path,
            "recommended_apply": True,
            "model_type": entry.get("model_type"),
            "accuracy": entry.get("accuracy"),
            "balanced_accuracy": entry.get("balanced_accuracy"),
            "roc_auc": entry.get("roc_auc"),
            "brier": entry.get("brier"),
            "fallback_mode": entry.get("fallback_mode", "by_side"),
        }

    out_obj = {
        "source_registry": str(src),
        "models": out_models,
    }

    out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    print("saved:", out)
    print("\n=== OPERATIONAL BY SIDE ===")
    if not out_models:
        print("(empty)")
    else:
        for k, v in out_models.items():
            print(
                k,
                "model_type=", v.get("model_type"),
                "auc=", v.get("roc_auc"),
                "path=", v.get("path"),
            )


if __name__ == "__main__":
    main()
