from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class PWinModelRegistryV2:
    def __init__(self, registry_path: str):
        self.registry_path = str(registry_path)
        p = Path(self.registry_path)
        if not p.exists():
            raise FileNotFoundError(f"Registry no encontrado: {p}")
        self.registry = json.loads(p.read_text(encoding="utf-8"))
        self._cache: dict[str, object] = {}

    def _load_artifact(self, path: str):
        if path not in self._cache:
            self._cache[path] = joblib.load(path)
        return self._cache[path]

    def _predict_with_artifact(self, artifact, values: dict) -> float | None:
        try:
            num_cols = list(getattr(artifact, "feature_numeric", []) or artifact["feature_numeric"])
            cat_cols = list(getattr(artifact, "feature_categorical", []) or artifact["feature_categorical"])
            model = getattr(artifact, "model", None) or artifact["model"]

            row = {}
            for c in num_cols:
                v = values.get(c, np.nan)
                try:
                    row[c] = float(v) if v is not None else np.nan
                except Exception:
                    row[c] = np.nan

            for c in cat_cols:
                row[c] = "" if values.get(c) is None else str(values.get(c))

            X = pd.DataFrame([row], columns=num_cols + cat_cols)
            p = model.predict_proba(X)[:, 1][0]
            p = float(np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1.0 - 1e-6))
            return p
        except Exception:
            return None

    def predict_from_feature_values(self, values: dict) -> float | None:
        symbol = str(values.get("symbol", "") or "")
        side = str(values.get("side", "") or "").lower()

        key = f"{symbol}|{side}"
        models = dict(self.registry.get("models", {}) or {})

        entry = dict(models.get(key, {}) or {})
        if entry.get("recommended_apply") and entry.get("artifact_path"):
            artifact = self._load_artifact(str(entry["artifact_path"]))
            p = self._predict_with_artifact(artifact, values)
            if p is not None:
                return p

        fallback = dict(self.registry.get("global_fallback", {}) or {})
        if fallback.get("artifact_path"):
            artifact = self._load_artifact(str(fallback["artifact_path"]))
            p = self._predict_with_artifact(artifact, values)
            if p is not None:
                return p

        return None
