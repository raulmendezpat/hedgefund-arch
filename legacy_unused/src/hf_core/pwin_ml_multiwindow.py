from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class PWinMLMultiWindow:
    def __init__(self, registry_path: str):
        self.registry_path = str(registry_path)
        self.registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        self.symbols = {}
        self._load_models()

    def _load_models(self) -> None:
        for symbol, cfg in (self.registry.get("symbols") or {}).items():
            mode = str(cfg.get("mode", "") or "")
            windows = list(cfg.get("windows", []) or [])
            loaded = []

            for w in windows:
                path = str(w.get("model_path", "") or "")
                if not path:
                    continue
                p = Path(path)
                if not p.exists():
                    continue
                bundle = joblib.load(p)
                loaded.append({
                    "window": str(w.get("window", "") or ""),
                    "window_days": int(w.get("window_days", 0) or 0),
                    "model_type": str(w.get("model_type", "") or ""),
                    "weight": float(w.get("weight", 0.0) or 0.0),
                    "model": bundle.get("model"),
                    "feature_numeric": list(bundle.get("feature_numeric", []) or []),
                    "feature_categorical": list(bundle.get("feature_categorical", []) or []),
                })

            self.symbols[symbol] = {
                "mode": mode,
                "windows": loaded,
            }

    def _predict_one_model(self, model, feature_numeric: list[str], feature_categorical: list[str], row: dict) -> float | None:
        try:
            payload = {}
            for c in feature_numeric:
                payload[c] = float(row.get(c, 0.0) or 0.0)
            for c in feature_categorical:
                payload[c] = str(row.get(c, "") or "")
            X = pd.DataFrame([payload])
            p = model.predict_proba(X)[0][1]
            return float(p)
        except Exception:
            return None

    def predict(self, row: dict) -> float | None:
        symbol = str(row.get("symbol", "") or "")
        cfg = self.symbols.get(symbol)
        if not cfg:
            return None

        if str(cfg.get("mode", "")) != "multiwindow_blend":
            return None

        preds = []
        weights = []

        for w in list(cfg.get("windows", []) or []):
            p = self._predict_one_model(
                model=w.get("model"),
                feature_numeric=list(w.get("feature_numeric", []) or []),
                feature_categorical=list(w.get("feature_categorical", []) or []),
                row=row,
            )
            if p is None:
                continue
            preds.append(float(p))
            weights.append(float(w.get("weight", 0.0) or 0.0))

        if not preds:
            return None

        wsum = float(sum(weights))
        if wsum <= 0.0:
            return float(np.mean(preds))

        return float(np.average(np.array(preds, dtype=float), weights=np.array(weights, dtype=float)))
