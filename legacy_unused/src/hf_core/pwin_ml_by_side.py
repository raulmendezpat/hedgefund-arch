from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


class PWinMLBySide:
    def __init__(self, registry_path: str):
        self.registry_path = str(registry_path)
        self.registry = {}
        self._models = {}

        p = Path(self.registry_path)
        if p.exists():
            self.registry = json.loads(p.read_text(encoding="utf-8"))
        else:
            self.registry = {"models": {}}

    def has_model(self, symbol: str, side: str) -> bool:
        key = f"{symbol}|{str(side).lower()}"
        return key in dict(self.registry.get("models", {}) or {})

    def _load_model(self, key: str):
        if key in self._models:
            return self._models[key]

        entry = dict((self.registry.get("models", {}) or {}).get(key, {}) or {})
        path = entry.get("path")
        if not path:
            return None

        p = Path(path)
        if not p.exists():
            return None

        model = joblib.load(p)
        self._models[key] = model
        return model

    def predict_from_feature_values(self, values: dict) -> float | None:
        symbol = str(values.get("symbol", "") or "")
        side = str(values.get("side", "") or "").lower()
        if not symbol or side not in {"long", "short"}:
            return None

        key = f"{symbol}|{side}"
        entry = dict((self.registry.get("models", {}) or {}).get(key, {}) or {})
        if not entry:
            return None
        if not bool(entry.get("recommended_apply", False)):
            return None

        model = self._load_model(key)
        if model is None:
            return None

        row = pd.DataFrame([dict(values or {})])

        try:
            proba = model.predict_proba(row)
            if proba is None or len(proba) == 0:
                return None
            return float(proba[0][1])
        except Exception:
            return None
