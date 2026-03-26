from __future__ import annotations

import json
import joblib
import numpy as np
from pathlib import Path


class PWinML:
    def __init__(self, registry_path: str):
        self.registry = json.loads(Path(registry_path).read_text())
        self.models = {}

        for symbol, cfg in (self.registry.get("models") or {}).items():
            path = cfg.get("path")
            if not path:
                continue

            p = Path(path)
            if not p.exists():
                continue

            self.models[symbol] = {
                "model": joblib.load(p),
                "recommended": bool(cfg.get("recommended_apply", False)),
            }

    def predict(self, row: dict) -> float | None:
        symbol = row.get("symbol")
        bundle = self.models.get(symbol)

        if not bundle:
            return None

        # SOLO aplicar si está aprobado
        if not bundle["recommended"]:
            return None

        try:
            # features que ya sabes que existen
            x = [
                float(row.get("signal_strength", 0.0) or 0.0),
                float(row.get("adx", 0.0) or 0.0),
                float(row.get("atrp", 0.0) or 0.0),
            ]

            X = np.array([x])

            proba = bundle["model"].predict_proba(X)[0][1]
            return float(proba)

        except Exception:
            return None
