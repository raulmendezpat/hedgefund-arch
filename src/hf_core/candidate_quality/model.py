from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _extract_value(candidate, key: str, default: Any = None) -> Any:
    sm = dict(getattr(candidate, "signal_meta", {}) or {})
    meta = dict(getattr(candidate, "meta", {}) or {})

    if key in sm:
        return sm.get(key)
    if key in meta:
        return meta.get(key)
    if hasattr(candidate, key):
        return getattr(candidate, key)

    return default


@lru_cache(maxsize=8)
def _load_manifest(manifest_path: str) -> dict:
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"candidate quality manifest not found: {manifest_path}")
    return json.loads(p.read_text())


@lru_cache(maxsize=4)
def _load_model(model_path: str):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"candidate quality model not found: {model_path}")
    return joblib.load(p)


class CandidateQualityModel:
    def __init__(self, model_path: str, manifest_path: str):
        self.model_path = str(model_path)
        self.manifest_path = str(manifest_path)
        self.manifest = _load_manifest(self.manifest_path)
        self.model = _load_model(self.model_path)

        self.numeric_features = list(self.manifest.get("numeric_features", []) or [])
        self.categorical_features = list(self.manifest.get("categorical_features", []) or [])
        self.features = self.numeric_features + self.categorical_features

        if not self.numeric_features:
            raise ValueError("candidate quality manifest has no numeric_features")

    def build_row(self, candidate) -> pd.DataFrame:
        row = {}

        for key in self.numeric_features:
            row[key] = _safe_float(_extract_value(candidate, key, 0.0), 0.0)

        for key in self.categorical_features:
            value = _extract_value(candidate, key, "")
            row[key] = str(value if value is not None else "")

        return pd.DataFrame([row], columns=self.features)

    def score_candidate(self, candidate) -> float:
        x = self.build_row(candidate)

        if hasattr(self.model, "predict_proba"):
            pred = self.model.predict_proba(x)
            score = float(pred[0, 1])
        else:
            pred = self.model.predict(x)
            score = float(pred[0])

        if not np.isfinite(score):
            score = 0.0

        return max(0.0, min(1.0, score))
