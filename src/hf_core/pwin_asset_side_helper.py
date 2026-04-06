from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _candidate_meta(candidate) -> dict[str, Any]:
    meta = {}
    try:
        meta.update(dict(getattr(candidate, "meta", {}) or {}))
    except Exception:
        pass
    try:
        meta.update(dict(getattr(candidate, "signal_meta", {}) or {}))
    except Exception:
        pass

    for key in ["symbol", "side", "strategy_id", "signal_strength", "base_weight", "ts"]:
        try:
            if key not in meta:
                meta[key] = getattr(candidate, key)
        except Exception:
            pass

    return meta


@lru_cache(maxsize=4)
def _load_registry_cached(registry_path: str) -> dict[str, Any]:
    p = Path(registry_path)
    if not p.exists():
        return {}

    data = json.loads(p.read_text(encoding="utf-8"))
    groups = dict(data.get("groups", {}) or {})

    loaded = {
        "version": str(data.get("version", "")),
        "target_col": str(data.get("target_col", "")),
        "groups": {},
    }

    for key, info in groups.items():
        model_path = Path(str(info.get("model_path", "") or ""))
        if not model_path.exists():
            continue
        try:
            with model_path.open("rb") as f:
                payload = pickle.load(f)
            loaded["groups"][str(key)] = {
                **dict(info or {}),
                "payload": payload,
            }
        except Exception:
            continue

    return loaded


def load_registry(registry_path: str | None = None) -> dict[str, Any]:
    path = str(
        registry_path
        or os.environ.get("PWIN_ASSET_SIDE_REGISTRY", "")
    ).strip()
    if not path:
        return {}
    return _load_registry_cached(path)


def resolve_group_key(candidate) -> str:
    meta = _candidate_meta(candidate)
    symbol = str(meta.get("symbol", "") or "")
    side = str(meta.get("side", "") or "").lower()
    return f"{symbol}|{side}"


def predict_pwin_for_candidate(candidate, registry_path: str | None = None) -> dict[str, Any]:
    registry = load_registry(registry_path=registry_path)
    if not registry:
        return {
            "enabled": False,
            "applied": False,
            "reason": "registry_unavailable",
            "group_key": "",
            "p_win": None,
            "model_name": "",
            "registry_version": "",
        }

    group_key = resolve_group_key(candidate)
    group_info = dict(registry.get("groups", {}).get(group_key, {}) or {})
    if not group_info:
        return {
            "enabled": True,
            "applied": False,
            "reason": "group_missing",
            "group_key": group_key,
            "p_win": None,
            "model_name": "",
            "registry_version": str(registry.get("version", "")),
        }

    payload = dict(group_info.get("payload", {}) or {})
    pipe = payload.get("pipeline")
    feature_cols = list(payload.get("feature_cols", []) or [])

    if pipe is None or not feature_cols:
        return {
            "enabled": True,
            "applied": False,
            "reason": "payload_invalid",
            "group_key": group_key,
            "p_win": None,
            "model_name": str(group_info.get("model_name", "") or ""),
            "registry_version": str(registry.get("version", "")),
        }

    meta = _candidate_meta(candidate)
    row = {}
    for c in feature_cols:
        row[c] = meta.get(c, None)

    try:
        import pandas as pd

        X = pd.DataFrame([row], columns=feature_cols)

        if hasattr(pipe, "predict_proba"):
            p = float(pipe.predict_proba(X)[:, 1][0])
        else:
            raw = float(pipe.decision_function(X)[0])
            p = float(1.0 / (1.0 + pow(2.718281828459045, -raw)))

        p = max(0.0, min(1.0, float(p)))

        return {
            "enabled": True,
            "applied": True,
            "reason": "ok",
            "group_key": group_key,
            "p_win": p,
            "model_name": str(group_info.get("model_name", "") or ""),
            "registry_version": str(registry.get("version", "")),
        }
    except Exception as e:
        return {
            "enabled": True,
            "applied": False,
            "reason": f"predict_error:{e}",
            "group_key": group_key,
            "p_win": None,
            "model_name": str(group_info.get("model_name", "") or ""),
            "registry_version": str(registry.get("version", "")),
        }


def override_candidate_pwin(candidate, fallback: float | None = None, registry_path: str | None = None) -> dict[str, Any]:
    out = predict_pwin_for_candidate(candidate, registry_path=registry_path)
    p_win = out.get("p_win", None)
    if p_win is None:
        p_win = fallback
    out["p_win_final"] = p_win
    out["fallback"] = fallback
    return out
