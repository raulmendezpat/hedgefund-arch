from __future__ import annotations

import json
import os
import pickle
import joblib
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
    if isinstance(candidate, dict):
        out = dict(candidate)
    else:
        meta = getattr(candidate, "meta", None)
        out = dict(meta) if isinstance(meta, dict) else {}

    direct_fields = [
        "symbol",
        "side",
        "strategy_id",
        "timestamp",
        "ts",
        "p_win",
        "policy_score",
        "post_ml_score",
        "post_ml_competitive_score",
    ]
    for k in direct_fields:
        if k not in out or out.get(k) in (None, ""):
            v = getattr(candidate, k, None) if not isinstance(candidate, dict) else out.get(k)
            if v not in (None, ""):
                out[k] = v

    return out


def _normalize_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"long", "short"}:
        return side
    if side in {"flat", "", "none", "nan", "null"}:
        return "flat"
    return side


@lru_cache(maxsize=4)
def _load_registry_cached(registry_path: str) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}

    registry_feature_cols = list(data.get("feature_cols", []) or [])
    groups = dict(data.get("groups", {}) or {})
    loaded = {
        "version": str(data.get("version", "")),
        "target_col": str(data.get("target_col", "")),
        "feature_cols": registry_feature_cols,
        "groups": {},
    }

    for key, info in groups.items():
        info = dict(info or {})
        model_path = Path(str(info.get("model_path", "") or ""))
        if not model_path.exists():
            continue

        try:
            try:
                artifact = joblib.load(model_path)
            except Exception:
                with model_path.open("rb") as f:
                    artifact = pickle.load(f)

            # Supported artifact contracts:
            # 1) Old style: {"pipeline": <model>, "feature_cols": [...]}
            # 2) New style: direct sklearn Pipeline/model object; feature_cols lives in registry JSON/group info.
            if isinstance(artifact, dict):
                pipeline = artifact.get("pipeline") or artifact.get("model")
                artifact_feature_cols = list(artifact.get("feature_cols", []) or [])
            else:
                pipeline = artifact
                artifact_feature_cols = []

            group_feature_cols = list(info.get("feature_cols", []) or [])
            feature_cols = artifact_feature_cols or group_feature_cols or registry_feature_cols

            loaded["groups"][str(key)] = {
                **info,
                "pipeline": pipeline,
                "feature_cols": feature_cols,
                "artifact_contract": "dict_payload" if isinstance(artifact, dict) else "direct_pipeline",
            }
        except Exception as e:
            loaded["groups"][str(key)] = {
                **info,
                "pipeline": None,
                "feature_cols": [],
                "artifact_contract": "load_error",
                "load_error": str(e),
            }

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

    if isinstance(candidate, dict):
        symbol_value = meta.get("symbol", "") or candidate.get("symbol", "")
        side_value = candidate.get("side", None)
    else:
        symbol_value = meta.get("symbol", "") or getattr(candidate, "symbol", "")
        side_value = getattr(candidate, "side", None)

    symbol = str(symbol_value or "").strip()
    side = _normalize_side(side_value if side_value is not None else meta.get("side", ""))

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
    side_for_group = group_key.split("|", 1)[1] if "|" in group_key else ""
    if side_for_group not in {"long", "short"}:
        return {
            "enabled": True,
            "applied": False,
            "reason": f"side_not_supported:{side_for_group}",
            "group_key": group_key,
            "p_win": None,
            "model_name": "",
            "registry_version": str(registry.get("version", "")),
        }

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

    pipe = group_info.get("pipeline")
    feature_cols = list(group_info.get("feature_cols", []) or registry.get("feature_cols", []) or [])

    if pipe is None or not feature_cols:
        reason = "payload_invalid"
        if group_info.get("artifact_contract") == "load_error":
            reason = f"model_load_error:{group_info.get('load_error', '')}"
        return {
            "enabled": True,
            "applied": False,
            "reason": reason,
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
