from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

from hf.core.types import Candle, Signal


FEATURE_COLUMNS = [
    "is_btc",
    "is_sol",
    "side_long",
    "side_short",
    "strength",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "adx",
    "atr",
    "atrp",
    "ema_fast",
    "ema_slow",
    "ema_gap",
    "ema_gap_pct",
    "rsi",
    "bb_mid",
    "bb_up",
    "bb_low",
    "bb_width",
    "bb_span",
    "bb_pos",
]


def load_model(path: Optional[str]) -> Any:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None

    suffix = p.suffix.lower()

    if suffix in {".joblib", ".jl"} and joblib is not None:
        try:
            return joblib.load(p)
        except Exception:
            pass

    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    if joblib is not None:
        try:
            return joblib.load(p)
        except Exception:
            pass

    return None


def load_model_registry(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    suffix = p.suffix.lower()

    # Registry serializado completo
    if suffix in {".joblib", ".jl", ".pkl", ".pickle"}:
        loaded = load_model(path)
        if isinstance(loaded, dict):
            return loaded
        return {}

    # JSON clásico: {"BTC/USDT:USDT|long": "artifacts/model.pkl", ...}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[str, Any] = {}
    if not isinstance(data, dict):
        return out

    for k, v in (data or {}).items():
        if isinstance(v, str) and v:
            out[str(k)] = load_model(v)

    return out


def select_model_for_signal(
    model: Any,
    registry: Dict[str, Any],
    symbol: str,
    side: str,
    strategy_id: Optional[str] = None,
) -> Any:
    registry = dict(registry or {})

    # Registry nuevo: artifact con {"type": "strategy_model_registry", "models": {...}}
    if isinstance(registry.get("models"), dict):
        models = dict(registry.get("models") or {})
        if strategy_id and strategy_id in models and models[strategy_id] is not None:
            return models[strategy_id]
        if "default" in models and models["default"] is not None:
            return models["default"]
        return model

    # Registry clásico symbol|side
    candidates = [
        f"{symbol}|{side}",
        f"{symbol}|*",
        f"*|{side}",
        "*|*",
    ]
    for key in candidates:
        if key in registry and registry[key] is not None:
            return registry[key]

    return model


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def build_feature_row(symbol: str, candle: Candle, signal: Signal) -> Dict[str, Any]:
    feats = dict(getattr(candle, "features", {}) or {})
    side = getattr(signal, "side", "flat")
    meta = dict(getattr(signal, "meta", {}) or {})

    row: Dict[str, Any] = {
        "is_btc": 1.0 if "BTC" in symbol.upper() else 0.0,
        "is_sol": 1.0 if "SOL" in symbol.upper() else 0.0,
        "side_long": 1.0 if side == "long" else 0.0,
        "side_short": 1.0 if side == "short" else 0.0,
        "strength": _safe_float(getattr(signal, "strength", 0.0), 0.0),
        "close": _safe_float(getattr(candle, "close", 0.0), 0.0),
        "open": _safe_float(getattr(candle, "open", 0.0), 0.0),
        "high": _safe_float(getattr(candle, "high", 0.0), 0.0),
        "low": _safe_float(getattr(candle, "low", 0.0), 0.0),
        "volume": _safe_float(getattr(candle, "volume", 0.0), 0.0),

        # requeridos por los modelos entrenados recientes
        "strategy_id": str(meta.get("strategy_id", "") or ""),
        "symbol": str(symbol),
        "side_raw": str(side),
        "base_weight": _safe_float(meta.get("base_weight", 1.0), 1.0),
        "competitive_score": _safe_float(meta.get("competitive_score", 0.0), 0.0),
    }

    for k, v in feats.items():
        row[str(k)] = _safe_float(v, 0.0)

    if "atr" in row and "atrp" not in row and row.get("close", 0.0) != 0.0:
        row["atrp"] = row["atr"] / row["close"]

    if "ema_fast" in row and "ema_slow" in row:
        row["ema_gap"] = row["ema_fast"] - row["ema_slow"]
        denom = abs(row["ema_slow"]) if row["ema_slow"] != 0.0 else 1.0
        row["ema_gap_pct"] = row["ema_gap"] / denom

    if "bb_up" in row and "bb_low" in row and "close" in row:
        span = row["bb_up"] - row["bb_low"]
        row["bb_span"] = span
        if span != 0.0:
            row["bb_pos"] = (row["close"] - row["bb_low"]) / span
        else:
            row["bb_pos"] = 0.0

    normalized: Dict[str, Any] = {
        col: _safe_float(row.get(col, 0.0), 0.0)
        for col in FEATURE_COLUMNS
    }

    normalized["strategy_id"] = str(row.get("strategy_id", "") or "")
    normalized["symbol"] = str(row.get("symbol", "") or "")
    normalized["side_raw"] = str(row.get("side_raw", "") or "")
    normalized["base_weight"] = _safe_float(row.get("base_weight", 1.0), 1.0)
    normalized["competitive_score"] = _safe_float(row.get("competitive_score", 0.0), 0.0)

    return normalized


def _artifact_model(model: Any) -> Any:
    if isinstance(model, dict) and model.get("model") is not None:
        return model.get("model")
    return model


def _features_to_frame(model: Any, features: Dict[str, Any]) -> pd.DataFrame:
    if isinstance(model, dict):
        num_cols = list(model.get("feature_cols_num") or [])
        cat_cols = list(model.get("feature_cols_cat") or [])
        if num_cols or cat_cols:
            row: Dict[str, Any] = {}
            for col in num_cols:
                row[col] = _safe_float(features.get(col, 0.0), 0.0)
            for col in cat_cols:
                row[col] = str(features.get(col, "") or "")
            return pd.DataFrame([row], columns=num_cols + cat_cols)

    row = {col: _safe_float(features.get(col, 0.0), 0.0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_proba(model: Any, features: Dict[str, Any]) -> float:
    raw_model = model
    model = _artifact_model(model)

    if model is not None and hasattr(model, "predict_proba"):
        try:
            X = _features_to_frame(raw_model, features)
            proba = model.predict_proba(X)
            if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
                return _safe_float(proba[0][1], 0.5)
            return _safe_float(proba[0], 0.5)
        except Exception:
            pass

    if callable(model):
        try:
            return _safe_float(model(features), 0.5)
        except Exception:
            pass

    # fallback heurístico
    score = 0.5
    score += min(max(features.get("adx", 0.0) - 18.0, -10.0), 10.0) * 0.01
    score += min(max(features.get("ema_gap_pct", 0.0), -0.05), 0.05) * 2.0
    score -= min(max(features.get("atrp", 0.0) - 0.02, -0.03), 0.03) * 3.0

    rsi = features.get("rsi", 50.0)
    bb_pos = features.get("bb_pos", 0.5)
    has_bb_context = any(
        abs(float(features.get(k, 0.0) or 0.0)) > 1e-12
        for k in ("bb_mid", "bb_up", "bb_low", "bb_width", "bb_span", "bb_pos")
    )

    if has_bb_context:
        if features.get("side_long", 0.0) > 0.5:
            score += max(0.0, (35.0 - rsi)) * 0.004
            score += max(0.0, (0.25 - bb_pos)) * 0.30
        elif features.get("side_short", 0.0) > 0.5:
            score += max(0.0, (rsi - 65.0)) * 0.004
            score += max(0.0, (bb_pos - 0.75)) * 0.30

    return max(0.0, min(1.0, score))


def apply_ml_filter_to_signals(
    *,
    candles: Dict[str, Candle],
    signals: Dict[str, Signal],
    model: Any,
    threshold: float,
    model_registry: Optional[Dict[str, Any]] = None,
    threshold_map: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, Signal], Dict[str, int]]:
    out: Dict[str, Signal] = {}
    rejected: Dict[str, int] = {}
    model_registry = dict(model_registry or {})

    for sym, sig in (signals or {}).items():
        if sig is None:
            out[sym] = sig
            continue

        side = getattr(sig, "side", "flat")
        meta = dict(getattr(sig, "meta", {}) or {})

        if "skip" in meta or side == "flat":
            out[sym] = sig
            continue

        candle = candles.get(sym)
        if candle is None:
            out[sym] = sig
            continue

        feats = build_feature_row(sym, candle, sig)
        chosen_model = select_model_for_signal(
            model,
            model_registry,
            sym,
            side,
            strategy_id=str(meta.get("strategy_id", "") or ""),
        )

        p_win = predict_proba(chosen_model, feats)

        strategy_id = str(meta.get("strategy_id", "") or "")
        effective_threshold = float(threshold)
        if threshold_map and strategy_id:
            try:
                effective_threshold = float(threshold_map.get(strategy_id, effective_threshold))
            except Exception:
                effective_threshold = float(threshold)

        if p_win < effective_threshold:
            rejected[sym] = int(rejected.get(sym, 0)) + 1
            out[sym] = Signal(
                symbol=sym,
                side="flat",
                strength=0.0,
                meta={
                    **meta,
                    "reason": "ml_filter",
                    "p_win": float(p_win),
                    "ml_threshold": float(effective_threshold),
                    "ml_rejected": 1,
                },
            )
        else:
            out[sym] = Signal(
                symbol=sym,
                side=side,
                strength=float(getattr(sig, "strength", 0.0) or 0.0),
                meta={
                    **meta,
                    "p_win": float(p_win),
                    "ml_threshold": float(effective_threshold),
                    "ml_rejected": 0,
                },
            )

    return out, rejected
