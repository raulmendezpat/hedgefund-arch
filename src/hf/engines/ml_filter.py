from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from hf.core.types import Candle, Signal


def load_model(path: Optional[str]) -> Any:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def build_feature_row(symbol: str, candle: Candle, signal: Signal) -> Dict[str, float]:
    feats = dict(getattr(candle, "features", {}) or {})
    side = getattr(signal, "side", "flat")

    row: Dict[str, float] = {
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
    }

    for k, v in feats.items():
        row[str(k)] = _safe_float(v, 0.0)

    # compatibility aliases
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

    return row


def predict_proba(model: Any, features: Dict[str, float]) -> float:
    # model contract 1: sklearn-like predict_proba(DataFrame)
    if model is not None and hasattr(model, "predict_proba"):
        X = pd.DataFrame([features])
        proba = model.predict_proba(X)
        try:
            if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
                return _safe_float(proba[0][1], 0.5)
            return _safe_float(proba[0], 0.5)
        except Exception:
            pass

    # model contract 2: callable(features_dict) -> float
    if callable(model):
        try:
            return _safe_float(model(features), 0.5)
        except Exception:
            pass

    # fallback stub heuristic
    score = 0.5
    score += min(max(features.get("adx", 0.0) - 18.0, -10.0), 10.0) * 0.01
    score += min(max(features.get("ema_gap_pct", 0.0), -0.05), 0.05) * 2.0
    score -= min(max(features.get("atrp", 0.0) - 0.02, -0.03), 0.03) * 3.0

    if features.get("is_sol", 0.0) > 0.5:
        rsi = features.get("rsi", 50.0)
        bb_pos = features.get("bb_pos", 0.5)
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
) -> tuple[Dict[str, Signal], Dict[str, int]]:
    out: Dict[str, Signal] = {}
    rejected: Dict[str, int] = {}

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
        p_win = predict_proba(model, feats)

        if p_win < float(threshold):
            rejected[sym] = int(rejected.get(sym, 0)) + 1
            out[sym] = Signal(
                symbol=sym,
                side="flat",
                strength=0.0,
                meta={
                    **meta,
                    "reason": "ml_filter",
                    "p_win": float(p_win),
                    "ml_threshold": float(threshold),
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
                    "ml_threshold": float(threshold),
                },
            )

    return out, rejected
