from __future__ import annotations


def apply_symbol_cap(weights: dict[str, float], symbol_cap: float) -> dict[str, float]:
    cap = max(0.0, float(symbol_cap))
    if cap <= 0.0:
        return dict(weights or {})

    out = {k: min(cap, max(0.0, float(v or 0.0))) for k, v in (weights or {}).items()}
    s = sum(out.values())
    if s <= 0.0:
        return {}
    return {k: float(v / s) for k, v in out.items()}


def apply_target_exposure(weights: dict[str, float], target_exposure: float, sides: dict[str, str]) -> dict[str, float]:
    te = max(0.0, float(target_exposure))
    out = {}
    for symbol, w in (weights or {}).items():
        side = str((sides or {}).get(symbol, "flat") or "flat").lower()
        signed = float(w) * (1.0 if side == "long" else -1.0 if side == "short" else 0.0)
        out[symbol] = float(signed * te)
    return out
