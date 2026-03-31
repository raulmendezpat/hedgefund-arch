from __future__ import annotations

from typing import Any

from hf_core.contracts import OpportunityCandidate


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def seed_candidate_meta(candidate: OpportunityCandidate) -> OpportunityCandidate:
    meta = dict(getattr(candidate, "signal_meta", {}) or {})

    # Normalizar aliases ya existentes para que FeatureBuilder / MetaModel los vean
    if "meta_p_win" not in meta:
        if "p_win" in meta:
            meta["meta_p_win"] = _f(meta.get("p_win"), 0.50)
        elif "ml_p_win" in meta:
            meta["meta_p_win"] = _f(meta.get("ml_p_win"), 0.50)
        else:
            meta["meta_p_win"] = 0.50

    if "meta_post_ml_score" not in meta:
        if "post_ml_score" in meta:
            meta["meta_post_ml_score"] = _f(meta.get("post_ml_score"), 0.0)
        else:
            meta["meta_post_ml_score"] = 0.0

    if "meta_competitive_score" not in meta:
        if "competitive_score" in meta:
            meta["meta_competitive_score"] = _f(meta.get("competitive_score"), 0.0)
        else:
            meta["meta_competitive_score"] = 0.0

    candidate.signal_meta = meta
    return candidate


def build_portfolio_context(
    candidates: list[OpportunityCandidate],
    *,
    score_mode: str = "early",
) -> dict[str, float | str]:
    active = [c for c in (candidates or []) if str(getattr(c, "side", "flat") or "flat").lower() in {"long", "short"}]

    if not active:
        return {
            "portfolio_regime": "normal",
            "portfolio_breadth": 0.0,
            "portfolio_avg_pwin": 0.50,
            "portfolio_avg_atrp": 0.0,
            "portfolio_avg_strength": 0.0,
            "portfolio_conviction": 0.0,
            "portfolio_regime_scale_applied": 1.0,
        }

    pwin_vals = []
    atrp_vals = []
    strength_vals = []
    n_long = 0
    n_short = 0

    for c in active:
        meta = dict(getattr(c, "signal_meta", {}) or {})
        side = str(getattr(c, "side", "flat") or "flat").lower()

        if side == "long":
            n_long += 1
        elif side == "short":
            n_short += 1

        if str(score_mode) == "allocation":
            pwin_vals.append(_f(meta.get("post_ml_score", meta.get("p_win", 0.0)), 0.0))
        else:
            pwin_vals.append(_f(meta.get("meta_p_win", meta.get("p_win", 0.50)), 0.50))
        atrp_vals.append(_f(meta.get("atrp", 0.0), 0.0))
        strength_vals.append(abs(_f(getattr(c, "signal_strength", 0.0), 0.0)))

    breadth = float(len(active))
    avg_pwin = sum(pwin_vals) / max(1, len(pwin_vals))
    avg_atrp = sum(atrp_vals) / max(1, len(atrp_vals))
    avg_strength = sum(strength_vals) / max(1, len(strength_vals))

    short_share = float(n_short) / max(1, len(active))
    conviction = _clip((avg_strength * 0.65) + (max(0.0, avg_pwin - 0.50) * 4.0 * 0.35), 0.0, 1.0)

    # Régimen simple pero útil para que MetaModel reciba contexto real
    regime = "defensive" if (short_share >= 0.60 and breadth >= 3.0) else "normal"

    return {
        "portfolio_regime": regime,
        "portfolio_breadth": float(breadth),
        "portfolio_avg_pwin": float(avg_pwin),
        "portfolio_avg_atrp": float(avg_atrp),
        "portfolio_avg_strength": float(avg_strength),
        "portfolio_conviction": float(conviction),
        "portfolio_regime_scale_applied": 1.0,
    }
