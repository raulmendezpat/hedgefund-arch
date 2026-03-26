from __future__ import annotations

from collections import defaultdict


def opp_score(opp) -> float:
    meta = dict(getattr(opp, "meta", {}) or {})
    for key in [
        "post_ml_competitive_score",
        "meta_post_ml_competitive_score",
        "post_ml_score",
        "meta_post_ml_score",
        "competitive_score",
        "meta_competitive_score",
    ]:
        try:
            v = float(meta.get(key, 0.0) or 0.0)
        except Exception:
            v = 0.0
        if v > 0.0:
            return float(v)
    return 0.0


def build_pre_allocator_trace(opps: list) -> dict:
    by_symbol = defaultdict(list)

    for opp in list(opps or []):
        symbol = str(getattr(opp, "symbol", "") or "")
        strategy_id = str(getattr(opp, "strategy_id", "") or "")
        side = str(getattr(opp, "side", "") or "")
        strength = float(getattr(opp, "strength", 0.0) or 0.0)
        score = float(opp_score(opp))
        meta = dict(getattr(opp, "meta", {}) or {})

        by_symbol[symbol].append(
            {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "side": side,
                "strength": strength,
                "score": score,
                "base_weight": float(meta.get("base_weight", 0.0) or 0.0),
                "competitive_score": float(meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0),
                "post_ml_competitive_score": float(
                    meta.get(
                        "post_ml_competitive_score",
                        meta.get(
                            "meta_post_ml_competitive_score",
                            meta.get("post_ml_score", meta.get("meta_post_ml_score", 0.0)),
                        ),
                    ) or 0.0
                ),
                "p_win": float(meta.get("p_win", 0.0) or 0.0),
                "policy_score": float(meta.get("policy_score", 0.0) or 0.0),
            }
        )

    ranked_by_symbol = {}
    top_global = []

    for symbol, rows in by_symbol.items():
        ranked = sorted(
            rows,
            key=lambda r: (
                float(r.get("score", 0.0) or 0.0),
                float(r.get("post_ml_competitive_score", 0.0) or 0.0),
                float(r.get("competitive_score", 0.0) or 0.0),
                float(r.get("strength", 0.0) or 0.0),
            ),
            reverse=True,
        )
        ranked_by_symbol[symbol] = ranked[:5]
        if ranked:
            top_global.append(ranked[0])

    top_global = sorted(
        top_global,
        key=lambda r: (
            float(r.get("score", 0.0) or 0.0),
            float(r.get("post_ml_competitive_score", 0.0) or 0.0),
            float(r.get("competitive_score", 0.0) or 0.0),
            float(r.get("strength", 0.0) or 0.0),
        ),
        reverse=True,
    )[:10]

    return {
        "counts_by_symbol": {str(k): int(len(v)) for k, v in by_symbol.items()},
        "ranked_by_symbol": ranked_by_symbol,
        "top_global": top_global,
    }


def apply_competition_mode(opps: list, config_snapshot: dict | None = None) -> tuple[list, dict]:
    cfg = dict(config_snapshot or {})
    mode = str(cfg.get("legacy_competition_mode", "off") or "off").lower().strip()

    if mode in {"", "off", "none"}:
        return list(opps or []), {
            "mode": "off",
            "input_count": int(len(list(opps or []))),
            "output_count": int(len(list(opps or []))),
        }

    opps = list(opps or [])

    def _sort_key(opp):
        meta = dict(getattr(opp, "meta", {}) or {})
        return (
            float(opp_score(opp)),
            float(meta.get("post_ml_competitive_score", meta.get("meta_post_ml_competitive_score", 0.0)) or 0.0),
            float(meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0),
            float(getattr(opp, "strength", 0.0) or 0.0),
        )

    selected = opps

    if mode == "best_per_symbol":
        best = {}
        for opp in opps:
            symbol = str(getattr(opp, "symbol", "") or "")
            prev = best.get(symbol)
            if prev is None or _sort_key(opp) > _sort_key(prev):
                best[symbol] = opp
        selected = list(best.values())
    elif mode == "top1_global":
        ranked = sorted(opps, key=_sort_key, reverse=True)
        selected = [opp for opp in ranked[:1] if float(opp_score(opp)) > 0.0]
    elif mode == "top2_global":
        ranked = sorted(opps, key=_sort_key, reverse=True)
        selected = [opp for opp in ranked[:2] if float(opp_score(opp)) > 0.0]

    summary_rows = []
    for opp in selected:
        meta = dict(getattr(opp, "meta", {}) or {})
        summary_rows.append(
            {
                "symbol": str(getattr(opp, "symbol", "") or ""),
                "strategy_id": str(getattr(opp, "strategy_id", "") or ""),
                "side": str(getattr(opp, "side", "") or ""),
                "score": float(opp_score(opp)),
                "post_ml_competitive_score": float(
                    meta.get("post_ml_competitive_score", meta.get("meta_post_ml_competitive_score", 0.0)) or 0.0
                ),
                "competitive_score": float(
                    meta.get("competitive_score", meta.get("meta_competitive_score", 0.0)) or 0.0
                ),
            }
        )

    summary = {
        "mode": mode,
        "input_count": int(len(opps)),
        "output_count": int(len(selected)),
        "selected": summary_rows,
    }

    if mode == "best_per_symbol":
        summary["kept_symbols"] = sorted(
            {str(getattr(opp, "symbol", "") or "") for opp in selected}
        )
    elif mode in {"top1_global", "top2_global"}:
        summary["top_n"] = 1 if mode == "top1_global" else 2

    return selected, summary
