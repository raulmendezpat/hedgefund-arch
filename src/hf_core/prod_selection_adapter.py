from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_bool(x: Any) -> bool:
    try:
        return bool(x)
    except Exception:
        return False


@dataclass
class ProdSelectionResult:
    selected_candidates: list
    selected_decisions: list
    meta: dict


def _score_competitive(candidate) -> float:
    side = str(getattr(candidate, "side", "flat") or "flat").lower()
    active_flag = 1.0 if side in {"long", "short"} else 0.0
    strength = abs(_safe_float(getattr(candidate, "signal_strength", 0.0), 0.0))
    base_weight = _safe_float(getattr(candidate, "base_weight", 1.0), 1.0)
    return float(active_flag * strength * base_weight)


def _score_post_ml(candidate) -> float:
    sm = dict(getattr(candidate, "signal_meta", {}) or {})
    competitive_score = _safe_float(
        sm.get("competitive_score", sm.get("meta_competitive_score", 0.0)),
        0.0,
    )
    if competitive_score <= 0.0:
        competitive_score = _score_competitive(candidate)

    p_win = _safe_float(
        sm.get("p_win_prod", sm.get("p_win_ml_raw", sm.get("p_win", 0.0))),
        0.0,
    )

    size_mult = _safe_float(
        sm.get("ml_position_size_mult", sm.get("policy_size_mult", 1.0)),
        1.0,
    )
    if size_mult <= 0.0:
        size_mult = 1.0
    size_mult = max(0.50, min(1.50, size_mult))

    return float(competitive_score * p_win * size_mult)


def _threshold_for_candidate(candidate, thresholds: dict[str, float]) -> float:
    sm = dict(getattr(candidate, "signal_meta", {}) or {})
    strategy_id = str(getattr(candidate, "strategy_id", "") or sm.get("strategy_id", "") or "")
    return _safe_float(thresholds.get(strategy_id, thresholds.get("default", 0.0)), 0.0)


def _apply_thresholds(candidates, decisions, thresholds: dict[str, float]):
    kept = []
    dropped = []

    for c, d in zip(candidates or [], decisions or []):
        sm = dict(getattr(c, "signal_meta", {}) or {})
        accepted = _safe_bool(getattr(d, "accept", False))
        p_win = _safe_float(sm.get("p_win", 0.0), 0.0)
        threshold = _threshold_for_candidate(c, thresholds)

        sm["prod_selection_threshold"] = float(threshold)
        sm["prod_selection_p_win"] = float(p_win)

        # Observability only:
        # no hard reject here; upstream policy / ranking already decide acceptance.
        sm["prod_selection_filtered_by_threshold"] = bool(accepted and p_win < threshold)
        if bool(sm["prod_selection_filtered_by_threshold"]):
            sm["prod_selection_threshold_observed_only"] = True
            if not str(sm.get("policy_reason", "") or ""):
                sm["policy_reason"] = "prod_ml_threshold_observed_only"

        c.signal_meta = sm
        kept.append((c, d))

    return kept, dropped


def _annotate_scores(candidates, preserve_upstream_post_ml_scores: bool = False):
    annotated = []
    for c in candidates or []:
        sm = dict(getattr(c, "signal_meta", {}) or {})

        if preserve_upstream_post_ml_scores:
            competitive_score = _safe_float(
                sm.get("competitive_score", sm.get("meta_competitive_score", 0.0)),
                0.0,
            )
            if competitive_score <= 0.0:
                competitive_score = _score_competitive(c)

            post_ml_score = _safe_float(
                sm.get(
                    "post_ml_competitive_score",
                    sm.get(
                        "post_ml_score",
                        sm.get(
                            "meta_post_ml_competitive_score",
                            sm.get("meta_post_ml_score", 0.0),
                        ),
                    ),
                ),
                0.0,
            )
            if post_ml_score <= 0.0:
                post_ml_score = _score_post_ml(c)
        else:
            competitive_score = _score_competitive(c)
            post_ml_score = _score_post_ml(c)

        sm["competitive_score"] = float(competitive_score)
        sm["meta_competitive_score"] = float(competitive_score)
        sm["post_ml_score"] = float(post_ml_score)
        sm["post_ml_competitive_score"] = float(post_ml_score)
        sm["meta_post_ml_score"] = float(post_ml_score)
        sm["meta_post_ml_competitive_score"] = float(post_ml_score)
        c.signal_meta = sm
        annotated.append(c)
    return annotated


def _select_pairs(pairs, mode: str):
    mode = str(mode or "best_per_symbol").strip().lower()

    if mode == "all":
        return list(pairs)

    ranked = sorted(
        list(pairs),
        key=lambda x: (
            _safe_float(
                dict(getattr(x[0], "signal_meta", {}) or {}).get(
                    "post_ml_competitive_score",
                    dict(getattr(x[0], "signal_meta", {}) or {}).get("post_ml_score", 0.0),
                ),
                0.0,
            ),
            abs(_safe_float(getattr(x[0], "signal_strength", 0.0), 0.0)),
        ),
        reverse=True,
    )

    if mode == "best_per_symbol":
        best = {}
        for c, d in ranked:
            sym = str(getattr(c, "symbol", "") or "")
            if sym not in best:
                best[sym] = (c, d)
        return list(best.values())

    if mode == "competitive":
        best = {}
        for c, d in list(pairs):
            sym = str(getattr(c, "symbol", "") or "")
            sm = dict(getattr(c, "signal_meta", {}) or {})
            opp_score = _safe_float(
                sm.get("post_ml_competitive_score", sm.get("post_ml_score", 0.0)),
                0.0,
            )
            opp_strength = abs(_safe_float(getattr(c, "signal_strength", 0.0), 0.0))

            prev = best.get(sym)
            if prev is None:
                best[sym] = (c, d)
                continue

            prev_c, _ = prev
            prev_sm = dict(getattr(prev_c, "signal_meta", {}) or {})
            prev_score = _safe_float(
                prev_sm.get("post_ml_competitive_score", prev_sm.get("post_ml_score", 0.0)),
                0.0,
            )
            prev_strength = abs(_safe_float(getattr(prev_c, "signal_strength", 0.0), 0.0))

            if opp_score > prev_score:
                best[sym] = (c, d)
                continue

            if opp_score == prev_score and opp_strength > prev_strength:
                best[sym] = (c, d)
                continue

        return list(best.values())

    if mode == "top1_global":
        return ranked[:1]
    if mode == "top2_global":
        return ranked[:2]
    if mode == "top3_global":
        return ranked[:3]

    return list(pairs)


def apply_prod_selection_semantics(
    *,
    candidates,
    decisions,
    selection_mode: str = "best_per_symbol",
    ml_threshold: float = 0.0,
    ml_thresholds: dict[str, float] | None = None,
    preserve_upstream_post_ml_scores: bool = False,
) -> ProdSelectionResult:
    candidates = list(candidates or [])
    decisions = list(decisions or [])
    ml_thresholds = dict(ml_thresholds or {})
    if "default" not in ml_thresholds:
        ml_thresholds["default"] = float(ml_threshold)

    input_pairs = list(zip(candidates, decisions))
    accepted_pairs = [(c, d) for c, d in input_pairs if _safe_bool(getattr(d, "accept", False))]

    accepted_pairs, dropped_by_threshold = _apply_thresholds(
        [c for c, _ in accepted_pairs],
        [d for _, d in accepted_pairs],
        ml_thresholds,
    )

    accepted_candidates = _annotate_scores(
        [c for c, _ in accepted_pairs],
        preserve_upstream_post_ml_scores=bool(preserve_upstream_post_ml_scores),
    )
    accepted_pairs = list(zip(accepted_candidates, [d for _, d in accepted_pairs]))

    selected_pairs = _select_pairs(accepted_pairs, mode=selection_mode)
    selected_set = {id(c) for c, _ in selected_pairs}

    dropped_by_mode = []
    for c, d in accepted_pairs:
        if id(c) not in selected_set:
            sm = dict(getattr(c, "signal_meta", {}) or {})
            sm["accept"] = False
            sm["prod_selection_filtered_by_mode"] = True
            sm["policy_reason"] = f"prod_selection_mode:{selection_mode}"
            c.signal_meta = sm
            try:
                d.accept = False
            except Exception:
                pass
            dropped_by_mode.append((c, d))

    meta = {
        "enabled": True,
        "selection_mode": str(selection_mode),
        "rows_in": int(len(input_pairs)),
        "accepted_in": int(len([1 for _, d in input_pairs if _safe_bool(getattr(d, "accept", False))])),
        "threshold_default": float(ml_thresholds.get("default", 0.0)),
        "threshold_keys": sorted([k for k in ml_thresholds.keys() if k != "default"]),
        "kept_count": int(len(selected_pairs)),
        "dropped_by_threshold_count": int(len(dropped_by_threshold)),
        "dropped_by_mode_count": int(len(dropped_by_mode)),
        "kept": [
            {
                "symbol": str(getattr(c, "symbol", "") or ""),
                "strategy_id": str(getattr(c, "strategy_id", "") or ""),
                "side": str(getattr(c, "side", "") or ""),
                "post_ml_score": _safe_float(dict(getattr(c, "signal_meta", {}) or {}).get("post_ml_score", 0.0), 0.0),
            }
            for c, _ in selected_pairs
        ],
    }

    return ProdSelectionResult(
        selected_candidates=[c for c, _ in selected_pairs],
        selected_decisions=[d for _, d in selected_pairs],
        meta=meta,
    )
