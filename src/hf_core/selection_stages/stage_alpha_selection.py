from __future__ import annotations

from .contracts import SelectionContext


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _ensure_stage_bucket(row, stage_name: str) -> dict:
    meta = getattr(row, "selection_meta", None)
    if not isinstance(meta, dict):
        meta = {}
        row.selection_meta = meta

    bucket = meta.get(stage_name)
    if not isinstance(bucket, dict):
        bucket = {}
        meta[stage_name] = bucket
    return bucket


class AlphaSelectionStage:
    STAGE_NAME = "alpha_selection"

    def __init__(self, cfg: dict | None = None, profile: str = "research"):
        self.cfg = dict(cfg or {})
        self.profile = str(profile or "research")

    def _profile_cfg(self) -> dict:
        profiles = self.cfg.get("profiles", {})
        prof = profiles.get(self.profile, {}) if isinstance(profiles, dict) else {}
        return dict(prof or {})

    def _compute_alpha_score(
        self,
        *,
        policy_score: float,
        p_win: float,
        expected_return: float,
        policy_size_mult: float,
        post_ml_score: float,
        competitive_score: float,
        contextual_score: float,
        contextual_penalty: float,
        regime_score_mult: float,
        regime_penalty: float,
        cfg: dict,
    ) -> tuple[float, float]:
        weights = dict(cfg.get("alpha_selection_score_weights", {}) or {})

        expected_return_scale = _safe_float(cfg.get("alpha_selection_expected_return_scale", 0.01), 0.01)
        size_mult_scale = _safe_float(cfg.get("alpha_selection_size_mult_scale", 1.0), 1.0)
        score_floor = _safe_float(cfg.get("alpha_selection_score_floor", 0.0), 0.0)
        score_cap = _safe_float(cfg.get("alpha_selection_score_cap", 1.0), 1.0)

        w_policy_score = _safe_float(weights.get("policy_score", 0.10), 0.10)
        w_p_win = _safe_float(weights.get("p_win", 0.20), 0.20)
        w_expected_return = _safe_float(weights.get("expected_return", 0.15), 0.15)
        w_policy_size_mult = _safe_float(weights.get("policy_size_mult", 0.05), 0.05)
        w_post_ml_score = _safe_float(weights.get("post_ml_score", 0.25), 0.25)
        w_competitive_score = _safe_float(weights.get("competitive_score", 0.10), 0.10)
        w_contextual_score = _safe_float(weights.get("contextual_score", 0.10), 0.10)
        w_regime_score_mult = _safe_float(weights.get("regime_score_mult", 0.05), 0.05)

        weight_sum = (
            w_policy_score
            + w_p_win
            + w_expected_return
            + w_policy_size_mult
            + w_post_ml_score
            + w_competitive_score
            + w_contextual_score
            + w_regime_score_mult
        )
        if weight_sum > 0:
            w_policy_score /= weight_sum
            w_p_win /= weight_sum
            w_expected_return /= weight_sum
            w_policy_size_mult /= weight_sum
            w_post_ml_score /= weight_sum
            w_competitive_score /= weight_sum
            w_contextual_score /= weight_sum
            w_regime_score_mult /= weight_sum

        policy_score_norm = _clamp(policy_score / max(1e-9, 0.00025))
        p_win_norm = _clamp((p_win - 0.50) / 0.10)
        expected_return_norm = _clamp(expected_return / expected_return_scale) if expected_return_scale > 0 else _clamp(expected_return)
        size_mult_norm = _clamp(policy_size_mult / size_mult_scale) if size_mult_scale > 0 else _clamp(policy_size_mult)
        post_ml_norm = _clamp(post_ml_score)
        competitive_norm = _clamp(competitive_score)
        contextual_norm = _clamp(contextual_score)
        regime_norm = _clamp(regime_score_mult)

        raw_score = (
            w_policy_score * policy_score_norm
            + w_p_win * p_win_norm
            + w_expected_return * expected_return_norm
            + w_policy_size_mult * size_mult_norm
            + w_post_ml_score * post_ml_norm
            + w_competitive_score * competitive_norm
            + w_contextual_score * contextual_norm
            + w_regime_score_mult * regime_norm
        )

        penalty = float(contextual_penalty) + float(regime_penalty)
        final_score = _clamp(raw_score - penalty, score_floor, score_cap)
        return float(final_score), float(penalty)

    def apply(self, ctx: SelectionContext) -> SelectionContext:
        if not ctx.rows:
            ctx.selected_idx = []
            return ctx

        cfg = self._profile_cfg()

        min_policy_score = float(cfg.get("alpha_selection_min_policy_score", 0.0))
        min_p_win = float(cfg.get("alpha_selection_min_p_win", 0.0))
        min_expected_return = float(cfg.get("alpha_selection_min_expected_return", 0.0))
        min_size_mult = float(cfg.get("alpha_selection_min_size_mult", 0.0))
        require_accept = bool(cfg.get("alpha_selection_require_accept", True))
        min_alpha_score = float(cfg.get("alpha_selection_min_alpha_score", 0.0))

        current = set(int(x) for x in list(ctx.selected_idx or []))
        kept = []
        trace_rows = []

        for r in ctx.rows:
            idx = int(r.idx)
            if idx not in current:
                continue

            accept = bool(getattr(r, "accept_in", False))
            policy_score = float(getattr(r, "policy_score", 0.0) or 0.0)
            p_win = float(getattr(r, "p_win", 0.0) or 0.0)
            expected_return = float(getattr(r, "expected_return", 0.0) or 0.0)
            policy_size_mult = float(getattr(r, "policy_size_mult", 0.0) or 0.0)
            post_ml_score = float(getattr(r, "post_ml_score", 0.0) or 0.0)
            competitive_score = float(getattr(r, "competitive_score", 0.0) or 0.0)

            selection_meta = getattr(r, "selection_meta", None)
            if not isinstance(selection_meta, dict):
                selection_meta = {}
                r.selection_meta = selection_meta

            contextual_meta = dict(selection_meta.get("contextual_eligibility", {}) or {})
            regime_meta = dict(selection_meta.get("strategy_regime_eligibility", {}) or {})

            contextual_score = _safe_float(contextual_meta.get("score", 0.0), 0.0)
            contextual_penalty = _safe_float(contextual_meta.get("penalty", 0.0), 0.0)
            regime_score_mult = _safe_float(regime_meta.get("score_mult", 0.0), 0.0)
            regime_penalty = _safe_float(regime_meta.get("penalty", 0.0), 0.0)

            alpha_score, alpha_penalty = self._compute_alpha_score(
                policy_score=policy_score,
                p_win=p_win,
                expected_return=expected_return,
                policy_size_mult=policy_size_mult,
                post_ml_score=post_ml_score,
                competitive_score=competitive_score,
                contextual_score=contextual_score,
                contextual_penalty=contextual_penalty,
                regime_score_mult=regime_score_mult,
                regime_penalty=regime_penalty,
                cfg=cfg,
            )

            reasons = []
            keep = True

            if require_accept and not accept:
                keep = False
                reasons.append("accept_false")
            if policy_score < min_policy_score:
                keep = False
                reasons.append("policy_score_below_min")
            if p_win < min_p_win:
                keep = False
                reasons.append("p_win_below_min")
            if expected_return < min_expected_return:
                keep = False
                reasons.append("expected_return_below_min")
            if policy_size_mult < min_size_mult:
                keep = False
                reasons.append("policy_size_mult_below_min")
            if alpha_score < min_alpha_score:
                keep = False
                reasons.append("alpha_score_below_min")

            if keep:
                kept.append(idx)

            stage_bucket = _ensure_stage_bucket(r, self.STAGE_NAME)
            stage_bucket.update(
                {
                    "pass": bool(keep),
                    "kept": bool(keep),
                    "score": float(alpha_score),
                    "penalty": float(alpha_penalty),
                    "reasons": list(reasons),
                    "inputs": {
                        "accept": bool(accept),
                        "policy_score": float(policy_score),
                        "p_win": float(p_win),
                        "expected_return": float(expected_return),
                        "policy_size_mult": float(policy_size_mult),
                        "post_ml_score": float(post_ml_score),
                        "competitive_score": float(competitive_score),
                        "contextual_score": float(contextual_score),
                        "contextual_penalty": float(contextual_penalty),
                        "regime_score_mult": float(regime_score_mult),
                        "regime_penalty": float(regime_penalty),
                    },
                }
            )

            trace_rows.append(
                {
                    "stage": self.STAGE_NAME,
                    "idx": idx,
                    "ts": int(r.ts),
                    "symbol": str(r.symbol),
                    "strategy_id": str(r.strategy_id),
                    "side": str(r.side),
                    "accept": bool(accept),
                    "policy_score": float(policy_score),
                    "p_win": float(p_win),
                    "expected_return": float(expected_return),
                    "policy_size_mult": float(policy_size_mult),
                    "post_ml_score": float(post_ml_score),
                    "competitive_score": float(competitive_score),
                    "contextual_score": float(contextual_score),
                    "contextual_penalty": float(contextual_penalty),
                    "regime_score_mult": float(regime_score_mult),
                    "regime_penalty": float(regime_penalty),
                    "alpha_score": float(alpha_score),
                    "alpha_penalty": float(alpha_penalty),
                    "kept": bool(keep),
                    "reasons": list(reasons),
                }
            )

        ctx.selected_idx = kept
        ctx.trace_rows.extend(trace_rows)
        ctx.meta["alpha_selection_kept"] = int(len(kept))
        return ctx
