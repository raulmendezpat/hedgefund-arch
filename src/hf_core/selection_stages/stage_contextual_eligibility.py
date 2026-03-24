from __future__ import annotations

from .contracts import SelectionContext
from .config import resolve_profile_config


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


class ContextualEligibilityStage:
    STAGE_NAME = "contextual_eligibility"

    def __init__(self, cfg: dict | None = None, profile: str = "research"):
        self.cfg = dict(cfg or {})
        self.profile = str(profile or "research")

    def _compute_contextual_score(
        self,
        *,
        p_win: float,
        expected_return: float,
        post_ml_score: float,
        competitive_score: float,
        policy_score: float,
        policy_size_mult: float,
        passed: bool,
        reasons: list[str],
        ccfg: dict,
    ) -> tuple[float, float]:
        score_weights = dict(ccfg.get("score_weights", {}) or {})
        penalty_weights = dict(ccfg.get("penalty_weights", {}) or {})

        expected_return_scale = _safe_float(ccfg.get("expected_return_scale", 0.01), 0.01)
        size_mult_scale = _safe_float(ccfg.get("size_mult_scale", 1.0), 1.0)
        score_floor = _safe_float(ccfg.get("score_floor", 0.0), 0.0)
        score_cap = _safe_float(ccfg.get("score_cap", 1.0), 1.0)

        w_p_win = _safe_float(score_weights.get("p_win", 0.35), 0.35)
        w_expected_return = _safe_float(score_weights.get("expected_return", 0.20), 0.20)
        w_post_ml_score = _safe_float(score_weights.get("post_ml_score", 0.15), 0.15)
        w_competitive_score = _safe_float(score_weights.get("competitive_score", 0.10), 0.10)
        w_policy_score = _safe_float(score_weights.get("policy_score", 0.15), 0.15)
        w_policy_size_mult = _safe_float(score_weights.get("policy_size_mult", 0.05), 0.05)

        pwin_norm = _clamp(p_win)
        er_norm = _clamp(expected_return / expected_return_scale) if expected_return_scale > 0 else _clamp(expected_return)
        post_ml_norm = _clamp(post_ml_score)
        competitive_norm = _clamp(competitive_score)
        policy_norm = _clamp(policy_score)
        size_mult_norm = _clamp(policy_size_mult / size_mult_scale) if size_mult_scale > 0 else _clamp(policy_size_mult)

        raw_score = (
            w_p_win * pwin_norm
            + w_expected_return * er_norm
            + w_post_ml_score * post_ml_norm
            + w_competitive_score * competitive_norm
            + w_policy_score * policy_norm
            + w_policy_size_mult * size_mult_norm
        )

        penalty = 0.0
        if not passed:
            penalty += _safe_float(penalty_weights.get("base_fail", 0.15), 0.15)

        for reason in reasons:
            penalty += _safe_float(penalty_weights.get(reason, 0.05), 0.05)

        final_score = _clamp(raw_score - penalty, score_floor, score_cap)
        return float(final_score), float(penalty)

    def apply(self, ctx: SelectionContext) -> SelectionContext:
        if not ctx.rows:
            ctx.selected_idx = []
            return ctx

        current = set(int(x) for x in list(ctx.selected_idx or []))
        kept = []
        trace_rows = []

        for r in ctx.rows:
            idx = int(r.idx)
            if idx not in current:
                continue

            rcfg = resolve_profile_config(
                self.cfg,
                symbol=str(r.symbol),
                side=str(r.side),
                profile=self.profile,
            )
            ccfg = dict(rcfg.get("contextual_eligibility", {}) or {})
            mode = str(ccfg.get("mode", "observe_only") or "observe_only").lower()

            require_accept = bool(ccfg.get("require_accept", True))
            min_p_win = _safe_float(ccfg.get("min_p_win", 0.0), 0.0)
            min_expected_return = _safe_float(ccfg.get("min_expected_return", 0.0), 0.0)
            min_policy_score = _safe_float(ccfg.get("min_policy_score", 0.0), 0.0)
            min_size_mult = _safe_float(ccfg.get("min_size_mult", 0.0), 0.0)
            allowed_sides = list(ccfg.get("allowed_sides", ["long", "short"]) or ["long", "short"])

            accept_in = bool(getattr(r, "accept_in", False))
            policy_score = _safe_float(getattr(r, "policy_score", 0.0), 0.0)
            p_win = _safe_float(getattr(r, "p_win", 0.0), 0.0)
            expected_return = _safe_float(getattr(r, "expected_return", 0.0), 0.0)
            policy_size_mult = _safe_float(getattr(r, "policy_size_mult", 0.0), 0.0)
            post_ml_score = _safe_float(getattr(r, "post_ml_score", 0.0), 0.0)
            competitive_score = _safe_float(getattr(r, "competitive_score", 0.0), 0.0)
            side = str(getattr(r, "side", "flat") or "flat").lower()

            reasons = []
            passed = True

            if require_accept and not accept_in:
                passed = False
                reasons.append("accept_in_false")
            if side not in allowed_sides:
                passed = False
                reasons.append("side_not_allowed")
            if p_win < min_p_win:
                passed = False
                reasons.append("p_win_below_min")
            if expected_return < min_expected_return:
                passed = False
                reasons.append("expected_return_below_min")
            if policy_score < min_policy_score:
                passed = False
                reasons.append("policy_score_below_min")
            if policy_size_mult < min_size_mult:
                passed = False
                reasons.append("policy_size_mult_below_min")

            contextual_score, contextual_penalty = self._compute_contextual_score(
                p_win=p_win,
                expected_return=expected_return,
                post_ml_score=post_ml_score,
                competitive_score=competitive_score,
                policy_score=policy_score,
                policy_size_mult=policy_size_mult,
                passed=passed,
                reasons=reasons,
                ccfg=ccfg,
            )

            keep = True if mode == "observe_only" else bool(passed)

            if keep:
                kept.append(idx)

            stage_bucket = _ensure_stage_bucket(r, self.STAGE_NAME)
            stage_bucket.update(
                {
                    "mode": str(mode),
                    "pass": bool(passed),
                    "kept": bool(keep),
                    "score": float(contextual_score),
                    "penalty": float(contextual_penalty),
                    "reasons": list(reasons),
                    "inputs": {
                        "accept_in": bool(accept_in),
                        "policy_score": float(policy_score),
                        "p_win": float(p_win),
                        "expected_return": float(expected_return),
                        "policy_size_mult": float(policy_size_mult),
                        "post_ml_score": float(post_ml_score),
                        "competitive_score": float(competitive_score),
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
                    "side": str(side),
                    "mode": str(mode),
                    "accept_in": bool(accept_in),
                    "policy_score": float(policy_score),
                    "p_win": float(p_win),
                    "expected_return": float(expected_return),
                    "policy_size_mult": float(policy_size_mult),
                    "post_ml_score": float(post_ml_score),
                    "competitive_score": float(competitive_score),
                    "contextual_score": float(contextual_score),
                    "contextual_penalty": float(contextual_penalty),
                    "passed": bool(passed),
                    "kept": bool(keep),
                    "reasons": list(reasons),
                    "min_p_win": float(min_p_win),
                    "min_expected_return": float(min_expected_return),
                    "min_policy_score": float(min_policy_score),
                    "min_size_mult": float(min_size_mult),
                }
            )

        ctx.selected_idx = kept
        ctx.trace_rows.extend(trace_rows)
        ctx.meta["contextual_eligibility_kept"] = int(len(kept))
        return ctx
