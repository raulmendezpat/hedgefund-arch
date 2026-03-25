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


def _safe_bool(x, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _safe_str(x, default: str = "") -> str:
    if x is None:
        return str(default)
    return str(x)


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


class StrategyRegimeEligibilityStage:
    STAGE_NAME = "strategy_regime_eligibility"

    def __init__(self, cfg: dict | None = None, profile: str = "research"):
        self.cfg = dict(cfg or {})
        self.profile = str(profile or "research")

    def _compute_regime_score_mult(
        self,
        *,
        breadth: float,
        conviction: float,
        avg_pwin: float,
        avg_strength: float,
        risk_scale: float,
        regime_on: bool,
        passed: bool,
        reasons: list[str],
        scfg: dict,
    ) -> tuple[float, float]:
        score_weights = dict(scfg.get("score_weights", {}) or {})
        penalty_weights = dict(scfg.get("penalty_weights", {}) or {})

        w_breadth = _safe_float(score_weights.get("portfolio_breadth", 0.25), 0.25)
        w_conviction = _safe_float(score_weights.get("portfolio_conviction", 0.30), 0.30)
        w_avg_pwin = _safe_float(score_weights.get("portfolio_avg_pwin", 0.20), 0.20)
        w_avg_strength = _safe_float(score_weights.get("portfolio_avg_strength", 0.15), 0.15)
        w_risk_scale = _safe_float(score_weights.get("portfolio_risk_scale", 0.10), 0.10)

        breadth_norm = _clamp(breadth)
        conviction_norm = _clamp(conviction)
        avg_pwin_norm = _clamp(avg_pwin)
        avg_strength_norm = _clamp(avg_strength)
        risk_scale_norm = _clamp(risk_scale)

        raw = (
            w_breadth * breadth_norm
            + w_conviction * conviction_norm
            + w_avg_pwin * avg_pwin_norm
            + w_avg_strength * avg_strength_norm
            + w_risk_scale * risk_scale_norm
        )

        penalty = 0.0
        if not regime_on:
            penalty += _safe_float(penalty_weights.get("regime_off", 0.10), 0.10)
        if not passed:
            penalty += _safe_float(penalty_weights.get("base_fail", 0.15), 0.15)
        for reason in reasons:
            penalty += _safe_float(penalty_weights.get(reason, 0.05), 0.05)

        final_mult = _clamp(raw - penalty, _safe_float(scfg.get("score_floor", 0.0), 0.0), _safe_float(scfg.get("score_cap", 1.0), 1.0))
        return float(final_mult), float(penalty)

    def apply(self, ctx: SelectionContext) -> SelectionContext:
        if not ctx.rows:
            ctx.selected_idx = []
            return ctx

        current = set(int(x) for x in list(ctx.selected_idx or []))
        kept = []
        trace_rows = []

        portfolio_meta = ctx.meta if isinstance(ctx.meta, dict) else {}

        for r in ctx.rows:
            idx = int(r.idx)
            if idx not in current:
                continue

            row_meta = getattr(r, "meta", None)
            if not isinstance(row_meta, dict):
                row_meta = {}
                r.meta = row_meta

            rcfg = resolve_profile_config(
                self.cfg,
                symbol=str(r.symbol),
                side=str(r.side),
                profile=self.profile,
                strategy_id=str(getattr(r, "strategy_id", "") or ""),
            )
            scfg = dict(rcfg.get("strategy_regime_eligibility", {}) or {})
            mode = str(scfg.get("mode", "observe_only") or "observe_only").lower()

            allowed_regimes = list(scfg.get("allowed_regimes", []) or [])
            min_portfolio_breadth = _safe_float(scfg.get("min_portfolio_breadth", 0.0), 0.0)
            min_portfolio_conviction = _safe_float(scfg.get("min_portfolio_conviction", 0.0), 0.0)
            min_portfolio_avg_pwin = _safe_float(scfg.get("min_portfolio_avg_pwin", 0.0), 0.0)
            min_portfolio_avg_strength = _safe_float(scfg.get("min_portfolio_avg_strength", 0.0), 0.0)
            min_portfolio_risk_scale = _safe_float(scfg.get("min_portfolio_risk_scale", 0.0), 0.0)
            require_regime_on = bool(scfg.get("require_regime_on", False))

            portfolio_regime = _safe_str(
                row_meta.get("portfolio_regime", portfolio_meta.get("portfolio_regime", "")),
                "",
            ).lower()
            portfolio_breadth = _safe_float(
                row_meta.get("portfolio_breadth", portfolio_meta.get("portfolio_breadth", 0.0)),
                0.0,
            )
            portfolio_conviction = _safe_float(
                row_meta.get("portfolio_conviction", portfolio_meta.get("portfolio_conviction", 0.0)),
                0.0,
            )
            portfolio_avg_pwin = _safe_float(
                row_meta.get("portfolio_avg_pwin", portfolio_meta.get("portfolio_avg_pwin", 0.0)),
                0.0,
            )
            portfolio_avg_strength = _safe_float(
                row_meta.get("portfolio_avg_strength", portfolio_meta.get("portfolio_avg_strength", 0.0)),
                0.0,
            )
            portfolio_risk_scale = _safe_float(
                row_meta.get("portfolio_risk_scale_by_symbol", portfolio_meta.get("portfolio_risk_scale_by_symbol", 1.0)),
                1.0,
            )
            regime_on = _safe_bool(
                row_meta.get("regime_on_by_symbol", portfolio_meta.get("regime_on_by_symbol", True)),
                True,
            )

            reasons = []
            passed = True

            if allowed_regimes and portfolio_regime not in {str(x).lower() for x in allowed_regimes}:
                passed = False
                reasons.append("regime_not_allowed")
            if portfolio_breadth < min_portfolio_breadth:
                passed = False
                reasons.append("portfolio_breadth_below_min")
            if portfolio_conviction < min_portfolio_conviction:
                passed = False
                reasons.append("portfolio_conviction_below_min")
            if portfolio_avg_pwin < min_portfolio_avg_pwin:
                passed = False
                reasons.append("portfolio_avg_pwin_below_min")
            if portfolio_avg_strength < min_portfolio_avg_strength:
                passed = False
                reasons.append("portfolio_avg_strength_below_min")
            if portfolio_risk_scale < min_portfolio_risk_scale:
                passed = False
                reasons.append("portfolio_risk_scale_below_min")
            if require_regime_on and not regime_on:
                passed = False
                reasons.append("regime_off_by_symbol")

            regime_score_mult, regime_penalty = self._compute_regime_score_mult(
                breadth=portfolio_breadth,
                conviction=portfolio_conviction,
                avg_pwin=portfolio_avg_pwin,
                avg_strength=portfolio_avg_strength,
                risk_scale=portfolio_risk_scale,
                regime_on=regime_on,
                passed=passed,
                reasons=reasons,
                scfg=scfg,
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
                    "score_mult": float(regime_score_mult),
                    "penalty": float(regime_penalty),
                    "reasons": list(reasons),
                    "inputs": {
                        "portfolio_regime": str(portfolio_regime),
                        "portfolio_breadth": float(portfolio_breadth),
                        "portfolio_conviction": float(portfolio_conviction),
                        "portfolio_avg_pwin": float(portfolio_avg_pwin),
                        "portfolio_avg_strength": float(portfolio_avg_strength),
                        "portfolio_risk_scale_by_symbol": float(portfolio_risk_scale),
                        "regime_on_by_symbol": bool(regime_on),
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
                    "mode": str(mode),
                    "portfolio_regime": str(portfolio_regime),
                    "portfolio_breadth": float(portfolio_breadth),
                    "portfolio_conviction": float(portfolio_conviction),
                    "portfolio_avg_pwin": float(portfolio_avg_pwin),
                    "portfolio_avg_strength": float(portfolio_avg_strength),
                    "portfolio_risk_scale_by_symbol": float(portfolio_risk_scale),
                    "regime_on_by_symbol": bool(regime_on),
                    "regime_score_mult": float(regime_score_mult),
                    "regime_penalty": float(regime_penalty),
                    "passed": bool(passed),
                    "kept": bool(keep),
                    "reasons": list(reasons),
                }
            )

        ctx.selected_idx = kept
        ctx.trace_rows.extend(trace_rows)
        ctx.meta["strategy_regime_eligibility_kept"] = int(len(kept))
        return ctx
