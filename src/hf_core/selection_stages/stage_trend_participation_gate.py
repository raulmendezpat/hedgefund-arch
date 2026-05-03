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


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(x)))


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


def _first_meta_float(meta: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        if k in meta and meta.get(k) is not None:
            return _safe_float(meta.get(k), default)
    return float(default)


class TrendParticipationGateStage:
    STAGE_NAME = "trend_participation_gate"

    def __init__(self, cfg: dict | None = None, profile: str = "research"):
        self.cfg = dict(cfg or {})
        self.profile = str(profile or "research")

    def _compute_score_penalty(
        self,
        *,
        tcfg: dict,
        side_allowed: bool,
        close_vs_ema_fast_ok: bool,
        ema_stack_ok: bool,
        ema_slope_ok: bool,
        adx_confirmed: bool,
        volume_confirmed: bool,
        volume_missing: bool,
        reasons: list[str],
    ) -> tuple[float, float]:
        score_weights = dict(tcfg.get("score_weights", {}) or {})
        penalty_weights = dict(tcfg.get("penalty_weights", {}) or {})

        w_close = _safe_float(score_weights.get("close_vs_ema_fast_ok", 0.20), 0.20)
        w_stack = _safe_float(score_weights.get("ema_stack_ok", 0.20), 0.20)
        w_slope = _safe_float(score_weights.get("ema_slope_ok", 0.30), 0.30)
        w_adx = _safe_float(score_weights.get("adx_confirmed", 0.15), 0.15)
        w_volume = _safe_float(score_weights.get("volume_confirmed", 0.15), 0.15)

        denom = max(1e-9, w_close + w_stack + w_slope + w_adx + w_volume)
        raw_score = (
            (w_close if close_vs_ema_fast_ok else 0.0)
            + (w_stack if ema_stack_ok else 0.0)
            + (w_slope if ema_slope_ok else 0.0)
            + (w_adx if adx_confirmed else 0.0)
            + (w_volume if volume_confirmed else 0.0)
        ) / denom
        raw_score = _clamp(raw_score)

        penalty = 0.0
        if not side_allowed:
            penalty += _safe_float(penalty_weights.get("side_not_allowed", 0.25), 0.25)

        for reason in reasons:
            penalty += _safe_float(penalty_weights.get(reason, 0.05), 0.05)

        if volume_missing:
            penalty += _safe_float(penalty_weights.get("volume_missing_extra", 0.0), 0.0)

        return float(raw_score), max(0.0, float(penalty))

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
                strategy_id=str(getattr(r, "strategy_id", "") or ""),
            )
            tcfg = dict(rcfg.get("trend_participation_gate", {}) or {})
            mode = str(tcfg.get("mode", "observe_only") or "observe_only").lower()

            allowed_sides = list(tcfg.get("allowed_sides", ["long", "short"]) or ["long", "short"])
            require_close_vs_ema_fast = _safe_bool(
                tcfg.get("require_close_vs_ema_fast", tcfg.get("require_trend_alignment", True)),
                True,
            )
            require_ema_stack = _safe_bool(
                tcfg.get("require_ema_stack", tcfg.get("require_trend_alignment", True)),
                True,
            )
            require_ema_slope = _safe_bool(tcfg.get("require_ema_slope", True), True)
            require_adx = _safe_bool(tcfg.get("require_adx", True), True)
            require_rvol = _safe_bool(
                tcfg.get("require_rvol", tcfg.get("require_rvol_expansion", False)),
                False,
            )
            allow_missing_rvol = _safe_bool(tcfg.get("allow_missing_rvol", True), True)

            min_adx = _safe_float(tcfg.get("min_adx", 18.0), 18.0)
            min_rvol20 = _safe_float(tcfg.get("min_rvol20", 1.0), 1.0)
            max_entry_bar_ret_pct_long = _safe_float(tcfg.get("max_entry_bar_ret_pct_long", 999.0), 999.0)
            max_entry_bar_ret_pct_short = _safe_float(tcfg.get("max_entry_bar_ret_pct_short", 999.0), 999.0)

            meta = dict(getattr(r, "meta", {}) or {})
            side = str(getattr(r, "side", "flat") or "flat").lower()

            close_field = str(tcfg.get("close_field", "close") or "close")
            ema_fast_field = str(tcfg.get("ema_fast_field", "ema_fast_runtime") or "ema_fast_runtime")
            ema_slow_field = str(tcfg.get("ema_slow_field", "ema_slow_runtime") or "ema_slow_runtime")
            ema_slope_field = str(tcfg.get("ema_slope_field", "ema_fast_slope_signed") or "ema_fast_slope_signed")
            adx_field = str(tcfg.get("adx_field", "adx_runtime") or "adx_runtime")
            rvol_field = str(tcfg.get("rvol_field", "rvol20") or "rvol20")
            entry_bar_ret_field = str(tcfg.get("entry_bar_ret_field", "entry_bar_ret_pct") or "entry_bar_ret_pct")

            close_v = _first_meta_float(meta, [close_field, "close", "close_now", "entry_close"], 0.0)
            ema_fast = _first_meta_float(meta, [ema_fast_field, "ema_fast_runtime", "ema20", "ema_20", "ema_fast"], 0.0)
            ema_slow = _first_meta_float(meta, [ema_slow_field, "ema_slow_runtime", "ema50", "ema_50", "ema_slow"], 0.0)
            ema_slope = _first_meta_float(meta, [ema_slope_field, "ema_fast_slope_signed"], 0.0)
            adx = _first_meta_float(meta, [adx_field, "adx_runtime", "adx14", "adx_14", "adx"], 0.0)
            rvol20 = _first_meta_float(meta, [rvol_field, "rvol20", "rvol_20", "relative_volume_20", "relative_volume"], 0.0)
            entry_bar_ret_pct = _first_meta_float(meta, [entry_bar_ret_field, "entry_bar_ret_pct", "bar_ret_pct"], 0.0)

            reasons = []
            hard_fail_reasons = []

            side_allowed = side in allowed_sides
            if not side_allowed:
                reasons.append("side_not_allowed")
                hard_fail_reasons.append("side_not_allowed")

            close_vs_ema_fast_ok = False
            ema_stack_ok = False
            ema_slope_ok = False

            if side == "long":
                close_vs_ema_fast_ok = close_v > ema_fast
                ema_stack_ok = ema_fast > ema_slow
                ema_slope_ok = ema_slope > 0.0
                if require_close_vs_ema_fast and not close_vs_ema_fast_ok:
                    reasons.append("close_below_ema_fast")
                if require_ema_stack and not ema_stack_ok:
                    reasons.append("ema_stack_not_long")
                if require_ema_slope and not ema_slope_ok:
                    reasons.append("ema_slope_not_positive")
                if entry_bar_ret_pct > max_entry_bar_ret_pct_long:
                    reasons.append("entry_bar_ret_too_high_long")
            elif side == "short":
                close_vs_ema_fast_ok = close_v < ema_fast
                ema_stack_ok = ema_fast < ema_slow
                ema_slope_ok = ema_slope < 0.0
                if require_close_vs_ema_fast and not close_vs_ema_fast_ok:
                    reasons.append("close_above_ema_fast")
                if require_ema_stack and not ema_stack_ok:
                    reasons.append("ema_stack_not_short")
                if require_ema_slope and not ema_slope_ok:
                    reasons.append("ema_slope_not_negative")
                if entry_bar_ret_pct > max_entry_bar_ret_pct_short:
                    reasons.append("entry_bar_ret_too_high_short")
            else:
                reasons.append("invalid_side")
                hard_fail_reasons.append("invalid_side")

            trend_confirmed = (
                (not require_close_vs_ema_fast or close_vs_ema_fast_ok)
                and (not require_ema_stack or ema_stack_ok)
                and (not require_ema_slope or ema_slope_ok)
                and ("invalid_side" not in hard_fail_reasons)
            )

            adx_confirmed = True
            if require_adx:
                adx_confirmed = adx >= min_adx
                if not adx_confirmed:
                    reasons.append("adx_below_min")

            volume_missing = rvol20 <= 0.0
            volume_confirmed = True
            if require_rvol:
                if volume_missing and allow_missing_rvol:
                    volume_confirmed = True
                    reasons.append("rvol_missing_observed_only")
                elif volume_missing and not allow_missing_rvol:
                    volume_confirmed = False
                    reasons.append("rvol_missing")
                    hard_fail_reasons.append("rvol_missing")
                else:
                    volume_confirmed = rvol20 >= min_rvol20
                    if not volume_confirmed:
                        reasons.append("rvol_below_min")
            else:
                if volume_missing:
                    reasons.append("rvol_missing_observed_only")
                elif rvol20 < min_rvol20:
                    reasons.append("rvol_below_min_observed_only")

            score, penalty = self._compute_score_penalty(
                tcfg=tcfg,
                side_allowed=side_allowed,
                close_vs_ema_fast_ok=close_vs_ema_fast_ok,
                ema_stack_ok=ema_stack_ok,
                ema_slope_ok=ema_slope_ok,
                adx_confirmed=adx_confirmed,
                volume_confirmed=volume_confirmed,
                volume_missing=volume_missing,
                reasons=reasons,
            )

            passed = bool(
                len(hard_fail_reasons) == 0
                and trend_confirmed
                and adx_confirmed
                and volume_confirmed
            )

            keep = True if mode in {"observe_only", "off", "bypass"} else bool(passed)
            if keep:
                kept.append(idx)

            stage_bucket = _ensure_stage_bucket(r, self.STAGE_NAME)
            stage_bucket.update(
                {
                    "mode": str(mode),
                    "pass": bool(passed),
                    "kept": bool(keep),
                    "score": float(score),
                    "penalty": float(penalty),
                    "reasons": list(reasons),
                    "hard_fail_reasons": list(hard_fail_reasons),
                    "inputs": {
                        "close": float(close_v),
                        "ema_fast": float(ema_fast),
                        "ema_slow": float(ema_slow),
                        "ema_slope": float(ema_slope),
                        "adx": float(adx),
                        "rvol20": float(rvol20),
                        "entry_bar_ret_pct": float(entry_bar_ret_pct),
                    },
                    "fields": {
                        "close_field": str(close_field),
                        "ema_fast_field": str(ema_fast_field),
                        "ema_slow_field": str(ema_slow_field),
                        "ema_slope_field": str(ema_slope_field),
                        "adx_field": str(adx_field),
                        "rvol_field": str(rvol_field),
                        "entry_bar_ret_field": str(entry_bar_ret_field),
                    },
                    "flags": {
                        "trend_confirmed": bool(trend_confirmed),
                        "close_vs_ema_fast_ok": bool(close_vs_ema_fast_ok),
                        "ema_stack_ok": bool(ema_stack_ok),
                        "ema_slope_ok": bool(ema_slope_ok),
                        "adx_confirmed": bool(adx_confirmed),
                        "volume_confirmed": bool(volume_confirmed),
                        "volume_missing": bool(volume_missing),
                        "side_allowed": bool(side_allowed),
                    },
                    "thresholds": {
                        "min_adx": float(min_adx),
                        "min_rvol20": float(min_rvol20),
                        "max_entry_bar_ret_pct_long": float(max_entry_bar_ret_pct_long),
                        "max_entry_bar_ret_pct_short": float(max_entry_bar_ret_pct_short),
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
                    "score": float(score),
                    "penalty": float(penalty),
                    "close": float(close_v),
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "ema_slope": float(ema_slope),
                    "adx": float(adx),
                    "rvol20": float(rvol20),
                    "entry_bar_ret_pct": float(entry_bar_ret_pct),
                    "close_field": str(close_field),
                    "ema_fast_field": str(ema_fast_field),
                    "ema_slow_field": str(ema_slow_field),
                    "ema_slope_field": str(ema_slope_field),
                    "adx_field": str(adx_field),
                    "rvol_field": str(rvol_field),
                    "entry_bar_ret_field": str(entry_bar_ret_field),
                    "close_vs_ema_fast_ok": bool(close_vs_ema_fast_ok),
                    "ema_stack_ok": bool(ema_stack_ok),
                    "ema_slope_ok": bool(ema_slope_ok),
                    "trend_confirmed": bool(trend_confirmed),
                    "adx_confirmed": bool(adx_confirmed),
                    "volume_confirmed": bool(volume_confirmed),
                    "volume_missing": bool(volume_missing),
                    "passed": bool(passed),
                    "kept": bool(keep),
                    "reasons": list(reasons),
                    "hard_fail_reasons": list(hard_fail_reasons),
                }
            )

        ctx.selected_idx = kept
        ctx.trace_rows.extend(trace_rows)
        ctx.meta["trend_participation_gate_kept"] = int(len(kept))
        return ctx
