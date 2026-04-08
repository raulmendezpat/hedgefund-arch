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


class MarketGuardStage:
    STAGE_NAME = "market_guard"

    def __init__(self, cfg: dict | None = None, profile: str = "research"):
        self.cfg = dict(cfg or {})
        self.profile = str(profile or "research")

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
            gcfg = dict(rcfg.get("market_guard", {}) or {})
            mode = str(gcfg.get("mode", "observe_only") or "observe_only").lower()
            allowed_sides = list(gcfg.get("allowed_sides", ["long", "short"]) or ["long", "short"])

            min_ema_gap = _safe_float(gcfg.get("min_ema_gap", 0.0), 0.0)
            ema_gap_field = str(gcfg.get("ema_gap_field", "ema_gap_fast_slow") or "ema_gap_fast_slow")

            min_atrp = _safe_float(gcfg.get("min_atrp", 0.0), 0.0)
            max_atrp = _safe_float(gcfg.get("max_atrp", 0.0), 0.0)
            min_bb_width = _safe_float(gcfg.get("min_bb_width", 0.0), 0.0)
            max_bb_width = _safe_float(gcfg.get("max_bb_width", 0.0), 0.0)
            min_adx = _safe_float(gcfg.get("min_adx", 0.0), 0.0)
            max_adx = _safe_float(gcfg.get("max_adx", 0.0), 0.0)

            max_ema_pullback_dist = _safe_float(gcfg.get("max_ema_pullback_dist", 0.0), 0.0)
            ema_fast_field = str(gcfg.get("ema_fast_field", "ema_fast") or "ema_fast")
            ema_slow_field = str(gcfg.get("ema_slow_field", "ema_slow") or "ema_slow")
            close_field = str(gcfg.get("close_field", "close") or "close")
            open_field = str(gcfg.get("open_field", "open") or "open")

            require_setup_window = bool(gcfg.get("require_setup_window", False))
            require_bb_rsi_setup = bool(gcfg.get("require_bb_rsi_setup", False))
            require_directional_close = bool(gcfg.get("require_directional_close", False))
            require_trend_alignment = bool(gcfg.get("require_trend_alignment", False))
            require_donchian_break = bool(gcfg.get("require_donchian_break", False))
            use_rsi_exhaustion_guard = bool(gcfg.get("use_rsi_exhaustion_guard", False))
            use_extension_guard = bool(gcfg.get("use_extension_guard", False))

            rsi_field = str(gcfg.get("rsi_field", "rsi") or "rsi")
            bb_low_field = str(gcfg.get("bb_low_field", "bb_low") or "bb_low")
            bb_up_field = str(gcfg.get("bb_up_field", "bb_up") or "bb_up")
            bb_width_field = str(gcfg.get("bb_width_field", "bb_width") or "bb_width")
            donchian_high_field = str(gcfg.get("donchian_high_field", "donchian_high") or "donchian_high")
            donchian_low_field = str(gcfg.get("donchian_low_field", "donchian_low") or "donchian_low")

            extension_ref_field_up = str(gcfg.get("extension_ref_field_up", bb_up_field) or bb_up_field)
            extension_ref_field_down = str(gcfg.get("extension_ref_field_down", bb_low_field) or bb_low_field)

            rsi_long_min = _safe_float(gcfg.get("rsi_long_min", 0.0), 0.0)
            rsi_long_max = _safe_float(gcfg.get("rsi_long_max", 0.0), 0.0)
            rsi_short_min = _safe_float(gcfg.get("rsi_short_min", 0.0), 0.0)
            rsi_short_max = _safe_float(gcfg.get("rsi_short_max", 0.0), 0.0)
            max_breakout_extension_pct = _safe_float(gcfg.get("max_breakout_extension_pct", 0.0), 0.0)
            max_breakdown_extension_pct = _safe_float(gcfg.get("max_breakdown_extension_pct", 0.0), 0.0)

            meta = dict(getattr(r, "meta", {}) or {})
            side = str(getattr(r, "side", "flat") or "flat").lower()
            adx = _safe_float(meta.get("adx", 0.0), 0.0)
            atrp = _safe_float(meta.get("atrp", 0.0), 0.0)
            rsi = _safe_float(meta.get(rsi_field, 0.0), 0.0)
            bb_low = _safe_float(meta.get(bb_low_field, 0.0), 0.0)
            bb_up = _safe_float(meta.get(bb_up_field, 0.0), 0.0)
            bb_width = _safe_float(meta.get(bb_width_field, 0.0), 0.0)

            ema_gap = abs(_safe_float(meta.get(ema_gap_field, 0.0), 0.0))
            ema_fast = _safe_float(meta.get(ema_fast_field, 0.0), 0.0)
            ema_slow = _safe_float(meta.get(ema_slow_field, 0.0), 0.0)
            open_v = _safe_float(meta.get(open_field, meta.get("open", 0.0)), 0.0)
            close_v = _safe_float(meta.get(close_field, meta.get("close", 0.0)), 0.0)

            extension_up_ref = _safe_float(meta.get(extension_ref_field_up, meta.get(bb_up_field, 0.0)), 0.0)
            extension_down_ref = _safe_float(meta.get(extension_ref_field_down, meta.get(bb_low_field, 0.0)), 0.0)
            donchian_high = _safe_float(meta.get(donchian_high_field, 0.0), 0.0)
            donchian_low = _safe_float(meta.get(donchian_low_field, 0.0), 0.0)

            breakout_extension_pct = 0.0
            if extension_up_ref > 0.0:
                breakout_extension_pct = max(0.0, (close_v / max(extension_up_ref, 1e-12)) - 1.0)

            breakdown_extension_pct = 0.0
            if extension_down_ref > 0.0:
                breakdown_extension_pct = max(0.0, 1.0 - (close_v / max(extension_down_ref, 1e-12)))

            ema_pullback_dist = 0.0
            if close_v != 0.0:
                ema_pullback_dist = abs(close_v - ema_fast) / max(abs(close_v), 1e-12)

            reasons = []
            passed = True

            if side not in allowed_sides:
                passed = False
                reasons.append("side_not_allowed")
            if min_atrp > 0.0 and atrp < min_atrp:
                passed = False
                reasons.append("atrp_below_min")
            if max_atrp > 0.0 and atrp > max_atrp:
                passed = False
                reasons.append("atrp_above_max")

            if min_bb_width > 0.0 and bb_width < min_bb_width:
                passed = False
                reasons.append("bb_width_below_min")
            if max_bb_width > 0.0 and bb_width > max_bb_width:
                passed = False
                reasons.append("bb_width_above_max")

            if min_adx > 0.0 and adx < min_adx:
                passed = False
                reasons.append("adx_below_min")
            if max_adx > 0.0 and adx > max_adx:
                passed = False
                reasons.append("adx_above_max")

            if min_ema_gap > 0.0 and ema_gap < min_ema_gap:
                passed = False
                reasons.append("ema_gap_below_min")
            if max_ema_pullback_dist > 0.0 and ema_pullback_dist > max_ema_pullback_dist:
                passed = False
                reasons.append("too_far_from_ema_fast")

            if require_setup_window:
                trend_up = ema_fast > ema_slow
                trend_down = ema_fast < ema_slow

                setup_ok = False
                if side == "long":
                    setup_ok = trend_up and (rsi >= rsi_long_min) and (rsi <= rsi_long_max)
                elif side == "short":
                    setup_ok = trend_down and (rsi >= rsi_short_min) and (rsi <= rsi_short_max)

                if not setup_ok:
                    passed = False
                    reasons.append("no_setup")

            if require_bb_rsi_setup:
                setup_ok = False
                if side == "long":
                    setup_ok = (close_v <= bb_low) and (rsi <= rsi_long_max)
                elif side == "short":
                    setup_ok = (close_v >= bb_up) and (rsi >= rsi_short_min)

                if not setup_ok:
                    passed = False
                    reasons.append("no_setup")

            if require_directional_close:
                directional_ok = True
                if side == "long":
                    directional_ok = close_v > open_v
                elif side == "short":
                    directional_ok = close_v < open_v
                if not directional_ok:
                    passed = False
                    reasons.append("non_directional_bar")

            if require_trend_alignment:
                trend_ok = True
                if side == "long":
                    trend_ok = (close_v > ema_fast) and (ema_fast > ema_slow)
                elif side == "short":
                    trend_ok = (close_v < ema_fast) and (ema_fast < ema_slow)
                if not trend_ok:
                    passed = False
                    reasons.append("trend_misaligned")

            if require_donchian_break:
                donchian_ok = True
                if side == "long":
                    donchian_ok = (donchian_high > 0.0) and (close_v > donchian_high)
                elif side == "short":
                    donchian_ok = (donchian_low > 0.0) and (close_v < donchian_low)
                if not donchian_ok:
                    passed = False
                    reasons.append("no_donchian_break")

            if use_rsi_exhaustion_guard:
                exhaustion_ok = True
                if side == "long" and rsi_long_max > 0.0:
                    exhaustion_ok = rsi <= rsi_long_max
                elif side == "short" and rsi_short_min > 0.0:
                    exhaustion_ok = rsi >= rsi_short_min
                if not exhaustion_ok:
                    passed = False
                    reasons.append("rsi_exhausted")

            if use_extension_guard:
                extension_ok = True
                if side == "long" and max_breakout_extension_pct > 0.0:
                    extension_ok = breakout_extension_pct <= max_breakout_extension_pct
                elif side == "short" and max_breakdown_extension_pct > 0.0:
                    extension_ok = breakdown_extension_pct <= max_breakdown_extension_pct
                if not extension_ok:
                    passed = False
                    reasons.append("overextended")

            keep = True if mode == "observe_only" else bool(passed)
            if keep:
                kept.append(idx)

            stage_bucket = _ensure_stage_bucket(r, self.STAGE_NAME)
            stage_bucket.update(
                {
                    "mode": str(mode),
                    "pass": bool(passed),
                    "kept": bool(keep),
                    "reasons": list(reasons),
                    "inputs": {
                        "adx": float(adx),
                        "atrp": float(atrp),
                        "rsi": float(rsi),
                        "bb_low": float(bb_low),
                        "bb_up": float(bb_up),
                        "bb_width": float(bb_width),
                        "ema_gap": float(ema_gap),
                        "ema_gap_field": str(ema_gap_field),
                        "ema_fast": float(ema_fast),
                        "ema_slow": float(ema_slow),
                        "open": float(open_v),
                        "close": float(close_v),
                        "donchian_high": float(donchian_high),
                        "donchian_low": float(donchian_low),
                        "breakout_extension_pct": float(breakout_extension_pct),
                        "breakdown_extension_pct": float(breakdown_extension_pct),
                        "ema_pullback_dist": float(ema_pullback_dist),
                        "ema_fast_field": str(ema_fast_field),
                        "ema_slow_field": str(ema_slow_field),
                        "close_field": str(close_field),
                        "bb_low_field": str(bb_low_field),
                        "bb_up_field": str(bb_up_field),
                        "bb_width_field": str(bb_width_field),
                        "require_setup_window": bool(require_setup_window),
                        "require_bb_rsi_setup": bool(require_bb_rsi_setup),
                        "require_directional_close": bool(require_directional_close),
                        "require_trend_alignment": bool(require_trend_alignment),
                        "require_donchian_break": bool(require_donchian_break),
                        "use_rsi_exhaustion_guard": bool(use_rsi_exhaustion_guard),
                        "use_extension_guard": bool(use_extension_guard),
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
                    "adx": float(adx),
                    "atrp": float(atrp),
                    "rsi": float(rsi),
                    "bb_low": float(bb_low),
                    "bb_up": float(bb_up),
                    "bb_width": float(bb_width),
                    "ema_gap": float(ema_gap),
                    "ema_gap_field": str(ema_gap_field),
                    "ema_fast": float(ema_fast),
                    "ema_slow": float(ema_slow),
                    "open": float(open_v),
                    "close": float(close_v),
                    "donchian_high": float(donchian_high),
                    "donchian_low": float(donchian_low),
                    "breakout_extension_pct": float(breakout_extension_pct),
                    "breakdown_extension_pct": float(breakdown_extension_pct),
                    "ema_pullback_dist": float(ema_pullback_dist),
                    "ema_fast_field": str(ema_fast_field),
                    "ema_slow_field": str(ema_slow_field),
                    "close_field": str(close_field),
                    "bb_low_field": str(bb_low_field),
                    "bb_up_field": str(bb_up_field),
                    "bb_width_field": str(bb_width_field),
                    "require_setup_window": bool(require_setup_window),
                    "require_bb_rsi_setup": bool(require_bb_rsi_setup),
                    "require_directional_close": bool(require_directional_close),
                    "require_trend_alignment": bool(require_trend_alignment),
                    "require_donchian_break": bool(require_donchian_break),
                    "use_rsi_exhaustion_guard": bool(use_rsi_exhaustion_guard),
                    "use_extension_guard": bool(use_extension_guard),
                    "rsi_field": str(rsi_field),
                    "rsi_long_min": float(rsi_long_min),
                    "rsi_long_max": float(rsi_long_max),
                    "rsi_short_min": float(rsi_short_min),
                    "rsi_short_max": float(rsi_short_max),
                    "min_atrp": float(min_atrp),
                    "max_atrp": float(max_atrp),
                    "min_bb_width": float(min_bb_width),
                    "max_bb_width": float(max_bb_width),
                    "min_adx": float(min_adx),
                    "max_adx": float(max_adx),
                    "min_ema_gap": float(min_ema_gap),
                    "max_ema_pullback_dist": float(max_ema_pullback_dist),
                    "passed": bool(passed),
                    "kept": bool(keep),
                    "reasons": list(reasons),
                }
            )

        ctx.selected_idx = kept
        ctx.trace_rows.extend(trace_rows)
        ctx.meta["market_guard_kept"] = int(len(kept))
        return ctx
