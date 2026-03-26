from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation
from hf_core.strategy_side_weight_rules import resolve_post_ml_score


@dataclass
class StrategySideWeightOverlay:
    rules: dict[tuple[str, str], tuple[float, float, float]]

    def apply(
        self,
        *,
        allocation: Allocation,
        prev_allocation: Allocation | None = None,
    ) -> Allocation:
        alloc_meta = dict(getattr(allocation, "meta", {}) or {})
        legacy_opps = list(alloc_meta.get("legacy_opportunities", []) or [])

        prev_weights = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(prev_allocation, "weights", {}) or {}).items()
        }

        symbol_shape_meta: dict[str, dict] = {}
        for row in legacy_opps:
            symbol = str(row.get("symbol", "") or "")
            meta = dict(row.get("meta", {}) or {})
            side = str(row.get("side", meta.get("side", "flat")) or "flat").lower()
            strategy_id = str(row.get("strategy_id", meta.get("strategy_id", "")) or "").lower()

            row_meta = dict(meta)
            row_meta["side"] = str(side)
            row_meta["strategy_id"] = str(strategy_id)

            current = symbol_shape_meta.get(symbol)

            if current is None:
                symbol_shape_meta[symbol] = row_meta
                continue

            curr_score = resolve_post_ml_score(current)
            new_score = resolve_post_ml_score(row_meta)

            if new_score > curr_score:
                symbol_shape_meta[symbol] = row_meta

        shaped_weights = {}
        applied = {}

        fallback_details: dict[str, dict] = {}

        for sym, w in dict(getattr(allocation, "weights", {}) or {}).items():
            sym_key = str(sym)
            raw_w = float(w or 0.0)
            shape_meta = dict(symbol_shape_meta.get(sym_key, {}) or {})

            side = str(shape_meta.get("side", "flat") or "flat").lower()
            strategy_id = str(shape_meta.get("strategy_id", "") or "").lower()

            prev_w_signed = float(prev_weights.get(sym_key, 0.0) or 0.0)
            used_prev_side_fallback = False

            if side not in {"long", "short"}:
                if prev_w_signed < 0.0:
                    side = "short"
                    used_prev_side_fallback = True
                elif prev_w_signed > 0.0:
                    side = "long"
                    used_prev_side_fallback = True
                else:
                    side = "flat"

            if side == "short":
                w2 = -abs(raw_w)
            elif side == "long":
                w2 = abs(raw_w)
            else:
                w2 = 0.0

            if used_prev_side_fallback and abs(raw_w) > 1e-12:
                fallback_details[sym_key] = {
                    "fallback_used": True,
                    "prev_weight_signed": float(prev_w_signed),
                    "resolved_side": str(side),
                    "raw_weight_in": float(raw_w),
                    "weight_out_pre_rule": float(w2),
                }

            rule = self.rules.get((strategy_id, side))

            if rule is not None and abs(w2) > 1e-12:
                ref, min_mult, max_mult = rule
                post_ml = resolve_post_ml_score(shape_meta)
                if float(ref) > 0.0:
                    shape_mult = float(post_ml) / float(ref)
                    shape_mult = max(float(min_mult), min(float(max_mult), float(shape_mult)))
                    w2 = float(w2) * float(shape_mult)
                    applied[sym_key] = {
                        "strategy_id": str(strategy_id),
                        "side": str(side),
                        "ref": float(ref),
                        "min_mult": float(min_mult),
                        "max_mult": float(max_mult),
                        "post_ml_score": float(post_ml),
                        "shape_mult": float(shape_mult),
                        "raw_weight_in": float(raw_w),
                        "signed_weight_pre_rule": float(-abs(raw_w) if side == "short" else abs(raw_w) if side == "long" else 0.0),
                        "weight_out": float(w2),
                    }

            shaped_weights[sym_key] = float(w2)

        out_meta = dict(alloc_meta)
        out_meta["strategy_side_weight_overlay_applied"] = bool(applied)
        out_meta["strategy_side_weight_overlay_rules"] = {
            f"{k[0]}|{k[1]}": [float(v[0]), float(v[1]), float(v[2])]
            for k, v in self.rules.items()
        }
        out_meta["strategy_side_weight_overlay_details"] = applied
        out_meta["strategy_side_weight_overlay_prev_side_fallback"] = fallback_details

        return Allocation(
            weights=shaped_weights,
            meta=out_meta,
        )
