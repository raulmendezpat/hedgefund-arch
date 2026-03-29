from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation


@dataclass
class ProductionLikeAllocationMlSizer:
    enabled: bool = False

    def apply(
        self,
        *,
        allocation: Allocation,
        selected_candidates,
    ) -> Allocation:
        alloc_meta = dict(getattr(allocation, "meta", {}) or {})
        base_w = {
            str(k): float(v or 0.0)
            for k, v in dict(getattr(allocation, "weights", {}) or {}).items()
        }

        if not bool(self.enabled):
            return Allocation(
                weights=dict(base_w),
                meta={
                    **alloc_meta,
                    "ml_position_sizing_applied": False,
                    "ml_position_size_mults": {},
                },
            )

        selected_by_symbol = {}
        for c in list(selected_candidates or []):
            sym = str(getattr(c, "symbol", "") or "")
            if not sym or sym in selected_by_symbol:
                continue
            selected_by_symbol[sym] = c

        out_w = {}
        applied = {}
        for sym, w in base_w.items():
            c = selected_by_symbol.get(sym)
            if c is None:
                out_w[sym] = float(w)
                continue

            sm = dict(getattr(c, "signal_meta", {}) or {})
            mult = sm.get("ml_position_size_mult", None)
            if mult is None:
                out_w[sym] = float(w)
                continue

            out_w[sym] = float(w) * float(mult)
            applied[sym] = float(mult)

        return Allocation(
            weights=out_w,
            meta={
                **alloc_meta,
                "ml_position_sizing_applied": bool(applied),
                "ml_position_size_mults": dict(applied),
            },
        )
