from __future__ import annotations

from dataclasses import dataclass

from hf.core.types import Allocation


def _apply_symbol_top_n_and_cluster_caps(
    weights: dict[str, float],
    *,
    symbol_cluster_map: dict[str, str] | None = None,
    cluster_cap_map: dict[str, float] | None = None,
    top_n_symbols: int | None = None,
) -> dict[str, float]:
    out = {str(k): float(v or 0.0) for k, v in (weights or {}).items()}

    if top_n_symbols is not None and int(top_n_symbols) > 0:
        ranked = sorted(
            [sym for sym, w in out.items() if abs(float(w or 0.0)) > 0.0],
            key=lambda sym: (abs(float(out.get(sym, 0.0) or 0.0)), str(sym)),
            reverse=True,
        )
        keep = set(ranked[: int(top_n_symbols)])
        for sym in list(out.keys()):
            if sym not in keep:
                out[sym] = 0.0

    symbol_cluster_map = dict(symbol_cluster_map or {})
    cluster_cap_map = dict(cluster_cap_map or {})

    if symbol_cluster_map and cluster_cap_map:
        cluster_abs_sum: dict[str, float] = {}
        for sym, w in out.items():
            cid = symbol_cluster_map.get(sym)
            if not cid:
                continue
            cluster_abs_sum[cid] = float(cluster_abs_sum.get(cid, 0.0) + abs(float(w or 0.0)))

        for cid, total_abs in cluster_abs_sum.items():
            cap = cluster_cap_map.get(cid, None)
            if cap is None:
                continue
            cap = float(cap)
            if cap <= 0.0 or total_abs <= cap:
                continue

            scale = float(cap / total_abs)
            for sym in list(out.keys()):
                if symbol_cluster_map.get(sym) == cid:
                    out[sym] = float(out[sym] * scale)

    return out


@dataclass
class ProductionLikeAllocationClusterControls:
    top_n_symbols: int = 0
    apply_cluster_caps: bool = False
    symbol_cluster_map: dict[str, str] | None = None
    cluster_cap_map: dict[str, float] | None = None

    def apply(
        self,
        *,
        allocation: Allocation,
    ) -> Allocation:
        alloc_meta = dict(getattr(allocation, "meta", {}) or {})
        in_w = {str(k): float(v or 0.0) for k, v in dict(getattr(allocation, "weights", {}) or {}).items()}

        out_w = _apply_symbol_top_n_and_cluster_caps(
            dict(in_w),
            symbol_cluster_map=(self.symbol_cluster_map if bool(self.apply_cluster_caps) else {}),
            cluster_cap_map=(self.cluster_cap_map if bool(self.apply_cluster_caps) else {}),
            top_n_symbols=(int(self.top_n_symbols) if int(self.top_n_symbols) > 0 else None),
        )

        return Allocation(
            weights=out_w,
            meta={
                **alloc_meta,
                "allocator_top_n_symbols": int(self.top_n_symbols or 0),
                "allocator_apply_cluster_caps": bool(self.apply_cluster_caps),
                "symbol_cluster_map": dict(self.symbol_cluster_map or {}),
                "cluster_cap_map": dict(self.cluster_cap_map or {}),
                "alloc_pre_cluster_controls": dict(in_w),
            },
        )
