from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import AllocatorContext


@dataclass
class CollectDeployableStep:
    def __call__(self, ctx: AllocatorContext) -> AllocatorContext:
        rows: list[dict[str, Any]] = []

        for c in list(ctx.candidates or []):
            side = str(c.side or "").lower()
            if side not in {"long", "short"}:
                continue

            base_weight = float(c.base_weight or 0.0)
            if abs(base_weight) <= 0.0:
                continue

            signed_weight = abs(base_weight) if side == "long" else -abs(base_weight)
            meta = dict(c.signal_meta or {})

            rows.append({
                "ts": int(c.ts),
                "symbol": str(c.symbol),
                "strategy_id": str(c.strategy_id),
                "side": side,
                "base_weight": float(base_weight),
                "signed_weight": float(signed_weight),
                "policy_band": str(meta.get("policy_band", "")),
                "policy_reason": str(meta.get("policy_reason", "")),
                "policy_size_mult": float(meta.get("policy_size_mult", 1.0) or 1.0),
                "policy_score": float(meta.get("policy_score", 0.0) or 0.0),
            })

        ctx.deployable_rows = rows
        ctx.meta["n_deployable_rows"] = len(rows)
        return ctx


@dataclass
class AggregateBySymbolStep:
    enable_symbol_netting: bool = True

    def __call__(self, ctx: AllocatorContext) -> AllocatorContext:
        out: dict[str, float] = {}

        for row in list(ctx.deployable_rows or []):
            sym = str(row["symbol"])
            signed_weight = float(row["signed_weight"])

            if self.enable_symbol_netting:
                out[sym] = out.get(sym, 0.0) + signed_weight
            else:
                # keep only additive gross by symbol-direction encoded in key
                side = str(row["side"])
                out[f"{sym}::{side}"] = out.get(f"{sym}::{side}", 0.0) + signed_weight

        ctx.symbol_raw_weights = out
        ctx.meta["symbol_raw_weights"] = dict(out)
        ctx.meta["enable_symbol_netting"] = bool(self.enable_symbol_netting)
        return ctx


@dataclass
class NormalizeToTargetExposureStep:
    target_exposure: float = 1.0

    def __call__(self, ctx: AllocatorContext) -> AllocatorContext:
        raw = dict(ctx.symbol_raw_weights or {})
        gross = sum(abs(v) for v in raw.values())

        if gross <= 0.0:
            ctx.symbol_weights = {}
            ctx.meta["gross_before_normalization"] = 0.0
            ctx.meta["normalization_scale"] = 0.0
            return ctx

        scale = float(self.target_exposure) / gross
        ctx.symbol_weights = {k: float(v) * scale for k, v in raw.items()}
        ctx.meta["gross_before_normalization"] = float(gross)
        ctx.meta["normalization_scale"] = float(scale)
        return ctx


@dataclass
class CapWeightsStep:
    symbol_cap: float = 0.35

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def __call__(self, ctx: AllocatorContext) -> AllocatorContext:
        weights = dict(ctx.symbol_weights or {})
        capped = {
            sym: self._clip(w, -float(self.symbol_cap), float(self.symbol_cap))
            for sym, w in weights.items()
        }

        gross_after_cap = sum(abs(v) for v in capped.values())
        ctx.symbol_weights = capped
        ctx.meta["gross_after_cap"] = float(gross_after_cap)
        ctx.meta["symbol_cap"] = float(self.symbol_cap)
        return ctx
