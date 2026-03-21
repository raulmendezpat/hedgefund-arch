from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf_core.contracts import OpportunityCandidate
from hf_core.allocator_parts import AllocatorContext, AllocatorFactory
from hf_core.execution_projection import ProjectionFactory


@dataclass
class Allocation:
    weights: dict[str, float]
    intents: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class Allocator:
    def __init__(
        self,
        *,
        target_exposure: float = 1.0,
        symbol_cap: float = 0.35,
        profile: str = "symbol_net",
        projection_profile: str = "net_symbol",
    ):
        self.target_exposure = float(target_exposure)
        self.symbol_cap = float(symbol_cap)
        self.profile = str(profile)
        self.projection_profile = str(projection_profile)

    def allocate_intents(
        self,
        *,
        candidates: list[OpportunityCandidate],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not candidates:
            return [], {
                "case": "no_inputs",
                "n_candidates": 0,
                "target_exposure": self.target_exposure,
                "symbol_cap": self.symbol_cap,
                "allocator_profile": self.profile,
            }

        pipeline = AllocatorFactory(profile=self.profile).build(
            target_exposure=float(self.target_exposure),
            symbol_cap=float(self.symbol_cap),
        )

        ctx = AllocatorContext(candidates=list(candidates or []))
        ctx = pipeline.run(ctx)

        intents: list[dict[str, Any]] = []
        symbol_weights = dict(ctx.symbol_weights or {})
        deployable_rows = list(ctx.deployable_rows or [])

        if str(self.profile).lower() == "symbol_net":
            # repartir el peso final del símbolo entre intents del mismo símbolo,
            # proporcional al signed_weight bruto absoluto de cada intent
            rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
            for row in deployable_rows:
                sym = str(row.get("symbol", "") or "")
                rows_by_symbol.setdefault(sym, []).append(row)

            for symbol, rows in rows_by_symbol.items():
                if symbol not in symbol_weights:
                    continue

                final_symbol_weight = float(symbol_weights.get(symbol, 0.0) or 0.0)
                gross_abs = sum(abs(float(r.get("signed_weight", 0.0) or 0.0)) for r in rows)

                if gross_abs <= 0.0:
                    continue

                for row in rows:
                    row_abs = abs(float(row.get("signed_weight", 0.0) or 0.0))
                    alloc_share = row_abs / gross_abs
                    intent_weight = final_symbol_weight * alloc_share

                    intents.append(
                        {
                            "symbol": symbol,
                            "strategy_id": str(row.get("strategy_id", "") or ""),
                            "side": str(row.get("side", "") or ""),
                            "target_weight": float(intent_weight),
                            "symbol_target_weight": float(final_symbol_weight),
                            "allocation_share_within_symbol": float(alloc_share),
                            "base_weight": float(row.get("base_weight", 0.0) or 0.0),
                            "signed_weight": float(row.get("signed_weight", 0.0) or 0.0),
                            "policy_score": float(row.get("policy_score", 0.0) or 0.0),
                            "policy_band": str(row.get("policy_band", "") or ""),
                            "policy_reason": str(row.get("policy_reason", "") or ""),
                            "policy_size_mult": float(row.get("policy_size_mult", 1.0) or 1.0),
                        }
                    )
        else:
            for row in deployable_rows:
                symbol = str(row.get("symbol", "") or "")
                side = str(row.get("side", "") or "").lower()
                key = f"{symbol}::{side}"
                if key not in symbol_weights:
                    continue

                intents.append(
                    {
                        "symbol": symbol,
                        "strategy_id": str(row.get("strategy_id", "") or ""),
                        "side": side,
                        "target_weight": float(symbol_weights.get(key, 0.0) or 0.0),
                        "base_weight": float(row.get("base_weight", 0.0) or 0.0),
                        "signed_weight": float(row.get("signed_weight", 0.0) or 0.0),
                        "policy_score": float(row.get("policy_score", 0.0) or 0.0),
                        "policy_band": str(row.get("policy_band", "") or ""),
                        "policy_reason": str(row.get("policy_reason", "") or ""),
                        "policy_size_mult": float(row.get("policy_size_mult", 1.0) or 1.0),
                    }
                )

        meta = {
            "case": "allocator_factory_pipeline",
            "n_candidates": len(candidates),
            "n_intents": len(intents),
            "target_exposure": self.target_exposure,
            "symbol_cap": self.symbol_cap,
            "allocator_profile": self.profile,
            "projection_profile": self.projection_profile,
            **dict(ctx.meta or {}),
        }
        return intents, meta

    def allocate(
        self,
        *,
        candidates: list[OpportunityCandidate],
    ) -> Allocation:
        intents, meta = self.allocate_intents(candidates=candidates)

        projector = ProjectionFactory(profile=self.projection_profile).build()
        projection = projector.project(intents)

        return Allocation(
            weights=dict(projection.weights or {}),
            intents=list(intents or []),
            meta={
                **meta,
                **dict(projection.meta or {}),
            },
        )
