from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf_core.contracts import OpportunityCandidate


@dataclass
class Allocation:
    weights: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


class Allocator:
    def __init__(
        self,
        *,
        target_exposure: float = 1.0,
        symbol_cap: float = 0.35,
        score_floor: float = 1e-12,
    ):
        self.target_exposure = float(target_exposure)
        self.symbol_cap = float(symbol_cap)
        self.score_floor = float(score_floor)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def allocate(
        self,
        *,
        candidates: list[OpportunityCandidate],
    ) -> Allocation:
        if not candidates:
            return Allocation(
                weights={},
                meta={
                    "case": "no_inputs",
                    "n_candidates": 0,
                    "target_exposure": self.target_exposure,
                    "symbol_cap": self.symbol_cap,
                },
            )

        raw_scores: dict[str, float] = {}
        debug_rows: list[dict[str, Any]] = []

        for c in candidates:
            meta = dict(c.signal_meta or {})
            side = str(c.side).lower()
            if side not in {"long", "short"}:
                continue

            base_weight = float(c.base_weight or 0.0)
            policy_score = float(meta.get("policy_score", 0.0) or 0.0)

            if abs(base_weight) <= 0.0:
                continue
            if policy_score <= self.score_floor:
                continue

            signed = 1.0 if side == "long" else -1.0
            raw = signed * abs(base_weight) * policy_score

            raw_scores[c.symbol] = raw_scores.get(c.symbol, 0.0) + raw
            debug_rows.append(
                {
                    "symbol": c.symbol,
                    "strategy_id": c.strategy_id,
                    "side": side,
                    "base_weight": base_weight,
                    "policy_score": policy_score,
                    "raw_contribution": raw,
                }
            )

        if not raw_scores:
            return Allocation(
                weights={},
                meta={
                    "case": "no_positive_scores",
                    "n_candidates": len(candidates),
                    "target_exposure": self.target_exposure,
                    "symbol_cap": self.symbol_cap,
                    "debug_rows": debug_rows,
                },
            )

        gross = sum(abs(v) for v in raw_scores.values())
        if gross <= 0.0:
            return Allocation(
                weights={},
                meta={
                    "case": "zero_gross",
                    "n_candidates": len(candidates),
                    "target_exposure": self.target_exposure,
                    "symbol_cap": self.symbol_cap,
                    "debug_rows": debug_rows,
                },
            )

        weights = {sym: (val / gross) * self.target_exposure for sym, val in raw_scores.items()}

        capped = {
            sym: self._clip(w, -self.symbol_cap, self.symbol_cap)
            for sym, w in weights.items()
        }

        capped_gross = sum(abs(v) for v in capped.values())
        if capped_gross > 0.0:
            scale = min(1.0, self.target_exposure / capped_gross)
            capped = {sym: v * scale for sym, v in capped.items()}
        else:
            scale = 0.0

        return Allocation(
            weights=capped,
            meta={
                "case": "policy_allocator",
                "n_candidates": len(candidates),
                "n_symbols": len(capped),
                "gross_raw_score": gross,
                "gross_weight_after_cap": sum(abs(v) for v in capped.values()),
                "target_exposure": self.target_exposure,
                "symbol_cap": self.symbol_cap,
                "post_cap_scale": scale,
                "raw_scores_by_symbol": raw_scores,
                "debug_rows": debug_rows,
            },
        )
