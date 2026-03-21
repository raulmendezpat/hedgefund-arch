from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectionResult:
    weights: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)


class NetSymbolProjection:
    @staticmethod
    def project(intents: list[dict[str, Any]]) -> ProjectionResult:
        weights: dict[str, float] = {}
        debug_rows: list[dict[str, Any]] = []

        for row in list(intents or []):
            symbol = str(row.get("symbol", "") or "")
            target_weight = float(row.get("target_weight", 0.0) or 0.0)
            if not symbol:
                continue
            weights[symbol] = weights.get(symbol, 0.0) + target_weight
            debug_rows.append(
                {
                    "symbol": symbol,
                    "strategy_id": str(row.get("strategy_id", "") or ""),
                    "side": str(row.get("side", "") or ""),
                    "target_weight": target_weight,
                }
            )

        return ProjectionResult(
            weights=weights,
            meta={
                "projection_profile": "net_symbol",
                "n_intents": len(intents or []),
                "n_projected_symbols": len(weights),
                "debug_rows": debug_rows,
            },
        )


class GrossSymbolSideProjection:
    @staticmethod
    def project(intents: list[dict[str, Any]]) -> ProjectionResult:
        weights: dict[str, float] = {}
        debug_rows: list[dict[str, Any]] = []

        for row in list(intents or []):
            symbol = str(row.get("symbol", "") or "")
            side = str(row.get("side", "") or "").lower()
            target_weight = float(row.get("target_weight", 0.0) or 0.0)
            if not symbol or side not in {"long", "short"}:
                continue

            key = f"{symbol}::{side}"
            weights[key] = weights.get(key, 0.0) + target_weight
            debug_rows.append(
                {
                    "projection_key": key,
                    "symbol": symbol,
                    "strategy_id": str(row.get("strategy_id", "") or ""),
                    "side": side,
                    "target_weight": target_weight,
                }
            )

        return ProjectionResult(
            weights=weights,
            meta={
                "projection_profile": "gross_symbol_side",
                "n_intents": len(intents or []),
                "n_projected_keys": len(weights),
                "debug_rows": debug_rows,
            },
        )


class ProjectionFactory:
    def __init__(self, profile: str = "net_symbol"):
        self.profile = str(profile or "net_symbol")

    def build(self):
        p = self.profile.lower()
        if p == "net_symbol":
            return NetSymbolProjection()
        if p == "gross_symbol_side":
            return GrossSymbolSideProjection()
        raise ValueError(f"Unknown projection profile: {self.profile}")
