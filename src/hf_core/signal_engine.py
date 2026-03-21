from __future__ import annotations

from hf_core.contracts import OpportunityCandidate


class SignalEngine:
    def build_candidates(
        self,
        *,
        ts: int,
        selected_opps_for_alloc: list[object],
    ) -> list[OpportunityCandidate]:
        out: list[OpportunityCandidate] = []

        for opp in list(selected_opps_for_alloc or []):
            if isinstance(opp, OpportunityCandidate):
                side = str(getattr(opp, "side", "flat") or "flat").lower()
                if side == "flat":
                    continue
                out.append(
                    OpportunityCandidate(
                        ts=int(getattr(opp, "ts", ts) or ts),
                        symbol=str(getattr(opp, "symbol", "") or ""),
                        strategy_id=str(getattr(opp, "strategy_id", "") or ""),
                        side=side,
                        signal_strength=float(getattr(opp, "signal_strength", 0.0) or 0.0),
                        base_weight=float(getattr(opp, "base_weight", 0.0) or 0.0),
                        signal_meta=dict(getattr(opp, "signal_meta", {}) or {}),
                    )
                )
                continue

            symbol = str(getattr(opp, "symbol", "") or "")
            side = str(getattr(opp, "side", "flat") or "flat").lower()
            strategy_id = str(getattr(opp, "strategy_id", "") or "")
            meta = dict(getattr(opp, "meta", {}) or {})

            if side == "flat":
                continue

            out.append(
                OpportunityCandidate(
                    ts=int(ts),
                    symbol=symbol,
                    strategy_id=strategy_id,
                    side=side,
                    signal_strength=float(getattr(opp, "signal_strength", getattr(opp, "strength", 0.0)) or 0.0),
                    base_weight=float(getattr(opp, "base_weight", meta.get("base_weight", 0.0)) or 0.0),
                    signal_meta=meta,
                )
            )

        return out
