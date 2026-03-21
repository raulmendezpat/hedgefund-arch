from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import PolicyRule, PolicyState, PolicyDecision


@dataclass
class PolicyPipeline:
    rules: list[PolicyRule] = field(default_factory=list)

    def run(self, state: PolicyState) -> PolicyDecision:
        for rule in self.rules:
            state = rule.apply(state)

        return PolicyDecision(
            ts=int(state.ts),
            symbol=str(state.symbol),
            strategy_id=str(state.strategy_id),
            side=str(state.side),
            accept=bool(state.accept),
            size_mult=float(state.size_mult if state.accept else 0.0),
            policy_score=float(state.score),
            band=str(state.band),
            reason=str(state.reason),
            policy_meta={
                "p_win": float(state.p_win),
                "expected_return": float(state.expected_return),
                "score": float(state.score),
                **dict(state.model_meta or {}),
                **dict(state.tags or {}),
            },
        )
