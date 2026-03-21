from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy_parts.contracts import PolicyDecision, PolicyState
from .policy_parts.factory import PolicyFactory


@dataclass
class PolicyModel:
    profile: str = "default"
    config: dict[str, Any] = field(default_factory=dict)

    def _resolve_profile_and_config(self, meta) -> tuple[str, dict[str, Any]]:
        base_cfg = dict(self.config or {})
        strategy_id = str(getattr(meta, "strategy_id", "") or "")

        default_profile = str(base_cfg.get("default_profile", self.profile) or self.profile)
        profiles = dict(base_cfg.get("profiles", {}) or {})
        strategy_overrides = dict(base_cfg.get("strategy_overrides", {}) or {})

        chosen_profile = default_profile
        chosen_cfg = dict(profiles.get(chosen_profile, {}) or {})

        if strategy_id in strategy_overrides:
            ov = dict(strategy_overrides.get(strategy_id, {}) or {})
            override_profile = str(ov.get("profile", chosen_profile) or chosen_profile)
            chosen_profile = override_profile
            chosen_cfg = dict(profiles.get(chosen_profile, {}) or {})
            chosen_cfg.update({k: v for k, v in ov.items() if k != "profile"})

        return chosen_profile, chosen_cfg

    def _decide_one(self, meta) -> PolicyDecision:
        profile, cfg = self._resolve_profile_and_config(meta)

        state = PolicyState(
            ts=int(getattr(meta, "ts", 0) or 0),
            symbol=str(getattr(meta, "symbol", "") or ""),
            strategy_id=str(getattr(meta, "strategy_id", "") or ""),
            side=str(getattr(meta, "side", "flat") or "flat").lower(),
            p_win=float(getattr(meta, "p_win", 0.0) or 0.0),
            expected_return=float(getattr(meta, "expected_return", 0.0) or 0.0),
            score=float(getattr(meta, "score", 0.0) or 0.0),
            model_meta=dict(getattr(meta, "model_meta", {}) or {}),
        )

        pipeline = PolicyFactory(profile=profile, config=cfg).build()
        decision = pipeline.run(state)
        decision.policy_meta["policy_profile"] = profile
        return decision

    def decide_many(self, metas) -> list[PolicyDecision]:
        return [self._decide_one(m) for m in list(metas or [])]
