from __future__ import annotations

from typing import Any

from .pipeline import SelectionPipeline
from .stage_asset_gate import AssetGateStage
from .stage_alpha_selection import AlphaSelectionStage
from .stage_best_per_symbol import BestPerSymbolStage
from .stage_contextual_eligibility import ContextualEligibilityStage
from .stage_strategy_regime_eligibility import StrategyRegimeEligibilityStage


class SelectionPipelineFactory:
    STAGE_REGISTRY = {
        "asset_gate": AssetGateStage,
        "contextual_eligibility": ContextualEligibilityStage,
        "strategy_regime_eligibility": StrategyRegimeEligibilityStage,
        "alpha_selection": AlphaSelectionStage,
        "best_per_symbol": BestPerSymbolStage,
    }

    DEFAULT_STAGE_SPECS = [
        {"name": "asset_gate"},
        {"name": "alpha_selection"},
        {"name": "best_per_symbol", "kwargs": {"score_field": "policy_score"}},
    ]

    @classmethod
    def build(cls, *, config: dict, profile: str, trace_path: str) -> SelectionPipeline:
        stage_specs = cls._resolve_stage_specs(config=config, profile=profile)
        stages = [cls._build_stage(spec=spec, config=config, profile=profile) for spec in stage_specs]
        return SelectionPipeline(
            stages=stages,
            trace_path=str(trace_path),
        )

    @classmethod
    def register_stage(cls, stage_name: str, stage_cls: type) -> None:
        name = str(stage_name or "").strip()
        if not name:
            raise ValueError("stage_name is required")
        cls.STAGE_REGISTRY[name] = stage_cls

    @classmethod
    def _resolve_stage_specs(cls, *, config: dict, profile: str) -> list[dict[str, Any]]:
        profiles = dict((config or {}).get("profiles", {}) or {})
        profile_cfg = dict(profiles.get(str(profile), {}) or {})
        pipeline_cfg = dict(profile_cfg.get("pipeline", {}) or {})
        stage_specs = list(pipeline_cfg.get("stages", []) or [])
        if stage_specs:
            return stage_specs
        return [dict(x) for x in cls.DEFAULT_STAGE_SPECS]

    @classmethod
    def _build_stage(cls, *, spec: dict[str, Any], config: dict, profile: str):
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid selection stage spec: {spec!r}")

        name = str(spec.get("name", "") or "").strip()
        if not name:
            raise ValueError(f"Selection stage spec missing name: {spec!r}")

        if name not in cls.STAGE_REGISTRY:
            known = ", ".join(sorted(cls.STAGE_REGISTRY.keys()))
            raise ValueError(f"Unknown selection stage '{name}'. Known stages: {known}")

        stage_cls = cls.STAGE_REGISTRY[name]
        kwargs = dict(spec.get("kwargs", {}) or {})

        if name == "best_per_symbol":
            return stage_cls(**kwargs)

        return stage_cls(config, profile=str(profile), **kwargs)
