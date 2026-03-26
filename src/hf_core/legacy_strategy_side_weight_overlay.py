from __future__ import annotations

from hf_core.strategy_side_weight_overlay import StrategySideWeightOverlay
from hf_core.strategy_side_weight_rules import parse_strategy_side_post_ml_weight_rules


class LegacyStrategySideWeightOverlay(StrategySideWeightOverlay):
    """Backward-compatible alias during migration."""
    pass
