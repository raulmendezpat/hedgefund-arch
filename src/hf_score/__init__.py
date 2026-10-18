"""Expected-value (EV) trade scoring.

Phase 1 (this package today): honest per-candidate labels.
- market_data: closed-candle bars + point-in-time features from the OHLCV cache
- candidates:  raw signal candidates produced by the strategy registry engines
- labels:      each candidate simulated with the production exit policy, no lookahead,
               taker fees and slippage, expressed in risk units (R)
"""

from .config import LabelConfig
from .market_data import build_symbol_frames, load_closed_bars
from .candidates import generate_candidates
from .labels import label_candidates

__all__ = [
    "LabelConfig",
    "build_symbol_frames",
    "generate_candidates",
    "label_candidates",
    "load_closed_bars",
]
