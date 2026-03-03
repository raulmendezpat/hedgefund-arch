from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from hf.core.types import Candle, Signal, Allocation, RegimeState
from hf.core.interfaces import CapitalAllocator


@dataclass
class TradeHoldAllocator(CapitalAllocator):
    """
    Bootstrap allocator: derives per-candle weights from a legacy portfolio_trades CSV.

    Rule:
    - For each candle timestamp (ms), find trades where entry_ts_aligned <= ts < exit_ts_aligned
    - Assign weight = alloc_w to that trade's symbol
    - If multiple concurrent trades (edge case), sum weights then (optionally) normalize to <=1
    - If none active: sticky to previous if sticky_when_flat else 0/0
    """
    trades_csv: str
    sticky_when_flat: bool = True
    normalize_if_over_1: bool = True

    def __post_init__(self) -> None:
        df = pd.read_csv(self.trades_csv)

        need = {"symbol", "alloc_w", "entry_ts_aligned", "exit_ts_aligned"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in trades CSV {self.trades_csv}: {sorted(missing)}")

        # ensure ints
        df["entry_ts_aligned"] = df["entry_ts_aligned"].astype("int64")
        df["exit_ts_aligned"] = df["exit_ts_aligned"].astype("int64")
        df["alloc_w"] = df["alloc_w"].astype(float)

        # keep only meaningful weights
        df = df[df["alloc_w"] > 0].copy()
        self._trades = df.sort_values(["entry_ts_aligned", "exit_ts_aligned"]).reset_index(drop=True)

    def allocate(
        self,
        candles: Dict[str, Candle],
        signals: Dict[str, Signal],
        regimes: Dict[str, RegimeState],
        prev_alloc: Optional[Allocation],
    ) -> Allocation:
        # candle ts in ms
        ts_ms = None
        for c in candles.values():
            ts_ms = int(pd.Timestamp(c.ts).value // 1_000_000)
            break
        if ts_ms is None:
            return Allocation(weights={}, meta={"case": "no_candles"})

        active = self._trades[(self._trades["entry_ts_aligned"] <= ts_ms) & (ts_ms < self._trades["exit_ts_aligned"])]

        if active.empty:
            if self.sticky_when_flat and prev_alloc is not None:
                return Allocation(weights=dict(prev_alloc.weights), meta={"case": "flat_sticky"})
            return Allocation(weights={sym: 0.0 for sym in candles.keys()}, meta={"case": "flat_zero"})

        w: Dict[str, float] = {sym: 0.0 for sym in candles.keys()}

        # map by symbol exact match (legacy uses full symbol like BTC/USDT:USDT)
        for _, r in active.iterrows():
            sym = str(r["symbol"])
            aw = float(r["alloc_w"])
            if sym in w:
                w[sym] += aw

        total = sum(max(0.0, x) for x in w.values())
        if self.normalize_if_over_1 and total > 1.0:
            for k in list(w.keys()):
                w[k] = w[k] / total

        # label case
        nonzero = [(k, v) for k, v in w.items() if v > 0]
        if len(nonzero) == 1:
            case = "single_active"
        else:
            case = "multi_active"
        return Allocation(weights=w, meta={"case": case, "n_active_trades": int(len(active))})
