from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from hf.core.types import Candle, Signal, RegimeState
from hf.core.interfaces import RegimeEngine


@dataclass
class CSVRegimeEngine(RegimeEngine):
    """
    Bootstrap RegimeEngine: reads per-candle regime flags from a CSV produced by legacy backtester.
    This avoids re-implementing Regime3+ML gates while we stabilize the new architecture.
    """
    csv_path: str
    ts_col: str = "timestamp"  # ms epoch (legacy CSV)
    btc_col: str = "btc_regime_on"  # <-- will adjust after inspecting CSV
    sol_col: str = "sol_regime_on"  # <-- will adjust after inspecting CSV

    def __post_init__(self) -> None:
        df = pd.read_csv(self.csv_path)
        if self.ts_col not in df.columns:
            raise ValueError(f"ts_col '{self.ts_col}' not found in {self.csv_path}")
        if self.btc_col not in df.columns:
            raise ValueError(f"btc_col '{self.btc_col}' not found in {self.csv_path}")
        if self.sol_col not in df.columns:
            raise ValueError(f"sol_col '{self.sol_col}' not found in {self.csv_path}")

        df[self.ts_col] = df[self.ts_col].astype("int64")
        self._df = df.set_index(self.ts_col).sort_index()

    def evaluate(self, candles: Dict[str, Candle], signals: Dict[str, Signal]) -> Dict[str, RegimeState]:
        # All candles dict keys are full symbols (e.g., BTC/USDT:USDT)
        # We map by presence (btc/sol) using simple contains.
        # NOTE: We'll tighten this mapping later.
        ts_ms = None
        # take first candle ts
        for c in candles.values():
            ts_ms = int(pd.Timestamp(c.ts).value // 1_000_000)  # ns->ms
            break
        if ts_ms is None or ts_ms not in self._df.index:
            # if missing, default OFF (conservative)
            return {sym: RegimeState(on=False, reason="missing_csv_ts") for sym in candles.keys()}

        row = self._df.loc[ts_ms]
        btc_on = bool(row[self.btc_col])
        sol_on = bool(row[self.sol_col])

        out = {}
        for sym in candles.keys():
            if "BTC" in sym:
                out[sym] = RegimeState(on=btc_on, reason="csv")
            elif "SOL" in sym:
                out[sym] = RegimeState(on=sol_on, reason="csv")
            else:
                out[sym] = RegimeState(on=False, reason="csv_unknown_symbol")
        return out
