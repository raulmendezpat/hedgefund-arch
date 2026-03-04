from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from hf.core.interfaces import PortfolioEngine
from hf.core.types import Candle, Allocation


@dataclass
class SimplePortfolioEngine(PortfolioEngine):
    """
    PortfolioEngine mínimo estilo research:

    - Construye returns por símbolo a partir de Candle.close (simple returns).
    - Aplica weights por candle (Allocation.weights) para obtener return del portfolio.
    - Calcula equity curve y drawdown.

    Nota: NO hace slippage/fees/latency. Eso va en Execution/Simulator luego.
    """

    initial_equity: float = 1000.0
    use_log_returns: bool = False  # futuro: activar log returns si quieres

    def run(
        self,
        candles_by_symbol: Dict[str, List[Candle]],
        allocations: List[Allocation],
        symbols: Optional[Tuple[str, ...]] = None,
    ) -> pd.DataFrame:
        if symbols is None:
            symbols = tuple(candles_by_symbol.keys())

        if not symbols:
            raise ValueError("No symbols provided to portfolio engine.")

        # Asumimos series alineadas por índice (mismo número de candles por símbolo)
        n = len(allocations)
        for sym in symbols:
            if sym not in candles_by_symbol:
                raise KeyError(f"Missing candles for symbol: {sym}")
            if len(candles_by_symbol[sym]) != n:
                raise ValueError(
                    f"Len mismatch for {sym}: candles={len(candles_by_symbol[sym])} != allocations={n}"
                )

        # timestamps (usamos el primer símbolo como referencia)
        ts = [c.ts for c in candles_by_symbol[symbols[0]]]

        # prices + returns
        prices = {}
        rets = {}
        for sym in symbols:
            close = pd.Series([c.close for c in candles_by_symbol[sym]], index=ts, dtype="float64")
            prices[sym] = close
            if self.use_log_returns:
                r = (close / close.shift(1)).apply(lambda x: float("nan") if pd.isna(x) else (0.0 if x <= 0 else __import__("math").log(x)))
            else:
                r = close.pct_change()
            rets[sym] = r

        df = pd.DataFrame(index=pd.Index(ts, name="ts"))
        for sym in symbols:
            df[f"px_{sym}"] = prices[sym].values
            df[f"ret_{sym}"] = rets[sym].values

        # weights por candle
        w_cols = []
        for sym in symbols:
            col = f"w_{sym}"
            w_cols.append(col)
            df[col] = [float(a.weights.get(sym, 0.0)) for a in allocations]

        # portfolio return (suma ponderada)
        # Nota: en t=0 return NaN; lo convertimos a 0 para equity
        port_ret = 0.0
        pr = pd.Series(0.0, index=df.index, dtype="float64")
        for sym in symbols:
            pr += df[f"w_{sym}"] * df[f"ret_{sym}"].fillna(0.0)
        df["port_ret"] = pr.values

        # equity curve
        equity = pd.Series(self.initial_equity, index=df.index, dtype="float64")
        for i in range(1, len(df)):
            equity.iloc[i] = equity.iloc[i - 1] * (1.0 + float(df["port_ret"].iloc[i]))
        df["equity"] = equity.values

        # drawdown
        peak = equity.cummax()
        dd = (equity / peak) - 1.0
        df["drawdown"] = dd.values
        df["drawdown_pct"] = (dd * 100.0).values

        return df
