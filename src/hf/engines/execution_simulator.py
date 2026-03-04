from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _weight_columns(df: pd.DataFrame) -> List[str]:
    # soporta allocations CSV actuales: w_btc, w_sol
    # y también formato genérico: w_<symbol>
    cols = [c for c in df.columns if c.startswith("w_")]
    if not cols:
        raise ValueError("No weight columns found (expected columns starting with 'w_').")
    return cols


@dataclass
class ExecutionCostModel:
    """
    Modelo de costos por turnover (estilo research / portfolio-level).

    - fee_bps: costo lineal en bps aplicado al turnover (ej: 2 bps por 100% turnover)
    - slippage_bps: slippage lineal en bps aplicado al turnover
    - turnover = sum_i |w_t(i) - w_{t-1}(i)|

    Nota:
      - Esto NO simula fills/órdenes. Es un approximation portfolio-level para research.
      - Si los pesos suman <= 1, el turnover queda en [0, 2] típicamente.
    """
    fee_bps: float = 0.0
    slippage_bps: float = 0.0

    def cost_per_period(self, turnover: pd.Series) -> pd.Series:
        bps_total = float(self.fee_bps) + float(self.slippage_bps)
        return turnover.astype("float64") * (bps_total / 10_000.0)


@dataclass
class ExecutionSimulator:
    """
    Aplica costos a una equity curve / returns ya calculados.

    Inputs típicos:
      - perf_df: results/pipeline_equity_<name>.csv
        requiere: ts, port_ret (gross), equity (gross) (equity se recalcula neta)
      - alloc_df: results/pipeline_allocations_<name>.csv
        requiere: columnas w_* (pesos por activo)

    Output:
      DataFrame con:
        - gross_ret (original)
        - turnover
        - cost
        - net_ret
        - net_equity
        - net_drawdown / net_drawdown_pct
    """
    cost_model: ExecutionCostModel = field(default_factory=ExecutionCostModel)
    initial_equity: float = 1000.0

    def apply_costs(
        self,
        perf_df: pd.DataFrame,
        alloc_df: pd.DataFrame,
        ts_col: str = "ts",
        gross_ret_col: str = "port_ret",
    ) -> pd.DataFrame:
        if ts_col not in perf_df.columns:
            raise ValueError(f"perf_df missing '{ts_col}'")
        if gross_ret_col not in perf_df.columns:
            raise ValueError(f"perf_df missing '{gross_ret_col}'")

        wcols = _weight_columns(alloc_df)

        # Alinear por índice (asumimos mismo orden/longitud del pipeline)
        if len(perf_df) != len(alloc_df):
            raise ValueError(f"Len mismatch perf_df={len(perf_df)} alloc_df={len(alloc_df)}")

        out = pd.DataFrame()
        out[ts_col] = perf_df[ts_col]

        gross_ret = pd.to_numeric(perf_df[gross_ret_col], errors="coerce").fillna(0.0).astype("float64")
        W = alloc_df[wcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float64")

        # turnover
        dW = W.diff().abs()
        dW.iloc[0] = W.iloc[0].abs()  # desde cash=0 en t=0
        turnover = dW.sum(axis=1)

        cost = self.cost_model.cost_per_period(turnover)
        net_ret = gross_ret - cost

        out["gross_ret"] = gross_ret.values
        out["turnover"] = turnover.values
        out["cost"] = cost.values
        out["net_ret"] = net_ret.values

        # net equity
        eq = pd.Series(float(self.initial_equity), index=out.index, dtype="float64")
        for i in range(1, len(out)):
            eq.iloc[i] = eq.iloc[i - 1] * (1.0 + float(net_ret.iloc[i]))
        out["net_equity"] = eq.values

        # net drawdown
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        out["net_drawdown"] = dd.values
        out["net_drawdown_pct"] = (dd * 100.0).values

        return out
