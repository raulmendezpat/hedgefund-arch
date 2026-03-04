from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import pandas as pd


def _parse_ts_series(ts: pd.Series) -> pd.Series:
    """
    Soporta:
      - strings datetime (con o sin tz)
      - int/float epoch en ms o s
    Devuelve pandas datetime64[ns, UTC]
    """
    # 1) intenta parsear como datetime string
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if dt.notna().mean() > 0.8:
        return dt

    # 2) si es numérico, intenta epoch ms vs s
    tnum = pd.to_numeric(ts, errors="coerce")
    if tnum.notna().mean() < 0.8:
        # peor caso: todo NaT
        return pd.to_datetime(ts, errors="coerce", utc=True)

    median = float(tnum.dropna().median())
    # heurística: epoch ms suele ser > 1e11 (2024ms ~ 1.7e12)
    unit = "ms" if median > 1e11 else "s"
    return pd.to_datetime(tnum, errors="coerce", utc=True, unit=unit)


def _infer_periods_per_year(ts_dt: pd.Series) -> Optional[float]:
    """Inferir periodicidad (velas por año) a partir de diffs de timestamps."""
    dt = ts_dt.dropna().sort_values()
    if len(dt) < 5:
        return None
    diffs = dt.diff().dropna()
    # usa la mediana para ser robusto
    sec = diffs.median().total_seconds()
    if not (sec > 0):
        return None
    return (365.0 * 24.0 * 3600.0) / sec


@dataclass
class PortfolioMetricsEngine:
    """
    Métricas de portfolio estilo hedge fund / research.

    Inputs esperados:
      - perf_df: DataFrame con columnas: ts, equity, drawdown_pct, port_ret
      - alloc_df (opcional): DataFrame con w_btc, w_sol para calcular exposure%

    Salida:
      dict con métricas principales.
    """

    risk_free_rate_annual: float = 0.0

    def compute(
        self,
        perf_df: pd.DataFrame,
        alloc_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        for col in ["ts", "equity", "port_ret"]:
            if col not in perf_df.columns:
                raise ValueError(f"perf_df missing required column: {col}")

        ts_dt = _parse_ts_series(perf_df["ts"])
        periods_per_year = _infer_periods_per_year(ts_dt)

        equity = pd.to_numeric(perf_df["equity"], errors="coerce")
        port_ret = pd.to_numeric(perf_df["port_ret"], errors="coerce").fillna(0.0)

        if equity.isna().any():
            raise ValueError("perf_df.equity has NaNs after coercion")

        start_eq = float(equity.iloc[0])
        end_eq = float(equity.iloc[-1])
        total_return = (end_eq / start_eq) - 1.0 if start_eq != 0 else float("nan")

        # max drawdown %
        if "drawdown_pct" in perf_df.columns:
            dd_pct = pd.to_numeric(perf_df["drawdown_pct"], errors="coerce")
            max_dd_pct = float(dd_pct.min())
        elif "drawdown" in perf_df.columns:
            dd = pd.to_numeric(perf_df["drawdown"], errors="coerce")
            max_dd_pct = float(dd.min() * 100.0)
        else:
            max_dd_pct = float("nan")

        # CAGR basado en tiempo real
        cagr = float("nan")
        t0 = ts_dt.dropna().iloc[0] if ts_dt.notna().any() else None
        t1 = ts_dt.dropna().iloc[-1] if ts_dt.notna().any() else None
        if t0 is not None and t1 is not None and t1 > t0 and start_eq > 0:
            years = (t1 - t0).total_seconds() / (365.0 * 24.0 * 3600.0)
            if years > 0:
                cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0

        # Volatilidad y Sharpe anualizados (si podemos inferir periodicidad)
        vol_annual = float("nan")
        sharpe_annual = float("nan")
        mean_ret = float(port_ret.mean())
        std_ret = float(port_ret.std(ddof=1))

        if periods_per_year is not None and periods_per_year > 0:
            vol_annual = std_ret * math.sqrt(periods_per_year)
            # Sharpe: (E[r] - rf/ppy) / std * sqrt(ppy)
            rf_per_period = float(self.risk_free_rate_annual) / periods_per_year
            excess_mean = mean_ret - rf_per_period
            if std_ret > 0:
                sharpe_annual = (excess_mean / std_ret) * math.sqrt(periods_per_year)

        # Exposure (% del tiempo con w_sum > 0)
        exposure_pct = float("nan")
        if alloc_df is not None:
            if ("w_btc" in alloc_df.columns) and ("w_sol" in alloc_df.columns):
                wsum = pd.to_numeric(alloc_df["w_btc"], errors="coerce").fillna(0.0) + pd.to_numeric(
                    alloc_df["w_sol"], errors="coerce"
                ).fillna(0.0)
                exposure_pct = float((wsum > 1e-12).mean() * 100.0)

        return {
            "start_equity": start_eq,
            "end_equity": end_eq,
            "total_return_pct": float(total_return * 100.0),
            "cagr_pct": float(cagr * 100.0) if not math.isnan(cagr) else float("nan"),
            "max_drawdown_pct": max_dd_pct,
            "mean_period_return": mean_ret,
            "vol_annual": vol_annual,
            "sharpe_annual": sharpe_annual,
            "periods_per_year": periods_per_year,
            "exposure_pct": exposure_pct,
        }
