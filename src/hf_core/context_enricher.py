from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hf_core.contracts import OpportunityCandidate




def _apply_strategy_exit_profile_overrides(
    *,
    strategy_id: str,
    side: str,
    portfolio_regime: str,
    portfolio_breadth: float,
    p_win: float,
    adx: float,
    atrp: float,
    ema_gap: float,
    exit_profile: str,
    tp_mult: float,
    sl_mult: float,
    time_stop_bars: int,
) -> tuple[str, float, float, int]:
    strategy_id = str(strategy_id or "")
    side = str(side or "").lower()
    portfolio_regime = str(portfolio_regime or "").lower()
    portfolio_breadth = float(portfolio_breadth or 0.0)
    p_win = float(p_win or 0.0)
    adx = float(adx or 0.0)
    atrp = float(atrp or 0.0)
    ema_gap = abs(float(ema_gap or 0.0))

    # BTC short necesita una segmentación propia.
    # No debemos heredar ciegamente el perfil base calculado por expected_holding_bars,
    # porque eso colapsó casi todo a fast_exit.
    if strategy_id == "btc_trend" and side == "short":
        is_fast_exit = (
            p_win < 0.470
            or ema_gap < 0.010
            or adx < 22.0
            or atrp < 0.0045
            or portfolio_breadth <= 3.0
        )

        is_runner = (
            p_win >= 0.530
            and ema_gap >= 0.020
            and adx >= 30.0
            and atrp >= 0.0060
            and portfolio_breadth >= 6.0
            and portfolio_regime in {"normal", "defensive"}
        )

        if is_runner:
            return ("runner", 1.2, 1.0, 24)

        if is_fast_exit:
            return ("fast_exit", 0.8, 0.8, 8)

        return ("normal", 1.0, 1.0, 12)

    return (str(exit_profile or ""), float(tp_mult or 1.0), float(sl_mult or 1.0), int(time_stop_bars or 12))

def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if np.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _pct_rank(series: pd.Series, value: float, window: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.5
    s = s.iloc[-window:] if len(s) > window else s
    if s.empty:
        return 0.5
    return float((s <= value).mean())


def _safe_loc(series: pd.Series | None, ts, default: float = 0.0) -> float:
    if series is None or len(series) == 0:
        return float(default)
    try:
        v = series.loc[ts]
        if hasattr(v, "iloc"):
            v = v.iloc[-1]
        return _f(v, default)
    except Exception:
        return float(default)


@dataclass
class AssetContextEnricher:
    pct_window: int = 180
    slope_lookback: int = 24

    def enrich_candidate(
        self,
        *,
        candidate: OpportunityCandidate,
        ts,
        symbol_df: pd.DataFrame,
        feature_map: dict[str, pd.Series],
    ) -> OpportunityCandidate:
        meta = dict(candidate.signal_meta or {})

        close_s = pd.to_numeric(symbol_df.get("close"), errors="coerce")
        ts_idx = symbol_df.index

        close_now = _f(_safe_loc(close_s, ts, 0.0))
        ema_fast_s = feature_map.get("ema_fast")
        ema_slow_s = feature_map.get("ema_slow")
        adx_s = feature_map.get("adx")
        atrp_s = feature_map.get("atrp")
        rsi_s = feature_map.get("rsi")
        bb_width_s = feature_map.get("bb_width")
        range_exp_s = feature_map.get("range_expansion")

        ema_fast = _safe_loc(ema_fast_s, ts, 0.0)
        ema_slow = _safe_loc(ema_slow_s, ts, 0.0)
        adx = _safe_loc(adx_s, ts, 0.0)
        atrp = _safe_loc(atrp_s, ts, 0.0)
        rsi = _safe_loc(rsi_s, ts, 0.0)
        bb_width = _safe_loc(bb_width_s, ts, 0.0)
        range_expansion = _safe_loc(range_exp_s, ts, 0.0)

        try:
            pos = ts_idx.get_loc(ts)
            if hasattr(pos, "__len__") and not isinstance(pos, int):
                pos = pos[-1]
        except Exception:
            pos = None

        def _ret_bars(n: int) -> float:
            try:
                if pos is None or pos - n < 0:
                    return 0.0
                prev = _f(close_s.iloc[pos - n], 0.0)
                return 0.0 if prev == 0.0 else float(close_now / prev - 1.0)
            except Exception:
                return 0.0

        def _slope_pct(n: int) -> float:
            try:
                if ema_fast_s is None or pos is None or pos - n < 0:
                    return 0.0
                cur = _f(ema_fast_s.iloc[pos], 0.0)
                prev = _f(ema_fast_s.iloc[pos - n], 0.0)
                return 0.0 if prev == 0.0 else float(cur / prev - 1.0)
            except Exception:
                return 0.0

        ret_24h = _ret_bars(24)
        ret_7d = _ret_bars(24 * 7)
        ret_30d = _ret_bars(24 * 30)
        slope_ema_fast_24h = _slope_pct(self.slope_lookback)

        close_vs_ema_slow = 0.0 if ema_slow == 0.0 else float(close_now / ema_slow - 1.0)
        ema_fast_vs_ema_slow = 0.0 if ema_slow == 0.0 else float(ema_fast / ema_slow - 1.0)

        adx_pct = _pct_rank(adx_s, adx, self.pct_window) if adx_s is not None else 0.5
        atrp_pct = _pct_rank(atrp_s, atrp, self.pct_window) if atrp_s is not None else 0.5
        rsi_pct = _pct_rank(rsi_s, rsi, self.pct_window) if rsi_s is not None else 0.5
        bb_width_pct = _pct_rank(bb_width_s, bb_width, self.pct_window) if bb_width_s is not None else 0.5
        range_exp_pct = _pct_rank(range_exp_s, range_expansion, self.pct_window) if range_exp_s is not None else 0.5

        backdrop = "neutral"
        if ret_30d > 0.05 and ema_fast_vs_ema_slow > 0.0 and close_vs_ema_slow > 0.0:
            backdrop = "bullish"
        elif ret_30d < -0.05 and ema_fast_vs_ema_slow < 0.0 and close_vs_ema_slow < 0.0:
            backdrop = "bearish"

        trend_alignment_score = 0.0
        trend_alignment_score += np.clip(ret_7d / 0.08, -1.0, 1.0) * 0.30
        trend_alignment_score += np.clip(ret_30d / 0.20, -1.0, 1.0) * 0.30
        trend_alignment_score += np.clip(ema_fast_vs_ema_slow / 0.03, -1.0, 1.0) * 0.25
        trend_alignment_score += np.clip(close_vs_ema_slow / 0.05, -1.0, 1.0) * 0.15
        trend_alignment_score = float(np.clip(trend_alignment_score, -1.0, 1.0))

        side = str(candidate.side or "flat").lower()
        side_backdrop_alignment = 0.0
        if side == "long":
            side_backdrop_alignment = trend_alignment_score
        elif side == "short":
            side_backdrop_alignment = -trend_alignment_score

        expected_holding_bars = 12
        if abs(side_backdrop_alignment) >= 0.6 and adx_pct >= 0.6:
            expected_holding_bars = 48
        elif abs(side_backdrop_alignment) >= 0.3:
            expected_holding_bars = 24
        elif abs(side_backdrop_alignment) < 0.15:
            expected_holding_bars = 8

        exit_profile = "normal"
        if expected_holding_bars <= 8:
            exit_profile = "fast_exit"
        elif expected_holding_bars >= 48:
            exit_profile = "runner"

        tp_mult = 1.0
        sl_mult = 1.0
        time_stop_bars = expected_holding_bars

        if exit_profile == "fast_exit":
            tp_mult = 0.8
            sl_mult = 0.8
        elif exit_profile == "runner":
            tp_mult = 1.4
            sl_mult = 1.1

        debug_strategy_id = str(getattr(candidate, "strategy_id", "") or meta.get("strategy_id", ""))
        debug_side = str(getattr(candidate, "side", "") or meta.get("side", ""))
        debug_portfolio_regime = str(meta.get("portfolio_regime", "") or "")
        debug_portfolio_breadth = float(meta.get("portfolio_breadth", 0.0) or 0.0)
        debug_p_win = float(meta.get("p_win", 0.0) or 0.0)
        debug_adx = float(adx or 0.0)
        debug_atrp = float(atrp or 0.0)
        debug_ema_gap = float(meta.get("ema_gap_fast_slow", meta.get("ema_gap_pct", ema_fast_vs_ema_slow)) or 0.0)

        exit_profile, tp_mult, sl_mult, time_stop_bars = _apply_strategy_exit_profile_overrides(
            strategy_id=debug_strategy_id,
            side=debug_side,
            portfolio_regime=debug_portfolio_regime,
            portfolio_breadth=debug_portfolio_breadth,
            p_win=debug_p_win,
            adx=debug_adx,
            atrp=debug_atrp,
            ema_gap=debug_ema_gap,
            exit_profile=str(exit_profile or ""),
            tp_mult=float(tp_mult or 1.0),
            sl_mult=float(sl_mult or 1.0),
            time_stop_bars=int(time_stop_bars or 12),
        )

        if debug_strategy_id == "btc_trend" and debug_side.lower() == "short":
            meta["debug_exit_probe"] = {
                "strategy_id": debug_strategy_id,
                "side": debug_side,
                "portfolio_regime": debug_portfolio_regime,
                "portfolio_breadth": debug_portfolio_breadth,
                "p_win": debug_p_win,
                "adx": debug_adx,
                "atrp": debug_atrp,
                "ema_gap": debug_ema_gap,
                "resolved_exit_profile": str(exit_profile),
                "resolved_tp_mult": float(tp_mult),
                "resolved_sl_mult": float(sl_mult),
                "resolved_time_stop_bars": int(time_stop_bars),
            }

        meta.update({
            "close": float(close_now),
            "ema_fast": float(ema_fast),
            "ema_slow": float(ema_slow),
            "adx": float(adx),
            "atrp": float(atrp),
            "rsi": float(rsi),
            "bb_width": float(bb_width),
            "ctx_close_vs_ema_slow": float(close_vs_ema_slow),
            "ctx_ema_fast_vs_ema_slow": float(ema_fast_vs_ema_slow),
            "ctx_ret_24h": float(ret_24h),
            "ctx_ret_7d": float(ret_7d),
            "ctx_ret_30d": float(ret_30d),
            "ctx_slope_ema_fast_24h": float(slope_ema_fast_24h),
            "ctx_adx_pct": float(adx_pct),
            "ctx_atrp_pct": float(atrp_pct),
            "ctx_rsi_pct": float(rsi_pct),
            "ctx_bb_width_pct": float(bb_width_pct),
            "ctx_range_expansion_pct": float(range_exp_pct),
            "ctx_backdrop": str(backdrop),
            "ctx_trend_alignment_score": float(trend_alignment_score),
            "ctx_side_backdrop_alignment": float(side_backdrop_alignment),
            "ctx_expected_holding_bars": int(expected_holding_bars),
            "ctx_exit_profile": str(exit_profile),
            "ctx_tp_mult": float(tp_mult),
            "ctx_sl_mult": float(sl_mult),
            "ctx_time_stop_bars": int(time_stop_bars),
        })

        return OpportunityCandidate(
            ts=int(candidate.ts),
            symbol=str(candidate.symbol),
            strategy_id=str(candidate.strategy_id),
            side=str(candidate.side),
            signal_strength=float(candidate.signal_strength),
            base_weight=float(candidate.base_weight),
            signal_meta=meta,
        )
