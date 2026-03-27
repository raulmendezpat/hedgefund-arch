from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc
from hf.engines.opportunity_book import RegistryOpportunityBook
from hf.pipeline.run_portfolio import _adx, _atr, _ema, _row_to_candle
from hf_core import FeatureBuilder, MetaModel, PolicyModel, AllocationBridge, Allocator, OpportunityCandidate, AssetContextEnricher, AllocationEngine, ResearchAllocationRouter, build_allocation_config_from_args, build_allocation_engine
from hf_core.selection_stages import load_selection_policy_config, SelectionPipelineFactory
from hf_core.pwin_math_v2 import MathPWinV2
from hf_core.pwin_math_v3 import MathPWinV3
from hf_core.ml.feature_expansion import build_symbol_feature_frame, merge_cross_asset_features
from hf_core.research_meta_inputs import seed_candidate_meta, build_portfolio_context
from hf_core.trade_lifecycle import TradeLifecycleEngine

from hf_core.score_projector import ScoreProjector


def _resolve_competitive_rank_score(candidate) -> float:
    sm = dict(getattr(candidate, "signal_meta", {}) or {})
    try:
        return float(
            sm.get(
                "post_ml_competitive_score",
                sm.get(
                    "meta_post_ml_competitive_score",
                    sm.get(
                        "post_ml_score",
                        sm.get(
                            "meta_post_ml_score",
                            sm.get("competitive_score", sm.get("meta_competitive_score", 0.0)),
                        ),
                    ),
                ),
            ) or 0.0
        )
    except Exception:
        return 0.0


def _apply_best_per_symbol_competition(
    candidates,
    decisions,
):
    pairs = list(zip(list(candidates or []), list(decisions or [])))
    if not pairs:
        return [], [], {
            "enabled": True,
            "mode": "best_per_symbol",
            "kept_count": 0,
            "dropped_count": 0,
            "kept": [],
            "dropped": [],
        }

    ranked_rows = []
    for idx, (c, d) in enumerate(pairs):
        score = _resolve_competitive_rank_score(c)
        sm = dict(getattr(c, "signal_meta", {}) or {})
        ranked_rows.append(
            {
                "idx": int(idx),
                "candidate": c,
                "decision": d,
                "symbol": str(getattr(c, "symbol", "") or ""),
                "strategy_id": str(getattr(c, "strategy_id", "") or ""),
                "side": str(getattr(c, "side", "") or ""),
                "score": float(score),
                "p_win": float(sm.get("p_win", 0.0) or 0.0),
                "competitive_score": float(
                    sm.get("competitive_score", sm.get("meta_competitive_score", 0.0)) or 0.0
                ),
                "post_ml_competitive_score": float(score),
            }
        )

    best_by_symbol = {}
    for row in ranked_rows:
        sym = str(row["symbol"])
        prev = best_by_symbol.get(sym)
        if prev is None or (
            row["score"], row["p_win"], row["competitive_score"], -row["idx"]
        ) > (
            prev["score"], prev["p_win"], prev["competitive_score"], -prev["idx"]
        ):
            best_by_symbol[sym] = row

    kept_idx = {int(v["idx"]) for v in best_by_symbol.values()}
    kept = []
    dropped = []

    for row in ranked_rows:
        target = kept if int(row["idx"]) in kept_idx else dropped
        target.append(
            {
                "symbol": str(row["symbol"]),
                "strategy_id": str(row["strategy_id"]),
                "side": str(row["side"]),
                "p_win": float(row["p_win"]),
                "competitive_score": float(row["competitive_score"]),
                "post_ml_competitive_score": float(row["post_ml_competitive_score"]),
            }
        )

    kept_candidates = []
    kept_decisions = []
    for idx, (c, d) in enumerate(pairs):
        if idx in kept_idx:
            sm = dict(getattr(c, "signal_meta", {}) or {})
            sm["best_per_symbol_kept"] = True
            sm["best_per_symbol_rank_score"] = float(_resolve_competitive_rank_score(c))
            c.signal_meta = sm
            kept_candidates.append(c)
            kept_decisions.append(d)

    meta = {
        "enabled": True,
        "mode": "best_per_symbol",
        "kept_count": int(len(kept)),
        "dropped_count": int(len(dropped)),
        "kept": kept,
        "dropped": dropped,
    }
    return kept_candidates, kept_decisions, meta


def load_registry(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Strategy registry must be a list")
    return data


def extract_symbols(rows: list[dict]) -> list[str]:
    out = []
    seen = set()
    for r in rows:
        s = str(r.get("symbol", "") or "")
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_symbol_df(symbol: str, start: str, end: str | None, exchange: str, cache_dir: str) -> pd.DataFrame:
    end_ms = dt_to_ms_utc(end) if end else None

    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe="1h",
        start_ms=dt_to_ms_utc(start),
        end_ms=end_ms,
        exchange_id=exchange,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_if_no_end=bool(end is None),
    ).copy()

    df["ts"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")].copy()
    if end:
        df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")].copy()
    df = df.set_index("ts").sort_index()
    return df


def build_feature_map(df: pd.DataFrame, symbol: str) -> dict[str, pd.Series]:
    close = df["close"].astype(float)
    atr = _atr(df, 14)
    atrp = atr / close.replace(0.0, np.nan)

    out = {
        "adx": _adx(df, 14),
        "atr": atr,
        "atrp": atrp,
        "ema_fast": _ema(close, 20),
        "ema_slow": _ema(close, 200),
    }

    if symbol.upper().startswith("SOL/"):
        bb_period = 20
        bb_std = 2.0
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        avg_gain = gain.rolling(bb_period, min_periods=bb_period).mean()
        avg_loss = loss.rolling(bb_period, min_periods=bb_period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        bb_mid = close.rolling(bb_period, min_periods=bb_period).mean()
        bb_stddev = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
        bb_up = bb_mid + bb_std * bb_stddev
        bb_low = bb_mid - bb_std * bb_stddev
        bb_width = (bb_up - bb_low) / bb_mid.replace(0.0, np.nan)

        out.update({
            "rsi": rsi,
            "bb_mid": bb_mid,
            "bb_up": bb_up,
            "bb_low": bb_low,
            "bb_width": bb_width,
        })

    return out


def compute_common_ts(data_by_symbol: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    common = None
    for df in data_by_symbol.values():
        idx = set(df.index)
        common = idx if common is None else common.intersection(idx)
    return sorted(common or [])




def build_selection_reason_summary(trace_path: str | Path) -> pd.DataFrame:
    path = Path(trace_path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    expanded = []
    for row in rows:
        stage = str(row.get("stage", "") or "")
        symbol = str(row.get("symbol", "") or "")
        side = str(row.get("side", "") or "")
        strategy_id = str(row.get("strategy_id", "") or "")

        reasons = row.get("reasons", [])
        if isinstance(reasons, str):
            reasons = [reasons] if reasons else []
        elif not isinstance(reasons, list):
            reasons = []

        if not reasons:
            expanded.append({
                "stage": stage,
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "reason": "",
                "count": 1,
            })
            continue

        for reason in reasons:
            expanded.append({
                "stage": stage,
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "reason": str(reason or ""),
                "count": 1,
            })

    if not expanded:
        return pd.DataFrame()

    df = pd.DataFrame(expanded)
    summary = (
        df.groupby(["stage", "symbol", "side", "strategy_id", "reason"], dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values(["stage", "count", "symbol", "side", "strategy_id", "reason"], ascending=[True, False, True, True, True, True])
    )
    return summary


def build_selection_stage_summary(trace_path: str | Path) -> pd.DataFrame:
    path = Path(trace_path)
    if not path.exists():
        return pd.DataFrame()

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["stage", "symbol", "side", "strategy_id"]:
        if col not in df.columns:
            df[col] = ""

    def _bool_int(series_name: str) -> pd.Series:
        if series_name not in df.columns:
            return pd.Series([0] * len(df), index=df.index, dtype=int)
        s = df[series_name]
        if s.dtype == bool:
            return s.astype(int)
        return s.fillna(False).astype(bool).astype(int)

    df["pass_true"] = _bool_int("pass")
    df["pass_false"] = 1 - df["pass_true"]
    df["passed_true"] = _bool_int("passed")
    df["passed_false"] = 1 - df["passed_true"]
    df["kept_true"] = _bool_int("kept")
    df["kept_false"] = 1 - df["kept_true"]
    df["winner_true"] = _bool_int("winner")
    df["winner_false"] = 1 - df["winner_true"]

    if "contextual_penalty" in df.columns:
        df["contextual_penalty_nonzero"] = pd.to_numeric(df["contextual_penalty"], errors="coerce").fillna(0.0).ne(0.0).astype(int)
        df["contextual_penalty_mean"] = pd.to_numeric(df["contextual_penalty"], errors="coerce").fillna(0.0)
    else:
        df["contextual_penalty_nonzero"] = 0
        df["contextual_penalty_mean"] = 0.0

    if "regime_penalty" in df.columns:
        df["regime_penalty_nonzero"] = pd.to_numeric(df["regime_penalty"], errors="coerce").fillna(0.0).ne(0.0).astype(int)
        df["regime_penalty_mean"] = pd.to_numeric(df["regime_penalty"], errors="coerce").fillna(0.0)
    else:
        df["regime_penalty_nonzero"] = 0
        df["regime_penalty_mean"] = 0.0

    for score_col in ["p_win", "policy_score", "contextual_score", "regime_score_mult", "alpha_score"]:
        if score_col not in df.columns:
            df[score_col] = np.nan
        else:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    group_cols = ["stage", "symbol", "side", "strategy_id"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("stage", "size"),
            pass_true=("pass_true", "sum"),
            pass_false=("pass_false", "sum"),
            passed_true=("passed_true", "sum"),
            passed_false=("passed_false", "sum"),
            kept_true=("kept_true", "sum"),
            kept_false=("kept_false", "sum"),
            winner_true=("winner_true", "sum"),
            winner_false=("winner_false", "sum"),
            contextual_penalty_nonzero=("contextual_penalty_nonzero", "sum"),
            regime_penalty_nonzero=("regime_penalty_nonzero", "sum"),
            contextual_penalty_mean=("contextual_penalty_mean", "mean"),
            regime_penalty_mean=("regime_penalty_mean", "mean"),
            p_win_mean=("p_win", "mean"),
            policy_score_mean=("policy_score", "mean"),
            contextual_score_mean=("contextual_score", "mean"),
            regime_score_mult_mean=("regime_score_mult", "mean"),
            alpha_score_mean=("alpha_score", "mean"),
        )
        .reset_index()
        .sort_values(["stage", "rows", "symbol", "side", "strategy_id"], ascending=[True, False, True, True, True])
    )
    return summary




def _f(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)



def _clip01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        x = 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _sigmoid(x: float) -> float:
    import math
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _safe_meta_float(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        if not isinstance(meta, dict):
            return float(default)
        v = meta.get(key, default)
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _compute_pwin_math_v1(meta: dict) -> float:
    policy_score = _safe_meta_float(meta, "policy_score", _safe_meta_float(meta, "score", 0.0))
    score = _safe_meta_float(meta, "score", 0.0)
    expected_return = _safe_meta_float(meta, "expected_return", 0.0)
    signal_strength = _safe_meta_float(meta, "strength", _safe_meta_float(meta, "signal_strength", 0.0))
    competitive_score = _safe_meta_float(meta, "competitive_score", 0.0)
    post_ml_score = _safe_meta_float(meta, "post_ml_score", 0.0)
    portfolio_conviction = _safe_meta_float(meta, "portfolio_conviction", 0.0)

    adx = _safe_meta_float(meta, "adx", 0.0)
    atrp = _safe_meta_float(meta, "atrp", 0.0)
    rsi = _safe_meta_float(meta, "rsi", 50.0)
    ema_gap = abs(_safe_meta_float(meta, "ema_gap_fast_slow", _safe_meta_float(meta, "ema_gap_pct", 0.0)))
    dist_fast = abs(_safe_meta_float(meta, "dist_close_ema_fast", 0.0))

    policy_norm = _clip01(policy_score / 0.00025)
    score_norm = _clip01(score / 0.00006)
    er_norm = _clip01(expected_return / 0.0010)
    signal_norm = _clip01(signal_strength / 2.0)
    comp_norm = _clip01(competitive_score)
    postml_norm = _clip01(post_ml_score)

    adx_low_bonus = _clip01((adx - 14.0) / 10.0) * 0.15
    adx_high_bonus = _clip01((adx - 35.0) / 15.0) * 0.25
    adx_mid_penalty = _clip01(1.0 - abs(adx - 30.0) / 6.0) * 0.35

    atrp_good_bonus = _clip01((0.0082 - atrp) / 0.0045) * 0.30
    atrp_mid_penalty = _clip01((atrp - 0.0083) / 0.0020) * 0.35
    atrp_high_penalty = _clip01((atrp - 0.0105) / 0.0040) * 0.45

    ema_gap_bonus = _clip01(ema_gap / 0.0060) * 0.15
    ema_gap_penalty = _clip01((0.0015 - ema_gap) / 0.0015) * 0.20
    dist_penalty = _clip01((dist_fast - 0.010) / 0.020) * 0.20

    rsi_extreme_penalty = (
        _clip01((25.0 - rsi) / 25.0) * 0.12
        + _clip01((rsi - 75.0) / 25.0) * 0.12
    )

    port_conv_bonus = _clip01((portfolio_conviction - 0.55) / 0.20) * 0.10

    edge = (
        0.22 * policy_norm
        + 0.16 * score_norm
        + 0.14 * er_norm
        + 0.08 * signal_norm
        + 0.06 * comp_norm
        + 0.04 * postml_norm
        + adx_low_bonus
        + adx_high_bonus
        + atrp_good_bonus
        + ema_gap_bonus
        + port_conv_bonus
        - adx_mid_penalty
        - atrp_mid_penalty
        - atrp_high_penalty
        - ema_gap_penalty
        - dist_penalty
        - rsi_extreme_penalty
    )

    p = 0.50 + 0.10 * (_sigmoid((edge - 0.18) * 8.0) - 0.5) * 2.0
    if p < 0.45:
        p = 0.45
    if p > 0.65:
        p = 0.65
    return float(p)


def _resolve_pwin(meta: dict, mode: str) -> tuple[float, float, float]:
    p_ml = _safe_meta_float(meta, "p_win", _safe_meta_float(meta, "ml_p_win", 0.5))
    if p_ml <= 0.0:
        p_ml = 0.5
    p_math = _compute_pwin_math_v1(meta)
    p_hybrid = 0.35 * float(p_ml) + 0.65 * float(p_math)

    mode = str(mode or "ml").lower()
    if mode == "math_v1":
        p_final = p_math
    elif mode == "hybrid_v1":
        p_final = p_hybrid
    else:
        p_final = p_ml

    if p_final < 0.0:
        p_final = 0.0
    if p_final > 1.0:
        p_final = 1.0

    return float(p_ml), float(p_math), float(p_hybrid if p_hybrid <= 1.0 else 1.0)

def load_exit_registry(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def resolve_exit_policy(exit_cfg: dict, strategy_id: str) -> dict:
    strategy_id = str(strategy_id or "")
    strategy_map = dict((exit_cfg or {}).get("strategies", {}) or {})
    if strategy_id in strategy_map:
        return dict(strategy_map.get(strategy_id, {}) or {})
    return dict((exit_cfg or {}).get("default", {}) or {})



def _resolve_runtime_exit_profile_for_candidate(candidate) -> dict:
    sm = dict(getattr(candidate, "signal_meta", {}) or {})
    strategy_id = str(getattr(candidate, "strategy_id", "") or sm.get("strategy_id", ""))
    side = str(getattr(candidate, "side", "") or sm.get("side", "")).lower()

    base_profile = str(sm.get("ctx_exit_profile", "normal") or "normal")
    base_tp = float(sm.get("ctx_tp_mult", 1.0) or 1.0)
    base_sl = float(sm.get("ctx_sl_mult", 1.0) or 1.0)
    base_time_stop = int(sm.get("ctx_time_stop_bars", 12) or 12)

    portfolio_regime = str(sm.get("portfolio_regime", "") or "").lower()
    portfolio_breadth = float(sm.get("portfolio_breadth", 0.0) or 0.0)
    p_win = float(sm.get("p_win", 0.0) or 0.0)
    adx = float(sm.get("adx", 0.0) or 0.0)
    atrp = float(sm.get("atrp", 0.0) or 0.0)
    ema_gap = abs(float(sm.get("ema_gap_fast_slow", sm.get("ema_gap_pct", 0.0)) or 0.0))

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
            return {
                "ctx_exit_profile": "runner",
                "ctx_tp_mult": 1.2,
                "ctx_sl_mult": 1.0,
                "ctx_time_stop_bars": 24,
            }
        if is_fast_exit:
            return {
                "ctx_exit_profile": "fast_exit",
                "ctx_tp_mult": 0.8,
                "ctx_sl_mult": 0.8,
                "ctx_time_stop_bars": 8,
            }
        return {
            "ctx_exit_profile": "normal",
            "ctx_tp_mult": 1.0,
            "ctx_sl_mult": 1.0,
            "ctx_time_stop_bars": 12,
        }

    return {
        "ctx_exit_profile": base_profile,
        "ctx_tp_mult": base_tp,
        "ctx_sl_mult": base_sl,
        "ctx_time_stop_bars": base_time_stop,
    }


def _resolve_shadow_exit_context(selected_candidates, weights: dict, symbol: str) -> dict:
    selected_candidates = list(selected_candidates or [])
    weights = dict(weights or {})

    signal_side = ""
    regime_on = True
    strategy_id = ""
    ctx_exit_profile = "normal"
    ctx_tp_mult = 1.0
    ctx_sl_mult = 1.0
    ctx_time_stop_bars = 12
    p_win = 0.0
    policy_score = 0.0
    size_mult = 0.0
    adx = 0.0
    atrp = 0.0
    ema_gap = 0.0
    portfolio_regime = ""
    portfolio_breadth = 0.0
    portfolio_avg_pwin = 0.0

    for c in selected_candidates:
        if str(getattr(c, "symbol", "") or "") != str(symbol):
            continue

        signal_side = str(getattr(c, "side", "") or "").lower()
        strategy_id = str(getattr(c, "strategy_id", "") or "")
        sm = dict(getattr(c, "signal_meta", {}) or {})

        if "regime_as_metadata" in sm:
            regime_on = bool(sm.get("regime_as_metadata", True))
        elif "portfolio_regime" in sm:
            regime_on = str(sm.get("portfolio_regime", "normal") or "normal").lower() != "off"

        ctx_exit_profile = str(sm.get("ctx_exit_profile", "normal") or "normal")
        ctx_tp_mult = float(sm.get("ctx_tp_mult", 1.0) or 1.0)
        ctx_sl_mult = float(sm.get("ctx_sl_mult", 1.0) or 1.0)
        ctx_time_stop_bars = int(sm.get("ctx_time_stop_bars", 12) or 12)

        p_win = float(sm.get("p_win", 0.0) or 0.0)
        policy_score = float(sm.get("policy_score", 0.0) or 0.0)
        size_mult = float(sm.get("policy_size_mult", sm.get("size_mult", 0.0)) or 0.0)
        adx = float(sm.get("adx", 0.0) or 0.0)
        atrp = float(sm.get("atrp", 0.0) or 0.0)
        ema_gap = abs(float(sm.get("ema_gap_fast_slow", sm.get("ema_gap_pct", 0.0)) or 0.0))
        portfolio_regime = str(sm.get("portfolio_regime", "") or "")
        portfolio_breadth = float(sm.get("portfolio_breadth", 0.0) or 0.0)
        portfolio_avg_pwin = float(sm.get("portfolio_avg_pwin", 0.0) or 0.0)
        break

    return {
        "target_weight": float(weights.get(symbol, 0.0) or 0.0),
        "signal_side": str(signal_side),
        "strategy_id": str(strategy_id),
        "regime_on": bool(regime_on),
        "ctx_exit_profile": str(ctx_exit_profile),
        "ctx_tp_mult": float(ctx_tp_mult),
        "ctx_sl_mult": float(ctx_sl_mult),
        "ctx_time_stop_bars": int(ctx_time_stop_bars),
        "p_win": float(p_win),
        "policy_score": float(policy_score),
        "size_mult": float(size_mult),
        "adx": float(adx),
        "atrp": float(atrp),
        "ema_gap": float(ema_gap),
        "portfolio_regime": str(portfolio_regime),
        "portfolio_breadth": float(portfolio_breadth),
        "portfolio_avg_pwin": float(portfolio_avg_pwin),
        "context_source": "research_runtime_shadow",
    }


def _build_lifecycle_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    equity = pd.to_numeric(equity_df.get("equity", pd.Series(dtype=float)), errors="coerce").dropna()

    if equity.empty:
        equity_start = 0.0
        equity_final = 0.0
        total_return_pct = 0.0
        sharpe_annual = 0.0
        max_drawdown_pct = 0.0
        vol_annual = 0.0
    else:
        equity_start = float(equity.iloc[0])
        equity_final = float(equity.iloc[-1])
        total_return_pct = ((equity_final / equity_start) - 1.0) * 100.0 if equity_start != 0.0 else 0.0
        rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        sharpe_annual = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(24 * 365)) if len(rets) and float(rets.std(ddof=0)) > 0 else 0.0
        peak = equity.cummax()
        dd = (equity / peak.replace(0.0, np.nan)) - 1.0
        dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        max_drawdown_pct = float(dd.min() * 100.0)
        vol_annual = float(rets.std(ddof=0) * np.sqrt(24 * 365)) if len(rets) else 0.0

    trade_count = int(len(trades_df))
    if trade_count > 0 and "pnl" in trades_df.columns:
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
        win_rate_pct = float((pnl.gt(0).sum() / max(1, len(pnl))) * 100.0)
    else:
        win_rate_pct = 0.0

    return {
        "equity_start": float(equity_start),
        "equity_final": float(equity_final),
        "total_return_pct": float(total_return_pct),
        "sharpe_annual": float(sharpe_annual),
        "max_drawdown_pct": float(max_drawdown_pct),
        "vol_annual": float(vol_annual),
        "trade_count": int(trade_count),
        "win_rate_pct": float(win_rate_pct),
    }


def compute_metrics(port_ret: pd.Series, equity: pd.Series) -> dict:
    ret = pd.to_numeric(port_ret, errors="coerce").fillna(0.0)
    eq = pd.to_numeric(equity, errors="coerce").ffill().fillna(1000.0)

    total_return_pct = float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0) if len(eq) else 0.0
    vol = float(ret.std(ddof=0) * np.sqrt(24 * 365)) if len(ret) else 0.0
    sharpe = 0.0
    if ret.std(ddof=0) > 0:
        sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(24 * 365))

    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd_pct = float(dd.min() * 100.0) if len(dd) else 0.0
    win_rate = float((ret > 0).mean() * 100.0) if len(ret) else 0.0

    return {
        "total_return_pct": total_return_pct,
        "sharpe_annual": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "equity_final": float(eq.iloc[-1]) if len(eq) else 1000.0,
        "win_rate_pct": win_rate,
        "vol_annual": vol,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--target-exposure", type=float, default=0.07)
    ap.add_argument("--symbol-cap", type=float, default=0.50)
    ap.add_argument("--allocator-profile", default="symbol_net")
    ap.add_argument("--projection-profile", default="net_symbol")
    ap.add_argument("--allocator-mode", default="snapshot", choices=["snapshot", "legacy_multi_strategy"])
    ap.add_argument("--legacy-competition-mode", default="off", choices=["off", "best_per_symbol", "top1_global", "top2_global", "top3_global"])
    ap.add_argument("--legacy-score-power", type=float, default=1.0)
    ap.add_argument("--legacy-min-score", type=float, default=1e-12)
    ap.add_argument("--legacy-symbol-score-agg", default="sum", choices=["sum", "max"])
    ap.add_argument("--legacy-normalize-total", action="store_true")
    ap.add_argument("--legacy-switch-hysteresis", type=float, default=0.10)
    ap.add_argument("--legacy-min-switch-bars", type=int, default=6)
    ap.add_argument("--legacy-rebalance-deadband", type=float, default=0.03)
    ap.add_argument("--legacy-weight-blend-alpha", type=float, default=0.55)
    ap.add_argument("--legacy-allocator-smoothing-alpha", type=float, default=0.0)
    ap.add_argument("--legacy-allocator-smoothing-snap-eps", type=float, default=0.0)
    ap.add_argument("--legacy-portfolio-regime-defensive-scale", type=float, default=1.0)
    ap.add_argument("--legacy-portfolio-regime-defensive-conviction-k", type=float, default=0.0)
    ap.add_argument("--legacy-portfolio-regime-aggressive-scale", type=float, default=1.0)
    ap.add_argument("--legacy-portfolio-breadth-high-risk", type=int, default=0)
    ap.add_argument("--legacy-portfolio-pwin-high-risk", type=float, default=0.0)
    ap.add_argument("--legacy-portfolio-high-risk-scale", type=float, default=1.0)
    ap.add_argument("--legacy-strategy-side-post-ml-weight-rules", default="")
    ap.add_argument("--legacy-allocator-max-step-per-bar", type=float, default=1.0)
    ap.add_argument("--policy-config", default="artifacts/policy_config.json")
    ap.add_argument("--policy-profile", default="default")
    ap.add_argument("--selection-policy-config", default="artifacts/selection_policy_config.json")
    ap.add_argument("--pwin-mode", choices=["math_v1", "math_v2"], default="math_v2")
    ap.add_argument("--selection-policy-profile", default="research")
    args = ap.parse_args()

    selection_cfg = load_selection_policy_config(str(args.selection_policy_config))
    selection_trace_path = Path(f"results/selection_trace_{str(args.name)}.jsonl")
    selection_pipeline = SelectionPipelineFactory.build(
        config=selection_cfg,
        profile=str(args.selection_policy_profile),
        trace_path=str(selection_trace_path),
    )

    t0 = time.perf_counter()

    registry_rows = load_registry(args.strategy_registry)
    symbols = extract_symbols(registry_rows)

    data_by_symbol = {sym: load_symbol_df(sym, args.start, args.end, args.exchange, args.cache_dir) for sym in symbols}
    feature_series_by_symbol = {sym: build_feature_map(df, sym) for sym, df in data_by_symbol.items()}

    enriched_feature_frames = {}
    for sym, df in data_by_symbol.items():
        base = df.copy()
        for k, series in feature_series_by_symbol[sym].items():
            if hasattr(series, "reindex"):
                base[k] = series.reindex(base.index)
        enriched_feature_frames[sym] = build_symbol_feature_frame(base)

    if "BTC/USDT:USDT" in enriched_feature_frames:
        btc_feat = enriched_feature_frames["BTC/USDT:USDT"]
        for sym in list(enriched_feature_frames.keys()):
            enriched_feature_frames[sym] = merge_cross_asset_features(
                enriched_feature_frames[sym],
                btc_feat,
                prefix="btc_",
            )
    common_ts = compute_common_ts(data_by_symbol)

    t_load = time.perf_counter()

    book = RegistryOpportunityBook(registry_path=args.strategy_registry)
    fb = FeatureBuilder()
    mm = MetaModel()
    pwin_math_v2 = MathPWinV2()
    pwin_math_v3 = MathPWinV3()

    policy_cfg = {}
    if args.policy_config and Path(args.policy_config).exists():
        policy_cfg = json.loads(Path(args.policy_config).read_text(encoding="utf-8"))

    disabled_strategy_side_pairs = set()
    for _raw in list((policy_cfg or {}).get("disabled_strategy_sides", []) or []):
        _item = str(_raw or "").strip()
        if not _item or "|" not in _item:
            continue
        _sid, _side = _item.split("|", 1)
        disabled_strategy_side_pairs.add((str(_sid).strip().lower(), str(_side).strip().lower()))

    pm = PolicyModel(
        profile=str(args.policy_profile),
        config=dict(policy_cfg or {}),
    )
    bridge = AllocationBridge()
    allocator = Allocator(
        target_exposure=float(args.target_exposure),
        symbol_cap=float(args.symbol_cap),
        profile=str(args.allocator_profile),
        projection_profile=str(args.projection_profile),
    )
    score_projector = ScoreProjector(
        use_policy_scaled_base_weight=False,
        inject_post_ml_competitive_score=True,
    )
    allocation_config = build_allocation_config_from_args(args)
    allocation_engine = build_allocation_engine(
        config=allocation_config,
    )
    allocation_router = ResearchAllocationRouter(
        snapshot_allocator=allocator,
        legacy_allocation_engine=allocation_engine,
    )
    context_enricher = AssetContextEnricher()

    rows = []
    candidate_rows = []
    equity = 1000.0
    prev_alloc = None

    lifecycle_engine = TradeLifecycleEngine(
        maker_fee=0.0002,
        taker_fee=0.0006,
        cooldown_after_close_bars=2,
    )
    lifecycle_exit_cfg = load_exit_registry("artifacts/exit_policy_registry.json")
    lifecycle_balance = 1000.0
    lifecycle_equity_rows = []
    lifecycle_open_rows = []
    lifecycle_event_rows = []
    alloc_trace_path = Path(f"results/allocation_trace_{str(args.name)}.jsonl")
    if alloc_trace_path.exists():
        alloc_trace_path.unlink()

    alloc_input_trace_path = Path(f"results/allocation_inputs_trace_{str(args.name)}.jsonl")
    if alloc_input_trace_path.exists():
        alloc_input_trace_path.unlink()

    for ts in common_ts:
        candles = {}
        for sym in symbols:
            df = data_by_symbol[sym]
            row = df.loc[ts]
            feats = {}
            for feat_name, feat_series in feature_series_by_symbol[sym].items():
                if ts in feat_series.index:
                    feats[feat_name] = float(feat_series.loc[ts])

            _extra_df = enriched_feature_frames.get(sym)
            if _extra_df is not None and ts in _extra_df.index:
                _extra_row = _extra_df.loc[ts]
                for _c, _v in _extra_row.items():
                    if pd.notna(_v):
                        feats[_c] = float(_v)
            candles[sym] = _row_to_candle(
                int(pd.to_numeric(row["timestamp"], errors="coerce")),
                row,
                features=feats,
            )

        opps = book.generate(
            candles=candles,
            ts=int(pd.to_numeric(data_by_symbol[symbols[0]].loc[ts]["timestamp"], errors="coerce")),
            print_debug=False,
        )

        candidates = []
        for opp in opps:
            meta = dict(getattr(opp, "meta", {}) or {})
            candidates.append(
                OpportunityCandidate(
                    ts=int(getattr(opp, "timestamp", 0) or 0),
                    symbol=str(getattr(opp, "symbol", "") or ""),
                    strategy_id=str(getattr(opp, "strategy_id", "") or ""),
                    side=str(getattr(opp, "side", "flat") or "flat"),
                    signal_strength=float(getattr(opp, "strength", 0.0) or 0.0),
                    base_weight=float(meta.get("base_weight", 1.0) or 1.0),
                    signal_meta=meta,
                )
            )

        enriched_candidates = []
        disabled_candidates = []
        for c in candidates:
            _c = context_enricher.enrich_candidate(
                candidate=c,
                ts=ts,
                symbol_df=data_by_symbol[c.symbol],
                feature_map=feature_series_by_symbol[c.symbol],
            )
            _c = seed_candidate_meta(_c)

            _sid_now = str(getattr(_c, "strategy_id", "") or "").lower()
            _side_now = str(getattr(_c, "side", "flat") or "flat").lower()
            if (_sid_now, _side_now) in disabled_strategy_side_pairs:
                _sm = dict(getattr(_c, "signal_meta", {}) or {})
                _sm["reason"] = "disabled_strategy_side"
                _sm["disabled_strategy_side"] = f"{_sid_now}|{_side_now}"
                _sm["disabled_strategy_side_applied"] = True
                _c.signal_meta = _sm
                disabled_candidates.append(_c)
                continue

            enriched_candidates.append(_c)

        portfolio_context = build_portfolio_context(enriched_candidates, score_mode="early")

        feature_rows = []
        for c in enriched_candidates:
            feature_rows.append(
                fb.build_feature_row(
                    candidate=c,
                    portfolio_context=portfolio_context,
                )
            )

        scores = mm.predict_many(feature_rows)

        for c, s in zip(enriched_candidates, scores):
            _sm_pre = dict(getattr(c, "signal_meta", {}) or {})
            _sm_pre["expected_return_ml"] = float(getattr(s, "expected_return", 0.0) or 0.0)
            _sm_pre["p_win_ml"] = float(getattr(s, "p_win", 0.0) or 0.0)

            _p_ml, _p_math, _p_hybrid = _resolve_pwin(_sm_pre, args.pwin_mode)
            _sm_pre["strategy_id"] = str(getattr(c, "strategy_id", "") or "")
            _sm_pre["side"] = str(getattr(c, "side", "flat") or "flat")
            _p_math_v2 = float(pwin_math_v2.predict_from_meta(_sm_pre))
            _p_math_v3 = float(pwin_math_v3.predict_from_meta(_sm_pre))

            if str(args.pwin_mode) == "math_v1":
                _p_final = float(_p_math)
            elif str(args.pwin_mode) == "math_v2":
                _p_final = float(_p_math_v2)
            elif str(args.pwin_mode) == "hybrid_v1":
                _p_final = float(_p_hybrid)
            else:
                _p_final = float(_p_ml)

            try:
                s.p_win = float(_p_final)
            except Exception:
                pass

            try:
                s.expected_return = float(_recompute_expected_return(_sm_pre, float(_p_final)))
            except Exception:
                pass

            try:
                s.score = float(max(0.0, float(_p_final) - 0.5) ** 0.90) * float(max(0.0, getattr(s, "expected_return", 0.0) or 0.0))
            except Exception:
                pass

            _sm_pre["p_win"] = float(_p_final)
            _sm_pre["p_win_math_v1"] = float(_p_math)
            _sm_pre["p_win_math_v2"] = float(_p_math_v2)

            # score floor operacional:
            # si el candidato fue aceptado pero el score quedó microscópico,
            # lo elevamos a un piso mínimo para que el allocator pueda convertirlo
            # en peso real y no lo deje en cero.
            _score_raw = float(_sm_pre.get("score", 0.0) or 0.0)
            _policy_raw = float(_sm_pre.get("policy_score", 0.0) or 0.0)
            _accepted_flag = bool(_sm_pre.get("accept", False))

            if _accepted_flag:
                _score_floor = 1.0e-4
                _policy_floor = 1.0e-4
                if _score_raw < _score_floor:
                    _sm_pre["score"] = float(_score_floor)
                if _policy_raw < _policy_floor:
                    _sm_pre["policy_score"] = float(_policy_floor)
            _sm_pre["p_win_math_v3"] = float(_p_math_v3)
            _sm_pre["p_win_hybrid_v1"] = float(_p_hybrid)
            _sm_pre["p_win_mode"] = str(args.pwin_mode)
            _sm_pre["expected_return"] = float(getattr(s, "expected_return", 0.0) or 0.0)
            _sm_pre["score"] = float(getattr(s, "score", 0.0) or 0.0)
            c.signal_meta = _sm_pre

        decisions = pm.decide_many(scores)

        accepted_pack = [
            (c, fr, s, d)
            for c, fr, s, d in zip(enriched_candidates, feature_rows, scores, decisions)
            if bool(getattr(d, "accept", False))
        ]

        opp_index = {}
        for _opp in opps:
            _key = (
                str(getattr(_opp, "symbol", "") or ""),
                str(getattr(_opp, "strategy_id", "") or ""),
                str(getattr(_opp, "side", "") or ""),
            )
            opp_index[_key] = _opp

        enriched_candidates_scored = []
        for c, fr, s, d in zip(enriched_candidates, feature_rows, scores, decisions):
            sm0 = dict(getattr(c, "signal_meta", {}) or {})
            sm0["expected_return"] = float(getattr(s, "expected_return", 0.0) or 0.0)
            sm0["p_win"] = float(getattr(s, "p_win", 0.0) or 0.0)

            _p_ml, _p_math, _p_hybrid = _resolve_pwin(sm0, args.pwin_mode)
            sm0["strategy_id"] = str(getattr(c, "strategy_id", "") or "")
            sm0["side"] = str(getattr(c, "side", "flat") or "flat")
            _p_math_v2 = float(pwin_math_v2.predict_from_meta(sm0))
            _p_math_v3 = float(pwin_math_v3.predict_from_meta(sm0))
            sm0["p_win_ml"] = float(_p_ml)
            sm0["p_win_math_v1"] = float(_p_math)
            sm0["p_win_math_v2"] = float(_p_math_v2)
            sm0["p_win_math_v3"] = float(_p_math_v3)
            sm0["p_win_hybrid_v1"] = float(_p_hybrid)
            sm0["p_win_mode"] = str(args.pwin_mode)

            if str(args.pwin_mode) == "math_v1":
                sm0["p_win"] = float(_p_math)
            elif str(args.pwin_mode) == "math_v2":
                sm0["p_win"] = float(_p_math_v2)
            elif str(args.pwin_mode) == "hybrid_v1":
                sm0["p_win"] = float(_p_hybrid)
            else:
                sm0["p_win"] = float(_p_ml)
            sm0["score"] = float(getattr(s, "score", 0.0) or 0.0)

            _key = (
                str(getattr(c, "symbol", "") or ""),
                str(getattr(c, "strategy_id", "") or ""),
                str(getattr(c, "side", "") or ""),
            )
            _opp = opp_index.get(_key)

            _competitive_score = float(sm0.get("meta_competitive_score", sm0.get("competitive_score", 0.0)) or 0.0)
            _post_ml_score = float(sm0.get("meta_post_ml_score", sm0.get("post_ml_score", 0.0)) or 0.0)

            if _opp is not None:
                try:
                    _opp.meta = dict(getattr(_opp, "meta", {}) or {})
                    _opp.meta.update(sm0)
                    _competitive_score = float(compute_competitive_score(_opp))
                except Exception:
                    pass

                try:
                    _opp.meta = dict(getattr(_opp, "meta", {}) or {})
                    _opp.meta.update(sm0)
                    _opp.meta["competitive_score"] = float(_competitive_score)
                    _opp.meta["post_ml_score"] = float(_competitive_score) * float(sm0.get("p_win", 0.0) or 0.0)
                    _post_ml_score = float(compute_post_ml_competitive_score(_opp))
                except Exception:
                    pass

            _accept_flag = bool(getattr(d, "accept", False))
            if _accept_flag and _competitive_score <= 0.0:
                _fallback_strength = abs(float(getattr(c, "signal_strength", 0.0) or 0.0))
                _fallback_base_weight = float(
                    getattr(c, "base_weight", sm0.get("base_weight", 1.0)) or sm0.get("base_weight", 1.0) or 1.0
                )
                _competitive_score = float(_fallback_strength * _fallback_base_weight)

            if _accept_flag and _post_ml_score <= 0.0 and _competitive_score > 0.0:
                _fallback_pwin = max(0.0, float(sm0.get("p_win", 0.0) or 0.0))
                _fallback_size_mult = float(getattr(d, "size_mult", sm0.get("policy_size_mult", 1.0)) or 1.0)
                _fallback_size_mult = max(0.50, min(1.50, _fallback_size_mult))
                _post_ml_score = float(_competitive_score * _fallback_pwin * _fallback_size_mult)

            sm0["competitive_score"] = float(_competitive_score)
            sm0["post_ml_score"] = float(_post_ml_score)
            sm0["post_ml_competitive_score"] = float(_post_ml_score)
            sm0["meta_competitive_score"] = float(_competitive_score)
            sm0["meta_post_ml_score"] = float(_post_ml_score)
            sm0["meta_post_ml_competitive_score"] = float(_post_ml_score)
            sm0["policy_score"] = float(getattr(d, "policy_score", getattr(s, "score", 0.0)) or 0.0)
            sm0["policy_band"] = str(getattr(d, "band", "") or "")
            sm0["policy_reason"] = str(getattr(d, "reason", "") or "")
            sm0["policy_size_mult"] = float(getattr(d, "size_mult", 0.0) or 0.0)
            sm0["accept"] = bool(getattr(d, "accept", False))

            if bool(sm0.get("accept", False)):
                _score_floor = 1.0e-4
                _policy_floor = 1.0e-4
                if float(sm0.get("score", 0.0) or 0.0) < _score_floor:
                    sm0["score"] = float(_score_floor)
                if float(sm0.get("policy_score", 0.0) or 0.0) < _policy_floor:
                    sm0["policy_score"] = float(_policy_floor)
            sm0["portfolio_regime"] = portfolio_context.get("portfolio_regime")
            sm0["portfolio_breadth"] = portfolio_context.get("portfolio_breadth")
            sm0["portfolio_avg_pwin"] = portfolio_context.get("portfolio_avg_pwin")
            sm0["portfolio_avg_atrp"] = portfolio_context.get("portfolio_avg_atrp")
            sm0["portfolio_avg_strength"] = portfolio_context.get("portfolio_avg_strength")
            sm0["portfolio_conviction"] = portfolio_context.get("portfolio_conviction")
            c.signal_meta = sm0
            enriched_candidates_scored.append(c)

        enriched_candidates = enriched_candidates_scored

        selected_candidates, selected_decisions, selection_meta = selection_pipeline.run(
            candidates=enriched_candidates,
            decisions=decisions,
        )

        selected_candidates, selected_decisions, global_competition_meta = _apply_best_per_symbol_competition(
            selected_candidates,
            selected_decisions,
        )

        selected_candidates = score_projector.enrich_many(selected_candidates)

        projected_selected_candidates = []
        for _c in selected_candidates:
            _sm = dict(getattr(_c, "signal_meta", {}) or {})
            _sm.update(_resolve_runtime_exit_profile_for_candidate(_c))
            _c.signal_meta = _sm
            projected_selected_candidates.append(_c)
        selected_candidates = projected_selected_candidates
        allocation_portfolio_context = build_portfolio_context(selected_candidates, score_mode="allocation")
        selection_meta["disabled_strategy_side_filtered"] = int(len(disabled_candidates))
        selection_meta["global_competition"] = dict(global_competition_meta or {})

        for c, fr, s, d in zip(enriched_candidates, feature_rows, scores, decisions):
            sm = dict(getattr(c, "signal_meta", {}) or {})
            mm_meta = dict(getattr(s, "model_meta", {}) or {})
            pm_meta = dict(getattr(d, "policy_meta", {}) or {})
            _feature_map = dict(getattr(fr, "values", {}) or {})
            candidate_rows.append({
                "portfolio_regime": portfolio_context.get("portfolio_regime"),
                "portfolio_breadth": portfolio_context.get("portfolio_breadth"),
                "portfolio_avg_pwin": portfolio_context.get("portfolio_avg_pwin"),
                "portfolio_avg_atrp": portfolio_context.get("portfolio_avg_atrp"),
                "portfolio_avg_strength": portfolio_context.get("portfolio_avg_strength"),
                "portfolio_conviction": portfolio_context.get("portfolio_conviction"),
                "ts": ts,
                "symbol": c.symbol,
                "strategy_id": c.strategy_id,
                "side": c.side,
                "signal_strength": c.signal_strength,
                "base_weight": c.base_weight,
                "p_win": float(sm.get("p_win", getattr(s, "p_win", 0.0)) or 0.0),
                "p_win_base": float(sm.get("p_win_base", getattr(s, "p_win", 0.0)) or 0.0),
                "p_win_math_v1": float(sm.get("p_win_math_v1", float("nan"))),
                "p_win_math_v2": float(sm.get("p_win_math_v2", float("nan"))),
                "p_win_math_v3": float(sm.get("p_win_math_v3", float("nan"))),
                "p_win_mode": str(sm.get("p_win_mode", "")),
                "expected_return": float(sm.get("expected_return", getattr(s, "expected_return", 0.0)) or 0.0),
                "score": float(sm.get("score", getattr(s, "score", 0.0)) or 0.0),
                "accept": getattr(d, "accept", False),
                "size_mult": getattr(d, "size_mult", 0.0),
                "band": getattr(d, "band", ""),
                "reason": getattr(d, "reason", ""),
                "policy_score": getattr(d, "policy_score", 0.0),
                "policy_profile": pm_meta.get("policy_profile", ""),
                "adx": sm.get("adx", mm_meta.get("adx", 0.0)),
                "atrp": sm.get("atrp", mm_meta.get("atrp", 0.0)),
                "rsi": sm.get("rsi", mm_meta.get("rsi", 0.0)),
                "adx_below_min": pm_meta.get("adx_below_min", mm_meta.get("adx_below_min", False)),
                "ema_gap_below_min": pm_meta.get("ema_gap_below_min", mm_meta.get("ema_gap_below_min", False)),
                "atrp_low": pm_meta.get("atrp_low", mm_meta.get("atrp_low", False)),
                "adx_low": pm_meta.get("adx_low", mm_meta.get("adx_low", False)),
                "range_expansion_low": pm_meta.get("range_expansion_low", mm_meta.get("range_expansion_low", False)),
                "ret_1h_lag": float(_feature_map.get("ret_1h_lag", 0.0) or 0.0),
                "ret_4h_lag": float(_feature_map.get("ret_4h_lag", 0.0) or 0.0),
                "ret_12h_lag": float(_feature_map.get("ret_12h_lag", 0.0) or 0.0),
                "ret_24h_lag": float(_feature_map.get("ret_24h_lag", 0.0) or 0.0),
                "ema_gap_fast_slow": float(_feature_map.get("ema_gap_fast_slow", 0.0) or 0.0),
                "dist_close_ema_fast": float(_feature_map.get("dist_close_ema_fast", 0.0) or 0.0),
                "dist_close_ema_slow": float(_feature_map.get("dist_close_ema_slow", 0.0) or 0.0),
                "range_pct": float(_feature_map.get("range_pct", 0.0) or 0.0),
                "rolling_vol_24h": float(_feature_map.get("rolling_vol_24h", 0.0) or 0.0),
                "rolling_vol_72h": float(_feature_map.get("rolling_vol_72h", 0.0) or 0.0),
                "atrp_zscore": float(_feature_map.get("atrp_zscore", 0.0) or 0.0),
                "breakout_distance_up": float(_feature_map.get("breakout_distance_up", 0.0) or 0.0),
                "breakout_distance_down": float(_feature_map.get("breakout_distance_down", 0.0) or 0.0),
                "pullback_depth": float(_feature_map.get("pullback_depth", 0.0) or 0.0),
                "btc_ret_24h_lag": float(_feature_map.get("btc_ret_24h_lag", 0.0) or 0.0),
                "btc_rolling_vol_24h": float(_feature_map.get("btc_rolling_vol_24h", 0.0) or 0.0),
                "btc_atrp": float(_feature_map.get("btc_atrp", 0.0) or 0.0),
                "btc_adx": float(_feature_map.get("btc_adx", 0.0) or 0.0),
            })

        alloc_inputs = bridge.to_allocator_inputs(
            candidates=selected_candidates,
            decisions=selected_decisions,
        )

        alloc = allocation_router.allocate(
            mode=str(args.allocator_mode),
            candles=candles,
            selected_candidates=selected_candidates,
            alloc_inputs=alloc_inputs,
            prev_allocation=prev_alloc,
            portfolio_context=allocation_portfolio_context,
        )

        prev_alloc = alloc
        _alloc_meta = dict(getattr(alloc, "meta", {}) or {})

        try:
            _alloc_input_trace_row = {
                "ts": str(ts),
                "n_selected_after_pipeline": int(len(selected_candidates)),
                "selection_meta": dict(selection_meta or {}),
                "global_competition": dict((selection_meta or {}).get("global_competition", {}) or {}),
                "selected_candidates": [
                    {
                        "symbol": str(getattr(c, "symbol", "") or ""),
                        "strategy_id": str(getattr(c, "strategy_id", "") or ""),
                        "side": str(getattr(c, "side", "") or ""),
                        "signal_strength": float(getattr(c, "signal_strength", 0.0) or 0.0),
                        "base_weight": float(getattr(c, "base_weight", 0.0) or 0.0),
                        "signal_meta": dict(getattr(c, "signal_meta", {}) or {}),
                    }
                    for c in selected_candidates
                ],
                "alloc_inputs": list(alloc_inputs or []),
                "legacy_opportunities": list(_alloc_meta.get("legacy_opportunities", []) or []),
                "legacy_opportunities_pre_competition": list(_alloc_meta.get("legacy_opportunities_pre_competition", []) or []),
                "legacy_pre_allocator_trace": dict(_alloc_meta.get("legacy_pre_allocator_trace", {}) or {}),
                "legacy_competition_summary": dict(_alloc_meta.get("legacy_competition_summary", {}) or {}),
                "weights": {str(k): float(v or 0.0) for k, v in dict(getattr(alloc, "weights", {}) or {}).items()},
                "legacy_stage_weights": dict(_alloc_meta.get("legacy_stage_weights", {}) or {}),
                "legacy_stage_gross_exposure": dict(_alloc_meta.get("legacy_stage_gross_exposure", {}) or {}),
                "raw_scores": dict(_alloc_meta.get("raw_scores", {}) or {}),
                "base_weights": dict(_alloc_meta.get("base_weights", {}) or {}),
                "capped_weights": dict(_alloc_meta.get("capped_weights", {}) or {}),
                "selected_meta": dict(_alloc_meta.get("selected_meta", {}) or {}),
                "allocation_mode": str(_alloc_meta.get("allocation_mode", "") or ""),
                "legacy_allocation_config": dict(_alloc_meta.get("legacy_allocation_config", {}) or {}),
                "allocation_portfolio_regime": allocation_portfolio_context.get("portfolio_regime"),
                "allocation_portfolio_breadth": allocation_portfolio_context.get("portfolio_breadth"),
                "allocation_portfolio_avg_pwin": allocation_portfolio_context.get("portfolio_avg_pwin"),
                "allocation_portfolio_avg_atrp": allocation_portfolio_context.get("portfolio_avg_atrp"),
                "allocation_portfolio_avg_strength": allocation_portfolio_context.get("portfolio_avg_strength"),
                "allocation_portfolio_conviction": allocation_portfolio_context.get("portfolio_conviction"),
            }
            with alloc_input_trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_alloc_input_trace_row) + "\n")
        except Exception:
            pass

        weights = dict(alloc.weights or {})
        _gross_weight = float(sum(abs(float(v or 0.0)) for v in weights.values()))

        lifecycle_engine.decrement_cooldowns()

        # shadow exits first
        for sym in symbols:
            pos = lifecycle_engine.get_open_position(sym)
            if pos is None:
                continue

            df = data_by_symbol[sym]
            i = df.index.get_loc(ts)
            if i <= 0:
                continue

            current_bar = dict(df.iloc[i].to_dict())
            current_bar["timestamp"] = int(pd.to_numeric(df.iloc[i]["timestamp"], errors="coerce"))

            prev_raw = df.iloc[i - 1]
            prev_bar = dict(prev_raw.to_dict())
            prev_bar["timestamp"] = int(pd.to_numeric(prev_raw["timestamp"], errors="coerce"))

            for feat_name, feat_series in feature_series_by_symbol[sym].items():
                try:
                    prev_bar[feat_name] = float(feat_series.iloc[i - 1])
                except Exception:
                    pass
                try:
                    current_bar[feat_name] = float(feat_series.iloc[i])
                except Exception:
                    pass

            exit_context = _resolve_shadow_exit_context(selected_candidates, weights, sym)
            strat_cfg = resolve_exit_policy(lifecycle_exit_cfg, pos.strategy_id)

            decision = lifecycle_engine.evaluate_exit(
                symbol=sym,
                prev_bar=prev_bar,
                current_bar=current_bar,
                exit_policy_cfg=strat_cfg,
                context=exit_context,
                exit_ts=int(current_bar["timestamp"]),
            )

            if str(decision.action).lower() == "close":
                rec = lifecycle_engine.trade_log[-1]
                lifecycle_balance += float(rec.pnl)
                lifecycle_event_rows.append(
                    {
                        "ts": str(ts),
                        "symbol": sym,
                        "strategy_id": pos.strategy_id,
                        "event": "exit",
                        "side": pos.side,
                        "exit_reason": rec.exit_reason,
                        "entry_px": rec.entry_px,
                        "exit_px": rec.exit_px,
                        "qty": rec.qty,
                        "pnl": rec.pnl,
                        "bars_held": rec.bars_held,
                    }
                )

        # shadow entries second
        opened_symbols = set()
        for c in selected_candidates:
            sym = str(getattr(c, "symbol", "") or "")
            strategy_id = str(getattr(c, "strategy_id", "") or "")
            if sym in opened_symbols:
                continue
            if not lifecycle_engine.can_open(sym):
                continue

            target_weight = float(weights.get(sym, 0.0) or 0.0)
            if abs(target_weight) <= 1e-12:
                continue

            side = "long" if target_weight > 0 else "short"
            df = data_by_symbol[sym]
            i = df.index.get_loc(ts)
            if i <= 0:
                continue

            prev_raw = df.iloc[i - 1]
            entry_px = _f(prev_raw.get("close"), 0.0)

            prev_feats = {}
            for feat_name, feat_series in feature_series_by_symbol[sym].items():
                if (i - 1) >= 0:
                    try:
                        prev_feats[feat_name] = float(feat_series.iloc[i - 1])
                    except Exception:
                        pass

            atr = _f(prev_feats.get("atr"), 0.0)
            adx = _f(prev_feats.get("adx"), float("nan"))
            atrp = _f(prev_feats.get("atrp"), float("nan"))

            if entry_px <= 0.0 or atr <= 0.0:
                continue

            notional_frac = min(max(abs(target_weight), 0.01), 0.30)
            notional = lifecycle_balance * notional_frac
            qty = notional / entry_px
            fee = notional * lifecycle_engine.maker_fee
            lifecycle_balance -= fee

            lifecycle_engine.open_position(
                symbol=sym,
                strategy_id=strategy_id,
                side=side,
                entry_ts=int(pd.to_numeric(df.iloc[i]["timestamp"], errors="coerce")),
                entry_px=float(entry_px),
                qty=float(qty),
                entry_atr=float(atr),
                entry_adx=float(adx),
                entry_atrp=float(atrp),
                entry_strength=_f(getattr(c, "signal_strength", 0.0), 0.0),
                entry_reason="shadow_allocator_target_weight",
                entry_meta={
                    "signal_meta": dict(getattr(c, "signal_meta", {}) or {}),
                    "cooldown_after_close_bars": 2,
                    "trace_target_weight": float(target_weight),
                    "entry_notional_frac": float(notional_frac),
                },
            )

            opened_symbols.add(sym)
            lifecycle_event_rows.append(
                {
                    "ts": str(ts),
                    "symbol": sym,
                    "strategy_id": strategy_id,
                    "event": "entry",
                    "side": side,
                    "entry_px": float(entry_px),
                    "qty": float(qty),
                    "target_weight": float(target_weight),
                    "notional_frac": float(notional_frac),
                }
            )

        lifecycle_mtm = 0.0
        for sym, pos in list(lifecycle_engine.open_positions.items()):
            df = data_by_symbol[sym]
            i = df.index.get_loc(ts)
            px = _f(df.iloc[i].get("close"), pos.entry_px)

            if str(pos.side).lower() == "long":
                lifecycle_mtm += (px - pos.entry_px) * pos.qty
            elif str(pos.side).lower() == "short":
                lifecycle_mtm += (pos.entry_px - px) * pos.qty

            lifecycle_open_rows.append(
                {
                    "ts": str(ts),
                    "symbol": pos.symbol,
                    "strategy_id": pos.strategy_id,
                    "side": pos.side,
                    "entry_ts": int(pos.entry_ts),
                    "entry_px": float(pos.entry_px),
                    "qty": float(pos.qty),
                    "bars_held": int(pos.bars_held),
                    "trail_stop": None if pos.trail_stop is None else float(pos.trail_stop),
                    "breakeven_armed": bool(pos.breakeven_armed),
                }
            )

        lifecycle_equity_rows.append(
            {
                "ts": str(ts),
                "equity": float(lifecycle_balance + lifecycle_mtm),
                "balance": float(lifecycle_balance),
                "open_positions": int(len(lifecycle_engine.open_positions)),
            }
        )

        try:
            _trace_row = {
                "ts": str(ts),
                "n_candidates": int(len(enriched_candidates)),
                "n_accepts": int(len(accepted_pack)),
                "n_selected_after_pipeline": int(len(selected_candidates)),
                "n_alloc_inputs": int(len(alloc_inputs)),
                "n_weighted": int(sum(1 for _, v in weights.items() if abs(float(v or 0.0)) > 0.0)),
                "gross_weight": float(_gross_weight),
                "accepted_symbols": [str(getattr(c, "symbol", "") or "") for c in selected_candidates],
                "accepted_scores": {
                    str(getattr(c, "symbol", "") or ""): float(dict(getattr(c, "signal_meta", {}) or {}).get("policy_score", 0.0) or 0.0)
                    for c in selected_candidates
                },
                "accepted_pwins": {
                    str(getattr(c, "symbol", "") or ""): float(dict(getattr(c, "signal_meta", {}) or {}).get("p_win", 0.0) or 0.0)
                    for c in selected_candidates
                },
                "weights": {str(k): float(v or 0.0) for k, v in weights.items()},
                "raw_scores": dict(_alloc_meta.get("raw_scores", {}) or {}),
                "base_weights": dict(_alloc_meta.get("base_weights", {}) or {}),
                "capped_weights": dict(_alloc_meta.get("capped_weights", {}) or {}),
                "selected_meta": dict(_alloc_meta.get("selected_meta", {}) or {}),
                "strategy_weights": dict(_alloc_meta.get("strategy_weights", {}) or {}),
                "symbol_scores": dict(_alloc_meta.get("symbol_scores", {}) or {}),
                "symbol_budget": dict(_alloc_meta.get("symbol_budget", {}) or {}),
                "gross_exposure_pre_target": float(_alloc_meta.get("gross_exposure_pre_target", 0.0) or 0.0),
                "gross_exposure_post_target": float(_alloc_meta.get("gross_exposure_post_target", 0.0) or 0.0),
                "target_exposure": float(_alloc_meta.get("target_exposure", 0.0) or 0.0),
                "target_exposure_scale": float(_alloc_meta.get("target_exposure_scale", 0.0) or 0.0),
                "allocator_hysteresis": dict(_alloc_meta.get("allocator_hysteresis", {}) or {}),
                "legacy_stage_weights": dict(_alloc_meta.get("legacy_stage_weights", {}) or {}),
                "legacy_stage_gross_exposure": dict(_alloc_meta.get("legacy_stage_gross_exposure", {}) or {}),
                "legacy_stage_order": list(_alloc_meta.get("legacy_stage_order", []) or []),
                "legacy_prev_weights_signed": dict(_alloc_meta.get("legacy_prev_weights_signed", {}) or {}),
                "legacy_prev_weights_abs_for_allocator": dict(_alloc_meta.get("legacy_prev_weights_abs_for_allocator", {}) or {}),
                "legacy_opportunities": list(_alloc_meta.get("legacy_opportunities", []) or []),
                "legacy_opportunities_pre_competition": list(_alloc_meta.get("legacy_opportunities_pre_competition", []) or []),
                "legacy_pre_allocator_trace": dict(_alloc_meta.get("legacy_pre_allocator_trace", {}) or {}),
                "legacy_competition_summary": dict(_alloc_meta.get("legacy_competition_summary", {}) or {}),
                "portfolio_regime": _alloc_meta.get("portfolio_regime", ""),
                "allocation_portfolio_regime": allocation_portfolio_context.get("portfolio_regime"),
                "allocation_portfolio_breadth": allocation_portfolio_context.get("portfolio_breadth"),
                "allocation_portfolio_avg_pwin": allocation_portfolio_context.get("portfolio_avg_pwin"),
                "allocation_portfolio_avg_atrp": allocation_portfolio_context.get("portfolio_avg_atrp"),
                "allocation_portfolio_avg_strength": allocation_portfolio_context.get("portfolio_avg_strength"),
                "allocation_portfolio_conviction": allocation_portfolio_context.get("portfolio_conviction"),
                "portfolio_conviction": float(_alloc_meta.get("portfolio_conviction", 0.0) or 0.0),
                "portfolio_conviction_upstream": float(_alloc_meta.get("portfolio_conviction_upstream", 0.0) or 0.0),
                "portfolio_regime_scaler_conviction": float(_alloc_meta.get("portfolio_regime_scaler_conviction", 0.0) or 0.0),
                "portfolio_breadth": float(_alloc_meta.get("portfolio_breadth", 0.0) or 0.0),
                "portfolio_avg_pwin": float(_alloc_meta.get("portfolio_avg_pwin", 0.0) or 0.0),
                "portfolio_avg_strength": float(_alloc_meta.get("portfolio_avg_strength", 0.0) or 0.0),
                "portfolio_regime_scaler_breadth_score": float(_alloc_meta.get("portfolio_regime_scaler_breadth_score", 0.0) or 0.0),
                "portfolio_regime_scaler_pwin_score": float(_alloc_meta.get("portfolio_regime_scaler_pwin_score", 0.0) or 0.0),
                "portfolio_regime_scaler_strength_score": float(_alloc_meta.get("portfolio_regime_scaler_strength_score", 0.0) or 0.0),
                "portfolio_context_seen_by_regime_scaler": dict(_alloc_meta.get("portfolio_context_seen_by_regime_scaler", {}) or {}),
                "portfolio_regime_scale": float(_alloc_meta.get("portfolio_regime_scale", 0.0) or 0.0),
                "postprocess_smoothing_applied": bool(_alloc_meta.get("postprocess_smoothing_applied", False)),
                "postprocess_smoothing_alpha": float(_alloc_meta.get("postprocess_smoothing_alpha", 0.0) or 0.0),
                "postprocess_smoothing_snap_eps": float(_alloc_meta.get("postprocess_smoothing_snap_eps", 0.0) or 0.0),
            }
            with alloc_trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_trace_row) + "\n")
        except Exception:
            pass
        alloc_intents = list(getattr(alloc, "intents", []) or [])
        port_ret = 0.0
        gross_weight = 0.0
        active_symbols = 0

        for sym in symbols:
            w = float(weights.get(sym, 0.0) or 0.0)
            gross_weight += abs(w)
            if abs(w) > 1e-12:
                active_symbols += 1

            df = data_by_symbol[sym]
            i = df.index.get_loc(ts)

            if i < len(df) - 1:
                cur_close = float(df.iloc[i]["close"])
                next_close = float(df.iloc[i + 1]["close"])
                if cur_close != 0:
                    ret = next_close / cur_close - 1.0
                    port_ret += w * ret

        equity *= (1.0 + port_ret)

        lifecycle_equity_now = float(lifecycle_balance + lifecycle_mtm)
        if lifecycle_equity_rows:
            lifecycle_equity_prev = float(lifecycle_equity_rows[-1]["equity"])
        else:
            lifecycle_equity_prev = 1000.0

        lifecycle_port_ret = (
            (lifecycle_equity_now / lifecycle_equity_prev) - 1.0
            if lifecycle_equity_prev != 0.0 else 0.0
        )

        rows.append({
            "ts": ts,
            "n_opps": len(opps),
            "n_features": len(feature_rows),
            "n_scores": len(scores),
            "n_accepts": int(sum(1 for d in decisions if bool(d.accept))),
            "gross_weight": gross_weight,
            "active_symbols": active_symbols,
            "port_ret_simple": port_ret,
            "equity_simple": equity,
            "port_ret": lifecycle_port_ret,
            "equity": lifecycle_equity_now,
        })

    t_run = time.perf_counter()

    out_df = pd.DataFrame(rows)
    out_csv = Path(f"results/research_runtime_{args.name}.csv")
    out_candidates_csv = Path(f"results/research_runtime_candidates_{args.name}.csv")

    selection_summary_df = build_selection_stage_summary(selection_trace_path)
    if not selection_summary_df.empty:
        selection_summary_path = Path(f"results/selection_stage_summary_{args.name}.csv")
        selection_summary_df.to_csv(selection_summary_path, index=False)
        print(f"saved: {selection_summary_path}")

    selection_reason_df = build_selection_reason_summary(selection_trace_path)
    if not selection_reason_df.empty:
        selection_reason_path = Path(f"results/selection_reason_summary_{args.name}.csv")
        selection_reason_df.to_csv(selection_reason_path, index=False)
        print(f"saved: {selection_reason_path}")

    out_json = Path(f"results/research_runtime_metrics_{args.name}.json")

    out_df.to_csv(out_csv, index=False)
    pd.DataFrame(candidate_rows).to_csv(out_candidates_csv, index=False)
    lifecycle_trades_df = pd.DataFrame([{
        "symbol": t.symbol,
        "strategy_id": t.strategy_id,
        "side": t.side,
        "entry_ts": t.entry_ts,
        "exit_ts": t.exit_ts,
        "entry_px": t.entry_px,
        "exit_px": t.exit_px,
        "qty": t.qty,
        "pnl": t.pnl,
        "exit_reason": t.exit_reason,
        "fee": t.fee,
        "bars_held": t.bars_held,
        "entry_adx": t.entry_adx,
        "entry_atrp": t.entry_atrp,
        "entry_atr": t.entry_atr,
    } for t in lifecycle_engine.trade_log])
    lifecycle_equity_df = pd.DataFrame(lifecycle_equity_rows)
    lifecycle_open_df = pd.DataFrame(lifecycle_open_rows)
    lifecycle_events_df = pd.DataFrame(lifecycle_event_rows)

    lifecycle_trades_csv = Path(f"results/research_runtime_lifecycle_trades_{args.name}.csv")
    lifecycle_equity_csv = Path(f"results/research_runtime_lifecycle_equity_{args.name}.csv")
    lifecycle_open_csv = Path(f"results/research_runtime_lifecycle_open_positions_{args.name}.csv")
    lifecycle_events_csv = Path(f"results/research_runtime_lifecycle_events_{args.name}.csv")
    lifecycle_metrics_json = Path(f"results/research_runtime_lifecycle_metrics_{args.name}.json")

    lifecycle_trades_df.to_csv(lifecycle_trades_csv, index=False)
    lifecycle_equity_df.to_csv(lifecycle_equity_csv, index=False)
    lifecycle_open_df.to_csv(lifecycle_open_csv, index=False)
    lifecycle_events_df.to_csv(lifecycle_events_csv, index=False)

    lifecycle_metrics = _build_lifecycle_metrics(lifecycle_trades_df, lifecycle_equity_df)
    lifecycle_metrics_json.write_text(json.dumps(lifecycle_metrics, indent=2), encoding="utf-8")

    print(f"saved: {lifecycle_trades_csv}")
    print(f"saved: {lifecycle_equity_csv}")
    print(f"saved: {lifecycle_open_csv}")
    print(f"saved: {lifecycle_events_csv}")
    print(f"saved: {lifecycle_metrics_json}")



    metrics = dict(lifecycle_metrics or {})
    metrics["rows"] = int(len(out_df))
    metrics["load_seconds"] = float(t_load - t0)
    metrics["run_seconds"] = float(t_run - t_load)
    metrics["total_seconds"] = float(t_run - t0)
    metrics["avg_n_opps"] = float(out_df["n_opps"].mean()) if len(out_df) else 0.0
    metrics["avg_n_accepts"] = float(out_df["n_accepts"].mean()) if len(out_df) else 0.0
    metrics["avg_active_symbols"] = float(out_df["active_symbols"].mean()) if len(out_df) else 0.0
    metrics["avg_gross_weight"] = float(out_df["gross_weight"].mean()) if len(out_df) else 0.0

    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"saved: {out_csv}")
    print(f"saved: {out_candidates_csv}")
    print(f"saved: {out_json}")
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
