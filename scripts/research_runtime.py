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
from hf_core import FeatureBuilder, MetaModel, PolicyModel, AllocationBridge, Allocator, OpportunityCandidate, AssetContextEnricher
from hf_core.selection_stages import load_selection_policy_config, SelectionPipelineFactory
from hf_core.pwin_ml_multiwindow import PWinMLMultiWindow
from hf_core.pwin_ml_by_side import PWinMLBySide
from hf_core.ml.feature_expansion import build_symbol_feature_frame, merge_cross_asset_features
from hf_core.research_meta_inputs import seed_candidate_meta, build_portfolio_context


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


def load_symbol_df(symbol: str, start: str, exchange: str, cache_dir: str) -> pd.DataFrame:
    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe="1h",
        start_ms=dt_to_ms_utc(start),
        end_ms=None,
        exchange_id=exchange,
        cache_dir=cache_dir,
        use_cache=True,
        refresh_if_no_end=True,
    ).copy()

    df["ts"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")].copy()
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
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--target-exposure", type=float, default=0.07)
    ap.add_argument("--symbol-cap", type=float, default=0.50)
    ap.add_argument("--allocator-profile", default="symbol_net")
    ap.add_argument("--projection-profile", default="net_symbol")
    ap.add_argument("--policy-config", default="artifacts/policy_config.json")
    ap.add_argument("--policy-profile", default="default")
    ap.add_argument("--selection-policy-config", default="artifacts/selection_policy_config.json")
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

    data_by_symbol = {sym: load_symbol_df(sym, args.start, args.exchange, args.cache_dir) for sym in symbols}
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
    mm.pwin_ml = PWinMLMultiWindow("artifacts/pwin_ml_multiwindow_v1/pwin_ml_operational_registry.json")
    mm.pwin_ml_by_side = PWinMLBySide("artifacts/pwin_ml_by_side_v1/pwin_ml_by_side_operational_registry.json")

    policy_cfg = {}
    if args.policy_config and Path(args.policy_config).exists():
        policy_cfg = json.loads(Path(args.policy_config).read_text(encoding="utf-8"))

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
    context_enricher = AssetContextEnricher()

    rows = []
    candidate_rows = []
    equity = 1000.0
    alloc_trace_path = Path(f"results/allocation_trace_{str(args.name)}.jsonl")
    if alloc_trace_path.exists():
        alloc_trace_path.unlink()

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
        for c in candidates:
            _c = context_enricher.enrich_candidate(
                candidate=c,
                ts=ts,
                symbol_df=data_by_symbol[c.symbol],
                feature_map=feature_series_by_symbol[c.symbol],
            )
            _c = seed_candidate_meta(_c)
            enriched_candidates.append(_c)

        portfolio_context = build_portfolio_context(enriched_candidates)

        feature_rows = []
        for c in enriched_candidates:
            feature_rows.append(
                fb.build_feature_row(
                    candidate=c,
                    portfolio_context=portfolio_context,
                )
            )

        scores = mm.predict_many(feature_rows)
        decisions = pm.decide_many(scores)

        accepted_pack = [
            (c, fr, s, d)
            for c, fr, s, d in zip(enriched_candidates, feature_rows, scores, decisions)
            if bool(getattr(d, "accept", False))
        ]

        enriched_candidates_scored = []
        for c, fr, s, d in zip(enriched_candidates, feature_rows, scores, decisions):
            sm0 = dict(getattr(c, "signal_meta", {}) or {})
            sm0["p_win"] = float(getattr(s, "p_win", 0.0) or 0.0)
            sm0["expected_return"] = float(getattr(s, "expected_return", 0.0) or 0.0)
            sm0["score"] = float(getattr(s, "score", 0.0) or 0.0)
            sm0["post_ml_score"] = float(sm0.get("meta_post_ml_score", sm0.get("post_ml_score", 0.0)) or 0.0)
            sm0["competitive_score"] = float(sm0.get("meta_competitive_score", sm0.get("competitive_score", 0.0)) or 0.0)
            sm0["policy_score"] = float(getattr(d, "policy_score", getattr(s, "score", 0.0)) or 0.0)
            sm0["policy_band"] = str(getattr(d, "band", "") or "")
            sm0["policy_reason"] = str(getattr(d, "reason", "") or "")
            sm0["policy_size_mult"] = float(getattr(d, "size_mult", 0.0) or 0.0)
            sm0["accept"] = bool(getattr(d, "accept", False))
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
                "p_win": getattr(s, "p_win", 0.0),
                "expected_return": getattr(s, "expected_return", 0.0),
                "score": getattr(s, "score", 0.0),
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

        alloc = allocator.allocate(candidates=alloc_inputs)

        weights = dict(alloc.weights or {})
        _gross_weight = float(sum(abs(float(v or 0.0)) for v in weights.values()))

        try:
            _alloc_meta = dict(getattr(alloc, "meta", {}) or {})
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
            if i > 0:
                prev_close = float(df.iloc[i - 1]["close"])
                cur_close = float(df.iloc[i]["close"])
                if prev_close != 0:
                    ret = cur_close / prev_close - 1.0
                    port_ret += w * ret

        equity *= (1.0 + port_ret)

        rows.append({
            "ts": ts,
            "n_opps": len(opps),
            "n_features": len(feature_rows),
            "n_scores": len(scores),
            "n_accepts": int(sum(1 for d in decisions if bool(d.accept))),
            "gross_weight": gross_weight,
            "active_symbols": active_symbols,
            "port_ret": port_ret,
            "equity": equity,
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

    metrics = compute_metrics(out_df["port_ret"], out_df["equity"])
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
