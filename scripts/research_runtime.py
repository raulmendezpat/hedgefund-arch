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
from hf_core import FeatureBuilder, MetaModel, PolicyModel, AllocationBridge, Allocator, OpportunityCandidate


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
    ap.add_argument("--policy-config", default="artifacts/policy_config_v1.json")
    ap.add_argument("--policy-profile", default="default")
    args = ap.parse_args()

    t0 = time.perf_counter()

    registry_rows = load_registry(args.strategy_registry)
    symbols = extract_symbols(registry_rows)

    data_by_symbol = {sym: load_symbol_df(sym, args.start, args.exchange, args.cache_dir) for sym in symbols}
    feature_series_by_symbol = {sym: build_feature_map(df, sym) for sym, df in data_by_symbol.items()}
    common_ts = compute_common_ts(data_by_symbol)

    t_load = time.perf_counter()

    book = RegistryOpportunityBook(registry_path=args.strategy_registry)
    fb = FeatureBuilder()
    mm = MetaModel()

    policy_cfg = {}
    if args.policy_config and Path(args.policy_config).exists():
        policy_cfg = json.loads(Path(args.policy_config).read_text(encoding="utf-8"))

    pm = PolicyModel(
        profile=str(args.policy_profile),
        config=dict(policy_cfg or {}),
    )
    bridge = AllocationBridge()
    allocator = Allocator(target_exposure=float(args.target_exposure), symbol_cap=float(args.symbol_cap))

    rows = []
    candidate_rows = []
    equity = 1000.0

    for ts in common_ts:
        candles = {}
        for sym in symbols:
            df = data_by_symbol[sym]
            row = df.loc[ts]
            feats = {}
            for feat_name, feat_series in feature_series_by_symbol[sym].items():
                if ts in feat_series.index:
                    feats[feat_name] = float(feat_series.loc[ts])
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

        feature_rows = []
        for c in candidates:
            portfolio_context = {
                "portfolio_regime": "research",
                "portfolio_breadth": 0.0,
                "portfolio_avg_pwin": 0.0,
                "portfolio_avg_atrp": 0.0,
                "portfolio_avg_strength": 0.0,
                "portfolio_conviction": 0.0,
                "portfolio_regime_scale_applied": 1.0,
            }
            feature_rows.append(
                fb.build_feature_row(
                    candidate=c,
                    portfolio_context=portfolio_context,
                )
            )

        scores = mm.predict_many(feature_rows)
        decisions = pm.decide_many(scores)

        for c, s, d in zip(candidates, scores, decisions):
            sm = dict(getattr(c, "signal_meta", {}) or {})
            mm_meta = dict(getattr(s, "model_meta", {}) or {})
            pm_meta = dict(getattr(d, "policy_meta", {}) or {})
            candidate_rows.append({
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
            })

        alloc_inputs = bridge.apply(candidates=candidates, decisions=decisions)
        alloc = allocator.allocate(candidates=alloc_inputs)

        weights = dict(alloc.weights or {})
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
