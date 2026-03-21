from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hf.data.ohlcv import fetch_ohlcv_ccxt, dt_to_ms_utc
from hf.engines.opportunity_book import RegistryOpportunityBook
from hf.pipeline.run_portfolio import _adx, _atr, _ema, _row_to_candle
from hf_core import (
    FeatureBuilder,
    MetaModel,
    PolicyModel,
    AllocationBridge,
    Allocator,
    OpportunityCandidate,
    AssetContextEnricher,
)


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
    atrp = atr / close.replace(0.0, pd.NA)

    out = {
        "adx": _adx(df, 14),
        "atr": atr,
        "atrp": atrp,
        "ema_fast": _ema(close, 20),
        "ema_slow": _ema(close, 200),
    }

    # SOL-style features only where available/needed
    if symbol.upper().startswith("SOL/"):
        bb_period = 20
        bb_std = 2.0
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta.clip(upper=0.0))
        avg_gain = gain.rolling(bb_period, min_periods=bb_period).mean()
        avg_loss = loss.rolling(bb_period, min_periods=bb_period).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        bb_mid = close.rolling(bb_period, min_periods=bb_period).mean()
        bb_stddev = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
        bb_up = bb_mid + bb_std * bb_stddev
        bb_low = bb_mid - bb_std * bb_stddev
        bb_width = (bb_up - bb_low) / bb_mid.replace(0.0, pd.NA)

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


def stage_counts(ts, name: str, rows: list[dict], key_field: str = "strategy_id") -> list[dict]:
    if not rows:
        return [{
            "ts": ts,
            "stage": name,
            "group_type": "all",
            "group_value": "__all__",
            "count": 0,
        }]
    out = [{
        "ts": ts,
        "stage": name,
        "group_type": "all",
        "group_value": "__all__",
        "count": len(rows),
    }]
    by_strategy = {}
    by_side = {}
    for r in rows:
        sid = str(r.get("strategy_id", "") or "")
        side = str(r.get("side", "") or "")
        by_strategy[sid] = by_strategy.get(sid, 0) + 1
        by_side[side] = by_side.get(side, 0) + 1
    for k, v in by_strategy.items():
        out.append({"ts": ts, "stage": name, "group_type": "strategy_id", "group_value": k, "count": v})
    for k, v in by_side.items():
        out.append({"ts": ts, "stage": name, "group_type": "side", "group_value": k, "count": v})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--policy-config", default="artifacts/policy_config.json")
    ap.add_argument("--policy-profile", default="default")
    ap.add_argument("--target-exposure", type=float, default=0.07)
    ap.add_argument("--symbol-cap", type=float, default=0.50)
    args = ap.parse_args()

    registry_rows = load_registry(args.strategy_registry)
    symbols = extract_symbols(registry_rows)

    data_by_symbol = {sym: load_symbol_df(sym, args.start, args.exchange, args.cache_dir) for sym in symbols}
    feature_series_by_symbol = {sym: build_feature_map(df, sym) for sym, df in data_by_symbol.items()}
    common_ts = compute_common_ts(data_by_symbol)

    book = RegistryOpportunityBook(registry_path=args.strategy_registry)
    fb = FeatureBuilder()
    mm = MetaModel()

    policy_cfg = {}
    if args.policy_config and Path(args.policy_config).exists():
        policy_cfg = json.loads(Path(args.policy_config).read_text(encoding="utf-8"))

    pm = PolicyModel(profile=str(args.policy_profile), config=dict(policy_cfg or {}))
    bridge = AllocationBridge()
    allocator = Allocator(target_exposure=float(args.target_exposure), symbol_cap=float(args.symbol_cap))
    context_enricher = AssetContextEnricher()

    stage_rows = []
    detail_rows = []

    for ts in common_ts:
        candles = {}
        for sym in symbols:
            df = data_by_symbol[sym]
            row = df.loc[ts]
            feats = {}
            for feat_name, feat_series in feature_series_by_symbol[sym].items():
                if ts in feat_series.index:
                    try:
                        feats[feat_name] = float(feat_series.loc[ts])
                    except Exception:
                        pass
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

        raw_rows = []
        for opp in opps:
            meta = dict(getattr(opp, "meta", {}) or {})
            raw_rows.append({
                "strategy_id": str(getattr(opp, "strategy_id", "") or ""),
                "symbol": str(getattr(opp, "symbol", "") or ""),
                "side": str(getattr(opp, "side", "flat") or "flat"),
                "signal_strength": float(getattr(opp, "strength", 0.0) or 0.0),
                "base_weight": float(meta.get("base_weight", 1.0) or 1.0),
                "meta": meta,
            })
        stage_rows.extend(stage_counts(ts, "raw_opportunities", raw_rows))

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
        cand_rows = [{"strategy_id": c.strategy_id, "symbol": c.symbol, "side": c.side} for c in candidates]
        stage_rows.extend(stage_counts(ts, "candidates", cand_rows))

        enriched_candidates = []
        for c in candidates:
            enriched_candidates.append(
                context_enricher.enrich_candidate(
                    candidate=c,
                    ts=ts,
                    symbol_df=data_by_symbol[c.symbol],
                    feature_map=feature_series_by_symbol[c.symbol],
                )
            )
        enr_rows = [{"strategy_id": c.strategy_id, "symbol": c.symbol, "side": c.side} for c in enriched_candidates]
        stage_rows.extend(stage_counts(ts, "context_enriched", enr_rows))

        feature_rows = []
        for c in enriched_candidates:
            portfolio_context = {
                "portfolio_regime": "research",
                "portfolio_breadth": 0.0,
                "portfolio_avg_pwin": 0.0,
                "portfolio_avg_atrp": 0.0,
                "portfolio_avg_strength": 0.0,
                "portfolio_conviction": 0.0,
                "portfolio_regime_scale_applied": 1.0,
            }
            feature_rows.append(fb.build_feature_row(candidate=c, portfolio_context=portfolio_context))
        feat_rows = [{"strategy_id": f.strategy_id, "symbol": f.symbol, "side": f.side} for f in feature_rows]
        stage_rows.extend(stage_counts(ts, "feature_rows", feat_rows))

        scores = mm.predict_many(feature_rows)
        score_rows = [{
            "strategy_id": s.strategy_id,
            "symbol": s.symbol,
            "side": s.side,
            "p_win": s.p_win,
            "expected_return": s.expected_return,
            "score": s.score,
        } for s in scores]
        stage_rows.extend(stage_counts(ts, "meta_scores", score_rows))

        decisions = pm.decide_many(scores)
        decision_rows = [{
            "strategy_id": d.strategy_id,
            "symbol": d.symbol,
            "side": d.side,
            "accept": bool(d.accept),
            "band": d.band,
            "size_mult": d.size_mult,
            "reason": d.reason,
        } for d in decisions]
        stage_rows.extend(stage_counts(ts, "policy_decisions_all", decision_rows))
        stage_rows.extend(stage_counts(
            ts,
            "policy_decisions_accept",
            [r for r in decision_rows if bool(r.get("accept", False))]
        ))

        alloc_inputs = bridge.apply(candidates=enriched_candidates, decisions=decisions)
        alloc_input_rows = [{
            "strategy_id": c.strategy_id,
            "symbol": c.symbol,
            "side": c.side,
            "base_weight": c.base_weight,
        } for c in alloc_inputs]
        stage_rows.extend(stage_counts(ts, "allocation_inputs", alloc_input_rows))

        alloc = allocator.allocate(candidates=alloc_inputs)
        weights = dict(alloc.weights or {})
        nonzero_weight_rows = []
        for sym, w in weights.items():
            if abs(float(w or 0.0)) > 1e-12:
                nonzero_weight_rows.append({
                    "strategy_id": "__symbol_weight__",
                    "symbol": sym,
                    "side": "weighted",
                    "weight": float(w),
                })
        stage_rows.extend(stage_counts(ts, "final_nonzero_weights", nonzero_weight_rows, key_field="symbol"))

        # intrastage detail rows
        for c, s, d in zip(enriched_candidates, scores, decisions):
            pm_meta = dict(getattr(d, "policy_meta", {}) or {})
            sm = dict(getattr(c, "signal_meta", {}) or {})
            mm_meta = dict(getattr(s, "model_meta", {}) or {})
            detail_rows.append({
                "ts": ts,
                "symbol": c.symbol,
                "strategy_id": c.strategy_id,
                "side": c.side,
                "signal_strength": c.signal_strength,
                "base_weight": c.base_weight,
                "p_win": s.p_win,
                "expected_return": s.expected_return,
                "score": s.score,
                "accept": d.accept,
                "band": d.band,
                "size_mult": d.size_mult,
                "reason": d.reason,
                "alloc_input_present": any(
                    (x.symbol == c.symbol and x.strategy_id == c.strategy_id and x.side == c.side)
                    for x in alloc_inputs
                ),
                "final_symbol_weight": float(weights.get(c.symbol, 0.0) or 0.0),
                "ctx_backdrop": sm.get("ctx_backdrop", mm_meta.get("backdrop", "")),
                "ctx_side_backdrop_alignment": sm.get("ctx_side_backdrop_alignment", mm_meta.get("side_backdrop_alignment", 0.0)),
                "ctx_expected_holding_bars": sm.get("ctx_expected_holding_bars", mm_meta.get("expected_holding_bars", 0)),
                "ctx_exit_profile": sm.get("ctx_exit_profile", mm_meta.get("exit_profile", "")),
                "adx_below_min": pm_meta.get("adx_below_min", mm_meta.get("adx_below_min", False)),
                "ema_gap_below_min": pm_meta.get("ema_gap_below_min", mm_meta.get("ema_gap_below_min", False)),
                "atrp_low": pm_meta.get("atrp_low", mm_meta.get("atrp_low", False)),
                "adx_low": pm_meta.get("adx_low", mm_meta.get("adx_low", False)),
                "range_expansion_low": pm_meta.get("range_expansion_low", mm_meta.get("range_expansion_low", False)),
            })

    stage_df = pd.DataFrame(stage_rows)
    detail_df = pd.DataFrame(detail_rows)

    out_stage = Path(f"results/pipeline_stage_audit_{args.name}.csv")
    out_detail = Path(f"results/pipeline_stage_detail_{args.name}.csv")
    out_summary = Path(f"results/pipeline_stage_summary_{args.name}.txt")

    stage_df.to_csv(out_stage, index=False)
    detail_df.to_csv(out_detail, index=False)

    with out_summary.open("w", encoding="utf-8") as f:
        f.write("=== STAGE SUMMARY ===\n")
        agg = (
            stage_df[stage_df["group_type"] == "all"]
            .groupby("stage", as_index=False)["count"]
            .agg(["mean", "min", "max"])
        )
        f.write(agg.to_string())
        f.write("\n\n=== DETAIL FLOW CHECKS ===\n")
        if not detail_df.empty:
            f.write(f"rows: {len(detail_df)}\n")
            f.write(f"accept_rate: {float(detail_df['accept'].mean())}\n")
            f.write(f"alloc_input_rate: {float(detail_df['alloc_input_present'].mean())}\n")
            f.write(f"nonzero_final_weight_rate: {float((detail_df['final_symbol_weight'].abs() > 1e-12).mean())}\n")

    print(f"saved: {out_stage}")
    print(f"saved: {out_detail}")
    print(f"saved: {out_summary}")

    print("\n=== STAGE SUMMARY ===")
    print(
        stage_df[stage_df["group_type"] == "all"]
        .groupby("stage")["count"]
        .agg(["mean", "min", "max"])
        .sort_index()
        .to_string()
    )

    print("\n=== DETAIL FLOW CHECKS ===")
    if not detail_df.empty:
        print("rows:", len(detail_df))
        print("accept_rate:", float(detail_df["accept"].mean()))
        print("alloc_input_rate:", float(detail_df["alloc_input_present"].mean()))
        print("nonzero_final_weight_rate:", float((detail_df["final_symbol_weight"].abs() > 1e-12).mean()))


if __name__ == "__main__":
    main()
