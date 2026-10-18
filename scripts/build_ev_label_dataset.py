"""Phase 1 of the EV score: build the labeled candidate dataset.

For every closed bar of every registry symbol, run the strategy engines, keep active
(long/short) candidates and simulate each one with the production exit policy (entry at
the signal bar's close, taker fees, slippage). Writes:

  <out>/candidates_labeled.csv.gz   one row per candidate with point-in-time features + labels
  <out>/summary_by_strategy_side.csv
  <out>/manifest.json               inputs, label config, row counts, data ranges

Example:
  PYTHONPATH=src python scripts/build_ev_label_dataset.py \
    --strategy-registry artifacts/asset_universe_ab/rt_registry_base_plus_first_crypto_discovery.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hf_score import LabelConfig, build_symbol_frames, generate_candidates, label_candidates, load_closed_bars
from hf_score.market_data import BTC_SYMBOL

# Point-in-time feature columns copied from the signal bar onto each candidate.
FEATURE_COLS = [
    "adx", "atr", "atrp", "ema_fast", "ema_slow",
    "ret_1h_lag", "ret_4h_lag", "ret_12h_lag", "ret_24h_lag",
    "ema_gap_fast_slow", "dist_close_ema_fast", "dist_close_ema_slow",
    "range_pct", "rolling_vol_24h", "rolling_vol_72h", "atrp_zscore",
    "breakout_distance_up", "breakout_distance_down", "pullback_depth",
    "btc_ret_24h_lag", "btc_rolling_vol_24h", "btc_atrp", "btc_adx",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _work(job: tuple) -> pd.DataFrame:
    rows, symbol, frames_payload, disabled, cfg_dict = job
    cfg = LabelConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg_dict.items() if k != "round_trip_cost"})
    frames = {symbol: frames_payload}
    cands = generate_candidates(rows, frames, disabled_strategy_sides=disabled)
    if cands.empty:
        return cands
    labeled = label_candidates(cands, frames, cfg)
    feats = frames_payload[[c for c in FEATURE_COLS if c in frames_payload.columns]]
    return labeled.merge(feats, left_on="ts", right_index=True, how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-registry", required=True)
    ap.add_argument("--policy-config", default="artifacts/policy_config.json")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--start", default=None, help="UTC 'YYYY-MM-DD HH:MM:SS'; default = full cache")
    ap.add_argument("--end", default=None)
    ap.add_argument("--warmup-bars", type=int, default=600, help="Bars dropped per symbol so EMA200/ADX are converged")
    ap.add_argument("--slippage-bps", type=float, default=LabelConfig.slippage_bps)
    ap.add_argument("--taker-fee", type=float, default=LabelConfig.taker_fee)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg = LabelConfig(slippage_bps=args.slippage_bps, taker_fee=args.taker_fee)
    registry_rows = json.loads(Path(args.strategy_registry).read_text(encoding="utf-8"))
    policy_cfg = json.loads(Path(args.policy_config).read_text(encoding="utf-8")) if Path(args.policy_config).exists() else {}
    disabled = {
        tuple(s.strip().lower() for s in str(item).split("|", 1))
        for item in (policy_cfg.get("disabled_strategy_sides", []) or [])
        if "|" in str(item)
    }

    symbols = sorted({str(r.get("symbol")) for r in registry_rows if r.get("symbol") and r.get("enabled", True)})
    load_syms = sorted(set(symbols) | {BTC_SYMBOL})
    bars = {s: load_closed_bars(s, cache_dir=args.cache_dir, exchange=args.exchange, start=args.start, end=args.end) for s in load_syms}
    frames = build_symbol_frames(bars)
    # Drop warm-up bars only after features are computed on the full history.
    frames = {s: f.iloc[args.warmup_bars:] for s, f in frames.items()}

    jobs = [
        ([r for r in registry_rows if r.get("symbol") == s], s, frames[s], disabled, cfg.to_dict())
        for s in symbols
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        parts = list(ex.map(_work, jobs))
    data = pd.concat([p for p in parts if not p.empty], ignore_index=True).sort_values(["timestamp", "symbol", "strategy_id"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir or f"artifacts/ev_score/labels_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / "candidates_labeled.csv.gz", index=False, compression="gzip")

    done = data[data["label_complete"] == True]  # noqa: E712
    summary = (
        done.groupby(["strategy_id", "symbol", "side"])
        .agg(
            n=("net_R", "size"),
            win_rate=("win", "mean"),
            mean_net_R=("net_R", "mean"),
            median_net_R=("net_R", "median"),
            mean_net_ret_bps=("net_ret", lambda s: 1e4 * s.mean()),
            mean_bars=("bars_held", "mean"),
            tp_share=("exit_reason", lambda s: (s == "tp").mean()),
            sl_share=("exit_reason", lambda s: (s == "sl").mean()),
            mean_fwd_net_atr_12=("fwd_net_atr_12", "mean"),
        )
        .reset_index()
        .sort_values("mean_net_R")
    )
    summary.to_csv(out_dir / "summary_by_strategy_side.csv", index=False)

    manifest = {
        "created_utc": stamp,
        "git_sha": _git_sha(),
        "strategy_registry": args.strategy_registry,
        "policy_config": args.policy_config,
        "exchange": args.exchange,
        "cache_dir": args.cache_dir,
        "warmup_bars": args.warmup_bars,
        "label_config": cfg.to_dict(),
        "rows": int(len(data)),
        "rows_label_complete": int(len(done)),
        "symbols": {
            s: {"first": str(frames[s].index.min()), "last": str(frames[s].index.max()), "bars": int(len(frames[s]))}
            for s in symbols
        },
        "overall": {
            "mean_net_R": float(done["net_R"].mean()),
            "win_rate": float(done["win"].mean()),
            "mean_net_ret_bps": float(1e4 * done["net_ret"].mean()),
        },
        "notes": [
            "Each candidate is labeled independently, as if taken alone (no cooldown, overlap or allocation).",
            "Entry at the signal bar close; exits by the production exit policy via TradeLifecycleEngine.",
            "Exit registry params were tuned with the lookahead simulator; labels measure them as they are.",
            "Allocator-driven closes (target weight to zero) are not modeled; the label is the exit-policy outcome.",
        ],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "rows_label_complete", "overall", "seconds")}, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
