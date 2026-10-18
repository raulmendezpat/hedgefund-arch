from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hf_core.trade_lifecycle import TradeLifecycleEngine

from .config import LabelConfig


def load_exit_registry(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def resolve_exit_cfg(registry: dict, strategy_id: str, side: str, symbol: str, cfg: LabelConfig) -> dict:
    """Same precedence as research_runtime.resolve_exit_policy (default < strategy < strategy:side),
    plus the Dynamic Exit V1.2 TP cap the production launcher applies to its allowlist."""
    resolved = dict(registry.get("default", {}) or {})
    resolved.update(dict((registry.get("strategies", {}) or {}).get(strategy_id, {}) or {}))
    resolved.update(dict((registry.get("strategy_sides", {}) or {}).get(f"{strategy_id}:{side}", {}) or {}))
    params = dict(resolved.get("params", {}) or {})
    if cfg.dynamic_exit_tp_cap_atr > 0.0 and f"{symbol}|{side}" in set(cfg.dynamic_exit_allowlist):
        params["tp_atr_mult"] = min(float(params.get("tp_atr_mult", 2.6) or 2.6), cfg.dynamic_exit_tp_cap_atr)
    return {"family": resolved.get("family", "trend_atr_dynamic"), "params": params}


def _bar_dicts(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict("records")


def _simulate_policy_trade(
    *,
    bars: list[dict],
    i: int,
    symbol: str,
    strategy_id: str,
    side: str,
    exit_cfg: dict,
    cfg: LabelConfig,
) -> dict:
    """Open at the close of bar i (the signal bar) and let the exit policy manage it.

    Uses TradeLifecycleEngine so the semantics match research_runtime exactly: at bar j
    the policy sees features of bar j-1 (prev_bar) and the high/low of bar j (current_bar).
    """
    slip = cfg.slippage_bps / 10000.0
    signal_bar = bars[i]
    close_i = float(signal_bar["close"])
    atr_i = float(signal_bar.get("atr", np.nan))
    entry_px = close_i * ((1.0 + slip) if side == "long" else (1.0 - slip))

    eng = TradeLifecycleEngine(taker_fee=cfg.taker_fee, exit_slippage_bps=cfg.slippage_bps)
    eng.open_position(
        symbol=symbol,
        strategy_id=strategy_id,
        side=side,
        entry_ts=int(signal_bar["timestamp"]),
        entry_px=entry_px,
        qty=1.0 / entry_px,  # notional 1.0 -> pnl is a return fraction
        entry_atr=atr_i,
        entry_adx=float(signal_bar.get("adx", np.nan)),
        entry_atrp=float(signal_bar.get("atrp", np.nan)),
    )
    context = {
        "strategy_id": strategy_id,
        "ctx_tp_mult": cfg.ctx_tp_mult,
        "ctx_sl_mult": cfg.ctx_sl_mult,
        "ctx_time_stop_bars": cfg.ctx_time_stop_bars,
    }

    last = min(i + cfg.max_horizon_bars, len(bars) - 1)
    if last <= i:
        return {"label_complete": False}
    for j in range(i + 1, last + 1):
        decision = eng.evaluate_exit(
            symbol=symbol,
            prev_bar=bars[j - 1],
            current_bar=bars[j],
            exit_policy_cfg=exit_cfg,
            context=dict(context),
            exit_ts=int(bars[j]["timestamp"]),
        )
        if str(decision.action).lower() == "close":
            break
    else:
        if j == i + cfg.max_horizon_bars:
            eng.force_close(
                symbol=symbol,
                exit_ts=int(bars[j]["timestamp"]),
                exit_price=float(bars[j]["close"]),
                exit_reason="label_horizon",
            )

    if not eng.trade_log:
        # Ran out of data before any exit: label unknown, not a loss.
        return {"label_complete": False}

    rec = eng.trade_log[-1]
    entry_fee = cfg.taker_fee  # on notional 1.0
    net_ret = float(rec.pnl) - entry_fee
    stop_mult = float(exit_cfg["params"].get("stop_atr_mult", 1.5) or 1.5) * cfg.ctx_sl_mult
    risk_frac = stop_mult * atr_i / close_i if close_i > 0 and atr_i > 0 else np.nan
    return {
        "label_complete": True,
        "exit_reason": str(rec.exit_reason),
        "bars_held": int(rec.bars_held),
        "exit_ts": int(rec.exit_ts),
        "entry_px": float(rec.entry_px),
        "exit_px": float(rec.exit_px),
        "net_ret": net_ret,
        "risk_frac": risk_frac,
        "net_R": net_ret / risk_frac if risk_frac and risk_frac > 0 else np.nan,
        "win": bool(net_ret > 0.0),
        "tp_atr_mult": float(exit_cfg["params"].get("tp_atr_mult", np.nan)),
        "stop_atr_mult": stop_mult,
    }


def _forward_labels(close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray, side: str, cfg: LabelConfig) -> dict[str, np.ndarray]:
    """Policy-agnostic labels for every bar: signed forward returns and excursions, entry at close t."""
    n = len(close)
    sign = 1.0 if side == "long" else -1.0
    out: dict[str, np.ndarray] = {}
    for h in cfg.forward_horizons:
        fwd = np.full(n, np.nan)
        if n > h:
            fwd[:-h] = close[h:] / close[:-h] - 1.0
        net = sign * fwd - cfg.round_trip_cost
        out[f"fwd_net_ret_{h}"] = net
        out[f"fwd_net_atr_{h}"] = net / (atr / close)
    H = cfg.excursion_horizon_bars
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    for t in range(n - H):
        hi = high[t + 1 : t + 1 + H].max()
        lo = low[t + 1 : t + 1 + H].min()
        up = (hi - close[t]) / atr[t]
        dn = (lo - close[t]) / atr[t]
        mfe[t], mae[t] = (up, dn) if side == "long" else (-dn, -up)
    out[f"mfe_atr_{H}"] = mfe
    out[f"mae_atr_{H}"] = mae
    return out


def label_candidates(
    candidates: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    cfg: LabelConfig,
    *,
    exit_registry: dict | None = None,
) -> pd.DataFrame:
    """Attach realized-outcome labels to every candidate row (same row order)."""
    registry = exit_registry if exit_registry is not None else load_exit_registry(cfg.exit_registry_path)
    labeled = []

    for symbol, group in candidates.groupby("symbol", sort=False):
        frame = frames[symbol]
        bars = _bar_dicts(frame)
        pos_by_ts = {ts: k for k, ts in enumerate(frame.index)}
        arrays = {c: frame[c].to_numpy(dtype=float) for c in ("close", "high", "low", "atr")}
        fwd_by_side = {
            s: _forward_labels(arrays["close"], arrays["high"], arrays["low"], arrays["atr"], s, cfg)
            for s in group["side"].unique()
        }
        exit_cfg_cache: dict[tuple[str, str], dict] = {}

        for idx, row in group.iterrows():
            i = pos_by_ts[row["ts"]]
            key = (row["strategy_id"], row["side"])
            if key not in exit_cfg_cache:
                exit_cfg_cache[key] = resolve_exit_cfg(registry, key[0], key[1], symbol, cfg)
            res = {"_idx": idx}
            if np.isfinite(arrays["atr"][i]) and arrays["atr"][i] > 0:
                res.update(
                    _simulate_policy_trade(
                        bars=bars, i=i, symbol=symbol, strategy_id=key[0], side=key[1],
                        exit_cfg=exit_cfg_cache[key], cfg=cfg,
                    )
                )
            else:
                res["label_complete"] = False
            for name, arr in fwd_by_side[row["side"]].items():
                res[name] = float(arr[i])
            labeled.append(res)

    lab = pd.DataFrame(labeled).set_index("_idx").sort_index()
    return candidates.join(lab)
