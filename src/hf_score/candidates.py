from __future__ import annotations

import math

import pandas as pd

from hf.core.types import Candle
from hf.engines.opportunity_book import RegistryOpportunityBook

_OHLCV = ("open", "high", "low", "close", "volume")
_NON_FEATURE_COLS = {"timestamp", *_OHLCV}


def _row_features(row: dict) -> dict[str, float]:
    feats = {}
    for k, v in row.items():
        if k in _NON_FEATURE_COLS:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(fv):
            feats[k] = fv
    return feats


def generate_candidates(
    registry_rows: list[dict],
    frames: dict[str, pd.DataFrame],
    *,
    disabled_strategy_sides: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Run every enabled registry engine on every closed bar of its symbol.

    Returns one row per active (long/short) signal. No selection, gating or
    allocation is applied: those are exactly the layers the EV score replaces.
    Engines are stateless per candle (RegistryOpportunityBook rebuilds them on each
    call), so evaluating each symbol on its own full history is equivalent to the
    runtime's bar-by-bar loop.
    """
    book = RegistryOpportunityBook(registry_path="")
    disabled = set(disabled_strategy_sides or set())
    out = []

    for cfg in registry_rows:
        if not bool(cfg.get("enabled", True)):
            continue
        symbol = str(cfg.get("symbol", "") or "")
        strategy_id = str(cfg.get("strategy_id", "") or "")
        engine_name = str(cfg.get("engine", "") or "")
        if symbol not in frames or not strategy_id or engine_name not in book.engine_factories:
            continue

        engine = book.engine_factories[engine_name](cfg)
        base_weight = float(cfg.get("base_weight", 1.0) or 1.0)
        frame = frames[symbol]

        for ts, row in zip(frame.index, frame.to_dict("records")):
            candle = Candle(
                ts=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0) or 0.0),
                features=_row_features(row),
            )
            try:
                generated = engine.generate({symbol: candle}, print_debug=False)
            except TypeError:
                generated = engine.generate({symbol: candle})

            for sig in (generated or {}).values():
                side = str(getattr(sig, "side", "flat") or "flat").lower()
                if side not in {"long", "short"}:
                    continue
                meta = dict(getattr(sig, "meta", {}) or {})
                out.append(
                    {
                        "ts": ts,
                        "timestamp": int(row["timestamp"]),
                        "symbol": symbol,
                        "strategy_id": strategy_id,
                        "engine": engine_name,
                        "side": side,
                        "signal_strength": float(getattr(sig, "strength", 0.0) or 0.0),
                        "base_weight": base_weight,
                        "signal_reason": str(meta.get("reason", "") or ""),
                        "disabled_strategy_side": (strategy_id.lower(), side) in disabled,
                    }
                )

    cols = [
        "ts", "timestamp", "symbol", "strategy_id", "engine", "side", "signal_strength",
        "base_weight", "signal_reason", "disabled_strategy_side",
    ]
    return pd.DataFrame(out, columns=cols)
