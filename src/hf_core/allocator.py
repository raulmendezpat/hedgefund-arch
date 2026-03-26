from __future__ import annotations

from types import SimpleNamespace

from hf.core.opportunity import Opportunity
from hf.core.types import Allocation, Candle
from hf.engines.alloc_multi_strategy import MultiStrategyAllocator

from hf_core.allocation_engine import AllocationCandidate
from hf_core.allocation_engine.factory import build_snapshot_allocator


class Allocator:
    def __init__(
        self,
        *,
        target_exposure: float = 0.07,
        symbol_cap: float = 0.25,
        profile: str = "snapshot_v1",
        projection_profile: str = "net_symbol",
    ):
        self.target_exposure = float(target_exposure)
        self.symbol_cap = float(symbol_cap)
        self.profile = str(profile)
        self.projection_profile = str(projection_profile)

        score_mode = "policy_first" if self.profile in {"symbol_net", "policy_first", "snapshot_v2"} else "pwin_expected_return"

        self.prev_allocation = None

        if self.profile == "multi_strategy_bridge_v1":
            self.engine = MultiStrategyAllocator(
                score_power=1.0,
                min_score=1e-12,
                symbol_score_agg="sum",
                normalize_total=True,
                switch_hysteresis=0.10,
                min_switch_bars=6,
                rebalance_deadband=0.03,
                weight_blend_alpha=0.55,
                symbol_cap=self.symbol_cap,
                target_exposure=self.target_exposure,
            )
        else:
            self.engine = build_snapshot_allocator(
                target_exposure=self.target_exposure,
                symbol_cap=self.symbol_cap,
                min_pwin=0.0 if score_mode == "policy_first" else 0.55,
                temperature=0.05,
                use_expected_return=True,
                score_mode=score_mode,
                min_policy_score=0.0,
                p0=0.50,
                pwin_scale=2.0,
                er_scale=500.0,
            )

    def _to_candidate(self, item) -> AllocationCandidate | None:
        if item is None:
            return None

        if isinstance(item, dict):
            symbol = str(item.get("symbol", "") or "")
            side = str(item.get("side", "flat") or "flat")
            score = float(item.get("score", 0.0) or 0.0)
            p_win = float(item.get("p_win", 0.5) or 0.5)
            expected_return = float(item.get("expected_return", 0.0) or 0.0)
            base_weight = float(item.get("base_weight", 1.0) or 1.0)
            meta = dict(item.get("meta", {}) or {})
        else:
            symbol = str(getattr(item, "symbol", "") or "")
            side = str(getattr(item, "side", "flat") or "flat")
            score = float(getattr(item, "score", 0.0) or 0.0)
            p_win = float(getattr(item, "p_win", 0.5) or 0.5)
            expected_return = float(getattr(item, "expected_return", 0.0) or 0.0)
            base_weight = float(getattr(item, "base_weight", 1.0) or 1.0)
            meta = dict(getattr(item, "meta", {}) or {})

        if not symbol:
            return None

        if score <= 0.0:
            meta_score = meta.get("score", None)
            if meta_score is not None:
                score = float(meta_score or 0.0)

        if expected_return <= 0.0:
            meta_er = meta.get("expected_return", None)
            if meta_er is not None:
                expected_return = float(meta_er or 0.0)

        if p_win <= 0.0:
            meta_pw = meta.get("p_win", None)
            if meta_pw is not None:
                p_win = float(meta_pw or 0.5)

        return AllocationCandidate(
            symbol=symbol,
            side=side,
            score=score,
            p_win=p_win,
            expected_return=expected_return,
            base_weight=base_weight,
            meta=meta,
        )

    def allocate(self, *args, **kwargs):
        candidates = []

        if "opportunities" in kwargs and kwargs["opportunities"] is not None:
            raw = list(kwargs["opportunities"] or [])
        elif "candidates" in kwargs and kwargs["candidates"] is not None:
            raw = list(kwargs["candidates"] or [])
        elif args:
            raw = list(args[0] or []) if isinstance(args[0], (list, tuple)) else []
        else:
            raw = []

        for item in raw:
            c = self._to_candidate(item)
            if c is not None:
                candidates.append(c)

        if self.profile == "multi_strategy_bridge_v1":
            opps = []
            candles = {}
            opp_side_by_key = {}

            for item in raw:
                c = self._to_candidate(item)
                if c is None:
                    continue

                strategy_id = str(c.meta.get("strategy_id", "unknown_strategy") or "unknown_strategy")
                side = str(c.side or "flat").lower()

                candles[c.symbol] = Candle(
                    ts=None,
                    open=0.0,
                    high=0.0,
                    low=0.0,
                    close=0.0,
                    volume=0.0,
                    features={},
                )

                opp_side_by_key[f"{c.symbol}::{strategy_id}"] = {
                    "symbol": str(c.symbol),
                    "strategy_id": str(strategy_id),
                    "side": str(side),
                }

                opps.append(
                    Opportunity(
                        timestamp=0,
                        symbol=str(c.symbol),
                        strategy_id=str(strategy_id),
                        side=str(side),
                        strength=float(c.score),
                        meta=dict(c.meta or {}),
                    )
                )

            raw_result = self.engine.allocate_from_opportunities(
                candles=candles,
                opportunities=opps,
                prev_allocation=self.prev_allocation,
            )

            # mantener estado raw del allocator real para hysteresis / blend / deadband
            self.prev_allocation = raw_result

            raw_symbol_weights = {
                str(k): float(v or 0.0)
                for k, v in dict(getattr(raw_result, "weights", {}) or {}).items()
            }
            strategy_weights = {
                str(k): float(v or 0.0)
                for k, v in dict((getattr(raw_result, "meta", {}) or {}).get("strategy_weights", {}) or {}).items()
            }

            # Proyección signed/net por símbolo
            signed_weights = {str(sym): 0.0 for sym in raw_symbol_weights.keys()}

            by_symbol = {}
            for key, w in strategy_weights.items():
                info = opp_side_by_key.get(key)
                if not info:
                    continue
                sym = str(info["symbol"])
                side = str(info["side"]).lower()
                by_symbol.setdefault(sym, {"long": 0.0, "short": 0.0, "total": 0.0})
                by_symbol[sym]["total"] += float(w)
                if side == "long":
                    by_symbol[sym]["long"] += float(w)
                elif side == "short":
                    by_symbol[sym]["short"] += float(w)

            for sym, raw_w in raw_symbol_weights.items():
                dist = by_symbol.get(sym, None)
                if not dist:
                    signed_weights[sym] = 0.0
                    continue

                total = float(dist.get("total", 0.0) or 0.0)
                if total <= 0.0:
                    signed_weights[sym] = 0.0
                    continue

                long_w = float(dist.get("long", 0.0) or 0.0)
                short_w = float(dist.get("short", 0.0) or 0.0)
                net_ratio = (long_w - short_w) / total
                signed_weights[sym] = float(raw_w) * float(net_ratio)

            # post-projection cap + target exposure scaling
            capped_signed_weights = {
                str(k): float(max(-self.symbol_cap, min(self.symbol_cap, float(v or 0.0))))
                for k, v in signed_weights.items()
            }

            gross = float(sum(abs(float(v or 0.0)) for v in capped_signed_weights.values()))
            target_exposure = float(getattr(self, "target_exposure", 0.0) or 0.0)

            if target_exposure > 0.0 and gross > 0.0:
                scale = float(target_exposure) / gross
                signed_weights = {
                    str(k): float(v) * scale
                    for k, v in capped_signed_weights.items()
                }
            else:
                signed_weights = capped_signed_weights

            out_meta = dict(getattr(raw_result, "meta", {}) or {})
            out_meta["raw_symbol_weights"] = dict(raw_symbol_weights)
            out_meta["projected_signed_weights"] = dict(signed_weights)
            out_meta["projection_mode"] = "strategy_side_net"

            return SimpleNamespace(weights=signed_weights, meta=out_meta)

        result = self.engine.allocate(candidates)
        return SimpleNamespace(weights=result.weights, meta=result.meta)
