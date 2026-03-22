from __future__ import annotations

from types import SimpleNamespace

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

        self.engine = build_snapshot_allocator(
            target_exposure=self.target_exposure,
            symbol_cap=self.symbol_cap,
            min_pwin=0.57,
            temperature=0.05,
            use_expected_return=True,
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

        result = self.engine.allocate(candidates)
        return SimpleNamespace(weights=result.weights, meta=result.meta)
