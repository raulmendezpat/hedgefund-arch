from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf_core.contracts import MetaScore


@dataclass
class PolicyDecision:
    ts: int
    symbol: str
    strategy_id: str
    side: str
    accept: bool
    size_mult: float
    policy_score: float
    band: str
    reason: str
    policy_meta: dict[str, Any] = field(default_factory=dict)


class PolicyModel:
    def __init__(
        self,
        *,
        pwin_min_normal: float = 0.53,
        pwin_min_defensive: float = 0.56,
        er_min_normal: float = 0.0005,
        er_min_defensive: float = 0.0008,
        score_min_normal: float = 0.00003,
        score_min_defensive: float = 0.00006,
        score_ref: float = 0.0015,
        size_min: float = 0.25,
        size_max: float = 1.25,
        defensive_size_penalty: float = 0.85,
    ):
        self.pwin_min_normal = float(pwin_min_normal)
        self.pwin_min_defensive = float(pwin_min_defensive)
        self.er_min_normal = float(er_min_normal)
        self.er_min_defensive = float(er_min_defensive)
        self.score_min_normal = float(score_min_normal)
        self.score_min_defensive = float(score_min_defensive)
        self.score_ref = float(score_ref)
        self.size_min = float(size_min)
        self.size_max = float(size_max)
        self.defensive_size_penalty = float(defensive_size_penalty)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def _thresholds(self, meta: MetaScore) -> tuple[float, float, float, str]:
        regime = str((meta.model_meta or {}).get("regime", "normal") or "normal").lower()
        if regime == "defensive":
            return self.pwin_min_defensive, self.er_min_defensive, self.score_min_defensive, regime
        return self.pwin_min_normal, self.er_min_normal, self.score_min_normal, regime

    def decide_one(self, meta: MetaScore) -> PolicyDecision:
        p = float(meta.p_win)
        er = float(meta.expected_return)
        s = float(meta.score)

        p_thr, er_thr, s_thr, regime = self._thresholds(meta)

        accept = (p >= p_thr) and (er >= er_thr) and (s >= s_thr)

        edge = max(0.0, p - 0.50)
        score_norm = self._clip(s / self.score_ref, 0.0, 1.0) if self.score_ref > 0.0 else 0.0

        raw_size = self.size_min + (self.size_max - self.size_min) * (
            0.65 * score_norm + 0.35 * self._clip(edge / 0.20, 0.0, 1.0)
        )

        if regime == "defensive":
            raw_size *= self.defensive_size_penalty

        size_mult = self._clip(raw_size, self.size_min, self.size_max)

        if not accept:
            size_mult = 0.0
            band = "reject"
            reason = "threshold_fail"
        else:
            if score_norm >= 0.75 and p >= (p_thr + 0.03):
                band = "high_conviction"
            elif score_norm >= 0.40:
                band = "normal"
            else:
                band = "small"
            reason = "accepted"

        return PolicyDecision(
            ts=int(meta.ts),
            symbol=str(meta.symbol),
            strategy_id=str(meta.strategy_id),
            side=str(meta.side),
            accept=bool(accept),
            size_mult=float(size_mult),
            policy_score=float(s),
            band=str(band),
            reason=str(reason),
            policy_meta={
                "regime": regime,
                "p_win": p,
                "expected_return": er,
                "score": s,
                "p_thr": p_thr,
                "er_thr": er_thr,
                "score_thr": s_thr,
                "score_norm": float(score_norm),
                "edge": float(edge),
            },
        )

    def decide_many(self, metas: list[MetaScore]) -> list[PolicyDecision]:
        return [self.decide_one(m) for m in list(metas or [])]
