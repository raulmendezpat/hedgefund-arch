from __future__ import annotations

from abc import ABC, abstractmethod

from hf_core.allocation_engine.contracts import AllocationCandidate


class ScoreStrategy(ABC):
    @abstractmethod
    def score(self, candidate: AllocationCandidate) -> float:
        raise NotImplementedError


class PWinExpectedReturnScoreStrategy(ScoreStrategy):
    def __init__(
        self,
        *,
        min_pwin: float = 0.57,
        use_expected_return: bool = True,
        pwin_power: float = 1.0,
    ):
        self.min_pwin = float(min_pwin)
        self.use_expected_return = bool(use_expected_return)
        self.pwin_power = float(pwin_power)

    def score(self, candidate: AllocationCandidate) -> float:
        if str(candidate.side).lower() == "flat":
            return 0.0

        p = float(candidate.p_win or 0.5)
        if p < self.min_pwin:
            return 0.0

        edge = max(0.0, p - 0.5) ** self.pwin_power

        if self.use_expected_return:
            er = max(0.0, float(candidate.expected_return or 0.0))
            if er <= 0.0:
                raw = float(candidate.score or 0.0)
                return max(0.0, raw)
            return edge * er

        raw = float(candidate.score or 0.0)
        return max(edge, raw, 0.0)

class PolicyFirstScoreStrategy(ScoreStrategy):
    def __init__(
        self,
        *,
        min_policy_score: float = 0.0,
        min_pwin: float = 0.0,
        p0: float = 0.50,
        pwin_scale: float = 2.0,
        use_expected_return: bool = True,
        er_scale: float = 500.0,
    ):
        self.min_policy_score = float(min_policy_score)
        self.min_pwin = float(min_pwin)
        self.p0 = float(p0)
        self.pwin_scale = float(pwin_scale)
        self.use_expected_return = bool(use_expected_return)
        self.er_scale = float(er_scale)

    def score(self, candidate: AllocationCandidate) -> float:
        if str(candidate.side).lower() == "flat":
            return 0.0

        raw = max(0.0, float(candidate.score or 0.0))
        if raw <= self.min_policy_score:
            return 0.0

        p = float(candidate.p_win or 0.5)
        if p < self.min_pwin:
            return 0.0

        p_boost = max(0.0, p - self.p0)
        score = raw * (1.0 + self.pwin_scale * p_boost)

        if self.use_expected_return:
            er = float(candidate.expected_return or 0.0)
            if er > 0.0:
                score *= (1.0 + min(er * self.er_scale, 1.0))

        return max(0.0, float(score))

