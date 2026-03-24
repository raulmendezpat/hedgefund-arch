from __future__ import annotations

from abc import ABC, abstractmethod


class WeightStrategy(ABC):
    @abstractmethod
    def transform(self, scores: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError


class SoftmaxWeightStrategy(WeightStrategy):
    def __init__(self, *, temperature: float = 0.05):
        self.temperature = max(float(temperature), 1e-6)

    def transform(self, scores: dict[str, float]) -> dict[str, float]:
        import math

        clean = {k: max(0.0, float(v or 0.0)) for k, v in scores.items() if float(v or 0.0) > 0.0}
        if not clean:
            return {}

        vals = {k: v / self.temperature for k, v in clean.items()}
        vmax = max(vals.values())
        exps = {k: math.exp(v - vmax) for k, v in vals.items()}
        s = sum(exps.values())
        if s <= 0.0:
            return {}
        return {k: float(v / s) for k, v in exps.items()}


class EliteTopWeightStrategy(WeightStrategy):
    def __init__(self, *, top_share: float = 0.90):
        self.top_share = float(top_share)

    def transform(self, scores: dict[str, float]) -> dict[str, float]:
        clean = {k: max(0.0, float(v or 0.0)) for k, v in scores.items() if float(v or 0.0) > 0.0}
        if not clean:
            return {}

        ranked = sorted(clean.items(), key=lambda kv: kv[1], reverse=True)

        if len(ranked) == 1:
            return {ranked[0][0]: 1.0}

        top_key, _ = ranked[0]
        rest = ranked[1:]

        out = {top_key: self.top_share}

        rest_total_score = sum(v for _, v in rest)
        rem = max(0.0, 1.0 - self.top_share)

        if rest_total_score <= 0.0 or rem <= 0.0:
            out[top_key] = 1.0
            return out

        for k, v in rest:
            out[k] = rem * (v / rest_total_score)

        s = sum(out.values())
        if s <= 0.0:
            return {}
        return {k: float(v / s) for k, v in out.items()}
