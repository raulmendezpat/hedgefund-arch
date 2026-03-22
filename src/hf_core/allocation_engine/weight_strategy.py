from __future__ import annotations

from abc import ABC, abstractmethod
import math


class WeightStrategy(ABC):
    @abstractmethod
    def transform(self, scores: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError


class SoftmaxWeightStrategy(WeightStrategy):
    def __init__(self, *, temperature: float = 0.05):
        self.temperature = max(float(temperature), 1e-6)

    def transform(self, scores: dict[str, float]) -> dict[str, float]:
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
