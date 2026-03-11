
from __future__ import annotations


class SlippageModel:

    def __init__(
        self,
        base_slippage_bps: float = 2.0,
        size_slippage_factor: float = 10.0,
    ):
        """
        base_slippage_bps:
            baseline spread/fees component

        size_slippage_factor:
            additional slippage proportional to order size
        """
        self.base_slippage_bps = float(base_slippage_bps)
        self.size_slippage_factor = float(size_slippage_factor)

    def _compute_slippage(self, target_weight: float) -> float:
        bps = self.base_slippage_bps + self.size_slippage_factor * abs(target_weight)
        return bps / 10000.0

    def apply_market_slippage(
        self,
        *,
        price: float,
        side: str,
        target_weight: float,
    ) -> float:

        slip = self._compute_slippage(target_weight)

        if side == "long":
            return price * (1.0 + slip)

        if side == "short":
            return price * (1.0 - slip)

        return price

    def apply_limit_slippage(
        self,
        *,
        price: float,
        side: str,
        target_weight: float,
    ) -> float:

        slip = self._compute_slippage(target_weight) * 0.25

        if side == "long":
            return price * (1.0 + slip)

        if side == "short":
            return price * (1.0 - slip)

        return price
