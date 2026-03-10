
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from hf.engines.execution_planner import ExecutionPlan
from hf.execution.slippage_model import SlippageModel


@dataclass
class Order:
    order_id: str
    cluster_id: str
    symbol: str
    side: str
    order_type: str
    target_weight: float
    limit_price: Optional[float] = None
    time_offset_bars: int = 0


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    fill_price: float
    filled_weight: float
    bar_index: int
    expected_price: float | None = None
    execution_cost_bps: float | None = None
    execution_cost_pct: float | None = None


@dataclass
class ExecutionState:
    active_orders: dict
    filled_orders: set

    def __init__(self):
        self.active_orders = {}
        self.filled_orders = set()


class OrderSimulator:

    def __init__(self):
        self.state = ExecutionState()
        self.slippage = SlippageModel()

    

    def submit_plan(self, plan: ExecutionPlan):

        for i, s in enumerate(plan.slices):

            oid = f"{plan.cluster_id}_{i}"

            if oid in self.state.active_orders:
                continue

            order = Order(
                order_id=oid,
                cluster_id=plan.cluster_id,
                symbol=plan.symbol,
                side=plan.side,
                order_type=s.order_type,
                target_weight=float(s.target_weight),
                limit_price=s.limit_price,
                time_offset_bars=int(s.time_offset_bars),
            )

            self.state.active_orders[oid] = order


    def build_orders(self, plan: ExecutionPlan) -> List[Order]:

        orders: List[Order] = []

        for i, s in enumerate(plan.slices):

            oid = f"{plan.cluster_id}_{i}"

            order = Order(
                order_id=oid,
                cluster_id=plan.cluster_id,
                symbol=plan.symbol,
                side=plan.side,
                order_type=s.order_type,
                target_weight=float(s.target_weight),
                limit_price=s.limit_price,
                time_offset_bars=int(s.time_offset_bars),
            )

            orders.append(order)
            self.state.active_orders[order.order_id] = order

        return orders

    

    def process_bar(
        self,
        *,
        bar_index: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> List[Fill]:

        fills: List[Fill] = []

        for oid, o in list(self.state.active_orders.items()):

            if oid in self.state.filled_orders:
                continue

            if bar_index < o.time_offset_bars:
                continue

            if o.order_type == "market":

                price = self.slippage.apply_market_slippage(
                    price=float(open_price),
                    side=o.side,
                    target_weight=o.target_weight,
                )

                expected_price = float(open_price)

                if expected_price > 0:
                    cost_pct = (price - expected_price) / expected_price
                else:
                    cost_pct = 0.0

                cost_bps = cost_pct * 10000.0


            elif o.order_type == "limit":

                if o.limit_price is None:
                    continue

                if o.side == "long" and low_price <= o.limit_price:
                    price = o.limit_price
                elif o.side == "short" and high_price >= o.limit_price:
                    price = o.limit_price
                else:
                    continue
            else:
                continue

            fill = Fill(
                order_id=o.order_id,
                symbol=o.symbol,
                side=o.side,
                fill_price=float(price),
                filled_weight=float(o.target_weight),
                bar_index=int(bar_index),
                expected_price=float(open_price),
                execution_cost_bps=float(cost_bps),
                execution_cost_pct=float(cost_pct),
            )

            fills.append(fill)
            self.state.filled_orders.add(oid)

        return fills


    def simulate_plan(
        self,
        *,
        plan: ExecutionPlan,
        bar_index: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
    ) -> List[Fill]:

        orders = self.build_orders(plan)

        fills: List[Fill] = []

        for o in orders:

            if bar_index < o.time_offset_bars:
                continue

            if o.order_type == "market":

                fills.append(
                    Fill(
                        order_id=o.order_id,
                        symbol=o.symbol,
                        side=o.side,
                        fill_price=float(open_price),
                        filled_weight=float(o.target_weight),
                        bar_index=int(bar_index),
                    )
                )

            elif o.order_type == "limit":

                if o.limit_price is None:
                    continue

                if o.side == "long" and low_price <= o.limit_price:
                    price = o.limit_price
                elif o.side == "short" and high_price >= o.limit_price:
                    price = o.limit_price
                else:
                    continue

                fills.append(
                    Fill(
                        order_id=o.order_id,
                        symbol=o.symbol,
                        side=o.side,
                        fill_price=float(price),
                        filled_weight=float(o.target_weight),
                        bar_index=int(bar_index),
                    )
                )

        return fills
