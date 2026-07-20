from __future__ import annotations

import unittest

from hf_core.trade_lifecycle.contracts import PositionState
from hf_core.trade_lifecycle.exit_policies import build_exit_policy


def make_position(
    *,
    side: str,
    bars_held: int,
    entry_px: float = 100.0,
    qty: float = 2.0,
) -> PositionState:
    return PositionState(
        symbol="TEST/USDT:USDT",
        strategy_id="test_trend",
        side=side,
        entry_ts=0,
        entry_px=entry_px,
        qty=qty,
        entry_atr=1.0,
        bars_held=bars_held,
    )


def make_bars(close: float) -> tuple[dict, dict]:
    return (
        {
            "close": close,
            "atr": 1.0,
            "ema_fast": close,
            "ema_slow": close,
        },
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        },
    )


class ExchangeTimeExitTests(unittest.TestCase):
    def build_policy(self, **overrides):
        params = {
            "stop_atr_mult": 50.0,
            "tp_atr_mult": 50.0,
            "max_hold_bars": 0,
            "invalidate_on_trend_break": False,
            "exchange_time_exit_enabled": True,
            "exchange_positive_after_bars": 10,
            "exchange_positive_min_roe_pct": 0.0,
            "exchange_hard_close_after_bars": 20,
        }
        params.update(overrides)

        return build_exit_policy(
            {
                "family": "trend_atr_dynamic",
                "params": params,
            }
        )

    def evaluate(
        self,
        *,
        side: str,
        bars_held: int,
        close: float,
        context: dict | None = None,
        policy=None,
    ):
        prev_bar, current_bar = make_bars(close)

        return (policy or self.build_policy()).evaluate(
            position=make_position(
                side=side,
                bars_held=bars_held,
            ),
            prev_bar=prev_bar,
            current_bar=current_bar,
            context=dict(context or {}),
        )

    def test_disabled_by_default_preserves_baseline(self):
        policy = build_exit_policy(
            {
                "family": "trend_atr_dynamic",
                "params": {
                    "stop_atr_mult": 50.0,
                    "tp_atr_mult": 50.0,
                    "max_hold_bars": 0,
                },
            }
        )

        decision = self.evaluate(
            side="long",
            bars_held=30,
            close=101.0,
            context={"leverage": 2.0},
            policy=policy,
        )

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.exit_reason, "hold")

    def test_long_positive_roe_after_ten_bars(self):
        decision = self.evaluate(
            side="long",
            bars_held=10,
            close=101.0,
            context={"leverage": 2.0},
        )

        self.assertEqual(decision.action, "close")
        self.assertEqual(
            decision.exit_reason,
            "exchange_positive_roe_timeout",
        )
        self.assertAlmostEqual(
            decision.meta["exchange_roe_pct"],
            2.0,
            places=9,
        )

    def test_short_positive_roe_after_ten_bars(self):
        decision = self.evaluate(
            side="short",
            bars_held=10,
            close=99.0,
            context={"leverage": 3.0},
        )

        self.assertEqual(
            decision.exit_reason,
            "exchange_positive_roe_timeout",
        )
        self.assertAlmostEqual(
            decision.meta["exchange_roe_pct"],
            3.0,
            places=9,
        )

    def test_negative_roe_does_not_trigger_positive_timeout(self):
        decision = self.evaluate(
            side="long",
            bars_held=10,
            close=99.0,
            context={"leverage": 2.0},
        )

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.exit_reason, "hold")

    def test_hard_timeout_closes_negative_position(self):
        decision = self.evaluate(
            side="long",
            bars_held=20,
            close=99.0,
            context={"leverage": 2.0},
        )

        self.assertEqual(decision.action, "close")
        self.assertEqual(
            decision.exit_reason,
            "exchange_max_hold_timeout",
        )

    def test_hard_timeout_has_priority_at_twenty_bars(self):
        decision = self.evaluate(
            side="long",
            bars_held=20,
            close=101.0,
            context={"leverage": 2.0},
        )

        self.assertEqual(
            decision.exit_reason,
            "exchange_max_hold_timeout",
        )

    def test_explicit_exchange_roe_overrides_simulation(self):
        decision = self.evaluate(
            side="long",
            bars_held=10,
            close=99.0,
            context={"exchange_roe_pct": 0.01},
        )

        self.assertEqual(
            decision.exit_reason,
            "exchange_positive_roe_timeout",
        )

    def test_context_can_enable_default_disabled_policy(self):
        policy = build_exit_policy(
            {
                "family": "trend_atr_dynamic",
                "params": {
                    "stop_atr_mult": 50.0,
                    "tp_atr_mult": 50.0,
                    "max_hold_bars": 0,
                },
            }
        )

        decision = self.evaluate(
            side="long",
            bars_held=10,
            close=101.0,
            context={
                "leverage": 2.0,
                "exchange_time_exit_enabled": True,
            },
            policy=policy,
        )

        self.assertEqual(
            decision.exit_reason,
            "exchange_positive_roe_timeout",
        )


if __name__ == "__main__":
    unittest.main()
